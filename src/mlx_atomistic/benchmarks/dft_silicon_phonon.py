"""Bounded source-bound Gamma-point phonon validation for diamond Silicon."""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.benchmarks.dft_silicon_bands import silicon_primitive_cell
from mlx_atomistic.dft import (
    GammaCenteredGrid,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicPhononConfig,
    PeriodicPhononSampleSet,
    PeriodicPhononSymmetry,
    PeriodicSCFConfig,
    RuntimeObserver,
    assemble_periodic_phonons,
    compare_periodic_phonon_displacements,
    cubic_reciprocal_symmetry_operations,
    evaluate_periodic_phonon_sample,
    plan_periodic_phonon_displacements,
    read_gth,
    read_periodic_phonon_samples,
    write_periodic_phonon_samples,
)

REFERENCE_SCHEMA = "mlx-atomistic.silicon-gamma-phonon-references.v1"
REFERENCE_SHA256 = "b3781f2e5ac3d11e67fcf6d86eea9ccdd698f654106d03fa6c77ddd22ea6df2e"
REPORT_SCHEMA = "mlx-atomistic.silicon-gamma-phonon-validation.v1"
GTH_RESOURCE_SHA256 = "da66fe0c10d015229e7ad88ea2c4204e6db54c4b9d39e05434d1c08df292d571"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "silicon_phonon_references.json"


def load_silicon_phonon_references() -> dict[str, Any]:
    """Load the hash-pinned Silicon Gamma-phonon reference context."""

    raw = _reference_path().read_bytes()
    if sha256_bytes(raw) != REFERENCE_SHA256:
        raise ValueError("Silicon phonon reference bundle hash mismatch")
    payload = json.loads(raw)
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("unsupported Silicon phonon reference schema")
    return payload


def _scf_config() -> PeriodicSCFConfig:
    return PeriodicSCFConfig(
        max_iterations=80,
        min_iterations=2,
        density_tolerance=1.0e-6,
        energy_tolerance=8.0e-6,
        orbital_tolerance=1.0e-6,
        mixing_beta=0.35,
        mixer="diis",
        max_batch_transient_bytes=512 * 1024**2,
        adaptive_eigensolver_tolerance=True,
        initial_eigensolver_tolerance=1.0e-2,
        eigensolver_tolerance_scale=0.1,
        davidson=PeriodicDavidsonConfig(
            max_iterations=48,
            tolerance=1.0e-6,
            max_subspace_size=64,
            preconditioner_floor=0.25,
        ),
    )


def _phonon_config(displacement_bohr: float) -> PeriodicPhononConfig:
    thresholds = load_silicon_phonon_references()["thresholds"]
    return PeriodicPhononConfig(
        displacement_bohr=displacement_bohr,
        reciprocity_tolerance_hartree_per_bohr2=float(
            thresholds["reciprocity_hartree_per_bohr2"]
        ),
        sum_rule_tolerance_hartree_per_bohr2=float(
            thresholds["sum_rule_hartree_per_bohr2"]
        ),
        acoustic_frequency_tolerance_cm1=float(
            thresholds["acoustic_max_abs_cm1"]
        ),
        frequency_convergence_tolerance_cm1=float(
            thresholds["displacement_frequency_drift_cm1"]
        ),
        eigenvalue_convergence_tolerance_au=float(
            thresholds["displacement_eigenvalue_drift_au"]
        ),
    )


def _displacement_symmetries() -> tuple[PeriodicPhononSymmetry, ...]:
    operations = []
    for index, operation in enumerate(cubic_reciprocal_symmetry_operations()):
        values = np.asarray(operation, dtype=np.float64)
        if np.all(values >= 0.0):
            operations.append(
                PeriodicPhononSymmetry(values, label=f"axis-permutation-{index}")
            )
    if len(operations) != 6:
        raise RuntimeError("Silicon phonon axis-permutation group is incomplete")
    return tuple(operations)


def _system_and_mesh(gth_source: str | Path) -> tuple[PeriodicDFTSystem, Any]:
    references = load_silicon_phonon_references()
    protocol = references["local_protocol"]
    lattice_bohr = float(protocol["lattice_constant_angstrom"]) * ANGSTROM_TO_BOHR
    cell = silicon_primitive_cell(lattice_bohr)
    fractional = np.asarray(((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)))
    pseudo = read_gth(gth_source, element="Si", name="GTH-PBE-q4")
    system = PeriodicDFTSystem(
        cell,
        tuple(protocol["fft_shape"]),
        fractional @ cell,
        pseudo,
    )
    mesh = GammaCenteredGrid(tuple(protocol["gamma_centered_kpoint_mesh"]))
    return system, mesh


def _workload(gth_source: str | Path) -> dict[str, Any]:
    source = Path(gth_source).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("Silicon phonon GTH source must be a regular file")
    if sha256_file(source) != GTH_RESOURCE_SHA256:
        raise ValueError("Silicon phonon GTH source hash mismatch")
    references = load_silicon_phonon_references()
    protocol = {
        "schema_version": "mlx-atomistic.silicon-gamma-phonon-workload.v1",
        "resource": {
            "element": "Si",
            "name": "GTH-PBE-q4",
            "sha256": GTH_RESOURCE_SHA256,
        },
        "reference_sha256": REFERENCE_SHA256,
        "local_protocol": references["local_protocol"],
        "thresholds": references["thresholds"],
        "scf_config": {
            "max_iterations": 80,
            "density_tolerance": 1.0e-6,
            "energy_tolerance_hartree": 8.0e-6,
            "orbital_tolerance": 1.0e-6,
            "davidson_tolerance": 1.0e-6,
        },
    }
    return {
        **protocol,
        "workload_fingerprint": sha256_bytes(canonical_json_bytes(protocol)),
    }


def describe_silicon_phonon_validation(gth_source: str | Path) -> dict[str, Any]:
    """Return the exact bounded material plan without running an SCF."""

    workload = _workload(gth_source)
    system, mesh = _system_and_mesh(gth_source)
    displacements = tuple(
        float(value) for value in workload["local_protocol"]["displacements_bohr"]
    )
    plans = [
        plan_periodic_phonon_displacements(
            system,
            config=_phonon_config(displacement),
            symmetry_operations=_displacement_symmetries(),
        )
        for displacement in displacements
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "planned",
        "workload_fingerprint": workload["workload_fingerprint"],
        "system_fingerprint": system.fingerprint,
        "electronic_kpoint_count": len(mesh.points),
        "electronic_kpoint_policy": (
            "full mesh; displaced structures must not reuse equilibrium point-group reduction"
        ),
        "displacements": [
            {
                "displacement_bohr": plan.displacement_bohr,
                "plan_fingerprint": plan.fingerprint,
                "representative_dofs": list(plan.representative_dofs),
                "central_scf_count": 2 * len(plan.representative_dofs),
            }
            for plan in plans
        ],
        "total_scf_count": sum(2 * len(plan.representative_dofs) for plan in plans),
        "asr_imposed": False,
    }


def _latest_samples(
    output: Path,
    plan: Any,
) -> tuple[PeriodicPhononSampleSet, bool]:
    files = sorted(output.glob("samples-*.npz"))
    if not files:
        return PeriodicPhononSampleSet.empty(plan), False
    latest = read_periodic_phonon_samples(files[-1], plan)
    expected_stage = len(latest.samples)
    observed_stage = int(files[-1].stem.rsplit("-", 1)[-1])
    if observed_stage != expected_stage or len(files) != expected_stage:
        raise ValueError("Silicon phonon sample stages are incomplete or non-canonical")
    return latest, True


def _compact_runtime_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: observation[key]
        for key in (
            "schema_version",
            "total_elapsed_seconds",
            "phase_seconds",
            "work_counters",
            "memory",
            "hpsi_shapes",
        )
        if key in observation
    }


def _prior_displacement_record(output: Path, plan_fingerprint: str) -> dict[str, Any] | None:
    report_path = output.parent / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("existing Silicon phonon report schema is invalid")
    matches = [
        record
        for record in report.get("displacements", ())
        if record.get("plan_fingerprint") == plan_fingerprint
    ]
    if len(matches) > 1:
        raise ValueError("existing Silicon phonon report duplicates a displacement plan")
    return None if not matches else dict(matches[0])


def _run_displacement(
    *,
    system: PeriodicDFTSystem,
    mesh: Any,
    displacement_bohr: float,
    output: Path,
) -> tuple[Any, dict[str, Any]]:
    config = _phonon_config(displacement_bohr)
    plan = plan_periodic_phonon_displacements(
        system,
        config=config,
        symmetry_operations=_displacement_symmetries(),
    )
    samples, resumed = _latest_samples(output, plan)
    missing_before = samples.missing_representatives(plan)
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    for representative in missing_before:
        sample = evaluate_periodic_phonon_sample(
            system,
            plan,
            representative,
            cutoff_hartree=25.0,
            kpoint_mesh=mesh,
            n_bands=4,
            scf_config=_scf_config(),
            observer=observer,
        )
        samples = samples.with_sample(sample)
        write_periodic_phonon_samples(
            output / f"samples-{len(samples.samples):02d}.npz",
            plan,
            samples,
        )
    wall = perf_counter() - started
    result = assemble_periodic_phonons(
        plan,
        samples,
        (28.0855, 28.0855),
        config=config,
    )
    observation = observer.snapshot()
    prior = (
        _prior_displacement_record(output, plan.fingerprint)
        if resumed and not missing_before
        else None
    )
    if prior is not None:
        wall = float(prior["evaluation_wall_seconds"])
        timing_eligible = bool(prior["fresh_timing_eligible"])
        compact_observation = _compact_runtime_observation(
            prior.get("runtime_observation", {})
        )
    else:
        timing_eligible = not resumed
        compact_observation = _compact_runtime_observation(observation)
    return result, {
        "plan_fingerprint": plan.fingerprint,
        "representative_dofs": list(plan.representative_dofs),
        "sample_count": len(samples.samples),
        "resumed": resumed,
        "reused_complete_samples": resumed and not missing_before,
        "fresh_timing_eligible": timing_eligible,
        "evaluation_wall_seconds": wall,
        "runtime_observation": compact_observation,
        "result": result.to_dict(),
    }


def _material_gates(
    fine: Any,
    comparison: Any | None,
    references: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, float | None]]:
    thresholds = references["thresholds"]
    if fine.frequencies_cm1 is None or fine.acoustic_mode_indices is None:
        return {
            "force_constants": False,
            "acoustic_modes": False,
            "stable": False,
            "optical_triplet": False,
            "reference_context": False,
            "displacement_convergence": False,
        }, {
            "optical_mean_cm1": None,
            "optical_spread_cm1": None,
            "reference_abs_error_cm1": None,
        }
    acoustic = set(fine.acoustic_mode_indices)
    optical = np.asarray(
        [
            value
            for index, value in enumerate(fine.frequencies_cm1)
            if index not in acoustic
        ]
    )
    optical_mean = float(np.mean(optical))
    optical_spread = float(np.max(optical) - np.min(optical))
    reference_error = abs(
        optical_mean - float(references["reference"]["optical_frequency_cm1"])
    )
    gates = {
        "force_constants": fine.force_constants_passed,
        "acoustic_modes": fine.acoustic_passed,
        "stable": fine.stable,
        "optical_triplet": optical.size == 3
        and optical_spread <= float(thresholds["optical_triplet_spread_cm1"]),
        "reference_context": reference_error
        <= float(thresholds["reference_optical_mean_abs_cm1"]),
        "displacement_convergence": comparison is not None and comparison.passed,
    }
    return gates, {
        "optical_mean_cm1": optical_mean,
        "optical_spread_cm1": optical_spread,
        "reference_abs_error_cm1": reference_error,
    }


def run_silicon_phonon_validation(
    *,
    gth_source: str | Path,
    out: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or describe the bounded two-displacement Silicon phonon gate."""

    if dry_run:
        return describe_silicon_phonon_validation(gth_source)
    workload = _workload(gth_source)
    references = load_silicon_phonon_references()
    system, mesh = _system_and_mesh(gth_source)
    output = Path(out).expanduser().resolve()
    results = []
    records = []
    for displacement in workload["local_protocol"]["displacements_bohr"]:
        label = str(displacement).replace(".", "p")
        result, record = _run_displacement(
            system=system,
            mesh=mesh,
            displacement_bohr=float(displacement),
            output=output / f"h-{label}",
        )
        results.append(result)
        records.append(record)
    comparison = None
    if all(result.valid for result in results):
        comparison = compare_periodic_phonon_displacements(
            results[0],
            results[1],
            config=_phonon_config(float(workload["local_protocol"]["displacements_bohr"][1])),
        )
    gates, metrics = _material_gates(results[1], comparison, references)
    passed = all(gates.values())
    source_fingerprints = build_source_fingerprints()
    protocol_path = Path(inspect.getsourcefile(run_silicon_phonon_validation) or __file__)
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "verified" if passed else "failed",
        "workload_fingerprint": workload["workload_fingerprint"],
        "system_fingerprint": system.fingerprint,
        "source_fingerprints": {
            "gth_resource_sha256": GTH_RESOURCE_SHA256,
            "reference_sha256": REFERENCE_SHA256,
            "material_protocol_sha256": sha256_file(protocol_path),
            "runtime_fingerprint": source_fingerprints["runtime_fingerprint"],
        },
        "reference": references["reference"],
        "thresholds": references["thresholds"],
        "displacements": records,
        "displacement_convergence": (
            None if comparison is None else comparison.to_dict()
        ),
        "metrics": metrics,
        "gates": gates,
        "asr_imposed": False,
        "blockers": sorted(name for name, value in gates.items() if not value),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".report.json.tmp"
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(output / "report.json")
    return payload


def main(argv: list[str] | None = None) -> None:
    """Run the bounded Silicon Gamma-point phonon validation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gth-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_silicon_phonon_validation(
        gth_source=args.gth_source,
        out=args.out,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(" ".join(f"{key}={value}" for key, value in payload.items()))


if __name__ == "__main__":
    main()
