"""Source-bound periodic fixed-cell Silicon relaxation validation."""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import (
    canonical_json_bytes,
    inspect_generation,
    sha256_bytes,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    collect_host_provenance,
    load_workload,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.dft import (
    MonkhorstPackGrid,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicGeometryOptimizationConfig,
    PeriodicSCFConfig,
    RuntimeObserver,
    optimize_periodic_geometry,
    read_gth,
)

REPORT_SCHEMA = "mlx-atomistic.silicon-periodic-relaxation.v1"
SOURCE_STRUCTURE_SHA256 = "96c53bf0a1caa8a5afc99baeabb19f727483644152a5ca9e0e68efea3d3c972e"
LATTICE_ANGSTROM = 5.460859
DISPLACED_ATOM_INDEX = 4
DISPLACEMENT_ANGSTROM = (0.04, -0.03, 0.02)
CUTOFF_HARTREE = 25.0
FFT_SHAPE = (56, 56, 56)
KPOINT_MESH = (6, 6, 6)
BAND_COUNT = 16
MAX_GEOMETRY_ERROR_ANGSTROM = 0.01
ENERGY_RESTART_TOLERANCE_HARTREE = 5.0e-6
POSITION_RESTART_TOLERANCE_BOHR = 5.0e-4


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _scf_config(manifest: Mapping[str, Any]) -> PeriodicSCFConfig:
    scf = manifest["solver"]["scf"]
    davidson = manifest["solver"]["davidson"]
    return PeriodicSCFConfig(
        max_iterations=int(scf["max_iterations"]),
        min_iterations=int(scf["min_iterations"]),
        density_tolerance=float(scf["density_tolerance"]),
        energy_tolerance=float(scf["energy_tolerance_hartree"]),
        orbital_tolerance=float(scf["orbital_tolerance"]),
        mixing_beta=float(scf["mixing_beta"]),
        mixer=str(scf["mixer"]),
        max_batch_transient_bytes=512 * 1024 * 1024,
        adaptive_eigensolver_tolerance=bool(scf["adaptive_eigensolver_tolerance"]),
        initial_eigensolver_tolerance=float(scf["initial_eigensolver_tolerance"]),
        eigensolver_tolerance_scale=float(scf["eigensolver_tolerance_scale"]),
        davidson=PeriodicDavidsonConfig(
            max_iterations=int(davidson["max_iterations"]),
            tolerance=float(davidson["tolerance"]),
            max_subspace_size=int(davidson["max_subspace_size"]),
            preconditioner_floor=float(davidson["preconditioner_floor"]),
        ),
    )


def _optimization_config(*, reuse_scf_state: bool) -> PeriodicGeometryOptimizationConfig:
    return PeriodicGeometryOptimizationConfig(
        max_steps=12,
        force_tolerance=5.0e-4,
        rms_force_tolerance=3.0e-4,
        displacement_tolerance=3.0e-3,
        initial_step_size=1.0,
        max_step=0.25,
        line_search_shrink=0.5,
        line_search_min_step=1.0e-4,
        max_line_search_iterations=8,
        armijo_constant=1.0e-4,
        history_size=5,
        optimizer="lbfgs",
        reuse_scf_state=reuse_scf_state,
    )


def _ideal_positions(manifest: Mapping[str, Any]) -> np.ndarray:
    fractional = np.asarray(manifest["system"]["fractional_positions"], dtype=np.float64)
    return fractional * (LATTICE_ANGSTROM * ANGSTROM_TO_BOHR)


def _initial_system(manifest: Mapping[str, Any], gth_source: str | Path) -> PeriodicDFTSystem:
    positions = _ideal_positions(manifest)
    positions[DISPLACED_ATOM_INDEX] += (
        np.asarray(DISPLACEMENT_ANGSTROM, dtype=np.float64) * ANGSTROM_TO_BOHR
    )
    lattice_bohr = LATTICE_ANGSTROM * ANGSTROM_TO_BOHR
    return PeriodicDFTSystem(
        (lattice_bohr, lattice_bohr, lattice_bohr),
        FFT_SHAPE,
        positions,
        read_gth(gth_source, element="Si", name="GTH-PBE-q4"),
        electron_count=float(manifest["system"]["electron_count"]),
    )


def _minimum_image(displacement: np.ndarray, lattice_bohr: float) -> np.ndarray:
    return displacement - lattice_bohr * np.rint(displacement / lattice_bohr)


def _geometry_comparison(
    positions_bohr: np.ndarray,
    ideal_bohr: np.ndarray,
) -> dict[str, Any]:
    lattice_bohr = LATTICE_ANGSTROM * ANGSTROM_TO_BOHR
    delta = _minimum_image(positions_bohr - ideal_bohr, lattice_bohr)
    translation = np.mean(delta, axis=0)
    aligned = _minimum_image(delta - translation, lattice_bohr)
    norms_angstrom = np.linalg.norm(aligned, axis=1) / ANGSTROM_TO_BOHR
    return {
        "maximum_translation_aligned_error_angstrom": float(np.max(norms_angstrom)),
        "rms_translation_aligned_error_angstrom": float(
            np.sqrt(np.mean(aligned * aligned)) / ANGSTROM_TO_BOHR
        ),
        "removed_uniform_translation_bohr": translation.tolist(),
        "threshold_angstrom": MAX_GEOMETRY_ERROR_ANGSTROM,
        "passed": float(np.max(norms_angstrom)) <= MAX_GEOMETRY_ERROR_ANGSTROM,
    }


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.silicon-periodic-relaxation-execution.v1",
        "report_schema": REPORT_SCHEMA,
        "scf_config_source": inspect.getsource(_scf_config),
        "optimization_config_source": inspect.getsource(_optimization_config),
        "run_source": inspect.getsource(run_silicon_relaxation),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def run_silicon_relaxation(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    reuse_scf_state: bool = True,
    checkpoint_to: str | Path | None = None,
    checkpoint_step: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run one bounded Silicon periodic relaxation or restart segment."""

    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    source_fingerprints = build_source_fingerprints()
    resume_fingerprint = None
    if resume_from is not None:
        resume_fingerprint = inspect_generation(resume_from)["manifest_sha256"]
    point_identity = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "runtime_fingerprint": source_fingerprints["runtime_fingerprint"],
        "implementation_fingerprint": _implementation_fingerprint(),
        "source_structure_sha256": SOURCE_STRUCTURE_SHA256,
        "reuse_scf_state": reuse_scf_state,
        "resume_checkpoint_manifest_sha256": resume_fingerprint,
    }
    point_identity["point_fingerprint"] = sha256_bytes(canonical_json_bytes(point_identity))
    system = _initial_system(manifest, gth_source)
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    result = optimize_periodic_geometry(
        system,
        cutoff_hartree=CUTOFF_HARTREE,
        kpoint_mesh=MonkhorstPackGrid(KPOINT_MESH),
        n_bands=BAND_COUNT,
        config=_optimization_config(reuse_scf_state=reuse_scf_state),
        scf_config=_scf_config(manifest),
        observer=observer,
        checkpoint_to=checkpoint_to,
        checkpoint_step=checkpoint_step,
        resume_from=resume_from,
        provenance={"point_fingerprint": point_identity["point_fingerprint"]},
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    geometry = _geometry_comparison(result.final_positions, _ideal_positions(manifest))
    final_step = None if not result.steps else result.steps[-1]
    numerical_passed = bool(
        result.status in {"converged", "checkpointed"}
        and result.final_scf is not None
        and result.final_scf.converged
        and np.isfinite(result.final_energy)
        and abs(result.final_scf.electron_count - system.electron_count) <= 1.0e-4
        and result.final_forces is not None
        and np.isfinite(result.final_forces).all()
    )
    scientifically_verified = bool(
        result.status == "converged"
        and numerical_passed
        and geometry["passed"]
        and final_step is not None
        and final_step.max_force <= result.config.force_tolerance
        and final_step.rms_force <= result.config.rms_force_tolerance
        and final_step.step_norm <= result.config.displacement_tolerance
        and all(step.energy <= step.armijo_limit for step in result.steps)
    )
    observation = observer.snapshot()
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": result.status,
        "numerical_passed": numerical_passed,
        "scientifically_verified": scientifically_verified,
        "point": point_identity,
        "host": collect_host_provenance(),
        "protocol": {
            "material": "diamond-silicon-conventional-cubic",
            "lattice_angstrom": LATTICE_ANGSTROM,
            "source_structure_sha256": SOURCE_STRUCTURE_SHA256,
            "source_record": "https://archive.materialscloud.org/records/yf0rj-w3r97",
            "source_doi": "10.24435/materialscloud:s4-3h",
            "displaced_atom_index": DISPLACED_ATOM_INDEX,
            "displacement_angstrom": list(DISPLACEMENT_ANGSTROM),
            "cutoff_hartree": CUTOFF_HARTREE,
            "fft_shape": list(FFT_SHAPE),
            "kpoint_mesh": list(KPOINT_MESH),
            "band_count": BAND_COUNT,
            "solver": manifest["solver"],
            "scf_batch_policy": _scf_config(manifest).batch_policy(),
            "optimization_config": result.config.to_dict(),
        },
        "geometry_comparison": geometry,
        "result": result.to_dict(),
        "runtime": {
            "complete_wall_seconds": elapsed,
            "observation": observation,
        },
    }
    _write_atomic(Path(out), payload)
    return payload


def compare_restart_reports(
    uninterrupted: Mapping[str, Any],
    resumed: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare final scientific state across an outer checkpoint boundary."""

    energy_error = abs(
        float(uninterrupted["result"]["final_energy_hartree"])
        - float(resumed["result"]["final_energy_hartree"])
    )
    position_error = float(
        np.max(
            np.abs(
                np.asarray(uninterrupted["result"]["final_positions_bohr"])
                - np.asarray(resumed["result"]["final_positions_bohr"])
            )
        )
    )
    passed = bool(
        uninterrupted["result"]["status"] == resumed["result"]["status"]
        and uninterrupted["result"]["accepted_step_count"]
        == resumed["result"]["accepted_step_count"]
        and energy_error <= ENERGY_RESTART_TOLERANCE_HARTREE
        and position_error <= POSITION_RESTART_TOLERANCE_BOHR
    )
    return {
        "passed": passed,
        "energy_error_hartree": energy_error,
        "energy_tolerance_hartree": ENERGY_RESTART_TOLERANCE_HARTREE,
        "maximum_position_error_bohr": position_error,
        "position_tolerance_bohr": POSITION_RESTART_TOLERANCE_BOHR,
    }


def compare_continuation_reports(
    continued: Mapping[str, Any],
    cold: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare bounded accepted-step work with and without electronic reuse."""

    continued_steps = continued["result"]["steps"]
    cold_steps = cold["result"]["steps"]
    continued_iterations = sum(int(step["scf_iterations"]) for step in continued_steps)
    cold_iterations = sum(int(step["scf_iterations"]) for step in cold_steps)
    position_error = float(
        np.max(
            np.abs(
                np.asarray(continued["result"]["final_positions_bohr"])
                - np.asarray(cold["result"]["final_positions_bohr"])
            )
        )
    )
    comparable = bool(
        continued["result"]["status"] == cold["result"]["status"]
        and continued["result"]["accepted_step_count"] == cold["result"]["accepted_step_count"]
        and position_error <= POSITION_RESTART_TOLERANCE_BOHR
    )
    return {
        "passed": comparable and continued_iterations <= cold_iterations,
        "comparable": comparable,
        "accepted_step_count": len(continued_steps),
        "continued_scf_iterations": continued_iterations,
        "cold_scf_iterations": cold_iterations,
        "iteration_reduction": cold_iterations - continued_iterations,
        "maximum_position_error_bohr": position_error,
        "position_tolerance_bohr": POSITION_RESTART_TOLERANCE_BOHR,
        "continued_complete_wall_seconds": continued["runtime"]["complete_wall_seconds"],
        "cold_complete_wall_seconds": cold["runtime"]["complete_wall_seconds"],
        "wall_time_is_diagnostic": True,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the bounded Silicon relaxation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gth-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--checkpoint-to", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_silicon_relaxation(
        manifest_path=args.manifest,
        gth_source=args.gth_source,
        out=args.out,
        reuse_scf_state=not args.cold,
        checkpoint_to=args.checkpoint_to,
        checkpoint_step=args.checkpoint_step,
        resume_from=args.resume_from,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"status={payload['status']} "
            f"verified={payload['scientifically_verified']} "
            f"wall_seconds={payload['runtime']['complete_wall_seconds']:.6f}"
        )


if __name__ == "__main__":
    main()
