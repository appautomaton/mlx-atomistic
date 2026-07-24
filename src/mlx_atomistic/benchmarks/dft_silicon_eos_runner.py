"""Bounded execution and admission for the silicon equation of state."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.benchmarks.dft_silicon_eos import (
    EOS_REPORT_SCHEMA,
    HARTREE_TO_EV,
    REFERENCE_SHA256,
    compare_eos_convergence,
    compare_fit_to_reference,
    fit_cubic_silicon_eos,
    load_silicon_eos_references,
    reference_fit,
    validation_lattice_constants,
)

POINT_SCHEMA = "mlx-atomistic.silicon-eos-point.v1"
MEMORY_LIMIT_BYTES = 40_000_000_000
_LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS = {
    "135c531070bb09982ccc5dcc824baca699c5b9efe4ea41882286695d50d85dd7",
}

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "cutoff_hartree": 25.0,
        "fft_shape": [56, 56, 56],
        "kpoint_mesh": [6, 6, 6],
        "max_batch_transient_bytes": 512 * 1024 * 1024,
        "timeout_seconds": 180.0,
    },
    "cutoff": {
        "cutoff_hartree": 30.0,
        "fft_shape": [64, 64, 64],
        "kpoint_mesh": [6, 6, 6],
        "max_batch_transient_bytes": 768 * 1024 * 1024,
        "timeout_seconds": 180.0,
    },
    "kpoint": {
        "cutoff_hartree": 25.0,
        "fft_shape": [56, 56, 56],
        "kpoint_mesh": [8, 8, 8],
        "max_batch_transient_bytes": 512 * 1024 * 1024,
        "timeout_seconds": 240.0,
    },
    "combined": {
        "cutoff_hartree": 30.0,
        "fft_shape": [64, 64, 64],
        "kpoint_mesh": [8, 8, 8],
        "max_batch_transient_bytes": 768 * 1024 * 1024,
        "timeout_seconds": 240.0,
    },
}

_SCREEN_INDICES = (2, 3, 4)
_FULL_INDICES = tuple(range(7))
_KPOINT_SPOT_CHECK_THRESHOLD_MEV_PER_ATOM = 1.0


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.silicon-eos-point-execution.v1",
        "point_schema": POINT_SCHEMA,
        "profiles": PROFILE_SPECS,
        "scf_config_source": inspect.getsource(_scf_config),
        "point_execution_source": inspect.getsource(run_silicon_eos_point),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def _point_spec(
    *,
    workload_fingerprint: str,
    runtime_fingerprint: str,
    profile: str,
    volume_index: int,
    lattice_angstrom: float,
    implementation_fingerprint: str | None = None,
) -> dict[str, Any]:
    spec = {
        "workload_fingerprint": workload_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "eos_implementation_fingerprint": (
            _implementation_fingerprint()
            if implementation_fingerprint is None
            else implementation_fingerprint
        ),
        "reference_sha256": REFERENCE_SHA256,
        "profile": profile,
        "volume_index": volume_index,
        "lattice_constant_angstrom": lattice_angstrom,
        **PROFILE_SPECS[profile],
    }
    return {**spec, "point_fingerprint": sha256_bytes(canonical_json_bytes(spec))}


def _scf_config(
    manifest: Mapping[str, Any],
    *,
    max_batch_transient_bytes: int,
) -> Any:
    from mlx_atomistic.dft import PeriodicDavidsonConfig, PeriodicSCFConfig

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
        max_batch_transient_bytes=max_batch_transient_bytes,
        adaptive_eigensolver_tolerance=bool(scf.get("adaptive_eigensolver_tolerance", True)),
        initial_eigensolver_tolerance=float(scf.get("initial_eigensolver_tolerance", 1.0e-2)),
        eigensolver_tolerance_scale=float(scf.get("eigensolver_tolerance_scale", 0.1)),
        davidson=PeriodicDavidsonConfig(
            max_iterations=int(davidson["max_iterations"]),
            tolerance=float(davidson["tolerance"]),
            max_subspace_size=int(davidson["max_subspace_size"]),
            preconditioner_floor=float(davidson["preconditioner_floor"]),
        ),
    )


def run_silicon_eos_point(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    profile: str,
    volume_index: int,
    out: str | Path,
) -> dict[str, Any]:
    """Run one isolated silicon EOS geometry and persist a compact result."""

    import mlx.core as mx

    from mlx_atomistic.benchmarks.dft_runtime_contract import (
        build_source_fingerprints,
        load_workload,
    )
    from mlx_atomistic.dft import (
        MonkhorstPackGrid,
        PeriodicDFTSystem,
        read_gth,
        run_periodic_scf,
    )
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver

    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown silicon EOS profile: {profile}")
    if volume_index not in _FULL_INDICES:
        raise ValueError("silicon EOS volume_index must lie in [0, 6]")
    manifest, _resource = load_workload(manifest_path, gth_source=gth_source)
    references = load_silicon_eos_references()
    lattice_angstrom = validation_lattice_constants(references)[volume_index]
    spec = _point_spec(
        workload_fingerprint=str(manifest["workload_fingerprint"]),
        runtime_fingerprint=str(build_source_fingerprints()["runtime_fingerprint"]),
        profile=profile,
        volume_index=volume_index,
        lattice_angstrom=lattice_angstrom,
    )
    settings = PROFILE_SPECS[profile]
    system_values = manifest["system"]
    lattice_bohr = lattice_angstrom * ANGSTROM_TO_BOHR
    fractional = np.asarray(system_values["fractional_positions"], dtype=np.float64)
    system = PeriodicDFTSystem(
        (lattice_bohr, lattice_bohr, lattice_bohr),
        settings["fft_shape"],
        fractional * lattice_bohr,
        read_gth(gth_source, element="Si", name="GTH-PBE-q4"),
        electron_count=float(system_values["electron_count"]),
    )
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    result = run_periodic_scf(
        system,
        cutoff_hartree=float(settings["cutoff_hartree"]),
        kpoint_mesh=MonkhorstPackGrid(tuple(settings["kpoint_mesh"])),
        n_bands=int(system_values["occupied_band_count"]),
        config=_scf_config(
            manifest,
            max_batch_transient_bytes=int(settings["max_batch_transient_bytes"]),
        ),
        observer=observer,
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    electron_error = abs(float(result.electron_count) - float(system_values["electron_count"]))
    maximum_overlap = max(float(item.eigen.orthonormality_error) for item in result.owned_kpoints)
    maximum_residual = max(
        float(np.max(np.asarray(item.eigen.residuals))) for item in result.owned_kpoints
    )
    gates = manifest["numerical_gates"]
    numerical_passed = bool(
        result.converged
        and np.isfinite(result.total_energy)
        and electron_error <= float(gates["electron_count_abs_per_cell"])
        and maximum_overlap <= float(gates["orthonormality_max"])
        and maximum_residual <= float(manifest["solver"]["davidson"]["tolerance"])
    )
    observation = observer.snapshot()
    payload = {
        "schema_version": POINT_SCHEMA,
        "status": "ok" if numerical_passed else "failed",
        "numerical_passed": numerical_passed,
        "point": spec,
        "method": {
            "functional": manifest["physics"]["exchange_correlation"],
            "pseudopotential": manifest["physics"]["pseudopotential"],
            "occupation": "zero-temperature fixed occupations",
            "bands": int(system_values["occupied_band_count"]),
            "electrons": int(system_values["electron_count"]),
            "atoms": int(system_values["atom_count"]),
        },
        "result": {
            "total_energy_hartree": float(result.total_energy),
            "converged": bool(result.converged),
            "scf_iterations": int(result.iterations),
            "electron_count": float(result.electron_count),
            "electron_count_error": electron_error,
            "maximum_orbital_residual": maximum_residual,
            "maximum_orthonormality_error": maximum_overlap,
            "density_residual": float(result.density_residual),
            "energy_delta_hartree": (
                None if result.energy_delta is None else float(result.energy_delta)
            ),
            "explicit_kpoint_count": len(result.kpoints),
            "representative_kpoint_count": len(result.owned_kpoints),
            "elapsed_wall_seconds": elapsed,
            "timings_ms": dict(result.timings),
            "observation": {
                "total_elapsed_seconds": observation["total_elapsed_seconds"],
                "phase_seconds": observation["phase_seconds"],
                "work_counters": observation["work_counters"],
                "memory": observation["memory"],
            },
        },
    }
    output = Path(out)
    _write_atomic(output, payload)
    return payload


def _triad_assessment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3:
        return {"status": "blocked", "blocker": "central_triad_incomplete", "passed": False}
    ordered = sorted(rows, key=lambda row: int(row["point"]["volume_index"]))
    if [row["point"]["volume_index"] for row in ordered] != list(_SCREEN_INDICES):
        return {"status": "blocked", "blocker": "central_triad_indices_mismatch", "passed": False}
    numerical = all(row.get("numerical_passed") is True for row in ordered)
    energies = [float(row["result"]["total_energy_hartree"]) for row in ordered]
    convex = energies[1] < energies[0] and energies[1] < energies[2]
    return {
        "status": "ok" if numerical and convex else "failed",
        "passed": numerical and convex,
        "numerical_passed": numerical,
        "center_is_local_minimum": convex,
        "energies_hartree": energies,
    }


def _fit_profile(rows: Sequence[Mapping[str, Any]], atom_count: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["point"]["volume_index"]))
    if len(ordered) != 7 or [row["point"]["volume_index"] for row in ordered] != list(
        _FULL_INDICES
    ):
        return {"status": "blocked", "blocker": "seven_point_profile_incomplete"}
    if not all(row.get("numerical_passed") is True for row in ordered):
        return {"status": "blocked", "blocker": "profile_contains_failed_scf"}
    return fit_cubic_silicon_eos(
        [float(row["point"]["lattice_constant_angstrom"]) for row in ordered],
        [float(row["result"]["total_energy_hartree"]) for row in ordered],
        atom_count=atom_count,
    )


def _kpoint_spot_check(
    baseline_rows: Sequence[Mapping[str, Any]],
    kpoint_rows: Sequence[Mapping[str, Any]],
    *,
    atom_count: int,
) -> dict[str, Any]:
    """Compare the central 8-cubed energy shape with the admitted 6-cubed baseline."""

    baseline = sorted(baseline_rows, key=lambda row: int(row["point"]["volume_index"]))
    candidate = sorted(kpoint_rows, key=lambda row: int(row["point"]["volume_index"]))
    expected = list(_SCREEN_INDICES)
    if (
        [row["point"]["volume_index"] for row in baseline] != expected
        or [row["point"]["volume_index"] for row in candidate] != expected
        or not all(row.get("numerical_passed") is True for row in [*baseline, *candidate])
    ):
        return {
            "status": "blocked",
            "blocker": "central_kpoint_spot_check_incomplete",
            "passed": False,
        }
    baseline_energy = np.asarray([float(row["result"]["total_energy_hartree"]) for row in baseline])
    candidate_energy = np.asarray(
        [float(row["result"]["total_energy_hartree"]) for row in candidate]
    )
    baseline_shape = baseline_energy - baseline_energy[1]
    candidate_shape = candidate_energy - candidate_energy[1]
    maximum_change = float(
        np.max(np.abs(candidate_shape - baseline_shape)) * HARTREE_TO_EV * 1000.0 / atom_count
    )
    passed = maximum_change <= _KPOINT_SPOT_CHECK_THRESHOLD_MEV_PER_ATOM
    return {
        "status": "ok" if passed else "failed",
        "scope": "central_three_volume_energy_shape",
        "passed": passed,
        "metrics": {"curve_max_mev_per_atom": maximum_change},
        "thresholds": {"curve_max_mev_per_atom": _KPOINT_SPOT_CHECK_THRESHOLD_MEV_PER_ATOM},
    }


def build_silicon_eos_report(
    *,
    manifest: Mapping[str, Any],
    level: str,
    point_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fail-closed scientific report from persisted point artifacts."""

    if level not in {"screen", "admission"}:
        raise ValueError("silicon EOS validation level must be screen or admission")
    references = load_silicon_eos_references()
    grouped = {
        profile: [row for row in point_reports if row["point"]["profile"] == profile]
        for profile in PROFILE_SPECS
    }
    triads = {
        profile: _triad_assessment(
            [row for row in rows if int(row["point"]["volume_index"]) in _SCREEN_INDICES]
        )
        for profile, rows in grouped.items()
        if rows
    }
    atom_count = int(manifest["system"]["atom_count"])
    fits = {
        profile: _fit_profile(rows, atom_count)
        for profile, rows in grouped.items()
        if len(rows) == 7
    }
    primary_reference = reference_fit(references["references"]["all_electron_average"])
    scientific = (
        compare_fit_to_reference(fits.get("baseline", {}), primary_reference)
        if level == "admission"
        else {
            "status": "blocked",
            "blocker": "screen_level_does_not_fit_or_certify_an_eos",
            "verified": False,
            "excellent": False,
        }
    )
    convergence = (
        {
            "cutoff": compare_eos_convergence(
                fits.get("baseline", {}),
                fits.get("cutoff", {}),
            ),
            "kpoint_spot_check": _kpoint_spot_check(
                [
                    row
                    for row in grouped["baseline"]
                    if int(row["point"]["volume_index"]) in _SCREEN_INDICES
                ],
                [
                    row
                    for row in grouped["kpoint"]
                    if int(row["point"]["volume_index"]) in _SCREEN_INDICES
                ],
                atom_count=atom_count,
            ),
        }
        if level == "admission"
        else {}
    )
    blockers: list[str] = []
    required_triads = ("baseline",) if level == "screen" else ("baseline", "cutoff", "kpoint")
    for profile in required_triads:
        if not triads.get(profile, {}).get("passed", False):
            blockers.append(f"central_triad_failed:{profile}")
    if level == "admission":
        for profile in ("baseline", "cutoff"):
            if fits.get(profile, {}).get("status") != "ok":
                blockers.append(f"eos_fit_failed:{profile}")
        if scientific.get("verified") is not True:
            blockers.append("all_electron_reference_thresholds_failed")
        if convergence["cutoff"].get("passed") is not True:
            blockers.append("convergence_failed:cutoff")
        if convergence["kpoint_spot_check"].get("passed") is not True:
            blockers.append("kpoint_spot_check_failed")
    passed = not blockers
    return {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "level": level,
        "admitted": passed and level == "admission",
        "blockers": blockers,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "runtime_fingerprints": sorted(
            {
                str(row["point"]["runtime_fingerprint"])
                for row in point_reports
                if row["point"].get("runtime_fingerprint") is not None
            }
        ),
        "eos_implementation_fingerprints": sorted(
            {
                str(row["point"]["eos_implementation_fingerprint"])
                for row in point_reports
                if row["point"].get("eos_implementation_fingerprint") is not None
            }
        ),
        "method_scope": {
            "claim": "eight-atom diamond-silicon PBE/GTH EOS with fixed occupations",
            "limitation": references["protocol"]["method_difference"],
            "not_certified": [
                "exact ACWF smearing/free-energy protocol parity",
                "forces",
                "stress",
                "band structure",
                "materials beyond diamond silicon",
                "full seven-point k-point convergence curve",
                "combined cutoff/k-point interaction stress profile",
            ],
        },
        "reference_bundle": {
            "sha256": REFERENCE_SHA256,
            "license": references["license"],
            "primary": references["references"]["all_electron_average"],
            "same_pseudopotential_family": references["references"]["cp2k_gth"],
            "context": {
                "qe_sssp": references["references"]["qe_sssp"],
                "experiment": references["references"]["experiment"],
            },
        },
        "profiles": {
            profile: {
                "settings": PROFILE_SPECS[profile],
                "point_count": len(grouped[profile]),
                "triad": triads.get(profile),
                "fit": fits.get(profile),
            }
            for profile in PROFILE_SPECS
        },
        "scientific_comparison": scientific,
        "numerical_convergence": convergence,
        "point_fingerprints": sorted(
            str(row["point"]["point_fingerprint"]) for row in point_reports
        ),
    }


def _required_points(
    level: str,
    *,
    include_combined: bool,
) -> list[tuple[str, int]]:
    if level == "screen":
        return [("baseline", index) for index in _SCREEN_INDICES]
    if level != "admission":
        raise ValueError("silicon EOS validation level must be screen or admission")
    independent_curves = [
        (profile, index) for profile in ("baseline", "cutoff") for index in _FULL_INDICES
    ]
    kpoint_spot_check = [("kpoint", index) for index in _SCREEN_INDICES]
    combined = [("combined", index) for index in _SCREEN_INDICES] if include_combined else []
    return independent_curves + kpoint_spot_check + combined


def _load_matching_point(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    point = payload.get("point", {})
    if payload.get("schema_version") != POINT_SCHEMA or not isinstance(point, dict):
        raise ValueError(f"refusing mismatched silicon EOS point artifact: {path}")
    if point.get("point_fingerprint") == expected["point_fingerprint"]:
        return payload
    identity_keys = set(expected).difference(
        {"eos_implementation_fingerprint", "point_fingerprint"}
    )
    same_science_identity = all(point.get(key) == expected[key] for key in identity_keys)
    if same_science_identity:
        for implementation_fingerprint in _LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS:
            legacy_spec = {
                key: value
                for key, value in expected.items()
                if key not in {"eos_implementation_fingerprint", "point_fingerprint"}
            }
            legacy_spec["eos_implementation_fingerprint"] = implementation_fingerprint
            legacy_fingerprint = sha256_bytes(canonical_json_bytes(legacy_spec))
            if point.get("point_fingerprint") == legacy_fingerprint:
                return payload
    raise ValueError(f"refusing mismatched silicon EOS point artifact: {path}")


def _summarize_existing_points(
    *,
    output: Path,
    manifest: Mapping[str, Any],
    level: str,
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    bounded_failures: list[dict[str, Any]] = []
    for spec in plan:
        profile = str(spec["profile"])
        index = int(spec["volume_index"])
        point_root = output / "points" / profile / f"v{index}"
        report = _load_matching_point(point_root / "report.json", spec)
        if report is not None:
            reports.append(report)
            continue
        missing.append({"profile": profile, "volume_index": index})
        trace_path = point_root / "memory.json"
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text())
            bounded_failures.append(
                {
                    "profile": profile,
                    "volume_index": index,
                    "timed_out": trace.get("bounded_process_timed_out"),
                    "memory_exceeded": trace.get("bounded_process_exceeded"),
                    "peak_physical_bytes": trace.get("bounded_process_peak_physical_bytes"),
                    "memory_plateau_passed": trace.get("memory_trace_summary", {}).get(
                        "plateau_passed"
                    ),
                    "trace": str(trace_path),
                }
            )
    payload = build_silicon_eos_report(
        manifest=manifest,
        level=level,
        point_reports=reports,
    )
    missing_blockers = [f"missing_point:{row['profile']}:v{row['volume_index']}" for row in missing]
    timeout_blockers = [
        f"point_timeout:{row['profile']}:v{row['volume_index']}"
        for row in bounded_failures
        if row["timed_out"] is True
    ]
    blockers = sorted(set([*payload["blockers"], *missing_blockers, *timeout_blockers]))
    payload.update(
        {
            "status": "passed" if not blockers else "failed",
            "admitted": not blockers and level == "admission",
            "blockers": blockers,
            "evidence_status": "complete" if not missing else "partial",
            "completed_point_count": len(reports),
            "missing_points": missing,
            "bounded_failures": bounded_failures,
        }
    )
    _write_atomic(output / "report.json", payload)
    return payload


def run_silicon_eos_validation(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    level: str,
    dry_run: bool = False,
    include_combined: bool = False,
    summarize_only: bool = False,
) -> dict[str, Any]:
    """Execute or describe the bounded silicon EOS validation ladder."""

    from mlx_atomistic.benchmarks.dft_runtime_contract import (
        build_source_fingerprints,
        load_workload,
    )

    manifest, _resource = load_workload(manifest_path, gth_source=gth_source)
    runtime_fingerprint = str(build_source_fingerprints()["runtime_fingerprint"])
    output = Path(out)
    lattice = validation_lattice_constants()
    required = _required_points(level, include_combined=include_combined)
    plan = [
        _point_spec(
            workload_fingerprint=str(manifest["workload_fingerprint"]),
            runtime_fingerprint=runtime_fingerprint,
            profile=profile,
            volume_index=index,
            lattice_angstrom=lattice[index],
        )
        for profile, index in required
    ]
    if dry_run and summarize_only:
        raise ValueError("--dry-run and --summarize-only are mutually exclusive")
    if dry_run:
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": "planned",
            "level": level,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "include_combined": include_combined,
            "point_count": len(plan),
            "points": plan,
        }
        _write_atomic(output / "plan.json", payload)
        return payload
    if summarize_only:
        return _summarize_existing_points(
            output=output,
            manifest=manifest,
            level=level,
            plan=plan,
        )

    reports: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(plan):
        profile = str(spec["profile"])
        index = int(spec["volume_index"])
        point_root = output / "points" / profile / f"v{index}"
        point_path = point_root / "report.json"
        existing = _load_matching_point(point_path, spec)
        if existing is not None:
            reports.append(existing)
            continue
        point_root.mkdir(parents=True, exist_ok=True)
        trace_path = point_root / "memory.json"
        command = [
            sys.executable,
            "scripts/run_bounded_process.py",
            "--max-bytes",
            str(MEMORY_LIMIT_BYTES),
            "--poll-seconds",
            "0.25",
            "--timeout-seconds",
            str(spec["timeout_seconds"]),
            "--trace-out",
            str(trace_path),
            "--",
            sys.executable,
            "-m",
            "mlx_atomistic.benchmarks.dft_silicon",
            "eos-point",
            "--manifest",
            str(manifest_path),
            "--gth-source",
            str(gth_source),
            "--profile",
            profile,
            "--volume-index",
            str(index),
            "--out",
            str(point_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        (point_root / "stdout.txt").write_text(completed.stdout)
        (point_root / "stderr.txt").write_text(completed.stderr)
        if completed.returncode != 0 or not point_path.is_file():
            failure = {
                "schema_version": EOS_REPORT_SCHEMA,
                "status": "failed",
                "level": level,
                "admitted": False,
                "blockers": [f"point_execution_failed:{profile}:v{index}"],
                "failed_point_ordinal": ordinal,
                "failed_point": spec,
                "returncode": completed.returncode,
                "memory_trace": str(trace_path),
            }
            _write_atomic(output / "report.json", failure)
            return failure
        report = _load_matching_point(point_path, spec)
        if report is None or report.get("numerical_passed") is not True:
            failure = {
                "schema_version": EOS_REPORT_SCHEMA,
                "status": "failed",
                "level": level,
                "admitted": False,
                "blockers": [f"point_numerical_failure:{profile}:v{index}"],
                "failed_point_ordinal": ordinal,
                "failed_point": spec,
            }
            _write_atomic(output / "report.json", failure)
            return failure
        reports.append(report)
        if index == _SCREEN_INDICES[-1]:
            profile_triad = [
                row
                for row in reports
                if row["point"]["profile"] == profile
                and int(row["point"]["volume_index"]) in _SCREEN_INDICES
            ]
            if not _triad_assessment(profile_triad)["passed"]:
                failure = build_silicon_eos_report(
                    manifest=manifest,
                    level=level,
                    point_reports=reports,
                )
                _write_atomic(output / "report.json", failure)
                return failure

    payload = build_silicon_eos_report(
        manifest=manifest,
        level=level,
        point_reports=reports,
    )
    _write_atomic(output / "report.json", payload)
    return payload
