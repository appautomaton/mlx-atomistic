"""Bounded cutoff, k-point, and EOS validation for rock-salt MgO."""

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
from mlx_atomistic.benchmarks.dft_mgo import load_mgo_workload
from mlx_atomistic.benchmarks.dft_mgo_eos import (
    EOS_REPORT_SCHEMA,
    HARTREE_TO_EV,
    REFERENCE_SHA256,
    compare_fit_to_reference,
    fit_cubic_mgo_eos,
    load_mgo_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR

POINT_SCHEMA = "mlx-atomistic.mgo-eos-point.v1"
MEMORY_LIMIT_BYTES = 40_000_000_000
POINT_TIMEOUT_SECONDS = 1800.0
SCREEN_INDICES = (2, 3, 4)
FULL_INDICES = tuple(range(7))
KPOINT_SHAPE_THRESHOLD_MEV_PER_ATOM = 1.0
_FFT_SHAPES = {
    25: 40,
    30: 44,
    40: 48,
    50: 56,
    60: 64,
    70: 68,
    80: 72,
}

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "smoke-q2": {
        "pseudopotential_mode": "q2",
        "cutoff_hartree": 25.0,
        "fft_shape": [40, 40, 40],
        "kpoint_mesh": [2, 2, 2],
        "max_batch_transient_bytes": 768 * 1024**2,
    },
    **{
        f"q2-c{cutoff}-k{mesh}": {
            "pseudopotential_mode": "q2",
            "cutoff_hartree": float(cutoff),
            "fft_shape": [_FFT_SHAPES[cutoff]] * 3,
            "kpoint_mesh": [mesh] * 3,
            "max_batch_transient_bytes": (
                1024 * 1024**2 if cutoff <= 50 else 1536 * 1024**2
            ),
        }
        for cutoff in _FFT_SHAPES
        for mesh in (4, 6)
    },
    **{
        f"seed-q2-c{cutoff}": {
            "pseudopotential_mode": "q2",
            "cutoff_hartree": float(cutoff),
            "fft_shape": [_FFT_SHAPES[cutoff]] * 3,
            "kpoint_mesh": [2, 2, 2],
            "max_batch_transient_bytes": 768 * 1024**2,
        }
        for cutoff in _FFT_SHAPES
        if cutoff != 25
    },
    "q10-feasibility": {
        "pseudopotential_mode": "q10",
        "cutoff_hartree": 40.0,
        "fft_shape": [48, 48, 48],
        "kpoint_mesh": [4, 4, 4],
        "max_batch_transient_bytes": 1024 * 1024**2,
    },
}


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.mgo-eos-point-execution.v1",
        "point_schema": POINT_SCHEMA,
        "profiles": PROFILE_SPECS,
        "scf_config_source": inspect.getsource(_scf_config),
        "point_execution_source": inspect.getsource(run_mgo_eos_point),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def _point_spec(
    *,
    workload_fingerprint: str,
    profile: str,
    volume_index: int,
    initial_density_sha256: str | None = None,
) -> dict[str, Any]:
    values = {
        "workload_fingerprint": workload_fingerprint,
        "eos_implementation_fingerprint": _implementation_fingerprint(),
        "reference_sha256": REFERENCE_SHA256,
        "profile": profile,
        "volume_index": volume_index,
        "lattice_constant_angstrom": validation_lattice_constants()[volume_index],
        "initial_density_sha256": initial_density_sha256,
        "timeout_seconds": POINT_TIMEOUT_SECONDS,
        **PROFILE_SPECS[profile],
    }
    return {
        **values,
        "point_fingerprint": sha256_bytes(canonical_json_bytes(values)),
    }


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


def run_mgo_eos_point(
    *,
    manifest_path: str | Path,
    profile: str,
    volume_index: int,
    out: str | Path,
    initial_density_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one isolated rock-salt MgO EOS point and persist compact evidence."""

    import mlx.core as mx

    from mlx_atomistic.dft import (
        MonkhorstPackGrid,
        PeriodicDFTSystem,
        read_gth,
        run_periodic_scf,
    )
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver

    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown MgO EOS profile: {profile}")
    if volume_index not in FULL_INDICES:
        raise ValueError("MgO EOS volume_index must lie in [0, 6]")
    manifest, resources = load_mgo_workload(manifest_path)
    settings = PROFILE_SPECS[profile]
    system_values = manifest["system"]
    mode = str(settings["pseudopotential_mode"])
    magnesium_id = "mg_q2" if mode == "q2" else "mg_q10"
    magnesium = read_gth(
        resources[magnesium_id],
        element="Mg",
        name="GTH-PBE-q2" if mode == "q2" else "GTH-PBE-q10",
    )
    oxygen = read_gth(resources["o_q6"], element="O", name="GTH-PBE-q6")
    pseudopotentials = (magnesium,) * 4 + (oxygen,) * 4
    electron_key = f"{mode}_electron_count"
    band_key = f"{mode}_occupied_band_count"

    initial_density = None
    initial_density_sha256 = None
    if initial_density_path is not None:
        density_path = Path(initial_density_path)
        if density_path.is_symlink() or not density_path.is_file():
            raise ValueError("initial density must be a regular existing file")
        density_bytes = density_path.read_bytes()
        initial_density_sha256 = sha256_bytes(density_bytes)
        initial_density = np.load(density_path, allow_pickle=False)
    spec = _point_spec(
        workload_fingerprint=str(manifest["workload_fingerprint"]),
        profile=profile,
        volume_index=volume_index,
        initial_density_sha256=initial_density_sha256,
    )
    lattice_angstrom = float(spec["lattice_constant_angstrom"])
    lattice_bohr = lattice_angstrom * ANGSTROM_TO_BOHR
    fractional = np.asarray(system_values["fractional_positions"], dtype=np.float64)
    system = PeriodicDFTSystem(
        (lattice_bohr, lattice_bohr, lattice_bohr),
        settings["fft_shape"],
        fractional * lattice_bohr,
        electron_count=float(system_values[electron_key]),
        pseudopotentials=pseudopotentials,
    )
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    result = run_periodic_scf(
        system,
        cutoff_hartree=float(settings["cutoff_hartree"]),
        kpoint_mesh=MonkhorstPackGrid(tuple(settings["kpoint_mesh"])),
        n_bands=int(system_values[band_key]),
        config=_scf_config(
            manifest,
            max_batch_transient_bytes=int(settings["max_batch_transient_bytes"]),
        ),
        observer=observer,
        initial_density=initial_density,
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    electron_error = abs(
        float(result.electron_count) - float(system_values[electron_key])
    )
    maximum_overlap = max(
        float(item.eigen.orthonormality_error) for item in result.owned_kpoints
    )
    maximum_residual = max(
        float(np.max(np.asarray(item.eigen.residuals)))
        for item in result.owned_kpoints
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
    density_output = Path(out).with_name("density.npy")
    density_temporary = density_output.with_name(f".{density_output.name}.tmp")
    density_output.parent.mkdir(parents=True, exist_ok=True)
    with density_temporary.open("wb") as handle:
        np.save(handle, np.asarray(result.density), allow_pickle=False)
    density_temporary.replace(density_output)
    density_sha256 = sha256_bytes(density_output.read_bytes())
    payload = {
        "schema_version": POINT_SCHEMA,
        "status": "ok" if numerical_passed else "failed",
        "numerical_passed": numerical_passed,
        "point": spec,
        "method": {
            "functional": manifest["physics"]["exchange_correlation"],
            "pseudopotential_mode": mode,
            "pseudopotentials": manifest["physics"][
                (
                    "accepted_pseudopotentials"
                    if mode == "q2"
                    else "high_accuracy_feasibility_pseudopotentials"
                )
            ],
            "occupation": manifest["physics"]["occupation"],
            "atoms": int(system_values["atom_count"]),
            "electrons": int(system_values[electron_key]),
            "bands": int(system_values[band_key]),
            "symbols": list(system.symbols),
            "system_fingerprint": system.fingerprint,
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
            "density_artifact": {
                "path": density_output.name,
                "sha256": density_sha256,
                "shape": list(result.density.shape),
            },
        },
    }
    _write_atomic(Path(out), payload)
    return payload


def _load_matching_point(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != POINT_SCHEMA
        or payload.get("point", {}).get("point_fingerprint")
        != expected["point_fingerprint"]
    ):
        raise ValueError(f"refusing mismatched MgO EOS point artifact: {path}")
    return payload


def _run_bounded_point(
    *,
    manifest_path: Path,
    output: Path,
    spec: Mapping[str, Any],
    initial_density_path: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    profile = str(spec["profile"])
    index = int(spec["volume_index"])
    point_root = output / "points" / profile / f"v{index}"
    report_path = point_root / "report.json"
    existing = _load_matching_point(report_path, spec)
    if existing is not None:
        failure = (
            None
            if existing.get("numerical_passed") is True
            else {"blocker": f"existing_point_numerical_failure:{profile}:v{index}"}
        )
        return existing, failure
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
        str(POINT_TIMEOUT_SECONDS),
        "--trace-out",
        str(trace_path),
        "--",
        sys.executable,
        "-m",
        "mlx_atomistic.benchmarks.dft_mgo",
        "eos-point",
        "--manifest",
        str(manifest_path),
        "--profile",
        profile,
        "--volume-index",
        str(index),
        "--out",
        str(report_path),
    ]
    if initial_density_path is not None:
        command.extend(["--initial-density", str(initial_density_path)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (point_root / "stdout.txt").write_text(completed.stdout)
    (point_root / "stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0 or not report_path.is_file():
        trace = json.loads(trace_path.read_text()) if trace_path.is_file() else {}
        return None, {
            "blocker": f"point_execution_failed:{profile}:v{index}",
            "returncode": completed.returncode,
            "timed_out": trace.get("bounded_process_timed_out"),
            "memory_exceeded": trace.get("bounded_process_exceeded"),
            "peak_physical_bytes": trace.get(
                "bounded_process_peak_physical_bytes"
            ),
            "stderr": str(point_root / "stderr.txt"),
        }
    report = _load_matching_point(report_path, spec)
    if report is None or report.get("numerical_passed") is not True:
        return report, {"blocker": f"point_numerical_failure:{profile}:v{index}"}
    return report, None


def _plan_spec(
    manifest: Mapping[str, Any],
    profile: str,
    index: int,
    initial_density_path: Path | None = None,
) -> dict[str, Any]:
    density_sha256 = (
        None
        if initial_density_path is None
        else sha256_bytes(initial_density_path.read_bytes())
    )
    return _point_spec(
        workload_fingerprint=str(manifest["workload_fingerprint"]),
        profile=profile,
        volume_index=index,
        initial_density_sha256=density_sha256,
    )


def _ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    profile: str,
    indices: Sequence[int],
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["point"]["profile"] == profile
        and int(row["point"]["volume_index"]) in indices
    ]
    return sorted(selected, key=lambda row: int(row["point"]["volume_index"]))


def _shape_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = _ordered_rows(baseline_rows, str(baseline_rows[0]["point"]["profile"]), SCREEN_INDICES)
    second = _ordered_rows(
        candidate_rows,
        str(candidate_rows[0]["point"]["profile"]),
        SCREEN_INDICES,
    )
    if (
        len(first) != 3
        or len(second) != 3
        or not all(row.get("numerical_passed") is True for row in [*first, *second])
    ):
        return {
            "status": "blocked",
            "blocker": "kpoint_shape_comparison_incomplete",
            "passed": False,
        }
    first_energy = np.asarray(
        [float(row["result"]["total_energy_hartree"]) for row in first]
    )
    second_energy = np.asarray(
        [float(row["result"]["total_energy_hartree"]) for row in second]
    )
    maximum = float(
        np.max(
            np.abs(
                (second_energy - second_energy[1])
                - (first_energy - first_energy[1])
            )
        )
        * HARTREE_TO_EV
        * 1000.0
        / 8.0
    )
    return {
        "status": (
            "ok" if maximum <= KPOINT_SHAPE_THRESHOLD_MEV_PER_ATOM else "failed"
        ),
        "passed": maximum <= KPOINT_SHAPE_THRESHOLD_MEV_PER_ATOM,
        "metrics": {"curve_max_mev_per_atom": maximum},
        "thresholds": {
            "curve_max_mev_per_atom": KPOINT_SHAPE_THRESHOLD_MEV_PER_ATOM
        },
    }


def _existing_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "points").glob("*/v*/report.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") == POINT_SCHEMA:
            rows.append(payload)
    return rows


def _execute_point(
    *,
    manifest_file: Path,
    manifest: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    profile: str,
    index: int,
    initial_density_path: Path | None = None,
) -> dict[str, Any] | None:
    existing = _ordered_rows(rows, profile, (index,))
    if existing:
        if existing[0].get("numerical_passed") is True:
            return None
        return {"blocker": f"existing_point_numerical_failure:{profile}:v{index}"}
    spec = _plan_spec(
        manifest,
        profile,
        index,
        initial_density_path=initial_density_path,
    )
    report, failure = _run_bounded_point(
        manifest_path=manifest_file,
        output=output,
        spec=spec,
        initial_density_path=initial_density_path,
    )
    if failure is not None:
        return failure
    if report is None:
        return {"blocker": f"point_artifact_missing:{profile}:v{index}"}
    rows.append(report)
    return None


def _failure_report(
    output: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": "failed",
        "admitted": False,
        "validation_complete": False,
        "core_properties_validated": False,
        "strict_reference_gate_passed": False,
        "scientifically_verified": False,
        "blockers": [str(detail["blocker"])],
        "completed_point_count": len(rows),
        "detail": dict(detail),
    }
    _write_atomic(output / "report.json", payload)
    return payload


def _completion_assessment(
    selected_fit: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify strict verification separately from an accepted B-prime residual."""

    metrics = scientific.get("metrics", {})
    thresholds = scientific.get("verified_thresholds", {})
    strict_passed = (
        selected_fit.get("status") == "ok"
        and scientific.get("verified") is True
    )
    failed_metrics = sorted(
        key
        for key, limit in thresholds.items()
        if key not in metrics or float(metrics[key]) > float(limit)
    )
    core_metric_names = (
        "delta_mev_per_atom",
        "lattice_relative",
        "bulk_modulus_relative",
    )
    core_properties_validated = (
        selected_fit.get("status") == "ok"
        and all(
            key in metrics
            and key in thresholds
            and float(metrics[key]) <= float(thresholds[key])
            for key in core_metric_names
        )
    )
    accepted_bprime_residual = (
        core_properties_validated
        and failed_metrics == ["bulk_derivative_relative"]
    )
    validation_complete = strict_passed or accepted_bprime_residual
    deviations: list[dict[str, Any]] = []
    if accepted_bprime_residual:
        deviations.append(
            {
                "property": "bulk_derivative",
                "candidate": float(selected_fit["bulk_derivative"]),
                "relative_error": float(metrics["bulk_derivative_relative"]),
                "strict_threshold": float(
                    thresholds["bulk_derivative_relative"]
                ),
                "status": "outside_strict_reference_gate",
                "likely_attribution": (
                    "Mg GTH-PBE-q2 pseudopotential transferability"
                ),
                "interpretation": (
                    "Known residual scientific deviation; the strict gate remains "
                    "failed and no Mg-q10 retry is required for this validation."
                ),
            }
        )
    if strict_passed:
        status = "passed"
    elif validation_complete:
        status = "complete_with_known_deviation"
    else:
        status = "failed"
    return {
        "status": status,
        "admitted": strict_passed,
        "validation_complete": validation_complete,
        "core_properties_validated": core_properties_validated,
        "strict_reference_gate_passed": strict_passed,
        "scientifically_verified": strict_passed,
        "failed_strict_metrics": failed_metrics,
        "known_residual_deviations": deviations,
        "blockers": (
            []
            if validation_complete
            else ["all_electron_reference_thresholds_failed"]
        ),
    }


def run_mgo_eos_validation(
    *,
    manifest_path: str | Path,
    out: str | Path,
    dry_run: bool = False,
    summarize_only: bool = False,
) -> dict[str, Any]:
    """Run the fail-early multi-element MgO validation ladder."""

    if dry_run and summarize_only:
        raise ValueError("--dry-run and --summarize-only are mutually exclusive")
    manifest_file = Path(manifest_path).resolve()
    manifest, _resources = load_mgo_workload(manifest_file)
    output = Path(out)
    if dry_run:
        cutoff_profiles = [f"q2-c{cutoff}-k4" for cutoff in _FFT_SHAPES]
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": "planned",
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "point_timeout_seconds": POINT_TIMEOUT_SECONDS,
            "maximum_point_count": 35,
            "initial_smoke_point": _plan_spec(manifest, "smoke-q2", 3),
            "cutoff_screen_points": [
                _plan_spec(manifest, profile, 3) for profile in cutoff_profiles
            ],
            "decision_ladder": [
                "run the Mg-q2/O-q6 2x2x2 equilibrium smoke point",
                "build one same-grid 2x2x2 density seed before each higher cutoff",
                (
                    "screen adjacent Mg-q2/O-q6 cutoffs with three-volume "
                    "4x4x4 energy shapes"
                ),
                "compare 4x4x4 and 6x6x6 three-volume energy shapes",
                "complete only the accepted 6x6x6 seven-point curve",
                "compare the fit with ACWF all-electron and CP2K/GTH references",
            ],
        }
        _write_atomic(output / "plan.json", payload)
        return payload

    rows = _existing_rows(output)
    if summarize_only:
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        admitted = report.get("admitted") is True
        validation_complete = (
            report.get("validation_complete") is True or admitted
        )
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": (
                str(report.get("status", "complete"))
                if validation_complete
                else "partial"
            ),
            "admitted": admitted,
            "validation_complete": validation_complete,
            "core_properties_validated": (
                report.get("core_properties_validated") is True or admitted
            ),
            "strict_reference_gate_passed": admitted,
            "scientifically_verified": report.get("scientifically_verified") is True,
            "evidence_status": "complete" if validation_complete else "partial",
            "completed_point_count": len(rows),
        }
        _write_atomic(output / "summary.json", payload)
        return payload

    failure = _execute_point(
        manifest_file=manifest_file,
        manifest=manifest,
        output=output,
        rows=rows,
        profile="smoke-q2",
        index=3,
    )
    if failure is not None:
        return _failure_report(output, rows=rows, detail=failure)

    cutoff_density_seeds = {
        25: output / "points" / "smoke-q2" / "v3" / "density.npy"
    }
    accepted_cutoff: int | None = None
    cutoff_evidence: dict[str, Any] | None = None
    previous_profile: str | None = None
    for cutoff in _FFT_SHAPES:
        if cutoff != 25:
            seed_profile = f"seed-q2-c{cutoff}"
            failure = _execute_point(
                manifest_file=manifest_file,
                manifest=manifest,
                output=output,
                rows=rows,
                profile=seed_profile,
                index=3,
            )
            if failure is not None:
                return _failure_report(output, rows=rows, detail=failure)
            cutoff_density_seeds[cutoff] = (
                output / "points" / seed_profile / "v3" / "density.npy"
            )
        density_seed = cutoff_density_seeds[cutoff]
        if not density_seed.is_file():
            return _failure_report(
                output,
                rows=rows,
                detail={"blocker": f"cutoff_density_seed_missing:{cutoff}"},
            )
        profile = f"q2-c{cutoff}-k4"
        failure = _execute_point(
            manifest_file=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=profile,
            index=3,
            initial_density_path=density_seed,
        )
        if failure is not None:
            return _failure_report(output, rows=rows, detail=failure)
        if previous_profile is not None:
            for candidate_profile in (previous_profile, profile):
                center_density = (
                    output
                    / "points"
                    / candidate_profile
                    / "v3"
                    / "density.npy"
                )
                for index in (2, 4):
                    failure = _execute_point(
                        manifest_file=manifest_file,
                        manifest=manifest,
                        output=output,
                        rows=rows,
                        profile=candidate_profile,
                        index=index,
                        initial_density_path=center_density,
                    )
                    if failure is not None:
                        return _failure_report(output, rows=rows, detail=failure)
            shape = _shape_comparison(
                _ordered_rows(rows, previous_profile, SCREEN_INDICES),
                _ordered_rows(rows, profile, SCREEN_INDICES),
            )
            if shape.get("passed") is True:
                accepted_cutoff = int(
                    PROFILE_SPECS[previous_profile]["cutoff_hartree"]
                )
                cutoff_evidence = {
                    **shape,
                    "scope": "adjacent_cutoff_three_volume_energy_shape",
                    "profiles": [previous_profile, profile],
                }
                break
        previous_profile = profile
    if accepted_cutoff is None or cutoff_evidence is None:
        return _failure_report(
            output,
            rows=rows,
            detail={"blocker": "no_q2_cutoff_pair_converged_through_80_hartree"},
        )

    profile4 = f"q2-c{accepted_cutoff}-k4"
    profile6 = f"q2-c{accepted_cutoff}-k6"
    for profile, indices in (
        (profile4, SCREEN_INDICES),
        (profile6, SCREEN_INDICES),
    ):
        for index in indices:
            if profile == profile4:
                seed = output / "points" / profile4 / "v3" / "density.npy"
            else:
                seed = output / "points" / profile4 / f"v{index}" / "density.npy"
            if not seed.is_file():
                return _failure_report(
                    output,
                    rows=rows,
                    detail={
                        "blocker": f"kpoint_density_seed_missing:{profile}:v{index}"
                    },
                )
            failure = _execute_point(
                manifest_file=manifest_file,
                manifest=manifest,
                output=output,
                rows=rows,
                profile=profile,
                index=index,
                initial_density_path=None if index == 3 and profile == profile4 else seed,
            )
            if failure is not None:
                return _failure_report(output, rows=rows, detail=failure)
    kpoint = _shape_comparison(
        _ordered_rows(rows, profile4, SCREEN_INDICES),
        _ordered_rows(rows, profile6, SCREEN_INDICES),
    )
    if kpoint.get("passed") is not True:
        return _failure_report(
            output,
            rows=rows,
            detail={"blocker": "six_cubed_kpoint_shape_not_converged", **kpoint},
        )

    density_path = output / "points" / profile6 / "v3" / "density.npy"
    if not density_path.is_file():
        return _failure_report(
            output,
            rows=rows,
            detail={"blocker": "accepted_equilibrium_density_missing"},
        )
    for index in FULL_INDICES:
        failure = _execute_point(
            manifest_file=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=profile6,
            index=index,
            initial_density_path=None if index == 3 else density_path,
        )
        if failure is not None:
            return _failure_report(output, rows=rows, detail=failure)
    full_rows = _ordered_rows(rows, profile6, FULL_INDICES)
    selected_fit = fit_cubic_mgo_eos(
        [float(row["point"]["lattice_constant_angstrom"]) for row in full_rows],
        [float(row["result"]["total_energy_hartree"]) for row in full_rows],
    )
    references = load_mgo_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    scientific = compare_fit_to_reference(selected_fit, primary)
    completion = _completion_assessment(selected_fit, scientific)
    payload = {
        "schema_version": EOS_REPORT_SCHEMA,
        **completion,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "accepted_workload": {
            "profile": profile6,
            "cutoff_hartree": accepted_cutoff,
            "fft_shape": PROFILE_SPECS[profile6]["fft_shape"],
            "kpoint_mesh": [6, 6, 6],
            "pseudopotentials": manifest["physics"][
                "accepted_pseudopotentials"
            ],
            "volume_point_count": 7,
        },
        "selected_fit": selected_fit,
        "cutoff_screen": cutoff_evidence,
        "kpoint_screen": kpoint,
        "scientific_comparison": scientific,
        "reference_bundle": {
            "sha256": REFERENCE_SHA256,
            "license": references["license"],
            "primary": references["references"]["all_electron_average"],
            "gth_family_context": references["references"]["cp2k_gth"],
            "context": {"qe_sssp": references["references"]["qe_sssp"]},
        },
        "completed_point_count": len(rows),
        "point_fingerprints": sorted(
            str(row["point"]["point_fingerprint"]) for row in rows
        ),
    }
    _write_atomic(output / "report.json", payload)
    return payload
