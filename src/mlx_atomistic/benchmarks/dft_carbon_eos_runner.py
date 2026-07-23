"""Bounded cutoff admission and EOS validation for diamond carbon."""

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
from mlx_atomistic.benchmarks.dft_carbon import load_carbon_workload
from mlx_atomistic.benchmarks.dft_carbon_eos import (
    EOS_REPORT_SCHEMA,
    HARTREE_TO_EV,
    REFERENCE_SHA256,
    compare_fit_to_reference,
    fit_cubic_carbon_eos,
    load_carbon_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR

POINT_SCHEMA = "mlx-atomistic.carbon-eos-point.v1"
MEMORY_LIMIT_BYTES = 40_000_000_000
POINT_TIMEOUT_SECONDS = 180.0
_LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS = {
    "bc41c8439da67b0f8b65c9f7a7b6e8d889a7ece25b3f4ac5fd495a852db19f7a",
}
SCREEN_INDICES = (2, 3, 4)
FULL_INDICES = tuple(range(7))
SHAPE_THRESHOLD_MEV_PER_ATOM = 1.0

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "cutoff30": {
        "cutoff_hartree": 30.0,
        "fft_shape": [40, 40, 40],
        "kpoint_mesh": [6, 6, 6],
        "max_batch_transient_bytes": 512 * 1024**2,
    },
    "cutoff40": {
        "cutoff_hartree": 40.0,
        "fft_shape": [48, 48, 48],
        "kpoint_mesh": [6, 6, 6],
        "max_batch_transient_bytes": 768 * 1024**2,
    },
    "cutoff50": {
        "cutoff_hartree": 50.0,
        "fft_shape": [56, 56, 56],
        "kpoint_mesh": [6, 6, 6],
        "max_batch_transient_bytes": 1024 * 1024**2,
    },
    "kpoint30": {
        "cutoff_hartree": 30.0,
        "fft_shape": [40, 40, 40],
        "kpoint_mesh": [8, 8, 8],
        "max_batch_transient_bytes": 512 * 1024**2,
    },
    "kpoint40": {
        "cutoff_hartree": 40.0,
        "fft_shape": [48, 48, 48],
        "kpoint_mesh": [8, 8, 8],
        "max_batch_transient_bytes": 768 * 1024**2,
    },
    "kpoint50": {
        "cutoff_hartree": 50.0,
        "fft_shape": [56, 56, 56],
        "kpoint_mesh": [8, 8, 8],
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
        "schema_version": "mlx-atomistic.carbon-eos-point-execution.v1",
        "point_schema": POINT_SCHEMA,
        "profiles": PROFILE_SPECS,
        "scf_config_source": inspect.getsource(_scf_config),
        "point_execution_source": inspect.getsource(run_carbon_eos_point),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def _point_spec(
    *,
    workload_fingerprint: str,
    profile: str,
    volume_index: int,
    lattice_angstrom: float,
    initial_density_sha256: str | None = None,
) -> dict[str, Any]:
    settings = PROFILE_SPECS[profile]
    values = {
        "workload_fingerprint": workload_fingerprint,
        "eos_implementation_fingerprint": _implementation_fingerprint(),
        "reference_sha256": REFERENCE_SHA256,
        "profile": profile,
        "volume_index": volume_index,
        "lattice_constant_angstrom": lattice_angstrom,
        "initial_density_sha256": initial_density_sha256,
        "timeout_seconds": POINT_TIMEOUT_SECONDS,
        **settings,
    }
    return {**values, "point_fingerprint": sha256_bytes(canonical_json_bytes(values))}


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


def run_carbon_eos_point(
    *,
    manifest_path: str | Path,
    profile: str,
    volume_index: int,
    out: str | Path,
    initial_density_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one isolated diamond-carbon EOS point and persist compact evidence."""

    import mlx.core as mx

    from mlx_atomistic.dft import (
        MonkhorstPackGrid,
        PeriodicDFTSystem,
        read_gth,
        run_periodic_scf,
    )
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver

    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown carbon EOS profile: {profile}")
    if volume_index not in FULL_INDICES:
        raise ValueError("carbon EOS volume_index must lie in [0, 6]")
    manifest, resource = load_carbon_workload(manifest_path)
    lattice_angstrom = validation_lattice_constants()[volume_index]
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
        lattice_angstrom=lattice_angstrom,
        initial_density_sha256=initial_density_sha256,
    )
    settings = PROFILE_SPECS[profile]
    system_values = manifest["system"]
    lattice_bohr = lattice_angstrom * ANGSTROM_TO_BOHR
    fractional = np.asarray(system_values["fractional_positions"], dtype=np.float64)
    system = PeriodicDFTSystem(
        (lattice_bohr, lattice_bohr, lattice_bohr),
        settings["fft_shape"],
        fractional * lattice_bohr,
        read_gth(resource, element="C", name="GTH-PBE-q4"),
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
        initial_density=initial_density,
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
            "pseudopotential": manifest["physics"]["pseudopotential"],
            "occupation": manifest["physics"]["occupation"],
            "atoms": int(system_values["atom_count"]),
            "electrons": int(system_values["electron_count"]),
            "bands": int(system_values["occupied_band_count"]),
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


def _triad_assessment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["point"]["volume_index"]))
    if len(ordered) != 3 or [row["point"]["volume_index"] for row in ordered] != list(
        SCREEN_INDICES
    ):
        return {"status": "blocked", "blocker": "central_triad_incomplete", "passed": False}
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


def _shape_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    atom_count: int,
) -> dict[str, Any]:
    first = sorted(baseline_rows, key=lambda row: int(row["point"]["volume_index"]))
    second = sorted(candidate_rows, key=lambda row: int(row["point"]["volume_index"]))
    if (
        len(first) != 3
        or len(second) != 3
        or [row["point"]["volume_index"] for row in first] != list(SCREEN_INDICES)
        or [row["point"]["volume_index"] for row in second] != list(SCREEN_INDICES)
        or not all(row.get("numerical_passed") is True for row in [*first, *second])
    ):
        return {"status": "blocked", "blocker": "shape_comparison_incomplete", "passed": False}
    first_energy = np.asarray([float(row["result"]["total_energy_hartree"]) for row in first])
    second_energy = np.asarray([float(row["result"]["total_energy_hartree"]) for row in second])
    first_shape = first_energy - first_energy[1]
    second_shape = second_energy - second_energy[1]
    maximum = float(
        np.max(np.abs(second_shape - first_shape)) * HARTREE_TO_EV * 1000.0 / atom_count
    )
    passed = maximum <= SHAPE_THRESHOLD_MEV_PER_ATOM
    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "scope": "central_three_volume_energy_shape",
        "metrics": {"curve_max_mev_per_atom": maximum},
        "thresholds": {"curve_max_mev_per_atom": SHAPE_THRESHOLD_MEV_PER_ATOM},
    }


def _fit_profile(
    rows: Sequence[Mapping[str, Any]],
    profile: str,
    *,
    atom_count: int,
) -> dict[str, Any]:
    ordered = _ordered_rows(rows, profile, FULL_INDICES)
    if len(ordered) != 7 or not all(row.get("numerical_passed") is True for row in ordered):
        return {"status": "blocked", "blocker": "seven_point_profile_incomplete_or_failed"}
    return fit_cubic_carbon_eos(
        [float(row["point"]["lattice_constant_angstrom"]) for row in ordered],
        [float(row["result"]["total_energy_hartree"]) for row in ordered],
        atom_count=atom_count,
    )


def _load_matching_point(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    point = payload.get("point", {})
    if payload.get("schema_version") != POINT_SCHEMA or not isinstance(point, dict):
        raise ValueError(f"refusing mismatched carbon EOS point artifact: {path}")
    if point.get("point_fingerprint") == expected["point_fingerprint"]:
        return payload
    identity_keys = set(expected).difference(
        {"eos_implementation_fingerprint", "point_fingerprint"}
    )
    legacy_matches = (
        point.get("eos_implementation_fingerprint")
        in _LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS
        and all(point.get(key) == expected[key] for key in identity_keys)
    )
    if not legacy_matches:
        raise ValueError(f"refusing mismatched carbon EOS point artifact: {path}")
    return payload


def _run_bounded_point(
    *,
    manifest_path: Path,
    output: Path,
    spec: Mapping[str, Any],
    initial_density_path: Path | None = None,
    root_name: str = "points",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    profile = str(spec["profile"])
    index = int(spec["volume_index"])
    point_root = output / root_name / profile / f"v{index}"
    report_path = point_root / "report.json"
    existing = _load_matching_point(report_path, spec)
    if existing is not None:
        return existing, None
    point_root.mkdir(parents=True, exist_ok=True)
    trace_path = point_root / "memory.json"
    attempt_path = point_root / "attempt.json"
    attempt = {
        "schema_version": "mlx-atomistic.carbon-eos-attempt.v1",
        "point": dict(spec),
    }
    if attempt_path.is_file():
        if json.loads(attempt_path.read_text()) != attempt:
            raise ValueError(f"refusing mismatched carbon EOS attempt artifact: {attempt_path}")
    elif trace_path.is_file():
        density_is_older = (
            initial_density_path is not None
            and initial_density_path.stat().st_mtime_ns <= trace_path.stat().st_mtime_ns
        )
        trace = json.loads(trace_path.read_text())
        same_bounds = (
            trace.get("bounded_process_limit_bytes") == MEMORY_LIMIT_BYTES
            and trace.get("bounded_process_timeout_seconds") == POINT_TIMEOUT_SECONDS
        )
        if not density_is_older or not same_bounds:
            raise ValueError(f"refusing unidentified carbon EOS bounded trace: {trace_path}")
        _write_atomic(attempt_path, attempt)
    else:
        _write_atomic(attempt_path, attempt)
    if trace_path.is_file() and not report_path.is_file():
        trace = json.loads(trace_path.read_text())
        if (
            trace.get("bounded_process_timed_out") is True
            or trace.get("bounded_process_exceeded") is True
        ):
            return None, {
                "blocker": f"point_execution_failed:{profile}:v{index}",
                "returncode": trace.get("bounded_process_returncode"),
                "timed_out": trace.get("bounded_process_timed_out"),
                "memory_exceeded": trace.get("bounded_process_exceeded"),
                "peak_physical_bytes": trace.get("bounded_process_peak_physical_bytes"),
                "stdout": str(point_root / "stdout.txt"),
                "stderr": str(point_root / "stderr.txt"),
                "memory_trace": str(trace_path),
            }
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
        "mlx_atomistic.benchmarks.dft_carbon",
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
            "peak_physical_bytes": trace.get("bounded_process_peak_physical_bytes"),
            "stdout": str(point_root / "stdout.txt"),
            "stderr": str(point_root / "stderr.txt"),
            "memory_trace": str(trace_path),
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
        lattice_angstrom=validation_lattice_constants()[index],
        initial_density_sha256=density_sha256,
    )


def _execute_profiles(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    profiles: Sequence[str],
    indices: Sequence[int],
    initial_density_by_profile: Mapping[str, Path] | None = None,
) -> dict[str, Any] | None:
    known = {
        (str(row["point"]["profile"]), int(row["point"]["volume_index"])) for row in rows
    }
    for profile in profiles:
        for index in indices:
            if (profile, index) in known:
                continue
            initial_density_path = (
                None
                if initial_density_by_profile is None
                else initial_density_by_profile.get(profile)
            )
            spec = _plan_spec(
                manifest,
                profile,
                index,
                initial_density_path=initial_density_path,
            )
            report, failure = _run_bounded_point(
                manifest_path=manifest_path,
                output=output,
                spec=spec,
                initial_density_path=initial_density_path,
            )
            if failure is not None:
                return failure
            if report is None:
                return {"blocker": f"point_artifact_missing:{profile}:v{index}"}
            rows.append(report)
            known.add((profile, index))
    return None


def _ensure_density_seed(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output: Path,
    profile: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Produce one bounded equilibrium density used to warm-start an EOS curve."""

    spec = _plan_spec(manifest, profile, 3)
    report, failure = _run_bounded_point(
        manifest_path=manifest_path,
        output=output,
        spec=spec,
        root_name="seeds",
    )
    if failure is not None:
        return None, failure
    density_path = output / "seeds" / profile / "v3" / "density.npy"
    if report is None or not density_path.is_file():
        return None, {"blocker": f"density_seed_missing:{profile}"}
    expected = report.get("result", {}).get("density_artifact", {}).get("sha256")
    observed = sha256_bytes(density_path.read_bytes())
    if expected != observed:
        return None, {"blocker": f"density_seed_hash_mismatch:{profile}"}
    return density_path, None


def _failure_report(
    output: Path,
    *,
    blocker: str,
    rows: Sequence[Mapping[str, Any]],
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": "failed",
        "admitted": False,
        "blockers": [blocker],
        "completed_point_count": len(rows),
        "detail": None if detail is None else dict(detail),
    }
    _write_atomic(output / "report.json", payload)
    return payload


def _final_report(
    *,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cutoff_pair: tuple[str, str],
    selected_profile: str,
    cutoff_shape: Mapping[str, Any],
    cutoff_convergence: Mapping[str, Any],
    selected_fit: Mapping[str, Any],
    kpoint_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    references = load_carbon_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    scientific = compare_fit_to_reference(selected_fit, primary)
    blockers = []
    if cutoff_convergence.get("passed") is not True:
        blockers.append("full_cutoff_curve_convergence_incomplete")
    if kpoint_comparison.get("passed") is not True:
        blockers.append("kpoint_spot_check_failed")
    if scientific.get("verified") is not True:
        blockers.append("all_electron_reference_thresholds_failed")
    scientific_verified = (
        scientific.get("verified") is True and kpoint_comparison.get("passed") is True
    )
    admitted = not blockers
    status = "passed" if admitted else "failed"
    settings = PROFILE_SPECS[selected_profile]
    return {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": status,
        "admitted": admitted,
        "scientifically_verified": scientific_verified,
        "blockers": blockers,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "screened_cutoff_pair": list(cutoff_pair),
        "accepted_workload": {
            "profile": selected_profile,
            "cutoff_hartree": settings["cutoff_hartree"],
            "fft_shape": settings["fft_shape"],
            "kpoint_mesh": settings["kpoint_mesh"],
            "volume_point_count": 7,
            "cutoff_evidence": "central-three-volume energy-shape comparison",
            "full_upper_cutoff_curve_required": False,
        },
        "method_scope": {
            "claim": "eight-atom diamond-carbon PBE/GTH equation of state",
            "limitation": references["protocol"]["method_difference"],
            "not_certified": [
                "exact ACWF smearing/free-energy protocol parity",
                "forces",
                "stress",
                "band structure",
                "materials beyond diamond carbon and silicon",
                "full seven-point 8-cubed k-point curve",
            ],
        },
        "selected_cutoff_profile": selected_profile,
        "selected_fit": dict(selected_fit),
        "cutoff_screen": dict(cutoff_shape),
        "cutoff_convergence": dict(cutoff_convergence),
        "kpoint_spot_check": dict(kpoint_comparison),
        "scientific_comparison": scientific,
        "reference_bundle": {
            "sha256": REFERENCE_SHA256,
            "license": references["license"],
            "primary": references["references"]["all_electron_average"],
            "same_pseudopotential_family": references["references"]["cp2k_gth"],
            "context": {
                "qe_sssp": references["references"]["qe_sssp"],
                "published_pbe": references["references"]["published_pbe"],
                "experiment": references["references"]["experiment"],
            },
        },
        "completed_point_count": len(rows),
        "point_fingerprints": sorted(
            str(row["point"]["point_fingerprint"]) for row in rows
        ),
    }


def _existing_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "points").glob("*/v*/report.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") == POINT_SCHEMA:
            rows.append(payload)
    return rows


def run_carbon_eos_validation(
    *,
    manifest_path: str | Path,
    out: str | Path,
    dry_run: bool = False,
    summarize_only: bool = False,
) -> dict[str, Any]:
    """Run the fail-early cutoff and k-point admission ladder."""

    if dry_run and summarize_only:
        raise ValueError("--dry-run and --summarize-only are mutually exclusive")
    manifest_file = Path(manifest_path).resolve()
    manifest, _resource = load_carbon_workload(manifest_file)
    output = Path(out)
    initial = [
        _plan_spec(manifest, profile, index)
        for profile in ("cutoff30", "cutoff40")
        for index in SCREEN_INDICES
    ]
    if dry_run:
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": "planned",
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "point_timeout_seconds": POINT_TIMEOUT_SECONDS,
            "initial_screen_points": initial,
            "initial_point_count": len(initial),
            "maximum_point_count_after_escalation": 16,
            "decision_ladder": [
                "screen cutoff30 versus cutoff40 at volume indices 2,3,4",
                "if needed, screen cutoff40 versus cutoff50",
                "accept the lower cutoff after a passing central-shape comparison",
                "complete only the accepted cutoff's seven-point EOS",
                "spot-check selected cutoff at 8x8x8 over indices 2,3,4",
                "compare the selected EOS with the all-electron PBE reference",
            ],
        }
        _write_atomic(output / "plan.json", payload)
        return payload
    rows = _existing_rows(output)
    if summarize_only:
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        admitted = report.get("admitted") is True
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": "passed" if admitted else "partial",
            "admitted": admitted,
            "scientifically_verified": report.get("scientifically_verified") is True,
            "evidence_status": "complete" if admitted else "partial",
            "completed_point_count": len(rows),
            "completed_points": sorted(
                (row["point"]["profile"], row["point"]["volume_index"]) for row in rows
            ),
        }
        _write_atomic(output / "summary.json", payload)
        return payload

    failure = _execute_profiles(
        manifest_path=manifest_file,
        manifest=manifest,
        output=output,
        rows=rows,
        profiles=("cutoff30", "cutoff40"),
        indices=SCREEN_INDICES,
    )
    if failure is not None:
        return _failure_report(
            output,
            blocker=str(failure["blocker"]),
            rows=rows,
            detail=failure,
        )
    atom_count = int(manifest["system"]["atom_count"])
    pair = ("cutoff30", "cutoff40")
    shape = _shape_comparison(
        _ordered_rows(rows, pair[0], SCREEN_INDICES),
        _ordered_rows(rows, pair[1], SCREEN_INDICES),
        atom_count=atom_count,
    )
    triads_pass = all(
        _triad_assessment(_ordered_rows(rows, profile, SCREEN_INDICES))["passed"]
        for profile in pair
    )
    if not shape["passed"] or not triads_pass:
        failure = _execute_profiles(
            manifest_path=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profiles=("cutoff50",),
            indices=SCREEN_INDICES,
        )
        if failure is not None:
            return _failure_report(
                output,
                blocker=str(failure["blocker"]),
                rows=rows,
                detail=failure,
            )
        pair = ("cutoff40", "cutoff50")
        shape = _shape_comparison(
            _ordered_rows(rows, pair[0], SCREEN_INDICES),
            _ordered_rows(rows, pair[1], SCREEN_INDICES),
            atom_count=atom_count,
        )
        triads_pass = all(
            _triad_assessment(_ordered_rows(rows, profile, SCREEN_INDICES))["passed"]
            for profile in pair
        )
        if not shape["passed"] or not triads_pass:
            return _failure_report(
                output,
                blocker="no_cutoff_pair_passed_central_shape_screen",
                rows=rows,
                detail={"pair": pair, "shape": shape, "triads_passed": triads_pass},
            )

    remaining = tuple(index for index in FULL_INDICES if index not in SCREEN_INDICES)
    selected_profile = pair[0]
    density_seed, seed_failure = _ensure_density_seed(
        manifest_path=manifest_file,
        manifest=manifest,
        output=output,
        profile=selected_profile,
    )
    if seed_failure is not None or density_seed is None:
        detail = seed_failure or {"blocker": f"density_seed_missing:{selected_profile}"}
        return _failure_report(
            output,
            blocker=str(detail["blocker"]),
            rows=rows,
            detail=detail,
        )
    density_seeds = {selected_profile: density_seed}
    failure = _execute_profiles(
        manifest_path=manifest_file,
        manifest=manifest,
        output=output,
        rows=rows,
        profiles=(selected_profile,),
        indices=remaining,
        initial_density_by_profile=density_seeds,
    )
    if failure is not None:
        return _failure_report(
            output,
            blocker=str(failure["blocker"]),
            rows=rows,
            detail=failure,
        )
    selected_fit = _fit_profile(rows, selected_profile, atom_count=atom_count)
    convergence = {
        "status": "ok",
        "passed": True,
        "scope": "central_three_volume_energy_shape",
        "full_upper_cutoff_curve_required": False,
        "screened_profiles": list(pair),
        "metrics": dict(shape["metrics"]),
        "thresholds": dict(shape["thresholds"]),
    }
    kpoint_profile = selected_profile.replace("cutoff", "kpoint")
    failure = _execute_profiles(
        manifest_path=manifest_file,
        manifest=manifest,
        output=output,
        rows=rows,
        profiles=(kpoint_profile,),
        indices=SCREEN_INDICES,
        initial_density_by_profile={
            kpoint_profile: density_seeds[selected_profile],
        },
    )
    if failure is not None:
        return _failure_report(
            output,
            blocker=str(failure["blocker"]),
            rows=rows,
            detail=failure,
        )
    kpoint = _shape_comparison(
        _ordered_rows(rows, selected_profile, SCREEN_INDICES),
        _ordered_rows(rows, kpoint_profile, SCREEN_INDICES),
        atom_count=atom_count,
    )
    payload = _final_report(
        manifest=manifest,
        rows=rows,
        cutoff_pair=pair,
        selected_profile=selected_profile,
        cutoff_shape=shape,
        cutoff_convergence=convergence,
        selected_fit=selected_fit,
        kpoint_comparison=kpoint,
    )
    _write_atomic(output / "report.json", payload)
    return payload
