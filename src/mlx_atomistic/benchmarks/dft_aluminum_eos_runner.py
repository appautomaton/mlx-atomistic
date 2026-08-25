"""Bounded metallic convergence and EOS validation for fcc Aluminum."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_aluminum import (
    kpoint_mesh_from_workload,
    load_aluminum_workload,
)
from mlx_atomistic.benchmarks.dft_aluminum_eos import (
    EOS_REPORT_SCHEMA,
    HARTREE_TO_EV,
    REFERENCE_SHA256,
    compare_eos_convergence,
    compare_fit_to_reference,
    fit_cubic_aluminum_eos,
    load_aluminum_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR

POINT_SCHEMA = "mlx-atomistic.aluminum-eos-point.v1"
SCREEN_INDICES = (2, 3, 4)
FULL_INDICES = tuple(range(7))
SCREEN_MESH_SIZE = 11
SHAPE_THRESHOLD_MEV_PER_ATOM = 1.0
SYMMETRY_ENERGY_THRESHOLD_HARTREE_PER_ATOM = 5.0e-5
SYMMETRY_ORACLE_BANDS = 12
_PROFILE_RE = re.compile(
    r"^c(?P<cutoff>\d+)-k(?P<mesh>\d+)(?:-(?P<mode>full|reduced))?-b(?P<bands>\d+)$"
)


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _profile_name(
    cutoff: int,
    mesh: int,
    bands: int,
    *,
    mode: str | None = None,
) -> str:
    suffix = "" if mode is None else f"-{mode}"
    return f"c{cutoff}-k{mesh}{suffix}-b{bands}"


def _profile_settings(manifest: Mapping[str, Any], profile: str) -> dict[str, Any]:
    match = _PROFILE_RE.fullmatch(profile)
    if match is None:
        raise ValueError(f"invalid Aluminum EOS profile: {profile}")
    cutoff = int(match.group("cutoff"))
    mesh = int(match.group("mesh"))
    bands = int(match.group("bands"))
    mode = match.group("mode")
    validation = manifest["validation"]
    if float(cutoff) not in validation["cutoff_candidates_hartree"]:
        raise ValueError(f"unsupported Aluminum cutoff profile: {profile}")
    if bands not in validation["band_capacity_candidates"]:
        raise ValueError(f"unsupported Aluminum band profile: {profile}")
    if mode is None:
        if mesh not in validation["kpoint_mesh_sizes"]:
            raise ValueError(f"unsupported Aluminum k-point profile: {profile}")
        mode = "reduced"
    elif mesh != int(validation["full_grid_oracle_size"]):
        raise ValueError("full/reduced oracle profiles require the locked oracle mesh")
    fft_size = int(validation["fft_shape_by_cutoff"][str(cutoff)])
    return {
        "cutoff_hartree": float(cutoff),
        "fft_shape": [fft_size, fft_size, fft_size],
        "kpoint_mesh_size": mesh,
        "kpoint_mode": mode,
        "n_bands": bands,
        "max_batch_transient_bytes": 1024 * 1024**2,
        "timeout_seconds": float(validation["point_timeout_seconds"]),
        "memory_limit_bytes": int(validation["memory_limit_bytes"]),
    }


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.aluminum-eos-point-execution.v1",
        "point_schema": POINT_SCHEMA,
        "profile_source": inspect.getsource(_profile_settings),
        "scf_config_source": inspect.getsource(_scf_config),
        "point_execution_source": inspect.getsource(run_aluminum_eos_point),
    }
    return sha256_bytes(canonical_json_bytes(contract))


@lru_cache(maxsize=1)
def _runtime_fingerprint() -> str:
    from mlx_atomistic.benchmarks.dft_runtime_contract import build_source_fingerprints

    return str(build_source_fingerprints()["runtime_fingerprint"])


def _point_spec(
    *,
    manifest: Mapping[str, Any],
    profile: str,
    volume_index: int,
    initial_density_sha256: str | None = None,
) -> dict[str, Any]:
    settings = _profile_settings(manifest, profile)
    values = {
        "workload_fingerprint": manifest["workload_fingerprint"],
        "runtime_fingerprint": _runtime_fingerprint(),
        "eos_implementation_fingerprint": _implementation_fingerprint(),
        "reference_sha256": REFERENCE_SHA256,
        "profile": profile,
        "volume_index": volume_index,
        "lattice_constant_angstrom": validation_lattice_constants()[volume_index],
        "initial_density_sha256": initial_density_sha256,
        **settings,
    }
    return {**values, "point_fingerprint": sha256_bytes(canonical_json_bytes(values))}


def _scf_config(
    manifest: Mapping[str, Any],
    *,
    max_batch_transient_bytes: int,
) -> Any:
    from mlx_atomistic.dft import (
        PeriodicDavidsonConfig,
        PeriodicFermiDiracSmearing,
        PeriodicSCFConfig,
    )

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
        smearing=PeriodicFermiDiracSmearing(
            width_hartree=float(manifest["physics"]["smearing_width_hartree"])
        ),
        davidson=PeriodicDavidsonConfig(
            max_iterations=int(davidson["max_iterations"]),
            tolerance=float(davidson["tolerance"]),
            max_subspace_size=int(davidson["max_subspace_size"]),
            preconditioner_floor=float(davidson["preconditioner_floor"]),
        ),
    )


def _mesh_for_profile(manifest: Mapping[str, Any], settings: Mapping[str, Any]) -> Any:
    from mlx_atomistic.dft import (
        GammaCenteredGrid,
        cubic_reciprocal_symmetry_operations,
        reduce_kpoint_mesh_by_symmetry,
    )

    size = int(settings["kpoint_mesh_size"])
    mode = settings["kpoint_mode"]
    if size in manifest["validation"]["kpoint_mesh_sizes"]:
        if mode != "reduced":
            raise ValueError("production Aluminum meshes must use fingerprinted reduction")
        return kpoint_mesh_from_workload(manifest, size)
    full = GammaCenteredGrid((size, size, size))
    if mode == "full":
        return full
    return reduce_kpoint_mesh_by_symmetry(
        full,
        cubic_reciprocal_symmetry_operations(),
    )


def run_aluminum_eos_point(
    *,
    manifest_path: str | Path,
    profile: str,
    volume_index: int,
    out: str | Path,
    initial_density_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one isolated Aluminum EOS point and persist compact evidence."""

    import mlx.core as mx

    from mlx_atomistic.benchmarks.dft_runtime_contract import collect_host_provenance
    from mlx_atomistic.dft import PeriodicDFTSystem, read_gth, run_periodic_scf
    from mlx_atomistic.dft._runtime_observer import RuntimeObserver

    if volume_index not in FULL_INDICES:
        raise ValueError("Aluminum EOS volume_index must lie in [0, 6]")
    manifest, resource = load_aluminum_workload(manifest_path)
    settings = _profile_settings(manifest, profile)
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
        manifest=manifest,
        profile=profile,
        volume_index=volume_index,
        initial_density_sha256=initial_density_sha256,
    )
    lattice_bohr = lattice_angstrom * ANGSTROM_TO_BOHR
    fractional = np.asarray(manifest["system"]["fractional_positions"], dtype=np.float64)
    system = PeriodicDFTSystem(
        (lattice_bohr, lattice_bohr, lattice_bohr),
        settings["fft_shape"],
        fractional * lattice_bohr,
        read_gth(resource, element="Al", name="GTH-PBE-q3"),
        electron_count=float(manifest["system"]["electron_count"]),
    )
    observer = RuntimeObserver(detail_events=False)
    host = collect_host_provenance()
    started = perf_counter()
    result = run_periodic_scf(
        system,
        cutoff_hartree=float(settings["cutoff_hartree"]),
        kpoint_mesh=_mesh_for_profile(manifest, settings),
        n_bands=int(settings["n_bands"]),
        config=_scf_config(
            manifest,
            max_batch_transient_bytes=int(settings["max_batch_transient_bytes"]),
        ),
        observer=observer,
        initial_density=initial_density,
    )
    mx.synchronize()
    elapsed = perf_counter() - started

    electron_error = abs(float(result.electron_count) - float(manifest["system"]["electron_count"]))
    maximum_overlap = max(float(item.eigen.orthonormality_error) for item in result.owned_kpoints)
    maximum_residual = max(
        float(np.max(np.asarray(item.eigen.residuals))) for item in result.owned_kpoints
    )
    occupations = [
        tuple(float(value) for value in point.occupations or ()) for point in result.kpoints
    ]
    if not occupations or any(len(values) != int(settings["n_bands"]) for values in occupations):
        raise RuntimeError("Aluminum metallic result did not publish band occupations")
    highest_band_occupation = max(values[-1] for values in occupations)
    fractional_occupation_count = sum(
        1 for values in occupations for value in values if 1.0e-6 < value < 2.0 - 1.0e-6
    )
    width = float(manifest["physics"]["smearing_width_hartree"])
    internal_energy = float(result.internal_energy)
    entropy = float(result.electronic_entropy)
    free_energy_identity_error = abs(
        float(result.total_energy) - (internal_energy - width * entropy)
    )
    gates = manifest["numerical_gates"]
    numerical_passed = bool(
        result.converged
        and np.isfinite(
            [
                result.total_energy,
                internal_energy,
                entropy,
                result.chemical_potential,
            ]
        ).all()
        and electron_error <= float(gates["electron_count_abs_per_cell"])
        and maximum_overlap <= float(gates["orthonormality_max"])
        and maximum_residual <= float(gates["orbital_residual_max"])
        and highest_band_occupation <= float(gates["highest_band_occupation_max"])
        and free_energy_identity_error <= float(gates["free_energy_identity_abs_hartree"])
        and fractional_occupation_count > 0
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
            "smearing_width_hartree": width,
            "energy_observable": manifest["physics"]["energy_observable"],
            "atoms": int(manifest["system"]["atom_count"]),
            "electrons": int(manifest["system"]["electron_count"]),
            "bands": int(settings["n_bands"]),
        },
        "host": host,
        "result": {
            "free_energy_hartree": float(result.total_energy),
            "internal_energy_hartree": internal_energy,
            "electronic_entropy": entropy,
            "chemical_potential_hartree": float(result.chemical_potential),
            "free_energy_identity_error_hartree": free_energy_identity_error,
            "converged": bool(result.converged),
            "scf_iterations": int(result.iterations),
            "electron_count": float(result.electron_count),
            "electron_count_error": electron_error,
            "maximum_orbital_residual": maximum_residual,
            "maximum_orthonormality_error": maximum_overlap,
            "highest_band_occupation": highest_band_occupation,
            "fractional_occupation_count": fractional_occupation_count,
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
        if row["point"]["profile"] == profile and int(row["point"]["volume_index"]) in indices
    ]
    return sorted(selected, key=lambda row: int(row["point"]["volume_index"]))


def _shape_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    atom_count: int,
) -> dict[str, Any]:
    first = sorted(baseline_rows, key=lambda row: int(row["point"]["volume_index"]))
    second = sorted(candidate_rows, key=lambda row: int(row["point"]["volume_index"]))
    expected = list(SCREEN_INDICES)
    if (
        len(first) != 3
        or len(second) != 3
        or [row["point"]["volume_index"] for row in first] != expected
        or [row["point"]["volume_index"] for row in second] != expected
        or not all(row.get("numerical_passed") is True for row in [*first, *second])
    ):
        return {"status": "blocked", "blocker": "shape_comparison_incomplete", "passed": False}
    first_energy = np.asarray([row["result"]["free_energy_hartree"] for row in first])
    second_energy = np.asarray([row["result"]["free_energy_hartree"] for row in second])
    maximum = float(
        np.max(np.abs((second_energy - second_energy[1]) - (first_energy - first_energy[1])))
        * HARTREE_TO_EV
        * 1000.0
        / atom_count
    )
    return {
        "status": "ok" if maximum <= SHAPE_THRESHOLD_MEV_PER_ATOM else "failed",
        "passed": maximum <= SHAPE_THRESHOLD_MEV_PER_ATOM,
        "scope": "central_three_volume_free_energy_shape",
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
    return fit_cubic_aluminum_eos(
        [float(row["point"]["lattice_constant_angstrom"]) for row in ordered],
        [float(row["result"]["free_energy_hartree"]) for row in ordered],
        atom_count=atom_count,
    )


def _load_matching_point(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    point = payload.get("point", {})
    if (
        payload.get("schema_version") != POINT_SCHEMA
        or not isinstance(point, dict)
        or point.get("point_fingerprint") != expected["point_fingerprint"]
    ):
        raise ValueError(f"refusing mismatched Aluminum EOS point artifact: {path}")
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
    command = [
        sys.executable,
        "scripts/run_bounded_process.py",
        "--max-bytes",
        str(spec["memory_limit_bytes"]),
        "--poll-seconds",
        "0.25",
        "--timeout-seconds",
        str(spec["timeout_seconds"]),
        "--trace-out",
        str(trace_path),
        "--",
        sys.executable,
        "-m",
        "mlx_atomistic.benchmarks.dft_aluminum",
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
        }
    return _load_matching_point(report_path, spec), None


def _plan_spec(
    manifest: Mapping[str, Any],
    profile: str,
    index: int,
    initial_density_path: Path | None = None,
) -> dict[str, Any]:
    digest = (
        None if initial_density_path is None else sha256_bytes(initial_density_path.read_bytes())
    )
    return _point_spec(
        manifest=manifest,
        profile=profile,
        volume_index=index,
        initial_density_sha256=digest,
    )


def _ensure_profile_indices(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    profile: str,
    indices: Sequence[int],
) -> dict[str, Any] | None:
    known = {(str(row["point"]["profile"]), int(row["point"]["volume_index"])): row for row in rows}
    failure = _ensure_profile_point(
        manifest_path=manifest_path,
        manifest=manifest,
        output=output,
        rows=rows,
        known=known,
        profile=profile,
        index=3,
    )
    if failure is not None:
        return failure
    center_key = (profile, 3)
    center = known[center_key]
    if center.get("numerical_passed") is not True:
        return None
    density_path = output / "points" / profile / "v3" / "density.npy"
    if not density_path.is_file():
        return {"blocker": f"density_seed_missing:{profile}"}
    for index in indices:
        if index == 3:
            continue
        failure = _ensure_profile_point(
            manifest_path=manifest_path,
            manifest=manifest,
            output=output,
            rows=rows,
            known=known,
            profile=profile,
            index=index,
            density_path=density_path,
        )
        if failure is not None:
            return failure
    return None


def _ensure_profile_point(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    known: dict[tuple[str, int], dict[str, Any]],
    profile: str,
    index: int,
    density_path: Path | None = None,
) -> dict[str, Any] | None:
    key = (profile, index)
    spec = _plan_spec(manifest, profile, index, density_path)
    if key in known:
        if known[key]["point"]["point_fingerprint"] != spec["point_fingerprint"]:
            raise ValueError(f"refusing stale Aluminum EOS point: {profile}:v{index}")
        return None
    report, failure = _run_bounded_point(
        manifest_path=manifest_path,
        output=output,
        spec=spec,
        initial_density_path=density_path,
    )
    if failure is not None:
        return failure
    if report is None:
        return {"blocker": f"point_artifact_missing:{profile}:v{index}"}
    rows.append(report)
    known[key] = report
    return None


def _existing_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "points").glob("*/v*/report.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") == POINT_SCHEMA:
            rows.append(payload)
    return rows


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


def _symmetry_comparison(
    rows: Sequence[Mapping[str, Any]],
    full_profile: str,
    reduced_profile: str,
    *,
    atom_count: int,
) -> dict[str, Any]:
    full = _ordered_rows(rows, full_profile, (3,))
    reduced = _ordered_rows(rows, reduced_profile, (3,))
    if (
        len(full) != 1
        or len(reduced) != 1
        or full[0].get("numerical_passed") is not True
        or reduced[0].get("numerical_passed") is not True
    ):
        return {"status": "blocked", "passed": False, "blocker": "symmetry_oracle_failed"}
    error = (
        abs(
            float(full[0]["result"]["free_energy_hartree"])
            - float(reduced[0]["result"]["free_energy_hartree"])
        )
        / atom_count
    )
    return {
        "status": "ok" if error <= SYMMETRY_ENERGY_THRESHOLD_HARTREE_PER_ATOM else "failed",
        "passed": error <= SYMMETRY_ENERGY_THRESHOLD_HARTREE_PER_ATOM,
        "free_energy_abs_hartree_per_atom": error,
        "threshold_hartree_per_atom": SYMMETRY_ENERGY_THRESHOLD_HARTREE_PER_ATOM,
        "full_point_count": int(full[0]["result"]["explicit_kpoint_count"]),
        "reduced_point_count": int(reduced[0]["result"]["explicit_kpoint_count"]),
    }


def _selected_runtime_summary(
    rows: Sequence[Mapping[str, Any]],
    profile: str,
    output: Path,
) -> dict[str, Any]:
    selected = _ordered_rows(rows, profile, FULL_INDICES)
    physical_peaks = []
    runtime_temporary_peaks = []
    runtime_persistent_payloads = []
    for row in selected:
        volume_index = int(row["point"]["volume_index"])
        trace_path = output / "points" / profile / f"v{volume_index}" / "memory.json"
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text())
            value = trace.get("bounded_process_peak_physical_bytes")
            if value is not None:
                physical_peaks.append(int(value))
        memory = row["result"]["observation"]["memory"]
        temporary = memory.get("peak_temporary_bytes")
        if temporary is not None:
            runtime_temporary_peaks.append(int(temporary))
        persistent = sum(
            int(memory.get(field) or 0)
            for field in (
                "persistent_coefficient_bytes",
                "persistent_projector_bytes",
                "shared_full_grid_bytes",
            )
        )
        runtime_persistent_payloads.append(persistent)
    power_states = sorted(
        {
            (
                row["host"].get("power_source"),
                row["host"].get("low_power_mode"),
                row["host"].get("thermal_pressure"),
            )
            for row in selected
        },
        key=str,
    )
    return {
        "complete_wall_seconds": sum(
            float(row["result"]["elapsed_wall_seconds"]) for row in selected
        ),
        "maximum_process_physical_bytes": max(physical_peaks, default=None),
        "maximum_runtime_peak_temporary_bytes": max(
            runtime_temporary_peaks,
            default=None,
        ),
        "maximum_runtime_persistent_payload_bytes": max(
            runtime_persistent_payloads,
            default=None,
        ),
        "point_walls_seconds": [float(row["result"]["elapsed_wall_seconds"]) for row in selected],
        "host_power_states": [
            {
                "power_source": state[0],
                "low_power_mode": state[1],
                "thermal_pressure": state[2],
            }
            for state in power_states
        ],
        "host_power_state_consistent": len(power_states) == 1,
    }


def _dry_run_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validation = manifest["validation"]
    cutoffs = [int(value) for value in validation["cutoff_candidates_hartree"]]
    meshes = [int(value) for value in validation["kpoint_mesh_sizes"]]
    bands = [int(value) for value in validation["band_capacity_candidates"]]
    return {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": "planned",
        "decision_ladder": [
            "validate 4x4x4 full versus cubic-symmetry-reduced SCF",
            "admit the first complete band capacity at the densest mesh and largest volume",
            "screen adjacent cutoff pairs over volume indices 2,3,4",
            "screen adjacent Gamma-centered k-point pairs over volume indices 2,3,4",
            "complete selected, upper-cutoff, and upper-kpoint seven-volume curves",
            "compare the selected EOS with the all-electron ACWF reference",
        ],
        "cutoff_candidates_hartree": cutoffs,
        "kpoint_mesh_sizes": meshes,
        "band_capacity_candidates": bands,
        "screen_bands": "selected by adjacent capacity convergence",
        "screen_mesh_size": SCREEN_MESH_SIZE,
        "maximum_point_count": 57,
        "memory_limit_bytes": validation["memory_limit_bytes"],
        "point_timeout_seconds": validation["point_timeout_seconds"],
    }


def run_aluminum_eos_validation(
    *,
    manifest_path: str | Path,
    out: str | Path,
    dry_run: bool = False,
    summarize_only: bool = False,
) -> dict[str, Any]:
    """Run the fail-early Aluminum metallic admission ladder."""

    if dry_run and summarize_only:
        raise ValueError("--dry-run and --summarize-only are mutually exclusive")
    manifest_file = Path(manifest_path).resolve()
    manifest, _resource = load_aluminum_workload(manifest_file)
    output = Path(out)
    if dry_run:
        payload = _dry_run_plan(manifest)
        _write_atomic(output / "plan.json", payload)
        return payload
    rows = _existing_rows(output)
    if summarize_only:
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        payload = {
            "schema_version": EOS_REPORT_SCHEMA,
            "status": "passed" if report.get("admitted") is True else "partial",
            "admitted": report.get("admitted") is True,
            "scientifically_verified": report.get("scientifically_verified") is True,
            "evidence_status": "complete" if report.get("admitted") is True else "partial",
            "completed_point_count": len(rows),
        }
        _write_atomic(output / "summary.json", payload)
        return payload

    validation = manifest["validation"]
    atom_count = int(manifest["system"]["atom_count"])
    oracle_size = int(validation["full_grid_oracle_size"])
    oracle_cutoff = int(validation["cutoff_candidates_hartree"][0])
    oracle_bands = SYMMETRY_ORACLE_BANDS
    if oracle_bands not in validation["band_capacity_candidates"]:
        raise ValueError("locked symmetry-oracle band count is not in the workload ladder")
    full_oracle = _profile_name(oracle_cutoff, oracle_size, oracle_bands, mode="full")
    reduced_oracle = _profile_name(oracle_cutoff, oracle_size, oracle_bands, mode="reduced")
    for profile in (full_oracle, reduced_oracle):
        failure = _ensure_profile_indices(
            manifest_path=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=profile,
            indices=(3,),
        )
        if failure is not None:
            return _failure_report(
                output, blocker=str(failure["blocker"]), rows=rows, detail=failure
            )
    symmetry = _symmetry_comparison(rows, full_oracle, reduced_oracle, atom_count=atom_count)
    if symmetry.get("passed") is not True:
        return _failure_report(output, blocker="symmetry_oracle_failed", rows=rows, detail=symmetry)

    cutoffs = [int(value) for value in validation["cutoff_candidates_hartree"]]
    meshes = [int(value) for value in validation["kpoint_mesh_sizes"]]
    bands = [int(value) for value in validation["band_capacity_candidates"]]
    selected_bands = None
    band_checks = []
    capacity_rows = []
    capacity_mesh = meshes[-1]
    capacity_volume = FULL_INDICES[-1]
    capacity_density = output / "points" / reduced_oracle / "v3" / "density.npy"
    for candidate_bands in bands:
        profile = _profile_name(cutoffs[0], capacity_mesh, candidate_bands)
        spec = _plan_spec(
            manifest,
            profile,
            capacity_volume,
            capacity_density,
        )
        report, failure = _run_bounded_point(
            manifest_path=manifest_file,
            output=output,
            spec=spec,
            initial_density_path=capacity_density,
            root_name="capacity",
        )
        if failure is not None:
            return _failure_report(
                output, blocker=str(failure["blocker"]), rows=rows, detail=failure
            )
        if report is None:
            return _failure_report(
                output,
                blocker=f"capacity_point_artifact_missing:{profile}",
                rows=rows,
            )
        capacity_rows.append(report)
        passed = report.get("numerical_passed") is True
        band_checks.append(
            {
                "bands": candidate_bands,
                "passed": passed,
                "highest_band_occupation": report["result"]["highest_band_occupation"],
                "maximum_orbital_residual": report["result"]["maximum_orbital_residual"],
            }
        )
        if passed:
            selected_bands = candidate_bands
            break
    if selected_bands is None:
        return _failure_report(output, blocker="band_capacity_exhausted", rows=rows)

    cutoff_choice: tuple[int, int] | None = None
    cutoff_screen: dict[str, Any] | None = None
    for lower, upper in zip(cutoffs, cutoffs[1:], strict=True):
        profiles = (
            _profile_name(lower, SCREEN_MESH_SIZE, selected_bands),
            _profile_name(upper, SCREEN_MESH_SIZE, selected_bands),
        )
        for profile in profiles:
            failure = _ensure_profile_indices(
                manifest_path=manifest_file,
                manifest=manifest,
                output=output,
                rows=rows,
                profile=profile,
                indices=SCREEN_INDICES,
            )
            if failure is not None:
                return _failure_report(
                    output, blocker=str(failure["blocker"]), rows=rows, detail=failure
                )
        comparison = _shape_comparison(
            _ordered_rows(rows, profiles[0], SCREEN_INDICES),
            _ordered_rows(rows, profiles[1], SCREEN_INDICES),
            atom_count=atom_count,
        )
        if comparison.get("passed") is True:
            cutoff_choice = (lower, upper)
            cutoff_screen = comparison
            break
    if cutoff_choice is None or cutoff_screen is None:
        return _failure_report(output, blocker="cutoff_screen_exhausted", rows=rows)

    kpoint_choice: tuple[int, int] | None = None
    kpoint_screen: dict[str, Any] | None = None
    for lower, upper in zip(meshes, meshes[1:], strict=True):
        profiles = (
            _profile_name(cutoff_choice[0], lower, selected_bands),
            _profile_name(cutoff_choice[0], upper, selected_bands),
        )
        for profile in profiles:
            failure = _ensure_profile_indices(
                manifest_path=manifest_file,
                manifest=manifest,
                output=output,
                rows=rows,
                profile=profile,
                indices=SCREEN_INDICES,
            )
            if failure is not None:
                return _failure_report(
                    output, blocker=str(failure["blocker"]), rows=rows, detail=failure
                )
        comparison = _shape_comparison(
            _ordered_rows(rows, profiles[0], SCREEN_INDICES),
            _ordered_rows(rows, profiles[1], SCREEN_INDICES),
            atom_count=atom_count,
        )
        if comparison.get("passed") is True:
            kpoint_choice = (lower, upper)
            kpoint_screen = comparison
            break
    if kpoint_choice is None or kpoint_screen is None:
        return _failure_report(output, blocker="kpoint_screen_exhausted", rows=rows)

    cutoff_index = cutoffs.index(cutoff_choice[0])
    kpoint_index = meshes.index(kpoint_choice[0])
    cutoff_history = []
    kpoint_history = []
    while True:
        selected_cutoff = cutoffs[cutoff_index]
        selected_kpoint = meshes[kpoint_index]
        selected_profile = _profile_name(
            selected_cutoff,
            selected_kpoint,
            selected_bands,
        )
        failure = _ensure_profile_indices(
            manifest_path=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=selected_profile,
            indices=FULL_INDICES,
        )
        if failure is not None:
            return _failure_report(
                output, blocker=str(failure["blocker"]), rows=rows, detail=failure
            )
        selected_fit = _fit_profile(rows, selected_profile, atom_count=atom_count)
        if selected_fit.get("status") != "ok":
            return _failure_report(
                output,
                blocker="selected_profile_not_admissible",
                rows=rows,
                detail=selected_fit,
            )
        if kpoint_index + 1 >= len(meshes):
            return _failure_report(output, blocker="kpoint_convergence_exhausted", rows=rows)
        upper_kpoint = meshes[kpoint_index + 1]
        upper_kpoint_profile = _profile_name(
            selected_cutoff,
            upper_kpoint,
            selected_bands,
        )
        failure = _ensure_profile_indices(
            manifest_path=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=upper_kpoint_profile,
            indices=FULL_INDICES,
        )
        if failure is not None:
            return _failure_report(
                output, blocker=str(failure["blocker"]), rows=rows, detail=failure
            )
        upper_kpoint_fit = _fit_profile(
            rows,
            upper_kpoint_profile,
            atom_count=atom_count,
        )
        if upper_kpoint_fit.get("status") != "ok":
            return _failure_report(
                output,
                blocker="upper_kpoint_profile_not_admissible",
                rows=rows,
                detail=upper_kpoint_fit,
            )
        kpoint_convergence = compare_eos_convergence(selected_fit, upper_kpoint_fit)
        kpoint_history.append(
            {
                "selected_mesh": [selected_kpoint] * 3,
                "upper_mesh": [upper_kpoint] * 3,
                "comparison": kpoint_convergence,
            }
        )
        if kpoint_convergence.get("passed") is not True:
            kpoint_index += 1
            continue
        if cutoff_index + 1 >= len(cutoffs):
            return _failure_report(output, blocker="cutoff_convergence_exhausted", rows=rows)
        upper_cutoff = cutoffs[cutoff_index + 1]
        upper_cutoff_profile = _profile_name(
            upper_cutoff,
            selected_kpoint,
            selected_bands,
        )
        failure = _ensure_profile_indices(
            manifest_path=manifest_file,
            manifest=manifest,
            output=output,
            rows=rows,
            profile=upper_cutoff_profile,
            indices=FULL_INDICES,
        )
        if failure is not None:
            return _failure_report(
                output, blocker=str(failure["blocker"]), rows=rows, detail=failure
            )
        upper_cutoff_fit = _fit_profile(
            rows,
            upper_cutoff_profile,
            atom_count=atom_count,
        )
        if upper_cutoff_fit.get("status") != "ok":
            return _failure_report(
                output,
                blocker="upper_cutoff_profile_not_admissible",
                rows=rows,
                detail=upper_cutoff_fit,
            )
        cutoff_convergence = compare_eos_convergence(selected_fit, upper_cutoff_fit)
        cutoff_history.append(
            {
                "selected_cutoff_hartree": selected_cutoff,
                "upper_cutoff_hartree": upper_cutoff,
                "comparison": cutoff_convergence,
            }
        )
        if cutoff_convergence.get("passed") is not True:
            cutoff_index += 1
            continue
        break

    references = load_aluminum_eos_references()
    scientific = compare_fit_to_reference(
        selected_fit,
        reference_fit(references["references"]["all_electron_average"]),
    )
    blockers = []
    if cutoff_convergence.get("passed") is not True:
        blockers.append("cutoff_convergence_failed")
    if kpoint_convergence.get("passed") is not True:
        blockers.append("kpoint_convergence_failed")
    if scientific.get("verified") is not True:
        blockers.append("all_electron_reference_thresholds_failed")
    admitted = not blockers
    payload = {
        "schema_version": EOS_REPORT_SCHEMA,
        "status": "passed" if admitted else "failed",
        "admitted": admitted,
        "scientifically_verified": admitted and scientific.get("verified") is True,
        "blockers": blockers,
        "workload_fingerprint": manifest["workload_fingerprint"],
        "accepted_workload": {
            "profile": selected_profile,
            "cutoff_hartree": selected_cutoff,
            "fft_shape": _profile_settings(manifest, selected_profile)["fft_shape"],
            "kpoint_mesh": [selected_kpoint] * 3,
            "representative_kpoint_count": manifest["reduced_kpoint_meshes"][str(selected_kpoint)][
                "representative_point_count"
            ],
            "bands": selected_bands,
            "smearing_width_hartree": manifest["physics"]["smearing_width_hartree"],
            "energy_observable": manifest["physics"]["energy_observable"],
            "volume_point_count": 7,
        },
        "symmetry_oracle": symmetry,
        "cutoff_screen": cutoff_screen,
        "kpoint_screen": kpoint_screen,
        "band_capacity": {
            "selected_bands": selected_bands,
            "checks": band_checks,
            "scope": "densest locked mesh at the largest EOS volume",
            "mesh": [capacity_mesh] * 3,
            "volume_index": capacity_volume,
            "highest_band_occupation_max": manifest["numerical_gates"][
                "highest_band_occupation_max"
            ],
        },
        "selected_fit": selected_fit,
        "cutoff_convergence": cutoff_convergence,
        "cutoff_convergence_history": cutoff_history,
        "kpoint_convergence": kpoint_convergence,
        "kpoint_convergence_history": kpoint_history,
        "scientific_comparison": scientific,
        "runtime": _selected_runtime_summary(rows, selected_profile, output),
        "reference_bundle": {
            "sha256": REFERENCE_SHA256,
            "license": references["license"],
            "primary": references["references"]["all_electron_average"],
            "same_pseudopotential_family": references["references"]["cp2k_gth"],
        },
        "completed_point_count": len(rows) + len(capacity_rows),
        "point_fingerprints": sorted(
            str(row["point"]["point_fingerprint"]) for row in [*rows, *capacity_rows]
        ),
    }
    _write_atomic(output / "report.json", payload)
    return payload
