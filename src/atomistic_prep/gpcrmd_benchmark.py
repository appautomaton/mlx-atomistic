"""Repeatable GPCRmd MLX runtime benchmarks."""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from atomistic_prep.io import JSON_NAME, NPZ_NAME, VIEW_PDB_NAME, load_prepared_system
from atomistic_prep.runner import TRAJECTORY_NAME, run_gpcrmd_mlx
from mlx_atomistic.benchmarks.gpcrmd_runtime import (
    directory_size_bytes,
    last_int,
    max_float,
    max_rss_mb,
    pme_mesh_summary,
)
from mlx_atomistic.io import load_npz_trajectory
from mlx_atomistic.runtime import get_runtime_info

GPCRMD_BENCHMARK_JSON_NAME = "gpcrmd_performance.json"
GPCRMD_BENCHMARK_CSV_NAME = "gpcrmd_performance.csv"


def benchmark_gpcrmd_mlx(
    *,
    out: str | Path,
    target_id: str | None = None,
    cache: str | Path | None = None,
    registry_path: str | Path | None = None,
    prepared: str | Path | None = None,
    durations_ps: tuple[float, ...] = (0.01,),
    electrostatics_modes: tuple[str, ...] = ("artifact",),
    dt: float = 0.001,
    sample_interval: int = 10,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int | None = 7,
    restraint_k: float = 5.0,
    minimize_steps: int = 0,
    equilibration_steps: int = 0,
    constraint_max_iterations: int = 4,
    diagnostic_interval: int | None = None,
    force: bool = False,
    write_json: bool = True,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Run short GPCRmd benchmark rows through `run_gpcrmd_mlx`."""

    if cache is None and prepared is None:
        msg = "benchmark_gpcrmd_mlx requires either cache or prepared"
        raise ValueError(msg)
    if dt <= 0.0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if sample_interval <= 0:
        msg = "sample_interval must be positive"
        raise ValueError(msg)
    durations = _validated_durations(durations_ps)
    modes = _validated_modes(electrostatics_modes)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for duration_ps in durations:
        steps = max(1, int(round(duration_ps / dt)))
        for requested_mode in modes:
            case_dir = out_dir / _case_dir_name(duration_ps, requested_mode)
            if force and case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            row = _run_case(
                case_dir=case_dir,
                target_id=target_id,
                cache=None if cache is None else Path(cache),
                registry_path=None if registry_path is None else Path(registry_path),
                prepared=None if prepared is None else Path(prepared),
                requested_electrostatics_mode=requested_mode,
                duration_ps=duration_ps,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                temperature=temperature,
                friction=friction,
                seed=seed,
                restraint_k=restraint_k,
                minimize_steps=minimize_steps,
                equilibration_steps=equilibration_steps,
                constraint_max_iterations=constraint_max_iterations,
                diagnostic_interval=diagnostic_interval,
                force=force,
            )
            rows.append(row)

    payload = {
        "runtime": asdict(get_runtime_info()),
        "scope_note": (
            "GPCRmd MLX runtime benchmark. Rows are short NVT timing probes, "
            "not biological sampling or binding/unbinding evidence."
        ),
        "electrostatics_note": (
            "The runnable row uses the electrostatics model encoded in the prepared "
            "artifact. Requested cutoff/ewald_reference/pme variants block unless "
            "they match that artifact; separate prepared variants are needed for "
            "physics-valid mode comparisons."
        ),
        "config": {
            "target_id": target_id,
            "cache": None if cache is None else str(cache),
            "prepared": None if prepared is None else str(prepared),
            "durations_ps": list(durations),
            "electrostatics_modes": list(modes),
            "dt": dt,
            "sample_interval": sample_interval,
            "temperature": temperature,
            "friction": friction,
            "minimize_steps": minimize_steps,
            "equilibration_steps": equilibration_steps,
            "constraint_max_iterations": constraint_max_iterations,
            "diagnostic_interval": diagnostic_interval,
        },
        "case_count": len(rows),
        "blocked_case_count": sum(1 for row in rows if row["status"] != "ran"),
        "cases": rows,
    }
    if write_json:
        (out_dir / GPCRMD_BENCHMARK_JSON_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if write_csv:
        _write_csv(out_dir / GPCRMD_BENCHMARK_CSV_NAME, rows)
    return payload


def _run_case(
    *,
    case_dir: Path,
    target_id: str | None,
    cache: Path | None,
    registry_path: Path | None,
    prepared: Path | None,
    requested_electrostatics_mode: str,
    duration_ps: float,
    steps: int,
    dt: float,
    sample_interval: int,
    temperature: float,
    friction: float,
    seed: int | None,
    restraint_k: float,
    minimize_steps: int,
    equilibration_steps: int,
    constraint_max_iterations: int,
    diagnostic_interval: int | None,
    force: bool,
) -> dict[str, Any]:
    if prepared is not None:
        try:
            _copy_prepared_artifact(prepared, case_dir)
            artifact_model = _prepared_electrostatics_model(case_dir)
        except (FileNotFoundError, ValueError) as exc:
            return _blocked_row(
                case_dir=case_dir,
                target_id=target_id,
                dynamics_id=None,
                requested_electrostatics_mode=requested_electrostatics_mode,
                electrostatics_model=None,
                duration_ps=duration_ps,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                blockers=(f"prepared_artifact:{exc}",),
            )
        if (
            requested_electrostatics_mode != "artifact"
            and requested_electrostatics_mode != artifact_model
        ):
            return _blocked_row(
                case_dir=case_dir,
                target_id=target_id,
                dynamics_id=None,
                requested_electrostatics_mode=requested_electrostatics_mode,
                electrostatics_model=artifact_model,
                duration_ps=duration_ps,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                blockers=(
                    "electrostatics_variant_not_prepared:"
                    f"requested={requested_electrostatics_mode}:artifact={artifact_model}",
                ),
            )
    elif requested_electrostatics_mode != "artifact":
        return _blocked_row(
            case_dir=case_dir,
            target_id=target_id,
            dynamics_id=None,
            requested_electrostatics_mode=requested_electrostatics_mode,
            electrostatics_model=None,
            duration_ps=duration_ps,
            steps=steps,
            dt=dt,
            sample_interval=sample_interval,
            blockers=(
                "electrostatics_variant_requires_prepared_artifact:"
                f"requested={requested_electrostatics_mode}",
            ),
        )

    started = time.perf_counter()
    payload = run_gpcrmd_mlx(
        target_id=target_id,
        cache=cache,
        registry_path=registry_path,
        out=case_dir,
        steps=steps,
        sample_interval=sample_interval,
        dt=dt,
        temperature=temperature,
        friction=friction,
        seed=seed,
        restraint_k=restraint_k,
        minimize_steps=minimize_steps,
        equilibration_steps=equilibration_steps,
        constraint_max_iterations=constraint_max_iterations,
        diagnostic_interval=diagnostic_interval,
        electrostatics="short-range-prototype",
        force=force,
    )
    total_wall_s = time.perf_counter() - started
    if payload.get("status") != "ran":
        return _blocked_row(
            case_dir=case_dir,
            target_id=payload.get("target_id") or target_id,
            dynamics_id=payload.get("dynamics_id"),
            requested_electrostatics_mode=requested_electrostatics_mode,
            electrostatics_model=None,
            duration_ps=duration_ps,
            steps=steps,
            dt=dt,
            sample_interval=sample_interval,
            blockers=tuple(str(item) for item in payload.get("blockers", ())),
            total_wall_s=total_wall_s,
        )
    prepared_system = load_prepared_system(case_dir)
    record = load_npz_trajectory(case_dir / TRAJECTORY_NAME)
    return _completed_row(
        case_dir=case_dir,
        payload=payload,
        prepared=prepared_system,
        record=record,
        requested_electrostatics_mode="short-range-prototype",
        duration_ps=duration_ps,
        steps=steps,
        dt=dt,
        sample_interval=sample_interval,
        total_wall_s=total_wall_s,
    )


def _completed_row(
    *,
    case_dir: Path,
    payload: dict[str, Any],
    prepared,
    record,
    requested_electrostatics_mode: str,
    duration_ps: float,
    steps: int,
    dt: float,
    sample_interval: int,
    total_wall_s: float,
) -> dict[str, Any]:
    metadata = dict(record.metadata)
    compatibility_report = dict(prepared.metadata.compatibility_report)
    run_wall_s = metadata.get("elapsed_wall_seconds")
    run_wall = None if run_wall_s is None else float(run_wall_s)
    import_wall = None if run_wall is None else max(0.0, total_wall_s - run_wall)
    pme_shape, pme_size = pme_mesh_summary(prepared.pme_mesh_shape, prepared.metadata.pme_config)
    electrostatics_model = (
        metadata.get("electrostatics_model")
        or compatibility_report.get("electrostatics_model")
        or "cutoff"
    )
    return {
        **_base_row(
            case_dir=case_dir,
            status="ran",
            target_id=payload.get("target_id"),
            dynamics_id=payload.get("dynamics_id"),
            requested_electrostatics_mode=requested_electrostatics_mode,
            electrostatics_model=str(electrostatics_model),
            duration_ps=duration_ps,
            steps=steps,
            dt=dt,
            sample_interval=sample_interval,
            blockers=(),
        ),
        "atom_count": int(prepared.atom_count),
        "water_atoms": int(np.count_nonzero(np.asarray(prepared.water_mask, dtype=bool))),
        "ion_atoms": int(np.count_nonzero(np.asarray(prepared.ion_mask, dtype=bool))),
        "lipid_atoms": int(np.count_nonzero(np.asarray(prepared.lipid_mask, dtype=bool))),
        "pme_mesh_shape": pme_shape,
        "pme_mesh_size": pme_size,
        "frame_count": int(np.asarray(record.sampled_positions).shape[0]),
        "diagnostic_count": int(np.asarray(record.diagnostic_steps).shape[0]),
        "final_pair_count": last_int(record.pair_count),
        "rebuild_count": last_int(record.rebuild_count),
        "total_wall_s": total_wall_s,
        "run_wall_s": run_wall,
        "import_wall_s": import_wall,
        "integration_steps_per_s": metadata.get("integration_steps_per_second"),
        "production_steps_per_s": steps / run_wall if run_wall and run_wall > 0.0 else None,
        "ps_per_s": metadata.get("simulated_ps_per_wall_second"),
        "max_constraint_error_A": max_float(record.constraint_max_error),
        "max_rss_mb": max_rss_mb(),
        "artifact_size_bytes": directory_size_bytes(case_dir),
        "trajectory_path": str(case_dir / TRAJECTORY_NAME),
        "positions_finite": bool(np.all(np.isfinite(np.asarray(record.sampled_positions)))),
        "total_energy_finite": bool(np.all(np.isfinite(np.asarray(record.total_energy)))),
        "temperature_finite": bool(np.all(np.isfinite(np.asarray(record.temperature)))),
    }


def _blocked_row(
    *,
    case_dir: Path,
    target_id: str | None,
    dynamics_id: Any,
    requested_electrostatics_mode: str,
    electrostatics_model: str | None,
    duration_ps: float,
    steps: int,
    dt: float,
    sample_interval: int,
    blockers: tuple[str, ...],
    total_wall_s: float | None = None,
) -> dict[str, Any]:
    return {
        **_base_row(
            case_dir=case_dir,
            status="blocked",
            target_id=target_id,
            dynamics_id=dynamics_id,
            requested_electrostatics_mode=requested_electrostatics_mode,
            electrostatics_model=electrostatics_model,
            duration_ps=duration_ps,
            steps=steps,
            dt=dt,
            sample_interval=sample_interval,
            blockers=blockers,
        ),
        "atom_count": None,
        "water_atoms": None,
        "ion_atoms": None,
        "lipid_atoms": None,
        "pme_mesh_shape": None,
        "pme_mesh_size": None,
        "frame_count": None,
        "diagnostic_count": None,
        "final_pair_count": None,
        "rebuild_count": None,
        "total_wall_s": total_wall_s,
        "run_wall_s": None,
        "import_wall_s": None,
        "integration_steps_per_s": None,
        "production_steps_per_s": None,
        "ps_per_s": None,
        "max_constraint_error_A": None,
        "max_rss_mb": max_rss_mb(),
        "artifact_size_bytes": directory_size_bytes(case_dir),
        "trajectory_path": None,
        "positions_finite": None,
        "total_energy_finite": None,
        "temperature_finite": None,
    }


def _base_row(
    *,
    case_dir: Path,
    status: str,
    target_id: str | None,
    dynamics_id: Any,
    requested_electrostatics_mode: str,
    electrostatics_model: str | None,
    duration_ps: float,
    steps: int,
    dt: float,
    sample_interval: int,
    blockers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "case": case_dir.name,
        "status": status,
        "target_id": target_id,
        "dynamics_id": dynamics_id,
        "requested_electrostatics_mode": requested_electrostatics_mode,
        "electrostatics_model": electrostatics_model,
        "duration_ps": duration_ps,
        "steps": steps,
        "dt": dt,
        "sample_interval": sample_interval,
        "blockers": ";".join(blockers),
        "output_dir": str(case_dir),
    }


def _copy_prepared_artifact(prepared: Path, case_dir: Path) -> None:
    for name in [JSON_NAME, NPZ_NAME]:
        source = prepared / name
        if not source.exists():
            msg = f"missing prepared artifact file: {source}"
            raise FileNotFoundError(msg)
        shutil.copy2(source, case_dir / name)
    view_pdb = prepared / VIEW_PDB_NAME
    if view_pdb.exists():
        shutil.copy2(view_pdb, case_dir / VIEW_PDB_NAME)


def _prepared_electrostatics_model(prepared_dir: Path) -> str:
    prepared = load_prepared_system(prepared_dir)
    report = dict(prepared.metadata.compatibility_report)
    return str(report.get("electrostatics_model") or "cutoff")


def _case_dir_name(duration_ps: float, requested_mode: str) -> str:
    safe_mode = requested_mode.replace("/", "_")
    return f"{duration_ps:g}ps-{safe_mode}"


def _validated_durations(values: tuple[float, ...]) -> tuple[float, ...]:
    durations = tuple(float(item) for item in values)
    if not durations or any(item <= 0.0 or not np.isfinite(item) for item in durations):
        msg = "durations_ps must contain positive finite values"
        raise ValueError(msg)
    return durations


def _validated_modes(values: tuple[str, ...]) -> tuple[str, ...]:
    modes = tuple(str(item).strip() for item in values if str(item).strip())
    allowed = {"artifact", "cutoff", "ewald_reference", "pme"}
    if not modes or any(item not in allowed for item in modes):
        msg = "electrostatics_modes must contain artifact, cutoff, ewald_reference, or pme"
        raise ValueError(msg)
    return modes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "GPCRMD_BENCHMARK_CSV_NAME",
    "GPCRMD_BENCHMARK_JSON_NAME",
    "benchmark_gpcrmd_mlx",
]
