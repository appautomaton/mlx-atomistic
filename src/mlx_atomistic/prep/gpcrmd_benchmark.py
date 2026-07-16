"""Repeatable GPCRmd MLX runtime benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.artifacts import load_prepared_mlx_artifact
from mlx_atomistic.benchmarks.gpcrmd_runtime import (
    directory_size_bytes,
    last_int,
    max_float,
    max_rss_mb,
    pme_mesh_summary,
)
from mlx_atomistic.io import load_npz_trajectory, load_simulation_checkpoint
from mlx_atomistic.prep.gpcrmd import (
    build_gpcrmd_mlx_workload_manifest,
    load_gpcrmd_targets,
    select_gpcrmd_target,
)
from mlx_atomistic.prep.io import JSON_NAME, NPZ_NAME, VIEW_PDB_NAME, load_prepared_system
from mlx_atomistic.prep.runner import TRAJECTORY_NAME, run_gpcrmd_mlx, run_mlx
from mlx_atomistic.runtime import get_runtime_info

GPCRMD_BENCHMARK_JSON_NAME = "gpcrmd_performance.json"
GPCRMD_BENCHMARK_CSV_NAME = "gpcrmd_performance.csv"
GPCRMD_CHECKPOINT_NAME = "checkpoint.npz"


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


def benchmark_gpcrmd_source_protocol(
    *,
    out: str | Path,
    prepared: str | Path,
    protocol_manifest: str | Path,
    target_id: str | None = None,
    registry_path: str | Path | None = None,
    warmups: int = 1,
    measured_steps: int = 2,
    checkpoint_restart: bool = False,
    seed: int | None = 7,
    constraint_max_iterations: int = 4,
    force: bool = False,
    write_json: bool = True,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Run a bounded source-protocol GPCRmd warmup, measurement, and restart."""

    if warmups <= 0:
        msg = "warmups must be positive in source-protocol mode"
        raise ValueError(msg)
    if measured_steps <= 0:
        msg = "measured_steps must be positive"
        raise ValueError(msg)
    if constraint_max_iterations <= 0:
        msg = "constraint_max_iterations must be positive"
        raise ValueError(msg)

    out_dir = Path(out)
    prepared_path = Path(prepared)
    manifest_path = Path(protocol_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_dirs = {
        "warmup": out_dir / "warmup",
        "measured": out_dir / "measured",
        "restart": out_dir / "restart",
    }
    if force:
        for phase_dir in phase_dirs.values():
            if phase_dir.exists():
                shutil.rmtree(phase_dir)

    manifest, prepared_system, settings, validation = _validate_source_protocol_inputs(
        prepared=prepared_path,
        protocol_manifest=manifest_path,
        target_id=target_id,
        registry_path=None if registry_path is None else Path(registry_path),
    )
    resolved_target_id = validation.get("target_id") or target_id
    dynamics_id = validation.get("dynamics_id")
    declared_differences = [
        {
            "field": "run_length",
            "source": settings.get("source_run_steps"),
            "bounded_evidence": warmups + measured_steps + int(checkpoint_restart),
            "effect": "evidence window only; integration parameters are unchanged",
        },
        {
            "field": "trajectory_and_diagnostic_interval",
            "source_steps": settings.get("source_trajectory_interval_steps"),
            "bounded_evidence_steps": 1,
            "effect": "denser observation only; force and integration semantics are unchanged",
        },
        {
            "field": "thermostat_rng_sequence",
            "source": "source engine RNG state is not represented by the public package",
            "bounded_evidence": f"deterministic MLX seed {seed}",
            "effect": "stochastic trajectories are not expected to be bitwise comparable",
        },
    ]
    payload: dict[str, Any] = {
        "kind": "gpcrmd_source_protocol_benchmark",
        "status": "blocked" if validation["status"] != "ready" else "running",
        "runtime": asdict(get_runtime_info()),
        "scope_note": (
            "Bounded fixed-cell source-protocol NVT evidence. The row is not "
            "production-length biological sampling."
        ),
        "source_protocol_note": (
            "Ensemble, cell, timestep, temperature, friction, constraints, HMR, "
            "nonbonded switching, and PME are admitted from the validated workload "
            "manifest; bounded output/RNG differences are declared separately."
        ),
        "protocol_validation": validation,
        "declared_bounded_evidence_differences": declared_differences,
        "config": {
            "target_id": resolved_target_id,
            "dynamics_id": dynamics_id,
            "prepared": str(prepared_path),
            "protocol_manifest": str(manifest_path),
            "protocol_manifest_sha256": (
                None if manifest is None else manifest.get("manifest_sha256")
            ),
            "warmups": warmups,
            "measured_steps": measured_steps,
            "checkpoint_restart": bool(checkpoint_restart),
            "restart_steps": 1 if checkpoint_restart else 0,
            "seed": seed,
            "constraint_max_iterations": constraint_max_iterations,
            "dt_ps": settings.get("dt_ps"),
            "temperature_K": settings.get("temperature_K"),
            "friction_ps^-1": settings.get("friction_ps^-1"),
            "ensemble": settings.get("ensemble"),
            "fixed_cell": settings.get("fixed_cell"),
            "restraint_k": 0.0,
            "minimize_steps": 0,
            "equilibration_steps": 0,
            "sample_interval": 1,
            "diagnostic_interval": 1,
            "initial_velocity_policy": "prepared_constraint_projected",
        },
        "case_count": 0,
        "blocked_case_count": 0,
        "cases": [],
        "continuation": {
            "status": "not_attempted",
            "warmup_to_measured": False,
            "measured_to_restart": None if not checkpoint_restart else False,
            "monotonic_step_time": False,
            "fixed_cell_preserved": False,
        },
    }
    if validation["status"] != "ready" or prepared_system is None or manifest is None:
        payload["cases"] = [
            _source_protocol_blocked_row(
                phase="validation",
                steps=0,
                blockers=tuple(str(item) for item in validation["blockers"]),
                out_dir=out_dir,
                phase_dir=out_dir,
            )
        ]
        payload["case_count"] = 1
        payload["blocked_case_count"] = 1
        _write_benchmark_payload(
            payload,
            out_dir=out_dir,
            write_json=write_json,
            write_csv=write_csv,
        )
        return payload

    existing_outputs = tuple(
        path
        for phase, phase_dir in phase_dirs.items()
        for path in (
            phase_dir / TRAJECTORY_NAME,
            phase_dir / GPCRMD_CHECKPOINT_NAME,
        )
        if (phase != "restart" or checkpoint_restart) and path.exists()
    )
    if existing_outputs and not force:
        payload["status"] = "blocked"
        payload["cases"] = [
            _source_protocol_blocked_row(
                phase="validation",
                steps=0,
                blockers=tuple(f"output_exists:{path}" for path in existing_outputs),
                out_dir=out_dir,
                phase_dir=out_dir,
            )
        ]
        payload["case_count"] = 1
        payload["blocked_case_count"] = 1
        _write_benchmark_payload(
            payload,
            out_dir=out_dir,
            write_json=write_json,
            write_csv=write_csv,
        )
        return payload

    phase_common = {
        "prepared": prepared_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "settings": settings,
        "target_id": str(resolved_target_id),
        "dynamics_id": dynamics_id,
        "seed": seed,
        "constraint_max_iterations": constraint_max_iterations,
        "out_dir": out_dir,
    }
    warmup = _run_source_protocol_phase(
        phase="warmup",
        phase_dir=phase_dirs["warmup"],
        steps=warmups,
        resume_checkpoint=None,
        **phase_common,
    )
    rows = [warmup]
    warmup_checkpoint = phase_dirs["warmup"] / GPCRMD_CHECKPOINT_NAME
    measured_checkpoint = phase_dirs["measured"] / GPCRMD_CHECKPOINT_NAME
    if warmup["status"] == "ran":
        measured = _run_source_protocol_phase(
            phase="measured",
            phase_dir=phase_dirs["measured"],
            steps=measured_steps,
            resume_checkpoint=warmup_checkpoint,
            **phase_common,
        )
    else:
        measured = _source_protocol_blocked_row(
            phase="measured",
            steps=measured_steps,
            blockers=("upstream_phase:warmup",),
            out_dir=out_dir,
            phase_dir=phase_dirs["measured"],
            resume_checkpoint=warmup_checkpoint,
        )
    rows.append(measured)

    restart = None
    if checkpoint_restart:
        if measured["status"] == "ran":
            restart = _run_source_protocol_phase(
                phase="restart",
                phase_dir=phase_dirs["restart"],
                steps=1,
                resume_checkpoint=measured_checkpoint,
                **phase_common,
            )
        else:
            restart = _source_protocol_blocked_row(
                phase="restart",
                steps=1,
                blockers=("upstream_phase:measured",),
                out_dir=out_dir,
                phase_dir=phase_dirs["restart"],
                resume_checkpoint=measured_checkpoint,
            )
        rows.append(restart)

    payload["cases"] = rows
    payload["case_count"] = len(rows)
    payload["blocked_case_count"] = sum(row["status"] != "ran" for row in rows)
    continuation = _source_protocol_continuation(warmup, measured, restart)
    payload["continuation"] = continuation
    payload["status"] = (
        "passed"
        if payload["blocked_case_count"] == 0 and continuation["status"] == "passed"
        else "failed"
    )
    _write_benchmark_payload(
        payload,
        out_dir=out_dir,
        write_json=write_json,
        write_csv=write_csv,
    )
    return payload


def _validate_source_protocol_inputs(
    *,
    prepared: Path,
    protocol_manifest: Path,
    target_id: str | None,
    registry_path: Path | None,
) -> tuple[
    dict[str, Any] | None,
    Any | None,
    dict[str, Any],
    dict[str, Any],
]:
    blockers: list[str] = []
    manifest = None
    prepared_system = None
    settings: dict[str, Any] = {}
    declared_hash = None
    rebuilt_hash = None
    resolved_target_id = target_id
    dynamics_id = None
    try:
        manifest = json.loads(protocol_manifest.read_text())
        if not isinstance(manifest, dict):
            msg = "manifest root must be an object"
            raise ValueError(msg)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        blockers.append(f"protocol_manifest:{exc}")
    if manifest is not None:
        declared_hash = manifest.get("manifest_sha256")
        payload_without_hash = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        computed_hash = _canonical_payload_hash(payload_without_hash)
        if declared_hash != computed_hash:
            blockers.append(
                "manifest_integrity:"
                f"declared={declared_hash}:computed={computed_hash}"
            )
        try:
            settings = _source_protocol_settings(manifest)
            resolved_target_id = str(manifest["workload"]["name"])
            dynamics_id = int(manifest["workload"]["dynamics_id"])
            if target_id is not None and str(target_id) != resolved_target_id:
                blockers.append(
                    "target_mismatch:"
                    f"requested={target_id}:manifest={resolved_target_id}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"protocol_semantics:{exc}")
    if manifest is not None and not blockers:
        try:
            prepared_system = load_prepared_system(prepared)
            artifact = load_prepared_mlx_artifact(prepared, require_production=True)
            targets = load_gpcrmd_targets(registry_path) if registry_path is not None else None
            target = select_gpcrmd_target(resolved_target_id, targets=targets)
            rebuilt = build_gpcrmd_mlx_workload_manifest(
                prepared_system,
                target=target,
                artifact_dir=prepared,
                source_manifest_path=manifest.get("source", {}).get("manifest_path"),
            )
            rebuilt_hash = rebuilt["manifest_sha256"]
            if rebuilt_hash != declared_hash:
                blockers.append(
                    "prepared_manifest_mismatch:"
                    f"declared={declared_hash}:rebuilt={rebuilt_hash}"
                )
            if artifact.atom_count != settings["atom_count"]:
                blockers.append(
                    "atom_count_mismatch:"
                    f"artifact={artifact.atom_count}:manifest={settings['atom_count']}"
                )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            blockers.append(f"prepared_artifact:{exc}")
    validation = {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "prepared": str(prepared),
        "protocol_manifest": str(protocol_manifest),
        "declared_manifest_sha256": declared_hash,
        "rebuilt_manifest_sha256": rebuilt_hash,
        "manifest_matches_prepared": bool(
            declared_hash is not None and declared_hash == rebuilt_hash and not blockers
        ),
        "target_id": resolved_target_id,
        "dynamics_id": dynamics_id,
        "strict_production_load": prepared_system is not None and not blockers,
        "source_settings": settings,
    }
    return manifest, prepared_system, settings, validation


def _source_protocol_settings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("kind") != "gpcrmd_mlx_workload":
        msg = "kind must be gpcrmd_mlx_workload"
        raise ValueError(msg)
    engine = _mapping_field(manifest, "engine")
    if engine.get("name") != "mlx_atomistic" or engine.get("role") != "product_runtime":
        msg = "manifest engine must identify the MLX product runtime"
        raise ValueError(msg)
    workload = _mapping_field(manifest, "workload")
    protocol = _mapping_field(manifest, "protocol")
    hmr = _mapping_field(manifest, "hydrogen_mass_repartitioning")
    constraints = _mapping_field(manifest, "constraints")
    pme = _mapping_field(manifest, "pme")
    runtime_contract = _mapping_field(manifest, "runtime_contract")
    if workload.get("ensemble") != "NVT" or protocol.get("ensemble") != "NVT":
        msg = "source-protocol execution requires NVT"
        raise ValueError(msg)
    if not workload.get("fixed_cell") or not protocol.get("fixed_cell"):
        msg = "source-protocol execution requires a fixed cell"
        raise ValueError(msg)
    dt_fs = _positive_float(protocol.get("time_step_fs"), "protocol.time_step_fs")
    temperature = _positive_float(protocol.get("temperature_K"), "protocol.temperature_K")
    friction = _nonnegative_float(
        protocol.get("langevin_friction_ps^-1"),
        "protocol.langevin_friction_ps^-1",
    )
    atom_count = int(workload.get("atom_count", 0))
    if atom_count <= 0:
        msg = "workload.atom_count must be positive"
        raise ValueError(msg)
    if int(constraints.get("count", 0)) <= 0:
        msg = "source 4 fs protocol requires represented constraints"
        raise ValueError(msg)
    if hmr.get("status") != "represented_by_masses":
        msg = "source 4 fs protocol requires HMR represented by artifact masses"
        raise ValueError(msg)
    if int(hmr.get("selected_hydrogen_count", 0)) <= 0:
        msg = "source 4 fs protocol requires a non-empty HMR selection"
        raise ValueError(msg)
    mesh_shape = tuple(int(value) for value in pme.get("mesh_shape", ()))
    if len(mesh_shape) != 3 or any(value <= 0 for value in mesh_shape):
        msg = "pme.mesh_shape must contain three positive dimensions"
        raise ValueError(msg)
    if pme.get("background_policy") != "reject_non_neutral":
        msg = "source-neutral GPCRmd PME must retain reject_non_neutral"
        raise ValueError(msg)
    expected_runtime = {
        "topology_pair_policy": "lazy",
        "eager_nonbonded_pair_limit": 0,
        "neighbor_backend": "mlx_cell_blocks",
        "neighbor_representation": "NeighborBlocks",
        "fixed_cell_pme_plan_reuse": True,
        "dense_or_tiled_fallback_allowed": False,
    }
    for name, expected in expected_runtime.items():
        if runtime_contract.get(name) != expected:
            msg = (
                f"runtime_contract.{name} must be {expected!r}; "
                f"got {runtime_contract.get(name)!r}"
            )
            raise ValueError(msg)
    trajectory = protocol.get("trajectory")
    trajectory = trajectory if isinstance(trajectory, Mapping) else {}
    return {
        "atom_count": atom_count,
        "ensemble": "NVT",
        "fixed_cell": True,
        "dt_ps": dt_fs / 1000.0,
        "temperature_K": temperature,
        "friction_ps^-1": friction,
        "source_run_steps": int(protocol.get("run_steps", 0)),
        "source_trajectory_interval_steps": trajectory.get("interval_steps"),
        "constraints_count": int(constraints["count"]),
        "hmr_status": hmr["status"],
        "hmr_selected_hydrogen_count": int(hmr["selected_hydrogen_count"]),
        "hmr_target_hydrogen_mass_da": hmr.get("target_hydrogen_mass_da"),
        "cell_lengths_angstrom": list(_mapping_field(manifest, "cell")["lengths_angstrom"]),
        "pme_mesh_shape": list(mesh_shape),
        "pme_alpha_per_angstrom": _positive_float(
            pme.get("alpha_per_angstrom"),
            "pme.alpha_per_angstrom",
        ),
        "pme_real_cutoff_angstrom": _positive_float(
            pme.get("real_cutoff_angstrom"),
            "pme.real_cutoff_angstrom",
        ),
        "pme_background_policy": pme["background_policy"],
        "runtime_contract": dict(runtime_contract),
    }


def _mapping_field(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        msg = f"{name} must be an object"
        raise ValueError(msg)
    return value


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        msg = f"{name} must be non-negative and finite"
        raise ValueError(msg)
    return number


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_source_protocol_phase(
    *,
    phase: str,
    phase_dir: Path,
    steps: int,
    resume_checkpoint: Path | None,
    prepared: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    settings: Mapping[str, Any],
    target_id: str,
    dynamics_id: Any,
    seed: int | None,
    constraint_max_iterations: int,
    out_dir: Path,
) -> dict[str, Any]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = phase_dir / TRAJECTORY_NAME
    checkpoint_path = phase_dir / GPCRMD_CHECKPOINT_NAME
    started = time.perf_counter()
    try:
        run_mlx(
            prepared,
            out=trajectory_path,
            steps=steps,
            sample_interval=1,
            dt=float(settings["dt_ps"]),
            temperature=float(settings["temperature_K"]),
            friction=float(settings["friction_ps^-1"]),
            seed=seed,
            restraint_k=0.0,
            require_production=True,
            minimize_steps=0,
            equilibration_steps=0,
            constraint_max_iterations=constraint_max_iterations,
            diagnostic_interval=1,
            eager_nonbonded_pair_limit=0,
            rescale_initial_velocities=False,
            checkpoint_out=checkpoint_path,
            resume_checkpoint=resume_checkpoint,
            metadata_overrides={
                "kind": f"gpcrmd_source_protocol_{phase}",
                "workflow": "benchmark_gpcrmd_source_protocol",
                "gpcrmd_target_id": target_id,
                "gpcrmd_dynamics_id": dynamics_id,
                "source_protocol_phase": phase,
                "source_protocol_manifest": str(manifest_path),
                "source_protocol_manifest_sha256": manifest["manifest_sha256"],
                "source_protocol_dynamics_parameters_exact": True,
                "source_stochastic_sequence_comparable": False,
            },
        )
        record = load_npz_trajectory(trajectory_path)
        checkpoint = load_simulation_checkpoint(checkpoint_path)
        input_checkpoint = (
            None
            if resume_checkpoint is None
            else load_simulation_checkpoint(resume_checkpoint)
        )
    except Exception as exc:  # benchmark evidence must retain the first runtime blocker.
        return _source_protocol_blocked_row(
            phase=phase,
            steps=steps,
            blockers=(f"mlx_run:{type(exc).__name__}:{exc}",),
            out_dir=out_dir,
            phase_dir=phase_dir,
            resume_checkpoint=resume_checkpoint,
            total_wall_s=time.perf_counter() - started,
        )
    return _source_protocol_completed_row(
        phase=phase,
        steps=steps,
        phase_dir=phase_dir,
        out_dir=out_dir,
        trajectory_path=trajectory_path,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=resume_checkpoint,
        record=record,
        checkpoint=checkpoint,
        input_checkpoint=input_checkpoint,
        settings=settings,
        total_wall_s=time.perf_counter() - started,
    )


def _source_protocol_completed_row(
    *,
    phase: str,
    steps: int,
    phase_dir: Path,
    out_dir: Path,
    trajectory_path: Path,
    checkpoint_path: Path,
    resume_checkpoint: Path | None,
    record,
    checkpoint,
    input_checkpoint,
    settings: Mapping[str, Any],
    total_wall_s: float,
) -> dict[str, Any]:
    metadata = dict(record.metadata)
    nonbonded_runtime = dict(metadata.get("nonbonded_runtime", {}))
    runtime_contract = dict(metadata.get("runtime_execution_contract", {}))
    pme_plans = list(metadata.get("pme_execution_plans", []))
    sampled_positions = np.asarray(record.sampled_positions)
    sampled_velocities = np.asarray(record.sampled_velocities)
    sampled_steps = np.asarray(record.sampled_steps)
    sampled_time = np.asarray(record.sampled_time)
    diagnostic_steps = np.asarray(record.diagnostic_steps)
    diagnostic_time = np.asarray(record.diagnostic_time)
    expected_start_step = 0 if input_checkpoint is None else int(input_checkpoint.step)
    expected_start_time = 0.0 if input_checkpoint is None else float(input_checkpoint.time)
    expected_final_step = expected_start_step + steps
    expected_final_time = expected_start_time + steps * float(settings["dt_ps"])
    expected_cell = np.asarray(settings["cell_lengths_angstrom"], dtype=np.float32)
    trajectory_cell = None if record.cell is None else np.asarray(record.cell)
    checkpoint_cell = None if checkpoint.cell is None else np.asarray(checkpoint.cell)
    trajectory_cell_matches = bool(
        trajectory_cell is not None
        and trajectory_cell.shape == expected_cell.shape
        and np.allclose(trajectory_cell, expected_cell, rtol=1e-6, atol=1e-6)
    )
    checkpoint_cell_matches = bool(
        checkpoint_cell is not None
        and checkpoint_cell.shape == expected_cell.shape
        and np.allclose(checkpoint_cell, expected_cell, rtol=1e-6, atol=1e-6)
    )
    finite_checks = {
        "positions_finite": bool(np.all(np.isfinite(sampled_positions))),
        "velocities_finite": bool(np.all(np.isfinite(sampled_velocities))),
        "forces_finite": bool(np.all(np.isfinite(np.asarray(checkpoint.forces)))),
        "potential_energy_finite": bool(
            np.all(np.isfinite(np.asarray(record.potential_energy)))
        ),
        "kinetic_energy_finite": bool(
            np.all(np.isfinite(np.asarray(record.kinetic_energy)))
        ),
        "total_energy_finite": bool(
            np.all(np.isfinite(np.asarray(record.total_energy)))
        ),
        "temperature_finite": bool(
            np.all(np.isfinite(np.asarray(record.temperature)))
        ),
        "constraint_error_finite": bool(
            np.all(np.isfinite(np.asarray(record.constraint_max_error)))
        ),
    }
    sampled_monotonic = bool(
        sampled_steps.size >= 2
        and np.all(np.diff(sampled_steps) > 0)
        and np.all(np.diff(sampled_time) > 0.0)
    )
    diagnostic_monotonic = bool(
        diagnostic_steps.size >= 2
        and np.all(np.diff(diagnostic_steps) > 0)
        and np.all(np.diff(diagnostic_time) > 0.0)
    )
    state_matches = bool(
        sampled_steps.size
        and sampled_time.size
        and int(sampled_steps[0]) == expected_start_step
        and np.isclose(float(sampled_time[0]), expected_start_time, rtol=1e-7, atol=1e-9)
        and int(sampled_steps[-1]) == expected_final_step
        and np.isclose(float(sampled_time[-1]), expected_final_time, rtol=1e-7, atol=1e-9)
        and checkpoint.step == expected_final_step
        and np.isclose(checkpoint.time, expected_final_time, rtol=1e-7, atol=1e-9)
    )
    hmr_state = dict(metadata.get("hydrogen_mass_repartitioning", {}))
    checkpoint_hmr = checkpoint.hmr_state
    hmr_preserved = bool(
        hmr_state.get("status") == settings["hmr_status"]
        and checkpoint_hmr.get("status") == settings["hmr_status"]
    )
    pme_plan_count = len(pme_plans)
    pme_build_count = sum(int(plan.get("build_count", 0)) for plan in pme_plans)
    pme_reuse_count = sum(int(plan.get("reuse_count", 0)) for plan in pme_plans)
    runtime_contract_matches = bool(
        runtime_contract.get("fixed_cell") is True
        and runtime_contract.get("topology_pair_policy") == "lazy"
        and runtime_contract.get("eager_nonbonded_pair_limit") == 0
        and runtime_contract.get("neighbor_backend") == "mlx_cell_blocks"
        and runtime_contract.get("neighbor_representation") == "NeighborBlocks"
        and runtime_contract.get("shared_direct_space_neighbors") is True
        and runtime_contract.get("dense_or_tiled_fallback_used") is False
    )
    checks = {
        **finite_checks,
        "sampled_step_time_monotonic": sampled_monotonic,
        "diagnostic_step_time_monotonic": diagnostic_monotonic,
        "expected_state_bounds": state_matches,
        "trajectory_cell_matches": trajectory_cell_matches,
        "checkpoint_cell_matches": checkpoint_cell_matches,
        "hmr_preserved": hmr_preserved,
        "runtime_contract_matches": runtime_contract_matches,
        "one_pme_plan": pme_plan_count == 1 and pme_build_count == 1,
        "pme_plan_reused": pme_reuse_count > 0,
        "resume_skipped_minimization": metadata.get("minimize_steps") == 0,
        "resume_skipped_equilibration": metadata.get("equilibration_steps") == 0,
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    row = _source_protocol_row_template(
        phase=phase,
        status="ran" if not blockers else "failed",
        steps=steps,
        blockers=blockers,
        out_dir=out_dir,
        phase_dir=phase_dir,
        resume_checkpoint=resume_checkpoint,
        total_wall_s=total_wall_s,
    )
    run_wall_s = metadata.get("elapsed_wall_seconds")
    row.update(
        {
            "dt_ps": float(settings["dt_ps"]),
            "duration_ps": steps * float(settings["dt_ps"]),
            "start_step": int(sampled_steps[0]),
            "final_step": int(sampled_steps[-1]),
            "start_time_ps": float(sampled_time[0]),
            "final_time_ps": float(sampled_time[-1]),
            "sampled_step_time_monotonic": sampled_monotonic,
            "diagnostic_step_time_monotonic": diagnostic_monotonic,
            "atom_count": int(sampled_positions.shape[1]),
            "frame_count": int(sampled_positions.shape[0]),
            "diagnostic_count": int(diagnostic_steps.shape[0]),
            "trajectory_path": _portable_output_path(trajectory_path, out_dir),
            "checkpoint_path": _portable_output_path(checkpoint_path, out_dir),
            "trajectory_loaded": True,
            "checkpoint_loaded": True,
            "checkpoint_atom_count": int(checkpoint.positions.shape[0]),
            "checkpoint_step": int(checkpoint.step),
            "checkpoint_time_ps": float(checkpoint.time),
            "checkpoint_rng_step_offset": int(
                checkpoint.thermostat.get("rng_step_offset", checkpoint.step)
            ),
            "checkpoint_resumed_from": checkpoint.metadata.get("resumed_from"),
            "fixed_cell": bool(metadata.get("fixed_cell")),
            "cell_lengths_angstrom": expected_cell.astype(float).tolist(),
            "trajectory_cell_matches": trajectory_cell_matches,
            "checkpoint_cell_matches": checkpoint_cell_matches,
            "hmr_status": hmr_state.get("status"),
            "hmr_preserved": hmr_preserved,
            "neighbor_backend": nonbonded_runtime.get("backend"),
            "neighbor_representation": runtime_contract.get(
                "neighbor_representation"
            ),
            "neighbor_fallback_reason": nonbonded_runtime.get("fallback_reason"),
            "topology_pair_policy": runtime_contract.get("topology_pair_policy"),
            "eager_nonbonded_pair_limit": runtime_contract.get(
                "eager_nonbonded_pair_limit"
            ),
            "shared_direct_space_neighbors": runtime_contract.get(
                "shared_direct_space_neighbors"
            ),
            "dense_or_tiled_fallback_used": runtime_contract.get(
                "dense_or_tiled_fallback_used"
            ),
            "compact_pair_count": nonbonded_runtime.get("compact_pair_count"),
            "candidate_count": nonbonded_runtime.get("candidate_count"),
            "candidate_waste_count": nonbonded_runtime.get(
                "candidate_waste_count"
            ),
            "candidate_waste_fraction": nonbonded_runtime.get(
                "candidate_waste_fraction"
            ),
            "neighbor_update_wall_s": nonbonded_runtime.get(
                "neighbor_update_wall_seconds"
            ),
            "neighbor_rebuild_wall_s": nonbonded_runtime.get(
                "neighbor_rebuild_wall_seconds"
            ),
            "force_eval_wall_s": nonbonded_runtime.get(
                "force_evaluation_wall_seconds"
            ),
            "pme_execution_plan_count": pme_plan_count,
            "pme_execution_plan_build_count": pme_build_count,
            "pme_execution_plan_reuse_count": pme_reuse_count,
            "pme_execution_plan_fingerprints": [
                plan.get("fingerprint") for plan in pme_plans
            ],
            "run_wall_s": None if run_wall_s is None else float(run_wall_s),
            "integration_steps_per_s": metadata.get(
                "integration_steps_per_second"
            ),
            "ps_per_s": metadata.get("simulated_ps_per_wall_second"),
            "max_constraint_error_A": max_float(record.constraint_max_error),
            "max_rss_mb": max_rss_mb(),
            **finite_checks,
            "checks": checks,
        }
    )
    return row


def _source_protocol_blocked_row(
    *,
    phase: str,
    steps: int,
    blockers: tuple[str, ...],
    out_dir: Path,
    phase_dir: Path,
    resume_checkpoint: Path | None = None,
    total_wall_s: float | None = None,
) -> dict[str, Any]:
    return _source_protocol_row_template(
        phase=phase,
        status="blocked",
        steps=steps,
        blockers=blockers,
        out_dir=out_dir,
        phase_dir=phase_dir,
        resume_checkpoint=resume_checkpoint,
        total_wall_s=total_wall_s,
    )


def _source_protocol_row_template(
    *,
    phase: str,
    status: str,
    steps: int,
    blockers: tuple[str, ...],
    out_dir: Path,
    phase_dir: Path,
    resume_checkpoint: Path | None,
    total_wall_s: float | None,
) -> dict[str, Any]:
    trajectory_path = phase_dir / TRAJECTORY_NAME
    checkpoint_path = phase_dir / GPCRMD_CHECKPOINT_NAME
    return {
        "phase": phase,
        "status": status,
        "blockers": ";".join(blockers),
        "steps": steps,
        "dt_ps": None,
        "duration_ps": None,
        "start_step": None,
        "final_step": None,
        "start_time_ps": None,
        "final_time_ps": None,
        "sampled_step_time_monotonic": None,
        "diagnostic_step_time_monotonic": None,
        "atom_count": None,
        "frame_count": None,
        "diagnostic_count": None,
        "output_dir": _portable_output_path(phase_dir, out_dir),
        "trajectory_path": _portable_output_path(trajectory_path, out_dir),
        "checkpoint_path": _portable_output_path(checkpoint_path, out_dir),
        "resume_checkpoint": (
            None
            if resume_checkpoint is None
            else _portable_output_path(resume_checkpoint, out_dir)
        ),
        "trajectory_loaded": False,
        "checkpoint_loaded": False,
        "checkpoint_atom_count": None,
        "checkpoint_step": None,
        "checkpoint_time_ps": None,
        "checkpoint_rng_step_offset": None,
        "checkpoint_resumed_from": None,
        "fixed_cell": None,
        "cell_lengths_angstrom": None,
        "trajectory_cell_matches": None,
        "checkpoint_cell_matches": None,
        "hmr_status": None,
        "hmr_preserved": None,
        "neighbor_backend": None,
        "neighbor_representation": None,
        "neighbor_fallback_reason": None,
        "topology_pair_policy": None,
        "eager_nonbonded_pair_limit": None,
        "shared_direct_space_neighbors": None,
        "dense_or_tiled_fallback_used": None,
        "compact_pair_count": None,
        "candidate_count": None,
        "candidate_waste_count": None,
        "candidate_waste_fraction": None,
        "neighbor_update_wall_s": None,
        "neighbor_rebuild_wall_s": None,
        "force_eval_wall_s": None,
        "pme_execution_plan_count": None,
        "pme_execution_plan_build_count": None,
        "pme_execution_plan_reuse_count": None,
        "pme_execution_plan_fingerprints": None,
        "total_wall_s": total_wall_s,
        "run_wall_s": None,
        "integration_steps_per_s": None,
        "ps_per_s": None,
        "max_constraint_error_A": None,
        "max_rss_mb": max_rss_mb(),
        "positions_finite": None,
        "velocities_finite": None,
        "forces_finite": None,
        "potential_energy_finite": None,
        "kinetic_energy_finite": None,
        "total_energy_finite": None,
        "temperature_finite": None,
        "constraint_error_finite": None,
        "checks": {},
    }


def _source_protocol_continuation(
    warmup: Mapping[str, Any],
    measured: Mapping[str, Any],
    restart: Mapping[str, Any] | None,
) -> dict[str, Any]:
    warmup_to_measured = bool(
        warmup.get("status") == "ran"
        and measured.get("status") == "ran"
        and measured.get("start_step") == warmup.get("final_step")
        and np.isclose(
            float(measured.get("start_time_ps")),
            float(warmup.get("final_time_ps")),
            rtol=1e-7,
            atol=1e-9,
        )
    )
    measured_to_restart = None
    if restart is not None:
        measured_to_restart = bool(
            measured.get("status") == "ran"
            and restart.get("status") == "ran"
            and restart.get("start_step") == measured.get("final_step")
            and np.isclose(
                float(restart.get("start_time_ps")),
                float(measured.get("final_time_ps")),
                rtol=1e-7,
                atol=1e-9,
            )
        )
    rows = [warmup, measured, *([] if restart is None else [restart])]
    all_ran = all(row.get("status") == "ran" for row in rows)
    monotonic = bool(
        all_ran
        and warmup_to_measured
        and (measured_to_restart is not False)
        and all(row.get("sampled_step_time_monotonic") is True for row in rows)
    )
    fixed_cell = bool(
        all_ran
        and all(
            row.get("fixed_cell") is True
            and row.get("trajectory_cell_matches") is True
            and row.get("checkpoint_cell_matches") is True
            for row in rows
        )
    )
    passed = bool(all_ran and monotonic and fixed_cell)
    return {
        "status": "passed" if passed else "failed",
        "warmup_to_measured": warmup_to_measured,
        "measured_to_restart": measured_to_restart,
        "monotonic_step_time": monotonic,
        "fixed_cell_preserved": fixed_cell,
        "warmup_checkpoint": warmup.get("checkpoint_path"),
        "measured_checkpoint": measured.get("checkpoint_path"),
        "restart_checkpoint": None if restart is None else restart.get("checkpoint_path"),
    }


def _portable_output_path(path: Path, out_dir: Path) -> str:
    try:
        return str(path.relative_to(out_dir))
    except ValueError:
        return str(path)


def _write_benchmark_payload(
    payload: Mapping[str, Any],
    *,
    out_dir: Path,
    write_json: bool,
    write_csv: bool,
) -> None:
    safe_payload = _json_safe(payload)
    if write_json:
        (out_dir / GPCRMD_BENCHMARK_JSON_NAME).write_text(
            json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    if write_csv:
        _write_csv(
            out_dir / GPCRMD_BENCHMARK_CSV_NAME,
            list(safe_payload["cases"]),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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
    nonbonded_runtime = dict(metadata.get("nonbonded_runtime", {}))
    neighbor_rebuild_wall_s = float(
        nonbonded_runtime.get("neighbor_rebuild_wall_seconds") or 0.0
    )
    force_eval_wall_s = float(
        nonbonded_runtime.get("force_evaluation_wall_seconds") or 0.0
    )
    decomposed_wall_s = neighbor_rebuild_wall_s + force_eval_wall_s
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
        "neighbor_backend": nonbonded_runtime.get("backend"),
        "neighbor_fallback_reason": nonbonded_runtime.get("fallback_reason"),
        "compact_pair_count": nonbonded_runtime.get("compact_pair_count"),
        "candidate_count": nonbonded_runtime.get("candidate_count"),
        "candidate_waste_count": nonbonded_runtime.get("candidate_waste_count"),
        "candidate_waste_fraction": nonbonded_runtime.get("candidate_waste_fraction"),
        "compaction_backend": nonbonded_runtime.get("compaction_backend"),
        "neighbor_update_wall_s": nonbonded_runtime.get("neighbor_update_wall_seconds"),
        "neighbor_rebuild_wall_s": neighbor_rebuild_wall_s,
        "force_eval_wall_s": force_eval_wall_s,
        "force_eval_ms_per_step": force_eval_wall_s * 1000.0 / max(steps, 1),
        "neighbor_build_fraction": (
            neighbor_rebuild_wall_s / decomposed_wall_s if decomposed_wall_s > 0.0 else None
        ),
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
        "neighbor_backend": None,
        "neighbor_fallback_reason": None,
        "compact_pair_count": None,
        "candidate_count": None,
        "candidate_waste_count": None,
        "candidate_waste_fraction": None,
        "compaction_backend": None,
        "neighbor_update_wall_s": None,
        "neighbor_rebuild_wall_s": None,
        "force_eval_wall_s": None,
        "force_eval_ms_per_step": None,
        "neighbor_build_fraction": None,
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


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_string_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    """Run GPCRmd benchmark rows from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--protocol-manifest", type=Path, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--measured-steps", type=int, default=None)
    parser.add_argument("--checkpoint-restart", action="store_true")
    parser.add_argument("--durations-ps", default="0.01")
    parser.add_argument("--electrostatics-modes", default="artifact")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    parser.add_argument("--minimize-steps", type=int, default=0)
    parser.add_argument("--equilibration-steps", type=int, default=0)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.protocol_manifest is not None:
        if args.prepared is None:
            parser.error("--protocol-manifest requires --prepared")
        payload = benchmark_gpcrmd_source_protocol(
            out=args.out,
            target_id=args.target_id,
            registry_path=args.registry_path,
            prepared=args.prepared,
            protocol_manifest=args.protocol_manifest,
            warmups=1 if args.warmups is None else args.warmups,
            measured_steps=2 if args.measured_steps is None else args.measured_steps,
            checkpoint_restart=args.checkpoint_restart,
            seed=args.seed,
            constraint_max_iterations=args.constraint_max_iterations,
            force=args.force,
        )
    else:
        if (
            args.warmups is not None
            or args.measured_steps is not None
            or args.checkpoint_restart
        ):
            parser.error(
                "--warmups, --measured-steps, and --checkpoint-restart require "
                "--protocol-manifest"
            )
        payload = benchmark_gpcrmd_mlx(
            out=args.out,
            target_id=args.target_id,
            cache=args.cache,
            registry_path=args.registry_path,
            prepared=args.prepared,
            durations_ps=_parse_float_tuple(args.durations_ps),
            electrostatics_modes=_parse_string_tuple(args.electrostatics_modes),
            dt=args.dt,
            sample_interval=args.sample_interval,
            temperature=args.temperature,
            friction=args.friction,
            seed=args.seed,
            restraint_k=args.restraint_k,
            minimize_steps=args.minimize_steps,
            equilibration_steps=args.equilibration_steps,
            constraint_max_iterations=args.constraint_max_iterations,
            diagnostic_interval=args.diagnostic_interval,
            force=args.force,
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"cases={payload['case_count']} blocked={payload['blocked_case_count']} "
            f"out={args.out}"
        )


__all__ = [
    "GPCRMD_BENCHMARK_CSV_NAME",
    "GPCRMD_BENCHMARK_JSON_NAME",
    "GPCRMD_CHECKPOINT_NAME",
    "benchmark_gpcrmd_mlx",
    "benchmark_gpcrmd_source_protocol",
    "main",
]


if __name__ == "__main__":
    main()
