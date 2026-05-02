"""Notebook helpers for the active GPCRmd MLX trajectory workflow."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlx_atomistic.io import TrajectoryRecord, load_npz_trajectory

from .motion_analysis import ProcessedTrajectory

GPCRMD_FAST_RUNTIME_ATOM_LIMIT = 50_000
GPCRMD_MIN_NEIGHBOR_SKIN = 1.0


@dataclass(frozen=True)
class GPCRmdMLXBundle:
    """GPCRmd prepared artifact plus the saved MLX trajectory or blockers."""

    prepared_dir: Path
    trajectory_path: Path
    report_path: Path
    processed_trajectory: ProcessedTrajectory | None
    diagnostics: pd.DataFrame
    metadata: dict[str, Any]
    run_report: dict[str, Any]
    blockers: tuple[str, ...]
    generated_trajectory: bool

    @property
    def runnable(self) -> bool:
        return self.processed_trajectory is not None and not self.blockers

    def blocker_json(self) -> str:
        payload = {
            "status": self.run_report.get("status"),
            "blockers": list(self.blockers),
            "trajectory_path": self.run_report.get("trajectory_path"),
            "planned_trajectory_path": self.run_report.get("planned_trajectory_path"),
            "run_report_path": str(self.report_path),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def ensure_gpcrmd_mlx_bundle(
    *,
    out_dir: str | Path,
    target_id: str | None = None,
    cache: str | Path | None = None,
    registry_path: str | Path | None = None,
    steps: int = 2000,
    dt: float = 0.001,
    sample_interval: int = 10,
    temperature: float = 300.0,
    friction: float = 1.0,
    seed: int | None = 7,
    restraint_k: float = 5.0,
    minimize_steps: int = 50,
    equilibration_steps: int = 100,
    constraint_max_iterations: int = 4,
    diagnostic_interval: int | None = None,
    electrostatics: str = "short-range-prototype",
    force: bool = False,
) -> GPCRmdMLXBundle:
    """Load or run the GPCRmd MLX trajectory consumed by the notebook."""

    prep_io, prep_runner = _reload_active_prep_modules()
    out_path = Path(out_dir)
    trajectory_path = out_path / prep_runner.TRAJECTORY_NAME
    report_path = out_path / prep_runner.GPCRMD_RUN_REPORT_NAME

    run_report: dict[str, Any] | None = None
    generated_trajectory = False
    if not force and trajectory_path.exists() and report_path.exists():
        run_report = _load_json_report(report_path)
        if not _report_allows_existing_trajectory(run_report):
            return _blocked_bundle(out_path, trajectory_path, report_path, run_report)
        requested_target_blocker = _requested_target_blocker(
            run_report,
            requested_target_id=target_id,
        )
        if requested_target_blocker is not None:
            return _blocked_bundle(
                out_path,
                trajectory_path,
                report_path,
                {
                    **run_report,
                    "status": "blocked",
                    "blockers": [requested_target_blocker],
                    "trajectory_path": None,
                },
            )
    else:
        run_report = prep_runner.run_gpcrmd_mlx(
            out=out_path,
            target_id=target_id,
            cache=cache,
            registry_path=registry_path,
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
            electrostatics=electrostatics,
            force=force,
        )
        generated_trajectory = bool(run_report.get("trajectory_written"))

    if run_report.get("status") != "ran":
        return _blocked_bundle(out_path, trajectory_path, report_path, run_report)

    trajectory_report_path = (
        _path_from_report(run_report, "trajectory_path", report_path=report_path)
        or trajectory_path
    )
    prepared_dir = (
        _path_from_report(run_report, "prepared_artifact_path", report_path=report_path)
        or out_path
    )
    if not trajectory_report_path.exists():
        return _blocked_bundle(
            out_path,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"missing_mlx_trajectory:{trajectory_report_path}"],
                "trajectory_path": None,
            },
        )

    try:
        record = load_npz_trajectory(trajectory_report_path)
    except Exception as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [
                    "trajectory_artifact_load_failed:"
                    f"{trajectory_report_path}:{type(exc).__name__}:{exc}"
                ],
                "trajectory_path": None,
            },
        )
    validation_blockers = _trajectory_metadata_blockers(record.metadata)
    if validation_blockers:
        return _blocked_bundle(
            out_path,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": list(validation_blockers),
                "trajectory_path": None,
            },
        )

    try:
        prepared = prep_io.load_prepared_system(prepared_dir)
    except FileNotFoundError as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"missing_prepared_artifact:{exc}"],
                "trajectory_path": None,
            },
        )
    except (KeyError, ValueError) as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"prepared_artifact:{exc}"],
                "trajectory_path": None,
            },
        )
    artifact_blockers = _prepared_artifact_blockers(
        prepared,
        record=record,
        run_report=run_report,
    )
    if artifact_blockers:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": list(artifact_blockers),
                "trajectory_path": None,
            },
        )
    trajectory = _processed_trajectory_from_record(
        record,
        prepared=prepared,
        run_report=run_report,
    )
    return GPCRmdMLXBundle(
        prepared_dir=prepared_dir,
        trajectory_path=trajectory_report_path,
        report_path=report_path,
        processed_trajectory=trajectory,
        diagnostics=_diagnostics_dataframe(record),
        metadata=dict(record.metadata),
        run_report=run_report,
        blockers=(),
        generated_trajectory=generated_trajectory,
    )


def load_gpcrmd_mlx_artifact(
    *,
    out_dir: str | Path,
    target_id: str | None = None,
) -> GPCRmdMLXBundle:
    """Load an existing GPCRmd MLX artifact without running MD."""

    prep_io, prep_runner = _reload_active_prep_modules()
    out_path = Path(out_dir)
    trajectory_path = out_path / prep_runner.TRAJECTORY_NAME
    report_path = out_path / prep_runner.GPCRMD_RUN_REPORT_NAME
    if not trajectory_path.exists() or not report_path.exists():
        missing = [str(path) for path in (trajectory_path, report_path) if not path.exists()]
        return _blocked_bundle(
            out_path,
            trajectory_path,
            report_path,
            {
                "status": "blocked",
                "blockers": [f"missing_mlx_artifact:{path}" for path in missing],
                "trajectory_path": None,
            },
        )

    run_report = _load_json_report(report_path)
    if not _report_allows_existing_trajectory(run_report):
        return _blocked_bundle(out_path, trajectory_path, report_path, run_report)
    requested_target_blocker = _requested_target_blocker(
        run_report,
        requested_target_id=target_id,
    )
    if requested_target_blocker is not None:
        return _blocked_bundle(
            out_path,
            trajectory_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [requested_target_blocker],
                "trajectory_path": None,
            },
        )
    return _bundle_from_verified_report(
        prep_io=prep_io,
        out_path=out_path,
        trajectory_path=trajectory_path,
        report_path=report_path,
        run_report=run_report,
        generated_trajectory=False,
    )


def ensure_mlx_ligand_receptor_bundle(**kwargs) -> GPCRmdMLXBundle:
    """Backward-compatible notebook entrypoint for the GPCRmd MLX workflow."""

    return ensure_gpcrmd_mlx_bundle(**kwargs)


def _reload_active_prep_modules():
    import atomistic_prep.io as prep_io
    import atomistic_prep.runner as prep_runner
    import atomistic_prep.schema as prep_schema

    importlib.reload(prep_schema)
    prep_io = importlib.reload(prep_io)
    prep_runner = importlib.reload(prep_runner)
    return prep_io, prep_runner


def _load_json_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _report_allows_existing_trajectory(report: dict[str, Any]) -> bool:
    metadata = report.get("run_metadata")
    return (
        report.get("status") == "ran"
        and report.get("trajectory_written") is True
        and isinstance(metadata, dict)
        and not _trajectory_metadata_blockers(metadata)
    )


def _requested_target_blocker(
    report: dict[str, Any],
    *,
    requested_target_id: str | None,
) -> str | None:
    requested = _normalized_optional(requested_target_id)
    if requested is None:
        return None
    metadata = report.get("run_metadata")
    report_target = _normalized_optional(report.get("target_id"))
    metadata_target = (
        None
        if not isinstance(metadata, dict)
        else _normalized_optional(metadata.get("gpcrmd_target_id"))
    )
    existing = report_target or metadata_target
    if existing == requested:
        return None
    return f"existing_gpcrmd_target_mismatch:requested={requested}:existing={existing}"


def _blocked_bundle(
    prepared_dir: Path,
    trajectory_path: Path,
    report_path: Path,
    run_report: dict[str, Any],
) -> GPCRmdMLXBundle:
    blockers = tuple(str(item) for item in run_report.get("blockers", ()))
    return GPCRmdMLXBundle(
        prepared_dir=prepared_dir,
        trajectory_path=trajectory_path,
        report_path=report_path,
        processed_trajectory=None,
        diagnostics=pd.DataFrame(),
        metadata=dict(run_report.get("run_metadata") or {}),
        run_report=run_report,
        blockers=blockers,
        generated_trajectory=False,
    )


def _path_from_report(
    report: dict[str, Any],
    key: str,
    *,
    report_path: Path,
) -> Path | None:
    value = report.get(key)
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    workspace_path = _workspace_root() / path
    if workspace_path.exists():
        return workspace_path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    report_sibling = report_path.parent / path.name
    if report_sibling.exists():
        return report_sibling
    return workspace_path


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def _bundle_from_verified_report(
    *,
    prep_io,
    out_path: Path,
    trajectory_path: Path,
    report_path: Path,
    run_report: dict[str, Any],
    generated_trajectory: bool,
) -> GPCRmdMLXBundle:
    trajectory_report_path = (
        _path_from_report(run_report, "trajectory_path", report_path=report_path)
        or trajectory_path
    )
    prepared_dir = (
        _path_from_report(run_report, "prepared_artifact_path", report_path=report_path)
        or out_path
    )
    if not trajectory_report_path.exists():
        return _blocked_bundle(
            out_path,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"missing_mlx_trajectory:{trajectory_report_path}"],
                "trajectory_path": None,
            },
        )
    try:
        record = load_npz_trajectory(trajectory_report_path)
    except Exception as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [
                    "trajectory_artifact_load_failed:"
                    f"{trajectory_report_path}:{type(exc).__name__}:{exc}"
                ],
                "trajectory_path": None,
            },
        )
    validation_blockers = _trajectory_metadata_blockers(record.metadata)
    if validation_blockers:
        return _blocked_bundle(
            out_path,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": list(validation_blockers),
                "trajectory_path": None,
            },
        )
    try:
        prepared = prep_io.load_prepared_system(prepared_dir)
    except FileNotFoundError as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"missing_prepared_artifact:{exc}"],
                "trajectory_path": None,
            },
        )
    except (KeyError, ValueError) as exc:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": [f"prepared_artifact:{exc}"],
                "trajectory_path": None,
            },
        )
    artifact_blockers = _prepared_artifact_blockers(
        prepared,
        record=record,
        run_report=run_report,
    )
    if artifact_blockers:
        return _blocked_bundle(
            prepared_dir,
            trajectory_report_path,
            report_path,
            {
                **run_report,
                "status": "blocked",
                "blockers": list(artifact_blockers),
                "trajectory_path": None,
            },
        )
    trajectory = _processed_trajectory_from_record(
        record,
        prepared=prepared,
        run_report=run_report,
    )
    return GPCRmdMLXBundle(
        prepared_dir=prepared_dir,
        trajectory_path=trajectory_report_path,
        report_path=report_path,
        processed_trajectory=trajectory,
        diagnostics=_diagnostics_dataframe(record),
        metadata=dict(record.metadata),
        run_report=run_report,
        blockers=(),
        generated_trajectory=generated_trajectory,
    )


def _trajectory_metadata_blockers(metadata: dict[str, Any]) -> tuple[str, ...]:
    blockers: list[str] = []
    if metadata.get("engine") != "mlx_atomistic":
        blockers.append(f"trajectory_engine:{metadata.get('engine')!r}")
    if metadata.get("source") != "mlx_atomistic":
        blockers.append(f"trajectory_source:{metadata.get('source')!r}")
    if metadata.get("kind") != "gpcrmd_mlx_nvt":
        blockers.append(f"trajectory_kind:{metadata.get('kind')!r}")
    if metadata.get("workflow") != "run_gpcrmd_mlx":
        blockers.append(f"trajectory_workflow:{metadata.get('workflow')!r}")
    return tuple(blockers)


def _prepared_artifact_blockers(
    prepared,
    *,
    record: TrajectoryRecord,
    run_report: dict[str, Any],
) -> tuple[str, ...]:
    blockers: list[str] = []
    source = dict(prepared.metadata.source)
    record_metadata = record.metadata
    prepared_target = _normalized_optional(source.get("gpcrmd_target_id"))
    prepared_dynamics = _normalized_optional(source.get("gpcrmd_dynamics_id"))
    trajectory_target = _normalized_optional(record_metadata.get("gpcrmd_target_id"))
    trajectory_dynamics = _normalized_optional(record_metadata.get("gpcrmd_dynamics_id"))
    report_target = _normalized_optional(run_report.get("target_id"))
    report_dynamics = _normalized_optional(run_report.get("dynamics_id"))

    blockers.extend(
        _identity_mismatch_blockers(
            "target_id",
            "target",
            prepared_target,
            trajectory=trajectory_target,
            report=report_target,
        )
    )
    blockers.extend(
        _identity_mismatch_blockers(
            "dynamics_id",
            "dynamics",
            prepared_dynamics,
            trajectory=trajectory_dynamics,
            report=report_dynamics,
        )
    )

    positions = np.asarray(record.sampled_positions)
    trajectory_atom_count = int(positions.shape[1]) if positions.ndim == 3 else None
    prepared_atom_count = int(prepared.atom_count)
    if trajectory_atom_count is None:
        blockers.append(f"trajectory_positions_shape:{positions.shape}")
    elif prepared_atom_count != trajectory_atom_count:
        blockers.append(
            "prepared_artifact_atom_count_mismatch:"
            f"prepared={prepared_atom_count}:trajectory={trajectory_atom_count}"
        )

    for name in ["ligand_mask", "receptor_mask", "water_mask", "ion_mask", "lipid_mask"]:
        mask = np.asarray(getattr(prepared, name))
        if mask.shape != (prepared_atom_count,):
            blockers.append(
                f"prepared_artifact_mask_length_mismatch:{name}:"
                f"shape={mask.shape}:atom_count={prepared_atom_count}"
            )
    blockers.extend(_runtime_route_blockers(record_metadata, atom_count=prepared_atom_count))
    return tuple(blockers)


def _runtime_route_blockers(
    metadata: dict[str, Any],
    *,
    atom_count: int,
) -> tuple[str, ...]:
    if atom_count <= GPCRMD_FAST_RUNTIME_ATOM_LIMIT:
        return ()
    runtime = metadata.get("nonbonded_runtime")
    if not isinstance(runtime, dict):
        return ("runtime_nonbonded_route_missing",)
    blockers: list[str] = []
    backend = runtime.get("backend")
    if backend != "periodic_cell_list":
        blockers.append(f"runtime_nonbonded_backend:{backend!r}")
    try:
        skin = float(runtime.get("skin"))
    except (TypeError, ValueError):
        blockers.append(f"runtime_neighbor_skin:{runtime.get('skin')!r}")
    else:
        if skin < GPCRMD_MIN_NEIGHBOR_SKIN:
            blockers.append(
                f"runtime_neighbor_skin:actual={skin:g}:minimum={GPCRMD_MIN_NEIGHBOR_SKIN:g}"
            )
    return tuple(blockers)


def _normalized_optional(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


def _identity_mismatch_blockers(
    field: str,
    label: str,
    prepared_value: str | None,
    *,
    trajectory: str | None,
    report: str | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if trajectory is not None and prepared_value != trajectory:
        blockers.append(
            f"prepared_artifact_{label}_mismatch:"
            f"prepared={prepared_value}:trajectory={trajectory}"
        )
    if report is not None and prepared_value != report:
        blockers.append(
            f"prepared_artifact_{label}_mismatch:"
            f"prepared={prepared_value}:report={report}"
        )
    if trajectory is not None and report is not None and trajectory != report:
        blockers.append(
            f"trajectory_report_{label}_mismatch:trajectory={trajectory}:report={report}"
        )
    if field == "target_id" and prepared_value is None and (trajectory is None or report is None):
        blockers.append("prepared_artifact_target_missing")
    return tuple(blockers)


def _processed_trajectory_from_record(
    record: TrajectoryRecord,
    *,
    prepared,
    run_report: dict[str, Any],
) -> ProcessedTrajectory:
    metadata = record.metadata
    prepared_source = dict(prepared.metadata.source)
    target_id = metadata.get("gpcrmd_target_id") or run_report.get("target_id")
    dynamics_id = metadata.get("gpcrmd_dynamics_id") or run_report.get("dynamics_id")
    return ProcessedTrajectory(
        positions=np.asarray(record.sampled_positions, dtype=np.float32),
        time_ps=np.asarray(record.sampled_time, dtype=np.float32),
        symbols=np.asarray(prepared.symbols).astype(str),
        atom_names=np.asarray(prepared.atom_names).astype(str),
        residue_names=np.asarray(prepared.residue_names).astype(str),
        residue_ids=np.asarray(prepared.residue_ids, dtype=np.int32),
        segment_ids=np.asarray(prepared.chain_ids).astype(str),
        ligand_indices=np.flatnonzero(np.asarray(prepared.ligand_mask, dtype=bool)).astype(
            np.int32
        ),
        receptor_indices=np.flatnonzero(
            np.asarray(prepared.receptor_mask, dtype=bool)
        ).astype(np.int32),
        water_indices=np.flatnonzero(np.asarray(prepared.water_mask, dtype=bool)).astype(
            np.int32
        ),
        ion_indices=np.flatnonzero(np.asarray(prepared.ion_mask, dtype=bool)).astype(
            np.int32
        ),
        lipid_indices=np.flatnonzero(np.asarray(prepared.lipid_mask, dtype=bool)).astype(
            np.int32
        ),
        cell_lengths_A=(
            None if record.cell is None else np.asarray(record.cell, dtype=np.float32)
        ),
        source={
            "kind": "gpcrmd_mlx_nvt",
            "engine": "mlx_atomistic",
            "workflow": "run_gpcrmd_mlx",
            "target_id": target_id,
            "dynamics_id": dynamics_id,
            "pdb_id": prepared_source.get("pdb_id"),
            "title": "MLX-generated GPCRmd short NVT",
            "parameter_source": prepared.metadata.parameter_source,
            "electrostatics_model": metadata.get("electrostatics_model"),
            "ensemble": metadata.get("ensemble"),
            "proof_mode": metadata.get("proof_mode"),
            "barostat_status": metadata.get("barostat_status"),
            "source": "mlx_atomistic",
            "reference_role": "comparison_only",
            "trajectory_public": False,
        },
    )


def _diagnostics_dataframe(record: TrajectoryRecord) -> pd.DataFrame:
    potential = np.asarray(record.potential_energy, dtype=np.float32)
    kinetic = np.asarray(record.kinetic_energy, dtype=np.float32)
    total = np.asarray(record.total_energy, dtype=np.float32)
    temperature = np.asarray(record.temperature, dtype=np.float32)
    pressure = np.asarray(record.pressure, dtype=np.float32)
    constraint_error = np.asarray(record.constraint_max_error, dtype=np.float32)
    return pd.DataFrame(
        {
            "step": np.asarray(record.diagnostic_steps, dtype=np.int32),
            "time_ps": np.asarray(record.diagnostic_time, dtype=np.float32),
            "potential_energy_kJ_mol": potential,
            "kinetic_energy_kJ_mol": kinetic,
            "total_energy_kJ_mol": total,
            "temperature_K": temperature,
            "pressure": pressure,
            "constraint_max_error_A": constraint_error,
            "pair_count": np.asarray(record.pair_count, dtype=np.int32),
            "rebuild_count": np.asarray(record.rebuild_count, dtype=np.int32),
            "energy_drift_kJ_mol": total - total[0],
        }
    )


__all__ = [
    "GPCRmdMLXBundle",
    "ensure_gpcrmd_mlx_bundle",
    "ensure_mlx_ligand_receptor_bundle",
    "load_gpcrmd_mlx_artifact",
]
