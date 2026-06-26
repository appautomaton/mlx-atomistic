"""Structure and trajectory I/O helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.md import ForceTerm, ReporterEvent, SimulationState


@dataclass(frozen=True)
class TrajectoryRecord:
    """Native loaded trajectory record."""

    sampled_positions: np.ndarray
    sampled_velocities: np.ndarray
    sampled_steps: np.ndarray
    sampled_time: np.ndarray
    diagnostic_steps: np.ndarray
    diagnostic_time: np.ndarray
    potential_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray
    potential_energy_by_term: dict[str, np.ndarray]
    temperature: np.ndarray
    pair_count: np.ndarray
    rebuild_count: np.ndarray
    constraint_max_error: np.ndarray
    symbols: tuple[str, ...]
    cell: np.ndarray | None
    metadata: dict[str, Any]
    virial_tensor: np.ndarray | None = None
    pressure_tensor: np.ndarray | None = None
    pressure: np.ndarray | None = None


@dataclass
class RuntimeTraceReporter:
    """Collect scalar runtime reporter events for parity and diagnostics traces."""

    include_samples: bool = True
    include_diagnostics: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, event: ReporterEvent) -> None:
        """Record a reporter event, honoring the include-sample/diagnostic flags.

        Args:
            event: The reporter event to record.
        """

        if event.event_type == "sample" and not self.include_samples:
            return
        if event.event_type == "diagnostic" and not self.include_diagnostics:
            return
        self.events.append(
            {
                "ensemble": event.ensemble,
                "event_type": event.event_type,
                "step": int(event.step),
                "time": float(event.time),
                "potential_energy": _scalar_or_none(event.potential_energy),
                "kinetic_energy": _scalar_or_none(event.kinetic_energy),
                "total_energy": _scalar_or_none(event.total_energy),
                "temperature": _scalar_or_none(event.temperature),
                "pressure": _scalar_or_none(event.pressure),
                "pair_count": _scalar_or_none(event.pair_count),
                "rebuild_count": _scalar_or_none(event.rebuild_count),
                "constraint_max_error": _scalar_or_none(event.constraint_max_error),
                "thermostat": dict(event.thermostat),
                "barostat": dict(event.barostat),
            }
        )

    def to_jsonable(self) -> list[dict[str, Any]]:
        """Return the collected events as a list of JSON-serializable dicts.

        Returns:
            One dict per recorded reporter event, in arrival order.
        """

        return list(self.events)


@dataclass(frozen=True)
class SimulationCheckpoint:
    """Serializable boundary for resuming a production MD run."""

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    forces: np.ndarray
    step: int
    time: float
    cell: np.ndarray | None
    thermostat: dict[str, Any]
    neighbor_policy: dict[str, Any]
    force_terms: tuple[str, ...]
    diagnostic_cursor: int
    metadata: dict[str, Any]

    def state(self) -> SimulationState:
        """Rebuild the in-memory simulation state from this checkpoint.

        Returns:
            A `SimulationState` with MLX-backed arrays.
        """

        return SimulationState(
            positions=as_mx_array(self.positions),
            velocities=as_mx_array(self.velocities),
            masses=as_mx_array(self.masses),
            forces=as_mx_array(self.forces),
            step=self.step,
            time=self.time,
        )

    @property
    def hmr_state(self) -> dict[str, Any]:
        """Hydrogen-mass-repartitioning state recovered from the checkpoint metadata."""

        return _hmr_state_from_metadata(self.metadata)


def _hmr_state_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_state = metadata.get("hydrogen_mass_repartitioning") or metadata.get("hmr")
    if isinstance(raw_state, dict):
        state = dict(raw_state)
        state.setdefault("status", "represented_by_masses")
        state.setdefault("provenance_available", True)
        policy = dict(state.get("policy", {}))
        policy.setdefault("virtual_sites_supported", False)
        state["policy"] = policy
        return state
    return {
        "status": "absent",
        "provenance_available": False,
        "policy": {"virtual_sites_supported": False},
    }


def _scalar_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    scalar = array.item()
    if isinstance(scalar, int | np.integer):
        return int(scalar)
    return float(scalar)


def _cell_payload(cell: Cell | None) -> np.ndarray:
    if cell is None:
        return np.asarray([], dtype=np.float32)
    if cell.is_orthorhombic:
        return np.asarray(cell.lengths, dtype=np.float32)
    return np.asarray(cell.matrix, dtype=np.float32)


def read_xyz(path: str | Path) -> tuple[tuple[str, ...], np.ndarray, str]:
    """Read a single-frame XYZ file.

    Args:
        path: Path to the ``.xyz`` file.

    Returns:
        A ``(symbols, positions, comment)`` tuple: element symbols, an
            ``(n_atoms, 3)`` float array of coordinates, and the comment line.

    Raises:
        ValueError: If the file has fewer than two lines or the declared atom
            count does not match the number of coordinate lines.
    """

    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        msg = "XYZ file must contain atom count and comment lines"
        raise ValueError(msg)
    atom_count = int(lines[0].strip())
    comment = lines[1]
    if len(lines) != atom_count + 2:
        msg = "XYZ atom count does not match coordinate lines"
        raise ValueError(msg)
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 4:
            msg = "XYZ coordinate lines must contain symbol x y z"
            raise ValueError(msg)
        symbols.append(fields[0])
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return tuple(symbols), np.asarray(positions, dtype=np.float32), comment


def write_xyz(
    path: str | Path,
    symbols: list[str] | tuple[str, ...],
    positions,
    *,
    comment: str = "",
) -> None:
    """Write a single-frame XYZ file.

    Args:
        path: Destination path for the ``.xyz`` file.
        symbols: Element symbols, one per atom.
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        comment: Comment written as the file's second line. Defaults to ``""``.

    Raises:
        ValueError: If ``positions`` does not have shape ``(len(symbols), 3)``.
    """

    positions_np = np.asarray(positions, dtype=np.float32)
    if positions_np.shape != (len(symbols), 3):
        msg = "positions must have shape (n_symbols, 3)"
        raise ValueError(msg)
    lines = [str(len(symbols)), comment]
    for symbol, position in zip(symbols, positions_np, strict=True):
        lines.append(
            f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def save_npz_trajectory(
    path: str | Path,
    result,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    cell: Cell | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a native trajectory record to a compressed ``.npz`` file.

    Args:
        path: Destination ``.npz`` path.
        result: A finished simulation result (e.g. `NVEResult`) to serialize.
        symbols: Optional element symbols stored with the trajectory. Defaults to ``None``.
        cell: Optional periodic cell stored with the frames. Defaults to ``None``.
        metadata: Optional JSON-serializable metadata dict. Defaults to ``None``.
    """

    payload = {
        "sampled_positions": np.asarray(result.sampled_positions),
        "sampled_velocities": np.asarray(result.sampled_velocities),
        "sampled_steps": np.asarray(result.sampled_steps),
        "sampled_time": np.asarray(result.sampled_time),
        "diagnostic_steps": np.asarray(
            getattr(
                result,
                "diagnostic_steps",
                np.arange(len(np.asarray(result.total_energy)), dtype=np.int32),
            )
        ),
        "diagnostic_time": np.asarray(
            getattr(
                result,
                "diagnostic_time",
                np.arange(len(np.asarray(result.total_energy)), dtype=np.float32),
            )
        ),
        "potential_energy": np.asarray(result.potential_energy),
        "kinetic_energy": np.asarray(result.kinetic_energy),
        "total_energy": np.asarray(result.total_energy),
        "temperature": np.asarray(result.temperature),
        "virial_tensor": np.asarray(
            getattr(result, "virial_tensor", _zero_diagnostic_tensor(result))
        ),
        "pressure_tensor": np.asarray(
            getattr(result, "pressure_tensor", _zero_diagnostic_tensor(result))
        ),
        "pressure": np.asarray(
            getattr(result, "pressure", _zero_diagnostic_scalar(result))
        ),
        "pair_count": np.asarray(result.pair_count),
        "rebuild_count": np.asarray(result.rebuild_count),
        "constraint_max_error": np.asarray(
            getattr(result, "constraint_max_error", np.zeros_like(np.asarray(result.total_energy)))
        ),
        "symbols": np.asarray([] if symbols is None else list(symbols), dtype=str),
        "cell": _cell_payload(cell),
        "metadata_json": np.asarray(json.dumps(metadata or {})),
        "energy_term_names": np.asarray(
            list(getattr(result, "potential_energy_by_term", {}).keys()),
            dtype=str,
        ),
    }
    for name, values in getattr(result, "potential_energy_by_term", {}).items():
        payload[f"energy_term::{name}"] = np.asarray(values)
    for name in [
        "sampled_cv",
        "sampled_target",
        "sampled_bias_energy",
        "sampled_work",
        "diagnostic_cv",
        "diagnostic_target",
        "diagnostic_bias_energy",
        "diagnostic_work",
    ]:
        if hasattr(result, name):
            payload[name] = np.asarray(getattr(result, name))
    np.savez_compressed(path, **payload)


def trajectory_record_from_result(
    result,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    cell: Cell | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryRecord:
    """Create a native loaded-record view from an in-memory simulation result.

    Args:
        result: A finished simulation result to convert.
        symbols: Optional element symbols. Defaults to ``None``.
        cell: Optional periodic cell. Defaults to ``None``.
        metadata: Optional JSON-serializable metadata dict. Defaults to ``None``.

    Returns:
        A `TrajectoryRecord` mirroring what `load_npz_trajectory`
            would return for the same run.
    """

    return TrajectoryRecord(
        sampled_positions=np.asarray(result.sampled_positions),
        sampled_velocities=np.asarray(result.sampled_velocities),
        sampled_steps=np.asarray(result.sampled_steps),
        sampled_time=np.asarray(result.sampled_time),
        diagnostic_steps=np.asarray(
            getattr(
                result,
                "diagnostic_steps",
                np.arange(len(np.asarray(result.total_energy)), dtype=np.int32),
            )
        ),
        diagnostic_time=np.asarray(
            getattr(
                result,
                "diagnostic_time",
                np.arange(len(np.asarray(result.total_energy)), dtype=np.float32),
            )
        ),
        potential_energy=np.asarray(result.potential_energy),
        kinetic_energy=np.asarray(result.kinetic_energy),
        total_energy=np.asarray(result.total_energy),
        potential_energy_by_term={
            name: np.asarray(values)
            for name, values in getattr(result, "potential_energy_by_term", {}).items()
        },
        temperature=np.asarray(result.temperature),
        pair_count=np.asarray(result.pair_count),
        rebuild_count=np.asarray(result.rebuild_count),
        constraint_max_error=np.asarray(
            getattr(result, "constraint_max_error", np.zeros_like(np.asarray(result.total_energy)))
        ),
        symbols=tuple() if symbols is None else tuple(str(item) for item in symbols),
        cell=None if cell is None else _cell_payload(cell),
        metadata=dict(metadata or {}),
        virial_tensor=np.asarray(getattr(result, "virial_tensor", _zero_diagnostic_tensor(result))),
        pressure_tensor=np.asarray(
            getattr(result, "pressure_tensor", _zero_diagnostic_tensor(result))
        ),
        pressure=np.asarray(getattr(result, "pressure", _zero_diagnostic_scalar(result))),
    )


def load_npz_trajectory(path: str | Path) -> TrajectoryRecord:
    """Load a native trajectory record from a ``.npz`` file.

    Args:
        path: Path to a file written by `save_npz_trajectory`.

    Returns:
        The reconstructed `TrajectoryRecord`.
    """

    with np.load(path, allow_pickle=False) as data:
        term_names = tuple(str(item) for item in data["energy_term_names"].tolist())
        terms = {name: np.asarray(data[f"energy_term::{name}"]) for name in term_names}
        cell_data = np.asarray(data["cell"], dtype=np.float32)
        symbols = tuple(str(item) for item in data["symbols"].tolist())
        metadata = json.loads(str(np.asarray(data["metadata_json"])))
        diagnostic_steps, diagnostic_time = _load_diagnostic_axis(data, metadata)
        return TrajectoryRecord(
            sampled_positions=np.asarray(data["sampled_positions"]),
            sampled_velocities=np.asarray(data["sampled_velocities"]),
            sampled_steps=np.asarray(data["sampled_steps"]),
            sampled_time=np.asarray(data["sampled_time"]),
            diagnostic_steps=diagnostic_steps,
            diagnostic_time=diagnostic_time,
            potential_energy=np.asarray(data["potential_energy"]),
            kinetic_energy=np.asarray(data["kinetic_energy"]),
            total_energy=np.asarray(data["total_energy"]),
            potential_energy_by_term=terms,
            temperature=np.asarray(data["temperature"]),
            pair_count=np.asarray(data["pair_count"]),
            rebuild_count=np.asarray(data["rebuild_count"]),
            constraint_max_error=np.asarray(data["constraint_max_error"]),
            symbols=symbols,
            cell=None if cell_data.size == 0 else cell_data,
            metadata=metadata,
            virial_tensor=_load_diagnostic_tensor(data, "virial_tensor"),
            pressure_tensor=_load_diagnostic_tensor(data, "pressure_tensor"),
            pressure=_load_diagnostic_scalar(data, "pressure"),
        )


def save_simulation_checkpoint(
    path: str | Path,
    state: SimulationState,
    *,
    cell: Cell | None = None,
    thermostat: dict[str, Any] | None = None,
    neighbor_policy: dict[str, Any] | None = None,
    force_terms: tuple[str, ...] | list[str] | None = None,
    diagnostic_cursor: int | None = None,
    metadata: dict[str, Any] | None = None,
    runtime_sync_report: dict[str, int | float] | None = None,
    runtime_nonbonded_report: dict[str, int | float | str | None] | None = None,
) -> None:
    """Write a restart checkpoint for a runner-level continuation.

    Args:
        path: Destination checkpoint path (parent directories are created).
        state: The `SimulationState` to serialize.
        cell: Optional periodic cell. Defaults to ``None``.
        thermostat: Optional thermostat state (RNG offset or Nose-Hoover state).
            Defaults to ``None``.
        neighbor_policy: Optional neighbor-list policy dict. Defaults to ``None``.
        force_terms: Optional names of the force terms in effect. Defaults to ``None``.
        diagnostic_cursor: Optional index into the diagnostic series for exact
            resumption. Defaults to ``None``.
        metadata: Optional JSON-serializable metadata dict. Defaults to ``None``.
        runtime_sync_report: Optional runtime-sync counters to persist. Defaults to ``None``.
        runtime_nonbonded_report: Optional nonbonded-runtime report to persist.
            Defaults to ``None``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thermostat_payload = dict(thermostat or {})
    if thermostat_payload.get("family") == "nose_hoover":
        thermostat_payload.setdefault("deterministic_state", True)
    elif "rng_step_offset" not in thermostat_payload:
        thermostat_payload["rng_step_offset"] = int(state.step)
    materialization_start = perf_counter()
    checkpoint_arrays = {
        "positions": np.asarray(state.positions),
        "velocities": np.asarray(state.velocities),
        "masses": np.asarray(state.masses),
        "forces": np.asarray(state.forces),
    }
    materialization_elapsed = perf_counter() - materialization_start
    _record_checkpoint_runtime_attribution(
        runtime_sync_report,
        runtime_nonbonded_report,
        elapsed=materialization_elapsed,
    )
    payload = {
        **checkpoint_arrays,
        "step": np.asarray([int(state.step)], dtype=np.int64),
        "time": np.asarray([float(state.time)], dtype=np.float64),
        "cell": _cell_payload(cell),
        "thermostat_json": np.asarray(json.dumps(thermostat_payload)),
        "neighbor_policy_json": np.asarray(json.dumps(neighbor_policy or {})),
        "force_terms": np.asarray([] if force_terms is None else list(force_terms), dtype=str),
        "diagnostic_cursor": np.asarray(
            [int(state.step if diagnostic_cursor is None else diagnostic_cursor)],
            dtype=np.int64,
        ),
        "metadata_json": np.asarray(json.dumps(metadata or {})),
    }
    np.savez_compressed(path, **payload)


def _record_checkpoint_runtime_attribution(
    *reports: dict[str, int | float | str | None] | None,
    elapsed: float,
) -> None:
    seen: set[int] = set()
    for report in reports:
        if report is None:
            continue
        report_id = id(report)
        if report_id in seen:
            continue
        seen.add(report_id)
        _increment_runtime_count(report, "runtime_sync_total_count")
        _increment_runtime_wall_seconds(report, "runtime_sync_total_wall_seconds", elapsed)
        _increment_runtime_count(report, "runtime_sync_checkpoint_count")
        _increment_runtime_wall_seconds(report, "runtime_sync_checkpoint_wall_seconds", elapsed)
        _increment_runtime_count(report, "runtime_materialization_total_count")
        _increment_runtime_wall_seconds(
            report,
            "runtime_materialization_total_wall_seconds",
            elapsed,
        )
        _increment_runtime_count(report, "runtime_materialization_checkpoint_count")
        _increment_runtime_wall_seconds(
            report,
            "runtime_materialization_checkpoint_wall_seconds",
            elapsed,
        )


def _increment_runtime_count(
    report: dict[str, int | float | str | None],
    key: str,
) -> None:
    report[key] = int(report.get(key) or 0) + 1


def _increment_runtime_wall_seconds(
    report: dict[str, int | float | str | None],
    key: str,
    elapsed: float,
) -> None:
    report[key] = float(report.get(key) or 0.0) + elapsed


def load_simulation_checkpoint(path: str | Path) -> SimulationCheckpoint:
    """Load a restart checkpoint written by `save_simulation_checkpoint`.

    Args:
        path: Path to a checkpoint file.

    Returns:
        The reconstructed `SimulationCheckpoint`.
    """

    with np.load(path, allow_pickle=False) as data:
        cell = np.asarray(data["cell"], dtype=np.float32)
        return SimulationCheckpoint(
            positions=np.asarray(data["positions"]),
            velocities=np.asarray(data["velocities"]),
            masses=np.asarray(data["masses"]),
            forces=np.asarray(data["forces"]),
            step=int(np.asarray(data["step"])[0]),
            time=float(np.asarray(data["time"])[0]),
            cell=None if cell.size == 0 else cell,
            thermostat=json.loads(str(np.asarray(data["thermostat_json"]))),
            neighbor_policy=json.loads(str(np.asarray(data["neighbor_policy_json"]))),
            force_terms=tuple(str(item) for item in data["force_terms"].tolist()),
            diagnostic_cursor=int(np.asarray(data["diagnostic_cursor"])[0]),
            metadata=json.loads(str(np.asarray(data["metadata_json"]))),
        )


def _load_diagnostic_axis(data, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "diagnostic_steps" in data.files:
        diagnostic_steps = np.asarray(data["diagnostic_steps"])
    else:
        diagnostic_steps = np.arange(len(np.asarray(data["total_energy"])), dtype=np.int32)
    if "diagnostic_time" in data.files:
        diagnostic_time = np.asarray(data["diagnostic_time"])
    else:
        dt = float(metadata.get("dt", 1.0))
        diagnostic_time = diagnostic_steps.astype(np.float32) * dt
    return diagnostic_steps, diagnostic_time


def _zero_diagnostic_tensor(result) -> np.ndarray:
    count = len(np.asarray(result.total_energy))
    return np.zeros((count, 3, 3), dtype=np.float32)


def _zero_diagnostic_scalar(result) -> np.ndarray:
    count = len(np.asarray(result.total_energy))
    return np.zeros((count,), dtype=np.float32)


def _load_diagnostic_tensor(data, name: str) -> np.ndarray:
    if name in data.files:
        return np.asarray(data[name])
    count = len(np.asarray(data["total_energy"]))
    return np.zeros((count, 3, 3), dtype=np.float32)


def _load_diagnostic_scalar(data, name: str) -> np.ndarray:
    if name in data.files:
        return np.asarray(data[name])
    count = len(np.asarray(data["total_energy"]))
    return np.zeros((count,), dtype=np.float32)


def restart_state_from_trajectory(
    record: TrajectoryRecord,
    masses,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
    *,
    cell: Cell | None = None,
    frame: int = -1,
) -> SimulationState:
    """Recompute forces for a continuation-ready state from a trajectory frame.

    Args:
        record: The `TrajectoryRecord` to restart from.
        masses: Per-particle masses, shape ``(n_particles,)``.
        force_terms: One or more force terms used to recompute forces at the frame.
        cell: Optional periodic cell. Defaults to ``None``.
        frame: Frame index into the sampled trajectory. Defaults to ``-1`` (last frame).

    Returns:
        A `SimulationState` with the frame's positions/velocities and
            freshly recomputed forces.

    Raises:
        ValueError: If ``force_terms`` is empty.
    """

    positions = as_mx_array(record.sampled_positions[frame])
    velocities = as_mx_array(record.sampled_velocities[frame])
    masses = as_mx_array(masses)
    terms = tuple(force_terms) if isinstance(force_terms, (list, tuple)) else (force_terms,)
    total_forces = None
    for term in terms:
        _, forces = term.energy_forces(positions, cell=cell)
        total_forces = forces if total_forces is None else total_forces + forces
    if total_forces is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=total_forces,
        step=int(record.sampled_steps[frame]),
        time=float(record.sampled_time[frame]),
    )


__all__ = [
    "RuntimeTraceReporter",
    "SimulationCheckpoint",
    "TrajectoryRecord",
    "load_simulation_checkpoint",
    "load_npz_trajectory",
    "read_xyz",
    "restart_state_from_trajectory",
    "save_npz_trajectory",
    "save_simulation_checkpoint",
    "trajectory_record_from_result",
    "write_xyz",
]
