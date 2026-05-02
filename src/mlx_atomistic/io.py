"""Structure and trajectory I/O helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.md import ForceTerm, SimulationState


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


def read_xyz(path: str | Path) -> tuple[tuple[str, ...], np.ndarray, str]:
    """Read a single-frame XYZ file."""

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
    """Write a single-frame XYZ file."""

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
    """Save a native trajectory record."""

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
        "cell": np.asarray([] if cell is None else np.asarray(cell.lengths), dtype=np.float32),
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


def load_npz_trajectory(path: str | Path) -> TrajectoryRecord:
    """Load a native trajectory record."""

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
    """Recompute forces for a continuation-ready state from a trajectory frame."""

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
    "TrajectoryRecord",
    "load_npz_trajectory",
    "read_xyz",
    "restart_state_from_trajectory",
    "save_npz_trajectory",
    "write_xyz",
]
