"""Energy minimization helpers for molecular mechanics systems."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.md import ForceTerm, _energy_forces_from_terms
from mlx_atomistic.neighbors import NeighborListManager


@dataclass(frozen=True)
class MinimizationResult:
    """Result of a simple force-based energy minimization."""

    positions: mx.array
    energy: mx.array
    energy_history: mx.array
    max_force_history: mx.array
    steps: int
    converged: bool


def _as_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    if isinstance(force_terms, (list, tuple)):
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        return tuple(force_terms)
    return (force_terms,)


def _energy_forces(
    positions: mx.array,
    terms: tuple[ForceTerm, ...],
    cell: Cell | None,
    neighbor_manager: NeighborListManager | None,
):
    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    return _energy_forces_from_terms(positions, terms, cell=cell, pairs=pairs)


def minimize_energy(
    positions,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
    *,
    cell: Cell | None = None,
    max_steps: int = 200,
    step_size: float = 1e-3,
    force_tolerance: float = 1e-3,
    backtracking: float = 0.5,
    min_step_size: float = 1e-12,
    neighbor_manager: NeighborListManager | None = None,
) -> MinimizationResult:
    """Minimize potential energy with conservative backtracking steepest descent."""

    if max_steps < 0:
        msg = "max_steps must be non-negative"
        raise ValueError(msg)
    if step_size <= 0.0 or min_step_size <= 0.0:
        msg = "step sizes must be positive"
        raise ValueError(msg)
    if not 0.0 < backtracking < 1.0:
        msg = "backtracking must be between 0 and 1"
        raise ValueError(msg)
    if force_tolerance <= 0.0:
        msg = "force_tolerance must be positive"
        raise ValueError(msg)

    terms = _as_terms(force_terms)
    current_positions = as_mx_array(positions)
    energy, forces = _energy_forces(current_positions, terms, cell, neighbor_manager)
    energy_history = [energy]
    max_force_history = [mx.max(mx.abs(forces))]
    converged = bool(np.asarray(max_force_history[-1]) <= force_tolerance)
    steps_taken = 0

    for step in range(1, max_steps + 1):
        if converged:
            break
        trial_step = step_size
        accepted = False
        while trial_step >= min_step_size:
            trial_positions = current_positions + trial_step * forces
            if cell is not None:
                trial_positions = cell.wrap(trial_positions)
            trial_energy, trial_forces = _energy_forces(
                trial_positions,
                terms,
                cell,
                neighbor_manager,
            )
            if float(np.asarray(trial_energy)) <= float(np.asarray(energy)):
                current_positions = trial_positions
                energy = trial_energy
                forces = trial_forces
                accepted = True
                break
            trial_step *= backtracking
        if not accepted:
            break
        steps_taken = step
        max_force = mx.max(mx.abs(forces))
        energy_history.append(energy)
        max_force_history.append(max_force)
        converged = bool(np.asarray(max_force) <= force_tolerance)

    return MinimizationResult(
        positions=current_positions,
        energy=energy,
        energy_history=mx.stack(energy_history),
        max_force_history=mx.stack(max_force_history),
        steps=steps_taken,
        converged=converged,
    )


__all__ = ["MinimizationResult", "minimize_energy"]
