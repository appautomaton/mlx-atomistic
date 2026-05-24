"""Energy minimization helpers for molecular mechanics systems."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from scipy.optimize import minimize as scipy_minimize

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
    method: str = "steepest_descent"
    convergence_reason: str = ""


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
    method: str = "steepest_descent",
) -> MinimizationResult:
    """Minimize potential energy with a selectable local optimizer."""

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

    method_name = _normalize_method(method)
    terms = _as_terms(force_terms)
    if method_name in {"l-bfgs", "conjugate-gradient"}:
        return _minimize_with_scipy(
            positions,
            terms,
            cell=cell,
            max_steps=max_steps,
            force_tolerance=force_tolerance,
            neighbor_manager=neighbor_manager,
            method=method_name,
        )

    return _minimize_steepest_descent(
        positions,
        terms,
        cell=cell,
        max_steps=max_steps,
        step_size=step_size,
        force_tolerance=force_tolerance,
        backtracking=backtracking,
        min_step_size=min_step_size,
        neighbor_manager=neighbor_manager,
    )


def _normalize_method(method: str) -> str:
    normalized = method.lower().replace("_", "-")
    aliases = {
        "sd": "steepest-descent",
        "steepest": "steepest-descent",
        "steepest-descent": "steepest-descent",
        "lbfgs": "l-bfgs",
        "l-bfgs": "l-bfgs",
        "l-bfgs-b": "l-bfgs",
        "cg": "conjugate-gradient",
        "conjugate-gradient": "conjugate-gradient",
    }
    try:
        return aliases[normalized]
    except KeyError as err:
        expected = sorted(set(aliases.values()))
        msg = f"unknown minimization method {method!r}; expected one of {expected}"
        raise ValueError(msg) from err


def _minimize_steepest_descent(
    positions,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    max_steps: int,
    step_size: float,
    force_tolerance: float,
    backtracking: float,
    min_step_size: float,
    neighbor_manager: NeighborListManager | None,
) -> MinimizationResult:
    current_positions = as_mx_array(positions)
    energy, forces = _energy_forces(current_positions, terms, cell, neighbor_manager)
    energy_history = [energy]
    max_force_history = [mx.max(mx.abs(forces))]
    converged = bool(np.asarray(max_force_history[-1]) <= force_tolerance)
    steps_taken = 0
    convergence_reason = "force_tolerance" if converged else "max_steps"

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
            convergence_reason = "line_search_failed"
            break
        steps_taken = step
        max_force = mx.max(mx.abs(forces))
        energy_history.append(energy)
        max_force_history.append(max_force)
        converged = bool(np.asarray(max_force) <= force_tolerance)

    if converged:
        convergence_reason = "force_tolerance"

    return MinimizationResult(
        positions=current_positions,
        energy=energy,
        energy_history=mx.stack(energy_history),
        max_force_history=mx.stack(max_force_history),
        steps=steps_taken,
        converged=converged,
        method="steepest_descent",
        convergence_reason=convergence_reason,
    )


def _minimize_with_scipy(
    positions,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    max_steps: int,
    force_tolerance: float,
    neighbor_manager: NeighborListManager | None,
    method: str,
) -> MinimizationResult:
    initial_positions = as_mx_array(positions)
    shape = initial_positions.shape
    if initial_positions.ndim != 2 or initial_positions.shape[1] != 3:
        msg = "positions must have shape (n_particles, 3)"
        raise ValueError(msg)

    energy_history: list[mx.array] = []
    max_force_history: list[mx.array] = []

    def evaluate(flat_positions: np.ndarray) -> tuple[float, np.ndarray]:
        trial_positions = as_mx_array(flat_positions.reshape(shape))
        if cell is not None:
            trial_positions = cell.wrap(trial_positions)
        energy, forces = _energy_forces(trial_positions, terms, cell, neighbor_manager)
        max_force = mx.max(mx.abs(forces))
        energy_history.append(energy)
        max_force_history.append(max_force)
        return float(np.asarray(energy)), -np.asarray(forces, dtype=np.float64).reshape(-1)

    initial_flat = np.asarray(initial_positions, dtype=np.float64).reshape(-1)
    initial_energy, initial_gradient = evaluate(initial_flat)
    if max_steps == 0 or float(np.max(np.abs(initial_gradient))) <= force_tolerance:
        final_positions = initial_positions if cell is None else cell.wrap(initial_positions)
        return MinimizationResult(
            positions=final_positions,
            energy=energy_history[-1],
            energy_history=mx.stack(energy_history),
            max_force_history=mx.stack(max_force_history),
            steps=0,
            converged=bool(np.asarray(max_force_history[-1]) <= force_tolerance),
            method=method,
            convergence_reason="force_tolerance"
            if bool(np.asarray(max_force_history[-1]) <= force_tolerance)
            else "max_steps",
        )

    scipy_method = "L-BFGS-B" if method == "l-bfgs" else "CG"
    result = scipy_minimize(
        fun=evaluate,
        x0=np.asarray(initial_positions, dtype=np.float64).reshape(-1),
        jac=True,
        method=scipy_method,
        options={"maxiter": max_steps, "gtol": force_tolerance},
    )

    final_positions = as_mx_array(result.x.reshape(shape))
    if cell is not None:
        final_positions = cell.wrap(final_positions)
    final_energy, final_forces = _energy_forces(final_positions, terms, cell, neighbor_manager)
    final_max_force = mx.max(mx.abs(final_forces))
    energy_history.append(final_energy)
    max_force_history.append(final_max_force)
    converged = bool(np.asarray(final_max_force) <= force_tolerance)
    if converged:
        convergence_reason = "force_tolerance"
    elif result.success:
        convergence_reason = "optimizer_success"
    elif result.nit >= max_steps:
        convergence_reason = "max_steps"
    else:
        convergence_reason = str(result.message)

    # Keep the initial-energy local alive for debuggers and to make the first
    # evaluation intentional rather than an unused side effect.
    _ = initial_energy
    return MinimizationResult(
        positions=final_positions,
        energy=final_energy,
        energy_history=mx.stack(energy_history),
        max_force_history=mx.stack(max_force_history),
        steps=int(result.nit),
        converged=converged,
        method=method,
        convergence_reason=convergence_reason,
    )


__all__ = ["MinimizationResult", "minimize_energy"]
