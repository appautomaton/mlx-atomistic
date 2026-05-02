"""Steered molecular dynamics utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import exp, sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.md import (
    ForceTerm,
    LangevinThermostat,
    SimulationConfig,
    SimulationState,
    _as_force_terms,
    _dense_pair_count,
    _energy_forces_by_term,
    _energy_forces_from_terms,
    _is_diagnostic_step,
    _named_force_terms,
    _temperature_degrees_of_freedom,
    _zero_constraint_error,
    instantaneous_temperature,
    kinetic_energy,
)
from mlx_atomistic.neighbors import NeighborListManager


@dataclass(frozen=True)
class SteeredCOMBiasPotential:
    """Moving harmonic restraint on a ligand center-of-mass projection."""

    ligand_indices: object
    direction: object
    target: float
    k: float
    masses: object | None = None
    name: str = "steered_com_bias"

    def __post_init__(self) -> None:
        ligand_indices = np.asarray(self.ligand_indices, dtype=np.int32)
        if ligand_indices.ndim != 1 or ligand_indices.size == 0:
            msg = "ligand_indices must be a non-empty 1D index array"
            raise ValueError(msg)
        if np.any(ligand_indices < 0):
            msg = "ligand_indices must be non-negative"
            raise ValueError(msg)
        direction = np.asarray(self.direction, dtype=np.float32)
        if direction.shape != (3,):
            msg = "direction must have shape (3,)"
            raise ValueError(msg)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            msg = "direction must be non-zero"
            raise ValueError(msg)
        if self.k < 0.0:
            msg = "bias force constant k must be non-negative"
            raise ValueError(msg)
        if self.masses is None:
            weights = np.full((ligand_indices.size,), 1.0 / ligand_indices.size, dtype=np.float32)
        else:
            masses = np.asarray(self.masses, dtype=np.float32)
            if masses.ndim != 1:
                msg = "masses must have shape (n_atoms,)"
                raise ValueError(msg)
            selected = masses[ligand_indices]
            if np.any(selected <= 0.0):
                msg = "selected ligand masses must be positive"
                raise ValueError(msg)
            weights = selected / np.sum(selected)
        object.__setattr__(self, "ligand_indices", mx.array(ligand_indices, dtype=mx.int32))
        object.__setattr__(self, "direction", as_mx_array(direction / norm))
        object.__setattr__(self, "weights", as_mx_array(weights))

    def collective_variable(self, positions: mx.array) -> mx.array:
        """Return ligand COM projection onto the steering direction."""

        positions = as_mx_array(positions)
        ligand_positions = positions[self.ligand_indices]
        center = mx.sum(ligand_positions * self.weights[:, None], axis=0)
        return mx.sum(center * self.direction)

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        del cell
        displacement = self.collective_variable(positions) - self.target
        return 0.5 * self.k * displacement * displacement

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del cell, pairs
        positions = as_mx_array(positions)
        cv = self.collective_variable(positions)
        displacement = cv - self.target
        energy = 0.5 * self.k * displacement * displacement
        ligand_force = -self.k * displacement * self.direction
        per_atom_forces = self.weights[:, None] * ligand_force[None, :]
        forces = mx.zeros_like(positions).at[self.ligand_indices].add(per_atom_forces)
        return energy, forces


@dataclass(frozen=True)
class SteeredNVTResult:
    """Trajectory and steering diagnostics from an NVT SMD run."""

    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    diagnostic_steps: mx.array
    diagnostic_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    potential_energy_by_term: dict[str, mx.array]
    temperature: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    sampled_cv: mx.array
    sampled_target: mx.array
    sampled_bias_energy: mx.array
    sampled_work: mx.array
    diagnostic_cv: mx.array
    diagnostic_target: mx.array
    diagnostic_bias_energy: mx.array
    diagnostic_work: mx.array
    final_state: SimulationState
    target_temperature: float


def simulate_steered_nvt(
    positions,
    velocities,
    *,
    masses=None,
    ligand_indices,
    direction,
    target_start: float,
    target_velocity: float,
    k: float,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: LangevinThermostat | None = None,
    constraints: DistanceConstraints | None = None,
) -> SteeredNVTResult:
    """Run Langevin NVT while steering a ligand COM projection."""

    if config is None:
        config = SimulationConfig()
    if thermostat is None:
        thermostat = LangevinThermostat()
    if force_terms is None:
        msg = "force_terms are required for steered NVT"
        raise ValueError(msg)
    base_terms = _as_force_terms(force_terms)
    named_base_terms = _named_force_terms(base_terms)

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    bias = _bias(ligand_indices, direction, target_start, k, masses)
    constraint_error = _zero_constraint_error(positions)
    if constraints is not None:
        positions, constraint_error = constraints.apply_positions(positions, masses, cell)
        velocities = constraints.apply_velocities(positions, velocities, masses, cell)
    temperature_dof = _temperature_degrees_of_freedom(positions, constraints)

    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    pair_count = _dense_pair_count(positions) if neighbor_list is None else neighbor_list.pair_count
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

    potential_energy, forces, energy_by_term = _energy_forces_with_bias_by_term(
        positions,
        named_base_terms,
        bias,
        cell=cell,
        pairs=pairs,
    )
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
    )

    cv = bias.collective_variable(state.positions)
    bias_energy = bias.potential_energy(state.positions)
    accumulated_work = cv * 0.0

    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [0]
    sampled_times = [0.0]
    sampled_cv = [cv]
    sampled_target = [as_mx_array(target_start)]
    sampled_bias_energy = [bias_energy]
    sampled_work = [accumulated_work]
    diagnostic_steps = [0]
    diagnostic_times = [0.0]
    diagnostic_cv = [cv]
    diagnostic_target = [as_mx_array(target_start)]
    diagnostic_bias_energy = [bias_energy]
    diagnostic_work = [accumulated_work]
    potential_energies = [potential_energy]
    potential_energy_by_term = {name: [energy] for name, energy in energy_by_term.items()}
    kinetic_energies = [
        kinetic_energy(state.velocities, masses, kinetic_energy_scale=config.kinetic_energy_scale)
    ]
    temperatures = [
        instantaneous_temperature(
            state.velocities,
            masses,
            dof=temperature_dof,
            kinetic_energy_scale=config.kinetic_energy_scale,
            boltzmann_constant=config.boltzmann_constant,
        )
    ]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]
    constraint_errors = [constraint_error]

    key = _local_prng_key(thermostat.seed)
    velocity_decay = exp(-thermostat.friction * config.dt)
    noise_scale = sqrt(
        (1.0 - velocity_decay * velocity_decay)
        * thermostat.temperature
        * config.boltzmann_constant
        / config.kinetic_energy_scale
    )
    previous_target = target_start

    for step in range(1, config.steps + 1):
        acceleration = config.force_to_acceleration_scale * state.forces / masses[:, None]
        velocities_half = state.velocities + 0.5 * config.dt * acceleration
        next_positions = state.positions + 0.5 * config.dt * velocities_half
        if cell is not None:
            next_positions = cell.wrap(next_positions)

        keys = mx.random.split(key, 2)
        key = keys[0]
        noise = mx.random.normal(state.velocities.shape, key=keys[1])
        thermal_scale = noise_scale / mx.sqrt(masses)[:, None]
        thermostatted_velocities = velocity_decay * velocities_half + thermal_scale * noise

        next_positions = next_positions + 0.5 * config.dt * thermostatted_velocities
        if cell is not None:
            next_positions = cell.wrap(next_positions)
        constraint_error = _zero_constraint_error(next_positions)
        if constraints is not None:
            next_positions, constraint_error = constraints.apply_positions(
                next_positions,
                masses,
                cell,
            )

        neighbor_list = (
            neighbor_manager.update(next_positions) if neighbor_manager is not None else None
        )
        pairs = None if neighbor_list is None else neighbor_list.pairs
        pair_count = (
            _dense_pair_count(next_positions) if neighbor_list is None else neighbor_list.pair_count
        )
        rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

        target = target_start + target_velocity * (step * config.dt)
        bias = _bias(ligand_indices, direction, target, k, masses)
        diagnostic_step = _is_diagnostic_step(step, config)
        if diagnostic_step:
            potential_energy, next_forces, energy_by_term = _energy_forces_with_bias_by_term(
                next_positions,
                named_base_terms,
                bias,
                cell=cell,
                pairs=pairs,
            )
        else:
            potential_energy, next_forces = _energy_forces_with_bias(
                next_positions,
                base_terms,
                bias,
                cell=cell,
                pairs=pairs,
            )
            energy_by_term = None
        cv = bias.collective_variable(next_positions)
        bias_energy = bias.potential_energy(next_positions)
        target_delta = target - previous_target
        accumulated_work = accumulated_work + (-k * (cv - target)) * target_delta
        previous_target = target

        next_acceleration = config.force_to_acceleration_scale * next_forces / masses[:, None]
        next_velocities = thermostatted_velocities + 0.5 * config.dt * next_acceleration
        if constraints is not None:
            next_velocities = constraints.apply_velocities(
                next_positions,
                next_velocities,
                masses,
                cell,
            )
        state = SimulationState(
            positions=next_positions,
            velocities=next_velocities,
            masses=masses,
            forces=next_forces,
            step=step,
            time=step * config.dt,
        )

        if step % config.sample_interval == 0 or step == config.steps:
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(step)
            sampled_times.append(state.time)
            sampled_cv.append(cv)
            sampled_target.append(as_mx_array(target))
            sampled_bias_energy.append(bias_energy)
            sampled_work.append(accumulated_work)
        if diagnostic_step:
            diagnostic_steps.append(step)
            diagnostic_times.append(state.time)
            diagnostic_cv.append(cv)
            diagnostic_target.append(as_mx_array(target))
            diagnostic_bias_energy.append(bias_energy)
            diagnostic_work.append(accumulated_work)
            potential_energies.append(potential_energy)
            if energy_by_term is not None:
                for name, energy in energy_by_term.items():
                    potential_energy_by_term[name].append(energy)
            kinetic_energies.append(
                kinetic_energy(
                    state.velocities,
                    masses,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                )
            )
            temperatures.append(
                instantaneous_temperature(
                    state.velocities,
                    masses,
                    dof=temperature_dof,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                    boltzmann_constant=config.boltzmann_constant,
                )
            )
            pair_counts.append(pair_count)
            rebuild_counts.append(rebuild_count)
            constraint_errors.append(constraint_error)
        if step % config.evaluation_interval == 0 or step == config.steps:
            mx.eval(state.positions, state.velocities, state.forces, potential_energy, cv)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return SteeredNVTResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        diagnostic_steps=mx.array(diagnostic_steps, dtype=mx.int32),
        diagnostic_time=mx.array(diagnostic_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        potential_energy_by_term={
            name: mx.stack(energies) for name, energies in potential_energy_by_term.items()
        },
        temperature=mx.stack(temperatures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        constraint_max_error=mx.stack(constraint_errors),
        sampled_cv=mx.stack(sampled_cv),
        sampled_target=mx.stack(sampled_target),
        sampled_bias_energy=mx.stack(sampled_bias_energy),
        sampled_work=mx.stack(sampled_work),
        diagnostic_cv=mx.stack(diagnostic_cv),
        diagnostic_target=mx.stack(diagnostic_target),
        diagnostic_bias_energy=mx.stack(diagnostic_bias_energy),
        diagnostic_work=mx.stack(diagnostic_work),
        final_state=state,
        target_temperature=thermostat.temperature,
    )


def _energy_forces_with_bias(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    bias: SteeredCOMBiasPotential,
    *,
    cell: Cell | None,
    pairs: mx.array | None,
) -> tuple[mx.array, mx.array]:
    base_energy, base_forces = _energy_forces_from_terms(
        positions,
        force_terms,
        cell=cell,
        pairs=pairs,
    )
    bias_energy, bias_forces = bias.energy_forces(positions, cell=cell, pairs=pairs)
    return base_energy + bias_energy, base_forces + bias_forces


def _energy_forces_with_bias_by_term(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    bias: SteeredCOMBiasPotential,
    *,
    cell: Cell | None,
    pairs: mx.array | None,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    base_energy, base_forces, energy_by_term = _energy_forces_by_term(
        positions,
        force_terms,
        cell=cell,
        pairs=pairs,
    )
    bias_energy, bias_forces = bias.energy_forces(positions, cell=cell, pairs=pairs)
    energy_by_term = dict(energy_by_term)
    energy_by_term[bias.name] = bias_energy
    return base_energy + bias_energy, base_forces + bias_forces, energy_by_term


def _bias(
    ligand_indices,
    direction,
    target: float,
    k: float,
    masses,
) -> SteeredCOMBiasPotential:
    return SteeredCOMBiasPotential(
        ligand_indices=ligand_indices,
        direction=direction,
        target=float(target),
        k=float(k),
        masses=np.asarray(masses, dtype=np.float32),
    )


def _local_prng_key(seed: int | None) -> mx.array:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return mx.random.key(seed)


__all__ = [
    "SteeredCOMBiasPotential",
    "SteeredNVTResult",
    "simulate_steered_nvt",
]
