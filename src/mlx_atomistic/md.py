"""Molecular dynamics primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import exp, sqrt
from typing import Protocol

import mlx.core as mx

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import NeighborListManager


class ForceTerm(Protocol):
    """Protocol for composable force terms."""

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return potential energy and forces."""


@dataclass(frozen=True)
class LennardJonesPotential:
    """Naive all-pairs Lennard-Jones potential in reduced units."""

    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float | None = 2.5
    shift: bool = True

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return potential energy and forces for positions with shape `(N, 3)`."""

        positions = as_mx_array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "positions must have shape (n_particles, 3)"
            raise ValueError(msg)
        if pairs is not None:
            return self._pair_energy_forces(positions, pairs, cell)

        displacement = positions[:, None, :] - positions[None, :, :]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = r2 > 0.0
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)

        safe_r2 = mx.where(pair_mask, r2, 1.0)
        sigma2_over_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6

        pair_energy = 4.0 * self.epsilon * (inv_r12 - inv_r6)
        if self.shift and self.cutoff is not None:
            sigma2_over_rc2 = (self.sigma * self.sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * self.epsilon * (inv_rc12 - inv_rc6)
        pair_energy = mx.where(pair_mask, pair_energy, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar, 0.0)
        forces = mx.sum(scalar[:, :, None] * displacement, axis=1)

        energy = 0.5 * mx.sum(pair_energy)
        return energy, forces

    def _pair_energy_forces(
        self,
        positions: mx.array,
        pairs: mx.array,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array]:
        pairs = mx.array(pairs, dtype=mx.int32)
        forces = mx.zeros_like(positions)
        if pairs.shape[0] == 0:
            return mx.sum(positions[:, 0] * 0.0), forces

        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = r2 > 0.0
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)

        safe_r2 = mx.where(pair_mask, r2, 1.0)
        sigma2_over_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6

        pair_energy = 4.0 * self.epsilon * (inv_r12 - inv_r6)
        if self.shift and self.cutoff is not None:
            sigma2_over_rc2 = (self.sigma * self.sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * self.epsilon * (inv_rc12 - inv_rc6)
        pair_energy = mx.where(pair_mask, pair_energy, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = forces.at[i].add(pair_forces).at[j].add(-pair_forces)

        return mx.sum(pair_energy), forces


@dataclass(frozen=True)
class StepState:
    """Single MD state."""

    positions: mx.array
    velocities: mx.array
    forces: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array

    @property
    def total_energy(self) -> mx.array:
        """Potential plus kinetic energy."""

        return self.potential_energy + self.kinetic_energy


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory and diagnostics from an MD run."""

    positions: mx.array
    velocities: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    temperature: mx.array


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for molecular dynamics."""

    dt: float = 0.005
    steps: int = 100
    sample_interval: int = 1

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if self.steps < 0:
            msg = "steps must be non-negative"
            raise ValueError(msg)
        if self.sample_interval <= 0:
            msg = "sample_interval must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class SimulationState:
    """Current NVE simulation state."""

    positions: mx.array
    velocities: mx.array
    masses: mx.array
    forces: mx.array
    step: int = 0
    time: float = 0.0


@dataclass(frozen=True)
class NVEResult:
    """Sparse trajectory and per-step diagnostics from an NVE simulation."""

    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    temperature: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    final_state: SimulationState

    @property
    def energy_drift(self) -> mx.array:
        """Total energy minus the initial total energy for each diagnostic step."""

        return self.total_energy - self.total_energy[0]

    @property
    def max_energy_drift(self) -> mx.array:
        """Maximum absolute total-energy drift over the run."""

        return mx.max(mx.abs(self.energy_drift))

    @property
    def relative_energy_drift(self) -> mx.array:
        """Energy drift normalized by the absolute initial total energy."""

        denominator = mx.maximum(mx.abs(self.total_energy[0]), 1e-12)
        return self.energy_drift / denominator


@dataclass(frozen=True)
class LangevinThermostat:
    """Langevin thermostat parameters in reduced units."""

    temperature: float = 1.0
    friction: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            msg = "temperature must be non-negative"
            raise ValueError(msg)
        if self.friction < 0.0:
            msg = "friction must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class NVTResult:
    """Sparse trajectory and per-step diagnostics from an NVT simulation."""

    sampled_positions: mx.array
    sampled_velocities: mx.array
    sampled_steps: mx.array
    sampled_time: mx.array
    potential_energy: mx.array
    kinetic_energy: mx.array
    total_energy: mx.array
    temperature: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    final_state: SimulationState
    target_temperature: float

    @property
    def temperature_error(self) -> mx.array:
        """Instantaneous temperature minus the target thermostat temperature."""

        return self.temperature - self.target_temperature


def kinetic_energy(velocities: mx.array, masses: mx.array) -> mx.array:
    """Return kinetic energy in reduced units."""

    velocities = as_mx_array(velocities)
    masses = as_mx_array(masses)
    return 0.5 * mx.sum(masses[:, None] * velocities * velocities)


def instantaneous_temperature(
    velocities: mx.array,
    masses: mx.array,
    *,
    dof: int | None = None,
) -> mx.array:
    """Return the instantaneous reduced temperature with `k_B = 1`."""

    if dof is None:
        dof = velocities.size
    return 2.0 * kinetic_energy(velocities, masses) / dof


def _as_force_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    if isinstance(force_terms, (list, tuple)):
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        return tuple(force_terms)
    return (force_terms,)


def _energy_forces_from_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
) -> tuple[mx.array, mx.array]:
    total_energy = None
    total_forces = mx.zeros_like(positions)
    for term in force_terms:
        energy, forces = term.energy_forces(positions, cell, pairs=pairs)
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, total_forces


def _dense_pair_count(positions: mx.array) -> int:
    n_particles = positions.shape[0]
    return n_particles * (n_particles - 1) // 2


def _local_prng_key(seed: int | None) -> mx.array:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return mx.random.key(seed)


@dataclass(frozen=True)
class VelocityVerlet:
    """Velocity Verlet integrator."""

    dt: float

    def step(
        self,
        positions: mx.array,
        velocities: mx.array,
        masses: mx.array,
        potential: LennardJonesPotential,
        *,
        cell: Cell | None = None,
        forces: mx.array | None = None,
        pairs: mx.array | None = None,
    ) -> StepState:
        """Advance one MD step."""

        if forces is None:
            _, forces = potential.energy_forces(positions, cell, pairs=pairs)

        acceleration = forces / masses[:, None]
        velocities_half = velocities + 0.5 * self.dt * acceleration
        next_positions = positions + self.dt * velocities_half
        if cell is not None:
            next_positions = cell.wrap(next_positions)

        potential_energy, next_forces = potential.energy_forces(next_positions, cell, pairs=pairs)
        next_acceleration = next_forces / masses[:, None]
        next_velocities = velocities_half + 0.5 * self.dt * next_acceleration

        return StepState(
            positions=next_positions,
            velocities=next_velocities,
            forces=next_forces,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energy(next_velocities, masses),
        )


def simulate(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    potential: LennardJonesPotential | None = None,
    pairs: mx.array | None = None,
    dt: float = 0.005,
    steps: int = 100,
) -> SimulationResult:
    """Run a short NVE MD simulation in reduced units."""

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    if potential is None:
        potential = LennardJonesPotential()

    potential_energy, forces = potential.energy_forces(positions, cell, pairs=pairs)
    state = StepState(
        positions=positions,
        velocities=velocities,
        forces=forces,
        potential_energy=potential_energy,
        kinetic_energy=kinetic_energy(velocities, masses),
    )

    position_frames = [state.positions]
    velocity_frames = [state.velocities]
    potential_energies = [state.potential_energy]
    kinetic_energies = [state.kinetic_energy]
    temperatures = [instantaneous_temperature(state.velocities, masses)]

    integrator = VelocityVerlet(dt)
    for _ in range(steps):
        state = integrator.step(
            state.positions,
            state.velocities,
            masses,
            potential,
            cell=cell,
            forces=state.forces,
            pairs=pairs,
        )
        position_frames.append(state.positions)
        velocity_frames.append(state.velocities)
        potential_energies.append(state.potential_energy)
        kinetic_energies.append(state.kinetic_energy)
        temperatures.append(instantaneous_temperature(state.velocities, masses))

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return SimulationResult(
        positions=mx.stack(position_frames),
        velocities=mx.stack(velocity_frames),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        temperature=mx.stack(temperatures),
    )


def simulate_nve(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
) -> NVEResult:
    """Run NVE molecular dynamics with sparse trajectory and dense diagnostics.

    `sample_interval` controls trajectory storage only. Energy, temperature,
    pair-count, and rebuild diagnostics are retained for every integration step.
    """

    if config is None:
        config = SimulationConfig()
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _as_force_terms(force_terms)

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)

    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    pair_count = _dense_pair_count(positions) if neighbor_list is None else neighbor_list.pair_count
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

    potential_energy, forces = _energy_forces_from_terms(positions, terms, cell=cell, pairs=pairs)
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
    )

    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [0]
    sampled_times = [0.0]
    potential_energies = [potential_energy]
    kinetic_energies = [kinetic_energy(state.velocities, masses)]
    temperatures = [instantaneous_temperature(state.velocities, masses)]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]

    for step in range(1, config.steps + 1):
        acceleration = state.forces / masses[:, None]
        velocities_half = state.velocities + 0.5 * config.dt * acceleration
        next_positions = state.positions + config.dt * velocities_half
        if cell is not None:
            next_positions = cell.wrap(next_positions)

        neighbor_list = (
            neighbor_manager.update(next_positions) if neighbor_manager is not None else None
        )
        pairs = None if neighbor_list is None else neighbor_list.pairs
        pair_count = (
            _dense_pair_count(next_positions) if neighbor_list is None else neighbor_list.pair_count
        )
        rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

        potential_energy, next_forces = _energy_forces_from_terms(
            next_positions,
            terms,
            cell=cell,
            pairs=pairs,
        )
        next_acceleration = next_forces / masses[:, None]
        next_velocities = velocities_half + 0.5 * config.dt * next_acceleration
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
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy(state.velocities, masses))
        temperatures.append(instantaneous_temperature(state.velocities, masses))
        pair_counts.append(pair_count)
        rebuild_counts.append(rebuild_count)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return NVEResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        temperature=mx.stack(temperatures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        final_state=state,
    )


def simulate_nvt(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: LangevinThermostat | None = None,
) -> NVTResult:
    """Run Langevin NVT molecular dynamics with BAOAB integration."""

    if config is None:
        config = SimulationConfig()
    if thermostat is None:
        thermostat = LangevinThermostat()
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _as_force_terms(force_terms)

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)

    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    pair_count = _dense_pair_count(positions) if neighbor_list is None else neighbor_list.pair_count
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

    potential_energy, forces = _energy_forces_from_terms(positions, terms, cell=cell, pairs=pairs)
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
    )

    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [0]
    sampled_times = [0.0]
    potential_energies = [potential_energy]
    kinetic_energies = [kinetic_energy(state.velocities, masses)]
    temperatures = [instantaneous_temperature(state.velocities, masses)]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]

    key = _local_prng_key(thermostat.seed)
    velocity_decay = exp(-thermostat.friction * config.dt)
    noise_scale = sqrt((1.0 - velocity_decay * velocity_decay) * thermostat.temperature)

    for step in range(1, config.steps + 1):
        acceleration = state.forces / masses[:, None]
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

        neighbor_list = (
            neighbor_manager.update(next_positions) if neighbor_manager is not None else None
        )
        pairs = None if neighbor_list is None else neighbor_list.pairs
        pair_count = (
            _dense_pair_count(next_positions) if neighbor_list is None else neighbor_list.pair_count
        )
        rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

        potential_energy, next_forces = _energy_forces_from_terms(
            next_positions,
            terms,
            cell=cell,
            pairs=pairs,
        )
        next_acceleration = next_forces / masses[:, None]
        next_velocities = thermostatted_velocities + 0.5 * config.dt * next_acceleration
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
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy(state.velocities, masses))
        temperatures.append(instantaneous_temperature(state.velocities, masses))
        pair_counts.append(pair_count)
        rebuild_counts.append(rebuild_count)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return NVTResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        temperature=mx.stack(temperatures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        final_state=state,
        target_temperature=thermostat.temperature,
    )
