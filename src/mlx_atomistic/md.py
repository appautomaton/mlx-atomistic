"""Molecular dynamics primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import exp, sqrt
from time import perf_counter
from typing import Protocol

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import NeighborList, NeighborListManager
from mlx_atomistic.nonbonded import (
    DEFAULT_DENSE_MEMORY_BUDGET_BYTES,
    NonbondedBackend,
    NonbondedExecutionConfig,
    choose_nonbonded_backend,
    dense_lj_energy_forces,
    estimate_dense_nonbonded_bytes,
)
from mlx_atomistic.topology import Topology


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
    topology: Topology | None = None
    one_four_scale: float = 1.0
    backend: NonbondedBackend = "auto"
    tile_size: int = 512
    memory_budget_bytes: int | None = DEFAULT_DENSE_MEMORY_BUDGET_BYTES
    name: str = "lj"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.one_four_scale < 0.0:
            msg = "one_four_scale must be non-negative"
            raise ValueError(msg)
        config = NonbondedExecutionConfig(
            backend=self.backend,
            tile_size=self.tile_size,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        object.__setattr__(self, "tile_size", config.tile_size)
        object.__setattr__(self, "memory_budget_bytes", config.memory_budget_bytes)
        object.__setattr__(self, "_pair_scale_cache", None)

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
        if (
            self.topology is not None
            and pairs is None
            and self.topology.nonbonded_pair_policy == "lazy"
        ):
            msg = (
                "lazy topology requires a runtime nonbonded pair provider; "
                "full dense pair materialization was not requested"
            )
            raise ValueError(msg)

        estimated_bytes = estimate_dense_nonbonded_bytes(positions.shape[0], components="lj")
        concrete_backend = choose_nonbonded_backend(
            requested=self.backend,
            n_atoms=positions.shape[0],
            pairs_provided=pairs is not None,
            estimated_dense_bytes=estimated_bytes,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        if concrete_backend in {"mlx_dense", "mlx_tiled"}:
            return dense_lj_energy_forces(
                positions,
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                shift=self.shift,
                cell=cell,
                topology=self.topology,
                one_four_scale=self.one_four_scale,
                tile_size=self.tile_size if concrete_backend == "mlx_tiled" else None,
            )

        if self.topology is not None:
            filtered_pairs, scales = self._topology_pairs_and_scales(pairs)
            return self._pair_energy_forces(positions, filtered_pairs, cell, scales=scales)
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

    def _topology_pairs_and_scales(self, pairs) -> tuple[mx.array, mx.array]:
        topology = self.topology
        if topology is None:
            msg = "topology is required"
            raise ValueError(msg)
        if pairs is None and topology.nonbonded_pair_policy == "lazy":
            msg = (
                "lazy topology requires a runtime nonbonded pair provider; "
                "full dense pair materialization was not requested"
            )
            raise ValueError(msg)
        if pairs is not None:
            cache_key = (id(pairs), self.one_four_scale)
            cache = self._pair_scale_cache
            if cache is not None and cache[0] == cache_key:
                return cache[1]
        filtered_pairs = topology.nonbonded_pairs(pairs)
        if float(self.one_four_scale) == 1.0:
            scales = mx.array(1.0, dtype=mx.float32)
        else:
            scales = topology.pair_scales(
                filtered_pairs,
                one_four_scale=self.one_four_scale,
            )
        if pairs is not None:
            object.__setattr__(self, "_pair_scale_cache", (cache_key, (filtered_pairs, scales)))
        return filtered_pairs, scales

    def _pair_energy_forces(
        self,
        positions: mx.array,
        pairs: mx.array,
        cell: Cell | None,
        *,
        scales: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        pairs = mx.array(pairs, dtype=mx.int32)
        forces = mx.zeros_like(positions)
        if pairs.shape[0] == 0:
            return mx.sum(positions[:, 0] * 0.0), forces
        if scales is None:
            scales = as_mx_array([1.0] * pairs.shape[0])

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
        pair_energy = mx.where(pair_mask, pair_energy * scales, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar * scales, 0.0)
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
    kinetic_energy_scale: float = 1.0
    force_to_acceleration_scale: float = 1.0
    boltzmann_constant: float = 1.0
    evaluation_interval: int = 25
    diagnostic_interval: int = 1
    compile_force_evaluator: bool = False
    pressure_diagnostics: bool = True
    initial_step: int = 0
    initial_time: float = 0.0

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
        if self.kinetic_energy_scale <= 0.0:
            msg = "kinetic_energy_scale must be positive"
            raise ValueError(msg)
        if self.force_to_acceleration_scale <= 0.0:
            msg = "force_to_acceleration_scale must be positive"
            raise ValueError(msg)
        if self.boltzmann_constant <= 0.0:
            msg = "boltzmann_constant must be positive"
            raise ValueError(msg)
        if self.evaluation_interval <= 0:
            msg = "evaluation_interval must be positive"
            raise ValueError(msg)
        if self.diagnostic_interval <= 0:
            msg = "diagnostic_interval must be positive"
            raise ValueError(msg)
        if self.initial_step < 0:
            msg = "initial_step must be non-negative"
            raise ValueError(msg)
        if self.initial_time < 0.0:
            msg = "initial_time must be non-negative"
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
class ReporterEvent:
    """State exposed to runtime reporter callbacks."""

    ensemble: str
    event_type: str
    step: int
    time: float
    state: SimulationState
    potential_energy: mx.array | None = None
    kinetic_energy: mx.array | None = None
    total_energy: mx.array | None = None
    temperature: mx.array | None = None
    energy_by_term: dict[str, mx.array] = field(default_factory=dict)
    virial_tensor: mx.array | None = None
    pressure_tensor: mx.array | None = None
    pressure: mx.array | None = None
    pair_count: int | mx.array | None = None
    rebuild_count: int | mx.array | None = None
    constraint_max_error: mx.array | None = None


class RuntimeReporter(Protocol):
    """Callable observer for sampled frames and diagnostic state."""

    def __call__(self, event: ReporterEvent) -> None:
        """Observe one runtime event."""


@dataclass(frozen=True)
class NVEResult:
    """Sparse trajectory and per-step diagnostics from an NVE simulation."""

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
    virial_tensor: mx.array
    pressure_tensor: mx.array
    pressure: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    final_state: SimulationState
    nonbonded_report: dict[str, int | float | str | None] = field(default_factory=dict)

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
    rng_step_offset: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            msg = "temperature must be non-negative"
            raise ValueError(msg)
        if self.friction < 0.0:
            msg = "friction must be non-negative"
            raise ValueError(msg)
        if self.rng_step_offset is not None and self.rng_step_offset < 0:
            msg = "rng_step_offset must be non-negative when provided"
            raise ValueError(msg)


@dataclass(frozen=True)
class NVTResult:
    """Sparse trajectory and per-step diagnostics from an NVT simulation."""

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
    virial_tensor: mx.array
    pressure_tensor: mx.array
    pressure: mx.array
    pair_count: mx.array
    rebuild_count: mx.array
    constraint_max_error: mx.array
    final_state: SimulationState
    target_temperature: float
    nonbonded_report: dict[str, int | float | str | None] = field(default_factory=dict)

    @property
    def temperature_error(self) -> mx.array:
        """Instantaneous temperature minus the target thermostat temperature."""

        return self.temperature - self.target_temperature


@dataclass(frozen=True)
class MonteCarloBarostat:
    """Minimal isotropic Monte Carlo barostat for orthorhombic cells."""

    pressure: float = 1.0
    temperature: float = 1.0
    interval: int = 25
    max_log_volume_scale: float = 0.02
    seed: int | None = 11

    def __post_init__(self) -> None:
        if self.pressure < 0.0:
            msg = "pressure must be non-negative"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if self.interval <= 0:
            msg = "barostat interval must be positive"
            raise ValueError(msg)
        if self.max_log_volume_scale <= 0.0:
            msg = "max_log_volume_scale must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class NPTResult:
    """NPT production result with delegated NVT trajectory fields."""

    production: NVTResult
    final_state: SimulationState
    final_cell: Cell
    cell_lengths: mx.array
    volume: mx.array
    target_pressure: float
    barostat_attempts: int
    barostat_accepted: int

    def __getattr__(self, name: str):
        return getattr(self.production, name)


def kinetic_energy(
    velocities: mx.array,
    masses: mx.array,
    *,
    kinetic_energy_scale: float = 1.0,
) -> mx.array:
    """Return kinetic energy using the configured unit conversion."""

    velocities = as_mx_array(velocities)
    masses = as_mx_array(masses)
    return kinetic_energy_scale * 0.5 * mx.sum(masses[:, None] * velocities * velocities)


def instantaneous_temperature(
    velocities: mx.array,
    masses: mx.array,
    *,
    dof: int | None = None,
    kinetic_energy_scale: float = 1.0,
    boltzmann_constant: float = 1.0,
) -> mx.array:
    """Return the instantaneous temperature for the configured unit system."""

    if dof is None:
        dof = velocities.size
    return (
        2.0
        * kinetic_energy(
            velocities,
            masses,
            kinetic_energy_scale=kinetic_energy_scale,
        )
        / (dof * boltzmann_constant)
    )


def virial_tensor(positions: mx.array, forces: mx.array) -> mx.array:
    """Return the non-periodic configurational virial tensor."""

    positions = as_mx_array(positions)
    forces = as_mx_array(forces)
    if positions.shape != forces.shape or positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions and forces must both have shape (n_particles, 3)"
        raise ValueError(msg)
    return mx.transpose(positions) @ forces


def configurational_virial_tensor(
    positions: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
    strain_epsilon: float = 1e-3,
) -> mx.array:
    """Return a configurational virial diagnostic after explicit support validation.

    Periodic orthorhombic cells use diagonal finite differences in cell strain
    with fractional coordinates held fixed. Off-diagonal strain is not part of
    this Slice 8 diagnostic convention and is reported as zero.
    """

    validate_virial_support(force_terms)
    positions = as_mx_array(positions)
    if cell is None:
        return virial_tensor(positions, forces)
    if strain_epsilon <= 0.0:
        msg = "strain_epsilon must be positive"
        raise ValueError(msg)
    if float(mx.min(cell.lengths)) <= 0.0:
        msg = "virial diagnostics require positive cell lengths"
        raise ValueError(msg)

    fractional = positions / cell.lengths
    fractional = fractional - mx.floor(fractional)
    diagonal = []
    for axis in range(3):
        plus_lengths = _strained_cell_lengths(cell.lengths, axis, strain_epsilon)
        minus_lengths = _strained_cell_lengths(cell.lengths, axis, -strain_epsilon)
        plus_energy = _potential_energy_for_virial(
            fractional * plus_lengths,
            force_terms,
            cell=Cell(plus_lengths),
            pairs=pairs,
        )
        minus_energy = _potential_energy_for_virial(
            fractional * minus_lengths,
            force_terms,
            cell=Cell(minus_lengths),
            pairs=pairs,
        )
        diagonal.append(-(plus_energy - minus_energy) / (2.0 * strain_epsilon))
    return mx.diag(mx.stack(diagonal))


def _strained_cell_lengths(lengths: mx.array, axis: int, strain: float) -> mx.array:
    factors = as_mx_array([1.0 + strain if item == axis else 1.0 for item in range(3)])
    return lengths * factors


def _potential_energy_for_virial(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell,
    pairs: mx.array | None,
) -> mx.array:
    energy, _ = _energy_forces_from_terms(
        positions,
        force_terms,
        cell=cell,
        pairs=pairs,
    )
    return energy


def kinetic_pressure_tensor(
    velocities: mx.array,
    masses: mx.array,
    *,
    kinetic_energy_scale: float = 1.0,
) -> mx.array:
    """Return ``sum_i m_i v_i outer v_i`` in the configured kinetic units."""

    velocities = as_mx_array(velocities)
    masses = as_mx_array(masses)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        msg = "velocities must have shape (n_particles, 3)"
        raise ValueError(msg)
    if masses.shape != (velocities.shape[0],):
        msg = "masses must have shape (n_particles,)"
        raise ValueError(msg)
    weighted_velocities = masses[:, None] * velocities
    return kinetic_energy_scale * mx.transpose(velocities) @ weighted_velocities


def pressure_tensor(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
    kinetic_energy_scale: float = 1.0,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return virial tensor, pressure tensor, and scalar pressure diagnostics.

    The pressure tensor uses the reduced-unit convention
    ``P = (kinetic tensor + configurational virial) / V``. Periodic virials
    are diagonal-only orthorhombic cell-strain diagnostics; non-periodic runs
    report finite zero pressure diagnostics because no volume is defined.
    """

    virial = configurational_virial_tensor(
        positions,
        forces,
        force_terms,
        cell=cell,
        pairs=pairs,
    )
    if cell is None:
        zeros = mx.zeros((3, 3), dtype=virial.dtype)
        return virial, zeros, mx.sum(virial * 0.0)
    volume = mx.prod(cell.lengths)
    if float(mx.min(cell.lengths)) <= 0.0:
        msg = "pressure diagnostics require positive cell lengths"
        raise ValueError(msg)
    kinetic_tensor = kinetic_pressure_tensor(
        velocities,
        masses,
        kinetic_energy_scale=kinetic_energy_scale,
    )
    tensor = (kinetic_tensor + virial) / volume
    scalar = mx.trace(tensor) / 3.0
    return virial, tensor, scalar


def _pressure_diagnostics(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
    kinetic_energy_scale: float,
    enabled: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    if enabled:
        return pressure_tensor(
            positions,
            velocities,
            masses,
            forces,
            force_terms,
            cell=cell,
            pairs=pairs,
            kinetic_energy_scale=kinetic_energy_scale,
        )
    zeros = mx.zeros((3, 3), dtype=positions.dtype)
    return zeros, zeros, mx.sum(positions[:, 0] * 0.0)


def _temperature_degrees_of_freedom(
    positions: mx.array,
    constraints: DistanceConstraints | None,
) -> int:
    dof = int(positions.size)
    if constraints is not None:
        dof -= int(constraints.pairs.shape[0])
    if positions.shape[0] > 1:
        dof -= 3
    return max(1, dof)


def _as_force_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    if isinstance(force_terms, (list, tuple)):
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        return tuple(force_terms)
    return (force_terms,)


def _named_force_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    terms = _as_force_terms(force_terms)
    seen: dict[str, int] = {}
    named_terms = []
    for term in terms:
        base_name = str(getattr(term, "name", type(term).__name__))
        seen[base_name] = seen.get(base_name, 0) + 1
        name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}"
        named_terms.append((name, term))
    return tuple(named_terms)


def missing_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> tuple[str, ...]:
    """Return exact force-term names without a supported virial diagnostics path."""

    missing = []
    for name, term in _named_force_terms(force_terms):
        if not _term_supports_virial(term):
            missing.append(name)
    return tuple(missing)


def validate_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> None:
    """Fail closed when future pressure-coupled runtimes see unsupported terms."""

    missing = missing_virial_support(force_terms)
    if missing:
        msg = "missing virial support for force terms: " + ", ".join(missing)
        raise ValueError(msg)


def _term_supports_virial(term: ForceTerm) -> bool:
    declared = getattr(term, "supports_virial", None)
    if declared is not None:
        return bool(declared)
    return callable(getattr(term, "virial_tensor", None)) or callable(
        getattr(term, "virial_diagnostics", None)
    )


def _energy_forces_from_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
) -> tuple[mx.array, mx.array]:
    grouped_terms = _groupable_potential_terms(force_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(positions)
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for term in force_terms:
        if id(term) in grouped_ids:
            continue
        energy, forces = term.energy_forces(positions, cell, pairs=pairs)
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, total_forces


def _energy_forces_by_term(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    unnamed_terms = tuple(term for _, term in force_terms)
    grouped_terms = _groupable_potential_terms(unnamed_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(positions)
    energy_by_term = {}
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for name, term in force_terms:
        if id(term) in grouped_ids:
            energy_by_term[name] = term.potential_energy(positions, cell)
            continue

        combined_components = getattr(term, "energy_forces_with_components", None)
        if callable(combined_components):
            energy, forces, components = combined_components(positions, cell, pairs=pairs)
            for component_name, component_energy in components.items():
                if _is_energy_component(component_energy):
                    energy_by_term[f"{name}.{component_name}"] = component_energy
        else:
            energy, forces = term.energy_forces(positions, cell, pairs=pairs)
            component_energies = getattr(term, "component_energies", None)
            if callable(component_energies):
                for component_name, component_energy in component_energies(
                    positions,
                    cell=cell,
                    pairs=pairs,
                ).items():
                    if _is_energy_component(component_energy):
                        energy_by_term[f"{name}.{component_name}"] = component_energy
            else:
                energy_by_term[name] = energy
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, total_forces, energy_by_term


def _make_energy_forces_by_term_evaluator(
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
    compile_evaluator: bool,
):
    def evaluate(pos: mx.array):
        return _energy_forces_by_term(
            pos,
            force_terms,
            cell=cell,
            pairs=pairs,
        )

    if compile_evaluator and pairs is None:
        return mx.compile(evaluate)
    return evaluate


def _is_energy_component(value: object) -> bool:
    return isinstance(value, (mx.array, int, float))


def _make_energy_forces_evaluator(
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: mx.array | None,
    compile_evaluator: bool,
):
    def evaluate(pos: mx.array):
        return _energy_forces_from_terms(
            pos,
            force_terms,
            cell=cell,
            pairs=pairs,
        )

    if compile_evaluator and pairs is None:
        return mx.compile(evaluate)
    return evaluate


def _groupable_potential_terms(
    force_terms: tuple[ForceTerm, ...],
    pairs: mx.array | None,
) -> tuple[ForceTerm, ...]:
    if pairs is not None:
        return ()
    # Production force terms provide analytical `energy_forces`; using those is
    # faster than differentiating summed potential energies each MD step.
    # Potential-only custom terms may opt into autograd grouping explicitly.
    return tuple(
        term
        for term in force_terms
        if bool(getattr(term, "use_autograd_forces", False))
        and callable(getattr(term, "potential_energy", None))
        and not callable(getattr(term, "energy_forces_with_components", None))
    )


def _grouped_potential_energy_forces(
    positions: mx.array,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
) -> tuple[mx.array, mx.array]:
    def total_potential_energy(pos: mx.array) -> mx.array:
        total = None
        for term in terms:
            energy = term.potential_energy(pos, cell)
            total = energy if total is None else total + energy
        if total is None:
            return _zero_constraint_error(pos)
        return total

    energy, gradient = mx.value_and_grad(total_potential_energy)(positions)
    return energy, -gradient


def _dense_pair_count(positions: mx.array) -> int:
    n_particles = positions.shape[0]
    return n_particles * (n_particles - 1) // 2


def _validate_compact_nonbonded_backend(
    force_terms: tuple[ForceTerm, ...],
    *,
    neighbor_manager: NeighborListManager | None,
) -> None:
    for term in force_terms:
        topology = getattr(term, "topology", None)
        if topology is None or getattr(topology, "nonbonded_pair_policy", None) != "lazy":
            continue
        if neighbor_manager is not None:
            continue
        msg = (
            "large lazy topology requires compact periodic neighbor pairs; "
            "dense/tiled all-pairs fallback is refused"
        )
        raise ValueError(msg)


def _nonbonded_runtime_report(
    positions: mx.array,
    *,
    neighbor_manager: NeighborListManager | None,
    neighbor_list: NeighborList | None,
    force_evaluation_wall_seconds: float = 0.0,
) -> dict[str, int | float | str | None]:
    if neighbor_list is None:
        return {
            "backend": "dense_all_pairs",
            "pair_count": _dense_pair_count(positions),
            "cutoff": None,
            "skin": None,
            "rebuild_count": 0,
            "estimated_pair_memory_bytes": _dense_pair_count(positions) * 2 * 4,
            "estimated_cell_list_memory_bytes": 0,
            "representation_kind": "pairs",
            "candidate_count": _dense_pair_count(positions),
            "estimated_candidate_memory_bytes": 0,
            "compaction_backend": None,
            "fallback_reason": None,
            "neighbor_update_wall_seconds": 0.0,
            "neighbor_rebuild_wall_seconds": 0.0,
            "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
        }
    return {
        "backend": neighbor_list.backend,
        "pair_count": neighbor_list.pair_count,
        "cutoff": neighbor_list.cutoff,
        "skin": neighbor_list.skin,
        "rebuild_count": 0 if neighbor_manager is None else neighbor_manager.rebuild_count,
        "estimated_pair_memory_bytes": neighbor_list.estimated_pair_bytes,
        "estimated_cell_list_memory_bytes": neighbor_list.estimated_cell_list_bytes,
        "representation_kind": neighbor_list.representation_kind,
        "candidate_count": neighbor_list.candidate_count,
        "estimated_candidate_memory_bytes": neighbor_list.estimated_candidate_bytes,
        "compaction_backend": neighbor_list.compaction_backend,
        "fallback_reason": neighbor_list.fallback_reason,
        "neighbor_update_wall_seconds": (
            0.0 if neighbor_manager is None else neighbor_manager.update_wall_seconds
        ),
        "neighbor_rebuild_wall_seconds": (
            0.0 if neighbor_manager is None else neighbor_manager.rebuild_wall_seconds
        ),
        "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
    }


def _eval_step_state(
    state: SimulationState,
    potential_energy: mx.array,
    kinetic_energy_value: mx.array,
    temperature_value: mx.array,
    virial_value: mx.array,
    pressure_tensor_value: mx.array,
    pressure_value: mx.array,
    constraint_error: mx.array,
    energy_by_term: dict[str, mx.array],
) -> None:
    mx.eval(
        state.positions,
        state.velocities,
        state.forces,
        potential_energy,
        kinetic_energy_value,
        temperature_value,
        virial_value,
        pressure_tensor_value,
        pressure_value,
        constraint_error,
        *energy_by_term.values(),
    )


def _eval_runtime_state(
    state: SimulationState,
    potential_energy: mx.array,
    constraint_error: mx.array,
) -> None:
    mx.eval(
        state.positions,
        state.velocities,
        state.forces,
        potential_energy,
        constraint_error,
    )


def _is_diagnostic_step(step: int, config: SimulationConfig, *, final: bool = False) -> bool:
    return step % config.diagnostic_interval == 0 or final


def _normalize_reporters(
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None,
) -> tuple[RuntimeReporter, ...]:
    if reporters is None:
        return ()
    if isinstance(reporters, (list, tuple)):
        return tuple(reporters)
    return (reporters,)


def _notify_reporters(
    reporters: tuple[RuntimeReporter, ...],
    event: ReporterEvent,
) -> None:
    for reporter in reporters:
        reporter(event)


def _zero_constraint_error(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _materialize_sampled_state(state: SimulationState) -> None:
    # Sampled frames may be retained until trajectory serialization; force
    # evaluation so long sampled runs do not retain unevaluated step graphs.
    mx.eval(state.positions, state.velocities)


def _local_prng_key(seed: int | None) -> mx.array:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return mx.random.key(seed)


def _advance_prng_key(key: mx.array, steps: int) -> mx.array:
    for _ in range(int(steps)):
        key = mx.random.split(key, 2)[0]
    return key


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
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NVEResult:
    """Run NVE molecular dynamics with sparse trajectory and configurable diagnostics.

    `sample_interval` controls trajectory storage. `diagnostic_interval`
    controls energy, temperature, pair-count, and constraint diagnostics.
    """

    if config is None:
        config = SimulationConfig()
    reporters_tuple = _normalize_reporters(reporters)
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _named_force_terms(force_terms)
    unnamed_terms = tuple(term for _, term in terms)
    validate_virial_support(unnamed_terms)
    _validate_compact_nonbonded_backend(
        unnamed_terms,
        neighbor_manager=neighbor_manager,
    )

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    constraint_error = _zero_constraint_error(positions)
    if constraints is not None:
        positions, constraint_error = constraints.apply_positions(positions, masses, cell)
        velocities = constraints.apply_velocities(positions, velocities, masses, cell)
    temperature_dof = _temperature_degrees_of_freedom(positions, constraints)

    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    pair_count = _dense_pair_count(positions) if neighbor_list is None else neighbor_list.pair_count
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
    )
    energy_forces = _make_energy_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
    )
    force_start = perf_counter()
    potential_energy, forces, energy_by_term = energy_forces_by_term(positions)
    force_evaluation_wall_seconds += perf_counter() - force_start
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
        step=config.initial_step,
        time=config.initial_time,
    )

    _materialize_sampled_state(state)
    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [config.initial_step]
    sampled_times = [config.initial_time]
    diagnostic_steps = [config.initial_step]
    diagnostic_times = [config.initial_time]
    potential_energies = [potential_energy]
    potential_energy_by_term = {name: [energy] for name, energy in energy_by_term.items()}
    kinetic_energies = [
        kinetic_energy(
            state.velocities,
            masses,
            kinetic_energy_scale=config.kinetic_energy_scale,
        )
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
    virial, pressure_tensor_value, pressure_value = _pressure_diagnostics(
        state.positions,
        state.velocities,
        masses,
        state.forces,
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        kinetic_energy_scale=config.kinetic_energy_scale,
        enabled=config.pressure_diagnostics,
    )
    virials = [virial]
    pressure_tensors = [pressure_tensor_value]
    pressures = [pressure_value]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]
    constraint_errors = [constraint_error]
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nve",
            event_type="sample",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
        ),
    )
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nve",
            event_type="diagnostic",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energies[-1],
            total_energy=potential_energy + kinetic_energies[-1],
            temperature=temperatures[-1],
            energy_by_term=energy_by_term,
            virial_tensor=virial,
            pressure_tensor=pressure_tensor_value,
            pressure=pressure_value,
            pair_count=pair_count,
            rebuild_count=rebuild_count,
            constraint_max_error=constraint_error,
        ),
    )

    for local_step in range(1, config.steps + 1):
        current_step = config.initial_step + local_step
        current_time = config.initial_time + local_step * config.dt
        acceleration = config.force_to_acceleration_scale * state.forces / masses[:, None]
        velocities_half = state.velocities + 0.5 * config.dt * acceleration
        next_positions = state.positions + config.dt * velocities_half
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

        diagnostic_step = _is_diagnostic_step(
            current_step,
            config,
            final=local_step == config.steps,
        )
        force_start = perf_counter()
        if neighbor_manager is None and diagnostic_step:
            potential_energy, next_forces, energy_by_term = energy_forces_by_term(next_positions)
        elif neighbor_manager is None:
            potential_energy, next_forces = energy_forces(next_positions)
            energy_by_term = None
        elif diagnostic_step:
            potential_energy, next_forces, energy_by_term = _energy_forces_by_term(
                next_positions,
                terms,
                cell=cell,
                pairs=pairs,
            )
        else:
            potential_energy, next_forces = _energy_forces_from_terms(
                next_positions,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
            )
            energy_by_term = None
        force_evaluation_wall_seconds += perf_counter() - force_start
        next_acceleration = config.force_to_acceleration_scale * next_forces / masses[:, None]
        next_velocities = velocities_half + 0.5 * config.dt * next_acceleration
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
            step=current_step,
            time=current_time,
        )

        if current_step % config.sample_interval == 0 or local_step == config.steps:
            _materialize_sampled_state(state)
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(current_step)
            sampled_times.append(state.time)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nve",
                    event_type="sample",
                    step=current_step,
                    time=state.time,
                    state=state,
                ),
            )
        if diagnostic_step:
            diagnostic_steps.append(current_step)
            diagnostic_times.append(state.time)
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
            virial, pressure_tensor_value, pressure_value = _pressure_diagnostics(
                state.positions,
                state.velocities,
                masses,
                state.forces,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
                kinetic_energy_scale=config.kinetic_energy_scale,
                enabled=config.pressure_diagnostics,
            )
            virials.append(virial)
            pressure_tensors.append(pressure_tensor_value)
            pressures.append(pressure_value)
            pair_counts.append(pair_count)
            rebuild_counts.append(rebuild_count)
            constraint_errors.append(constraint_error)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nve",
                    event_type="diagnostic",
                    step=current_step,
                    time=state.time,
                    state=state,
                    potential_energy=potential_energy,
                    kinetic_energy=kinetic_energies[-1],
                    total_energy=potential_energy + kinetic_energies[-1],
                    temperature=temperatures[-1],
                    energy_by_term={} if energy_by_term is None else energy_by_term,
                    virial_tensor=virial,
                    pressure_tensor=pressure_tensor_value,
                    pressure=pressure_value,
                    pair_count=pair_count,
                    rebuild_count=rebuild_count,
                    constraint_max_error=constraint_error,
                ),
            )
        if current_step % config.evaluation_interval == 0 or local_step == config.steps:
            if diagnostic_step and energy_by_term is not None:
                _eval_step_state(
                    state,
                    potential_energy,
                    kinetic_energies[-1],
                    temperatures[-1],
                    virials[-1],
                    pressure_tensors[-1],
                    pressures[-1],
                    constraint_error,
                    energy_by_term,
                )
            else:
                _eval_runtime_state(state, potential_energy, constraint_error)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return NVEResult(
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
        virial_tensor=mx.stack(virials),
        pressure_tensor=mx.stack(pressure_tensors),
        pressure=mx.stack(pressures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        constraint_max_error=mx.stack(constraint_errors),
        final_state=state,
        nonbonded_report=_nonbonded_runtime_report(
            state.positions,
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
        ),
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
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NVTResult:
    """Run Langevin NVT molecular dynamics with BAOAB integration."""

    if config is None:
        config = SimulationConfig()
    reporters_tuple = _normalize_reporters(reporters)
    if thermostat is None:
        thermostat = LangevinThermostat()
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _named_force_terms(force_terms)
    unnamed_terms = tuple(term for _, term in terms)
    validate_virial_support(unnamed_terms)
    _validate_compact_nonbonded_backend(
        unnamed_terms,
        neighbor_manager=neighbor_manager,
    )

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    constraint_error = _zero_constraint_error(positions)
    if constraints is not None:
        positions, constraint_error = constraints.apply_positions(positions, masses, cell)
        velocities = constraints.apply_velocities(positions, velocities, masses, cell)
    temperature_dof = _temperature_degrees_of_freedom(positions, constraints)

    neighbor_list = neighbor_manager.update(positions) if neighbor_manager is not None else None
    pairs = None if neighbor_list is None else neighbor_list.pairs
    pair_count = _dense_pair_count(positions) if neighbor_list is None else neighbor_list.pair_count
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
    )
    energy_forces = _make_energy_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
    )
    force_start = perf_counter()
    potential_energy, forces, energy_by_term = energy_forces_by_term(positions)
    force_evaluation_wall_seconds += perf_counter() - force_start
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
        step=config.initial_step,
        time=config.initial_time,
    )

    _materialize_sampled_state(state)
    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [config.initial_step]
    sampled_times = [config.initial_time]
    diagnostic_steps = [config.initial_step]
    diagnostic_times = [config.initial_time]
    potential_energies = [potential_energy]
    potential_energy_by_term = {name: [energy] for name, energy in energy_by_term.items()}
    kinetic_energies = [
        kinetic_energy(
            state.velocities,
            masses,
            kinetic_energy_scale=config.kinetic_energy_scale,
        )
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
    virial, pressure_tensor_value, pressure_value = _pressure_diagnostics(
        state.positions,
        state.velocities,
        masses,
        state.forces,
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        kinetic_energy_scale=config.kinetic_energy_scale,
        enabled=config.pressure_diagnostics,
    )
    virials = [virial]
    pressure_tensors = [pressure_tensor_value]
    pressures = [pressure_value]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]
    constraint_errors = [constraint_error]
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nvt",
            event_type="sample",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
        ),
    )
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nvt",
            event_type="diagnostic",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energies[-1],
            total_energy=potential_energy + kinetic_energies[-1],
            temperature=temperatures[-1],
            energy_by_term=energy_by_term,
            virial_tensor=virial,
            pressure_tensor=pressure_tensor_value,
            pressure=pressure_value,
            pair_count=pair_count,
            rebuild_count=rebuild_count,
            constraint_max_error=constraint_error,
        ),
    )

    key = _local_prng_key(thermostat.seed)
    rng_step_offset = (
        config.initial_step
        if thermostat.rng_step_offset is None
        else thermostat.rng_step_offset
    )
    key = _advance_prng_key(key, rng_step_offset)
    velocity_decay = exp(-thermostat.friction * config.dt)
    noise_scale = sqrt(
        (1.0 - velocity_decay * velocity_decay)
        * thermostat.temperature
        * config.boltzmann_constant
        / config.kinetic_energy_scale
    )

    for local_step in range(1, config.steps + 1):
        current_step = config.initial_step + local_step
        current_time = config.initial_time + local_step * config.dt
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

        diagnostic_step = _is_diagnostic_step(
            current_step,
            config,
            final=local_step == config.steps,
        )
        force_start = perf_counter()
        if neighbor_manager is None and diagnostic_step:
            potential_energy, next_forces, energy_by_term = energy_forces_by_term(next_positions)
        elif neighbor_manager is None:
            potential_energy, next_forces = energy_forces(next_positions)
            energy_by_term = None
        elif diagnostic_step:
            potential_energy, next_forces, energy_by_term = _energy_forces_by_term(
                next_positions,
                terms,
                cell=cell,
                pairs=pairs,
            )
        else:
            potential_energy, next_forces = _energy_forces_from_terms(
                next_positions,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
            )
            energy_by_term = None
        force_evaluation_wall_seconds += perf_counter() - force_start
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
            step=current_step,
            time=current_time,
        )

        if current_step % config.sample_interval == 0 or local_step == config.steps:
            _materialize_sampled_state(state)
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(current_step)
            sampled_times.append(state.time)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nvt",
                    event_type="sample",
                    step=current_step,
                    time=state.time,
                    state=state,
                ),
            )
        if diagnostic_step:
            diagnostic_steps.append(current_step)
            diagnostic_times.append(state.time)
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
            virial, pressure_tensor_value, pressure_value = _pressure_diagnostics(
                state.positions,
                state.velocities,
                masses,
                state.forces,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
                kinetic_energy_scale=config.kinetic_energy_scale,
                enabled=config.pressure_diagnostics,
            )
            virials.append(virial)
            pressure_tensors.append(pressure_tensor_value)
            pressures.append(pressure_value)
            pair_counts.append(pair_count)
            rebuild_counts.append(rebuild_count)
            constraint_errors.append(constraint_error)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nvt",
                    event_type="diagnostic",
                    step=current_step,
                    time=state.time,
                    state=state,
                    potential_energy=potential_energy,
                    kinetic_energy=kinetic_energies[-1],
                    total_energy=potential_energy + kinetic_energies[-1],
                    temperature=temperatures[-1],
                    energy_by_term={} if energy_by_term is None else energy_by_term,
                    virial_tensor=virial,
                    pressure_tensor=pressure_tensor_value,
                    pressure=pressure_value,
                    pair_count=pair_count,
                    rebuild_count=rebuild_count,
                    constraint_max_error=constraint_error,
                ),
            )
        if current_step % config.evaluation_interval == 0 or local_step == config.steps:
            if diagnostic_step and energy_by_term is not None:
                _eval_step_state(
                    state,
                    potential_energy,
                    kinetic_energies[-1],
                    temperatures[-1],
                    virials[-1],
                    pressure_tensors[-1],
                    pressures[-1],
                    constraint_error,
                    energy_by_term,
                )
            else:
                _eval_runtime_state(state, potential_energy, constraint_error)

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return NVTResult(
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
        virial_tensor=mx.stack(virials),
        pressure_tensor=mx.stack(pressure_tensors),
        pressure=mx.stack(pressures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        constraint_max_error=mx.stack(constraint_errors),
        final_state=state,
        target_temperature=thermostat.temperature,
        nonbonded_report=_nonbonded_runtime_report(
            state.positions,
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
        ),
    )


def simulate_npt(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: LangevinThermostat | None = None,
    barostat: MonteCarloBarostat | None = None,
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NPTResult:
    """Run NVT dynamics followed by isotropic Monte Carlo volume attempts."""

    if cell is None:
        msg = "NPT simulation requires an orthorhombic periodic cell"
        raise ValueError(msg)
    if config is None:
        config = SimulationConfig()
    if thermostat is None:
        thermostat = LangevinThermostat()
    if barostat is None:
        barostat = MonteCarloBarostat(temperature=thermostat.temperature)
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = tuple(force_terms) if isinstance(force_terms, (list, tuple)) else (force_terms,)

    production = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=terms,
        neighbor_manager=neighbor_manager,
        config=config,
        thermostat=thermostat,
        constraints=constraints,
        reporters=reporters,
    )
    final_state, final_cell, accepted = _attempt_isotropic_barostat_move(
        production.final_state,
        terms,
        cell,
        barostat=barostat,
        constraints=constraints,
        boltzmann_constant=config.boltzmann_constant,
    )
    volumes = mx.array(
        [float(np.prod(np.asarray(cell.lengths))), float(np.prod(np.asarray(final_cell.lengths)))],
        dtype=mx.float32,
    )
    return NPTResult(
        production=production,
        final_state=final_state,
        final_cell=final_cell,
        cell_lengths=mx.stack([cell.lengths, final_cell.lengths]),
        volume=volumes,
        target_pressure=barostat.pressure,
        barostat_attempts=1,
        barostat_accepted=int(accepted),
    )


def _attempt_isotropic_barostat_move(
    state: SimulationState,
    force_terms: tuple[ForceTerm, ...],
    cell: Cell,
    *,
    barostat: MonteCarloBarostat,
    constraints: DistanceConstraints | None,
    boltzmann_constant: float,
) -> tuple[SimulationState, Cell, bool]:
    rng = np.random.default_rng(barostat.seed)
    log_volume_scale = rng.uniform(
        -barostat.max_log_volume_scale,
        barostat.max_log_volume_scale,
    )
    volume_scale = float(np.exp(log_volume_scale))
    length_scale = volume_scale ** (1.0 / 3.0)
    proposed_lengths = cell.lengths * length_scale
    proposed_cell = Cell(proposed_lengths)
    fractional = state.positions / cell.lengths
    proposed_positions = fractional * proposed_cell.lengths
    proposed_velocities = state.velocities
    constraint_error = _zero_constraint_error(proposed_positions)
    if constraints is not None:
        proposed_positions, constraint_error = constraints.apply_positions(
            proposed_positions,
            state.masses,
            proposed_cell,
        )
        proposed_velocities = constraints.apply_velocities(
            proposed_positions,
            proposed_velocities,
            state.masses,
            proposed_cell,
        )

    old_energy, _ = _energy_forces_from_terms(state.positions, force_terms, cell=cell, pairs=None)
    new_energy, new_forces = _energy_forces_from_terms(
        proposed_positions,
        force_terms,
        cell=proposed_cell,
        pairs=None,
    )
    old_volume = float(np.prod(np.asarray(cell.lengths)))
    new_volume = float(np.prod(np.asarray(proposed_cell.lengths)))
    beta = 1.0 / (boltzmann_constant * barostat.temperature)
    atom_count = int(state.positions.shape[0])
    delta = (
        float(np.asarray(new_energy - old_energy))
        + barostat.pressure * (new_volume - old_volume)
        - atom_count / beta * float(np.log(new_volume / old_volume))
    )
    accepted = delta <= 0.0 or rng.random() < float(np.exp(-beta * delta))
    if not accepted:
        return state, cell, False
    mx.eval(proposed_positions, proposed_velocities, new_forces, constraint_error)
    return (
        SimulationState(
            positions=proposed_positions,
            velocities=proposed_velocities,
            masses=state.masses,
            forces=new_forces,
            step=state.step,
            time=state.time,
        ),
        proposed_cell,
        True,
    )
