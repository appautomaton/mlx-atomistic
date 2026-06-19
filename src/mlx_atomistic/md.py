"""Molecular dynamics primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from math import exp, sqrt
from time import perf_counter
from typing import Any, Protocol

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.metal_kernels import fused_lj_forces
from mlx_atomistic.neighbors import (
    NeighborBlocks,
    NeighborList,
    NeighborListManager,
    build_neighbor_list,
)
from mlx_atomistic.nonbonded import (
    DEFAULT_DENSE_MEMORY_BUDGET_BYTES,
    NonbondedBackend,
    NonbondedExecutionConfig,
    choose_nonbonded_backend,
    dense_lj_energy_forces,
    estimate_dense_nonbonded_bytes,
)
from mlx_atomistic.topology import Topology, _isin_sorted_codes
from mlx_atomistic.virtual_sites import VirtualSiteManager

RUNTIME_SYNC_REASONS = (
    "reporter",
    "diagnostic",
    "checkpoint",
    "final_state",
    "failure_check",
    "explicit_user_output",
)


class ForceTerm(Protocol):
    """Protocol for composable force terms."""

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: object | None = None,
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
    use_fused_kernel: bool = False

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
        pairs: object | None = None,
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

        if isinstance(pairs, NeighborBlocks):
            return self._block_energy_forces(positions, pairs, cell)
        if self.topology is not None:
            filtered_pairs, scales = self._topology_pairs_and_scales(pairs)
            return self._pair_energy_forces(positions, filtered_pairs, cell, scales=scales)
        if (
            self.use_fused_kernel
            and pairs is not None
            and isinstance(pairs, mx.array)
            and cell is not None
            and cell.is_orthorhombic
            and self.cutoff is not None
        ):
            return fused_lj_forces(
                positions,
                pairs,
                mx.diag(cell.matrix),
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                shift=self.shift,
            )
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
        pairs = as_mx_array(pairs, dtype=mx.int32)
        forces = mx.zeros_like(positions)
        if pairs.shape[0] == 0:
            return mx.sum(positions[:, 0] * 0.0), forces
        if scales is None:
            scales = mx.array(1.0, dtype=mx.float32)

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

    def _block_mask_and_scales(self, blocks: NeighborBlocks) -> tuple[mx.array, mx.array]:
        if self.topology is None:
            return blocks.valid_mask, mx.array(1.0, dtype=mx.float32)

        cache_key = ("blocks", id(blocks), self.one_four_scale)
        cache = self._pair_scale_cache
        if cache is not None and cache[0] == cache_key:
            return cache[1]

        left = np.asarray(blocks.left, dtype=np.int32).reshape(-1)
        right = np.asarray(blocks.right, dtype=np.int32).reshape(-1)
        valid = np.asarray(blocks.valid_mask, dtype=bool).reshape(-1)
        n_atoms = self.topology.n_atoms
        if np.any(left[valid] < 0) or np.any(right[valid] < 0):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        if np.any(left[valid] >= n_atoms) or np.any(right[valid] >= n_atoms):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)

        normalized_left = np.minimum(left, right).astype(np.int64, copy=False)
        normalized_right = np.maximum(left, right).astype(np.int64, copy=False)
        codes = normalized_left * np.int64(n_atoms) + normalized_right
        keep = valid & ~_isin_sorted_codes(codes, self.topology._exclusion_codes)
        mask = mx.array(keep.reshape(blocks.left.shape))
        if float(self.one_four_scale) == 1.0 or self.topology._one_four_codes.size == 0:
            scales = mx.array(1.0, dtype=mx.float32)
        else:
            one_four = _isin_sorted_codes(codes, self.topology._one_four_codes)
            scales_np = np.where(one_four, float(self.one_four_scale), 1.0).astype(np.float32)
            scales = mx.array(scales_np.reshape(blocks.left.shape), dtype=mx.float32)
        object.__setattr__(self, "_pair_scale_cache", (cache_key, (mask, scales)))
        return mask, scales

    def _block_energy_forces(
        self,
        positions: mx.array,
        blocks: NeighborBlocks,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array]:
        forces = mx.zeros_like(positions)
        if blocks.candidate_count == 0:
            return mx.sum(positions[:, 0] * 0.0), forces

        i = blocks.left
        j = blocks.right
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        topology_mask, scales = self._block_mask_and_scales(blocks)
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = topology_mask & (r2 > 0.0)
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
        pair_forces = scalar[..., None] * displacement
        flat_i = mx.reshape(i, (-1,))
        flat_j = mx.reshape(j, (-1,))
        flat_forces = mx.reshape(pair_forces, (-1, 3))
        forces = forces.at[flat_i].add(flat_forces).at[flat_j].add(-flat_forces)

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
    virtual_sites: VirtualSiteManager | None = None
    block_size: int = 1

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
        if self.block_size < 1:
            msg = "block_size must be a positive integer (1 = per-step execution)"
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
    thermostat: dict[str, Any] = field(default_factory=dict)
    barostat: dict[str, Any] = field(default_factory=dict)


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
    runtime_sync_report: dict[str, int | float] = field(default_factory=dict)

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
class NoseHooverThermostat:
    """Deterministic single-variable Nose-Hoover thermostat parameters."""

    temperature: float = 1.0
    relaxation_time: float = 0.1
    thermal_mass: float | None = None
    chain_position: float = 0.0
    chain_velocity: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            msg = "temperature must be positive for Nose-Hoover"
            raise ValueError(msg)
        if self.relaxation_time <= 0.0:
            msg = "relaxation_time must be positive"
            raise ValueError(msg)
        if self.thermal_mass is not None and self.thermal_mass <= 0.0:
            msg = "thermal_mass must be positive when provided"
            raise ValueError(msg)


Thermostat = LangevinThermostat | NoseHooverThermostat


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
    thermostat_metadata: dict[str, Any] = field(default_factory=dict)
    nonbonded_report: dict[str, int | float | str | None] = field(default_factory=dict)
    runtime_sync_report: dict[str, int | float] = field(default_factory=dict)

    @property
    def temperature_error(self) -> mx.array:
        """Instantaneous temperature minus the target thermostat temperature."""

        return self.temperature - self.target_temperature


@dataclass(frozen=True)
class MonteCarloBarostat:
    """Monte Carlo barostat parameters for isotropic, anisotropic, and membrane NPT."""

    pressure: float = 1.0
    temperature: float = 1.0
    interval: int = 25
    max_log_volume_scale: float = 0.02
    seed: int | None = 11
    mode: str = "isotropic"
    axes: tuple[bool, bool, bool] = (True, True, True)
    membrane_plane: str = "xy"
    normal_axis: str = "z"

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
        mode = _normalize_barostat_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        axes = tuple(bool(axis) for axis in self.axes)
        if len(axes) != 3:
            msg = "barostat axes must contain three booleans"
            raise ValueError(msg)
        object.__setattr__(self, "axes", axes)
        if mode == "anisotropic" and not any(axes):
            msg = "anisotropic barostat requires at least one enabled axis"
            raise ValueError(msg)
        plane_axes = _barostat_plane_axes(self.membrane_plane)
        normal_axis = _barostat_axis_index(self.normal_axis)
        if mode == "membrane" and normal_axis in plane_axes:
            msg = "membrane normal_axis must be outside membrane_plane"
            raise ValueError(msg)
        object.__setattr__(self, "membrane_plane", "".join("xyz"[axis] for axis in plane_axes))
        object.__setattr__(self, "normal_axis", "xyz"[normal_axis])


@dataclass(frozen=True)
class NPTResult:
    """NPT production result with delegated NVT trajectory fields."""

    production: NVTResult
    final_state: SimulationState
    final_cell: Cell
    cell_lengths: mx.array
    cell_matrix: mx.array
    volume: mx.array
    target_pressure: float
    barostat_attempts: int
    barostat_accepted: int
    barostat_metadata: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        return getattr(self.production, name)

    @property
    def cell_history(self) -> mx.array:
        """Return the sampled cell-matrix history for pressure-coupled runs."""

        return self.cell_matrix


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
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
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
    if float(np.asarray(cell.volume)) <= 0.0:
        msg = "virial diagnostics require positive cell volume"
        raise ValueError(msg)

    fractional = cell.fractional_coordinates(positions)
    fractional = fractional - mx.floor(fractional)
    diagonal = []
    for axis in range(3):
        plus_cell = Cell(_strained_cell_matrix(cell.matrix, axis, strain_epsilon))
        minus_cell = Cell(_strained_cell_matrix(cell.matrix, axis, -strain_epsilon))
        plus_energy = _potential_energy_for_virial(
            plus_cell.cartesian_coordinates(fractional),
            force_terms,
            cell=plus_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        minus_energy = _potential_energy_for_virial(
            minus_cell.cartesian_coordinates(fractional),
            force_terms,
            cell=minus_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        diagonal.append(-(plus_energy - minus_energy) / (2.0 * strain_epsilon))
    return mx.diag(mx.stack(diagonal))


def _strained_cell_matrix(matrix: mx.array, axis: int, strain: float) -> mx.array:
    factors = as_mx_array([1.0 + strain if item == axis else 1.0 for item in range(3)])
    return matrix * factors[:, None]


def _potential_energy_for_virial(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> mx.array:
    energy, _ = _energy_forces_from_terms(
        positions,
        force_terms,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
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
    pairs: object | None,
    kinetic_energy_scale: float = 1.0,
    virtual_sites: VirtualSiteManager | None = None,
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
        virtual_sites=virtual_sites,
    )
    if cell is None:
        zeros = mx.zeros((3, 3), dtype=virial.dtype)
        return virial, zeros, mx.sum(virial * 0.0)
    volume = cell.volume
    if float(np.asarray(volume)) <= 0.0:
        msg = "pressure diagnostics require positive cell volume"
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
    pairs: object | None,
    kinetic_energy_scale: float,
    enabled: bool,
    virtual_sites: VirtualSiteManager | None = None,
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
            virtual_sites=virtual_sites,
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
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> tuple[mx.array, mx.array]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    grouped_terms = _groupable_potential_terms(force_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(eval_positions)
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            eval_positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for term in force_terms:
        if id(term) in grouped_ids:
            continue
        energy, forces = term.energy_forces(eval_positions, cell, pairs=pairs)
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, _redistribute_virtual_site_forces(
        total_forces,
        eval_positions,
        virtual_sites,
    )


def _energy_forces_by_term(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    unnamed_terms = tuple(term for _, term in force_terms)
    grouped_terms = _groupable_potential_terms(unnamed_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(eval_positions)
    energy_by_term = {}
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            eval_positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for name, term in force_terms:
        if id(term) in grouped_ids:
            energy_by_term[name] = term.potential_energy(eval_positions, cell)
            continue

        combined_components = getattr(term, "energy_forces_with_components", None)
        if callable(combined_components):
            energy, forces, components = combined_components(eval_positions, cell, pairs=pairs)
            for component_name, component_energy in components.items():
                if _is_energy_component(component_energy):
                    energy_by_term[f"{name}.{component_name}"] = component_energy
        else:
            energy, forces = term.energy_forces(eval_positions, cell, pairs=pairs)
            component_energies = getattr(term, "component_energies", None)
            if callable(component_energies):
                for component_name, component_energy in component_energies(
                    eval_positions,
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
    return (
        total_energy,
        _redistribute_virtual_site_forces(total_forces, eval_positions, virtual_sites),
        energy_by_term,
    )


def _virtual_site_evaluation_positions(
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return positions
    if positions.shape[0] != virtual_sites.n_real_atoms:
        msg = "positions must contain real atoms only when virtual sites are configured"
        raise ValueError(msg)
    return virtual_sites.extend_positions(positions)


def _redistribute_virtual_site_forces(
    forces: mx.array,
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return forces
    return virtual_sites.redistribute_forces(forces, positions)


def _neighbor_evaluation_positions(
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    return _virtual_site_evaluation_positions(positions, virtual_sites)


def _make_energy_forces_by_term_evaluator(
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    compile_evaluator: bool,
    virtual_sites: VirtualSiteManager | None = None,
):
    def evaluate(pos: mx.array):
        return _energy_forces_by_term(
            pos,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )

    if compile_evaluator and pairs is None and virtual_sites is None:
        return mx.compile(evaluate)
    return evaluate


def _is_energy_component(value: object) -> bool:
    return isinstance(value, (mx.array, int, float))


def _make_energy_forces_evaluator(
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    compile_evaluator: bool,
    virtual_sites: VirtualSiteManager | None = None,
):
    def evaluate(pos: mx.array):
        return _energy_forces_from_terms(
            pos,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )

    if compile_evaluator and pairs is None and virtual_sites is None:
        return mx.compile(evaluate)
    return evaluate


def _groupable_potential_terms(
    force_terms: tuple[ForceTerm, ...],
    pairs: object | None,
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


def _zero_reason_ints() -> dict[str, int]:
    return {reason: 0 for reason in RUNTIME_SYNC_REASONS}


def _zero_reason_floats() -> dict[str, float]:
    return {reason: 0.0 for reason in RUNTIME_SYNC_REASONS}


@dataclass
class _RuntimeSyncRecorder:
    sync_counts: dict[str, int] = field(default_factory=_zero_reason_ints)
    sync_wall_seconds: dict[str, float] = field(default_factory=_zero_reason_floats)
    materialization_counts: dict[str, int] = field(default_factory=_zero_reason_ints)
    materialization_wall_seconds: dict[str, float] = field(default_factory=_zero_reason_floats)

    def record_sync(self, reason: str, *values: mx.array) -> float:
        _validate_runtime_sync_reason(reason)
        start = perf_counter()
        mx.eval(*values)
        elapsed = perf_counter() - start
        self.sync_counts[reason] += 1
        self.sync_wall_seconds[reason] += elapsed
        return elapsed

    def record_materialization(self, reason: str, elapsed: float = 0.0) -> None:
        _validate_runtime_sync_reason(reason)
        self.materialization_counts[reason] += 1
        self.materialization_wall_seconds[reason] += elapsed

    def record_callback(self, reason: str, callback) -> None:
        _validate_runtime_sync_reason(reason)
        start = perf_counter()
        callback()
        self.record_materialization(reason, perf_counter() - start)

    def to_report(self) -> dict[str, int | float]:
        report: dict[str, int | float] = {
            "runtime_sync_total_count": sum(self.sync_counts.values()),
            "runtime_sync_total_wall_seconds": sum(self.sync_wall_seconds.values()),
            "runtime_materialization_total_count": sum(self.materialization_counts.values()),
            "runtime_materialization_total_wall_seconds": sum(
                self.materialization_wall_seconds.values()
            ),
        }
        for reason in RUNTIME_SYNC_REASONS:
            report[f"runtime_sync_{reason}_count"] = self.sync_counts[reason]
            report[f"runtime_sync_{reason}_wall_seconds"] = self.sync_wall_seconds[reason]
            report[f"runtime_materialization_{reason}_count"] = (
                self.materialization_counts[reason]
            )
            report[f"runtime_materialization_{reason}_wall_seconds"] = (
                self.materialization_wall_seconds[reason]
            )
        return report


def _validate_runtime_sync_reason(reason: str) -> None:
    if reason not in RUNTIME_SYNC_REASONS:
        msg = f"unknown runtime sync reason: {reason}"
        raise ValueError(msg)


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
    runtime_sync_report: dict[str, int | float] | None = None,
) -> dict[str, int | float | str | None]:
    if neighbor_list is None:
        dense_pair_count = _dense_pair_count(positions)
        report = {
            "backend": "dense_all_pairs",
            "pair_count": dense_pair_count,
            "compact_pair_count": dense_pair_count,
            "cutoff": None,
            "skin": None,
            "rebuild_count": 0,
            "estimated_pair_memory_bytes": dense_pair_count * 2 * 4,
            "estimated_compact_pair_memory_bytes": dense_pair_count * 2 * 4,
            "estimated_cell_list_memory_bytes": 0,
            "representation_kind": "pairs",
            "candidate_count": dense_pair_count,
            "candidate_waste_count": 0,
            "candidate_waste_fraction": 0.0,
            "estimated_candidate_memory_bytes": 0,
            "compaction_backend": None,
            "fallback_reason": None,
            "neighbor_update_wall_seconds": 0.0,
            "neighbor_rebuild_wall_seconds": 0.0,
            "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
        }
        return _with_runtime_sync_report(report, runtime_sync_report)
    report = {
        "backend": neighbor_list.backend,
        "pair_count": neighbor_list.pair_count,
        "compact_pair_count": neighbor_list.compact_pair_count,
        "cutoff": neighbor_list.cutoff,
        "skin": neighbor_list.skin,
        "rebuild_count": 0 if neighbor_manager is None else neighbor_manager.rebuild_count,
        "estimated_pair_memory_bytes": neighbor_list.estimated_pair_bytes,
        "estimated_compact_pair_memory_bytes": neighbor_list.estimated_compact_pair_bytes,
        "estimated_cell_list_memory_bytes": neighbor_list.estimated_cell_list_bytes,
        "representation_kind": neighbor_list.representation_kind,
        "candidate_count": neighbor_list.candidate_count,
        "candidate_waste_count": neighbor_list.candidate_waste_count,
        "candidate_waste_fraction": neighbor_list.candidate_waste_fraction,
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
    return _with_runtime_sync_report(report, runtime_sync_report)


def _with_runtime_sync_report(
    report: dict[str, int | float | str | None],
    runtime_sync_report: dict[str, int | float] | None,
) -> dict[str, int | float | str | None]:
    if runtime_sync_report is None:
        return report
    report.update(runtime_sync_report)
    return report


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
    *,
    evaluate_sampled_state: bool = True,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    sync_reason: str = "diagnostic",
) -> float:
    state_values = (
        (state.positions, state.velocities)
        if evaluate_sampled_state
        else ()
    )
    values = (
        *state_values,
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
    if runtime_sync is not None:
        return runtime_sync.record_sync(sync_reason, *values)
    start = perf_counter()
    mx.eval(*values)
    return perf_counter() - start


def _eval_runtime_state(
    state: SimulationState,
    potential_energy: mx.array,
    constraint_error: mx.array,
    *,
    evaluate_sampled_state: bool = True,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    sync_reason: str = "failure_check",
) -> float:
    state_values = (
        (state.positions, state.velocities)
        if evaluate_sampled_state
        else ()
    )
    values = (
        *state_values,
        state.forces,
        potential_energy,
        constraint_error,
    )
    if runtime_sync is not None:
        return runtime_sync.record_sync(sync_reason, *values)
    start = perf_counter()
    mx.eval(*values)
    return perf_counter() - start


def _is_diagnostic_step(step: int, config: SimulationConfig, *, final: bool = False) -> bool:
    return step % config.diagnostic_interval == 0 or final


def _langevin_block_execution_enabled(
    config: SimulationConfig,
    *,
    thermostat: object,
    neighbor_manager: NeighborListManager | None,
    constraints: object | None,
    virtual_sites: VirtualSiteManager | None,
) -> bool:
    """Whether the compiled batched-block fast path applies to this NVT run.

    The fast path runs `block_size` velocity-Verlet/Langevin substeps as one
    compiled block between neighbor rebuilds, syncing to the host once per block
    instead of every step. It is only safe when nothing in the loop needs
    per-step host interaction or per-step force bookkeeping: a Langevin
    thermostat (deterministic threaded PRNG, compiles cleanly), a managed
    neighbor list, no constraints, and no virtual sites. Block length is capped
    at the next sampling/diagnostic boundary at run time, so recording cadences
    need not divide `block_size` and no recorded step is ever skipped.
    """

    return (
        config.block_size > 1
        and isinstance(thermostat, LangevinThermostat)
        and neighbor_manager is not None
        and constraints is None
        and virtual_sites is None
    )


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
    *,
    runtime_sync: _RuntimeSyncRecorder | None = None,
) -> None:
    for reporter in reporters:
        if runtime_sync is None:
            reporter(event)
        else:
            runtime_sync.record_callback(
                "reporter",
                lambda reporter=reporter, event=event: reporter(event),
            )


def _zero_constraint_error(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _materialize_sampled_state(
    state: SimulationState,
    *,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    reason: str = "explicit_user_output",
) -> None:
    # Sampled frames may be retained until trajectory serialization; force
    # evaluation so long sampled runs do not retain unevaluated step graphs.
    if runtime_sync is None:
        mx.eval(state.positions, state.velocities)
        return
    runtime_sync.record_sync(reason, state.positions, state.velocities)
    runtime_sync.record_materialization(reason)


def _local_prng_key(seed: int | None) -> mx.array:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return mx.random.key(seed)


def _advance_prng_key(key: mx.array, steps: int) -> mx.array:
    for _ in range(int(steps)):
        key = mx.random.split(key, 2)[0]
    return key


def _nose_hoover_thermal_mass(
    thermostat: NoseHooverThermostat,
    *,
    dof: int,
    boltzmann_constant: float,
) -> float:
    if thermostat.thermal_mass is not None:
        return float(thermostat.thermal_mass)
    return (
        float(dof)
        * boltzmann_constant
        * thermostat.temperature
        * thermostat.relaxation_time
        * thermostat.relaxation_time
    )


def _thermostat_metadata(
    thermostat: Thermostat,
    *,
    dof: int,
    boltzmann_constant: float,
    chain_position: float | None = None,
    chain_velocity: float | None = None,
    rng_step_offset: int | None = None,
) -> dict[str, Any]:
    if isinstance(thermostat, NoseHooverThermostat):
        return {
            "family": "nose_hoover",
            "integrator": "nose_hoover_velocity_verlet",
            "deterministic_state": True,
            "temperature": float(thermostat.temperature),
            "relaxation_time": float(thermostat.relaxation_time),
            "thermal_mass": _nose_hoover_thermal_mass(
                thermostat,
                dof=dof,
                boltzmann_constant=boltzmann_constant,
            ),
            "chain_position": float(
                thermostat.chain_position if chain_position is None else chain_position
            ),
            "chain_velocity": float(
                thermostat.chain_velocity if chain_velocity is None else chain_velocity
            ),
        }
    return {
        "family": "langevin_baoab",
        "integrator": "baoab",
        "temperature": float(thermostat.temperature),
        "friction": float(thermostat.friction),
        "seed": thermostat.seed,
        "rng_step_offset": (
            thermostat.rng_step_offset if rng_step_offset is None else int(rng_step_offset)
        ),
    }


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
        pairs: object | None = None,
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
    pairs: object | None = None,
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
    virtual_sites = config.virtual_sites
    reporters_tuple = _normalize_reporters(reporters)
    runtime_sync = _RuntimeSyncRecorder()
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

    eval_positions = _neighbor_evaluation_positions(positions, virtual_sites)
    neighbor_list = (
        neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
    )
    pairs = None if neighbor_list is None else neighbor_list.interactions
    pair_count = (
        _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
    )
    energy_forces = _make_energy_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
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

    _materialize_sampled_state(
        state,
        runtime_sync=runtime_sync,
        reason="explicit_user_output",
    )
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
        virtual_sites=virtual_sites,
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
        runtime_sync=runtime_sync,
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
        runtime_sync=runtime_sync,
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

        eval_positions = _neighbor_evaluation_positions(next_positions, virtual_sites)
        neighbor_list = (
            neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
        )
        pairs = None if neighbor_list is None else neighbor_list.interactions
        pair_count = (
            _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
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
                virtual_sites=virtual_sites,
            )
        else:
            potential_energy, next_forces = _energy_forces_from_terms(
                next_positions,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
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

        sampled_state_evaluated = False
        if current_step % config.sample_interval == 0 or local_step == config.steps:
            _materialize_sampled_state(
                state,
                runtime_sync=runtime_sync,
                reason="final_state" if local_step == config.steps else "explicit_user_output",
            )
            sampled_state_evaluated = True
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
                runtime_sync=runtime_sync,
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
                virtual_sites=virtual_sites,
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
                runtime_sync=runtime_sync,
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
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="diagnostic",
                )
            else:
                _eval_runtime_state(
                    state,
                    potential_energy,
                    constraint_error,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="failure_check",
                )

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    runtime_sync_report = runtime_sync.to_report()
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
            _neighbor_evaluation_positions(state.positions, virtual_sites),
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
        runtime_sync_report=runtime_sync_report,
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
    """Run NVT molecular dynamics with Langevin BAOAB or Nose-Hoover dynamics."""

    if config is None:
        config = SimulationConfig()
    virtual_sites = config.virtual_sites
    reporters_tuple = _normalize_reporters(reporters)
    runtime_sync = _RuntimeSyncRecorder()
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

    eval_positions = _neighbor_evaluation_positions(positions, virtual_sites)
    neighbor_list = (
        neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
    )
    pairs = None if neighbor_list is None else neighbor_list.interactions
    pair_count = (
        _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
    )
    energy_forces = _make_energy_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
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

    _materialize_sampled_state(
        state,
        runtime_sync=runtime_sync,
        reason="explicit_user_output",
    )
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
    rng_step_offset = None
    key = None
    velocity_decay = None
    noise_scale = None
    nh_chain_position = None
    nh_chain_velocity = None
    nh_thermal_mass = None
    nh_target_kinetic = None
    if isinstance(thermostat, LangevinThermostat):
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
        thermostat_metadata = _thermostat_metadata(
            thermostat,
            dof=temperature_dof,
            boltzmann_constant=config.boltzmann_constant,
            rng_step_offset=rng_step_offset,
        )
    else:
        nh_chain_position = mx.array(float(thermostat.chain_position))
        nh_chain_velocity = mx.array(float(thermostat.chain_velocity))
        nh_thermal_mass = mx.array(
            _nose_hoover_thermal_mass(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
            )
        )
        nh_target_kinetic = mx.array(
            float(temperature_dof) * config.boltzmann_constant * thermostat.temperature
        )
        thermostat_metadata = _thermostat_metadata(
            thermostat,
            dof=temperature_dof,
            boltzmann_constant=config.boltzmann_constant,
            chain_position=float(np.asarray(nh_chain_position)),
            chain_velocity=float(np.asarray(nh_chain_velocity)),
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
        virtual_sites=virtual_sites,
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
            thermostat=thermostat_metadata,
        ),
        runtime_sync=runtime_sync,
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
            thermostat=thermostat_metadata,
        ),
        runtime_sync=runtime_sync,
    )

    _batched = _langevin_block_execution_enabled(
        config,
        thermostat=thermostat,
        neighbor_manager=neighbor_manager,
        constraints=constraints,
        virtual_sites=virtual_sites,
    )
    if _batched:
        fscale = config.force_to_acceleration_scale
        dt = config.dt
        # Use the same arithmetic as the per-step loop below (division by the
        # mass column, not multiply-by-reciprocal) so the batched trajectory is
        # bit-for-bit identical, not just close.
        masses_col = masses[:, None]
        sqrt_masses_col = mx.sqrt(masses)[:, None]

        def _langevin_substep(pos, vel, forces, prng, block_pairs):
            accel = fscale * forces / masses_col
            vel_half = vel + 0.5 * dt * accel
            pos = pos + 0.5 * dt * vel_half
            if cell is not None:
                pos = cell.wrap(pos)
            split_keys = mx.random.split(prng, 2)
            prng = split_keys[0]
            noise = mx.random.normal(vel.shape, key=split_keys[1])
            middle = velocity_decay * vel_half + (noise_scale / sqrt_masses_col) * noise
            pos = pos + 0.5 * dt * middle
            if cell is not None:
                pos = cell.wrap(pos)
            _, next_forces = _energy_forces_from_terms(
                pos, unnamed_terms, cell=cell, pairs=block_pairs, virtual_sites=None
            )
            next_accel = fscale * next_forces / masses_col
            vel = vel_half + 0.5 * dt * next_accel
            return pos, vel, next_forces, prng

        _block_cache: dict[int, object] = {}

        def _compiled_block(n_substeps: int):
            cached = _block_cache.get(n_substeps)
            if cached is not None:
                return cached

            def block(pos, vel, forces, prng, block_pairs):
                for _ in range(n_substeps):
                    pos, vel, forces, prng = _langevin_substep(
                        pos, vel, forces, prng, block_pairs
                    )
                return pos, vel, forces, prng

            compiled = mx.compile(block)
            _block_cache[n_substeps] = compiled
            return compiled

        def _next_recording_local_step(local_step: int) -> int:
            """Smallest local step > `local_step` that is a sampling, diagnostic,
            or final step — so a block never steps past a recorded boundary."""
            current = config.initial_step + local_step
            next_sample = ((current // config.sample_interval) + 1) * config.sample_interval
            next_diag = ((current // config.diagnostic_interval) + 1) * config.diagnostic_interval
            next_step = min(next_sample, next_diag) - config.initial_step
            return min(next_step, config.steps)

        def _run_langevin_batched(
            state, key, thermostat_metadata, pairs, pair_count, rebuild_count, fe_wall
        ):
            pos, vel, forces = state.positions, state.velocities, state.forces
            local_step = 0
            while local_step < config.steps:
                n = min(config.block_size, _next_recording_local_step(local_step) - local_step)
                pos, vel, forces, key = _compiled_block(n)(pos, vel, forces, key, pairs)
                local_step += n
                current_step = config.initial_step + local_step
                current_time = config.initial_time + local_step * config.dt

                force_start = perf_counter()
                neighbor_list = neighbor_manager.update(pos)
                pairs = neighbor_list.interactions
                pair_count = neighbor_list.pair_count
                rebuild_count = neighbor_manager.rebuild_count
                fe_wall += perf_counter() - force_start

                state = SimulationState(
                    positions=pos,
                    velocities=vel,
                    masses=masses,
                    forces=forces,
                    step=current_step,
                    time=current_time,
                )
                thermostat_metadata = _thermostat_metadata(
                    thermostat,
                    dof=temperature_dof,
                    boltzmann_constant=config.boltzmann_constant,
                    rng_step_offset=rng_step_offset + local_step,
                )

                diagnostic_step = _is_diagnostic_step(
                    current_step, config, final=local_step == config.steps
                )
                energy_by_term = None
                potential_energy = None
                if diagnostic_step:
                    force_start = perf_counter()
                    potential_energy, _, energy_by_term = _energy_forces_by_term(
                        pos, terms, cell=cell, pairs=pairs, virtual_sites=None
                    )
                    fe_wall += perf_counter() - force_start

                sampled_state_evaluated = False
                if current_step % config.sample_interval == 0 or local_step == config.steps:
                    _materialize_sampled_state(
                        state,
                        runtime_sync=runtime_sync,
                        reason="final_state"
                        if local_step == config.steps
                        else "explicit_user_output",
                    )
                    sampled_state_evaluated = True
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
                            thermostat=thermostat_metadata,
                        ),
                        runtime_sync=runtime_sync,
                    )
                if diagnostic_step:
                    diagnostic_steps.append(current_step)
                    diagnostic_times.append(state.time)
                    potential_energies.append(potential_energy)
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
                        virtual_sites=virtual_sites,
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
                            energy_by_term=energy_by_term,
                            virial_tensor=virial,
                            pressure_tensor=pressure_tensor_value,
                            pressure=pressure_value,
                            pair_count=pair_count,
                            rebuild_count=rebuild_count,
                            constraint_max_error=constraint_error,
                            thermostat=thermostat_metadata,
                        ),
                        runtime_sync=runtime_sync,
                    )
                if (
                    (current_step % config.evaluation_interval == 0 or local_step == config.steps)
                    and diagnostic_step
                    and energy_by_term is not None
                ):
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
                        evaluate_sampled_state=not sampled_state_evaluated,
                        runtime_sync=runtime_sync,
                        sync_reason="diagnostic",
                    )
                # Bound the lazy graph and catch divergence: one sync per block
                # (every block_size steps) instead of per step. The manager
                # update above already materialized positions; this covers the
                # velocity/force/PRNG state carried into the next block.
                runtime_sync.record_sync("failure_check", vel, forces, key)
            return (
                state,
                key,
                thermostat_metadata,
                pairs,
                pair_count,
                rebuild_count,
                fe_wall,
            )

        (
            state,
            key,
            thermostat_metadata,
            pairs,
            pair_count,
            rebuild_count,
            force_evaluation_wall_seconds,
        ) = _run_langevin_batched(
            state,
            key,
            thermostat_metadata,
            pairs,
            pair_count,
            rebuild_count,
            force_evaluation_wall_seconds,
        )

    step_range = range(0) if _batched else range(1, config.steps + 1)
    for local_step in step_range:
        current_step = config.initial_step + local_step
        current_time = config.initial_time + local_step * config.dt
        acceleration = config.force_to_acceleration_scale * state.forces / masses[:, None]
        if isinstance(thermostat, LangevinThermostat):
            velocities_half = state.velocities + 0.5 * config.dt * acceleration
            next_positions = state.positions + 0.5 * config.dt * velocities_half
            if cell is not None:
                next_positions = cell.wrap(next_positions)

            keys = mx.random.split(key, 2)
            key = keys[0]
            noise = mx.random.normal(state.velocities.shape, key=keys[1])
            thermal_scale = noise_scale / mx.sqrt(masses)[:, None]
            middle_velocities = velocity_decay * velocities_half + thermal_scale * noise

            next_positions = next_positions + 0.5 * config.dt * middle_velocities
        else:
            current_kinetic = kinetic_energy(
                state.velocities,
                masses,
                kinetic_energy_scale=config.kinetic_energy_scale,
            )
            nh_chain_velocity = nh_chain_velocity + 0.5 * config.dt * (
                (2.0 * current_kinetic - nh_target_kinetic) / nh_thermal_mass
            )
            thermostat_scale = mx.exp(-0.5 * config.dt * nh_chain_velocity)
            scaled_velocities = state.velocities * thermostat_scale
            velocities_half = scaled_velocities + 0.5 * config.dt * acceleration
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

        eval_positions = _neighbor_evaluation_positions(next_positions, virtual_sites)
        neighbor_list = (
            neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
        )
        pairs = None if neighbor_list is None else neighbor_list.interactions
        pair_count = (
            _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
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
                virtual_sites=virtual_sites,
            )
        else:
            potential_energy, next_forces = _energy_forces_from_terms(
                next_positions,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
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
        if isinstance(thermostat, NoseHooverThermostat):
            next_velocities = next_velocities * mx.exp(-0.5 * config.dt * nh_chain_velocity)
            next_kinetic = kinetic_energy(
                next_velocities,
                masses,
                kinetic_energy_scale=config.kinetic_energy_scale,
            )
            nh_chain_velocity = nh_chain_velocity + 0.5 * config.dt * (
                (2.0 * next_kinetic - nh_target_kinetic) / nh_thermal_mass
            )
            nh_chain_position = nh_chain_position + config.dt * nh_chain_velocity
        state = SimulationState(
            positions=next_positions,
            velocities=next_velocities,
            masses=masses,
            forces=next_forces,
            step=current_step,
            time=current_time,
        )
        if isinstance(thermostat, LangevinThermostat):
            thermostat_metadata = _thermostat_metadata(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
                rng_step_offset=rng_step_offset + local_step,
            )
        else:
            thermostat_metadata = _thermostat_metadata(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
                chain_position=float(np.asarray(nh_chain_position)),
                chain_velocity=float(np.asarray(nh_chain_velocity)),
            )

        sampled_state_evaluated = False
        if current_step % config.sample_interval == 0 or local_step == config.steps:
            _materialize_sampled_state(
                state,
                runtime_sync=runtime_sync,
                reason="final_state" if local_step == config.steps else "explicit_user_output",
            )
            sampled_state_evaluated = True
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
                    thermostat=thermostat_metadata,
                ),
                runtime_sync=runtime_sync,
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
                virtual_sites=virtual_sites,
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
                    thermostat=thermostat_metadata,
                ),
                runtime_sync=runtime_sync,
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
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="diagnostic",
                )
            else:
                _eval_runtime_state(
                    state,
                    potential_energy,
                    constraint_error,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="failure_check",
                )

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    runtime_sync_report = runtime_sync.to_report()
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
        thermostat_metadata=thermostat_metadata,
        nonbonded_report=_nonbonded_runtime_report(
            _neighbor_evaluation_positions(state.positions, virtual_sites),
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
        runtime_sync_report=runtime_sync_report,
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
    """Run NVT dynamics followed by a Monte Carlo pressure-coupling attempt."""

    if cell is None:
        msg = "NPT simulation requires a periodic cell"
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
    validate_virial_support(terms)
    _validate_barostat_cell_support(cell, barostat)

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
    final_state, final_cell, accepted = _attempt_barostat_move(
        production.final_state,
        terms,
        cell,
        barostat=barostat,
        constraints=constraints,
        boltzmann_constant=config.boltzmann_constant,
        neighbor_manager=neighbor_manager,
        virtual_sites=config.virtual_sites,
    )
    production = _npt_production_with_final_barostat_state(
        production,
        final_state,
        terms,
        final_cell,
        constraints=constraints,
        config=config,
        neighbor_manager=neighbor_manager,
    )
    volumes = mx.array(
        [float(np.asarray(cell.volume)), float(np.asarray(final_cell.volume))],
        dtype=mx.float32,
    )
    cell_matrix = mx.stack([cell.matrix, final_cell.matrix])
    barostat_metadata = _barostat_metadata(barostat)
    barostat_metadata.update(
        {
            "attempts": 1,
            "accepted": int(accepted),
            "target_pressure": barostat.pressure,
            "initial_volume": float(np.asarray(cell.volume)),
            "final_volume": float(np.asarray(final_cell.volume)),
        }
    )
    _notify_barostat_reporters(
        reporters,
        final_state=final_state,
        final_cell=final_cell,
        metadata=barostat_metadata,
    )
    return NPTResult(
        production=production,
        final_state=final_state,
        final_cell=final_cell,
        cell_lengths=mx.stack([cell.lengths, final_cell.lengths]),
        cell_matrix=cell_matrix,
        volume=volumes,
        target_pressure=barostat.pressure,
        barostat_attempts=1,
        barostat_accepted=int(accepted),
        barostat_metadata=barostat_metadata,
    )


def _npt_production_with_final_barostat_state(
    production: NVTResult,
    final_state: SimulationState,
    force_terms: tuple[ForceTerm, ...],
    final_cell: Cell,
    *,
    constraints: DistanceConstraints | None,
    config: SimulationConfig,
    neighbor_manager: NeighborListManager | None,
) -> NVTResult:
    virtual_sites = config.virtual_sites
    named_terms = _named_force_terms(force_terms)
    eval_positions = _neighbor_evaluation_positions(final_state.positions, virtual_sites)
    neighbor_list = (
        neighbor_manager.update(eval_positions)
        if neighbor_manager is not None
        else None
    )
    pairs = None if neighbor_list is None else neighbor_list.interactions
    potential_energy, _, energy_by_term = _energy_forces_by_term(
        final_state.positions,
        named_terms,
        cell=final_cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
    )
    kinetic_energy_value = kinetic_energy(
        final_state.velocities,
        final_state.masses,
        kinetic_energy_scale=config.kinetic_energy_scale,
    )
    temperature_dof = _temperature_degrees_of_freedom(final_state.positions, constraints)
    temperature_value = instantaneous_temperature(
        final_state.velocities,
        final_state.masses,
        dof=temperature_dof,
        kinetic_energy_scale=config.kinetic_energy_scale,
        boltzmann_constant=config.boltzmann_constant,
    )
    virial, pressure_tensor_value, pressure_value = _pressure_diagnostics(
        final_state.positions,
        final_state.velocities,
        final_state.masses,
        final_state.forces,
        tuple(term for _, term in named_terms),
        cell=final_cell,
        pairs=pairs,
        kinetic_energy_scale=config.kinetic_energy_scale,
        enabled=config.pressure_diagnostics,
        virtual_sites=virtual_sites,
    )
    constraint_error = (
        _zero_constraint_error(final_state.positions)
        if constraints is None
        else constraints.max_error(final_state.positions, final_cell)
    )
    updated_terms = {
        name: _replace_last_frame(values, energy_by_term[name])
        for name, values in production.potential_energy_by_term.items()
        if name in energy_by_term
    }
    pair_count = (
        _dense_pair_count(eval_positions)
        if neighbor_list is None
        else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = float(
        production.nonbonded_report.get("force_evaluation_wall_seconds", 0.0)
    )
    runtime_sync_report = production.runtime_sync_report
    return replace(
        production,
        sampled_positions=_replace_last_frame(production.sampled_positions, final_state.positions),
        sampled_velocities=_replace_last_frame(
            production.sampled_velocities,
            final_state.velocities,
        ),
        potential_energy=_replace_last_frame(production.potential_energy, potential_energy),
        kinetic_energy=_replace_last_frame(production.kinetic_energy, kinetic_energy_value),
        total_energy=_replace_last_frame(
            production.total_energy,
            potential_energy + kinetic_energy_value,
        ),
        potential_energy_by_term=updated_terms,
        temperature=_replace_last_frame(production.temperature, temperature_value),
        virial_tensor=_replace_last_frame(production.virial_tensor, virial),
        pressure_tensor=_replace_last_frame(production.pressure_tensor, pressure_tensor_value),
        pressure=_replace_last_frame(production.pressure, pressure_value),
        pair_count=_replace_last_frame(
            production.pair_count,
            mx.array(pair_count, dtype=mx.int32),
        ),
        rebuild_count=_replace_last_frame(
            production.rebuild_count,
            mx.array(rebuild_count, dtype=mx.int32),
        ),
        constraint_max_error=_replace_last_frame(
            production.constraint_max_error,
            constraint_error,
        ),
        final_state=final_state,
        nonbonded_report=_nonbonded_runtime_report(
            eval_positions,
            neighbor_manager=neighbor_manager,
            neighbor_list=neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
    )


def _replace_last_frame(frames: mx.array, frame: mx.array) -> mx.array:
    frame = as_mx_array(frame)
    if frames.shape[0] <= 1:
        return frame[None, ...]
    return mx.concatenate([frames[:-1], frame[None, ...]], axis=0)


def _attempt_barostat_move(
    state: SimulationState,
    force_terms: tuple[ForceTerm, ...],
    cell: Cell,
    *,
    barostat: MonteCarloBarostat,
    constraints: DistanceConstraints | None,
    boltzmann_constant: float,
    neighbor_manager: NeighborListManager | None = None,
    virtual_sites: VirtualSiteManager | None = None,
) -> tuple[SimulationState, Cell, bool]:
    rng = np.random.default_rng(barostat.seed)
    scale_factors = _barostat_scale_factors(barostat, rng)
    proposed_cell = _scaled_cell(cell, scale_factors)
    fractional = cell.fractional_coordinates(state.positions)
    proposed_positions = proposed_cell.cartesian_coordinates(fractional)
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

    old_eval_positions = _neighbor_evaluation_positions(state.positions, virtual_sites)
    proposed_eval_positions = _neighbor_evaluation_positions(proposed_positions, virtual_sites)
    old_neighbor_list = (
        neighbor_manager.update(old_eval_positions) if neighbor_manager is not None else None
    )
    old_pairs = None if old_neighbor_list is None else old_neighbor_list.interactions
    proposed_neighbor_list = _barostat_neighbor_list(
        proposed_eval_positions,
        proposed_cell,
        neighbor_manager,
    )
    proposed_pairs = None if proposed_neighbor_list is None else proposed_neighbor_list.interactions
    old_energy, _ = _energy_forces_from_terms(
        state.positions,
        force_terms,
        cell=cell,
        pairs=old_pairs,
        virtual_sites=virtual_sites,
    )
    new_energy, new_forces = _energy_forces_from_terms(
        proposed_positions,
        force_terms,
        cell=proposed_cell,
        pairs=proposed_pairs,
        virtual_sites=virtual_sites,
    )
    old_volume = float(np.asarray(cell.volume))
    new_volume = float(np.asarray(proposed_cell.volume))
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
    if neighbor_manager is not None:
        neighbor_manager.cell = proposed_cell
        neighbor_manager.rebuild(proposed_eval_positions)
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


def _barostat_neighbor_list(
    positions: mx.array,
    cell: Cell,
    neighbor_manager: NeighborListManager | None,
) -> NeighborList | None:
    if neighbor_manager is None:
        return None
    return build_neighbor_list(
        positions,
        cell,
        cutoff=neighbor_manager.cutoff,
        skin=neighbor_manager.skin,
        sort_pairs=neighbor_manager.sort_pairs,
        max_workers=neighbor_manager.max_workers,
        backend=neighbor_manager.backend,
        max_mlx_dense_atoms=neighbor_manager.max_mlx_dense_atoms,
        block_size=neighbor_manager.block_size,
    )


def _normalize_barostat_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "isotropic": "isotropic",
        "iso": "isotropic",
        "anisotropic": "anisotropic",
        "aniso": "anisotropic",
        "membrane": "membrane",
        "semi_isotropic": "membrane",
        "semiisotropic": "membrane",
    }
    if normalized not in aliases:
        msg = "barostat mode must be isotropic, anisotropic, or membrane"
        raise ValueError(msg)
    return aliases[normalized]


def _barostat_axis_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis in {0, 1, 2}:
            return axis
        msg = "barostat axis must be x, y, z, 0, 1, or 2"
        raise ValueError(msg)
    normalized = str(axis).strip().lower()
    if normalized in {"x", "0"}:
        return 0
    if normalized in {"y", "1"}:
        return 1
    if normalized in {"z", "2"}:
        return 2
    msg = "barostat axis must be x, y, z, 0, 1, or 2"
    raise ValueError(msg)


def _barostat_plane_axes(plane: str | tuple[str | int, str | int]) -> tuple[int, int]:
    if isinstance(plane, tuple):
        axes = tuple(_barostat_axis_index(axis) for axis in plane)
    else:
        normalized = str(plane).strip().lower().replace("-", "").replace("_", "")
        axes = tuple(_barostat_axis_index(axis) for axis in normalized)
    if len(axes) != 2 or len(set(axes)) != 2:
        msg = "membrane_plane must name two distinct axes"
        raise ValueError(msg)
    return tuple(sorted(axes))


def _validate_barostat_cell_support(cell: Cell, barostat: MonteCarloBarostat) -> None:
    volume = float(np.asarray(cell.volume))
    if not np.isfinite(volume) or volume <= 0.0:
        msg = "NPT barostat requires a positive finite cell volume"
        raise ValueError(msg)
    if barostat.mode == "anisotropic" and not any(barostat.axes):
        msg = "anisotropic barostat requires at least one enabled axis"
        raise ValueError(msg)


def _barostat_scale_factors(
    barostat: MonteCarloBarostat,
    rng: np.random.Generator,
) -> np.ndarray:
    max_scale = barostat.max_log_volume_scale
    if barostat.mode == "isotropic":
        log_volume_scale = rng.uniform(-max_scale, max_scale)
        return np.full(3, np.exp(log_volume_scale / 3.0), dtype=np.float64)
    if barostat.mode == "anisotropic":
        log_axis_scale = np.zeros(3, dtype=np.float64)
        enabled = np.asarray(barostat.axes, dtype=bool)
        log_axis_scale[enabled] = rng.uniform(-max_scale, max_scale, size=int(enabled.sum()))
        return np.exp(log_axis_scale)

    plane_axes = _barostat_plane_axes(barostat.membrane_plane)
    normal_axis = _barostat_axis_index(barostat.normal_axis)
    log_area_scale = rng.uniform(-max_scale, max_scale)
    log_normal_scale = rng.uniform(-max_scale, max_scale)
    log_axis_scale = np.zeros(3, dtype=np.float64)
    for axis in plane_axes:
        log_axis_scale[axis] = log_area_scale / 2.0
    log_axis_scale[normal_axis] = log_normal_scale
    return np.exp(log_axis_scale)


def _scaled_cell(cell: Cell, scale_factors: np.ndarray) -> Cell:
    matrix = np.asarray(cell.matrix, dtype=np.float64).copy()
    matrix *= np.asarray(scale_factors, dtype=np.float64)[:, None]
    return Cell(matrix)


def _barostat_metadata(barostat: MonteCarloBarostat) -> dict[str, Any]:
    metadata = {
        "family": "monte_carlo",
        "mode": barostat.mode,
        "pressure": barostat.pressure,
        "temperature": barostat.temperature,
        "interval": barostat.interval,
        "max_log_volume_scale": barostat.max_log_volume_scale,
    }
    if barostat.mode == "anisotropic":
        metadata["axes"] = {
            axis: enabled for axis, enabled in zip(("x", "y", "z"), barostat.axes, strict=True)
        }
    elif barostat.mode == "membrane":
        metadata["membrane_plane"] = barostat.membrane_plane
        metadata["normal_axis"] = barostat.normal_axis
        metadata["plane_policy"] = "coupled_area"
        metadata["normal_policy"] = "independent_length"
    return metadata


def _notify_barostat_reporters(
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None,
    *,
    final_state: SimulationState,
    final_cell: Cell,
    metadata: dict[str, Any],
) -> None:
    if reporters is None:
        return
    event_metadata = dict(metadata)
    event_metadata["final_cell"] = np.asarray(final_cell.matrix, dtype=np.float32).tolist()
    event = ReporterEvent(
        ensemble="NPT",
        event_type="barostat",
        step=final_state.step,
        time=final_state.time,
        state=final_state,
        barostat=event_metadata,
    )
    for reporter in _normalize_reporters(reporters):
        reporter(event)
