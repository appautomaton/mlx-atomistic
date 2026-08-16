"""Compiled Langevin block execution for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from mlx_atomistic._md_state import SimulationConfig
from mlx_atomistic.core import Cell
from mlx_atomistic.force_runtime import _PreparedForcePipeline
from mlx_atomistic.neighbors import NeighborList, NeighborListManager


class _PairForceEvaluator(Protocol):
    """Evaluate forces for one positions and pair-representation binding."""

    def __call__(
        self,
        positions: mx.array,
        *,
        pairs: object | None,
    ) -> mx.array: ...


@dataclass(frozen=True)
class _LangevinBlockState:
    """Carry the mutable simulation state across compiled blocks."""

    positions: mx.array
    velocities: mx.array
    forces: mx.array
    key: mx.array
    pairs: object
    pair_count: int
    rebuild_count: int
    neighbor_list: NeighborList


class _LangevinBlockExecutor:
    """Advance compiled Langevin blocks against one neighbor-list manager."""

    def __init__(
        self,
        *,
        cell: Cell,
        wrap_positions: bool,
        dt: float,
        force_to_acceleration_scale: float,
        masses: mx.array,
        velocity_decay: float,
        noise_scale: float,
        neighbor_manager: NeighborListManager,
        force_evaluator: _PairForceEvaluator,
        prepared_force_pipeline: _PreparedForcePipeline | None,
    ) -> None:
        self.cell = cell
        self.wrap_positions = wrap_positions
        self.dt = dt
        self.force_to_acceleration_scale = force_to_acceleration_scale
        self.masses_col = masses[:, None]
        self.sqrt_masses_col = mx.sqrt(masses)[:, None]
        self.velocity_decay = velocity_decay
        self.noise_scale = noise_scale
        self.neighbor_manager = neighbor_manager
        self.force_evaluator = force_evaluator
        self.prepared_force_pipeline = prepared_force_pipeline
        self._compiled_blocks: dict[int, object] = {}

    def advance(
        self,
        state: _LangevinBlockState,
        n_substeps: int,
    ) -> _LangevinBlockState:
        """Advance one proposed block, replaying it if its Verlet list expires."""

        if n_substeps < 1:
            msg = "n_substeps must be positive"
            raise ValueError(msg)
        reference_positions = self.neighbor_manager.reference_positions
        if reference_positions is None:
            msg = "compiled block execution requires neighbor reference positions"
            raise RuntimeError(msg)
        proposed = self._compiled_block(n_substeps)(
            state.positions,
            state.velocities,
            state.forces,
            state.key,
            state.pairs,
            reference_positions,
        )
        if self.neighbor_manager._admit_block(proposed[-2], proposed[-1], n_substeps):
            positions, velocities, forces, key = proposed[:4]
        else:
            positions, velocities, forces, key = self._replay(
                state.positions,
                state.velocities,
                state.forces,
                state.key,
                n_substeps,
            )

        neighbor_list = self.neighbor_manager.neighbor_list
        if neighbor_list is None:
            msg = "Langevin block execution lost its current neighbor list"
            raise RuntimeError(msg)
        return _LangevinBlockState(
            positions=positions,
            velocities=velocities,
            forces=forces,
            key=key,
            pairs=neighbor_list.force_candidates(prefer_tiles=False),
            pair_count=neighbor_list.pair_count,
            rebuild_count=self.neighbor_manager.rebuild_count,
            neighbor_list=neighbor_list,
        )

    def _substep(self, positions, velocities, forces, key, pairs):
        acceleration = self.force_to_acceleration_scale * forces / self.masses_col
        half_velocity = velocities + 0.5 * self.dt * acceleration
        positions = positions + 0.5 * self.dt * half_velocity
        if self.wrap_positions:
            positions = self.cell.wrap(positions)
        split_keys = mx.random.split(key, 2)
        key = split_keys[0]
        noise = mx.random.normal(velocities.shape, key=split_keys[1])
        middle_velocity = (
            self.velocity_decay * half_velocity + (self.noise_scale / self.sqrt_masses_col) * noise
        )
        positions = positions + 0.5 * self.dt * middle_velocity
        if self.wrap_positions:
            positions = self.cell.wrap(positions)
        next_forces = self.force_evaluator(positions, pairs=pairs)
        next_acceleration = self.force_to_acceleration_scale * next_forces / self.masses_col
        velocities = middle_velocity + 0.5 * self.dt * next_acceleration
        return positions, velocities, next_forces, key

    def _compiled_block(self, n_substeps: int):
        cached = self._compiled_blocks.get(n_substeps)
        if cached is not None:
            return cached

        def block(positions, velocities, forces, key, pairs, reference_positions):
            block_max_displacement = mx.array(0.0, dtype=positions.dtype)
            block_admissible = mx.array(True)
            for _ in range(n_substeps):
                positions, velocities, forces, key = self._substep(
                    positions,
                    velocities,
                    forces,
                    key,
                    pairs,
                )
                displacement = self.cell.minimum_image(positions - reference_positions)
                distance2 = mx.sum(displacement * displacement, axis=1)
                step_max_displacement = (
                    mx.array(0.0, dtype=positions.dtype)
                    if positions.shape[0] == 0
                    else mx.sqrt(mx.max(distance2))
                )
                block_max_displacement = mx.maximum(
                    block_max_displacement,
                    step_max_displacement,
                )
                block_admissible = (
                    block_admissible
                    & mx.all(mx.isfinite(positions))
                    & (step_max_displacement <= self.neighbor_manager.rebuild_threshold)
                )
            return (
                positions,
                velocities,
                forces,
                key,
                block_max_displacement,
                block_admissible,
            )

        compiled = mx.compile(block)
        self._compiled_blocks[n_substeps] = compiled
        return compiled

    def _replay(self, positions, velocities, forces, key, n_substeps: int):
        for _ in range(n_substeps):
            acceleration = self.force_to_acceleration_scale * forces / self.masses_col
            half_velocity = velocities + 0.5 * self.dt * acceleration
            positions = positions + 0.5 * self.dt * half_velocity
            if self.wrap_positions:
                positions = self.cell.wrap(positions)
            split_keys = mx.random.split(key, 2)
            key = split_keys[0]
            noise = mx.random.normal(velocities.shape, key=split_keys[1])
            middle_velocity = (
                self.velocity_decay * half_velocity
                + (self.noise_scale / self.sqrt_masses_col) * noise
            )
            positions = positions + 0.5 * self.dt * middle_velocity
            if self.wrap_positions:
                positions = self.cell.wrap(positions)

            neighbor_list = self.neighbor_manager.rebuild(positions)
            pairs = neighbor_list.force_candidates(prefer_tiles=False)
            if self.prepared_force_pipeline is None:
                forces = self.force_evaluator(positions, pairs=pairs)
            else:
                binding = self.prepared_force_pipeline.bind(neighbor_list)
                forces = binding.forces(
                    positions,
                    evaluation_positions=positions,
                )
            next_acceleration = self.force_to_acceleration_scale * forces / self.masses_col
            velocities = middle_velocity + 0.5 * self.dt * next_acceleration
        return positions, velocities, forces, key


def _next_recording_local_step(config: SimulationConfig, local_step: int) -> int:
    """Return the next sampling, diagnostic, or final local step."""

    current_step = config.initial_step + local_step
    next_sample = ((current_step // config.sample_interval) + 1) * config.sample_interval
    next_diagnostic = (
        (current_step // config.diagnostic_interval) + 1
    ) * config.diagnostic_interval
    next_step = min(next_sample, next_diagnostic) - config.initial_step
    return min(next_step, config.steps)
