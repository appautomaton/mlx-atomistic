"""Interaction32 neighbor-generation construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_atomistic.cell_list import PairListStats
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.interaction_engine import (
    _DeviceFusedHalfSchedule32,
    _filter_device_fused_half_schedule32,
    _interaction32_profile_finish_build,
    _interaction32_profile_finish_stage,
    _interaction32_profile_start,
    _Interaction32ScheduleCapacity,
    _PreparedInteraction32Topology,
    _retry_device_fused_half_schedule32,
    _try_build_device_fused_half_schedule32,
)

if TYPE_CHECKING:
    from mlx_atomistic.neighbors import NeighborList

_FLOAT_BYTES = np.dtype(np.float32).itemsize
_INT_BYTES = np.dtype(np.int32).itemsize


@dataclass(frozen=True)
class _Interaction32NeighborBuildSpec:
    """Describe one device-resident Interaction32 neighbor generation."""

    positions: object
    cell: Cell
    cutoff: float
    outer_skin: float
    inner_skin: float | None
    generation: int
    exclusion_pairs: object
    one_four_pairs: object
    ordinary_tiles_per_group: int
    capacity: _Interaction32ScheduleCapacity | None
    topology: _PreparedInteraction32Topology | None
    two_level_admitted: bool | None


@dataclass(frozen=True)
class _Interaction32NeighborBuild:
    """Return schedules, statistics, and reusable state from one build."""

    positions: mx.array
    outer_schedule: _DeviceFusedHalfSchedule32
    outer_stats: PairListStats
    inner_schedule: _DeviceFusedHalfSchedule32 | None
    inner_stats: PairListStats | None
    capacity: _Interaction32ScheduleCapacity
    topology: _PreparedInteraction32Topology


@dataclass
class _Interaction32NeighborLifecycle:
    """Own reusable and adaptive state for Interaction32 generations."""

    capacity: _Interaction32ScheduleCapacity | None = None
    topology: _PreparedInteraction32Topology | None = None
    outer_neighbor_list: NeighborList | None = None
    inner_neighbor_list: NeighborList | None = None
    inner_threshold: float | None = None
    two_level_admitted: bool | None = None
    generation_updates: int = 0
    observed_updates: int = 0
    observed_generations: int = 0

    def record_updates(self, count: int) -> None:
        """Record integration updates covered by the current generation."""

        if count < 0:
            raise ValueError("interaction32 update count must be non-negative")
        if self.inner_neighbor_list is not None:
            self.generation_updates += count

    def finish_generation(
        self,
        *,
        admission_generations: int,
        minimum_generation_updates: int,
    ) -> None:
        """Disable short-lived two-level schedules after enough observations."""

        updates = self.generation_updates
        if self.inner_neighbor_list is not None and updates > 0:
            self.observed_updates += updates
            self.observed_generations += 1
            if self.observed_generations >= admission_generations:
                mean_updates = self.observed_updates / self.observed_generations
                if mean_updates < minimum_generation_updates:
                    self.two_level_admitted = False
        self.generation_updates = 0

    def select_schedule(self, max_displacement: float) -> NeighborList | None:
        """Select the inner schedule until its displacement margin expires."""

        if (
            self.inner_neighbor_list is None
            or self.outer_neighbor_list is None
            or self.inner_threshold is None
        ):
            return None
        if max_displacement <= self.inner_threshold:
            return self.inner_neighbor_list
        return self.outer_neighbor_list

    def active_rebuild_threshold(
        self,
        neighbor_list: NeighborList | None,
        default: float,
    ) -> float:
        """Return the threshold belonging to the active schedule."""

        if neighbor_list is self.inner_neighbor_list and self.inner_threshold is not None:
            return self.inner_threshold
        return default

    def install_generation(
        self,
        build: _Interaction32NeighborBuild,
        *,
        outer_neighbor_list: NeighborList,
        inner_neighbor_list: NeighborList | None,
        inner_skin: float | None,
    ) -> NeighborList:
        """Commit build outputs and return the active schedule."""

        self.capacity = build.capacity
        self.topology = build.topology
        self.outer_neighbor_list = outer_neighbor_list
        self.inner_neighbor_list = inner_neighbor_list
        self.inner_threshold = None if inner_skin is None else 0.5 * inner_skin
        if inner_neighbor_list is not None:
            self.two_level_admitted = True
            return inner_neighbor_list
        return outer_neighbor_list

    def fork_build_candidate(self) -> _Interaction32NeighborLifecycle:
        """Copy reusable policy state without sharing a current generation."""

        return _Interaction32NeighborLifecycle(
            capacity=self.capacity,
            topology=self.topology,
            two_level_admitted=self.two_level_admitted,
            observed_updates=self.observed_updates,
            observed_generations=self.observed_generations,
        )


def _build_interaction32_neighbor_generation(
    spec: _Interaction32NeighborBuildSpec,
) -> _Interaction32NeighborBuild:
    """Build one outer schedule and its optional shorter-skin schedule."""

    if not _uses_metal_device():
        msg = "mlx_interaction32 requires the Metal device"
        raise ValueError(msg)
    if not spec.cell.is_orthorhombic:
        msg = "mlx_interaction32 requires an orthorhombic periodic cell"
        raise ValueError(msg)

    positions = as_mx_array(spec.positions, dtype=mx.float32)
    box_lengths = np.asarray(np.diag(np.asarray(spec.cell.matrix)), dtype=np.float32)
    outer_search_radius = spec.cutoff + spec.outer_skin
    inner_search_radius = None if spec.inner_skin is None else spec.cutoff + spec.inner_skin
    build_started = _interaction32_profile_start()
    attempt = _try_build_device_fused_half_schedule32(
        positions,
        box_lengths,
        search_radius=outer_search_radius,
        capacity=spec.capacity,
        generation_value=spec.generation,
        lj_exclusion_pairs=spec.exclusion_pairs,
        lj_one_four_pairs=spec.one_four_pairs,
        ordinary_tiles_per_group=spec.ordinary_tiles_per_group,
        topology=spec.topology,
        ordering_search_radius=inner_search_radius,
    )
    topology = attempt.topology
    overflow_retry = attempt.overflow
    if attempt.overflow:
        attempt = _retry_device_fused_half_schedule32(attempt)
    outer_schedule = attempt.schedule
    if outer_schedule is None or outer_schedule.generation is None:
        msg = "Interaction32 schedule build did not commit a generation"
        raise RuntimeError(msg)

    completion_started = _interaction32_profile_start()
    arrays = (
        outer_schedule.atom_order,
        outer_schedule.ordinary_left_blocks,
        outer_schedule.ordinary_right_atoms,
        outer_schedule.ordinary_half_modes,
        outer_schedule.ordinary_group_starts,
        outer_schedule.ordinary_group_counts,
        outer_schedule.special_work_left_blocks,
        outer_schedule.special_work_left_slices,
        outer_schedule.special_work_right_atoms,
        outer_schedule.special_work_lj_enabled,
        outer_schedule.special_work_lj_one_four,
        outer_schedule.special_work_diagonal,
    )
    mx.eval(*arrays)
    capacity = outer_schedule.generation.capacity
    inventory = attempt.inventory
    inner_build = (
        None
        if inner_search_radius is None
        else _filter_device_fused_half_schedule32(
            positions,
            inventory,
            outer_schedule,
            search_radius=inner_search_radius,
        )
    )
    inner_schedule = None if inner_build is None else inner_build.schedule
    outer_pair_lanes = inventory.logical_pair_lanes + outer_schedule.special_work_count * 16 * 32
    cell_counts = inventory.geometry.cell_counts
    cell_count = int(np.prod(np.asarray(cell_counts), dtype=np.int64))
    resident_schedule_bytes = outer_schedule.estimated_bytes + (
        0 if inner_schedule is None else inner_schedule.estimated_bytes
    )
    estimated_cell_bytes = (
        positions.shape[0] * (3 * _FLOAT_BYTES + 2 * _INT_BYTES) + cell_count * 2 * _INT_BYTES
    )

    def make_stats(
        pair_lanes: int,
        radius: float,
        compaction_backend: str,
    ) -> PairListStats:
        return PairListStats(
            pair_count=pair_lanes,
            n_cells=cell_counts,
            cell_count=cell_count,
            occupied_cell_count=inventory.occupied_cell_count,
            search_radius=radius,
            estimated_pair_bytes=resident_schedule_bytes,
            estimated_cell_list_bytes=estimated_cell_bytes,
            backend="mlx_interaction32",
            representation_kind="interaction32",
            candidate_count=pair_lanes,
            estimated_candidate_bytes=resident_schedule_bytes,
            compaction_backend=compaction_backend,
            fallback_reason=None,
            adaptation_reason=(
                "interaction32_two_level_short_generation"
                if spec.two_level_admitted is False
                else None
            ),
        )

    outer_stats = make_stats(
        outer_pair_lanes,
        outer_search_radius,
        "metal_interaction32_device_builder",
    )
    inner_stats = None
    if inner_build is not None and inner_search_radius is not None:
        inner_pair_lanes = (
            inner_build.logical_pair_lanes + inner_build.schedule.special_work_count * 16 * 32
        )
        inner_stats = make_stats(
            inner_pair_lanes,
            inner_search_radius,
            "metal_interaction32_outer_inner_compactor",
        )

    _interaction32_profile_finish_stage("schedule_completion", completion_started)
    _interaction32_profile_finish_build(
        build_started,
        inventory={
            "atom_count": inventory.geometry.atom_count,
            "occupied_cell_count": inventory.occupied_cell_count,
            "ordinary_right_entry_count": inventory.right_entry_count,
            "ordinary_logical_pair_lanes": inventory.logical_pair_lanes,
            "ordinary_tile_count": inventory.ordinary_tile_count,
            "ordinary_group_count": inventory.ordinary_group_count,
            "special_tile_count": inventory.special_tile_count,
            "mode_cache_bytes": inventory.mode_cache_bytes,
            "ordinary_tile_capacity": capacity.ordinary_tiles,
            "ordinary_group_capacity": capacity.ordinary_groups,
            "special_tile_capacity": capacity.special_tiles,
            "overflow_retry_count": int(overflow_retry),
        },
    )
    return _Interaction32NeighborBuild(
        positions=positions,
        outer_schedule=outer_schedule,
        outer_stats=outer_stats,
        inner_schedule=inner_schedule,
        inner_stats=inner_stats,
        capacity=capacity,
        topology=topology,
    )


def _uses_metal_device() -> bool:
    return mx.metal.is_available() and "gpu" in str(mx.default_device()).lower()
