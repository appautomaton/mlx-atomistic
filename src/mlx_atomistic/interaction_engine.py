"""Experimental 32-atom interaction schedules for the Metal force prototype."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil, isfinite
from time import perf_counter

import mlx.core as mx
import numpy as np
from scipy.spatial import cKDTree

from mlx_atomistic.core import as_mx_array
from mlx_atomistic.metal_kernels import (
    _interaction32_block_geometry,
    _interaction32_fused_half_pme_direct_force_only,
    _interaction32_ordinary_mode_counts,
    _interaction32_ordinary_scatter_sized,
    _interaction32_pme_direct_force_only,
    _interaction32_special_blocks_sized,
    _interaction32_special_work_two_halves,
    _Interaction32ForceStages,
    _neighbor_tile_force_groups_sized,
    _owner_compute32_pme_direct_force_only,
)

_INTERACTION_TILE_SIZE = 32
_DEFAULT_ORDINARY_TILES_PER_GROUP = 3
_INTERACTION32_CAPACITY_RESERVE = 1.25
_INTERACTION32_CAPACITY_QUANTUM = 64
_INTERACTION32_MODE_CACHE_LIMIT_BYTES = 64 * 1024 * 1024
_INTERACTION32_REBUILD_PROFILE_STAGES = (
    "geometry_validation_and_sort",
    "topology_preparation",
    "special_block_inventory",
    "ordinary_count_and_prefix_readback",
    "capacity_admission",
    "ordinary_scatter",
    "special_scatter",
    "schedule_completion",
)


def _interaction32_timing_summary(
    samples: list[float],
    *,
    total_build_seconds: float,
) -> dict[str, object]:
    """Summarize one synchronized Interaction32 builder-stage sample vector."""

    total = float(sum(samples))
    return {
        "count": len(samples),
        "total_seconds": total,
        "median_seconds": None if not samples else float(np.median(samples)),
        "minimum_seconds": None if not samples else float(min(samples)),
        "maximum_seconds": None if not samples else float(max(samples)),
        "fraction_of_profiled_rebuild_wall": (
            0.0 if total_build_seconds <= 0.0 else total / total_build_seconds
        ),
        "samples_seconds": list(samples),
    }


def _interaction32_inventory_summary(
    samples: list[dict[str, int]],
) -> dict[str, dict[str, int | float] | int]:
    """Summarize integer Interaction32 inventories across profiled rebuilds."""

    if not samples:
        return {"count": 0}
    names = tuple(samples[0])
    summary: dict[str, dict[str, int | float] | int] = {"count": len(samples)}
    for name in names:
        values = [int(sample[name]) for sample in samples]
        summary[name] = {
            "minimum": min(values),
            "median": float(np.median(values)),
            "maximum": max(values),
        }
    return summary


@dataclass
class _Interaction32RebuildProfiler:
    """Collect synchronized stage timings for the Interaction32 builder."""

    stage_samples: dict[str, list[float]] = field(
        default_factory=lambda: {
            name: [] for name in _INTERACTION32_REBUILD_PROFILE_STAGES
        }
    )
    build_samples: list[float] = field(default_factory=list)
    inventory_samples: list[dict[str, int]] = field(default_factory=list)

    def record_stage(self, name: str, elapsed_seconds: float) -> None:
        """Record one synchronized builder-stage sample."""

        if name not in self.stage_samples:
            msg = f"unknown Interaction32 rebuild stage {name!r}"
            raise ValueError(msg)
        self.stage_samples[name].append(float(elapsed_seconds))

    def record_build(
        self,
        elapsed_seconds: float,
        *,
        inventory: dict[str, int],
    ) -> None:
        """Record one complete profiled build and its interaction inventory."""

        self.build_samples.append(float(elapsed_seconds))
        self.inventory_samples.append(dict(inventory))

    def report(self) -> dict[str, object]:
        """Return a JSON-serializable synchronized stage report."""

        build_total = float(sum(self.build_samples))
        stages = {
            name: _interaction32_timing_summary(
                self.stage_samples[name],
                total_build_seconds=build_total,
            )
            for name in _INTERACTION32_REBUILD_PROFILE_STAGES
        }
        accounted = float(sum(float(stage["total_seconds"]) for stage in stages.values()))
        unattributed = build_total - accounted
        tolerance = max(1.0e-6, build_total * 0.05)
        stage_counts_match = all(
            len(samples) == len(self.build_samples)
            for samples in self.stage_samples.values()
        )
        return {
            "schema": "mlx_atomistic.interaction32_rebuild_profile.v1",
            "mode": "sequential_synchronized_builder_stages",
            "backend": "mlx_interaction32",
            "stage_order": list(_INTERACTION32_REBUILD_PROFILE_STAGES),
            "rebuild_count": len(self.build_samples),
            "profiled_rebuild_wall_seconds": build_total,
            "accounted_stage_seconds": accounted,
            "unattributed_seconds": unattributed,
            "reconciled": stage_counts_match and -tolerance <= unattributed <= tolerance,
            "build_samples_seconds": list(self.build_samples),
            "stages": stages,
            "inventories": _interaction32_inventory_summary(self.inventory_samples),
        }


_ACTIVE_INTERACTION32_REBUILD_PROFILER: ContextVar[
    _Interaction32RebuildProfiler | None
] = ContextVar(
    "mlx_atomistic_active_interaction32_rebuild_profiler",
    default=None,
)


@contextmanager
def _profile_interaction32_rebuilds() -> Iterator[_Interaction32RebuildProfiler]:
    """Collect synchronized Interaction32 stage timings inside the context."""

    if _ACTIVE_INTERACTION32_REBUILD_PROFILER.get() is not None:
        msg = "Interaction32 rebuild profiling contexts cannot be nested"
        raise RuntimeError(msg)
    profiler = _Interaction32RebuildProfiler()
    token = _ACTIVE_INTERACTION32_REBUILD_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_INTERACTION32_REBUILD_PROFILER.reset(token)


def _interaction32_profile_start() -> float | None:
    """Start one opt-in wall-clock interval without affecting the fast path."""

    if _ACTIVE_INTERACTION32_REBUILD_PROFILER.get() is None:
        return None
    return perf_counter()


def _interaction32_profile_finish_stage(
    name: str,
    started: float | None,
    *values: object,
) -> None:
    """Synchronize and record one opt-in builder stage."""

    if started is None:
        return
    if values:
        mx.eval(*values)
    profiler = _ACTIVE_INTERACTION32_REBUILD_PROFILER.get()
    if profiler is None:
        raise RuntimeError("Interaction32 rebuild profiling context ended during a stage")
    profiler.record_stage(name, perf_counter() - started)


def _interaction32_profile_finish_build(
    started: float | None,
    *,
    inventory: dict[str, int],
) -> None:
    """Record one complete Interaction32 rebuild when profiling is active."""

    if started is None:
        return
    profiler = _ACTIVE_INTERACTION32_REBUILD_PROFILER.get()
    if profiler is None:
        raise RuntimeError("Interaction32 rebuild profiling context ended during a build")
    profiler.record_build(perf_counter() - started, inventory=inventory)


def _normalize_pairs(pairs: object, atom_count: int, name: str) -> np.ndarray:
    array = np.asarray(pairs, dtype=np.int32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_pairs, 2)")
    if np.any(array < 0) or np.any(array >= atom_count):
        raise ValueError(f"{name} contains atom indices outside [0, n_atoms)")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError(f"{name} cannot contain self pairs")
    normalized = np.sort(array, axis=1)
    codes = normalized[:, 0].astype(np.int64) * np.int64(atom_count) + normalized[:, 1].astype(
        np.int64
    )
    unique_codes = np.unique(codes)
    return np.stack((unique_codes // atom_count, unique_codes % atom_count), axis=1).astype(
        np.int32
    )


def _pair_codes(pairs: np.ndarray, atom_count: int) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    return pairs[:, 0].astype(np.int64) * np.int64(atom_count) + pairs[:, 1].astype(np.int64)


def _contains_sorted(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sorted_values, values)
    in_bounds = indices < sorted_values.size
    result = np.zeros(values.shape, dtype=bool)
    result[in_bounds] = sorted_values[indices[in_bounds]] == values[in_bounds]
    return result


def _group_ordinary_tiles(
    left_blocks: np.ndarray,
    left_slices: np.ndarray,
    max_tiles_per_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts: list[int] = []
    counts: list[int] = []
    run_start = 0
    tile_count = int(left_blocks.shape[0])
    while run_start < tile_count:
        run_stop = run_start + 1
        while (
            run_stop < tile_count
            and left_blocks[run_stop] == left_blocks[run_start]
            and left_slices[run_stop] == left_slices[run_start]
        ):
            run_stop += 1
        for start in range(run_start, run_stop, max_tiles_per_group):
            starts.append(start)
            counts.append(min(max_tiles_per_group, run_stop - start))
        run_start = run_stop
    return np.asarray(starts, dtype=np.int32), np.asarray(counts, dtype=np.int32)


def _cell_atom_order(
    positions: np.ndarray,
    box: np.ndarray,
    search_radius: float,
) -> np.ndarray:
    cell_width = search_radius / 3.0
    cell_counts = np.maximum(np.floor(box / cell_width).astype(np.int64), 1)
    wrapped = positions - box * np.floor(positions / box)
    cells = np.floor(wrapped * cell_counts / box).astype(np.int64)
    cells = np.minimum(cells, cell_counts - 1)
    keys = cells[:, 0].astype(np.int64) + cell_counts[0] * (
        cells[:, 1].astype(np.int64) + cell_counts[1] * cells[:, 2].astype(np.int64)
    )
    return np.argsort(keys, kind="stable").astype(np.int32)


def _make_atom_blocks(
    positions: np.ndarray,
    box: np.ndarray,
    atom_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atom_count = positions.shape[0]
    padded_count = (
        (atom_count + _INTERACTION_TILE_SIZE - 1) // _INTERACTION_TILE_SIZE
    ) * _INTERACTION_TILE_SIZE
    padded_order = np.full((padded_count,), -1, dtype=np.int32)
    padded_order[:atom_count] = atom_order
    block_atoms = padded_order.reshape((-1, _INTERACTION_TILE_SIZE))
    valid = block_atoms >= 0
    safe_atoms = np.maximum(block_atoms, 0)
    block_positions = positions[safe_atoms]
    reference = block_positions[:, :1, :]
    unwrapped = reference + (
        block_positions - reference - box * np.rint((block_positions - reference) / box)
    )
    valid_count = np.sum(valid, axis=1, keepdims=True)
    centers = np.sum(unwrapped * valid[..., None], axis=1) / valid_count
    centered = np.where(valid[..., None], unwrapped - centers[:, None, :], 0.0)
    extents = np.max(np.abs(centered), axis=1)
    radii = np.sqrt(np.max(np.sum(centered * centered, axis=2), axis=1))
    centers -= box * np.floor(centers / box)

    inverse_order = np.empty((atom_count,), dtype=np.int32)
    inverse_order[atom_order] = np.arange(atom_count, dtype=np.int32)
    return block_atoms, valid, centers, extents, radii, inverse_order


def _special_block_codes(
    block_count: int,
    inverse_order: np.ndarray,
    *pair_groups: np.ndarray,
) -> np.ndarray:
    diagonal = np.arange(block_count, dtype=np.int64)
    codes = [diagonal * np.int64(block_count) + diagonal]
    for pairs in pair_groups:
        if pairs.shape[0] == 0:
            continue
        ordered = inverse_order[pairs]
        blocks = ordered // _INTERACTION_TILE_SIZE
        left = np.minimum(blocks[:, 0], blocks[:, 1]).astype(np.int64)
        right = np.maximum(blocks[:, 0], blocks[:, 1]).astype(np.int64)
        codes.append(left * np.int64(block_count) + right)
    return np.unique(np.concatenate(codes))


def _special_lj_masks(
    block_atoms: np.ndarray,
    special_blocks: np.ndarray,
    exclusion_codes: np.ndarray,
    one_four_codes: np.ndarray,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tile_count = special_blocks.shape[0]
    enabled_words = np.zeros((tile_count, _INTERACTION_TILE_SIZE), dtype=np.uint32)
    one_four_words = np.zeros_like(enabled_words)
    weights = np.left_shift(np.uint32(1), np.arange(_INTERACTION_TILE_SIZE, dtype=np.uint32))
    batch_size = 1024
    for start in range(0, tile_count, batch_size):
        stop = min(start + batch_size, tile_count)
        block_pairs = special_blocks[start:stop]
        left = block_atoms[block_pairs[:, 0]][:, :, None]
        right = block_atoms[block_pairs[:, 1]][:, None, :]
        safe_left = np.maximum(left, 0).astype(np.int64, copy=False)
        safe_right = np.maximum(right, 0).astype(np.int64, copy=False)
        low = np.minimum(safe_left, safe_right)
        high = np.maximum(safe_left, safe_right)
        codes = low * np.int64(atom_count) + high
        valid = (left >= 0) & (right >= 0) & (left != right)
        enabled = valid & ~_contains_sorted(exclusion_codes, codes)
        one_four = enabled & _contains_sorted(one_four_codes, codes)
        enabled_words[start:stop] = np.sum(
            enabled.astype(np.uint32) * weights[None, :, None],
            axis=1,
            dtype=np.uint32,
        )
        one_four_words[start:stop] = np.sum(
            one_four.astype(np.uint32) * weights[None, :, None],
            axis=1,
            dtype=np.uint32,
        )
    return enabled_words, one_four_words


def _build_special_work(
    positions: np.ndarray,
    box: np.ndarray,
    block_atoms: np.ndarray,
    valid: np.ndarray,
    special_blocks: np.ndarray,
    lj_enabled: np.ndarray,
    lj_one_four: np.ndarray,
    search_radius2: float,
    left_slice_size: int,
) -> tuple[np.ndarray, ...]:
    sentinel = block_atoms.size
    left_blocks: list[int] = []
    left_slices: list[int] = []
    right_rows: list[np.ndarray] = []
    enabled_rows: list[np.ndarray] = []
    one_four_rows: list[np.ndarray] = []
    diagonal_flags: list[int] = []
    for tile, (left_block, right_block) in enumerate(special_blocks):
        left_atoms = block_atoms[left_block]
        right_atoms = block_atoms[right_block]
        left_positions = positions[np.maximum(left_atoms, 0)]
        right_positions = positions[np.maximum(right_atoms, 0)]
        delta = left_positions[:, None, :] - right_positions[None, :, :]
        delta -= box * np.rint(delta / box)
        distance2 = np.sum(delta * delta, axis=2)
        member = valid[left_block, :, None] & valid[right_block, None, :]
        member &= distance2 < search_radius2
        diagonal = left_block == right_block
        if diagonal:
            left_slots = np.arange(_INTERACTION_TILE_SIZE)[:, None]
            right_slots = np.arange(_INTERACTION_TILE_SIZE)[None, :]
            member &= left_slots < right_slots
        for left_slice in range(_INTERACTION_TILE_SIZE // left_slice_size):
            atom_slice = slice(
                left_slice_size * left_slice,
                left_slice_size * (left_slice + 1),
            )
            right_mask = np.any(member[atom_slice], axis=0)
            if not np.any(right_mask):
                continue
            slots = np.nonzero(right_mask)[0].astype(np.int32)
            right_row = np.full((_INTERACTION_TILE_SIZE,), sentinel, dtype=np.int32)
            right_row[: slots.size] = np.int32(right_block * _INTERACTION_TILE_SIZE) + slots
            enabled_row = np.zeros((_INTERACTION_TILE_SIZE,), dtype=np.uint32)
            enabled_row[: slots.size] = lj_enabled[tile, slots]
            one_four_row = np.zeros((_INTERACTION_TILE_SIZE,), dtype=np.uint32)
            one_four_row[: slots.size] = lj_one_four[tile, slots]
            left_blocks.append(int(left_block))
            left_slices.append(left_slice)
            right_rows.append(right_row)
            enabled_rows.append(enabled_row)
            one_four_rows.append(one_four_row)
            diagonal_flags.append(int(diagonal))

    work_count = len(left_blocks)
    empty_rows = (0, _INTERACTION_TILE_SIZE)
    return (
        np.asarray(left_blocks, dtype=np.int32),
        np.asarray(left_slices, dtype=np.int32),
        np.stack(right_rows) if right_rows else np.empty(empty_rows, dtype=np.int32),
        np.stack(enabled_rows) if enabled_rows else np.empty(empty_rows, dtype=np.uint32),
        np.stack(one_four_rows) if one_four_rows else np.empty(empty_rows, dtype=np.uint32),
        np.asarray(diagonal_flags, dtype=np.int32).reshape(work_count),
    )


@dataclass(frozen=True)
class _InteractionSchedule32:
    atom_count: int
    search_radius: float
    left_slice_size: int
    box_lengths: np.ndarray
    atom_order: np.ndarray
    inverse_order: np.ndarray
    ordinary_left_blocks: np.ndarray
    ordinary_left_slices: np.ndarray
    ordinary_right_atoms: np.ndarray
    ordinary_group_starts: np.ndarray
    ordinary_group_counts: np.ndarray
    special_blocks: np.ndarray
    special_lj_enabled: np.ndarray
    special_lj_one_four: np.ndarray
    special_work_left_blocks: np.ndarray
    special_work_left_slices: np.ndarray
    special_work_right_atoms: np.ndarray
    special_work_lj_enabled: np.ndarray
    special_work_lj_one_four: np.ndarray
    special_work_diagonal: np.ndarray

    @property
    def padded_atom_count(self) -> int:
        return int(self.atom_order.shape[0])

    @property
    def block_count(self) -> int:
        return self.padded_atom_count // _INTERACTION_TILE_SIZE

    @property
    def ordinary_tile_count(self) -> int:
        return int(self.ordinary_left_blocks.shape[0])

    @property
    def ordinary_group_count(self) -> int:
        return int(self.ordinary_group_starts.shape[0])

    @property
    def special_tile_count(self) -> int:
        return int(self.special_blocks.shape[0])

    @property
    def special_work_count(self) -> int:
        return int(self.special_work_left_blocks.shape[0])


@dataclass(frozen=True)
class _DeviceInteractionSchedule32:
    atom_count: int
    search_radius: float
    left_slice_size: int
    padded_atom_count: int
    ordinary_tile_count: int
    ordinary_group_count: int
    special_tile_count: int
    atom_order: mx.array
    ordinary_left_blocks: mx.array
    ordinary_left_slices: mx.array
    ordinary_right_atoms: mx.array
    ordinary_group_starts: mx.array
    ordinary_group_counts: mx.array
    special_blocks: mx.array
    special_lj_enabled: mx.array
    special_lj_one_four: mx.array
    special_work_left_blocks: mx.array
    special_work_left_slices: mx.array
    special_work_right_atoms: mx.array
    special_work_lj_enabled: mx.array
    special_work_lj_one_four: mx.array
    special_work_diagonal: mx.array


@dataclass(frozen=True)
class _DeviceBlockGeometry32:
    atom_count: int
    search_radius: float
    box_lengths: np.ndarray
    cell_counts: tuple[int, int, int]
    occupied_cell_count: mx.array
    atom_order: mx.array
    inverse_order: mx.array
    center_radius: mx.array
    half_extent: mx.array
    block_traversal: mx.array

    @property
    def padded_atom_count(self) -> int:
        return int(self.atom_order.shape[0])

    @property
    def block_count(self) -> int:
        return self.padded_atom_count // _INTERACTION_TILE_SIZE


@dataclass(frozen=True)
class _PreparedInteraction32Topology:
    """Immutable topology payload shared by every spatial generation."""

    atom_count: int
    exclusion_source_id: int
    one_four_source_id: int
    device: str
    digest: str
    exclusion_pairs: mx.array
    one_four_pairs: mx.array
    topology_offsets: mx.array
    topology_neighbors: mx.array
    topology_classes: mx.array


@dataclass(frozen=True)
class _DeviceSpecialBlockInventory32:
    atom_count: int
    block_count: int
    exclusion_pairs: mx.array
    one_four_pairs: mx.array
    block_codes: mx.array
    block_code_unique: mx.array
    special_count: mx.array
    topology_offsets: mx.array
    topology_neighbors: mx.array
    topology_classes: mx.array


@dataclass(frozen=True)
class _DeviceOrdinarySchedule32:
    atom_count: int
    search_radius: float
    padded_atom_count: int
    right_entry_count: int
    logical_pair_lanes: int
    ordinary_tile_count: int
    ordinary_group_count: int
    ordinary_tile_capacity: int
    ordinary_group_capacity: int
    special_tile_count_inventory: int
    mode_entry_counts: mx.array
    ordinary_left_blocks: mx.array
    ordinary_right_atoms: mx.array
    ordinary_half_modes: mx.array
    ordinary_group_starts: mx.array
    ordinary_group_counts: mx.array


@dataclass(frozen=True)
class _DeviceSpecialSchedule32:
    atom_count: int
    padded_atom_count: int
    special_tile_count: int
    special_work_count: int
    special_tile_capacity: int
    special_work_capacity: int
    special_blocks: mx.array
    special_work_left_blocks: mx.array
    special_work_left_slices: mx.array
    special_work_right_atoms: mx.array
    special_work_lj_enabled: mx.array
    special_work_lj_one_four: mx.array
    special_work_diagonal: mx.array


@dataclass(frozen=True)
class _DeviceScheduleInventory32:
    positions: mx.array
    geometry: _DeviceBlockGeometry32
    special: _DeviceSpecialBlockInventory32
    ordinary_tiles_per_group: int
    occupied_cell_count: int
    right_entry_count: int
    logical_pair_lanes: int
    ordinary_tile_count: int
    ordinary_group_count: int
    special_tile_count: int
    mode_cache_bytes: int
    mode_entry_counts: mx.array
    mode_words: mx.array | None
    mode_tile_counts: mx.array
    mode_tile_prefix: mx.array
    mode_group_counts: mx.array
    mode_group_prefix: mx.array


@dataclass(frozen=True)
class _Interaction32ScheduleCapacity:
    ordinary_tiles: int
    ordinary_groups: int
    special_tiles: int

    @property
    def special_work(self) -> int:
        return 2 * self.special_tiles


@dataclass(frozen=True)
class _Interaction32Generation:
    value: int
    atom_count: int
    box_lengths: tuple[float, float, float]
    search_radius: float
    topology_digest: str
    capacity: _Interaction32ScheduleCapacity


@dataclass(frozen=True)
class _DeviceScheduleBuildAttempt32:
    inventory: _DeviceScheduleInventory32
    topology: _PreparedInteraction32Topology
    requested_capacity: _Interaction32ScheduleCapacity | None
    recommended_capacity: _Interaction32ScheduleCapacity
    overflow_fields: tuple[str, ...]
    generation_value: int
    topology_digest: str
    schedule: _DeviceFusedHalfSchedule32 | None

    @property
    def overflow(self) -> bool:
        return bool(self.overflow_fields)


@dataclass(frozen=True)
class _FusedHalfSchedule32:
    atom_count: int
    search_radius: float
    atom_order: np.ndarray
    ordinary_left_blocks: np.ndarray
    ordinary_right_atoms: np.ndarray
    ordinary_half_modes: np.ndarray
    ordinary_group_starts: np.ndarray
    ordinary_group_counts: np.ndarray
    special_work_left_blocks: np.ndarray
    special_work_left_slices: np.ndarray
    special_work_right_atoms: np.ndarray
    special_work_lj_enabled: np.ndarray
    special_work_lj_one_four: np.ndarray
    special_work_diagonal: np.ndarray

    @property
    def padded_atom_count(self) -> int:
        return int(self.atom_order.shape[0])

    @property
    def block_count(self) -> int:
        return self.padded_atom_count // _INTERACTION_TILE_SIZE

    @property
    def ordinary_tile_count(self) -> int:
        return int(self.ordinary_left_blocks.shape[0])

    @property
    def ordinary_group_count(self) -> int:
        return int(self.ordinary_group_starts.shape[0])

    @property
    def ordinary_right_entry_count(self) -> int:
        return int(np.count_nonzero(self.ordinary_right_atoms < self.padded_atom_count))

    @property
    def ordinary_logical_pair_lanes(self) -> int:
        valid_per_tile = np.count_nonzero(
            self.ordinary_right_atoms < self.padded_atom_count,
            axis=1,
        )
        left_widths = np.where(self.ordinary_half_modes == 3, 32, 16)
        return int(np.sum(valid_per_tile * left_widths))

    @property
    def special_work_count(self) -> int:
        return int(self.special_work_left_blocks.shape[0])


@dataclass(frozen=True)
class _DeviceFusedHalfSchedule32:
    atom_count: int
    search_radius: float
    padded_atom_count: int
    ordinary_tile_count: int
    ordinary_group_count: int
    atom_order: mx.array
    ordinary_left_blocks: mx.array
    ordinary_right_atoms: mx.array
    ordinary_half_modes: mx.array
    ordinary_group_starts: mx.array
    ordinary_group_counts: mx.array
    special_work_left_blocks: mx.array
    special_work_left_slices: mx.array
    special_work_right_atoms: mx.array
    special_work_lj_enabled: mx.array
    special_work_lj_one_four: mx.array
    special_work_diagonal: mx.array
    generation: _Interaction32Generation | None = None

    @property
    def special_work_count(self) -> int:
        """Return the number of scheduled special half-block work items."""

        return int(self.special_work_left_blocks.shape[0])

    @property
    def estimated_bytes(self) -> int:
        """Estimate resident schedule storage from its packed device arrays."""

        if self.generation is not None:
            capacity = self.generation.capacity
            ordinary_tile_bytes = 4 + 32 * 4 + 4
            ordinary_group_bytes = 2 * 4
            special_work_bytes = 4 + 4 + 32 * 4 + 32 * 4 + 32 * 4 + 4
            return (
                self.padded_atom_count * 4
                + capacity.ordinary_tiles * ordinary_tile_bytes
                + capacity.ordinary_groups * ordinary_group_bytes
                + capacity.special_work * special_work_bytes
            )
        arrays = (
            self.atom_order,
            self.ordinary_left_blocks,
            self.ordinary_right_atoms,
            self.ordinary_half_modes,
            self.ordinary_group_starts,
            self.ordinary_group_counts,
            self.special_work_left_blocks,
            self.special_work_left_slices,
            self.special_work_right_atoms,
            self.special_work_lj_enabled,
            self.special_work_lj_one_four,
            self.special_work_diagonal,
        )
        return sum(int(np.prod(array.shape, dtype=np.int64)) * 4 for array in arrays)


@dataclass(frozen=True)
class _OwnerComputeSchedule32:
    atom_count: int
    search_radius: float
    atom_order: np.ndarray
    owner_offsets: np.ndarray
    right_atoms: np.ndarray
    topology_offsets: np.ndarray
    topology_neighbors: np.ndarray
    topology_classes: np.ndarray

    @property
    def padded_atom_count(self) -> int:
        return int(self.atom_order.shape[0])

    @property
    def block_count(self) -> int:
        return self.padded_atom_count // _INTERACTION_TILE_SIZE

    @property
    def scheduled_pair_lanes(self) -> int:
        return int(self.right_atoms.shape[0]) * _INTERACTION_TILE_SIZE


@dataclass(frozen=True)
class _DeviceOwnerComputeSchedule32:
    atom_count: int
    search_radius: float
    padded_atom_count: int
    block_count: int
    atom_order: mx.array
    owner_offsets: mx.array
    right_atoms: mx.array
    topology_offsets: mx.array
    topology_neighbors: mx.array
    topology_classes: mx.array


def _schedule_to_device32(
    schedule: _InteractionSchedule32,
) -> _DeviceInteractionSchedule32:
    return _DeviceInteractionSchedule32(
        atom_count=schedule.atom_count,
        search_radius=schedule.search_radius,
        left_slice_size=schedule.left_slice_size,
        padded_atom_count=schedule.padded_atom_count,
        ordinary_tile_count=schedule.ordinary_tile_count,
        ordinary_group_count=schedule.ordinary_group_count,
        special_tile_count=schedule.special_tile_count,
        atom_order=mx.array(schedule.atom_order, dtype=mx.int32),
        ordinary_left_blocks=mx.array(schedule.ordinary_left_blocks, dtype=mx.int32),
        ordinary_left_slices=mx.array(schedule.ordinary_left_slices, dtype=mx.int32),
        ordinary_right_atoms=mx.array(schedule.ordinary_right_atoms, dtype=mx.int32),
        ordinary_group_starts=mx.array(schedule.ordinary_group_starts, dtype=mx.int32),
        ordinary_group_counts=mx.array(schedule.ordinary_group_counts, dtype=mx.int32),
        special_blocks=mx.array(schedule.special_blocks, dtype=mx.int32),
        special_lj_enabled=mx.array(schedule.special_lj_enabled, dtype=mx.uint32),
        special_lj_one_four=mx.array(schedule.special_lj_one_four, dtype=mx.uint32),
        special_work_left_blocks=mx.array(schedule.special_work_left_blocks, dtype=mx.int32),
        special_work_left_slices=mx.array(schedule.special_work_left_slices, dtype=mx.int32),
        special_work_right_atoms=mx.array(schedule.special_work_right_atoms, dtype=mx.int32),
        special_work_lj_enabled=mx.array(schedule.special_work_lj_enabled, dtype=mx.uint32),
        special_work_lj_one_four=mx.array(
            schedule.special_work_lj_one_four,
            dtype=mx.uint32,
        ),
        special_work_diagonal=mx.array(schedule.special_work_diagonal, dtype=mx.int32),
    )


def _build_device_block_geometry32(
    positions: object,
    box_lengths: object,
    *,
    search_radius: float,
) -> _DeviceBlockGeometry32:
    """Build the spatial atom order and periodic 32-atom block bounds on device."""

    positions_mx = as_mx_array(positions, dtype=mx.float32)
    box = np.asarray(box_lengths, dtype=np.float32)
    if positions_mx.ndim != 2 or positions_mx.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions_mx.shape[0])
    if atom_count == 0:
        raise ValueError("device block geometry requires atoms")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be a finite positive vector with shape (3,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    if 2.0 * float(search_radius) >= float(np.min(box)):
        raise ValueError("search_radius must be smaller than half the shortest box length")

    cell_width = float(search_radius) / 3.0
    cell_counts = np.maximum(np.floor(box / cell_width).astype(np.int64), 1)
    cell_count = int(np.prod(cell_counts, dtype=np.int64))
    if cell_count > np.iinfo(np.int32).max:
        raise ValueError("device spatial cell inventory exceeds int32 capacity")
    box_mx = mx.array(box, dtype=mx.float32)
    cell_counts_mx = mx.array(cell_counts.astype(np.int32), dtype=mx.int32)
    wrapped = positions_mx - box_mx * mx.floor(positions_mx / box_mx)
    cells = mx.floor(wrapped * cell_counts_mx / box_mx).astype(mx.int32)
    cells = mx.minimum(cells, cell_counts_mx - 1)
    keys = cells[:, 0] + cell_counts_mx[0] * (
        cells[:, 1] + cell_counts_mx[1] * cells[:, 2]
    )
    atom_order = mx.argsort(keys).astype(mx.int32)
    sorted_keys = keys[atom_order]
    occupied_cell_count = (
        1 + mx.sum((sorted_keys[1:] != sorted_keys[:-1]).astype(mx.int32))
    ).astype(mx.int32)
    padded_atom_count = (
        (atom_count + _INTERACTION_TILE_SIZE - 1) // _INTERACTION_TILE_SIZE
    ) * _INTERACTION_TILE_SIZE
    padding = padded_atom_count - atom_count
    if padding:
        atom_order = mx.concatenate(
            (atom_order, mx.full((padding,), -1, dtype=mx.int32)),
        )
    inverse_order = (
        mx.zeros((atom_count,), dtype=mx.int32)
        .at[atom_order[:atom_count]]
        .add(mx.arange(atom_count, dtype=mx.int32))
    )
    box_lengths_and_inverses = mx.concatenate((box_mx, 1.0 / box_mx))
    center_radius, half_extent = _interaction32_block_geometry(
        positions_mx,
        atom_order,
        box_lengths_and_inverses,
    )
    block_traversal = mx.argsort(mx.sum(half_extent, axis=1)).astype(mx.int32)
    return _DeviceBlockGeometry32(
        atom_count=atom_count,
        search_radius=float(search_radius),
        box_lengths=box,
        cell_counts=tuple(int(value) for value in cell_counts),
        occupied_cell_count=occupied_cell_count,
        atom_order=atom_order,
        inverse_order=inverse_order,
        center_radius=center_radius,
        half_extent=half_extent,
        block_traversal=block_traversal,
    )


def _build_device_special_block_inventory32(
    geometry: _DeviceBlockGeometry32,
    *,
    lj_exclusion_pairs: object = (),
    lj_one_four_pairs: object = (),
    topology: _PreparedInteraction32Topology | None = None,
) -> _DeviceSpecialBlockInventory32:
    """Mark diagonal and topology-bearing packed block pairs on device."""

    atom_count = geometry.atom_count
    block_count = geometry.block_count
    block_code_capacity = block_count * block_count
    if block_code_capacity > np.iinfo(np.int32).max:
        raise ValueError("32-atom block-pair inventory exceeds int32 capacity")
    if topology is None:
        topology = _prepare_interaction32_topology(
            atom_count,
            lj_exclusion_pairs=lj_exclusion_pairs,
            lj_one_four_pairs=lj_one_four_pairs,
        )
    if topology.atom_count != atom_count:
        raise ValueError("prepared Interaction32 topology must match the geometry")
    exclusion_pairs = topology.exclusion_pairs
    one_four_pairs = topology.one_four_pairs
    code_groups = [
        mx.arange(block_count, dtype=mx.int32) * (block_count + 1),
    ]
    for pairs in (exclusion_pairs, one_four_pairs):
        if int(pairs.shape[0]) == 0:
            continue
        ordered = geometry.inverse_order[pairs]
        blocks = ordered // _INTERACTION_TILE_SIZE
        low = mx.minimum(blocks[:, 0], blocks[:, 1])
        high = mx.maximum(blocks[:, 0], blocks[:, 1])
        code_groups.append(low * block_count + high)
    codes = mx.concatenate(code_groups)
    block_codes = mx.sort(codes)
    block_code_unique = mx.concatenate(
        (
            mx.ones((1,), dtype=mx.int32),
            (block_codes[1:] != block_codes[:-1]).astype(mx.int32),
        )
    )
    special_count = mx.sum(block_code_unique).astype(mx.int32)
    return _DeviceSpecialBlockInventory32(
        atom_count=atom_count,
        block_count=block_count,
        exclusion_pairs=exclusion_pairs,
        one_four_pairs=one_four_pairs,
        block_codes=block_codes,
        block_code_unique=block_code_unique,
        special_count=special_count,
        topology_offsets=topology.topology_offsets,
        topology_neighbors=topology.topology_neighbors,
        topology_classes=topology.topology_classes,
    )


def _count_device_schedule_inventory32(
    positions: object,
    geometry: _DeviceBlockGeometry32,
    special: _DeviceSpecialBlockInventory32,
    *,
    ordinary_tiles_per_group: int = _DEFAULT_ORDINARY_TILES_PER_GROUP,
    _mode_cache_limit_bytes: int = _INTERACTION32_MODE_CACHE_LIMIT_BYTES,
) -> _DeviceScheduleInventory32:
    """Count the logical ordinary and special schedule inventory on device."""

    if special.atom_count != geometry.atom_count or special.block_count != geometry.block_count:
        raise ValueError("special inventory must match the device block geometry")
    if ordinary_tiles_per_group < 1:
        raise ValueError("ordinary_tiles_per_group must be positive")
    if _mode_cache_limit_bytes < 0:
        raise ValueError("mode cache limit must be non-negative")
    positions_mx = as_mx_array(positions, dtype=mx.float32)
    if positions_mx.shape != (geometry.atom_count, 3):
        raise ValueError(f"positions must have shape ({geometry.atom_count}, 3)")
    box_mx = mx.array(geometry.box_lengths, dtype=mx.float32)
    box_lengths_and_inverses = mx.concatenate((box_mx, 1.0 / box_mx))
    mode_cache_bytes = 4 * geometry.block_count * (geometry.block_count - 1)
    mode_entry_counts, mode_words = _interaction32_ordinary_mode_counts(
        positions_mx,
        geometry.atom_order,
        geometry.center_radius,
        geometry.half_extent,
        geometry.block_traversal,
        special.block_codes,
        box_lengths_and_inverses,
        search_radius=geometry.search_radius,
        retain_modes=0 < mode_cache_bytes <= _mode_cache_limit_bytes,
    )
    flat_entry_counts = mode_entry_counts.reshape((-1,))
    mode_tile_counts = (
        flat_entry_counts + _INTERACTION_TILE_SIZE - 1
    ) // _INTERACTION_TILE_SIZE
    mode_tile_prefix = mx.cumsum(mode_tile_counts)
    mode_group_counts = (
        mode_tile_counts + ordinary_tiles_per_group - 1
    ) // ordinary_tiles_per_group
    mode_group_prefix = mx.cumsum(mode_group_counts)
    logical_pair_lanes = mx.sum(
        mode_entry_counts
        * mx.array([16, 16, 32], dtype=mx.int32)[None, :]
    )
    inventory = mx.stack(
        (
            mx.sum(flat_entry_counts),
            mode_tile_prefix[-1],
            mode_group_prefix[-1],
            logical_pair_lanes,
            special.special_count,
            geometry.occupied_cell_count,
        )
    )
    mx.eval(inventory)
    (
        right_entry_count,
        ordinary_tile_count,
        ordinary_group_count,
        logical_pair_lane_count,
        special_tile_count_inventory,
        occupied_cell_count,
    ) = (int(value) for value in np.asarray(inventory))
    return _DeviceScheduleInventory32(
        positions=positions_mx,
        geometry=geometry,
        special=special,
        ordinary_tiles_per_group=ordinary_tiles_per_group,
        occupied_cell_count=occupied_cell_count,
        right_entry_count=right_entry_count,
        logical_pair_lanes=logical_pair_lane_count,
        ordinary_tile_count=ordinary_tile_count,
        ordinary_group_count=ordinary_group_count,
        special_tile_count=special_tile_count_inventory,
        mode_cache_bytes=mode_cache_bytes if mode_words is not None else 0,
        mode_entry_counts=mode_entry_counts,
        mode_words=mode_words,
        mode_tile_counts=mode_tile_counts,
        mode_tile_prefix=mode_tile_prefix,
        mode_group_counts=mode_group_counts,
        mode_group_prefix=mode_group_prefix,
    )


def _materialize_device_ordinary_schedule32(
    inventory: _DeviceScheduleInventory32,
    *,
    tile_capacity: int,
    group_capacity: int,
) -> _DeviceOrdinarySchedule32:
    """Scatter ordinary schedule payload only after capacity admission."""

    geometry = inventory.geometry
    if tile_capacity < inventory.ordinary_tile_count:
        raise ValueError("ordinary tile capacity is below the logical inventory")
    if group_capacity < inventory.ordinary_group_count:
        raise ValueError("ordinary group capacity is below the logical inventory")
    positions_mx = inventory.positions
    box_mx = mx.array(geometry.box_lengths, dtype=mx.float32)
    box_lengths_and_inverses = mx.concatenate((box_mx, 1.0 / box_mx))
    ordinary_left_blocks, ordinary_right_atoms, ordinary_half_modes = (
        _interaction32_ordinary_scatter_sized(
            positions_mx,
            geometry.atom_order,
            geometry.center_radius,
            geometry.half_extent,
            geometry.block_traversal,
            inventory.special.block_codes,
            inventory.mode_words,
            inventory.mode_tile_counts,
            inventory.mode_tile_prefix,
            box_lengths_and_inverses,
            search_radius=geometry.search_radius,
            accepted_tile_count=tile_capacity,
        )
    )
    ordinary_group_starts, ordinary_group_counts = _neighbor_tile_force_groups_sized(
        inventory.mode_tile_counts,
        inventory.mode_tile_prefix,
        inventory.mode_group_counts,
        inventory.mode_group_prefix,
        accepted_count=group_capacity,
        items_per_group=inventory.ordinary_tiles_per_group,
    )
    ordinary_tile_count = inventory.ordinary_tile_count
    ordinary_group_count = inventory.ordinary_group_count
    return _DeviceOrdinarySchedule32(
        atom_count=geometry.atom_count,
        search_radius=geometry.search_radius,
        padded_atom_count=geometry.padded_atom_count,
        right_entry_count=inventory.right_entry_count,
        logical_pair_lanes=inventory.logical_pair_lanes,
        ordinary_tile_count=ordinary_tile_count,
        ordinary_group_count=ordinary_group_count,
        ordinary_tile_capacity=tile_capacity,
        ordinary_group_capacity=group_capacity,
        special_tile_count_inventory=inventory.special_tile_count,
        mode_entry_counts=inventory.mode_entry_counts,
        ordinary_left_blocks=ordinary_left_blocks[:ordinary_tile_count],
        ordinary_right_atoms=ordinary_right_atoms[:ordinary_tile_count],
        ordinary_half_modes=ordinary_half_modes[:ordinary_tile_count],
        ordinary_group_starts=ordinary_group_starts[:ordinary_group_count],
        ordinary_group_counts=ordinary_group_counts[:ordinary_group_count],
    )


def _build_device_ordinary_schedule32(
    positions: object,
    geometry: _DeviceBlockGeometry32,
    special: _DeviceSpecialBlockInventory32,
    *,
    ordinary_tiles_per_group: int = _DEFAULT_ORDINARY_TILES_PER_GROUP,
    _mode_cache_limit_bytes: int = _INTERACTION32_MODE_CACHE_LIMIT_BYTES,
) -> _DeviceOrdinarySchedule32:
    """Build an exact-sized ordinary schedule for research callers."""

    inventory = _count_device_schedule_inventory32(
        positions,
        geometry,
        special,
        ordinary_tiles_per_group=ordinary_tiles_per_group,
        _mode_cache_limit_bytes=_mode_cache_limit_bytes,
    )
    return _materialize_device_ordinary_schedule32(
        inventory,
        tile_capacity=inventory.ordinary_tile_count,
        group_capacity=inventory.ordinary_group_count,
    )


def _build_device_special_schedule32(
    geometry: _DeviceBlockGeometry32,
    special: _DeviceSpecialBlockInventory32,
    *,
    special_tile_count: int | None = None,
    tile_capacity: int | None = None,
) -> _DeviceSpecialSchedule32:
    """Build compact special blocks and conservative two-half work on device."""

    if special.atom_count != geometry.atom_count or special.block_count != geometry.block_count:
        raise ValueError("special inventory must match the device block geometry")
    special_prefix = mx.cumsum(special.block_code_unique)
    if special_tile_count is None:
        mx.eval(special.special_count)
        special_tile_count = int(np.asarray(special.special_count))
    if not 0 <= special_tile_count <= special.block_count * special.block_count:
        raise ValueError("special tile count is incompatible with the block inventory")
    if tile_capacity is None:
        tile_capacity = special_tile_count
    if tile_capacity < special_tile_count:
        raise ValueError("special tile capacity is below the logical inventory")
    special_blocks = _interaction32_special_blocks_sized(
        special.block_codes,
        special.block_code_unique,
        special_prefix,
        block_count=geometry.block_count,
        special_count=special_tile_count,
        block_capacity=tile_capacity,
    )
    (
        special_work_left_blocks,
        special_work_left_slices,
        special_work_right_atoms,
        special_work_lj_enabled,
        special_work_lj_one_four,
        special_work_diagonal,
    ) = _interaction32_special_work_two_halves(
        geometry.atom_order,
        special_blocks[:special_tile_count],
        special.topology_offsets,
        special.topology_neighbors,
        special.topology_classes,
        work_capacity=2 * tile_capacity,
    )
    special_work_count = 2 * special_tile_count
    return _DeviceSpecialSchedule32(
        atom_count=geometry.atom_count,
        padded_atom_count=geometry.padded_atom_count,
        special_tile_count=special_tile_count,
        special_work_count=special_work_count,
        special_tile_capacity=tile_capacity,
        special_work_capacity=2 * tile_capacity,
        special_blocks=special_blocks[:special_tile_count],
        special_work_left_blocks=special_work_left_blocks[:special_work_count],
        special_work_left_slices=special_work_left_slices[:special_work_count],
        special_work_right_atoms=special_work_right_atoms[:special_work_count],
        special_work_lj_enabled=special_work_lj_enabled[:special_work_count],
        special_work_lj_one_four=special_work_lj_one_four[:special_work_count],
        special_work_diagonal=special_work_diagonal[:special_work_count],
    )


def _interaction32_reserved_capacity(logical_count: int) -> int:
    """Round a logical count to the stable 25 percent reserve policy."""

    if logical_count < 0:
        raise ValueError("logical capacity count must be non-negative")
    if logical_count == 0:
        return 0
    reserved = ceil(logical_count * _INTERACTION32_CAPACITY_RESERVE)
    quantum = _INTERACTION32_CAPACITY_QUANTUM
    return ((reserved + quantum - 1) // quantum) * quantum


def _interaction32_capacity_for_inventory(
    inventory: _DeviceScheduleInventory32,
    current: _Interaction32ScheduleCapacity | None = None,
) -> _Interaction32ScheduleCapacity:
    """Return retained or grown capacity for one logical inventory."""

    required = _Interaction32ScheduleCapacity(
        ordinary_tiles=_interaction32_reserved_capacity(
            inventory.ordinary_tile_count
        ),
        ordinary_groups=_interaction32_reserved_capacity(
            inventory.ordinary_group_count
        ),
        special_tiles=_interaction32_reserved_capacity(
            inventory.special_tile_count
        ),
    )
    if current is None:
        return required
    return _Interaction32ScheduleCapacity(
        ordinary_tiles=max(current.ordinary_tiles, required.ordinary_tiles),
        ordinary_groups=max(current.ordinary_groups, required.ordinary_groups),
        special_tiles=max(current.special_tiles, required.special_tiles),
    )


def _interaction32_capacity_overflow_fields(
    inventory: _DeviceScheduleInventory32,
    capacity: _Interaction32ScheduleCapacity | None,
) -> tuple[str, ...]:
    """Name every logical inventory that cannot fit the supplied capacity."""

    if capacity is None:
        return ("ordinary_tiles", "ordinary_groups", "special_tiles")
    fields: list[str] = []
    if inventory.ordinary_tile_count > capacity.ordinary_tiles:
        fields.append("ordinary_tiles")
    if inventory.ordinary_group_count > capacity.ordinary_groups:
        fields.append("ordinary_groups")
    if inventory.special_tile_count > capacity.special_tiles:
        fields.append("special_tiles")
    return tuple(fields)


def _interaction32_topology_digest(
    atom_count: int,
    exclusions: np.ndarray,
    one_four: np.ndarray,
) -> str:
    """Fingerprint canonical topology inputs for generation ownership."""

    digest = sha256()
    digest.update(np.asarray([atom_count], dtype="<i8").tobytes())
    for label, pairs in ((b"exclusions", exclusions), (b"one_four", one_four)):
        digest.update(label)
        digest.update(np.asarray(pairs, dtype="<i4").tobytes())
    return digest.hexdigest()


def _prepare_interaction32_topology(
    atom_count: int,
    *,
    lj_exclusion_pairs: object = (),
    lj_one_four_pairs: object = (),
) -> _PreparedInteraction32Topology:
    """Prepare immutable topology data for reuse across spatial generations."""

    exclusions = _normalize_pairs(lj_exclusion_pairs, atom_count, "lj_exclusion_pairs")
    one_four = _normalize_pairs(lj_one_four_pairs, atom_count, "lj_one_four_pairs")
    exclusion_codes = _pair_codes(exclusions, atom_count)
    one_four_codes = _pair_codes(one_four, atom_count)
    if np.any(_contains_sorted(exclusion_codes, one_four_codes)):
        raise ValueError("LJ exclusion and one-four pair sets must be disjoint")
    topology_offsets, topology_neighbors, topology_classes = _build_owner_topology(
        atom_count,
        exclusions,
        one_four,
    )
    return _PreparedInteraction32Topology(
        atom_count=atom_count,
        exclusion_source_id=id(lj_exclusion_pairs),
        one_four_source_id=id(lj_one_four_pairs),
        device=str(mx.default_device()),
        digest=_interaction32_topology_digest(atom_count, exclusions, one_four),
        exclusion_pairs=mx.array(exclusions, dtype=mx.int32),
        one_four_pairs=mx.array(one_four, dtype=mx.int32),
        topology_offsets=mx.array(topology_offsets, dtype=mx.int32),
        topology_neighbors=mx.array(topology_neighbors, dtype=mx.int32),
        topology_classes=mx.array(topology_classes, dtype=mx.int32),
    )


def _interaction32_topology_matches_sources(
    topology: _PreparedInteraction32Topology,
    *,
    atom_count: int,
    lj_exclusion_pairs: object,
    lj_one_four_pairs: object,
) -> bool:
    """Return whether a topology snapshot still owns the declared inputs."""

    return (
        topology.atom_count == atom_count
        and topology.exclusion_source_id == id(lj_exclusion_pairs)
        and topology.one_four_source_id == id(lj_one_four_pairs)
        and topology.device == str(mx.default_device())
    )


def _materialize_device_schedule_attempt32(
    inventory: _DeviceScheduleInventory32,
    *,
    topology: _PreparedInteraction32Topology,
    capacity: _Interaction32ScheduleCapacity,
    generation_value: int,
) -> _DeviceScheduleBuildAttempt32:
    """Materialize one admitted generation after all capacity checks pass."""

    overflow_fields = _interaction32_capacity_overflow_fields(inventory, capacity)
    recommended = _interaction32_capacity_for_inventory(inventory, capacity)
    if overflow_fields:
        return _DeviceScheduleBuildAttempt32(
            inventory=inventory,
            topology=topology,
            requested_capacity=capacity,
            recommended_capacity=recommended,
            overflow_fields=overflow_fields,
            generation_value=generation_value,
            topology_digest=topology.digest,
            schedule=None,
        )
    ordinary_started = _interaction32_profile_start()
    ordinary = _materialize_device_ordinary_schedule32(
        inventory,
        tile_capacity=capacity.ordinary_tiles,
        group_capacity=capacity.ordinary_groups,
    )
    _interaction32_profile_finish_stage(
        "ordinary_scatter",
        ordinary_started,
        ordinary.ordinary_left_blocks,
        ordinary.ordinary_right_atoms,
        ordinary.ordinary_half_modes,
        ordinary.ordinary_group_starts,
        ordinary.ordinary_group_counts,
    )
    special_started = _interaction32_profile_start()
    special = _build_device_special_schedule32(
        inventory.geometry,
        inventory.special,
        special_tile_count=inventory.special_tile_count,
        tile_capacity=capacity.special_tiles,
    )
    _interaction32_profile_finish_stage(
        "special_scatter",
        special_started,
        special.special_blocks,
        special.special_work_left_blocks,
        special.special_work_left_slices,
        special.special_work_right_atoms,
        special.special_work_lj_enabled,
        special.special_work_lj_one_four,
        special.special_work_diagonal,
    )
    generation = _Interaction32Generation(
        value=generation_value,
        atom_count=inventory.geometry.atom_count,
        box_lengths=tuple(float(value) for value in inventory.geometry.box_lengths),
        search_radius=inventory.geometry.search_radius,
        topology_digest=topology.digest,
        capacity=capacity,
    )
    schedule = _assemble_device_fused_half_schedule32(
        inventory.geometry,
        ordinary,
        special,
        generation=generation,
    )
    return _DeviceScheduleBuildAttempt32(
        inventory=inventory,
        topology=topology,
        requested_capacity=capacity,
        recommended_capacity=capacity,
        overflow_fields=(),
        generation_value=generation_value,
        topology_digest=topology.digest,
        schedule=schedule,
    )


def _try_build_device_fused_half_schedule32(
    positions: object,
    box_lengths: object,
    *,
    search_radius: float,
    capacity: _Interaction32ScheduleCapacity | None,
    generation_value: int,
    lj_exclusion_pairs: object = (),
    lj_one_four_pairs: object = (),
    ordinary_tiles_per_group: int = _DEFAULT_ORDINARY_TILES_PER_GROUP,
    topology: _PreparedInteraction32Topology | None = None,
) -> _DeviceScheduleBuildAttempt32:
    """Count one candidate generation and stop before scatter on overflow."""

    if generation_value < 0:
        raise ValueError("generation_value must be non-negative")
    geometry_started = _interaction32_profile_start()
    geometry = _build_device_block_geometry32(
        positions,
        box_lengths,
        search_radius=search_radius,
    )
    _interaction32_profile_finish_stage(
        "geometry_validation_and_sort",
        geometry_started,
        geometry.occupied_cell_count,
        geometry.atom_order,
        geometry.inverse_order,
        geometry.center_radius,
        geometry.half_extent,
        geometry.block_traversal,
    )
    topology_started = _interaction32_profile_start()
    if topology is None or not _interaction32_topology_matches_sources(
        topology,
        atom_count=geometry.atom_count,
        lj_exclusion_pairs=lj_exclusion_pairs,
        lj_one_four_pairs=lj_one_four_pairs,
    ):
        topology = _prepare_interaction32_topology(
            geometry.atom_count,
            lj_exclusion_pairs=lj_exclusion_pairs,
            lj_one_four_pairs=lj_one_four_pairs,
        )
    _interaction32_profile_finish_stage(
        "topology_preparation",
        topology_started,
        topology.exclusion_pairs,
        topology.one_four_pairs,
        topology.topology_offsets,
        topology.topology_neighbors,
        topology.topology_classes,
    )
    special_inventory_started = _interaction32_profile_start()
    special = _build_device_special_block_inventory32(
        geometry,
        topology=topology,
    )
    _interaction32_profile_finish_stage(
        "special_block_inventory",
        special_inventory_started,
        special.exclusion_pairs,
        special.one_four_pairs,
        special.block_codes,
        special.block_code_unique,
        special.special_count,
        special.topology_offsets,
        special.topology_neighbors,
        special.topology_classes,
    )
    count_started = _interaction32_profile_start()
    inventory = _count_device_schedule_inventory32(
        positions,
        geometry,
        special,
        ordinary_tiles_per_group=ordinary_tiles_per_group,
    )
    _interaction32_profile_finish_stage(
        "ordinary_count_and_prefix_readback",
        count_started,
    )
    admission_started = _interaction32_profile_start()
    recommended = _interaction32_capacity_for_inventory(inventory, capacity)
    overflow_fields = _interaction32_capacity_overflow_fields(inventory, capacity)
    _interaction32_profile_finish_stage("capacity_admission", admission_started)
    if overflow_fields:
        return _DeviceScheduleBuildAttempt32(
            inventory=inventory,
            topology=topology,
            requested_capacity=capacity,
            recommended_capacity=recommended,
            overflow_fields=overflow_fields,
            generation_value=generation_value,
            topology_digest=topology.digest,
            schedule=None,
        )
    return _materialize_device_schedule_attempt32(
        inventory,
        topology=topology,
        capacity=capacity,
        generation_value=generation_value,
    )


def _retry_device_fused_half_schedule32(
    attempt: _DeviceScheduleBuildAttempt32,
    *,
    capacity: _Interaction32ScheduleCapacity | None = None,
) -> _DeviceScheduleBuildAttempt32:
    """Retry an overflowed inventory without repeating spatial search."""

    if not attempt.overflow or attempt.schedule is not None:
        raise ValueError("only an overflowed build attempt can be retried")
    selected = attempt.recommended_capacity if capacity is None else capacity
    return _materialize_device_schedule_attempt32(
        attempt.inventory,
        topology=attempt.topology,
        capacity=selected,
        generation_value=attempt.generation_value,
    )


def _assemble_device_fused_half_schedule32(
    geometry: _DeviceBlockGeometry32,
    ordinary: _DeviceOrdinarySchedule32,
    special: _DeviceSpecialSchedule32,
    *,
    generation: _Interaction32Generation | None = None,
) -> _DeviceFusedHalfSchedule32:
    """Assemble matching device-built ordinary and special schedule sections."""

    if (
        ordinary.atom_count != geometry.atom_count
        or special.atom_count != geometry.atom_count
        or ordinary.padded_atom_count != geometry.padded_atom_count
        or special.padded_atom_count != geometry.padded_atom_count
        or ordinary.search_radius != geometry.search_radius
        or special.special_tile_count != ordinary.special_tile_count_inventory
    ):
        raise ValueError("device schedule sections must share one block generation")
    return _DeviceFusedHalfSchedule32(
        atom_count=geometry.atom_count,
        search_radius=geometry.search_radius,
        padded_atom_count=geometry.padded_atom_count,
        ordinary_tile_count=ordinary.ordinary_tile_count,
        ordinary_group_count=ordinary.ordinary_group_count,
        atom_order=geometry.atom_order,
        ordinary_left_blocks=ordinary.ordinary_left_blocks,
        ordinary_right_atoms=ordinary.ordinary_right_atoms,
        ordinary_half_modes=ordinary.ordinary_half_modes,
        ordinary_group_starts=ordinary.ordinary_group_starts,
        ordinary_group_counts=ordinary.ordinary_group_counts,
        special_work_left_blocks=special.special_work_left_blocks,
        special_work_left_slices=special.special_work_left_slices,
        special_work_right_atoms=special.special_work_right_atoms,
        special_work_lj_enabled=special.special_work_lj_enabled,
        special_work_lj_one_four=special.special_work_lj_one_four,
        special_work_diagonal=special.special_work_diagonal,
        generation=generation,
    )


def _owner_schedule_to_device32(
    schedule: _OwnerComputeSchedule32,
) -> _DeviceOwnerComputeSchedule32:
    return _DeviceOwnerComputeSchedule32(
        atom_count=schedule.atom_count,
        search_radius=schedule.search_radius,
        padded_atom_count=schedule.padded_atom_count,
        block_count=schedule.block_count,
        atom_order=mx.array(schedule.atom_order, dtype=mx.int32),
        owner_offsets=mx.array(schedule.owner_offsets, dtype=mx.int32),
        right_atoms=mx.array(schedule.right_atoms, dtype=mx.int32),
        topology_offsets=mx.array(schedule.topology_offsets, dtype=mx.int32),
        topology_neighbors=mx.array(schedule.topology_neighbors, dtype=mx.int32),
        topology_classes=mx.array(schedule.topology_classes, dtype=mx.int32),
    )


def _fused_half_schedule_to_device32(
    schedule: _FusedHalfSchedule32,
) -> _DeviceFusedHalfSchedule32:
    return _DeviceFusedHalfSchedule32(
        atom_count=schedule.atom_count,
        search_radius=schedule.search_radius,
        padded_atom_count=schedule.padded_atom_count,
        ordinary_tile_count=schedule.ordinary_tile_count,
        ordinary_group_count=schedule.ordinary_group_count,
        atom_order=mx.array(schedule.atom_order, dtype=mx.int32),
        ordinary_left_blocks=mx.array(schedule.ordinary_left_blocks, dtype=mx.int32),
        ordinary_right_atoms=mx.array(schedule.ordinary_right_atoms, dtype=mx.int32),
        ordinary_half_modes=mx.array(schedule.ordinary_half_modes, dtype=mx.int32),
        ordinary_group_starts=mx.array(schedule.ordinary_group_starts, dtype=mx.int32),
        ordinary_group_counts=mx.array(schedule.ordinary_group_counts, dtype=mx.int32),
        special_work_left_blocks=mx.array(schedule.special_work_left_blocks, dtype=mx.int32),
        special_work_left_slices=mx.array(schedule.special_work_left_slices, dtype=mx.int32),
        special_work_right_atoms=mx.array(schedule.special_work_right_atoms, dtype=mx.int32),
        special_work_lj_enabled=mx.array(schedule.special_work_lj_enabled, dtype=mx.uint32),
        special_work_lj_one_four=mx.array(
            schedule.special_work_lj_one_four,
            dtype=mx.uint32,
        ),
        special_work_diagonal=mx.array(schedule.special_work_diagonal, dtype=mx.int32),
    )


def _fuse_interaction_halves32(
    schedule: _InteractionSchedule32,
    *,
    ordinary_tiles_per_group: int = _DEFAULT_ORDINARY_TILES_PER_GROUP,
) -> _FusedHalfSchedule32:
    if schedule.left_slice_size != 16:
        raise ValueError("fused-half scheduling requires a 16-atom base schedule")
    if ordinary_tiles_per_group < 1:
        raise ValueError("ordinary_tiles_per_group must be positive")

    sentinel = schedule.padded_atom_count
    fused_left: list[int] = []
    fused_right: list[np.ndarray] = []
    fused_modes: list[int] = []
    run_start = 0
    while run_start < schedule.ordinary_tile_count:
        left_block = int(schedule.ordinary_left_blocks[run_start])
        run_stop = run_start + 1
        while (
            run_stop < schedule.ordinary_tile_count
            and schedule.ordinary_left_blocks[run_stop] == left_block
        ):
            run_stop += 1
        rights = schedule.ordinary_right_atoms[run_start:run_stop].reshape(-1)
        slices = np.repeat(
            schedule.ordinary_left_slices[run_start:run_stop],
            _INTERACTION_TILE_SIZE,
        )
        valid = rights < sentinel
        rights = rights[valid]
        slices = slices[valid]
        unique_rights, inverse = np.unique(rights, return_inverse=True)
        half_masks = np.zeros((unique_rights.shape[0],), dtype=np.uint32)
        np.bitwise_or.at(half_masks, inverse, np.left_shift(np.uint32(1), slices))
        for mode in (1, 2, 3):
            selected = unique_rights[half_masks == mode]
            if selected.size == 0:
                continue
            padded_size = (
                (selected.size + _INTERACTION_TILE_SIZE - 1)
                // _INTERACTION_TILE_SIZE
            ) * _INTERACTION_TILE_SIZE
            padded_rights = np.full((padded_size,), sentinel, dtype=np.int32)
            padded_rights[: selected.size] = selected
            tile_count = padded_size // _INTERACTION_TILE_SIZE
            fused_left.extend([left_block] * tile_count)
            fused_modes.extend([mode] * tile_count)
            fused_right.append(padded_rights.reshape((-1, _INTERACTION_TILE_SIZE)))
        run_start = run_stop

    ordinary_left_blocks = np.asarray(fused_left, dtype=np.int32)
    empty_shape = (0, _INTERACTION_TILE_SIZE)
    ordinary_right_atoms = (
        np.concatenate(fused_right, axis=0)
        if fused_right
        else np.empty(empty_shape, dtype=np.int32)
    )
    ordinary_half_modes = np.asarray(fused_modes, dtype=np.int32)
    ordinary_group_starts, ordinary_group_counts = _group_ordinary_tiles(
        ordinary_left_blocks,
        ordinary_half_modes,
        ordinary_tiles_per_group,
    )
    return _FusedHalfSchedule32(
        atom_count=schedule.atom_count,
        search_radius=schedule.search_radius,
        atom_order=schedule.atom_order,
        ordinary_left_blocks=ordinary_left_blocks,
        ordinary_right_atoms=ordinary_right_atoms,
        ordinary_half_modes=ordinary_half_modes,
        ordinary_group_starts=ordinary_group_starts,
        ordinary_group_counts=ordinary_group_counts,
        special_work_left_blocks=schedule.special_work_left_blocks,
        special_work_left_slices=schedule.special_work_left_slices,
        special_work_right_atoms=schedule.special_work_right_atoms,
        special_work_lj_enabled=schedule.special_work_lj_enabled,
        special_work_lj_one_four=schedule.special_work_lj_one_four,
        special_work_diagonal=schedule.special_work_diagonal,
    )


def _build_owner_topology(
    atom_count: int,
    exclusions: np.ndarray,
    one_four: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owners: list[np.ndarray] = []
    neighbors: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for pairs, topology_class in ((exclusions, 0), (one_four, 1)):
        if pairs.shape[0] == 0:
            continue
        owners.append(np.concatenate((pairs[:, 0], pairs[:, 1])))
        neighbors.append(np.concatenate((pairs[:, 1], pairs[:, 0])))
        classes.append(np.full((2 * pairs.shape[0],), topology_class, dtype=np.int32))
    if not owners:
        return (
            np.zeros((atom_count + 1,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    owner_array = np.concatenate(owners).astype(np.int32, copy=False)
    neighbor_array = np.concatenate(neighbors).astype(np.int32, copy=False)
    class_array = np.concatenate(classes)
    order = np.argsort(owner_array, kind="stable")
    owner_array = owner_array[order]
    neighbor_array = neighbor_array[order]
    class_array = class_array[order]
    offsets = np.empty((atom_count + 1,), dtype=np.int32)
    offsets[0] = 0
    np.cumsum(np.bincount(owner_array, minlength=atom_count), out=offsets[1:])
    return offsets, neighbor_array, class_array


def _build_owner_compute_schedule32(
    positions: object,
    box_lengths: object,
    *,
    search_radius: float,
    lj_exclusion_pairs: object = (),
    lj_one_four_pairs: object = (),
) -> _OwnerComputeSchedule32:
    positions_np = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box_lengths, dtype=np.float64)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions_np.shape[0])
    if atom_count == 0:
        raise ValueError("the owner-computes schedule requires atoms")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be a finite positive vector with shape (3,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    if 2.0 * float(search_radius) >= float(np.min(box)):
        raise ValueError("search_radius must be smaller than half the shortest box length")

    exclusions = _normalize_pairs(lj_exclusion_pairs, atom_count, "lj_exclusion_pairs")
    one_four = _normalize_pairs(lj_one_four_pairs, atom_count, "lj_one_four_pairs")
    if np.any(
        _contains_sorted(
            _pair_codes(exclusions, atom_count),
            _pair_codes(one_four, atom_count),
        )
    ):
        raise ValueError("LJ exclusion and one-four pair sets must be disjoint")

    atom_order = _cell_atom_order(positions_np, box, float(search_radius))
    padded_count = (
        (atom_count + _INTERACTION_TILE_SIZE - 1) // _INTERACTION_TILE_SIZE
    ) * _INTERACTION_TILE_SIZE
    padded_order = np.full((padded_count,), -1, dtype=np.int32)
    padded_order[:atom_count] = atom_order
    blocks = padded_order.reshape((-1, _INTERACTION_TILE_SIZE))
    inverse_order = np.empty((atom_count,), dtype=np.int32)
    inverse_order[atom_order] = np.arange(atom_count, dtype=np.int32)
    wrapped = positions_np - box * np.floor(positions_np / box)
    tree = cKDTree(wrapped, boxsize=box)
    sentinel = padded_count
    owner_offsets = [0]
    right_rows: list[np.ndarray] = []
    for owners in blocks:
        owners = owners[owners >= 0]
        neighborhoods = tree.query_ball_point(
            wrapped[owners],
            float(search_radius),
            return_sorted=False,
        )
        canonical_neighbors = np.unique(np.concatenate(neighborhoods))
        ordered_neighbors = np.sort(inverse_order[canonical_neighbors])
        padded_size = (
            (ordered_neighbors.size + _INTERACTION_TILE_SIZE - 1)
            // _INTERACTION_TILE_SIZE
        ) * _INTERACTION_TILE_SIZE
        row = np.full((padded_size,), sentinel, dtype=np.int32)
        row[: ordered_neighbors.size] = ordered_neighbors
        right_rows.append(row)
        owner_offsets.append(owner_offsets[-1] + padded_size)
    topology_offsets, topology_neighbors, topology_classes = _build_owner_topology(
        atom_count,
        exclusions,
        one_four,
    )
    return _OwnerComputeSchedule32(
        atom_count=atom_count,
        search_radius=float(search_radius),
        atom_order=padded_order,
        owner_offsets=np.asarray(owner_offsets, dtype=np.int32),
        right_atoms=np.concatenate(right_rows),
        topology_offsets=topology_offsets,
        topology_neighbors=topology_neighbors,
        topology_classes=topology_classes,
    )


def _build_interaction_schedule32(
    positions: object,
    box_lengths: object,
    *,
    search_radius: float,
    lj_exclusion_pairs: object = (),
    lj_one_four_pairs: object = (),
    ordinary_tiles_per_group: int = _DEFAULT_ORDINARY_TILES_PER_GROUP,
    left_slice_size: int = 16,
) -> _InteractionSchedule32:
    positions_np = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box_lengths, dtype=np.float64)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions_np.shape[0])
    if atom_count == 0:
        raise ValueError("the 32-atom interaction schedule requires atoms")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be a finite positive vector with shape (3,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    if ordinary_tiles_per_group < 1:
        raise ValueError("ordinary_tiles_per_group must be positive")
    if left_slice_size not in (4, 8, 16):
        raise ValueError("left_slice_size must be 4, 8, or 16")
    if 2.0 * float(search_radius) >= float(np.min(box)):
        raise ValueError("search_radius must be smaller than half the shortest box length")

    exclusions = _normalize_pairs(lj_exclusion_pairs, atom_count, "lj_exclusion_pairs")
    one_four = _normalize_pairs(lj_one_four_pairs, atom_count, "lj_one_four_pairs")
    exclusion_codes = _pair_codes(exclusions, atom_count)
    one_four_codes = _pair_codes(one_four, atom_count)
    if np.any(_contains_sorted(exclusion_codes, one_four_codes)):
        raise ValueError("LJ exclusion and one-four pair sets must be disjoint")

    atom_order = _cell_atom_order(positions_np, box, float(search_radius))
    block_atoms, valid, centers, extents, radii, inverse_order = _make_atom_blocks(
        positions_np, box, atom_order
    )
    block_count = int(block_atoms.shape[0])
    special_codes = _special_block_codes(block_count, inverse_order, exclusions, one_four)
    special_blocks = np.stack(
        (special_codes // block_count, special_codes % block_count), axis=1
    ).astype(np.int32)
    special_lj_enabled, special_lj_one_four = _special_lj_masks(
        block_atoms,
        special_blocks,
        exclusion_codes,
        one_four_codes,
        atom_count,
    )
    search_radius2 = float(search_radius) ** 2
    (
        special_work_left_blocks,
        special_work_left_slices,
        special_work_right_atoms,
        special_work_lj_enabled,
        special_work_lj_one_four,
        special_work_diagonal,
    ) = _build_special_work(
        positions_np,
        box,
        block_atoms,
        valid,
        special_blocks,
        special_lj_enabled,
        special_lj_one_four,
        search_radius2,
        left_slice_size,
    )

    traversal = np.argsort(np.sum(extents, axis=1), kind="stable")
    sentinel = block_atoms.size
    ordinary_left: list[int] = []
    ordinary_slice: list[int] = []
    ordinary_right: list[np.ndarray] = []
    for traversal_index, left_block in enumerate(traversal[:-1]):
        right_blocks = traversal[traversal_index + 1 :]
        delta = centers[right_blocks] - centers[left_block]
        delta -= box * np.rint(delta / box)
        center_distance2 = np.sum(delta * delta, axis=1)
        sphere_limit = float(search_radius) + radii[left_block] + radii[right_blocks]
        keep = center_distance2 < sphere_limit * sphere_limit
        if not np.any(keep):
            continue
        candidate_right = right_blocks[keep]
        candidate_delta = delta[keep]
        aabb_delta = np.maximum(
            np.abs(candidate_delta) - extents[left_block] - extents[candidate_right],
            0.0,
        )
        keep = np.sum(aabb_delta * aabb_delta, axis=1) < search_radius2
        candidate_right = candidate_right[keep]
        if candidate_right.size == 0:
            continue
        low = np.minimum(left_block, candidate_right).astype(np.int64)
        high = np.maximum(left_block, candidate_right).astype(np.int64)
        codes = low * np.int64(block_count) + high
        candidate_right = candidate_right[~_contains_sorted(special_codes, codes)]

        left_atoms = block_atoms[left_block]
        left_valid = valid[left_block]
        left_positions = positions_np[np.maximum(left_atoms, 0)]
        admitted: list[list[np.ndarray]] = [
            [] for _ in range(_INTERACTION_TILE_SIZE // left_slice_size)
        ]
        for right_block in candidate_right:
            right_atoms = block_atoms[right_block]
            right_valid = valid[right_block]
            right_positions = positions_np[np.maximum(right_atoms, 0)]
            pair_delta = left_positions[:, None, :] - right_positions[None, :, :]
            pair_delta -= box * np.rint(pair_delta / box)
            distance2 = np.sum(pair_delta * pair_delta, axis=2)
            pair_valid = left_valid[:, None] & right_valid[None, :]
            for left_slice in range(_INTERACTION_TILE_SIZE // left_slice_size):
                atom_slice = slice(
                    left_slice_size * left_slice,
                    left_slice_size * (left_slice + 1),
                )
                right_mask = np.any(
                    pair_valid[atom_slice] & (distance2[atom_slice] < search_radius2),
                    axis=0,
                )
                if np.any(right_mask):
                    slots = np.nonzero(right_mask)[0].astype(np.int32)
                    admitted[left_slice].append(
                        np.int32(right_block * _INTERACTION_TILE_SIZE) + slots
                    )
        for left_slice, entries in enumerate(admitted):
            if not entries:
                continue
            row = np.concatenate(entries)
            padded_size = (
                (row.size + _INTERACTION_TILE_SIZE - 1) // _INTERACTION_TILE_SIZE
            ) * _INTERACTION_TILE_SIZE
            padded = np.full((padded_size,), sentinel, dtype=np.int32)
            padded[: row.size] = row
            tiles = padded.reshape((-1, _INTERACTION_TILE_SIZE))
            ordinary_left.extend([int(left_block)] * tiles.shape[0])
            ordinary_slice.extend([left_slice] * tiles.shape[0])
            ordinary_right.append(tiles)

    ordinary_left_array = np.asarray(ordinary_left, dtype=np.int32)
    ordinary_slice_array = np.asarray(ordinary_slice, dtype=np.int32)
    ordinary_right_array = (
        np.concatenate(ordinary_right, axis=0)
        if ordinary_right
        else np.empty((0, _INTERACTION_TILE_SIZE), dtype=np.int32)
    )
    ordinary_group_starts, ordinary_group_counts = _group_ordinary_tiles(
        ordinary_left_array,
        ordinary_slice_array,
        ordinary_tiles_per_group,
    )
    return _InteractionSchedule32(
        atom_count=atom_count,
        search_radius=float(search_radius),
        left_slice_size=left_slice_size,
        box_lengths=box.astype(np.float32),
        atom_order=block_atoms.reshape(-1),
        inverse_order=inverse_order,
        ordinary_left_blocks=ordinary_left_array,
        ordinary_left_slices=ordinary_slice_array,
        ordinary_right_atoms=ordinary_right_array,
        ordinary_group_starts=ordinary_group_starts,
        ordinary_group_counts=ordinary_group_counts,
        special_blocks=special_blocks,
        special_lj_enabled=special_lj_enabled,
        special_lj_one_four=special_lj_one_four,
        special_work_left_blocks=special_work_left_blocks,
        special_work_left_slices=special_work_left_slices,
        special_work_right_atoms=special_work_right_atoms,
        special_work_lj_enabled=special_work_lj_enabled,
        special_work_lj_one_four=special_work_lj_one_four,
        special_work_diagonal=special_work_diagonal,
    )


def _interaction32_direct_force_only(
    positions: mx.array,
    schedule: _InteractionSchedule32 | _DeviceInteractionSchedule32,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    _return_stages: bool = False,
    _canonical_records: bool = True,
    _simdgroups_per_threadgroup: int = 4,
) -> mx.array | _Interaction32ForceStages:
    if isinstance(schedule, _InteractionSchedule32):
        schedule = _schedule_to_device32(schedule)
    positions = as_mx_array(positions, dtype=mx.float32)
    if positions.shape != (schedule.atom_count, 3):
        raise ValueError(f"positions must have shape ({schedule.atom_count}, 3) for the schedule")
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    parameter_shape = (schedule.atom_count,)
    if (
        half_sigma.shape != parameter_shape
        or sqrt_epsilon.shape != parameter_shape
        or charges.shape != parameter_shape
    ):
        raise ValueError("prepared nonbonded parameters must match the atom count")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if cutoff > schedule.search_radius:
        raise ValueError("cutoff cannot exceed the schedule search radius")
    return _interaction32_pme_direct_force_only(
        positions,
        schedule.atom_order,
        schedule.ordinary_left_blocks,
        schedule.ordinary_left_slices,
        schedule.ordinary_right_atoms,
        schedule.ordinary_group_starts,
        schedule.ordinary_group_counts,
        schedule.special_work_left_blocks,
        schedule.special_work_left_slices,
        schedule.special_work_right_atoms,
        schedule.special_work_lj_enabled,
        schedule.special_work_lj_one_four,
        schedule.special_work_diagonal,
        box,
        half_sigma,
        sqrt_epsilon,
        charges,
        cutoff=cutoff,
        shift=shift,
        switch_distance=switch_distance,
        one_four_scale=one_four_scale,
        coulomb_constant=coulomb_constant,
        alpha=alpha,
        _return_stages=_return_stages,
        _canonical_records=_canonical_records,
        _simdgroups_per_threadgroup=_simdgroups_per_threadgroup,
        _left_slice_size=schedule.left_slice_size,
    )


def _fused_half32_direct_force_only(
    positions: mx.array,
    schedule: _FusedHalfSchedule32 | _DeviceFusedHalfSchedule32,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    atom_type_ids: mx.array | None = None,
    nbfix_type_sigma: mx.array | None = None,
    nbfix_type_epsilon: mx.array | None = None,
    nbfix_type_count: int = 0,
    expected_generation: int | None = None,
    _simdgroups_per_threadgroup: int = 4,
) -> mx.array:
    if isinstance(schedule, _FusedHalfSchedule32):
        schedule = _fused_half_schedule_to_device32(schedule)
    if expected_generation is not None and (
        schedule.generation is None
        or schedule.generation.value != expected_generation
    ):
        raise ValueError("interaction schedule generation does not match the force binding")
    positions = as_mx_array(positions, dtype=mx.float32)
    if positions.shape != (schedule.atom_count, 3):
        raise ValueError(f"positions must have shape ({schedule.atom_count}, 3) for the schedule")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if cutoff > schedule.search_radius:
        raise ValueError("cutoff cannot exceed the schedule search radius")
    return _interaction32_fused_half_pme_direct_force_only(
        positions,
        schedule.atom_order,
        schedule.ordinary_left_blocks,
        schedule.ordinary_right_atoms,
        schedule.ordinary_half_modes,
        schedule.ordinary_group_starts,
        schedule.ordinary_group_counts,
        schedule.special_work_left_blocks,
        schedule.special_work_left_slices,
        schedule.special_work_right_atoms,
        schedule.special_work_lj_enabled,
        schedule.special_work_lj_one_four,
        schedule.special_work_diagonal,
        box_lengths_and_inverses,
        half_sigma,
        sqrt_epsilon,
        charges,
        cutoff=cutoff,
        shift=shift,
        switch_distance=switch_distance,
        one_four_scale=one_four_scale,
        coulomb_constant=coulomb_constant,
        alpha=alpha,
        atom_type_ids=atom_type_ids,
        nbfix_type_sigma=nbfix_type_sigma,
        nbfix_type_epsilon=nbfix_type_epsilon,
        nbfix_type_count=nbfix_type_count,
        _simdgroups_per_threadgroup=_simdgroups_per_threadgroup,
    )


def _owner_compute32_direct_force_only(
    positions: mx.array,
    schedule: _OwnerComputeSchedule32 | _DeviceOwnerComputeSchedule32,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    _simdgroups_per_threadgroup: int = 4,
) -> mx.array:
    if isinstance(schedule, _OwnerComputeSchedule32):
        schedule = _owner_schedule_to_device32(schedule)
    positions = as_mx_array(positions, dtype=mx.float32)
    if positions.shape != (schedule.atom_count, 3):
        raise ValueError(f"positions must have shape ({schedule.atom_count}, 3) for the schedule")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if cutoff > schedule.search_radius:
        raise ValueError("cutoff cannot exceed the schedule search radius")
    return _owner_compute32_pme_direct_force_only(
        positions,
        schedule.atom_order,
        schedule.owner_offsets,
        schedule.right_atoms,
        schedule.topology_offsets,
        schedule.topology_neighbors,
        schedule.topology_classes,
        box_lengths_and_inverses,
        half_sigma,
        sqrt_epsilon,
        charges,
        cutoff=cutoff,
        shift=shift,
        switch_distance=switch_distance,
        one_four_scale=one_four_scale,
        coulomb_constant=coulomb_constant,
        alpha=alpha,
        _simdgroups_per_threadgroup=_simdgroups_per_threadgroup,
    )
