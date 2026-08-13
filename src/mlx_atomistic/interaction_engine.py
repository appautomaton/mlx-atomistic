"""Experimental 32-atom interaction schedules for the Metal force prototype."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import mlx.core as mx
import numpy as np
from scipy.spatial import cKDTree

from mlx_atomistic.core import as_mx_array
from mlx_atomistic.metal_kernels import (
    _interaction32_fused_half_pme_direct_force_only,
    _interaction32_pme_direct_force_only,
    _Interaction32ForceStages,
    _owner_compute32_pme_direct_force_only,
)

_INTERACTION_TILE_SIZE = 32
_DEFAULT_ORDINARY_TILES_PER_GROUP = 3


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
    _simdgroups_per_threadgroup: int = 4,
) -> mx.array:
    if isinstance(schedule, _FusedHalfSchedule32):
        schedule = _fused_half_schedule_to_device32(schedule)
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
