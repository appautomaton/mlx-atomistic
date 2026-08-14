"""Neighbor-list construction for periodic MD."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import product
from time import perf_counter
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.cell_list import (
    PairListStats,
    build_periodic_cell_list,
    build_periodic_pair_list,
    estimate_pair_list_bytes,
)
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.metal_kernels import (
    _TILE_PME_COLUMN_DESCRIPTOR_INDEX_MASK,
    _TILE_PME_COLUMN_DESCRIPTOR_MEMBER_DIVISOR,
    _neighbor_cell_pair_candidates,
    _neighbor_cell_tile_candidates,
    _neighbor_pair_ordered_scatter_sized,
    _neighbor_tile_column_scatter_sized,
    _neighbor_tile_force_groups_sized,
    _neighbor_tile_member_counts,
    _neighbor_tile_member_pairs_sized,
    _neighbor_tile_membership,
    _neighbor_tile_ordered_scatter_sized,
    neighbor_pair_cutoff_mask,
    neighbor_pair_ordered_scatter,
)

NeighborBackend = Literal[
    "auto",
    "periodic_cell_list",
    "mlx_dense_pairs",
    "mlx_cell_pairs",
    "mlx_cell_blocks",
    "mlx_cell_tiles",
]
NeighborCheckBackend = Literal["numpy", "mlx_scalar"]
_ALLOWED_NEIGHBOR_BACKENDS = {
    "auto",
    "periodic_cell_list",
    "mlx_dense_pairs",
    "mlx_cell_pairs",
    "mlx_cell_blocks",
    "mlx_cell_tiles",
}
_ALLOWED_NEIGHBOR_CHECK_BACKENDS = {"numpy", "mlx_scalar"}
DEFAULT_MLX_DENSE_PAIR_LIMIT = 4096
DEFAULT_MLX_CELL_PAIR_CANDIDATE_CHUNK = 1_000_000
DEFAULT_MLX_SPATIAL_CANDIDATE_BATCH = 16_000_000
DEFAULT_MLX_SPATIAL_CELL_SUBDIVISION = 3
DEFAULT_MLX_SPATIAL_MAX_CELLS_PER_ATOM = 4
DEFAULT_MLX_CELL_BLOCK_SIZE = 256
DEFAULT_MLX_CELL_TILE_BLOCK_SIZE = 4
DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE = 8
DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE = 32
DEFAULT_MLX_CELL_TILE_BATCH = 65_536
_MLX_MD_CACHE_LIMIT_BYTES = 4_000_000_000
# Default neighbor backend for systems above the dense-pair limit. Measured on
# M5 Max (4k/16k/50k LJ, 2026-06-18): compacting candidates to real pairs
# ("mlx_cell_pairs") beats the fixed-shape padded-block path ("mlx_cell_blocks")
# by 5.9-7.1x in a managed loop (incl. rebuild) at identical physics (dE=0).
# Compact real-pair lists avoid evaluating the much larger padded candidate
# inventory. Metal uses deterministic prefix-scan compaction while CPU retains
# the NumPy fallback. Blocks remain available as an explicit backend.
DEFAULT_LARGE_SYSTEM_NEIGHBOR_BACKEND: NeighborBackend = "mlx_cell_pairs"
_BOOL_BYTES = np.dtype(np.bool_).itemsize
_FLOAT_BYTES = np.dtype(np.float32).itemsize
_INT_BYTES = np.dtype(np.int32).itemsize
_MLX_MD_CACHE_POLICY_ACTIVE: ContextVar[bool] = ContextVar(
    "mlx_atomistic_mlx_md_cache_policy_active",
    default=False,
)
_MLX_CELL_TILE_REBUILD_PROFILE_STAGES = (
    "validation_and_grid",
    "cell_assignment_sort",
    "host_task_inventory",
    "candidate_membership",
    "tile_inventory_readback",
    "exact_tile_scatter_and_order",
    "force_inventory_readback",
    "force_schedule_scatter",
    "result_assembly",
)


@dataclass
class _MLXCellTileRebuildProfiler:
    """Collect synchronized stage timings for the Metal tile-list builder."""

    stage_samples: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in _MLX_CELL_TILE_REBUILD_PROFILE_STAGES}
    )
    build_samples: list[float] = field(default_factory=list)
    inventory_samples: list[dict[str, int]] = field(default_factory=list)

    def record_stage(self, name: str, elapsed_seconds: float) -> None:
        """Record one synchronized builder-stage sample."""

        if name not in self.stage_samples:
            msg = f"unknown MLX cell-tile rebuild stage {name!r}"
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
            name: _timing_sample_summary(
                self.stage_samples[name],
                total_build_seconds=build_total,
            )
            for name in _MLX_CELL_TILE_REBUILD_PROFILE_STAGES
        }
        accounted = float(sum(float(stage["total_seconds"]) for stage in stages.values()))
        unattributed = build_total - accounted
        tolerance = max(1.0e-6, build_total * 0.05)
        return {
            "schema": "mlx_atomistic.mlx_cell_tile_rebuild_profile.v1",
            "mode": "sequential_synchronized_builder_stages",
            "backend": "mlx_cell_tiles",
            "stage_order": list(_MLX_CELL_TILE_REBUILD_PROFILE_STAGES),
            "rebuild_count": len(self.build_samples),
            "profiled_rebuild_wall_seconds": build_total,
            "accounted_stage_seconds": accounted,
            "unattributed_seconds": unattributed,
            "reconciled": -tolerance <= unattributed <= tolerance,
            "build_samples_seconds": list(self.build_samples),
            "stages": stages,
            "inventories": _inventory_sample_summary(self.inventory_samples),
        }


_ACTIVE_MLX_CELL_TILE_REBUILD_PROFILER: ContextVar[_MLXCellTileRebuildProfiler | None] = ContextVar(
    "mlx_atomistic_active_mlx_cell_tile_rebuild_profiler",
    default=None,
)


@contextmanager
def _profile_mlx_cell_tile_rebuilds() -> Iterator[_MLXCellTileRebuildProfiler]:
    """Collect Metal tile-list stage timings inside the active context."""

    if _ACTIVE_MLX_CELL_TILE_REBUILD_PROFILER.get() is not None:
        msg = "MLX cell-tile rebuild profiling contexts cannot be nested"
        raise RuntimeError(msg)
    profiler = _MLXCellTileRebuildProfiler()
    token = _ACTIVE_MLX_CELL_TILE_REBUILD_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_MLX_CELL_TILE_REBUILD_PROFILER.reset(token)


def _timing_sample_summary(
    samples: list[float],
    *,
    total_build_seconds: float,
) -> dict[str, object]:
    """Summarize one rebuild-stage sample vector."""

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


def _inventory_sample_summary(
    samples: list[dict[str, int]],
) -> dict[str, dict[str, int | float] | int]:
    """Summarize integer interaction inventories across profiled rebuilds."""

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


@contextmanager
def _bounded_metal_md_cache() -> Iterator[None]:
    """Bound reusable Metal buffers for one MD execution and restore policy."""

    if not _uses_metal_device():
        yield
        return
    previous_limit = mx.set_cache_limit(_MLX_MD_CACHE_LIMIT_BYTES)
    token = _MLX_MD_CACHE_POLICY_ACTIVE.set(True)
    try:
        yield
    finally:
        _MLX_MD_CACHE_POLICY_ACTIVE.reset(token)
        mx.set_cache_limit(previous_limit)


@dataclass(frozen=True)
class _CandidateEmissionStats:
    candidate_count: int
    peak_candidate_count: int

    @property
    def estimated_candidate_bytes(self) -> int:
        return self.peak_candidate_count * (3 * _FLOAT_BYTES + _BOOL_BYTES)


def validate_neighbor_backend(backend: str) -> NeighborBackend:
    """Validate and normalize a neighbor-list construction backend."""

    if backend not in _ALLOWED_NEIGHBOR_BACKENDS:
        expected = sorted(_ALLOWED_NEIGHBOR_BACKENDS)
        msg = f"unknown neighbor backend {backend!r}; expected one of {expected}"
        raise ValueError(msg)
    return backend  # type: ignore[return-value]


def validate_neighbor_check_backend(backend: str) -> NeighborCheckBackend:
    """Validate and normalize a neighbor-list displacement check backend."""

    if backend not in _ALLOWED_NEIGHBOR_CHECK_BACKENDS:
        expected = sorted(_ALLOWED_NEIGHBOR_CHECK_BACKENDS)
        msg = f"unknown neighbor check backend {backend!r}; expected one of {expected}"
        raise ValueError(msg)
    return backend  # type: ignore[return-value]


@dataclass(frozen=True)
class NeighborBlocks:
    """Fixed-shape candidate pair blocks for MLX-side cutoff filtering."""

    left: mx.array
    right: mx.array
    valid_mask: mx.array
    block_size: int
    candidate_count: int
    compact_pair_count: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            msg = "block_size must be positive"
            raise ValueError(msg)
        if self.left.shape != self.right.shape or self.left.shape != self.valid_mask.shape:
            msg = "left, right, and valid_mask must have matching shapes"
            raise ValueError(msg)
        if self.left.ndim != 2 or self.left.shape[1] != self.block_size:
            msg = "neighbor blocks must have shape (n_blocks, block_size)"
            raise ValueError(msg)
        if self.candidate_count < 0 or self.candidate_count > self.padded_candidate_count:
            msg = "candidate_count must fit within padded block storage"
            raise ValueError(msg)
        if self.compact_pair_count < 0 or self.compact_pair_count > self.candidate_count:
            msg = "compact_pair_count must fit within candidate_count"
            raise ValueError(msg)

    @property
    def block_count(self) -> int:
        """Number of fixed-size candidate blocks."""

        return int(self.left.shape[0])

    @property
    def padded_candidate_count(self) -> int:
        """Number of candidate slots including padding."""

        return int(self.left.size)

    @property
    def estimated_bytes(self) -> int:
        """Estimated storage bytes for block indices and validity mask."""

        return self.padded_candidate_count * (2 * _INT_BYTES + _BOOL_BYTES)

    @property
    def candidate_waste_count(self) -> int:
        """Number of emitted block candidates outside the neighbor radius."""

        return self.candidate_count - self.compact_pair_count


@dataclass(frozen=True)
class NeighborTiles:
    """Exact Verlet membership encoded over fixed four-atom block tiles.

    The representation is geometry-only. Each bit in ``member_mask`` records
    one pair that was inside ``cutoff + skin`` at rebuild time. Empty tiles are
    omitted. Each force-column descriptor packs its four membership bits above
    the tile and right-column index so the recurring Metal force kernel does
    not reload the tile membership word. Materializing explicit pairs remains
    an opt-in diagnostic action.
    """

    atom_blocks: mx.array
    tile_blocks: mx.array
    member_mask: mx.array
    exact_pair_count: int
    raw_candidate_count: int
    force_columns: mx.array | None = None
    force_group_starts: mx.array | None = None
    force_group_counts: mx.array | None = None
    generation: int = 0
    block_size: int = DEFAULT_MLX_CELL_TILE_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.block_size != DEFAULT_MLX_CELL_TILE_BLOCK_SIZE:
            msg = (
                "NeighborTiles require the fixed atom block size "
                f"{DEFAULT_MLX_CELL_TILE_BLOCK_SIZE}"
            )
            raise ValueError(msg)
        if self.atom_blocks.ndim != 2 or self.atom_blocks.shape[1] != self.block_size:
            msg = "atom_blocks must have shape (n_blocks, block_size)"
            raise ValueError(msg)
        if self.tile_blocks.ndim != 2 or self.tile_blocks.shape[1] != 2:
            msg = "tile_blocks must have shape (n_tiles, 2)"
            raise ValueError(msg)
        if self.member_mask.shape != (self.tile_count, self.mask_word_count):
            msg = "member_mask must contain one bit per tile lane"
            raise ValueError(msg)
        if self.atom_blocks.dtype != mx.int32 or self.tile_blocks.dtype != mx.int32:
            msg = "atom_blocks and tile_blocks must use int32 indices"
            raise ValueError(msg)
        if self.member_mask.dtype != mx.uint32:
            msg = "member_mask must use uint32 words"
            raise ValueError(msg)
        schedule = (
            self.force_columns,
            self.force_group_starts,
            self.force_group_counts,
        )
        if any(value is None for value in schedule) and not all(
            value is None for value in schedule
        ):
            msg = "force columns, group starts, and group counts must all be present or absent"
            raise ValueError(msg)
        if (
            self.force_columns is not None
            and self.force_group_starts is not None
            and self.force_group_counts is not None
        ):
            if self.force_columns.ndim != 1 or self.force_columns.dtype != mx.int32:
                msg = "force columns must be an int32 vector"
                raise ValueError(msg)
            if self.force_group_starts.ndim != 1 or (
                self.force_group_counts.shape != self.force_group_starts.shape
            ):
                msg = "force group starts and counts must have matching vector shapes"
                raise ValueError(msg)
            if (
                self.force_group_starts.dtype != mx.int32
                or self.force_group_counts.dtype != mx.int32
            ):
                msg = "force group starts and counts must use int32"
                raise ValueError(msg)
            column_count = int(self.force_columns.shape[0])
            group_count = int(self.force_group_starts.shape[0])
            if self.tile_count == 0:
                if column_count or group_count:
                    msg = "empty tile geometry cannot contain force columns or groups"
                    raise ValueError(msg)
            elif column_count == 0 or group_count == 0:
                msg = "non-empty tile geometry requires force columns and groups"
                raise ValueError(msg)
            else:
                starts = self.force_group_starts
                counts = self.force_group_counts
                ends = starts + counts
                safe_starts = mx.clip(starts, 0, column_count - 1)
                safe_ends = mx.clip(ends - 1, 0, column_count - 1)
                descriptor_bits = self.force_columns.astype(mx.uint32)
                first_descriptors = descriptor_bits[safe_starts]
                last_descriptors = descriptor_bits[safe_ends]
                descriptor_index_mask = np.uint32(_TILE_PME_COLUMN_DESCRIPTOR_INDEX_MASK)
                member_divisor = np.uint32(_TILE_PME_COLUMN_DESCRIPTOR_MEMBER_DIVISOR)
                first_columns = first_descriptors & descriptor_index_mask
                last_columns = last_descriptors & descriptor_index_mask
                first_tiles = first_columns // self.block_size
                last_tiles = last_columns // self.block_size
                first_members = first_descriptors // member_divisor
                last_members = last_descriptors // member_divisor
                safe_first_tiles = mx.clip(first_tiles, 0, self.tile_count - 1)
                safe_last_tiles = mx.clip(last_tiles, 0, self.tile_count - 1)
                valid = (
                    (starts >= 0)
                    & (counts >= 1)
                    & (counts <= DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE)
                    & (ends <= column_count)
                    & (last_columns < self.tile_count * self.block_size)
                    & (first_members >= 1)
                    & (last_members >= 1)
                    & (
                        self.tile_blocks[safe_first_tiles, 0]
                        == self.tile_blocks[safe_last_tiles, 0]
                    )
                )
                schedule_valid = (
                    mx.all(valid)
                    & (starts[0] == 0)
                    & (ends[-1] == column_count)
                    & mx.all(starts[1:] == ends[:-1])
                )
                mx.eval(schedule_valid)
                if not bool(np.asarray(schedule_valid)):
                    msg = (
                        "force groups must cover active columns contiguously in same-left "
                        f"groups of at most {DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE}"
                    )
                    raise ValueError(msg)
        if self.exact_pair_count < 0 or self.exact_pair_count > self.padded_lane_count:
            msg = "exact_pair_count must fit within padded tile lanes"
            raise ValueError(msg)
        if self.raw_candidate_count < self.exact_pair_count:
            msg = "raw_candidate_count must include every exact Verlet pair"
            raise ValueError(msg)
        if self.generation < 0:
            msg = "generation must be non-negative"
            raise ValueError(msg)

    @property
    def block_count(self) -> int:
        """Number of fixed-width atom blocks."""

        return int(self.atom_blocks.shape[0])

    @property
    def tile_count(self) -> int:
        """Number of retained non-empty block-pair tiles."""

        return int(self.tile_blocks.shape[0])

    @property
    def lanes_per_tile(self) -> int:
        """Number of atom-pair lanes in one padded tile."""

        return self.block_size * self.block_size

    @property
    def mask_word_count(self) -> int:
        """Number of 32-bit membership words stored for each tile."""

        return (self.lanes_per_tile + 31) // 32

    @property
    def padded_lane_count(self) -> int:
        """Number of scheduled tile lanes including inactive padding."""

        return self.tile_count * self.lanes_per_tile

    @property
    def force_group_count(self) -> int:
        """Number of same-left-block column groups in the direct-force schedule."""

        if self.force_group_starts is None:
            return 0
        return int(self.force_group_starts.shape[0])

    @property
    def active_column_count(self) -> int:
        """Number of non-empty right-atom columns in the direct-force schedule."""

        if self.force_columns is None:
            return 0
        return int(self.force_columns.shape[0])

    @property
    def scheduled_column_count(self) -> int:
        """Number of dispatched SIMD column lanes including group-tail padding."""

        return self.force_group_count * DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE

    @property
    def padding_waste_count(self) -> int:
        """Number of scheduled lanes that are not exact Verlet members."""

        return self.padded_lane_count - self.exact_pair_count

    @property
    def padding_waste_fraction(self) -> float:
        """Fraction of scheduled tile lanes outside exact Verlet membership."""

        if self.padded_lane_count == 0:
            return 0.0
        return self.padding_waste_count / self.padded_lane_count

    @property
    def estimated_bytes(self) -> int:
        """Estimated persistent bytes for block, tile, and membership arrays."""

        schedule_size = 0
        if (
            self.force_columns is not None
            and self.force_group_starts is not None
            and self.force_group_counts is not None
        ):
            schedule_size = (
                self.force_columns.size
                + self.force_group_starts.size
                + self.force_group_counts.size
            )
        return int(
            (self.atom_blocks.size + self.tile_blocks.size + self.member_mask.size + schedule_size)
            * _INT_BYTES
        )

    def materialize_pairs(self, *, sort: bool = True) -> mx.array:
        """Decode exact Verlet members for diagnostics and tests.

        Args:
            sort: Whether to return canonical lexicographic pair order.

        Returns:
            Exact unique atom pairs with shape ``(exact_pair_count, 2)``.
        """

        if _uses_metal_device():
            member_counts = _neighbor_tile_member_counts(self.member_mask)
            member_prefix = mx.cumsum(member_counts)
            pairs = _neighbor_tile_member_pairs_sized(
                self.atom_blocks,
                self.tile_blocks,
                self.member_mask,
                member_counts,
                member_prefix,
                accepted_count=self.exact_pair_count,
            )
            if sort and self.exact_pair_count:
                right_order = mx.argsort(pairs[:, 1])
                pairs = pairs[right_order]
                left_order = mx.argsort(pairs[:, 0])
                pairs = pairs[left_order]
            return pairs

        atom_blocks = np.asarray(self.atom_blocks, dtype=np.int32)
        tile_blocks = np.asarray(self.tile_blocks, dtype=np.int32)
        member_mask = np.asarray(self.member_mask, dtype=np.uint32)
        pairs = np.empty((self.exact_pair_count, 2), dtype=np.int32)
        output = 0
        for tile_index, (left_block, right_block) in enumerate(tile_blocks):
            for lane in self._active_lanes(member_mask[tile_index]):
                left_slot, right_slot = divmod(lane, self.block_size)
                left_atom = int(atom_blocks[left_block, left_slot])
                right_atom = int(atom_blocks[right_block, right_slot])
                pairs[output, 0] = min(left_atom, right_atom)
                pairs[output, 1] = max(left_atom, right_atom)
                output += 1
        if sort and pairs.shape[0]:
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs = pairs[order]
        return mx.array(pairs, dtype=mx.int32)

    @staticmethod
    def _active_lanes(mask_words: np.ndarray):
        for word_index, raw_word in enumerate(mask_words):
            word = int(raw_word)
            while word:
                bit = (word & -word).bit_length() - 1
                yield word_index * 32 + bit
                word &= word - 1


@dataclass(frozen=True, init=False)
class NeighborList:
    """Neighbor interactions for pairwise potentials."""

    _pairs: mx.array | None = field(repr=False)
    cutoff: float
    skin: float = 0.0
    stats: PairListStats | None = None
    blocks: NeighborBlocks | None = None
    tiles: NeighborTiles | None = None
    _sort_diagnostic_pairs: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        pairs: mx.array | None,
        cutoff: float,
        skin: float = 0.0,
        stats: PairListStats | None = None,
        blocks: NeighborBlocks | None = None,
        tiles: NeighborTiles | None = None,
        *,
        sort_diagnostic_pairs: bool = False,
    ) -> None:
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "skin", skin)
        object.__setattr__(self, "stats", stats)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "tiles", tiles)
        object.__setattr__(self, "_sort_diagnostic_pairs", sort_diagnostic_pairs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.blocks is not None and self.tiles is not None:
            msg = "a neighbor list may contain blocks or tiles, not both"
            raise ValueError(msg)
        if self._pairs is None and self.tiles is None:
            msg = "pairs may be deferred only when exact tiles are present"
            raise ValueError(msg)
        if self._pairs is not None and (self._pairs.ndim != 2 or self._pairs.shape[1] != 2):
            msg = "pairs must have shape (n_pairs, 2)"
            raise ValueError(msg)
        if self._pairs is not None and self._pairs.dtype != mx.int32:
            msg = "pairs must use int32 indices"
            raise ValueError(msg)
        if self.blocks is not None and self.blocks.compact_pair_count != self._pairs.shape[0]:
            msg = "block compact_pair_count must match the diagnostic pairs"
            raise ValueError(msg)
        if (
            self.tiles is not None
            and self._pairs is not None
            and self.tiles.exact_pair_count != self._pairs.shape[0]
        ):
            msg = "tile exact_pair_count must match the diagnostic pairs"
            raise ValueError(msg)

    @property
    def pairs(self) -> mx.array:
        """Return compact pairs, materializing deferred tile diagnostics once."""

        return self.diagnostic_pairs

    @property
    def pair_count(self) -> int:
        """Number of unique pairs or candidate block entries."""

        if self.blocks is not None:
            return self.blocks.candidate_count
        if self.tiles is not None:
            return self.tiles.exact_pair_count
        return self._pairs.shape[0]

    @property
    def compact_pair_count(self) -> int:
        """Number of compact pairs accepted by the neighbor search radius."""

        if self.tiles is not None:
            return self.tiles.exact_pair_count
        return int(self._pairs.shape[0])

    @property
    def materialized_diagnostic_pairs(self) -> mx.array | None:
        """Return resident diagnostic pairs without triggering materialization."""

        return self._pairs

    @property
    def diagnostic_pairs_materialized(self) -> bool:
        """Whether the optional exact-pair diagnostic currently occupies memory."""

        return self._pairs is not None

    @property
    def diagnostic_pairs(self) -> mx.array:
        """Return the compact cutoff-plus-skin pairs for this generation."""

        pairs = self._pairs
        if pairs is None:
            if self.tiles is None:
                msg = "deferred diagnostic pairs require exact tiles"
                raise RuntimeError(msg)
            pairs = self.tiles.materialize_pairs(sort=self._sort_diagnostic_pairs)
            object.__setattr__(self, "_pairs", pairs)
        return pairs

    def force_candidates(
        self,
        *,
        prefer_tiles: bool,
    ) -> mx.array | NeighborTiles:
        """Select compact pairs or exact tiles for force binding.

        Args:
            prefer_tiles: Return exact tiles when they are present. Otherwise,
                return the compact pair representation.

        Returns:
            Exact tiles only when requested and available, or compact pairs.
        """

        if prefer_tiles and self.tiles is not None:
            return self.tiles
        return self.diagnostic_pairs

    @property
    def interactions(self) -> mx.array | NeighborBlocks | NeighborTiles:
        """Return the active force-evaluation representation."""

        # Keep the compatibility surface pair-oriented for exact tiles until a
        # production tile force route is selected explicitly.
        if self.tiles is not None:
            return self.diagnostic_pairs
        return self.blocks if self.blocks is not None else self.diagnostic_pairs

    @property
    def backend(self) -> str:
        """Pair-construction backend name."""

        return "periodic_cell_list" if self.stats is None else self.stats.backend

    @property
    def estimated_pair_bytes(self) -> int:
        """Estimated bytes for the compact int32 pair array."""

        if self.blocks is not None:
            return self.blocks.estimated_bytes + estimate_pair_list_bytes(
                self.compact_pair_count,
            )
        if self.tiles is not None:
            diagnostic_bytes = (
                0 if self._pairs is None else estimate_pair_list_bytes(self.compact_pair_count)
            )
            return self.tiles.estimated_bytes + diagnostic_bytes
        if self.stats is None:
            return int(self.pair_count) * 2 * np.dtype(np.int32).itemsize
        return self.stats.estimated_pair_bytes

    @property
    def estimated_cell_list_bytes(self) -> int:
        """Estimated bytes for cell-list construction arrays."""

        return 0 if self.stats is None else self.stats.estimated_cell_list_bytes

    @property
    def representation_kind(self) -> str:
        """Neighbor interaction representation shape."""

        return "pairs" if self.stats is None else self.stats.representation_kind

    @property
    def candidate_count(self) -> int | None:
        """Number of candidate interactions tested before cutoff filtering."""

        return None if self.stats is None else self.stats.candidate_count

    @property
    def candidate_waste_count(self) -> int | None:
        """Number of candidate interactions rejected by compaction/filtering."""

        if self.candidate_count is None:
            return None
        return max(0, int(self.candidate_count) - int(self.compact_pair_count))

    @property
    def candidate_waste_fraction(self) -> float | None:
        """Fraction of emitted candidates rejected by compaction/filtering."""

        if self.candidate_count is None:
            return None
        if self.candidate_count == 0:
            return 0.0
        return float(self.candidate_waste_count or 0) / float(self.candidate_count)

    @property
    def estimated_candidate_bytes(self) -> int:
        """Estimated bytes for backend candidate testing arrays."""

        return 0 if self.stats is None else self.stats.estimated_candidate_bytes

    @property
    def estimated_compact_pair_bytes(self) -> int:
        """Estimated bytes for compact int32 pairs accepted by the search radius."""

        if self.tiles is not None and self._pairs is None:
            return 0
        return estimate_pair_list_bytes(self.compact_pair_count)

    @property
    def compaction_backend(self) -> str | None:
        """Backend used to compact candidates into explicit pairs, if any."""

        return None if self.stats is None else self.stats.compaction_backend

    @property
    def fallback_reason(self) -> str | None:
        """Reason an accelerated representation fell back or used a hybrid step."""

        return None if self.stats is None else self.stats.fallback_reason


@dataclass
class NeighborListManager:
    """Manage Verlet neighbor-list rebuilds during an MD trajectory."""

    cell: Cell
    cutoff: float
    skin: float = 0.3
    check_interval: int = 1
    # Pair sorting is off by default in the MD loop: it is pure force-kernel
    # locality, not correctness, and MLX scatter-add is insensitive to pair order.
    # Measured on M5 Max (50k LJ, 2026-06-18): the per-rebuild np.lexsort of ~4.8M
    # pairs is ~700ms and dominates the rebuild (~77%); disabling it ~2x'd 50k NVT
    # throughput (68->134 steps/s) with energy identical to ULPs. Unsorted lists are
    # still fully deterministic (same positions -> same array). Set True to restore
    # canonical (i, j) ordering.
    sort_pairs: bool = False
    max_workers: int | None = None
    backend: NeighborBackend = "auto"
    max_mlx_dense_atoms: int = DEFAULT_MLX_DENSE_PAIR_LIMIT
    block_size: int = DEFAULT_MLX_CELL_BLOCK_SIZE
    displacement_check_backend: NeighborCheckBackend = "numpy"
    neighbor_list: NeighborList | None = None
    reference_positions: mx.array | None = None
    rebuild_count: int = 0
    last_max_displacement: float = 0.0
    updates_since_check: int = 0
    rebuild_wall_seconds: float = 0.0
    update_wall_seconds: float = 0.0
    _cache_clear_pending: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.check_interval <= 0:
            msg = "check_interval must be positive"
            raise ValueError(msg)
        self.backend = validate_neighbor_backend(self.backend)
        self.displacement_check_backend = validate_neighbor_check_backend(
            self.displacement_check_backend
        )
        if self.max_mlx_dense_atoms <= 0:
            msg = "max_mlx_dense_atoms must be positive"
            raise ValueError(msg)
        if self.block_size <= 0:
            msg = "block_size must be positive"
            raise ValueError(msg)

    @property
    def rebuild_threshold(self) -> float:
        """Maximum displacement before the Verlet list must be rebuilt."""

        return 0.5 * self.skin

    def needs_rebuild(self, positions) -> bool:
        """Return true when positions have moved too far from the reference frame."""

        if self.displacement_check_backend == "mlx_scalar":
            return self._needs_rebuild_mlx_scalar(positions)
        return self._needs_rebuild_numpy(positions)

    def _needs_rebuild_numpy(self, positions) -> bool:
        """Return true using the legacy NumPy displacement check."""

        if self.neighbor_list is None or self.reference_positions is None:
            positions_np = np.asarray(positions, dtype=np.float32)
            if positions_np.ndim != 2 or positions_np.shape[1] != 3:
                msg = "positions must have shape (n_particles, 3)"
                raise ValueError(msg)
            if not np.all(np.isfinite(positions_np)):
                msg = "positions must be finite"
                raise ValueError(msg)
            self.last_max_displacement = float("inf")
            return True
        self.updates_since_check += 1
        if self.updates_since_check < self.check_interval:
            if isinstance(positions, np.ndarray):
                positions_np = np.asarray(positions, dtype=np.float32)
                if positions_np.ndim != 2 or positions_np.shape[1] != 3:
                    msg = "positions must have shape (n_particles, 3)"
                    raise ValueError(msg)
                if not np.all(np.isfinite(positions_np)):
                    msg = "positions must be finite"
                    raise ValueError(msg)
            return False
        self.updates_since_check = 0

        positions_np = np.asarray(positions, dtype=np.float32)
        if positions_np.ndim != 2 or positions_np.shape[1] != 3:
            msg = "positions must have shape (n_particles, 3)"
            raise ValueError(msg)
        if not np.all(np.isfinite(positions_np)):
            msg = "positions must be finite"
            raise ValueError(msg)
        reference_np = np.asarray(self.reference_positions, dtype=np.float32)
        if reference_np.shape != positions_np.shape:
            msg = "positions must match the neighbor-list reference shape"
            raise ValueError(msg)
        displacement = positions_np - reference_np
        displacement = np.asarray(self.cell.minimum_image(as_mx_array(displacement)))
        distance2 = np.sum(displacement * displacement, axis=1)
        self.last_max_displacement = float(np.sqrt(np.max(distance2))) if len(distance2) else 0.0
        return self.last_max_displacement > self.rebuild_threshold

    def _needs_rebuild_mlx_scalar(self, positions) -> bool:
        """Return true using MLX displacement reduction plus scalar materialization."""

        positions_mx = as_mx_array(positions)
        if positions_mx.ndim != 2 or positions_mx.shape[1] != 3:
            msg = "positions must have shape (n_particles, 3)"
            raise ValueError(msg)

        if self.neighbor_list is None or self.reference_positions is None:
            finite = mx.all(mx.isfinite(positions_mx))
            mx.eval(finite)
            if not bool(np.asarray(finite)):
                msg = "positions must be finite"
                raise ValueError(msg)
            self.last_max_displacement = float("inf")
            return True

        self.updates_since_check += 1
        if self.updates_since_check < self.check_interval:
            return False
        self.updates_since_check = 0

        reference = as_mx_array(self.reference_positions)
        if reference.shape != positions_mx.shape:
            msg = "positions must match the neighbor-list reference shape"
            raise ValueError(msg)
        finite = mx.all(mx.isfinite(positions_mx))
        if positions_mx.shape[0] == 0:
            max_displacement = mx.array(0.0, dtype=mx.float32)
        else:
            displacement = positions_mx - reference
            displacement = self.cell.minimum_image(displacement)
            distance2 = mx.sum(displacement * displacement, axis=1)
            max_displacement = mx.sqrt(mx.max(distance2))
        mx.eval(finite, max_displacement)
        if not bool(np.asarray(finite)):
            msg = "positions must be finite"
            raise ValueError(msg)
        self.last_max_displacement = float(np.asarray(max_displacement))
        return self.last_max_displacement > self.rebuild_threshold

    def rebuild(self, positions) -> NeighborList:
        """Force a neighbor-list rebuild from current positions."""

        self._release_pending_metal_cache()
        start = perf_counter()
        neighbor_list = build_neighbor_list(
            positions,
            self.cell,
            cutoff=self.cutoff,
            skin=self.skin,
            sort_pairs=self.sort_pairs,
            max_workers=self.max_workers,
            backend=self.backend,
            max_mlx_dense_atoms=self.max_mlx_dense_atoms,
            block_size=self.block_size,
            generation=self.rebuild_count + 1,
        )
        self.neighbor_list = neighbor_list
        self.rebuild_wall_seconds += perf_counter() - start
        self.reference_positions = as_mx_array(positions)
        self.rebuild_count += 1
        self.last_max_displacement = 0.0
        self.updates_since_check = 0
        self._cache_clear_pending = self.neighbor_list.compaction_backend in {
            "metal_spatial_prefix_scan",
            "metal_spatial_tile_prefix_scan",
        }
        return self.neighbor_list

    def update(self, positions) -> NeighborList:
        """Return a current neighbor list, rebuilding if needed."""

        start = perf_counter()
        try:
            self._release_pending_metal_cache()
            if self.needs_rebuild(positions):
                return self.rebuild(positions)
            if self.neighbor_list is None:
                msg = "neighbor list manager has no current neighbor list"
                raise RuntimeError(msg)
            return self.neighbor_list
        finally:
            self.update_wall_seconds += perf_counter() - start

    def _admit_block(
        self,
        max_displacement: mx.array,
        admissible: mx.array,
    ) -> bool:
        """Commit one device-measured integration block when its list stayed valid.

        Args:
            max_displacement: Maximum minimum-image displacement reached anywhere
                in the proposed block.
            admissible: Scalar boolean that is true only when every proposed
                position was finite and stayed within the rebuild threshold.

        Returns:
            ``True`` when the current neighbor list safely covers the whole
            block. ``False`` requests exact per-step replay from the block start.
        """

        start = perf_counter()
        try:
            mx.eval(max_displacement, admissible)
            if not bool(np.asarray(admissible)):
                return False
            displacement = float(np.asarray(max_displacement))
            if not np.isfinite(displacement):
                return False
            if displacement > self.rebuild_threshold:
                return False
            if self.neighbor_list is None or self.reference_positions is None:
                msg = "neighbor list manager has no current neighbor list"
                raise RuntimeError(msg)
            self.last_max_displacement = displacement
            self.updates_since_check = 0
            return True
        finally:
            self.update_wall_seconds += perf_counter() - start

    def _release_pending_metal_cache(self) -> None:
        """Release inactive rebuild temporaries at the next safe call boundary."""

        if not self._cache_clear_pending:
            return
        self._cache_clear_pending = False
        if _uses_metal_device() and not _MLX_MD_CACHE_POLICY_ACTIVE.get():
            mx.clear_cache()

    def build_cell_candidate(
        self,
        positions,
        cell: Cell,
    ) -> NeighborListManager:
        """Build an isolated neighbor state for a proposed periodic cell.

        Args:
            positions: Candidate particle positions.
            cell: Candidate periodic cell.

        Returns:
            A distinct manager with one neighbor list built for the candidate
                state. This manager does not mutate the current manager.
        """

        candidate = NeighborListManager(
            cell,
            cutoff=self.cutoff,
            skin=self.skin,
            check_interval=self.check_interval,
            sort_pairs=self.sort_pairs,
            max_workers=self.max_workers,
            backend=self.backend,
            max_mlx_dense_atoms=self.max_mlx_dense_atoms,
            block_size=self.block_size,
            displacement_check_backend=self.displacement_check_backend,
            rebuild_count=self.rebuild_count,
            rebuild_wall_seconds=self.rebuild_wall_seconds,
            update_wall_seconds=self.update_wall_seconds,
        )
        candidate.rebuild(positions)
        return candidate

    def commit_cell_candidate(
        self,
        candidate: NeighborListManager,
    ) -> None:
        """Replace the current cell-bound state with a compatible candidate.

        Args:
            candidate: Isolated candidate returned by
                `build_cell_candidate`.

        Raises:
            ValueError: If the candidate uses different neighbor-list policy.
        """

        policy = (
            "cutoff",
            "skin",
            "check_interval",
            "sort_pairs",
            "max_workers",
            "backend",
            "max_mlx_dense_atoms",
            "block_size",
            "displacement_check_backend",
        )
        mismatches = tuple(
            name for name in policy if getattr(self, name) != getattr(candidate, name)
        )
        if mismatches:
            msg = "neighbor candidate policy mismatch: " + ",".join(mismatches)
            raise ValueError(msg)
        if candidate.neighbor_list is None or candidate.reference_positions is None:
            msg = "neighbor candidate must contain a built neighbor list"
            raise ValueError(msg)
        self.cell = candidate.cell
        self.neighbor_list = candidate.neighbor_list
        self.reference_positions = candidate.reference_positions
        self.rebuild_count = candidate.rebuild_count
        self.last_max_displacement = candidate.last_max_displacement
        self.updates_since_check = candidate.updates_since_check
        self.rebuild_wall_seconds = candidate.rebuild_wall_seconds
        self.update_wall_seconds = candidate.update_wall_seconds
        self._cache_clear_pending = candidate._cache_clear_pending


def build_neighbor_list(
    positions,
    cell: Cell,
    *,
    cutoff: float,
    skin: float = 0.3,
    sort_pairs: bool = True,
    max_workers: int | None = None,
    backend: NeighborBackend = "periodic_cell_list",
    max_mlx_dense_atoms: int = DEFAULT_MLX_DENSE_PAIR_LIMIT,
    block_size: int = DEFAULT_MLX_CELL_BLOCK_SIZE,
    generation: int = 0,
) -> NeighborList:
    """Build a periodic cell-list neighbor list with unique `i < j` pairs.

    Args:
        positions: Particle coordinates.
        cell: Periodic cell owning the minimum-image convention.
        cutoff: Interaction cutoff distance.
        skin: Verlet-list skin added to the cutoff. Defaults to ``0.3``.
        sort_pairs: Order the compact pairs by left then right atom. Defaults
            to ``True``.
        max_workers: Worker count for the threaded CPU builder. Defaults to
            ``None``.
        backend: Neighbor construction backend. Defaults to
            ``"periodic_cell_list"``.
        max_mlx_dense_atoms: Largest atom count the dense MLX pair backend
            accepts. Defaults to `DEFAULT_MLX_DENSE_PAIR_LIMIT`.
        block_size: Atom-block width for block-structured backends. Defaults to
            `DEFAULT_MLX_CELL_BLOCK_SIZE`.
        generation: Rebuild generation stamped onto tile geometry, so a caller
            never has to rebuild the tiles through `dataclasses.replace` and
            repeat their schedule validation. Defaults to ``0``.

    Returns:
        The constructed neighbor list.
    """

    backend = validate_neighbor_backend(backend)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        msg = "cutoff must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(skin) or skin < 0.0:
        msg = "skin must be finite and non-negative"
        raise ValueError(msg)
    if block_size <= 0:
        msg = "block_size must be positive"
        raise ValueError(msg)

    positions_mx = as_mx_array(positions)
    if positions_mx.ndim != 2 or positions_mx.shape[1] != 3:
        msg = "positions must have shape (n_particles, 3)"
        raise ValueError(msg)
    search_radius = cutoff + skin
    fallback_reason = None
    if backend == "auto":
        if positions_mx.shape[0] <= max_mlx_dense_atoms:
            backend = "mlx_dense_pairs"
        else:
            backend = DEFAULT_LARGE_SYSTEM_NEIGHBOR_BACKEND
    if backend == "mlx_cell_pairs" and _uses_metal_device():
        return _build_mlx_spatial_cell_pair_list(
            positions_mx,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            sort_pairs=sort_pairs,
        )
    if backend == "mlx_cell_tiles" and _uses_metal_device():
        return _build_mlx_spatial_cell_tile_list(
            positions_mx,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            sort_pairs=sort_pairs,
            generation=generation,
        )

    positions_np = np.asarray(positions_mx, dtype=np.float32)
    if not np.all(np.isfinite(positions_np)):
        msg = "positions must be finite"
        raise ValueError(msg)
    if backend == "mlx_dense_pairs":
        return _build_mlx_dense_pair_list(
            positions_np,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            max_atoms=max_mlx_dense_atoms,
        )
    if backend == "mlx_cell_pairs":
        return _build_mlx_cell_pair_list_cpu(
            positions_np,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            sort_pairs=sort_pairs,
        )
    if backend == "mlx_cell_blocks":
        return _build_mlx_cell_blocks(
            positions_np,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            sort_pairs=sort_pairs,
            block_size=block_size,
        )
    if backend == "mlx_cell_tiles":
        return _build_mlx_cell_tiles(
            positions_np,
            cell,
            cutoff=cutoff,
            skin=skin,
            search_radius=search_radius,
            sort_pairs=sort_pairs,
            generation=generation,
        )
    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, backend)
    pair_array, stats = build_periodic_pair_list(
        positions_np,
        cell,
        search_radius=search_radius,
        sort_pairs=sort_pairs,
        max_workers=max_workers,
    )
    candidate_stats = _periodic_candidate_emission_stats(
        positions_np,
        cell,
        search_radius=search_radius,
    )
    stats = replace(
        stats,
        candidate_count=candidate_stats.candidate_count,
        estimated_candidate_bytes=candidate_stats.estimated_candidate_bytes,
        compaction_backend="cpu_distance_filter",
    )
    if fallback_reason is not None:
        stats = replace(stats, fallback_reason=fallback_reason)
    return NeighborList(
        mx.array(pair_array, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
    )


def _uses_metal_device() -> bool:
    return "gpu" in str(mx.default_device()).lower()


def _require_orthorhombic_cell_for_compact_neighbor_backend(
    cell: Cell,
    backend: str,
) -> None:
    if cell.is_orthorhombic:
        return
    msg = (
        f"{backend} neighbor backend currently supports orthorhombic cells only; "
        "triclinic cell matrices require mlx_dense_pairs or another minimum-image-safe path"
    )
    raise ValueError(msg)


def _periodic_candidate_emission_stats(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    search_radius: float,
) -> _CandidateEmissionStats:
    cell_list = build_periodic_cell_list(positions_np, cell, search_radius=search_radius)
    return _cell_list_candidate_emission_stats(cell_list.bins, cell_list.n_cells)


def _cell_list_candidate_emission_stats(
    bins: dict[tuple[int, int, int], np.ndarray],
    n_cells: tuple[int, int, int],
) -> _CandidateEmissionStats:
    offsets = tuple(product((-1, 0, 1), repeat=3))
    candidate_count = 0
    peak_candidate_count = 0

    for cell_index, members in tuple(bins.items()):
        neighbor_indices = sorted(
            {
                tuple((cell_index[axis] + offset[axis]) % n_cells[axis] for axis in range(3))
                for offset in offsets
            }
        )
        for neighbor_index in neighbor_indices:
            if neighbor_index < cell_index:
                continue
            neighbors = bins.get(neighbor_index)
            if neighbors is None:
                continue
            if neighbor_index == cell_index:
                emitted_count = members.shape[0] * max(members.shape[0] - 1, 0) // 2
            else:
                emitted_count = members.shape[0] * neighbors.shape[0]
            candidate_count += int(emitted_count)
            peak_candidate_count = max(peak_candidate_count, int(emitted_count))

    return _CandidateEmissionStats(
        candidate_count=candidate_count,
        peak_candidate_count=peak_candidate_count,
    )


def _build_mlx_dense_pair_list(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    max_atoms: int,
) -> NeighborList:
    if max_atoms <= 0:
        msg = "max_mlx_dense_atoms must be positive"
        raise ValueError(msg)
    n_atoms = int(positions_np.shape[0])
    if n_atoms > max_atoms:
        msg = (
            "mlx_dense_pairs is limited to small-system candidate checks: "
            f"n_atoms={n_atoms}, max_mlx_dense_atoms={max_atoms}; "
            "fallback_backend=periodic_cell_list; "
            f"fallback_reason=mlx_dense_pairs_atom_limit_exceeded:"
            f"n_atoms={n_atoms}:max_mlx_dense_atoms={max_atoms}"
        )
        raise ValueError(msg)
    lengths = np.asarray(cell.lengths, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)

    positions_mx = as_mx_array(positions_np)
    displacement = positions_mx[:, None, :] - positions_mx[None, :, :]
    displacement = cell.minimum_image(displacement)
    r2 = mx.sum(displacement * displacement, axis=-1)
    indices = mx.arange(n_atoms)
    pair_mask = (indices[:, None] < indices[None, :]) & (r2 < search_radius * search_radius)
    mx.eval(pair_mask)
    pair_array = np.argwhere(np.asarray(pair_mask)).astype(np.int32, copy=False)

    candidate_count = n_atoms * max(n_atoms - 1, 0) // 2
    stats = PairListStats(
        pair_count=int(pair_array.shape[0]),
        n_cells=(1, 1, 1),
        cell_count=1,
        occupied_cell_count=1 if n_atoms else 0,
        search_radius=search_radius,
        estimated_pair_bytes=estimate_pair_list_bytes(int(pair_array.shape[0])),
        estimated_cell_list_bytes=0,
        backend="mlx_dense_pairs",
        representation_kind="pairs",
        candidate_count=candidate_count,
        estimated_candidate_bytes=n_atoms * n_atoms * (3 * _FLOAT_BYTES + _BOOL_BYTES),
        compaction_backend="cpu_argwhere",
        fallback_reason="mlx_argwhere_or_nonzero_unavailable",
    )
    return NeighborList(
        mx.array(pair_array, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
    )


def _build_mlx_cell_blocks(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
    block_size: int,
) -> NeighborList:
    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, "mlx_cell_blocks")
    lengths = np.asarray(cell.lengths, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)

    cell_list = build_periodic_cell_list(positions_np, cell, search_radius=search_radius)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    pair_chunks: list[np.ndarray] = []
    candidate_count = 0

    for cell_index, members in tuple(cell_list.bins.items()):
        neighbor_indices = sorted(
            {
                tuple(
                    (cell_index[axis] + offset[axis]) % cell_list.n_cells[axis] for axis in range(3)
                )
                for offset in offsets
            }
        )
        for neighbor_index in neighbor_indices:
            if neighbor_index < cell_index:
                continue
            neighbors = cell_list.bins.get(neighbor_index)
            if neighbors is None:
                continue
            if neighbor_index == cell_index:
                left, right = _same_cell_member_pairs(members)
            else:
                left, right = _cross_cell_member_pairs(members, neighbors)
            if left.shape[0] == 0:
                continue
            candidate_count += int(left.shape[0])
            normalized = np.stack((np.minimum(left, right), np.maximum(left, right)), axis=1)
            pair_chunks.append(normalized.astype(np.int32, copy=False))

    if pair_chunks:
        candidate_pairs = np.concatenate(pair_chunks, axis=0).astype(np.int32, copy=False)
    else:
        candidate_pairs = np.empty((0, 2), dtype=np.int32)
    if sort_pairs and candidate_pairs.shape[0]:
        order = np.lexsort((candidate_pairs[:, 1], candidate_pairs[:, 0]))
        candidate_pairs = candidate_pairs[order]

    compact_pairs = _filter_candidate_pairs_within_radius(
        positions_np,
        cell,
        candidate_pairs,
        search_radius=search_radius,
    )
    blocks = _candidate_pairs_to_blocks(
        candidate_pairs,
        block_size=block_size,
        compact_pair_count=int(compact_pairs.shape[0]),
    )
    stats = PairListStats(
        pair_count=blocks.candidate_count,
        n_cells=cell_list.n_cells,
        cell_count=cell_list.cell_count,
        occupied_cell_count=cell_list.occupied_cell_count,
        search_radius=search_radius,
        estimated_pair_bytes=(
            blocks.estimated_bytes + estimate_pair_list_bytes(int(compact_pairs.shape[0]))
        ),
        estimated_cell_list_bytes=cell_list.estimated_bytes,
        backend="mlx_cell_blocks",
        representation_kind="blocks",
        candidate_count=candidate_count,
        estimated_candidate_bytes=blocks.estimated_bytes,
        compaction_backend=None,
        fallback_reason=None,
    )
    return NeighborList(
        mx.array(compact_pairs, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
        blocks=blocks,
    )


def _candidate_pairs_to_blocks(
    pairs: np.ndarray,
    *,
    block_size: int,
    compact_pair_count: int,
) -> NeighborBlocks:
    count = int(pairs.shape[0])
    block_count = (count + block_size - 1) // block_size
    padded_count = block_count * block_size
    if padded_count:
        left = np.zeros((padded_count,), dtype=np.int32)
        right = np.zeros((padded_count,), dtype=np.int32)
        valid = np.zeros((padded_count,), dtype=np.bool_)
        left[:count] = pairs[:, 0]
        right[:count] = pairs[:, 1]
        valid[:count] = True
        left = left.reshape(block_count, block_size)
        right = right.reshape(block_count, block_size)
        valid = valid.reshape(block_count, block_size)
    else:
        left = np.empty((0, block_size), dtype=np.int32)
        right = np.empty((0, block_size), dtype=np.int32)
        valid = np.empty((0, block_size), dtype=np.bool_)
    return NeighborBlocks(
        left=mx.array(left, dtype=mx.int32),
        right=mx.array(right, dtype=mx.int32),
        valid_mask=mx.array(valid),
        block_size=block_size,
        candidate_count=count,
        compact_pair_count=compact_pair_count,
    )


def _build_mlx_spatial_cell_tile_list(
    positions_mx: mx.array,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
    subdivision: int = DEFAULT_MLX_SPATIAL_CELL_SUBDIVISION,
    generation: int = 0,
) -> NeighborList:
    """Build exact 4x4 tiles from coarse 8x8 Metal membership."""

    profiler = _ACTIVE_MLX_CELL_TILE_REBUILD_PROFILER.get()
    build_started = perf_counter() if profiler is not None else 0.0
    stage_started = build_started

    def finish_stage(name: str, *completion_values: mx.array) -> None:
        nonlocal stage_started
        if profiler is None:
            return
        if completion_values:
            mx.eval(*completion_values)
        completed = perf_counter()
        profiler.record_stage(name, completed - stage_started)
        stage_started = perf_counter()

    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, "mlx_cell_tiles")
    lengths_mx = as_mx_array(cell.lengths, dtype=mx.float32)
    mx.eval(lengths_mx)
    lengths = np.asarray(lengths_mx, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)
    if subdivision <= 0:
        msg = "spatial cell subdivision must be positive"
        raise ValueError(msg)

    finite = mx.all(mx.isfinite(positions_mx))
    mx.eval(finite)
    if not bool(np.asarray(finite)):
        msg = "positions must be finite"
        raise ValueError(msg)

    n_atoms = int(positions_mx.shape[0])
    lengths64 = lengths.astype(np.float64)
    n_cells_array = _bounded_spatial_grid(
        lengths64,
        search_radius=search_radius,
        subdivision=subdivision,
        n_atoms=n_atoms,
    )
    n_cells = tuple(int(value) for value in n_cells_array)
    cell_count = int(np.prod(n_cells_array.astype(np.int64)))
    cell_widths = lengths64 / n_cells_array.astype(np.float64)
    finish_stage("validation_and_grid")

    n_cells_mx = mx.array(n_cells_array, dtype=mx.int32)
    wrapped = positions_mx - mx.floor(positions_mx / lengths_mx) * lengths_mx
    cell_indices = mx.floor(wrapped / lengths_mx * n_cells_mx).astype(mx.int32)
    cell_indices = mx.minimum(cell_indices, n_cells_mx - 1)
    cell_ids = (cell_indices[:, 0] * n_cells[1] + cell_indices[:, 1]) * n_cells[2] + cell_indices[
        :, 2
    ]
    sorted_atoms = mx.argsort(cell_ids).astype(mx.int32)
    cell_counts_mx = (
        mx.zeros((cell_count,), dtype=mx.int32)
        .at[cell_ids]
        .add(mx.ones((n_atoms,), dtype=mx.int32))
    )
    mx.eval(sorted_atoms, cell_counts_mx)
    cell_counts = np.asarray(cell_counts_mx, dtype=np.int32)
    finish_stage("cell_assignment_sort")
    cell_starts = np.empty_like(cell_counts)
    if cell_count:
        cell_starts[0] = 0
        np.cumsum(cell_counts[:-1], dtype=np.int64, out=cell_starts[1:])

    cell_block_counts = (
        cell_counts + DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE - 1
    ) // DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE
    cell_block_starts = np.empty_like(cell_block_counts)
    if cell_count:
        cell_block_starts[0] = 0
        np.cumsum(
            cell_block_counts[:-1],
            dtype=np.int64,
            out=cell_block_starts[1:],
        )
    block_count = int(np.sum(cell_block_counts, dtype=np.int64))
    if block_count:
        block_cells = np.repeat(
            np.arange(cell_count, dtype=np.int32),
            cell_block_counts,
        )
        block_local = np.arange(block_count, dtype=np.int32) - np.repeat(
            cell_block_starts,
            cell_block_counts,
        )
        block_cells_mx = mx.array(block_cells, dtype=mx.int32)
        within_cell = (
            mx.array(
                block_local * DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE,
                dtype=mx.int32,
            )[:, None]
            + mx.arange(DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE, dtype=mx.int32)[None, :]
        )
        sorted_offsets = (
            mx.array(cell_starts, dtype=mx.int32)[block_cells_mx][:, None] + within_cell
        )
        valid = within_cell < cell_counts_mx[block_cells_mx][:, None]
        safe_offsets = mx.minimum(sorted_offsets, n_atoms - 1)
        atom_blocks = mx.where(
            valid,
            sorted_atoms[safe_offsets],
            mx.array(-1, dtype=mx.int32),
        ).astype(mx.int32)
    else:
        atom_blocks = mx.zeros(
            (0, DEFAULT_MLX_CELL_TILE_BUILD_BLOCK_SIZE),
            dtype=mx.int32,
        )

    task_left, task_right, task_atom_candidate_counts = _spatial_cell_pair_tasks(
        cell_counts,
        n_cells,
        cell_widths,
        search_radius=search_radius,
    )
    left_block_counts = cell_block_counts[task_left].astype(np.int64)
    right_block_counts = cell_block_counts[task_right].astype(np.int64)
    task_tile_counts = left_block_counts * right_block_counts
    same_cell = task_left == task_right
    same_counts = left_block_counts[same_cell]
    task_tile_counts[same_cell] = same_counts * (same_counts + 1) // 2
    candidate_tile_count = int(np.sum(task_tile_counts, dtype=np.int64))
    if candidate_tile_count > np.iinfo(np.int32).max:
        msg = "spatial tile candidate count exceeds int32 capacity"
        raise ValueError(msg)
    task_offsets = np.empty((task_tile_counts.shape[0],), dtype=np.int32)
    if task_offsets.size:
        task_offsets[0] = 0
        np.cumsum(task_tile_counts[:-1], dtype=np.int64, out=task_offsets[1:])
    cell_pairs = np.stack((task_left, task_right), axis=1).astype(
        np.int32,
        copy=False,
    )
    finish_stage("host_task_inventory", atom_blocks)
    candidate_tiles = _neighbor_cell_tile_candidates(
        mx.array(cell_block_starts, dtype=mx.int32),
        mx.array(cell_block_counts, dtype=mx.int32),
        mx.array(cell_pairs, dtype=mx.int32),
        mx.array(task_offsets, dtype=mx.int32),
        candidate_count=candidate_tile_count,
    )
    candidate_subtile_mask, member_counts, subtile_counts = _neighbor_tile_membership(
        positions_mx,
        atom_blocks,
        candidate_tiles,
        lengths_mx,
        search_radius=search_radius,
    )
    finish_stage(
        "candidate_membership",
        candidate_tiles,
        candidate_subtile_mask,
        member_counts,
        subtile_counts,
    )

    if candidate_tile_count:
        subtile_prefix = mx.cumsum(subtile_counts)
        member_prefix = mx.cumsum(member_counts)
        inventory_counts = mx.stack(
            (subtile_prefix[-1], member_prefix[-1]),
        )
        mx.eval(subtile_prefix, member_prefix, inventory_counts)
        tile_count, exact_pair_count = (int(value) for value in np.asarray(inventory_counts))
        finish_stage("tile_inventory_readback")
        tile_blocks, member_mask = _neighbor_tile_ordered_scatter_sized(
            candidate_tiles,
            candidate_subtile_mask,
            subtile_counts,
            subtile_prefix,
            accepted_count=tile_count,
        )
        atom_blocks = atom_blocks.reshape((-1, DEFAULT_MLX_CELL_TILE_BLOCK_SIZE))
        block_count = int(atom_blocks.shape[0])
        # Cell-pair tasks are spatially coherent but periodic canonicalization
        # does not guarantee monotonic block IDs. Group only the much smaller
        # non-empty tile inventory so every force group owns one left block.
        force_order = mx.argsort(tile_blocks[:, 0])
        tile_blocks = tile_blocks[force_order]
        member_mask = member_mask[force_order]
        finish_stage("exact_tile_scatter_and_order", tile_blocks, member_mask)
        column_patterns = mx.array(
            [0x1111, 0x2222, 0x4444, 0x8888],
            dtype=mx.uint32,
        )
        active_column_counts = mx.sum(
            (member_mask[:, :, None] & column_patterns[None, None, :]) != 0,
            axis=(1, 2),
        ).astype(mx.int32)
        active_column_prefix = mx.cumsum(active_column_counts)
        column_counts_by_left = (
            mx.zeros((block_count,), dtype=mx.int32).at[tile_blocks[:, 0]].add(active_column_counts)
        )
        column_prefix_by_left = mx.cumsum(column_counts_by_left)
        force_groups_by_left = (
            column_counts_by_left + DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE - 1
        ) // DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE
        force_group_prefix = mx.cumsum(force_groups_by_left)
        # These inventories are independent until their sized scatter kernels.
        mx.eval(active_column_prefix, column_prefix_by_left, force_group_prefix)
        active_column_count = int(np.asarray(active_column_prefix[-1]))
        force_group_count = int(np.asarray(force_group_prefix[-1]))
        finish_stage("force_inventory_readback")
        force_columns = _neighbor_tile_column_scatter_sized(
            member_mask,
            active_column_counts,
            active_column_prefix,
            accepted_count=active_column_count,
        )
        force_group_starts, force_group_counts = _neighbor_tile_force_groups_sized(
            column_counts_by_left,
            column_prefix_by_left,
            force_groups_by_left,
            force_group_prefix,
            accepted_count=force_group_count,
            items_per_group=DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE,
        )
    else:
        atom_blocks = atom_blocks.reshape((-1, DEFAULT_MLX_CELL_TILE_BLOCK_SIZE))
        tile_count = 0
        exact_pair_count = 0
        tile_blocks = mx.zeros((0, 2), dtype=mx.int32)
        member_mask = mx.zeros((0, 1), dtype=mx.uint32)
        force_columns = mx.zeros((0,), dtype=mx.int32)
        force_group_starts = mx.zeros((0,), dtype=mx.int32)
        force_group_counts = mx.zeros((0,), dtype=mx.int32)
        finish_stage("tile_inventory_readback")
        finish_stage("exact_tile_scatter_and_order")
        finish_stage("force_inventory_readback")
    mx.eval(
        atom_blocks,
        tile_blocks,
        member_mask,
        force_columns,
        force_group_starts,
        force_group_counts,
    )
    finish_stage("force_schedule_scatter")

    raw_candidate_count = int(np.sum(task_atom_candidate_counts, dtype=np.int64))
    tiles = NeighborTiles(
        atom_blocks=atom_blocks,
        tile_blocks=tile_blocks,
        member_mask=member_mask,
        exact_pair_count=exact_pair_count,
        raw_candidate_count=raw_candidate_count,
        force_columns=force_columns,
        force_group_starts=force_group_starts,
        force_group_counts=force_group_counts,
        generation=generation,
    )
    estimated_cell_list_bytes = (
        n_atoms * (3 * _FLOAT_BYTES + 2 * _INT_BYTES) + cell_count * 2 * _INT_BYTES
    )
    stats = PairListStats(
        pair_count=exact_pair_count,
        n_cells=n_cells,
        cell_count=cell_count,
        occupied_cell_count=int(np.count_nonzero(cell_counts)),
        search_radius=search_radius,
        estimated_pair_bytes=tiles.estimated_bytes,
        estimated_cell_list_bytes=estimated_cell_list_bytes,
        backend="mlx_cell_tiles",
        representation_kind="tiles",
        candidate_count=raw_candidate_count,
        estimated_candidate_bytes=candidate_tile_count * (10 * _INT_BYTES),
        compaction_backend="metal_spatial_tile_prefix_scan",
        fallback_reason=None,
    )
    neighbor_list = NeighborList(
        None,
        cutoff=cutoff,
        skin=skin,
        stats=stats,
        tiles=tiles,
        sort_diagnostic_pairs=sort_pairs,
    )
    finish_stage("result_assembly")
    if profiler is not None:
        profiler.record_build(
            perf_counter() - build_started,
            inventory={
                "atom_count": n_atoms,
                "cell_count": cell_count,
                "coarse_candidate_tile_count": candidate_tile_count,
                "exact_tile_count": tile_count,
                "exact_pair_count": exact_pair_count,
                "force_column_count": int(force_columns.shape[0]),
                "force_group_count": int(force_group_starts.shape[0]),
            },
        )
    return neighbor_list


def _build_mlx_cell_tiles(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
    generation: int = 0,
) -> NeighborList:
    """Build exact cutoff-plus-skin membership over four-atom block tiles."""

    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, "mlx_cell_tiles")
    lengths = np.asarray(cell.lengths, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)

    n_atoms = int(positions_np.shape[0])
    n_cells_array = _bounded_spatial_grid(
        lengths.astype(np.float64),
        search_radius=search_radius,
        subdivision=DEFAULT_MLX_SPATIAL_CELL_SUBDIVISION,
        n_atoms=n_atoms,
    )
    n_cells = tuple(int(value) for value in n_cells_array)
    cell_count = int(np.prod(n_cells_array.astype(np.int64)))
    cell_widths = lengths.astype(np.float64) / n_cells_array.astype(np.float64)
    wrapped = positions_np - np.floor(positions_np / lengths) * lengths
    indices = np.floor(wrapped / lengths * n_cells_array).astype(np.int32)
    indices = np.minimum(indices, n_cells_array - 1)
    cell_ids = (indices[:, 0] * n_cells[1] + indices[:, 1]) * n_cells[2] + indices[:, 2]
    sorted_atoms = np.argsort(cell_ids, kind="stable").astype(np.int32, copy=False)
    cell_counts = np.bincount(cell_ids, minlength=cell_count).astype(np.int32)
    cell_starts = np.empty_like(cell_counts)
    if cell_count:
        cell_starts[0] = 0
        np.cumsum(cell_counts[:-1], dtype=np.int64, out=cell_starts[1:])

    atom_block_rows: list[np.ndarray] = []
    cell_block_ids: list[tuple[int, ...]] = [() for _ in range(cell_count)]
    for cell_id in np.flatnonzero(cell_counts):
        start = int(cell_starts[cell_id])
        stop = start + int(cell_counts[cell_id])
        members = sorted_atoms[start:stop]
        block_ids: list[int] = []
        for start in range(0, int(members.shape[0]), DEFAULT_MLX_CELL_TILE_BLOCK_SIZE):
            block = np.full((DEFAULT_MLX_CELL_TILE_BLOCK_SIZE,), -1, dtype=np.int32)
            chunk = members[start : start + DEFAULT_MLX_CELL_TILE_BLOCK_SIZE]
            block[: chunk.shape[0]] = chunk
            block_ids.append(len(atom_block_rows))
            atom_block_rows.append(block)
        cell_block_ids[int(cell_id)] = tuple(block_ids)

    if atom_block_rows:
        atom_blocks = np.stack(atom_block_rows, axis=0)
    else:
        atom_blocks = np.empty(
            (0, DEFAULT_MLX_CELL_TILE_BLOCK_SIZE),
            dtype=np.int32,
        )

    task_left, task_right, _ = _spatial_cell_pair_tasks(
        cell_counts,
        n_cells,
        cell_widths,
        search_radius=search_radius,
    )
    candidate_tile_chunks: list[np.ndarray] = []
    for left_cell, right_cell in zip(task_left, task_right, strict=True):
        left_blocks = np.asarray(cell_block_ids[int(left_cell)], dtype=np.int32)
        right_blocks = np.asarray(cell_block_ids[int(right_cell)], dtype=np.int32)
        if int(left_cell) == int(right_cell):
            left_index, right_index = np.triu_indices(left_blocks.shape[0])
            tile_chunk = np.stack(
                (left_blocks[left_index], right_blocks[right_index]),
                axis=1,
            )
        else:
            tile_chunk = np.stack(
                (
                    np.repeat(left_blocks, right_blocks.shape[0]),
                    np.tile(right_blocks, left_blocks.shape[0]),
                ),
                axis=1,
            )
        if tile_chunk.shape[0]:
            candidate_tile_chunks.append(tile_chunk.astype(np.int32, copy=False))
    candidate_tiles = (
        np.concatenate(candidate_tile_chunks, axis=0)
        if candidate_tile_chunks
        else np.empty((0, 2), dtype=np.int32)
    )

    search_radius2 = float(search_radius) * float(search_radius)
    tile_rows: list[np.ndarray] = []
    membership_rows: list[np.ndarray] = []
    raw_candidate_count = 0
    peak_tile_candidate_count = 0
    exact_pair_count = 0
    upper_triangle = np.triu(
        np.ones(
            (DEFAULT_MLX_CELL_TILE_BLOCK_SIZE, DEFAULT_MLX_CELL_TILE_BLOCK_SIZE),
            dtype=np.bool_,
        ),
        k=1,
    )
    bit_weights = np.left_shift(
        np.uint32(1),
        np.arange(32, dtype=np.uint32),
    )
    positions_mx = mx.array(positions_np, dtype=mx.float32) if _uses_metal_device() else None
    lengths_mx = mx.array(lengths, dtype=mx.float32) if positions_mx is not None else None
    for batch_start in range(0, candidate_tiles.shape[0], DEFAULT_MLX_CELL_TILE_BATCH):
        tile_batch = candidate_tiles[batch_start : batch_start + DEFAULT_MLX_CELL_TILE_BATCH]
        left_atoms = atom_blocks[tile_batch[:, 0]]
        right_atoms = atom_blocks[tile_batch[:, 1]]
        valid = (left_atoms[:, :, None] >= 0) & (right_atoms[:, None, :] >= 0)
        same_block = tile_batch[:, 0] == tile_batch[:, 1]
        valid[same_block] &= upper_triangle
        candidates_per_tile = np.sum(valid, axis=(1, 2), dtype=np.int32)
        raw_candidate_count += int(np.sum(candidates_per_tile, dtype=np.int64))
        if candidates_per_tile.shape[0]:
            peak_tile_candidate_count = max(
                peak_tile_candidate_count,
                int(np.max(candidates_per_tile)),
            )

        safe_left = np.maximum(left_atoms, 0)
        safe_right = np.maximum(right_atoms, 0)
        if positions_mx is None or lengths_mx is None:
            displacement = (
                positions_np[safe_left][:, :, None, :] - positions_np[safe_right][:, None, :, :]
            )
            displacement -= lengths * np.round(displacement / lengths)
            distance2 = np.sum(displacement * displacement, axis=-1)
            close = distance2 < search_radius2
        else:
            lane_left = np.broadcast_to(safe_left[:, :, None], valid.shape)
            lane_right = np.broadcast_to(safe_right[:, None, :], valid.shape)
            close_mx = neighbor_pair_cutoff_mask(
                positions_mx,
                mx.array(lane_left.reshape(-1), dtype=mx.int32),
                mx.array(lane_right.reshape(-1), dtype=mx.int32),
                lengths_mx,
                search_radius=search_radius,
            )
            mx.eval(close_mx)
            close = np.asarray(close_mx).reshape(valid.shape)
        member = valid & close
        lanes_per_tile = DEFAULT_MLX_CELL_TILE_BLOCK_SIZE**2
        flat_member = member.reshape((-1, lanes_per_tile)).astype(
            np.uint32,
            copy=False,
        )
        mask_words = np.sum(
            flat_member * bit_weights[:lanes_per_tile],
            axis=1,
            dtype=np.uint32,
        )[:, None]
        nonempty = np.any(mask_words != 0, axis=1)
        if np.any(nonempty):
            tile_rows.append(tile_batch[nonempty])
            membership_rows.append(mask_words[nonempty])
            exact_pair_count += int(np.count_nonzero(member))

    if tile_rows:
        tile_blocks = np.concatenate(tile_rows, axis=0)
        member_mask = np.concatenate(membership_rows, axis=0)
    else:
        tile_blocks = np.empty((0, 2), dtype=np.int32)
        member_mask = np.empty((0, 1), dtype=np.uint32)
    tiles = NeighborTiles(
        atom_blocks=mx.array(atom_blocks, dtype=mx.int32),
        tile_blocks=mx.array(tile_blocks, dtype=mx.int32),
        member_mask=mx.array(member_mask, dtype=mx.uint32),
        exact_pair_count=exact_pair_count,
        raw_candidate_count=raw_candidate_count,
        generation=generation,
    )
    stats = PairListStats(
        pair_count=exact_pair_count,
        n_cells=n_cells,
        cell_count=cell_count,
        occupied_cell_count=int(np.count_nonzero(cell_counts)),
        search_radius=search_radius,
        estimated_pair_bytes=tiles.estimated_bytes,
        estimated_cell_list_bytes=(
            n_atoms * (3 * _FLOAT_BYTES + 2 * _INT_BYTES) + cell_count * 2 * _INT_BYTES
        ),
        backend="mlx_cell_tiles",
        representation_kind="tiles",
        candidate_count=raw_candidate_count,
        estimated_candidate_bytes=(peak_tile_candidate_count * (3 * _FLOAT_BYTES + _BOOL_BYTES)),
        compaction_backend=(
            "metal_tile_membership_mask"
            if positions_mx is not None
            else "cpu_vectorized_tile_membership_mask"
        ),
        fallback_reason=None,
    )
    return NeighborList(
        None,
        cutoff=cutoff,
        skin=skin,
        stats=stats,
        tiles=tiles,
        sort_diagnostic_pairs=sort_pairs,
    )


def _count_candidate_pairs_within_radius(
    positions_np: np.ndarray,
    cell: Cell,
    pairs: np.ndarray,
    *,
    search_radius: float,
) -> int:
    return int(
        _filter_candidate_pairs_within_radius(
            positions_np,
            cell,
            pairs,
            search_radius=search_radius,
        ).shape[0]
    )


def _filter_candidate_pairs_within_radius(
    positions_np: np.ndarray,
    cell: Cell,
    pairs: np.ndarray,
    *,
    search_radius: float,
) -> np.ndarray:
    """Return candidate pairs inside the strict periodic search radius."""

    if pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    displacement = positions_np[pairs[:, 0]] - positions_np[pairs[:, 1]]
    displacement = np.asarray(cell.minimum_image(as_mx_array(displacement)))
    distance2 = np.sum(displacement * displacement, axis=1)
    return pairs[distance2 < search_radius * search_radius].astype(
        np.int32,
        copy=False,
    )


def _build_mlx_spatial_cell_pair_list(
    positions_mx: mx.array,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
    subdivision: int = DEFAULT_MLX_SPATIAL_CELL_SUBDIVISION,
    candidate_batch: int = DEFAULT_MLX_SPATIAL_CANDIDATE_BATCH,
) -> NeighborList:
    """Build compact pairs from a device-resident, spatially pruned cell grid."""

    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, "mlx_cell_pairs")
    lengths_mx = as_mx_array(cell.lengths, dtype=mx.float32)
    mx.eval(lengths_mx)
    lengths = np.asarray(lengths_mx, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)
    if subdivision <= 0:
        msg = "spatial cell subdivision must be positive"
        raise ValueError(msg)
    if candidate_batch <= 0 or candidate_batch > np.iinfo(np.int32).max:
        msg = "spatial candidate batch must be a positive int32-sized count"
        raise ValueError(msg)

    finite = mx.all(mx.isfinite(positions_mx))
    mx.eval(finite)
    if not bool(np.asarray(finite)):
        msg = "positions must be finite"
        raise ValueError(msg)

    n_atoms = int(positions_mx.shape[0])
    lengths64 = lengths.astype(np.float64)
    n_cells_array = _bounded_spatial_grid(
        lengths64,
        search_radius=search_radius,
        subdivision=subdivision,
        n_atoms=n_atoms,
    )
    n_cells = tuple(int(value) for value in n_cells_array)
    cell_count = int(np.prod(n_cells_array.astype(np.int64)))
    cell_widths = lengths64 / n_cells_array.astype(np.float64)

    n_cells_mx = mx.array(n_cells_array, dtype=mx.int32)
    wrapped = positions_mx - mx.floor(positions_mx / lengths_mx) * lengths_mx
    cell_indices = mx.floor(wrapped / lengths_mx * n_cells_mx).astype(mx.int32)
    cell_indices = mx.minimum(cell_indices, n_cells_mx - 1)
    cell_ids = (cell_indices[:, 0] * n_cells[1] + cell_indices[:, 1]) * n_cells[2] + cell_indices[
        :, 2
    ]
    sorted_atoms = mx.argsort(cell_ids).astype(mx.int32)
    cell_counts_mx = (
        mx.zeros((cell_count,), dtype=mx.int32)
        .at[cell_ids]
        .add(mx.ones((n_atoms,), dtype=mx.int32))
    )
    mx.eval(sorted_atoms, cell_counts_mx)
    cell_counts = np.asarray(cell_counts_mx, dtype=np.int32)
    cell_starts = np.empty_like(cell_counts)
    if cell_count:
        cell_starts[0] = 0
        np.cumsum(cell_counts[:-1], dtype=np.int64, out=cell_starts[1:])

    task_left, task_right, task_candidate_counts = _spatial_cell_pair_tasks(
        cell_counts,
        n_cells,
        cell_widths,
        search_radius=search_radius,
    )
    candidate_count = int(np.sum(task_candidate_counts, dtype=np.int64))
    if task_candidate_counts.size and int(np.max(task_candidate_counts)) > candidate_batch:
        largest = int(np.max(task_candidate_counts))
        msg = (
            "one spatial cell-pair task exceeds the bounded Metal candidate batch: "
            f"candidate_count={largest}, candidate_batch={candidate_batch}; "
            "increase spatial subdivision or reduce local occupancy"
        )
        raise ValueError(msg)

    cell_starts_mx = mx.array(cell_starts, dtype=mx.int32)
    compact_chunks: list[mx.array] = []
    compact_pair_count = 0
    peak_candidate_count = 0
    for task_start, task_stop in _spatial_task_batches(
        task_candidate_counts,
        candidate_batch=candidate_batch,
    ):
        batch_counts = task_candidate_counts[task_start:task_stop]
        batch_candidate_count = int(np.sum(batch_counts, dtype=np.int64))
        peak_candidate_count = max(peak_candidate_count, batch_candidate_count)
        task_offsets = np.empty((batch_counts.shape[0],), dtype=np.int32)
        task_offsets[0] = 0
        if task_offsets.shape[0] > 1:
            np.cumsum(batch_counts[:-1], dtype=np.int64, out=task_offsets[1:])
        cell_pairs = np.stack(
            (
                task_left[task_start:task_stop],
                task_right[task_start:task_stop],
            ),
            axis=1,
        ).astype(np.int32, copy=False)
        candidates_i, candidates_j = _neighbor_cell_pair_candidates(
            sorted_atoms,
            cell_starts_mx,
            cell_counts_mx,
            mx.array(cell_pairs, dtype=mx.int32),
            mx.array(task_offsets, dtype=mx.int32),
            candidate_count=batch_candidate_count,
        )
        close = neighbor_pair_cutoff_mask(
            positions_mx,
            candidates_i,
            candidates_j,
            lengths_mx,
            search_radius=search_radius,
        )
        prefix = mx.cumsum(close.astype(mx.int32))
        mx.eval(prefix)
        accepted_count = int(np.asarray(prefix[-1]))
        if accepted_count == 0:
            continue
        accepted_i, accepted_j = _neighbor_pair_ordered_scatter_sized(
            candidates_i,
            candidates_j,
            close,
            prefix,
            accepted_count=accepted_count,
        )
        compact = mx.stack((accepted_i, accepted_j), axis=1)
        mx.eval(compact)
        compact_chunks.append(compact)
        compact_pair_count += accepted_count

    if len(compact_chunks) == 1:
        pairs = compact_chunks[0]
    elif compact_chunks:
        pairs = mx.concatenate(compact_chunks, axis=0)
    else:
        pairs = mx.zeros((0, 2), dtype=mx.int32)
    if sort_pairs and compact_pair_count:
        right_order = mx.argsort(pairs[:, 1])
        pairs = pairs[right_order]
        left_order = mx.argsort(pairs[:, 0])
        pairs = pairs[left_order]
    if len(compact_chunks) > 1 or sort_pairs:
        mx.eval(pairs)

    occupied_cell_count = int(np.count_nonzero(cell_counts))
    estimated_cell_list_bytes = (
        n_atoms * (3 * _FLOAT_BYTES + 2 * _INT_BYTES) + cell_count * 2 * _INT_BYTES
    )
    stats = PairListStats(
        pair_count=compact_pair_count,
        n_cells=n_cells,
        cell_count=cell_count,
        occupied_cell_count=occupied_cell_count,
        search_radius=search_radius,
        estimated_pair_bytes=estimate_pair_list_bytes(compact_pair_count),
        estimated_cell_list_bytes=estimated_cell_list_bytes,
        backend="mlx_cell_pairs",
        representation_kind="pairs",
        candidate_count=candidate_count,
        estimated_candidate_bytes=(peak_candidate_count * (3 * _FLOAT_BYTES + _BOOL_BYTES)),
        compaction_backend="metal_spatial_prefix_scan",
        fallback_reason=None,
    )
    return NeighborList(
        pairs,
        cutoff=cutoff,
        skin=skin,
        stats=stats,
    )


def _spatial_cell_pair_tasks(
    cell_counts: np.ndarray,
    n_cells: tuple[int, int, int],
    cell_widths: np.ndarray,
    *,
    search_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupied canonical cell-pair tasks allowed by an AABB stencil."""

    cell_count = int(np.prod(np.asarray(n_cells, dtype=np.int64)))
    if cell_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, empty.astype(np.int64)
    left, right, same = _spatial_cell_pair_template(
        n_cells,
        tuple(float(value) for value in cell_widths),
        float(search_radius),
    )
    left_counts = cell_counts[left].astype(np.int64)
    right_counts = cell_counts[right].astype(np.int64)
    candidate_counts = left_counts * right_counts
    same_counts = left_counts[same]
    candidate_counts[same] = same_counts * np.maximum(same_counts - 1, 0) // 2
    occupied = candidate_counts > 0
    # Boolean indexing would copy both cached templates on dense solvent grids.
    if bool(np.all(occupied)):
        return left, right, candidate_counts
    return (
        left[occupied],
        right[occupied],
        candidate_counts[occupied],
    )


@lru_cache(maxsize=8)
def _spatial_cell_pair_template(
    n_cells: tuple[int, int, int],
    cell_widths: tuple[float, float, float],
    search_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immutable canonical cell-pair indices for fixed geometry."""

    cell_count = int(np.prod(np.asarray(n_cells, dtype=np.int64)))
    if cell_count == 0:
        empty = np.empty((0,), dtype=np.int32)
        empty.flags.writeable = False
        same = np.empty((0,), dtype=np.bool_)
        same.flags.writeable = False
        return empty, empty, same
    cell_ids = np.arange(cell_count, dtype=np.int64)
    cell_x, cell_y, cell_z = np.unravel_index(cell_ids, n_cells)
    encoded_chunks: list[np.ndarray] = []
    for offset_x, offset_y, offset_z in _aabb_pruned_half_stencil(
        np.asarray(cell_widths, dtype=np.float64),
        search_radius=search_radius,
    ):
        neighbor_x = (cell_x + offset_x) % n_cells[0]
        neighbor_y = (cell_y + offset_y) % n_cells[1]
        neighbor_z = (cell_z + offset_z) % n_cells[2]
        neighbor_ids = (neighbor_x * n_cells[1] + neighbor_y) * n_cells[2] + neighbor_z
        lower = np.minimum(cell_ids, neighbor_ids)
        upper = np.maximum(cell_ids, neighbor_ids)
        encoded_chunks.append(lower * cell_count + upper)
    encoded = np.unique(np.concatenate(encoded_chunks))
    left = (encoded // cell_count).astype(np.int32, copy=False)
    right = (encoded % cell_count).astype(np.int32, copy=False)
    same = left == right
    left.flags.writeable = False
    right.flags.writeable = False
    same.flags.writeable = False
    return left, right, same


def _bounded_spatial_grid(
    lengths: np.ndarray,
    *,
    search_radius: float,
    subdivision: int,
    n_atoms: int,
) -> np.ndarray:
    """Return a fine grid without allocating a pathological number of empty cells."""

    requested = np.maximum(
        np.floor(lengths * float(subdivision) / float(search_radius)).astype(np.int64),
        1,
    )
    max_cells = max(1, DEFAULT_MLX_SPATIAL_MAX_CELLS_PER_ATOM * n_atoms)
    requested_count = int(np.prod(requested, dtype=np.int64))
    if requested_count <= max_cells:
        return requested.astype(np.int32)

    scale = (float(max_cells) / float(requested_count)) ** (1.0 / 3.0)
    bounded = np.maximum(np.floor(requested.astype(np.float64) * scale).astype(np.int64), 1)
    while int(np.prod(bounded, dtype=np.int64)) > max_cells:
        reducible = np.flatnonzero(bounded > 1)
        if reducible.size == 0:
            break
        axis = int(reducible[np.argmax(bounded[reducible] / requested[reducible])])
        bounded[axis] -= 1
    return bounded.astype(np.int32)


def _aabb_pruned_half_stencil(
    cell_widths: np.ndarray,
    *,
    search_radius: float,
) -> tuple[tuple[int, int, int], ...]:
    """Return canonical cell offsets whose axis-aligned boxes can be in range."""

    max_offsets = np.ceil(search_radius / cell_widths).astype(np.int32)
    radius2 = float(search_radius) * float(search_radius)
    tolerance = np.finfo(np.float64).eps * max(radius2, 1.0) * 16.0
    offsets: list[tuple[int, int, int]] = []
    for offset in product(
        range(-int(max_offsets[0]), int(max_offsets[0]) + 1),
        range(-int(max_offsets[1]), int(max_offsets[1]) + 1),
        range(-int(max_offsets[2]), int(max_offsets[2]) + 1),
    ):
        if not _is_canonical_half_offset(offset):
            continue
        gap = np.maximum(np.abs(np.asarray(offset, dtype=np.float64)) - 1.0, 0.0)
        minimum_distance2 = float(np.sum((gap * cell_widths) ** 2))
        if minimum_distance2 <= radius2 + tolerance:
            offsets.append(offset)
    return tuple(offsets)


def _is_canonical_half_offset(offset: tuple[int, int, int]) -> bool:
    for component in offset:
        if component:
            return component > 0
    return True


def _spatial_task_batches(
    task_candidate_counts: np.ndarray,
    *,
    candidate_batch: int,
) -> tuple[tuple[int, int], ...]:
    if task_candidate_counts.size == 0:
        return ()
    cumulative = np.cumsum(task_candidate_counts, dtype=np.int64)
    batches: list[tuple[int, int]] = []
    start = 0
    prior = 0
    while start < task_candidate_counts.shape[0]:
        stop = int(np.searchsorted(cumulative, prior + candidate_batch, side="right"))
        if stop <= start:
            stop = start + 1
        batches.append((start, stop))
        prior = int(cumulative[stop - 1])
        start = stop
    return tuple(batches)


def _build_mlx_cell_pair_list_cpu(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
) -> NeighborList:
    _require_orthorhombic_cell_for_compact_neighbor_backend(cell, "mlx_cell_pairs")
    lengths = np.asarray(cell.lengths, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)

    cell_list = build_periodic_cell_list(positions_np, cell, search_radius=search_radius)
    wrapped = positions_np - np.floor(positions_np / lengths) * lengths
    positions_mx = mx.array(wrapped, dtype=mx.float32)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    pair_chunks: list[np.ndarray] = []
    pending_left: list[np.ndarray] = []
    pending_right: list[np.ndarray] = []
    pending_candidate_count = 0
    candidate_count = 0
    peak_candidate_count = 0

    for cell_index, members in tuple(cell_list.bins.items()):
        neighbor_indices = sorted(
            {
                tuple(
                    (cell_index[axis] + offset[axis]) % cell_list.n_cells[axis] for axis in range(3)
                )
                for offset in offsets
            }
        )
        for neighbor_index in neighbor_indices:
            if neighbor_index < cell_index:
                continue
            neighbors = cell_list.bins.get(neighbor_index)
            if neighbors is None:
                continue
            if neighbor_index == cell_index:
                left, right = _same_cell_member_pairs(members)
            else:
                left, right = _cross_cell_member_pairs(members, neighbors)
            if left.shape[0] == 0:
                continue
            left_count = int(left.shape[0])
            candidate_count += left_count
            pending_left.append(left)
            pending_right.append(right)
            pending_candidate_count += left_count
            if pending_candidate_count >= DEFAULT_MLX_CELL_PAIR_CANDIDATE_CHUNK:
                peak_candidate_count = max(peak_candidate_count, pending_candidate_count)
                _flush_mlx_candidate_chunks(
                    pair_chunks,
                    positions_mx,
                    pending_left,
                    pending_right,
                    cell,
                    search_radius=search_radius,
                )
                pending_left.clear()
                pending_right.clear()
                pending_candidate_count = 0

    if pending_candidate_count > 0:
        peak_candidate_count = max(peak_candidate_count, pending_candidate_count)
        _flush_mlx_candidate_chunks(
            pair_chunks,
            positions_mx,
            pending_left,
            pending_right,
            cell,
            search_radius=search_radius,
        )

    if pair_chunks:
        pair_array = np.concatenate(pair_chunks, axis=0).astype(np.int32, copy=False)
    else:
        pair_array = np.empty((0, 2), dtype=np.int32)
    if sort_pairs and pair_array.shape[0]:
        order = np.lexsort((pair_array[:, 1], pair_array[:, 0]))
        pair_array = pair_array[order]

    stats = PairListStats(
        pair_count=int(pair_array.shape[0]),
        n_cells=cell_list.n_cells,
        cell_count=cell_list.cell_count,
        occupied_cell_count=cell_list.occupied_cell_count,
        search_radius=search_radius,
        estimated_pair_bytes=estimate_pair_list_bytes(int(pair_array.shape[0])),
        estimated_cell_list_bytes=cell_list.estimated_bytes,
        backend="mlx_cell_pairs",
        representation_kind="pairs",
        candidate_count=candidate_count,
        estimated_candidate_bytes=peak_candidate_count * (3 * _FLOAT_BYTES + _BOOL_BYTES),
        compaction_backend=(
            "metal_prefix_scan" if "gpu" in str(mx.default_device()).lower() else "cpu_argwhere"
        ),
        fallback_reason=None,
    )
    return NeighborList(
        mx.array(pair_array, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
    )


def _same_cell_member_pairs(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if members.shape[0] < 2:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    left, right = np.triu_indices(members.shape[0], k=1)
    return members[left].astype(np.int32, copy=False), members[right].astype(np.int32, copy=False)


def _cross_cell_member_pairs(
    members: np.ndarray,
    neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if members.shape[0] == 0 or neighbors.shape[0] == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    return (
        np.repeat(members, neighbors.shape[0]).astype(np.int32, copy=False),
        np.tile(neighbors, members.shape[0]).astype(np.int32, copy=False),
    )


def _mlx_filter_index_pairs_within_radius(
    positions_mx: mx.array,
    left: np.ndarray,
    right: np.ndarray,
    cell: Cell,
    *,
    search_radius: float,
) -> np.ndarray:
    left_mx = mx.array(left, dtype=mx.int32)
    right_mx = mx.array(right, dtype=mx.int32)
    if "gpu" in str(mx.default_device()).lower():
        close = neighbor_pair_cutoff_mask(
            positions_mx,
            left_mx,
            right_mx,
            cell.lengths,
            search_radius=search_radius,
        )
        prefix = mx.cumsum(close.astype(mx.int32))
        mx.eval(prefix)
        selected_count = int(np.asarray(prefix[-1]))
        if selected_count == 0:
            return np.empty((0, 2), dtype=np.int32)
        accepted_i, accepted_j = neighbor_pair_ordered_scatter(
            left_mx,
            right_mx,
            close,
            prefix,
        )
        compact = mx.stack(
            (
                accepted_i[:selected_count],
                accepted_j[:selected_count],
            ),
            axis=1,
        )
        mx.eval(compact)
        return np.asarray(compact).astype(np.int32, copy=False)
    else:
        displacement = positions_mx[left_mx] - positions_mx[right_mx]
        displacement = cell.minimum_image(displacement)
        r2 = mx.sum(displacement * displacement, axis=1)
        close = r2 < search_radius * search_radius
    mx.eval(close)
    selected = np.argwhere(np.asarray(close)).reshape(-1)
    if selected.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    selected_left = left[selected]
    selected_right = right[selected]
    return np.stack(
        (
            np.minimum(selected_left, selected_right),
            np.maximum(selected_left, selected_right),
        ),
        axis=1,
    ).astype(np.int32, copy=False)


def _flush_mlx_candidate_chunks(
    pair_chunks: list[np.ndarray],
    positions_mx: mx.array,
    left_chunks: list[np.ndarray],
    right_chunks: list[np.ndarray],
    cell: Cell,
    *,
    search_radius: float,
) -> None:
    if not left_chunks:
        return
    left = np.concatenate(left_chunks).astype(np.int32, copy=False)
    right = np.concatenate(right_chunks).astype(np.int32, copy=False)
    pair_chunk = _mlx_filter_index_pairs_within_radius(
        positions_mx,
        left,
        right,
        cell,
        search_radius=search_radius,
    )
    if pair_chunk.shape[0] > 0:
        pair_chunks.append(pair_chunk)
