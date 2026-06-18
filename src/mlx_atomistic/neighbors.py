"""Neighbor-list construction for periodic MD."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

NeighborBackend = Literal[
    "auto",
    "periodic_cell_list",
    "mlx_dense_pairs",
    "mlx_cell_pairs",
    "mlx_cell_blocks",
]
NeighborCheckBackend = Literal["numpy", "mlx_scalar"]
_ALLOWED_NEIGHBOR_BACKENDS = {
    "auto",
    "periodic_cell_list",
    "mlx_dense_pairs",
    "mlx_cell_pairs",
    "mlx_cell_blocks",
}
_ALLOWED_NEIGHBOR_CHECK_BACKENDS = {"numpy", "mlx_scalar"}
DEFAULT_MLX_DENSE_PAIR_LIMIT = 4096
DEFAULT_MLX_CELL_PAIR_CANDIDATE_CHUNK = 1_000_000
DEFAULT_MLX_CELL_BLOCK_SIZE = 256
# Default neighbor backend for systems above the dense-pair limit. Measured on
# M5 Max (4k/16k/50k LJ, 2026-06-18): compacting candidates to real pairs
# ("mlx_cell_pairs") beats the fixed-shape padded-block path ("mlx_cell_blocks")
# by 5.9-7.1x in a managed loop (incl. rebuild) at identical physics (dE=0).
# Host compaction scales near-linearly (50k build ~0.34s) and does not OOM, so
# there is no scaling reason to prefer padded blocks. Blocks remain available as
# an explicit backend for the future on-device fused path.
DEFAULT_LARGE_SYSTEM_NEIGHBOR_BACKEND: NeighborBackend = "mlx_cell_pairs"
_BOOL_BYTES = np.dtype(np.bool_).itemsize
_FLOAT_BYTES = np.dtype(np.float32).itemsize
_INT_BYTES = np.dtype(np.int32).itemsize


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
class NeighborList:
    """Neighbor interactions for pairwise potentials."""

    pairs: mx.array
    cutoff: float
    skin: float = 0.0
    stats: PairListStats | None = None
    blocks: NeighborBlocks | None = None

    @property
    def pair_count(self) -> int:
        """Number of unique pairs or candidate block entries."""

        if self.blocks is not None:
            return self.blocks.candidate_count
        return self.pairs.shape[0]

    @property
    def compact_pair_count(self) -> int:
        """Number of compact pairs accepted by the neighbor search radius."""

        if self.blocks is not None:
            return self.blocks.compact_pair_count
        if self.stats is not None:
            return self.stats.pair_count
        return int(self.pairs.shape[0])

    @property
    def interactions(self) -> mx.array | NeighborBlocks:
        """Return the active force-evaluation representation."""

        return self.blocks if self.blocks is not None else self.pairs

    @property
    def backend(self) -> str:
        """Pair-construction backend name."""

        return "periodic_cell_list" if self.stats is None else self.stats.backend

    @property
    def estimated_pair_bytes(self) -> int:
        """Estimated bytes for the compact int32 pair array."""

        if self.blocks is not None:
            return self.blocks.estimated_bytes
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
    sort_pairs: bool = True
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

        start = perf_counter()
        self.neighbor_list = build_neighbor_list(
            positions,
            self.cell,
            cutoff=self.cutoff,
            skin=self.skin,
            sort_pairs=self.sort_pairs,
            max_workers=self.max_workers,
            backend=self.backend,
            max_mlx_dense_atoms=self.max_mlx_dense_atoms,
            block_size=self.block_size,
        )
        self.rebuild_wall_seconds += perf_counter() - start
        self.reference_positions = as_mx_array(positions)
        self.rebuild_count += 1
        self.last_max_displacement = 0.0
        self.updates_since_check = 0
        return self.neighbor_list

    def update(self, positions) -> NeighborList:
        """Return a current neighbor list, rebuilding if needed."""

        start = perf_counter()
        try:
            if self.needs_rebuild(positions):
                return self.rebuild(positions)
            if self.neighbor_list is None:
                msg = "neighbor list manager has no current neighbor list"
                raise RuntimeError(msg)
            return self.neighbor_list
        finally:
            self.update_wall_seconds += perf_counter() - start


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
) -> NeighborList:
    """Build a periodic cell-list neighbor list with unique `i < j` pairs."""

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

    positions_np = np.asarray(positions, dtype=np.float32)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        msg = "positions must have shape (n_particles, 3)"
        raise ValueError(msg)
    if not np.all(np.isfinite(positions_np)):
        msg = "positions must be finite"
        raise ValueError(msg)
    search_radius = cutoff + skin
    fallback_reason = None
    if backend == "auto":
        if positions_np.shape[0] <= max_mlx_dense_atoms:
            backend = "mlx_dense_pairs"
        else:
            backend = DEFAULT_LARGE_SYSTEM_NEIGHBOR_BACKEND
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
        return _build_mlx_cell_pair_list(
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
                    (cell_index[axis] + offset[axis]) % cell_list.n_cells[axis]
                    for axis in range(3)
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

    compact_pair_count = _count_candidate_pairs_within_radius(
        positions_np,
        cell,
        candidate_pairs,
        search_radius=search_radius,
    )
    blocks = _candidate_pairs_to_blocks(
        candidate_pairs,
        block_size=block_size,
        compact_pair_count=compact_pair_count,
    )
    stats = PairListStats(
        pair_count=blocks.candidate_count,
        n_cells=cell_list.n_cells,
        cell_count=cell_list.cell_count,
        occupied_cell_count=cell_list.occupied_cell_count,
        search_radius=search_radius,
        estimated_pair_bytes=blocks.estimated_bytes,
        estimated_cell_list_bytes=cell_list.estimated_bytes,
        backend="mlx_cell_blocks",
        representation_kind="blocks",
        candidate_count=candidate_count,
        estimated_candidate_bytes=blocks.estimated_bytes,
        compaction_backend=None,
        fallback_reason=None,
    )
    return NeighborList(
        mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32),
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


def _count_candidate_pairs_within_radius(
    positions_np: np.ndarray,
    cell: Cell,
    pairs: np.ndarray,
    *,
    search_radius: float,
) -> int:
    if pairs.shape[0] == 0:
        return 0
    displacement = positions_np[pairs[:, 0]] - positions_np[pairs[:, 1]]
    displacement = np.asarray(cell.minimum_image(as_mx_array(displacement)))
    distance2 = np.sum(displacement * displacement, axis=1)
    return int(np.count_nonzero(distance2 < search_radius * search_radius))


def _build_mlx_cell_pair_list(
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
                    (cell_index[axis] + offset[axis]) % cell_list.n_cells[axis]
                    for axis in range(3)
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
        compaction_backend="cpu_argwhere",
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
