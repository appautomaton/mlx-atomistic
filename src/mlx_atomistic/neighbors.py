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

NeighborBackend = Literal["auto", "periodic_cell_list", "mlx_dense_pairs", "mlx_cell_pairs"]
_ALLOWED_NEIGHBOR_BACKENDS = {
    "auto",
    "periodic_cell_list",
    "mlx_dense_pairs",
    "mlx_cell_pairs",
}
DEFAULT_MLX_DENSE_PAIR_LIMIT = 4096
DEFAULT_MLX_CELL_PAIR_CANDIDATE_CHUNK = 1_000_000
_BOOL_BYTES = np.dtype(np.bool_).itemsize
_FLOAT_BYTES = np.dtype(np.float32).itemsize


def validate_neighbor_backend(backend: str) -> NeighborBackend:
    """Validate and normalize a neighbor-list construction backend."""

    if backend not in _ALLOWED_NEIGHBOR_BACKENDS:
        expected = sorted(_ALLOWED_NEIGHBOR_BACKENDS)
        msg = f"unknown neighbor backend {backend!r}; expected one of {expected}"
        raise ValueError(msg)
    return backend  # type: ignore[return-value]


@dataclass(frozen=True)
class NeighborList:
    """Compact unique neighbor pairs for pairwise potentials."""

    pairs: mx.array
    cutoff: float
    skin: float = 0.0
    stats: PairListStats | None = None

    @property
    def pair_count(self) -> int:
        """Number of unique pairs."""

        return self.pairs.shape[0]

    @property
    def backend(self) -> str:
        """Pair-construction backend name."""

        return "periodic_cell_list" if self.stats is None else self.stats.backend

    @property
    def estimated_pair_bytes(self) -> int:
        """Estimated bytes for the compact int32 pair array."""

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
    def estimated_candidate_bytes(self) -> int:
        """Estimated bytes for backend candidate testing arrays."""

        return 0 if self.stats is None else self.stats.estimated_candidate_bytes

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
        if self.max_mlx_dense_atoms <= 0:
            msg = "max_mlx_dense_atoms must be positive"
            raise ValueError(msg)

    @property
    def rebuild_threshold(self) -> float:
        """Maximum displacement before the Verlet list must be rebuilt."""

        return 0.5 * self.skin

    def needs_rebuild(self, positions) -> bool:
        """Return true when positions have moved too far from the reference frame."""

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
        lengths = np.asarray(self.cell.lengths, dtype=np.float32)
        displacement = positions_np - reference_np
        displacement -= lengths * np.round(displacement / lengths)
        distance2 = np.sum(displacement * displacement, axis=1)
        self.last_max_displacement = float(np.sqrt(np.max(distance2))) if len(distance2) else 0.0
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
) -> NeighborList:
    """Build a periodic cell-list neighbor list with unique `i < j` pairs."""

    backend = validate_neighbor_backend(backend)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        msg = "cutoff must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(skin) or skin < 0.0:
        msg = "skin must be finite and non-negative"
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
            backend = "mlx_cell_pairs"
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
    pair_array, stats = build_periodic_pair_list(
        positions_np,
        cell,
        search_radius=search_radius,
        sort_pairs=sort_pairs,
        max_workers=max_workers,
    )
    if fallback_reason is not None:
        stats = replace(stats, fallback_reason=fallback_reason)
    return NeighborList(
        mx.array(pair_array, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
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
            f"n_atoms={n_atoms}, max_mlx_dense_atoms={max_atoms}; use periodic_cell_list"
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


def _build_mlx_cell_pair_list(
    positions_np: np.ndarray,
    cell: Cell,
    *,
    cutoff: float,
    skin: float,
    search_radius: float,
    sort_pairs: bool,
) -> NeighborList:
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
