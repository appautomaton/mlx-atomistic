"""Periodic cell-list pair construction."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from math import isfinite

import numpy as np

from mlx_atomistic.core import Cell

_INT_BYTES = np.dtype(np.int32).itemsize
_FLOAT_BYTES = np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class PeriodicCellList:
    """Binned particles for orthorhombic periodic pair construction."""

    bins: dict[tuple[int, int, int], np.ndarray]
    n_cells: tuple[int, int, int]
    cell_lengths: np.ndarray
    search_radius: float
    n_particles: int
    estimated_bytes: int

    @property
    def cell_count(self) -> int:
        """Total grid cell count."""

        return int(np.prod(np.asarray(self.n_cells, dtype=np.int64)))

    @property
    def occupied_cell_count(self) -> int:
        """Number of cells containing at least one particle."""

        return len(self.bins)


@dataclass(frozen=True)
class PairListStats:
    """Diagnostics for a constructed periodic pair list."""

    pair_count: int
    n_cells: tuple[int, int, int]
    cell_count: int
    occupied_cell_count: int
    search_radius: float
    estimated_pair_bytes: int
    estimated_cell_list_bytes: int
    backend: str = "periodic_cell_list"


def estimate_pair_list_bytes(pair_count: int) -> int:
    """Return storage bytes for an int32 `(pair_count, 2)` pair array."""

    if pair_count < 0:
        msg = "pair_count must be non-negative"
        raise ValueError(msg)
    return int(pair_count) * 2 * _INT_BYTES


def build_periodic_cell_list(
    positions,
    cell: Cell,
    *,
    search_radius: float,
) -> PeriodicCellList:
    """Bin wrapped positions into a periodic orthorhombic grid."""

    if not isfinite(search_radius) or search_radius <= 0.0:
        msg = "search_radius must be finite and positive"
        raise ValueError(msg)

    positions_np = np.asarray(positions, dtype=np.float32)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        msg = "positions must have shape (n_particles, 3)"
        raise ValueError(msg)
    if not np.all(np.isfinite(positions_np)):
        msg = "positions must be finite"
        raise ValueError(msg)

    lengths = np.asarray(cell.lengths, dtype=np.float32)
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "cell lengths must be finite and positive"
        raise ValueError(msg)

    n_cells_array = np.maximum(np.floor(lengths / search_radius).astype(np.int32), 1)
    wrapped = positions_np - np.floor(positions_np / lengths) * lengths
    cell_indices = np.floor(wrapped / lengths * n_cells_array).astype(np.int32)
    cell_indices = np.minimum(cell_indices, n_cells_array - 1)

    bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for particle_index, index in enumerate(cell_indices):
        bins[tuple(int(axis_index) for axis_index in index)].append(particle_index)

    packed_bins = {
        key: np.asarray(members, dtype=np.int32)
        for key, members in sorted(bins.items(), key=lambda item: item[0])
    }
    estimated_bytes = (
        positions_np.shape[0] * 3 * _FLOAT_BYTES
        + positions_np.shape[0] * 3 * _INT_BYTES
        + positions_np.shape[0] * _INT_BYTES
        + len(packed_bins) * 3 * _INT_BYTES
    )
    return PeriodicCellList(
        bins=packed_bins,
        n_cells=tuple(int(value) for value in n_cells_array),
        cell_lengths=lengths,
        search_radius=float(search_radius),
        n_particles=int(positions_np.shape[0]),
        estimated_bytes=int(estimated_bytes),
    )


def build_periodic_pair_list(
    positions,
    cell: Cell,
    *,
    search_radius: float,
    sort_pairs: bool = True,
    max_workers: int | None = None,
) -> tuple[np.ndarray, PairListStats]:
    """Build deterministic unique `i < j` pairs within `search_radius`."""

    cell_list = build_periodic_cell_list(positions, cell, search_radius=search_radius)
    positions_np = np.asarray(positions, dtype=np.float32)
    positions_np = positions_np - np.floor(positions_np / cell_list.cell_lengths) * (
        cell_list.cell_lengths
    )
    search_radius2 = float(search_radius) * float(search_radius)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    pair_chunks: list[np.ndarray] = []
    cell_items = tuple(cell_list.bins.items())

    if max_workers is not None and max_workers > 1 and len(cell_items) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunks in executor.map(
                lambda item: _pair_chunks_for_cell(
                    item[0],
                    item[1],
                    cell_list.bins,
                    cell_list.n_cells,
                    positions_np,
                    cell_list.cell_lengths,
                    search_radius2,
                    offsets,
                ),
                cell_items,
            ):
                pair_chunks.extend(chunks)
    else:
        for cell_index, members in cell_items:
            pair_chunks.extend(
                _pair_chunks_for_cell(
                    cell_index,
                    members,
                    cell_list.bins,
                    cell_list.n_cells,
                    positions_np,
                    cell_list.cell_lengths,
                    search_radius2,
                    offsets,
                )
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
        search_radius=float(search_radius),
        estimated_pair_bytes=estimate_pair_list_bytes(int(pair_array.shape[0])),
        estimated_cell_list_bytes=cell_list.estimated_bytes,
    )
    return pair_array, stats


def _pair_chunks_for_cell(
    cell_index: tuple[int, int, int],
    members: np.ndarray,
    bins: dict[tuple[int, int, int], np.ndarray],
    n_cells: tuple[int, int, int],
    positions: np.ndarray,
    lengths: np.ndarray,
    search_radius2: float,
    offsets: tuple[tuple[int, int, int], ...],
) -> list[np.ndarray]:
    pair_chunks: list[np.ndarray] = []
    neighbor_indices = sorted(
        {
            tuple(
                (cell_index[axis] + offset[axis]) % n_cells[axis]
                for axis in range(3)
            )
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
            _append_same_cell_pairs(
                pair_chunks,
                positions,
                members,
                lengths,
                search_radius2,
            )
        else:
            _append_cross_cell_pairs(
                pair_chunks,
                positions,
                members,
                neighbors,
                lengths,
                search_radius2,
            )
    return pair_chunks


def _append_same_cell_pairs(
    pair_chunks: list[np.ndarray],
    positions: np.ndarray,
    members: np.ndarray,
    lengths: np.ndarray,
    search_radius2: float,
) -> None:
    if members.shape[0] < 2:
        return
    left, right = np.triu_indices(members.shape[0], k=1)
    _append_index_pairs_within_radius(
        pair_chunks,
        positions,
        members[left],
        members[right],
        lengths,
        search_radius2,
    )


def _append_cross_cell_pairs(
    pair_chunks: list[np.ndarray],
    positions: np.ndarray,
    members: np.ndarray,
    neighbors: np.ndarray,
    lengths: np.ndarray,
    search_radius2: float,
) -> None:
    if members.shape[0] == 0 or neighbors.shape[0] == 0:
        return
    left = np.repeat(members, neighbors.shape[0])
    right = np.tile(neighbors, members.shape[0])
    _append_index_pairs_within_radius(
        pair_chunks,
        positions,
        left,
        right,
        lengths,
        search_radius2,
    )


def _append_index_pairs_within_radius(
    pair_chunks: list[np.ndarray],
    positions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    lengths: np.ndarray,
    search_radius2: float,
) -> None:
    displacement = positions[left] - positions[right]
    displacement -= lengths * np.round(displacement / lengths)
    close = np.sum(displacement * displacement, axis=1) < search_radius2
    if not np.any(close):
        return
    selected_left = left[close]
    selected_right = right[close]
    pair_chunks.append(
        np.stack(
            (
                np.minimum(selected_left, selected_right),
                np.maximum(selected_left, selected_right),
            ),
            axis=1,
        )
    )
