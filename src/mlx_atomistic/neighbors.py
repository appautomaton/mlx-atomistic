"""Neighbor-list construction for periodic MD."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


@dataclass(frozen=True)
class NeighborList:
    """Compact unique neighbor pairs for pairwise potentials."""

    pairs: mx.array
    cutoff: float
    skin: float = 0.0

    @property
    def pair_count(self) -> int:
        """Number of unique pairs."""

        return self.pairs.shape[0]


@dataclass
class NeighborListManager:
    """Manage Verlet neighbor-list rebuilds during an MD trajectory."""

    cell: Cell
    cutoff: float
    skin: float = 0.3
    neighbor_list: NeighborList | None = None
    reference_positions: mx.array | None = None
    rebuild_count: int = 0
    last_max_displacement: float = 0.0

    @property
    def rebuild_threshold(self) -> float:
        """Maximum displacement before the Verlet list must be rebuilt."""

        return 0.5 * self.skin

    def needs_rebuild(self, positions) -> bool:
        """Return true when positions have moved too far from the reference frame."""

        if self.neighbor_list is None or self.reference_positions is None:
            self.last_max_displacement = float("inf")
            return True

        positions_np = np.asarray(positions, dtype=np.float32)
        reference_np = np.asarray(self.reference_positions, dtype=np.float32)
        lengths = np.asarray(self.cell.lengths, dtype=np.float32)
        displacement = positions_np - reference_np
        displacement -= lengths * np.round(displacement / lengths)
        distance2 = np.sum(displacement * displacement, axis=1)
        self.last_max_displacement = float(np.sqrt(np.max(distance2))) if len(distance2) else 0.0
        return self.last_max_displacement > self.rebuild_threshold

    def rebuild(self, positions) -> NeighborList:
        """Force a neighbor-list rebuild from current positions."""

        self.neighbor_list = build_neighbor_list(
            positions,
            self.cell,
            cutoff=self.cutoff,
            skin=self.skin,
        )
        self.reference_positions = as_mx_array(positions)
        self.rebuild_count += 1
        self.last_max_displacement = 0.0
        return self.neighbor_list

    def update(self, positions) -> NeighborList:
        """Return a current neighbor list, rebuilding if needed."""

        if self.needs_rebuild(positions):
            return self.rebuild(positions)
        if self.neighbor_list is None:
            msg = "neighbor list manager has no current neighbor list"
            raise RuntimeError(msg)
        return self.neighbor_list


def build_neighbor_list(
    positions,
    cell: Cell,
    *,
    cutoff: float,
    skin: float = 0.3,
) -> NeighborList:
    """Build a periodic cell-list neighbor list with unique `i < j` pairs."""

    if cutoff <= 0.0:
        msg = "cutoff must be positive"
        raise ValueError(msg)
    if skin < 0.0:
        msg = "skin must be non-negative"
        raise ValueError(msg)

    positions_np = np.asarray(positions, dtype=np.float32)
    if positions_np.ndim != 2 or positions_np.shape[1] != 3:
        msg = "positions must have shape (n_particles, 3)"
        raise ValueError(msg)

    lengths = np.asarray(cell.lengths, dtype=np.float32)
    search_radius = cutoff + skin
    n_cells = np.maximum(np.floor(lengths / search_radius).astype(np.int32), 1)
    wrapped = positions_np - np.floor(positions_np / lengths) * lengths
    cell_indices = np.floor(wrapped / lengths * n_cells).astype(np.int32)
    cell_indices = np.minimum(cell_indices, n_cells - 1)

    bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for particle_index, index in enumerate(cell_indices):
        bins[tuple(int(x) for x in index)].append(particle_index)

    pairs: set[tuple[int, int]] = set()
    search_radius2 = search_radius * search_radius
    offsets = list(product((-1, 0, 1), repeat=3))
    for cell_index, members in bins.items():
        for offset in offsets:
            neighbor_index = tuple(
                (cell_index[axis] + offset[axis]) % n_cells[axis] for axis in range(3)
            )
            neighbors = bins.get(neighbor_index)
            if not neighbors:
                continue
            for i in members:
                for j in neighbors:
                    if i >= j:
                        continue
                    displacement = positions_np[i] - positions_np[j]
                    displacement -= lengths * np.round(displacement / lengths)
                    if float(np.dot(displacement, displacement)) < search_radius2:
                        pairs.add((i, j))

    pair_array = np.array(sorted(pairs), dtype=np.int32).reshape((-1, 2))
    return NeighborList(mx.array(pair_array, dtype=mx.int32), cutoff=cutoff, skin=skin)
