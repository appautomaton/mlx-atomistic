"""Neighbor-list construction for periodic MD."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.cell_list import PairListStats, build_periodic_pair_list
from mlx_atomistic.core import Cell, as_mx_array


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


@dataclass
class NeighborListManager:
    """Manage Verlet neighbor-list rebuilds during an MD trajectory."""

    cell: Cell
    cutoff: float
    skin: float = 0.3
    check_interval: int = 1
    sort_pairs: bool = True
    max_workers: int | None = None
    neighbor_list: NeighborList | None = None
    reference_positions: mx.array | None = None
    rebuild_count: int = 0
    last_max_displacement: float = 0.0
    updates_since_check: int = 0

    def __post_init__(self) -> None:
        if self.check_interval <= 0:
            msg = "check_interval must be positive"
            raise ValueError(msg)

    @property
    def rebuild_threshold(self) -> float:
        """Maximum displacement before the Verlet list must be rebuilt."""

        return 0.5 * self.skin

    def needs_rebuild(self, positions) -> bool:
        """Return true when positions have moved too far from the reference frame."""

        positions_np = np.asarray(positions, dtype=np.float32)
        if positions_np.ndim != 2 or positions_np.shape[1] != 3:
            msg = "positions must have shape (n_particles, 3)"
            raise ValueError(msg)
        if not np.all(np.isfinite(positions_np)):
            msg = "positions must be finite"
            raise ValueError(msg)

        if self.neighbor_list is None or self.reference_positions is None:
            self.last_max_displacement = float("inf")
            return True
        self.updates_since_check += 1
        if self.updates_since_check < self.check_interval:
            return False
        self.updates_since_check = 0

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

        self.neighbor_list = build_neighbor_list(
            positions,
            self.cell,
            cutoff=self.cutoff,
            skin=self.skin,
            sort_pairs=self.sort_pairs,
            max_workers=self.max_workers,
        )
        self.reference_positions = as_mx_array(positions)
        self.rebuild_count += 1
        self.last_max_displacement = 0.0
        self.updates_since_check = 0
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
    sort_pairs: bool = True,
    max_workers: int | None = None,
) -> NeighborList:
    """Build a periodic cell-list neighbor list with unique `i < j` pairs."""

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
    search_radius = cutoff + skin
    pair_array, stats = build_periodic_pair_list(
        positions_np,
        cell,
        search_radius=search_radius,
        sort_pairs=sort_pairs,
        max_workers=max_workers,
    )
    return NeighborList(
        mx.array(pair_array, dtype=mx.int32),
        cutoff=cutoff,
        skin=skin,
        stats=stats,
    )
