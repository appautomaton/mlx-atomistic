"""Real- and reciprocal-space grids for toy plane-wave DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


def _shape_tuple(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        msg = "grid shape must contain exactly three dimensions"
        raise ValueError(msg)
    parsed = tuple(int(item) for item in shape)
    if any(item <= 0 for item in parsed):
        msg = "grid shape dimensions must be positive"
        raise ValueError(msg)
    return parsed


@dataclass(frozen=True)
class RealSpaceGrid:
    """Uniform orthorhombic real-space grid in atomic units."""

    shape: tuple[int, int, int]
    cell: Cell

    def __init__(self, shape: Sequence[int], cell: Cell | Sequence[float]):
        object.__setattr__(self, "shape", _shape_tuple(shape))
        parsed_cell = cell if isinstance(cell, Cell) else Cell.orthorhombic(cell)
        object.__setattr__(self, "cell", parsed_cell)
        lengths = np.array(parsed_cell.lengths, dtype=np.float64)
        if np.any(lengths <= 0.0):
            msg = "cell lengths must be positive"
            raise ValueError(msg)

    @property
    def ndim(self) -> int:
        """Number of grid dimensions."""

        return 3

    @property
    def size(self) -> int:
        """Total number of grid points."""

        return int(np.prod(self.shape))

    @property
    def lengths(self) -> mx.array:
        """Cell lengths in atomic units."""

        return self.cell.lengths

    @property
    def spacing(self) -> mx.array:
        """Grid spacing in atomic units."""

        return self.cell.lengths / as_mx_array(self.shape)

    @property
    def volume(self) -> float:
        """Cell volume in bohr cubed."""

        return float(np.prod(np.array(self.cell.lengths, dtype=np.float64)))

    @property
    def dv(self) -> float:
        """Volume represented by one grid point."""

        return self.volume / self.size

    def coordinates(self) -> mx.array:
        """Return cell-centered Cartesian grid coordinates with shape ``(*shape, 3)``."""

        lengths = np.array(self.cell.lengths, dtype=np.float64)
        axes = [
            (np.arange(count, dtype=np.float64) + 0.5) * length / count
            for count, length in zip(self.shape, lengths, strict=True)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return as_mx_array(np.stack(mesh, axis=-1).astype(np.float32))


@dataclass(frozen=True)
class ReciprocalGrid:
    """Reciprocal-space vectors matching a `RealSpaceGrid` FFT layout."""

    real_grid: RealSpaceGrid
    vectors: mx.array
    g2: mx.array
    zero_mask: mx.array

    @classmethod
    def from_real_space(cls, grid: RealSpaceGrid) -> ReciprocalGrid:
        """Build reciprocal vectors in NumPy FFT frequency order."""

        spacing = np.array(grid.spacing, dtype=np.float64)
        axes = [
            2.0 * pi * np.fft.fftfreq(count, d=delta)
            for count, delta in zip(grid.shape, spacing, strict=True)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        vectors_np = np.stack(mesh, axis=-1).astype(np.float32)
        g2_np = np.sum(vectors_np * vectors_np, axis=-1, dtype=np.float32)
        zero_mask_np = g2_np == 0.0
        return cls(
            real_grid=grid,
            vectors=as_mx_array(vectors_np),
            g2=as_mx_array(g2_np),
            zero_mask=mx.array(zero_mask_np),
        )
