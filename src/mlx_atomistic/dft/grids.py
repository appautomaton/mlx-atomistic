"""Real- and reciprocal-space grids for toy plane-wave DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
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


def _fft_integer_g(shape: tuple[int, int, int]) -> mx.array:
    integer_axes = []
    for count in shape:
        indices = mx.arange(count, dtype=mx.int32)
        integer_axes.append(
            mx.where(indices <= (count - 1) // 2, indices, indices - count)
        )
    return mx.stack(
        [
            mx.broadcast_to(integer_axes[0][:, None, None], shape),
            mx.broadcast_to(integer_axes[1][None, :, None], shape),
            mx.broadcast_to(integer_axes[2][None, None, :], shape),
        ],
        axis=-1,
    )


def _reciprocal_fingerprint(grid: RealSpaceGrid) -> str:
    digest = sha256()
    digest.update(b"mlx-atomistic.reciprocal-grid.v1\0")
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(grid.cell.matrix, dtype=np.float64).tobytes())
    digest.update(b"exact-numpy-fftfreq-integer-order-v1\0")
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class ReciprocalGrid:
    """Reciprocal-space vectors matching a `RealSpaceGrid` FFT layout."""

    real_grid: RealSpaceGrid
    vectors: mx.array
    g2: mx.array
    zero_mask: mx.array
    integer_g: mx.array
    fingerprint: str

    def __init__(
        self,
        real_grid: RealSpaceGrid,
        vectors: mx.array,
        g2: mx.array,
        zero_mask: mx.array,
        integer_g: mx.array | None = None,
        fingerprint: str | None = None,
    ) -> None:
        object.__setattr__(self, "real_grid", real_grid)
        object.__setattr__(self, "vectors", mx.array(vectors))
        object.__setattr__(self, "g2", mx.array(g2))
        object.__setattr__(self, "zero_mask", mx.array(zero_mask))
        object.__setattr__(
            self,
            "integer_g",
            _fft_integer_g(real_grid.shape) if integer_g is None else mx.array(integer_g),
        )
        object.__setattr__(
            self,
            "fingerprint",
            _reciprocal_fingerprint(real_grid) if fingerprint is None else fingerprint,
        )

    @classmethod
    def from_real_space(cls, grid: RealSpaceGrid) -> ReciprocalGrid:
        """Build reciprocal vectors in exact NumPy FFT frequency order.

        Args:
            grid: Real-space grid whose FFT ordering and cell define the
                reciprocal descriptor.

        Returns:
            Reciprocal metadata with exact integer ``G`` coordinates and a
            deterministic grid fingerprint.
        """

        integer_g = _fft_integer_g(grid.shape)
        spacing = np.asarray(grid.spacing, dtype=np.float64)
        reciprocal_axes = [
            mx.array(
                (2.0 * pi * np.fft.fftfreq(count, d=delta)).astype(np.float32)
            )
            for count, delta in zip(grid.shape, spacing, strict=True)
        ]
        vectors = mx.stack(
            [
                mx.broadcast_to(reciprocal_axes[0][:, None, None], grid.shape),
                mx.broadcast_to(reciprocal_axes[1][None, :, None], grid.shape),
                mx.broadcast_to(reciprocal_axes[2][None, None, :], grid.shape),
            ],
            axis=-1,
        )
        g2 = mx.sum(vectors * vectors, axis=-1)
        return cls(
            real_grid=grid,
            vectors=vectors,
            g2=g2,
            zero_mask=g2 == 0.0,
            integer_g=integer_g,
            fingerprint=_reciprocal_fingerprint(grid),
        )
