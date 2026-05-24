"""Core atomistic data structures."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

DEFAULT_DTYPE = mx.float32


def use_cpu_device() -> None:
    """Route subsequent MLX operations to the CPU device."""

    cpu = mx.Device(mx.cpu, 0)
    mx.set_default_device(cpu)
    mx.set_default_stream(mx.new_stream(cpu))


def as_mx_array(value, *, dtype=DEFAULT_DTYPE) -> mx.array:
    """Convert a value to an MLX array with the project default dtype."""

    if isinstance(value, mx.array):
        return value if value.dtype == dtype else value.astype(dtype)
    if os.environ.get("MLX_ATOMISTIC_DEVICE") == "cpu":
        use_cpu_device()
        with mx.stream(mx.cpu):
            return mx.array(value, dtype=dtype)
    try:
        return mx.array(value, dtype=dtype)
    except RuntimeError as err:
        if "No Metal device available" not in str(err):
            raise
        use_cpu_device()
        with mx.stream(mx.cpu):
            return mx.array(value, dtype=dtype)


@dataclass(frozen=True, init=False)
class Cell:
    """Periodic cell in MD reduced units.

    One-dimensional input preserves the historical orthorhombic API. Full
    `(3, 3)` input stores row-vector cell vectors for triclinic boxes.
    """

    matrix: mx.array
    _is_orthorhombic: bool

    def __init__(self, lengths: Sequence[float] | Sequence[Sequence[float]] | mx.array):
        values = as_mx_array(lengths)
        if values.shape == (3,):
            if np.any(np.asarray(values) <= 0.0):
                msg = "positive orthorhombic cell lengths required"
                raise ValueError(msg)
            matrix = mx.diag(values)
            is_orthorhombic = True
        elif values.shape == (3, 3):
            values_np = np.asarray(values, dtype=np.float64)
            determinant = float(np.linalg.det(values_np))
            if not np.isfinite(determinant) or determinant <= 0.0:
                msg = "cell matrix must have a positive non-singular determinant"
                raise ValueError(msg)
            matrix = values
            off_diagonal = values_np - np.diag(np.diag(values_np))
            is_orthorhombic = bool(np.allclose(off_diagonal, 0.0, atol=1e-7))
        else:
            msg = "cell must have shape (3,) or (3, 3)"
            raise ValueError(msg)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "_is_orthorhombic", is_orthorhombic)

    @classmethod
    def cubic(cls, length: float) -> Cell:
        """Create a cubic periodic cell."""

        return cls(as_mx_array([length, length, length]))

    @classmethod
    def orthorhombic(cls, lengths: Sequence[float]) -> Cell:
        """Create an orthorhombic periodic cell."""

        if len(lengths) != 3:
            msg = "orthorhombic cell requires exactly three lengths"
            raise ValueError(msg)
        return cls(as_mx_array(lengths))

    @classmethod
    def triclinic(cls, matrix: Sequence[Sequence[float]]) -> Cell:
        """Create a triclinic periodic cell from row-vector cell vectors."""

        return cls(matrix)

    @property
    def lengths(self) -> mx.array:
        """Return cell-vector lengths for compatibility with existing callers."""

        return mx.sqrt(mx.sum(self.matrix * self.matrix, axis=1))

    @property
    def is_orthorhombic(self) -> bool:
        """Return true when the cell is axis-aligned orthorhombic."""

        return self._is_orthorhombic

    @property
    def volume(self) -> mx.array:
        """Return the periodic cell volume."""

        matrix = self.matrix
        determinant = (
            matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
            - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
            + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
        )
        return determinant

    def fractional_coordinates(self, positions: mx.array) -> mx.array:
        """Convert Cartesian row-vector coordinates to fractional coordinates."""

        positions = as_mx_array(positions)
        if self.is_orthorhombic:
            return positions / mx.diag(self.matrix)
        return positions @ mx.linalg.inv(self.matrix)

    def cartesian_coordinates(self, fractional: mx.array) -> mx.array:
        """Convert fractional row-vector coordinates to Cartesian coordinates."""

        return as_mx_array(fractional) @ self.matrix

    def wrap(self, positions: mx.array) -> mx.array:
        """Wrap positions back into the periodic cell."""

        fractional = self.fractional_coordinates(positions)
        return self.cartesian_coordinates(fractional - mx.floor(fractional))

    def minimum_image(self, displacement: mx.array) -> mx.array:
        """Apply the minimum-image convention to displacement vectors."""

        displacement = as_mx_array(displacement)
        if self.is_orthorhombic:
            lengths = mx.diag(self.matrix)
            return displacement - lengths * mx.round(displacement / lengths)
        fractional = self.fractional_coordinates(displacement)
        return self.cartesian_coordinates(fractional - mx.round(fractional))


@dataclass(frozen=True)
class Atoms:
    """Atoms or particles with MLX-backed coordinates."""

    symbols: tuple[str, ...]
    positions: mx.array
    masses: mx.array
    velocities: mx.array | None = None

    @classmethod
    def from_sequences(
        cls,
        symbols: Sequence[str],
        positions: Sequence[Sequence[float]],
        *,
        masses: Sequence[float] | None = None,
        velocities: Sequence[Sequence[float]] | None = None,
    ) -> Atoms:
        """Create an atom collection from Python sequences."""

        atom_count = len(symbols)
        if masses is None:
            masses = [1.0] * atom_count

        return cls(
            symbols=tuple(symbols),
            positions=as_mx_array(positions),
            masses=as_mx_array(masses),
            velocities=None if velocities is None else as_mx_array(velocities),
        )

    def __post_init__(self) -> None:
        atom_count = len(self.symbols)
        if self.positions.shape != (atom_count, 3):
            msg = "positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        if self.masses.shape != (atom_count,):
            msg = "masses must have shape (n_atoms,)"
            raise ValueError(msg)
        if self.velocities is not None and self.velocities.shape != (atom_count, 3):
            msg = "velocities must have shape (n_atoms, 3)"
            raise ValueError(msg)

    @property
    def count(self) -> int:
        """Number of atoms or particles."""

        return len(self.symbols)
