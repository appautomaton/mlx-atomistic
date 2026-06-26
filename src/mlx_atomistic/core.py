"""Core atomistic data structures."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

DEFAULT_DTYPE = mx.float32


def use_cpu_device() -> None:
    """Route subsequent MLX operations to the CPU device.

    Sets the default MLX device and stream to the CPU. Used as a fallback when no
    Metal GPU is available, e.g. in CI or other headless environments.
    """

    cpu = mx.Device(mx.cpu, 0)
    mx.set_default_device(cpu)
    mx.set_default_stream(mx.new_stream(cpu))


def as_mx_array(
    value: Sequence | np.ndarray | mx.array, *, dtype: mx.Dtype = DEFAULT_DTYPE
) -> mx.array:
    """Convert a value to an MLX array in the project's default dtype.

    An existing ``mx.array`` is returned unchanged unless its dtype differs. If no
    Metal device is available the conversion transparently falls back to the CPU
    device (also forced when the ``MLX_ATOMISTIC_DEVICE=cpu`` environment variable
    is set).

    Args:
        value: Array-like data to convert (a sequence, NumPy array, or
            ``mx.array``).
        dtype: Target MLX dtype. Defaults to ``mx.float32``.

    Returns:
        The data as an ``mx.array`` of ``dtype``.
    """

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
    """Periodic simulation cell in MD reduced units.

    The cell is stored as a row-vector matrix: row ``i`` is the ``i``-th cell
    vector. One-dimensional input preserves the historical orthorhombic API; a
    full ``(3, 3)`` matrix stores an arbitrary triclinic box.

    Attributes:
        matrix: ``(3, 3)`` row-vector cell matrix.
    """

    matrix: mx.array
    _is_orthorhombic: bool

    def __init__(self, lengths: Sequence[float] | Sequence[Sequence[float]] | mx.array):
        """Build a cell from edge lengths or a full cell matrix.

        Args:
            lengths: Either three positive box lengths with shape ``(3,)`` for an
                orthorhombic cell, or a ``(3, 3)`` row-vector matrix for a
                triclinic cell.

        Raises:
            ValueError: If the lengths are non-positive, the matrix is singular or
                has a non-positive determinant, or the shape is neither ``(3,)``
                nor ``(3, 3)``.
        """
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
        """Create a cubic periodic cell.

        Args:
            length: Edge length of the cube.

        Returns:
            A cubic :class:`Cell` with all three edges equal to ``length``.
        """

        return cls(as_mx_array([length, length, length]))

    @classmethod
    def orthorhombic(cls, lengths: Sequence[float]) -> Cell:
        """Create an orthorhombic (axis-aligned) periodic cell.

        Args:
            lengths: The three box edge lengths ``(a, b, c)``.

        Returns:
            An axis-aligned :class:`Cell`.

        Raises:
            ValueError: If ``lengths`` does not contain exactly three values.
        """

        if len(lengths) != 3:
            msg = "orthorhombic cell requires exactly three lengths"
            raise ValueError(msg)
        return cls(as_mx_array(lengths))

    @classmethod
    def triclinic(cls, matrix: Sequence[Sequence[float]]) -> Cell:
        """Create a triclinic periodic cell from row-vector cell vectors.

        Args:
            matrix: ``(3, 3)`` matrix whose rows are the three cell vectors.

        Returns:
            A :class:`Cell` for the given (possibly non-orthogonal) box.
        """

        return cls(matrix)

    @property
    def lengths(self) -> mx.array:
        """Cell-vector lengths ``(3,)``: the Euclidean norm of each row of
        :attr:`matrix`."""

        return mx.sqrt(mx.sum(self.matrix * self.matrix, axis=1))

    @property
    def is_orthorhombic(self) -> bool:
        """Whether the cell is axis-aligned orthorhombic (no off-diagonal terms)."""

        return self._is_orthorhombic

    @property
    def volume(self) -> mx.array:
        """Cell volume, computed as the determinant of :attr:`matrix`."""

        matrix = self.matrix
        determinant = (
            matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
            - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
            + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
        )
        return determinant

    def fractional_coordinates(self, positions: mx.array) -> mx.array:
        """Map Cartesian coordinates into the fractional cell basis.

        Args:
            positions: Cartesian row-vector coordinates, shape ``(..., 3)``.

        Returns:
            Fractional coordinates (Cartesian expressed in the cell basis), with
            the same shape as ``positions``.
        """

        positions = as_mx_array(positions)
        if self.is_orthorhombic:
            return positions / mx.diag(self.matrix)
        return positions @ mx.linalg.inv(self.matrix)

    def cartesian_coordinates(self, fractional: mx.array) -> mx.array:
        """Map fractional cell coordinates back to Cartesian coordinates.

        Args:
            fractional: Fractional row-vector coordinates, shape ``(..., 3)``.

        Returns:
            Cartesian coordinates, with the same shape as ``fractional``.
        """

        return as_mx_array(fractional) @ self.matrix

    def wrap(self, positions: mx.array) -> mx.array:
        """Wrap positions back into the primary periodic cell.

        For orthorhombic cells this subtracts an integer number of box lengths
        directly (``x - L*floor(x/L)``) instead of round-tripping through
        fractional coordinates. The round-trip form ``(x/L - floor(x/L))*L`` is
        algebraically identical but, in float32, does not return a position that
        is exactly ``x`` minus an integer multiple of ``L`` -- it nudges atoms
        near a cell boundary by up to ~1e-2. Applied every MD step, that spurious
        displacement does work against the forces and injects energy, breaking
        energy conservation over long runs (invisible in short-run tests). The
        direct form keeps the wrap a pure lattice translation, consistent with
        :meth:`minimum_image`.

        Args:
            positions: Cartesian row-vector coordinates, shape ``(..., 3)``.

        Returns:
            Positions translated into the primary cell, with the same shape as
            ``positions``.
        """

        positions = as_mx_array(positions)
        if self.is_orthorhombic:
            lengths = mx.diag(self.matrix)
            return positions - lengths * mx.floor(positions / lengths)
        fractional = self.fractional_coordinates(positions)
        return self.cartesian_coordinates(fractional - mx.floor(fractional))

    def minimum_image(self, displacement: mx.array) -> mx.array:
        """Apply the minimum-image convention to displacement vectors.

        Returns the shortest periodic image of each displacement, so that pair
        distances use the nearest copy of each atom across cell boundaries.

        Args:
            displacement: Cartesian displacement vectors, shape ``(..., 3)``.

        Returns:
            Minimum-image displacements, with the same shape as the input.
        """

        displacement = as_mx_array(displacement)
        if self.is_orthorhombic:
            lengths = mx.diag(self.matrix)
            return displacement - lengths * mx.round(displacement / lengths)
        fractional = self.fractional_coordinates(displacement)
        return self.cartesian_coordinates(fractional - mx.round(fractional))


@dataclass(frozen=True)
class Atoms:
    """A collection of atoms (or particles) with MLX-backed arrays.

    Attributes:
        symbols: Per-atom chemical symbols, one entry per atom.
        positions: ``(n_atoms, 3)`` Cartesian coordinates.
        masses: ``(n_atoms,)`` per-atom masses.
        velocities: Optional ``(n_atoms, 3)`` per-atom velocities.
    """

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
        """Build an :class:`Atoms` collection from plain Python sequences.

        Args:
            symbols: Chemical symbols, one per atom.
            positions: ``(n_atoms, 3)`` Cartesian coordinates.
            masses: Optional per-atom masses; defaults to unit mass for every
                atom.
            velocities: Optional ``(n_atoms, 3)`` initial velocities.

        Returns:
            An :class:`Atoms` instance with MLX-backed arrays.
        """

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
        """Number of atoms in the collection."""

        return len(self.symbols)
