"""Core atomistic data structures."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

DEFAULT_DTYPE = mx.float32


def use_cpu_device() -> None:
    """Route subsequent MLX operations to the CPU device."""

    cpu = mx.Device(mx.cpu, 0)
    mx.set_default_device(cpu)
    mx.set_default_stream(mx.new_stream(cpu))


def as_mx_array(value, *, dtype=DEFAULT_DTYPE) -> mx.array:
    """Convert a value to an MLX array with the project default dtype."""

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


@dataclass(frozen=True)
class Cell:
    """Orthorhombic periodic cell in MD reduced units."""

    lengths: mx.array

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

    def __post_init__(self) -> None:
        if self.lengths.shape != (3,):
            msg = "cell lengths must have shape (3,)"
            raise ValueError(msg)

    def wrap(self, positions: mx.array) -> mx.array:
        """Wrap positions back into the periodic cell."""

        return positions - mx.floor(positions / self.lengths) * self.lengths

    def minimum_image(self, displacement: mx.array) -> mx.array:
        """Apply the minimum-image convention to displacement vectors."""

        return displacement - self.lengths * mx.round(displacement / self.lengths)


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
