"""Spin and orbital occupation models for DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.grids import RealSpaceGrid

SpinMode = Literal["unpolarized", "polarized"]


@dataclass(frozen=True)
class OccupationResult:
    """Orbital occupations and chemical-potential diagnostics."""

    occupations: mx.array
    electron_count: float
    chemical_potential: float | None = None
    temperature: float | None = None
    spin_mode: SpinMode = "unpolarized"

    def to_dict(self) -> dict:
        """Return a JSON-safe occupation summary."""

        return {
            "occupations": np.array(self.occupations).tolist(),
            "electron_count": self.electron_count,
            "chemical_potential": self.chemical_potential,
            "temperature": self.temperature,
            "spin_mode": self.spin_mode,
        }


@dataclass(frozen=True)
class FixedOccupations:
    """Explicit fixed orbital occupations."""

    occupations: tuple[float, ...]
    spin_mode: SpinMode = "unpolarized"

    def __init__(
        self,
        occupations: Sequence[float],
        *,
        spin_mode: SpinMode = "unpolarized",
    ):
        values = tuple(float(value) for value in occupations)
        if not values:
            msg = "occupations cannot be empty"
            raise ValueError(msg)
        max_value = 2.0 if spin_mode == "unpolarized" else 1.0
        if any(value < 0.0 or value > max_value for value in values):
            msg = f"occupations must be in [0, {max_value}] for {spin_mode} mode"
            raise ValueError(msg)
        object.__setattr__(self, "occupations", values)
        object.__setattr__(self, "spin_mode", spin_mode)

    def resolve(self, _eigenvalues: Sequence[float] | mx.array | None = None) -> OccupationResult:
        """Return fixed occupations."""

        return OccupationResult(
            occupations=mx.array(self.occupations, dtype=mx.float32),
            electron_count=float(sum(self.occupations)),
            spin_mode=self.spin_mode,
        )


@dataclass(frozen=True)
class FermiDiracOccupations:
    """Fermi-Dirac occupations with electron-count conservation."""

    electron_count: float
    temperature: float = 0.01
    spin_mode: SpinMode = "unpolarized"
    tolerance: float = 1e-10
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.electron_count <= 0.0:
            msg = "electron_count must be positive"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if self.spin_mode not in {"unpolarized", "polarized"}:
            msg = "spin_mode must be 'unpolarized' or 'polarized'"
            raise ValueError(msg)

    def resolve(self, eigenvalues: Sequence[float] | mx.array) -> OccupationResult:
        """Return Fermi-Dirac occupations for sorted or unsorted eigenvalues."""

        values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        if values.size == 0:
            msg = "eigenvalues cannot be empty"
            raise ValueError(msg)
        degeneracy = 2.0 if self.spin_mode == "unpolarized" else 1.0
        if self.electron_count > degeneracy * values.size + 1e-12:
            msg = "not enough orbitals for requested electron count"
            raise ValueError(msg)
        low = float(np.min(values) - 100.0 * self.temperature - 10.0)
        high = float(np.max(values) + 100.0 * self.temperature + 10.0)
        mu = 0.5 * (low + high)
        occupations = np.zeros_like(values)
        for _ in range(self.max_iterations):
            mu = 0.5 * (low + high)
            x = np.clip((values - mu) / self.temperature, -80.0, 80.0)
            occupations = degeneracy / (np.exp(x) + 1.0)
            count = float(np.sum(occupations))
            if abs(count - self.electron_count) <= self.tolerance:
                break
            if count > self.electron_count:
                high = mu
            else:
                low = mu
        return OccupationResult(
            occupations=mx.array(occupations.astype(np.float32)),
            electron_count=float(np.sum(occupations)),
            chemical_potential=mu,
            temperature=self.temperature,
            spin_mode=self.spin_mode,
        )


def spin_density_from_orbitals(
    up_orbitals: mx.array,
    down_orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    up_occupations: Sequence[float],
    down_occupations: Sequence[float],
) -> tuple[mx.array, mx.array]:
    """Build collinear spin densities ``ρ↑`` and ``ρ↓``."""

    return (
        _density(up_orbitals, grid, up_occupations),
        _density(down_orbitals, grid, down_occupations),
    )


def magnetization_density(up_density: mx.array, down_density: mx.array) -> mx.array:
    """Return ``m(r) = ρ↑(r) - ρ↓(r)``."""

    return mx.real(mx.array(up_density) - mx.array(down_density))


def _density(orbitals: mx.array, grid: RealSpaceGrid, occupations: Sequence[float]) -> mx.array:
    stack = mx.array(orbitals)
    if stack.shape == grid.shape:
        stack = mx.reshape(stack, (1, *grid.shape))
    if len(occupations) != int(stack.shape[0]):
        msg = "occupations length must match number of orbitals"
        raise ValueError(msg)
    weights = mx.reshape(mx.array(occupations, dtype=mx.float32), (-1, 1, 1, 1))
    return mx.real(mx.sum(weights * mx.abs(stack) ** 2, axis=0))
