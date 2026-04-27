"""Orbital normalization and density construction."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.grids import RealSpaceGrid


def _as_orbital_stack(orbitals: mx.array, grid: RealSpaceGrid) -> tuple[mx.array, bool]:
    array = mx.array(orbitals)
    if array.shape == grid.shape:
        return mx.reshape(array, (1, *grid.shape)), True
    if len(array.shape) == 4 and array.shape[1:] == grid.shape:
        return array, False
    msg = "orbitals must have shape grid.shape or (n_orbitals, *grid.shape)"
    raise ValueError(msg)


def _occupations_array(
    occupations: Sequence[float] | None,
    n_orbitals: int,
) -> mx.array:
    if occupations is None:
        values = np.full(n_orbitals, 2.0, dtype=np.float32)
    else:
        if len(occupations) != n_orbitals:
            msg = "occupations length must match number of orbitals"
            raise ValueError(msg)
        values = np.array(occupations, dtype=np.float32)
    if np.any(values < 0.0):
        msg = "occupations must be non-negative"
        raise ValueError(msg)
    if np.any(values > 2.0):
        msg = "spin-unpolarized occupations cannot exceed 2.0"
        raise ValueError(msg)
    return mx.array(values)


def normalize_orbitals(orbitals: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Normalize each orbital so that ``∫ |ψᵢ(r)|² dr = 1``."""

    stack, was_single = _as_orbital_stack(orbitals, grid)
    norms = mx.sum(mx.abs(stack) ** 2, axis=(1, 2, 3)) * grid.dv
    if bool(mx.any(norms <= 0.0)):
        msg = "cannot normalize an orbital with zero norm"
        raise ValueError(msg)
    normalized = stack / mx.reshape(mx.sqrt(norms), (-1, 1, 1, 1))
    if was_single:
        return normalized[0]
    return normalized


def density_from_orbitals(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float] | None = None,
) -> mx.array:
    """Build spin-unpolarized electron density from occupied orbitals.

    With no explicit occupations, each orbital is treated as doubly occupied:
    ``ρ(r) = 2Σᵢ |ψᵢ(r)|²``.
    """

    stack, _ = _as_orbital_stack(orbitals, grid)
    occupation_array = _occupations_array(occupations, int(stack.shape[0]))
    weights = mx.reshape(occupation_array, (-1, 1, 1, 1))
    density = mx.sum(weights * (mx.abs(stack) ** 2), axis=0)
    return mx.real(density)
