"""Shared validation and geometry for periodic pseudopotential operators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx_atomistic.dft.pseudopotentials import (
    PseudopotentialData,
    PseudopotentialFormat,
)


def _periodic_positions(positions: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.array(positions, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("positions must have shape (n_ions, 3)")
    if not np.isfinite(values).all():
        raise ValueError("positions must contain only finite values")
    values.setflags(write=False)
    return values


def _periodic_pseudopotentials(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    ion_count: int,
    *,
    expected_format: PseudopotentialFormat,
) -> tuple[PseudopotentialData, ...]:
    if isinstance(pseudopotential, PseudopotentialData):
        values = (pseudopotential,) * ion_count
    else:
        values = tuple(pseudopotential)
        if len(values) != ion_count:
            raise ValueError("pseudopotentials length must match the ion count")
    if any(not isinstance(value, PseudopotentialData) for value in values):
        raise TypeError("pseudopotentials must contain PseudopotentialData values")
    if any(value.format != expected_format for value in values):
        raise ValueError(
            f"periodic operator requires {expected_format.value.upper()} pseudopotentials"
        )
    return values


def _periodic_structure_factor(
    vectors: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    phase = np.einsum("...d,id->i...", vectors, positions, optimize=True)
    return np.sum(np.exp(-1j * phase), axis=0)
