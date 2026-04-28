"""Lightweight DFT system model for toy Γ-point calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.dft.grids import RealSpaceGrid
from mlx_atomistic.dft.potentials import LocalGaussianPseudopotential


@dataclass(frozen=True)
class DFTSystem:
    """Minimal spin-unpolarized DFT system with local Gaussian centers."""

    grid_shape: tuple[int, int, int]
    cell: Cell
    electron_count: float
    pseudopotential: LocalGaussianPseudopotential
    charges: tuple[float, ...]

    def __init__(
        self,
        *,
        cell: Cell | Sequence[float],
        grid_shape: Sequence[int],
        electron_count: float,
        centers: Sequence[Sequence[float]] | None = None,
        amplitudes: Sequence[float] | float | None = None,
        widths: Sequence[float] | float | None = None,
        charges: Sequence[float] | None = None,
        pseudopotential: LocalGaussianPseudopotential | None = None,
    ):
        parsed_cell = cell if isinstance(cell, Cell) else Cell.orthorhombic(cell)
        shape = tuple(int(item) for item in grid_shape)
        if len(shape) != 3 or any(item <= 0 for item in shape):
            msg = "grid_shape must contain three positive dimensions"
            raise ValueError(msg)
        if electron_count <= 0.0:
            msg = "electron_count must be positive"
            raise ValueError(msg)
        if pseudopotential is None:
            if centers is None or amplitudes is None or widths is None:
                msg = "centers, amplitudes, and widths are required without pseudopotential"
                raise ValueError(msg)
            pseudopotential = LocalGaussianPseudopotential(centers, amplitudes, widths)
        n_centers = int(pseudopotential.centers.shape[0])
        if charges is None:
            parsed_charges = tuple(
                float(-amplitude) for amplitude in np.array(pseudopotential.amplitudes)
            )
        else:
            if len(charges) != n_centers:
                msg = "charges length must match number of pseudopotential centers"
                raise ValueError(msg)
            parsed_charges = tuple(float(value) for value in charges)

        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "cell", parsed_cell)
        object.__setattr__(self, "electron_count", float(electron_count))
        object.__setattr__(self, "pseudopotential", pseudopotential)
        object.__setattr__(self, "charges", parsed_charges)

    @property
    def grid(self) -> RealSpaceGrid:
        """Return the real-space grid for this system."""

        return RealSpaceGrid(self.grid_shape, self.cell)

    @property
    def centers(self):
        """Pseudopotential center coordinates."""

        return self.pseudopotential.centers

    @property
    def center_count(self) -> int:
        """Number of local pseudopotential centers."""

        return int(self.pseudopotential.centers.shape[0])
