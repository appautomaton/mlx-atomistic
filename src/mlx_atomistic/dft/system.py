"""Lightweight DFT system model for toy Γ-point calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.dft.grids import RealSpaceGrid
from mlx_atomistic.dft.potentials import LocalGaussianPseudopotential
from mlx_atomistic.dft.pseudopotentials import IonCollection, LocalPseudopotentialField


def center_center_energy(system: DFTSystem) -> float:
    """Return pairwise center-center Coulomb energy for toy DFT centers."""

    centers = np.array(system.centers, dtype=np.float64)
    charges = np.array(system.charges, dtype=np.float64)
    energy = 0.0
    for i in range(system.center_count):
        for j in range(i + 1, system.center_count):
            displacement = system.cell.minimum_image(centers[i] - centers[j])
            distance = float(np.linalg.norm(np.array(displacement, dtype=np.float64)))
            if distance <= 1e-12:
                msg = "center-center energy is undefined for overlapping centers"
                raise ValueError(msg)
            energy += float(charges[i] * charges[j] / distance)
    return energy


def center_center_forces(system: DFTSystem) -> np.ndarray:
    """Return pairwise Coulomb forces between toy DFT centers."""

    centers = np.array(system.centers, dtype=np.float64)
    charges = np.array(system.charges, dtype=np.float64)
    forces = np.zeros_like(centers)
    for i in range(system.center_count):
        for j in range(i + 1, system.center_count):
            displacement = np.array(
                system.cell.minimum_image(centers[i] - centers[j]),
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(displacement))
            if distance <= 1e-12:
                msg = "center-center force is undefined for overlapping centers"
                raise ValueError(msg)
            pair_force = charges[i] * charges[j] * displacement / (distance**3)
            forces[i] += pair_force
            forces[j] -= pair_force
    return forces


@dataclass(frozen=True)
class DFTSystem:
    """Minimal spin-unpolarized DFT system with local Gaussian centers."""

    grid_shape: tuple[int, int, int]
    cell: Cell
    electron_count: float
    pseudopotential: LocalGaussianPseudopotential | LocalPseudopotentialField
    charges: tuple[float, ...]
    ions: IonCollection | None

    def __init__(
        self,
        *,
        cell: Cell | Sequence[float],
        grid_shape: Sequence[int],
        electron_count: float | None = None,
        centers: Sequence[Sequence[float]] | None = None,
        amplitudes: Sequence[float] | float | None = None,
        widths: Sequence[float] | float | None = None,
        charges: Sequence[float] | None = None,
        pseudopotential: LocalGaussianPseudopotential | LocalPseudopotentialField | None = None,
        ions: IonCollection | None = None,
    ):
        parsed_cell = cell if isinstance(cell, Cell) else Cell.orthorhombic(cell)
        shape = tuple(int(item) for item in grid_shape)
        if len(shape) != 3 or any(item <= 0 for item in shape):
            msg = "grid_shape must contain three positive dimensions"
            raise ValueError(msg)
        if ions is not None and (
            pseudopotential is not None
            or centers is not None
            or amplitudes is not None
            or widths is not None
        ):
            msg = "ions cannot be combined with explicit pseudopotential or Gaussian parameters"
            raise ValueError(msg)
        if ions is not None:
            pseudopotential = LocalPseudopotentialField(ions)
            if electron_count is None:
                electron_count = ions.valence_electron_count
        if pseudopotential is None:
            if centers is None or amplitudes is None or widths is None:
                msg = "centers, amplitudes, and widths are required without pseudopotential"
                raise ValueError(msg)
            pseudopotential = LocalGaussianPseudopotential(centers, amplitudes, widths)
        if electron_count is None:
            msg = "electron_count is required without ions"
            raise ValueError(msg)
        if electron_count <= 0.0:
            msg = "electron_count must be positive"
            raise ValueError(msg)
        n_centers = int(pseudopotential.centers.shape[0])
        if charges is None:
            if ions is None:
                parsed_charges = tuple(
                    float(-amplitude) for amplitude in np.array(pseudopotential.amplitudes)
                )
            else:
                parsed_charges = ions.charges
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
        object.__setattr__(self, "ions", ions)

    @classmethod
    def one_center(
        cls,
        *,
        cell: Cell | Sequence[float] = (8.0, 8.0, 8.0),
        grid_shape: Sequence[int] = (8, 8, 8),
        center: Sequence[float] = (4.0, 4.0, 4.0),
        electron_count: float = 2.0,
        amplitude: float = -3.0,
        width: float = 0.9,
        charge: float | None = None,
    ) -> DFTSystem:
        """Build a one-center toy DFT system."""

        return cls(
            cell=cell,
            grid_shape=grid_shape,
            electron_count=electron_count,
            centers=[center],
            amplitudes=amplitude,
            widths=width,
            charges=None if charge is None else [charge],
        )

    @classmethod
    def two_center(
        cls,
        *,
        cell: Cell | Sequence[float] = (8.0, 8.0, 8.0),
        grid_shape: Sequence[int] = (8, 8, 8),
        centers: Sequence[Sequence[float]] = ((3.4, 4.0, 4.0), (4.6, 4.0, 4.0)),
        electron_count: float = 2.0,
        amplitudes: Sequence[float] | float = (-2.0, -2.0),
        widths: Sequence[float] | float = (0.8, 0.8),
        charges: Sequence[float] | None = None,
    ) -> DFTSystem:
        """Build a two-center toy DFT system."""

        return cls(
            cell=cell,
            grid_shape=grid_shape,
            electron_count=electron_count,
            centers=centers,
            amplitudes=amplitudes,
            widths=widths,
            charges=charges,
        )

    @classmethod
    def cluster(
        cls,
        *,
        cell: Cell | Sequence[float] = (10.0, 10.0, 10.0),
        grid_shape: Sequence[int] = (8, 8, 8),
        electron_count: float = 4.0,
        centers: Sequence[Sequence[float]] = (
            (4.0, 4.0, 5.0),
            (6.0, 4.0, 5.0),
            (5.0, 6.0, 5.0),
        ),
        amplitudes: Sequence[float] | float = (-2.0, -2.0, -1.5),
        widths: Sequence[float] | float = (0.85, 0.85, 0.95),
        charges: Sequence[float] | None = None,
    ) -> DFTSystem:
        """Build a small multi-center toy DFT system."""

        return cls(
            cell=cell,
            grid_shape=grid_shape,
            electron_count=electron_count,
            centers=centers,
            amplitudes=amplitudes,
            widths=widths,
            charges=charges,
        )

    def with_centers(self, centers: Sequence[Sequence[float]]) -> DFTSystem:
        """Return a copy with shifted center coordinates."""

        if self.ions is not None:
            return DFTSystem(
                cell=self.cell,
                grid_shape=self.grid_shape,
                electron_count=self.electron_count,
                ions=self.ions.with_positions(centers),
                charges=self.charges,
            )
        return DFTSystem(
            cell=self.cell,
            grid_shape=self.grid_shape,
            electron_count=self.electron_count,
            centers=centers,
            amplitudes=np.array(self.pseudopotential.amplitudes, dtype=np.float32),
            widths=np.array(self.pseudopotential.widths, dtype=np.float32),
            charges=self.charges,
        )

    def with_cell(self, cell: Cell | Sequence[float], *, scale_centers: bool = False) -> DFTSystem:
        """Return a copy with a new orthorhombic cell."""

        parsed_cell = cell if isinstance(cell, Cell) else Cell.orthorhombic(cell)
        centers = np.array(self.centers, dtype=np.float64)
        if scale_centers:
            old_lengths = np.array(self.cell.lengths, dtype=np.float64)
            new_lengths = np.array(parsed_cell.lengths, dtype=np.float64)
            centers = centers / old_lengths * new_lengths
        if self.ions is not None:
            return DFTSystem(
                cell=parsed_cell,
                grid_shape=self.grid_shape,
                electron_count=self.electron_count,
                ions=self.ions.with_positions(centers),
                charges=self.charges,
            )
        return DFTSystem(
            cell=parsed_cell,
            grid_shape=self.grid_shape,
            electron_count=self.electron_count,
            centers=centers,
            amplitudes=np.array(self.pseudopotential.amplitudes, dtype=np.float32),
            widths=np.array(self.pseudopotential.widths, dtype=np.float32),
            charges=self.charges,
        )

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

    @property
    def center_center_energy(self) -> float:
        """Pairwise center-center Coulomb energy."""

        return center_center_energy(self)

    @property
    def center_center_forces(self) -> np.ndarray:
        """Pairwise center-center Coulomb forces."""

        return center_center_forces(self)
