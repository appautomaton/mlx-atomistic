"""Separable nonlocal pseudopotential operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.grids import RealSpaceGrid
from mlx_atomistic.dft.pseudopotentials import IonCollection, NonlocalProjectorData


@dataclass(frozen=True)
class ProjectorSet:
    """Real-space separable nonlocal projectors on one grid."""

    projectors: mx.array
    couplings: mx.array
    angular_momenta: tuple[int, ...]
    metadata: tuple[dict, ...]

    def __init__(
        self,
        projectors: Sequence[np.ndarray] | mx.array,
        couplings: Sequence[float] | mx.array,
        *,
        angular_momenta: Sequence[int] | None = None,
        metadata: Sequence[dict] | None = None,
    ):
        projectors_mx = mx.array(projectors)
        if len(projectors_mx.shape) != 4:
            msg = "projectors must have shape (n_projectors, nx, ny, nz)"
            raise ValueError(msg)
        couplings_mx = mx.array(couplings, dtype=mx.float32)
        if couplings_mx.shape != (projectors_mx.shape[0],):
            msg = "couplings must have shape (n_projectors,)"
            raise ValueError(msg)
        n_projectors = int(projectors_mx.shape[0])
        if angular_momenta is None:
            parsed_angular = tuple(0 for _ in range(n_projectors))
        else:
            if len(angular_momenta) != n_projectors:
                msg = "angular_momenta length must match number of projectors"
                raise ValueError(msg)
            parsed_angular = tuple(int(value) for value in angular_momenta)
        if metadata is None:
            parsed_metadata = tuple({} for _ in range(n_projectors))
        else:
            if len(metadata) != n_projectors:
                msg = "metadata length must match number of projectors"
                raise ValueError(msg)
            parsed_metadata = tuple(dict(item) for item in metadata)
        object.__setattr__(self, "projectors", projectors_mx)
        object.__setattr__(self, "couplings", couplings_mx)
        object.__setattr__(self, "angular_momenta", parsed_angular)
        object.__setattr__(self, "metadata", parsed_metadata)

    @property
    def count(self) -> int:
        """Number of projectors."""

        return int(self.projectors.shape[0])

    @property
    def available(self) -> bool:
        """Whether any projector is present."""

        return self.count > 0

    @classmethod
    def from_ions(cls, ions: IonCollection, grid: RealSpaceGrid) -> ProjectorSet:
        """Build normalized real-space projectors for all parsed ion projectors."""

        coordinates = np.array(grid.coordinates(), dtype=np.float64)
        projectors: list[np.ndarray] = []
        couplings: list[float] = []
        angular_momenta: list[int] = []
        metadata: list[dict] = []
        for ion_index, ion in enumerate(ions.ions):
            center = np.array(ion.position, dtype=np.float64)
            displacement = np.array(grid.cell.minimum_image(coordinates - center), dtype=np.float64)
            radius = np.linalg.norm(displacement, axis=-1)
            for projector_index, projector in enumerate(ion.pseudopotential.nonlocal_projectors):
                field = _projector_field(projector, radius, displacement)
                norm = float(np.sqrt(np.sum(field * field) * grid.dv))
                if norm <= 1e-14 or not np.isfinite(norm):
                    continue
                projectors.append((field / norm).astype(np.float32))
                couplings.append(float(projector.coupling))
                angular_momenta.append(int(projector.angular_momentum))
                metadata.append(
                    {
                        "ion_index": ion_index,
                        "symbol": ion.symbol,
                        "projector_index": projector_index,
                        "format": str(ion.pseudopotential.format),
                    }
                )
        if not projectors:
            return cls(
                np.zeros((0, *grid.shape), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return cls(
            np.stack(projectors),
            np.asarray(couplings, dtype=np.float32),
            angular_momenta=angular_momenta,
            metadata=metadata,
        )


@dataclass(frozen=True)
class NonlocalPseudopotentialOperator:
    """Hermitian separable nonlocal pseudopotential operator."""

    grid: RealSpaceGrid
    projectors: ProjectorSet

    @classmethod
    def from_ions(cls, ions: IonCollection, grid: RealSpaceGrid) -> NonlocalPseudopotentialOperator:
        """Build the nonlocal operator for an ion collection."""

        return cls(grid=grid, projectors=ProjectorSet.from_ions(ions, grid))

    @property
    def available(self) -> bool:
        """Whether nonlocal projectors are available."""

        return self.projectors.available

    def apply(self, orbitals: mx.array) -> mx.array:
        """Apply the separable nonlocal operator to one orbital or an orbital stack."""

        return apply_nonlocal_pseudopotential(
            orbitals,
            self.grid,
            self.projectors,
        )

    def energy(self, orbitals: mx.array, *, occupations: Sequence[float]) -> mx.array:
        """Return the occupied nonlocal pseudopotential energy."""

        return nonlocal_pseudopotential_energy(
            orbitals,
            self.grid,
            self.projectors,
            occupations=occupations,
        )

    def to_dict(self) -> dict:
        """Return JSON-safe projector metadata."""

        return {
            "projector_count": self.projectors.count,
            "couplings": np.array(self.projectors.couplings).tolist(),
            "angular_momenta": list(self.projectors.angular_momenta),
            "metadata": list(self.projectors.metadata),
        }


def apply_nonlocal_pseudopotential(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    projectors: ProjectorSet,
) -> mx.array:
    """Apply ``Σᵢ |βᵢ>Dᵢ<βᵢ|`` to one orbital or an orbital stack."""

    stack, was_single = _as_stack(orbitals, grid)
    if not projectors.available:
        return orbitals
    applied = []
    beta = projectors.projectors
    couplings = projectors.couplings
    for orbital in stack:
        values = mx.zeros(grid.shape, dtype=orbital.dtype)
        for index in range(projectors.count):
            overlap = mx.sum(mx.conjugate(beta[index]) * orbital) * grid.dv
            values = values + couplings[index] * beta[index] * overlap
        applied.append(values)
    result = mx.stack(applied, axis=0)
    return result[0] if was_single else result


def nonlocal_pseudopotential_energy(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    projectors: ProjectorSet,
    *,
    occupations: Sequence[float],
) -> mx.array:
    """Return occupied expectation value of a separable nonlocal operator."""

    stack, _ = _as_stack(orbitals, grid)
    if len(occupations) != int(stack.shape[0]):
        msg = "occupations length must match number of orbitals"
        raise ValueError(msg)
    if not projectors.available:
        return mx.array(0.0, dtype=mx.float32)
    applied, _ = _as_stack(apply_nonlocal_pseudopotential(stack, grid, projectors), grid)
    energy = mx.array(0.0, dtype=mx.float32)
    for index, occupation in enumerate(occupations):
        expectation = mx.sum(mx.conjugate(stack[index]) * applied[index]) * grid.dv
        energy = energy + float(occupation) * mx.real(expectation)
    return energy


def _as_stack(orbitals: mx.array, grid: RealSpaceGrid) -> tuple[mx.array, bool]:
    array = mx.array(orbitals)
    if array.shape == grid.shape:
        return mx.reshape(array, (1, *grid.shape)), True
    if len(array.shape) == 4 and array.shape[1:] == grid.shape:
        return array, False
    msg = "orbitals must have shape grid.shape or (n_orbitals, *grid.shape)"
    raise ValueError(msg)


def _projector_field(
    projector: NonlocalProjectorData,
    radius: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    if projector.radial_grid is not None and projector.values:
        radial = np.interp(
            radius,
            projector.radial_grid.radii,
            np.asarray(projector.values, dtype=np.float64),
            left=0.0,
            right=0.0,
        )
    else:
        cutoff = 1.0 if projector.cutoff_radius is None else max(projector.cutoff_radius, 1e-6)
        x = radius / cutoff
        radial = np.exp(-0.5 * x * x) * x ** max(projector.angular_momentum, 0)
    angular = _angular_factor(projector.angular_momentum, radius, displacement)
    return np.asarray(radial * angular, dtype=np.float64)


def _angular_factor(
    angular_momentum: int,
    radius: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    if angular_momentum <= 0:
        return np.ones_like(radius)
    safe_radius = np.maximum(radius, 1e-12)
    unit = displacement / safe_radius[..., None]
    if angular_momentum == 1:
        return unit[..., 0]
    if angular_momentum == 2:
        return 0.5 * (3.0 * unit[..., 2] * unit[..., 2] - 1.0)
    return unit[..., 0] ** min(angular_momentum, 4)
