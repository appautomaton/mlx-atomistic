"""Toy DFT potentials and energy helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array
from mlx_atomistic.dft.density import density_from_orbitals
from mlx_atomistic.dft.fft import fft3, ifft3
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid


@dataclass(frozen=True)
class LocalGaussianPseudopotential:
    """Simple periodic local Gaussian pseudopotential for toy SCF examples."""

    centers: mx.array
    amplitudes: mx.array
    widths: mx.array

    def __init__(
        self,
        centers: Sequence[Sequence[float]],
        amplitudes: Sequence[float] | float,
        widths: Sequence[float] | float,
    ):
        centers_np = np.array(centers, dtype=np.float32)
        if centers_np.ndim != 2 or centers_np.shape[1] != 3:
            msg = "centers must have shape (n_centers, 3)"
            raise ValueError(msg)
        n_centers = centers_np.shape[0]
        amplitudes_np = np.broadcast_to(np.array(amplitudes, dtype=np.float32), (n_centers,))
        widths_np = np.broadcast_to(np.array(widths, dtype=np.float32), (n_centers,))
        if np.any(widths_np <= 0.0):
            msg = "Gaussian widths must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "centers", as_mx_array(centers_np))
        object.__setattr__(self, "amplitudes", as_mx_array(amplitudes_np))
        object.__setattr__(self, "widths", as_mx_array(widths_np))

    def field(self, grid: RealSpaceGrid) -> mx.array:
        """Evaluate the local potential on a real-space grid."""

        coordinates = grid.coordinates()
        potential = mx.zeros(grid.shape, dtype=mx.float32)
        for index in range(int(self.centers.shape[0])):
            center = self.centers[index]
            width = self.widths[index]
            amplitude = self.amplitudes[index]
            displacement = grid.cell.minimum_image(coordinates - center)
            r2 = mx.sum(displacement * displacement, axis=-1)
            potential = potential + amplitude * mx.exp(-0.5 * r2 / (width * width))
        return potential

    __call__ = field


def hartree_potential(density: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Solve the periodic Hartree potential for ``ρ`` with the ``G = 0`` term removed."""

    reciprocal = ReciprocalGrid.from_real_space(grid)
    density_g = fft3(density)
    safe_g2 = mx.where(reciprocal.zero_mask, mx.ones_like(reciprocal.g2), reciprocal.g2)
    potential_g = 4.0 * pi * density_g / safe_g2
    potential_g = mx.where(reciprocal.zero_mask, mx.zeros_like(potential_g), potential_g)
    return mx.real(ifft3(potential_g))


def lda_exchange_energy_potential(
    density: mx.array,
    grid: RealSpaceGrid | None = None,
    *,
    density_floor: float = 1e-12,
) -> tuple[mx.array, mx.array]:
    """Return Dirac LDA exchange energy and potential for an unpolarized density."""

    rho = mx.maximum(mx.array(density), density_floor)
    coefficient = (3.0 / pi) ** (1.0 / 3.0)
    potential = -coefficient * rho ** (1.0 / 3.0)
    energy_density = -0.75 * coefficient * rho ** (4.0 / 3.0)
    dv = 1.0 if grid is None else grid.dv
    return mx.sum(energy_density) * dv, potential


def apply_kinetic(orbital: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Apply the plane-wave kinetic operator ``-1/2 ∇²`` to one orbital."""

    reciprocal = ReciprocalGrid.from_real_space(grid)
    return mx.real(ifft3(0.5 * reciprocal.g2 * fft3(orbital)))


def kinetic_energy(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
) -> mx.array:
    """Return the occupied one-particle kinetic energy."""

    stack = mx.array(orbitals)
    if stack.shape == grid.shape:
        stack = mx.reshape(stack, (1, *grid.shape))
    energy = mx.array(0.0, dtype=mx.float32)
    for index, occupation in enumerate(occupations):
        applied = apply_kinetic(stack[index], grid)
        expectation = mx.real(mx.sum(mx.conjugate(stack[index]) * applied))
        energy = energy + float(occupation) * expectation * grid.dv
    return energy


def energy_decomposition(
    orbitals: mx.array,
    density: mx.array,
    local_potential: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
    density_floor: float = 1e-12,
) -> dict[str, mx.array]:
    """Return core toy-DFT energy terms."""

    v_hartree = hartree_potential(density, grid)
    exchange_energy, _ = lda_exchange_energy_potential(
        density,
        grid,
        density_floor=density_floor,
    )
    kinetic = kinetic_energy(orbitals, grid, occupations=occupations)
    local = mx.sum(density * local_potential) * grid.dv
    hartree = 0.5 * mx.sum(density * v_hartree) * grid.dv
    total = kinetic + local + hartree + exchange_energy
    return {
        "kinetic": kinetic,
        "local": local,
        "hartree": hartree,
        "exchange": exchange_energy,
        "total": total,
    }


def electron_count(density: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Integrate a density over the real-space grid."""

    return mx.sum(density) * grid.dv


def density_from_normalized_orbitals(
    orbitals: mx.array,
    grid: RealSpaceGrid,
    *,
    occupations: Sequence[float],
) -> mx.array:
    """Build a density after normalizing orbitals."""

    return density_from_orbitals(orbitals, grid, occupations=occupations)
