"""Spin-polarized PBE exchange-correlation for periodic DFT."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import mlx.core as mx

from mlx_atomistic.dft.gga import (
    _BETA,
    _GAMMA,
    _pbe_exchange_energy_density,
    density_gradient,
)
from mlx_atomistic.dft.grids import RealSpaceGrid

_PW92_UNPOLARIZED = (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_PW92_POLARIZED = (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_PW92_SPIN_STIFFNESS = (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
_SPIN_INTERPOLATION_SECOND_DERIVATIVE = (8.0 / 9.0) / (2.0 ** (4.0 / 3.0) - 2.0)


@dataclass(frozen=True)
class SpinXCResult:
    """Spin-resolved exchange-correlation energy and potentials."""

    name: str
    energy_density: mx.array
    potentials: mx.array
    total_energy: mx.array

    @property
    def up_potential(self) -> mx.array:
        """Return the spin-up exchange-correlation potential."""

        return self.potentials[0]

    @property
    def down_potential(self) -> mx.array:
        """Return the spin-down exchange-correlation potential."""

        return self.potentials[1]


def _pw92_energy(
    rs: mx.array,
    parameters: tuple[float, float, float, float, float, float],
) -> mx.array:
    a, alpha1, beta1, beta2, beta3, beta4 = parameters
    denominator = 2.0 * a * (
        beta1 * mx.sqrt(rs)
        + beta2 * rs
        + beta3 * rs**1.5
        + beta4 * rs * rs
    )
    return -2.0 * a * (1.0 + alpha1 * rs) * mx.log1p(1.0 / denominator)


def _pw92_spin_correlation_per_particle(
    density: mx.array,
    polarization: mx.array,
) -> mx.array:
    rs = (3.0 / (4.0 * pi * density)) ** (1.0 / 3.0)
    unpolarized = _pw92_energy(rs, _PW92_UNPOLARIZED)
    polarized = _pw92_energy(rs, _PW92_POLARIZED)
    stiffness = -_pw92_energy(rs, _PW92_SPIN_STIFFNESS)
    zeta4 = polarization**4
    interpolation = (
        (1.0 + polarization) ** (4.0 / 3.0)
        + (1.0 - polarization) ** (4.0 / 3.0)
        - 2.0
    ) / (2.0 ** (4.0 / 3.0) - 2.0)
    return (
        unpolarized
        + stiffness
        * interpolation
        * (1.0 - zeta4)
        / _SPIN_INTERPOLATION_SECOND_DERIVATIVE
        + (polarized - unpolarized) * interpolation * zeta4
    )


def _spin_pbe_energy_density(
    spin_density: mx.array,
    grid: RealSpaceGrid,
    density_floor: float,
) -> mx.array:
    up = mx.maximum(spin_density[0], density_floor)
    down = mx.maximum(spin_density[1], density_floor)
    up_gradient = density_gradient(up, grid)
    down_gradient = density_gradient(down, grid)

    exchange = 0.5 * (
        _pbe_exchange_energy_density(
            2.0 * up,
            4.0 * mx.sum(up_gradient * up_gradient, axis=0),
        )
        + _pbe_exchange_energy_density(
            2.0 * down,
            4.0 * mx.sum(down_gradient * down_gradient, axis=0),
        )
    )

    density = up + down
    polarization = mx.clip((up - down) / density, -1.0 + 1.0e-7, 1.0 - 1.0e-7)
    phi = 0.5 * (
        (1.0 + polarization) ** (2.0 / 3.0)
        + (1.0 - polarization) ** (2.0 / 3.0)
    )
    uniform_correlation = _pw92_spin_correlation_per_particle(
        density,
        polarization,
    )
    total_gradient = up_gradient + down_gradient
    sigma = mx.sum(total_gradient * total_gradient, axis=0)
    kf = (3.0 * pi * pi * density) ** (1.0 / 3.0)
    ks = mx.sqrt(4.0 * kf / pi)
    t2 = sigma / (4.0 * phi * phi * ks * ks * density * density)
    a = (_BETA / _GAMMA) / mx.expm1(
        -uniform_correlation / (_GAMMA * phi**3)
    )
    y = a * t2
    gradient_ratio = (1.0 + y) / (1.0 + y + y * y)
    correlation_gradient = _GAMMA * phi**3 * mx.log1p(
        (_BETA / _GAMMA) * t2 * gradient_ratio
    )
    return exchange + density * (uniform_correlation + correlation_gradient)


@dataclass(frozen=True)
class ProductionSpinPBEExchangeCorrelation:
    """Collinear spin-PBE with the PW92 uniform-gas baseline."""

    name: str = "spin-pbe-pw92-gga"

    def evaluate(
        self,
        up_density: mx.array,
        down_density: mx.array,
        grid: RealSpaceGrid,
        *,
        density_floor: float = 1.0e-12,
    ) -> SpinXCResult:
        """Evaluate spin-PBE energy and both local potentials.

        Args:
            up_density: Spin-up electron density on the real-space grid.
            down_density: Spin-down electron density on the real-space grid.
            grid: Periodic real-space integration grid.
            density_floor: Positive per-channel numerical density floor.

        Returns:
            Spin-resolved PBE energy density, total energy, and potentials.

        Raises:
            ValueError: If density shapes or the floor are invalid.
        """

        up = mx.real(mx.array(up_density)).astype(mx.float32)
        down = mx.real(mx.array(down_density)).astype(mx.float32)
        if up.shape != grid.shape or down.shape != grid.shape:
            raise ValueError("spin densities must match the real-space grid")
        if density_floor <= 0.0:
            raise ValueError("density_floor must be positive")
        spin_density = mx.stack((up, down), axis=0)

        def energy_and_density(field: mx.array) -> tuple[mx.array, mx.array]:
            energy_density = _spin_pbe_energy_density(
                field,
                grid,
                density_floor,
            )
            return mx.sum(energy_density) * grid.dv, energy_density

        (total_energy, energy_density), derivative = mx.value_and_grad(
            energy_and_density
        )(spin_density)
        return SpinXCResult(
            name=self.name,
            energy_density=energy_density,
            potentials=derivative / grid.dv,
            total_energy=total_energy,
        )
