"""PBE-form GGA exchange-correlation with an autodiff-derived potential.

The defining idea: a GGA energy is ``E_xc = ∫ ε(ρ, ∇ρ) dr``, and its potential is
the functional derivative ``v_xc = δE_xc/δρ = ∂ε/∂ρ - ∇·(∂ε/∂∇ρ)``. The gradient
term is the part that is famously error-prone to hand-derive. Here we write *only*
the energy density and obtain ``v_xc`` with ``mx.value_and_grad``. The autodiff
machinery reconstructs the divergence term automatically, provided the density
gradient is built with a differentiable (MLX-native) FFT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log, pi

import mlx.core as mx

from mlx_atomistic.dft.fft import fft3, ifft3
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.xc import LDACorrelationPW92, LDACorrelationPZ81, XCResult

# PBE (Perdew-Burke-Ernzerhof 1996) gradient constants.
_KAPPA = 0.804
_MU = 0.2195149727645171
_BETA = 0.06672455060314922
_GAMMA = (1.0 - log(2.0)) / (pi * pi)

_EnergyDensityFunction = Callable[[mx.array, RealSpaceGrid, float], mx.array]


def density_gradient(rho: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Return ``∇ρ`` (shape ``(3, *grid.shape)``) via reciprocal-space differentiation."""

    reciprocal = ReciprocalGrid.from_real_space(grid)
    rho_reciprocal = fft3(rho)
    vectors = reciprocal.vectors
    components = [mx.real(ifft3(1j * vectors[..., axis] * rho_reciprocal)) for axis in range(3)]
    return mx.stack(components, axis=0)


def _pbe_exchange_energy_density(rho: mx.array, sigma: mx.array) -> mx.array:
    """PBE exchange energy density: Dirac times the enhancement factor ``F_x(s)``."""

    cx = (3.0 / pi) ** (1.0 / 3.0)
    dirac = -0.75 * cx * rho ** (4.0 / 3.0)
    kf = (3.0 * pi * pi * rho) ** (1.0 / 3.0)
    numerator = _MU * sigma
    denominator = 4.0 * _KAPPA * kf * kf * rho * rho + numerator
    enhancement = 1.0 + _KAPPA * numerator / denominator
    return dirac * enhancement


def _pbe_correlation_energy_density(
    rho: mx.array, sigma: mx.array, eps_c_unif: mx.array
) -> mx.array:
    """PBE correlation: uniform-gas correlation plus the gradient term ``H(rs, t)``."""

    kf = (3.0 * pi * pi * rho) ** (1.0 / 3.0)
    ks = mx.sqrt(4.0 * kf / pi)
    a = (_BETA / _GAMMA) / mx.expm1(-eps_c_unif / _GAMMA)
    numerator = a * sigma
    z = numerator / (4.0 * ks * ks * rho * rho + numerator)
    reduced_gradient_term = z / (a * (1.0 - z + z * z))
    h = _GAMMA * mx.log1p((_BETA / _GAMMA) * reduced_gradient_term)
    return rho * (eps_c_unif + h)


def _evaluate_pbe(
    name: str,
    energy_density_function: _EnergyDensityFunction,
    density: mx.array,
    grid: RealSpaceGrid,
    density_floor: float,
) -> XCResult:
    """Evaluate one PBE variant without repeating its forward energy graph."""

    rho = mx.maximum(mx.array(density), density_floor)

    def energy_and_density(field: mx.array) -> tuple[mx.array, mx.array]:
        energy_density = energy_density_function(field, grid, density_floor)
        return mx.sum(energy_density) * grid.dv, energy_density

    (total_energy, energy_density), derivative = mx.value_and_grad(energy_and_density)(rho)
    return XCResult(
        name=name,
        energy_density=energy_density,
        potential=derivative / grid.dv,
        total_energy=total_energy,
    )


@dataclass(frozen=True)
class PBEExchangeCorrelation:
    """Alpha PBE-form GGA exchange-correlation with a PZ81 uniform baseline.

    The exchange enhancement and correlation gradient terms follow the PBE form, while
    the uniform-gas correlation baseline reuses the PZ81 parameterization already in
    the package. Full production PBE uses a PW92 baseline, so this public alpha class
    intentionally reports an explicit alpha result name.
    """

    name: str = "pbe-pz81-gga-alpha"

    def _energy_density(self, rho: mx.array, grid: RealSpaceGrid, density_floor: float) -> mx.array:
        rho = mx.maximum(rho, density_floor)
        gradient = density_gradient(rho, grid)
        sigma = mx.sum(gradient * gradient, axis=0)
        eps_c_unif = (
            LDACorrelationPZ81().evaluate(rho, grid, density_floor=density_floor).energy_density
            / rho
        )
        return _pbe_exchange_energy_density(rho, sigma) + _pbe_correlation_energy_density(
            rho, sigma, eps_c_unif
        )

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        """Evaluate alpha PBE-form GGA energy density, potential, and total energy.

        Args:
            density: Electron density ``rho`` sampled on the grid.
            grid: Real-space grid; required for GGA to evaluate the density gradient.
                Defaults to ``None``.
            density_floor: Lower clamp applied to the density for numerical stability.
                Defaults to ``1e-12``.

        Returns:
            An `XCResult` with the energy density, potential, and total energy.

        Raises:
            ValueError: If ``grid`` is ``None`` (a grid is required for GGA).
        """

        if grid is None:
            msg = "PBE (GGA) requires a real-space grid to evaluate the density gradient"
            raise ValueError(msg)
        return _evaluate_pbe(
            self.name,
            self._energy_density,
            density,
            grid,
            density_floor,
        )


@dataclass(frozen=True)
class ProductionPBEExchangeCorrelation:
    """PBE GGA exchange-correlation with the PW92 uniform-gas baseline."""

    name: str = "pbe-pw92-gga"

    def _energy_density(self, rho: mx.array, grid: RealSpaceGrid, density_floor: float) -> mx.array:
        rho = mx.maximum(rho, density_floor)
        gradient = density_gradient(rho, grid)
        sigma = mx.sum(gradient * gradient, axis=0)
        eps_c_unif = LDACorrelationPW92().correlation_per_particle(
            rho,
            density_floor=density_floor,
        )
        return _pbe_exchange_energy_density(rho, sigma) + _pbe_correlation_energy_density(
            rho,
            sigma,
            eps_c_unif,
        )

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        """Evaluate production PBE energy density, potential, and total energy.

        Args:
            density: Electron density ``rho`` sampled on the grid.
            grid: Real-space grid required for the GGA gradient. Defaults to
                ``None``.
            density_floor: Lower density clamp. Defaults to ``1e-12``.

        Returns:
            Production PBE energy density, potential, and total energy.

        Raises:
            ValueError: If no real-space grid is provided.
        """

        if grid is None:
            msg = "production PBE requires a real-space grid"
            raise ValueError(msg)
        return _evaluate_pbe(
            self.name,
            self._energy_density,
            density,
            grid,
            density_floor,
        )
