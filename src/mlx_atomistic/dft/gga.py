"""PBE GGA exchange-correlation with an autodiff-derived potential.

The defining idea: a GGA energy is ``E_xc = ∫ ε(ρ, ∇ρ) dr``, and its potential is
the functional derivative ``v_xc = δE_xc/δρ = ∂ε/∂ρ - ∇·(∂ε/∂∇ρ)``. The gradient
term is the part that is famously error-prone to hand-derive. Here we write *only*
the energy density and obtain ``v_xc`` from ``mx.grad`` of the total energy — the
autodiff machinery reconstructs the divergence term automatically, provided the
density gradient is built with a differentiable (MLX-native) FFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import mlx.core as mx

from mlx_atomistic.dft.fft import fft3, ifft3
from mlx_atomistic.dft.grids import RealSpaceGrid, ReciprocalGrid
from mlx_atomistic.dft.xc import LDACorrelationPZ81, XCResult

# PBE (Perdew-Burke-Ernzerhof 1996) constants.
_KAPPA = 0.804
_MU = 0.2195149727645171
_BETA = 0.06672455060314922
_GAMMA = (1.0 - log(2.0)) / (pi * pi)


def density_gradient(rho: mx.array, grid: RealSpaceGrid) -> mx.array:
    """Return ``∇ρ`` (shape ``(3, *grid.shape)``) via reciprocal-space differentiation."""

    reciprocal = ReciprocalGrid.from_real_space(grid)
    rho_reciprocal = fft3(rho)
    vectors = reciprocal.vectors
    components = [
        mx.real(ifft3(1j * vectors[..., axis] * rho_reciprocal)) for axis in range(3)
    ]
    return mx.stack(components, axis=0)


def _pbe_exchange_energy_density(rho: mx.array, sigma: mx.array) -> mx.array:
    """PBE exchange energy density: Dirac times the enhancement factor ``F_x(s)``."""

    cx = (3.0 / pi) ** (1.0 / 3.0)
    dirac = -0.75 * cx * rho ** (4.0 / 3.0)
    kf = (3.0 * pi * pi * rho) ** (1.0 / 3.0)
    s2 = sigma / (4.0 * kf * kf * rho * rho)  # reduced gradient squared, s²
    enhancement = 1.0 + _KAPPA - _KAPPA / (1.0 + _MU * s2 / _KAPPA)
    return dirac * enhancement


def _pbe_correlation_energy_density(
    rho: mx.array, sigma: mx.array, eps_c_unif: mx.array
) -> mx.array:
    """PBE correlation: uniform-gas correlation plus the gradient term ``H(rs, t)``."""

    kf = (3.0 * pi * pi * rho) ** (1.0 / 3.0)
    ks = mx.sqrt(4.0 * kf / pi)
    t2 = sigma / (4.0 * ks * ks * rho * rho)  # reduced gradient squared, t²
    a = (_BETA / _GAMMA) / (mx.exp(-eps_c_unif / _GAMMA) - 1.0)
    at2 = a * t2
    h = _GAMMA * mx.log(
        1.0 + (_BETA / _GAMMA) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2)
    )
    return rho * (eps_c_unif + h)


@dataclass(frozen=True)
class PBEExchangeCorrelation:
    """PBE GGA exchange-correlation; ``v_xc`` is the autodiff functional derivative.

    The uniform-gas correlation baseline reuses the PZ81 parameterization already in
    the package (true PBE uses PW92; the difference is sub-mHa and the baseline is
    swappable).
    """

    name: str = "pbe-gga"

    def _energy_density(
        self, rho: mx.array, grid: RealSpaceGrid, density_floor: float
    ) -> mx.array:
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
        """Evaluate the PBE GGA exchange-correlation energy density, potential, and total energy.

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
        rho = mx.maximum(mx.array(density), density_floor)

        def total_energy(field: mx.array) -> mx.array:
            return mx.sum(self._energy_density(field, grid, density_floor)) * grid.dv

        potential = mx.grad(total_energy)(rho) / grid.dv
        energy_density = self._energy_density(rho, grid, density_floor)
        return XCResult(
            name=self.name,
            energy_density=energy_density,
            potential=potential,
            total_energy=mx.sum(energy_density) * grid.dv,
        )
