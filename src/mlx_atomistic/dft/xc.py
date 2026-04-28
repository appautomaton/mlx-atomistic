"""Exchange-correlation functionals for the spin-unpolarized DFT prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Protocol

import mlx.core as mx

from mlx_atomistic.dft.grids import RealSpaceGrid


@dataclass(frozen=True)
class XCResult:
    """Exchange-correlation energy density, total energy, and potential."""

    name: str
    energy_density: mx.array
    potential: mx.array
    total_energy: mx.array


class ExchangeCorrelationFunctional(Protocol):
    """Protocol for local exchange-correlation functionals."""

    name: str

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        """Evaluate energy density, total energy, and potential for ``ρ``."""


@dataclass(frozen=True)
class DiracExchange:
    """Unpolarized Dirac LDA exchange."""

    name: str = "dirac-exchange"

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        rho = mx.maximum(mx.array(density), density_floor)
        coefficient = (3.0 / pi) ** (1.0 / 3.0)
        potential = -coefficient * rho ** (1.0 / 3.0)
        energy_density = -0.75 * coefficient * rho ** (4.0 / 3.0)
        dv = 1.0 if grid is None else grid.dv
        return XCResult(
            name=self.name,
            energy_density=energy_density,
            potential=potential,
            total_energy=mx.sum(energy_density) * dv,
        )


@dataclass(frozen=True)
class LDACorrelationPZ81:
    """Perdew-Zunger 1981 unpolarized LDA correlation parameterization."""

    name: str = "lda-correlation-pz81"
    a: float = 0.0311
    b: float = -0.048
    c: float = 0.0020
    d: float = -0.0116
    gamma: float = -0.1423
    beta1: float = 1.0529
    beta2: float = 0.3334

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        rho = mx.maximum(mx.array(density), density_floor)
        rs = (3.0 / (4.0 * pi * rho)) ** (1.0 / 3.0)
        sqrt_rs = mx.sqrt(rs)

        high_density_energy = (
            self.a * mx.log(rs) + self.b + self.c * rs * mx.log(rs) + self.d * rs
        )
        high_density_derivative = self.a / rs + self.c * mx.log(rs) + self.c + self.d

        denominator = 1.0 + self.beta1 * sqrt_rs + self.beta2 * rs
        low_density_energy = self.gamma / denominator
        low_density_derivative = (
            -self.gamma * (0.5 * self.beta1 / sqrt_rs + self.beta2) / (denominator**2)
        )

        correlation_per_particle = mx.where(rs < 1.0, high_density_energy, low_density_energy)
        derivative_rs = mx.where(rs < 1.0, high_density_derivative, low_density_derivative)
        potential = correlation_per_particle - (rs / 3.0) * derivative_rs
        energy_density = rho * correlation_per_particle
        dv = 1.0 if grid is None else grid.dv
        return XCResult(
            name=self.name,
            energy_density=energy_density,
            potential=potential,
            total_energy=mx.sum(energy_density) * dv,
        )


@dataclass(frozen=True)
class LDAExchangeCorrelation:
    """Combined Dirac exchange plus PZ81 LDA correlation."""

    exchange: ExchangeCorrelationFunctional = field(default_factory=DiracExchange)
    correlation: ExchangeCorrelationFunctional = field(default_factory=LDACorrelationPZ81)
    name: str = "lda-xc-pz81"

    def evaluate(
        self,
        density: mx.array,
        grid: RealSpaceGrid | None = None,
        *,
        density_floor: float = 1e-12,
    ) -> XCResult:
        exchange = self.exchange.evaluate(density, grid, density_floor=density_floor)
        correlation = self.correlation.evaluate(density, grid, density_floor=density_floor)
        energy_density = exchange.energy_density + correlation.energy_density
        potential = exchange.potential + correlation.potential
        dv = 1.0 if grid is None else grid.dv
        return XCResult(
            name=self.name,
            energy_density=energy_density,
            potential=potential,
            total_energy=mx.sum(energy_density) * dv,
        )
