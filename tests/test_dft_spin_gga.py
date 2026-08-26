from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    ProductionPBEExchangeCorrelation,
    ProductionSpinPBEExchangeCorrelation,
    RealSpaceGrid,
)


def _density(grid: RealSpaceGrid) -> mx.array:
    coordinates = grid.coordinates()
    phase = (
        2.0 * np.pi * coordinates[..., 0] / float(grid.lengths[0])
        + 4.0 * np.pi * coordinates[..., 1] / float(grid.lengths[1])
    )
    return 0.08 + 0.01 * mx.cos(phase)


def _qe_pw92_spin(rs: float, zeta: float) -> tuple[float, float, float]:
    def component(parameters):
        a, a1, b1, b2, b3, b4 = parameters
        rs12 = np.sqrt(rs)
        rs32 = rs * rs12
        rs2 = rs * rs
        omega = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
        derivative = 2.0 * a * (
            0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2.0 * b4 * rs2
        )
        logarithm = np.log1p(1.0 / omega)
        energy = -2.0 * a * (1.0 + a1 * rs) * logarithm
        potential = (
            -2.0 * a * (1.0 + 2.0 * a1 * rs / 3.0) * logarithm
            - 2.0
            * a
            * (1.0 + a1 * rs)
            * derivative
            / (3.0 * omega * (omega + 1.0))
        )
        return energy, potential

    unpolarized = component((0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294))
    polarized = component((0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517))
    stiffness_energy, stiffness_potential = component(
        (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    )
    alpha = -stiffness_energy
    alpha_potential = -stiffness_potential
    zeta3 = zeta**3
    zeta4 = zeta**4
    denominator = 2.0 ** (4.0 / 3.0) - 2.0
    interpolation = (
        (1.0 + zeta) ** (4.0 / 3.0)
        + (1.0 - zeta) ** (4.0 / 3.0)
        - 2.0
    ) / denominator
    derivative = (
        4.0
        * ((1.0 + zeta) ** (1.0 / 3.0) - (1.0 - zeta) ** (1.0 / 3.0))
        / (3.0 * denominator)
    )
    fz0 = 1.709921
    delta_energy = polarized[0] - unpolarized[0]
    common = (
        unpolarized[1]
        + alpha_potential * interpolation * (1.0 - zeta4) / fz0
        + (polarized[1] - unpolarized[1]) * interpolation * zeta4
    )
    spin_derivative = (
        alpha
        / fz0
        * (derivative * (1.0 - zeta4) - 4.0 * interpolation * zeta3)
        + delta_energy * (derivative * zeta4 + 4.0 * interpolation * zeta3)
    )
    energy = (
        unpolarized[0]
        + alpha * interpolation * (1.0 - zeta4) / fz0
        + delta_energy * interpolation * zeta4
    )
    return (
        energy,
        common + (1.0 - zeta) * spin_derivative,
        common - (1.0 + zeta) * spin_derivative,
    )


def test_spin_pbe_equal_channels_reproduce_unpolarized_pbe():
    grid = RealSpaceGrid((4, 4, 4), (7.0, 8.0, 9.0))
    density = _density(grid)
    unpolarized = ProductionPBEExchangeCorrelation().evaluate(density, grid)
    spin = ProductionSpinPBEExchangeCorrelation().evaluate(
        0.5 * density,
        0.5 * density,
        grid,
    )
    mx.eval(
        unpolarized.total_energy,
        unpolarized.potential,
        spin.total_energy,
        spin.potentials,
    )

    np.testing.assert_allclose(
        np.asarray(spin.total_energy),
        np.asarray(unpolarized.total_energy),
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(spin.up_potential),
        np.asarray(unpolarized.potential),
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(spin.down_potential),
        np.asarray(unpolarized.potential),
        atol=2.0e-5,
    )


def test_spin_pbe_is_invariant_to_channel_exchange():
    grid = RealSpaceGrid((4, 4, 4), (7.0, 8.0, 9.0))
    density = _density(grid)
    functional = ProductionSpinPBEExchangeCorrelation()
    first = functional.evaluate(0.65 * density, 0.35 * density, grid)
    swapped = functional.evaluate(0.35 * density, 0.65 * density, grid)
    mx.eval(first.total_energy, first.potentials, swapped.total_energy, swapped.potentials)

    np.testing.assert_allclose(first.total_energy, swapped.total_energy, atol=2.0e-6)
    np.testing.assert_allclose(first.up_potential, swapped.down_potential, atol=2.0e-5)
    np.testing.assert_allclose(first.down_potential, swapped.up_potential, atol=2.0e-5)


def test_spin_pbe_potentials_match_directional_energy_derivative():
    grid = RealSpaceGrid((4, 4, 4), (7.0, 8.0, 9.0))
    density = _density(grid)
    up = 0.6 * density
    down = 0.4 * density
    direction = 0.01 * mx.sin(grid.coordinates()[..., 2])
    functional = ProductionSpinPBEExchangeCorrelation()
    result = functional.evaluate(up, down, grid)
    step = 5.0e-2
    plus = functional.evaluate(up + step * direction, down, grid)
    minus = functional.evaluate(up - step * direction, down, grid)
    mx.eval(result.up_potential, plus.total_energy, minus.total_energy)

    analytic = float(mx.sum(result.up_potential * direction) * grid.dv)
    central = float(plus.total_energy - minus.total_energy) / (2.0 * step)
    np.testing.assert_allclose(analytic, central, atol=3.0e-5, rtol=1.5e-4)


@pytest.mark.parametrize("polarization", [0.2, 0.6, 0.9])
def test_uniform_spin_pbe_matches_quantum_espresso_pw92_oracle(polarization):
    grid = RealSpaceGrid((3, 3, 3), (4.0, 4.0, 4.0))
    density = 0.05
    up = 0.5 * density * (1.0 + polarization)
    down = density - up
    result = ProductionSpinPBEExchangeCorrelation().evaluate(
        mx.full(grid.shape, up),
        mx.full(grid.shape, down),
        grid,
    )
    rs = (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    correlation_energy, correlation_up, correlation_down = _qe_pw92_spin(
        rs,
        polarization,
    )
    exchange_coefficient = (3.0 / np.pi) ** (1.0 / 3.0)
    exchange_density = -0.375 * exchange_coefficient * (
        (2.0 * up) ** (4.0 / 3.0) + (2.0 * down) ** (4.0 / 3.0)
    )
    expected_energy = grid.volume * (exchange_density + density * correlation_energy)
    expected_up = -exchange_coefficient * (2.0 * up) ** (1.0 / 3.0) + correlation_up
    expected_down = (
        -exchange_coefficient * (2.0 * down) ** (1.0 / 3.0) + correlation_down
    )

    assert float(result.total_energy) == pytest.approx(expected_energy, abs=3.0e-5)
    np.testing.assert_allclose(np.asarray(result.up_potential), expected_up, atol=3.0e-5)
    np.testing.assert_allclose(np.asarray(result.down_potential), expected_down, atol=3.0e-5)
