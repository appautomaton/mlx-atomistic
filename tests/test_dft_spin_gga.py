from __future__ import annotations

import mlx.core as mx
import numpy as np

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
