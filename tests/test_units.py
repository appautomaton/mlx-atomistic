import numpy as np

from mlx_atomistic.units import LJ_REDUCED_UNITS, LennardJonesReducedUnits


def test_default_lj_reduced_units_are_dimensionless():
    assert LJ_REDUCED_UNITS.length == 1.0
    assert LJ_REDUCED_UNITS.energy == 1.0
    assert LJ_REDUCED_UNITS.mass == 1.0
    assert LJ_REDUCED_UNITS.time == 1.0
    assert LJ_REDUCED_UNITS.force == 1.0
    assert LJ_REDUCED_UNITS.velocity == 1.0
    assert LJ_REDUCED_UNITS.temperature == 1.0


def test_lj_reduced_unit_conversions_round_trip():
    units = LennardJonesReducedUnits(sigma=3.4, epsilon=0.997, mass=39.948, boltzmann=1.0)

    assert units.to_reduced_length(6.8) == 2.0
    assert units.from_reduced_length(2.0) == 6.8
    assert units.to_reduced_energy(1.994) == 2.0
    assert units.from_reduced_energy(2.0) == 1.994
    assert units.to_reduced_temperature(1.994) == 2.0
    assert units.from_reduced_temperature(2.0) == 1.994
    np.testing.assert_allclose(units.time, 3.4 * np.sqrt(39.948 / 0.997))

