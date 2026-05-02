import numpy as np

from mlx_atomistic.units import (
    BOLTZMANN_CONSTANT_KJ_MOL_K,
    COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
    DALTON_ANGSTROM2_PER_PS2_TO_KJ_PER_MOL,
    LJ_REDUCED_UNITS,
    LennardJonesReducedUnits,
    MDUnitSystem,
)


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


def test_production_unit_constants_are_physical():
    units = MDUnitSystem()

    np.testing.assert_allclose(units.coulomb_constant, COULOMB_CONSTANT_KJ_MOL_ANGSTROM)
    np.testing.assert_allclose(units.boltzmann_constant, BOLTZMANN_CONSTANT_KJ_MOL_K)
    np.testing.assert_allclose(
        units.kinetic_energy_scale,
        DALTON_ANGSTROM2_PER_PS2_TO_KJ_PER_MOL,
    )
    np.testing.assert_allclose(units.force_to_acceleration_scale, 100.0)
