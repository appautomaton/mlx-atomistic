import numpy as np

from mlx_atomistic.initialize import (
    fcc_lattice,
    remove_center_of_mass_velocity,
    rescale_temperature,
    simple_cubic_lattice,
    thermal_velocities,
)
from mlx_atomistic.md import instantaneous_temperature


def test_lattice_helpers_return_positions_inside_cell():
    for builder in (simple_cubic_lattice, fcc_lattice):
        positions, cell = builder(10, density=0.8)
        positions_np = np.array(positions)
        lengths = np.array(cell.lengths)

        assert positions_np.shape == (10, 3)
        assert np.all(positions_np >= 0.0)
        assert np.all(positions_np < lengths)


def test_remove_center_of_mass_velocity():
    velocities = remove_center_of_mass_velocity([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], [1.0, 2.0])
    momentum = np.sum(np.array(velocities) * np.array([[1.0], [2.0]]), axis=0)

    np.testing.assert_allclose(momentum, np.zeros(3), atol=1e-6)


def test_thermal_velocities_reach_target_temperature():
    velocities = thermal_velocities(16, temperature=1.5, seed=3)
    velocities = rescale_temperature(velocities, temperature=1.5)
    temperature = float(np.array(instantaneous_temperature(velocities, np.ones(16))))

    np.testing.assert_allclose(temperature, 1.5, rtol=1e-5)
