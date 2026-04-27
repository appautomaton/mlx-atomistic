import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, simulate


def test_lj_energy_minimum_near_sigma_root_sixth_two():
    potential = LennardJonesPotential(cutoff=None)
    r_min = 2.0 ** (1.0 / 6.0)
    energy, forces = potential.energy_forces(np.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]]))

    np.testing.assert_allclose(np.array(energy), -1.0, atol=1e-6)
    np.testing.assert_allclose(np.array(forces), np.zeros((2, 3)), atol=1e-5)


def test_lj_force_repulsive_and_attractive_regions():
    potential = LennardJonesPotential(cutoff=None)

    _, repulsive = potential.energy_forces(np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]))
    _, attractive = potential.energy_forces(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))

    assert np.array(repulsive)[0, 0] < 0.0
    assert np.array(attractive)[0, 0] > 0.0


def test_lj_forces_are_pairwise_antisymmetric():
    potential = LennardJonesPotential(cutoff=None)
    _, forces = potential.energy_forces(np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]))
    forces = np.array(forces)

    np.testing.assert_allclose(forces[0], -forces[1], atol=1e-6)


def test_lj_periodic_minimum_image_energy_matches_short_distance():
    potential = LennardJonesPotential(cutoff=None)
    cell = Cell.cubic(10.0)

    periodic_energy, _ = potential.energy_forces(
        np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        cell,
    )
    direct_energy, _ = potential.energy_forces(np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(np.array(periodic_energy), np.array(direct_energy), atol=1e-6)


def test_short_nve_simulation_keeps_total_energy_bounded():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.3, 1.0, 1.0],
            [1.0, 2.3, 1.0],
            [2.3, 2.3, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [0.02, 0.01, 0.0],
            [-0.02, 0.01, 0.0],
            [0.02, -0.01, 0.0],
            [-0.02, -0.01, 0.0],
        ],
        dtype=np.float32,
    )
    result = simulate(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        potential=LennardJonesPotential(cutoff=2.5),
        dt=0.002,
        steps=25,
    )

    total_energy = np.array(result.total_energy)
    drift = np.max(np.abs(total_energy - total_energy[0]))
    assert np.isfinite(total_energy).all()
    assert drift < 1e-3
