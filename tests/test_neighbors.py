import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, simulate
from mlx_atomistic.neighbors import build_neighbor_list


def test_neighbor_list_has_unique_expected_pairs():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    neighbors = build_neighbor_list(positions, Cell.cubic(8.0), cutoff=1.5, skin=0.0)
    pairs = np.array(neighbors.pairs)

    assert pairs.tolist() == [[0, 1]]


def test_neighbor_list_lj_matches_all_pairs():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5)
    neighbors = build_neighbor_list(positions, cell, cutoff=2.5, skin=0.0)

    dense_energy, dense_forces = potential.energy_forces(positions, cell)
    pair_energy, pair_forces = potential.energy_forces(positions, cell, pairs=neighbors.pairs)

    np.testing.assert_allclose(np.array(pair_energy), np.array(dense_energy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(pair_forces), np.array(dense_forces), rtol=1e-5, atol=1e-5)


def test_neighbor_list_simulation_runs():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5)
    neighbors = build_neighbor_list(positions, cell, cutoff=2.5)

    result = simulate(
        positions,
        velocities,
        cell=cell,
        potential=potential,
        pairs=neighbors.pairs,
        steps=3,
    )

    assert np.isfinite(np.array(result.total_energy)).all()
