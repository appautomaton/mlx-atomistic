import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, simulate
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
from mlx_atomistic.nonbonded import estimate_dense_nonbonded_bytes


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
    assert neighbors.backend == "periodic_cell_list"
    assert neighbors.estimated_pair_bytes == 8


def test_neighbor_list_is_deterministic_when_periodic_offsets_alias():
    positions = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.35, 0.05, 0.05],
            [1.85, 0.05, 0.05],
            [1.85, 1.85, 1.85],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(2.0)

    first = build_neighbor_list(positions, cell, cutoff=0.45, skin=0.0)
    second = build_neighbor_list(positions, cell, cutoff=0.45, skin=0.0)

    expected = [[0, 1], [0, 2], [0, 3], [2, 3]]
    assert np.array(first.pairs).tolist() == expected
    assert np.array(second.pairs).tolist() == expected
    assert first.pair_count == 4


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


def test_neighbor_list_large_periodic_fixture_uses_compact_pair_storage():
    grid = np.indices((10, 10, 10), dtype=np.float32).reshape(3, -1).T
    positions = 1.2 * grid
    cell = Cell.cubic(12.0)

    neighbors = build_neighbor_list(positions, cell, cutoff=1.25, skin=0.0)

    assert neighbors.pair_count == 3000
    assert tuple(neighbors.stats.n_cells) == (9, 9, 9)
    assert neighbors.estimated_pair_bytes == neighbors.pair_count * 8
    assert neighbors.estimated_pair_bytes < estimate_dense_nonbonded_bytes(
        positions.shape[0],
        components="lj",
    )


def test_neighbor_list_manager_rebuild_policy_is_deterministic():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(Cell.cubic(6.0), cutoff=1.5, skin=0.4, check_interval=2)

    initial = manager.update(positions)
    small_move = positions + np.array([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    still_current = manager.update(small_move)
    checked_current = manager.update(small_move)
    large_move = positions + np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    skipped_check = manager.update(large_move)
    rebuilt = manager.update(large_move)

    assert initial is still_current
    assert initial is checked_current
    assert initial is skipped_check
    assert rebuilt is not initial
    assert manager.rebuild_count == 2
    assert manager.last_max_displacement == 0.0
    np.testing.assert_allclose(np.asarray(manager.reference_positions), large_move)


def test_neighbor_list_manager_rejects_non_finite_positions_after_valid_build():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(Cell.cubic(6.0), cutoff=1.5, skin=0.4, check_interval=2)
    initial = manager.update(positions)
    invalid = positions.copy()
    invalid[0, 0] = np.nan

    with pytest.raises(ValueError, match="positions must be finite"):
        manager.update(invalid)

    assert manager.neighbor_list is initial
    assert manager.rebuild_count == 1


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
