import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate, simulate_nve
from mlx_atomistic.neighbors import NeighborListManager


def test_neighbor_manager_rebuild_threshold():
    cell = Cell.cubic(10.0)
    manager = NeighborListManager(cell, cutoff=2.5, skin=0.4)
    positions = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)

    manager.update(positions)
    assert manager.rebuild_count == 1
    assert not manager.needs_rebuild(positions + np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert manager.needs_rebuild(positions + np.array([[0.21, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def test_simulate_nve_sparse_sampling_counts():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [2.2, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.002, steps=5, sample_interval=2),
    )

    assert np.array(result.sampled_steps).tolist() == [0, 2, 4, 5]
    assert np.array(result.sampled_positions).shape == (4, 4, 3)
    assert np.array(result.sampled_velocities).shape == (4, 4, 3)
    np.testing.assert_allclose(np.array(result.sampled_time), [0.0, 0.004, 0.008, 0.01])
    assert np.array(result.total_energy).shape == (6,)
    assert np.array(result.temperature).shape == (6,)


def test_simulate_nve_sparse_diagnostics_use_diagnostic_axis():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [2.2, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(
            dt=0.002,
            steps=5,
            sample_interval=5,
            diagnostic_interval=2,
        ),
    )

    assert np.array(result.sampled_steps).tolist() == [0, 5]
    assert np.array(result.diagnostic_steps).tolist() == [0, 2, 4, 5]
    assert np.array(result.total_energy).shape == (4,)


def test_dynamic_neighbor_nve_matches_static_neighbor_for_short_run():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [2.2, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.005, 0.0, 0.0], [-0.005, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, -0.005, 0.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5)
    manager = NeighborListManager(cell, cutoff=2.5, skin=0.4)

    dynamic = simulate_nve(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        neighbor_manager=manager,
        config=SimulationConfig(dt=0.001, steps=5, sample_interval=5),
    )
    dense = simulate(
        positions,
        velocities,
        cell=cell,
        potential=potential,
        dt=0.001,
        steps=5,
    )

    np.testing.assert_allclose(
        np.array(dynamic.total_energy),
        np.array(dense.total_energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(dynamic.energy_drift),
        np.array(dynamic.total_energy) - np.array(dynamic.total_energy)[0],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.array(dynamic.max_energy_drift) < 1e-4
    assert int(np.array(dynamic.rebuild_count)[-1]) == 1
