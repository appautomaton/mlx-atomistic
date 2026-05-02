import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)
from mlx_atomistic.neighbors import NeighborListManager


def _small_system():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [2.2, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.02, 0.0, 0.0], [-0.01, 0.01, 0.0], [0.0, -0.02, 0.0], [0.0, 0.01, 0.01]],
        dtype=np.float32,
    )
    return positions, velocities, Cell.cubic(6.0), LennardJonesPotential(cutoff=2.5)


def test_langevin_thermostat_validation():
    with pytest.raises(ValueError, match="temperature"):
        LangevinThermostat(temperature=-1.0)
    with pytest.raises(ValueError, match="friction"):
        LangevinThermostat(friction=-1.0)


def test_simulate_nvt_sparse_sampling_counts_and_temperature_error():
    positions, velocities, cell, potential = _small_system()
    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=SimulationConfig(dt=0.002, steps=5, sample_interval=2),
        thermostat=LangevinThermostat(temperature=1.25, friction=0.5, seed=11),
    )

    assert np.array(result.sampled_steps).tolist() == [0, 2, 4, 5]
    assert np.array(result.sampled_positions).shape == (4, 4, 3)
    assert np.array(result.sampled_velocities).shape == (4, 4, 3)
    np.testing.assert_allclose(np.array(result.sampled_time), [0.0, 0.004, 0.008, 0.01])
    assert np.array(result.total_energy).shape == (6,)
    assert np.array(result.temperature).shape == (6,)
    np.testing.assert_allclose(
        np.array(result.temperature_error),
        np.array(result.temperature) - 1.25,
        rtol=1e-6,
        atol=1e-6,
    )


def test_simulate_nvt_sparse_diagnostics_use_diagnostic_axis():
    positions, velocities, cell, potential = _small_system()
    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=SimulationConfig(
            dt=0.002,
            steps=5,
            sample_interval=5,
            diagnostic_interval=2,
        ),
        thermostat=LangevinThermostat(temperature=1.25, friction=0.5, seed=11),
    )

    assert np.array(result.sampled_steps).tolist() == [0, 5]
    assert np.array(result.diagnostic_steps).tolist() == [0, 2, 4, 5]
    np.testing.assert_allclose(np.array(result.diagnostic_time), [0.0, 0.004, 0.008, 0.01])
    assert np.array(result.total_energy).shape == (4,)
    assert np.array(result.temperature).shape == (4,)


def test_seeded_nvt_runs_are_reproducible():
    positions, velocities, cell, potential = _small_system()
    config = SimulationConfig(dt=0.002, steps=5, sample_interval=5)
    thermostat = LangevinThermostat(temperature=1.0, friction=1.0, seed=3)

    first = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=config,
        thermostat=thermostat,
    )
    second = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=config,
        thermostat=thermostat,
    )

    np.testing.assert_allclose(
        np.array(first.sampled_positions),
        np.array(second.sampled_positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(first.total_energy),
        np.array(second.total_energy),
        rtol=1e-6,
        atol=1e-6,
    )


def test_zero_friction_nvt_matches_nve():
    positions, velocities, cell, potential = _small_system()
    config = SimulationConfig(dt=0.001, steps=5, sample_interval=5)

    nvt = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=config,
        thermostat=LangevinThermostat(temperature=1.0, friction=0.0, seed=19),
    )
    nve = simulate_nve(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=config,
    )

    np.testing.assert_allclose(
        np.array(nvt.total_energy),
        np.array(nve.total_energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(nvt.sampled_positions),
        np.array(nve.sampled_positions),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dynamic_neighbor_nvt_reports_rebuilds():
    positions, velocities, cell, potential = _small_system()
    manager = NeighborListManager(cell, cutoff=2.5, skin=0.4)

    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        neighbor_manager=manager,
        config=SimulationConfig(dt=0.001, steps=3, sample_interval=3),
        thermostat=LangevinThermostat(temperature=1.0, friction=0.2, seed=5),
    )

    assert int(np.array(result.rebuild_count)[-1]) >= 1
    assert int(np.array(result.pair_count)[-1]) > 0
