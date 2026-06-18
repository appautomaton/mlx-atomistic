import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    NoseHooverThermostat,
    SimulationConfig,
    _langevin_block_execution_enabled,
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


def test_nose_hoover_thermostat_validation():
    with pytest.raises(ValueError, match="temperature"):
        NoseHooverThermostat(temperature=0.0)
    with pytest.raises(ValueError, match="relaxation_time"):
        NoseHooverThermostat(relaxation_time=0.0)
    with pytest.raises(ValueError, match="thermal_mass"):
        NoseHooverThermostat(thermal_mass=-1.0)


def _batched_fcc_system(n=256):
    positions, cell = fcc_lattice(n, density=0.8)
    velocities = thermal_velocities(n, temperature=1.0, seed=7)
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        cell,
        LennardJonesPotential(cutoff=2.5),
    )


def test_block_execution_gate_requires_supported_langevin_config():
    config = SimulationConfig(steps=10, block_size=8)
    manager = NeighborListManager(Cell.cubic(6.0), cutoff=2.5, skin=1.0)
    langevin = LangevinThermostat(temperature=1.0, friction=0.5, seed=1)
    # Supported: Langevin + managed neighbors + no constraints/virtual sites.
    assert _langevin_block_execution_enabled(
        config, thermostat=langevin, neighbor_manager=manager,
        constraints=None, virtual_sites=None,
    )
    # block_size == 1 is the per-step path.
    assert not _langevin_block_execution_enabled(
        SimulationConfig(steps=10, block_size=1), thermostat=langevin,
        neighbor_manager=manager, constraints=None, virtual_sites=None,
    )
    # No neighbor manager (dense path) cannot batch.
    assert not _langevin_block_execution_enabled(
        config, thermostat=langevin, neighbor_manager=None,
        constraints=None, virtual_sites=None,
    )
    # Nose-Hoover is not supported by the fast path.
    assert not _langevin_block_execution_enabled(
        config, thermostat=NoseHooverThermostat(temperature=1.0),
        neighbor_manager=manager, constraints=None, virtual_sites=None,
    )


@pytest.mark.parametrize("block_size", [4, 16])
def test_batched_langevin_matches_per_step(block_size):
    """The compiled batched-block fast path must reproduce the per-step loop.

    Same seed + same Langevin substep arithmetic => the batched trajectory
    matches the per-step loop to floating-point precision (the only differences
    are summation-order ULPs from the larger skin's neighbor list, the same class
    of difference as changing the rebuild interval). Sampling/diagnostic cadences
    here are deliberately NOT multiples of block_size to exercise boundary
    capping.
    """
    positions, velocities, cell, potential = _batched_fcc_system()

    def run(bs, skin):
        manager = NeighborListManager(
            cell, cutoff=2.5, skin=skin, check_interval=1, backend="mlx_cell_pairs"
        )
        config = SimulationConfig(
            dt=0.002, steps=120, sample_interval=30, diagnostic_interval=30,
            evaluation_interval=25, block_size=bs,
        )
        return simulate_nvt(
            positions, velocities, cell=cell, force_terms=potential,
            neighbor_manager=manager, config=config,
            thermostat=LangevinThermostat(temperature=1.0, friction=0.5, seed=7),
        )

    reference = run(1, 0.4)
    batched = run(block_size, 1.2)

    assert np.array_equal(
        np.asarray(batched.diagnostic_steps), np.asarray(reference.diagnostic_steps)
    )
    assert np.array_equal(
        np.asarray(batched.sampled_steps), np.asarray(reference.sampled_steps)
    )
    assert np.allclose(
        np.asarray(batched.total_energy), np.asarray(reference.total_energy),
        rtol=0.0, atol=1e-3,
    )
    assert np.allclose(
        np.asarray(batched.sampled_positions), np.asarray(reference.sampled_positions),
        rtol=0.0, atol=1e-3,
    )
    assert bool(np.isfinite(np.asarray(batched.total_energy)).all())


def test_batched_block_size_falls_back_without_neighbor_manager():
    """block_size > 1 on the dense path (no manager) must still run correctly."""
    positions, velocities, cell, potential = _batched_fcc_system(n=108)
    config = SimulationConfig(
        dt=0.002, steps=40, sample_interval=40, diagnostic_interval=40, block_size=8
    )
    result = simulate_nvt(
        positions, velocities, cell=cell, force_terms=potential,
        neighbor_manager=None, config=config,
        thermostat=LangevinThermostat(temperature=1.0, friction=0.5, seed=7),
    )
    assert bool(np.isfinite(np.asarray(result.total_energy)).all())


def test_nose_hoover_nvt_produces_finite_state_and_metadata():
    positions, velocities, cell, potential = _small_system()
    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        config=SimulationConfig(dt=0.001, steps=8, sample_interval=4),
        thermostat=NoseHooverThermostat(temperature=1.0, relaxation_time=0.2),
    )

    assert result.thermostat_metadata["family"] == "nose_hoover"
    assert result.thermostat_metadata["integrator"] == "nose_hoover_velocity_verlet"
    assert result.thermostat_metadata["deterministic_state"] is True
    assert np.isfinite(np.asarray(result.sampled_positions)).all()
    assert np.isfinite(np.asarray(result.sampled_velocities)).all()
    assert np.isfinite(np.asarray(result.total_energy)).all()
    assert np.isfinite(np.asarray(result.temperature)).all()


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
