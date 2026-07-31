import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    NoseHooverThermostat,
    SimulationConfig,
    _diagnostic_cutoff_strain_pairs,
    _evaluate_force_terms,
    _ForceEvaluationRequest,
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


class _DemandCountingTerm:
    name = "demand_counting"
    supports_virial = True
    analytic_virial_supported = True

    def __init__(self):
        self.calls = []

    def _runtime_forces(self, positions, *, cell=None, pairs=None):
        del cell, pairs
        self.calls.append("forces")
        return mx.zeros_like(positions)

    def _runtime_energy(self, positions, *, cell=None, pairs=None):
        del cell, pairs
        self.calls.append("energy")
        return mx.sum(positions * 0.0)

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        self.calls.append("energy_forces")
        return mx.sum(positions * 0.0), mx.zeros_like(positions)

    def energy_forces_with_components(self, positions, cell=None, pairs=None):
        del cell, pairs
        self.calls.append("diagnostic")
        energy = mx.sum(positions * 0.0)
        return energy, mx.zeros_like(positions), {"zero": energy}

    def analytic_virial_tensor(
        self,
        positions,
        *,
        cell=None,
        pairs=None,
        masses=None,
        molecule_ids=None,
    ):
        del cell, pairs, masses, molecule_ids
        self.calls.append("virial")
        return mx.zeros((3, 3), dtype=positions.dtype)


@pytest.mark.parametrize(
    ("evaluation_request", "expected_calls"),
    [
        (_ForceEvaluationRequest("forces"), ["forces"]),
        (_ForceEvaluationRequest("energy_forces"), ["energy_forces"]),
        (_ForceEvaluationRequest("energy"), ["energy"]),
        (
            _ForceEvaluationRequest("diagnostic", virial_mode="analytic"),
            ["diagnostic", "virial"],
        ),
    ],
)
def test_internal_force_evaluation_dispatches_exact_requested_mode(
    evaluation_request,
    expected_calls,
):
    term = _DemandCountingTerm()
    positions = mx.zeros((2, 3), dtype=mx.float32)

    result = _evaluate_force_terms(
        positions,
        (term,),
        request=evaluation_request,
        cell=Cell.cubic(6.0),
        pairs=None,
    )

    assert term.calls == expected_calls
    assert (result.energy is not None) == (
        evaluation_request.mode in {"energy_forces", "energy", "diagnostic"}
    )
    assert (result.forces is not None) == (
        evaluation_request.mode in {"forces", "energy_forces", "diagnostic"}
    )
    assert (result.virial is not None) == (evaluation_request.mode == "diagnostic")


def test_diagnostic_cutoff_strain_pairs_require_complete_verlet_margin():
    positions, _, cell, _ = _small_system()
    manager = NeighborListManager(
        cell,
        cutoff=2.5,
        skin=0.4,
        backend="mlx_cell_pairs",
    )
    neighbor_list = manager.update(positions)

    assert (
        _diagnostic_cutoff_strain_pairs(
            manager,
            neighbor_list,
            cell,
        )
        is neighbor_list.interactions
    )

    manager.updates_since_check = 1
    assert _diagnostic_cutoff_strain_pairs(manager, neighbor_list, cell) is None
    manager.updates_since_check = 0
    assert (
        _diagnostic_cutoff_strain_pairs(
            manager,
            neighbor_list,
            Cell.cubic(7.0),
        )
        is None
    )

    manager.last_max_displacement = 0.2
    assert (
        _diagnostic_cutoff_strain_pairs(
            manager,
            neighbor_list,
            cell,
        )
        is None
    )


def test_internal_analytic_diagnostic_uses_one_combined_owner():
    class CombinedDiagnosticTerm(_DemandCountingTerm):
        def _runtime_energy_forces_with_components_virial(
            self,
            positions,
            cell=None,
            pairs=None,
            *,
            masses=None,
            molecule_ids=None,
        ):
            del cell, pairs, masses, molecule_ids
            self.calls.append("combined_diagnostic")
            energy = mx.sum(positions * 0.0)
            return (
                energy,
                mx.zeros_like(positions),
                {"zero": energy},
                mx.zeros((3, 3), dtype=positions.dtype),
            )

    term = CombinedDiagnosticTerm()
    result = _evaluate_force_terms(
        mx.zeros((2, 3), dtype=mx.float32),
        (term,),
        request=_ForceEvaluationRequest(
            "diagnostic",
            virial_mode="analytic",
        ),
        cell=Cell.cubic(6.0),
        pairs=mx.array([[0, 1]], dtype=mx.int32),
    )

    assert term.calls == ["combined_diagnostic"]
    assert result.optimized_terms == 1
    assert result.fallback_terms == 0
    assert result.virial is not None


def test_internal_analytic_diagnostic_uses_proven_cutoff_strain_pairs():
    class ReusedPairDiagnosticTerm(_DemandCountingTerm):
        def _runtime_energy_forces_with_components_virial_reusing_pairs(
            self,
            positions,
            cell=None,
            pairs=None,
            *,
            masses=None,
            molecule_ids=None,
            cutoff_strain_pairs=None,
        ):
            del cell, pairs, masses, molecule_ids
            self.calls.append(("reused_diagnostic", cutoff_strain_pairs))
            energy = mx.sum(positions * 0.0)
            return (
                energy,
                mx.zeros_like(positions),
                {"zero": energy},
                mx.zeros((3, 3), dtype=positions.dtype),
            )

    term = ReusedPairDiagnosticTerm()
    strain_pairs = mx.array([[0, 1]], dtype=mx.int32)
    result = _evaluate_force_terms(
        mx.zeros((2, 3), dtype=mx.float32),
        (term,),
        request=_ForceEvaluationRequest(
            "diagnostic",
            virial_mode="analytic",
        ),
        cell=Cell.cubic(6.0),
        pairs=strain_pairs,
        cutoff_strain_pairs=strain_pairs,
    )

    assert len(term.calls) == 1
    assert term.calls[0][0] == "reused_diagnostic"
    assert term.calls[0][1] is strain_pairs
    assert result.optimized_terms == 1
    assert result.fallback_terms == 0


@pytest.mark.parametrize("mode", ["forces", "energy"])
def test_internal_force_evaluation_falls_back_when_private_hook_declines(mode):
    class DecliningTerm:
        name = "declining"

        def __init__(self):
            self.calls = []

        def _runtime_forces(self, positions, *, cell=None, pairs=None):
            del positions, cell, pairs
            self.calls.append("private_forces")
            return NotImplemented

        def _runtime_energy(self, positions, *, cell=None, pairs=None):
            del positions, cell, pairs
            self.calls.append("private_energy")
            return NotImplemented

        def energy_forces(self, positions, cell=None, pairs=None):
            del cell, pairs
            self.calls.append("energy_forces")
            return mx.sum(positions * 0.0), mx.zeros_like(positions)

    term = DecliningTerm()
    result = _evaluate_force_terms(
        mx.zeros((2, 3), dtype=mx.float32),
        (term,),
        request=_ForceEvaluationRequest(mode),
        cell=None,
        pairs=mx.array([[0, 1]], dtype=mx.int32),
    )

    assert term.calls == [f"private_{mode}", "energy_forces"]
    assert result.optimized_terms == 0
    assert result.fallback_terms == 1


def test_nvt_ordinary_steps_request_forces_and_boundaries_request_diagnostics():
    term = _DemandCountingTerm()
    positions = np.zeros((2, 3), dtype=np.float32)

    simulate_nvt(
        positions,
        np.zeros_like(positions),
        cell=Cell.cubic(6.0),
        force_terms=term,
        config=SimulationConfig(
            dt=0.001,
            steps=3,
            sample_interval=3,
            diagnostic_interval=3,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=3,
        ),
    )

    assert term.calls == ["diagnostic", "forces", "forces", "diagnostic"]


def test_langevin_thermostat_validation():
    with pytest.raises(ValueError, match="temperature"):
        LangevinThermostat(temperature=-1.0)
    with pytest.raises(ValueError, match="friction"):
        LangevinThermostat(friction=-1.0)


def test_langevin_zero_force_carries_thermostatted_velocity():
    """A zero-force Langevin step must retain its stochastic velocity kick."""

    positions = np.zeros((8, 3), dtype=np.float32)
    result = simulate_nvt(
        positions,
        np.zeros_like(positions),
        masses=np.ones(8, dtype=np.float32),
        force_terms=_DemandCountingTerm(),
        config=SimulationConfig(
            dt=0.01,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=1.0,
            friction=1.0,
            seed=13,
        ),
    )

    final_positions = np.asarray(result.final_state.positions)
    final_velocities = np.asarray(result.final_state.velocities)
    assert float(np.asarray(result.temperature)[-1]) > 0.0
    np.testing.assert_allclose(
        final_velocities,
        2.0 * (final_positions - positions) / 0.01,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_constrained_langevin_matches_rattle_baoab_ordering():
    """A forced constrained step follows the complete RATTLE BAOAB sequence."""

    class ConstantForce:
        def __init__(self, forces):
            self.forces = mx.array(forces, dtype=mx.float32)

        def energy_forces(self, positions, cell=None, pairs=None):
            del cell, pairs
            return -mx.sum(positions * self.forces), self.forces

    dt = 0.1
    positions = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=mx.float32)
    velocities = mx.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=mx.float32)
    masses = mx.ones((2,), dtype=mx.float32)
    forces = mx.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=mx.float32)
    constraints = DistanceConstraints([(0, 1)], distances=[1.0], max_iterations=8)

    velocity_half = velocities + 0.5 * dt * forces / masses[:, None]
    predicted_positions = positions + dt * velocity_half
    expected_positions, _ = constraints.apply_position_step(
        positions,
        predicted_positions,
        masses,
    )
    velocity_after_drift = velocity_half + (
        expected_positions - predicted_positions
    ) / dt
    velocity_after_drift = constraints.apply_velocities(
        expected_positions,
        velocity_after_drift,
        masses,
    )
    expected_velocities = velocity_after_drift + 0.5 * dt * forces / masses[:, None]
    expected_velocities = constraints.apply_velocities(
        expected_positions,
        expected_velocities,
        masses,
    )

    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        force_terms=ConstantForce(forces),
        constraints=constraints,
        config=SimulationConfig(
            dt=dt,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=19,
        ),
    )

    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        np.asarray(expected_positions),
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.velocities),
        np.asarray(expected_velocities),
        rtol=0.0,
        atol=1.0e-7,
    )


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
    # A scheduled center-of-mass operation requires the per-step path.
    assert not _langevin_block_execution_enabled(
        SimulationConfig(
            steps=10,
            block_size=8,
            center_of_mass_motion_interval=1,
        ),
        thermostat=langevin,
        neighbor_manager=manager,
        constraints=None,
        virtual_sites=None,
    )


def test_nvt_center_of_mass_motion_uses_global_step_schedule():
    positions = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    masses = np.asarray([1.0, 2.0], dtype=np.float32)

    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        force_terms=LennardJonesPotential(epsilon=0.0, cutoff=None),
        config=SimulationConfig(
            dt=0.001,
            steps=3,
            initial_step=1,
            initial_time=0.001,
            sample_interval=1,
            diagnostic_interval=1,
            center_of_mass_motion_interval=3,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
    )

    sampled = np.asarray(result.sampled_velocities)
    np.testing.assert_allclose(sampled[1], velocities, atol=1.0e-7)
    momentum_at_step_3 = np.sum(sampled[2] * masses[:, None], axis=0)
    momentum_at_step_4 = np.sum(sampled[3] * masses[:, None], axis=0)
    np.testing.assert_allclose(momentum_at_step_3, np.zeros(3), atol=1.0e-7)
    np.testing.assert_allclose(momentum_at_step_4, np.zeros(3), atol=1.0e-7)


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
    # Batched and per-step agree to float32 summation precision. Use a relative
    # band: at total energies ~1e3 a pure absolute tolerance is backend-fragile,
    # where the mlx-cpu reorder is ~4e-3 absolute but ~3e-6 relative.
    assert np.allclose(
        np.asarray(batched.total_energy), np.asarray(reference.total_energy),
        rtol=1e-5, atol=1e-3,
    )
    assert np.allclose(
        np.asarray(batched.sampled_positions), np.asarray(reference.sampled_positions),
        rtol=0.0, atol=1e-3,
    )
    assert bool(np.isfinite(np.asarray(batched.total_energy)).all())


def test_batched_langevin_zero_force_retains_thermal_noise():
    """Compiled Langevin blocks must carry the random velocity between steps."""

    positions, cell = fcc_lattice(32, density=0.8)
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.zeros_like(positions)
    potential = LennardJonesPotential(epsilon=0.0, cutoff=2.5)

    def run(block_size):
        manager = NeighborListManager(
            cell,
            cutoff=2.5,
            skin=1.2,
            check_interval=1,
            backend="mlx_cell_pairs",
        )
        return simulate_nvt(
            positions,
            velocities,
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            config=SimulationConfig(
                dt=0.002,
                steps=4,
                sample_interval=4,
                diagnostic_interval=4,
                block_size=block_size,
            ),
            thermostat=LangevinThermostat(
                temperature=1.0,
                friction=1.0,
                seed=17,
            ),
        )

    reference = run(1)
    batched = run(4)

    assert float(np.asarray(batched.temperature)[-1]) > 0.0
    np.testing.assert_allclose(
        np.asarray(batched.final_state.positions),
        np.asarray(reference.final_state.positions),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(batched.final_state.velocities),
        np.asarray(reference.final_state.velocities),
        rtol=2.0e-4,
        atol=1.0e-5,
    )


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
