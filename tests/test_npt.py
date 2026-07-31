import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.md as md_module
from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.io import RuntimeTraceReporter, load_npz_trajectory, save_npz_trajectory
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    MonteCarloBarostat,
    SimulationConfig,
    SimulationState,
    _attempt_barostat_move,
    _barostat_log_acceptance_probability,
    simulate_npt,
    simulate_nvt,
)
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.pme import PMEConfig
from mlx_atomistic.protocols import validate_gpcrmd_protocol_request
from mlx_atomistic.topology import Topology


class _ZeroForceTerm:
    name = "zero"
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        return mx.array(0.0, dtype=positions.dtype), mx.zeros_like(positions)


class _CellScaledHarmonicTerm:
    name = "cell_scaled_harmonic"
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        scale = cell.lengths[0]
        energy = 0.5 * mx.sum(positions * positions) / scale
        return energy, -positions / scale


class _RejectCellChangeTerm:
    name = "reject_cell_change"
    supports_virial = True

    def __init__(self, volume):
        self.volume = float(volume)

    def energy_forces(self, positions, cell=None, pairs=None):
        penalty = 1.0e9 * mx.abs(cell.volume - self.volume)
        return penalty, mx.zeros_like(positions)


class _SourceCellOnlyTerm:
    name = "source_cell_only"
    supports_virial = True

    def __init__(self, cell):
        self.cell_matrix = np.asarray(cell.matrix).copy()
        self.evaluation_count = 0

    def energy_forces(self, positions, cell=None, pairs=None):
        if not np.array_equal(np.asarray(cell.matrix), self.cell_matrix):
            raise AssertionError("candidate energy evaluated before cell admission")
        self.evaluation_count += 1
        return mx.array(0.0, dtype=positions.dtype), mx.zeros_like(positions)


class _ConstraintProjectionTrap:
    pairs = np.asarray([[0, 1]], dtype=np.int32)

    def apply_positions(self, positions, masses, cell=None):
        raise AssertionError("rigid molecular strain must not re-project constraints")

    def max_error(self, positions, cell=None):
        return mx.array(0.0, dtype=positions.dtype)


def _small_bound_pme_term(cell, *, real_cutoff=3.0):
    term = NonbondedPotential(
        sigma=np.ones((4,), dtype=np.float32),
        epsilon=np.zeros((4,), dtype=np.float32),
        charges=np.asarray([0.7, -0.2, -0.3, -0.2], dtype=np.float32),
        cutoff=real_cutoff,
        electrostatics="pme",
        pme_config=PMEConfig(
            mesh_shape=(8, 8, 8),
            alpha=0.35,
            real_cutoff=real_cutoff,
        ),
    )
    return term.bind_pme_plan(cell)


def test_monte_carlo_npt_path_scales_orthorhombic_volume_with_constraints():
    positions = np.array([[1.0, 1.0, 1.0], [2.25, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    cell = Cell.cubic(8.0)
    constraints = DistanceConstraints(
        [(0, 1)],
        distances=[1.25],
        max_iterations=8,
    )

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=cell,
        force_terms=LennardJonesPotential(cutoff=3.0),
        config=SimulationConfig(dt=0.001, steps=4, sample_interval=2, diagnostic_interval=2),
        thermostat=LangevinThermostat(temperature=1.0, friction=1.0, seed=3),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=2,
            seed=3,
            max_log_volume_scale=0.01,
        ),
        constraints=constraints,
    )

    assert result.final_state.step == 4
    assert result.barostat_attempts == 2
    assert result.barostat_accepted in {0, 1, 2}
    assert result.target_pressure == 0.0
    assert result.cell_lengths.shape == (3, 3)
    assert result.volume.shape == (3,)
    np.testing.assert_allclose(np.asarray(result.cell_lengths)[0], np.asarray(cell.lengths))
    assert np.isfinite(np.asarray(result.volume)).all()
    assert np.isfinite(np.asarray(result.cell_lengths)).all()
    assert np.isfinite(np.asarray(result.final_state.positions)).all()
    assert np.isfinite(np.asarray(result.final_state.velocities)).all()
    assert np.all(np.asarray(result.final_cell.lengths) > 0.0)
    final_distance = np.linalg.norm(
        np.asarray(result.final_state.positions)[0] - np.asarray(result.final_state.positions)[1]
    )
    np.testing.assert_allclose(final_distance, 1.25, atol=1e-4)
    final_constraint_error = constraints.max_error(
        result.final_state.positions,
        result.final_cell,
    )
    assert float(np.asarray(final_constraint_error)) < 1e-4


def test_monte_carlo_npt_accepts_isotropic_orthorhombic_box_update(tmp_path):
    positions = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    cell = Cell.orthorhombic([8.0, 9.0, 10.0])

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=cell,
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            max_log_volume_scale=0.02,
        ),
    )

    initial_lengths = np.asarray(cell.lengths)
    final_lengths = np.asarray(result.final_cell.lengths)
    length_ratios = final_lengths / initial_lengths

    assert result.barostat_attempts == 1
    assert result.barostat_accepted == 1
    assert np.asarray(result.volume)[1] > np.asarray(result.volume)[0]
    np.testing.assert_allclose(length_ratios, np.full(3, length_ratios[0]), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        positions * length_ratios[0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.sampled_positions)[-1],
        np.asarray(result.final_state.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.sampled_velocities)[-1],
        np.asarray(result.final_state.velocities),
        rtol=1e-6,
        atol=1e-6,
    )

    trajectory_path = tmp_path / "accepted-npt.npz"
    save_npz_trajectory(trajectory_path, result, cell=result.final_cell)
    record = load_npz_trajectory(trajectory_path)

    np.testing.assert_allclose(
        record.sampled_positions[-1],
        np.asarray(result.final_state.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(record.cell, np.asarray(result.final_cell.lengths), rtol=1e-6)
    np.testing.assert_allclose(
        record.cell_history,
        np.asarray(result.cell_history),
        rtol=1e-6,
        atol=1e-6,
    )


def test_npt_barostat_rebuilds_neighbor_pairs_for_lazy_topology():
    positions = np.array(
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [1.1, 1.1, 0.1]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )
    manager = NeighborListManager(cell, cutoff=1.6, skin=0.2)

    result = simulate_npt(
        positions,
        velocities,
        masses=np.ones((4,), dtype=np.float32),
        cell=cell,
        force_terms=(term,),
        neighbor_manager=manager,
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            max_log_volume_scale=0.02,
        ),
    )

    assert result.barostat_accepted == 1
    assert manager.neighbor_list is not None
    np.testing.assert_allclose(
        np.asarray(manager.cell.matrix),
        np.asarray(result.final_cell.matrix),
    )
    np.testing.assert_allclose(
        np.asarray(result.sampled_positions)[-1],
        np.asarray(result.final_state.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    assert int(np.asarray(result.pair_count)[-1]) == manager.neighbor_list.pair_count
    assert int(np.asarray(result.rebuild_count)[-1]) == manager.rebuild_count
    assert result.nonbonded_report["pair_count"] == manager.neighbor_list.pair_count
    assert result.nonbonded_report["rebuild_count"] == manager.rebuild_count


def test_anisotropic_barostat_scales_enabled_matrix_axes_independently():
    positions = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    cell = Cell.triclinic(
        [
            [8.0, 0.0, 0.0],
            [1.0, 9.0, 0.0],
            [0.5, 0.25, 10.0],
        ]
    )

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=cell,
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            max_log_volume_scale=0.02,
            mode="anisotropic",
            axes=(True, False, True),
        ),
    )

    initial_matrix = np.asarray(cell.matrix)
    final_matrix = np.asarray(result.final_cell.matrix)
    row_scales = np.linalg.norm(final_matrix, axis=1) / np.linalg.norm(initial_matrix, axis=1)

    assert result.barostat_accepted == 1
    assert result.cell_matrix.shape == (2, 3, 3)
    assert result.cell_history.shape == (2, 3, 3)
    assert row_scales[0] != row_scales[2]
    np.testing.assert_allclose(final_matrix[1], initial_matrix[1], rtol=1e-6, atol=1e-6)
    expected_positions = np.asarray(
        result.final_cell.cartesian_coordinates(cell.fractional_coordinates(mx.array(positions)))
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        expected_positions,
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.barostat_metadata["mode"] == "anisotropic"
    assert result.barostat_metadata["axes"] == {"x": True, "y": False, "z": True}


def test_membrane_barostat_reports_explicit_plane_and_normal_policy():
    positions = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    cell = Cell.orthorhombic([8.0, 9.0, 10.0])
    reporter = RuntimeTraceReporter()

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=cell,
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            max_log_volume_scale=0.02,
            mode="semi_isotropic",
            membrane_plane="xy",
            normal_axis="z",
        ),
        reporters=reporter,
    )

    length_ratios = np.asarray(result.final_cell.lengths) / np.asarray(cell.lengths)

    assert result.barostat_accepted == 1
    np.testing.assert_allclose(length_ratios[0], length_ratios[1], rtol=1e-6)
    assert length_ratios[2] != pytest.approx(length_ratios[0])
    assert result.barostat_metadata["mode"] == "membrane"
    assert result.barostat_metadata["membrane_plane"] == "xy"
    assert result.barostat_metadata["normal_axis"] == "z"
    assert result.barostat_metadata["plane_policy"] == "coupled_area"
    assert result.barostat_metadata["normal_policy"] == "independent_length"
    barostat_events = [event for event in reporter.events if event["event_type"] == "barostat"]
    assert len(barostat_events) == 1
    assert barostat_events[0]["barostat"]["mode"] == "membrane"
    assert barostat_events[0]["barostat"]["accepted"] == 1


def test_npt_fails_closed_before_unsupported_virial_pressure_claim():
    class UnsupportedForceTerm:
        name = "unsupported_bias"

        def energy_forces(self, positions, cell=None, pairs=None):
            return positions[:, 0].sum() * 0.0, positions * 0.0

    positions = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)

    with pytest.raises(ValueError, match="unsupported_bias"):
        simulate_npt(
            positions,
            velocities,
            masses=np.asarray([1.0, 1.0], dtype=np.float32),
            cell=Cell.cubic(8.0),
            force_terms=UnsupportedForceTerm(),
            config=SimulationConfig(dt=0.001, steps=1),
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
            barostat=MonteCarloBarostat(mode="anisotropic"),
        )


def test_npt_analytic_pressure_fails_before_oracle_only_term_is_evaluated():
    class OracleOnlyForceTerm:
        name = "oracle_only_bias"
        supports_virial = True

        def energy_forces(self, positions, cell=None, pairs=None):
            raise AssertionError("integration must not start")

    positions = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)

    with pytest.raises(ValueError, match="missing analytic virial.*oracle_only_bias"):
        simulate_npt(
            positions,
            velocities,
            masses=np.asarray([1.0, 1.0], dtype=np.float32),
            cell=Cell.cubic(8.0),
            force_terms=OracleOnlyForceTerm(),
            config=SimulationConfig(
                dt=0.001,
                steps=1,
                pressure_virial_mode="analytic",
            ),
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=11),
            barostat=MonteCarloBarostat(mode="anisotropic"),
        )


def test_monte_carlo_barostat_validates_interval_state():
    with pytest.raises(ValueError, match="barostat interval must be positive"):
        MonteCarloBarostat(interval=0)


def test_center_of_mass_motion_interval_must_be_positive():
    with pytest.raises(ValueError, match="center_of_mass_motion_interval"):
        SimulationConfig(center_of_mass_motion_interval=0)


def test_npt_shorter_than_interval_matches_nvt_without_attempt():
    positions = np.asarray(
        [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    masses = np.asarray([1.0, 2.0], dtype=np.float32)
    cell = Cell.cubic(8.0)
    config = SimulationConfig(
        dt=0.001,
        steps=4,
        sample_interval=2,
        diagnostic_interval=2,
    )
    thermostat = LangevinThermostat(
        temperature=1.0,
        friction=1.0,
        seed=17,
    )

    nvt = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=_ZeroForceTerm(),
        config=config,
        thermostat=thermostat,
    )
    npt = simulate_npt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=_ZeroForceTerm(),
        config=config,
        thermostat=thermostat,
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=5,
            seed=3,
        ),
    )

    assert npt.barostat_attempts == 0
    assert npt.barostat_accepted == 0
    np.testing.assert_array_equal(
        np.asarray(npt.sampled_positions),
        np.asarray(nvt.sampled_positions),
    )
    np.testing.assert_array_equal(
        np.asarray(npt.sampled_velocities),
        np.asarray(nvt.sampled_velocities),
    )
    np.testing.assert_array_equal(
        np.asarray(npt.final_state.forces),
        np.asarray(nvt.final_state.forces),
    )
    np.testing.assert_array_equal(
        np.asarray(npt.cell_matrix),
        np.broadcast_to(
            np.asarray(cell.matrix),
            npt.cell_matrix.shape,
        ),
    )


def test_npt_preserves_continuous_molecule_images_between_cell_moves():
    positions = np.asarray(
        [[7.9, 1.0, 1.0], [8.9, 1.0, 1.0]],
        dtype=np.float32,
    )
    constraints = DistanceConstraints(
        [(0, 1)],
        distances=[1.0],
        max_iterations=4,
    )

    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((2,), dtype=np.float32),
        molecule_ids=np.asarray([0, 0], dtype=np.int32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=2,
            seed=4,
            mode="anisotropic",
        ),
        constraints=constraints,
    )

    final = np.asarray(result.final_state.positions)
    assert final[1, 0] > 8.0
    np.testing.assert_allclose(final[1] - final[0], [1.0, 0.0, 0.0], atol=1.0e-6)
    assert float(np.max(np.asarray(result.constraint_max_error))) <= 1.0e-6


def test_npt_schedules_global_steps_and_advances_one_rng_stream():
    positions = np.asarray(
        [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]],
        dtype=np.float32,
    )
    reporter = RuntimeTraceReporter()
    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((2,), dtype=np.float32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=8,
            initial_step=7,
            initial_time=0.007,
            sample_interval=5,
            diagnostic_interval=5,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=5,
            seed=9,
            mode="anisotropic",
        ),
        reporters=reporter,
    )

    events = [
        event
        for event in reporter.events
        if event["event_type"] == "barostat"
    ]
    history = result.barostat_metadata["proposal_history"]
    assert result.final_state.step == 15
    assert result.barostat_attempts == 2
    assert [event["step"] for event in events] == [10, 15]
    assert len(history) == 2
    assert history[0]["scale_factors"] != history[1]["scale_factors"]
    assert all(item["log_reverse_over_forward"] == 0.0 for item in history)
    assert sum(result.barostat_metadata["axis_attempts"].values()) == 2


def test_anisotropic_barostat_adapts_each_axis_proposal_width():
    cell = Cell.cubic(8.0)
    positions = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros((1, 3), dtype=np.float32)
    masses = np.ones((1,), dtype=np.float32)
    thermostat = LangevinThermostat(
        temperature=0.0,
        friction=0.0,
        seed=11,
    )
    barostat = MonteCarloBarostat(
        pressure=0.0,
        temperature=1.0,
        interval=1,
        seed=9,
        mode="anisotropic",
        axes=(True, False, False),
        max_log_volume_scale=float(np.log1p(0.01)),
    )
    result = simulate_npt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=_RejectCellChangeTerm(float(np.asarray(cell.volume))),
        config=SimulationConfig(
            dt=0.001,
            steps=11,
            sample_interval=1,
            diagnostic_interval=1,
        ),
        thermostat=thermostat,
        barostat=barostat,
    )

    initial_step = 0.01 * float(np.asarray(cell.volume))
    history = result.barostat_metadata["proposal_history"]
    assert result.barostat_attempts == 11
    assert result.barostat_accepted == 0
    assert [record["volume_step"] for record in history[:10]] == pytest.approx(
        [initial_step] * 10
    )
    assert history[10]["volume_step"] == pytest.approx(initial_step / 1.1)
    assert result.barostat_metadata["proposal_volume_steps"]["x"] == (
        pytest.approx(initial_step / 1.1)
    )
    assert result.barostat_metadata["adaptation_attempts"]["x"] == 1
    assert result.barostat_metadata["adaptation_accepted"]["x"] == 0

    first = simulate_npt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=_RejectCellChangeTerm(float(np.asarray(cell.volume))),
        config=SimulationConfig(
            dt=0.001,
            steps=10,
            sample_interval=1,
            diagnostic_interval=1,
        ),
        thermostat=thermostat,
        barostat=barostat,
    )
    resumed = simulate_npt(
        first.final_state.positions,
        first.final_state.velocities,
        masses=first.final_state.masses,
        cell=first.final_cell,
        force_terms=_RejectCellChangeTerm(float(np.asarray(cell.volume))),
        config=SimulationConfig(
            dt=0.001,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            initial_step=first.final_state.step,
            initial_time=first.final_state.time,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
            rng_step_offset=first.final_state.step,
        ),
        barostat=barostat,
        barostat_state=first.barostat_metadata,
    )

    assert resumed.barostat_metadata["proposal_history"] == history
    assert resumed.barostat_metadata["proposal_volume_steps"] == (
        result.barostat_metadata["proposal_volume_steps"]
    )


def test_npt_records_selected_center_of_mass_motion_schedule():
    result = simulate_npt(
        np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        masses=np.asarray([1.0, 2.0], dtype=np.float32),
        molecule_ids=np.asarray([0, 1], dtype=np.int32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=2,
            sample_interval=1,
            diagnostic_interval=1,
            center_of_mass_motion_interval=1,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=5,
            seed=9,
        ),
    )

    final_momentum = np.sum(
        np.asarray(result.final_state.velocities)
        * np.asarray(result.final_state.masses)[:, None],
        axis=0,
    )
    np.testing.assert_allclose(final_momentum, np.zeros(3), atol=1.0e-7)
    assert result.barostat_metadata["center_of_mass_motion_interval"] == 1


def test_anisotropic_proposal_translates_molecule_centers_without_stretching():
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.2, 1.1],
            [4.0, 4.0, 4.0],
            [4.8, 4.1, 4.3],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([12.0, 1.0, 16.0, 1.0], dtype=np.float32),
        molecule_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            mode="anisotropic",
        ),
    )

    final = np.asarray(result.final_state.positions)
    assert result.barostat_accepted == 1
    assert result.barostat_metadata["molecule_count"] == 2
    np.testing.assert_allclose(
        final[1] - final[0],
        positions[1] - positions[0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        final[3] - final[2],
        positions[3] - positions[2],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(result.final_state.velocities),
        velocities,
    )


def test_barostat_molecular_translation_does_not_reproject_constraints():
    positions = mx.array(
        [[1.0, 1.0, 1.0], [2.0, 1.2, 1.1]],
        dtype=mx.float32,
    )
    cell = Cell.cubic(8.0)
    state = SimulationState(
        positions=positions,
        velocities=mx.zeros_like(positions),
        masses=mx.array([12.0, 1.0], dtype=mx.float32),
        forces=mx.zeros_like(positions),
        step=1,
        time=0.001,
    )

    _, _, _, accepted, proposal = _attempt_barostat_move(
        state,
        (_ZeroForceTerm(),),
        cell,
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
            mode="anisotropic",
        ),
        rng=np.random.default_rng(4),
        volume_step=0.01 * float(np.asarray(cell.volume)),
        constraints=_ConstraintProjectionTrap(),
        boltzmann_constant=1.0,
        molecule_ids=np.asarray([0, 0], dtype=np.int32),
    )

    assert accepted
    assert proposal.delta_energy == pytest.approx(0.0)
    assert proposal.log_acceptance is not None


def test_accepted_move_rebuilds_constraint_free_block_path_for_new_cell():
    positions = np.asarray(
        [[1.0, 1.0, 1.0], [2.0, 1.5, 1.25], [3.0, 2.0, 1.5]],
        dtype=np.float32,
    )
    cell = Cell.cubic(8.0)
    term = _CellScaledHarmonicTerm()
    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((3,), dtype=np.float32),
        cell=cell,
        force_terms=term,
        neighbor_manager=NeighborListManager(
            cell,
            cutoff=2.0,
            skin=0.5,
        ),
        config=SimulationConfig(
            dt=0.001,
            steps=5,
            sample_interval=5,
            diagnostic_interval=5,
            pressure_diagnostics=False,
            block_size=3,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0e9,
            interval=3,
            seed=4,
            mode="anisotropic",
        ),
    )

    fresh_energy, fresh_forces = term.energy_forces(
        result.final_state.positions,
        cell=result.final_cell,
    )
    assert result.barostat_attempts == 1
    assert result.barostat_accepted == 1
    np.testing.assert_allclose(
        np.asarray(result.potential_energy)[-1],
        np.asarray(fresh_energy),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.forces),
        np.asarray(fresh_forces),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_accepted_pme_move_commits_plan_neighbor_energy_and_forces_together():
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(12.0)
    term = _small_bound_pme_term(cell)
    manager = NeighborListManager(
        cell,
        cutoff=3.0,
        skin=0.2,
        backend="mlx_cell_blocks",
        block_size=2,
    )

    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((4,), dtype=np.float32),
        cell=cell,
        force_terms=term,
        neighbor_manager=manager,
        config=SimulationConfig(
            dt=0.001,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0e9,
            interval=1,
            seed=4,
        ),
    )

    final_term = result.final_force_terms[0]
    fresh_term = term.bind_pme_plan(result.final_cell)
    fresh_energy, fresh_forces = fresh_term.energy_forces(
        result.final_state.positions,
        cell=result.final_cell,
        pairs=manager.neighbor_list.interactions,
    )
    history = result.barostat_metadata["proposal_history"][0]
    assert result.barostat_accepted == 1
    assert manager.cell is result.final_cell
    assert final_term.pme_plan.cell is result.final_cell
    assert final_term.pme_plan.config.mesh_shape == term.pme_plan.config.mesh_shape
    assert final_term.pme_plan.config.alpha == term.pme_plan.config.alpha
    assert final_term.pme_plan.config.real_cutoff == term.pme_plan.config.real_cutoff
    assert history["source_pme_plan_fingerprints"] == [term.pme_plan.fingerprint]
    assert history["candidate_pme_plan_fingerprints"] == [
        final_term.pme_plan.fingerprint
    ]
    assert final_term.pme_plan.fingerprint != term.pme_plan.fingerprint
    np.testing.assert_allclose(
        np.asarray(result.potential_energy)[-1],
        np.asarray(fresh_energy),
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.forces),
        np.asarray(fresh_forces),
        atol=2.0e-5,
    )


@pytest.mark.gpu
def test_dynamic_cell_pme_plan_gpu_matches_cpu(monkeypatch):
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )

    def run(device):
        mx.set_default_device(device)
        mx.set_default_stream(mx.new_stream(device))
        cell = Cell.cubic(12.0)
        return simulate_npt(
            positions,
            np.zeros_like(positions),
            masses=np.ones((4,), dtype=np.float32),
            cell=cell,
            force_terms=_small_bound_pme_term(cell),
            neighbor_manager=NeighborListManager(
                cell,
                cutoff=3.0,
                skin=0.2,
                backend="mlx_cell_blocks",
                block_size=2,
            ),
            config=SimulationConfig(
                dt=0.001,
                steps=1,
                sample_interval=1,
                diagnostic_interval=1,
                pressure_diagnostics=False,
            ),
            thermostat=LangevinThermostat(
                temperature=0.0,
                friction=0.0,
                seed=11,
            ),
            barostat=MonteCarloBarostat(
                pressure=0.0,
                temperature=1.0e9,
                interval=1,
                seed=4,
            ),
        )

    previous = mx.default_device()
    cpu = mx.Device(mx.cpu, 0)
    cpu_result = run(cpu)
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    try:
        gpu = mx.Device(mx.gpu, 0)
        gpu_result = run(gpu)
        mx.eval(
            gpu_result.final_state.positions,
            gpu_result.final_state.forces,
            gpu_result.potential_energy,
        )
    except Exception:  # noqa: BLE001 - any Metal load failure means skip.
        mx.set_default_device(previous)
        mx.set_default_stream(mx.new_stream(previous))
        pytest.skip("Metal GPU unavailable")
    try:
        np.testing.assert_allclose(
            np.asarray(gpu_result.final_cell.matrix),
            np.asarray(cpu_result.final_cell.matrix),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(gpu_result.potential_energy),
            np.asarray(cpu_result.potential_energy),
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            np.asarray(gpu_result.final_state.forces),
            np.asarray(cpu_result.final_state.forces),
            atol=2.0e-5,
        )
    finally:
        mx.set_default_device(previous)
        mx.set_default_stream(mx.new_stream(previous))


def test_rejected_pme_moves_preserve_complete_plan_and_neighbor_state():
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(12.0)
    term = _small_bound_pme_term(cell)
    reject = _RejectCellChangeTerm(cell.volume)
    manager = NeighborListManager(
        cell,
        cutoff=3.0,
        skin=0.2,
        backend="mlx_cell_blocks",
        block_size=2,
    )
    original_neighbors = manager.update(positions)
    original_reference = manager.reference_positions
    original_rebuild_count = manager.rebuild_count

    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((4,), dtype=np.float32),
        cell=cell,
        force_terms=(term, reject),
        neighbor_manager=manager,
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
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
        ),
    )

    assert result.barostat_attempts == 3
    assert result.barostat_accepted == 0
    assert result.final_cell is cell
    assert result.final_force_terms[0] is term
    assert result.final_force_terms[0].pme_plan is term.pme_plan
    assert manager.cell is cell
    assert manager.neighbor_list is original_neighbors
    assert manager.reference_positions is original_reference
    assert manager.rebuild_count == original_rebuild_count
    assert all(
        record["candidate_pme_plan_fingerprints"]
        != record["source_pme_plan_fingerprints"]
        for record in result.barostat_metadata["proposal_history"]
    )


def test_rejected_move_computes_the_committed_diagnostic_once(monkeypatch):
    positions = np.asarray(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(8.0)
    recompute_count = 0
    committed_diagnostic = md_module._npt_production_with_final_barostat_state

    def recording_recompute(*args, **kwargs):
        nonlocal recompute_count
        recompute_count += 1
        return committed_diagnostic(*args, **kwargs)

    monkeypatch.setattr(
        md_module,
        "_npt_production_with_final_barostat_state",
        recording_recompute,
    )
    result = simulate_npt(
        positions,
        np.zeros_like(positions),
        masses=np.ones((2,), dtype=np.float32),
        cell=cell,
        force_terms=_RejectCellChangeTerm(cell.volume),
        config=SimulationConfig(
            dt=0.001,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
        ),
    )

    assert result.barostat_attempts == 1
    assert result.barostat_accepted == 0
    assert recompute_count == 1


def test_npt_computes_one_pressure_frame_per_committed_boundary(monkeypatch):
    legacy_pressure_call_count = 0
    pressure_from_virial_call_count = 0
    pressure_diagnostics = md_module._pressure_diagnostics
    pressure_from_virial = md_module._pressure_diagnostics_from_virial

    def recording_pressure(*args, **kwargs):
        nonlocal legacy_pressure_call_count
        legacy_pressure_call_count += 1
        return pressure_diagnostics(*args, **kwargs)

    def recording_pressure_from_virial(*args, **kwargs):
        nonlocal pressure_from_virial_call_count
        pressure_from_virial_call_count += 1
        return pressure_from_virial(*args, **kwargs)

    monkeypatch.setattr(
        md_module,
        "_pressure_diagnostics",
        recording_pressure,
    )
    monkeypatch.setattr(
        md_module,
        "_pressure_diagnostics_from_virial",
        recording_pressure_from_virial,
    )
    result = simulate_npt(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        masses=np.ones((1,), dtype=np.float32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=4,
            sample_interval=2,
            diagnostic_interval=2,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=2,
            seed=4,
        ),
    )

    assert legacy_pressure_call_count == 0
    assert pressure_from_virial_call_count == 3
    assert np.asarray(result.pressure).shape == (3,)


def test_barostat_uses_supplied_current_energy_without_recomputing_it():
    class CountingTerm:
        name = "counting"

        def __init__(self):
            self.calls = 0

        def energy_forces(self, positions, cell=None, pairs=None):
            del cell, pairs
            self.calls += 1
            return mx.sum(positions * 0.0), mx.zeros_like(positions)

    term = CountingTerm()
    positions = mx.array([[1.0, 1.0, 1.0]], dtype=mx.float32)
    state = SimulationState(
        positions=positions,
        velocities=mx.zeros_like(positions),
        masses=mx.ones((1,), dtype=mx.float32),
        forces=mx.zeros_like(positions),
        step=0,
        time=0.0,
    )
    _attempt_barostat_move(
        state,
        (term,),
        Cell.cubic(8.0),
        current_energy=mx.array(0.0, dtype=mx.float32),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=1,
            seed=4,
        ),
        rng=np.random.default_rng(4),
        volume_step=0.01,
        constraints=None,
        boltzmann_constant=1.0,
    )

    assert term.calls == 1


def test_npt_reuses_committed_energy_and_forces_at_the_next_segment(monkeypatch):
    evaluator_calls = 0
    make_evaluator = md_module._make_energy_forces_evaluator

    def recording_factory(*args, **kwargs):
        evaluator = make_evaluator(*args, **kwargs)

        def recording_evaluator(positions):
            nonlocal evaluator_calls
            evaluator_calls += 1
            return evaluator(positions)

        return recording_evaluator

    monkeypatch.setattr(
        md_module,
        "_make_energy_forces_evaluator",
        recording_factory,
    )
    result = simulate_npt(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        masses=np.ones((1,), dtype=np.float32),
        cell=Cell.cubic(8.0),
        force_terms=_ZeroForceTerm(),
        config=SimulationConfig(
            dt=0.001,
            steps=4,
            sample_interval=2,
            diagnostic_interval=2,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=11,
        ),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            interval=2,
            seed=4,
        ),
    )

    assert result.barostat_attempts == 2
    assert evaluator_calls == result.barostat_attempts


def test_invalid_candidate_pme_cutoff_fails_before_candidate_energy():
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(12.0)
    guard = _SourceCellOnlyTerm(cell)

    with pytest.raises(ValueError, match="half the minimum box length"):
        simulate_npt(
            positions,
            np.zeros_like(positions),
            masses=np.ones((4,), dtype=np.float32),
            cell=cell,
            force_terms=(
                _small_bound_pme_term(cell, real_cutoff=5.9),
                guard,
            ),
            neighbor_manager=NeighborListManager(
                cell,
                cutoff=5.9,
                skin=0.0,
                backend="mlx_cell_blocks",
                block_size=2,
            ),
            config=SimulationConfig(
                dt=0.001,
                steps=1,
                sample_interval=1,
                diagnostic_interval=1,
                pressure_diagnostics=False,
            ),
            thermostat=LangevinThermostat(
                temperature=0.0,
                friction=0.0,
                seed=11,
            ),
            barostat=MonteCarloBarostat(
                pressure=0.0,
                temperature=1.0,
                interval=1,
                seed=3,
                max_log_volume_scale=0.1,
            ),
        )

    assert guard.evaluation_count > 0


def test_barostat_acceptance_uses_molecule_count_and_proposal_ratio():
    two_molecules = _barostat_log_acceptance_probability(
        delta_energy=0.5,
        pressure=0.2,
        old_volume=10.0,
        new_volume=11.0,
        molecule_count=2,
        beta=3.0,
        log_reverse_over_forward=0.25,
    )
    four_molecules = _barostat_log_acceptance_probability(
        delta_energy=0.5,
        pressure=0.2,
        old_volume=10.0,
        new_volume=11.0,
        molecule_count=4,
        beta=3.0,
        log_reverse_over_forward=0.25,
    )

    assert four_molecules - two_molecules == pytest.approx(
        2.0 * np.log(1.1)
    )


def test_protocol_gate_accepts_first_monte_carlo_npt_path():
    report = validate_gpcrmd_protocol_request(
        {"ensemble": "NPT", "barostat": "monte_carlo"},
        raise_on_blockers=True,
    )

    assert report.accepted
    assert report.ensemble == "NPT"
    assert report.metadata["proof_mode"] == "short_npt"
    assert report.metadata["barostat"] == "monte_carlo"
    assert report.metadata["barostat_status"] == "supported_monte_carlo"


def test_protocol_gate_accepts_membrane_monte_carlo_npt_path():
    report = validate_gpcrmd_protocol_request(
        {"ensemble": "NPT", "barostat": "monte_carlo", "membrane_barostat": "xy"},
        raise_on_blockers=True,
    )

    assert report.accepted
    assert report.metadata["barostat"] == "monte_carlo_membrane"
    assert report.metadata["barostat_mode"] == "membrane"
    assert report.metadata["membrane_barostat"] is True


def test_protocol_gate_rejects_npt_with_nvt_proof_mode():
    report = validate_gpcrmd_protocol_request(
        {"ensemble": "NPT", "barostat": "monte_carlo", "proof_mode": "short_nvt"},
    )

    assert not report.accepted
    assert report.ensemble == "NPT"
    assert report.metadata["proof_mode"] == "short_npt"
    assert report.blockers == ("unsupported_proof_mode",)
    assert report.metadata["unsupported_protocol_blockers"] == ["unsupported_proof_mode"]
