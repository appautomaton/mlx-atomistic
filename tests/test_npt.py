import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.io import RuntimeTraceReporter, load_npz_trajectory, save_npz_trajectory
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    MonteCarloBarostat,
    SimulationConfig,
    simulate_npt,
)
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.protocols import validate_gpcrmd_protocol_request
from mlx_atomistic.topology import Topology


class _ZeroForceTerm:
    name = "zero"
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        return mx.array(0.0, dtype=positions.dtype), mx.zeros_like(positions)


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
    assert result.barostat_attempts == 1
    assert result.barostat_accepted in {0, 1}
    assert result.target_pressure == 0.0
    assert result.cell_lengths.shape == (2, 3)
    assert result.volume.shape == (2,)
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
            interval=5,
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


def test_monte_carlo_barostat_validates_interval_state():
    with pytest.raises(ValueError, match="barostat interval must be positive"):
        MonteCarloBarostat(interval=0)


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
