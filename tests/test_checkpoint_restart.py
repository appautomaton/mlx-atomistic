import numpy as np
import pytest

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.io import (
    load_npz_trajectory,
    load_simulation_checkpoint,
    save_npz_trajectory,
    save_simulation_checkpoint,
)
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    MonteCarloBarostat,
    NoseHooverThermostat,
    SimulationConfig,
    SimulationState,
    simulate_npt,
    simulate_nvt,
)

pytestmark = pytest.mark.integration


def _run_nvt(positions, velocities, *, steps, initial_step=0, initial_time=0.0):
    return simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(
            dt=0.001,
            steps=steps,
            sample_interval=2,
            diagnostic_interval=2,
            initial_step=initial_step,
            initial_time=initial_time,
        ),
        thermostat=LangevinThermostat(
            temperature=1.5,
            friction=1.0,
            seed=17,
            rng_step_offset=initial_step,
        ),
    )


def _run_nose_hoover_nvt(
    positions,
    velocities,
    *,
    steps,
    chain_position=0.0,
    chain_velocity=0.0,
    initial_step=0,
    initial_time=0.0,
):
    return simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(
            dt=0.001,
            steps=steps,
            sample_interval=2,
            diagnostic_interval=2,
            initial_step=initial_step,
            initial_time=initial_time,
        ),
        thermostat=NoseHooverThermostat(
            temperature=1.0,
            relaxation_time=0.2,
            chain_position=chain_position,
            chain_velocity=chain_velocity,
        ),
    )


def test_simulation_checkpoint_resumes_nvt_deterministically(tmp_path):
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)

    continuous = _run_nvt(positions, velocities, steps=6)
    partial = _run_nvt(positions, velocities, steps=3)
    checkpoint_path = tmp_path / "checkpoint.npz"
    save_simulation_checkpoint(
        checkpoint_path,
        partial.final_state,
        cell=Cell.cubic(6.0),
        thermostat={
            "temperature": 1.5,
            "friction": 1.0,
            "seed": 17,
            "rng_step_offset": partial.final_state.step,
        },
        neighbor_policy={"skin": 0.0, "check_interval": 1},
        force_terms=("lj",),
        diagnostic_cursor=int(np.asarray(partial.diagnostic_steps)[-1]),
    )

    checkpoint = load_simulation_checkpoint(checkpoint_path)
    resumed = _run_nvt(
        checkpoint.positions,
        checkpoint.velocities,
        steps=3,
        initial_step=checkpoint.step,
        initial_time=checkpoint.time,
    )

    assert checkpoint.step == 3
    assert checkpoint.diagnostic_cursor == 3
    assert checkpoint.thermostat["rng_step_offset"] == 3
    assert checkpoint.neighbor_policy["check_interval"] == 1
    assert checkpoint.force_terms == ("lj",)
    np.testing.assert_allclose(
        np.asarray(resumed.final_state.positions),
        np.asarray(continuous.final_state.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(resumed.final_state.velocities),
        np.asarray(continuous.final_state.velocities),
        rtol=1e-6,
        atol=1e-6,
    )


def test_nose_hoover_checkpoint_resumes_nvt_deterministically(tmp_path):
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.02, 0.0, 0.0], [-0.01, 0.01, 0.0], [0.0, -0.02, 0.0]],
        dtype=np.float32,
    )

    continuous = _run_nose_hoover_nvt(positions, velocities, steps=6)
    partial = _run_nose_hoover_nvt(positions, velocities, steps=3)
    checkpoint_path = tmp_path / "nose-hoover-checkpoint.npz"
    save_simulation_checkpoint(
        checkpoint_path,
        partial.final_state,
        cell=Cell.cubic(6.0),
        thermostat=partial.thermostat_metadata,
        neighbor_policy={"skin": 0.0, "check_interval": 1},
        force_terms=("lj",),
        diagnostic_cursor=int(np.asarray(partial.diagnostic_steps)[-1]),
    )

    checkpoint = load_simulation_checkpoint(checkpoint_path)
    thermostat_state = checkpoint.thermostat
    resumed = _run_nose_hoover_nvt(
        checkpoint.positions,
        checkpoint.velocities,
        steps=3,
        chain_position=thermostat_state["chain_position"],
        chain_velocity=thermostat_state["chain_velocity"],
        initial_step=checkpoint.step,
        initial_time=checkpoint.time,
    )

    assert checkpoint.thermostat["family"] == "nose_hoover"
    assert checkpoint.thermostat["deterministic_state"] is True
    np.testing.assert_allclose(
        np.asarray(resumed.final_state.positions),
        np.asarray(continuous.final_state.positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(resumed.final_state.velocities),
        np.asarray(continuous.final_state.velocities),
        rtol=1e-6,
        atol=1e-6,
    )
    assert resumed.thermostat_metadata["chain_velocity"] == continuous.thermostat_metadata[
        "chain_velocity"
    ]


def test_run_mlx_writes_and_resumes_checkpoint(tmp_path):
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    first_checkpoint = tmp_path / "checkpoint-2.npz"
    first = run_mlx(
        tmp_path,
        out=tmp_path / "first.npz",
        steps=2,
        sample_interval=2,
        diagnostic_interval=2,
        temperature=0.0,
        minimize_steps=0,
        equilibration_steps=0,
        checkpoint_out=first_checkpoint,
    )

    assert first.runtime_sync_report["runtime_sync_checkpoint_count"] == 1
    assert first.runtime_sync_report["runtime_materialization_checkpoint_count"] == 1
    assert first.nonbonded_report["runtime_sync_checkpoint_count"] == 1
    assert first.nonbonded_report["runtime_materialization_checkpoint_count"] == 1
    first_record = load_npz_trajectory(tmp_path / "first.npz")
    first_runtime = first_record.metadata["nonbonded_runtime"]
    assert first_runtime["runtime_sync_checkpoint_count"] == 1
    assert first_runtime["runtime_materialization_checkpoint_count"] == 1
    assert first_record.metadata["platform_boundary"]["product_runtime"] == "mlx_atomistic"
    assert first_record.metadata["platform_readiness"]["artifact"]["name"] == "artifact"
    checkpoint_record = load_simulation_checkpoint(first_checkpoint)
    assert checkpoint_record.metadata["platform_boundary"]["product_runtime"] == "mlx_atomistic"
    assert checkpoint_record.metadata["platform_readiness"]["protocol"]["status"] == "proof-level"

    resumed_path = tmp_path / "resumed.npz"
    resumed = run_mlx(
        tmp_path,
        out=resumed_path,
        steps=2,
        sample_interval=2,
        diagnostic_interval=2,
        temperature=0.0,
        minimize_steps=0,
        equilibration_steps=0,
        resume_checkpoint=first_checkpoint,
    )
    record = load_npz_trajectory(resumed_path)

    assert resumed.final_state.step == 4
    assert record.sampled_steps.tolist() == [2, 4]
    assert record.metadata["resume_checkpoint"] == str(first_checkpoint)


def test_npt_checkpoint_preserves_final_cell_for_restart_continuation(tmp_path):
    positions = np.array([[1.0, 1.0, 1.0], [2.1, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=Cell.cubic(7.0),
        force_terms=LennardJonesPotential(cutoff=3.0),
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.5, friction=1.0, seed=23),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=0.5,
            interval=2,
            seed=4,
            max_log_volume_scale=0.01,
        ),
    )

    checkpoint_path = tmp_path / "npt-checkpoint.npz"
    save_simulation_checkpoint(
        checkpoint_path,
        result.final_state,
        cell=result.final_cell,
        thermostat={
            "temperature": 0.5,
            "friction": 1.0,
            "seed": 23,
            "rng_step_offset": result.final_state.step,
        },
        neighbor_policy={"ensemble": "NPT", "barostat": "monte_carlo"},
        force_terms=("lj",),
        diagnostic_cursor=result.final_state.step,
        metadata={
            "ensemble": "NPT",
            "barostat": "monte_carlo",
            "barostat_attempts": result.barostat_attempts,
            "barostat_accepted": result.barostat_accepted,
        },
    )

    checkpoint = load_simulation_checkpoint(checkpoint_path)
    restart_cell = Cell.orthorhombic(checkpoint.cell.tolist())
    resumed = simulate_nvt(
        checkpoint.positions,
        checkpoint.velocities,
        masses=checkpoint.masses,
        cell=restart_cell,
        force_terms=LennardJonesPotential(cutoff=3.0),
        config=SimulationConfig(
            dt=0.001,
            steps=2,
            sample_interval=1,
            diagnostic_interval=1,
            initial_step=checkpoint.step,
            initial_time=checkpoint.time,
        ),
        thermostat=LangevinThermostat(
            temperature=0.5,
            friction=1.0,
            seed=23,
            rng_step_offset=checkpoint.step,
        ),
    )

    assert checkpoint.step == result.final_state.step
    assert checkpoint.thermostat["rng_step_offset"] == result.final_state.step
    assert checkpoint.neighbor_policy == {"ensemble": "NPT", "barostat": "monte_carlo"}
    assert checkpoint.metadata["ensemble"] == "NPT"
    assert checkpoint.metadata["barostat_attempts"] == 1
    np.testing.assert_allclose(checkpoint.cell, np.asarray(result.final_cell.lengths))
    assert np.isfinite(checkpoint.positions).all()
    assert np.isfinite(checkpoint.velocities).all()
    assert resumed.final_state.step == result.final_state.step + 2
    assert np.isfinite(np.asarray(resumed.final_state.positions)).all()


def test_triclinic_checkpoint_and_trajectory_preserve_cell_matrix(tmp_path):
    matrix = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    positions = np.asarray([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    result = simulate_nvt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=cell,
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=23),
    )
    checkpoint_state = SimulationState(
        positions=as_mx_array(positions),
        velocities=as_mx_array(velocities),
        masses=as_mx_array([1.0, 1.0]),
        forces=as_mx_array(np.zeros_like(positions)),
        step=1,
        time=0.001,
    )

    checkpoint_path = tmp_path / "triclinic-checkpoint.npz"
    trajectory_path = tmp_path / "triclinic-trajectory.npz"
    save_simulation_checkpoint(checkpoint_path, checkpoint_state, cell=cell)
    save_npz_trajectory(trajectory_path, result, cell=cell)

    checkpoint = load_simulation_checkpoint(checkpoint_path)
    record = load_npz_trajectory(trajectory_path)
    np.testing.assert_allclose(checkpoint.cell, matrix)
    np.testing.assert_allclose(record.cell, matrix)


def test_checkpoint_reports_hmr_state_from_metadata(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    state = SimulationState(
        positions=as_mx_array(positions),
        velocities=as_mx_array(np.zeros_like(positions)),
        masses=as_mx_array([3.024, 13.983]),
        forces=as_mx_array(np.zeros_like(positions)),
        step=2,
        time=0.002,
    )
    hmr_state = {
        "status": "represented_by_masses",
        "policy": {
            "target_hydrogen_mass": 3.024,
            "virtual_sites_supported": False,
        },
        "original_masses": [1.008, 15.999],
        "transformed_masses": [3.024, 13.983],
        "selected_hydrogens": [{"hydrogen_index": 0, "heavy_atom_index": 1}],
    }
    checkpoint_path = tmp_path / "hmr-checkpoint.npz"

    save_simulation_checkpoint(
        checkpoint_path,
        state,
        metadata={"hydrogen_mass_repartitioning": hmr_state},
    )

    checkpoint = load_simulation_checkpoint(checkpoint_path)
    assert checkpoint.hmr_state["status"] == "represented_by_masses"
    assert checkpoint.hmr_state["policy"]["virtual_sites_supported"] is False
    assert checkpoint.hmr_state["transformed_masses"] == [3.024, 13.983]
