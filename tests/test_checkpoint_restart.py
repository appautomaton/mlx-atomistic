import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.io import (
    load_npz_trajectory,
    load_simulation_checkpoint,
    save_simulation_checkpoint,
)
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nvt,
)


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


def test_run_mlx_writes_and_resumes_checkpoint(tmp_path):
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    first_checkpoint = tmp_path / "checkpoint-2.npz"
    run_mlx(
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
