import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.io import RuntimeTraceReporter, load_npz_trajectory
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nvt,
)


def test_simulate_nvt_reporter_observes_samples_and_diagnostics_without_changing_result():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    config = SimulationConfig(dt=0.001, steps=4, sample_interval=2, diagnostic_interval=2)
    thermostat = LangevinThermostat(temperature=1.5, friction=1.0, seed=5)

    baseline = simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=config,
        thermostat=thermostat,
    )
    reporter = RuntimeTraceReporter()
    observed = simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=config,
        thermostat=thermostat,
        reporters=reporter,
    )

    np.testing.assert_allclose(
        np.asarray(observed.sampled_positions),
        np.asarray(baseline.sampled_positions),
    )
    np.testing.assert_allclose(np.asarray(observed.total_energy), np.asarray(baseline.total_energy))
    assert [(event["event_type"], event["step"]) for event in reporter.events] == [
        ("sample", 0),
        ("diagnostic", 0),
        ("sample", 2),
        ("diagnostic", 2),
        ("sample", 4),
        ("diagnostic", 4),
    ]
    diagnostic = [event for event in reporter.events if event["event_type"] == "diagnostic"][-1]
    assert diagnostic["total_energy"] is not None
    assert diagnostic["temperature"] is not None
    assert diagnostic["pair_count"] == 1


def test_run_mlx_accepts_reporter_and_keeps_npz_output(tmp_path):
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    trajectory_path = tmp_path / "trajectory.npz"
    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    reporter = RuntimeTraceReporter()

    run_mlx(
        tmp_path,
        out=trajectory_path,
        steps=4,
        sample_interval=2,
        diagnostic_interval=2,
        dt=0.0005,
        temperature=0.0,
        minimize_steps=0,
        equilibration_steps=0,
        reporters=reporter,
    )
    record = load_npz_trajectory(trajectory_path)

    assert trajectory_path.exists()
    assert record.sampled_steps.tolist() == [0, 2, 4]
    assert [event["step"] for event in reporter.events if event["event_type"] == "sample"] == [
        0,
        2,
        4,
    ]
