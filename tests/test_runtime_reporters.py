import numpy as np
import pytest

import mlx_atomistic.md as md
from mlx_atomistic.core import Cell
from mlx_atomistic.io import RuntimeTraceReporter, load_npz_trajectory
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    NoseHooverThermostat,
    SimulationConfig,
    simulate_nve,
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
    assert diagnostic["thermostat"]["family"] == "langevin_baoab"


def test_nose_hoover_reporter_events_identify_thermostat_family():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    reporter = RuntimeTraceReporter()

    result = simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=3, sample_interval=3),
        thermostat=NoseHooverThermostat(temperature=1.0, relaxation_time=0.2),
        reporters=reporter,
    )

    assert result.thermostat_metadata["family"] == "nose_hoover"
    assert {event["thermostat"]["family"] for event in reporter.events} == {"nose_hoover"}
    diagnostic = [event for event in reporter.events if event["event_type"] == "diagnostic"][-1]
    assert diagnostic["thermostat"]["chain_velocity"] == result.thermostat_metadata[
        "chain_velocity"
    ]


@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_sampled_runtime_evaluation_reuses_materialized_state(monkeypatch, ensemble):
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    config = SimulationConfig(
        dt=0.001,
        steps=2,
        sample_interval=1,
        diagnostic_interval=1,
        evaluation_interval=1,
        pressure_diagnostics=False,
    )
    reporter = RuntimeTraceReporter()
    eval_argument_counts = []
    original_eval = md.mx.eval

    def recording_eval(*args):
        eval_argument_counts.append(len(args))
        return original_eval(*args)

    monkeypatch.setattr(md.mx, "eval", recording_eval)
    kwargs = {
        "cell": Cell.cubic(6.0),
        "force_terms": LennardJonesPotential(cutoff=2.5),
        "config": config,
        "reporters": reporter,
    }
    if ensemble == "nve":
        result = simulate_nve(positions, velocities, **kwargs)
    else:
        result = simulate_nvt(
            positions,
            velocities,
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=5),
            **kwargs,
        )

    assert np.asarray(result.sampled_steps).tolist() == [0, 1, 2]
    assert np.asarray(result.diagnostic_steps).tolist() == [0, 1, 2]
    assert [(event["event_type"], event["step"]) for event in reporter.events] == [
        ("sample", 0),
        ("diagnostic", 0),
        ("sample", 1),
        ("diagnostic", 1),
        ("sample", 2),
        ("diagnostic", 2),
    ]
    assert result.runtime_sync_report["runtime_sync_explicit_user_output_count"] == 2
    assert result.runtime_sync_report["runtime_sync_final_state_count"] == 1
    assert result.runtime_sync_report["runtime_sync_diagnostic_count"] == 2
    assert result.runtime_sync_report["runtime_sync_failure_check_count"] == 0
    assert result.runtime_sync_report["runtime_materialization_reporter_count"] == 6
    assert result.nonbonded_report["runtime_sync_diagnostic_count"] == 2
    assert result.nonbonded_report["runtime_materialization_checkpoint_count"] == 0
    assert eval_argument_counts.count(2) >= 3
    assert 11 not in eval_argument_counts
    assert 9 in eval_argument_counts


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
    assert np.isfinite(record.sampled_positions).all()
    assert np.isfinite(record.total_energy).all()
    assert record.metadata["platform_boundary"]["product_runtime"] == "mlx_atomistic"
    assert "validation" in record.metadata["platform_boundary"]["sections"]
    assert record.metadata["platform_readiness"]["artifact"]["status"] == "proof-level"
    assert record.metadata["platform_readiness"]["protocol"]["status"] == "proof-level"
    assert [event["step"] for event in reporter.events if event["event_type"] == "sample"] == [
        0,
        2,
        4,
    ]
