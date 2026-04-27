import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.diagnostics import summarize_md_result
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)


def test_summarize_nve_result_reports_scalar_diagnostics():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=2),
    )

    summary = summarize_md_result(result)

    assert summary["ensemble"] == "nve"
    assert summary["steps"] == 2
    assert isinstance(summary["max_energy_drift"], float)
    assert isinstance(summary["final_pair_count"], int)
    assert isinstance(summary["final_rebuild_count"], int)


def test_summarize_nvt_result_reports_temperature_diagnostics():
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    result = simulate_nvt(
        positions,
        velocities,
        cell=Cell.cubic(6.0),
        force_terms=LennardJonesPotential(cutoff=2.5),
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=2),
        thermostat=LangevinThermostat(temperature=1.5, friction=1.0, seed=2),
    )

    summary = summarize_md_result(result)

    assert summary["ensemble"] == "nvt"
    assert summary["target_temperature"] == 1.5
    assert isinstance(summary["final_temperature_error"], float)
    assert isinstance(summary["mean_temperature_error"], float)
