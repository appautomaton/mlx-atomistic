import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    MonteCarloBarostat,
    SimulationConfig,
    simulate_npt,
)
from mlx_atomistic.protocols import validate_gpcrmd_protocol_request


def test_monte_carlo_npt_path_scales_orthorhombic_volume_with_constraints():
    positions = np.array([[1.0, 1.0, 1.0], [2.25, 1.0, 1.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)

    result = simulate_npt(
        positions,
        velocities,
        masses=np.asarray([1.0, 1.0], dtype=np.float32),
        cell=Cell.cubic(8.0),
        force_terms=LennardJonesPotential(cutoff=3.0),
        config=SimulationConfig(dt=0.001, steps=4, sample_interval=2, diagnostic_interval=2),
        thermostat=LangevinThermostat(temperature=1.0, friction=1.0, seed=3),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            seed=3,
            max_log_volume_scale=0.01,
        ),
    )

    assert result.final_state.step == 4
    assert result.barostat_attempts == 1
    assert result.volume.shape == (2,)
    assert np.isfinite(np.asarray(result.volume)).all()
    assert np.all(np.asarray(result.final_cell.lengths) > 0.0)


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
