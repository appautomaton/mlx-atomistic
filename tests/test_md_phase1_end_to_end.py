import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.constraints import SettleWaterConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    MonteCarloBarostat,
    NoseHooverThermostat,
    SimulationConfig,
    simulate_npt,
    simulate_nvt,
)
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.protocols import validate_gpcrmd_protocol_request

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class _AnchoredWaterFixture:
    name = "anchored_water"
    supports_virial = True

    def __init__(self, target):
        self.target = np.asarray(target, dtype=np.float32)

    def energy_forces(self, positions, cell=None, pairs=None):
        target = mx.array(self.target, dtype=positions.dtype)
        displacement = positions - target
        return 0.5 * mx.sum(displacement * displacement), -displacement


def _water_rich_fixture():
    target = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [0.925, 1.0992157, 1.0],
            [2.0, 2.0, 2.0],
            [2.1, 2.0, 2.0],
            [1.925, 2.0992157, 2.0],
        ],
        dtype=np.float32,
    )
    initial = target + np.asarray(
        [
            [0.03, -0.02, 0.01],
            [-0.02, 0.01, -0.01],
            [0.01, 0.02, 0.0],
            [-0.02, 0.03, -0.01],
            [0.02, -0.01, 0.01],
            [-0.01, -0.02, 0.0],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, -0.01],
        ],
        dtype=np.float32,
    )
    masses = np.asarray([16.0, 1.0, 1.0, 16.0, 1.0, 1.0], dtype=np.float32)
    constraints = SettleWaterConstraints(
        [(0, 1, 2), (3, 4, 5)],
        oh_distance=0.1,
        hh_distance=0.15,
    )
    return initial, target, velocities, masses, constraints


def test_phase1_minimize_nose_hoover_nvt_anisotropic_npt_records_finite_state():
    positions, target, velocities, masses, constraints = _water_rich_fixture()
    cell = Cell.triclinic(
        [
            [4.0, 0.0, 0.0],
            [0.2, 4.2, 0.0],
            [0.1, 0.3, 4.4],
        ]
    )
    force_term = _AnchoredWaterFixture(target)

    minimized = minimize_energy(
        positions,
        force_term,
        cell=cell,
        method="l-bfgs",
        max_steps=50,
        force_tolerance=1.0e-5,
    )
    constrained_positions, constraint_error = constraints.apply_positions(
        minimized.positions,
        masses,
        cell,
    )
    nvt = simulate_nvt(
        constrained_positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=force_term,
        constraints=constraints,
        config=SimulationConfig(dt=0.001, steps=6, sample_interval=3, diagnostic_interval=3),
        thermostat=NoseHooverThermostat(temperature=1.0, relaxation_time=0.2),
    )
    npt = simulate_npt(
        nvt.final_state.positions,
        nvt.final_state.velocities,
        masses=masses,
        cell=cell,
        force_terms=force_term,
        constraints=constraints,
        config=SimulationConfig(dt=0.001, steps=4, sample_interval=2, diagnostic_interval=2),
        thermostat=NoseHooverThermostat(temperature=1.0, relaxation_time=0.2),
        barostat=MonteCarloBarostat(
            pressure=0.0,
            temperature=1.0,
            seed=4,
            max_log_volume_scale=0.01,
            mode="anisotropic",
            axes=(True, False, True),
        ),
    )
    protocol_report = validate_gpcrmd_protocol_request(
        {"ensemble": "NPT", "barostat": "monte_carlo"},
        raise_on_blockers=True,
    )

    finite_arrays = [
        np.asarray(minimized.energy_history),
        np.asarray(minimized.positions),
        np.asarray(nvt.total_energy),
        np.asarray(nvt.sampled_positions),
        np.asarray(npt.total_energy),
        np.asarray(npt.sampled_positions),
        np.asarray(npt.cell_matrix),
        np.asarray(npt.volume),
    ]
    for values in finite_arrays:
        assert np.isfinite(values).all()
    assert minimized.convergence_reason in {"force_tolerance", "optimizer_success"}
    assert float(np.asarray(constraint_error)) < 1.0e-5
    assert float(np.max(np.asarray(nvt.constraint_max_error))) < 1.0e-5
    assert float(np.max(np.asarray(npt.constraint_max_error))) < 1.0e-5
    assert nvt.thermostat_metadata["family"] == "nose_hoover"
    assert npt.barostat_metadata["mode"] == "anisotropic"
    assert npt.barostat_attempts == 1
    assert npt.barostat_accepted in {0, 1}
    assert np.all(np.asarray(npt.final_cell.lengths) > 0.0)
    assert protocol_report.accepted
    assert protocol_report.blockers == ()
