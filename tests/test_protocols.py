import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.protocols import (
    MinimizeThenNVTProtocol,
    ProtocolCompatibilityError,
    run_minimize_then_nvt,
    validate_gpcrmd_protocol_request,
)


class _ExplodingPotential:
    name = "exploding"

    def energy_forces(self, positions, cell=None, pairs=None):
        raise AssertionError("protocol gate allowed integration to start")


def _small_periodic_fixture():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.2, 1.0, 1.0], [1.0, 2.2, 1.0], [2.2, 2.2, 1.0]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.02, 0.0, 0.0], [-0.01, 0.01, 0.0], [0.0, -0.02, 0.0], [0.0, 0.01, 0.01]],
        dtype=np.float32,
    )
    masses = np.ones((4,), dtype=np.float32)
    return positions, velocities, masses, Cell.cubic(6.0), LennardJonesPotential(cutoff=2.5)


def test_gpcrmd_protocol_gate_accepts_short_nvt_metadata():
    report = validate_gpcrmd_protocol_request({"ensemble": "NVT"})

    assert report.accepted is True
    assert report.blockers == ()
    assert report.metadata["ensemble"] == "NVT"
    assert report.metadata["proof_mode"] == "short_nvt"
    assert report.metadata["barostat"] == "none"
    assert report.metadata["barostat_status"] == "not_required_for_nvt_proof"
    assert report.metadata["npt_barostat"] is False
    assert report.metadata["membrane_barostat"] is False


@pytest.mark.parametrize(
    ("protocol_request", "blockers"),
    [
        ({"ensemble": "NPT"}, ("npt_barostat",)),
        ({"ensemble": "NVT", "barostat": "monte_carlo"}, ("barostat",)),
        ({"ensemble": "NVT", "membrane_barostat": "semiisotropic"}, ("membrane_barostat",)),
    ],
)
def test_gpcrmd_protocol_gate_rejects_npt_and_barostat_requests(protocol_request, blockers):
    report = validate_gpcrmd_protocol_request(protocol_request)

    assert report.accepted is False
    assert report.blockers == blockers
    assert report.metadata["unsupported_protocol_blockers"] == list(blockers)

    with pytest.raises(ProtocolCompatibilityError, match=", ".join(blockers)) as exc_info:
        validate_gpcrmd_protocol_request(protocol_request, raise_on_blockers=True)
    assert exc_info.value.blockers == blockers


def test_run_minimize_then_nvt_fails_closed_before_npt_force_evaluation():
    positions, velocities, masses, cell, _ = _small_periodic_fixture()

    with pytest.raises(ProtocolCompatibilityError) as exc_info:
        run_minimize_then_nvt(
            positions,
            velocities,
            masses,
            _ExplodingPotential(),
            protocol=MinimizeThenNVTProtocol(
                ensemble="NPT",
                minimize_steps=0,
                equilibration_steps=0,
                production_steps=1,
            ),
            cell=cell,
        )

    assert exc_info.value.blockers == ("npt_barostat",)


def test_run_minimize_then_nvt_exposes_nvt_protocol_metadata():
    positions, velocities, masses, cell, potential = _small_periodic_fixture()

    result = run_minimize_then_nvt(
        positions,
        velocities,
        masses,
        potential,
        protocol=MinimizeThenNVTProtocol(
            minimize_steps=0,
            equilibration_steps=0,
            production_steps=2,
            dt=0.001,
            sample_interval=1,
            temperature=1.0,
            friction=0.1,
            seed=13,
            compile_force_evaluator=False,
        ),
        cell=cell,
    )

    assert np.isfinite(np.asarray(result.production.total_energy)).all()
    assert result.protocol_metadata["ensemble"] == "NVT"
    assert result.protocol_metadata["proof_mode"] == "short_nvt"
    assert result.protocol_metadata["barostat"] == "none"
    assert result.protocol_metadata["barostat_status"] == "not_required_for_nvt_proof"
