from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.artifacts import MLXCompatibilityError, validate_mlx_compatibility
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.md import SimulationConfig, SimulationState
from mlx_atomistic.replica_exchange import (
    simulate_replica_exchange,
    temperature_exchange_probability,
)


@dataclass(frozen=True)
class HarmonicWell:
    k: float = 1.0
    name: str = "harmonic"
    supports_virial: bool = True

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        positions = mx.array(positions)
        return 0.5 * self.k * mx.sum(positions * positions), -self.k * positions


def _state(x):
    positions = mx.array([[x, 0.0, 0.0]], dtype=mx.float32)
    return SimulationState(
        positions=positions,
        velocities=mx.zeros_like(positions),
        masses=mx.array([1.0], dtype=mx.float32),
        forces=mx.zeros_like(positions),
    )


def test_temperature_exchange_probability_matches_metropolis_formula():
    probability, log_acceptance = temperature_exchange_probability(
        energy_left=1.0,
        energy_right=3.0,
        beta_left=0.8,
        beta_right=0.2,
    )

    expected_log = (0.8 - 0.2) * (1.0 - 3.0)

    assert log_acceptance == pytest.approx(expected_log)
    assert probability == pytest.approx(np.exp(expected_log))


def test_exchange_histogram_proxy_satisfies_boltzmann_ratio():
    forward, _ = temperature_exchange_probability(
        energy_left=0.5,
        energy_right=2.0,
        beta_left=0.9,
        beta_right=0.3,
    )
    reverse, _ = temperature_exchange_probability(
        energy_left=2.0,
        energy_right=0.5,
        beta_left=0.9,
        beta_right=0.3,
    )

    expected_ratio = np.exp((0.9 - 0.3) * (0.5 - 2.0))

    assert forward / reverse == pytest.approx(expected_ratio)


def test_simulate_replica_exchange_swaps_adjacent_temperature_replicas():
    result = simulate_replica_exchange(
        [_state(2.0), _state(0.0)],
        HarmonicWell(),
        temperatures=[1.0, 2.0],
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        swap_interval=1,
        thermostat_friction=0.0,
        seed=7,
    )

    assert len(result.replica_states) == 2
    assert len(result.swap_attempts) == 1
    assert result.accepted_swaps == 1
    np.testing.assert_array_equal(np.asarray(result.state_index_history[-1]), np.array([1, 0]))
    assert np.asarray(result.energy_history).shape == (2, 2)


def test_simulate_replica_exchange_accepts_lambda_scaled_hamiltonians():
    term = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.2, 0.2],
        charges=[0.5, -0.5],
        cutoff=None,
        lj_shift=False,
    )
    state_a = SimulationState(
        positions=mx.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=mx.float32),
        velocities=mx.zeros((2, 3), dtype=mx.float32),
        masses=mx.ones((2,), dtype=mx.float32),
        forces=mx.zeros((2, 3), dtype=mx.float32),
    )
    state_b = SimulationState(
        positions=mx.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=mx.float32),
        velocities=mx.zeros((2, 3), dtype=mx.float32),
        masses=mx.ones((2,), dtype=mx.float32),
        forces=mx.zeros((2, 3), dtype=mx.float32),
    )

    result = simulate_replica_exchange(
        [state_a, state_b],
        term,
        temperatures=[1.0, 1.0],
        lambdas=[1.0, 0.5],
        config=SimulationConfig(dt=0.001, steps=1, sample_interval=1, diagnostic_interval=1),
        swap_interval=1,
        thermostat_friction=0.0,
        seed=3,
    )

    assert result.lambdas == (1.0, 0.5)
    assert len(result.swap_attempts) == 1
    assert np.isfinite(np.asarray(result.energy_history)).all()


def test_simulate_replica_exchange_accepts_shared_multi_term_hamiltonian_matching_replica_count():
    result = simulate_replica_exchange(
        [_state(1.0), _state(0.0)],
        [HarmonicWell(k=1.0), HarmonicWell(k=0.1)],
        temperatures=[1.0, 1.0],
        config=SimulationConfig(dt=0.001, steps=1),
        swap_interval=1,
        thermostat_friction=0.0,
        seed=5,
    )

    assert len(result.swap_attempts) == 1
    assert np.isfinite(np.asarray(result.energy_history)).all()


def test_simulate_replica_exchange_accepts_explicit_single_term_hamiltonians():
    result = simulate_replica_exchange(
        [_state(1.0), _state(0.0)],
        [(HarmonicWell(k=1.0),), (HarmonicWell(k=0.5),)],
        temperatures=[1.0, 1.0],
        config=SimulationConfig(dt=0.001, steps=1),
        swap_interval=1,
        thermostat_friction=0.0,
        seed=5,
    )

    assert len(result.swap_attempts) == 1
    assert np.isfinite(np.asarray(result.energy_history)).all()


def test_simulate_replica_exchange_fails_closed_for_unsupported_runtime_inputs():
    with pytest.raises(ValueError, match="neighbor_manager"):
        simulate_replica_exchange(
            [_state(1.0), _state(0.0)],
            HarmonicWell(),
            temperatures=[1.0, 1.0],
            config=SimulationConfig(dt=0.001, steps=1),
            neighbor_manager=object(),
        )


def test_simulate_replica_exchange_uses_odd_even_swap_pairing():
    result = simulate_replica_exchange(
        [_state(3.0), _state(2.0), _state(0.0)],
        HarmonicWell(),
        temperatures=[1.0, 2.0, 3.0],
        config=SimulationConfig(dt=0.001, steps=2),
        swap_interval=1,
        thermostat_friction=0.0,
        seed=7,
    )

    assert [(attempt.step, attempt.left, attempt.right) for attempt in result.swap_attempts] == [
        (1, 0, 1),
        (2, 1, 2),
    ]


def test_artifact_accepts_replica_exchange_metadata():
    metadata = {
        "compatibility_report": {
            "production_force_field": False,
            "supported_terms": ["replica_exchange"],
            "required_terms": ["replica_exchange"],
            "unsupported_terms": [],
        },
        "replica_exchange": {
            "swap_interval": 10,
            "temperatures": [300.0, 310.0],
            "lambdas": [1.0, 0.5],
        },
    }

    assert validate_mlx_compatibility(metadata, require_production=False) is None


def test_artifact_rejects_invalid_replica_exchange_metadata():
    metadata = {
        "compatibility_report": {
            "production_force_field": False,
            "supported_terms": ["replica_exchange"],
            "required_terms": ["replica_exchange"],
            "unsupported_terms": [],
        },
        "replica_exchange": "enabled",
    }

    with pytest.raises(MLXCompatibilityError, match="replica_exchange metadata"):
        validate_mlx_compatibility(metadata, require_production=False)


@pytest.mark.parametrize(
    "replica_exchange",
    [
        {"swap_interval": 0, "temperatures": [300.0, 310.0]},
        {"swap_interval": 1.5, "temperatures": [300.0, 310.0]},
        {"swap_interval": 10, "temperatures": [300.0, -1.0]},
        {"swap_interval": 10, "temperatures": [300.0]},
        {"swap_interval": 10, "temperatures": [300.0, 310.0], "lambdas": [1.2, 0.5]},
        {"swap_interval": 10, "temperatures": [300.0, 310.0], "lambdas": [1.0]},
    ],
)
def test_artifact_rejects_invalid_replica_exchange_ranges(replica_exchange):
    metadata = {
        "compatibility_report": {
            "production_force_field": False,
            "supported_terms": ["replica_exchange"],
            "required_terms": ["replica_exchange"],
            "unsupported_terms": [],
        },
        "replica_exchange": replica_exchange,
    }

    with pytest.raises(MLXCompatibilityError, match="replica_exchange"):
        validate_mlx_compatibility(metadata, require_production=False)
