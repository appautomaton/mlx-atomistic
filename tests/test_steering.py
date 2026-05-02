from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_atomistic.md import LangevinThermostat, SimulationConfig
from mlx_atomistic.steering import SteeredCOMBiasPotential, simulate_steered_nvt


class _ZeroForce:
    name = "zero"

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        positions = mx.array(positions)
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)


def test_steered_com_bias_matches_finite_difference_force():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    masses = np.asarray([12.0, 1.0, 1.0], dtype=np.float32)
    bias = SteeredCOMBiasPotential(
        ligand_indices=np.asarray([0, 1], dtype=np.int32),
        direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        target=0.25,
        k=7.0,
        masses=masses,
    )

    energy, forces = bias.energy_forces(mx.array(positions))
    epsilon = 1e-3
    shifted_plus = positions.copy()
    shifted_minus = positions.copy()
    shifted_plus[0, 0] += epsilon
    shifted_minus[0, 0] -= epsilon
    energy_plus = float(np.asarray(bias.potential_energy(mx.array(shifted_plus))))
    energy_minus = float(np.asarray(bias.potential_energy(mx.array(shifted_minus))))
    finite_difference_force = -(energy_plus - energy_minus) / (2.0 * epsilon)

    assert float(np.asarray(energy)) > 0.0
    np.testing.assert_allclose(
        float(np.asarray(forces)[0, 0]),
        finite_difference_force,
        rtol=2e-3,
        atol=2e-3,
    )


def test_tiny_steered_nvt_moves_ligand_com_along_target():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    masses = np.asarray([12.0, 12.0, 12.0], dtype=np.float32)

    result = simulate_steered_nvt(
        positions,
        velocities,
        masses=masses,
        ligand_indices=np.asarray([0, 1], dtype=np.int32),
        direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        target_start=0.5,
        target_velocity=2.0,
        k=50.0,
        force_terms=[_ZeroForce()],
        config=SimulationConfig(
            dt=0.001,
            steps=20,
            sample_interval=5,
            diagnostic_interval=5,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=1.0,
            seed=3,
        ),
    )

    sampled_cv = np.asarray(result.sampled_cv)
    sampled_target = np.asarray(result.sampled_target)

    assert np.all(np.diff(sampled_target) > 0.0)
    assert sampled_cv[-1] > sampled_cv[0]
    assert "steered_com_bias" in result.potential_energy_by_term
