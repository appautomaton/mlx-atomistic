import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.constraints import (
    CompositeConstraints,
    DistanceConstraints,
    SettleWaterConstraints,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate_nve


def _distances(positions):
    oh_a = np.linalg.norm(positions[1] - positions[0])
    oh_b = np.linalg.norm(positions[2] - positions[0])
    hh = np.linalg.norm(positions[1] - positions[2])
    return np.asarray([oh_a, oh_b, hh], dtype=np.float32)


class ZeroForce:
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)


def test_distance_constraints_remain_stable_in_short_long_run():
    positions = np.array([[1.0, 1.0, 1.0], [2.25, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]], dtype=np.float32)
    constraints = DistanceConstraints(
        np.asarray([[0, 1]], dtype=np.int32),
        distances=np.asarray([1.25], dtype=np.float32),
        max_iterations=8,
    )

    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(8.0),
        force_terms=LennardJonesPotential(cutoff=3.0),
        constraints=constraints,
        config=SimulationConfig(
            dt=0.001,
            steps=100,
            sample_interval=25,
            diagnostic_interval=10,
        ),
    )

    assert float(np.max(np.asarray(result.constraint_max_error))) < 1e-4
    assert result.final_state.step == 100


def test_settle_water_constraints_project_positions_exactly():
    constraints = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.1, 0.9, 0.0]],
        dtype=np.float32,
    )

    projected, error = constraints.apply_positions(positions, masses=np.asarray([16.0, 1.0, 1.0]))

    np.testing.assert_allclose(_distances(np.asarray(projected)), [1.0, 1.0, 1.5], atol=1e-6)
    assert float(np.asarray(error)) <= 1e-6


def test_settle_water_constraints_remove_pair_relative_velocity():
    constraints = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    projected, _ = constraints.apply_positions(
        np.asarray([[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.1, 0.9, 0.0]], dtype=np.float32),
        masses=np.asarray([16.0, 1.0, 1.0]),
    )
    velocities = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [-0.4, 0.3, 0.0]])

    constrained = constraints.apply_velocities(
        projected,
        velocities,
        masses=np.asarray([16.0, 1.0, 1.0]),
    )

    for left, right in np.asarray(constraints.pairs):
        displacement = np.asarray(projected)[left] - np.asarray(projected)[right]
        unit = displacement / np.linalg.norm(displacement)
        relative = np.asarray(constrained)[left] - np.asarray(constrained)[right]
        assert abs(float(np.dot(relative, unit))) < 1e-6


def test_settle_interoperates_with_generic_distance_constraints_in_nve():
    settle = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    tether = DistanceConstraints([(0, 3)], distances=[2.0], max_iterations=4)
    constraints = CompositeConstraints((tether, settle))
    positions = np.asarray(
        [[2.0, 2.0, 2.0], [3.1, 2.1, 2.0], [1.9, 2.9, 2.0], [4.1, 2.0, 2.0]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)

    result = simulate_nve(
        positions,
        velocities,
        masses=np.asarray([16.0, 1.0, 1.0, 12.0], dtype=np.float32),
        cell=Cell.cubic(8.0),
        force_terms=ZeroForce(),
        constraints=constraints,
        config=SimulationConfig(dt=0.001, steps=2, sample_interval=1),
    )

    assert constraints.pairs.shape[0] == 4
    assert float(np.max(np.asarray(result.constraint_max_error))) < 1e-5


def test_settle_rejects_malformed_water_topology():
    with pytest.raises(ValueError, match="shape"):
        SettleWaterConstraints([(0, 1)])
    with pytest.raises(ValueError, match="distinct"):
        SettleWaterConstraints([(0, 1, 1)])
    constraints = SettleWaterConstraints([(0, 1, 4)])
    with pytest.raises(ValueError, match="outside positions"):
        constraints.apply_positions(
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
        )
