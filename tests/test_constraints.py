import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.constraints import (
    CompositeConstraints,
    DistanceConstraints,
    SettleWaterConstraints,
    _ShakeClusterConstraints,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nve,
    simulate_nvt,
)


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


def test_periodic_distance_constraints_preserve_continuous_molecules():
    constraints = DistanceConstraints(
        [(0, 1)],
        distances=[1.0],
        max_iterations=4,
    )
    positions = np.asarray(
        [[7.9, 1.0, 1.0], [8.9, 1.0, 1.0]],
        dtype=np.float32,
    )

    projected, error = constraints.apply_positions(
        positions,
        masses=np.ones((2,), dtype=np.float32),
        cell=Cell.cubic(8.0),
    )

    np.testing.assert_allclose(
        np.asarray(projected)[1] - np.asarray(projected)[0],
        [1.0, 0.0, 0.0],
        atol=1.0e-6,
    )
    assert float(np.asarray(error)) <= 1.0e-6


def test_distance_dynamics_step_matches_openmm_reference_oracle():
    """One coupled SHAKE drift matches OpenMM's deterministic positions."""

    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.12, -0.08, 0.04],
            [-0.31, 0.27, 0.09],
            [0.22, -0.14, -0.05],
        ],
        dtype=np.float32,
    )
    constraints = DistanceConstraints(
        [(0, 1), (0, 2)],
        distances=[0.1, 0.1],
        max_iterations=20,
    )

    result = simulate_nvt(
        positions,
        velocities,
        masses=np.asarray([12.0, 1.0, 1.0], dtype=np.float32),
        force_terms=ZeroForce(),
        constraints=constraints,
        config=SimulationConfig(
            dt=0.004,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=7,
        ),
    )

    expected_positions = np.asarray(
        [
            [0.00034848142, -0.00033830303, 0.00016],
            [0.10033822298, 0.00108, 0.00036],
            [0.00088, 0.09965963639, -0.0002],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        expected_positions,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_disjoint_shake_clusters_match_generic_cpu_projection():
    cluster_atoms = np.asarray(
        [[0, 1, 2, 3], [4, 5, -1, -1]],
        dtype=np.int32,
    )
    constraints = _ShakeClusterConstraints(
        cluster_atoms,
        peripheral_counts=[3, 1],
        distances=[1.0, 1.2],
        max_iterations=8,
    )
    reference = np.asarray(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [6.2, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    predicted = reference + np.asarray(
        [
            [0.02, -0.01, 0.01],
            [-0.03, 0.02, 0.0],
            [0.01, -0.02, 0.02],
            [0.0, 0.01, -0.02],
            [-0.01, 0.0, 0.02],
            [0.03, -0.01, 0.0],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.2, -0.1, 0.3],
            [-0.3, 0.2, 0.1],
            [0.1, -0.2, 0.0],
            [0.0, 0.1, -0.2],
            [0.4, -0.2, 0.1],
            [-0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    masses = np.asarray([12.0, 1.0, 1.0, 1.0, 14.0, 1.0], dtype=np.float32)
    cell = Cell.cubic(12.0)

    projected, error = constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    expected_positions, expected_error = constraints._pair_constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    projected_velocities = constraints.apply_velocities(
        projected,
        velocities,
        masses,
        cell,
    )
    expected_velocities = constraints._pair_constraints.apply_velocities(
        expected_positions,
        velocities,
        masses,
        cell,
    )

    np.testing.assert_allclose(np.asarray(projected), np.asarray(expected_positions))
    np.testing.assert_allclose(
        np.asarray(projected_velocities),
        np.asarray(expected_velocities),
    )
    np.testing.assert_allclose(np.asarray(error), np.asarray(expected_error))


def test_settle_water_constraints_project_positions_exactly():
    constraints = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.1, 0.9, 0.0]],
        dtype=np.float32,
    )
    masses = np.asarray([16.0, 1.0, 1.0])
    center_before = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)

    projected, error = constraints.apply_positions(positions, masses=masses)

    np.testing.assert_allclose(_distances(np.asarray(projected)), [1.0, 1.0, 1.5], atol=1e-6)
    center_after = np.sum(np.asarray(projected) * masses[:, None], axis=0) / np.sum(masses)
    np.testing.assert_allclose(center_after, center_before, atol=1e-6)
    assert float(np.asarray(error)) <= 1e-6


def test_periodic_settle_preserves_continuous_water_coordinates():
    constraints = SettleWaterConstraints(
        [(0, 1, 2)],
        oh_distance=1.0,
        hh_distance=1.5,
    )
    positions = np.asarray(
        [[7.9, 2.0, 2.0], [8.9, 2.0, 2.0], [7.6, 2.9, 2.0]],
        dtype=np.float32,
    )

    projected, error = constraints.apply_positions(
        positions,
        masses=np.asarray([16.0, 1.0, 1.0]),
        cell=Cell.cubic(8.0),
    )

    assert np.asarray(projected)[1, 0] > 8.0
    assert float(np.asarray(error)) <= 1.0e-6


def test_settle_water_constraints_remove_pair_relative_velocity():
    constraints = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    masses = np.asarray([16.0, 1.0, 1.0])
    projected, _ = constraints.apply_positions(
        np.asarray([[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.1, 0.9, 0.0]], dtype=np.float32),
        masses=masses,
    )
    velocities = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [-0.4, 0.3, 0.0]])
    momentum_before = np.sum(velocities * masses[:, None], axis=0)

    constrained = constraints.apply_velocities(
        projected,
        velocities,
        masses=masses,
    )

    for left, right in np.asarray(constraints.pairs):
        displacement = np.asarray(projected)[left] - np.asarray(projected)[right]
        unit = displacement / np.linalg.norm(displacement)
        relative = np.asarray(constrained)[left] - np.asarray(constrained)[right]
        assert abs(float(np.dot(relative, unit))) < 1e-6
    momentum_after = np.sum(np.asarray(constrained) * masses[:, None], axis=0)
    np.testing.assert_allclose(momentum_after, momentum_before, atol=1e-6)


def test_settle_dynamics_step_matches_openmm_reference_oracle():
    """One constrained drift matches OpenMM's deterministic SETTLE step."""

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.0125, 0.099215674, 0.0],
        ],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [
            [0.12, -0.08, 0.04],
            [-0.31, 0.27, 0.09],
            [0.22, -0.14, -0.05],
        ],
        dtype=np.float32,
    )
    constraints = SettleWaterConstraints(
        [(0, 1, 2)],
        oh_distance=0.1,
        hh_distance=0.15,
    )

    result = simulate_nvt(
        positions,
        velocities,
        masses=np.asarray([16.0, 1.0, 1.0], dtype=np.float32),
        force_terms=ZeroForce(),
        constraints=constraints,
        config=SimulationConfig(
            dt=0.004,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(
            temperature=0.0,
            friction=0.0,
            seed=5,
        ),
    )

    expected_positions = np.asarray(
        [
            [0.00043289735, -0.00027857105, 0.00016],
            [0.10043156888, 0.00019681351, 0.00036],
            [-0.01253792644, 0.09887599732, -0.0002],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        expected_positions,
        rtol=0.0,
        atol=2.0e-7,
    )
    final_positions = np.asarray(result.final_state.positions)
    final_velocities = np.asarray(result.final_state.velocities)
    for left, right in np.asarray(constraints.pairs):
        displacement = final_positions[left] - final_positions[right]
        relative_velocity = final_velocities[left] - final_velocities[right]
        assert abs(float(np.dot(displacement, relative_velocity))) < 2.0e-6


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


def test_overlapping_composite_constraints_project_all_relative_velocities():
    settle = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    tether = DistanceConstraints([(0, 3)], distances=[2.0], max_iterations=4)
    constraints = CompositeConstraints((tether, settle))
    masses = np.asarray([16.0, 1.0, 1.0, 12.0], dtype=np.float32)
    positions, _ = constraints.apply_positions(
        np.asarray(
            [
                [2.0, 2.0, 2.0],
                [3.1, 2.1, 2.0],
                [1.9, 2.9, 2.0],
                [4.1, 2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        masses,
        Cell.cubic(8.0),
    )
    velocities = np.asarray(
        [
            [0.3, -0.1, 0.2],
            [-0.2, 0.4, 0.0],
            [0.1, -0.3, 0.2],
            [-0.4, 0.2, -0.1],
        ],
        dtype=np.float32,
    )

    projected = constraints.apply_velocities(
        positions,
        velocities,
        masses,
        Cell.cubic(8.0),
    )

    for left, right in np.asarray(constraints.pairs):
        displacement = np.asarray(positions)[left] - np.asarray(positions)[right]
        relative = np.asarray(projected)[left] - np.asarray(projected)[right]
        assert abs(float(np.dot(relative, displacement))) < 1e-5


def test_disjoint_composite_skips_only_redundant_pre_force_settle_projection():
    settle = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    tether = DistanceConstraints([(3, 4)], distances=[1.2], max_iterations=4)
    constraints = CompositeConstraints((settle, tether))
    positions = mx.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.125, 0.99215674, 0.0],
            [3.0, 0.0, 0.0],
            [4.2, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    velocities = mx.array(
        [
            [0.3, -0.2, 0.1],
            [-0.1, 0.4, 0.2],
            [0.2, -0.3, 0.1],
            [0.5, 0.1, -0.2],
            [-0.4, 0.2, 0.3],
        ],
        dtype=mx.float32,
    )
    kick = mx.array(
        [
            [0.01, 0.03, -0.02],
            [-0.02, 0.01, 0.04],
            [0.03, -0.02, 0.01],
            [-0.01, 0.02, 0.03],
            [0.02, -0.03, -0.01],
        ],
        dtype=mx.float32,
    )
    masses = mx.array([16.0, 1.0, 1.0, 12.0, 1.0], dtype=mx.float32)

    full_pre = constraints.apply_velocities(positions, velocities, masses)
    full_final = constraints.apply_velocities(positions, full_pre + kick, masses)
    reduced_pre = constraints._apply_pre_force_velocities(
        positions,
        velocities,
        masses,
    )
    reduced_final = constraints.apply_velocities(
        positions,
        reduced_pre + kick,
        masses,
    )

    np.testing.assert_allclose(
        np.asarray(reduced_final),
        np.asarray(full_final),
        rtol=1e-6,
        atol=2e-7,
    )


def test_overlapping_composite_retains_full_pre_force_projection():
    settle = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    tether = DistanceConstraints([(0, 3)], distances=[2.0], max_iterations=4)
    constraints = CompositeConstraints((settle, tether))
    positions = mx.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.125, 0.99215674, 0.0], [2.0, 0.0, 0.0]],
        dtype=mx.float32,
    )
    velocities = mx.array(
        [[0.3, -0.1, 0.2], [-0.2, 0.4, 0.0], [0.1, -0.3, 0.2], [-0.4, 0.2, -0.1]],
        dtype=mx.float32,
    )
    masses = mx.array([16.0, 1.0, 1.0, 12.0], dtype=mx.float32)

    expected = constraints.apply_velocities(positions, velocities, masses)
    actual = constraints._apply_pre_force_velocities(positions, velocities, masses)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0)


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
