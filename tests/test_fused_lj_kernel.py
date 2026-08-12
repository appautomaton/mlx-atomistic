"""Physics-lock tests for the fused Metal LJ force kernel (perf lever #4).

The fused kernel runs only on a Metal GPU; ``conftest.py`` forces the CPU device,
so each test switches to the GPU and skips when Metal is unavailable (headless CI).
Equivalence is locked with loose tolerances, not bit-identical results: the kernel's
atomic scatter is summation-order non-deterministic, the same property as the existing
``.at[].add()`` op-chain (see tests/test_neighbors.py).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.constraints as constraints_module
import mlx_atomistic.md as md_module
from mlx_atomistic.constraints import (
    CompositeConstraints,
    DistanceConstraints,
    SettleWaterConstraints,
    _project_constraint_positions_unchecked,
    _ShakeClusterConstraints,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.force_runtime import _PreparedForcePipeline
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nvt,
)
from mlx_atomistic.metal_kernels import (
    _prepared_parameterized_pme_direct_force_only,
    _tile_parameterized_pme_direct_force_only,
    fused_lj_forces,
    fused_parameterized_pme_direct_components,
    fused_parameterized_pme_direct_force_only,
    fused_sparse_pme_correction_forces,
    neighbor_pair_cutoff_mask,
    neighbor_pair_ordered_scatter,
)
from mlx_atomistic.neighbors import (
    _MLX_MD_CACHE_LIMIT_BYTES,
    DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE,
    NeighborListManager,
    NeighborTiles,
    _bounded_metal_md_cache,
    build_neighbor_list,
)
from mlx_atomistic.pme import PMEConfig
from mlx_atomistic.topology import Topology


@pytest.fixture(autouse=True)
def _on_gpu(monkeypatch):
    """Run each test on the Metal GPU; skip if it cannot be reached.

    conftest sets MLX_ATOMISTIC_DEVICE=cpu, which makes as_mx_array() reset the
    default device to CPU whenever it converts a non-mx input -- that would yank the
    kernel off the GPU mid-test. Override the env so conversions stay on the GPU.
    """

    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    prev_device = mx.default_device()
    try:
        gpu = mx.Device(mx.gpu, 0)
        mx.set_default_device(gpu)
        mx.set_default_stream(mx.new_stream(gpu))
        mx.eval(mx.array([1.0], dtype=mx.float32) + 1.0)
    except Exception:  # noqa: BLE001 - any Metal load failure means skip
        mx.set_default_device(prev_device)
        mx.set_default_stream(mx.new_stream(prev_device))
        pytest.skip("Metal GPU unavailable")
    yield
    mx.set_default_device(prev_device)
    mx.set_default_stream(mx.new_stream(prev_device))


@pytest.mark.gpu
def test_disjoint_shake_cluster_kernel_matches_mlx_reference():
    """Cluster-owned SHAKE/RATTLE matches the pairwise MLX oracle on Metal."""

    constraints = _ShakeClusterConstraints(
        [[0, 1, 2, 3], [4, 5, -1, -1]],
        peripheral_counts=[3, 1],
        distances=[1.0, 1.2],
        max_iterations=8,
    )
    reference = mx.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
            [6.2, 5.0, 5.0],
        ],
        dtype=mx.float32,
    )
    predicted = reference + mx.array(
        [
            [0.02, -0.01, 0.01],
            [-0.03, 0.02, 0.0],
            [0.01, -0.02, 0.02],
            [0.0, 0.01, -0.02],
            [-0.01, 0.0, 0.02],
            [0.03, -0.01, 0.0],
        ],
        dtype=mx.float32,
    )
    velocities = mx.array(
        [
            [0.2, -0.1, 0.3],
            [-0.3, 0.2, 0.1],
            [0.1, -0.2, 0.0],
            [0.0, 0.1, -0.2],
            [0.4, -0.2, 0.1],
            [-0.1, 0.3, -0.2],
        ],
        dtype=mx.float32,
    )
    masses = mx.array([12.0, 1.0, 1.0, 1.0, 14.0, 1.0], dtype=mx.float32)
    cell = Cell.cubic(12.0)

    expected_positions, _ = constraints._pair_constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    actual_positions, actual_error = constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    expected_velocities = constraints._pair_constraints.apply_velocities(
        expected_positions,
        velocities,
        masses,
        cell,
    )
    actual_velocities = constraints.apply_velocities(
        actual_positions,
        velocities,
        masses,
        cell,
    )
    mx.eval(
        expected_positions,
        actual_positions,
        expected_velocities,
        actual_velocities,
        actual_error,
    )

    np.testing.assert_allclose(
        np.asarray(actual_positions),
        np.asarray(expected_positions),
        rtol=1.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_velocities),
        np.asarray(expected_velocities),
        rtol=1.0e-5,
        atol=2.0e-6,
    )
    assert float(np.asarray(actual_error)) <= 1.0e-5


@pytest.mark.gpu
def test_dense_disjoint_composite_matches_sequential_constraint_oracle(monkeypatch):
    """Dense SETTLE+SHAKE writes preserve every constrained and free atom."""

    settle = SettleWaterConstraints(
        [(5, 2, 9)],
        oh_distance=1.0,
        hh_distance=1.5,
    )
    shake = _ShakeClusterConstraints(
        [[0, 4, 7, 11], [3, 8, -1, -1]],
        peripheral_counts=[3, 1],
        distances=[1.0, 1.2],
        max_iterations=8,
    )
    constraints = CompositeConstraints((shake, settle))
    reference = mx.array(
        [
            [5.0, 5.0, 5.0],
            [8.0, 8.0, 8.0],
            [3.0, 2.0, 2.0],
            [8.0, 3.0, 3.0],
            [6.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
            [1.0, 9.0, 4.0],
            [5.0, 6.0, 5.0],
            [9.2, 3.0, 3.0],
            [1.875, 2.9921567, 2.0],
            [10.0, 8.0, 1.0],
            [5.0, 5.0, 6.0],
        ],
        dtype=mx.float32,
    )
    perturbation = mx.array(
        np.random.default_rng(73).uniform(-0.01, 0.01, size=reference.shape),
        dtype=mx.float32,
    )
    predicted = reference + perturbation
    velocities = mx.array(
        np.random.default_rng(74).uniform(-0.2, 0.2, size=reference.shape),
        dtype=mx.float32,
    )
    kick = mx.array(
        np.random.default_rng(75).uniform(-0.02, 0.02, size=reference.shape),
        dtype=mx.float32,
    )
    masses = mx.array(
        [12.0, 14.0, 1.0, 12.0, 1.0, 16.0, 10.0, 1.0, 1.0, 1.0, 9.0, 1.0],
        dtype=mx.float32,
    )
    cell = Cell.cubic(12.0)
    dense_calls = []
    original_dense_apply = constraints_module._dense_constraint_apply

    def record_dense_apply(*args, **kwargs):
        dense_calls.append(args)
        return original_dense_apply(*args, **kwargs)

    monkeypatch.setattr(
        constraints_module,
        "_dense_constraint_apply",
        record_dense_apply,
    )

    expected_positions = predicted
    for child in constraints.constraints:
        expected_positions, _ = child.apply_position_step(
            reference,
            expected_positions,
            masses,
            cell,
        )
    actual_positions, actual_error = constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    expected_pre_force = shake.apply_velocities(
        actual_positions,
        velocities,
        masses,
        cell,
    )
    actual_pre_force = constraints._apply_pre_force_velocities(
        actual_positions,
        velocities,
        masses,
        cell,
    )
    expected_final = velocities + kick
    for child in constraints.constraints:
        expected_final = child.apply_velocities(
            actual_positions,
            expected_final,
            masses,
            cell,
        )
    actual_final = constraints.apply_velocities(
        actual_positions,
        velocities + kick,
        masses,
        cell,
    )
    mx.eval(
        expected_positions,
        actual_positions,
        expected_pre_force,
        actual_pre_force,
        expected_final,
        actual_final,
        actual_error,
    )

    assert len(dense_calls) == 3
    for actual, expected in (
        (actual_positions, expected_positions),
        (actual_pre_force, expected_pre_force),
        (actual_final, expected_final),
    ):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-6,
            atol=2.0e-6,
        )
    np.testing.assert_array_equal(
        np.asarray(actual_positions)[[1, 6, 10]],
        np.asarray(predicted)[[1, 6, 10]],
    )
    assert float(np.asarray(actual_error)) <= 2.0e-5


@pytest.mark.gpu
def test_dense_composite_declines_overlap_and_runtime_profiling(monkeypatch):
    """Overlap and synchronized profiling retain the sequential child routes."""

    class ZeroForce:
        supports_virial = True

        def energy_forces(self, values, cell=None, pairs=None):
            del cell, pairs
            return mx.sum(values[:, 0] * 0.0), mx.zeros_like(values)

    settle = SettleWaterConstraints([(0, 1, 2)], oh_distance=1.0, hh_distance=1.5)
    overlapping_shake = _ShakeClusterConstraints(
        [[0, 3, -1, -1]],
        peripheral_counts=[1],
        distances=[2.0],
        max_iterations=8,
    )
    overlapping = CompositeConstraints((settle, overlapping_shake))
    disjoint_shake = _ShakeClusterConstraints(
        [[3, 4, -1, -1]],
        peripheral_counts=[1],
        distances=[1.2],
        max_iterations=8,
    )
    disjoint = CompositeConstraints((settle, disjoint_shake))
    positions = mx.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [1.875, 2.9921567, 2.0],
            [4.0, 2.0, 2.0],
            [5.2, 2.0, 2.0],
        ],
        dtype=mx.float32,
    )
    masses = mx.array([16.0, 1.0, 1.0, 12.0, 1.0], dtype=mx.float32)

    def fail_dense_apply(*args, **kwargs):
        raise AssertionError("fallback path must not invoke dense constraint apply")

    monkeypatch.setattr(
        constraints_module,
        "_dense_constraint_apply",
        fail_dense_apply,
    )
    projected, _ = overlapping.apply_position_step(
        positions,
        positions + 0.001,
        masses,
    )
    mx.eval(projected)
    assert bool(np.all(np.isfinite(np.asarray(projected))))

    profiled = simulate_nvt(
        positions,
        mx.zeros_like(positions),
        masses=masses,
        force_terms=ZeroForce(),
        constraints=disjoint,
        config=SimulationConfig(
            dt=0.0005,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
            runtime_profile=True,
        ),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=9),
    )
    assert profiled.route_profile["reconciled"] is True


@pytest.mark.gpu
def test_settle_water_kernels_match_openmm_position_oracle():
    """Fused SETTLE position and velocity kernels preserve the reference step."""

    class ZeroForce:
        supports_virial = True

        def energy_forces(self, values, cell=None, pairs=None):
            del cell, pairs
            return mx.sum(values[:, 0] * 0.0), mx.zeros_like(values)

    positions = mx.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.0125, 0.099215674, 0.0]],
        dtype=mx.float32,
    )
    velocities = mx.array(
        [[0.12, -0.08, 0.04], [-0.31, 0.27, 0.09], [0.22, -0.14, -0.05]],
        dtype=mx.float32,
    )
    constraints = SettleWaterConstraints(
        [(0, 1, 2)],
        oh_distance=0.1,
        hh_distance=0.15,
    )
    result = simulate_nvt(
        positions,
        velocities,
        masses=mx.array([16.0, 1.0, 1.0], dtype=mx.float32),
        force_terms=ZeroForce(),
        constraints=constraints,
        config=SimulationConfig(
            dt=0.004,
            steps=1,
            sample_interval=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
        ),
        thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=5),
    )
    mx.eval(result.final_state.positions, result.final_state.velocities)
    expected_positions = np.asarray(
        [
            [0.00043289735, -0.00027857105, 0.00016],
            [0.10043156888, 0.00019681351, 0.00036],
            [-0.01253792644, 0.09887599732, -0.0002],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(result.final_state.positions),
        expected_positions,
        rtol=0.0,
        atol=3.0e-7,
    )
    final_positions = np.asarray(result.final_state.positions)
    final_velocities = np.asarray(result.final_state.velocities)
    for left, right in np.asarray(constraints.pairs):
        displacement = final_positions[left] - final_positions[right]
        relative_velocity = final_velocities[left] - final_velocities[right]
        assert abs(float(np.dot(displacement, relative_velocity))) < 2.0e-6


@pytest.mark.gpu
def test_fused_langevin_baoab_drift_matches_eager_trajectory(monkeypatch):
    """Fused BAOAB drift preserves the seeded eager trajectory on Metal."""

    class HarmonicForce:
        def energy_forces(self, values, cell=None, pairs=None):
            del cell, pairs
            return 0.015 * mx.sum(values * values), -0.03 * values

    positions = mx.array(
        [
            [0.2, 0.3, 0.4],
            [1.1, 0.8, 0.5],
            [2.0, 1.5, 0.9],
            [2.8, 2.4, 1.7],
        ],
        dtype=mx.float32,
    )
    velocities = mx.array(
        [
            [0.10, -0.04, 0.02],
            [-0.08, 0.03, 0.01],
            [0.05, 0.02, -0.07],
            [-0.03, -0.06, 0.04],
        ],
        dtype=mx.float32,
    )
    masses = mx.array([12.0, 16.0, 14.0, 10.0], dtype=mx.float32)
    cell = Cell.orthorhombic([3.0, 3.5, 4.0])
    config = SimulationConfig(
        dt=0.002,
        steps=5,
        sample_interval=5,
        diagnostic_interval=5,
        pressure_diagnostics=False,
    )
    thermostat = LangevinThermostat(temperature=250.0, friction=1.5, seed=23)

    fused = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=HarmonicForce(),
        config=config,
        thermostat=thermostat,
    )

    def eager_baoab(
        current_positions,
        current_velocities,
        forces,
        force_scale_over_mass,
        thermal_scale,
        noise,
        box,
        parameters,
        counts,
    ):
        del counts
        half_dt = parameters[0]
        velocity_half = current_velocities + (half_dt * force_scale_over_mass[:, None] * forces)
        next_positions = current_positions + half_dt * velocity_half
        next_positions = next_positions - box * mx.floor(next_positions / box)
        middle_velocities = parameters[1] * velocity_half + thermal_scale[:, None] * noise
        next_positions = next_positions + half_dt * middle_velocities
        return (
            next_positions - box * mx.floor(next_positions / box),
            middle_velocities,
        )

    monkeypatch.setattr(md_module, "_fused_langevin_baoab_drift", eager_baoab)
    eager = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=HarmonicForce(),
        config=config,
        thermostat=thermostat,
    )
    mx.eval(
        fused.final_state.positions,
        fused.final_state.velocities,
        fused.final_state.forces,
        eager.final_state.positions,
        eager.final_state.velocities,
        eager.final_state.forces,
    )

    for fused_values, eager_values in (
        (fused.final_state.positions, eager.final_state.positions),
        (fused.final_state.velocities, eager.final_state.velocities),
        (fused.final_state.forces, eager.final_state.forces),
        (fused.potential_energy, eager.potential_energy),
    ):
        np.testing.assert_allclose(
            np.asarray(fused_values),
            np.asarray(eager_values),
            rtol=2.0e-6,
            atol=2.0e-6,
        )


@pytest.mark.gpu
def test_unchecked_constraint_projection_skips_error_graph_and_matches_checked_positions(
    monkeypatch,
):
    """Ordinary Metal projection skips errors without changing positions."""

    settle = SettleWaterConstraints(
        [(0, 1, 2)],
        oh_distance=0.1,
        hh_distance=0.15,
    )
    shake = _ShakeClusterConstraints(
        [[3, 4, 5, 6]],
        peripheral_counts=[3],
        distances=[0.1],
        max_iterations=8,
    )
    constraints = CompositeConstraints((settle, shake))
    reference = mx.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [0.9875, 1.099215674, 1.0],
            [3.0, 3.0, 3.0],
            [3.1, 3.0, 3.0],
            [3.0, 3.1, 3.0],
            [3.0, 3.0, 3.1],
        ],
        dtype=mx.float32,
    )
    predicted = reference + mx.array(
        [
            [0.001, -0.002, 0.0],
            [-0.002, 0.001, 0.0],
            [0.001, 0.001, 0.0],
            [0.002, -0.001, 0.001],
            [-0.001, 0.002, 0.0],
            [0.0, -0.002, 0.001],
            [0.001, 0.0, -0.002],
        ],
        dtype=mx.float32,
    )
    masses = mx.array([16.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0])
    cell = Cell.cubic(8.0)
    calls = {"settle": 0, "shake": 0}
    original_settle_error = SettleWaterConstraints.max_error
    original_shake_error = _ShakeClusterConstraints.max_error

    def counted_settle_error(self, positions, cell=None):
        calls["settle"] += 1
        return original_settle_error(self, positions, cell)

    def counted_shake_error(self, positions, cell=None):
        calls["shake"] += 1
        return original_shake_error(self, positions, cell)

    monkeypatch.setattr(SettleWaterConstraints, "max_error", counted_settle_error)
    monkeypatch.setattr(_ShakeClusterConstraints, "max_error", counted_shake_error)
    checked, error = constraints.apply_position_step(
        reference,
        predicted,
        masses,
        cell,
    )
    assert calls["settle"] > 0
    assert calls["shake"] > 0

    calls.update(settle=0, shake=0)
    unchecked = _project_constraint_positions_unchecked(
        constraints,
        predicted,
        masses,
        cell,
        reference_positions=reference,
    )
    assert calls == {"settle": 0, "shake": 0}
    mx.eval(checked, unchecked, error)

    np.testing.assert_allclose(
        np.asarray(unchecked),
        np.asarray(checked),
        rtol=0.0,
        atol=2.0e-7,
    )
    assert np.isfinite(float(np.asarray(error)))


@pytest.mark.gpu
def test_spatial_constrained_nvt_async_submission_is_guarded_and_state_preserving(
    monkeypatch,
):
    """One async force submission overlaps only ordinary prepared Metal steps."""

    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.3, 1.0],
            [2.4, 2.2, 1.1],
            [4.0, 4.0, 4.0],
            [5.1, 4.0, 4.0],
            [4.0, 5.2, 4.0],
            [5.2, 5.1, 4.2],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    masses = np.ones((positions.shape[0],), dtype=np.float32)
    cell = Cell.cubic(8.0)
    constraints = DistanceConstraints([(0, 1)], distances=[1.0], max_iterations=8)

    def run(*, runtime_profile=False):
        potential = NonbondedPotential(
            sigma=np.full((positions.shape[0],), 0.8, dtype=np.float32),
            epsilon=np.full((positions.shape[0],), 0.05, dtype=np.float32),
            charges=np.zeros((positions.shape[0],), dtype=np.float32),
            cutoff=2.5,
        )
        manager = NeighborListManager(
            cell,
            cutoff=2.5,
            skin=0.4,
            check_interval=1,
            backend="mlx_cell_tiles",
            displacement_check_backend="mlx_scalar",
        )
        return simulate_nvt(
            positions,
            velocities,
            masses=masses,
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            constraints=constraints,
            config=SimulationConfig(
                dt=0.0005,
                steps=5,
                sample_interval=5,
                diagnostic_interval=2,
                pressure_diagnostics=False,
                runtime_profile=runtime_profile,
            ),
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=17),
        )

    original_async_eval = mx.async_eval
    submissions = []

    def record_async_eval(*values):
        submissions.append(values)
        return original_async_eval(*values)

    monkeypatch.setattr(mx, "async_eval", record_async_eval)
    asynchronous = run()
    assert len(submissions) == 2

    profiled = run(runtime_profile=True)
    assert profiled.route_profile["reconciled"] is True
    assert len(submissions) == 2

    monkeypatch.setattr(
        md_module,
        "_async_force_submission_enabled",
        lambda *args, **kwargs: False,
    )
    synchronous = run()
    mx.eval(
        asynchronous.final_state.positions,
        asynchronous.final_state.velocities,
        asynchronous.final_state.forces,
        synchronous.final_state.positions,
        synchronous.final_state.velocities,
        synchronous.final_state.forces,
    )

    for asynchronous_values, synchronous_values in (
        (asynchronous.final_state.positions, synchronous.final_state.positions),
        (asynchronous.final_state.velocities, synchronous.final_state.velocities),
        (asynchronous.final_state.forces, synchronous.final_state.forces),
        (asynchronous.potential_energy, synchronous.potential_energy),
        (asynchronous.constraint_max_error, synchronous.constraint_max_error),
        (asynchronous.pair_count, synchronous.pair_count),
        (asynchronous.rebuild_count, synchronous.rebuild_count),
    ):
        np.testing.assert_allclose(
            np.asarray(asynchronous_values),
            np.asarray(synchronous_values),
            rtol=1.0e-6,
            atol=2.0e-6,
        )
    assert {
        key: value
        for key, value in asynchronous.runtime_sync_report.items()
        if key.endswith("_count")
    } == {
        key: value
        for key, value in synchronous.runtime_sync_report.items()
        if key.endswith("_count")
    }


@pytest.mark.gpu
def test_fused_lj_matches_op_chain():
    """Fused kernel reproduces the op-chain energy and forces on the same pair list."""

    positions, cell = fcc_lattice(512, density=0.8)
    pos_np = np.asarray(positions, dtype=np.float32)
    pos = mx.array(pos_np)
    pairs = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_pairs"
    ).interactions

    op_chain = LennardJonesPotential(cutoff=2.5, use_fused_kernel=False)
    e_ref, f_ref = op_chain.energy_forces(pos, cell, pairs=pairs)

    # Direct kernel call.
    e_fused, f_fused = fused_lj_forces(
        pos, pairs, mx.diag(cell.matrix), epsilon=1.0, sigma=1.0, cutoff=2.5, shift=True
    )
    mx.eval(e_ref, f_ref, e_fused, f_fused)
    assert abs(float(e_ref) - float(e_fused)) < 1e-2
    assert float(mx.max(mx.abs(f_ref - f_fused))) < 1e-3

    # Routed through the potential's use_fused_kernel gate.
    fused_potential = LennardJonesPotential(cutoff=2.5, use_fused_kernel=True)
    e_gate, f_gate = fused_potential.energy_forces(pos, cell, pairs=pairs)
    mx.eval(e_gate, f_gate)
    assert abs(float(e_ref) - float(e_gate)) < 1e-2
    assert float(mx.max(mx.abs(f_ref - f_gate))) < 1e-3


@pytest.mark.gpu
def test_neighbor_cutoff_mask_matches_mlx_and_preserves_compact_pair_order():
    """Fused neighbor masking preserves cutoff membership and deterministic order."""

    rng = np.random.default_rng(37)
    positions_np = rng.uniform(0.0, 8.0, size=(96, 3)).astype(np.float32)
    positions_np[0] = [0.0, 0.0, 0.0]
    positions_np[1] = [2.0, 0.0, 0.0]
    positions_np[95] = [6.0, 0.0, 0.0]
    positions = mx.array(positions_np)
    cell = Cell.cubic(8.0)
    pairs_i = mx.array([0, 0, 1, 3, 17, 31], dtype=mx.int32)
    pairs_j = mx.array([1, 95, 2, 4, 18, 63], dtype=mx.int32)
    search_radius = 2.0

    displacement = cell.minimum_image(positions[pairs_i] - positions[pairs_j])
    expected_mask = mx.sum(displacement * displacement, axis=1) < search_radius * search_radius
    fused_mask = neighbor_pair_cutoff_mask(
        positions,
        pairs_i,
        pairs_j,
        cell.lengths,
        search_radius=search_radius,
    )
    mx.eval(expected_mask, fused_mask)
    assert np.array_equal(np.asarray(fused_mask), np.asarray(expected_mask))
    assert not bool(np.asarray(fused_mask)[0])
    assert not bool(np.asarray(fused_mask)[1])

    prefix = mx.cumsum(fused_mask.astype(mx.int32))
    mx.eval(prefix)
    accepted_count = int(np.asarray(prefix[-1]))
    accepted_i, accepted_j = neighbor_pair_ordered_scatter(
        pairs_i,
        pairs_j,
        fused_mask,
        prefix,
    )
    compact = mx.stack(
        (accepted_i[:accepted_count], accepted_j[:accepted_count]),
        axis=1,
    )
    mx.eval(compact)
    mask_np = np.asarray(expected_mask)
    expected_pairs = np.stack(
        (
            np.asarray(pairs_i)[mask_np],
            np.asarray(pairs_j)[mask_np],
        ),
        axis=1,
    )
    assert np.array_equal(np.asarray(compact), expected_pairs)

    first = build_neighbor_list(
        positions_np,
        cell,
        cutoff=1.8,
        skin=0.3,
        sort_pairs=False,
        backend="mlx_cell_pairs",
    )
    second = build_neighbor_list(
        positions_np,
        cell,
        cutoff=1.8,
        skin=0.3,
        sort_pairs=False,
        backend="mlx_cell_pairs",
    )
    oracle = build_neighbor_list(
        positions_np,
        cell,
        cutoff=1.8,
        skin=0.3,
        backend="periodic_cell_list",
    )
    assert np.array_equal(np.asarray(first.pairs), np.asarray(second.pairs))
    assert {tuple(pair) for pair in np.asarray(first.pairs).tolist()} == {
        tuple(pair) for pair in np.asarray(oracle.pairs).tolist()
    }
    assert first.compaction_backend == "metal_spatial_prefix_scan"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    ["empty", "single", "periodic-boundary", "periodic-alias", "dense"],
)
def test_spatial_neighbor_pipeline_matches_cpu_oracle_for_edge_cases(case):
    """Spatial Metal emission is exact, unique, and repeatable at edge cases."""

    if case == "empty":
        positions = np.empty((0, 3), dtype=np.float32)
        cell = Cell.cubic(4.0)
        cutoff = 0.8
    elif case == "single":
        positions = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
        cell = Cell.cubic(4.0)
        cutoff = 0.8
    elif case == "periodic-boundary":
        positions = np.array(
            [[0.01, 1.0, 1.0], [3.99, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        cell = Cell.cubic(4.0)
        cutoff = 0.1
    elif case == "periodic-alias":
        positions = np.array(
            [
                [0.05, 0.05, 0.05],
                [0.35, 0.05, 0.05],
                [1.85, 0.05, 0.05],
                [1.85, 1.85, 1.85],
            ],
            dtype=np.float32,
        )
        cell = Cell.cubic(2.0)
        cutoff = 1.6
    else:
        rng = np.random.default_rng(91)
        positions = rng.uniform(0.0, 0.4, size=(64, 3)).astype(np.float32)
        cell = Cell.cubic(4.0)
        cutoff = 0.3

    oracle = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        sort_pairs=True,
        backend="periodic_cell_list",
    )
    first = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        sort_pairs=False,
        backend="mlx_cell_pairs",
    )
    second = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        sort_pairs=False,
        backend="mlx_cell_pairs",
    )
    sorted_pairs = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        sort_pairs=True,
        backend="mlx_cell_pairs",
    )
    tiled = build_neighbor_list(
        positions,
        cell,
        cutoff=cutoff,
        skin=0.0,
        sort_pairs=True,
        backend="mlx_cell_tiles",
    )

    expected = np.asarray(oracle.pairs)
    observed = np.asarray(first.pairs)
    assert np.array_equal(observed, np.asarray(second.pairs))
    assert {tuple(pair) for pair in observed.tolist()} == {
        tuple(pair) for pair in expected.tolist()
    }
    assert first.pair_count == len({tuple(pair) for pair in observed.tolist()})
    assert np.array_equal(np.asarray(sorted_pairs.pairs), expected)
    assert np.array_equal(np.asarray(tiled.pairs), expected)
    assert tiled.tiles is not None
    assert np.array_equal(np.asarray(tiled.tiles.materialize_pairs()), expected)
    assert tiled.tiles.force_columns is not None
    assert tiled.tiles.force_group_starts is not None
    assert tiled.tiles.force_group_counts is not None
    assert first.candidate_count is not None
    assert first.candidate_count >= first.pair_count
    assert first.compaction_backend == "metal_spatial_prefix_scan"
    assert tiled.compaction_backend == "metal_spatial_tile_prefix_scan"


@pytest.mark.gpu
def test_spatial_tile_builder_and_direct_kernel_match_compact_pair_route():
    """Device-built spatial tiles preserve membership, topology, and forces."""

    rng = np.random.default_rng(41)
    lattice = (
        np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3)), axis=-1)
        .reshape((-1, 3))
        .astype(np.float32)
    )
    positions_np = 0.6 + 1.65 * lattice[:23]
    positions_np += rng.uniform(-0.08, 0.08, size=positions_np.shape).astype(np.float32)
    positions_np[0, 0] = 0.05
    positions_np[-1, 0] = 8.95
    positions = mx.array(positions_np, dtype=mx.float32)
    cell = Cell.cubic(9.0)
    pair_neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=2.6,
        skin=0.35,
        sort_pairs=True,
        backend="mlx_cell_pairs",
    )
    tile_neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=2.6,
        skin=0.35,
        sort_pairs=True,
        backend="mlx_cell_tiles",
    )
    assert tile_neighbors.tiles is not None
    assert not tile_neighbors.diagnostic_pairs_materialized
    assert tile_neighbors.estimated_pair_bytes == tile_neighbors.tiles.estimated_bytes
    assert tile_neighbors.compaction_backend == "metal_spatial_tile_prefix_scan"
    tile_blocks = np.asarray(tile_neighbors.tiles.tile_blocks)
    force_columns = np.asarray(tile_neighbors.tiles.force_columns)
    group_starts = np.asarray(tile_neighbors.tiles.force_group_starts)
    group_counts = np.asarray(tile_neighbors.tiles.force_group_counts)
    group_ends = group_starts + group_counts
    scheduled_tiles = force_columns // tile_neighbors.tiles.block_size
    scheduled_columns = force_columns % tile_neighbors.tiles.block_size
    assert np.all(tile_blocks[1:, 0] >= tile_blocks[:-1, 0])
    assert group_starts[0] == 0
    assert np.all(group_starts[1:] == group_ends[:-1])
    assert group_ends[-1] == tile_neighbors.tiles.active_column_count
    assert np.all(group_counts >= 1)
    assert np.all(group_counts <= DEFAULT_MLX_CELL_TILE_FORCE_GROUP_SIZE)
    assert np.all(
        tile_blocks[scheduled_tiles[group_starts], 0]
        == tile_blocks[scheduled_tiles[group_ends - 1], 0]
    )
    member_words = np.asarray(tile_neighbors.tiles.member_mask)[:, 0]
    column_patterns = np.array([0x1111, 0x2222, 0x4444, 0x8888], dtype=np.uint32)
    assert np.all(member_words[scheduled_tiles] & column_patterns[scheduled_columns])
    assert len(force_columns) == sum(
        np.count_nonzero(word & column_patterns) for word in member_words
    )
    np.testing.assert_array_equal(
        np.asarray(tile_neighbors.diagnostic_pairs),
        np.asarray(pair_neighbors.diagnostic_pairs),
    )
    np.testing.assert_array_equal(
        np.asarray(tile_neighbors.tiles.materialize_pairs()),
        np.asarray(pair_neighbors.diagnostic_pairs),
    )

    sigma = rng.uniform(0.85, 1.15, size=23).astype(np.float32)
    epsilon = rng.uniform(0.1, 0.35, size=23).astype(np.float32)
    charges = rng.uniform(-0.45, 0.45, size=23).astype(np.float32)
    charges -= np.mean(charges, dtype=np.float32)
    topology = Topology.from_sequences(
        n_atoms=23,
        bonds=[(0, 1), (5, 6), (12, 13)],
        one_four_pairs=[(0, 4), (5, 9)],
        eager_nonbonded_pair_limit=0,
    )
    potential = NonbondedPotential(
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        cutoff=2.6,
        lj_shift=True,
        switch_distance=2.1,
        electrostatics="pme",
        pme_config=PMEConfig(
            mesh_shape=(16, 16, 16),
            alpha=0.34,
            real_cutoff=2.6,
        ),
        topology=topology,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
        exception_pairs=[(1, 9)],
        exception_charge_products=[0.025],
        exception_sigma=[1.05],
        exception_epsilon=[0.08],
    ).bind_pme_plan(cell)
    deferred_neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=2.6,
        skin=0.35,
        sort_pairs=False,
        backend="mlx_cell_tiles",
    )
    deferred_tiles = deferred_neighbors.tiles
    assert deferred_tiles is not None
    pipeline = _PreparedForcePipeline.prepare((potential,), cell=cell)
    prepared = pipeline.bind(deferred_neighbors)
    assert prepared.interactions is None
    assert not deferred_neighbors.diagnostic_pairs_materialized
    deferred_forces = prepared.forces(positions, evaluation_positions=positions)
    mx.eval(deferred_forces)
    assert np.all(np.isfinite(np.asarray(deferred_forces)))
    assert not deferred_neighbors.diagnostic_pairs_materialized
    tile_energy, tile_diagnostic_forces, tile_components = (
        potential._runtime_energy_forces_with_components(
            positions,
            cell,
            deferred_tiles,
        )
    )
    pair_energy, pair_diagnostic_forces, pair_components = (
        potential._runtime_energy_forces_with_components(
            positions,
            cell,
            pair_neighbors.diagnostic_pairs,
        )
    )
    mx.eval(
        tile_energy,
        tile_diagnostic_forces,
        pair_energy,
        pair_diagnostic_forces,
        *tile_components.values(),
        *pair_components.values(),
    )
    np.testing.assert_allclose(
        np.asarray(tile_energy),
        np.asarray(pair_energy),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(tile_diagnostic_forces),
        np.asarray(pair_diagnostic_forces),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    for name in pair_components:
        np.testing.assert_allclose(
            np.asarray(tile_components[name]),
            np.asarray(pair_components[name]),
            rtol=2.0e-5,
            atol=2.0e-3,
        )
    assert not deferred_neighbors.diagnostic_pairs_materialized
    tile_binding = potential._prepare_tile_force_binding(
        cell,
        tile_neighbors.diagnostic_pairs,
        tile_neighbors.tiles,
    )
    assert tile_binding is not NotImplemented
    assert tile_binding.tile_decline_reason is None
    assert tile_binding.aligned_lj_scales is None
    pair_binding = potential._prepare_force_binding(
        cell,
        pair_neighbors.diagnostic_pairs,
    )
    assert pair_binding is not NotImplemented

    tile_direct = potential._direct_forces_from_binding(positions, tile_binding)
    pair_direct = potential._direct_forces_from_binding(positions, pair_binding)
    fused_corrections = potential._prepared_sparse_correction_forces(
        positions,
        tile_binding,
    )
    reference_corrections = potential._exception_lj_forces(
        positions,
        cell,
    ) + potential._periodic_coulomb_correction_forces(
        positions,
        cell,
    )
    tile_full = potential._forces_from_binding(positions, tile_binding)
    pair_full = potential._forces_from_binding(positions, pair_binding)
    pipeline = _PreparedForcePipeline.prepare((potential,), cell=cell)
    routed_forces = pipeline.bind(tile_neighbors).forces(positions)
    mx.eval(
        tile_direct,
        pair_direct,
        fused_corrections,
        reference_corrections,
        tile_full,
        pair_full,
        routed_forces,
    )
    np.testing.assert_allclose(
        np.asarray(tile_direct),
        np.asarray(pair_direct),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(tile_full),
        np.asarray(pair_full),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(fused_corrections),
        np.asarray(reference_corrections),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(routed_forces),
        np.asarray(pair_full),
        rtol=2.0e-5,
        atol=2.0e-3,
    )


@pytest.mark.gpu
def test_sparse_pme_correction_uses_reference_half_box_tie_convention():
    """Sparse correction forces match Cell.minimum_image at exactly half a box."""

    positions = mx.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=mx.float32)
    pairs = mx.array([[0, 1]], dtype=mx.int32)
    cell = Cell.cubic(6.0)
    box_lengths = mx.diag(cell.matrix)
    box = mx.concatenate((box_lengths, 1.0 / box_lengths))
    observed = fused_sparse_pme_correction_forces(
        positions,
        pairs,
        box,
        mx.array([1.0], dtype=mx.float32),
        mx.array([0.0], dtype=mx.float32),
        mx.array([0.0], dtype=mx.float32),
        coulomb_constant=1.0,
    )
    displacement = cell.minimum_image(positions[0] - positions[1])
    expected_force = displacement / mx.power(mx.sum(displacement * displacement), 1.5)
    expected = mx.stack((expected_force, -expected_force))
    mx.eval(observed, expected)
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=1e-6)


@pytest.mark.gpu
def test_direct_plus_sparse_correction_matches_reference_at_half_box():
    """Pair and tile PME force sums share Cell's exact half-box convention."""

    positions = mx.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=mx.float32)
    pairs = mx.array([[0, 1]], dtype=mx.int32)
    cell = Cell.cubic(6.0)
    box_lengths = mx.diag(cell.matrix)
    box = mx.concatenate((box_lengths, 1.0 / box_lengths))
    charges = mx.array([1.0, -1.0], dtype=mx.float32)
    alpha = 0.35
    direct = fused_parameterized_pme_direct_force_only(
        positions,
        pairs,
        box_lengths,
        mx.ones((2,), dtype=mx.float32),
        mx.zeros((2,), dtype=mx.float32),
        charges,
        mx.zeros((1,), dtype=mx.float32),
        cutoff=3.1,
        shift=False,
        switch_distance=None,
        coulomb_constant=1.0,
        alpha=alpha,
    )
    prepared_direct = _prepared_parameterized_pme_direct_force_only(
        positions,
        pairs,
        box,
        mx.full((2,), 0.5, dtype=mx.float32),
        mx.zeros((2,), dtype=mx.float32),
        charges,
        mx.zeros((1,), dtype=mx.float32),
        cutoff=3.1,
        shift=False,
        switch_distance=None,
        coulomb_constant=1.0,
        alpha=alpha,
    )
    tile_direct = _tile_parameterized_pme_direct_force_only(
        positions,
        mx.array([[0, 1, -1, -1]], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[2]], dtype=mx.uint32),
        mx.zeros((1, 1), dtype=mx.uint32),
        mx.zeros((1, 1), dtype=mx.uint32),
        mx.array([1], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        mx.array([1], dtype=mx.int32),
        box,
        mx.full((2,), 0.5, dtype=mx.float32),
        mx.zeros((2,), dtype=mx.float32),
        charges,
        cutoff=3.1,
        shift=False,
        switch_distance=None,
        one_four_scale=0.5,
        coulomb_constant=1.0,
        alpha=alpha,
    )
    correction = fused_sparse_pme_correction_forces(
        positions,
        pairs,
        box,
        mx.array([1.0], dtype=mx.float32),
        mx.array([0.0], dtype=mx.float32),
        mx.array([0.0], dtype=mx.float32),
        coulomb_constant=1.0,
    )

    displacement = cell.minimum_image(positions[0] - positions[1])
    r2 = mx.sum(displacement * displacement)
    distance = mx.sqrt(r2)
    erfc_term = 1.0 - mx.erf(alpha * distance)
    direct_scalar = -(
        erfc_term / (r2 * distance)
        + (2.0 * alpha / np.sqrt(np.pi)) * mx.exp(-(alpha * alpha) * r2) / r2
    )
    correction_scalar = 1.0 / (r2 * distance)
    expected_atom = (direct_scalar + correction_scalar) * displacement
    expected = mx.stack((expected_atom, -expected_atom))
    observed = [value + correction for value in (direct, prepared_direct, tile_direct)]
    mx.eval(expected, *observed)
    for value in observed:
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected), rtol=2e-6)


@pytest.mark.gpu
def test_neighbor_tiles_reject_invalid_force_group_schedule():
    """Tile geometry rejects schedules that mix different left blocks."""

    with pytest.raises(ValueError, match="same-left"):
        NeighborTiles(
            atom_blocks=mx.array(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=mx.int32,
            ),
            tile_blocks=mx.array([[0, 0], [1, 1]], dtype=mx.int32),
            member_mask=mx.array([[1], [1]], dtype=mx.uint32),
            exact_pair_count=2,
            raw_candidate_count=2,
            force_columns=mx.array([0, 4], dtype=mx.int32),
            force_group_starts=mx.array([0], dtype=mx.int32),
            force_group_counts=mx.array([2], dtype=mx.int32),
        )


@pytest.mark.gpu
def test_spatial_neighbor_manager_releases_rebuild_cache_once(monkeypatch):
    """A completed spatial rebuild releases inactive Metal buffers at the next update."""

    positions, cell = fcc_lattice(256, density=0.8)
    manager = NeighborListManager(
        cell,
        cutoff=2.5,
        skin=0.4,
        backend="mlx_cell_pairs",
    )
    clear_calls = 0
    clear_cache = mx.clear_cache

    def counted_clear_cache():
        nonlocal clear_calls
        clear_calls += 1
        clear_cache()

    monkeypatch.setattr(mx, "clear_cache", counted_clear_cache)
    manager.update(positions)
    assert clear_calls == 0
    manager.update(positions)
    assert clear_calls == 1
    manager.update(positions)
    assert clear_calls == 1


@pytest.mark.gpu
def test_md_cache_scope_bounds_allocator_without_clearing_each_rebuild(monkeypatch):
    """MD uses a bounded reusable cache and restores the caller's policy."""

    positions, cell = fcc_lattice(256, density=0.8)
    manager = NeighborListManager(
        cell,
        cutoff=2.5,
        skin=0.4,
        backend="mlx_cell_pairs",
    )
    clear_calls = 0
    cache_transitions = []
    clear_cache = mx.clear_cache
    set_cache_limit = mx.set_cache_limit

    def counted_clear_cache():
        nonlocal clear_calls
        clear_calls += 1
        clear_cache()

    def recorded_cache_limit(limit):
        previous = set_cache_limit(limit)
        cache_transitions.append((limit, previous))
        return previous

    monkeypatch.setattr(mx, "clear_cache", counted_clear_cache)
    monkeypatch.setattr(mx, "set_cache_limit", recorded_cache_limit)
    with _bounded_metal_md_cache():
        manager.update(positions)
        manager.update(positions)

    assert clear_calls == 0
    assert cache_transitions[0][0] == _MLX_MD_CACHE_LIMIT_BYTES
    assert cache_transitions[1][0] == cache_transitions[0][1]
    assert len(cache_transitions) == 2


@pytest.mark.gpu
def test_fused_falls_back_when_unsupported():
    """use_fused_kernel=True with no cell takes the op-chain fallback (gate requires a cell).

    The two runs agree only to ULP, not bit-for-bit: MLX's own GPU ``.at[].add()`` scatter
    is itself summation-order non-deterministic, so even op-chain-vs-op-chain differs by ~1e-7.
    """

    positions, cell = fcc_lattice(256, density=0.8)
    pos_np = np.asarray(positions, dtype=np.float32)
    pos = mx.array(pos_np)
    pairs = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_pairs"
    ).interactions

    fused = LennardJonesPotential(cutoff=2.5, use_fused_kernel=True)
    op_chain = LennardJonesPotential(cutoff=2.5, use_fused_kernel=False)
    # cell=None fails the orthorhombic gate -> both take the op-chain.
    e_f, f_f = fused.energy_forces(pos, None, pairs=pairs)
    e_o, f_o = op_chain.energy_forces(pos, None, pairs=pairs)
    mx.eval(e_f, f_f, e_o, f_o)
    assert float(mx.max(mx.abs(f_f - f_o))) < 1e-4
    assert abs(float(e_f) - float(e_o)) < 1e-2


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("shift", "switch_distance", "one_four_scale"),
    [
        (False, None, 1.0),
        (True, None, 0.5),
        (False, 2.0, 0.5),
    ],
)
def test_parameterized_fused_lj_matches_topology_op_chain(
    shift,
    switch_distance,
    one_four_scale,
):
    """Parameterized Metal LJ matches exclusions, scales, shifts, and switching."""

    positions, cell = fcc_lattice(512, density=0.8)
    indices = np.arange(positions.shape[0], dtype=np.float32)
    sigma = 0.95 + 0.1 * (indices % 7.0) / 6.0
    epsilon = 0.8 + 0.4 * (indices % 11.0) / 10.0
    topology = Topology.from_sequences(
        n_atoms=positions.shape[0],
        bonds=[(0, 1)],
        one_four_pairs=[(0, 3)],
        eager_nonbonded_pair_limit=0,
    )
    potential = NonbondedPotential(
        sigma=sigma,
        epsilon=epsilon,
        charges=np.zeros((positions.shape[0],), dtype=np.float32),
        cutoff=2.5,
        lj_shift=shift,
        switch_distance=switch_distance,
        topology=topology,
        lj_one_four_scale=one_four_scale,
    )
    pairs = build_neighbor_list(
        np.asarray(positions),
        cell,
        cutoff=2.5,
        skin=0.4,
        backend="mlx_cell_pairs",
    ).interactions
    reference_energy, reference_forces = potential._regular_lj_components(
        positions,
        cell,
        pairs,
        allow_fused_metal=False,
    )
    fused_energy, fused_forces = potential._regular_lj_components(
        positions,
        cell,
        pairs,
    )

    mx.eval(reference_energy, reference_forces, fused_energy, fused_forces)
    np.testing.assert_allclose(
        np.asarray(fused_energy),
        np.asarray(reference_energy),
        rtol=1e-5,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(fused_forces),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=2e-3,
    )


@pytest.mark.gpu
def test_fused_parameterized_pme_direct_matches_decomposed_path():
    """One-dispatch LJ/PME direct space matches the decomposed production formulas."""

    positions = mx.array(
        [
            [0.0, 0.0, 0.0],
            [1.18, 0.0, 0.0],
            [0.0, 1.35, 0.0],
            [1.25, 1.10, 0.2],
        ],
        dtype=mx.float32,
    )
    cell = Cell.cubic(6.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        one_four_pairs=[(0, 3)],
        eager_nonbonded_pair_limit=0,
    )
    config = PMEConfig(
        mesh_shape=(16, 16, 16),
        alpha=0.35,
        real_cutoff=2.5,
        assignment_order=5,
    )
    potential = NonbondedPotential(
        sigma=[1.0, 1.1, 0.9, 1.05],
        epsilon=[0.2, 0.3, 0.25, 0.35],
        charges=[0.25, -0.25, 0.1, -0.1],
        cutoff=2.5,
        lj_shift=False,
        electrostatics="pme",
        pme_config=config,
        topology=topology,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
    ).bind_pme_plan(cell)
    pairs = build_neighbor_list(
        positions,
        cell,
        cutoff=2.5,
        skin=0.2,
        backend="mlx_cell_pairs",
    ).interactions

    reference_energy, reference_forces, reference_components = (
        potential._pme_energy_forces_with_components(
            positions,
            cell,
            pairs,
        )
    )
    fused_energy, fused_forces = potential._pme_energy_forces(
        positions,
        cell,
        pairs,
    )
    runtime_energy, runtime_forces, runtime_components = (
        potential._runtime_energy_forces_with_components(
            positions,
            cell,
            pairs,
        )
    )
    aligned_lj_scales = potential._compact_aligned_lj_scales(pairs)
    _, direct_reference_forces, _, _ = fused_parameterized_pme_direct_components(
        positions,
        pairs,
        mx.diag(cell.matrix),
        potential.sigma,
        potential.epsilon,
        potential.charges,
        aligned_lj_scales,
        cutoff=potential.cutoff,
        shift=potential.lj_shift,
        switch_distance=potential.switch_distance,
        coulomb_constant=potential.coulomb_constant,
        alpha=config.alpha,
    )
    direct_force_only = fused_parameterized_pme_direct_force_only(
        positions,
        pairs,
        mx.diag(cell.matrix),
        potential.sigma,
        potential.epsilon,
        potential.charges,
        aligned_lj_scales,
        cutoff=potential.cutoff,
        shift=potential.lj_shift,
        switch_distance=potential.switch_distance,
        coulomb_constant=potential.coulomb_constant,
        alpha=config.alpha,
    )
    runtime_force_only = potential._runtime_forces(
        positions,
        cell=cell,
        pairs=pairs,
    )
    assert runtime_force_only is not NotImplemented
    prepared_binding = potential._prepare_force_binding(cell, pairs)
    assert prepared_binding is not NotImplemented
    prepared_force_only = potential._forces_from_binding(
        positions,
        prepared_binding,
    )

    mx.eval(
        reference_energy,
        reference_forces,
        fused_energy,
        fused_forces,
        runtime_energy,
        runtime_forces,
        direct_reference_forces,
        direct_force_only,
        runtime_force_only,
        prepared_force_only,
        *runtime_components.values(),
    )
    assert potential._aligned_lj_scale_cache is not None
    np.testing.assert_allclose(
        np.asarray(fused_energy),
        np.asarray(reference_energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(fused_forces),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(runtime_energy),
        np.asarray(reference_energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(runtime_forces),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(direct_force_only),
        np.asarray(direct_reference_forces),
        rtol=1e-5,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(runtime_force_only),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(prepared_force_only),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=2e-4,
    )
    assert set(runtime_components) == set(reference_components) - {"pme_diagnostics"}
    for name, value in runtime_components.items():
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(reference_components[name]),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.gpu
def test_fused_pme_diagnostic_virial_matches_existing_analytic_route():
    """The fused diagnostic preserves energy, forces, components, and virial."""

    positions = mx.array(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            [3.999, 1.0, 1.0],
            [4.2, 1.0, 1.0],
        ],
        dtype=mx.float32,
    )
    cell = Cell.cubic(8.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1), (2, 3)],
        partial_charges=[0.4, -0.4, 0.25, -0.25],
        nonbonded_cutoff=3.0,
        eager_nonbonded_pair_limit=0,
    )
    potential = NonbondedPotential(
        sigma=[0.9, 1.0, 1.1, 0.95],
        epsilon=[0.15, 0.2, 0.18, 0.12],
        charges=[0.4, -0.4, 0.25, -0.25],
        cutoff=3.0,
        lj_shift=False,
        electrostatics="pme",
        pme_config=PMEConfig(
            mesh_shape=(8, 8, 8),
            alpha=0.4,
            real_cutoff=3.0,
            assignment_order=5,
        ),
        topology=topology,
    ).bind_pme_plan(cell)
    pairs = build_neighbor_list(
        positions,
        cell,
        cutoff=3.0,
        skin=0.3,
        backend="mlx_cell_pairs",
    ).interactions
    masses = mx.ones((4,), dtype=mx.float32)
    molecule_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)

    reference_energy, reference_forces, reference_components = (
        potential._runtime_energy_forces_with_components(
            positions,
            cell,
            pairs,
        )
    )
    reference_virial = potential.analytic_virial_tensor(
        positions,
        cell=cell,
        pairs=pairs,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    fused = potential._runtime_energy_forces_with_components_virial(
        positions,
        cell,
        pairs,
        masses=masses,
        molecule_ids=molecule_ids,
    )
    assert fused is not NotImplemented
    energy, forces, components, virial = fused
    reused = potential._runtime_energy_forces_with_components_virial_reusing_pairs(
        positions,
        cell,
        pairs,
        masses=masses,
        molecule_ids=molecule_ids,
        cutoff_strain_pairs=pairs,
    )
    assert reused is not NotImplemented
    reused_energy, reused_forces, reused_components, reused_virial = reused
    mx.eval(
        reference_energy,
        reference_forces,
        reference_virial,
        energy,
        forces,
        virial,
        reused_energy,
        reused_forces,
        reused_virial,
        *reference_components.values(),
        *components.values(),
        *reused_components.values(),
    )

    np.testing.assert_allclose(
        np.asarray(energy),
        np.asarray(reference_energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(forces),
        np.asarray(reference_forces),
        rtol=1e-5,
        atol=3e-4,
    )
    assert set(components) == set(reference_components)
    for name, value in components.items():
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(reference_components[name]),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        np.diag(np.asarray(virial)),
        np.diag(np.asarray(reference_virial)),
        rtol=3e-3,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(reused_energy),
        np.asarray(energy),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(reused_forces),
        np.asarray(forces),
        rtol=1e-5,
        atol=3e-4,
    )
    assert set(reused_components) == set(components)
    np.testing.assert_allclose(
        np.diag(np.asarray(reused_virial)),
        np.diag(np.asarray(virial)),
        rtol=3e-3,
        atol=5e-2,
    )

    translated_positions = positions + mx.array(
        [
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [-8.0, 8.0, 0.0],
            [16.0, -8.0, 8.0],
        ],
        dtype=mx.float32,
    )
    translated_pairs = build_neighbor_list(
        translated_positions,
        cell,
        cutoff=3.0,
        skin=0.3,
        backend="mlx_cell_pairs",
    ).interactions
    translated_reference = potential.analytic_virial_tensor(
        translated_positions,
        cell=cell,
        pairs=translated_pairs,
        masses=masses,
        molecule_ids=None,
    )
    translated_fused = potential._runtime_energy_forces_with_components_virial(
        translated_positions,
        cell,
        translated_pairs,
        masses=masses,
        molecule_ids=None,
    )
    assert translated_fused is not NotImplemented
    translated_virial = translated_fused[3]
    mx.eval(translated_reference, translated_virial)
    np.testing.assert_allclose(
        np.diag(np.asarray(translated_virial)),
        np.diag(np.asarray(translated_reference)),
        rtol=3e-3,
        atol=5e-2,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_fused_nvt_matches_op_chain_end_to_end():
    """A batched-block NVT run with the fused kernel tracks the op-chain trajectory.

    Also proves the kernel composes inside the mx.compile'd Langevin block.
    """

    n = 256
    positions, cell = fcc_lattice(n, density=0.8)
    pos_np = np.asarray(positions, dtype=np.float32)
    vel_np = np.asarray(thermal_velocities(n, temperature=1.0, seed=7), dtype=np.float32)

    def run(use_fused):
        potential = LennardJonesPotential(cutoff=2.5, use_fused_kernel=use_fused)
        manager = NeighborListManager(
            cell, cutoff=2.5, skin=0.4, check_interval=1, backend="mlx_cell_pairs"
        )
        config = SimulationConfig(
            dt=0.002,
            steps=120,
            sample_interval=30,
            diagnostic_interval=30,
            evaluation_interval=25,
            block_size=8,
        )
        return simulate_nvt(
            mx.array(pos_np),
            mx.array(vel_np),
            cell=cell,
            force_terms=potential,
            neighbor_manager=manager,
            config=config,
            thermostat=LangevinThermostat(temperature=1.0, friction=0.5, seed=7),
        )

    reference = run(use_fused=False)
    fused = run(use_fused=True)

    assert np.allclose(
        np.asarray(fused.total_energy), np.asarray(reference.total_energy), rtol=0.0, atol=1e-3
    )
    assert np.allclose(
        np.asarray(fused.sampled_positions),
        np.asarray(reference.sampled_positions),
        rtol=0.0,
        atol=1e-3,
    )
