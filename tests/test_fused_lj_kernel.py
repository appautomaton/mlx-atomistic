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

from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nvt,
)
from mlx_atomistic.metal_kernels import (
    fused_lj_forces,
    fused_parameterized_pme_direct_components,
    fused_parameterized_pme_direct_force_only,
    neighbor_pair_cutoff_mask,
    neighbor_pair_ordered_scatter,
)
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
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
    expected_mask = (
        mx.sum(displacement * displacement, axis=1) < search_radius * search_radius
    )
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
    assert {
        tuple(pair) for pair in np.asarray(first.pairs).tolist()
    } == {
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

    expected = np.asarray(oracle.pairs)
    observed = np.asarray(first.pairs)
    assert np.array_equal(observed, np.asarray(second.pairs))
    assert {tuple(pair) for pair in observed.tolist()} == {
        tuple(pair) for pair in expected.tolist()
    }
    assert first.pair_count == len({tuple(pair) for pair in observed.tolist()})
    assert np.array_equal(np.asarray(sorted_pairs.pairs), expected)
    assert first.candidate_count is not None
    assert first.candidate_count >= first.pair_count
    assert first.compaction_backend == "metal_spatial_prefix_scan"


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
    reused = (
        potential
        ._runtime_energy_forces_with_components_virial_reusing_pairs(
            positions,
            cell,
            pairs,
            masses=masses,
            molecule_ids=molecule_ids,
            cutoff_strain_pairs=pairs,
        )
    )
    assert reused is not NotImplemented
    reused_energy, reused_forces, reused_components, reused_virial = (
        reused
    )
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
    translated_fused = (
        potential._runtime_energy_forces_with_components_virial(
            translated_positions,
            cell,
            translated_pairs,
            masses=masses,
            molecule_ids=None,
        )
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
