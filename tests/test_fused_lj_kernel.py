"""Physics-lock tests for the fused Metal LJ force kernel (perf lever #4).

The fused kernel runs only on a Metal GPU; ``conftest.py`` forces the CPU device,
so each test switches to the GPU and skips when Metal is unavailable (headless CI).
Equivalence is locked with loose tolerances, not bit-identical results: the kernel's
atomic scatter is summation-order non-deterministic, the same property as the existing
``.at[].add()`` op-chain (see tests/test_neighbors.py).
"""

from __future__ import annotations

from math import erf, exp, sqrt

import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.forcefields as forcefields_module
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
    _tile_parameterized_pme_direct_force_only,
    fused_lj_forces,
    fused_parameterized_pme_direct_components,
    fused_parameterized_pme_direct_force_only,
    neighbor_pair_cutoff_mask,
    neighbor_pair_ordered_scatter,
)
from mlx_atomistic.neighbors import NeighborListManager, NeighborTiles, build_neighbor_list
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


def _mask_words(lanes: list[int]) -> np.ndarray:
    words = np.zeros((2,), dtype=np.uint32)
    for lane in lanes:
        word, bit = divmod(lane, 32)
        words[word] |= np.uint32(1) << np.uint32(bit)
    return words


def _explicit_pme_direct_forces(
    positions: np.ndarray,
    pairs: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    charges: np.ndarray,
    lj_scales: np.ndarray,
    box_lengths: np.ndarray,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> np.ndarray:
    """Return a host reference for the prepared direct-force equations."""

    forces = np.zeros_like(positions, dtype=np.float64)
    cutoff2 = cutoff * cutoff
    for pair_index, (left, right) in enumerate(pairs):
        displacement = positions[left].astype(np.float64) - positions[right]
        displacement -= box_lengths * np.floor(displacement / box_lengths + 0.5)
        r2 = float(np.dot(displacement, displacement))
        if r2 <= 0.0 or r2 >= cutoff2:
            continue
        distance = sqrt(r2)
        scalar = 0.0
        lj_scale = float(lj_scales[pair_index])
        if lj_scale != 0.0:
            sigma_ij = 0.5 * (float(sigma[left]) + float(sigma[right]))
            epsilon_ij = sqrt(float(epsilon[left]) * float(epsilon[right]))
            sigma2_over_r2 = sigma_ij * sigma_ij / r2
            inv_r6 = sigma2_over_r2**3
            inv_r12 = inv_r6 * inv_r6
            unswitched_energy = 4.0 * epsilon_ij * (inv_r12 - inv_r6)
            if shift:
                sigma2_over_rc2 = sigma_ij * sigma_ij / cutoff2
                inv_rc6 = sigma2_over_rc2**3
                unswitched_energy -= 4.0 * epsilon_ij * (
                    inv_rc6 * inv_rc6 - inv_rc6
                )
            switch_value = 1.0
            switch_derivative = 0.0
            if switch_distance is not None:
                width = cutoff - switch_distance
                x = min(max((distance - switch_distance) / width, 0.0), 1.0)
                switch_value = 1.0 - (10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5)
                if switch_distance < distance < cutoff:
                    switch_derivative = -(
                        30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4
                    ) / width
            scalar += (
                24.0 * epsilon_ij * (2.0 * inv_r12 - inv_r6)
                / r2
                * switch_value
                - unswitched_energy * switch_derivative / distance
            ) * lj_scale
        qij = float(charges[left]) * float(charges[right])
        erfc_term = 1.0 - erf(alpha * distance)
        scalar += coulomb_constant * qij * (
            erfc_term / (r2 * distance)
            + 2.0
            * alpha
            / sqrt(np.pi)
            * exp(-(alpha * alpha) * r2)
            / r2
        )
        pair_force = scalar * displacement
        forces[left] += pair_force
        forces[right] -= pair_force
    return forces.astype(np.float32)


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
def test_tile_pme_direct_reduces_shared_endpoints_and_repeats_without_races(
    monkeypatch,
):
    """The 64-lane tile candidate reduces both endpoints before atomics."""

    positions_np = np.asarray(
        [
            [0.20, 0.20, 0.20],
            [2.80, 0.25, 0.20],
            [3.35, 0.20, 0.20],
            [0.30, 2.10, 0.25],
            [1.65, 2.15, 0.35],
            [3.10, 2.25, 0.40],
            [0.25, 4.05, 0.30],
            [2.05, 4.15, 0.25],
            [1.15, 0.95, 1.70],
            [3.75, 0.90, 1.75],
            [0.95, 2.95, 1.85],
            [2.80, 3.05, 1.80],
            [4.55, 2.90, 1.70],
            [0.35, 4.85, 1.90],
            [2.25, 5.10, 1.75],
            [4.10, 4.80, 1.80],
            [1.30, 1.80, 3.45],
            [3.65, 2.00, 3.55],
        ],
        dtype=np.float32,
    )
    atom_blocks = np.asarray(
        [
            np.arange(0, 8, dtype=np.int32),
            np.arange(8, 16, dtype=np.int32),
            [16, 17, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    tile_blocks = np.asarray(
        [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],
        dtype=np.int32,
    )
    lane_rows = [
        [1, 2, 3 * 8 + 4, 5 * 8 + 7],
        [0, 1 * 8 + 1, 2 * 8 + 4, 7 * 8 + 7],
        [0, 1 * 8 + 1, 6 * 8 + 0],
        [1, 2, 3 * 8 + 5, 6 * 8 + 7],
        [0, 2 * 8 + 1, 7 * 8 + 0],
        [1],
    ]
    member_mask = np.stack([_mask_words(lanes) for lanes in lane_rows])
    exact_pair_count = sum(len(lanes) for lanes in lane_rows)
    tiles = NeighborTiles(
        atom_blocks=mx.array(atom_blocks, dtype=mx.int32),
        tile_blocks=mx.array(tile_blocks, dtype=mx.int32),
        member_mask=mx.array(member_mask, dtype=mx.uint32),
        exact_pair_count=exact_pair_count,
        raw_candidate_count=exact_pair_count + 9,
    )
    pairs = tiles.materialize_pairs(sort=False)
    cell = Cell.cubic(12.0)
    sigma_np = np.linspace(0.90, 1.20, positions_np.shape[0], dtype=np.float32)
    epsilon_np = np.linspace(0.12, 0.30, positions_np.shape[0], dtype=np.float32)
    charges_np = np.linspace(-0.35, 0.40, positions_np.shape[0], dtype=np.float32)
    charges_np -= np.mean(charges_np, dtype=np.float32)
    topology = Topology.from_sequences(
        n_atoms=positions_np.shape[0],
        bonds=[(0, 1), (8, 9)],
        one_four_pairs=[(0, 8), (7, 15)],
        eager_nonbonded_pair_limit=0,
    )
    config = PMEConfig(
        mesh_shape=(16, 16, 16),
        alpha=0.31,
        real_cutoff=3.0,
        assignment_order=5,
    )
    potential = NonbondedPotential(
        sigma=sigma_np,
        epsilon=epsilon_np,
        charges=charges_np,
        cutoff=3.0,
        lj_shift=True,
        switch_distance=2.4,
        electrostatics="pme",
        pme_config=config,
        topology=topology,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
        exception_pairs=[(1, 9)],
        exception_charge_products=[0.025],
        exception_sigma=[1.05],
        exception_epsilon=[0.08],
    ).bind_pme_plan(cell)
    binding = potential._prepare_tile_force_binding(cell, pairs, tiles)
    assert binding is not NotImplemented
    assert binding.tile_force_ready
    assert binding.tile_decline_reason == "tile_force_route_not_selected"
    assert binding.tile_launch_grid == (tiles.tile_count * 64, 1, 1)
    assert binding.tile_threadgroup == (64, 1, 1)
    assert binding.tile_global_update_proxy < 6 * tiles.exact_pair_count
    assert np.any(tile_blocks[:, 0] == tile_blocks[:, 1])
    assert np.any(atom_blocks < 0)
    block_frequency = np.bincount(tile_blocks.reshape(-1))
    assert int(np.max(block_frequency)) > 1

    pair_mask, pair_lj_scales, _ = potential._compact_pair_masks_and_scales(pairs)
    aligned_lj_scales = np.where(
        np.asarray(pair_mask),
        np.asarray(pair_lj_scales),
        0.0,
    ).astype(np.float32)
    reference_direct = _explicit_pme_direct_forces(
        positions_np,
        np.asarray(pairs, dtype=np.int32),
        sigma_np,
        epsilon_np,
        charges_np,
        aligned_lj_scales,
        np.asarray(cell.lengths, dtype=np.float64),
        cutoff=potential.cutoff,
        shift=potential.lj_shift,
        switch_distance=potential.switch_distance,
        coulomb_constant=potential.coulomb_constant,
        alpha=config.alpha,
    )
    positions = mx.array(positions_np, dtype=mx.float32)
    assert (
        potential._tile_direct_forces_from_binding(
            positions.astype(mx.float16),
            binding,
        )
        is NotImplemented
    )
    repeated = [
        potential._tile_direct_forces_from_binding(positions, binding)
        for _ in range(24)
    ]
    assert all(result is not NotImplemented for result in repeated)
    mx.eval(*repeated)
    for result in repeated:
        np.testing.assert_allclose(
            np.asarray(result),
            reference_direct,
            rtol=2.0e-5,
            atol=2.0e-3,
        )

    reference_full = potential._forces_from_binding(positions, binding)
    candidate_full = potential._tile_forces_from_binding(positions, binding)
    assert candidate_full is not NotImplemented
    mx.eval(reference_full, candidate_full)
    np.testing.assert_allclose(
        np.asarray(candidate_full),
        np.asarray(reference_full),
        rtol=2.0e-5,
        atol=2.0e-3,
    )

    def _unexpected_tile_route(*args, **kwargs):
        raise AssertionError("production force binding selected the tile candidate")

    monkeypatch.setattr(
        forcefields_module,
        "_tile_parameterized_pme_direct_force_only",
        _unexpected_tile_route,
    )
    production_forces = potential._forces_from_binding(positions, binding)
    mx.eval(production_forces)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [19, 41])
def test_tile_pme_direct_random_geometry_matches_host_explicit_pairs(seed):
    """Random tile inventories match a host explicit-pair force reference."""

    rng = np.random.default_rng(seed)
    lattice = (
        np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3)), axis=-1)
        .reshape((-1, 3))
        .astype(np.float32)
    )
    positions_np = 0.7 + 1.65 * lattice[:23]
    positions_np += rng.uniform(-0.08, 0.08, size=positions_np.shape).astype(np.float32)
    sigma_np = rng.uniform(0.85, 1.15, size=23).astype(np.float32)
    epsilon_np = rng.uniform(0.1, 0.35, size=23).astype(np.float32)
    charges_np = rng.uniform(-0.45, 0.45, size=23).astype(np.float32)
    charges_np -= np.mean(charges_np, dtype=np.float32)
    cell = Cell.cubic(9.0)
    topology = Topology.from_sequences(
        n_atoms=23,
        bonds=[(0, 1), (5, 6), (12, 13)],
        one_four_pairs=[(0, 4), (5, 9)],
        eager_nonbonded_pair_limit=0,
    )
    config = PMEConfig(
        mesh_shape=(16, 16, 16),
        alpha=0.34,
        real_cutoff=2.6,
    )
    potential = NonbondedPotential(
        sigma=sigma_np,
        epsilon=epsilon_np,
        charges=charges_np,
        cutoff=2.6,
        lj_shift=bool(seed % 2),
        switch_distance=None if seed % 2 else 2.1,
        electrostatics="pme",
        pme_config=config,
        topology=topology,
        lj_one_four_scale=0.5,
    ).bind_pme_plan(cell)
    neighbors = build_neighbor_list(
        positions_np,
        cell,
        cutoff=2.6,
        skin=0.35,
        sort_pairs=False,
        backend="mlx_cell_tiles",
    )
    tiles = neighbors.tiles
    assert tiles is not None
    binding = potential._prepare_tile_force_binding(
        cell,
        neighbors.diagnostic_pairs,
        tiles,
    )
    assert binding is not NotImplemented
    assert binding.tile_force_ready
    pairs = neighbors.diagnostic_pairs
    pair_mask, pair_lj_scales, _ = potential._compact_pair_masks_and_scales(pairs)
    aligned_lj_scales = np.where(
        np.asarray(pair_mask),
        np.asarray(pair_lj_scales),
        0.0,
    ).astype(np.float32)
    reference = _explicit_pme_direct_forces(
        positions_np,
        np.asarray(pairs, dtype=np.int32),
        sigma_np,
        epsilon_np,
        charges_np,
        aligned_lj_scales,
        np.asarray(cell.lengths, dtype=np.float64),
        cutoff=potential.cutoff,
        shift=potential.lj_shift,
        switch_distance=potential.switch_distance,
        coulomb_constant=potential.coulomb_constant,
        alpha=config.alpha,
    )
    observed = potential._tile_direct_forces_from_binding(
        mx.array(positions_np, dtype=mx.float32),
        binding,
    )
    assert observed is not NotImplemented
    mx.eval(observed)
    np.testing.assert_allclose(
        np.asarray(observed),
        reference,
        rtol=2.0e-5,
        atol=2.0e-3,
    )


@pytest.mark.gpu
def test_tile_pme_direct_empty_sparse_mask_and_skin_only_lane():
    """Empty masks stay zero and member lanes still honor the force cutoff."""

    positions = mx.array(
        [[0.0, 0.0, 0.0], [3.2, 0.0, 0.0]],
        dtype=mx.float32,
    )
    atom_blocks = mx.array(
        [[0, 1, -1, -1, -1, -1, -1, -1]],
        dtype=mx.int32,
    )
    tile_blocks = mx.array([[0, 0]], dtype=mx.int32)
    empty_mask = mx.zeros((1, 2), dtype=mx.uint32)
    box = mx.array([12.0, 12.0, 12.0, 1 / 12.0, 1 / 12.0, 1 / 12.0])
    common = {
        "box_lengths_and_inverses": box,
        "half_sigma": mx.array([0.5, 0.55], dtype=mx.float32),
        "sqrt_epsilon": mx.array([0.4, 0.5], dtype=mx.float32),
        "charges": mx.array([0.3, -0.2], dtype=mx.float32),
        "cutoff": 3.0,
        "shift": True,
        "switch_distance": 2.4,
        "one_four_scale": 0.5,
        "coulomb_constant": 1.0,
        "alpha": 0.35,
    }
    empty = _tile_parameterized_pme_direct_force_only(
        positions,
        atom_blocks,
        tile_blocks,
        empty_mask,
        empty_mask,
        empty_mask,
        **common,
    )
    skin_only_mask = mx.array(
        np.asarray([_mask_words([1])], dtype=np.uint32),
        dtype=mx.uint32,
    )
    skin_only = _tile_parameterized_pme_direct_force_only(
        positions,
        atom_blocks,
        tile_blocks,
        skin_only_mask,
        skin_only_mask,
        empty_mask,
        **common,
    )
    mx.eval(empty, skin_only)
    np.testing.assert_array_equal(np.asarray(empty), np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(skin_only),
        np.zeros((2, 3), dtype=np.float32),
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
