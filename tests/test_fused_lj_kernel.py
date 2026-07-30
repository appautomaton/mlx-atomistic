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
from mlx_atomistic.metal_kernels import fused_lj_forces
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

    mx.eval(
        reference_energy,
        reference_forces,
        fused_energy,
        fused_forces,
        runtime_energy,
        runtime_forces,
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
    assert set(runtime_components) == set(reference_components) - {"pme_diagnostics"}
    for name, value in runtime_components.items():
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(reference_components[name]),
            rtol=1e-5,
            atol=1e-5,
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
