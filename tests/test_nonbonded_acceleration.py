import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.benchmarks import md_acceleration, md_performance
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate_nvt
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
from mlx_atomistic.nonbonded import (
    DEFAULT_FULL_LOOP_DENSE_THRESHOLD,
    NonbondedExecutionConfig,
    choose_full_loop_nonbonded_policy,
    choose_nonbonded_backend,
    estimate_dense_nonbonded_bytes,
    normalize_nonbonded_electrostatics,
    validate_nonbonded_electrostatics,
)
from mlx_atomistic.pme import PMEConfig
from mlx_atomistic.topology import Topology


def _positions():
    return as_mx_array(
        [
            [0.0, 0.0, 0.0],
            [1.18, 0.0, 0.0],
            [0.0, 1.35, 0.0],
            [1.25, 1.10, 0.2],
        ]
    )


def _all_pairs(n_atoms: int):
    return np.array([(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)], dtype=np.int32)


def test_dense_and_tiled_lj_match_indexed_pairs():
    positions = _positions()
    cell = Cell.cubic(6.0)
    pairs = _all_pairs(positions.shape[0])
    reference = LennardJonesPotential(backend="mlx_pairs")
    reference_energy, reference_forces = reference.energy_forces(positions, cell, pairs=pairs)

    for backend in ("mlx_dense", "mlx_tiled"):
        energy, forces = LennardJonesPotential(backend=backend, tile_size=2).energy_forces(
            positions,
            cell,
        )
        mx.eval(energy, forces, reference_energy, reference_forces)
        np.testing.assert_allclose(np.asarray(energy), np.asarray(reference_energy), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(forces), np.asarray(reference_forces), rtol=1e-5)


def test_lj_pairs_backend_requires_pair_provider_for_lazy_topology():
    positions = _positions()
    topology = Topology.from_sequences(
        n_atoms=positions.shape[0],
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=2,
    )
    potential = LennardJonesPotential(topology=topology, backend="mlx_pairs")

    with pytest.raises(
        ValueError,
        match="lazy topology requires a runtime nonbonded pair provider",
    ):
        potential.energy_forces(positions)

    assert topology._nonbonded_pairs is None


def test_lj_auto_backend_requires_pair_provider_for_lazy_topology_before_dense_fallback():
    positions = _positions()
    topology = Topology.from_sequences(
        n_atoms=positions.shape[0],
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=2,
    )
    potential = LennardJonesPotential(
        topology=topology,
        backend="auto",
        memory_budget_bytes=1,
    )

    with pytest.raises(
        ValueError,
        match="lazy topology requires a runtime nonbonded pair provider",
    ):
        potential.energy_forces(positions)

    assert topology._nonbonded_pairs is None


def test_dense_and_tiled_combined_nonbonded_match_indexed_pairs():
    positions = _positions()
    cell = Cell.cubic(6.0)
    pairs = _all_pairs(positions.shape[0])
    kwargs = {
        "sigma": [1.0, 1.1, 0.95, 1.05],
        "epsilon": [1.0, 0.8, 1.2, 0.7],
        "charges": [0.5, -0.25, 0.75, -0.5],
        "cutoff": 3.0,
    }
    reference = NonbondedPotential(**kwargs, backend="mlx_pairs")
    reference_energy, reference_forces = reference.energy_forces(positions, cell, pairs=pairs)

    for backend in ("mlx_dense", "mlx_tiled"):
        energy, forces = NonbondedPotential(**kwargs, backend=backend, tile_size=2).energy_forces(
            positions,
            cell,
        )
        mx.eval(energy, forces, reference_energy, reference_forces)
        np.testing.assert_allclose(np.asarray(energy), np.asarray(reference_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(forces), np.asarray(reference_forces), rtol=1e-5)


def test_nonbonded_auto_backend_requires_pair_provider_for_lazy_topology_before_dense_fallback():
    positions = _positions()
    topology = Topology.from_sequences(
        n_atoms=positions.shape[0],
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=2,
    )
    potential = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        backend="auto",
        memory_budget_bytes=1,
    )

    with pytest.raises(
        ValueError,
        match="lazy topology requires a runtime nonbonded pair provider",
    ):
        potential.energy_forces(positions, Cell.cubic(4.0))

    assert topology._nonbonded_pairs is None


def test_periodic_neighbor_pairs_match_dense_cutoff_nonbonded():
    positions = as_mx_array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [3.0, 3.0, 3.0],
            [4.2, 3.0, 3.0],
        ]
    )
    cell = Cell.cubic(6.0)
    kwargs = {
        "sigma": [1.0, 1.05, 0.95, 1.1, 0.9],
        "epsilon": [1.0, 0.8, 1.2, 0.7, 0.6],
        "charges": [0.5, -0.25, 0.75, -0.5, -0.5],
        "cutoff": 1.6,
    }
    neighbors = build_neighbor_list(positions, cell, cutoff=kwargs["cutoff"], skin=0.0)
    dense = NonbondedPotential(**kwargs, backend="mlx_dense")
    pairs = NonbondedPotential(**kwargs, backend="mlx_pairs")

    dense_energy, dense_forces = dense.energy_forces(positions, cell)
    pair_energy, pair_forces = pairs.energy_forces(positions, cell, pairs=neighbors.pairs)

    mx.eval(dense_energy, dense_forces, pair_energy, pair_forces)
    np.testing.assert_allclose(np.asarray(pair_energy), np.asarray(dense_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_forces), np.asarray(dense_forces), rtol=1e-5)


def test_triclinic_nonbonded_displacements_use_minimum_image():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    positions = as_mx_array(
        np.array(
            [
                [0.95, 0.5, 0.5],
                [0.05, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        @ matrix
    )
    term = NonbondedPotential(
        sigma=[0.3, 0.3],
        epsilon=[1.0, 1.0],
        charges=[0.0, 0.0],
        cutoff=0.6,
        lj_shift=False,
        backend="mlx_dense",
    )
    pairs_term = NonbondedPotential(
        sigma=[0.3, 0.3],
        epsilon=[1.0, 1.0],
        charges=[0.0, 0.0],
        cutoff=0.6,
        lj_shift=False,
        backend="mlx_pairs",
    )
    pairs = as_mx_array([[0, 1]], dtype=mx.int32)

    dense_energy, dense_forces = term.energy_forces(positions, cell)
    pair_energy, pair_forces = pairs_term.energy_forces(positions, cell, pairs=pairs)
    unwrapped_energy, _ = term.energy_forces(positions, None)

    mx.eval(dense_energy, dense_forces, pair_energy, pair_forces, unwrapped_energy)
    assert float(np.asarray(dense_energy)) != pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(unwrapped_energy), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(pair_energy), np.asarray(dense_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_forces), np.asarray(dense_forces), rtol=1e-5)


def test_lazy_topology_neighbor_pairs_filter_exclusions_and_exceptions_without_dense_cache():
    positions = as_mx_array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.2, 1.1, 0.1],
        ]
    )
    cell = Cell.cubic(4.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        nonbonded_exception_pairs=[(1, 3)],
        eager_nonbonded_pair_limit=0,
    )
    neighbors = build_neighbor_list(positions, cell, cutoff=1.6, skin=0.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        exception_pairs=[(1, 3)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
        cutoff=1.6,
        lj_shift=False,
        backend="mlx_pairs",
    )

    pairs, _, _ = term._pairs_and_scales(positions, neighbors.pairs)
    energy, forces = term.energy_forces(positions, cell, pairs=neighbors.pairs)

    assert topology._nonbonded_pairs is None
    assert np.asarray(pairs).tolist() == [[0, 2], [0, 3], [1, 2], [2, 3]]
    mx.eval(energy, forces)
    assert np.all(np.isfinite(np.asarray(forces)))


def test_mlx_dense_pairs_preserve_lazy_topology_pair_semantics():
    positions = as_mx_array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.2, 1.1, 0.1],
        ]
    )
    cell = Cell.cubic(4.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        one_four_pairs=[(0, 3)],
        nonbonded_exception_pairs=[(1, 3)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.1, 1.2, 1.0],
        epsilon=[0.2, 0.25, 0.3, 0.2],
        charges=[0.25, -0.25, 0.1, -0.1],
        atom_types=["H", "C", "O", "H"],
        topology=topology,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
        exception_pairs=[(1, 3)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.35],
        nbfix_type_epsilon=[0.6],
        cutoff=2.0,
        lj_shift=False,
        backend="auto",
    )
    oracle = build_neighbor_list(positions, cell, cutoff=2.0, skin=0.0)
    mlx_pairs = build_neighbor_list(
        positions,
        cell,
        cutoff=2.0,
        skin=0.0,
        backend="mlx_dense_pairs",
    )

    oracle_energy, oracle_forces = term.energy_forces(positions, cell, pairs=oracle.pairs)
    mlx_energy, mlx_forces = term.energy_forces(positions, cell, pairs=mlx_pairs.pairs)

    mx.eval(oracle_energy, oracle_forces, mlx_energy, mlx_forces)
    assert topology._nonbonded_pairs is None
    assert mlx_pairs.fallback_reason == "mlx_argwhere_or_nonzero_unavailable"
    np.testing.assert_allclose(np.asarray(mlx_energy), np.asarray(oracle_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(mlx_forces), np.asarray(oracle_forces), rtol=1e-5)


def test_mlx_cell_blocks_preserve_lazy_topology_pair_semantics():
    positions = as_mx_array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.2, 1.1, 0.1],
        ]
    )
    cell = Cell.cubic(4.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        one_four_pairs=[(0, 3)],
        nonbonded_exception_pairs=[(1, 3)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.1, 1.2, 1.0],
        epsilon=[0.2, 0.25, 0.3, 0.2],
        charges=[0.25, -0.25, 0.1, -0.1],
        atom_types=["H", "C", "O", "H"],
        topology=topology,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.75,
        exception_pairs=[(1, 3)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.35],
        nbfix_type_epsilon=[0.6],
        cutoff=2.0,
        lj_shift=False,
        backend="auto",
    )
    oracle = build_neighbor_list(positions, cell, cutoff=2.0, skin=0.0)
    blocks = build_neighbor_list(
        positions,
        cell,
        cutoff=2.0,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=2,
    )

    oracle_energy, oracle_forces = term.energy_forces(positions, cell, pairs=oracle.pairs)
    block_energy, block_forces = term.energy_forces(
        positions,
        cell,
        pairs=blocks.interactions,
    )

    mx.eval(oracle_energy, oracle_forces, block_energy, block_forces)
    assert topology._nonbonded_pairs is None
    assert blocks.representation_kind == "blocks"
    assert blocks.compaction_backend is None
    np.testing.assert_allclose(np.asarray(block_energy), np.asarray(oracle_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(block_forces), np.asarray(oracle_forces), rtol=1e-5)


def test_nbfix_type_pair_substitution_works_on_neighbor_pairs():
    positions = as_mx_array(
        [[0.1, 0.1, 0.1], [1.4, 0.1, 0.1], [0.1, 1.5, 0.1]]
    )
    cell = Cell.cubic(5.0)
    neighbors = build_neighbor_list(positions, cell, cutoff=2.0, skin=0.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.2, 1.0],
        epsilon=[0.2, 0.3, 0.2],
        charges=[0.0, 0.0, 0.0],
        atom_types=["H", "O", "H"],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.1],
        nbfix_type_epsilon=[0.5],
        cutoff=2.0,
        lj_shift=False,
        backend="mlx_pairs",
    )

    sigma_ij, epsilon_ij = term.mixed_pair_parameters(neighbors.pairs)
    energy, forces = term.energy_forces(positions, cell, pairs=neighbors.pairs)

    np.testing.assert_allclose(np.asarray(sigma_ij), [1.1, 1.0, 1.1], atol=1e-6)
    np.testing.assert_allclose(np.asarray(epsilon_ij), [0.5, 0.2, 0.5], atol=1e-6)
    mx.eval(energy, forces)
    assert np.all(np.isfinite(np.asarray(forces)))


def test_lazy_large_periodic_nonbonded_refuses_dense_fallback_and_reports_compact_backend():
    positions = as_mx_array(
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [1.1, 1.1, 0.1]]
    )
    velocities = mx.zeros_like(positions)
    masses = mx.ones((4,), dtype=mx.float32)
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )

    with pytest.raises(ValueError, match="dense/tiled all-pairs fallback is refused"):
        simulate_nvt(
            positions,
            velocities,
            masses=masses,
            cell=cell,
            force_terms=(term,),
            config=SimulationConfig(steps=0),
        )

    manager = NeighborListManager(cell, cutoff=1.6, skin=0.2)
    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=(term,),
        neighbor_manager=manager,
        config=SimulationConfig(steps=1, diagnostic_interval=1),
    )

    assert result.nonbonded_report["backend"] == "mlx_dense_pairs"
    assert result.nonbonded_report["representation_kind"] == "pairs"
    assert result.nonbonded_report["pair_count"] == int(result.pair_count[-1])
    assert result.nonbonded_report["cutoff"] == 1.6
    assert result.nonbonded_report["skin"] == 0.2
    assert result.nonbonded_report["rebuild_count"] == int(result.rebuild_count[-1])
    assert (
        result.nonbonded_report["estimated_pair_memory_bytes"]
        == manager.neighbor_list.estimated_pair_bytes
    )
    assert (
        result.nonbonded_report["estimated_cell_list_memory_bytes"]
        == manager.neighbor_list.estimated_cell_list_bytes
    )
    assert result.nonbonded_report["neighbor_rebuild_wall_seconds"] >= 0.0
    assert result.nonbonded_report["force_evaluation_wall_seconds"] >= 0.0


def test_auto_neighbor_backend_selects_mlx_cell_pairs_above_dense_limit():
    positions = as_mx_array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.1, 1.1, 0.1],
        ]
    )
    velocities = mx.zeros_like(positions)
    masses = mx.ones((4,), dtype=mx.float32)
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )
    manager = NeighborListManager(cell, cutoff=1.6, skin=0.2, max_mlx_dense_atoms=3)

    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=(term,),
        neighbor_manager=manager,
        config=SimulationConfig(steps=1, diagnostic_interval=1),
    )

    assert result.nonbonded_report["backend"] == "mlx_cell_pairs"
    assert result.nonbonded_report["representation_kind"] == "pairs"
    assert result.nonbonded_report["candidate_count"] is not None
    assert result.nonbonded_report["compaction_backend"] == "cpu_argwhere"
    assert result.nonbonded_report["fallback_reason"] is None
    assert result.nonbonded_report["neighbor_update_wall_seconds"] >= 0.0
    assert result.nonbonded_report["neighbor_rebuild_wall_seconds"] >= 0.0
    assert result.nonbonded_report["force_evaluation_wall_seconds"] >= 0.0


def test_explicit_block_neighbor_backend_reports_block_representation():
    positions = as_mx_array(
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [1.1, 1.1, 0.1]]
    )
    velocities = mx.zeros_like(positions)
    masses = mx.ones((4,), dtype=mx.float32)
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )
    manager = NeighborListManager(
        cell,
        cutoff=1.6,
        skin=0.2,
        backend="mlx_cell_blocks",
        block_size=2,
    )

    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=(term,),
        neighbor_manager=manager,
        config=SimulationConfig(steps=1, diagnostic_interval=1),
    )

    assert result.nonbonded_report["backend"] == "mlx_cell_blocks"
    assert result.nonbonded_report["representation_kind"] == "blocks"
    assert result.nonbonded_report["pair_count"] == int(result.pair_count[-1])
    assert result.nonbonded_report["compact_pair_count"] <= result.nonbonded_report["pair_count"]
    assert result.nonbonded_report["candidate_count"] == int(result.pair_count[-1])
    assert result.nonbonded_report["candidate_waste_count"] is not None
    assert result.nonbonded_report["candidate_waste_fraction"] is not None
    assert result.nonbonded_report["compaction_backend"] is None
    assert result.nonbonded_report["fallback_reason"] is None
    assert result.nonbonded_report["force_evaluation_wall_seconds"] >= 0.0


def test_mlx_dense_pairs_neighbor_manager_drives_lazy_nonbonded_md_report():
    positions = as_mx_array(
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [1.1, 1.1, 0.1]]
    )
    velocities = mx.zeros_like(positions)
    masses = mx.ones((4,), dtype=mx.float32)
    cell = Cell.cubic(5.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.2, 0.2, 0.2, 0.2],
        charges=[0.0, 0.0, 0.0, 0.0],
        topology=topology,
        cutoff=1.6,
        backend="auto",
    )
    manager = NeighborListManager(cell, cutoff=1.6, skin=0.2, backend="mlx_dense_pairs")

    result = simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=(term,),
        neighbor_manager=manager,
        config=SimulationConfig(steps=1, diagnostic_interval=1),
    )

    assert result.nonbonded_report["backend"] == "mlx_dense_pairs"
    assert result.nonbonded_report["representation_kind"] == "pairs"
    assert result.nonbonded_report["pair_count"] == int(result.pair_count[-1])
    assert result.nonbonded_report["compact_pair_count"] == int(result.pair_count[-1])
    assert result.nonbonded_report["candidate_count"] == 6
    assert result.nonbonded_report["candidate_waste_count"] == 6 - int(result.pair_count[-1])
    assert result.nonbonded_report["candidate_waste_fraction"] is not None
    assert result.nonbonded_report["estimated_candidate_memory_bytes"] > 0
    assert result.nonbonded_report["compaction_backend"] == "cpu_argwhere"
    assert result.nonbonded_report["fallback_reason"] == "mlx_argwhere_or_nonzero_unavailable"


def test_dense_nonbonded_respects_topology_exclusions_and_one_four_scaling():
    positions = _positions()
    cell = Cell.cubic(6.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        dihedrals=[(0, 1, 2, 3)],
        one_four_pairs=[(0, 3)],
    )
    kwargs = {
        "sigma": [1.0, 1.0, 1.0, 1.0],
        "epsilon": [1.0, 1.0, 1.0, 1.0],
        "charges": [1.0, -1.0, 0.5, -0.5],
        "cutoff": 4.0,
        "topology": topology,
        "lj_one_four_scale": 0.25,
        "coulomb_one_four_scale": 0.5,
    }
    reference = NonbondedPotential(**kwargs, backend="mlx_pairs")
    reference_energy, reference_forces = reference.energy_forces(positions, cell)

    dense = NonbondedPotential(**kwargs, backend="mlx_dense")
    dense_energy, dense_forces = dense.energy_forces(positions, cell)
    mx.eval(reference_energy, reference_forces, dense_energy, dense_forces)
    np.testing.assert_allclose(np.asarray(dense_energy), np.asarray(reference_energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dense_forces), np.asarray(reference_forces), rtol=1e-5)


def test_backend_validation_and_memory_policy():
    with pytest.raises(ValueError, match="unknown nonbonded backend"):
        NonbondedExecutionConfig(backend="bad")  # type: ignore[arg-type]

    estimated = estimate_dense_nonbonded_bytes(64, components="combined")
    assert estimated > 0
    assert (
        choose_nonbonded_backend(
            requested="auto",
            n_atoms=64,
            pairs_provided=False,
            estimated_dense_bytes=estimated,
            memory_budget_bytes=1,
        )
        == "mlx_tiled"
    )
    with pytest.raises(MemoryError, match="exceeds memory budget"):
        NonbondedPotential(
            sigma=[1.0, 1.0],
            epsilon=[1.0, 1.0],
            charges=[0.0, 0.0],
            backend="mlx_dense",
            memory_budget_bytes=1,
        ).energy_forces(as_mx_array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]), Cell.cubic(4.0))


def test_full_loop_auto_policy_selects_dynamic_neighbor_from_evidence():
    assert DEFAULT_FULL_LOOP_DENSE_THRESHOLD == 1536

    policy = choose_full_loop_nonbonded_policy(
        mode="auto",
        n_atoms=512,
        cutoff=2.5,
        cell_provided=True,
    )
    assert policy.use_neighbor_list is False
    assert policy.potential_backend == "auto"
    assert policy.selected_backend == "mlx_dense"
    assert policy.neighbor_backend is None
    assert policy.selected_policy == "auto:evidence_dense_below_threshold; dense_threshold:1536"
    assert policy.evidence == "s3_small_system_dense_threshold"
    assert policy.blocker is None

    large = choose_full_loop_nonbonded_policy(
        mode="auto",
        n_atoms=2000,
        cutoff=2.5,
        cell_provided=True,
    )
    assert large.use_neighbor_list is True
    assert large.potential_backend == "mlx_pairs"
    assert large.selected_backend == "dynamic-neighbor+mlx_cell_pairs"
    assert large.neighbor_backend == "mlx_cell_pairs"
    assert large.selected_policy == "auto:evidence_dynamic_neighbor; dense_threshold:1536"
    assert large.evidence == "s1_s2_s3_full_loop_policy_evidence"
    assert large.blocker is None

    fallback = choose_full_loop_nonbonded_policy(
        mode="auto",
        n_atoms=512,
        cutoff=None,
        cell_provided=False,
    )
    assert fallback.use_neighbor_list is False
    assert fallback.selected_backend == "mlx_dense"
    assert fallback.blocker == "dynamic_neighbor_requires_periodic_cutoff"


@pytest.mark.parametrize("bad_cutoff", [float("nan"), float("inf"), 0.0, -1.0])
def test_full_loop_policy_rejects_invalid_dynamic_neighbor_cutoff(bad_cutoff):
    with pytest.raises(ValueError, match="finite positive cutoff"):
        choose_full_loop_nonbonded_policy(
            mode="dynamic-neighbor",
            n_atoms=512,
            cutoff=bad_cutoff,
            cell_provided=True,
        )

    fallback = choose_full_loop_nonbonded_policy(
        mode="auto",
        n_atoms=512,
        cutoff=bad_cutoff,
        cell_provided=True,
    )
    assert fallback.use_neighbor_list is False
    assert fallback.selected_backend == "mlx_dense"
    assert fallback.blocker == "dynamic_neighbor_requires_finite_positive_cutoff"


def test_electrostatics_mode_contract():
    assert normalize_nonbonded_electrostatics("direct-cutoff") == "cutoff"
    assert normalize_nonbonded_electrostatics("ewald") == "ewald_reference"
    assert normalize_nonbonded_electrostatics("pme_ewald_periodic_electrostatics") == "pme"
    assert (
        NonbondedExecutionConfig(
            electrostatics="short_range_electrostatics_prototype",
        ).electrostatics
        == "cutoff"
    )

    with pytest.raises(ValueError, match="unknown electrostatics mode"):
        normalize_nonbonded_electrostatics("reaction_field")
    assert validate_nonbonded_electrostatics("pme") == "pme"
    assert NonbondedExecutionConfig(electrostatics="pme").electrostatics == "pme"
    with pytest.raises(ValueError, match="pme_config"):
        NonbondedPotential(
            sigma=[1.0, 1.0],
            epsilon=[0.0, 0.0],
            charges=[0.0, 0.0],
            electrostatics="pme",
        )
    assert (
        NonbondedPotential(
            sigma=[1.0, 1.0],
            epsilon=[0.0, 0.0],
            charges=[0.0, 0.0],
            electrostatics="pme",
            pme_config=PMEConfig(mesh_shape=(8, 8, 8)),
        ).electrostatics
        == "pme"
    )


def test_ewald_reference_mode_requires_periodic_cell():
    potential = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.0, 0.0],
        charges=[1.0, -1.0],
        electrostatics="ewald_reference",
    )
    assert potential.electrostatics == "ewald_reference"
    with pytest.raises(ValueError, match="requires a periodic cell"):
        potential.energy_forces(as_mx_array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))


def test_small_degenerate_cases_are_finite():
    cell = Cell.cubic(4.0)
    one = as_mx_array([[0.0, 0.0, 0.0]])
    energy, forces = LennardJonesPotential(backend="mlx_dense").energy_forces(one, cell)
    mx.eval(energy, forces)
    assert float(energy) == 0.0
    np.testing.assert_allclose(np.asarray(forces), np.zeros((1, 3)))

    two = as_mx_array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    potential = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[0.0, 0.0],
        charges=[0.0, 0.0],
        cutoff=1.0,
        backend="mlx_dense",
    )
    energy, forces = potential.energy_forces(two, cell)
    mx.eval(energy, forces)
    assert float(energy) == 0.0
    np.testing.assert_allclose(np.asarray(forces), np.zeros((2, 3)))


def test_md_acceleration_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "md_acceleration.csv"

    md_acceleration.main(
        [
            "--sizes",
            "16",
            "--backends",
            "mlx_dense,mlx_tiled",
            "--evaluations",
            "1",
            "--tile-size",
            "8",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 2
    assert {case["backend"] for case in payload["cases"]} == {"mlx_dense", "mlx_tiled"}
    assert all(case["estimated_dense_bytes"] > 0 for case in payload["cases"])
    assert all("rebuild_count" in case for case in payload["cases"])
    assert all("estimated_pair_bytes" in case for case in payload["cases"])
    assert all("neighbor_rebuild_ms_per_eval" in case for case in payload["cases"])
    assert all("force_eval_ms_per_eval" in case for case in payload["cases"])
    assert csv_path.read_text().startswith("backend,particles")


def test_md_performance_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "md_performance.csv"

    md_performance.main(
        [
            "--sizes",
            "16",
            "--steps",
            "2",
            "--sample-interval",
            "1",
            "--diagnostic-interval",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 1
    case = payload["cases"][0]
    assert case["case"] == "synthetic_lj"
    assert case["particles"] == 16
    assert case["steps_per_s"] > 0.0
    assert case["estimated_dense_bytes"] > 0
    assert case["finite"] is True
    assert csv_path.read_text().startswith("case,mode,particles")


def test_md_performance_neighbor_policy_knobs_are_reported():
    payload = md_performance.build_payload(
        sizes=(64,),
        steps=1,
        dt=0.002,
        mode="dynamic-neighbor",
        dense_threshold=1536,
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=20,
        neighbor_skin=1.0,
    )

    case = payload["cases"][0]
    assert case["backend"] == "dynamic-neighbor+mlx_cell_pairs"
    assert case["finite"] is True
    assert payload["config"]["neighbor_check_interval"] == 20
    assert payload["config"]["neighbor_skin"] == 1.0


def test_md_performance_include_large_adds_documented_s5_large_case_once():
    assert md_performance._with_large_sizes((16,)) == (16, 5000)
    assert md_performance._with_large_sizes((2000, 5000)) == (2000, 5000)


def test_md_performance_auto_default_routes_2000_to_dynamic_neighbor():
    result = md_performance.run_synthetic_case(
        particles=2000,
        steps=1,
        dt=0.002,
        mode="auto",
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=1,
    )

    assert result.backend == "dynamic-neighbor+mlx_cell_pairs"
    assert result.selected_policy == "auto:evidence_dynamic_neighbor; dense_threshold:1536"
    assert result.selected_representation == "pairs"
    assert result.compaction_backend == "cpu_argwhere"
    assert result.neighbor_candidate_count is not None
    assert result.candidate_waste_count is not None
    assert result.candidate_waste_fraction is not None
    assert result.final_pair_count < 2000 * 1999 // 2
    assert result.rebuild_count >= 1
    assert result.finite is True


def test_md_performance_auto_threshold_override_can_keep_dense_for_small_system():
    result = md_performance.run_synthetic_case(
        particles=128,
        steps=1,
        dt=0.002,
        mode="auto",
        dense_threshold=256,
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=1,
    )

    assert result.backend == "mlx_dense"
    assert result.selected_policy == "auto:evidence_dense_below_threshold; dense_threshold:256"
    assert result.neighbor_backend is None
    assert result.candidate_waste_count == 0
    assert result.rebuild_count == 0
    assert result.finite is True


def test_md_performance_auto_default_keeps_512_dense():
    result = md_performance.run_synthetic_case(
        particles=512,
        steps=1,
        dt=0.002,
        mode="auto",
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=1,
    )

    assert result.backend == "mlx_dense"
    assert result.selected_policy == "auto:evidence_dense_below_threshold; dense_threshold:1536"
    assert result.selected_representation == "pairs"
    assert result.neighbor_backend is None
    assert result.rebuild_count == 0
    assert result.finite is True


def test_md_performance_dynamic_neighbor_mode_smoke():
    result = md_performance.run_synthetic_case(
        particles=128,
        steps=1,
        dt=0.002,
        mode="dynamic-neighbor",
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=1,
    )

    assert result.backend == "dynamic-neighbor+mlx_cell_pairs"
    assert result.final_pair_count > 0
    assert result.rebuild_count >= 1
    assert result.finite is True


def test_md_performance_batched_replicas_smoke():
    payload = md_performance.build_payload(
        sizes=(8,),
        steps=2,
        dt=0.002,
        mode="auto",
        dense_threshold=2048,
        sample_interval=1,
        diagnostic_interval=1,
        neighbor_check_interval=1,
        replicas=2,
    )

    case = payload["cases"][0]
    assert case["case"] == "synthetic_lj_replicas"
    assert case["replicas"] == 2
    assert case["backend"] == "batched_mlx_dense"
    assert case["finite"] is True
