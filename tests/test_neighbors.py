import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import LennardJonesPotential, simulate
from mlx_atomistic.neighbors import NeighborListManager, build_neighbor_list
from mlx_atomistic.nonbonded import estimate_dense_nonbonded_bytes


def _block_pair_set(neighbors):
    blocks = neighbors.blocks
    assert blocks is not None
    valid = np.asarray(blocks.valid_mask).reshape(-1)
    left = np.asarray(blocks.left).reshape(-1)[valid]
    right = np.asarray(blocks.right).reshape(-1)[valid]
    return {tuple(pair) for pair in np.stack((left, right), axis=1).tolist()}


def _neighbor_pair_set(neighbors):
    if neighbors.blocks is not None:
        return _block_pair_set(neighbors)
    return {tuple(pair) for pair in np.asarray(neighbors.pairs).tolist()}


def test_neighbor_list_has_unique_expected_pairs():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    neighbors = build_neighbor_list(positions, Cell.cubic(8.0), cutoff=1.5, skin=0.0)
    pairs = np.array(neighbors.pairs)

    assert pairs.tolist() == [[0, 1]]
    assert neighbors.backend == "periodic_cell_list"
    assert neighbors.representation_kind == "pairs"
    assert neighbors.candidate_count is not None
    assert neighbors.candidate_count >= neighbors.compact_pair_count
    assert neighbors.compact_pair_count == neighbors.pair_count
    assert neighbors.candidate_waste_count is not None
    assert neighbors.candidate_waste_fraction is not None
    assert neighbors.compaction_backend == "cpu_distance_filter"
    assert neighbors.fallback_reason is None
    assert neighbors.estimated_pair_bytes == 8
    assert neighbors.estimated_compact_pair_bytes == 8
    assert neighbors.estimated_candidate_bytes >= 0
    assert neighbors.estimated_cell_list_bytes > 0


def test_neighbor_backend_validation_rejects_unknown_backend():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(4.0)

    with pytest.raises(ValueError, match="unknown neighbor backend"):
        build_neighbor_list(positions, cell, cutoff=1.5, backend="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown neighbor backend"):
        NeighborListManager(cell, cutoff=1.5, backend="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown neighbor check backend"):
        NeighborListManager(  # type: ignore[arg-type]
            cell,
            cutoff=1.5,
            displacement_check_backend="bad",
        )


def test_mlx_dense_pairs_matches_periodic_cell_list_pairs_ignoring_order():
    positions = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.35, 0.05, 0.05],
            [1.85, 0.05, 0.05],
            [1.85, 1.85, 1.85],
            [1.1, 1.1, 0.05],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(2.0)

    oracle = build_neighbor_list(positions, cell, cutoff=0.45, skin=0.0)
    candidate = build_neighbor_list(
        positions,
        cell,
        cutoff=0.45,
        skin=0.0,
        backend="mlx_dense_pairs",
    )

    assert {tuple(pair) for pair in np.asarray(candidate.pairs).tolist()} == {
        tuple(pair) for pair in np.asarray(oracle.pairs).tolist()
    }
    assert candidate.backend == "mlx_dense_pairs"
    assert candidate.representation_kind == "pairs"
    assert candidate.candidate_count == positions.shape[0] * (positions.shape[0] - 1) // 2
    assert candidate.compact_pair_count == candidate.pair_count
    assert candidate.candidate_waste_count == candidate.candidate_count - candidate.pair_count
    assert candidate.candidate_waste_fraction is not None
    assert candidate.candidate_waste_fraction > 0.0
    assert candidate.estimated_pair_bytes == candidate.pair_count * 8
    assert candidate.estimated_compact_pair_bytes == candidate.compact_pair_count * 8
    assert candidate.estimated_candidate_bytes > 0
    assert candidate.estimated_cell_list_bytes == 0
    assert candidate.compaction_backend == "cpu_argwhere"
    assert candidate.fallback_reason == "mlx_argwhere_or_nonzero_unavailable"


def test_mlx_cell_pairs_matches_periodic_cell_list_pairs_ignoring_order():
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.0, 8.0, size=(96, 3)).astype(np.float32)
    cell = Cell.cubic(8.0)

    oracle = build_neighbor_list(positions, cell, cutoff=1.4, skin=0.2)
    candidate = build_neighbor_list(
        positions,
        cell,
        cutoff=1.4,
        skin=0.2,
        backend="mlx_cell_pairs",
    )

    assert {tuple(pair) for pair in np.asarray(candidate.pairs).tolist()} == {
        tuple(pair) for pair in np.asarray(oracle.pairs).tolist()
    }
    assert candidate.backend == "mlx_cell_pairs"
    assert candidate.representation_kind == "pairs"
    assert candidate.candidate_count is not None
    assert candidate.candidate_count < positions.shape[0] * (positions.shape[0] - 1) // 2
    assert candidate.compact_pair_count == candidate.pair_count
    assert candidate.candidate_waste_count == candidate.candidate_count - candidate.pair_count
    assert candidate.candidate_waste_fraction is not None
    assert candidate.estimated_pair_bytes == candidate.pair_count * 8
    assert candidate.estimated_candidate_bytes > 0
    assert candidate.estimated_cell_list_bytes > 0
    assert candidate.compaction_backend == "cpu_argwhere"
    assert candidate.fallback_reason is None


def test_mlx_cell_blocks_cover_periodic_cell_list_pairs_without_compaction():
    rng = np.random.default_rng(17)
    positions = rng.uniform(0.0, 8.0, size=(96, 3)).astype(np.float32)
    cell = Cell.cubic(8.0)

    oracle = build_neighbor_list(positions, cell, cutoff=1.4, skin=0.2)
    candidate = build_neighbor_list(
        positions,
        cell,
        cutoff=1.4,
        skin=0.2,
        backend="mlx_cell_blocks",
        block_size=32,
    )

    assert {tuple(pair) for pair in np.asarray(oracle.pairs).tolist()} <= _block_pair_set(
        candidate
    )
    assert candidate.backend == "mlx_cell_blocks"
    assert candidate.representation_kind == "blocks"
    assert candidate.blocks is not None
    assert candidate.interactions is candidate.blocks
    assert candidate.pairs.shape == (0, 2)
    assert candidate.pair_count >= oracle.pair_count
    assert candidate.compact_pair_count == oracle.pair_count
    assert candidate.candidate_count == candidate.pair_count
    assert candidate.candidate_waste_count == candidate.candidate_count - oracle.pair_count
    assert candidate.candidate_waste_fraction is not None
    assert candidate.estimated_pair_bytes == candidate.blocks.estimated_bytes
    assert candidate.estimated_compact_pair_bytes == oracle.pair_count * 8
    assert candidate.estimated_candidate_bytes == candidate.blocks.estimated_bytes
    assert candidate.compaction_backend is None
    assert candidate.fallback_reason is None


def test_staged_neighbor_metadata_reports_candidate_waste_for_all_cell_backends():
    positions = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.6, 0.1, 0.1],
            [2.5, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(4.0)

    cpu = build_neighbor_list(
        positions,
        cell,
        cutoff=1.1,
        skin=0.0,
        backend="periodic_cell_list",
    )
    mlx_pairs = build_neighbor_list(
        positions,
        cell,
        cutoff=1.1,
        skin=0.0,
        backend="mlx_cell_pairs",
    )
    mlx_blocks = build_neighbor_list(
        positions,
        cell,
        cutoff=1.1,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=2,
    )

    expected_pairs = {(0, 1)}
    assert _neighbor_pair_set(cpu) == expected_pairs
    assert _neighbor_pair_set(mlx_pairs) == expected_pairs
    assert expected_pairs <= _block_pair_set(mlx_blocks)

    for neighbors in (cpu, mlx_pairs, mlx_blocks):
        assert neighbors.candidate_count == 3
        assert neighbors.compact_pair_count == 1
        assert neighbors.candidate_waste_count == 2
        assert neighbors.candidate_waste_fraction == pytest.approx(2.0 / 3.0)
        assert neighbors.estimated_candidate_bytes > 0
        assert neighbors.estimated_compact_pair_bytes == 8
        assert neighbors.fallback_reason is None

    assert cpu.backend == "periodic_cell_list"
    assert cpu.compaction_backend == "cpu_distance_filter"
    assert mlx_pairs.backend == "mlx_cell_pairs"
    assert mlx_pairs.compaction_backend == "cpu_argwhere"
    assert mlx_blocks.backend == "mlx_cell_blocks"
    assert mlx_blocks.compaction_backend is None
    assert mlx_blocks.pair_count == 3


def test_mlx_dense_pairs_fails_closed_for_large_candidate_sets():
    positions = np.zeros((5, 3), dtype=np.float32)
    cell = Cell.cubic(4.0)

    with pytest.raises(ValueError, match="fallback_backend=periodic_cell_list"):
        build_neighbor_list(
            positions,
            cell,
            cutoff=1.5,
            backend="mlx_dense_pairs",
            max_mlx_dense_atoms=4,
        )


def test_triclinic_dense_neighbor_pairs_use_minimum_image():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    positions = np.array(
        [
            [0.95, 0.5, 0.5],
            [0.05, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ) @ matrix

    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=0.6,
        skin=0.0,
        backend="mlx_dense_pairs",
    )

    assert _neighbor_pair_set(neighbors) == {(0, 1)}


@pytest.mark.parametrize(
    "backend",
    ["periodic_cell_list", "mlx_cell_pairs", "mlx_cell_blocks"],
)
def test_triclinic_compact_neighbor_backends_fail_closed(backend):
    cell = Cell.triclinic(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ]
    )
    positions = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="triclinic"):
        build_neighbor_list(
            positions,
            cell,
            cutoff=0.6,
            skin=0.0,
            backend=backend,  # type: ignore[arg-type]
        )


def test_triclinic_numpy_rebuild_check_uses_minimum_image():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    start = np.array([[0.95, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32) @ matrix
    wrapped_small_move = np.array(
        [[0.05, 0.5, 0.5], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    ) @ matrix
    manager = NeighborListManager(
        cell,
        cutoff=0.6,
        skin=1.0,
        backend="mlx_dense_pairs",
        displacement_check_backend="numpy",
    )

    initial = manager.update(start)
    still_current = manager.update(wrapped_small_move)

    assert still_current is initial
    assert manager.rebuild_count == 1
    assert manager.last_max_displacement < manager.rebuild_threshold


def test_neighbor_list_manager_selects_mlx_dense_pairs_backend():
    positions = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.35, 0.05, 0.05],
            [1.85, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(
        Cell.cubic(2.0),
        cutoff=0.45,
        skin=0.0,
        backend="mlx_dense_pairs",
    )

    neighbors = manager.update(positions)

    assert neighbors.backend == "mlx_dense_pairs"
    assert neighbors.representation_kind == "pairs"
    assert manager.rebuild_count == 1


def test_neighbor_list_manager_auto_selects_mlx_cell_pairs_above_dense_limit():
    rng = np.random.default_rng(9)
    positions = rng.uniform(0.0, 8.0, size=(16, 3)).astype(np.float32)
    cell = Cell.cubic(8.0)
    manager = NeighborListManager(
        cell,
        cutoff=1.5,
        skin=0.2,
        max_mlx_dense_atoms=8,
    )

    neighbors = manager.update(positions)
    oracle = build_neighbor_list(
        positions,
        cell,
        cutoff=1.5,
        skin=0.2,
        backend="periodic_cell_list",
    )

    assert neighbors.backend == "mlx_cell_pairs"
    assert neighbors.representation_kind == "pairs"
    assert neighbors.compaction_backend == "cpu_argwhere"
    assert {tuple(pair) for pair in np.asarray(oracle.pairs).tolist()} <= {
        tuple(pair) for pair in np.asarray(neighbors.pairs).tolist()
    }
    assert manager.rebuild_count == 1


def test_default_backend_switch_preserves_lj_physics():
    """Regression lock for the production neighbor default.

    Above the dense limit the default moved from ``mlx_cell_blocks`` (padded
    candidate blocks) to ``mlx_cell_pairs`` (host-compacted real pairs) for
    throughput. The two representations are the SAME physics -- masked
    candidates contribute zero -- so the Lennard-Jones energy and forces must
    agree. This locks "switching the neighbor backend is an optimization, not a
    physics change". N=2048 (a full FCC box above the 1536 dense threshold)
    exercises the production regime where the switch takes effect.
    """
    positions, cell = fcc_lattice(2048, density=0.8)
    pos_np = np.asarray(positions, dtype=np.float32)
    pos = mx.array(pos_np)
    lj = LennardJonesPotential(cutoff=2.5)

    nl_pairs = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_pairs"
    )
    nl_blocks = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_blocks"
    )
    assert nl_pairs.backend == "mlx_cell_pairs"
    assert nl_blocks.backend == "mlx_cell_blocks"

    e_pairs, f_pairs = lj.energy_forces(pos, cell, pairs=nl_pairs.interactions)
    e_blocks, f_blocks = lj.energy_forces(pos, cell, pairs=nl_blocks.interactions)
    mx.eval(e_pairs, f_pairs, e_blocks, f_blocks)

    assert abs(float(e_pairs) - float(e_blocks)) < 1e-2
    assert float(mx.max(mx.abs(f_pairs - f_blocks))) < 1e-3


def test_unsorted_pairs_are_physics_neutral_and_default_in_md_loop():
    """Lock: dropping the per-rebuild pair sort is an optimization, not a physics change.

    The MD-loop neighbor manager defaults to ``sort_pairs=False`` because the
    per-rebuild ``np.lexsort`` of the compacted pair list dominates large-system
    rebuilds (~77% of a 50k rebuild on M5 Max) while buying nothing: MLX
    scatter-add is insensitive to pair order. Sorted and unsorted lists must
    therefore hold the SAME pairs and yield identical Lennard-Jones energy and
    forces (differences are summation-order ULPs). The unsorted list is still
    deterministic (same positions -> same array), so reproducibility is preserved.
    """
    positions, cell = fcc_lattice(2048, density=0.8)
    pos_np = np.asarray(positions, dtype=np.float32)
    pos = mx.array(pos_np)
    lj = LennardJonesPotential(cutoff=2.5)

    # The MD loop must not pay for canonical pair ordering by default.
    assert NeighborListManager(cell, cutoff=2.5).sort_pairs is False

    sorted_nl = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_pairs", sort_pairs=True
    )
    unsorted_nl = build_neighbor_list(
        pos_np, cell, cutoff=2.5, skin=0.4, backend="mlx_cell_pairs", sort_pairs=False
    )
    assert _neighbor_pair_set(sorted_nl) == _neighbor_pair_set(unsorted_nl)

    e_sorted, f_sorted = lj.energy_forces(pos, cell, pairs=sorted_nl.interactions)
    e_unsorted, f_unsorted = lj.energy_forces(pos, cell, pairs=unsorted_nl.interactions)
    mx.eval(e_sorted, f_sorted, e_unsorted, f_unsorted)

    # Identical pair sets => identical physics; residuals are float32 summation
    # order from MLX's (atomically non-deterministic) scatter-add. The band is
    # relative: at this energy magnitude (~1e4) an absolute tolerance is
    # backend-fragile, where a few-ULP reorder is ~5e-2 absolute but ~5e-6
    # relative (mlx-cpu reorders more than the Metal build).
    assert abs(float(e_sorted) - float(e_unsorted)) < 1e-5 * abs(float(e_sorted))
    assert float(mx.max(mx.abs(f_sorted - f_unsorted))) < 1e-3


def test_neighbor_list_is_deterministic_when_periodic_offsets_alias():
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

    first = build_neighbor_list(positions, cell, cutoff=0.45, skin=0.0)
    second = build_neighbor_list(positions, cell, cutoff=0.45, skin=0.0)

    expected = [[0, 1], [0, 2], [0, 3], [2, 3]]
    assert np.array(first.pairs).tolist() == expected
    assert np.array(second.pairs).tolist() == expected
    assert first.pair_count == 4


def test_neighbor_list_lj_matches_all_pairs():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5)
    neighbors = build_neighbor_list(positions, cell, cutoff=2.5, skin=0.0)

    dense_energy, dense_forces = potential.energy_forces(positions, cell)
    pair_energy, pair_forces = potential.energy_forces(positions, cell, pairs=neighbors.pairs)

    np.testing.assert_allclose(np.array(pair_energy), np.array(dense_energy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(pair_forces), np.array(dense_forces), rtol=1e-5, atol=1e-5)


def test_neighbor_block_lj_matches_all_pairs():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5, backend="mlx_pairs")
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=2.5,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=3,
    )

    dense_energy, dense_forces = potential.energy_forces(positions, cell)
    block_energy, block_forces = potential.energy_forces(
        positions,
        cell,
        pairs=neighbors.interactions,
    )

    np.testing.assert_allclose(np.array(block_energy), np.array(dense_energy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(block_forces), np.array(dense_forces), rtol=1e-5, atol=1e-5)


def test_neighbor_list_large_periodic_fixture_uses_compact_pair_storage():
    grid = np.indices((10, 10, 10), dtype=np.float32).reshape(3, -1).T
    positions = 1.2 * grid
    cell = Cell.cubic(12.0)

    neighbors = build_neighbor_list(positions, cell, cutoff=1.25, skin=0.0)

    assert neighbors.pair_count == 3000
    assert tuple(neighbors.stats.n_cells) == (9, 9, 9)
    assert neighbors.backend == "periodic_cell_list"
    assert neighbors.estimated_pair_bytes == neighbors.pair_count * 8
    assert neighbors.estimated_pair_bytes < estimate_dense_nonbonded_bytes(
        positions.shape[0],
        components="lj",
    )


def test_neighbor_list_manager_rebuild_policy_is_deterministic():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(Cell.cubic(6.0), cutoff=1.5, skin=0.4, check_interval=2)

    initial = manager.update(positions)
    small_move = positions + np.array([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    still_current = manager.update(small_move)
    checked_current = manager.update(small_move)
    large_move = positions + np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    skipped_check = manager.update(large_move)
    rebuilt = manager.update(large_move)

    assert initial is still_current
    assert initial is checked_current
    assert initial is skipped_check
    assert rebuilt is not initial
    assert manager.rebuild_count == 2
    assert manager.last_max_displacement == 0.0
    np.testing.assert_allclose(np.asarray(manager.reference_positions), large_move)


def test_neighbor_list_manager_mlx_scalar_rebuild_policy_matches_numpy():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )

    def exercise(displacement_check_backend: str) -> tuple[int, int, float]:
        manager = NeighborListManager(
            Cell.cubic(6.0),
            cutoff=1.5,
            skin=0.4,
            check_interval=2,
            displacement_check_backend=displacement_check_backend,  # type: ignore[arg-type]
        )

        def payload(array):
            return as_mx_array(array) if displacement_check_backend == "mlx_scalar" else array

        initial = manager.update(payload(positions))
        small_move = positions + np.array(
            [[0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        still_current = manager.update(payload(small_move))
        checked_current = manager.update(payload(small_move))
        large_move = positions + np.array(
            [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        skipped_check = manager.update(payload(large_move))
        rebuilt = manager.update(payload(large_move))

        assert initial is still_current
        assert initial is checked_current
        assert initial is skipped_check
        assert rebuilt is not initial
        return manager.rebuild_count, rebuilt.pair_count, manager.last_max_displacement

    assert exercise("mlx_scalar") == exercise("numpy")


def test_neighbor_list_manager_rejects_non_finite_positions_after_valid_build():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(Cell.cubic(6.0), cutoff=1.5, skin=0.4, check_interval=2)
    initial = manager.update(positions)
    invalid = positions.copy()
    invalid[0, 0] = np.nan

    with pytest.raises(ValueError, match="positions must be finite"):
        manager.update(invalid)

    assert manager.neighbor_list is initial
    assert manager.rebuild_count == 1


def test_neighbor_list_manager_mlx_scalar_rejects_non_finite_positions_when_checked():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    manager = NeighborListManager(
        Cell.cubic(6.0),
        cutoff=1.5,
        skin=0.4,
        displacement_check_backend="mlx_scalar",
    )
    initial = manager.update(as_mx_array(positions))
    invalid = positions.copy()
    invalid[0, 0] = np.nan

    with pytest.raises(ValueError, match="positions must be finite"):
        manager.update(as_mx_array(invalid))

    assert manager.neighbor_list is initial
    assert manager.rebuild_count == 1


def test_neighbor_list_manager_mlx_scalar_rejects_reference_shape_mismatch():
    positions = as_mx_array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ]
    )
    manager = NeighborListManager(
        Cell.cubic(6.0),
        cutoff=1.5,
        skin=0.4,
        displacement_check_backend="mlx_scalar",
    )
    manager.update(positions)

    with pytest.raises(ValueError, match="positions must match"):
        manager.update(positions[:2])


def test_tuned_mlx_scalar_neighbor_policy_covers_cutoff_pairs_during_motion():
    positions, cell = fcc_lattice(512, density=0.8)
    velocities = thermal_velocities(512, temperature=1.0, seed=13)
    manager = NeighborListManager(
        cell,
        cutoff=2.5,
        skin=1.0,
        check_interval=30,
        backend="mlx_dense_pairs",
        displacement_check_backend="mlx_scalar",
    )

    for step in range(201):
        neighbors = manager.update(positions)
        if step % 20 == 0 or step == 200:
            oracle = build_neighbor_list(
                positions,
                cell,
                cutoff=2.5,
                skin=0.0,
                backend="mlx_dense_pairs",
            )
            assert _neighbor_pair_set(oracle) <= _neighbor_pair_set(neighbors)
        positions = cell.wrap(positions + 0.002 * velocities)
        mx.eval(positions)


def test_neighbor_list_simulation_runs():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
            [2.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    cell = Cell.cubic(6.0)
    potential = LennardJonesPotential(cutoff=2.5)
    neighbors = build_neighbor_list(positions, cell, cutoff=2.5)

    result = simulate(
        positions,
        velocities,
        cell=cell,
        potential=potential,
        pairs=neighbors.pairs,
        steps=3,
    )

    assert np.isfinite(np.array(result.total_energy)).all()
