from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.force_runtime import _PreparedForcePipeline
from mlx_atomistic.neighbors import NeighborListManager


class _FallbackTerm:
    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        return mx.sum(positions * 0.0), -2.0 * positions


class _PreparedTerm:
    def __init__(self):
        self.prepare_calls = 0
        self.force_calls = 0

    def _prepare_force_binding(self, cell, pairs):
        self.prepare_calls += 1
        return (cell, pairs)

    def _forces_from_binding(self, positions, binding):
        self.force_calls += 1
        cell, pairs = binding
        assert cell is not None
        assert pairs is not None
        return mx.ones_like(positions)


class _PreparedTileTerm(_PreparedTerm):
    def __init__(self, *, ready=True):
        super().__init__()
        self.ready = ready
        self.tile_force_calls = 0

    def _prepare_tile_force_binding(self, cell, pairs, tiles):
        self.prepare_calls += 1
        return SimpleNamespace(
            cell=cell,
            pairs=pairs,
            tiles=tiles,
            tile_force_ready=self.ready,
            tile_decline_reason=None if self.ready else "test_decline",
        )

    def _prepare_force_binding(self, cell, pairs):
        self.prepare_calls += 1
        return SimpleNamespace(
            cell=cell,
            pairs=pairs,
            tiles=None,
            tile_force_ready=False,
            tile_decline_reason="tile_geometry_unavailable",
        )

    def _tile_forces_from_binding(self, positions, binding):
        assert binding.tiles is not None
        self.tile_force_calls += 1
        return mx.ones_like(positions) * 3.0

    def _forces_from_binding(self, positions, binding):
        assert binding.pairs is not None
        self.force_calls += 1
        return mx.ones_like(positions)


def test_prepared_pipeline_preserves_generic_force_fallback():
    positions = mx.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    pipeline = _PreparedForcePipeline.prepare(
        (_FallbackTerm(),),
        cell=None,
    )

    forces = pipeline.bind(None).forces(positions)

    np.testing.assert_allclose(np.asarray(forces), -2.0 * np.asarray(positions))


def test_prepared_pipeline_binds_once_per_neighbor_generation():
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    manager = NeighborListManager(cell, cutoff=1.5, skin=0.4)
    first_list = manager.update(positions)
    term = _PreparedTerm()
    pipeline = _PreparedForcePipeline.prepare((term,), cell=cell)

    first = pipeline.bind(first_list)
    repeated = pipeline.bind(first_list)
    first_forces = repeated.forces(mx.array(positions))

    moved = positions.copy()
    moved[0, 0] += 0.21
    second_list = manager.update(moved)
    second = pipeline.bind(second_list)
    second_forces = second.forces(mx.array(moved))

    assert repeated is first
    assert second is not first
    assert term.prepare_calls == 2
    assert term.force_calls == 2
    np.testing.assert_array_equal(np.asarray(first_forces), np.ones_like(positions))
    np.testing.assert_array_equal(np.asarray(second_forces), np.ones_like(positions))


def test_prepared_pipeline_selects_tiles_only_when_requested_and_ready():
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    cell = Cell.cubic(6.0)
    tile_neighbors = NeighborListManager(
        cell,
        cutoff=1.5,
        skin=0.4,
        backend="mlx_cell_tiles",
    ).update(positions)

    selected_term = _PreparedTileTerm()
    selected = _PreparedForcePipeline.prepare(
        (selected_term,),
        cell=cell,
        direct_force_backend="atom-tiles",
    ).bind(tile_neighbors)
    selected_forces = selected.forces(mx.array(positions))

    control_term = _PreparedTileTerm()
    control = _PreparedForcePipeline.prepare(
        (control_term,),
        cell=cell,
    ).bind(tile_neighbors)
    control_forces = control.forces(mx.array(positions))

    np.testing.assert_array_equal(
        np.asarray(selected_forces),
        np.full_like(positions, 3.0),
    )
    np.testing.assert_array_equal(np.asarray(control_forces), np.ones_like(positions))
    assert selected.route_report()["selected_backend"] == "atom-tiles"
    assert selected.route_report()["tile_decline_reasons"] == []
    assert control.route_report()["selected_backend"] == "explicit-pairs"


def test_prepared_pipeline_records_declined_tile_fallback():
    positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    cell = Cell.cubic(6.0)
    neighbors = NeighborListManager(
        cell,
        cutoff=1.5,
        skin=0.4,
        backend="mlx_cell_tiles",
    ).update(positions)
    term = _PreparedTileTerm(ready=False)
    bound = _PreparedForcePipeline.prepare(
        (term,),
        cell=cell,
        direct_force_backend="atom-tiles",
    ).bind(neighbors)

    forces = bound.forces(mx.array(positions))

    np.testing.assert_array_equal(np.asarray(forces), np.ones_like(positions))
    assert bound.route_report()["selected_backend"] == "explicit-pairs"
    assert bound.route_report()["tile_decline_reasons"] == ["test_decline"]
