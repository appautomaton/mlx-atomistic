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


def test_prepared_pipeline_preserves_generic_force_fallback():
    positions = mx.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    pipeline = _PreparedForcePipeline.prepare(
        (_FallbackTerm(),),
        cell=None,
    )

    forces = pipeline.bind(None).forces(positions)

    np.testing.assert_allclose(np.asarray(forces), -2.0 * np.asarray(positions))


def test_prepared_pipeline_accumulates_multiple_force_terms():
    positions = mx.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    pipeline = _PreparedForcePipeline.prepare(
        (_FallbackTerm(), _FallbackTerm()),
        cell=None,
    )

    forces = pipeline.bind(None).forces(positions)

    np.testing.assert_allclose(np.asarray(forces), -4.0 * np.asarray(positions))


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
