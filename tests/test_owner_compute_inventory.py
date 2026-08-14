from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_owner_compute.py"
_SPEC = importlib.util.spec_from_file_location("analyze_owner_compute", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_INVENTORY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INVENTORY)


def test_owner_neighbor_counts_cover_periodic_directed_neighbors():
    positions = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [9.9, 0.1, 0.1],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    counts = _INVENTORY._owner_neighbor_counts(
        positions,
        np.asarray([10.0, 10.0, 10.0]),
        np.arange(3, dtype=np.int32),
        0.5,
    )

    np.testing.assert_array_equal(counts, np.asarray([3], dtype=np.int32))


def test_cell_order_is_a_stable_permutation():
    positions = np.asarray(
        [[8.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.1, 1.0, 1.0]],
        dtype=np.float64,
    )
    order = _INVENTORY._cell_atom_order(
        positions,
        np.asarray([10.0, 10.0, 10.0]),
        2.0,
    )

    np.testing.assert_array_equal(np.sort(order), np.arange(3))
    assert list(order).index(1) < list(order).index(2)
