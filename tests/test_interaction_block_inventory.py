from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_interaction_blocks.py"
_SPEC = importlib.util.spec_from_file_location("analyze_interaction_blocks", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_INVENTORY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INVENTORY)


def test_special_block_codes_include_diagonal_and_topology_pairs():
    atom_to_block = np.asarray([0, 0, 1, 1], dtype=np.int32)
    exceptions = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    codes = _INVENTORY._special_block_codes(2, atom_to_block, exceptions)

    np.testing.assert_array_equal(codes, np.asarray([0, 1, 3]))


def test_inventory_packs_one_ordinary_cross_block():
    first = np.stack(
        (
            np.linspace(1.0, 1.31, 32),
            np.full((32,), 1.0),
            np.full((32,), 1.0),
        ),
        axis=1,
    )
    second = first + np.asarray([1.0, 0.0, 0.0])
    positions = np.concatenate((first, second), axis=0)
    box = np.asarray([20.0, 20.0, 20.0])

    force_pairs = _INVENTORY._geometric_pair_count(positions, box, 3.0)
    inventory = _INVENTORY._inventory_ordering(
        positions=positions,
        box=box,
        exception_pairs=np.empty((0, 2), dtype=np.int32),
        cutoff=3.0,
        skin=0.0,
        ordering="canonical",
        force_pair_count=force_pairs,
        search_pair_count=force_pairs,
    )

    assert inventory["block_count"] == 2
    assert inventory["special_tile_count"] == 2
    assert inventory["ordinary_right_atom_entries"] == 32
    assert inventory["ordinary_tile_count"] == 1
    assert inventory["ordinary_partial_tile_padding"] == 0
    assert inventory["total_tile_count"] == 3
    assert inventory["scheduled_pair_lanes"] == 3 * 32 * 32
    assert inventory["estimated_persistent_state_bytes"]["inverse_atom_order"] == 256
