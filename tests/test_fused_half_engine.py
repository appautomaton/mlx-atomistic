from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.interaction_engine import (
    _build_interaction_schedule32,
    _fuse_interaction_halves32,
)


def _base_memberships(schedule):
    represented = set()
    for left_block, left_slice, right_row in zip(
        schedule.ordinary_left_blocks,
        schedule.ordinary_left_slices,
        schedule.ordinary_right_atoms,
        strict=True,
    ):
        left_start = 16 * int(left_slice)
        for right in right_row[right_row < schedule.padded_atom_count]:
            for left_slot in range(left_start, left_start + 16):
                represented.add((int(left_block), left_slot, int(right)))
    return represented


def _fused_memberships(schedule):
    represented = set()
    rights_per_block = {}
    for left_block, right_row, mode in zip(
        schedule.ordinary_left_blocks,
        schedule.ordinary_right_atoms,
        schedule.ordinary_half_modes,
        strict=True,
    ):
        observed = rights_per_block.setdefault(int(left_block), set())
        for right in right_row:
            if right == schedule.padded_atom_count:
                continue
            assert right not in observed
            observed.add(int(right))
            for half in range(2):
                if int(mode) & (1 << half):
                    for left_slot in range(16 * half, 16 * (half + 1)):
                        represented.add((int(left_block), left_slot, int(right)))
    return represented


def test_fused_half_schedule_preserves_base_memberships_and_special_work():
    rng = np.random.default_rng(61)
    positions = rng.random((256, 3)) * np.asarray([20.0, 21.0, 22.0])
    base = _build_interaction_schedule32(
        positions,
        [20.0, 21.0, 22.0],
        search_radius=4.5,
        lj_exclusion_pairs=[[0, 1]],
        lj_one_four_pairs=[[2, 3]],
        left_slice_size=16,
    )

    fused = _fuse_interaction_halves32(base)

    assert _fused_memberships(fused) == _base_memberships(base)
    assert fused.ordinary_logical_pair_lanes == (
        16 * np.count_nonzero(base.ordinary_right_atoms < base.padded_atom_count)
    )
    assert fused.ordinary_right_entry_count < np.count_nonzero(
        base.ordinary_right_atoms < base.padded_atom_count
    )
    np.testing.assert_array_equal(
        fused.special_work_right_atoms,
        base.special_work_right_atoms,
    )
    np.testing.assert_array_equal(
        fused.special_work_lj_enabled,
        base.special_work_lj_enabled,
    )
    np.testing.assert_array_equal(
        fused.special_work_lj_one_four,
        base.special_work_lj_one_four,
    )


def test_fused_half_schedule_rejects_non_half_base_schedule():
    rng = np.random.default_rng(63)
    positions = rng.random((96, 3)) * 20.0
    base = _build_interaction_schedule32(
        positions,
        [20.0, 20.0, 20.0],
        search_radius=4.5,
        left_slice_size=8,
    )

    with pytest.raises(ValueError, match="16-atom base"):
        _fuse_interaction_halves32(base)


def test_fused_half_schedule_accepts_special_only_system():
    rng = np.random.default_rng(65)
    positions = rng.random((16, 3)) * 10.0
    base = _build_interaction_schedule32(
        positions,
        [20.0, 20.0, 20.0],
        search_radius=4.5,
        left_slice_size=16,
    )

    fused = _fuse_interaction_halves32(base)

    assert fused.ordinary_tile_count == 0
    assert fused.ordinary_group_count == 0
    assert fused.ordinary_right_atoms.shape == (0, 32)
    assert fused.ordinary_half_modes.shape == (0,)
    assert fused.special_work_count == base.special_work_count
