from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mlx_atomistic.interaction_engine import (
    _build_interaction_schedule32,
    _InteractionSchedule32,
)


def _two_block_positions() -> np.ndarray:
    first = np.stack(
        (
            np.linspace(1.0, 1.31, 32),
            np.full((32,), 1.0),
            np.full((32,), 1.0),
        ),
        axis=1,
    )
    return np.concatenate((first, first + np.asarray([1.0, 0.0, 0.0])), axis=0)


def _represented_pairs(
    schedule: _InteractionSchedule32,
    positions: np.ndarray,
) -> Counter[tuple[int, int]]:
    box = schedule.box_lengths.astype(np.float64)
    radius2 = schedule.search_radius**2
    represented: Counter[tuple[int, int]] = Counter()

    def add_if_close(atom_i: int, atom_j: int) -> None:
        if atom_i < 0 or atom_j < 0 or atom_i == atom_j:
            return
        delta = positions[atom_i] - positions[atom_j]
        delta -= box * np.rint(delta / box)
        if float(np.dot(delta, delta)) < radius2:
            represented[tuple(sorted((atom_i, atom_j)))] += 1

    blocks = schedule.atom_order.reshape((-1, 32))
    for left_block, left_slice, right_atoms in zip(
        schedule.ordinary_left_blocks,
        schedule.ordinary_left_slices,
        schedule.ordinary_right_atoms,
        strict=True,
    ):
        start = schedule.left_slice_size * left_slice
        stop = start + schedule.left_slice_size
        for left_atom in blocks[left_block, start:stop]:
            for right_ordered in right_atoms:
                right_atom = (
                    -1
                    if right_ordered >= schedule.padded_atom_count
                    else schedule.atom_order[right_ordered]
                )
                add_if_close(int(left_atom), int(right_atom))
    for left_block, right_block in schedule.special_blocks:
        for left_slot, left_atom in enumerate(blocks[left_block]):
            for right_slot, right_atom in enumerate(blocks[right_block]):
                if left_block == right_block and left_slot >= right_slot:
                    continue
                add_if_close(int(left_atom), int(right_atom))
    return represented


def _brute_pairs(
    positions: np.ndarray,
    box: np.ndarray,
    radius: float,
) -> set[tuple[int, int]]:
    pairs = set()
    for atom_i in range(positions.shape[0]):
        delta = positions[atom_i + 1 :] - positions[atom_i]
        delta -= box * np.rint(delta / box)
        distance2 = np.sum(delta * delta, axis=1)
        for offset in np.nonzero(distance2 < radius**2)[0]:
            pairs.add((atom_i, atom_i + 1 + int(offset)))
    return pairs


def test_schedule_covers_each_periodic_search_pair_once():
    positions = _two_block_positions()
    box = np.asarray([20.0, 20.0, 20.0])
    schedule = _build_interaction_schedule32(
        positions,
        box,
        search_radius=3.0,
    )

    represented = _represented_pairs(schedule, positions)

    assert schedule.block_count == 2
    assert schedule.special_tile_count == 2
    assert schedule.ordinary_tile_count == 2
    assert schedule.ordinary_group_count == 2
    assert set(represented) == _brute_pairs(positions, box, 3.0)
    assert set(represented.values()) == {1}


@pytest.mark.parametrize("left_slice_size", (4, 8, 16))
def test_random_periodic_schedule_covers_each_pair_once(left_slice_size):
    rng = np.random.default_rng(17)
    box = np.asarray([12.0, 13.0, 14.0])
    positions = rng.random((96, 3)) * box
    base = _build_interaction_schedule32(positions, box, search_radius=2.5)
    excluded = [[int(base.atom_order[0]), int(base.atom_order[32])]]
    one_four = [[int(base.atom_order[1]), int(base.atom_order[33])]]
    schedule = _build_interaction_schedule32(
        positions,
        box,
        search_radius=2.5,
        lj_exclusion_pairs=excluded,
        lj_one_four_pairs=one_four,
        left_slice_size=left_slice_size,
    )

    represented = _represented_pairs(schedule, positions)

    assert schedule.special_tile_count > schedule.block_count
    assert set(represented) == _brute_pairs(positions, box, 2.5)
    assert set(represented.values()) == {1}


def test_special_masks_encode_exclusions_and_one_four_pairs():
    positions = _two_block_positions()
    base = _build_interaction_schedule32(
        positions,
        [20.0, 20.0, 20.0],
        search_radius=3.0,
    )
    excluded = np.sort([base.atom_order[0], base.atom_order[32]])
    one_four = np.sort([base.atom_order[1], base.atom_order[33]])
    schedule = _build_interaction_schedule32(
        positions,
        [20.0, 20.0, 20.0],
        search_radius=3.0,
        lj_exclusion_pairs=[excluded],
        lj_one_four_pairs=[one_four],
    )
    excluded_ordered = schedule.inverse_order[excluded]
    one_four_ordered = schedule.inverse_order[one_four]
    excluded_blocks = excluded_ordered // 32
    one_four_blocks = one_four_ordered // 32
    assert np.array_equal(excluded_blocks, one_four_blocks)
    assert excluded_blocks[0] != excluded_blocks[1]
    tile = int(
        np.nonzero(np.all(schedule.special_blocks == excluded_blocks[None, :], axis=1))[0][0]
    )

    excluded_left_slot, excluded_right_slot = excluded_ordered % 32
    one_four_left_slot, one_four_right_slot = one_four_ordered % 32
    excluded_word = schedule.special_lj_enabled[tile, excluded_right_slot]
    one_four_enabled_word = schedule.special_lj_enabled[tile, one_four_right_slot]
    one_four_word = schedule.special_lj_one_four[tile, one_four_right_slot]

    assert ((excluded_word >> excluded_left_slot) & 1) == 0
    assert ((one_four_enabled_word >> one_four_left_slot) & 1) == 1
    assert ((one_four_word >> one_four_left_slot) & 1) == 1


def test_schedule_rejects_overlapping_lj_pair_classes():
    with pytest.raises(ValueError, match="must be disjoint"):
        _build_interaction_schedule32(
            _two_block_positions(),
            [20.0, 20.0, 20.0],
            search_radius=3.0,
            lj_exclusion_pairs=[[0, 1]],
            lj_one_four_pairs=[[1, 0]],
        )


def test_ordinary_groups_preserve_left_block_runs_and_tile_coverage():
    rng = np.random.default_rng(23)
    box = np.asarray([18.0, 19.0, 20.0])
    positions = rng.random((256, 3)) * box
    schedule = _build_interaction_schedule32(
        positions,
        box,
        search_radius=3.0,
        ordinary_tiles_per_group=3,
    )

    covered: list[int] = []
    for start, count in zip(
        schedule.ordinary_group_starts,
        schedule.ordinary_group_counts,
        strict=True,
    ):
        assert 1 <= count <= 3
        stop = int(start + count)
        left_blocks = schedule.ordinary_left_blocks[start:stop]
        left_slices = schedule.ordinary_left_slices[start:stop]
        assert np.all(left_blocks == left_blocks[0])
        assert np.all(left_slices == left_slices[0])
        covered.extend(range(int(start), stop))

    assert covered == list(range(schedule.ordinary_tile_count))
