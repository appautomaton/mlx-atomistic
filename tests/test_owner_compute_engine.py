from __future__ import annotations

from collections import Counter

import numpy as np

from mlx_atomistic.interaction_engine import _build_owner_compute_schedule32


def _represented_directed_pairs(schedule, positions):
    box = np.asarray([12.0, 13.0, 14.0])
    radius2 = schedule.search_radius**2
    blocks = schedule.atom_order.reshape((-1, 32))
    represented = Counter()
    for block, owners in enumerate(blocks):
        right = schedule.right_atoms[
            schedule.owner_offsets[block] : schedule.owner_offsets[block + 1]
        ]
        right = right[right < schedule.padded_atom_count]
        right_atoms = schedule.atom_order[right]
        for owner in owners[owners >= 0]:
            delta = positions[owner] - positions[right_atoms]
            delta -= box * np.rint(delta / box)
            members = right_atoms[np.sum(delta * delta, axis=1) < radius2]
            for neighbor in members:
                if owner != neighbor:
                    represented[(int(owner), int(neighbor))] += 1
    return represented


def _brute_directed_pairs(positions, box, radius):
    represented = set()
    for owner in range(positions.shape[0]):
        delta = positions[owner] - positions
        delta -= box * np.rint(delta / box)
        for neighbor in np.nonzero(np.sum(delta * delta, axis=1) < radius**2)[0]:
            if owner != neighbor:
                represented.add((owner, int(neighbor)))
    return represented


def test_owner_schedule_covers_every_directed_periodic_pair_once():
    rng = np.random.default_rng(41)
    box = np.asarray([12.0, 13.0, 14.0])
    positions = rng.random((96, 3)) * box
    schedule = _build_owner_compute_schedule32(
        positions,
        box,
        search_radius=2.5,
    )

    represented = _represented_directed_pairs(schedule, positions)

    assert set(represented) == _brute_directed_pairs(positions, box, 2.5)
    assert set(represented.values()) == {1}
    assert schedule.owner_offsets.shape == (schedule.block_count + 1,)
    assert schedule.right_atoms.shape[0] % 32 == 0


def test_owner_topology_is_directed_and_classified():
    rng = np.random.default_rng(43)
    box = np.asarray([12.0, 13.0, 14.0])
    positions = rng.random((64, 3)) * box
    schedule = _build_owner_compute_schedule32(
        positions,
        box,
        search_radius=2.5,
        lj_exclusion_pairs=[[0, 1]],
        lj_one_four_pairs=[[2, 3]],
    )

    observed = set()
    for owner in range(positions.shape[0]):
        start = schedule.topology_offsets[owner]
        stop = schedule.topology_offsets[owner + 1]
        for neighbor, topology_class in zip(
            schedule.topology_neighbors[start:stop],
            schedule.topology_classes[start:stop],
            strict=True,
        ):
            observed.add((owner, int(neighbor), int(topology_class)))

    assert observed == {(0, 1, 0), (1, 0, 0), (2, 3, 1), (3, 2, 1)}
