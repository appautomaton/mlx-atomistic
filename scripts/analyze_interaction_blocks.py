#!/usr/bin/env python3
"""Inventory a proposed 32-atom Metal interaction-block schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TILE_SIZE = 32
DEFAULT_SKIN_ANGSTROM = 5.5


def _spread_three_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False) & np.uint64(0x1FFFFF)
    values = (values | (values << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    values = (values | (values << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    values = (values | (values << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    values = (values | (values << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    return (values | (values << np.uint64(2))) & np.uint64(0x1249249249249249)


def _spatial_keys(
    positions: np.ndarray,
    box: np.ndarray,
    search_radius: float,
    ordering: str,
) -> np.ndarray:
    if ordering == "canonical":
        return np.arange(positions.shape[0], dtype=np.uint64)

    cell_width = search_radius / 3.0
    cell_counts = np.maximum(np.floor(box / cell_width).astype(np.int64), 1)
    wrapped = positions - box * np.floor(positions / box)
    cells = np.floor(wrapped * cell_counts / box).astype(np.int64)
    cells = np.minimum(cells, cell_counts - 1)
    if ordering == "cell":
        return cells[:, 0].astype(np.uint64) + np.uint64(cell_counts[0]) * (
            cells[:, 1].astype(np.uint64)
            + np.uint64(cell_counts[1]) * cells[:, 2].astype(np.uint64)
        )
    if ordering == "morton":
        return (
            _spread_three_bits(cells[:, 0])
            | (_spread_three_bits(cells[:, 1]) << np.uint64(1))
            | (_spread_three_bits(cells[:, 2]) << np.uint64(2))
        )
    raise ValueError(f"unknown ordering: {ordering}")


def _make_blocks(
    positions: np.ndarray,
    box: np.ndarray,
    atom_order: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    atom_count = positions.shape[0]
    padded_count = ((atom_count + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    padded_order = np.full((padded_count,), -1, dtype=np.int32)
    padded_order[:atom_count] = atom_order
    block_atoms = padded_order.reshape(-1, TILE_SIZE)

    valid = block_atoms >= 0
    safe_atoms = np.maximum(block_atoms, 0)
    block_positions = positions[safe_atoms]
    reference = block_positions[:, :1, :]
    unwrapped = reference + (
        block_positions - reference - box * np.rint((block_positions - reference) / box)
    )
    valid_count = np.sum(valid, axis=1, keepdims=True)
    centers = np.sum(unwrapped * valid[..., None], axis=1) / valid_count
    centered = np.where(valid[..., None], unwrapped - centers[:, None, :], 0.0)
    half_extents = np.max(np.abs(centered), axis=1)
    radii = np.sqrt(np.max(np.sum(centered * centered, axis=2), axis=1))
    centers -= box * np.floor(centers / box)

    atom_to_block = np.empty((atom_count,), dtype=np.int32)
    atom_to_block[atom_order] = np.arange(atom_count, dtype=np.int32) // TILE_SIZE
    return block_atoms, valid, centers, half_extents, radii, atom_to_block


def _special_block_codes(
    block_count: int,
    atom_to_block: np.ndarray,
    exception_pairs: np.ndarray,
) -> np.ndarray:
    diagonal = np.arange(block_count, dtype=np.int64)
    codes = diagonal * np.int64(block_count) + diagonal
    if exception_pairs.size:
        mapped = atom_to_block[exception_pairs]
        left = np.minimum(mapped[:, 0], mapped[:, 1]).astype(np.int64)
        right = np.maximum(mapped[:, 0], mapped[:, 1]).astype(np.int64)
        codes = np.concatenate((codes, left * np.int64(block_count) + right))
    return np.unique(codes)


def _contains_sorted(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sorted_values, values)
    in_bounds = indices < sorted_values.size
    result = np.zeros(values.shape, dtype=bool)
    result[in_bounds] = sorted_values[indices[in_bounds]] == values[in_bounds]
    return result


def _geometric_pair_count(
    positions: np.ndarray,
    box: np.ndarray,
    radius: float,
) -> int:
    wrapped = positions - box * np.floor(positions / box)
    tree = cKDTree(wrapped, boxsize=box)
    directed_with_self = int(tree.count_neighbors(tree, radius))
    return (directed_with_self - positions.shape[0]) // 2


def _inventory_ordering(
    *,
    positions: np.ndarray,
    box: np.ndarray,
    exception_pairs: np.ndarray,
    cutoff: float,
    skin: float,
    ordering: str,
    force_pair_count: int,
    search_pair_count: int,
) -> dict[str, object]:
    search_radius = cutoff + skin
    keys = _spatial_keys(positions, box, search_radius, ordering)
    atom_order = np.argsort(keys, kind="stable").astype(np.int32)
    block_atoms, valid, centers, extents, radii, atom_to_block = _make_blocks(
        positions, box, atom_order
    )
    block_count = block_atoms.shape[0]
    special_codes = _special_block_codes(block_count, atom_to_block, exception_pairs)

    # OpenMM searches smaller blocks first.  The exact bins are not relevant to
    # this inventory, but extent ordering captures the pair-orientation effect
    # that changes how right atoms are packed for each left block.
    extent_metric = np.sum(extents, axis=1)
    traversal = np.argsort(extent_metric, kind="stable")
    right_entries = np.zeros((block_count,), dtype=np.int64)
    candidate_block_pairs = 0
    admitted_block_pairs = 0

    for traversal_index, left_block in enumerate(traversal[:-1]):
        right_blocks = traversal[traversal_index + 1 :]
        delta = centers[right_blocks] - centers[left_block]
        delta -= box * np.rint(delta / box)
        center_distance2 = np.sum(delta * delta, axis=1)
        sphere_limit = search_radius + radii[left_block] + radii[right_blocks]
        keep = center_distance2 < sphere_limit * sphere_limit
        if not np.any(keep):
            continue

        candidate_right = right_blocks[keep]
        candidate_delta = delta[keep]
        aabb_delta = np.maximum(
            np.abs(candidate_delta) - extents[left_block] - extents[candidate_right],
            0.0,
        )
        keep_aabb = np.sum(aabb_delta * aabb_delta, axis=1) < search_radius**2
        candidate_right = candidate_right[keep_aabb]
        if candidate_right.size == 0:
            continue
        candidate_block_pairs += int(candidate_right.size)

        low = np.minimum(left_block, candidate_right).astype(np.int64)
        high = np.maximum(left_block, candidate_right).astype(np.int64)
        codes = low * np.int64(block_count) + high
        ordinary_right = candidate_right[~_contains_sorted(special_codes, codes)]

        left_atoms = block_atoms[left_block]
        left_valid = valid[left_block]
        left_positions = positions[np.maximum(left_atoms, 0)]
        for right_block in ordinary_right:
            right_atoms = block_atoms[right_block]
            right_valid = valid[right_block]
            right_positions = positions[np.maximum(right_atoms, 0)]
            pair_delta = left_positions[:, None, :] - right_positions[None, :, :]
            pair_delta -= box * np.rint(pair_delta / box)
            distance2 = np.sum(pair_delta * pair_delta, axis=2)
            pair_valid = left_valid[:, None] & right_valid[None, :]
            admitted_right = np.any(pair_valid & (distance2 < search_radius**2), axis=0)
            admitted_count = int(np.count_nonzero(admitted_right))
            if admitted_count:
                right_entries[left_block] += admitted_count
                admitted_block_pairs += 1

    ordinary_tiles_per_left = (right_entries + TILE_SIZE - 1) // TILE_SIZE
    ordinary_tile_count = int(np.sum(ordinary_tiles_per_left))
    special_tile_count = int(special_codes.size)
    total_tile_count = ordinary_tile_count + special_tile_count
    scheduled_lanes = total_tile_count * TILE_SIZE * TILE_SIZE
    capacity_tiles = max(int(np.ceil(ordinary_tile_count * 1.25)), ordinary_tile_count)

    state_bytes = {
        "atom_order": int(block_atoms.size * np.dtype(np.int32).itemsize),
        "inverse_atom_order": int(positions.shape[0] * np.dtype(np.int32).itemsize),
        "block_bounds": int(block_count * 2 * 4 * np.dtype(np.float32).itemsize),
        "ordinary_left_blocks": int(capacity_tiles * np.dtype(np.int32).itemsize),
        "ordinary_right_atoms": int(capacity_tiles * TILE_SIZE * np.dtype(np.int32).itemsize),
        "special_block_pairs": int(special_tile_count * 2 * np.dtype(np.int32).itemsize),
        "special_lj_masks": int(special_tile_count * TILE_SIZE * 2 * np.dtype(np.uint32).itemsize),
        "old_positions_float4": int(positions.shape[0] * 4 * np.dtype(np.float32).itemsize),
    }
    state_bytes["total"] = int(sum(state_bytes.values()))

    return {
        "ordering": ordering,
        "block_count": int(block_count),
        "special_tile_count": special_tile_count,
        "candidate_block_pairs_after_bounds": candidate_block_pairs,
        "admitted_ordinary_block_pairs": admitted_block_pairs,
        "ordinary_right_atom_entries": int(np.sum(right_entries)),
        "ordinary_tile_count": ordinary_tile_count,
        "ordinary_partial_tile_padding": int(
            ordinary_tile_count * TILE_SIZE - np.sum(right_entries)
        ),
        "total_tile_count": total_tile_count,
        "scheduled_pair_lanes": scheduled_lanes,
        "force_pair_occupancy": force_pair_count / scheduled_lanes,
        "search_pair_occupancy": search_pair_count / scheduled_lanes,
        "ordinary_tiles_per_left": {
            "max": int(np.max(ordinary_tiles_per_left)),
            "p50": float(np.percentile(ordinary_tiles_per_left, 50)),
            "p95": float(np.percentile(ordinary_tiles_per_left, 95)),
            "p99": float(np.percentile(ordinary_tiles_per_left, 99)),
        },
        "estimated_persistent_state_bytes": state_bytes,
    }


def analyze(
    prepared_path: Path,
    *,
    skin: float,
    orderings: tuple[str, ...],
) -> dict[str, object]:
    with np.load(prepared_path) as prepared:
        positions = np.asarray(prepared["positions"], dtype=np.float64)
        box = np.asarray(prepared["cell_lengths"], dtype=np.float64)
        cutoff = float(np.asarray(prepared["pme_real_cutoff"]).reshape(-1)[0])
        exception_pairs = np.asarray(prepared["nonbonded_exception_pairs"], dtype=np.int32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("prepared positions must have shape (n_atoms, 3)")
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError("prepared cell_lengths must be a positive orthorhombic box")
    if cutoff <= 0.0 or skin < 0.0:
        raise ValueError("cutoff must be positive and skin must be non-negative")

    force_pair_count = _geometric_pair_count(positions, box, cutoff)
    search_pair_count = _geometric_pair_count(positions, box, cutoff + skin)
    inventories = [
        _inventory_ordering(
            positions=positions,
            box=box,
            exception_pairs=exception_pairs,
            cutoff=cutoff,
            skin=skin,
            ordering=ordering,
            force_pair_count=force_pair_count,
            search_pair_count=search_pair_count,
        )
        for ordering in orderings
    ]
    return {
        "schema": "mlx_atomistic.interaction_block_inventory.v1",
        "prepared_path": str(prepared_path),
        "atom_count": int(positions.shape[0]),
        "box_angstrom": box.tolist(),
        "cutoff_angstrom": cutoff,
        "skin_angstrom": skin,
        "search_radius_angstrom": cutoff + skin,
        "force_pair_count": force_pair_count,
        "search_pair_count": search_pair_count,
        "exception_pair_count": int(exception_pairs.shape[0]),
        "tile_size": TILE_SIZE,
        "inventories": inventories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="Prepared-system NPZ path")
    parser.add_argument("--skin", type=float, default=DEFAULT_SKIN_ANGSTROM)
    parser.add_argument(
        "--ordering",
        action="append",
        choices=("canonical", "cell", "morton"),
        dest="orderings",
        help="Atom ordering to inventory; may be repeated",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    payload = analyze(
        args.prepared,
        skin=args.skin,
        orderings=tuple(args.orderings or ("canonical", "cell", "morton")),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
