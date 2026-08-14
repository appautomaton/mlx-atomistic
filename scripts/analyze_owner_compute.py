#!/usr/bin/env python3
"""Inventory a directed no-atomic owner-computes interaction schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TILE_SIZE = 32
DEFAULT_SKIN_ANGSTROM = 5.5


def _cell_atom_order(
    positions: np.ndarray,
    box: np.ndarray,
    search_radius: float,
) -> np.ndarray:
    cell_width = search_radius / 3.0
    cell_counts = np.maximum(np.floor(box / cell_width).astype(np.int64), 1)
    wrapped = positions - box * np.floor(positions / box)
    cells = np.floor(wrapped * cell_counts / box).astype(np.int64)
    cells = np.minimum(cells, cell_counts - 1)
    keys = cells[:, 0].astype(np.int64) + cell_counts[0] * (
        cells[:, 1].astype(np.int64) + cell_counts[1] * cells[:, 2].astype(np.int64)
    )
    return np.argsort(keys, kind="stable").astype(np.int32)


def _owner_neighbor_counts(
    positions: np.ndarray,
    box: np.ndarray,
    atom_order: np.ndarray,
    search_radius: float,
) -> np.ndarray:
    atom_count = positions.shape[0]
    padded_count = ((atom_count + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    padded_order = np.full((padded_count,), -1, dtype=np.int32)
    padded_order[:atom_count] = atom_order
    blocks = padded_order.reshape((-1, TILE_SIZE))
    wrapped = positions - box * np.floor(positions / box)
    tree = cKDTree(wrapped, boxsize=box)
    counts = np.zeros((blocks.shape[0],), dtype=np.int32)
    for block_index, owners in enumerate(blocks):
        owners = owners[owners >= 0]
        neighborhoods = tree.query_ball_point(
            wrapped[owners],
            search_radius,
            return_sorted=False,
        )
        counts[block_index] = np.unique(np.concatenate(neighborhoods)).size
    return counts


def analyze(prepared_path: Path, *, skin: float) -> dict[str, object]:
    with np.load(prepared_path) as prepared:
        positions = np.asarray(prepared["positions"], dtype=np.float64)
        box = np.asarray(prepared["cell_lengths"], dtype=np.float64)
        cutoff = float(np.asarray(prepared["pme_real_cutoff"]).reshape(-1)[0])
        topology_pair_count = int(prepared["nonbonded_exception_pairs"].shape[0])
    search_radius = cutoff + skin
    atom_order = _cell_atom_order(positions, box, search_radius)
    neighbor_counts = _owner_neighbor_counts(
        positions,
        box,
        atom_order,
        search_radius,
    )
    padded_counts = ((neighbor_counts + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    right_entry_count = int(np.sum(padded_counts))
    scheduled_pair_lanes = right_entry_count * TILE_SIZE
    block_count = int(neighbor_counts.shape[0])
    atom_count = int(positions.shape[0])

    common_state = {
        "atom_order": atom_count * np.dtype(np.int32).itemsize,
        "owner_offsets": (block_count + 1) * np.dtype(np.int32).itemsize,
        "right_atoms": right_entry_count * np.dtype(np.int32).itemsize,
    }
    inline_mask_state = {
        **common_state,
        "lj_enabled_words": right_entry_count * np.dtype(np.uint32).itemsize,
        "lj_one_four_words": right_entry_count * np.dtype(np.uint32).itemsize,
    }
    inline_mask_state["total"] = int(sum(inline_mask_state.values()))
    sparse_correction_state = {
        **common_state,
        "topology_offsets": (atom_count + 1) * np.dtype(np.int32).itemsize,
        "topology_neighbors": topology_pair_count * 2 * np.dtype(np.int32).itemsize,
        "topology_classes": topology_pair_count * 2 * np.dtype(np.int32).itemsize,
    }
    sparse_correction_state["total"] = int(sum(sparse_correction_state.values()))

    return {
        "schema": "mlx_atomistic.owner_compute_inventory.v1",
        "prepared_path": str(prepared_path),
        "atom_count": atom_count,
        "block_count": block_count,
        "cutoff_angstrom": cutoff,
        "skin_angstrom": skin,
        "search_radius_angstrom": search_radius,
        "topology_pair_count": topology_pair_count,
        "right_atom_entries": right_entry_count,
        "scheduled_pair_lanes": scheduled_pair_lanes,
        "neighbors_per_owner_block": {
            "min": int(np.min(neighbor_counts)),
            "p50": float(np.percentile(neighbor_counts, 50)),
            "p95": float(np.percentile(neighbor_counts, 95)),
            "p99": float(np.percentile(neighbor_counts, 99)),
            "max": int(np.max(neighbor_counts)),
            "padding_fraction": 1.0 - float(np.sum(neighbor_counts)) / right_entry_count,
        },
        "estimated_state_bytes": {
            "inline_masks": inline_mask_state,
            "sparse_topology_correction": sparse_correction_state,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="Prepared-system NPZ path")
    parser.add_argument("--skin", type=float, default=DEFAULT_SKIN_ANGSTROM)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = analyze(args.prepared, skin=args.skin)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
