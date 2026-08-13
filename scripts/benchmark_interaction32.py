#!/usr/bin/env python3
"""Benchmark the experimental 32-atom direct force against production tiles."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.benchmarks.charged_pme import _bind_pme_plans, _find_pme_term
from mlx_atomistic.interaction_engine import (
    _build_interaction_schedule32,
    _interaction32_direct_force_only,
    _schedule_to_device32,
)
from mlx_atomistic.neighbors import _MLX_MD_CACHE_LIMIT_BYTES, NeighborListManager


def _activate_metal() -> None:
    os.environ["MLX_ATOMISTIC_DEVICE"] = "gpu"
    device = mx.Device(mx.gpu, 0)
    mx.set_default_device(device)
    mx.set_default_stream(mx.new_stream(device))
    mx.set_cache_limit(_MLX_MD_CACHE_LIMIT_BYTES)
    probe = mx.array([1.0], dtype=mx.float32) + 1.0
    mx.eval(probe)


def _interleaved_samples(
    control_call,
    candidate_call,
    *,
    warmups: int,
    samples: int,
) -> dict[str, list[float]]:
    calls = {"control": control_call, "candidate": candidate_call}
    for _ in range(warmups):
        for call in calls.values():
            mx.eval(call())
    timings = {"control": [], "candidate": []}
    for sample in range(samples):
        order = ("control", "candidate") if sample % 2 == 0 else ("candidate", "control")
        for name in order:
            started = time.perf_counter()
            mx.eval(calls[name]())
            timings[name].append(time.perf_counter() - started)
    return timings


def _marginal_samples(
    control_call,
    candidate_call,
    *,
    warmups: int,
    samples: int,
    batch_count: int = 9,
) -> dict[str, object]:
    calls = {"control": control_call, "candidate": candidate_call}

    def evaluate(call, count: int) -> float:
        values = [call() for _ in range(count)]
        started = time.perf_counter()
        mx.eval(*values)
        return time.perf_counter() - started

    for _ in range(warmups):
        for call in calls.values():
            evaluate(call, 1)
            evaluate(call, batch_count)
    raw = {
        name: {"single_seconds": [], "batch_seconds": [], "marginal_seconds": []} for name in calls
    }
    for sample in range(samples):
        order = ("control", "candidate") if sample % 2 == 0 else ("candidate", "control")
        for name in order:
            call = calls[name]
            if sample % 2 == 0:
                single = evaluate(call, 1)
                batch = evaluate(call, batch_count)
            else:
                batch = evaluate(call, batch_count)
                single = evaluate(call, 1)
            raw[name]["single_seconds"].append(single)
            raw[name]["batch_seconds"].append(batch)
            raw[name]["marginal_seconds"].append((batch - single) / (batch_count - 1))
    medians = {name: float(np.median(values["marginal_seconds"])) for name, values in raw.items()}
    return {
        "batch_count": batch_count,
        "samples": raw,
        "control_marginal_median_seconds": medians["control"],
        "candidate_marginal_median_seconds": medians["candidate"],
        "candidate_speedup_fraction": 1.0 - medians["candidate"] / medians["control"],
    }


def benchmark(
    prepared: Path,
    *,
    skin: float,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    _activate_metal()
    artifact = load_prepared_mlx_artifact(prepared, require_production=True)
    system, force_terms, _ = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if system.cell is None or not system.cell.is_orthorhombic:
        raise ValueError("interaction32 benchmark requires an orthorhombic cell")
    bound_terms = _bind_pme_plans(force_terms, system.cell)
    nonbonded = _find_pme_term(bound_terms)
    cutoff = float(nonbonded.cutoff)
    box_lengths = np.asarray(np.diag(np.asarray(system.cell.matrix)), dtype=np.float32)

    tile_manager = NeighborListManager(
        system.cell,
        cutoff=cutoff,
        skin=skin,
        check_interval=1,
        sort_pairs=False,
        backend="mlx_cell_tiles",
        displacement_check_backend="mlx_scalar",
    )
    tile_build_started = time.perf_counter()
    neighbor_list = tile_manager.update(system.positions)
    tiles = neighbor_list.tiles
    if tiles is None:
        raise RuntimeError("production tile manager did not produce tiles")
    tile_binding = nonbonded._prepare_tile_force_binding(
        system.cell,
        neighbor_list.diagnostic_pairs,
        tiles,
    )
    if tile_binding is NotImplemented:
        raise RuntimeError("production direct tile binding returned NotImplemented")
    if tile_binding.tile_decline_reason is not None:
        raise RuntimeError(
            "production direct tile route was not admitted: "
            + tile_binding.tile_decline_reason
            + "; diagnostics="
            + repr(
                {
                    "device": str(mx.default_device()),
                    "orthorhombic": system.cell.is_orthorhombic,
                    "cutoff": nonbonded.cutoff,
                    "pme_real_cutoff": nonbonded.pme_config.real_cutoff,
                    "has_nbfix": nonbonded.has_nbfix,
                    "force_columns": tiles.force_columns is not None,
                    "force_group_starts": tiles.force_group_starts is not None,
                    "force_group_counts": tiles.force_group_counts is not None,
                }
            )
        )
    mx.eval(
        tiles.atom_blocks,
        tiles.tile_blocks,
        tiles.member_mask,
        tiles.force_columns,
        tiles.force_group_starts,
        tiles.force_group_counts,
        tile_binding.tile_lj_enabled_mask,
        tile_binding.tile_lj_one_four_mask,
    )
    tile_build_seconds = time.perf_counter() - tile_build_started

    mx.eval(
        system.positions,
        nonbonded._aligned_lj_exclusion_pairs,
        nonbonded._aligned_lj_one_four_pairs,
    )
    positions_np = np.asarray(system.positions)
    exclusion_pairs = np.asarray(nonbonded._aligned_lj_exclusion_pairs)
    one_four_pairs = np.asarray(nonbonded._aligned_lj_one_four_pairs)
    schedule_started = time.perf_counter()
    schedule = _build_interaction_schedule32(
        positions_np,
        box_lengths,
        search_radius=cutoff + skin,
        lj_exclusion_pairs=exclusion_pairs,
        lj_one_four_pairs=one_four_pairs,
    )
    schedule_build_seconds = time.perf_counter() - schedule_started
    device_schedule = _schedule_to_device32(schedule)
    mx.eval(
        device_schedule.atom_order,
        device_schedule.ordinary_left_blocks,
        device_schedule.ordinary_right_atoms,
        device_schedule.special_blocks,
        device_schedule.special_lj_enabled,
        device_schedule.special_lj_one_four,
    )

    def control_call():
        return nonbonded._direct_forces_from_binding(system.positions, tile_binding)

    def candidate_call():
        return _interaction32_direct_force_only(
            system.positions,
            device_schedule,
            tile_binding.box_lengths_and_inverses,
            tile_binding.half_sigma,
            tile_binding.sqrt_epsilon,
            nonbonded.charges,
            cutoff=cutoff,
            shift=nonbonded.lj_shift,
            switch_distance=nonbonded.switch_distance,
            one_four_scale=nonbonded.lj_one_four_scale,
            coulomb_constant=nonbonded.coulomb_constant,
            alpha=nonbonded.pme_config.alpha,
        )

    control = control_call()
    candidate = candidate_call()
    mx.eval(control, candidate)
    delta = np.asarray(candidate) - np.asarray(control)
    force_rms_delta = float(np.sqrt(np.mean(delta * delta)))
    force_max_delta = float(np.max(np.abs(delta)))
    force_max_reference = float(np.max(np.abs(np.asarray(control))))
    timings = _interleaved_samples(
        control_call,
        candidate_call,
        warmups=warmups,
        samples=samples,
    )
    marginal = _marginal_samples(
        control_call,
        candidate_call,
        warmups=max(1, warmups // 2),
        samples=samples,
    )
    control_median = float(np.median(timings["control"]))
    candidate_median = float(np.median(timings["candidate"]))
    return {
        "schema": "mlx_atomistic.interaction32_force_benchmark.v1",
        "prepared": str(prepared),
        "atom_count": schedule.atom_count,
        "cutoff_angstrom": cutoff,
        "skin_angstrom": skin,
        "search_radius_angstrom": cutoff + skin,
        "production": {
            "block_size": tiles.block_size,
            "tile_count": tiles.tile_count,
            "force_group_count": tiles.force_group_count,
            "build_seconds": tile_build_seconds,
        },
        "interaction32": {
            "block_count": schedule.block_count,
            "ordinary_tile_count": schedule.ordinary_tile_count,
            "special_tile_count": schedule.special_tile_count,
            "scheduled_pair_lanes": (schedule.ordinary_tile_count + schedule.special_tile_count)
            * 32
            * 32,
            "oracle_build_seconds": schedule_build_seconds,
        },
        "force_parity": {
            "rms_delta_kj_mol_angstrom": force_rms_delta,
            "max_delta_kj_mol_angstrom": force_max_delta,
            "max_reference_kj_mol_angstrom": force_max_reference,
            "finite": bool(np.all(np.isfinite(delta))),
        },
        "timing": {
            "warmups": warmups,
            "samples": samples,
            "control_seconds": timings["control"],
            "candidate_seconds": timings["candidate"],
            "control_median_seconds": control_median,
            "candidate_median_seconds": candidate_median,
            "candidate_speedup_fraction": 1.0 - candidate_median / control_median,
        },
        "marginal_timing": marginal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--skin", type=float, default=5.5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.prepared,
        skin=args.skin,
        warmups=args.warmups,
        samples=args.samples,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
