#!/usr/bin/env python3
"""Benchmark the experimental 32-atom direct force against production tiles."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
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
    _build_owner_compute_schedule32,
    _fuse_interaction_halves32,
    _fused_half32_direct_force_only,
    _fused_half_schedule_to_device32,
    _interaction32_direct_force_only,
    _owner_compute32_direct_force_only,
    _owner_schedule_to_device32,
    _retry_device_fused_half_schedule32,
    _schedule_to_device32,
    _try_build_device_fused_half_schedule32,
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


def _sequential_block_samples(
    control_call,
    candidate_call,
    *,
    warmups: int,
    samples: int,
    block_count: int,
) -> dict[str, object]:
    if block_count < 1:
        raise ValueError("timing block count must be positive")
    calls = {"control": control_call, "candidate": candidate_call}

    def evaluate(call) -> float:
        started = time.perf_counter()
        for _ in range(block_count):
            mx.eval(call())
        return (time.perf_counter() - started) / block_count

    for _ in range(warmups):
        for call in calls.values():
            evaluate(call)
    raw = {"control": [], "candidate": []}
    for sample in range(samples):
        order = ("control", "candidate") if sample % 2 == 0 else ("candidate", "control")
        for name in order:
            raw[name].append(evaluate(calls[name]))

    def direction(parity: int) -> dict[str, float]:
        control = float(np.median(raw["control"][parity::2]))
        candidate = float(np.median(raw["candidate"][parity::2]))
        return {
            "control_median_seconds": control,
            "candidate_median_seconds": candidate,
            "candidate_speedup_fraction": 1.0 - candidate / control,
        }

    control = float(np.median(raw["control"]))
    candidate = float(np.median(raw["candidate"]))
    return {
        "method": "sequential synchronized calls per timing block",
        "block_count": block_count,
        "samples": raw,
        "control_median_seconds": control,
        "candidate_median_seconds": candidate,
        "candidate_speedup_fraction": 1.0 - candidate / control,
        "control_first": direction(0),
        "candidate_first": direction(1),
    }


def _named_block_samples(
    calls,
    *,
    warmups: int,
    samples: int,
    block_count: int,
) -> dict[str, object]:
    """Measure several fixed-input calls in a position-balanced order."""

    if block_count < 1:
        raise ValueError("timing block count must be positive")
    names = tuple(calls)
    if not names:
        raise ValueError("named timing requires at least one call")

    def evaluate(call) -> float:
        started = time.perf_counter()
        for _ in range(block_count):
            mx.eval(call())
        return (time.perf_counter() - started) / block_count

    for _ in range(warmups):
        for name in names:
            evaluate(calls[name])
    raw = {name: [] for name in names}
    for sample in range(samples):
        rotation = (sample // 2) % len(names)
        order = names[rotation:] + names[:rotation]
        if sample % 2:
            order = tuple(reversed(order))
        for name in order:
            raw[name].append(evaluate(calls[name]))
    return {
        "method": "position-balanced synchronized calls per timing block",
        "block_count": block_count,
        "samples": raw,
        "median_seconds": {name: float(np.median(values)) for name, values in raw.items()},
    }


def _fused_half_component_schedules(schedule) -> dict[str, object]:
    """Return full and work-isolated views of one device fused-half schedule."""

    no_special = {
        "special_work_left_blocks": schedule.special_work_left_blocks[:0],
        "special_work_left_slices": schedule.special_work_left_slices[:0],
        "special_work_right_atoms": schedule.special_work_right_atoms[:0],
        "special_work_lj_enabled": schedule.special_work_lj_enabled[:0],
        "special_work_lj_one_four": schedule.special_work_lj_one_four[:0],
        "special_work_diagonal": schedule.special_work_diagonal[:0],
    }
    no_ordinary = {
        "ordinary_tile_count": 0,
        "ordinary_group_count": 0,
        "ordinary_left_blocks": schedule.ordinary_left_blocks[:0],
        "ordinary_right_atoms": schedule.ordinary_right_atoms[:0],
        "ordinary_half_modes": schedule.ordinary_half_modes[:0],
        "ordinary_group_starts": schedule.ordinary_group_starts[:0],
        "ordinary_group_counts": schedule.ordinary_group_counts[:0],
    }
    return {
        "full": schedule,
        "ordinary_only": replace(schedule, **no_special),
        "special_only": replace(schedule, **no_ordinary),
        "empty": replace(schedule, **no_ordinary, **no_special),
    }


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


def _stage_samples(stage_call, *, warmups: int, samples: int) -> dict[str, object]:
    stage_names = ("pack", "interaction", "scatter")

    def evaluate() -> dict[str, float]:
        stages = stage_call()
        arrays = (
            (stages.packed_posq, stages.packed_lj),
            (stages.ordered_forces,),
            (stages.forces,),
        )
        observed = {}
        for name, values in zip(stage_names, arrays, strict=True):
            started = time.perf_counter()
            mx.eval(*values)
            observed[name] = time.perf_counter() - started
        return observed

    for _ in range(warmups):
        evaluate()
    raw = {name: [] for name in stage_names}
    for _ in range(samples):
        observed = evaluate()
        for name in stage_names:
            raw[name].append(observed[name])
    medians = {name: float(np.median(values)) for name, values in raw.items()}
    return {
        "method": "sequential synchronized lazy-graph stages",
        "samples": raw,
        "median_seconds": medians,
        "median_sum_seconds": float(sum(medians.values())),
    }


def benchmark(
    prepared: Path,
    *,
    architecture: str,
    skin: float,
    control_skin: float | None,
    warmups: int,
    samples: int,
    ordinary_tiles_per_group: int,
    timing_block_count: int,
    canonical_records: bool,
    simdgroups_per_threadgroup: int,
    left_slice_size: int,
    builder_rebuild_samples: int = 3,
) -> dict[str, object]:
    _activate_metal()
    resolved_control_skin = skin if control_skin is None else control_skin
    if not np.isfinite(resolved_control_skin) or resolved_control_skin < 0.0:
        raise ValueError("control skin must be finite and non-negative")
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
        skin=resolved_control_skin,
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
    if architecture == "interaction32":
        schedule = _build_interaction_schedule32(
            positions_np,
            box_lengths,
            search_radius=cutoff + skin,
            lj_exclusion_pairs=exclusion_pairs,
            lj_one_four_pairs=one_four_pairs,
            ordinary_tiles_per_group=ordinary_tiles_per_group,
            left_slice_size=left_slice_size,
        )
        schedule_build_seconds = time.perf_counter() - schedule_started
        device_schedule = _schedule_to_device32(schedule)
        mx.eval(
            device_schedule.atom_order,
            device_schedule.ordinary_left_blocks,
            device_schedule.ordinary_left_slices,
            device_schedule.ordinary_right_atoms,
            device_schedule.ordinary_group_starts,
            device_schedule.ordinary_group_counts,
            device_schedule.special_work_left_blocks,
            device_schedule.special_work_left_slices,
            device_schedule.special_work_right_atoms,
            device_schedule.special_work_lj_enabled,
            device_schedule.special_work_lj_one_four,
            device_schedule.special_work_diagonal,
        )
        candidate_inventory = {
            "block_count": schedule.block_count,
            "ordinary_tile_count": schedule.ordinary_tile_count,
            "ordinary_group_count": schedule.ordinary_group_count,
            "ordinary_tiles_per_group_limit": ordinary_tiles_per_group,
            "canonical_records": canonical_records,
            "simdgroups_per_threadgroup": simdgroups_per_threadgroup,
            "left_slice_size": left_slice_size,
            "special_tile_count": schedule.special_tile_count,
            "special_work_count": schedule.special_work_count,
            "scheduled_pair_lanes": (
                schedule.ordinary_tile_count * left_slice_size * 32
                + schedule.special_work_count * left_slice_size * 32
            ),
            "oracle_build_seconds": schedule_build_seconds,
        }
    elif architecture == "fused_half32":
        base_schedule = _build_interaction_schedule32(
            positions_np,
            box_lengths,
            search_radius=cutoff + skin,
            lj_exclusion_pairs=exclusion_pairs,
            lj_one_four_pairs=one_four_pairs,
            ordinary_tiles_per_group=ordinary_tiles_per_group,
            left_slice_size=16,
        )
        schedule = _fuse_interaction_halves32(
            base_schedule,
            ordinary_tiles_per_group=ordinary_tiles_per_group,
        )
        schedule_build_seconds = time.perf_counter() - schedule_started
        device_schedule = _fused_half_schedule_to_device32(schedule)
        mx.eval(
            device_schedule.atom_order,
            device_schedule.ordinary_left_blocks,
            device_schedule.ordinary_right_atoms,
            device_schedule.ordinary_half_modes,
            device_schedule.ordinary_group_starts,
            device_schedule.ordinary_group_counts,
            device_schedule.special_work_left_blocks,
            device_schedule.special_work_left_slices,
            device_schedule.special_work_right_atoms,
            device_schedule.special_work_lj_enabled,
            device_schedule.special_work_lj_one_four,
            device_schedule.special_work_diagonal,
        )
        base_right_entries = int(
            np.count_nonzero(
                base_schedule.ordinary_right_atoms < base_schedule.padded_atom_count
            )
        )
        candidate_inventory = {
            "block_count": schedule.block_count,
            "ordinary_tile_count": schedule.ordinary_tile_count,
            "ordinary_group_count": schedule.ordinary_group_count,
            "ordinary_tiles_per_group_limit": ordinary_tiles_per_group,
            "base_half_right_atom_entries": base_right_entries,
            "fused_right_atom_entries": schedule.ordinary_right_entry_count,
            "right_atomic_candidate_reduction_fraction": (
                0.0
                if base_right_entries == 0
                else 1.0 - schedule.ordinary_right_entry_count / base_right_entries
            ),
            "ordinary_logical_pair_lanes": schedule.ordinary_logical_pair_lanes,
            "special_work_count": schedule.special_work_count,
            "scheduled_pair_lanes": (
                schedule.ordinary_logical_pair_lanes
                + schedule.special_work_count * 16 * 32
            ),
            "simdgroups_per_threadgroup": simdgroups_per_threadgroup,
            "oracle_build_seconds": schedule_build_seconds,
        }
    elif architecture == "device_fused_half32":
        if builder_rebuild_samples < 1:
            raise ValueError("builder_rebuild_samples must be positive")

        def evaluate_schedule_payload(schedule) -> None:
            mx.eval(
                schedule.atom_order,
                schedule.ordinary_left_blocks,
                schedule.ordinary_right_atoms,
                schedule.ordinary_half_modes,
                schedule.ordinary_group_starts,
                schedule.ordinary_group_counts,
                schedule.special_work_left_blocks,
                schedule.special_work_left_slices,
                schedule.special_work_right_atoms,
                schedule.special_work_lj_enabled,
                schedule.special_work_lj_one_four,
                schedule.special_work_diagonal,
            )

        attempt = _try_build_device_fused_half_schedule32(
            system.positions,
            box_lengths,
            search_radius=cutoff + skin,
            capacity=None,
            generation_value=1,
            lj_exclusion_pairs=exclusion_pairs,
            lj_one_four_pairs=one_four_pairs,
            ordinary_tiles_per_group=ordinary_tiles_per_group,
        )
        if not attempt.overflow or attempt.schedule is not None:
            raise RuntimeError("initial lifecycle probe must request reserved capacity")
        attempt = _retry_device_fused_half_schedule32(attempt)
        if attempt.overflow or attempt.schedule is None:
            raise RuntimeError("reserved lifecycle retry did not produce a schedule")
        device_schedule = attempt.schedule
        inventory = attempt.inventory
        capacity = attempt.recommended_capacity
        evaluate_schedule_payload(device_schedule)
        schedule_build_seconds = time.perf_counter() - schedule_started
        schedule = device_schedule
        retained_rebuild_seconds = []
        for sample in range(builder_rebuild_samples):
            rebuild_started = time.perf_counter()
            rebuilt = _try_build_device_fused_half_schedule32(
                system.positions,
                box_lengths,
                search_radius=cutoff + skin,
                capacity=capacity,
                generation_value=sample + 2,
                lj_exclusion_pairs=exclusion_pairs,
                lj_one_four_pairs=one_four_pairs,
                ordinary_tiles_per_group=ordinary_tiles_per_group,
            )
            if rebuilt.overflow or rebuilt.schedule is None:
                raise RuntimeError("retained device capacity overflowed unchanged positions")
            evaluate_schedule_payload(rebuilt.schedule)
            retained_rebuild_seconds.append(time.perf_counter() - rebuild_started)
        reserved_schedule_bytes = (
            capacity.ordinary_tiles * (4 + 32 * 4 + 4)
            + capacity.ordinary_groups * (2 * 4)
            + capacity.special_tiles * (2 * 4)
            + capacity.special_work * (4 + 4 + 32 * 4 + 32 * 4 + 32 * 4 + 4)
        )
        candidate_inventory = {
            "block_count": inventory.geometry.block_count,
            "ordinary_tile_count": inventory.ordinary_tile_count,
            "ordinary_group_count": inventory.ordinary_group_count,
            "ordinary_tiles_per_group_limit": ordinary_tiles_per_group,
            "ordinary_right_atom_entries": inventory.right_entry_count,
            "ordinary_logical_pair_lanes": inventory.logical_pair_lanes,
            "special_tile_count": inventory.special_tile_count,
            "special_work_count": 2 * inventory.special_tile_count,
            "sparse_special_code_entries": int(inventory.special.block_codes.shape[0]),
            "scheduled_pair_lanes": (
                inventory.logical_pair_lanes
                + 2 * inventory.special_tile_count * 16 * 32
            ),
            "capacity": {
                "ordinary_tiles": capacity.ordinary_tiles,
                "ordinary_groups": capacity.ordinary_groups,
                "special_tiles": capacity.special_tiles,
                "special_work": capacity.special_work,
                "reserve_fraction": 0.25,
                "allocation_quantum": 64,
                "estimated_schedule_bytes": reserved_schedule_bytes,
            },
            "generation": device_schedule.generation.value,
            "simdgroups_per_threadgroup": simdgroups_per_threadgroup,
            "device_build_seconds": schedule_build_seconds,
            "retained_capacity_rebuild": {
                "samples_seconds": retained_rebuild_seconds,
                "median_seconds": float(np.median(retained_rebuild_seconds)),
            },
            "host_payload_arrays": 0,
            "scalar_inventory_materializations": 1,
        }
    elif architecture == "owner_compute32":
        schedule = _build_owner_compute_schedule32(
            positions_np,
            box_lengths,
            search_radius=cutoff + skin,
            lj_exclusion_pairs=exclusion_pairs,
            lj_one_four_pairs=one_four_pairs,
        )
        schedule_build_seconds = time.perf_counter() - schedule_started
        device_schedule = _owner_schedule_to_device32(schedule)
        mx.eval(
            device_schedule.atom_order,
            device_schedule.owner_offsets,
            device_schedule.right_atoms,
            device_schedule.topology_offsets,
            device_schedule.topology_neighbors,
            device_schedule.topology_classes,
        )
        candidate_inventory = {
            "block_count": schedule.block_count,
            "right_atom_entries": int(schedule.right_atoms.shape[0]),
            "scheduled_pair_lanes": schedule.scheduled_pair_lanes,
            "topology_directed_entries": int(schedule.topology_neighbors.shape[0]),
            "simdgroups_per_threadgroup": simdgroups_per_threadgroup,
            "oracle_build_seconds": schedule_build_seconds,
        }
    else:
        raise ValueError(f"unsupported interaction architecture: {architecture}")

    def control_call():
        return nonbonded._direct_forces_from_binding(system.positions, tile_binding)

    if architecture == "interaction32":

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
                _canonical_records=canonical_records,
                _simdgroups_per_threadgroup=simdgroups_per_threadgroup,
            )

        def stage_call():
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
                _return_stages=True,
                _canonical_records=canonical_records,
                _simdgroups_per_threadgroup=simdgroups_per_threadgroup,
            )

    elif architecture == "owner_compute32":

        def candidate_call():
            return _owner_compute32_direct_force_only(
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
                _simdgroups_per_threadgroup=simdgroups_per_threadgroup,
            )

    else:

        def candidate_for_schedule(selected_schedule):
            return _fused_half32_direct_force_only(
                system.positions,
                selected_schedule,
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
                atom_type_ids=tile_binding.tile_atom_type_ids,
                nbfix_type_sigma=tile_binding.tile_nbfix_type_sigma,
                nbfix_type_epsilon=tile_binding.tile_nbfix_type_epsilon,
                nbfix_type_count=tile_binding.tile_nbfix_type_count,
                _simdgroups_per_threadgroup=simdgroups_per_threadgroup,
            )

        def candidate_call():
            return candidate_for_schedule(device_schedule)

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
    steady_state = _sequential_block_samples(
        control_call,
        candidate_call,
        warmups=max(1, warmups // 2),
        samples=samples,
        block_count=timing_block_count,
    )
    marginal = _marginal_samples(
        control_call,
        candidate_call,
        warmups=max(1, warmups // 2),
        samples=samples,
    )
    stage_timing = None
    component_timing = None
    if architecture == "interaction32":
        stage_timing = _stage_samples(
            stage_call,
            warmups=max(1, warmups // 2),
            samples=samples,
        )
    elif architecture in {"fused_half32", "device_fused_half32"}:
        component_schedules = _fused_half_component_schedules(device_schedule)
        component_timing = _named_block_samples(
            {
                name: (
                    lambda selected_schedule=selected_schedule: candidate_for_schedule(
                        selected_schedule
                    )
                )
                for name, selected_schedule in component_schedules.items()
            },
            warmups=max(1, warmups // 2),
            samples=samples,
            block_count=timing_block_count,
        )
    control_median = float(np.median(timings["control"]))
    candidate_median = float(np.median(timings["candidate"]))
    return {
        "schema": "mlx_atomistic.interaction32_force_benchmark.v6",
        "prepared": str(prepared),
        "architecture": architecture,
        "atom_count": schedule.atom_count,
        "cutoff_angstrom": cutoff,
        "skin_angstrom": skin,
        "search_radius_angstrom": cutoff + skin,
        "nonbonded_surface": {
            "nbfix_type_table": tile_binding.tile_atom_type_ids is not None,
            "nbfix_type_count": tile_binding.tile_nbfix_type_count,
        },
        "production": {
            "skin_angstrom": resolved_control_skin,
            "block_size": tiles.block_size,
            "tile_count": tiles.tile_count,
            "force_group_count": tiles.force_group_count,
            "build_seconds": tile_build_seconds,
        },
        architecture: candidate_inventory,
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
        "steady_state_timing": steady_state,
        "marginal_timing": marginal,
        "stage_timing": stage_timing,
        "component_timing": component_timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--architecture",
        choices=(
            "interaction32",
            "fused_half32",
            "device_fused_half32",
            "owner_compute32",
        ),
        default="interaction32",
    )
    parser.add_argument("--skin", type=float, default=5.5)
    parser.add_argument(
        "--control-skin",
        type=float,
        help=(
            "Production tile reference skin. Defaults to --skin; set it independently "
            "when screening an inner schedule against a validated outer schedule."
        ),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--ordinary-tiles-per-group", type=int, default=3)
    parser.add_argument("--timing-block-count", type=int, default=8)
    parser.add_argument(
        "--canonical-records",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--simdgroups-per-threadgroup", type=int, default=4)
    parser.add_argument("--left-slice-size", type=int, default=16)
    parser.add_argument("--builder-rebuild-samples", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.prepared,
        architecture=args.architecture,
        skin=args.skin,
        control_skin=args.control_skin,
        warmups=args.warmups,
        samples=args.samples,
        ordinary_tiles_per_group=args.ordinary_tiles_per_group,
        timing_block_count=args.timing_block_count,
        canonical_records=args.canonical_records,
        simdgroups_per_threadgroup=args.simdgroups_per_threadgroup,
        left_slice_size=args.left_slice_size,
        builder_rebuild_samples=args.builder_rebuild_samples,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
