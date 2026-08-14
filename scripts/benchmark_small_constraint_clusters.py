#!/usr/bin/env python3
"""Benchmark the small-component Metal solver against vectorized constraints."""

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
from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.neighbors import _MLX_MD_CACHE_LIMIT_BYTES


def _activate_metal() -> None:
    os.environ["MLX_ATOMISTIC_DEVICE"] = "gpu"
    device = mx.Device(mx.gpu, 0)
    mx.set_default_device(device)
    mx.set_default_stream(mx.new_stream(device))
    mx.set_cache_limit(_MLX_MD_CACHE_LIMIT_BYTES)
    probe = mx.array([1.0], dtype=mx.float32) + 1.0
    mx.eval(probe)


def _control_position_step(
    constraints: DistanceConstraints,
    reference: mx.array,
    predicted: mx.array,
    masses: mx.array,
    cell,
) -> mx.array:
    return constraints._generic_position_step(
        reference,
        predicted,
        masses,
        cell,
    )


def _control_velocities(
    constraints: DistanceConstraints,
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    cell,
) -> mx.array:
    return constraints._generic_velocities(
        positions,
        velocities,
        masses,
        cell,
    )


def _timed_block(call, count: int) -> float:
    started = time.perf_counter()
    for _ in range(count):
        mx.eval(call())
    return (time.perf_counter() - started) / count


def _interleaved_samples(
    control_call,
    candidate_call,
    *,
    warmups: int,
    samples: int,
    block_count: int,
) -> dict[str, object]:
    calls = {"control": control_call, "candidate": candidate_call}
    for _ in range(warmups):
        for call in calls.values():
            _timed_block(call, block_count)
    raw = {"control": [], "candidate": []}
    for sample in range(samples):
        order = ("control", "candidate") if sample % 2 == 0 else ("candidate", "control")
        for name in order:
            raw[name].append(_timed_block(calls[name], block_count))
    control = float(np.median(raw["control"]))
    candidate = float(np.median(raw["candidate"]))
    return {
        "block_count": block_count,
        "samples": raw,
        "control_median_seconds": control,
        "candidate_median_seconds": candidate,
        "candidate_speedup_fraction": 1.0 - candidate / control,
    }


def _delta_report(candidate: mx.array, control: mx.array) -> dict[str, float | bool]:
    delta = np.asarray(candidate) - np.asarray(control)
    return {
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "max_delta": float(np.max(np.abs(delta))),
        "finite": bool(np.all(np.isfinite(delta))),
    }


def benchmark(
    prepared: Path,
    *,
    warmups: int,
    samples: int,
    block_count: int,
) -> dict[str, object]:
    _activate_metal()
    artifact = load_prepared_mlx_artifact(prepared, require_production=True)
    system, _, constraints = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if not isinstance(constraints, DistanceConstraints):
        raise ValueError("small-component benchmark requires DistanceConstraints")
    if not constraints._small_cluster_supported:
        raise ValueError("constraint graph does not fit the small-component solver")
    reference = system.positions
    predicted = reference + 0.004 * system.velocities
    velocities = system.velocities
    masses = system.masses

    def control_position_call():
        return _control_position_step(
            constraints,
            reference,
            predicted,
            masses,
            system.cell,
        )

    def candidate_position_call():
        return constraints._apply_position_step_unchecked(
            reference,
            predicted,
            masses,
            system.cell,
        )

    control_positions = control_position_call()
    candidate_positions = candidate_position_call()
    mx.eval(control_positions, candidate_positions)

    def control_velocity_call():
        return _control_velocities(
            constraints,
            candidate_positions,
            velocities,
            masses,
            system.cell,
        )

    def candidate_velocity_call():
        return constraints.apply_velocities(
            candidate_positions,
            velocities,
            masses,
            system.cell,
        )

    control_velocities = control_velocity_call()
    candidate_velocities = candidate_velocity_call()
    mx.eval(control_velocities, candidate_velocities)
    candidate_error = constraints.max_error(candidate_positions, system.cell)
    control_error = constraints.max_error(control_positions, system.cell)
    mx.eval(candidate_error, control_error)
    return {
        "schema": "mlx_atomistic.small_constraint_cluster_benchmark.v1",
        "prepared": str(prepared),
        "atom_count": int(reference.shape[0]),
        "constraint_count": int(constraints.pairs.shape[0]),
        "cluster_count": int(constraints._small_cluster_atoms.shape[0]),
        "max_iterations": constraints.max_iterations,
        "position_parity": _delta_report(candidate_positions, control_positions),
        "velocity_parity": _delta_report(candidate_velocities, control_velocities),
        "candidate_max_constraint_error": float(np.asarray(candidate_error)),
        "control_max_constraint_error": float(np.asarray(control_error)),
        "position_timing": _interleaved_samples(
            control_position_call,
            candidate_position_call,
            warmups=warmups,
            samples=samples,
            block_count=block_count,
        ),
        "velocity_timing": _interleaved_samples(
            control_velocity_call,
            candidate_velocity_call,
            warmups=warmups,
            samples=samples,
            block_count=block_count,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--block-count", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.prepared,
        warmups=args.warmups,
        samples=args.samples,
        block_count=args.block_count,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
