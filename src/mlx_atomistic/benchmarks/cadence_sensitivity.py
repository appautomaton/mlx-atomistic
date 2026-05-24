"""Compare full-MD timing across reporting, diagnostic, and sync cadences."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from mlx_atomistic.benchmarks import (
    default_benchmark_command,
    get_hardware_info,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)
from mlx_atomistic.benchmarks.md_performance import BenchmarkMode, run_synthetic_case
from mlx_atomistic.nonbonded import DEFAULT_FULL_LOOP_DENSE_THRESHOLD
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class CadenceConfig:
    """One full-loop cadence setting."""

    name: str
    sample_interval: int
    diagnostic_interval: int
    evaluation_interval: int

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CADENCES = (
    CadenceConfig("sparse_output", 100, 100, 25),
    CadenceConfig("high_output_sync", 1, 1, 1),
)


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        msg = "value must contain positive integers"
        raise ValueError(msg)
    return values


def _parse_cadences(value: str) -> tuple[CadenceConfig, ...]:
    cadences: list[CadenceConfig] = []
    for item in value.split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 4:
            msg = "cadences must use name:sample_interval:diagnostic_interval:evaluation_interval"
            raise ValueError(msg)
        name, sample, diagnostic, evaluation = parts
        if not name:
            msg = "cadence name must not be empty"
            raise ValueError(msg)
        cadence = CadenceConfig(name, int(sample), int(diagnostic), int(evaluation))
        if (
            cadence.sample_interval <= 0
            or cadence.diagnostic_interval <= 0
            or cadence.evaluation_interval <= 0
        ):
            msg = "cadence intervals must be positive"
            raise ValueError(msg)
        cadences.append(cadence)
    if len(cadences) < 2:
        msg = "at least two cadences are required"
        raise ValueError(msg)
    return tuple(cadences)


def _ratio(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return value / baseline


def _comparison_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    rows_by_size: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_size.setdefault(int(row["particles"]), []).append(row)

    comparisons: list[dict] = []
    for particles, size_rows in rows_by_size.items():
        baseline = size_rows[0]
        for row in size_rows[1:]:
            comparisons.append(
                {
                    "particles": particles,
                    "atom_count": row["atom_count"],
                    "cadence_name": row["cadence_name"],
                    "baseline_cadence_name": baseline["cadence_name"],
                    "steps_per_s_ratio": _ratio(row["steps_per_s"], baseline["steps_per_s"]),
                    "wall_s_delta": row["wall_s"] - baseline["wall_s"],
                    "materialized_frame_delta": (
                        row["materialized_frame_count"] - baseline["materialized_frame_count"]
                    ),
                    "diagnostic_sync_delta": row["diagnostic_sync_count"]
                    - baseline["diagnostic_sync_count"],
                    "evaluation_sync_delta": row["evaluation_sync_count"]
                    - baseline["evaluation_sync_count"],
                    "force_eval_ms_per_step_delta": (
                        row["force_eval_ms_per_step"] - baseline["force_eval_ms_per_step"]
                    ),
                }
            )
    return comparisons


def build_payload(
    *,
    sizes: tuple[int, ...] = (2000,),
    steps: int = 200,
    dt: float = 0.002,
    mode: BenchmarkMode = "auto",
    dense_threshold: int = DEFAULT_FULL_LOOP_DENSE_THRESHOLD,
    cadences: tuple[CadenceConfig, ...] = DEFAULT_CADENCES,
    neighbor_check_interval: int = 1,
    neighbor_skin: float = 0.4,
) -> dict:
    """Run cadence sensitivity cases and return a durable JSON payload."""

    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if len(cadences) < 2:
        msg = "at least two cadences are required"
        raise ValueError(msg)

    rows: list[dict] = []
    for size in sizes:
        for cadence in cadences:
            result = run_synthetic_case(
                particles=size,
                steps=steps,
                dt=dt,
                mode=mode,
                dense_threshold=dense_threshold,
                sample_interval=cadence.sample_interval,
                diagnostic_interval=cadence.diagnostic_interval,
                evaluation_interval=cadence.evaluation_interval,
                neighbor_check_interval=neighbor_check_interval,
                neighbor_skin=neighbor_skin,
            )
            row = result.to_dict()
            row["cadence_name"] = cadence.name
            row["cadence"] = cadence.to_dict()
            row["sync_materialization_counts"] = {
                "materialized_frame_count": row["materialized_frame_count"],
                "diagnostic_sync_count": row["diagnostic_sync_count"],
                "evaluation_sync_count": row["evaluation_sync_count"],
            }
            rows.append(
                normalize_benchmark_row(
                    row,
                    benchmark_name="cadence_sensitivity",
                    timing_metric="steps_per_s",
                )
            )

    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": "cadence_sensitivity",
        "fixture": "synthetic_lj",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "sizes": list(sizes),
            "steps": steps,
            "dt": dt,
            "mode": mode,
            "dense_threshold": dense_threshold,
            "neighbor_check_interval": neighbor_check_interval,
            "neighbor_skin": neighbor_skin,
            "cadences": [cadence.to_dict() for cadence in cadences],
        },
        "case_count": len(rows),
        "cases": rows,
        "comparisons": _comparison_rows(rows),
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="cadence_sensitivity",
        fixture="synthetic_lj",
        timing_metric="steps_per_s",
        hardware=hardware,
        runtime=runtime,
        step_count=steps,
        finite=all(bool(row["finite"]) for row in rows),
        command=default_benchmark_command("cadence_sensitivity"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="2000")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--mode",
        choices=["auto", "dense", "dynamic-neighbor"],
        default="auto",
    )
    parser.add_argument("--dense-threshold", type=int, default=DEFAULT_FULL_LOOP_DENSE_THRESHOLD)
    parser.add_argument(
        "--cadences",
        default=",".join(
            f"{item.name}:{item.sample_interval}:{item.diagnostic_interval}:"
            f"{item.evaluation_interval}"
            for item in DEFAULT_CADENCES
        ),
    )
    parser.add_argument("--neighbor-check-interval", type=int, default=1)
    parser.add_argument("--neighbor-skin", type=float, default=0.4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        sizes=_parse_ints(args.sizes),
        steps=args.steps,
        dt=args.dt,
        mode=args.mode,
        dense_threshold=args.dense_threshold,
        cadences=_parse_cadences(args.cadences),
        neighbor_check_interval=args.neighbor_check_interval,
        neighbor_skin=args.neighbor_skin,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("cadence,particles,steps,steps_per_s,frames,diagnostics,mx_eval_syncs")
    for row in payload["cases"]:
        counts = row["sync_materialization_counts"]
        print(
            f"{row['cadence_name']},{row['particles']},{row['steps']},"
            f"{row['steps_per_s']:.3f},{counts['materialized_frame_count']},"
            f"{counts['diagnostic_sync_count']},{counts['evaluation_sync_count']}"
        )


if __name__ == "__main__":
    main()
