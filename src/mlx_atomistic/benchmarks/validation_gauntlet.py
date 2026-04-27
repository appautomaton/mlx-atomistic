"""Run finite-difference force validation across supported force terms."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.validation import (
    run_force_validation_suite,
    summarize_validation_results,
)


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_payload(
    *,
    seed: int,
    cases_per_term: int,
    epsilon: float,
    tolerance: float,
) -> dict:
    """Run the validation suite and return a CLI-friendly payload."""

    results = run_force_validation_suite(
        seed=seed,
        cases_per_term=cases_per_term,
        epsilon=epsilon,
        tolerance=tolerance,
    )
    return {
        "runtime": asdict(get_runtime_info()),
        "summary": summarize_validation_results(results),
        "cases": [result.to_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cases-per-term", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        seed=args.seed,
        cases_per_term=args.cases_per_term,
        epsilon=args.epsilon,
        tolerance=args.tolerance,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    summary = payload["summary"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    print(
        f"validation cases={summary['total_cases']} passed={summary['passed_cases']} "
        f"failed={summary['failed_cases']} max_error={summary['max_abs_error']:.6g}"
    )
    for case in payload["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(
            f"{status:4s} {case['case_name']:12s} term={case['term_name']:9s} "
            f"max_error={case['max_abs_error']:.6g} rms={case['rms_abs_error']:.6g} "
            f"coord=({case['failing_atom']},{case['failing_axis']})"
        )


if __name__ == "__main__":
    main()
