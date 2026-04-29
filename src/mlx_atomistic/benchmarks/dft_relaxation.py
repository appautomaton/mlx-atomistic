"""Benchmark DFT relaxation and stress diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from mlx_atomistic.dft import (
    GeometryOptimizationConfig,
    SCFConfig,
    finite_difference_stress,
    geometry_demo_system,
    optimize_geometry,
)
from mlx_atomistic.runtime import get_runtime_info


def run_case() -> dict:
    """Run one compact relaxation/stress case."""

    system = geometry_demo_system("gaussian-dimer", grid_shape=(4, 4, 4))
    scf_config = SCFConfig(max_iterations=20, solver="dense", seed=17, convergence_mode="either")
    relaxation = optimize_geometry(
        system,
        config=GeometryOptimizationConfig(max_steps=1, scf_config=scf_config),
    )
    stress = finite_difference_stress(system, config=scf_config)
    return {
        "status": relaxation.status,
        "steps_completed": len(relaxation.steps),
        "final_energy": relaxation.final_energy,
        "final_max_force": relaxation.final_max_force,
        "stress": stress.to_dict(),
    }


def build_payload() -> dict:
    """Run relaxation benchmark smoke."""

    case = run_case()
    return {"runtime": asdict(get_runtime_info()), "cases": [case], "case_count": 1, **case}


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    flattened = [_csv_row(row) for row in rows]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


def _csv_row(row: dict) -> dict:
    return {
        key: json.dumps(value) if isinstance(value, dict | list) else value
        for key, value in row.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload()
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"dft_relaxation status={payload['status']} energy={payload['final_energy']:.8g}")


if __name__ == "__main__":
    main()
