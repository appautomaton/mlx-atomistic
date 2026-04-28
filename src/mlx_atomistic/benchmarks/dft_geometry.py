"""Benchmark fixed-cell DFT geometry optimization workflows."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mlx_atomistic.dft import (
    GeometryOptimizationConfig,
    SCFConfig,
    geometry_demo_system,
    optimize_geometry,
)
from mlx_atomistic.runtime import get_runtime_info


def _parse_grid(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        msg = "--grid must use the form nx,ny,nz"
        raise ValueError(msg)
    shape = tuple(int(part) for part in parts)
    if any(item <= 0 for item in shape):
        msg = "--grid dimensions must be positive"
        raise ValueError(msg)
    return shape


def _systems(value: str | None) -> list[str]:
    if value is None or value == "all":
        return ["gaussian-dimer", "gth-h2", "upf-si2"]
    names = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"gaussian-dimer", "gth-h2", "upf-si2"}
    if any(name not in allowed for name in names):
        msg = "--systems may contain gaussian-dimer, gth-h2, upf-si2, or all"
        raise ValueError(msg)
    return names


def run_case(
    *,
    system_name: str,
    grid_shape: tuple[int, int, int],
    steps: int,
    optimizer: str,
) -> dict:
    """Run one DFT geometry benchmark case."""

    system = geometry_demo_system(system_name, grid_shape=grid_shape)
    config = GeometryOptimizationConfig(
        max_steps=steps,
        optimizer=optimizer,  # type: ignore[arg-type]
        scf_config=SCFConfig(
            max_iterations=20,
            tolerance=1e-8,
            mixing=0.45,
            solver="dense",
            seed=37,
            convergence_mode="either",
            record_timing=True,
        ),
    )
    start = perf_counter()
    result = optimize_geometry(system, config=config)
    elapsed_ms = (perf_counter() - start) * 1000.0
    scf_iterations = sum(step.scf_iterations for step in result.steps)
    return {
        "case": system_name,
        "grid_shape": list(grid_shape),
        "grid_points": system.grid.size,
        "optimizer": optimizer,
        "steps_requested": steps,
        "steps_completed": len(result.steps),
        "status": result.status,
        "converged": result.converged,
        "final_energy": result.final_energy,
        "final_max_force": result.final_max_force,
        "ms_total": elapsed_ms,
        "ms_per_step": elapsed_ms / max(len(result.steps), 1),
        "scf_iterations": scf_iterations,
        "history": [step.to_dict() for step in result.steps],
        "timings": None if result.final_scf is None else result.final_scf.timings,
    }


def build_payload(
    *,
    grid_shape: tuple[int, int, int] = (4, 4, 4),
    steps: int = 2,
    optimizer: str = "lbfgs",
    systems: str | None = None,
) -> dict:
    """Run compact DFT geometry benchmark cases."""

    cases = [
        run_case(
            system_name=name,
            grid_shape=grid_shape,
            steps=steps,
            optimizer=optimizer,
        )
        for name in _systems(systems)
    ]
    return {
        "runtime": asdict(get_runtime_info()),
        "cases": cases,
        "case_count": len(cases),
        "grid_shape": list(grid_shape),
        "steps_requested": steps,
        "optimizer": optimizer,
    }


def _csv_row(case: dict) -> dict:
    return {
        key: json.dumps(value) if isinstance(value, dict | list) else value
        for key, value in case.items()
    }


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    flattened = [_csv_row(row) for row in rows]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="4,4,4", help="Grid shape as nx,ny,nz.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--systems", default="all")
    parser.add_argument("--optimizer", choices=["lbfgs", "steepest_descent"], default="lbfgs")
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        msg = "--steps must be positive"
        raise ValueError(msg)

    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        steps=args.steps,
        optimizer=args.optimizer,
        systems=args.systems,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    for case in payload["cases"]:
        print(
            f"dft_geometry case={case['case']} grid={tuple(case['grid_shape'])} "
            f"optimizer={case['optimizer']} steps={case['steps_completed']} "
            f"status={case['status']} ms/step={case['ms_per_step']:.3f} "
            f"energy={_format_float(case['final_energy'])} "
            f"max_force={_format_float(case['final_max_force'])}"
        )


def _format_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.8g}"


if __name__ == "__main__":
    main()
