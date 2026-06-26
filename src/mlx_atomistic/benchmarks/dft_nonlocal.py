"""Benchmark DFT nonlocal pseudopotential application."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft import (
    DenseHamiltonianReference,
    DFTSystem,
    Ion,
    IonCollection,
    KohnShamOperator,
    NonlocalPseudopotentialOperator,
    SCFConfig,
    read_upf,
    run_scf,
)
from mlx_atomistic.runtime import get_runtime_info


def _parse_grid(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        msg = "--grid must use the form nx,ny,nz"
        raise ValueError(msg)
    return tuple(int(part) for part in parts)


def run_case(*, grid_shape: tuple[int, int, int], iterations: int, upf_path: str | Path) -> dict:
    """Run one compact nonlocal benchmark case."""

    upf = read_upf(upf_path)
    system = DFTSystem(
        cell=(8.0, 8.0, 8.0),
        grid_shape=grid_shape,
        ions=IonCollection([Ion("Si", (4.0, 4.0, 4.0), upf)]),
    )
    result = run_scf(
        system,
        config=SCFConfig(max_iterations=iterations, solver="dense", seed=13),
    )
    nonlocal_operator = NonlocalPseudopotentialOperator.from_ions(system.ions, system.grid)
    operator = KohnShamOperator.from_density(
        system.grid,
        system.pseudopotential.field(system.grid),
        result.density,
        nonlocal_operator=nonlocal_operator,
    )
    trial = result.orbitals[0]
    start = perf_counter()
    applied = operator.apply_hamiltonian(trial)
    mx.eval(applied)
    apply_ms = (perf_counter() - start) * 1000.0
    dense = DenseHamiltonianReference(operator)
    matrix = dense.matrix()
    dense_applied = matrix @ np.array(trial).reshape(system.grid.size)
    error = float(np.max(np.abs(dense_applied - np.array(applied).reshape(system.grid.size))))
    return {
        "case": "upf-si-nonlocal",
        "grid_shape": list(grid_shape),
        "grid_points": system.grid.size,
        "projector_count": nonlocal_operator.projectors.count,
        "nonlocal_applied": result.nonlocal_applied,
        "final_energy": result.total_energy,
        "nonlocal_energy": result.energy_by_term["nonlocal_pseudopotential"],
        "operator_apply_ms": apply_ms,
        "dense_vs_operator_max_error": error,
        "timings": result.timings,
    }


def _blocked_case(
    *,
    grid_shape: tuple[int, int, int],
    iterations: int,
    blocker: str,
) -> dict:
    return {
        "case": "upf-si-nonlocal",
        "status": "blocked",
        "blocker": blocker,
        "grid_shape": list(grid_shape),
        "grid_points": int(np.prod(grid_shape)),
        "iterations_requested": iterations,
        "projector_count": 0,
        "nonlocal_applied": False,
        "final_energy": None,
        "nonlocal_energy": None,
        "operator_apply_ms": None,
        "dense_vs_operator_max_error": None,
        "timings": None,
    }


def build_payload(
    *,
    grid_shape: tuple[int, int, int] = (4, 4, 4),
    iterations: int = 1,
    upf_path: str | Path | None = None,
) -> dict:
    """Run nonlocal benchmark smoke."""

    case = (
        _blocked_case(
            grid_shape=grid_shape,
            iterations=iterations,
            blocker="nonlocal benchmark requires an explicit --upf pseudopotential path",
        )
        if upf_path is None
        else run_case(grid_shape=grid_shape, iterations=iterations, upf_path=upf_path)
    )
    return {
        "runtime": asdict(get_runtime_info()),
        "cases": [case],
        "case_count": 1,
        "status": case.get("status", "ok"),
        "blocker": case.get("blocker"),
        **case,
    }


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
    parser.add_argument("--grid", default="4,4,4")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--upf", type=Path, default=None, help="UPF file with nonlocal projectors.")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        iterations=args.iterations,
        upf_path=args.upf,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    if payload["status"] == "blocked":
        print(f"dft_nonlocal blocked: {payload['blocker']}")
    else:
        print(
            f"dft_nonlocal grid={tuple(payload['grid_shape'])} "
            f"projectors={payload['projector_count']} energy={payload['final_energy']:.8g}"
        )


if __name__ == "__main__":
    main()
