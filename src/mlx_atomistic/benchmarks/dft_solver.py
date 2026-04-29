"""Benchmark dense reference versus Davidson DFT solvers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from mlx_atomistic.dft import (
    DavidsonDiagonalizer,
    DenseHamiltonianReference,
    DFTSystem,
    KohnShamOperator,
    SCFConfig,
    run_scf,
)
from mlx_atomistic.runtime import get_runtime_info


def _parse_grid(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        msg = "--grid must use the form nx,ny,nz"
        raise ValueError(msg)
    return tuple(int(part) for part in parts)


def run_case(*, grid_shape: tuple[int, int, int], iterations: int) -> dict:
    """Run one solver benchmark case."""

    system = DFTSystem.one_center(grid_shape=grid_shape)
    result = run_scf(system, config=SCFConfig(max_iterations=iterations, solver="dense", seed=7))
    operator = KohnShamOperator.from_density(
        system.grid,
        system.pseudopotential.field(system.grid),
        result.density,
    )
    start = perf_counter()
    dense = DenseHamiltonianReference(operator).diagonalize(1)
    dense_ms = (perf_counter() - start) * 1000.0
    start = perf_counter()
    davidson = DavidsonDiagonalizer().solve(operator, n_orbitals=1)
    davidson_ms = (perf_counter() - start) * 1000.0
    return {
        "grid_shape": list(grid_shape),
        "grid_points": system.grid.size,
        "dense_ms": dense_ms,
        "davidson_ms": davidson_ms,
        "dense_eigenvalue": float(np.array(dense.eigenvalues)[0]),
        "davidson_eigenvalue": float(np.array(davidson.eigenvalues)[0]),
        "eigenvalue_error": float(
            abs(np.array(dense.eigenvalues)[0] - np.array(davidson.eigenvalues)[0])
        ),
        "davidson_residual": float(np.max(np.array(davidson.residuals))),
        "davidson_metadata": davidson.metadata,
    }


def build_payload(*, grid_shape: tuple[int, int, int] = (4, 4, 4), iterations: int = 1) -> dict:
    """Run solver benchmark smoke."""

    case = run_case(grid_shape=grid_shape, iterations=iterations)
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
    parser.add_argument("--grid", default="4,4,4")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(grid_shape=_parse_grid(args.grid), iterations=args.iterations)
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(
        f"dft_solver grid={tuple(payload['grid_shape'])} "
        f"dense={payload['dense_ms']:.3f}ms davidson={payload['davidson_ms']:.3f}ms"
    )


if __name__ == "__main__":
    main()
