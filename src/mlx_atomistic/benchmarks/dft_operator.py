"""Benchmark DFT Kohn-Sham operator numerics."""

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
    KohnShamOperator,
    SCFConfig,
    SubspaceDiagonalizer,
    run_scf,
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


def _grid_shapes(grid: str, sizes: str | None) -> list[tuple[int, int, int]]:
    if sizes is None:
        return [_parse_grid(grid)]
    parsed = [int(item) for item in sizes.split(",") if item]
    if any(item <= 0 for item in parsed):
        msg = "--sizes values must be positive"
        raise ValueError(msg)
    return [(size, size, size) for size in parsed]


def _system_for_grid(grid_shape: tuple[int, int, int]) -> DFTSystem:
    cell_length = 8.0
    return DFTSystem.one_center(
        cell=(cell_length, cell_length, cell_length),
        grid_shape=grid_shape,
        center=(cell_length / 2.0, cell_length / 2.0, cell_length / 2.0),
        electron_count=2.0,
        amplitude=-3.0,
        width=0.9,
    )


def run_case(
    *,
    grid_shape: tuple[int, int, int],
    iterations: int,
    n_orbitals: int,
) -> dict:
    """Run one Kohn-Sham operator benchmark case."""

    system = _system_for_grid(grid_shape)
    config = SCFConfig(
        max_iterations=iterations,
        tolerance=1e-8,
        solver="dense" if system.grid.size <= 512 else "gradient",
        seed=23,
        record_timing=True,
    )
    result = run_scf(system, config=config, n_orbitals=n_orbitals)
    start = perf_counter()
    operator = KohnShamOperator.from_density(
        system.grid,
        system.pseudopotential.field(system.grid),
        result.density,
    )
    mx.eval(operator.effective_potential)
    operator_build_ms = (perf_counter() - start) * 1000.0

    trial = result.orbitals[0]
    reference = DenseHamiltonianReference(operator)
    start = perf_counter()
    matrix = reference.matrix()
    dense_build_ms = (perf_counter() - start) * 1000.0
    flat_trial = np.array(trial, dtype=np.complex128).reshape(system.grid.size)

    start = perf_counter()
    dense_applied = matrix @ flat_trial
    dense_matvec_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    operator_applied = operator.apply_hamiltonian(trial)
    mx.eval(operator_applied)
    operator_apply_ms = (perf_counter() - start) * 1000.0

    operator_applied_np = np.array(operator_applied, dtype=np.complex128).reshape(system.grid.size)
    dense_vs_operator_max_error = float(np.max(np.abs(dense_applied - operator_applied_np)))

    start = perf_counter()
    eigenvalues, _ = np.linalg.eigh(matrix)
    dense_diagonalize_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    subspace = SubspaceDiagonalizer(tolerance=1e-5).solve(operator, n_orbitals=n_orbitals)
    subspace_solve_ms = (perf_counter() - start) * 1000.0

    return {
        "grid_shape": list(grid_shape),
        "grid_points": system.grid.size,
        "iterations_requested": iterations,
        "iterations_completed": result.iterations,
        "n_orbitals": n_orbitals,
        "scf_status": result.status,
        "fft_backend": result.fft_backend,
        "operator_build_ms": operator_build_ms,
        "operator_apply_ms": operator_apply_ms,
        "dense_build_ms": dense_build_ms,
        "dense_matvec_ms": dense_matvec_ms,
        "dense_diagonalize_ms": dense_diagonalize_ms,
        "subspace_solve_ms": subspace_solve_ms,
        "dense_vs_operator_max_error": dense_vs_operator_max_error,
        "lowest_dense_eigenvalue": float(eigenvalues[0]),
        "lowest_subspace_eigenvalue": float(np.array(subspace.eigenvalues)[0]),
        "max_subspace_residual": float(np.max(np.array(subspace.residuals))),
        "subspace_orthonormality_error": subspace.orthonormality_error,
        "scf_timings": result.timings,
        "scf_orbital_residual_max": float(np.max(np.array(result.orbital_residuals))),
        "scf_orthonormality_error": result.orthonormality_error,
    }


def build_payload(
    *,
    grid_shape: tuple[int, int, int] = (4, 4, 4),
    sizes: str | None = None,
    iterations: int = 2,
    n_orbitals: int = 1,
) -> dict:
    """Run operator benchmark cases and return JSON-safe metadata."""

    cases = [
        run_case(grid_shape=shape, iterations=iterations, n_orbitals=n_orbitals)
        for shape in _grid_shapes(",".join(str(item) for item in grid_shape), sizes)
    ]
    first = cases[0]
    return {
        "runtime": asdict(get_runtime_info()),
        "cases": cases,
        "case_count": len(cases),
        "grid_shape": first["grid_shape"],
        "grid_points": first["grid_points"],
        "iterations_requested": first["iterations_requested"],
        "iterations_completed": first["iterations_completed"],
        "fft_backend": first["fft_backend"],
        "dense_vs_operator_max_error": first["dense_vs_operator_max_error"],
        "operator_apply_ms": first["operator_apply_ms"],
        "dense_build_ms": first["dense_build_ms"],
        "dense_diagonalize_ms": first["dense_diagonalize_ms"],
        "subspace_solve_ms": first["subspace_solve_ms"],
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
    parser.add_argument("--sizes", default=None, help="Optional cubic grid sizes, e.g. 4,6,8.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--orbitals", type=int, default=1)
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        msg = "--iterations must be positive"
        raise ValueError(msg)
    if args.orbitals <= 0:
        msg = "--orbitals must be positive"
        raise ValueError(msg)

    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        sizes=args.sizes,
        iterations=args.iterations,
        n_orbitals=args.orbitals,
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
            f"dft_operator grid={tuple(case['grid_shape'])} points={case['grid_points']} "
            f"operator_apply_ms={case['operator_apply_ms']:.3f} "
            f"dense_build_ms={case['dense_build_ms']:.3f} "
            f"dense_diag_ms={case['dense_diagonalize_ms']:.3f} "
            f"max_error={case['dense_vs_operator_max_error']:.3g}"
        )


if __name__ == "__main__":
    main()
