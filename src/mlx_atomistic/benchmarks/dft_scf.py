"""Benchmark the DFT SCF prototype."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from mlx_atomistic.dft import DFTSystem, SCFConfig, fft3, reciprocal_to_real, run_scf
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


def _mixers(value: str) -> list[str]:
    if value == "both":
        return ["linear", "diis"]
    return [value]


def _fft_probe_ms(shape: tuple[int, int, int]) -> float:
    field = mx.ones(shape, dtype=mx.float32)
    start = perf_counter()
    round_trip = reciprocal_to_real(fft3(field))
    mx.eval(round_trip)
    return (perf_counter() - start) * 1000.0


def run_case(
    *,
    grid_shape: tuple[int, int, int],
    iterations: int,
    solver: str,
    mixer: str,
) -> dict:
    """Run one compact toy SCF benchmark case."""

    cell_length = 8.0
    system = DFTSystem(
        cell=[cell_length, cell_length, cell_length],
        grid_shape=grid_shape,
        electron_count=2.0,
        centers=[[cell_length / 2.0, cell_length / 2.0, cell_length / 2.0]],
        amplitudes=-3.0,
        widths=0.9,
    )
    config = SCFConfig(
        max_iterations=iterations,
        tolerance=1e-8,
        mixing=0.4,
        mixer=mixer,  # type: ignore[arg-type]
        solver=solver,  # type: ignore[arg-type]
        seed=11,
        record_timing=True,
    )
    fft_probe_ms = _fft_probe_ms(grid_shape)
    start = perf_counter()
    result = run_scf(system, config=config)
    elapsed = perf_counter() - start
    row = {
        "grid_shape": list(grid_shape),
        "grid_points": system.grid.size,
        "iterations_requested": iterations,
        "iterations_completed": result.iterations,
        "solver": result.solver,
        "mixer": mixer,
        "fft_backend": result.fft_backend,
        "converged": result.converged,
        "status": result.status,
        "final_energy": result.total_energy,
        "final_residual": result.residual,
        "ms_total": elapsed * 1000.0,
        "ms_per_iteration": elapsed * 1000.0 / max(result.iterations, 1),
        "fft_probe_ms": fft_probe_ms,
        "energy_by_term": result.energy_by_term,
        "timings": result.timings,
    }
    for key, value in result.timings.items():
        row[f"timing_{key}"] = value
    return row


def build_payload(
    *,
    grid_shape: tuple[int, int, int] = (16, 16, 16),
    iterations: int,
    solver: str = "auto",
    mixer: str = "linear",
    sizes: str | None = None,
) -> dict:
    """Run DFT SCF benchmark cases and return JSON-safe metadata."""

    cases = [
        run_case(grid_shape=shape, iterations=iterations, solver=solver, mixer=mixer_name)
        for shape in _grid_shapes(",".join(str(item) for item in grid_shape), sizes)
        for mixer_name in _mixers(mixer)
    ]
    first = cases[0]
    payload = {
        "runtime": asdict(get_runtime_info()),
        "cases": cases,
        "case_count": len(cases),
        "grid_shape": first["grid_shape"],
        "grid_points": first["grid_points"],
        "iterations_requested": first["iterations_requested"],
        "iterations_completed": first["iterations_completed"],
        "solver": first["solver"],
        "mixer": first["mixer"],
        "fft_backend": first["fft_backend"],
        "converged": first["converged"],
        "final_energy": first["final_energy"],
        "final_residual": first["final_residual"],
        "ms_total": first["ms_total"],
        "ms_per_iteration": first["ms_per_iteration"],
        "energy_by_term": first["energy_by_term"],
        "timings": first["timings"],
    }
    return payload


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
    parser.add_argument("--grid", default="16,16,16", help="Grid shape as nx,ny,nz.")
    parser.add_argument("--sizes", default=None, help="Optional cubic grid sizes, e.g. 8,16,24,32.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--solver", choices=["auto", "dense", "gradient"], default="auto")
    parser.add_argument("--mixer", choices=["linear", "diis", "both"], default="linear")
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        msg = "--iterations must be positive"
        raise ValueError(msg)

    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        iterations=args.iterations,
        solver=args.solver,
        mixer=args.mixer,
        sizes=args.sizes,
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
            f"dft_scf grid={tuple(case['grid_shape'])} points={case['grid_points']} "
            f"solver={case['solver']} mixer={case['mixer']} fft={case['fft_backend']} "
            f"iters={case['iterations_completed']} ms/iter={case['ms_per_iteration']:.3f} "
            f"energy={case['final_energy']:.8g} residual={case['final_residual']:.3g}"
        )


if __name__ == "__main__":
    main()
