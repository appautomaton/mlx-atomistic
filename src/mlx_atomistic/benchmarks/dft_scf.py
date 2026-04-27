"""Benchmark the minimal DFT SCF prototype."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from time import perf_counter

from mlx_atomistic.dft import (
    LocalGaussianPseudopotential,
    RealSpaceGrid,
    SCFConfig,
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


def build_payload(
    *,
    grid_shape: tuple[int, int, int],
    iterations: int,
    solver: str = "auto",
) -> dict:
    """Run a compact toy SCF benchmark and return JSON-safe metadata."""

    grid = RealSpaceGrid(grid_shape, [8.0, 8.0, 8.0])
    local = LocalGaussianPseudopotential(
        centers=[[4.0, 4.0, 4.0]],
        amplitudes=-3.0,
        widths=0.9,
    )
    config = SCFConfig(
        max_iterations=iterations,
        tolerance=1e-8,
        mixing=0.4,
        solver=solver,  # type: ignore[arg-type]
        seed=11,
    )
    start = perf_counter()
    result = run_scf(grid, local, electron_count=2.0, config=config)
    elapsed = perf_counter() - start
    return {
        "runtime": asdict(get_runtime_info()),
        "grid_shape": list(grid_shape),
        "grid_points": grid.size,
        "iterations_requested": iterations,
        "iterations_completed": result.iterations,
        "solver": result.solver,
        "fft_backend": result.fft_backend,
        "converged": result.converged,
        "final_energy": result.total_energy,
        "final_residual": result.residual,
        "ms_total": elapsed * 1000.0,
        "ms_per_iteration": elapsed * 1000.0 / max(result.iterations, 1),
        "energy_by_term": result.energy_by_term,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="16,16,16", help="Grid shape as nx,ny,nz.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--solver", choices=["auto", "dense", "gradient"], default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        msg = "--iterations must be positive"
        raise ValueError(msg)

    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        iterations=args.iterations,
        solver=args.solver,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    print(
        f"dft_scf grid={tuple(payload['grid_shape'])} points={payload['grid_points']} "
        f"solver={payload['solver']} fft={payload['fft_backend']} "
        f"iters={payload['iterations_completed']} ms/iter={payload['ms_per_iteration']:.3f} "
        f"energy={payload['final_energy']:.8g} residual={payload['final_residual']:.3g}"
    )


if __name__ == "__main__":
    main()
