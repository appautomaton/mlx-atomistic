"""Command-line fixed-cell DFT geometry optimization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from mlx_atomistic.dft.optimization import (
    GeometryOptimizationConfig,
    GeometryOptimizationResult,
    geometry_demo_system,
    optimize_geometry,
    save_geometry_optimization,
)
from mlx_atomistic.dft.scf import SCFConfig
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
    system_name: str,
    steps: int,
    grid_shape: tuple[int, int, int] = (4, 4, 4),
    optimizer: str = "lbfgs",
) -> tuple[dict, GeometryOptimizationResult]:
    """Run one compact geometry-optimization workflow."""

    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    system = geometry_demo_system(system_name, grid_shape=grid_shape)
    result = optimize_geometry(
        system,
        config=GeometryOptimizationConfig(
            max_steps=steps,
            optimizer=optimizer,  # type: ignore[arg-type]
            scf_config=SCFConfig(
                max_iterations=20,
                tolerance=1e-8,
                mixing=0.45,
                solver="dense",
                seed=29,
                convergence_mode="either",
                record_timing=True,
            ),
        ),
    )
    payload = {
        "runtime": asdict(get_runtime_info()),
        "system": system_name,
        "grid_shape": list(grid_shape),
        "optimizer": optimizer,
        "result": result.to_dict(),
        "status": result.status,
        "final_energy": result.final_energy,
        "final_max_force": result.final_max_force,
        "step_count": len(result.steps),
        "history": [step.to_dict() for step in result.steps],
    }
    return payload, result


def main(argv: list[str] | None = None) -> None:
    """Run the DFT geometry-optimization benchmark from the command line.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv`` when ``None``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=["gaussian-dimer"],
        default="gaussian-dimer",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--grid", default="4,4,4", help="Grid shape as nx,ny,nz.")
    parser.add_argument("--optimizer", choices=["lbfgs", "steepest_descent"], default="lbfgs")
    parser.add_argument("--output", default=None, help="Optional NPZ output path.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload, result = build_payload(
        system_name=args.system,
        steps=args.steps,
        grid_shape=_parse_grid(args.grid),
        optimizer=args.optimizer,
    )
    if args.output is not None:
        save_geometry_optimization(args.output, result, metadata={"system": args.system})
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    print(
        f"dft_geometry system={payload['system']} grid={tuple(payload['grid_shape'])} "
        f"optimizer={payload['optimizer']} status={payload['status']} "
        f"steps={payload['step_count']} energy={_format_float(payload['final_energy'])} "
        f"max_force={_format_float(payload['final_max_force'])}"
    )


def _format_float(value: float | None) -> str:
    return "none" if value is None else f"{value:.8g}"


if __name__ == "__main__":
    main()
