"""Benchmark DFT local pseudopotential paths."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mlx_atomistic.dft import (
    DFTSystem,
    DiracExchange,
    Ion,
    IonCollection,
    SCFConfig,
    read_gth,
    read_upf,
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


def _case_systems(
    grid_shape: tuple[int, int, int],
    *,
    upf_path: str | Path | None = None,
    gth_path: str | Path | None = None,
    gth_element: str | None = None,
    gth_name: str | None = None,
) -> list[tuple[str, DFTSystem]]:
    cases = [
        (
            "gaussian",
            DFTSystem.one_center(
                cell=(8.0, 8.0, 8.0),
                grid_shape=grid_shape,
                center=(4.0, 4.0, 4.0),
                electron_count=2.0,
            ),
        ),
    ]
    if upf_path is not None:
        upf = read_upf(upf_path)
        cases.append(
            (
                "upf-local",
                DFTSystem(
                    cell=(8.0, 8.0, 8.0),
                    grid_shape=grid_shape,
                    ions=IonCollection([Ion(upf.element, (4.0, 4.0, 4.0), upf)]),
                ),
            )
        )
    if gth_path is not None:
        gth = read_gth(gth_path, element=gth_element, name=gth_name)
        cases.append(
            (
                "gth-local",
                DFTSystem(
                    cell=(8.0, 8.0, 8.0),
                    grid_shape=grid_shape,
                    ions=IonCollection([Ion(gth.element, (4.0, 4.0, 4.0), gth)]),
                ),
            )
        )
    return cases


def run_case(*, label: str, system: DFTSystem, iterations: int) -> dict:
    """Run one pseudopotential benchmark case."""

    config = SCFConfig(max_iterations=iterations, solver="dense", seed=31, record_timing=True)
    start = perf_counter()
    result = run_scf(system, config=config, xc_functional=DiracExchange())
    elapsed_ms = (perf_counter() - start) * 1000.0
    summary = result.to_dict()
    return {
        "case": label,
        "grid_shape": list(system.grid_shape),
        "grid_points": system.grid.size,
        "iterations_requested": iterations,
        "iterations_completed": result.iterations,
        "pseudopotential_format": summary["pseudopotential_format"],
        "ion_count": summary["ion_count"],
        "valence_electron_count": summary["valence_electron_count"],
        "nonlocal_available": summary["nonlocal_available"],
        "nonlocal_applied": summary["nonlocal_applied"],
        "final_energy": result.total_energy,
        "final_residual": result.residual,
        "ms_total": elapsed_ms,
        "ms_per_iteration": elapsed_ms / max(result.iterations, 1),
        "energy_by_term": result.energy_by_term,
        "timings": result.timings,
    }


def build_payload(
    *,
    grid_shape: tuple[int, int, int] = (4, 4, 4),
    iterations: int = 2,
    upf_path: str | Path | None = None,
    gth_path: str | Path | None = None,
    gth_element: str | None = None,
    gth_name: str | None = None,
) -> dict:
    """Run compact local-pseudopotential benchmark cases."""

    cases = [
        run_case(label=label, system=system, iterations=iterations)
        for label, system in _case_systems(
            grid_shape,
            upf_path=upf_path,
            gth_path=gth_path,
            gth_element=gth_element,
            gth_name=gth_name,
        )
    ]
    return {
        "runtime": asdict(get_runtime_info()),
        "cases": cases,
        "case_count": len(cases),
        "grid_shape": list(grid_shape),
        "iterations_requested": iterations,
        "external_pseudopotentials": {
            "upf_path": None if upf_path is None else str(upf_path),
            "gth_path": None if gth_path is None else str(gth_path),
            "gth_element": gth_element,
            "gth_name": gth_name,
        },
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
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--upf", type=Path, default=None, help="Optional UPF file to include.")
    parser.add_argument("--gth", type=Path, default=None, help="Optional GTH file or database.")
    parser.add_argument("--gth-element", default=None, help="Element for a GTH database lookup.")
    parser.add_argument("--gth-name", default=None, help="Optional named GTH database entry.")
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        msg = "--iterations must be positive"
        raise ValueError(msg)
    payload = build_payload(
        grid_shape=_parse_grid(args.grid),
        iterations=args.iterations,
        upf_path=args.upf,
        gth_path=args.gth,
        gth_element=args.gth_element,
        gth_name=args.gth_name,
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
            f"dft_pseudopotential case={case['case']} grid={tuple(case['grid_shape'])} "
            f"format={case['pseudopotential_format']} iters={case['iterations_completed']} "
            f"ms/iter={case['ms_per_iteration']:.3f} energy={case['final_energy']:.8g}"
        )


if __name__ == "__main__":
    main()
