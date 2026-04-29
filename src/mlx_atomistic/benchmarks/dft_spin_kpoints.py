"""Benchmark DFT spin occupations and k-point diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mlx_atomistic.dft import (
    BandPath,
    DFTSystem,
    FermiDiracOccupations,
    KPointMesh,
    SCFConfig,
    magnetization_density,
    run_band_structure,
    run_scf,
    spin_density_from_orbitals,
)
from mlx_atomistic.runtime import get_runtime_info


def run_case() -> dict:
    """Run one spin/k-point smoke case."""

    system = DFTSystem.one_center(grid_shape=(4, 4, 4), electron_count=2.0)
    result = run_scf(system, config=SCFConfig(max_iterations=1, solver="dense", seed=5))
    occupations = FermiDiracOccupations(2.0, temperature=0.05).resolve(result.orbital_eigenvalues)
    up, down = spin_density_from_orbitals(
        result.orbitals,
        result.orbitals,
        system.grid,
        up_occupations=[1.0],
        down_occupations=[1.0],
    )
    magnetization = magnetization_density(up, down)
    bands = run_band_structure(
        system,
        result,
        BandPath.line((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), count=3),
    )
    return {
        "kpoint_count": len(KPointMesh.gamma().points),
        "occupation_count": occupations.electron_count,
        "magnetization_integral": float(np.sum(np.array(magnetization)) * system.grid.dv),
        "band_shape": list(bands.eigenvalues.shape),
        "band_reused_density": bands.reused_density,
    }


def build_payload() -> dict:
    """Run spin/k-point benchmark smoke."""

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
    print(f"dft_spin_kpoints band_shape={payload['band_shape']}")


if __name__ == "__main__":
    main()
