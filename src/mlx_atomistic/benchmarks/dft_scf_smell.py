"""Run the bounded representative-k-point periodic SCF development gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.dft import (
    KPoint,
    KPointMesh,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    read_gth,
    run_periodic_scf,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver

SCHEMA = "mlx-atomistic.dft-scf-smell.v1"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlx_atomistic.benchmarks.dft_scf_smell",
        description=(
            "Run a partial-Brillouin-zone SCF gate. This is not a complete "
            "216-explicit/108-representative-k-point production result."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gth-source", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--representatives", type=_positive_integer, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def _owner_points(
    workload: dict[str, Any], representatives: int
) -> list[dict[str, Any]]:
    owners = [
        point
        for point in workload["physics"]["kpoints"]
        if point["role"] == "owner"
    ]
    if representatives > len(owners):
        msg = (
            f"requested {representatives} representative k-points, but the "
            f"manifest contains only {len(owners)} owners"
        )
        raise ValueError(msg)
    return owners[:representatives]


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    workload_bytes = arguments.manifest.read_bytes()
    workload = json.loads(workload_bytes)
    system_values = workload["system"]
    physics = workload["physics"]
    lattice = float(system_values["lattice_constant_bohr"])
    selected = _owner_points(workload, arguments.representatives)
    system = PeriodicDFTSystem(
        (lattice, lattice, lattice),
        physics["fft_shape"],
        np.asarray(system_values["fractional_positions"], dtype=np.float64)
        * lattice,
        read_gth(arguments.gth_source, element="Si", name="GTH-PBE-q4"),
        electron_count=float(system_values["electron_count"]),
    )
    mesh = KPointMesh(
        [
            KPoint(
                point["reduced_coordinates"],
                weight=float(point["weight"]["numerator"])
                / float(point["weight"]["denominator"]),
                coordinate_system="reduced",
            )
            for point in selected
        ]
    )
    observer = RuntimeObserver(synchronize=mx.synchronize, detail_events=False)
    config = PeriodicSCFConfig(
        max_iterations=80,
        min_iterations=2,
        density_tolerance=1e-6,
        energy_tolerance=8e-6,
        orbital_tolerance=1e-6,
        mixing_beta=0.35,
        mixer="diis",
        adaptive_eigensolver_tolerance=arguments.mode == "adaptive",
        initial_eigensolver_tolerance=1e-2,
        eigensolver_tolerance_scale=0.1,
        davidson=PeriodicDavidsonConfig(
            max_iterations=48,
            tolerance=1e-6,
            max_subspace_size=64,
            preconditioner_floor=0.25,
        ),
        kpoint_batch_size=8,
    )
    started = perf_counter()
    result = run_periodic_scf(
        system,
        cutoff_hartree=float(physics["kinetic_cutoff_hartree"]),
        kpoint_mesh=mesh,
        n_bands=int(system_values["occupied_band_count"]),
        config=config,
        observer=observer,
    )
    mx.synchronize()
    elapsed = perf_counter() - started
    observation = observer.snapshot()
    maximum_residual = max(
        float(mx.max(point.eigen.residuals)) for point in result.kpoints
    )
    maximum_overlap = max(point.eigen.orthonormality_error for point in result.kpoints)
    return {
        "schema": SCHEMA,
        "scope": "partial-brillouin-zone-development-gate",
        "production_full_scf_result": False,
        "includes_scf_density_loop": True,
        "includes_persistence": False,
        "manifest": str(arguments.manifest),
        "manifest_sha256": sha256_bytes(workload_bytes),
        "gth_source": str(arguments.gth_source),
        "mode": arguments.mode,
        "elapsed_seconds": elapsed,
        "converged": result.converged,
        "iterations": result.iterations,
        "representative_kpoints": len(result.kpoints),
        "selected_owner_indices": [point["index"] for point in selected],
        "total_energy_hartree": result.total_energy,
        "energy_hartree_per_atom": result.total_energy
        / int(system_values["atom_count"]),
        "electron_error": abs(
            result.electron_count - float(system_values["electron_count"])
        ),
        "density_residual": result.density_residual,
        "maximum_orbital_residual": maximum_residual,
        "maximum_overlap_error": maximum_overlap,
        "eigensolver_tolerances": [
            row["eigensolver_tolerance"] for row in result.history
        ],
        "eigensolver_methods": [row["eigensolver_method"] for row in result.history],
        "scf_history": list(result.history),
        "work_counters": observation["work_counters"],
        "phase_seconds": observation["phase_seconds"],
        "memory": observation["memory"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = _run(arguments)
    except (KeyError, OSError, TypeError, ValueError) as error:
        _parser().error(str(error))
    payload = canonical_json_bytes(report)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_bytes(payload + b"\n")
    if arguments.json:
        print(payload.decode(), flush=True)
    else:
        print(
            f"{report['mode']}: {report['elapsed_seconds']:.3f} s, "
            f"{report['iterations']} SCF iterations, "
            f"{report['representative_kpoints']} representative k-points",
            flush=True,
        )
    return 0 if report["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
