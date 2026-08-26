"""Source-bound 2H-Silicon validation for full-rank periodic DFT cells."""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from math import pi, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    collect_host_provenance,
    load_workload,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR
from mlx_atomistic.benchmarks.dft_silicon_relaxation import (
    _optimization_config,
    _scf_config,
)
from mlx_atomistic.dft import (
    MonkhorstPackGrid,
    PeriodicDFTSystem,
    RuntimeObserver,
    optimize_periodic_geometry,
    read_gth,
)

REPORT_SCHEMA = "mlx-atomistic.hexagonal-silicon-cell-validation.v2"
CUBIC_LATTICE_ANGSTROM = 5.460859
INTERNAL_COORDINATE = 1.0 / 16.0
CUTOFF_HARTREE = 25.0
FFT_SHAPE = (40, 40, 64)
KPOINT_MESH = (6, 6, 4)
BAND_COUNT = 8
MAX_NET_FORCE_HARTREE_PER_BOHR = 5.0e-5
MAX_RECIPROCAL_DUALITY_ERROR = 2.0e-6

_FRACTIONAL_POSITIONS = (
    (1.0 / 3.0, 2.0 / 3.0, INTERNAL_COORDINATE),
    (2.0 / 3.0, 1.0 / 3.0, 0.5 + INTERNAL_COORDINATE),
    (2.0 / 3.0, 1.0 / 3.0, 1.0 - INTERNAL_COORDINATE),
    (1.0 / 3.0, 2.0 / 3.0, 0.5 - INTERNAL_COORDINATE),
)

_SOURCE_CONTRACT = {
    "structure": "ideal-2H-silicon-lonsdaleite",
    "prototype": "A_hP4_194_f-001",
    "space_group": "P63/mmc (194)",
    "wyckoff_site": "4f",
    "internal_coordinate": INTERNAL_COORDINATE,
    "lattice_relation": {
        "a_hexagonal": "a_cubic/sqrt(2)",
        "c_over_a": "sqrt(8/3)",
    },
    "sources": [
        {
            "doi": "10.1088/0953-8984/26/4/045801",
            "url": "https://arxiv.org/abs/1210.7392",
        },
        {
            "prototype": "A_hP4_194_f-001",
            "url": "https://aflowlib.duke.edu/p/A_hP4_194_f-001/",
        },
    ],
}


def hexagonal_silicon_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return the ideal 2H-Silicon cell and Cartesian positions in bohr."""

    a_angstrom = CUBIC_LATTICE_ANGSTROM / sqrt(2.0)
    c_angstrom = sqrt(8.0 / 3.0) * a_angstrom
    cell_angstrom = np.asarray(
        (
            (a_angstrom, 0.0, 0.0),
            (-0.5 * a_angstrom, 0.5 * sqrt(3.0) * a_angstrom, 0.0),
            (0.0, 0.0, c_angstrom),
        ),
        dtype=np.float64,
    )
    cell_bohr = cell_angstrom * ANGSTROM_TO_BOHR
    positions_bohr = np.asarray(_FRACTIONAL_POSITIONS, dtype=np.float64) @ cell_bohr
    return cell_bohr, positions_bohr


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.hexagonal-silicon-execution.v1",
        "report_schema": REPORT_SCHEMA,
        "geometry_source": inspect.getsource(hexagonal_silicon_geometry),
        "scf_config_source": inspect.getsource(_scf_config),
        "optimization_config_source": inspect.getsource(_optimization_config),
        "run_source": inspect.getsource(run_hexagonal_silicon),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def run_hexagonal_silicon(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Run the bounded 2H-Silicon full-rank cell validation."""

    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    cell, positions = hexagonal_silicon_geometry()
    scf_config = _scf_config(manifest)
    optimization_config = _optimization_config(reuse_scf_state=True)
    protocol = {
        "source": _SOURCE_CONTRACT,
        "source_fingerprint": sha256_bytes(canonical_json_bytes(_SOURCE_CONTRACT)),
        "cell_matrix_bohr": cell.tolist(),
        "fractional_positions": [list(values) for values in _FRACTIONAL_POSITIONS],
        "cutoff_hartree": CUTOFF_HARTREE,
        "fft_shape": list(FFT_SHAPE),
        "kpoint_mesh": list(KPOINT_MESH),
        "band_count": BAND_COUNT,
        "solver": manifest["solver"],
        "optimization_config": optimization_config.to_dict(),
        "thresholds": {
            "max_force_hartree_per_bohr": optimization_config.force_tolerance,
            "max_net_force_hartree_per_bohr": MAX_NET_FORCE_HARTREE_PER_BOHR,
            "max_reciprocal_duality_error": MAX_RECIPROCAL_DUALITY_ERROR,
        },
    }
    workload = {
        "schema_version": "mlx-atomistic.hexagonal-silicon-workload.v1",
        "silicon_solver_resource_workload_fingerprint": manifest["workload_fingerprint"],
        "protocol": protocol,
    }
    workload_fingerprint = sha256_bytes(canonical_json_bytes(workload))
    source_fingerprints = build_source_fingerprints()
    point = {
        "workload_fingerprint": workload_fingerprint,
        "runtime_fingerprint": source_fingerprints["runtime_fingerprint"],
        "implementation_fingerprint": _implementation_fingerprint(),
    }
    point["point_fingerprint"] = sha256_bytes(canonical_json_bytes(point))

    system = PeriodicDFTSystem(
        cell,
        FFT_SHAPE,
        positions,
        read_gth(gth_source, element="Si", name="GTH-PBE-q4"),
        electron_count=16.0,
    )
    mesh = MonkhorstPackGrid(KPOINT_MESH)
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    optimization = optimize_periodic_geometry(
        system,
        cutoff_hartree=CUTOFF_HARTREE,
        kpoint_mesh=mesh,
        n_bands=BAND_COUNT,
        config=optimization_config,
        scf_config=scf_config,
        observer=observer,
    )
    mx.synchronize()
    elapsed = perf_counter() - started

    result = optimization.final_scf
    force = optimization.final_force
    reciprocal = 2.0 * pi * np.linalg.inv(cell).T
    duality_error = float(np.max(np.abs(cell @ reciprocal.T - 2.0 * pi * np.eye(3))))
    electron_error = (
        float("inf")
        if result is None
        else abs(float(result.electron_count) - system.electron_count)
    )
    max_force = None if force is None else force.max_force
    net_force_norm = (
        None if force is None else float(np.linalg.norm(np.asarray(force.net_force)))
    )
    final_step = None if not optimization.steps else optimization.steps[-1]
    initial_fractional = np.asarray(_FRACTIONAL_POSITIONS, dtype=np.float64)
    final_fractional = optimization.final_positions @ np.linalg.inv(cell)
    fractional_delta = final_fractional - initial_fractional
    fractional_delta -= np.rint(fractional_delta)
    cartesian_delta = fractional_delta @ cell
    translation = np.mean(cartesian_delta, axis=0)
    aligned_delta = cartesian_delta - translation
    geometry = {
        "final_fractional_positions": np.mod(final_fractional, 1.0).tolist(),
        "removed_uniform_translation_bohr": translation.tolist(),
        "maximum_aligned_displacement_angstrom": float(
            np.max(np.linalg.norm(aligned_delta, axis=1)) / ANGSTROM_TO_BOHR
        ),
    }
    gates = {
        "optimization_converged": optimization.converged,
        "scf_converged": bool(result is not None and result.converged),
        "finite_energy": bool(result is not None and np.isfinite(result.total_energy)),
        "electron_count": electron_error <= 1.0e-4,
        "reciprocal_duality": duality_error <= MAX_RECIPROCAL_DUALITY_ERROR,
        "analytic_force_available": force is not None,
        "maximum_force": bool(
            max_force is not None and max_force <= optimization_config.force_tolerance
        ),
        "net_force": bool(
            net_force_norm is not None
            and net_force_norm <= MAX_NET_FORCE_HARTREE_PER_BOHR
        ),
        "final_displacement": bool(
            final_step is not None
            and final_step.step_norm <= optimization_config.displacement_tolerance
        ),
        "armijo_history": bool(
            optimization.steps
            and all(step.energy <= step.armijo_limit for step in optimization.steps)
        ),
    }
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "scientifically_verified": all(gates.values()),
        "point": point,
        "host": collect_host_provenance(),
        "protocol": protocol,
        "geometry": geometry,
        "optimization": optimization.to_dict(),
        "result": None if result is None else result.to_dict(),
        "force": None if force is None else force.to_dict(),
        "checks": {
            "gates": gates,
            "electron_count_error": electron_error,
            "reciprocal_duality_error": duality_error,
            "net_force_norm_hartree_per_bohr": net_force_norm,
        },
        "runtime": {
            "complete_wall_seconds": elapsed,
            "observation": observer.snapshot(),
        },
    }
    _write_atomic(Path(out), payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    """Run the bounded 2H-Silicon validation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gth-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_hexagonal_silicon(
        manifest_path=args.manifest,
        gth_source=args.gth_source,
        out=args.out,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"status={payload['status']} "
            f"verified={payload['scientifically_verified']} "
            f"wall_seconds={payload['runtime']['complete_wall_seconds']:.6f}"
        )


if __name__ == "__main__":
    main()
