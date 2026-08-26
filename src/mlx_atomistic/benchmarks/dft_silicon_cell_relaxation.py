"""Source-bound 2H-Silicon variable-cell relaxation validation."""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.benchmarks.dft_hexagonal_silicon import (
    BAND_COUNT,
    CUTOFF_HARTREE,
    FFT_SHAPE,
    KPOINT_MESH,
    hexagonal_silicon_geometry,
)
from mlx_atomistic.benchmarks.dft_runtime_contract import (
    build_source_fingerprints,
    collect_host_provenance,
    load_workload,
)
from mlx_atomistic.benchmarks.dft_silicon_relaxation import _scf_config
from mlx_atomistic.dft import (
    MonkhorstPackGrid,
    PeriodicCellOptimizationConfig,
    PeriodicDFTSystem,
    PeriodicStressConfig,
    RuntimeObserver,
    optimize_periodic_cell,
    periodic_analytic_stress,
    read_gth,
    run_periodic_scf,
)

REPORT_SCHEMA = "mlx-atomistic.silicon-periodic-cell-relaxation.v1"
INITIAL_LINEAR_SCALE = 0.995
MAX_LATTICE_RELATIVE_ERROR = 3.0e-3
MAX_FRACTIONAL_POSITION_ERROR = 2.0e-6
STRESS_TOLERANCE_HARTREE_PER_BOHR3 = 5.0e-6
PULAY_CHECK_CUTOFF_HARTREE = 35.0
PULAY_PRESSURE_DELTA_TOLERANCE_HARTREE_PER_BOHR3 = 5.0e-6


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _initial_geometry() -> tuple[np.ndarray, np.ndarray]:
    accepted_cell, accepted_positions = hexagonal_silicon_geometry()
    fractional = accepted_positions @ np.linalg.inv(accepted_cell)
    initial_cell = INITIAL_LINEAR_SCALE * accepted_cell
    return initial_cell, fractional @ initial_cell


def _cell_config() -> PeriodicCellOptimizationConfig:
    return PeriodicCellOptimizationConfig(
        max_steps=4,
        relaxation_mode="cell",
        external_pressure=0.0,
        stress_tolerance=STRESS_TOLERANCE_HARTREE_PER_BOHR3,
        strain_tolerance=5.0e-4,
        cell_compliance=100.0,
        max_strain=0.015,
        minimum_volume_ratio=0.9,
        stress_config=PeriodicStressConfig(
            mode="isotropic",
            strain_step=1.0e-3,
            reuse_scf_state=True,
            require_fixed_basis_topology=True,
        ),
    )


def _implementation_fingerprint() -> str:
    contract = {
        "schema_version": "mlx-atomistic.silicon-cell-execution.v1",
        "report_schema": REPORT_SCHEMA,
        "initial_geometry_source": inspect.getsource(_initial_geometry),
        "cell_config_source": inspect.getsource(_cell_config),
        "run_source": inspect.getsource(run_silicon_cell_relaxation),
    }
    return sha256_bytes(canonical_json_bytes(contract))


def run_silicon_cell_relaxation(
    *,
    manifest_path: str | Path,
    gth_source: str | Path,
    out: str | Path,
    checkpoint_to: str | Path | None = None,
    checkpoint_step: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run the bounded 2H-Silicon isotropic cell validation."""

    manifest, _selected = load_workload(manifest_path, gth_source=gth_source)
    accepted_cell, accepted_positions = hexagonal_silicon_geometry()
    initial_cell, initial_positions = _initial_geometry()
    system = PeriodicDFTSystem(
        initial_cell,
        FFT_SHAPE,
        initial_positions,
        read_gth(gth_source, element="Si", name="GTH-PBE-q4"),
        electron_count=16.0,
    )
    base_scf_config = _scf_config(manifest)
    scf_config = replace(
        base_scf_config,
        density_tolerance=5.0e-7,
        energy_tolerance=5.0e-8,
    )
    config = _cell_config()
    source_fingerprints = build_source_fingerprints()
    protocol_identity = {
        "material": "ideal-2H-silicon-lonsdaleite",
        "accepted_cell_matrix_bohr": accepted_cell.tolist(),
        "initial_linear_scale": INITIAL_LINEAR_SCALE,
        "cutoff_hartree": CUTOFF_HARTREE,
        "pulay_check_cutoff_hartree": PULAY_CHECK_CUTOFF_HARTREE,
        "fft_shape": list(FFT_SHAPE),
        "kpoint_mesh": list(KPOINT_MESH),
        "band_count": BAND_COUNT,
        "solver": manifest["solver"],
        "scf_overrides": {
            "density_tolerance": scf_config.density_tolerance,
            "energy_tolerance_hartree": scf_config.energy_tolerance,
        },
        "scf_batch_policy": scf_config.batch_policy(),
        "cell_config": config.to_dict(),
    }
    workload = {
        "schema_version": "mlx-atomistic.silicon-periodic-cell-workload.v1",
        "solver_resource_workload_fingerprint": manifest["workload_fingerprint"],
        "protocol": protocol_identity,
    }
    point = {
        "workload_fingerprint": sha256_bytes(canonical_json_bytes(workload)),
        "runtime_fingerprint": source_fingerprints["runtime_fingerprint"],
        "implementation_fingerprint": _implementation_fingerprint(),
    }
    point["point_fingerprint"] = sha256_bytes(canonical_json_bytes(point))
    observer = RuntimeObserver(detail_events=False)
    started = perf_counter()
    optimization = optimize_periodic_cell(
        system,
        cutoff_hartree=CUTOFF_HARTREE,
        kpoint_mesh=MonkhorstPackGrid(KPOINT_MESH),
        n_bands=BAND_COUNT,
        config=config,
        scf_config=scf_config,
        observer=observer,
        checkpoint_to=checkpoint_to,
        checkpoint_step=checkpoint_step,
        resume_from=resume_from,
        provenance={"point_fingerprint": point["point_fingerprint"]},
    )

    final_cell = np.asarray(optimization.final_system.grid.cell.matrix, dtype=np.float64)
    final_positions = np.asarray(optimization.final_system.positions, dtype=np.float64)
    accepted_volume = abs(float(np.linalg.det(accepted_cell)))
    final_linear_scale = (abs(float(np.linalg.det(final_cell))) / accepted_volume) ** (
        1.0 / 3.0
    )
    accepted_fractional = accepted_positions @ np.linalg.inv(accepted_cell)
    final_fractional = final_positions @ np.linalg.inv(final_cell)
    fractional_delta = final_fractional - accepted_fractional
    fractional_delta -= np.rint(fractional_delta)
    lattice_relative_error = abs(final_linear_scale - 1.0)
    fractional_error = float(np.max(np.abs(fractional_delta)))
    final_stress = optimization.final_stress
    pulay_check: dict[str, Any]
    if optimization.converged and optimization.final_scf is not None and final_stress is not None:
        pulay_started = perf_counter()
        high_cutoff_scf = run_periodic_scf(
            optimization.final_system,
            cutoff_hartree=PULAY_CHECK_CUTOFF_HARTREE,
            kpoint_mesh=MonkhorstPackGrid(KPOINT_MESH),
            n_bands=BAND_COUNT,
            config=scf_config,
            initial_density=optimization.final_scf.density,
            observer=observer,
        )
        if high_cutoff_scf.converged:
            high_cutoff_stress = periodic_analytic_stress(
                optimization.final_system,
                cutoff_hartree=PULAY_CHECK_CUTOFF_HARTREE,
                kpoint_mesh=MonkhorstPackGrid(KPOINT_MESH),
                n_bands=BAND_COUNT,
                config=config.stress_config,
                scf_config=scf_config,
                observer=observer,
                base_result=high_cutoff_scf,
            )
            pressure_delta = abs(high_cutoff_stress.pressure - final_stress.pressure)
            mx.synchronize()
            pulay_check = {
                "status": "passed"
                if pressure_delta
                <= PULAY_PRESSURE_DELTA_TOLERANCE_HARTREE_PER_BOHR3
                else "failed",
                "production_cutoff_hartree": CUTOFF_HARTREE,
                "higher_cutoff_hartree": PULAY_CHECK_CUTOFF_HARTREE,
                "production_pressure_hartree_per_bohr3": final_stress.pressure,
                "higher_cutoff_pressure_hartree_per_bohr3": (
                    high_cutoff_stress.pressure
                ),
                "pressure_delta_hartree_per_bohr3": pressure_delta,
                "maximum_pressure_delta_hartree_per_bohr3": (
                    PULAY_PRESSURE_DELTA_TOLERANCE_HARTREE_PER_BOHR3
                ),
                "higher_cutoff_scf_iterations": high_cutoff_scf.iterations,
                "complete_wall_seconds": perf_counter() - pulay_started,
            }
        else:
            pulay_check = {
                "status": "failed",
                "reason": f"higher-cutoff SCF ended with {high_cutoff_scf.status}",
                "production_cutoff_hartree": CUTOFF_HARTREE,
                "higher_cutoff_hartree": PULAY_CHECK_CUTOFF_HARTREE,
            }
    else:
        pulay_check = {
            "status": "not_run",
            "reason": "production cell relaxation did not converge",
            "production_cutoff_hartree": CUTOFF_HARTREE,
            "higher_cutoff_hartree": PULAY_CHECK_CUTOFF_HARTREE,
        }
    mx.synchronize()
    elapsed = perf_counter() - started
    gates = {
        "optimization_converged": optimization.converged,
        "scf_converged": bool(
            optimization.final_scf is not None and optimization.final_scf.converged
        ),
        "stress_available": final_stress is not None,
        "pressure": bool(
            final_stress is not None
            and abs(final_stress.pressure) <= STRESS_TOLERANCE_HARTREE_PER_BOHR3
        ),
        "accepted_lattice": lattice_relative_error <= MAX_LATTICE_RELATIVE_ERROR,
        "fractional_positions": fractional_error <= MAX_FRACTIONAL_POSITION_ERROR,
        "armijo_history": bool(
            optimization.steps
            and all(step.enthalpy <= step.armijo_limit for step in optimization.steps)
        ),
        "pulay_convergence": pulay_check["status"] == "passed",
    }
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "scientifically_verified": all(gates.values()),
        "point": point,
        "host": collect_host_provenance(),
        "protocol": {
            **protocol_identity,
            "solver_resource_workload_fingerprint": manifest[
                "workload_fingerprint"
            ],
            "thresholds": {
                "maximum_lattice_relative_error": MAX_LATTICE_RELATIVE_ERROR,
                "maximum_fractional_position_error": MAX_FRACTIONAL_POSITION_ERROR,
                "maximum_pressure_hartree_per_bohr3": (
                    STRESS_TOLERANCE_HARTREE_PER_BOHR3
                ),
                "maximum_pulay_pressure_delta_hartree_per_bohr3": (
                    PULAY_PRESSURE_DELTA_TOLERANCE_HARTREE_PER_BOHR3
                ),
            },
        },
        "checks": {
            "gates": gates,
            "final_linear_scale": final_linear_scale,
            "lattice_relative_error": lattice_relative_error,
            "maximum_fractional_position_error": fractional_error,
        },
        "optimization": optimization.to_dict(),
        "pulay_check": pulay_check,
        "runtime": {
            "complete_wall_seconds": elapsed,
            "observation": observer.snapshot(),
        },
    }
    _write_atomic(Path(out), payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    """Run the bounded 2H-Silicon variable-cell validation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gth-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-to", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_silicon_cell_relaxation(
        manifest_path=args.manifest,
        gth_source=args.gth_source,
        out=args.out,
        checkpoint_to=args.checkpoint_to,
        checkpoint_step=args.checkpoint_step,
        resume_from=args.resume_from,
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
