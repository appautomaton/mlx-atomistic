"""Run a manifest-bound OpenMM NVT timing for the charged JAC PME workload.

This is reference-only tooling. OpenMM remains outside the MLX product runtime.
The runner reuses the independently validated JAC replica construction from
``run_charged_pme_parity.py`` and fails before timing when the MLX and OpenMM
workload manifests differ.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform as platform_module
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks.charged_pme import audit_openmm_runtime_artifacts
from mlx_atomistic.benchmarks.pme_validation import (
    manifest_hash,
    manifest_mismatches,
    require_matching_manifest,
)
from mlx_atomistic.prep.io import load_prepared_system

try:
    parity = importlib.import_module("scripts.run_charged_pme_parity")
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    parity = importlib.import_module("run_charged_pme_parity")

OPENMM_RUNTIME_NAME = "openmm_runtime.json"
OPENMM_MANIFEST_NAME = "openmm_workload_manifest.json"
MLX_MANIFEST_NAME = "mlx_workload_manifest.json"
MANIFEST_COMPARISON_NAME = "manifest_comparison.json"
RUNTIME_COMPARISON_NAME = "runtime_comparison.json"
ADMISSION_NAME = "openmm_runtime_admission.json"

RUNTIME_MANIFEST_FIELDS = (
    *parity.MANIFEST_FIELDS,
    "dynamics.integrator",
    "dynamics.dt_ps",
    "dynamics.temperature_k",
    "dynamics.friction_per_ps",
    "dynamics.constraint_tolerance",
    "dynamics.constraint_count",
    "dynamics.fixed_cell",
    "dynamics.warmup_steps",
    "dynamics.measured_steps",
    "dynamics.initial_positions",
    "dynamics.initial_velocities",
)


def _parse_replicas(value: str) -> tuple[int, int, int]:
    replicas = tuple(int(part) for part in value.split(","))
    if len(replicas) != 3 or any(part <= 0 for part in replicas):
        msg = "replicas must be three positive comma-separated integers"
        raise argparse.ArgumentTypeError(msg)
    return replicas


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _runtime_manifest(
    fixed_coordinate_manifest: dict[str, Any],
    *,
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Promote a validated fixed-coordinate manifest to the timed NVT operation."""

    manifest = deepcopy(fixed_coordinate_manifest)
    fixed_coordinate_hash = manifest.pop("manifest_hash")
    manifest["workload"]["operation"] = "fixed_cell_nvt_step"
    manifest["dynamics"] = deepcopy(dynamics)
    manifest["provenance"] = {
        "fixed_coordinate_manifest_hash": fixed_coordinate_hash,
        "derivation": "same validated topology and force model; NVT protocol added",
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def _manifest_comparison(
    mlx_manifest: dict[str, Any],
    openmm_manifest: dict[str, Any],
) -> dict[str, Any]:
    mismatches = manifest_mismatches(
        mlx_manifest,
        openmm_manifest,
        fields=RUNTIME_MANIFEST_FIELDS,
    )
    return {
        "status": "matched" if not mismatches else "mismatched",
        "matched": not mismatches,
        "required_fields": list(RUNTIME_MANIFEST_FIELDS),
        "mismatches": mismatches,
        "mlx_manifest_hash": mlx_manifest["manifest_hash"],
        "openmm_manifest_hash": openmm_manifest["manifest_hash"],
    }


def _platform_properties(platform: Any, context: Any) -> dict[str, str]:
    properties = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific diagnostic.
            properties[name] = f"<unavailable:{exc}>"
    return properties


def _constraint_error_angstrom(
    positions_angstrom: np.ndarray,
    constraint_pairs: np.ndarray,
    target_distances_angstrom: np.ndarray,
    cell_lengths_angstrom: np.ndarray,
) -> float:
    if constraint_pairs.size == 0:
        return 0.0
    delta = positions_angstrom[constraint_pairs[:, 1]] - positions_angstrom[constraint_pairs[:, 0]]
    delta -= cell_lengths_angstrom * np.rint(delta / cell_lengths_angstrom)
    lengths = np.linalg.norm(delta, axis=1)
    return float(np.max(np.abs(lengths - target_distances_angstrom)))


def _run_openmm(
    *,
    api: parity.OpenMMApi,
    reference: dict[str, Any],
    prepared: Any,
    replicas: tuple[int, int, int],
    platform_name: str,
    precision: str,
    warmups: int,
    steps: int,
    dt_ps: float,
    temperature_k: float,
    friction_per_ps: float,
    constraint_tolerance: float,
    seed: int,
    workload_manifest_hash: str,
) -> dict[str, Any]:
    platform = api.mm.Platform.getPlatformByName(platform_name)
    properties: dict[str, str] = {}
    if "Precision" in list(platform.getPropertyNames()):
        properties["Precision"] = precision
    integrator = api.mm.LangevinMiddleIntegrator(
        temperature_k * api.unit.kelvin,
        friction_per_ps / api.unit.picosecond,
        dt_ps * api.unit.picoseconds,
    )
    integrator.setConstraintTolerance(constraint_tolerance)
    integrator.setRandomNumberSeed(seed)
    context = api.mm.Context(reference["system"], integrator, platform, properties)
    context.setPeriodicBoxVectors(*reference["box_vectors"])
    context.setPositions(reference["positions_angstrom"] * 0.1 * api.unit.nanometer)
    context.setVelocities(
        np.asarray(prepared.velocities, dtype=np.float64)
        * 0.1
        * api.unit.nanometer
        / api.unit.picosecond
    )

    integrator.step(warmups)
    context.getState(getEnergy=True)
    started = time.perf_counter()
    integrator.step(steps)
    state = context.getState(
        getEnergy=True,
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=False,
    )
    elapsed_seconds = time.perf_counter() - started

    potential = float(state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole))
    kinetic = float(state.getKineticEnergy().value_in_unit(api.unit.kilojoule_per_mole))
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(api.unit.angstrom),
        dtype=np.float64,
    )
    velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(api.unit.nanometer / api.unit.picosecond),
        dtype=np.float64,
    )
    constraint_pairs = np.asarray(prepared.constraints, dtype=np.int32)
    constraint_distances = np.asarray(
        prepared.constraint_distance,
        dtype=np.float64,
    )
    cell_lengths = np.asarray(reference["cell_lengths_angstrom"], dtype=np.float64)
    constraint_error = _constraint_error_angstrom(
        positions,
        constraint_pairs,
        constraint_distances,
        cell_lengths,
    )
    degrees_of_freedom = 3 * prepared.atom_count - constraint_pairs.shape[0]
    temperature = 2.0 * kinetic / (degrees_of_freedom * 0.00831446261815324)

    nonbonded = parity._find_openmm_nonbonded(api, reference["system"])
    alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
    alpha_per_nanometer = (
        float(alpha.value_in_unit(api.unit.nanometer**-1))
        if hasattr(alpha, "value_in_unit")
        else float(alpha)
    )
    finite = bool(
        elapsed_seconds > 0.0
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and all(
            math.isfinite(value) for value in (potential, kinetic, temperature, constraint_error)
        )
    )
    return {
        "kind": "mlx_atomistic.openmm_charged_pme_runtime.v1",
        "status": "ok" if finite else "failed",
        "passed": finite,
        "engine": "openmm-reference",
        "operation": "fixed_cell_nvt_step",
        "atom_count": prepared.atom_count,
        "replicas": list(replicas),
        "warmups": warmups,
        "steps": steps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / steps,
        "steps_per_second": steps / elapsed_seconds,
        "dt_ps": dt_ps,
        "thermostat": {
            "kind": "langevin_middle",
            "temperature_k": temperature_k,
            "friction_per_ps": friction_per_ps,
            "seed": seed,
        },
        "constraint_protocol": {
            "source": "OpenMM AMBER HBonds plus rigidWater",
            "count": int(constraint_pairs.shape[0]),
            "tolerance": constraint_tolerance,
        },
        "pme": {
            "mesh_shape": [int(nx), int(ny), int(nz)],
            "alpha_per_angstrom": round(alpha_per_nanometer / 10.0, 7),
            "cutoff_angstrom": parity.DEFAULT_CUTOFF_ANGSTROM,
        },
        "cell_lengths_angstrom": cell_lengths.tolist(),
        "platform": context.getPlatform().getName(),
        "platform_properties": _platform_properties(platform, context),
        "openmm_version": parity._openmm_version(api),
        "hardware": {
            "machine": platform_module.machine(),
            "platform": platform_module.platform(),
            "processor": platform_module.processor(),
            "python_version": platform_module.python_version(),
        },
        "completion_barrier": "explicit_device_completion_inside_timer",
        "timing_boundary": {
            "initialization_included": False,
            "warmup_included": False,
            "io_included": False,
            "final_state_materialization_included": True,
        },
        "workload_manifest_hash": workload_manifest_hash,
        "finite": finite,
        "state": {
            "potential_energy_kj_mol": potential,
            "kinetic_energy_kj_mol": kinetic,
            "temperature_k": temperature,
            "constraint_max_error_angstrom": constraint_error,
        },
    }


def _runtime_comparison(
    mlx_runtime: dict[str, Any],
    openmm_runtime: dict[str, Any],
    *,
    expected_prepared: Path,
    mlx_manifest_hash: str,
    openmm_manifest_hash: str,
) -> dict[str, Any]:
    mlx_pme = mlx_runtime.get("pme", {})
    openmm_pme = openmm_runtime.get("pme", {})
    mlx_seconds = float(mlx_runtime.get("timings", {}).get("measured_seconds", math.nan))
    openmm_seconds = float(openmm_runtime.get("elapsed_seconds", math.nan))
    prepared_value = mlx_runtime.get("prepared")
    prepared_matches = isinstance(prepared_value, str) and (
        Path(prepared_value).resolve() == expected_prepared.resolve()
    )
    mlx_cell = np.asarray(mlx_runtime.get("cell_lengths_angstrom", []), dtype=np.float64)
    openmm_cell = np.asarray(
        openmm_runtime.get("cell_lengths_angstrom", []),
        dtype=np.float64,
    )
    checks = {
        "mlx_passed": mlx_runtime.get("passed") is True,
        "openmm_passed": openmm_runtime.get("passed") is True,
        "mlx_prepared_artifact": prepared_matches,
        "atom_count": mlx_runtime.get("atom_count") == openmm_runtime.get("atom_count"),
        "fixed_cell": bool(
            mlx_cell.shape == (3,)
            and openmm_cell.shape == (3,)
            and np.allclose(mlx_cell, openmm_cell, rtol=0.0, atol=5.0e-6)
        ),
        "warmup_steps": mlx_runtime.get("warmup_steps") == openmm_runtime.get("warmups"),
        "measured_steps": mlx_runtime.get("measured_steps") == openmm_runtime.get("steps"),
        "final_only_cadence": (
            mlx_runtime.get("sample_interval") == openmm_runtime.get("steps")
            and mlx_runtime.get("diagnostic_interval") == openmm_runtime.get("steps")
        ),
        "dt_ps": mlx_runtime.get("dt_ps") == openmm_runtime.get("dt_ps"),
        "temperature_k": mlx_runtime.get("temperature_target_k")
        == openmm_runtime.get("thermostat", {}).get("temperature_k"),
        "seed": mlx_runtime.get("seed") == openmm_runtime.get("thermostat", {}).get("seed"),
        "pme_mesh": mlx_pme.get("mesh_shape") == openmm_pme.get("mesh_shape"),
        "pme_alpha": bool(
            np.isclose(
                float(mlx_pme.get("alpha", math.nan)),
                float(openmm_pme.get("alpha_per_angstrom", math.nan)),
                rtol=0.0,
                atol=1.0e-7,
            )
        ),
        "pme_cutoff": mlx_pme.get("real_cutoff") == openmm_pme.get("cutoff_angstrom"),
        "positive_elapsed": (
            math.isfinite(mlx_seconds)
            and mlx_seconds > 0.0
            and math.isfinite(openmm_seconds)
            and openmm_seconds > 0.0
        ),
    }
    matched = all(checks.values())
    ratio = mlx_seconds / openmm_seconds if matched else None
    return {
        "kind": "mlx_atomistic.charged_pme_runtime_comparison.v1",
        "status": "comparable" if ratio is not None else "blocked",
        "matched": matched,
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "mlx_manifest_hash": mlx_manifest_hash,
        "openmm_manifest_hash": openmm_manifest_hash,
        "mlx_elapsed_seconds": mlx_seconds,
        "openmm_elapsed_seconds": openmm_seconds,
        "mlx_over_openmm_ratio": ratio,
        "within_ten_times": None if ratio is None else ratio <= 10.0,
        "note": (
            "The engines use independent random-number implementations; this "
            "is matched protocol throughput, not trajectory identity."
        ),
    }


def run_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Run OpenMM and write self-contained workload and timing evidence."""

    if args.warmups <= 0 or args.steps <= 0:
        msg = "warmups and steps must be positive"
        raise ValueError(msg)
    if args.dt_ps <= 0.0 or args.temperature_k <= 0.0:
        msg = "dt-ps and temperature-k must be positive"
        raise ValueError(msg)
    if args.friction_per_ps < 0.0 or args.constraint_tolerance <= 0.0:
        msg = "friction must be nonnegative and constraint tolerance positive"
        raise ValueError(msg)
    if not np.isclose(args.friction_per_ps, 1.0, rtol=0.0, atol=1.0e-12):
        msg = "the MLX charged-PME runtime fixes friction at 1/ps"
        raise ValueError(msg)
    if not np.isclose(args.constraint_tolerance, 1.0e-5, rtol=0.0, atol=1.0e-12):
        msg = "the MLX charged-PME runtime fixes constraint tolerance at 1e-5"
        raise ValueError(msg)
    required = (
        args.mlx_prepared / "prepared_system.json",
        args.mlx_prepared / "prepared_system.npz",
        args.mlx_runtime,
        args.amber_prmtop,
        args.amber_coordinates,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        msg = "missing required input: " + ", ".join(missing)
        raise FileNotFoundError(msg)

    args.out.mkdir(parents=True, exist_ok=True)
    api = parity._load_openmm()
    source = parity._load_openmm_source(
        api,
        prmtop_path=args.amber_prmtop,
        coordinates_path=args.amber_coordinates,
    )
    config = parity._jac_pme_config(args.replicas)
    prepared = parity._normalize_mlx_prepared(
        load_prepared_system(args.mlx_prepared),
        source_atom_count=source["atom_count"],
        replicas=args.replicas,
        expected_cell_lengths=(
            source["base_cell_lengths_angstrom"] * np.asarray(args.replicas, dtype=np.float64)
        ),
        config=config,
    )
    reference = parity._build_openmm_replicas(
        api,
        source=source,
        replicas=args.replicas,
        config=config,
    )
    mlx_fixed = parity._mlx_manifest(
        prepared,
        source_atom_count=source["atom_count"],
        replicas=args.replicas,
        config=config,
    )
    openmm_fixed = parity._openmm_manifest(
        api,
        reference=reference,
        replicas=args.replicas,
        config=config,
        platform_name=args.platform,
        precision=args.precision,
    )
    require_matching_manifest(
        mlx_fixed,
        openmm_fixed,
        fields=parity.MANIFEST_FIELDS,
    )

    dynamics = {
        "integrator": "langevin_middle_baoab",
        "dt_ps": args.dt_ps,
        "temperature_k": args.temperature_k,
        "friction_per_ps": args.friction_per_ps,
        "constraint_tolerance": args.constraint_tolerance,
        "constraint_count": int(np.asarray(prepared.constraints).shape[0]),
        "fixed_cell": True,
        "warmup_steps": args.warmups,
        "measured_steps": args.steps,
        "initial_positions": "validated AMBER replica coordinates",
        "initial_velocities": "MLX prepared artifact velocities",
        "randomness": "engine-native streams; statistical rather than pathwise parity",
    }
    mlx_manifest = _runtime_manifest(mlx_fixed, dynamics=dynamics)
    openmm_manifest = _runtime_manifest(openmm_fixed, dynamics=dynamics)
    comparison = _manifest_comparison(mlx_manifest, openmm_manifest)
    require_matching_manifest(
        mlx_manifest,
        openmm_manifest,
        fields=RUNTIME_MANIFEST_FIELDS,
    )
    _write_json(args.out / MLX_MANIFEST_NAME, mlx_manifest)
    _write_json(args.out / OPENMM_MANIFEST_NAME, openmm_manifest)
    _write_json(args.out / MANIFEST_COMPARISON_NAME, comparison)

    openmm_runtime = _run_openmm(
        api=api,
        reference=reference,
        prepared=prepared,
        replicas=args.replicas,
        platform_name=args.platform,
        precision=args.precision,
        warmups=args.warmups,
        steps=args.steps,
        dt_ps=args.dt_ps,
        temperature_k=args.temperature_k,
        friction_per_ps=args.friction_per_ps,
        constraint_tolerance=args.constraint_tolerance,
        seed=args.seed,
        workload_manifest_hash=openmm_manifest["manifest_hash"],
    )
    _write_json(args.out / OPENMM_RUNTIME_NAME, openmm_runtime)
    admission = audit_openmm_runtime_artifacts(
        runtime_path=args.out / OPENMM_RUNTIME_NAME,
        workload_manifest_path=args.out / OPENMM_MANIFEST_NAME,
        manifest_comparison_path=args.out / MANIFEST_COMPARISON_NAME,
        out=args.out / ADMISSION_NAME,
    )

    mlx_runtime = json.loads(args.mlx_runtime.read_text())
    _write_json(args.out / "mlx_runtime.json", mlx_runtime)
    runtime_comparison = _runtime_comparison(
        mlx_runtime,
        openmm_runtime,
        expected_prepared=args.mlx_prepared,
        mlx_manifest_hash=mlx_manifest["manifest_hash"],
        openmm_manifest_hash=openmm_manifest["manifest_hash"],
    )
    _write_json(args.out / RUNTIME_COMPARISON_NAME, runtime_comparison)
    passed = admission["admitted"] is True and runtime_comparison["status"] == "comparable"
    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "openmm_runtime": openmm_runtime,
        "openmm_admission": admission,
        "runtime_comparison": runtime_comparison,
        "output_directory": str(args.out),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-prepared", type=Path, required=True)
    parser.add_argument("--mlx-runtime", type=Path, required=True)
    parser.add_argument("--amber-prmtop", type=Path, required=True)
    parser.add_argument("--amber-coordinates", type=Path, required=True)
    parser.add_argument("--replicas", type=_parse_replicas, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument(
        "--precision",
        choices=("single", "mixed", "double"),
        default="single",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--dt-ps", type=float, default=0.004)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--constraint-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_runtime(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
