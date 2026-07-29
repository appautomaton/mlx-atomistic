"""Run isolated, resumable DHFR NPT runtime measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform as host_platform
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks import dhfr_npt_runtime as runtime
from mlx_atomistic.benchmarks.dhfr_npt import DHFRNPTValidationError
from mlx_atomistic.benchmarks.dhfr_npt_v2 import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_PROCESS_SCRIPT = REPO_ROOT / "scripts" / "run_bounded_process.py"


def build_bounded_worker_command(
    *,
    engine: str,
    prepared: Path,
    sample_dir: Path,
    steps: int,
    seed: int,
    platform_name: str,
    precision: str,
    contract: Path,
    max_bytes: int,
    timeout_seconds: float,
) -> list[str]:
    """Build one fresh-process worker command under the macOS supervisor."""

    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--engine",
        engine,
        "--prepared",
        str(prepared),
        "--out",
        str(sample_dir),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--platform",
        platform_name,
        "--precision",
        precision,
        "--contract",
        str(contract),
    ]
    return [
        sys.executable,
        str(BOUNDED_PROCESS_SCRIPT),
        "--max-bytes",
        str(max_bytes),
        "--timeout-seconds",
        str(timeout_seconds),
        "--trace-out",
        str(sample_dir / "memory.json"),
        "--",
        *worker,
    ]


def run_batch(
    *,
    engine: str,
    prepared: Path,
    out: Path,
    steps: int,
    repetitions: int,
    seed: int,
    platform_name: str,
    precision: str,
    contract_path: Path,
    max_bytes: int,
    sample_timeout_seconds: float,
) -> dict[str, Any]:
    """Run or resume an isolated runtime batch."""

    if repetitions <= 0:
        raise runtime.DHFRNPTRuntimeError("repetitions must be positive")
    if max_bytes != runtime.PROCESS_TREE_MAX_BYTES:
        raise runtime.DHFRNPTRuntimeError("runtime batch memory limit drifted")
    if not math.isclose(
        sample_timeout_seconds,
        runtime.SAMPLE_TIMEOUT_SECONDS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise runtime.DHFRNPTRuntimeError("runtime sample timeout drifted")
    workload = runtime.load_runtime_workload(
        prepared,
        steps=steps,
        seed=seed,
        contract_path=contract_path,
    )
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    samples: list[dict[str, Any]] = []
    _write_status(
        status_path,
        state="running",
        engine=engine,
        completed=0,
        requested=repetitions,
        active_sample=None,
    )
    try:
        for index in range(1, repetitions + 1):
            sample_dir = out / f"sample-{index:03d}"
            report_path = sample_dir / "report.json"
            if report_path.is_file():
                report = runtime.load_sample_report(
                    report_path,
                    expected_engine=engine,
                    artifact_root=sample_dir,
                )
                _require_workload(report, workload)
                samples.append(report)
                _write_status(
                    status_path,
                    state="running",
                    engine=engine,
                    completed=len(samples),
                    requested=repetitions,
                    active_sample=None,
                )
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_status(
                status_path,
                state="running",
                engine=engine,
                completed=len(samples),
                requested=repetitions,
                active_sample=index,
            )
            command = build_bounded_worker_command(
                engine=engine,
                prepared=prepared,
                sample_dir=sample_dir,
                steps=steps,
                seed=seed,
                platform_name=platform_name,
                precision=precision,
                contract=contract_path,
                max_bytes=max_bytes,
                timeout_seconds=sample_timeout_seconds,
            )
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            complete_wall_seconds = time.perf_counter() - started
            if completed.returncode != 0:
                raise runtime.DHFRNPTRuntimeError(
                    f"runtime sample {index} failed with exit code "
                    f"{completed.returncode}"
                )
            worker_path = sample_dir / "worker.json"
            memory_path = sample_dir / "memory.json"
            worker = runtime.load_worker_report(
                worker_path,
                expected_engine=engine,
            )
            _require_workload(worker, workload)
            worker["artifacts"] = [
                *worker["artifacts"],
                runtime.artifact_record(worker_path, relative_to=sample_dir),
            ]
            memory_payload = _load_json(memory_path)
            sample = runtime.finalize_sample_report(
                worker,
                complete_wall_seconds=complete_wall_seconds,
                memory_trace=memory_payload,
                memory_record=runtime.artifact_record(
                    memory_path,
                    relative_to=sample_dir,
                ),
            )
            runtime.atomic_write_json(report_path, sample)
            report = runtime.load_sample_report(
                report_path,
                expected_engine=engine,
                artifact_root=sample_dir,
            )
            samples.append(report)
            _write_status(
                status_path,
                state="running",
                engine=engine,
                completed=len(samples),
                requested=repetitions,
                active_sample=None,
            )
        batch = runtime.build_batch_report(samples, engine=engine)
        runtime.atomic_write_json(out / "report.json", batch)
        verified = verify_batch(out / "report.json", engine=engine)
        _write_status(
            status_path,
            state="passed",
            engine=engine,
            completed=repetitions,
            requested=repetitions,
            active_sample=None,
        )
        return verified
    except BaseException as error:
        _write_status(
            status_path,
            state="failed",
            engine=engine,
            completed=len(samples),
            requested=repetitions,
            active_sample=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def verify_batch(path: Path, *, engine: str) -> dict[str, Any]:
    """Verify a batch report and every independently persisted sample."""

    batch = runtime.load_batch_report(path, expected_engine=engine)
    root = path.parent
    samples = []
    for index in range(1, int(batch["sample_count"]) + 1):
        sample_dir = root / f"sample-{index:03d}"
        sample = runtime.load_sample_report(
            sample_dir / "report.json",
            expected_engine=engine,
            artifact_root=sample_dir,
        )
        samples.append(sample)
    rebuilt = runtime.build_batch_report(samples, engine=engine)
    if rebuilt != batch:
        raise runtime.DHFRNPTRuntimeError(
            "runtime batch does not reconcile with its sample reports"
        )
    return batch


def read_status(path: Path) -> dict[str, Any]:
    """Return the current batch status without mutating it."""

    status_path = path / "status.json"
    if status_path.is_file():
        payload = _load_json(status_path)
        if payload.get("schema") != runtime.STATUS_SCHEMA:
            raise runtime.DHFRNPTRuntimeError("runtime status schema is unsupported")
        return payload
    report_path = path / "report.json"
    if report_path.is_file():
        report = runtime.load_batch_report(report_path)
        return {
            "schema": runtime.STATUS_SCHEMA,
            "state": "passed",
            "engine": report["engine"],
            "completed": report["sample_count"],
            "requested": report["sample_count"],
            "active_sample": None,
        }
    return {
        "schema": runtime.STATUS_SCHEMA,
        "state": "not-started",
        "completed": 0,
        "requested": None,
        "active_sample": None,
    }


def run_worker(
    *,
    engine: str,
    prepared: Path,
    out: Path,
    steps: int,
    seed: int,
    platform_name: str,
    precision: str,
    contract_path: Path,
) -> dict[str, Any]:
    """Run one engine sample inside a bounded child process."""

    out.mkdir(parents=True, exist_ok=True)
    if engine == "openmm":
        report = _run_openmm_worker(
            prepared=prepared,
            out=out,
            steps=steps,
            seed=seed,
            platform_name=platform_name,
            precision=precision,
            contract_path=contract_path,
        )
    elif engine == "mlx":
        report = _run_mlx_worker(
            prepared=prepared,
            out=out,
            steps=steps,
            seed=seed,
            contract_path=contract_path,
        )
    else:
        raise runtime.DHFRNPTRuntimeError(f"unsupported runtime engine: {engine}")
    runtime.atomic_write_json(out / "worker.json", report)
    runtime.load_worker_report(out / "worker.json", expected_engine=engine)
    return report


def _run_openmm_worker(
    *,
    prepared: Path,
    out: Path,
    steps: int,
    seed: int,
    platform_name: str,
    precision: str,
    contract_path: Path,
) -> dict[str, Any]:
    worker_started = time.perf_counter()
    from scripts import run_openmm_mlx_dhfr_npt as v1_runner

    setup_started = time.perf_counter()
    contract = load_contract(contract_path)
    workload_identity = runtime.load_runtime_workload(
        prepared,
        steps=steps,
        seed=seed,
        contract_path=contract_path,
    )
    prepared_system = v1_runner._npt_prepared(
        v1_runner.load_prepared_system(prepared)
    )
    api, system = v1_runner._build_openmm_system(contract)
    mm = api.openmm
    unit = api.unit
    workload = workload_identity["protocol"]
    interval = int(workload["barostat"]["interval"])
    barostat = mm.MonteCarloAnisotropicBarostat(
        mm.Vec3(
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
        )
        * unit.bar,
        float(workload["temperature_K"]) * unit.kelvin,
        True,
        True,
        True,
        interval,
    )
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        float(workload["temperature_K"]) * unit.kelvin,
        float(workload["friction_per_ps"]) / unit.picosecond,
        float(workload["dt_ps"]) * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    selected_platform = mm.Platform.getPlatformByName(platform_name)
    context = mm.Context(
        system,
        integrator,
        selected_platform,
        {"Precision": precision},
    )
    positions = np.asarray(prepared_system.positions, dtype=np.float64)
    initial_lengths = np.diag(
        np.asarray(prepared_system.cell_matrix, dtype=np.float64)
    )
    context.setPeriodicBoxVectors(
        *v1_runner._openmm_box(mm, unit, initial_lengths)
    )
    context.setPositions(positions * 0.1 * unit.nanometer)
    context.applyConstraints(integrator.getConstraintTolerance())
    context.setVelocitiesToTemperature(
        float(workload["temperature_K"]) * unit.kelvin,
        seed,
    )
    context.applyVelocityConstraints(integrator.getConstraintTolerance())
    setup_seconds = time.perf_counter() - setup_started

    integration_seconds = 0.0
    diagnostics_seconds = 0.0
    sample_started = time.perf_counter()
    sampled = [v1_runner._openmm_sample(context, prepared_system, api)]
    diagnostics_seconds += time.perf_counter() - sample_started
    for _ in range(steps // interval):
        step_started = time.perf_counter()
        integrator.step(interval)
        integration_seconds += time.perf_counter() - step_started
        sample_started = time.perf_counter()
        sampled.append(v1_runner._openmm_sample(context, prepared_system, api))
        diagnostics_seconds += time.perf_counter() - sample_started

    barrier_started = time.perf_counter()
    final_state = context.getState(getEnergy=True)
    final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    diagnostics_seconds += time.perf_counter() - barrier_started
    arrays = {
        name: np.asarray([sample[name] for sample in sampled])
        for name in (
            "cell_matrix_angstrom",
            "volume_angstrom3",
            "potential_energy_kj_mol",
            "kinetic_energy_kj_mol",
            "total_energy_kj_mol",
            "temperature_K",
            "pressure_bar",
            "constraint_error_angstrom",
        )
    }
    numerical = _openmm_numerical_summary(
        arrays,
        atom_count=prepared_system.atom_count,
        expected_attempts=steps // interval,
    )
    persistence_started = time.perf_counter()
    samples_path = out / "samples.npz"
    v1_runner._atomic_savez(samples_path, arrays)
    persistence_seconds = time.perf_counter() - persistence_started
    worker_wall_seconds = time.perf_counter() - worker_started
    unaccounted = max(
        0.0,
        worker_wall_seconds
        - setup_seconds
        - integration_seconds
        - diagnostics_seconds
        - persistence_seconds,
    )
    actual_platform = context.getPlatform()
    actual_precision = actual_platform.getPropertyValue(context, "Precision")
    properties = {
        name: actual_platform.getPropertyValue(context, name)
        for name in actual_platform.getPropertyNames()
        if name in {"DeviceIndex", "DeviceName", "Precision"}
    }
    report = {
        "schema": runtime.WORKER_SCHEMA,
        "engine": {
            "name": "openmm",
            "platform": actual_platform.getName(),
            "precision": actual_precision,
            "version": api.openmm.version.version,
            "device": properties,
            "role": "reference-only",
        },
        "host": _host_metadata(),
        "workload": workload_identity,
        "process": {
            "pid": os.getpid(),
            "run_id": str(uuid.uuid4()),
        },
        "completion_barrier": {
            "performed": True,
            "boundary": "before_timer_stop",
            "kind": "OpenMM Context.getState energy materialization",
        },
        "timing": {
            "worker_wall_seconds": worker_wall_seconds,
            "setup_wall_seconds": setup_seconds,
            "steady_state_integration_wall_seconds": integration_seconds,
            "synchronization_diagnostics_wall_seconds": diagnostics_seconds,
            "persistence_wall_seconds": persistence_seconds,
            "unaccounted_wall_seconds": unaccounted,
        },
        "numerical": numerical,
        "checks": _numerical_checks(
            numerical,
            contract=contract,
            expected_attempts=steps // interval,
        ),
        "artifacts": [
            runtime.artifact_record(samples_path, relative_to=out),
        ],
    }
    del context
    del integrator
    return report


def _run_mlx_worker(
    *,
    prepared: Path,
    out: Path,
    steps: int,
    seed: int,
    contract_path: Path,
) -> dict[str, Any]:
    worker_started = time.perf_counter()
    import mlx
    import mlx.core as mx

    from scripts import run_openmm_mlx_dhfr_npt as v1_runner

    mx.set_default_device(mx.gpu)
    setup_started = time.perf_counter()
    contract = load_contract(contract_path)
    workload_identity = runtime.load_runtime_workload(
        prepared,
        steps=steps,
        seed=seed,
        contract_path=contract_path,
    )
    prepared_system = v1_runner._npt_prepared(
        v1_runner.load_prepared_system(prepared)
    )
    trajectory = out / "trajectory.npz"
    checkpoint = out / "checkpoint.npz"
    setup_seconds = time.perf_counter() - setup_started
    integration_started = time.perf_counter()
    result = v1_runner._run_mlx_npt(
        prepared_system,
        workload=workload_identity["protocol"],
        seed=seed,
        out=trajectory,
        checkpoint_out=checkpoint,
        steps=steps,
        constraint_max_iterations=int(
            workload_identity["protocol"]["constraint_max_iterations"]
        ),
    )
    integration_seconds = time.perf_counter() - integration_started
    barrier_started = time.perf_counter()
    mx.eval(
        result.final_state.positions,
        result.final_state.velocities,
        result.final_state.forces,
    )
    synchronization_seconds = time.perf_counter() - barrier_started
    numerical = v1_runner._mlx_npt_summary(
        result,
        atom_count=prepared_system.atom_count,
    )
    numerical["configured_barostat_attempts"] = int(
        numerical.pop("barostat_attempts")
    )
    numerical["observed_cell_changes"] = int(
        numerical.pop("barostat_accepted")
    )
    worker_wall_seconds = time.perf_counter() - worker_started
    unaccounted = max(
        0.0,
        worker_wall_seconds
        - setup_seconds
        - integration_seconds
        - synchronization_seconds,
    )
    return {
        "schema": runtime.WORKER_SCHEMA,
        "engine": {
            "name": "mlx",
            "backend": "Metal",
            "device": str(mx.default_device()),
            "version": mlx.__version__,
            "role": "product-runtime",
        },
        "host": _host_metadata(),
        "workload": workload_identity,
        "process": {
            "pid": os.getpid(),
            "run_id": str(uuid.uuid4()),
        },
        "completion_barrier": {
            "performed": True,
            "boundary": "before_timer_stop",
            "kind": "mlx.core.eval final state materialization",
        },
        "timing": {
            "worker_wall_seconds": worker_wall_seconds,
            "setup_wall_seconds": setup_seconds,
            "steady_state_integration_wall_seconds": integration_seconds,
            "synchronization_diagnostics_wall_seconds": synchronization_seconds,
            "persistence_wall_seconds": 0.0,
            "unaccounted_wall_seconds": unaccounted,
        },
        "numerical": numerical,
        "checks": _numerical_checks(
            numerical,
            contract=contract,
            expected_attempts=steps // 25,
        ),
        "runtime": {
            "nonbonded": dict(result.nonbonded_report),
            "synchronization": dict(result.runtime_sync_report),
            "barostat": dict(result.barostat_metadata),
        },
        "artifacts": [
            runtime.artifact_record(trajectory, relative_to=out),
            runtime.artifact_record(checkpoint, relative_to=out),
        ],
    }


def _openmm_numerical_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    atom_count: int,
    expected_attempts: int,
) -> dict[str, Any]:
    cells = np.asarray(arrays["cell_matrix_angstrom"], dtype=np.float64)
    volumes = np.asarray(arrays["volume_angstrom3"], dtype=np.float64)
    volume_ratios = volumes / volumes[0]
    total_energies = np.asarray(
        arrays["total_energy_kj_mol"],
        dtype=np.float64,
    )
    cell_changes = int(
        np.count_nonzero(
            np.max(np.abs(np.diff(cells, axis=0)), axis=(1, 2)) > 1.0e-7
        )
    )
    return {
        "finite": bool(
            all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values())
        ),
        "sample_count": int(cells.shape[0]),
        "configured_barostat_attempts": int(expected_attempts),
        "observed_cell_changes": cell_changes,
        "minimum_volume_ratio": float(np.min(volume_ratios)),
        "maximum_volume_ratio": float(np.max(volume_ratios)),
        "mean_volume_ratio": float(np.mean(volume_ratios)),
        "maximum_cell_off_diagonal_angstrom": _maximum_off_diagonal(cells),
        "maximum_constraint_error_angstrom": float(
            np.max(np.asarray(arrays["constraint_error_angstrom"]))
        ),
        "maximum_temperature_K": float(
            np.max(np.asarray(arrays["temperature_K"]))
        ),
        "mean_pressure_bar": float(np.mean(np.asarray(arrays["pressure_bar"]))),
        "maximum_abs_pressure_bar": float(
            np.max(np.abs(np.asarray(arrays["pressure_bar"])))
        ),
        "maximum_energy_excursion_per_atom_kj_mol": (
            float(np.max(total_energies) - np.min(total_energies)) / atom_count
        ),
    }


def _numerical_checks(
    numerical: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_attempts: int,
) -> dict[str, bool]:
    gates = contract["npt_gates"]
    return {
        "finite": numerical.get("finite") is True,
        "attempt_schedule": (
            numerical.get("configured_barostat_attempts") == expected_attempts
        ),
        "sample_count": numerical.get("sample_count") == expected_attempts + 1,
        "constraints": (
            float(numerical["maximum_constraint_error_angstrom"])
            <= float(gates["maximum_constraint_error_angstrom"])
        ),
        "volume_bounds": (
            float(gates["minimum_volume_ratio"])
            <= float(numerical["minimum_volume_ratio"])
            <= float(gates["maximum_volume_ratio"])
            and float(gates["minimum_volume_ratio"])
            <= float(numerical["maximum_volume_ratio"])
            <= float(gates["maximum_volume_ratio"])
        ),
        "orthorhombic_cells": (
            float(numerical["maximum_cell_off_diagonal_angstrom"])
            <= float(gates["orthorhombic_off_diagonal_tolerance_angstrom"])
        ),
        "temperature_bounds": (
            float(numerical["maximum_temperature_K"])
            <= float(gates["maximum_temperature_K"])
        ),
        "pressure_bounds": (
            float(numerical["maximum_abs_pressure_bar"])
            <= float(gates["maximum_abs_pressure_bar"])
        ),
        "energy_stability": (
            float(numerical["maximum_energy_excursion_per_atom_kj_mol"])
            <= float(gates["maximum_energy_excursion_per_atom_kj_mol"])
        ),
    }


def _maximum_off_diagonal(cells: np.ndarray) -> float:
    diagonal = np.zeros_like(cells)
    indices = np.arange(3)
    diagonal[:, indices, indices] = cells[:, indices, indices]
    return float(np.max(np.abs(cells - diagonal)))


def _host_metadata() -> dict[str, Any]:
    return {
        "machine": host_platform.machine(),
        "platform": host_platform.platform(),
        "chip": _read_command(("sysctl", "-n", "machdep.cpu.brand_string")),
        "power": {
            "source": _read_command(("pmset", "-g", "batt")),
            "settings": _read_command(("pmset", "-g", "custom")),
        },
    }


def _read_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            tuple(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _require_workload(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    actual = report["workload"]["workload_fingerprint"]
    if actual != expected["workload_fingerprint"]:
        raise runtime.DHFRNPTRuntimeError("runtime sample workload drifted")


def _write_status(
    path: Path,
    *,
    state: str,
    engine: str,
    completed: int,
    requested: int,
    active_sample: int | None,
    error: str | None = None,
) -> None:
    payload = {
        "schema": runtime.STATUS_SCHEMA,
        "state": state,
        "engine": engine,
        "completed": completed,
        "requested": requested,
        "active_sample": active_sample,
        "error": error,
    }
    runtime.atomic_write_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise runtime.DHFRNPTRuntimeError(
            f"cannot load runtime JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise runtime.DHFRNPTRuntimeError(f"runtime JSON must be an object: {path}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch")
    batch.add_argument("--engine", choices=("openmm", "mlx"), required=True)
    batch.add_argument("--prepared", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument(
        "--steps",
        type=int,
        choices=(runtime.PROFILE_STEPS, runtime.BASELINE_STEPS),
        required=True,
    )
    batch.add_argument("--repetitions", type=int, required=True)
    batch.add_argument("--seed", type=int, default=runtime.RUNTIME_SEED)
    batch.add_argument("--platform", default="OpenCL")
    batch.add_argument("--precision", default="single")
    batch.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    batch.add_argument(
        "--max-bytes",
        type=int,
        default=runtime.PROCESS_TREE_MAX_BYTES,
    )
    batch.add_argument(
        "--sample-timeout-seconds",
        type=float,
        default=runtime.SAMPLE_TIMEOUT_SECONDS,
    )

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--engine", choices=("openmm", "mlx"), required=True)
    worker.add_argument("--prepared", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument(
        "--steps",
        type=int,
        choices=(runtime.PROFILE_STEPS, runtime.BASELINE_STEPS),
        required=True,
    )
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--platform", required=True)
    worker.add_argument("--precision", required=True)
    worker.add_argument("--contract", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--input", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--engine", choices=("openmm", "mlx"), required=True)
    verify.add_argument("--input", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run, inspect, or verify one DHFR NPT runtime batch."""

    args = _parse_args(argv)
    if args.command == "batch":
        report = run_batch(
            engine=args.engine,
            prepared=args.prepared,
            out=args.out,
            steps=args.steps,
            repetitions=args.repetitions,
            seed=args.seed,
            platform_name=args.platform,
            precision=args.precision,
            contract_path=args.contract,
            max_bytes=args.max_bytes,
            sample_timeout_seconds=args.sample_timeout_seconds,
        )
    elif args.command == "_worker":
        report = run_worker(
            engine=args.engine,
            prepared=args.prepared,
            out=args.out,
            steps=args.steps,
            seed=args.seed,
            platform_name=args.platform,
            precision=args.precision,
            contract_path=args.contract,
        )
    elif args.command == "status":
        report = read_status(args.input)
    else:
        report = verify_batch(args.input, engine=args.engine)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (runtime.DHFRNPTRuntimeError, DHFRNPTValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
