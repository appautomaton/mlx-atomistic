"""Lightweight matched-workload evidence for DHFR NPT runtime diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

from mlx_atomistic.benchmarks.dhfr_npt import (
    payload_fingerprint,
    validate_prepared_boundary,
)
from mlx_atomistic.benchmarks.dhfr_npt_v2 import (
    DEFAULT_CONTRACT_PATH,
    contract_fingerprint,
    load_contract,
)

WORKLOAD_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-workload.v1"
WORKER_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-worker.v1"
SAMPLE_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-sample.v1"
BATCH_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-batch.v1"
PROFILE_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-profile.v1"
HOTSPOT_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-hotspots.v1"
STATUS_SCHEMA = "mlx-atomistic.dhfr-npt-runtime-status.v1"

RUNTIME_SEED = 313
BASELINE_STEPS = 750
PROFILE_STEPS = 75
PROCESS_TREE_MAX_BYTES = 40_000_000_000
SAMPLE_TIMEOUT_SECONDS = 600.0
EXPECTED_OPENMM_VERSION = "8.5.1.dev-f7fa0c2"


class DHFRNPTRuntimeError(RuntimeError):
    """Raised when DHFR NPT runtime evidence is incomplete or inconsistent."""


def load_runtime_workload(
    prepared_dir: str | Path,
    *,
    steps: int,
    seed: int = RUNTIME_SEED,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load the historical v2 workload as a runtime-only identity.

    Args:
        prepared_dir: Provenance-locked prepared DHFR artifact directory.
        steps: Exact integration-step count for this measurement.
        seed: Runtime diagnostic seed.
        contract_path: Historical v2 contract path.

    Returns:
        Engine-neutral workload identity and fingerprint.
    """

    contract = load_contract(contract_path)
    source = validate_prepared_boundary(prepared_dir, contract)
    return build_runtime_workload(
        contract,
        source_identity=source,
        steps=steps,
        seed=seed,
    )


def build_runtime_workload(
    contract: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    steps: int,
    seed: int = RUNTIME_SEED,
) -> dict[str, Any]:
    """Build one exact runtime workload identity from the historical contract."""

    workload = _mapping(contract.get("workload"), name="contract workload")
    target = _mapping(contract.get("target"), name="contract target")
    interval = _positive_int(
        _mapping(workload.get("barostat"), name="barostat").get("interval"),
        name="barostat interval",
    )
    if isinstance(steps, bool) or steps not in {BASELINE_STEPS, PROFILE_STEPS}:
        raise DHFRNPTRuntimeError(
            f"runtime steps must be {PROFILE_STEPS} or {BASELINE_STEPS}"
        )
    if steps % interval != 0:
        raise DHFRNPTRuntimeError("runtime steps must preserve barostat intervals")
    if isinstance(seed, bool) or seed != RUNTIME_SEED:
        raise DHFRNPTRuntimeError(
            f"runtime diagnostic seed must be {RUNTIME_SEED}"
        )
    source_fingerprint = source_identity.get("manifest_fingerprint")
    if not _is_sha256(source_fingerprint):
        raise DHFRNPTRuntimeError("runtime source fingerprint is invalid")
    protocol = json.loads(json.dumps(workload))
    protocol["steps"] = int(steps)
    protocol["seeds"] = [int(seed)]
    protocol["barostat"]["expected_attempts"] = int(steps) // interval
    unsigned = {
        "schema": WORKLOAD_SCHEMA,
        "historical_contract_schema": contract.get("schema"),
        "historical_contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source_fingerprint,
        "target": {
            "case_id": target.get("case_id"),
            "atom_count": target.get("atom_count"),
            "molecule_count": target.get("molecule_count"),
        },
        "protocol": protocol,
        "seed": int(seed),
        "steps": int(steps),
        "expected_attempts": int(steps) // interval,
        "role": "baseline" if steps == BASELINE_STEPS else "profile-prefix",
    }
    return {
        **unsigned,
        "workload_fingerprint": payload_fingerprint(unsigned),
    }


def artifact_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    """Return a content-addressed artifact record."""

    artifact = Path(path)
    root = Path(relative_to)
    relative = artifact.relative_to(root).as_posix()
    if not artifact.is_file() or artifact.is_symlink():
        raise DHFRNPTRuntimeError(f"runtime artifact is missing or unsafe: {relative}")
    return {
        "path": relative,
        "byte_size": artifact.stat().st_size,
        "sha256": _file_sha256(artifact),
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write finite deterministic JSON."""

    _require_finite_json(payload, context="JSON output")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_worker_report(
    path: str | Path,
    *,
    expected_engine: str | None = None,
) -> dict[str, Any]:
    """Load and validate one engine worker report."""

    report = _load_json(path, context="runtime worker report")
    if report.get("schema") != WORKER_SCHEMA:
        raise DHFRNPTRuntimeError("runtime worker schema is unsupported")
    _validate_engine(report, expected_engine=expected_engine)
    _validate_workload(_mapping(report.get("workload"), name="worker workload"))
    barrier = _mapping(report.get("completion_barrier"), name="completion barrier")
    if (
        barrier.get("performed") is not True
        or barrier.get("boundary") != "before_timer_stop"
        or not isinstance(barrier.get("kind"), str)
    ):
        raise DHFRNPTRuntimeError("runtime completion barrier is missing")
    timings = _mapping(report.get("timing"), name="worker timing")
    for name in (
        "worker_wall_seconds",
        "setup_wall_seconds",
        "steady_state_integration_wall_seconds",
        "synchronization_diagnostics_wall_seconds",
        "persistence_wall_seconds",
        "unaccounted_wall_seconds",
    ):
        _nonnegative_float(timings.get(name), name=name)
    _validate_timing_reconciliation(
        timings,
        total_name="worker_wall_seconds",
    )
    process = _mapping(report.get("process"), name="worker process")
    _positive_int(process.get("pid"), name="worker pid")
    if not isinstance(process.get("run_id"), str) or not process["run_id"]:
        raise DHFRNPTRuntimeError("worker run identifier is missing")
    checks = _mapping(report.get("checks"), name="worker checks")
    if not checks or any(value is not True for value in checks.values()):
        raise DHFRNPTRuntimeError("runtime worker numerical checks did not pass")
    _sequence(report.get("artifacts"), name="worker artifacts")
    _require_finite_json(report, context="runtime worker report")
    return report


def finalize_sample_report(
    worker_report: Mapping[str, Any],
    *,
    complete_wall_seconds: float,
    memory_trace: Mapping[str, Any],
    memory_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine worker, process, and memory evidence into one sample report."""

    worker = load_worker_report_payload(worker_report)
    complete = _positive_float(
        complete_wall_seconds,
        name="complete sample wall seconds",
    )
    memory = validate_memory_trace(
        memory_trace,
        expected_limit_bytes=PROCESS_TREE_MAX_BYTES,
    )
    worker_timing = _mapping(worker["timing"], name="worker timing")
    named = sum(
        float(worker_timing[name])
        for name in (
            "setup_wall_seconds",
            "steady_state_integration_wall_seconds",
            "synchronization_diagnostics_wall_seconds",
            "persistence_wall_seconds",
        )
    )
    unaccounted = max(0.0, complete - named)
    timing = {
        "complete_wall_seconds": complete,
        "setup_wall_seconds": float(worker_timing["setup_wall_seconds"]),
        "steady_state_integration_wall_seconds": float(
            worker_timing["steady_state_integration_wall_seconds"]
        ),
        "synchronization_diagnostics_wall_seconds": float(
            worker_timing["synchronization_diagnostics_wall_seconds"]
        ),
        "persistence_wall_seconds": float(worker_timing["persistence_wall_seconds"]),
        "unaccounted_wall_seconds": unaccounted,
    }
    unsigned = {
        "schema": SAMPLE_SCHEMA,
        "status": "passed",
        "engine": worker["engine"],
        "host": worker["host"],
        "workload": worker["workload"],
        "process": worker["process"],
        "completion_barrier": worker["completion_barrier"],
        "timing": timing,
        "numerical": worker["numerical"],
        "checks": worker["checks"],
        "artifacts": worker["artifacts"],
        "memory": {
            "limit_bytes": int(memory["bounded_process_limit_bytes"]),
            "peak_physical_bytes": int(memory["bounded_process_peak_physical_bytes"]),
            "trace": dict(memory_record),
        },
    }
    return {
        **unsigned,
        "report_fingerprint": payload_fingerprint(unsigned),
    }


def load_worker_report_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory worker report."""

    temporary = dict(payload)
    if temporary.get("schema") != WORKER_SCHEMA:
        raise DHFRNPTRuntimeError("runtime worker schema is unsupported")
    _validate_engine(temporary)
    _validate_workload(_mapping(temporary.get("workload"), name="worker workload"))
    barrier = _mapping(
        temporary.get("completion_barrier"),
        name="completion barrier",
    )
    if (
        barrier.get("performed") is not True
        or barrier.get("boundary") != "before_timer_stop"
        or not isinstance(barrier.get("kind"), str)
    ):
        raise DHFRNPTRuntimeError("runtime completion barrier is missing")
    timings = _mapping(temporary.get("timing"), name="worker timing")
    for name in (
        "worker_wall_seconds",
        "setup_wall_seconds",
        "steady_state_integration_wall_seconds",
        "synchronization_diagnostics_wall_seconds",
        "persistence_wall_seconds",
        "unaccounted_wall_seconds",
    ):
        _nonnegative_float(timings.get(name), name=name)
    _validate_timing_reconciliation(timings, total_name="worker_wall_seconds")
    process = _mapping(temporary.get("process"), name="worker process")
    _positive_int(process.get("pid"), name="worker pid")
    if not isinstance(process.get("run_id"), str) or not process["run_id"]:
        raise DHFRNPTRuntimeError("worker run identifier is missing")
    checks = _mapping(temporary.get("checks"), name="worker checks")
    if not checks or any(value is not True for value in checks.values()):
        raise DHFRNPTRuntimeError("runtime worker numerical checks did not pass")
    _require_finite_json(temporary, context="runtime worker report")
    return temporary


def load_sample_report(
    path: str | Path,
    *,
    expected_engine: str | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate one complete runtime sample report."""

    report = _load_json(path, context="runtime sample report")
    if report.get("schema") != SAMPLE_SCHEMA or report.get("status") != "passed":
        raise DHFRNPTRuntimeError("runtime sample is incomplete or unsupported")
    claimed = report.get("report_fingerprint")
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime sample fingerprint mismatch")
    _validate_engine(report, expected_engine=expected_engine)
    _validate_workload(_mapping(report.get("workload"), name="sample workload"))
    barrier = _mapping(report.get("completion_barrier"), name="completion barrier")
    if (
        barrier.get("performed") is not True
        or barrier.get("boundary") != "before_timer_stop"
    ):
        raise DHFRNPTRuntimeError("runtime sample lacks a completion barrier")
    timings = _mapping(report.get("timing"), name="sample timing")
    for name in (
        "complete_wall_seconds",
        "setup_wall_seconds",
        "steady_state_integration_wall_seconds",
        "synchronization_diagnostics_wall_seconds",
        "persistence_wall_seconds",
        "unaccounted_wall_seconds",
    ):
        _nonnegative_float(timings.get(name), name=name)
    _validate_timing_reconciliation(
        timings,
        total_name="complete_wall_seconds",
    )
    memory = _mapping(report.get("memory"), name="sample memory")
    if (
        memory.get("limit_bytes") != PROCESS_TREE_MAX_BYTES
        or not 0 <= int(memory.get("peak_physical_bytes", -1)) <= PROCESS_TREE_MAX_BYTES
    ):
        raise DHFRNPTRuntimeError("runtime sample memory boundary failed")
    checks = _mapping(report.get("checks"), name="sample checks")
    if not checks or any(value is not True for value in checks.values()):
        raise DHFRNPTRuntimeError("runtime sample numerical checks did not pass")
    if artifact_root is not None:
        for record in _sequence(report.get("artifacts"), name="sample artifacts"):
            _validate_artifact_record(artifact_root, record)
        _validate_artifact_record(
            artifact_root,
            _mapping(memory.get("trace"), name="memory trace record"),
        )
    _require_finite_json(report, context="runtime sample report")
    return report


def build_batch_report(
    samples: Sequence[Mapping[str, Any]],
    *,
    engine: str,
) -> dict[str, Any]:
    """Aggregate independently validated samples without hiding raw timings."""

    if not samples:
        raise DHFRNPTRuntimeError("runtime batch requires at least one sample")
    validated = [validate_sample_report_payload(sample, engine=engine) for sample in samples]
    workload_fingerprints = {
        sample["workload"]["workload_fingerprint"] for sample in validated
    }
    run_ids = {sample["process"]["run_id"] for sample in validated}
    if len(workload_fingerprints) != 1:
        raise DHFRNPTRuntimeError("runtime batch mixed workload identities")
    if len(run_ids) != len(validated):
        raise DHFRNPTRuntimeError("runtime batch reused a worker process identity")
    raw = [float(sample["timing"]["complete_wall_seconds"]) for sample in validated]
    unsigned = {
        "schema": BATCH_SCHEMA,
        "status": "passed",
        "engine": engine,
        "sample_count": len(validated),
        "workload_fingerprint": next(iter(workload_fingerprints)),
        "complete_wall_seconds_samples": raw,
        "median_complete_wall_seconds": float(median(raw)),
        "samples": [
            {
                "index": index,
                "report_fingerprint": sample["report_fingerprint"],
                "process_run_id": sample["process"]["run_id"],
                "complete_wall_seconds": sample["timing"]["complete_wall_seconds"],
                "power": sample["host"].get("power"),
            }
            for index, sample in enumerate(validated, start=1)
        ],
    }
    return {
        **unsigned,
        "report_fingerprint": payload_fingerprint(unsigned),
    }


def load_batch_report(
    path: str | Path,
    *,
    expected_engine: str | None = None,
) -> dict[str, Any]:
    """Load and validate one aggregate runtime report."""

    report = _load_json(path, context="runtime batch report")
    if report.get("schema") != BATCH_SCHEMA or report.get("status") != "passed":
        raise DHFRNPTRuntimeError("runtime batch is incomplete or unsupported")
    if expected_engine is not None and report.get("engine") != expected_engine:
        raise DHFRNPTRuntimeError("runtime batch engine mismatch")
    claimed = report.get("report_fingerprint")
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime batch fingerprint mismatch")
    raw = [
        _positive_float(value, name="batch sample wall seconds")
        for value in _sequence(
            report.get("complete_wall_seconds_samples"),
            name="batch raw samples",
        )
    ]
    if len(raw) != _positive_int(report.get("sample_count"), name="sample count"):
        raise DHFRNPTRuntimeError("runtime batch sample count mismatch")
    if not math.isclose(
        float(report.get("median_complete_wall_seconds", math.nan)),
        float(median(raw)),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise DHFRNPTRuntimeError("runtime batch median mismatch")
    return report


def validate_sample_report_payload(
    payload: Mapping[str, Any],
    *,
    engine: str,
) -> dict[str, Any]:
    """Validate a sample payload already loaded by the batch runner."""

    report = dict(payload)
    if report.get("schema") != SAMPLE_SCHEMA or report.get("status") != "passed":
        raise DHFRNPTRuntimeError("runtime sample is incomplete or unsupported")
    claimed = report.get("report_fingerprint")
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime sample fingerprint mismatch")
    _validate_engine(report, expected_engine=engine)
    _validate_workload(_mapping(report.get("workload"), name="sample workload"))
    _validate_timing_reconciliation(
        _mapping(report.get("timing"), name="sample timing"),
        total_name="complete_wall_seconds",
    )
    return report


def validate_memory_trace(
    payload: Mapping[str, Any],
    *,
    expected_limit_bytes: int,
) -> dict[str, Any]:
    """Validate a bounded-process memory trace."""

    trace = dict(payload)
    if (
        trace.get("bounded_process_exceeded") is not False
        or trace.get("bounded_process_timed_out") is not False
        or trace.get("bounded_process_returncode") != 0
        or trace.get("bounded_process_limit_bytes") != expected_limit_bytes
    ):
        raise DHFRNPTRuntimeError("runtime process supervisor did not pass")
    peak = int(trace.get("bounded_process_peak_physical_bytes", -1))
    if not 0 <= peak <= expected_limit_bytes:
        raise DHFRNPTRuntimeError("runtime process exceeded its memory boundary")
    return trace


def _validate_engine(
    report: Mapping[str, Any],
    *,
    expected_engine: str | None = None,
) -> None:
    engine = _mapping(report.get("engine"), name="runtime engine")
    name = engine.get("name")
    if expected_engine is not None and name != expected_engine:
        raise DHFRNPTRuntimeError("runtime engine label mismatch")
    if name == "openmm":
        if (
            engine.get("platform") != "OpenCL"
            or engine.get("precision") != "single"
            or engine.get("version") != EXPECTED_OPENMM_VERSION
        ):
            raise DHFRNPTRuntimeError("OpenMM runtime identity mismatch")
    elif name == "mlx":
        if engine.get("backend") != "Metal":
            raise DHFRNPTRuntimeError("MLX runtime did not use Metal")
    else:
        raise DHFRNPTRuntimeError("runtime engine is unsupported")


def _validate_workload(workload: Mapping[str, Any]) -> None:
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise DHFRNPTRuntimeError("runtime workload schema is unsupported")
    claimed = workload.get("workload_fingerprint")
    unsigned = dict(workload)
    unsigned.pop("workload_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime workload fingerprint mismatch")
    if workload.get("seed") != RUNTIME_SEED:
        raise DHFRNPTRuntimeError("runtime workload seed mismatch")
    steps = workload.get("steps")
    if steps not in {PROFILE_STEPS, BASELINE_STEPS}:
        raise DHFRNPTRuntimeError("runtime workload step count mismatch")
    if workload.get("expected_attempts") != int(steps) // 25:
        raise DHFRNPTRuntimeError("runtime workload attempt count mismatch")


def _validate_timing_reconciliation(
    timings: Mapping[str, Any],
    *,
    total_name: str,
) -> None:
    total = _nonnegative_float(timings.get(total_name), name=total_name)
    components = sum(
        _nonnegative_float(timings.get(name), name=name)
        for name in (
            "setup_wall_seconds",
            "steady_state_integration_wall_seconds",
            "synchronization_diagnostics_wall_seconds",
            "persistence_wall_seconds",
            "unaccounted_wall_seconds",
        )
    )
    tolerance = max(1.0e-6, 1.0e-6 * max(total, 1.0))
    if not math.isclose(total, components, rel_tol=0.0, abs_tol=tolerance):
        raise DHFRNPTRuntimeError("runtime timing components do not reconcile")


def _validate_artifact_record(root: str | Path, record: Mapping[str, Any]) -> None:
    relative = record.get("path")
    if not isinstance(relative, str):
        raise DHFRNPTRuntimeError("runtime artifact path is missing")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise DHFRNPTRuntimeError("runtime artifact path escapes its root")
    path = Path(root) / Path(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise DHFRNPTRuntimeError(f"runtime artifact is missing or unsafe: {relative}")
    if path.stat().st_size != int(record.get("byte_size", -1)):
        raise DHFRNPTRuntimeError(f"runtime artifact size mismatch: {relative}")
    if _file_sha256(path) != record.get("sha256"):
        raise DHFRNPTRuntimeError(f"runtime artifact digest mismatch: {relative}")


def _load_json(path: str | Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DHFRNPTRuntimeError(f"cannot load {context}") from error
    if not isinstance(payload, dict):
        raise DHFRNPTRuntimeError(f"{context} must be a JSON object")
    return payload


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DHFRNPTRuntimeError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DHFRNPTRuntimeError(f"{name} must be an array")
    return list(value)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DHFRNPTRuntimeError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, *, name: str) -> float:
    number = _nonnegative_float(value, name=name)
    if number <= 0.0:
        raise DHFRNPTRuntimeError(f"{name} must be positive")
    return number


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DHFRNPTRuntimeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DHFRNPTRuntimeError(f"{name} must be finite and non-negative")
    return number


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_json(child, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            _require_finite_json(child, context=context)
    elif isinstance(value, float) and not math.isfinite(value):
        raise DHFRNPTRuntimeError(f"{context} contains non-finite values")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
