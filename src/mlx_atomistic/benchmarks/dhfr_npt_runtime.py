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
    for name in ("mode", "runtime", "profile"):
        if name in worker:
            unsigned[name] = worker[name]
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


def build_profile_report(
    clean_sample: Mapping[str, Any],
    instrumented_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one diagnostic clean-versus-instrumented MLX profile report."""

    clean = validate_sample_report_payload(clean_sample, engine="mlx")
    instrumented = validate_sample_report_payload(
        instrumented_sample,
        engine="mlx",
    )
    clean_workload = _mapping(clean["workload"], name="clean profile workload")
    instrumented_workload = _mapping(
        instrumented["workload"],
        name="instrumented profile workload",
    )
    if (
        clean_workload["workload_fingerprint"]
        != instrumented_workload["workload_fingerprint"]
        or clean_workload["steps"] != PROFILE_STEPS
    ):
        raise DHFRNPTRuntimeError("profile samples do not share the 75-step workload")
    if instrumented.get("mode") != "instrumented":
        raise DHFRNPTRuntimeError("instrumented profile sample is not labeled")
    profile = _mapping(
        instrumented.get("profile"),
        name="instrumented route profile",
    )
    if (
        profile.get("profile_steps") != PROFILE_STEPS
        or profile.get("projection_steps") != BASELINE_STEPS
    ):
        raise DHFRNPTRuntimeError("instrumented route profile schedule drifted")
    profile = json.loads(json.dumps(profile))
    routes = _mapping(profile.get("routes"), name="instrumented profile routes")
    if not routes:
        raise DHFRNPTRuntimeError("instrumented route profile is empty")
    for route, details in routes.items():
        if not isinstance(route, str) or not route:
            raise DHFRNPTRuntimeError("profile route name is invalid")
        route_details = _mapping(details, name=f"profile route {route}")
        calls = route_details.get("calls")
        projected_calls = route_details.get("projected_750_calls")
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(projected_calls, bool)
            or not isinstance(projected_calls, int)
            or projected_calls < 0
        ):
            raise DHFRNPTRuntimeError("profile route call counts are invalid")
        for name in (
            "first_inclusive_wall_seconds",
            "first_exclusive_wall_seconds",
            "recurring_inclusive_mean_wall_seconds",
            "recurring_exclusive_mean_wall_seconds",
            "total_inclusive_wall_seconds",
            "total_exclusive_wall_seconds",
            "projected_750_exclusive_wall_seconds",
            "recurring_exclusive_stddev_wall_seconds",
            "projected_750_uncertainty_wall_seconds",
        ):
            _nonnegative_float(
                route_details.get(name, 0.0),
                name=f"{route} {name}",
            )
        projected_calls = _projected_route_call_count(
            calls=calls,
        )
        route_details["projected_750_calls"] = projected_calls
        route_details["projected_750_exclusive_wall_seconds"] = (
            float(route_details["first_exclusive_wall_seconds"])
            + max(0, projected_calls - 1)
            * float(route_details["recurring_exclusive_mean_wall_seconds"])
        )
        recurring_samples = int(route_details.get("recurring_sample_count", 0))
        if recurring_samples >= 2:
            standard_error = float(
                route_details.get(
                    "recurring_exclusive_stddev_wall_seconds",
                    0.0,
                )
            ) / math.sqrt(recurring_samples)
            route_details["projected_750_uncertainty_wall_seconds"] = (
                max(0, projected_calls - 1) * standard_error
            )
            route_details["uncertainty_status"] = "estimated"
        else:
            route_details["projected_750_uncertainty_wall_seconds"] = 0.0
            route_details["uncertainty_status"] = "insufficient"
        route_details["first_occurrence_classification"] = (
            "cold compilation/allocation/cache-population candidate"
        )
        route_details["later_occurrences_classification"] = (
            "recurring route cost"
        )
        routes[route] = route_details
    profile["routes"] = routes
    profile["timing_semantics"] = {
        "boundary": "explicit output materialization per selected route",
        "exclusive_accounting": (
            "Nested selected-route wall is subtracted from its selected parent."
        ),
        "lazy_attribution_caveat": (
            "Queued upstream MLX work may complete at the next selected output "
            "boundary; the bounded clean A/B gate, not this profile alone, "
            "decides whether an optimization is retained."
        ),
    }

    clean_timing = _mapping(clean["timing"], name="clean profile timing")
    instrumented_timing = _mapping(
        instrumented["timing"],
        name="instrumented profile timing",
    )
    complete = float(instrumented_timing["complete_wall_seconds"])
    named_outside_profile = sum(
        float(instrumented_timing[name])
        for name in (
            "setup_wall_seconds",
            "synchronization_diagnostics_wall_seconds",
            "persistence_wall_seconds",
        )
    )
    root_wall = _nonnegative_float(
        profile.get("root_wall_seconds"),
        name="profile root wall seconds",
    )
    residual = max(0.0, complete - named_outside_profile - root_wall)
    reconciled = named_outside_profile + root_wall + residual
    tolerance = max(1.0e-6, 0.10 * complete)
    if not math.isclose(complete, reconciled, rel_tol=0.0, abs_tol=tolerance):
        raise DHFRNPTRuntimeError("instrumented profile does not reconcile")
    unsigned = {
        "schema": PROFILE_SCHEMA,
        "status": "passed",
        "role": "diagnostic-prefix-only",
        "workload_fingerprint": clean_workload["workload_fingerprint"],
        "profile_steps": PROFILE_STEPS,
        "projection_steps": BASELINE_STEPS,
        "clean": {
            "mode": "clean",
            "sample_fingerprint": clean["report_fingerprint"],
            "complete_wall_seconds": clean_timing["complete_wall_seconds"],
            "memory_peak_bytes": clean["memory"]["peak_physical_bytes"],
            "timing": clean_timing,
            "runtime": clean.get("runtime"),
        },
        "instrumented": {
            "mode": "instrumented",
            "sample_fingerprint": instrumented["report_fingerprint"],
            "complete_wall_seconds": complete,
            "memory_peak_bytes": instrumented["memory"]["peak_physical_bytes"],
            "timing": instrumented_timing,
            "runtime": instrumented.get("runtime"),
        },
        "profiling_overhead_wall_seconds": (
            complete - float(clean_timing["complete_wall_seconds"])
        ),
        "reconciliation": {
            "setup_wall_seconds": instrumented_timing["setup_wall_seconds"],
            "profiled_root_wall_seconds": root_wall,
            "synchronization_diagnostics_wall_seconds": instrumented_timing[
                "synchronization_diagnostics_wall_seconds"
            ],
            "persistence_wall_seconds": instrumented_timing[
                "persistence_wall_seconds"
            ],
            "residual_unaccounted_wall_seconds": residual,
            "reconciled_wall_seconds": reconciled,
            "relative_error": abs(complete - reconciled) / complete,
        },
        "profile": profile,
    }
    return {
        **unsigned,
        "report_fingerprint": payload_fingerprint(unsigned),
    }


def build_hotspot_report(
    profile_report: Mapping[str, Any],
    *,
    route_audit: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank actionable MLX routes from an accepted diagnostic profile."""

    profile = validate_profile_report_payload(profile_report)
    route_timings = _mapping(
        _mapping(profile["profile"], name="route profile").get("routes"),
        name="route timings",
    )
    candidates = []
    for route, details in route_timings.items():
        timing = _mapping(details, name=f"route timing {route}")
        if int(timing["calls"]) == 0:
            continue
        audit = _mapping(route_audit.get(route), name=f"source audit {route}")
        hypothesis = audit.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis:
            raise DHFRNPTRuntimeError(f"route {route} lacks an optimization hypothesis")
        candidates.append(
            {
                "route": route,
                "projected_750_exclusive_wall_seconds": float(
                    timing["projected_750_exclusive_wall_seconds"]
                ),
                "projected_750_uncertainty_wall_seconds": float(
                    timing.get("projected_750_uncertainty_wall_seconds", 0.0)
                ),
                "uncertainty_status": timing.get(
                    "uncertainty_status",
                    "insufficient",
                ),
                "observed_75_exclusive_wall_seconds": float(
                    timing["total_exclusive_wall_seconds"]
                ),
                "observed_calls": int(timing["calls"]),
                "projected_750_calls": int(timing["projected_750_calls"]),
                "first_occurrence_wall_seconds": float(
                    timing["first_exclusive_wall_seconds"]
                ),
                "recurring_mean_wall_seconds": float(
                    timing["recurring_exclusive_mean_wall_seconds"]
                ),
                "source": audit,
                "optimization_hypothesis": hypothesis,
                "bounded_smell_test": (
                    "Run the identical 75-step seed-313 prefix with only this "
                    "route changed; compare complete wall, route work, and outputs."
                ),
                "numerical_gates": (
                    "All existing finite, attempt-schedule, sample-count, "
                    "constraint, volume, cell, temperature, pressure, and "
                    "energy-stability checks must pass."
                ),
                "memory_gate": "Process-tree peak must remain below 40,000,000,000 bytes.",
                "rollback_rule": (
                    "Revert unless clean complete wall improves materially "
                    "without worse numerical checks or unsafe memory growth."
                ),
            }
        )
    candidates.sort(
        key=lambda candidate: candidate[
            "projected_750_exclusive_wall_seconds"
        ],
        reverse=True,
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    if not candidates:
        raise DHFRNPTRuntimeError("profile contains no observed hotspot routes")

    profile_root = float(
        _mapping(profile["reconciliation"], name="profile reconciliation")[
            "profiled_root_wall_seconds"
        ]
    )
    instrumented_complete = float(
        _mapping(profile["instrumented"], name="instrumented profile")[
            "complete_wall_seconds"
        ]
    )
    explained_share = profile_root / instrumented_complete
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    uncertainty_overlap = second is not None and (
        top["projected_750_exclusive_wall_seconds"]
        - top["projected_750_uncertainty_wall_seconds"]
        <= second["projected_750_exclusive_wall_seconds"]
        + second["projected_750_uncertainty_wall_seconds"]
    )
    close = second is not None and (
        top["projected_750_exclusive_wall_seconds"]
        - second["projected_750_exclusive_wall_seconds"]
    ) <= 0.10 * top["projected_750_exclusive_wall_seconds"]
    limited_recurrence = second is not None and min(
        top["observed_calls"],
        second["observed_calls"],
    ) <= 3
    ranking_resolution = (
        "ambiguous-overlapping-uncertainty"
        if uncertainty_overlap
        else "ambiguous-close-low-recurrence"
        if close and limited_recurrence
        else "resolved-by-projected-exclusive-time"
    )
    reconciliation = _mapping(
        profile["reconciliation"],
        name="profile reconciliation",
    )
    projected_routes = sum(
        candidate["projected_750_exclusive_wall_seconds"]
        for candidate in candidates
    )
    residual_75 = float(reconciliation["residual_unaccounted_wall_seconds"])
    projected_residual = residual_75 * BASELINE_STEPS / PROFILE_STEPS
    projected_total = (
        float(reconciliation["setup_wall_seconds"])
        + float(reconciliation["synchronization_diagnostics_wall_seconds"])
        + float(reconciliation["persistence_wall_seconds"])
        + projected_routes
        + projected_residual
    )
    unsigned = {
        "schema": HOTSPOT_SCHEMA,
        "status": "passed",
        "role": "diagnostic-decision-packet",
        "profile_report_fingerprint": profile["report_fingerprint"],
        "explained_complete_wall_share": explained_share,
        "unresolved_remainder": (
            None
            if explained_share >= 0.90
            else {
                "status": "blocker",
                "complete_wall_share": 1.0 - explained_share,
                "reason": (
                    "Selected route timers explain less than 90% of the "
                    "instrumented complete wall."
                ),
            }
        ),
        "ranking_resolution": ranking_resolution,
        "selected_route": top["route"],
        "projection_750": {
            "role": "diagnostic-only-not-a-parity-result",
            "one_time_setup_wall_seconds": float(
                reconciliation["setup_wall_seconds"]
            ),
            "one_time_final_synchronization_wall_seconds": float(
                reconciliation["synchronization_diagnostics_wall_seconds"]
            ),
            "one_time_persistence_wall_seconds": float(
                reconciliation["persistence_wall_seconds"]
            ),
            "profiled_routes_wall_seconds": projected_routes,
            "unresolved_residual_wall_seconds": projected_residual,
            "projected_complete_wall_seconds": projected_total,
            "residual_assumption": (
                "The unresolved 75-step residual is scaled linearly; it cannot "
                "select an optimization candidate."
            ),
        },
        "candidates": candidates,
    }
    return {
        **unsigned,
        "report_fingerprint": payload_fingerprint(unsigned),
    }


def load_profile_report(path: str | Path) -> dict[str, Any]:
    """Load and validate one diagnostic MLX profile report."""

    return validate_profile_report_payload(
        _load_json(path, context="runtime profile report")
    )


def validate_profile_report_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an in-memory diagnostic MLX profile report."""

    report = dict(payload)
    if report.get("schema") != PROFILE_SCHEMA or report.get("status") != "passed":
        raise DHFRNPTRuntimeError("runtime profile is incomplete or unsupported")
    claimed = report.get("report_fingerprint")
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime profile fingerprint mismatch")
    if (
        report.get("role") != "diagnostic-prefix-only"
        or report.get("profile_steps") != PROFILE_STEPS
        or report.get("projection_steps") != BASELINE_STEPS
    ):
        raise DHFRNPTRuntimeError("runtime profile identity drifted")
    reconciliation = _mapping(
        report.get("reconciliation"),
        name="profile reconciliation",
    )
    relative_error = _nonnegative_float(
        reconciliation.get("relative_error"),
        name="profile reconciliation error",
    )
    if relative_error > 0.10:
        raise DHFRNPTRuntimeError("runtime profile exceeds reconciliation tolerance")
    _require_finite_json(report, context="runtime profile report")
    return report


def load_hotspot_report(
    path: str | Path,
    *,
    expected_profile_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load and validate one diagnostic MLX hotspot packet."""

    report = _load_json(path, context="runtime hotspot report")
    if report.get("schema") != HOTSPOT_SCHEMA or report.get("status") != "passed":
        raise DHFRNPTRuntimeError("runtime hotspot packet is incomplete")
    claimed = report.get("report_fingerprint")
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if claimed != payload_fingerprint(unsigned):
        raise DHFRNPTRuntimeError("runtime hotspot fingerprint mismatch")
    if (
        expected_profile_fingerprint is not None
        and report.get("profile_report_fingerprint")
        != expected_profile_fingerprint
    ):
        raise DHFRNPTRuntimeError("runtime hotspot profile identity mismatch")
    candidates = _sequence(report.get("candidates"), name="hotspot candidates")
    if not candidates:
        raise DHFRNPTRuntimeError("runtime hotspot candidates are missing")
    first = _mapping(candidates[0], name="first hotspot candidate")
    if first.get("rank") != 1 or first.get("route") != report.get("selected_route"):
        raise DHFRNPTRuntimeError("runtime hotspot selection is inconsistent")
    share = _nonnegative_float(
        report.get("explained_complete_wall_share"),
        name="hotspot explained share",
    )
    if share > 1.0 + 1.0e-6:
        raise DHFRNPTRuntimeError("runtime hotspot explained share is invalid")
    _require_finite_json(report, context="runtime hotspot report")
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


def _projected_route_call_count(*, calls: int) -> int:
    return max(0, calls) * BASELINE_STEPS // PROFILE_STEPS


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
