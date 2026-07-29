"""Frozen v2 contract and evidence helpers for bounded DHFR NPT validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks.dhfr_npt import (
    DEFAULT_CONTRACT_PATH as V1_CONTRACT_PATH,
)
from mlx_atomistic.benchmarks.dhfr_npt import (
    DHFRNPTValidationError,
    payload_fingerprint,
    validate_prepared_boundary,
)
from mlx_atomistic.benchmarks.dhfr_npt import (
    load_validation_contract as load_v1_contract,
)

CONTRACT_SCHEMA = "mlx-atomistic.dhfr-npt-validation-contract.v2"
ENGINE_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-v2-engine-report.v1"
SEED_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-v2-seed-report.v1"
FINAL_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-v2-final-report.v1"
FREEZE_MANIFEST_SCHEMA = "mlx-atomistic.dhfr-npt-v2-source-freeze.v1"
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("data") / (
    "dhfr_npt_v2_validation_contract.json"
)
FORMAL_SEEDS = (7, 19)
PROCESS_TREE_MAX_BYTES = 40_000_000_000
TIMEOUT_FORMULA_VERSION = "dhfr-npt-v2-timeout-v1"
SEED_TIMEOUT_CAPS = {7: 43_200, 19: 28_800}
TIMEOUT_ROUNDING_SECONDS = 300
OPENMM_INITIAL_VOLUME_FRACTION = 0.01
CONSTRAINT_MAX_ITERATIONS = 40
FORMAL_ROOT_NAME = "formal"
FREEZE_MANIFEST_NAME = "source-freeze.json"
NPT_GATES = {
    "required_attempts_per_seed": 30,
    "maximum_constraint_error_angstrom": 1.0e-4,
    "minimum_volume_ratio": 0.8,
    "maximum_volume_ratio": 1.25,
    "orthorhombic_off_diagonal_tolerance_angstrom": 1.0e-6,
    "maximum_temperature_K": 1200.0,
    "maximum_abs_pressure_bar": 100_000.0,
    "maximum_energy_excursion_per_atom_kj_mol": 50.0,
    "maximum_engine_mean_volume_ratio_delta": 0.1,
    "maximum_engine_mean_pressure_delta_bar": 10_000.0,
    "minimum_pooled_accepted_moves_per_engine": 1,
}
RESTART_GATE = {
    "seed": 7,
    "required": True,
    "maximum_abs_error": 1.0e-6,
}
CLAIM = {
    "status": "frozen-pre-formal",
    "scope": "bounded repeated orthorhombic anisotropic NPT for OpenMM 5dfr DHFR",
    "limitations": [
        "short stochastic validation, not thermodynamic convergence",
        "failed v1 evidence remains historical and unchanged",
    ],
}


def load_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load and validate the frozen DHFR NPT v2 contract."""

    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DHFRNPTValidationError("cannot load DHFR NPT v2 contract") from error
    if not isinstance(raw, Mapping):
        raise DHFRNPTValidationError("DHFR NPT v2 contract must be an object")
    contract = dict(raw)
    _validate_contract(contract)
    return contract


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    """Return the canonical v2 contract fingerprint."""

    return payload_fingerprint(dict(contract))


def derive_formal_timeouts(
    *,
    attempts: int,
    mlx_seconds: float,
    reference_seconds_per_attempt: float,
) -> dict[str, Any]:
    """Derive the two predeclared formal timeout limits."""

    if isinstance(attempts, bool) or attempts <= 0:
        raise DHFRNPTValidationError("timeout attempts must be positive")
    numeric = (float(mlx_seconds), float(reference_seconds_per_attempt))
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise DHFRNPTValidationError("timeout measurements must be finite and positive")
    unrounded = {
        7: 1.35
        * (
            2.0 * numeric[0]
            + int(attempts) * numeric[1]
        )
        + 900.0,
        19: 1.35
        * (
            numeric[0]
            + int(attempts) * numeric[1]
        )
        + 900.0,
    }
    rounded = {
        seed: int(
            math.ceil(value / TIMEOUT_ROUNDING_SECONDS)
            * TIMEOUT_ROUNDING_SECONDS
        )
        for seed, value in unrounded.items()
    }
    overflow = [
        seed
        for seed in FORMAL_SEEDS
        if rounded[seed] > SEED_TIMEOUT_CAPS[seed]
    ]
    if overflow:
        raise DHFRNPTValidationError(
            "derived formal timeout exceeds hard cap for seed "
            + ", ".join(str(seed) for seed in overflow)
        )
    return {
        "formula_version": TIMEOUT_FORMULA_VERSION,
        "selected_attempts": int(attempts),
        "mlx_diagnostic_seconds": numeric[0],
        "reference_max_seconds_per_attempt": numeric[1],
        "rounding_seconds": TIMEOUT_ROUNDING_SECONDS,
        "unrounded_seconds": {
            str(seed): unrounded[seed] for seed in FORMAL_SEEDS
        },
        "rounded_seconds": {
            str(seed): rounded[seed] for seed in FORMAL_SEEDS
        },
        "hard_caps_seconds": {
            str(seed): SEED_TIMEOUT_CAPS[seed] for seed in FORMAL_SEEDS
        },
    }


def implementation_inventory(
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fingerprint every source and runner covered by the v2 freeze."""

    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    paths = sorted(
        path
        for path in (root / "src" / "mlx_atomistic").rglob("*.py")
        if path.is_file()
    )
    paths.extend(
        root / name
        for name in (
            "scripts/calibrate_openmm_dhfr_npt.py",
            "scripts/run_bounded_process.py",
            "scripts/run_openmm_mlx_dhfr_npt.py",
            "scripts/run_openmm_mlx_dhfr_npt_v2.py",
            "scripts/run_bounded_dhfr_npt_v2.py",
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise DHFRNPTValidationError(
            "v2 implementation inventory is incomplete: "
            + ", ".join(path.name for path in missing)
        )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def create_contract(
    *,
    prepared_dir: str | Path,
    calibration_report: Mapping[str, Any],
    diagnostic_report: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the v2 contract from completed pre-target evidence."""

    base = load_v1_contract(V1_CONTRACT_PATH)
    source = validate_prepared_boundary(prepared_dir, base)
    calibration = _validated_calibration(calibration_report)
    diagnostic = _validated_diagnostic(diagnostic_report)
    selected = int(calibration["selected_formal_attempts"])
    if (
        selected != int(diagnostic["selected_formal_attempts"])
        or calibration["report_fingerprint"]
        != diagnostic["calibration_report_fingerprint"]
        or calibration["protocol_fingerprint"]
        != diagnostic["calibration_protocol_fingerprint"]
        or calibration["openmm_version"] != diagnostic["openmm_version"]
        or source["manifest_fingerprint"]
        != calibration["source_manifest_fingerprint"]
        or source["manifest_fingerprint"]
        != diagnostic["source_manifest_fingerprint"]
    ):
        raise DHFRNPTValidationError("v2 calibration and diagnostic identities do not reconcile")
    reference_seconds = max(
        float(run["elapsed_seconds"]) / int(run["scheduled_attempts"])
        for run in calibration["runs"]
    )
    timing = derive_formal_timeouts(
        attempts=selected,
        mlx_seconds=float(diagnostic["mlx_elapsed_seconds"]),
        reference_seconds_per_attempt=reference_seconds,
    )
    workload = json.loads(json.dumps(base["workload"]))
    interval = int(workload["barostat"]["interval"])
    workload["steps"] = selected * interval
    workload["seeds"] = list(FORMAL_SEEDS)
    workload["barostat"]["expected_attempts"] = selected
    workload["barostat"]["max_log_volume_scale"] = math.log1p(
        OPENMM_INITIAL_VOLUME_FRACTION
    )
    workload["constraint_max_iterations"] = CONSTRAINT_MAX_ITERATIONS
    inventory = implementation_inventory(repo_root)
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    lock_path = root / "uv.lock"
    contract = {
        "schema": CONTRACT_SCHEMA,
        "claim": dict(CLAIM),
        "target": json.loads(json.dumps(base["target"])),
        "workload": workload,
        "engines": {
            "runtime": "MLX/Metal",
            "reference": "OpenMM Reference",
            "openmm_version": str(calibration["openmm_version"]),
        },
        "resource_limits": {
            "process_tree_max_bytes": PROCESS_TREE_MAX_BYTES,
            "seed_timeout_seconds": dict(timing["rounded_seconds"]),
            "seed_timeout_hard_caps_seconds": dict(timing["hard_caps_seconds"]),
            "mlx_plateau_required": True,
        },
        "timeout_derivation": timing,
        "npt_gates": {
            **NPT_GATES,
            "required_attempts_per_seed": selected,
        },
        "restart_gate": dict(RESTART_GATE),
        "provenance": {
            "source_manifest_fingerprint": source["manifest_fingerprint"],
            "calibration_report_fingerprint": calibration["report_fingerprint"],
            "calibration_protocol_fingerprint": calibration["protocol_fingerprint"],
            "diagnostic_report_fingerprint": diagnostic["report_fingerprint"],
            "diagnostic_implementation_fingerprint": diagnostic[
                "implementation_fingerprint"
            ],
        },
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": _file_sha256(lock_path),
            "openmm_requirement": "openmm>=8.5.1",
        },
        "implementation_freeze": {
            "inventory": inventory,
            "fingerprint": payload_fingerprint(inventory),
        },
    }
    _validate_contract(contract)
    return contract


def freeze_check(
    *,
    contract_path: str | Path,
    prepared_dir: str | Path,
    formal_root: str | Path,
    calibration_path: str | Path | None = None,
    diagnostic_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate frozen source and emit or verify the formal freeze manifest."""

    contract = load_contract(contract_path)
    source = validate_prepared_boundary(prepared_dir, contract)
    inventory = implementation_inventory(repo_root)
    freeze = _mapping(contract["implementation_freeze"], name="implementation freeze")
    if (
        inventory != freeze["inventory"]
        or payload_fingerprint(inventory) != freeze["fingerprint"]
    ):
        raise DHFRNPTValidationError("v2 source drifted after contract freeze")
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    lock = _mapping(contract["dependency_lock"], name="dependency lock")
    if _file_sha256(root / str(lock["path"])) != lock["sha256"]:
        raise DHFRNPTValidationError("v2 dependency lock drifted after contract freeze")
    provenance = _mapping(contract["provenance"], name="contract provenance")
    if calibration_path is not None:
        calibration = _validated_calibration(_read_json(calibration_path))
        if calibration["report_fingerprint"] != provenance[
            "calibration_report_fingerprint"
        ]:
            raise DHFRNPTValidationError("v2 calibration provenance drifted")
    if diagnostic_dir is not None:
        diagnostic = _validated_diagnostic(
            _read_json(Path(diagnostic_dir) / "report.json")
        )
        if diagnostic["report_fingerprint"] != provenance[
            "diagnostic_report_fingerprint"
        ]:
            raise DHFRNPTValidationError("v2 diagnostic provenance drifted")
    output_root = Path(formal_root)
    manifest_path = output_root / FREEZE_MANIFEST_NAME
    unsigned = {
        "schema": FREEZE_MANIFEST_SCHEMA,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "implementation_fingerprint": freeze["fingerprint"],
        "dependency_lock_sha256": lock["sha256"],
        "formal_seeds": list(FORMAL_SEEDS),
    }
    expected = {**unsigned, "manifest_fingerprint": payload_fingerprint(unsigned)}
    if manifest_path.exists():
        actual = _read_json(manifest_path)
        if actual != expected:
            raise DHFRNPTValidationError("v2 source-freeze manifest drifted")
    else:
        forbidden = [
            output_root / f"seed-{seed}"
            for seed in FORMAL_SEEDS
            if (output_root / f"seed-{seed}").exists()
        ]
        if (output_root / "report.json").exists():
            forbidden.append(output_root / "report.json")
        if forbidden:
            raise DHFRNPTValidationError(
                "formal output exists before source-freeze preflight"
            )
        _write_json_atomic(manifest_path, expected)
    return {
        "status": "passed",
        "contract_fingerprint": expected["contract_fingerprint"],
        "implementation_fingerprint": expected["implementation_fingerprint"],
        "source_manifest_fingerprint": expected["source_manifest_fingerprint"],
        "freeze_manifest": str(manifest_path),
    }


def engine_report_path(root: str | Path, *, seed: int, engine: str) -> Path:
    """Return the canonical formal engine report path."""

    _validate_seed(seed)
    normalized = _validate_engine(engine)
    return Path(root) / f"seed-{seed}" / normalized / "report.json"


def build_engine_report(
    *,
    contract: Mapping[str, Any],
    source_manifest_fingerprint: str,
    seed: int,
    engine: str,
    summary: Mapping[str, Any],
    checks: Mapping[str, bool],
    artifacts: Mapping[str, Any],
    checkpoint_resume: Mapping[str, Any] | None = None,
    openmm_version: str | None = None,
) -> dict[str, Any]:
    """Build one fail-closed formal engine report."""

    _validate_contract(contract)
    _validate_seed(seed)
    normalized = _validate_engine(engine)
    check_payload = _boolean_checks(checks)
    blockers = [name for name, passed in check_payload.items() if not passed]
    unsigned = {
        "schema": ENGINE_REPORT_SCHEMA,
        "seed": seed,
        "engine": normalized,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "checks": check_payload,
        "summary": dict(summary),
        "checkpoint_resume": (
            None if checkpoint_resume is None else dict(checkpoint_resume)
        ),
        "openmm_version": openmm_version,
        "artifacts": dict(artifacts),
    }
    _require_finite(unsigned, context="formal engine report")
    report = {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}
    validate_engine_report(report, contract=contract)
    return report


def build_engine_checks(
    *,
    contract: Mapping[str, Any],
    seed: int,
    engine: str,
    summary: Mapping[str, Any],
    checkpoint_resume: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Build the frozen numerical checks for one formal engine phase."""

    _validate_contract(contract)
    _validate_seed(seed)
    normalized = _validate_engine(engine)
    gates = _mapping(contract["npt_gates"], name="NPT gates")
    attempts_field = (
        "barostat_attempts"
        if normalized == "mlx"
        else "configured_barostat_attempts"
    )
    checks = {
        "finite": bool(summary["finite"]),
        "attempt_schedule": (
            int(summary[attempts_field])
            == int(gates["required_attempts_per_seed"])
        ),
        "constraints": (
            float(summary["maximum_constraint_error_angstrom"])
            <= float(gates["maximum_constraint_error_angstrom"])
        ),
        "volume_bounds": (
            float(gates["minimum_volume_ratio"])
            <= float(summary["minimum_volume_ratio"])
            <= float(gates["maximum_volume_ratio"])
            and float(gates["minimum_volume_ratio"])
            <= float(summary["maximum_volume_ratio"])
            <= float(gates["maximum_volume_ratio"])
        ),
        "orthorhombic_cell": (
            float(summary["maximum_cell_off_diagonal_angstrom"])
            <= float(gates["orthorhombic_off_diagonal_tolerance_angstrom"])
        ),
        "temperature_bound": (
            float(summary["maximum_temperature_K"])
            <= float(gates["maximum_temperature_K"])
        ),
        "pressure_bound": (
            float(summary["maximum_abs_pressure_bar"])
            <= float(gates["maximum_abs_pressure_bar"])
        ),
        "energy_stability": (
            float(summary["maximum_energy_excursion_per_atom_kj_mol"])
            <= float(gates["maximum_energy_excursion_per_atom_kj_mol"])
        ),
    }
    if normalized == "mlx":
        checks["dynamic_pme"] = bool(summary["final_pme_plan_fingerprints"])
        restart_seed = int(
            _mapping(contract["restart_gate"], name="restart gate")["seed"]
        )
        checks["restart_policy"] = (
            checkpoint_resume is not None
            and checkpoint_resume.get("passed") is True
            if seed == restart_seed
            else checkpoint_resume is None
        )
    else:
        checks["reference_platform"] = summary["platform"] == "Reference"
    return checks


def validate_engine_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_seed: int | None = None,
    expected_engine: str | None = None,
) -> dict[str, Any]:
    """Validate one formal engine report without accepting diagnostics."""

    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if payload.get("schema") != ENGINE_REPORT_SCHEMA:
        raise DHFRNPTValidationError("formal engine report schema is unsupported")
    if fingerprint != payload_fingerprint(payload):
        raise DHFRNPTValidationError("formal engine report fingerprint mismatch")
    seed = int(payload.get("seed", -1))
    engine = _validate_engine(str(payload.get("engine", "")))
    _validate_seed(seed)
    if expected_seed is not None and seed != expected_seed:
        raise DHFRNPTValidationError("formal engine report seed mismatch")
    if expected_engine is not None and engine != _validate_engine(expected_engine):
        raise DHFRNPTValidationError("formal engine report engine mismatch")
    if payload.get("contract_fingerprint") != contract_fingerprint(contract):
        raise DHFRNPTValidationError("formal engine report contract mismatch")
    provenance = _mapping(contract["provenance"], name="contract provenance")
    if payload.get("source_manifest_fingerprint") != provenance[
        "source_manifest_fingerprint"
    ]:
        raise DHFRNPTValidationError("formal engine report source mismatch")
    summary = _mapping(payload.get("summary"), name="formal engine summary")
    checkpoint = payload.get("checkpoint_resume")
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise DHFRNPTValidationError("formal restart evidence must be an object")
    _validate_checkpoint_resume(
        checkpoint,
        contract=contract,
        seed=seed,
        engine=engine,
    )
    checks = _boolean_checks(payload.get("checks"))
    expected_checks = build_engine_checks(
        contract=contract,
        seed=seed,
        engine=engine,
        summary=summary,
        checkpoint_resume=checkpoint,
    )
    if checks != expected_checks:
        raise DHFRNPTValidationError("formal engine checks drifted")
    blockers = [name for name, passed in checks.items() if not passed]
    if payload.get("blockers") != blockers:
        raise DHFRNPTValidationError("formal engine blockers do not reconcile")
    if payload.get("status") != ("passed" if not blockers else "failed"):
        raise DHFRNPTValidationError("formal engine status does not reconcile")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _expected_artifacts(
        seed=seed,
        engine=engine,
    ):
        raise DHFRNPTValidationError("formal engine artifacts are missing")
    if engine == "openmm":
        expected_version = _mapping(contract["engines"], name="engines")[
            "openmm_version"
        ]
        if payload.get("openmm_version") != expected_version:
            raise DHFRNPTValidationError("formal OpenMM version drifted")
        if checkpoint is not None:
            raise DHFRNPTValidationError("OpenMM report cannot contain MLX restart evidence")
    elif payload.get("openmm_version") is not None:
        raise DHFRNPTValidationError("MLX report cannot declare an OpenMM version")
    _require_finite(payload, context="formal engine report")
    return {**payload, "report_fingerprint": str(fingerprint)}


def write_engine_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Validate and atomically write one engine report."""

    _write_json_atomic(path, report)


def load_engine_report(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    seed: int,
    engine: str,
) -> dict[str, Any]:
    """Load and verify an engine report and all declared artifacts."""

    report_path = Path(path)
    report = validate_engine_report(
        _read_json(report_path),
        contract=contract,
        expected_seed=seed,
        expected_engine=engine,
    )
    _validate_artifacts(report_path.parent, report["artifacts"])
    return report


def reconcile_seed_directory(
    *,
    contract_path: str | Path,
    prepared_dir: str | Path,
    formal_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Reconcile separate MLX and OpenMM phases for one formal seed."""

    contract = load_contract(contract_path)
    _validate_seed(seed)
    source = validate_prepared_boundary(prepared_dir, contract)
    seed_dir = Path(formal_root) / f"seed-{seed}"
    reports = {
        engine: load_engine_report(
            engine_report_path(formal_root, seed=seed, engine=engine),
            contract=contract,
            seed=seed,
            engine=engine,
        )
        for engine in ("mlx", "openmm")
    }
    memory = {
        engine: _validate_memory_trace(
            _read_json(seed_dir / f"{engine}-memory.json"),
            contract=contract,
            seed=seed,
            require_plateau=engine == "mlx",
        )
        for engine in ("mlx", "openmm")
    }
    delta = _engine_delta(reports)
    checks = _seed_checks(
        contract=contract,
        seed=seed,
        engines=reports,
        memory=memory,
        engine_delta=delta,
    )
    blockers = [name for name, passed in checks.items() if not passed]
    unsigned = {
        "schema": SEED_REPORT_SCHEMA,
        "seed": seed,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "checks": checks,
        "engines": reports,
        "memory": memory,
        "engine_delta": delta,
    }
    report = {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}
    _write_json_atomic(seed_dir / "report.json", report)
    return report


def finalize(
    *,
    contract_path: str | Path,
    prepared_dir: str | Path,
    input_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the pooled two-seed v2 report."""

    contract = load_contract(contract_path)
    source = validate_prepared_boundary(prepared_dir, contract)
    seed_reports = [
        _validate_seed_report(
            _read_json(Path(input_root) / f"seed-{seed}" / "report.json"),
            contract=contract,
            seed=seed,
        )
        for seed in FORMAL_SEEDS
    ]
    accepted = {
        engine: sum(
            int(report["engines"][engine]["summary"][_accepted_field(engine)])
            for report in seed_reports
        )
        for engine in ("mlx", "openmm")
    }
    minimum = int(
        _mapping(contract["npt_gates"], name="NPT gates")[
            "minimum_pooled_accepted_moves_per_engine"
        ]
    )
    checks = {
        "all_seed_reports": all(report["status"] == "passed" for report in seed_reports),
        "mlx_pooled_cell_evolution": accepted["mlx"] >= minimum,
        "openmm_pooled_cell_evolution": accepted["openmm"] >= minimum,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    unsigned = {
        "schema": FINAL_REPORT_SCHEMA,
        "contract_fingerprint": contract_fingerprint(contract),
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "checks": checks,
        "pooled_accepted_moves": accepted,
        "seeds": seed_reports,
        "claim": dict(contract["claim"]),
    }
    report = {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}
    _write_json_atomic(output_path, report)
    return report


def verify_final(
    *,
    contract_path: str | Path,
    prepared_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Validate the pooled v2 final report."""

    contract = load_contract(contract_path)
    source = validate_prepared_boundary(prepared_dir, contract)
    payload = dict(_read_json(report_path))
    fingerprint = payload.pop("report_fingerprint", None)
    if payload.get("schema") != FINAL_REPORT_SCHEMA:
        raise DHFRNPTValidationError("v2 final report schema is unsupported")
    if fingerprint != payload_fingerprint(payload):
        raise DHFRNPTValidationError("v2 final report fingerprint mismatch")
    if (
        payload.get("contract_fingerprint") != contract_fingerprint(contract)
        or payload.get("source_manifest_fingerprint")
        != source["manifest_fingerprint"]
    ):
        raise DHFRNPTValidationError("v2 final report identity mismatch")
    if payload.get("status") != "passed" or payload.get("blockers") != []:
        raise DHFRNPTValidationError("v2 final report is not passing")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(FORMAL_SEEDS):
        raise DHFRNPTValidationError("v2 final report seed inventory is incomplete")
    for expected_seed, report in zip(FORMAL_SEEDS, seeds, strict=True):
        _validate_seed_report(report, contract=contract, seed=expected_seed)
    accepted = {
        engine: sum(
            int(report["engines"][engine]["summary"][_accepted_field(engine)])
            for report in seeds
        )
        for engine in ("mlx", "openmm")
    }
    minimum = int(contract["npt_gates"]["minimum_pooled_accepted_moves_per_engine"])
    expected_checks = {
        "all_seed_reports": all(report["status"] == "passed" for report in seeds),
        "mlx_pooled_cell_evolution": accepted["mlx"] >= minimum,
        "openmm_pooled_cell_evolution": accepted["openmm"] >= minimum,
    }
    if (
        payload.get("pooled_accepted_moves") != accepted
        or payload.get("checks") != expected_checks
        or payload.get("claim") != contract["claim"]
    ):
        raise DHFRNPTValidationError("v2 final report does not reconcile")
    expected_blockers = [
        name for name, passed in expected_checks.items() if not passed
    ]
    if (
        payload.get("blockers") != expected_blockers
        or payload.get("status")
        != ("passed" if not expected_blockers else "failed")
    ):
        raise DHFRNPTValidationError("v2 final report status does not reconcile")
    return {**payload, "report_fingerprint": str(fingerprint)}


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise DHFRNPTValidationError("DHFR NPT v2 contract schema is unsupported")
    required = (
        "claim",
        "target",
        "workload",
        "engines",
        "resource_limits",
        "timeout_derivation",
        "npt_gates",
        "restart_gate",
        "provenance",
        "dependency_lock",
        "implementation_freeze",
    )
    for name in required:
        _mapping(contract.get(name), name=f"contract {name}")
    base = load_v1_contract(V1_CONTRACT_PATH)
    if contract["claim"] != CLAIM:
        raise DHFRNPTValidationError("DHFR NPT v2 claim drifted")
    if contract["target"] != base["target"]:
        raise DHFRNPTValidationError("DHFR NPT v2 target drifted")
    workload = _mapping(contract["workload"], name="workload")
    barostat = _mapping(workload.get("barostat"), name="barostat")
    attempts = int(barostat.get("expected_attempts", -1))
    expected_workload = json.loads(json.dumps(base["workload"]))
    expected_workload["steps"] = attempts * int(
        expected_workload["barostat"]["interval"]
    )
    expected_workload["seeds"] = list(FORMAL_SEEDS)
    expected_workload["barostat"]["expected_attempts"] = attempts
    expected_workload["barostat"]["max_log_volume_scale"] = math.log1p(
        OPENMM_INITIAL_VOLUME_FRACTION
    )
    expected_workload["constraint_max_iterations"] = CONSTRAINT_MAX_ITERATIONS
    if (
        dict(workload) != expected_workload
        or workload.get("steps") != attempts * 25
        or workload.get("dt_ps") != 0.001
        or workload.get("temperature_K") != 300.0
        or workload.get("pressure_bar") != 1.0
        or barostat.get("mode") != "anisotropic"
        or barostat.get("interval") != 25
        or barostat.get("axes") != [True, True, True]
        or not math.isclose(
            math.expm1(float(barostat.get("max_log_volume_scale", math.nan))),
            OPENMM_INITIAL_VOLUME_FRACTION,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or workload.get("constraint_max_iterations")
        != CONSTRAINT_MAX_ITERATIONS
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 workload drifted")
    resources = _mapping(contract["resource_limits"], name="resource limits")
    if (
        resources.get("process_tree_max_bytes") != PROCESS_TREE_MAX_BYTES
        or resources.get("mlx_plateau_required") is not True
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 resource limits drifted")
    timing = _mapping(contract["timeout_derivation"], name="timeout derivation")
    derived = derive_formal_timeouts(
        attempts=int(timing.get("selected_attempts", -1)),
        mlx_seconds=float(timing.get("mlx_diagnostic_seconds", math.nan)),
        reference_seconds_per_attempt=float(
            timing.get("reference_max_seconds_per_attempt", math.nan)
        ),
    )
    if dict(timing) != derived:
        raise DHFRNPTValidationError("DHFR NPT v2 timeout derivation drifted")
    if resources.get("seed_timeout_seconds") != derived["rounded_seconds"]:
        raise DHFRNPTValidationError("DHFR NPT v2 seed timeout mapping drifted")
    if resources.get("seed_timeout_hard_caps_seconds") != derived["hard_caps_seconds"]:
        raise DHFRNPTValidationError("DHFR NPT v2 timeout hard caps drifted")
    if (
        dict(_mapping(contract["npt_gates"], name="NPT gates")) != NPT_GATES
        or attempts != NPT_GATES["required_attempts_per_seed"]
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 numerical gates drifted")
    if dict(_mapping(contract["restart_gate"], name="restart gate")) != RESTART_GATE:
        raise DHFRNPTValidationError("DHFR NPT v2 restart gate drifted")
    engines = _mapping(contract["engines"], name="engines")
    if (
        engines.get("runtime") != "MLX/Metal"
        or engines.get("reference") != "OpenMM Reference"
        or not isinstance(engines.get("openmm_version"), str)
        or not engines["openmm_version"]
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 engine identity drifted")
    provenance = _mapping(contract["provenance"], name="provenance")
    if set(provenance) != {
        "source_manifest_fingerprint",
        "calibration_report_fingerprint",
        "calibration_protocol_fingerprint",
        "diagnostic_report_fingerprint",
        "diagnostic_implementation_fingerprint",
    } or any(not _is_sha256(value) for value in provenance.values()):
        raise DHFRNPTValidationError("DHFR NPT v2 provenance is invalid")
    lock = _mapping(contract["dependency_lock"], name="dependency lock")
    if (
        lock.get("path") != "uv.lock"
        or not _is_sha256(lock.get("sha256"))
        or lock.get("openmm_requirement") != "openmm>=8.5.1"
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 dependency lock is invalid")
    freeze = _mapping(contract["implementation_freeze"], name="implementation freeze")
    inventory = freeze.get("inventory")
    if (
        not isinstance(inventory, list)
        or not inventory
        or freeze.get("fingerprint") != payload_fingerprint(inventory)
    ):
        raise DHFRNPTValidationError("DHFR NPT v2 implementation freeze is invalid")
    paths: list[str] = []
    for raw in inventory:
        record = _mapping(raw, name="implementation record")
        path = Path(str(record.get("path", "")))
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or not _is_sha256(record.get("sha256"))
        ):
            raise DHFRNPTValidationError(
                "DHFR NPT v2 implementation record is invalid"
            )
        paths.append(path.as_posix())
    if len(paths) != len(set(paths)):
        raise DHFRNPTValidationError(
            "DHFR NPT v2 implementation inventory has duplicates"
        )
    _require_finite(contract, context="v2 contract")


def _validated_calibration(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    runs = payload.get("runs")
    if (
        payload.get("schema")
        != "mlx-atomistic.dhfr-npt-v2-calibration-report.v1"
        or fingerprint != payload_fingerprint(payload)
        or payload.get("status") != "selected"
        or int(payload.get("selected_formal_attempts", -1)) != 30
        or not isinstance(runs, list)
        or len(runs) != 6
    ):
        raise DHFRNPTValidationError("v2 calibration report is invalid")
    if (
        not _is_sha256(payload.get("source_manifest_fingerprint"))
        or not _is_sha256(payload.get("protocol_fingerprint"))
        or not isinstance(payload.get("openmm_version"), str)
        or not payload["openmm_version"]
    ):
        raise DHFRNPTValidationError("v2 calibration identity is invalid")
    identities: set[tuple[int, str]] = set()
    run_fingerprints: list[str] = []
    for raw in runs:
        run = dict(_mapping(raw, name="calibration run"))
        run_fingerprint = run.pop("report_fingerprint", None)
        identity = (int(run.get("seed", -1)), str(run.get("axis", "")))
        elapsed = float(run.get("elapsed_seconds", math.nan))
        if (
            run.get("schema")
            != "mlx-atomistic.dhfr-npt-v2-calibration-run.v1"
            or run_fingerprint != payload_fingerprint(run)
            or identity[0] not in {101, 211}
            or identity[1] not in {"x", "y", "z"}
            or int(run.get("scheduled_attempts", -1)) != 40
            or run.get("platform") != "Reference"
            or run.get("openmm_version") != payload.get("openmm_version")
            or run.get("source_manifest_fingerprint")
            != payload.get("source_manifest_fingerprint")
            or run.get("protocol_fingerprint")
            != payload.get("protocol_fingerprint")
            or not math.isfinite(elapsed)
            or elapsed <= 0.0
        ):
            raise DHFRNPTValidationError("v2 calibration run is invalid")
        identities.add(identity)
        run_fingerprints.append(str(run_fingerprint))
    if identities != {
        (seed, axis)
        for seed in (101, 211)
        for axis in ("x", "y", "z")
    } or payload.get("run_fingerprints") != sorted(run_fingerprints):
        raise DHFRNPTValidationError("v2 calibration run inventory is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise DHFRNPTValidationError("v2 calibration candidates are missing")
    qualifying = [
        int(candidate["formal_total_attempts"])
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("qualifies") is True
    ]
    if not qualifying or min(qualifying) != 30:
        raise DHFRNPTValidationError("v2 calibration selection is invalid")
    return {**payload, "report_fingerprint": str(fingerprint)}


def _validated_diagnostic(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    checks = payload.get("checks")
    evidence = payload.get("evidence")
    summary = evidence.get("mlx") if isinstance(evidence, Mapping) else None
    if (
        payload.get("schema")
        != "mlx-atomistic.dhfr-npt-v2-diagnostic-report.v1"
        or fingerprint != payload_fingerprint(payload)
        or payload.get("status") != "passed"
        or payload.get("blockers") != []
        or payload.get("seed") != 313
        or payload.get("selected_formal_attempts") != 30
        or not math.isfinite(
            float(payload.get("mlx_elapsed_seconds", math.nan))
        )
        or float(payload.get("mlx_elapsed_seconds", math.nan)) <= 0.0
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(summary, Mapping)
        or int(summary.get("barostat_attempts", -1)) != 30
    ):
        raise DHFRNPTValidationError("v2 diagnostic report is invalid")
    for name in (
        "source_manifest_fingerprint",
        "calibration_report_fingerprint",
        "calibration_protocol_fingerprint",
        "implementation_fingerprint",
    ):
        if not _is_sha256(payload.get(name)):
            raise DHFRNPTValidationError("v2 diagnostic identity is invalid")
    if not isinstance(payload.get("openmm_version"), str) or not payload[
        "openmm_version"
    ]:
        raise DHFRNPTValidationError("v2 diagnostic OpenMM version is invalid")
    return {**payload, "report_fingerprint": str(fingerprint)}


def _validate_seed_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if (
        payload.get("schema") != SEED_REPORT_SCHEMA
        or fingerprint != payload_fingerprint(payload)
        or payload.get("seed") != seed
        or payload.get("contract_fingerprint") != contract_fingerprint(contract)
        or payload.get("source_manifest_fingerprint")
        != contract["provenance"]["source_manifest_fingerprint"]
    ):
        raise DHFRNPTValidationError("v2 seed report is invalid or incomplete")
    raw_engines = _mapping(payload.get("engines"), name="seed engines")
    if set(raw_engines) != {"mlx", "openmm"}:
        raise DHFRNPTValidationError("v2 seed engine inventory is incomplete")
    engines = {
        engine: validate_engine_report(
            _mapping(raw_engines[engine], name=f"{engine} engine report"),
            contract=contract,
            expected_seed=seed,
            expected_engine=engine,
        )
        for engine in ("mlx", "openmm")
    }
    raw_memory = _mapping(payload.get("memory"), name="seed memory")
    if set(raw_memory) != {"mlx", "openmm"}:
        raise DHFRNPTValidationError("v2 seed memory inventory is incomplete")
    memory = {
        engine: _validate_memory_record(
            _mapping(raw_memory[engine], name=f"{engine} memory record"),
            contract=contract,
            seed=seed,
            require_plateau=engine == "mlx",
        )
        for engine in ("mlx", "openmm")
    }
    expected_delta = _engine_delta(engines)
    if payload.get("engine_delta") != expected_delta:
        raise DHFRNPTValidationError("v2 seed engine deltas do not reconcile")
    expected_checks = _seed_checks(
        contract=contract,
        seed=seed,
        engines=engines,
        memory=memory,
        engine_delta=expected_delta,
    )
    checks = _boolean_checks(payload.get("checks"))
    if checks != expected_checks:
        raise DHFRNPTValidationError("v2 seed checks do not reconcile")
    blockers = [name for name, passed in checks.items() if not passed]
    if (
        payload.get("blockers") != blockers
        or payload.get("status") != ("passed" if not blockers else "failed")
    ):
        raise DHFRNPTValidationError("v2 seed status does not reconcile")
    return {**payload, "report_fingerprint": str(fingerprint)}


def _validate_memory_trace(
    trace: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    seed: int,
    require_plateau: bool,
) -> dict[str, Any]:
    resources = _mapping(contract["resource_limits"], name="resource limits")
    summary = _mapping(trace.get("memory_trace_summary"), name="memory summary")
    checks = {
        "limit": trace.get("bounded_process_limit_bytes")
        == resources["process_tree_max_bytes"],
        "below_limit": int(trace.get("bounded_process_peak_physical_bytes", -1))
        <= resources["process_tree_max_bytes"],
        "not_exceeded": trace.get("bounded_process_exceeded") is False,
        "not_timed_out": trace.get("bounded_process_timed_out") is False,
        "worker_passed": trace.get("bounded_process_returncode") == 0,
        "plateau": (
            summary.get("plateau_evaluated") is True
            and summary.get("plateau_passed") is True
            if require_plateau
            else True
        ),
    }
    record = {
        "passed": all(checks.values()),
        "checks": checks,
        "peak_physical_bytes": int(
            trace.get("bounded_process_peak_physical_bytes", -1)
        ),
        "timeout_seconds": float(
            trace.get("bounded_process_timeout_seconds", math.nan)
        ),
        "plateau_summary": dict(summary),
    }
    return _validate_memory_record(
        record,
        contract=contract,
        seed=seed,
        require_plateau=require_plateau,
    )


def _validate_artifacts(root: Path, records: Mapping[str, Any]) -> None:
    resolved_root = root.resolve()
    for name, raw in records.items():
        record = _mapping(raw, name=f"artifact record {name}")
        relative = Path(str(record.get("path", "")))
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.as_posix() != str(name)
        ):
            raise DHFRNPTValidationError("formal artifact path is unsafe")
        path = (resolved_root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise DHFRNPTValidationError("formal artifact escapes report directory") from error
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != int(record.get("byte_size", -1))
            or _file_sha256(path) != record.get("sha256")
        ):
            raise DHFRNPTValidationError(f"formal artifact is missing or tampered: {name}")


def _validate_memory_record(
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    seed: int,
    require_plateau: bool,
) -> dict[str, Any]:
    checks = _boolean_checks(record.get("checks"))
    if set(checks) != {
        "limit",
        "below_limit",
        "not_exceeded",
        "not_timed_out",
        "worker_passed",
        "plateau",
    }:
        raise DHFRNPTValidationError("formal memory checks are incomplete")
    plateau = _mapping(record.get("plateau_summary"), name="memory plateau")
    expected_plateau = (
        plateau.get("plateau_evaluated") is True
        and plateau.get("plateau_passed") is True
        if require_plateau
        else True
    )
    timeout = float(record.get("timeout_seconds", math.nan))
    timeout_limit = float(
        contract["resource_limits"]["seed_timeout_seconds"][str(seed)]
    )
    peak = int(record.get("peak_physical_bytes", -1))
    if (
        checks["plateau"] is not expected_plateau
        or record.get("passed") is not all(checks.values())
        or not math.isfinite(timeout)
        or timeout <= 0.0
        or timeout > timeout_limit
        or peak < 0
        or peak > PROCESS_TREE_MAX_BYTES
    ):
        raise DHFRNPTValidationError("formal memory record is invalid")
    return dict(record)


def _expected_artifacts(*, seed: int, engine: str) -> set[str]:
    if engine == "openmm":
        return {"openmm_samples.npz"}
    artifacts = {"mlx_trajectory.npz", "mlx_checkpoint.npz"}
    if seed == RESTART_GATE["seed"]:
        artifacts.update(
            {
                "mlx_resume_first_trajectory.npz",
                "mlx_resume_split_checkpoint.npz",
                "mlx_resume_second_trajectory.npz",
                "mlx_resume_final_checkpoint.npz",
            }
        )
    return artifacts


def _validate_checkpoint_resume(
    value: Any,
    *,
    contract: Mapping[str, Any],
    seed: int,
    engine: str,
) -> None:
    restart = _mapping(contract["restart_gate"], name="restart gate")
    required = engine == "mlx" and seed == int(restart["seed"])
    if not required:
        if value is not None:
            raise DHFRNPTValidationError("unexpected formal restart evidence")
        return
    evidence = _mapping(value, name="formal restart evidence")
    required_fields = {
        "passed",
        "split_step",
        "tolerance",
        "maximum_abs_error",
        "final_state_max_abs_errors",
        "sampled_max_abs_errors",
        "energy_term_max_abs_errors",
        "index_fields_match",
        "barostat_and_rng_state_match",
    }
    if set(evidence) != required_fields:
        raise DHFRNPTValidationError("formal restart evidence is incomplete")
    maximum_error = float(evidence["maximum_abs_error"])
    expected_pass = (
        maximum_error <= float(restart["maximum_abs_error"])
        and evidence["index_fields_match"] is True
        and evidence["barostat_and_rng_state_match"] is True
    )
    if (
        evidence["split_step"] != int(contract["workload"]["steps"]) // 2
        or evidence["tolerance"] != restart["maximum_abs_error"]
        or not math.isfinite(maximum_error)
        or evidence["passed"] is not expected_pass
    ):
        raise DHFRNPTValidationError("formal restart evidence is invalid")
    _require_finite(evidence, context="formal restart evidence")


def _engine_delta(engines: Mapping[str, Any]) -> dict[str, float]:
    mlx = engines["mlx"]["summary"]
    openmm = engines["openmm"]["summary"]
    return {
        "mean_volume_ratio": abs(
            float(mlx["mean_volume_ratio"])
            - float(openmm["mean_volume_ratio"])
        ),
        "mean_pressure_bar": abs(
            float(mlx["mean_pressure_bar"])
            - float(openmm["mean_pressure_bar"])
        ),
    }


def _seed_checks(
    *,
    contract: Mapping[str, Any],
    seed: int,
    engines: Mapping[str, Any],
    memory: Mapping[str, Any],
    engine_delta: Mapping[str, float],
) -> dict[str, bool]:
    gates = _mapping(contract["npt_gates"], name="NPT gates")
    restart = engines["mlx"]["checkpoint_resume"]
    return {
        "mlx_phase": engines["mlx"]["status"] == "passed",
        "openmm_phase": engines["openmm"]["status"] == "passed",
        "mlx_memory": memory["mlx"]["passed"],
        "openmm_memory": memory["openmm"]["passed"],
        "aggregate_volume_compatibility": (
            float(engine_delta["mean_volume_ratio"])
            <= float(gates["maximum_engine_mean_volume_ratio_delta"])
        ),
        "aggregate_pressure_compatibility": (
            float(engine_delta["mean_pressure_bar"])
            <= float(gates["maximum_engine_mean_pressure_delta_bar"])
        ),
        "restart_policy": (
            restart is not None and restart.get("passed") is True
            if seed == int(contract["restart_gate"]["seed"])
            else restart is None
        ),
    }


def _accepted_field(engine: str) -> str:
    return "barostat_accepted" if engine == "mlx" else "observed_cell_changes"


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or int(seed) not in FORMAL_SEEDS:
        raise DHFRNPTValidationError("formal seed must be 7 or 19")


def _validate_engine(engine: str) -> str:
    normalized = str(engine).strip().lower()
    if normalized not in {"mlx", "openmm"}:
        raise DHFRNPTValidationError("formal engine must be mlx or openmm")
    return normalized


def _boolean_checks(value: Any) -> dict[str, bool]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(type(item) is not bool for item in value.values())
    ):
        raise DHFRNPTValidationError("formal checks must be non-empty booleans")
    return {str(name): bool(item) for name, item in value.items()}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DHFRNPTValidationError(f"{name} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DHFRNPTValidationError(f"cannot load JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise DHFRNPTValidationError("JSON evidence must be an object")
    return value


def _write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, context=context)
    elif isinstance(value, list | tuple):
        for item in value:
            _require_finite(item, context=context)
    elif isinstance(value, float) and not math.isfinite(value):
        raise DHFRNPTValidationError(f"{context} contains a non-finite number")


def _create_command(args: argparse.Namespace) -> int:
    calibration = _read_json(args.calibration)
    diagnostic = _read_json(Path(args.diagnostic) / "report.json")
    contract = create_contract(
        prepared_dir=args.prepared,
        calibration_report=calibration,
        diagnostic_report=diagnostic,
    )
    _write_json_atomic(args.output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run v2 contract creation, freeze, reconciliation, and verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-contract")
    create.add_argument("--prepared", type=Path, required=True)
    create.add_argument("--calibration", type=Path, required=True)
    create.add_argument("--diagnostic", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-check")
    freeze.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    freeze.add_argument("--prepared", type=Path, required=True)
    freeze.add_argument("--calibration", type=Path)
    freeze.add_argument("--diagnostic", type=Path)
    freeze.add_argument("--formal-root", type=Path, required=True)
    reconcile = commands.add_parser("reconcile-seed")
    reconcile.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    reconcile.add_argument("--prepared", type=Path, required=True)
    reconcile.add_argument("--formal-root", type=Path, required=True)
    reconcile.add_argument("--seed", type=int, required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    final.add_argument("--prepared", type=Path, required=True)
    final.add_argument("--input", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    verify.add_argument("--prepared", type=Path, required=True)
    verify.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create-contract":
        return _create_command(args)
    if args.command == "freeze-check":
        result = freeze_check(
            contract_path=args.contract,
            prepared_dir=args.prepared,
            calibration_path=args.calibration,
            diagnostic_dir=args.diagnostic,
            formal_root=args.formal_root,
        )
    elif args.command == "reconcile-seed":
        result = reconcile_seed_directory(
            contract_path=args.contract,
            prepared_dir=args.prepared,
            formal_root=args.formal_root,
            seed=args.seed,
        )
    elif args.command == "finalize":
        result = finalize(
            contract_path=args.contract,
            prepared_dir=args.prepared,
            input_root=args.input,
            output_path=args.output,
        )
    elif args.command == "verify":
        result = verify_final(
            contract_path=args.contract,
            prepared_dir=args.prepared,
            report_path=args.input,
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTRACT_SCHEMA",
    "DEFAULT_CONTRACT_PATH",
    "ENGINE_REPORT_SCHEMA",
    "FINAL_REPORT_SCHEMA",
    "FORMAL_SEEDS",
    "PROCESS_TREE_MAX_BYTES",
    "SEED_REPORT_SCHEMA",
    "build_engine_report",
    "contract_fingerprint",
    "create_contract",
    "derive_formal_timeouts",
    "engine_report_path",
    "finalize",
    "freeze_check",
    "implementation_inventory",
    "load_contract",
    "load_engine_report",
    "main",
    "reconcile_seed_directory",
    "validate_engine_report",
    "verify_final",
    "write_engine_report",
]
