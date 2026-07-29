"""Run and verify staged DHFR NPT v2 diagnostic and formal evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx

from mlx_atomistic.benchmarks import dhfr_npt_v2
from mlx_atomistic.benchmarks.dhfr_npt import (
    DEFAULT_CONTRACT_PATH as V1_CONTRACT_PATH,
)
from mlx_atomistic.benchmarks.dhfr_npt import (
    DHFRNPTValidationError,
    load_validation_contract,
    payload_fingerprint,
    validate_prepared_boundary,
)
from mlx_atomistic.prep.io import load_prepared_system

try:
    from scripts import calibrate_openmm_dhfr_npt as calibration
    from scripts import run_openmm_mlx_dhfr_npt as v1_runner
except ImportError:  # pragma: no cover - direct script execution.
    import calibrate_openmm_dhfr_npt as calibration
    import run_openmm_mlx_dhfr_npt as v1_runner

DIAGNOSTIC_REPORT_SCHEMA = "mlx-atomistic.dhfr-npt-v2-diagnostic-report.v1"
DIAGNOSTIC_SCOPE = "openmm-5dfr-v2-diagnostic-only"
DIAGNOSTIC_SEED = 313
PROCESS_TREE_MAX_BYTES = 40_000_000_000
VOLUME_RATIO_BOUNDS = (0.8, 1.25)
MAXIMUM_CONSTRAINT_ERROR_ANGSTROM = 1.0e-4
MAXIMUM_CELL_OFF_DIAGONAL_ANGSTROM = 1.0e-6
MAXIMUM_TEMPERATURE_K = 1200.0
MAXIMUM_ABS_PRESSURE_BAR = 100_000.0
MAXIMUM_ENERGY_EXCURSION_PER_ATOM_KJ_MOL = 50.0
OPENMM_INITIAL_VOLUME_FRACTION = 0.01
CONSTRAINT_MAX_ITERATIONS = 40


def load_selected_calibration(path: str | Path) -> dict[str, Any]:
    """Load a complete selected calibration report."""

    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DHFRNPTValidationError("cannot load v2 calibration report") from error
    if not isinstance(raw, Mapping):
        raise DHFRNPTValidationError("v2 calibration report must be an object")
    return calibration.validate_calibration_report(raw, require_selected=True)


def build_draft_workload(
    v1_contract: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact diagnostic workload from selected calibration evidence."""

    selected = int(calibration_report.get("selected_formal_attempts", -1))
    if selected not in calibration.FORMAL_BUDGETS:
        raise DHFRNPTValidationError("calibration selected an undeclared budget")
    workload = copy.deepcopy(
        dict(_mapping(v1_contract.get("workload"), name="v1 workload"))
    )
    interval = int(workload["barostat"]["interval"])
    workload["steps"] = selected * interval
    workload["seeds"] = [7, 19]
    workload["barostat"]["expected_attempts"] = selected
    workload["barostat"]["max_log_volume_scale"] = math.log1p(
        OPENMM_INITIAL_VOLUME_FRACTION
    )
    workload["constraint_max_iterations"] = CONSTRAINT_MAX_ITERATIONS
    return workload


def diagnostic_report_path(out_dir: str | Path) -> Path:
    """Return the canonical seed-313 diagnostic report path."""

    return Path(out_dir) / f"seed-{DIAGNOSTIC_SEED}" / "report.json"


def implementation_fingerprint(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint product Python source and every runner used by diagnostics."""

    root = (
        Path(__file__).resolve().parents[1]
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
        )
    )
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    return {
        "inventory": inventory,
        "fingerprint": payload_fingerprint(inventory),
    }


def validate_diagnostic_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one diagnostic-only seed-313 report."""

    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if payload.get("schema") != DIAGNOSTIC_REPORT_SCHEMA:
        raise DHFRNPTValidationError("diagnostic report schema is unsupported")
    if fingerprint != payload_fingerprint(payload):
        raise DHFRNPTValidationError("diagnostic report fingerprint mismatch")
    if payload.get("scope") != DIAGNOSTIC_SCOPE:
        raise DHFRNPTValidationError("diagnostic report scope is invalid")
    if payload.get("seed") != DIAGNOSTIC_SEED:
        raise DHFRNPTValidationError("diagnostic report seed is invalid")
    if int(payload.get("selected_formal_attempts", -1)) not in (
        calibration.FORMAL_BUDGETS
    ):
        raise DHFRNPTValidationError("diagnostic attempt budget is invalid")
    for name in (
        "source_manifest_fingerprint",
        "calibration_report_fingerprint",
        "implementation_fingerprint",
    ):
        if not _is_sha256(payload.get(name)):
            raise DHFRNPTValidationError(
                f"diagnostic {name} is missing or invalid"
            )
    elapsed = float(payload.get("mlx_elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise DHFRNPTValidationError("diagnostic MLX elapsed time is invalid")
    checks = payload.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(type(value) is not bool for value in checks.values())
    ):
        raise DHFRNPTValidationError("diagnostic checks are invalid")
    blockers = [name for name, passed in checks.items() if not passed]
    if payload.get("blockers") != blockers:
        raise DHFRNPTValidationError("diagnostic blockers do not reconcile")
    expected_status = "passed" if not blockers else "failed"
    if payload.get("status") != expected_status:
        raise DHFRNPTValidationError("diagnostic status does not reconcile")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise DHFRNPTValidationError("diagnostic evidence is missing")
    summary = evidence.get("mlx")
    if not isinstance(summary, Mapping):
        raise DHFRNPTValidationError("diagnostic MLX summary is missing")
    if int(summary.get("barostat_attempts", -1)) != int(
        payload["selected_formal_attempts"]
    ):
        raise DHFRNPTValidationError("diagnostic attempts do not reconcile")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "mlx_trajectory.npz",
        "mlx_checkpoint.npz",
    }:
        raise DHFRNPTValidationError("diagnostic artifacts are incomplete")
    return {**payload, "report_fingerprint": str(fingerprint)}


def verify_diagnostic_directory(path: str | Path) -> dict[str, Any]:
    """Verify diagnostic report, artifacts, and bounded memory evidence."""

    directory = Path(path)
    try:
        report_raw = json.loads((directory / "report.json").read_text())
        memory = json.loads((directory / "memory.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DHFRNPTValidationError(
            "diagnostic report or memory trace is unreadable"
        ) from error
    if not isinstance(report_raw, Mapping) or not isinstance(memory, Mapping):
        raise DHFRNPTValidationError("diagnostic evidence must be JSON objects")
    report = validate_diagnostic_report(report_raw)
    for name, record in report["evidence"]["artifacts"].items():
        _validate_artifact(directory, name=name, record=record)
    summary = _mapping(
        memory.get("memory_trace_summary"),
        name="diagnostic memory summary",
    )
    memory_checks = {
        "limit_declared": (
            int(memory.get("bounded_process_limit_bytes", -1))
            == PROCESS_TREE_MAX_BYTES
        ),
        "below_limit": (
            int(memory.get("bounded_process_peak_physical_bytes", -1))
            <= PROCESS_TREE_MAX_BYTES
        ),
        "not_exceeded": memory.get("bounded_process_exceeded") is False,
        "not_timed_out": memory.get("bounded_process_timed_out") is False,
        "worker_passed": int(memory.get("bounded_process_returncode", -1)) == 0,
        "plateau_evaluated": summary.get("plateau_evaluated") is True,
        "plateau_passed": summary.get("plateau_passed") is True,
    }
    blockers = [name for name, passed in memory_checks.items() if not passed]
    if report["status"] != "passed":
        blockers.append("diagnostic_numerics")
    result = {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "checks": memory_checks,
        "report_fingerprint": report["report_fingerprint"],
        "selected_formal_attempts": report["selected_formal_attempts"],
        "mlx_elapsed_seconds": report["mlx_elapsed_seconds"],
        "peak_physical_bytes": int(
            memory.get("bounded_process_peak_physical_bytes", -1)
        ),
    }
    if blockers:
        raise DHFRNPTValidationError(
            "diagnostic verification failed: " + ", ".join(blockers)
        )
    return result


def _run_diagnostic(
    *,
    prepared_dir: Path,
    calibration_path: Path,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if seed != DIAGNOSTIC_SEED:
        if seed in {7, 19}:
            raise DHFRNPTValidationError(
                "formal target seeds cannot be used for diagnostics"
            )
        raise DHFRNPTValidationError("diagnostic seed must be 313")
    calibration_report = load_selected_calibration(calibration_path)
    v1_contract = load_validation_contract(V1_CONTRACT_PATH)
    source_identity = validate_prepared_boundary(prepared_dir, v1_contract)
    if (
        calibration_report["source_manifest_fingerprint"]
        != source_identity["manifest_fingerprint"]
    ):
        raise DHFRNPTValidationError(
            "diagnostic source does not match calibration"
        )
    workload = build_draft_workload(v1_contract, calibration_report)
    selected = int(calibration_report["selected_formal_attempts"])
    source_fingerprint = implementation_fingerprint()
    report_path = diagnostic_report_path(out_dir)
    if report_path.is_file():
        existing = validate_diagnostic_report(
            json.loads(report_path.read_text())
        )
        expected = {
            "source_manifest_fingerprint": source_identity[
                "manifest_fingerprint"
            ],
            "calibration_report_fingerprint": calibration_report[
                "report_fingerprint"
            ],
            "implementation_fingerprint": source_fingerprint["fingerprint"],
            "selected_formal_attempts": selected,
        }
        mismatched = [
            name
            for name, value in expected.items()
            if existing.get(name) != value
        ]
        if mismatched:
            raise DHFRNPTValidationError(
                "existing diagnostic identity mismatch: "
                + ", ".join(mismatched)
            )
        for name, record in existing["evidence"]["artifacts"].items():
            _validate_artifact(report_path.parent, name=name, record=record)
        return existing

    report_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = report_path.parent / "mlx_trajectory.npz"
    checkpoint = report_path.parent / "mlx_checkpoint.npz"
    prepared = v1_runner._npt_prepared(load_prepared_system(prepared_dir))
    started = perf_counter()
    result = v1_runner._run_mlx_npt(
        prepared,
        workload=workload,
        seed=seed,
        out=trajectory,
        checkpoint_out=checkpoint,
        steps=int(workload["steps"]),
        constraint_max_iterations=int(workload["constraint_max_iterations"]),
    )
    mx.eval(
        result.final_state.positions,
        result.final_state.velocities,
        result.final_state.forces,
    )
    elapsed = perf_counter() - started
    summary = v1_runner._mlx_npt_summary(
        result,
        atom_count=prepared.atom_count,
    )
    checks = {
        "finite": bool(summary["finite"]),
        "attempt_schedule": summary["barostat_attempts"] == selected,
        "cell_evolution": summary["barostat_accepted"] >= 1,
        "constraints": (
            summary["maximum_constraint_error_angstrom"]
            <= MAXIMUM_CONSTRAINT_ERROR_ANGSTROM
        ),
        "volume_bounds": (
            VOLUME_RATIO_BOUNDS[0]
            <= summary["minimum_volume_ratio"]
            <= VOLUME_RATIO_BOUNDS[1]
            and VOLUME_RATIO_BOUNDS[0]
            <= summary["maximum_volume_ratio"]
            <= VOLUME_RATIO_BOUNDS[1]
        ),
        "orthorhombic_cell": (
            summary["maximum_cell_off_diagonal_angstrom"]
            <= MAXIMUM_CELL_OFF_DIAGONAL_ANGSTROM
        ),
        "temperature_bound": (
            summary["maximum_temperature_K"] <= MAXIMUM_TEMPERATURE_K
        ),
        "pressure_bound": (
            summary["maximum_abs_pressure_bar"] <= MAXIMUM_ABS_PRESSURE_BAR
        ),
        "energy_stability": (
            summary["maximum_energy_excursion_per_atom_kj_mol"]
            <= MAXIMUM_ENERGY_EXCURSION_PER_ATOM_KJ_MOL
        ),
        "dynamic_pme": bool(summary["final_pme_plan_fingerprints"]),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    del result
    mx.clear_cache()
    evidence = {
        "mlx": summary,
        "artifacts": {
            trajectory.name: _artifact_record(trajectory),
            checkpoint.name: _artifact_record(checkpoint),
        },
    }
    unsigned = {
        "schema": DIAGNOSTIC_REPORT_SCHEMA,
        "scope": DIAGNOSTIC_SCOPE,
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "checks": checks,
        "seed": seed,
        "selected_formal_attempts": selected,
        "mlx_elapsed_seconds": elapsed,
        "source_manifest_fingerprint": source_identity[
            "manifest_fingerprint"
        ],
        "calibration_report_fingerprint": calibration_report[
            "report_fingerprint"
        ],
        "calibration_protocol_fingerprint": calibration_report[
            "protocol_fingerprint"
        ],
        "openmm_version": calibration_report["openmm_version"],
        "implementation_fingerprint": source_fingerprint["fingerprint"],
        "implementation_inventory": source_fingerprint["inventory"],
        "evidence": evidence,
    }
    report = {**unsigned, "report_fingerprint": payload_fingerprint(unsigned)}
    validate_diagnostic_report(report)
    _write_json_atomic(report_path, report)
    return report


def _run_formal_mlx(
    *,
    prepared_dir: Path,
    stage_dir: Path,
    contract: dict[str, Any],
    source_manifest_fingerprint: str,
    seed: int,
    split_resume: bool,
) -> dict[str, Any]:
    workload = contract["workload"]
    prepared = v1_runner._npt_prepared(load_prepared_system(prepared_dir))
    stage_dir.mkdir(parents=True, exist_ok=True)
    trajectory = stage_dir / "mlx_trajectory.npz"
    checkpoint = stage_dir / "mlx_checkpoint.npz"
    started = perf_counter()
    result = v1_runner._run_mlx_npt(
        prepared,
        workload=workload,
        seed=seed,
        out=trajectory,
        checkpoint_out=checkpoint,
        steps=int(workload["steps"]),
        constraint_max_iterations=int(workload["constraint_max_iterations"]),
    )
    mx.eval(
        result.final_state.positions,
        result.final_state.velocities,
        result.final_state.forces,
    )
    summary = v1_runner._mlx_npt_summary(
        result,
        atom_count=prepared.atom_count,
    )
    summary["seed"] = seed
    summary["elapsed_wall_seconds"] = perf_counter() - started
    resume_evidence = None
    resume_artifacts: dict[str, dict[str, Any]] = {}
    if split_resume:
        snapshot = v1_runner._mlx_resume_snapshot(result)
        del result
        mx.clear_cache()
        resume_evidence, resume_artifacts = v1_runner._run_mlx_resume_parity(
            prepared,
            workload=workload,
            seed=seed,
            stage_dir=stage_dir,
            continuous=snapshot,
            constraint_max_iterations=int(
                workload["constraint_max_iterations"]
            ),
        )
    else:
        del result
        mx.clear_cache()
    artifacts = {
        trajectory.name: _artifact_record(trajectory),
        checkpoint.name: _artifact_record(checkpoint),
        **resume_artifacts,
    }
    return dhfr_npt_v2.build_engine_report(
        contract=contract,
        source_manifest_fingerprint=source_manifest_fingerprint,
        seed=seed,
        engine="mlx",
        summary=summary,
        checks=dhfr_npt_v2.build_engine_checks(
            contract=contract,
            seed=seed,
            engine="mlx",
            summary=summary,
            checkpoint_resume=resume_evidence,
        ),
        artifacts=artifacts,
        checkpoint_resume=resume_evidence,
    )


def _run_formal_openmm(
    *,
    prepared_dir: Path,
    stage_dir: Path,
    contract: dict[str, Any],
    source_manifest_fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    api = v1_runner._preparation._common._load_openmm()
    expected_version = str(contract["engines"]["openmm_version"])
    if api.openmm.version.version != expected_version:
        raise DHFRNPTValidationError(
            "formal OpenMM version does not match the frozen contract"
        )
    prepared = v1_runner._npt_prepared(load_prepared_system(prepared_dir))
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    result = v1_runner._run_openmm_npt(
        prepared,
        contract=contract,
        platform_name="Reference",
        seed=seed,
    )
    samples = stage_dir / "openmm_samples.npz"
    v1_runner._atomic_savez(samples, result.pop("arrays"))
    result["seed"] = seed
    result["elapsed_wall_seconds"] = perf_counter() - started
    return dhfr_npt_v2.build_engine_report(
        contract=contract,
        source_manifest_fingerprint=source_manifest_fingerprint,
        seed=seed,
        engine="openmm",
        summary=result,
        checks=dhfr_npt_v2.build_engine_checks(
            contract=contract,
            seed=seed,
            engine="openmm",
            summary=result,
            checkpoint_resume=None,
        ),
        artifacts={samples.name: _artifact_record(samples)},
        openmm_version=str(result["openmm_version"]),
    )


def _run_formal(
    *,
    prepared_dir: Path,
    contract_path: Path,
    out_dir: Path,
    seed: int,
    engine: str,
    split_resume: bool,
) -> dict[str, Any]:
    contract = dhfr_npt_v2.load_contract(contract_path)
    if seed not in dhfr_npt_v2.FORMAL_SEEDS:
        raise DHFRNPTValidationError("formal seed must be 7 or 19")
    restart_seed = int(contract["restart_gate"]["seed"])
    if engine == "mlx" and split_resume != (seed == restart_seed):
        raise DHFRNPTValidationError(
            "formal MLX split/resume policy does not match the contract"
        )
    if engine == "openmm" and split_resume:
        raise DHFRNPTValidationError(
            "formal OpenMM phase cannot use --split-resume"
        )
    dhfr_npt_v2.freeze_check(
        contract_path=contract_path,
        prepared_dir=prepared_dir,
        formal_root=out_dir,
    )
    source = validate_prepared_boundary(prepared_dir, contract)
    report_path = dhfr_npt_v2.engine_report_path(
        out_dir,
        seed=seed,
        engine=engine,
    )
    if report_path.exists():
        return dhfr_npt_v2.load_engine_report(
            report_path,
            contract=contract,
            seed=seed,
            engine=engine,
        )
    if engine == "mlx":
        report = _run_formal_mlx(
            prepared_dir=prepared_dir,
            stage_dir=report_path.parent,
            contract=contract,
            source_manifest_fingerprint=source["manifest_fingerprint"],
            seed=seed,
            split_resume=split_resume,
        )
    else:
        report = _run_formal_openmm(
            prepared_dir=prepared_dir,
            stage_dir=report_path.parent,
            contract=contract,
            source_manifest_fingerprint=source["manifest_fingerprint"],
            seed=seed,
        )
    dhfr_npt_v2.write_engine_report(report_path, report)
    return report


def _validate_artifact(
    directory: Path,
    *,
    name: str,
    record: Mapping[str, Any],
) -> None:
    path = directory / name
    if (
        record.get("path") != name
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(record.get("byte_size", -1))
        or _file_sha256(path) != record.get("sha256")
    ):
        raise DHFRNPTValidationError(
            f"diagnostic artifact is missing or tampered: {name}"
        )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-diagnostic", type=Path)
    parser.add_argument("--stage", choices=("diagnostic", "formal"))
    parser.add_argument("--engine", choices=("mlx", "openmm"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--split-resume", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.verify_diagnostic is not None:
        if any(
            value is not None
            for value in (
                args.stage,
                args.seed,
                args.prepared,
                args.calibration,
                args.contract,
                args.out,
            )
        ) or args.split_resume:
            parser.error("--verify-diagnostic cannot be combined with run arguments")
        return args
    missing = [
        name
        for name, value in (
            ("--stage", args.stage),
            ("--seed", args.seed),
            ("--prepared", args.prepared),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error("missing run arguments: " + ", ".join(missing))
    if args.stage == "diagnostic":
        if args.calibration is None:
            parser.error("--calibration is required for diagnostics")
        if args.engine is not None or args.contract is not None or args.split_resume:
            parser.error(
                "diagnostics cannot use --engine, --contract, or --split-resume"
            )
    else:
        if args.engine is None or args.contract is None:
            parser.error("--engine and --contract are required for formal runs")
        if args.calibration is not None:
            parser.error("--calibration is only valid for diagnostics")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run or verify the bounded v2 diagnostic stage."""

    args = _parse_args(argv)
    if args.verify_diagnostic is not None:
        result = verify_diagnostic_directory(args.verify_diagnostic)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.stage == "diagnostic":
        report = _run_diagnostic(
            prepared_dir=args.prepared,
            calibration_path=args.calibration,
            out_dir=args.out,
            seed=args.seed,
        )
    else:
        report = _run_formal(
            prepared_dir=args.prepared,
            contract_path=args.contract,
            out_dir=args.out,
            seed=args.seed,
            engine=args.engine,
            split_resume=args.split_resume,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "blockers": report["blockers"],
                "report_fingerprint": report["report_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
