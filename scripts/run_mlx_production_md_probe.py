"""Run the selected production-MD fixture through the MLX probe path."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    artifact_readiness_report,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.prep.gpcrmd import attempt_gpcrmd_prepared_artifact_import
from mlx_atomistic.prep.runner import _production_neighbor_manager
from mlx_atomistic.protocols import MinimizeThenNVTProtocol, protocol_readiness_report
from mlx_atomistic.runtime import get_platform_boundary_report, get_runtime_info

SCHEMA_VERSION = 1
DEFAULT_MAX_BOUNDED_RUN_ATOMS = 100_000
PROBE_TMP_PLACEHOLDER = "<mlx-production-md-probe>"
PROBE_TMP_PATH_PATTERN = re.compile(
    r"(?:/tmp|/var/folders)/(?:[^\s\"';,)]+/)*"
    r"mlx-production-md-probe-[^\s\"';,)]+"
    r"(?:/[^\s\"';,)]+)*"
)


def build_mlx_probe_record(
    *,
    candidate_path: Path,
    out_path: Path,
    root: Path | None = None,
    prep_importer: Callable[
        [str | Path, str | Path],
        Any,
    ] = attempt_gpcrmd_prepared_artifact_import,
    max_bounded_run_atoms: int = DEFAULT_MAX_BOUNDED_RUN_ATOMS,
) -> dict[str, Any]:
    """Attempt the selected fixture and return JSON-safe MLX probe evidence."""

    root = Path.cwd().resolve() if root is None else root.resolve()
    candidate = json.loads(candidate_path.read_text())
    command = _probe_command(candidate_path, out_path)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "change": "production-md-readiness-fixture-probe",
        "fixture": dict(candidate.get("fixture", {})),
        "command": command,
        "status": "running",
        "earliest_blocker": None,
        "taxonomy_blockers": [],
        "stages": {
            "prep": _stage("pending"),
            "load": _stage("pending"),
            "readiness": _stage("pending"),
            "run": _stage("pending"),
        },
        "platform_readiness": _platform_readiness_metadata(),
        "finite_checks": _empty_finite_checks(),
        "runtime_performance": {
            "bounded_run_attempted": False,
            "bounded_run_completed": False,
            "wall_time_seconds": None,
            "max_bounded_run_atoms": max_bounded_run_atoms,
        },
        "dependency_boundary": {
            "status": "passed",
            "product_runtime": "mlx_atomistic",
            "reference_engines_imported": False,
            "vendor_runtime_imports": False,
        },
    }

    if not candidate.get("selected", False):
        blocker = _blocker(
            category="artifact_source",
            status="blocked",
            command="read candidate fixture evidence",
            observed_result="candidate fixture is not selected",
            context=_display_path(candidate_path, root),
            next_decision="provide selected candidate evidence before MLX probe",
            prevents_bounded_pass=True,
        )
        _block_report(report, "prep", blocker, started)
        return report

    cache_path = root / str(candidate["fixture"]["source_path"])
    with tempfile.TemporaryDirectory(prefix="mlx-production-md-probe-") as tmp:
        prep_out = Path(tmp) / "prepared"
        prep_started = time.perf_counter()
        attempt = prep_importer(cache_path, prep_out)
        attempt_payload = attempt.to_json_dict()
        prepared_artifact_evidence_path = _prepared_artifact_evidence_path(
            attempt_payload.get("prepared_artifact_path"),
            probe_tmp=Path(tmp),
        )
        report["stages"]["prep"] = {
            **_stage("passed" if attempt_payload.get("exported") else "blocked"),
            "command": (
                "mlx_atomistic.prep.gpcrmd.attempt_gpcrmd_prepared_artifact_import"
            ),
            "duration_seconds": _elapsed(prep_started),
            "prepared_artifact_path": prepared_artifact_evidence_path,
            "exported": bool(attempt_payload.get("exported")),
            "blockers": list(attempt_payload.get("blockers", [])),
            "compatibility_report": _compact_compatibility_report(
                attempt_payload.get("compatibility_report", {})
            ),
        }
        if not attempt_payload.get("exported"):
            blocker = _blocker_from_prep_attempt(
                attempt_payload,
                candidate_path=candidate_path,
                root=root,
            )
            _block_report(report, "prep", blocker, started)
            return report

        artifact_path = Path(str(attempt_payload["prepared_artifact_path"]))
        load_started = time.perf_counter()
        try:
            artifact = load_prepared_mlx_artifact(artifact_path, require_production=True)
        except (FileNotFoundError, ValueError, MLXCompatibilityError) as exc:
            blocker = _blocker(
                category=_category_from_text(str(exc), default="topology_terms"),
                status="blocked",
                command="load_prepared_mlx_artifact(require_production=True)",
                observed_result=str(exc),
                context=f"prepared_artifact_path={prepared_artifact_evidence_path}",
                next_decision="fix prepared artifact production compatibility before runtime",
                prevents_bounded_pass=True,
            )
            report["stages"]["load"] = {
                **_stage("blocked"),
                "duration_seconds": _elapsed(load_started),
                "error": str(exc),
            }
            _block_report(report, "load", blocker, started)
            return report

        report["stages"]["load"] = {
            **_stage("passed"),
            "duration_seconds": _elapsed(load_started),
            "atom_count": artifact.atom_count,
        }
        report["finite_checks"].update(_artifact_finite_checks(artifact.arrays))

        readiness_started = time.perf_counter()
        artifact_readiness = artifact_readiness_report(
            artifact.metadata,
            require_production=True,
            arrays=artifact.arrays,
        )
        protocol_readiness = protocol_readiness_report(
            candidate.get("protocol_relevance", {})
        )
        readiness_reports = {
            "artifact": artifact_readiness.to_dict(),
            "protocol": protocol_readiness.to_dict(),
        }
        readiness_blockers = [
            (name, item)
            for name, payload in readiness_reports.items()
            for item in payload.get("blockers", [])
        ]
        report["stages"]["readiness"] = {
            **_stage("blocked" if readiness_blockers else "passed"),
            "duration_seconds": _elapsed(readiness_started),
            "reports": readiness_reports,
        }
        if readiness_blockers:
            name, observed = readiness_blockers[0]
            blocker = _blocker(
                category=_category_from_text(str(observed), default=name),
                status="blocked",
                command=f"{name}_readiness_report",
                observed_result=str(observed),
                context=f"prepared_artifact_path={prepared_artifact_evidence_path}",
                next_decision="resolve MLX readiness blocker before bounded execution",
                prevents_bounded_pass=True,
            )
            _block_report(report, "readiness", blocker, started)
            return report

        if artifact.atom_count > max_bounded_run_atoms:
            blocker = _blocker(
                category="performance_runtime",
                status="blocked",
                command="bounded MLX production-MD proof run",
                observed_result=(
                    f"atom_count={artifact.atom_count} exceeds bounded probe cap "
                    f"{max_bounded_run_atoms}"
                ),
                context=f"prepared_artifact_path={prepared_artifact_evidence_path}",
                next_decision="add an approved scalable runtime proof before running this scale",
                prevents_bounded_pass=True,
            )
            _block_report(report, "run", blocker, started)
            return report

        _run_bounded_probe(
            report,
            artifact,
            started,
            prepared_artifact_evidence_path=prepared_artifact_evidence_path,
        )
        return report


def write_mlx_probe_record(record: Mapping[str, Any], out: Path) -> None:
    """Write MLX probe evidence as stable JSON."""

    out.parent.mkdir(parents=True, exist_ok=True)
    portable_record = _redact_probe_temp_paths(record)
    out.write_text(json.dumps(portable_record, indent=2, sort_keys=True) + "\n")


def _run_bounded_probe(
    report: dict[str, Any],
    artifact: Any,
    started: float,
    *,
    prepared_artifact_evidence_path: str | None,
) -> None:
    run_started = time.perf_counter()
    report["runtime_performance"]["bounded_run_attempted"] = True
    try:
        system, terms, constraints = build_mlx_system_from_artifact(artifact)
        neighbor_manager = _production_neighbor_manager(
            system,
            terms,
            require_production=True,
        )
        result = MinimizeThenNVTProtocol(
            minimize_steps=0,
            equilibration_steps=0,
            production_steps=2,
            sample_interval=1,
            compile_force_evaluator=False,
        )
        from mlx_atomistic.protocols import run_minimize_then_nvt

        trajectory = run_minimize_then_nvt(
            artifact.arrays["positions"],
            artifact.arrays["velocities"],
            artifact.arrays["masses"],
            terms,
            protocol=result,
            cell=artifact.cell,
            constraints=constraints,
            unit_system=artifact.unit_system,
            neighbor_manager=neighbor_manager,
        )
    except Exception as exc:  # pragma: no cover - exercised only by large live artifacts.
        blocker = _blocker(
            category=_category_from_text(str(exc), default="stability_finiteness"),
            status="blocked",
            command="run_minimize_then_nvt bounded production probe",
            observed_result=str(exc),
            context=f"prepared_artifact_path={prepared_artifact_evidence_path}",
            next_decision="fix MLX runtime blocker before bounded fixture execution",
            prevents_bounded_pass=True,
        )
        report["stages"]["run"] = {
            **_stage("blocked"),
            "duration_seconds": _elapsed(run_started),
            "error": str(exc),
        }
        _block_report(report, "run", blocker, started)
        return

    production = trajectory.production
    total_energy = np.asarray(production.total_energy)
    nonbonded_runtime = dict(production.nonbonded_report)
    report["stages"]["run"] = {
        **_stage("passed"),
        "duration_seconds": _elapsed(run_started),
        "production_steps": 2,
        "sample_interval": 1,
        "nonbonded_runtime": nonbonded_runtime,
    }
    report["finite_checks"]["energies"] = bool(np.all(np.isfinite(total_energy)))
    report["runtime_performance"].update(
        {
            "bounded_run_completed": True,
            "wall_time_seconds": _elapsed(started),
            "backend": nonbonded_runtime.get("backend"),
            "fallback_reason": nonbonded_runtime.get("fallback_reason"),
            "pair_count": nonbonded_runtime.get("pair_count"),
            "candidate_count": nonbonded_runtime.get("candidate_count"),
            "candidate_waste_count": nonbonded_runtime.get("candidate_waste_count"),
            "candidate_waste_fraction": nonbonded_runtime.get("candidate_waste_fraction"),
            "neighbor_update_wall_seconds": nonbonded_runtime.get(
                "neighbor_update_wall_seconds"
            ),
            "neighbor_rebuild_wall_seconds": nonbonded_runtime.get(
                "neighbor_rebuild_wall_seconds"
            ),
            "force_evaluation_wall_seconds": nonbonded_runtime.get(
                "force_evaluation_wall_seconds"
            ),
        }
    )
    report["status"] = "passed"


def _block_report(
    report: dict[str, Any],
    stage: str,
    blocker: dict[str, Any],
    started: float,
) -> None:
    report["status"] = "blocked"
    report["earliest_blocker"] = blocker
    report["taxonomy_blockers"].append(blocker)
    report["stages"][stage]["status"] = "blocked"
    report["runtime_performance"]["wall_time_seconds"] = _elapsed(started)


def _blocker_from_prep_attempt(
    attempt: Mapping[str, Any],
    *,
    candidate_path: Path,
    root: Path,
) -> dict[str, Any]:
    blockers = [str(item) for item in attempt.get("blockers", [])]
    observed = blockers[0] if blockers else "prepared artifact was not exported"
    category = _category_from_text(observed, default="preparation")
    return _blocker(
        category=category,
        status="blocked",
        command="attempt_gpcrmd_prepared_artifact_import",
        observed_result=observed,
        context=(
            f"candidate={_display_path(candidate_path, root)}; "
            f"target_id={attempt.get('target_id')}; dynamics_id={attempt.get('dynamics_id')}"
        ),
        next_decision=str(
            dict(attempt.get("compatibility_report", {})).get(
                "next_engine_slice",
                "fix MLX preparation blocker before artifact loading",
            )
        ),
        prevents_bounded_pass=True,
    )


def _compact_compatibility_report(payload: Any) -> dict[str, Any]:
    """Keep prep evidence small while preserving readiness and runtime decisions."""

    if not isinstance(payload, Mapping):
        return {}
    keys = (
        "target_id",
        "dynamics_id",
        "runnable_now",
        "supported_now",
        "missing_input",
        "unsupported_physics",
        "runtime_risk",
        "next_engine_slice",
        "warnings",
        "required_terms",
        "supported_terms",
        "rejected_terms",
        "rejection_reasons",
        "term_counts",
    )
    return {key: payload[key] for key in keys if key in payload}


def _category_from_text(text: str, *, default: str) -> str:
    normalized = text.lower()
    if "missing" in normalized and ("cache" in normalized or "file:" in normalized):
        return "artifact_source"
    if "parse_failed" in normalized or "parmed" in normalized:
        return "preparation"
    if "atom type" in normalized or "parameter" in normalized or "force-field" in normalized:
        return "forcefield_terms"
    if "topology" in normalized or "bond" in normalized or "residue" in normalized:
        return "topology_terms"
    if "hmr" in normalized or "hydrogen_mass" in normalized or "virtual_site" in normalized:
        return "constraints_hmr_virtual_sites"
    if "pme" in normalized or "ewald" in normalized or "electrostatic" in normalized:
        return "electrostatics_pme"
    if "npt" in normalized or "barostat" in normalized:
        return "npt_barostat"
    if "protocol" in normalized or "ensemble" in normalized or "timestep" in normalized:
        return "integrator_protocol"
    if "restart" in normalized or "checkpoint" in normalized or "trajectory" in normalized:
        return "output_restart"
    if "finite" in normalized or "nan" in normalized or "inf" in normalized:
        return "stability_finiteness"
    return default


def _blocker(
    *,
    category: str,
    status: str,
    command: str,
    observed_result: str,
    context: str,
    next_decision: str,
    prevents_bounded_pass: bool,
) -> dict[str, Any]:
    return {
        "category": category,
        "status": status,
        "fixture": "gpcrmd-729-beta1-5f8u-cyanopindolol",
        "command": command,
        "observed_result": observed_result,
        "smallest_reproduction_context": context,
        "affected_acceptance_criteria": ["AC4", "AC6", "AC7", "AC8"],
        "next_implementation_decision": next_decision,
        "prevents_bounded_pass": prevents_bounded_pass,
    }


def _stage(status: str) -> dict[str, Any]:
    return {"status": status, "duration_seconds": None}


def _empty_finite_checks() -> dict[str, Any]:
    return {
        "positions": None,
        "velocities": None,
        "energies": None,
        "reason": "not available before artifact load/run",
    }


def _artifact_finite_checks(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "positions": bool(np.all(np.isfinite(np.asarray(arrays["positions"])))),
        "velocities": bool(np.all(np.isfinite(np.asarray(arrays["velocities"])))),
        "energies": None,
        "reason": "energies are only available after bounded run",
    }


def _platform_readiness_metadata() -> dict[str, Any]:
    runtime = get_runtime_info()
    boundary = get_platform_boundary_report(runtime_info=runtime)
    return {
        "runtime": runtime.to_dict(),
        "boundary": boundary.to_dict(),
    }


def _probe_command(candidate_path: Path, out_path: Path) -> str:
    return (
        "UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python "
        f"scripts/run_mlx_production_md_probe.py --candidate {candidate_path} --out {out_path}"
    )


def _prepared_artifact_evidence_path(path: Any, *, probe_tmp: Path) -> str | None:
    """Return a portable evidence path for temporary probe artifacts."""

    if path is None:
        return None
    artifact_path = Path(str(path))
    try:
        relative = artifact_path.resolve().relative_to(probe_tmp.resolve())
    except ValueError:
        return str(path)
    return f"{PROBE_TMP_PLACEHOLDER}/{relative}"


def _redact_probe_temp_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _redact_probe_temp_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_probe_temp_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_probe_temp_paths(item) for item in value)
    if isinstance(value, str):
        return PROBE_TMP_PATH_PATTERN.sub(PROBE_TMP_PLACEHOLDER, value)
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    record = build_mlx_probe_record(candidate_path=args.candidate, out_path=args.out)
    write_mlx_probe_record(record, args.out)


if __name__ == "__main__":
    main()
