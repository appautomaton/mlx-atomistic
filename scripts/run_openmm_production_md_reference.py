"""Write reference-only OpenMM evidence for the production-MD fixture probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = Path("notebooks/ligand-receptor-motion/data/openmm-md")
SCHEMA_VERSION = 1


def main() -> None:
    args = _parse_args()
    candidate_path = Path(args.candidate)
    out_path = Path(args.out)
    candidate = _read_json(candidate_path)

    report = build_openmm_reference_evidence(
        candidate=candidate,
        candidate_path=candidate_path,
        reference_root=Path(args.reference_root),
        argv=sys.argv,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_openmm_reference_evidence(
    *,
    candidate: dict[str, Any],
    candidate_path: Path,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    dependency_status = _openmm_dependency_status()
    command = _command_record(candidate_path, argv)

    if candidate.get("status") != "selected":
        return _blocked_report(
            candidate=candidate,
            command=command,
            dependency_status=dependency_status,
            category="artifact_source",
            observed_result="candidate fixture was not selected",
        )

    fixture = candidate.get("fixture", {})
    dynamics_id = fixture.get("dynamics_id")
    reference = _select_reference_evidence(reference_root, dynamics_id)
    if reference is None:
        return _blocked_report(
            candidate=candidate,
            command=command,
            dependency_status=dependency_status,
            category="reference_evidence",
            observed_result=(
                "no existing OpenMM reference report found under "
                f"{reference_root} for dynamics_id={dynamics_id}"
            ),
        )

    run_report = reference["run_report"]
    preview_summary = reference.get("preview_summary")
    finite_checks = _finite_output_checks(run_report, preview_summary)
    platform = {
        "name": run_report.get("platform"),
        "properties": run_report.get("platform_properties", {}),
        "product_runtime": "mlx_atomistic",
        "reference_engine": "openmm",
        "reference_only": True,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ran" if finite_checks["finite_outputs"] else "blocked",
        "change": candidate.get("change"),
        "fixture_id": fixture.get("id"),
        "reference_engine": "openmm",
        "reference_engine_role": "reference-only/dev evidence; not product runtime",
        "command": command,
        "platform": platform,
        "protocol_settings": _protocol_settings(candidate, run_report, preview_summary),
        "supported_divergences_from_mlx_protocol": _supported_divergences(candidate, run_report),
        "finite_output_checks": finite_checks,
        "reference_only_dependency_status": dependency_status,
        "evidence_source": {
            "kind": "existing_reference_data",
            "run_report": _portable_evidence_path(reference["run_report_path"]),
            "preview_summary": (
                _portable_evidence_path(reference["preview_summary_path"])
                if reference.get("preview_summary_path") is not None
                else None
            ),
            "heavy_artifacts_reused": False,
            "new_trajectory_written": False,
        },
        "blockers": []
        if finite_checks["finite_outputs"]
        else [
            {
                "category": "finite_outputs",
                "status": "blocked",
                "observed_result": finite_checks["observed_result"],
                "product_runtime_boundary_changed": False,
            }
        ],
        "product_runtime_boundary": {
            "status": "preserved",
            "product_runtime": "mlx_atomistic",
            "openmm_scope": "scripts/tests/notebooks/dev reference evidence only",
            "src_imports_openmm": False,
        },
    }


def _select_reference_evidence(reference_root: Path, dynamics_id: Any) -> dict[str, Any] | None:
    root = _resolve(reference_root)
    if not root.exists():
        return None

    candidates: list[dict[str, Any]] = []
    prefix = f"{dynamics_id}-" if dynamics_id is not None else ""
    for run_report_path in sorted(root.glob(f"{prefix}*/openmm_charmm_md_run_report.json")):
        run_report = _read_json(run_report_path)
        preview_path = run_report_path.with_name("preview_summary.json")
        preview_summary = _read_json(preview_path) if preview_path.exists() else None
        candidates.append(
            {
                "run_report_path": run_report_path,
                "run_report": run_report,
                "preview_summary_path": preview_path if preview_path.exists() else None,
                "preview_summary": preview_summary,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            int(item["preview_summary"] is not None),
            int(item["run_report"].get("steps", 0)),
            int(item["run_report"].get("sampled_frame_count", 0)),
        ),
    )


def _protocol_settings(
    candidate: dict[str, Any],
    run_report: dict[str, Any],
    preview_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    relevance = candidate.get("protocol_relevance", {})
    return {
        "candidate_protocol": {
            "ensemble": relevance.get("ensemble"),
            "time_step_fs": relevance.get("time_step_fs"),
            "frame_stride_ns": relevance.get("frame_stride_ns"),
            "pme_electrostatics_relevance": relevance.get("pme_electrostatics_relevance"),
            "constraints_hmr_virtual_sites_relevance": relevance.get(
                "constraints_hmr_virtual_sites_relevance"
            ),
        },
        "openmm_reference_protocol": {
            "workflow": run_report.get("workflow"),
            "engine": run_report.get("engine"),
            "steps": run_report.get("steps"),
            "sample_interval": run_report.get("sample_interval"),
            "sampled_frame_count": run_report.get("sampled_frame_count"),
            "dt_ps": run_report.get("dt_ps"),
            "simulated_time_ps": run_report.get("simulated_time_ps"),
            "temperature_min_max_K": (
                preview_summary.get("temperature_min_max_K")
                if preview_summary is not None
                else _min_max(run_report.get("temperature_K", []))
            ),
            "cutoff_A": run_report.get("cutoff_A"),
            "switch_A": run_report.get("switch_A"),
            "ewald_tolerance": run_report.get("ewald_tolerance"),
            "minimize_steps": run_report.get("minimize_steps"),
            "nonbonded_method": "PME",
            "constraints": "HBonds",
            "rigid_water": True,
        },
    }


def _supported_divergences(
    candidate: dict[str, Any], run_report: dict[str, Any]
) -> list[dict[str, Any]]:
    relevance = candidate.get("protocol_relevance", {})
    reference_dt_fs = (
        float(run_report["dt_ps"]) * 1000.0 if run_report.get("dt_ps") is not None else None
    )
    return [
        {
            "category": "reference_engine",
            "candidate": "MLX product runtime probe",
            "openmm_reference": "OpenMM CHARMM/PME dev reference evidence",
            "reason": "S2 records comparator evidence only; it does not change runtime scope.",
        },
        {
            "category": "time_step",
            "candidate": relevance.get("time_step_fs"),
            "openmm_reference": reference_dt_fs,
            "unit": "fs",
            "reason": (
                "Existing reference preview used a smaller timestep for bounded stability evidence."
            ),
        },
        {
            "category": "trajectory_length",
            "candidate": candidate.get("scale", {}).get("accumulated_time_us"),
            "openmm_reference": run_report.get("simulated_time_ps"),
            "candidate_unit": "us",
            "openmm_reference_unit": "ps",
            "reason": "Reference evidence is a short probe, not a production-length reproduction.",
        },
        {
            "category": "platform",
            "candidate": "mlx runtime selected by S3",
            "openmm_reference": run_report.get("platform"),
            "reason": "OpenMM platform is reference-only platform evidence.",
        },
    ]


def _finite_output_checks(
    run_report: dict[str, Any], preview_summary: dict[str, Any] | None
) -> dict[str, Any]:
    energy_fields = ("potential_energy_kj_mol", "kinetic_energy_kj_mol", "temperature_K")
    energy_checks = {field: _all_finite(run_report.get(field, [])) for field in energy_fields}
    preview_checks = {
        "atoms_positive": _positive(preview_summary.get("atoms"))
        if preview_summary is not None
        else None,
        "frames_positive": _positive(preview_summary.get("frames"))
        if preview_summary is not None
        else None,
        "temperature_min_max_finite": _all_finite(preview_summary.get("temperature_min_max_K", []))
        if preview_summary is not None
        else None,
        "ligand_displacement_finite": _all_finite(
            [
                preview_summary.get("final_ligand_displacement_A"),
                preview_summary.get("max_ligand_displacement_A"),
            ]
        )
        if preview_summary is not None
        else None,
    }
    finite_outputs = all(energy_checks.values()) and all(
        value is not False for value in preview_checks.values()
    )
    observed = "finite OpenMM energies/temperature and preview metadata"
    if not finite_outputs:
        observed = "one or more OpenMM energy, temperature, or preview fields were non-finite"
    return {
        "finite_outputs": finite_outputs,
        "energy_fields": energy_checks,
        "preview_fields": preview_checks,
        "sampled_frame_count": run_report.get("sampled_frame_count"),
        "observed_result": observed,
    }


def _blocked_report(
    *,
    candidate: dict[str, Any],
    command: dict[str, Any],
    dependency_status: dict[str, Any],
    category: str,
    observed_result: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "change": candidate.get("change"),
        "fixture_id": candidate.get("fixture", {}).get("id"),
        "reference_engine": "openmm",
        "reference_engine_role": "reference-only/dev evidence; not product runtime",
        "command": command,
        "platform": {
            "name": None,
            "properties": {},
            "product_runtime": "mlx_atomistic",
            "reference_engine": "openmm",
            "reference_only": True,
        },
        "protocol_settings": {
            "candidate_protocol": candidate.get("protocol_relevance", {}),
            "openmm_reference_protocol": None,
        },
        "supported_divergences_from_mlx_protocol": [],
        "finite_output_checks": {
            "finite_outputs": False,
            "observed_result": observed_result,
        },
        "reference_only_dependency_status": dependency_status,
        "blockers": [
            {
                "category": category,
                "status": "blocked",
                "observed_result": observed_result,
                "product_runtime_boundary_changed": False,
            }
        ],
        "product_runtime_boundary": {
            "status": "preserved",
            "product_runtime": "mlx_atomistic",
            "openmm_scope": "scripts/tests/notebooks/dev reference evidence only",
            "src_imports_openmm": False,
        },
    }


def _openmm_dependency_status() -> dict[str, Any]:
    installed = importlib.util.find_spec("openmm") is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version("openmm")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return {
        "package": "openmm",
        "installed": installed,
        "version": version,
        "dependency_group": "dev",
        "runtime_role": "reference-only",
        "product_runtime_dependency": False,
    }


def _command_record(candidate_path: Path, argv: list[str] | None) -> dict[str, Any]:
    args = list(argv) if argv is not None else []
    return {
        "argv": args,
        "command": " ".join(args),
        "candidate": str(candidate_path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text())


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _portable_evidence_path(path: Path) -> str:
    resolved = _resolve(path)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _all_finite(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    return all(isinstance(value, int | float) and math.isfinite(float(value)) for value in values)


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and float(value) > 0


def _min_max(values: list[Any]) -> list[float] | None:
    finite = [float(value) for value in values if isinstance(value, int | float)]
    if not finite:
        return None
    return [min(finite), max(finite)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    main()
