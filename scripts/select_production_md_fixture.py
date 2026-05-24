"""Select the Phase 3 production-MD readiness fixture candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_atomistic.prep.gpcrmd import (
    GPCRMD_IMPORT_REPORT_NAME,
    REQUIRED_MLX_IMPORT_FILE_ROLES,
    default_gpcrmd_targets,
)

DEFAULT_CACHE_DIR = Path("notebooks/ligand-receptor-motion/data/gpcrmd-cache/729")
DEFAULT_IMPORT_REPORT = (
    Path("notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-smoke")
    / GPCRMD_IMPORT_REPORT_NAME
)
SCHEMA_VERSION = 1


def build_candidate_record(
    *,
    root: Path,
    cache_dir: Path | None = None,
    import_report: Path | None = None,
) -> dict[str, Any]:
    """Build the selected fixture record or an artifact-source blocker."""

    root = root.resolve()
    target = default_gpcrmd_targets()[0]
    resolved_cache = _resolve(root, cache_dir or DEFAULT_CACHE_DIR)
    resolved_report = _resolve(root, import_report or DEFAULT_IMPORT_REPORT)
    file_statuses = _expected_file_statuses(target.files, resolved_cache, root)
    missing_required = sorted(
        status["role"]
        for status in file_statuses
        if status["role"] in REQUIRED_MLX_IMPORT_FILE_ROLES and not status["present"]
    )
    report_payload = _load_json(resolved_report)
    known_blockers = _known_pre_execution_blockers(
        missing_required=missing_required,
        import_report=report_payload,
        target_time_step_fs=target.time_step_fs,
    )
    selected = not missing_required

    return {
        "schema_version": SCHEMA_VERSION,
        "change": "production-md-readiness-fixture-probe",
        "status": "selected" if selected else "blocked",
        "selected": selected,
        "fixture": {
            "id": target.target_id,
            "dynamics_id": target.dynamics_id,
            "name": target.name,
            "source_url": target.source_url,
            "source_path": _display_path(resolved_cache, root),
            "source_kind": "ignored_local_cache" if resolved_cache.exists() else "external",
            "source_reproduction_command": (
                "UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python "
                "scripts/select_production_md_fixture.py --out "
                ".agent/work/production-md-readiness-fixture-probe/evidence/"
                "candidate-fixture.json"
            ),
            "prep_reproduction_command": (
                "UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run "
                "mlx_atomistic.prep Python API gpcrmd-import --cache "
                f"{_display_path(resolved_cache, root)} --out <ignored-output-dir> --json"
            ),
        },
        "scale": {
            "atom_count": target.total_atoms,
            "system_size": "large" if target.total_atoms >= 50_000 else "moderate",
            "dense_pair_count": target.total_atoms * (target.total_atoms - 1) // 2,
            "replicates": target.replicates,
            "accumulated_time_us": target.accumulated_time_us,
        },
        "topology_and_force_field": {
            "force_field_family": target.force_field,
            "topology_source": "CHARMM PSF/PDB/parameter package",
            "software": target.software,
            "expected_file_statuses": file_statuses,
            "missing_required_roles": missing_required,
        },
        "periodic_box": {
            "status": (
                "expected_from_gpcrmd_protocol"
                if target.periodic_box_expected and selected
                else "blocked_missing_required_inputs"
            ),
            "periodic_box_expected": target.periodic_box_expected,
            "source_files": ["model", "protocol"],
        },
        "components": {
            "water_model": target.solvent_type,
            "water_count": target.molecule_counts.get("Water", 0),
            "sodium_count": target.molecule_counts.get("Sodium ion", 0),
            "chloride_count": target.molecule_counts.get("Chloride", 0),
            "membrane": target.membrane_composition,
            "membrane_count": target.molecule_counts.get(
                str(target.membrane_composition), 0
            )
            if target.membrane_composition
            else 0,
            "receptor": target.receptor,
            "ligands": list(target.ligand_names),
        },
        "protocol_relevance": {
            "ensemble": target.ensemble,
            "time_step_fs": target.time_step_fs,
            "frame_stride_ns": target.frame_stride_ns,
            "pme_electrostatics_relevance": "required_for_periodic_explicit_membrane",
            "npt_barostat_relevance": (
                "not_protocol_required; target ensemble is NVT"
                if target.ensemble.upper() == "NVT"
                else "required"
            ),
            "constraints_hmr_virtual_sites_relevance": (
                "required_to_explain_4_fs_protocol"
                if target.time_step_fs >= 3.0
                else "standard_constraint_policy"
            ),
        },
        "artifact_policy": {
            "committed_evidence_only": True,
            "do_not_commit": [
                "raw GPCRmd downloads",
                "trajectory files",
                "prepared_system.npz",
                "binary checkpoints",
                "local cache directories",
            ],
            "allowed_committed_outputs": [
                "candidate-fixture.json",
                "openmm-reference.json",
                "mlx-probe.json",
                "blocker-matrix.json",
                "final-readiness-report.md",
            ],
        },
        "readiness_questions": [
            "artifact_source",
            "preparation",
            "topology_terms",
            "forcefield_terms",
            "constraints_hmr_virtual_sites",
            "electrostatics_pme",
            "integrator_protocol",
            "performance_runtime",
            "dependency_boundary",
        ],
        "known_pre_execution_blockers": known_blockers,
        "blockers": (
            [
                _blocker(
                    "artifact_source",
                    "blocked",
                    (
                        "missing required GPCRmd cache roles: "
                        + ", ".join(missing_required)
                    ),
                    command="inspect local cache before reference or MLX probe",
                    next_decision="provide local cache or record source blocker",
                )
            ]
            if missing_required
            else []
        ),
        "downstream": {
            "openmm_reference": "attempt_or_record_reference_blocker",
            "mlx_probe": "attempt_or_record_mlx_blocker",
            "parallel_safe_after_this_record": selected,
        },
    }


def write_candidate_record(record: dict[str, Any], out: Path) -> None:
    """Write candidate metadata as stable, sorted JSON."""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _known_pre_execution_blockers(
    *,
    missing_required: list[str],
    import_report: dict[str, Any] | None,
    target_time_step_fs: float,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if missing_required:
        return blockers
    if import_report:
        for item in import_report.get("blockers", []):
            blockers.append(
                _blocker(
                    "preparation",
                    "blocked",
                    str(item),
                    command="mlx_atomistic.prep Python API gpcrmd-import",
                    next_decision="route through S3 MLX probe and blocker matrix",
                )
            )
        compatibility = import_report.get("compatibility_report", {})
        for item in compatibility.get("unsupported_physics", []):
            category = (
                "constraints_hmr_virtual_sites"
                if "virtual" in str(item) or "hmr" in str(item).lower()
                else "forcefield_terms"
            )
            blockers.append(
                _blocker(
                    category,
                    "partial",
                    str(item),
                    command="read existing gpcrmd import report",
                    next_decision="verify in S3 whether this remains blocking",
                )
            )
    if target_time_step_fs >= 3.0 and not any(
        blocker["category"] == "constraints_hmr_virtual_sites" for blocker in blockers
    ):
        blockers.append(
            _blocker(
                "constraints_hmr_virtual_sites",
                "partial",
                "4 fs protocol requires explicit constraint/HMR/virtual-site policy",
                command="inspect GPCRmd protocol and topology before MLX run",
                next_decision="verify policy in S3 MLX probe",
            )
        )
    blockers.append(
        _blocker(
            "electrostatics_pme",
            "partial",
            "periodic explicit membrane system requires PME-scale electrostatics",
            command="inspect selected fixture protocol",
            next_decision="compare OpenMM reference and MLX readiness in S2/S3",
        )
    )
    return blockers


def _blocker(
    category: str,
    status: str,
    observed: str,
    *,
    command: str,
    next_decision: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "status": status,
        "observed_result": observed,
        "command": command,
        "next_implementation_decision": next_decision,
    }


def _expected_file_statuses(
    files: Any,
    cache_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for expected in files:
        matches = sorted(cache_dir.glob(f"{expected.file_id}_*")) if cache_dir.exists() else []
        path = matches[0] if matches else None
        statuses.append(
            {
                "role": expected.role,
                "file_id": expected.file_id,
                "label": expected.label,
                "format_hint": expected.format_hint,
                "present": path is not None,
                "path": _display_path(path, root) if path is not None else None,
                "size_bytes": path.stat().st_size
                if path is not None and path.is_file()
                else None,
            }
        )
    return statuses


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--import-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = build_candidate_record(
        root=args.root,
        cache_dir=args.cache_dir,
        import_report=args.import_report,
    )
    write_candidate_record(record, args.out)


if __name__ == "__main__":
    main()
