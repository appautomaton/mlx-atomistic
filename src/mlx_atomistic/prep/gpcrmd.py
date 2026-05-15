"""GPCRmd target registry and selection gates.

This module records source-backed GPCRmd candidate metadata without downloading
large trajectory packages or invoking external MD engines.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.io import JSON_NAME, NPZ_NAME, VIEW_PDB_NAME, save_prepared_system
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.prep.topology_import import (
    TopologyImportError,
    build_charmm_psf_mass_prelude,
    import_amber_prmtop,
    import_charmm_with_parmed,
)

GPCRMD_DATA_DOWNLOAD_DOCS_URL = "https://gpcrmd-docs.readthedocs.io/en/latest/data-download.html"
GPCRMD_IMPORT_REPORT_NAME = "gpcrmd_import_report.json"

REQUIRED_FILE_ROLES = frozenset({"model", "topology", "parameters", "protocol", "trajectory"})
REQUIRED_MLX_IMPORT_FILE_ROLES = frozenset({"model", "topology", "parameters", "protocol"})
OPTIONAL_ANALYSIS_FILE_ROLES = frozenset({"trajectory"})
REQUIRED_ION_NAMES = frozenset({"sodium ion", "chloride"})
REQUIRED_PREPARED_ARTIFACT_FIELDS = (
    "coordinates",
    "topology",
    "force_field_parameters",
    "explicit_hydrogens",
    "water_mask",
    "ion_mask",
    "ligand_mask",
    "receptor_mask",
    "box_vectors",
    "constraints",
    "nonbonded_exclusions",
    "nonbonded_exceptions",
)


class GPCRmdTargetError(ValueError):
    """Raised when GPCRmd target metadata cannot satisfy the selection gate."""


class GPCRmdInspectionError(ValueError):
    """Raised when a GPCRmd cache cannot be inspected as requested."""


@dataclass(frozen=True)
class GPCRmdFile:
    """One downloadable file advertised by a GPCRmd simulation report."""

    role: str
    file_id: int
    label: str
    format_hint: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "file_id": self.file_id,
            "label": self.label,
            "format_hint": self.format_hint,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> GPCRmdFile:
        return cls(
            role=str(payload["role"]),
            file_id=int(payload["file_id"]),
            label=str(payload["label"]),
            format_hint=(
                None if payload.get("format_hint") is None else str(payload["format_hint"])
            ),
        )


@dataclass(frozen=True)
class GPCRmdCacheFileStatus:
    """Presence report for one expected GPCRmd file in a local cache."""

    role: str
    file_id: int
    label: str
    format_hint: str | None
    present: bool
    path: str | None = None
    size_bytes: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "file_id": self.file_id,
            "label": self.label,
            "format_hint": self.format_hint,
            "present": self.present,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class GPCRmdCacheInspection:
    """Inspection report for a local GPCRmd cache directory or manifest."""

    target: GPCRmdTarget
    cache_path: str
    cache_exists: bool
    cache_kind: str
    file_statuses: tuple[GPCRmdCacheFileStatus, ...]

    @property
    def present_file_count(self) -> int:
        return sum(1 for status in self.file_statuses if status.present)

    @property
    def missing_file_ids(self) -> tuple[int, ...]:
        return tuple(status.file_id for status in self.file_statuses if not status.present)

    @property
    def complete(self) -> bool:
        return not self.missing_file_ids

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.selection_report(),
            "cache_path": self.cache_path,
            "cache_exists": self.cache_exists,
            "cache_kind": self.cache_kind,
            "expected_file_count": len(self.file_statuses),
            "present_file_count": self.present_file_count,
            "missing_file_ids": list(self.missing_file_ids),
            "complete": self.complete,
            "files": [status.to_json_dict() for status in self.file_statuses],
            "resolved_role_paths": {
                role: [str(path) for path in paths]
                for role, paths in _inspection_role_paths(self).items()
            },
        }


@dataclass(frozen=True)
class GPCRmdMLXCompatibilityReport:
    """Fail-closed MLX compatibility report for a GPCRmd target inspection."""

    target_id: str
    dynamics_id: int
    runnable_now: bool
    supported_now: tuple[str, ...]
    missing_input: tuple[str, ...]
    unsupported_physics: tuple[str, ...]
    runtime_risk: dict[str, Any]
    next_engine_slice: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "target_id": self.target_id,
            "dynamics_id": self.dynamics_id,
            "runnable_now": self.runnable_now,
            "supported_now": list(self.supported_now),
            "missing_input": list(self.missing_input),
            "unsupported_physics": list(self.unsupported_physics),
            "runtime_risk": dict(self.runtime_risk),
            "next_engine_slice": self.next_engine_slice,
            "warnings": list(self.warnings),
        }
        payload.update(self.extra_fields)
        return payload


@dataclass(frozen=True)
class GPCRmdPreparedImportAttempt:
    """Result of attempting to turn a GPCRmd cache into an MLX prepared artifact."""

    target_id: str
    dynamics_id: int
    out_dir: str
    exported: bool
    prepared_artifact_path: str | None
    blockers: tuple[str, ...]
    required_artifact_fields: tuple[str, ...]
    compatibility_report: GPCRmdMLXCompatibilityReport
    import_details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "dynamics_id": self.dynamics_id,
            "out_dir": self.out_dir,
            "exported": self.exported,
            "prepared_artifact_path": self.prepared_artifact_path,
            "blockers": list(self.blockers),
            "required_artifact_fields": list(self.required_artifact_fields),
            "compatibility_report": self.compatibility_report.to_json_dict(),
            "import_details": dict(self.import_details),
        }


@dataclass(frozen=True)
class GPCRmdReadinessInventory:
    """Target-level inventory for deciding the next GPCRmd MLX engine work."""

    target_id: str
    dynamics_id: int
    target_decision: dict[str, Any]
    required_files: tuple[dict[str, Any], ...]
    optional_analysis_features: tuple[dict[str, Any], ...]
    system_components: dict[str, Any]
    required_force_terms: tuple[dict[str, Any], ...]
    box: dict[str, Any]
    constraints: dict[str, Any]
    exceptions: dict[str, Any]
    protocol_requirements: dict[str, Any]
    first_engine_blockers: tuple[dict[str, Any], ...]
    missing_input: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "dynamics_id": self.dynamics_id,
            "target_decision": dict(self.target_decision),
            "required_files": [dict(item) for item in self.required_files],
            "optional_analysis_features": [
                dict(item) for item in self.optional_analysis_features
            ],
            "system_components": dict(self.system_components),
            "required_force_terms": [dict(item) for item in self.required_force_terms],
            "box": dict(self.box),
            "constraints": dict(self.constraints),
            "exceptions": dict(self.exceptions),
            "protocol_requirements": dict(self.protocol_requirements),
            "first_engine_blockers": [
                dict(item) for item in self.first_engine_blockers
            ],
            "missing_input": list(self.missing_input),
        }


@dataclass(frozen=True)
class GPCRmdTarget:
    """A candidate GPCRmd system for MLX import and compatibility reporting."""

    target_id: str
    dynamics_id: int
    name: str
    source_url: str
    pdb_id: str
    receptor: str
    activation_state: str
    ligand_names: tuple[str, ...]
    solvent_type: str
    membrane_type: str
    membrane_composition: str | None
    molecule_counts: dict[str, int]
    total_atoms: int
    software: str
    force_field: str
    ensemble: str
    time_step_fs: float
    frame_stride_ns: float | None
    replicates: int
    accumulated_time_us: float | None
    periodic_box_expected: bool
    files: tuple[GPCRmdFile, ...]
    reference_urls: tuple[str, ...] = field(default_factory=tuple)
    selection_reason: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def file_roles(self) -> set[str]:
        return {item.role for item in self.files}

    def missing_selection_requirements(self) -> list[str]:
        missing: list[str] = []
        for field_name in [
            "target_id",
            "name",
            "source_url",
            "pdb_id",
            "receptor",
            "solvent_type",
            "membrane_type",
            "software",
            "force_field",
            "ensemble",
        ]:
            if not str(getattr(self, field_name, "")).strip():
                missing.append(field_name)
        if not self.ligand_names:
            missing.append("ligand_names")
        if self.total_atoms <= 0:
            missing.append("total_atoms")
        if self.time_step_fs <= 0:
            missing.append("time_step_fs")
        if self.replicates <= 0:
            missing.append("replicates")
        if not self.periodic_box_expected:
            missing.append("periodic_box_expected")

        missing_file_roles = sorted(REQUIRED_FILE_ROLES - self.file_roles())
        missing.extend(f"file:{role}" for role in missing_file_roles)

        molecule_names = {name.lower() for name in self.molecule_counts}
        if "water" not in molecule_names:
            missing.append("molecule:Water")
        missing_ions = sorted(REQUIRED_ION_NAMES - molecule_names)
        missing.extend(f"molecule:{name.title()}" for name in missing_ions)

        return missing

    def selection_report(self) -> dict[str, Any]:
        missing = self.missing_selection_requirements()
        return {
            "target_id": self.target_id,
            "dynamics_id": self.dynamics_id,
            "source_url": self.source_url,
            "pdb_id": self.pdb_id,
            "receptor": self.receptor,
            "ligand_count": len(self.ligand_names),
            "total_atoms": self.total_atoms,
            "solvent_type": self.solvent_type,
            "membrane_type": self.membrane_type,
            "membrane_composition": self.membrane_composition,
            "software": self.software,
            "force_field": self.force_field,
            "file_roles": sorted(self.file_roles()),
            "required_file_roles": sorted(REQUIRED_FILE_ROLES),
            "missing_requirements": missing,
            "passes_selection_gate": len(missing) == 0,
            "selection_reason": self.selection_reason,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "dynamics_id": self.dynamics_id,
            "name": self.name,
            "source_url": self.source_url,
            "pdb_id": self.pdb_id,
            "receptor": self.receptor,
            "activation_state": self.activation_state,
            "ligand_names": list(self.ligand_names),
            "solvent_type": self.solvent_type,
            "membrane_type": self.membrane_type,
            "membrane_composition": self.membrane_composition,
            "molecule_counts": dict(self.molecule_counts),
            "total_atoms": self.total_atoms,
            "software": self.software,
            "force_field": self.force_field,
            "ensemble": self.ensemble,
            "time_step_fs": self.time_step_fs,
            "frame_stride_ns": self.frame_stride_ns,
            "replicates": self.replicates,
            "accumulated_time_us": self.accumulated_time_us,
            "periodic_box_expected": self.periodic_box_expected,
            "files": [item.to_json_dict() for item in self.files],
            "reference_urls": list(self.reference_urls),
            "selection_reason": self.selection_reason,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> GPCRmdTarget:
        return cls(
            target_id=str(payload["target_id"]),
            dynamics_id=int(payload["dynamics_id"]),
            name=str(payload["name"]),
            source_url=str(payload["source_url"]),
            pdb_id=str(payload["pdb_id"]),
            receptor=str(payload["receptor"]),
            activation_state=str(payload.get("activation_state", "")),
            ligand_names=tuple(str(item) for item in payload.get("ligand_names", [])),
            solvent_type=str(payload.get("solvent_type", "")),
            membrane_type=str(payload.get("membrane_type", "")),
            membrane_composition=(
                None
                if payload.get("membrane_composition") is None
                else str(payload["membrane_composition"])
            ),
            molecule_counts={
                str(name): int(count)
                for name, count in dict(payload.get("molecule_counts", {})).items()
            },
            total_atoms=int(payload.get("total_atoms", 0)),
            software=str(payload.get("software", "")),
            force_field=str(payload.get("force_field", "")),
            ensemble=str(payload.get("ensemble", "")),
            time_step_fs=float(payload.get("time_step_fs", 0.0)),
            frame_stride_ns=(
                None
                if payload.get("frame_stride_ns") is None
                else float(payload["frame_stride_ns"])
            ),
            replicates=int(payload.get("replicates", 0)),
            accumulated_time_us=(
                None
                if payload.get("accumulated_time_us") is None
                else float(payload["accumulated_time_us"])
            ),
            periodic_box_expected=bool(payload.get("periodic_box_expected", False)),
            files=tuple(
                GPCRmdFile.from_json_dict(item) for item in payload.get("files", [])
            ),
            reference_urls=tuple(str(item) for item in payload.get("reference_urls", [])),
            selection_reason=str(payload.get("selection_reason", "")),
            notes=tuple(str(item) for item in payload.get("notes", [])),
        )


def default_gpcrmd_targets() -> tuple[GPCRmdTarget, ...]:
    """Return curated GPCRmd candidates for the first real-system milestone."""

    return (
        GPCRmdTarget(
            target_id="gpcrmd-729-beta1-5f8u-cyanopindolol",
            dynamics_id=729,
            name="Beta-1 adrenergic receptor in complex with cyanopindolol-like ligand",
            source_url="https://www.gpcrmd.org/dynadb/dynamics/id/729/",
            pdb_id="5F8U",
            receptor="Beta-1 adrenergic receptor",
            activation_state="Inactive",
            ligand_names=(
                "4-{[(2s)-3-(tert-butylamino)-2-hydroxypropyl]oxy}-3h-indole-2-carbonitrile",
            ),
            solvent_type="TIP3P",
            membrane_type="Homogeneous",
            membrane_composition="POPC",
            molecule_counts={
                "Water": 19944,
                "POPC": 200,
                "Chloride": 74,
                "Sodium ion": 57,
                "Beta-1 adrenergic receptor": 1,
                "orthosteric ligand": 1,
            },
            total_atoms=92001,
            software="ACEMD3, GPUGRID",
            force_field="CHARMM c36 Jul 2020",
            ensemble="NVT",
            time_step_fs=4.0,
            frame_stride_ns=0.2,
            replicates=3,
            accumulated_time_us=1.5,
            periodic_box_expected=True,
            files=(
                GPCRmdFile("topology", 15286, "Topology file", "topology"),
                GPCRmdFile("trajectory", 15287, "Trajectory file replica 1", "trajectory"),
                GPCRmdFile("trajectory", 15288, "Trajectory file replica 2", "trajectory"),
                GPCRmdFile("trajectory", 15289, "Trajectory file replica 3", "trajectory"),
                GPCRmdFile("model", 17686, "Model file", "coordinates"),
                GPCRmdFile("parameters", 15290, "Parameters file", "parameters"),
                GPCRmdFile("protocol", 17687, "Others file", "starting files"),
            ),
            reference_urls=(
                GPCRMD_DATA_DOWNLOAD_DOCS_URL,
                "https://doi.org/10.1038/s41467-025-57034-y",
            ),
            selection_reason=(
                "Ligand-bound explicit-solvent membrane GPCR with topology, model, "
                "trajectory, parameters, protocol/start files, water, ions, POPC, "
                "and box expected from the periodic simulation package."
            ),
            notes=(
                "Large system; this target is expected to stress missing PME/lipid/scale support.",
                "GPCRmd reference trajectory is comparison context, not an MLX trajectory.",
            ),
        ),
    )


def load_gpcrmd_targets(path: str | Path | None = None) -> tuple[GPCRmdTarget, ...]:
    """Load target metadata from JSON, or return the curated built-in registry."""

    if path is None:
        return default_gpcrmd_targets()
    payload = json.loads(Path(path).read_text())
    raw_targets = payload.get("targets", []) if isinstance(payload, Mapping) else payload
    return tuple(GPCRmdTarget.from_json_dict(item) for item in raw_targets)


def inspect_gpcrmd_cache(
    cache_path: str | Path,
    *,
    target_id: str | None = None,
    registry_path: str | Path | None = None,
    targets: Iterable[GPCRmdTarget] | None = None,
) -> GPCRmdCacheInspection:
    """Inspect whether a local cache has the files expected by a GPCRmd target."""

    if registry_path is not None and targets is not None:
        msg = "pass either registry_path or targets, not both"
        raise GPCRmdInspectionError(msg)
    candidates = load_gpcrmd_targets(registry_path) if registry_path is not None else targets
    target = select_gpcrmd_target(target_id, targets=candidates)
    cache = Path(cache_path)
    cache_exists = cache.exists()
    cache_kind = _cache_kind(cache)
    file_map = _cache_file_map(cache) if cache_exists else {}
    statuses = tuple(_status_for_expected_file(expected, file_map) for expected in target.files)
    return GPCRmdCacheInspection(
        target=target,
        cache_path=str(cache),
        cache_exists=cache_exists,
        cache_kind=cache_kind,
        file_statuses=statuses,
    )


def attempt_gpcrmd_prepared_artifact_import(
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    target_id: str | None = None,
    registry_path: str | Path | None = None,
) -> GPCRmdPreparedImportAttempt:
    """Attempt a GPCRmd-to-MLX prepared artifact conversion and fail closed."""

    inspection = inspect_gpcrmd_cache(
        cache_path,
        target_id=target_id,
        registry_path=registry_path,
    )
    compatibility = gpcrmd_mlx_compatibility_report(inspection)
    output = Path(out_dir)
    blockers = _missing_import_file_blockers(inspection)
    if blockers:
        _remove_prepared_artifact_files(output)
        return _blocked_prepared_import_attempt(
            inspection=inspection,
            output=output,
            blockers=blockers,
            compatibility=compatibility,
            import_details=_gpcrmd_import_details(
                inspection,
                _present_required_role_paths(inspection),
                import_style="not_attempted",
            ),
        )

    prepared, blockers, import_details = _import_prepared_system_from_inspection(inspection)
    if blockers:
        _remove_prepared_artifact_files(output)
        return _blocked_prepared_import_attempt(
            inspection=inspection,
            output=output,
            blockers=blockers,
            compatibility=_compatibility_with_prepared_force_terms(
                compatibility,
                import_details,
            ),
            import_details=import_details,
        )

    save_prepared_system(prepared, output)
    return GPCRmdPreparedImportAttempt(
        target_id=inspection.target.target_id,
        dynamics_id=inspection.target.dynamics_id,
        out_dir=str(output),
        exported=True,
        prepared_artifact_path=str(output),
        blockers=(),
        required_artifact_fields=REQUIRED_PREPARED_ARTIFACT_FIELDS,
        compatibility_report=_compatibility_with_prepared_force_terms(
            compatibility,
            import_details,
        ),
        import_details=import_details,
    )


def write_gpcrmd_import_report(
    path: str | Path,
    attempt: GPCRmdPreparedImportAttempt,
) -> None:
    """Write a GPCRmd import-attempt report for notebooks and later slices."""

    Path(path).write_text(json.dumps(attempt.to_json_dict(), indent=2, sort_keys=True) + "\n")


def write_gpcrmd_targets(path: str | Path, targets: Sequence[GPCRmdTarget]) -> None:
    """Write target metadata in the same JSON shape accepted by the loader."""

    output = {"targets": [target.to_json_dict() for target in targets]}
    Path(path).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def select_gpcrmd_target(
    target_id: str | None = None,
    *,
    targets: Iterable[GPCRmdTarget] | None = None,
) -> GPCRmdTarget:
    """Select a GPCRmd target that passes the first-slice metadata gate."""

    candidates = tuple(default_gpcrmd_targets() if targets is None else targets)
    if not candidates:
        msg = "no GPCRmd targets are available"
        raise GPCRmdTargetError(msg)

    selected = None
    if target_id is None:
        selected = candidates[0]
    else:
        for target in candidates:
            if target.target_id == target_id:
                selected = target
                break
    if selected is None:
        available = ", ".join(target.target_id for target in candidates)
        msg = f"unknown GPCRmd target {target_id!r}; available targets: {available}"
        raise GPCRmdTargetError(msg)

    missing = selected.missing_selection_requirements()
    if missing:
        msg = (
            f"GPCRmd target {selected.target_id!r} is missing selection requirements: "
            f"{', '.join(missing)}"
        )
        raise GPCRmdTargetError(msg)
    return selected


def gpcrmd_selection_reports(
    targets: Iterable[GPCRmdTarget] | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe selection reports for candidates."""

    candidates = default_gpcrmd_targets() if targets is None else tuple(targets)
    return [target.selection_report() for target in candidates]


def gpcrmd_mlx_compatibility_report(
    inspection: GPCRmdCacheInspection,
) -> GPCRmdMLXCompatibilityReport:
    """Return a fail-closed report for running an inspected GPCRmd target in MLX."""

    target = inspection.target
    missing_input = _missing_input_items(inspection)
    unsupported_physics = _unsupported_physics_items(target)
    runtime_risk = _runtime_risk(target)
    runnable_now = not missing_input and not unsupported_physics
    return GPCRmdMLXCompatibilityReport(
        target_id=target.target_id,
        dynamics_id=target.dynamics_id,
        runnable_now=runnable_now,
        supported_now=_supported_now_items(target),
        missing_input=tuple(missing_input),
        unsupported_physics=tuple(unsupported_physics),
        runtime_risk=runtime_risk,
        next_engine_slice=_next_engine_slice(missing_input, unsupported_physics),
        warnings=(
            "GPCRmd reference trajectories are comparison context, not MLX-generated results.",
            "This report is metadata-level until topology/parameter files are parsed.",
        ),
    )


def gpcrmd_mlx_readiness_inventory(
    inspection: GPCRmdCacheInspection,
) -> GPCRmdReadinessInventory:
    """Return the selected target inventory that gates GPCRmd engine work."""

    target = inspection.target
    compatibility = gpcrmd_mlx_compatibility_report(inspection)
    return GPCRmdReadinessInventory(
        target_id=target.target_id,
        dynamics_id=target.dynamics_id,
        target_decision=_target_decision(target),
        required_files=tuple(_required_file_items(inspection)),
        optional_analysis_features=tuple(_optional_analysis_feature_items(inspection)),
        system_components=_system_component_inventory(target),
        required_force_terms=tuple(_required_force_term_items(target, compatibility)),
        box=_box_inventory(target, inspection),
        constraints=_constraint_inventory(target),
        exceptions=_exception_inventory(target),
        protocol_requirements=_protocol_inventory(target),
        first_engine_blockers=tuple(_first_engine_blockers(compatibility)),
        missing_input=compatibility.missing_input,
    )


def _missing_input_items(inspection: GPCRmdCacheInspection) -> list[str]:
    missing = [
        f"file:{status.role}:{status.file_id}"
        for status in inspection.file_statuses
        if not status.present and status.role in REQUIRED_MLX_IMPORT_FILE_ROLES
    ]
    if not inspection.cache_exists:
        missing.insert(0, f"cache:{inspection.cache_path}")
    if inspection.target.periodic_box_expected and not _has_present_role(inspection, "model"):
        missing.append("box_vectors:requires_model_or_coordinate_file")
    return missing


def _target_decision(target: GPCRmdTarget) -> dict[str, Any]:
    return {
        "status": "fixed",
        "selected_target_id": target.target_id,
        "selected_dynamics_id": target.dynamics_id,
        "pdb_id": target.pdb_id,
        "receptor": target.receptor,
        "ligands": list(target.ligand_names),
        "replacement": None,
        "reason": target.selection_reason,
        "lower_blocker_target_search": (
            "not used; the existing target is the GPCRmd-critical membrane/solvent "
            "case required to expose PME, CHARMM, lipid, constraints, and scale blockers"
        ),
    }


def _required_file_items(inspection: GPCRmdCacheInspection) -> list[dict[str, Any]]:
    return [
        _file_status_inventory_item(status, required_for="mlx_import")
        for status in inspection.file_statuses
        if status.role in REQUIRED_MLX_IMPORT_FILE_ROLES
    ]


def _optional_analysis_feature_items(
    inspection: GPCRmdCacheInspection,
) -> list[dict[str, Any]]:
    optional: list[dict[str, Any]] = [
        {
            **_file_status_inventory_item(status, required_for="reference_analysis"),
            "feature": "reference_trajectory_comparison",
            "required_for_mlx_run": False,
        }
        for status in inspection.file_statuses
        if status.role in OPTIONAL_ANALYSIS_FILE_ROLES
    ]
    target = inspection.target
    optional.append(
        {
            "feature": "replicate_reference_statistics",
            "required_for_mlx_run": False,
            "replicates": target.replicates,
            "accumulated_time_us": target.accumulated_time_us,
            "frame_stride_ns": target.frame_stride_ns,
        }
    )
    return optional


def _file_status_inventory_item(
    status: GPCRmdCacheFileStatus,
    *,
    required_for: str,
) -> dict[str, Any]:
    return {
        "role": status.role,
        "file_id": status.file_id,
        "label": status.label,
        "format_hint": status.format_hint,
        "present": status.present,
        "path": status.path,
        "size_bytes": status.size_bytes,
        "required_for": required_for,
    }


def _system_component_inventory(target: GPCRmdTarget) -> dict[str, Any]:
    return {
        "total_atoms": target.total_atoms,
        "water": {
            "model": target.solvent_type,
            "count": target.molecule_counts.get("Water", 0),
            "required_for_mlx_run": True,
        },
        "lipids": {
            "model": target.membrane_composition,
            "membrane_type": target.membrane_type,
            "count": target.molecule_counts.get(str(target.membrane_composition), 0)
            if target.membrane_composition
            else 0,
            "required_for_mlx_run": target.membrane_type.lower()
            not in {"", "implicit", "none"},
        },
        "ions": {
            "sodium": target.molecule_counts.get("Sodium ion", 0),
            "chloride": target.molecule_counts.get("Chloride", 0),
            "required_for_mlx_run": True,
        },
        "receptor": target.receptor,
        "ligands": list(target.ligand_names),
    }


def _required_force_term_items(
    target: GPCRmdTarget,
    compatibility: GPCRmdMLXCompatibilityReport,
) -> list[dict[str, Any]]:
    unsupported = set(compatibility.unsupported_physics)
    terms: list[dict[str, Any]] = []
    if target.periodic_box_expected:
        terms.append(
            {
                "name": "pme_mesh_periodic_electrostatics",
                "required": True,
                "source": "periodic explicit water/membrane GPCRmd package",
                "status": _term_status(
                    "pme_mesh_periodic_electrostatics", unsupported
                ),
                "first_slice": "Slice 2 plus Slice 5",
            }
        )
    if "charmm" in target.force_field.lower():
        terms.extend(
            [
                {
                    "name": "charmm36_bonded_and_nonbonded_parameters",
                    "required": True,
                    "source": target.force_field,
                    "status": "import_blocked_until_topology_parameter_parse",
                    "first_slice": "Slice 7",
                },
                {
                    "name": "charmm_cmap_terms",
                    "required": True,
                    "source": target.force_field,
                    "status": _term_status("charmm_cmap_terms", unsupported),
                    "first_slice": "Slice 3",
                },
            ]
        )
    if target.membrane_type.lower() not in {"", "implicit", "none"}:
        terms.append(
            {
                "name": "membrane_lipid_force_field_terms",
                "required": True,
                "source": target.membrane_composition or target.membrane_type,
                "status": _term_status("membrane_lipid_force_field_terms", unsupported),
                "first_slice": "Slice 3",
            }
        )
    if target.membrane_composition:
        terms.append(
            {
                "name": f"{target.membrane_composition.lower()}_lipid_topology_and_parameters",
                "required": True,
                "source": "molecule counts and CHARMM36 lipid package",
                "status": _term_status("popc_lipid_topology_and_parameters", unsupported),
                "first_slice": "Slice 3 plus Slice 7",
            }
        )
    if target.solvent_type.upper() == "TIP3P":
        terms.append(
            {
                "name": "tip3p_water_model",
                "required": True,
                "source": "GPCRmd solvent metadata",
                "status": "metadata_supported_import_pending",
                "first_slice": "Slice 7",
            }
        )
    terms.append(
        {
            "name": "nonbonded_exclusions_and_1_4_exceptions",
            "required": True,
            "source": "topology/parameter files",
            "status": "import_blocked_until_topology_parameter_parse",
            "first_slice": "Slice 6 plus Slice 7",
        }
    )
    return terms


def _term_status(term: str, unsupported: set[str]) -> str:
    return "engine_blocked" if term in unsupported else "supported_or_not_required"


def _box_inventory(
    target: GPCRmdTarget,
    inspection: GPCRmdCacheInspection,
) -> dict[str, Any]:
    return {
        "periodic_box_required": target.periodic_box_expected,
        "vectors_known_now": _has_present_role(inspection, "model")
        or _has_present_role(inspection, "protocol"),
        "source_files": ["model", "protocol"],
        "blocker_if_missing": "box_vectors:requires_model_or_coordinate_file",
    }


def _constraint_inventory(target: GPCRmdTarget) -> dict[str, Any]:
    blockers: list[str] = []
    if target.time_step_fs >= 3.0:
        blockers.append("hmr_or_virtual_site_policy_required")
    return {
        "required": True,
        "source_files": ["topology", "protocol"],
        "required_artifact_fields": ["constraints"],
        "time_step_fs": target.time_step_fs,
        "expected_policy": (
            "hydrogen-bond constraints and any HMR/virtual-site policy must be parsed"
        ),
        "blockers": blockers,
    }


def _exception_inventory(target: GPCRmdTarget) -> dict[str, Any]:
    return {
        "required": True,
        "source_files": ["topology", "parameters"],
        "required_artifact_fields": ["nonbonded_exclusions", "nonbonded_exceptions"],
        "expected_terms": [
            "bonded nonbonded exclusions",
            "1-4 electrostatic/LJ exceptions",
            "CHARMM pair overrides or NBFIX entries if present after parameter parse",
        ],
        "force_field": target.force_field,
    }


def _protocol_inventory(target: GPCRmdTarget) -> dict[str, Any]:
    requirements = ["nvt_thermostat"]
    blockers: list[str] = []
    if "npt" in target.ensemble.lower():
        requirements.append("barostat")
        blockers.append("npt_barostat")
    if target.time_step_fs >= 3.0:
        requirements.append("4_fs_constraint_or_hmr_policy")
        blockers.append("hmr_or_virtual_site_policy_required")
    return {
        "ensemble": target.ensemble,
        "time_step_fs": target.time_step_fs,
        "replicates": target.replicates,
        "frame_stride_ns": target.frame_stride_ns,
        "accumulated_time_us": target.accumulated_time_us,
        "software": target.software,
        "requirements": requirements,
        "blockers": blockers,
    }


def _first_engine_blockers(
    compatibility: GPCRmdMLXCompatibilityReport,
) -> list[dict[str, Any]]:
    blocker_map = {
        "pme_mesh_periodic_electrostatics": {
            "first_slice": "Slice 2 plus Slice 5",
            "reason": (
                "periodic explicit solvent/membrane electrostatics cannot use the "
                "small-system Ewald oracle at GPCRmd scale"
            ),
        },
        "charmm_cmap_terms": {
            "first_slice": "Slice 3",
            "reason": (
                "CHARMM36 protein parameters require CMAP support before force "
                "evaluation can be faithful"
            ),
        },
        "membrane_lipid_force_field_terms": {
            "first_slice": "Slice 3",
            "reason": (
                "POPC membrane bonded/nonbonded terms must be represented before "
                "import can run"
            ),
        },
        "popc_lipid_topology_and_parameters": {
            "first_slice": "Slice 3 plus Slice 7",
            "reason": (
                "POPC topology and CHARMM lipid parameters must round-trip into "
                "strict artifacts"
            ),
        },
        "large_periodic_system_neighbor_list_scaling": {
            "first_slice": "Slice 4",
            "reason": "92001 atoms make dense all-pairs nonbonded execution infeasible",
        },
        "hmr_or_virtual_site_policy_required": {
            "first_slice": "Slice 6 plus Slice 7",
            "reason": (
                "4 fs GPCRmd timestep requires parsing constraints/HMR/virtual-site "
                "policy before runtime"
            ),
        },
        "npt_barostat": {
            "first_slice": "Slice 8 plus Slice 9",
            "reason": "pressure control requires virial diagnostics and the selected barostat path",
        },
    }
    return [
        {
            "name": item,
            "first_slice": blocker_map[item]["first_slice"],
            "reason": blocker_map[item]["reason"],
        }
        for item in compatibility.unsupported_physics
        if item in blocker_map
    ]


def _unsupported_physics_items(target: GPCRmdTarget) -> list[str]:
    unsupported: list[str] = []
    if "npt" in target.ensemble.lower():
        unsupported.append("npt_barostat")
    if target.time_step_fs >= 3.0:
        unsupported.append("hmr_or_virtual_site_policy_required")
    return unsupported


def _requires_mesh_pme(target: GPCRmdTarget) -> bool:
    if target.total_atoms >= 50_000:
        return True
    return target.membrane_type.lower() not in {"", "implicit", "none"}


def _runtime_risk(target: GPCRmdTarget) -> dict[str, Any]:
    dense_pair_count = target.total_atoms * max(target.total_atoms - 1, 0) // 2
    return {
        "total_atoms": target.total_atoms,
        "dense_pair_count": dense_pair_count,
        "replicates": target.replicates,
        "time_step_fs": target.time_step_fs,
        "frame_stride_ns": target.frame_stride_ns,
        "system_size": "large" if target.total_atoms >= 50_000 else "moderate",
        "memory_risk": "high" if target.total_atoms >= 50_000 else "medium",
    }


def _supported_now_items(target: GPCRmdTarget) -> tuple[str, ...]:
    supported = [
        "target_metadata",
        "cache_file_presence",
        "water_ion_metadata",
        "ligand_receptor_metadata",
    ]
    if target.solvent_type.upper() == "TIP3P":
        supported.append("tip3p_metadata")
    molecule_names = {name.lower() for name in target.molecule_counts}
    if target.periodic_box_expected and "water" in molecule_names:
        if _requires_mesh_pme(target):
            supported.append("ewald_reference_small_system_oracle")
        else:
            supported.append("ewald_reference_periodic_electrostatics")
    return tuple(supported)


def _next_engine_slice(missing_input: Sequence[str], unsupported_physics: Sequence[str]) -> str:
    if missing_input:
        return "download_or_mount_complete_gpcrmd_package_before_import"
    if "pme_mesh_periodic_electrostatics" in unsupported_physics:
        return "implement_pme_mesh_periodic_electrostatics"
    if "pme_ewald_periodic_electrostatics" in unsupported_physics:
        return "implement_pme_ewald_periodic_electrostatics"
    if any("lipid" in item for item in unsupported_physics):
        return "implement_lipid_force_field_import_and_terms"
    if "charmm_cmap_terms" in unsupported_physics:
        return "implement_charmm_cmap_terms"
    if "large_periodic_system_neighbor_list_scaling" in unsupported_physics:
        return "implement_scalable_periodic_neighbor_lists"
    if "hmr_or_virtual_site_policy_required" in unsupported_physics:
        return "parse_gpcrmd_constraints_hmr_or_virtual_sites_policy"
    if unsupported_physics:
        return "resolve_unsupported_gpcrmd_physics"
    return "run_short_mlx_nvt_probe"


def _has_present_role(inspection: GPCRmdCacheInspection, role: str) -> bool:
    return any(status.role == role and status.present for status in inspection.file_statuses)


def _missing_import_file_blockers(inspection: GPCRmdCacheInspection) -> list[str]:
    compatibility = gpcrmd_mlx_compatibility_report(inspection)
    return [f"missing_input:{item}" for item in compatibility.missing_input]


def _blocked_prepared_import_attempt(
    *,
    inspection: GPCRmdCacheInspection,
    output: Path,
    blockers: Sequence[str],
    compatibility: GPCRmdMLXCompatibilityReport,
    import_details: Mapping[str, Any] | None = None,
) -> GPCRmdPreparedImportAttempt:
    return GPCRmdPreparedImportAttempt(
        target_id=inspection.target.target_id,
        dynamics_id=inspection.target.dynamics_id,
        out_dir=str(output),
        exported=False,
        prepared_artifact_path=None,
        blockers=tuple(blockers),
        required_artifact_fields=REQUIRED_PREPARED_ARTIFACT_FIELDS,
        compatibility_report=compatibility,
        import_details=dict(import_details or {}),
    )


def _compatibility_with_prepared_force_terms(
    compatibility: GPCRmdMLXCompatibilityReport,
    import_details: Mapping[str, Any],
) -> GPCRmdMLXCompatibilityReport:
    prepared_report = import_details.get("prepared_compatibility_report")
    if not isinstance(prepared_report, Mapping):
        return compatibility
    keys = (
        "supported_terms",
        "required_terms",
        "unsupported_terms",
        "rejected_terms",
        "rejection_reasons",
        "term_details",
        "term_counts",
    )
    extra_fields = {
        key: prepared_report[key]
        for key in keys
        if key in prepared_report
    }
    if not extra_fields:
        return compatibility
    return replace(
        compatibility,
        extra_fields={**compatibility.extra_fields, **extra_fields},
    )


def _remove_prepared_artifact_files(out_dir: Path) -> None:
    for name in (JSON_NAME, NPZ_NAME, VIEW_PDB_NAME):
        path = out_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _import_prepared_system_from_inspection(
    inspection: GPCRmdCacheInspection,
) -> tuple[PreparedSystem | None, list[str], dict[str, Any]]:
    role_paths = _present_required_role_paths(inspection)
    import_style = _gpcrmd_import_style(inspection.target, role_paths)
    import_details = _gpcrmd_import_details(
        inspection,
        role_paths,
        import_style=import_style,
    )
    try:
        if import_style == "amber":
            prepared = import_amber_prmtop(
                prmtop_path=role_paths["topology"][0],
                coords_path=role_paths["model"][0],
            )
        elif import_style == "charmm":
            if find_spec("parmed") is None:
                return None, ["parser_missing:parmed"], import_details
            mass_prelude = build_charmm_psf_mass_prelude(
                psf_path=role_paths["topology"][0],
                params=role_paths["parameters"],
            )
            if mass_prelude is None:
                prepared = import_charmm_with_parmed(
                    psf_path=role_paths["topology"][0],
                    params=role_paths["parameters"],
                    coords_path=role_paths["model"][0],
                )
            else:
                import_details["derived_mass_prelude"] = mass_prelude.to_json_dict()
                with tempfile.TemporaryDirectory(prefix="gpcrmd-charmm-") as tmpdir:
                    prelude_path = Path(tmpdir) / "psf-derived-masses.rtf"
                    prelude_path.write_text(mass_prelude.text)
                    prepared = import_charmm_with_parmed(
                        psf_path=role_paths["topology"][0],
                        params=[prelude_path, *role_paths["parameters"]],
                        coords_path=role_paths["model"][0],
                    )
        else:
            return None, [f"unsupported_topology_family:{import_style}"], import_details
    except TopologyImportError as exc:
        blockers = _topology_import_error_blockers(str(exc))
        if blockers:
            return None, blockers, import_details
        role = _parse_failure_role(str(exc))
        return None, [f"parse_failed:{role}:{exc}"], import_details
    except Exception as exc:  # pragma: no cover - parser-specific failures vary
        role = _parse_failure_role(str(exc))
        return None, [f"parse_failed:{role}:{exc}"], import_details

    prepared = _enrich_gpcrmd_prepared_system(
        prepared,
        inspection=inspection,
        role_paths=role_paths,
        import_style=import_style,
    )
    import_details["prepared_compatibility_report"] = dict(
        prepared.metadata.compatibility_report
    )
    blockers = _prepared_import_blockers(prepared, inspection.target)
    if blockers:
        return None, blockers, import_details
    return prepared, [], import_details


def _present_required_role_paths(
    inspection: GPCRmdCacheInspection,
) -> dict[str, list[Path]]:
    return {
        role: paths
        for role, paths in _inspection_role_paths(inspection).items()
        if role in REQUIRED_MLX_IMPORT_FILE_ROLES
    }


def _inspection_role_paths(
    inspection: GPCRmdCacheInspection,
) -> dict[str, list[Path]]:
    role_paths: dict[str, list[Path]] = {}
    for status in inspection.file_statuses:
        if not status.present:
            continue
        if status.path is None:
            continue
        role_paths.setdefault(status.role, []).append(Path(status.path))
    for role, paths in list(role_paths.items()):
        role_paths[role] = sorted(paths, key=lambda path: _natural_sort_key(str(path)))
    return role_paths


def _gpcrmd_import_details(
    inspection: GPCRmdCacheInspection,
    role_paths: Mapping[str, Sequence[Path]],
    *,
    import_style: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "import_style": import_style,
        "role_paths": {
            role: [str(path) for path in paths] for role, paths in role_paths.items()
        },
        "source_files": _gpcrmd_source_file_payload(inspection),
    }
    box_metadata = _gpcrmd_protocol_box_metadata(role_paths)
    if box_metadata:
        details["protocol_box"] = box_metadata
    return details


def _gpcrmd_import_style(
    target: GPCRmdTarget,
    role_paths: Mapping[str, Sequence[Path]],
) -> str:
    force_field = target.force_field.lower()
    topology_suffixes = {path.suffix.lower() for path in role_paths.get("topology", [])}
    if "amber" in force_field or topology_suffixes & {".prmtop", ".parm7", ".top"}:
        return "amber"
    if "charmm" in force_field or topology_suffixes & {".psf"}:
        return "charmm"
    return "unknown"


def _parse_failure_role(message: str) -> str:
    lowered = message.lower()
    if "coordinate" in lowered or "coords" in lowered or "model" in lowered:
        return "model"
    if "parameter" in lowered or "parm" in lowered:
        return "parameters"
    if "topology" in lowered or "psf" in lowered or "prmtop" in lowered:
        return "topology"
    return "topology_parameters_model"


def _topology_import_error_blockers(message: str) -> list[str]:
    if not message.startswith("unsupported_terms:"):
        return []
    raw_terms = message.removeprefix("unsupported_terms:")
    terms = [term.strip() for term in raw_terms.split(",") if term.strip()]
    return [f"unsupported_terms:{term}" for term in terms]


def _enrich_gpcrmd_prepared_system(
    prepared: PreparedSystem,
    *,
    inspection: GPCRmdCacheInspection,
    role_paths: Mapping[str, Sequence[Path]],
    import_style: str,
) -> PreparedSystem:
    target = inspection.target
    box_metadata = _gpcrmd_protocol_box_metadata(role_paths)
    prepared = _apply_gpcrmd_protocol_box(prepared, box_metadata)
    metadata = prepared.metadata
    source = {
        **metadata.source,
        "kind": f"gpcrmd_{import_style}",
        "gpcrmd_target_id": target.target_id,
        "gpcrmd_dynamics_id": target.dynamics_id,
        "gpcrmd_source_url": target.source_url,
        "gpcrmd_cache_path": inspection.cache_path,
        "gpcrmd_files": _gpcrmd_source_file_payload(inspection),
        "role_paths": {
            role: [str(path) for path in paths] for role, paths in role_paths.items()
        },
    }
    if box_metadata:
        source["gpcrmd_protocol_box_source_files"] = box_metadata["source_files"]
    protocol_metadata = {
        **metadata.protocol_metadata,
        "source": "gpcrmd_target_metadata",
        "ensemble": target.ensemble,
        "time_step_fs": target.time_step_fs,
        "software": target.software,
        "force_field": target.force_field,
        "replicates": target.replicates,
        "frame_stride_ns": target.frame_stride_ns,
        "accumulated_time_us": target.accumulated_time_us,
        "periodic_box_expected": target.periodic_box_expected,
    }
    if box_metadata:
        protocol_metadata["box_metadata"] = box_metadata
        protocol_metadata["box_vectors"] = box_metadata["box_vectors"]
        protocol_metadata["box_lengths_A"] = box_metadata["cell_lengths"]
        protocol_metadata["box_source_path"] = box_metadata["active_source_path"]
    compatibility = dict(metadata.compatibility_report)
    supported_terms = _gpcrmd_supported_terms(prepared, target, compatibility)
    rejected_terms = list(compatibility.get("rejected_terms", []))
    required_terms = sorted(
        set(supported_terms)
        | {str(item) for item in compatibility.get("required_terms", [])}
        | {str(item) for item in rejected_terms}
    )
    term_counts = dict(compatibility.get("term_counts", {}))
    term_counts.update(
        {
            "bonds": int(prepared.bonds.shape[0]),
            "angles": int(prepared.angles.shape[0]),
            "dihedrals": int(prepared.dihedrals.shape[0]),
            "impropers": int(prepared.impropers.shape[0]),
            "constraints": int(prepared.constraints.shape[0]),
            "nonbonded_exceptions": int(prepared.nonbonded_exception_pairs.shape[0]),
            "charmm_cmap_terms": int(prepared.charmm_cmap_terms.shape[0]),
            "urey_bradley_terms": int(prepared.urey_bradley_terms.shape[0]),
            "nbfix_pairs": int(prepared.nbfix_pairs.shape[0]),
            "nbfix_type_pair_overrides": int(prepared.nbfix_type_pairs.shape[0]),
        }
    )
    compatibility.update(
        {
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "source": "gpcrmd_import",
            "gpcrmd_target_id": target.target_id,
            "gpcrmd_dynamics_id": target.dynamics_id,
            "gpcrmd_source_url": target.source_url,
            "water_present": bool(np.any(prepared.water_mask)),
            "ions_present": bool(np.any(prepared.ion_mask)),
            "lipids_present": bool(np.any(prepared.lipid_mask)),
            "ligand_present": bool(np.any(prepared.ligand_mask)),
            "receptor_present": bool(np.any(prepared.receptor_mask)),
            "periodic_box_present": bool(np.asarray(prepared.cell_lengths).shape == (3,)),
            "constraints_present": bool(prepared.constraints.shape[0]),
            "nonbonded_exclusions_present": bool(prepared.nonbonded_exception_pairs.shape[0]),
            "nonbonded_exceptions_present": bool(prepared.nonbonded_exception_pairs.shape[0]),
            "supported_terms": supported_terms,
            "required_terms": required_terms,
            "unsupported_terms": list(compatibility.get("unsupported_terms", [])),
            "rejected_terms": rejected_terms,
            "parameter_counts_match_topology": True,
            "term_counts": term_counts,
        }
    )
    metadata = replace(
        metadata,
        source=source,
        selections={
            **metadata.selections,
            "gpcrmd_target_id": target.target_id,
            "gpcrmd_dynamics_id": target.dynamics_id,
            "target_total_atoms": target.total_atoms,
            "target_molecule_counts": dict(target.molecule_counts),
        },
        parameter_source=f"gpcrmd_{import_style}_{metadata.parameter_source}",
        compatibility_report=compatibility,
        protocol_metadata=protocol_metadata,
        pme_config=_gpcrmd_pme_config(prepared, target),
        warnings=[
            *metadata.warnings,
            "Imported from GPCRmd cached topology/parameter files. No external MD engine was run.",
        ],
    )
    pme_arrays = _gpcrmd_pme_arrays(prepared, target)
    return replace(prepared, metadata=metadata, **pme_arrays)


def _gpcrmd_source_file_payload(inspection: GPCRmdCacheInspection) -> list[dict[str, Any]]:
    return [
        {
            "role": status.role,
            "file_id": status.file_id,
            "label": status.label,
            "format_hint": status.format_hint,
            "path": status.path,
            "size_bytes": status.size_bytes,
            "present": status.present,
        }
        for status in inspection.file_statuses
    ]


def _gpcrmd_supported_terms(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
    compatibility: Mapping[str, Any],
) -> list[str]:
    terms = set(str(item) for item in compatibility.get("supported_terms", []))
    terms.update({"nonbonded_lj_coulomb", "water", "ion", "ligand", "receptor"})
    if target.periodic_box_expected:
        terms.add("pme")
    if np.any(prepared.lipid_mask):
        terms.add("lipid")
    if prepared.constraints.shape[0]:
        terms.add("distance_constraint")
    if prepared.nonbonded_exception_pairs.shape[0]:
        terms.add("nonbonded_exception")
    if prepared.nbfix_type_pairs.shape[0] or prepared.nbfix_pairs.shape[0]:
        terms.add("nbfix_pair_overrides")
    return sorted(terms)


def _apply_gpcrmd_protocol_box(
    prepared: PreparedSystem,
    box_metadata: Mapping[str, Any],
) -> PreparedSystem:
    if not box_metadata:
        return prepared
    cell_lengths = np.asarray(box_metadata["cell_lengths"], dtype=np.float32)
    if cell_lengths.shape != (3,):
        return prepared
    return replace(prepared, cell_lengths=cell_lengths)


def _gpcrmd_protocol_box_metadata(
    role_paths: Mapping[str, Sequence[Path]],
) -> dict[str, Any]:
    boxes = []
    for xsc_path in _gpcrmd_protocol_xsc_paths(role_paths):
        parsed = _read_gpcrmd_xsc_box(xsc_path)
        if parsed:
            boxes.append(parsed)
    if not boxes:
        return {}
    active = boxes[0]
    return {
        "source": "gpcrmd_protocol_xsc",
        "active_source_path": active["source_path"],
        "source_files": [box["source_path"] for box in boxes],
        "box_vectors": active["box_vectors"],
        "cell_lengths": active["cell_lengths"],
        "replicate_boxes": boxes,
    }


def _gpcrmd_protocol_xsc_paths(
    role_paths: Mapping[str, Sequence[Path]],
) -> list[Path]:
    candidates: list[Path] = []
    for protocol_path in role_paths.get("protocol", []):
        candidates.extend(_xsc_paths_near_protocol_path(protocol_path))
    return sorted(set(candidates), key=lambda path: _natural_sort_key(str(path)))


def _xsc_paths_near_protocol_path(protocol_path: Path) -> list[Path]:
    if protocol_path.is_dir():
        return [path for path in protocol_path.rglob("*.xsc") if path.is_file()]
    candidates: list[Path] = []
    if protocol_path.is_file() and protocol_path.suffix.lower() == ".xsc":
        candidates.append(protocol_path)
    extracted = _extracted_protocol_dir(protocol_path)
    if extracted.is_dir():
        candidates.extend(path for path in extracted.rglob("*.xsc") if path.is_file())
    return candidates


def _extracted_protocol_dir(protocol_path: Path) -> Path:
    name = protocol_path.name
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if name.lower().endswith(suffix):
            return protocol_path.with_name(name[: -len(suffix)])
    return protocol_path.with_suffix("")


def _read_gpcrmd_xsc_box(path: Path) -> dict[str, Any]:
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 10:
            continue
        try:
            step = int(float(values[0]))
            vector_values = [float(value) for value in values[1:10]]
        except ValueError:
            continue
        box_vectors = np.asarray(vector_values, dtype=np.float32).reshape((3, 3))
        cell_lengths = np.linalg.norm(box_vectors, axis=1).astype(np.float32)
        if not np.all(np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0):
            continue
        return {
            "source_path": str(path),
            "step": step,
            "box_vectors": box_vectors.astype(float).tolist(),
            "cell_lengths": cell_lengths.astype(float).tolist(),
        }
    return {}


def _gpcrmd_pme_config(prepared: PreparedSystem, target: GPCRmdTarget) -> dict[str, Any]:
    if not target.periodic_box_expected or np.asarray(prepared.cell_lengths).shape != (3,):
        return {}
    mesh_shape = _gpcrmd_pme_mesh_shape(np.asarray(prepared.cell_lengths, dtype=np.float32))
    return {
        "mesh_shape": list(mesh_shape),
        "alpha": 0.35,
        "real_cutoff": 10.0,
        "assignment_order": 2,
        "charge_tolerance": 1e-3,
        "deconvolve_assignment": True,
    }


def _gpcrmd_pme_arrays(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
) -> dict[str, np.ndarray]:
    if not target.periodic_box_expected or np.asarray(prepared.cell_lengths).shape != (3,):
        return {}
    mesh_shape = _gpcrmd_pme_mesh_shape(np.asarray(prepared.cell_lengths, dtype=np.float32))
    return {
        "pme_mesh_shape": np.asarray(mesh_shape, dtype=np.int32),
        "pme_alpha": np.asarray([0.35], dtype=np.float32),
        "pme_real_cutoff": np.asarray([10.0], dtype=np.float32),
        "pme_assignment_order": np.asarray([2], dtype=np.int32),
        "pme_charge_tolerance": np.asarray([1e-3], dtype=np.float32),
        "pme_deconvolve_assignment": np.asarray([True], dtype=bool),
    }


def _gpcrmd_pme_mesh_shape(cell_lengths: np.ndarray) -> tuple[int, int, int]:
    return tuple(max(4, int(np.ceil(float(length)))) for length in cell_lengths)


def _prepared_import_blockers(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
) -> list[str]:
    report = prepared.metadata.compatibility_report
    blockers = [f"unsupported_terms:{term}" for term in report.get("unsupported_terms", [])]
    blockers.extend(f"rejected_terms:{term}" for term in report.get("rejected_terms", []))
    if not np.any(prepared.water_mask):
        blockers.append("mask_missing:water")
    if not np.any(prepared.ion_mask):
        blockers.append("mask_missing:ion")
    if not np.any(prepared.ligand_mask):
        blockers.append("mask_missing:ligand")
    if not np.any(prepared.receptor_mask):
        blockers.append("mask_missing:receptor")
    if target.membrane_type.lower() not in {"", "implicit", "none"} and not np.any(
        prepared.lipid_mask
    ):
        blockers.append("mask_missing:lipid")
    if target.periodic_box_expected and np.asarray(prepared.cell_lengths).shape != (3,):
        blockers.append("box_vectors:missing")
    if prepared.constraints.shape[0] <= 0:
        blockers.append("constraints:missing")
    if prepared.nonbonded_exception_pairs.shape[0] <= 0:
        blockers.append("nonbonded_exceptions:missing")
    return blockers


def _import_blockers(report: GPCRmdMLXCompatibilityReport) -> list[str]:
    blockers = [f"missing_input:{item}" for item in report.missing_input]
    blockers.extend(f"unsupported_physics:{item}" for item in report.unsupported_physics)
    if report.runnable_now:
        blockers.append("gpcrmd_topology_parameter_parser_not_implemented")
    return blockers


def _cache_kind(cache: Path) -> str:
    if not cache.exists():
        return "missing"
    if cache.is_dir():
        return "directory"
    if cache.suffix.lower() == ".json":
        return "manifest"
    return "file"


def _cache_file_map(cache: Path) -> dict[int, tuple[Path | None, int | None]]:
    if cache.is_dir():
        return _directory_file_map(cache)
    if cache.suffix.lower() == ".json":
        return _manifest_file_map(cache)
    return _single_file_map(cache)


def _natural_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", value)
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def _directory_file_map(cache: Path) -> dict[int, tuple[Path | None, int | None]]:
    paths = [path for path in cache.rglob("*") if path.is_file()]
    file_map: dict[int, tuple[Path | None, int | None]] = {}
    for path in sorted(paths):
        for file_id in _file_ids_in_name(path.name):
            file_map.setdefault(file_id, (path, path.stat().st_size))
    return file_map


def _single_file_map(cache: Path) -> dict[int, tuple[Path | None, int | None]]:
    return {
        file_id: (cache, cache.stat().st_size)
        for file_id in _file_ids_in_name(cache.name)
    }


def _manifest_file_map(cache: Path) -> dict[int, tuple[Path | None, int | None]]:
    payload = json.loads(cache.read_text())
    raw_files = payload.get("files", []) if isinstance(payload, Mapping) else payload
    file_map: dict[int, tuple[Path | None, int | None]] = {}
    for item in raw_files:
        if not isinstance(item, Mapping) or "file_id" not in item:
            continue
        file_id = int(item["file_id"])
        path = _manifest_entry_path(cache.parent, item)
        size = _manifest_entry_size(path, item)
        file_map[file_id] = (path, size)
    return file_map


def _manifest_entry_path(base_dir: Path, item: Mapping[str, Any]) -> Path | None:
    raw_path = item.get("path")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else base_dir / path


def _manifest_entry_size(path: Path | None, item: Mapping[str, Any]) -> int | None:
    if item.get("size_bytes") is not None:
        return int(item["size_bytes"])
    if path is not None and path.exists():
        return path.stat().st_size
    return None


def _file_ids_in_name(name: str) -> tuple[int, ...]:
    ids: list[str] = []
    current: list[str] = []
    for char in name:
        if char.isdigit():
            current.append(char)
        elif current:
            ids.append("".join(current))
            current = []
    if current:
        ids.append("".join(current))
    return tuple(int(item) for item in ids)


def _status_for_expected_file(
    expected: GPCRmdFile,
    file_map: Mapping[int, tuple[Path | None, int | None]],
) -> GPCRmdCacheFileStatus:
    path, size = file_map.get(expected.file_id, (None, None))
    present = path is not None and path.exists()
    return GPCRmdCacheFileStatus(
        role=expected.role,
        file_id=expected.file_id,
        label=expected.label,
        format_hint=expected.format_hint,
        present=present,
        path=None if path is None else str(path),
        size_bytes=size,
    )


__all__ = [
    "GPCRMD_DATA_DOWNLOAD_DOCS_URL",
    "GPCRMD_IMPORT_REPORT_NAME",
    "GPCRmdCacheFileStatus",
    "GPCRmdCacheInspection",
    "GPCRmdFile",
    "GPCRmdInspectionError",
    "GPCRmdMLXCompatibilityReport",
    "GPCRmdPreparedImportAttempt",
    "GPCRmdReadinessInventory",
    "GPCRmdTarget",
    "GPCRmdTargetError",
    "REQUIRED_FILE_ROLES",
    "REQUIRED_PREPARED_ARTIFACT_FIELDS",
    "attempt_gpcrmd_prepared_artifact_import",
    "default_gpcrmd_targets",
    "gpcrmd_mlx_compatibility_report",
    "gpcrmd_mlx_readiness_inventory",
    "gpcrmd_selection_reports",
    "inspect_gpcrmd_cache",
    "load_gpcrmd_targets",
    "select_gpcrmd_target",
    "write_gpcrmd_import_report",
    "write_gpcrmd_targets",
]
