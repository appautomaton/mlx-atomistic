"""GPCRmd target registry and selection gates.

This module records source-backed GPCRmd candidate metadata without downloading
large trajectory packages or invoking external MD engines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.hmr import apply_hydrogen_mass_repartitioning
from mlx_atomistic.prep.io import (
    JSON_NAME,
    NPZ_NAME,
    VIEW_PDB_NAME,
    load_prepared_system,
    save_prepared_system,
)
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.prep.topology_import import (
    TopologyImportError,
    import_amber_prmtop,
    import_charmm_psf,
    import_gromacs_top_gro,
)

GPCRMD_DATA_DOWNLOAD_DOCS_URL = "https://gpcrmd-docs.readthedocs.io/en/latest/data-download.html"
GPCRMD_API_DOCS_URL = "https://gpcrmd-docs.readthedocs.io/en/latest/api.html"
GPCRMD_DYNAMICS_METADATA_URL_TEMPLATE = (
    "https://www.gpcrmd.org/api/search_dyn/info/{dynamics_id}"
)
GPCRMD_FILE_DOWNLOAD_REQUIRES_ACCOUNT = True
GPCRMD_IMPORT_REPORT_NAME = "gpcrmd_import_report.json"
GPCRMD_WORKLOAD_MANIFEST_NAME = "mlx-workload-manifest.json"
ACEMD_BINARY_VELOCITY_TO_ANGSTROM_PER_PS = 20.45482706
GPCRMD_PME_ASSIGNMENT_ORDER = 5
GPCRMD_NET_CHARGE_TOLERANCE_E = 1.0e-4

GPCRMD_729_SOURCE_BASELINE = {
    "atom_count": 92_001,
    "water_atom_count": 59_832,
    "ion_atom_count": 131,
    "lipid_atom_count": 26_800,
    "ligand_atom_count": 43,
    "receptor_atom_count": 5_195,
    "bonds": 91_734,
    "angles": 80_726,
    "source_proper_dihedrals": 85_210,
    "runtime_proper_dihedrals": 109_071,
    "harmonic_impropers": 1_214,
    "urey_bradley_terms": 49_223,
    "charmm_cmap_terms": 317,
    "constraints": 78_896,
    "nonbonded_exclusions": 152_516,
    "charmm_14_exceptions": 84_967,
    "nonbonded_exceptions": 237_483,
    "source_nbfix_overrides": 95,
    "applicable_nbfix_overrides": 5,
    "hydrogen_count": 58_952,
    "box_lengths_A": (87.17032, 87.15242, 118.5805),
    "temperature_K": 310.0,
    "langevin_friction_ps^-1": 0.1,
    "time_step_fs": 4.0,
    "cutoff_A": 9.0,
    "switch_distance_A": 7.5,
    "ewald_error_tolerance": 5.0e-4,
    "constraint_tolerance": 1.0e-6,
    "hmr_target_hydrogen_mass": 4.032,
    "trajectory_interval_steps": 50_000,
    "pme_mesh_shape": (78, 78, 106),
    "pme_alpha_per_A": 0.2920289872,
}

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
    filename_hint: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "file_id": self.file_id,
            "label": self.label,
            "format_hint": self.format_hint,
            "filename_hint": self.filename_hint,
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
            filename_hint=(
                None
                if payload.get("filename_hint") is None
                else str(payload["filename_hint"])
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
                GPCRmdFile(
                    "topology",
                    15286,
                    "Topology file",
                    "topology",
                    "15286_dyn_729.psf",
                ),
                GPCRmdFile("trajectory", 15287, "Trajectory file replica 1", "trajectory"),
                GPCRmdFile("trajectory", 15288, "Trajectory file replica 2", "trajectory"),
                GPCRmdFile("trajectory", 15289, "Trajectory file replica 3", "trajectory"),
                GPCRmdFile(
                    "model",
                    17686,
                    "Model file",
                    "coordinates",
                    "17686_dyn_729.pdb",
                ),
                GPCRmdFile(
                    "parameters",
                    15290,
                    "Parameters file",
                    "parameters",
                    "15290_prm_729.prm",
                ),
                GPCRmdFile(
                    "protocol",
                    17687,
                    "Others file",
                    "starting files",
                    "17687_oth_729",
                ),
            ),
            reference_urls=(
                GPCRMD_DATA_DOWNLOAD_DOCS_URL,
                GPCRMD_API_DOCS_URL,
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
                "GPCRmd file downloads require an authenticated account; metadata remains public.",
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
    source_manifest_path: str | Path | None = None,
) -> GPCRmdPreparedImportAttempt:
    """Attempt a GPCRmd-to-MLX prepared artifact conversion and fail closed."""

    inspection = inspect_gpcrmd_cache(
        cache_path,
        target_id=target_id,
        registry_path=registry_path,
    )
    compatibility = gpcrmd_mlx_compatibility_report(inspection)
    output = Path(out_dir)
    try:
        source_manifest = _validated_source_manifest(
            source_manifest_path,
            inspection=inspection,
            cache_path=Path(cache_path),
        )
    except GPCRmdInspectionError as exc:
        _remove_prepared_artifact_files(output)
        return _blocked_prepared_import_attempt(
            inspection=inspection,
            output=output,
            blockers=[f"artifact_source:{exc}"],
            compatibility=compatibility,
            import_details={"source_manifest_path": str(source_manifest_path)},
        )
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
                source_manifest=source_manifest,
            ),
        )

    prepared, blockers, import_details = _import_prepared_system_from_inspection(
        inspection,
        source_manifest=source_manifest,
    )
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

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(attempt.to_json_dict(), indent=2, sort_keys=True) + "\n")


def prepare_gpcrmd_artifact(
    *,
    cache_path: str | Path,
    out_dir: str | Path,
    report_path: str | Path,
    target_id: str | None = None,
    registry_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare, strictly load, and describe one GPCRmd MLX artifact.

    Args:
        cache_path: Caller-owned GPCRmd input cache.
        out_dir: Output directory for the artifact and workload manifest.
        report_path: JSON path for the preparation result.
        target_id: Optional target identifier; defaults to the curated target.
        registry_path: Optional custom target registry.
        source_manifest_path: Acquisition manifest whose hashes and protocol
            members are authoritative.

    Returns:
        JSON-serializable preparation report. A successful report includes a
        strict artifact-v3 load result and workload-manifest path.
    """

    output = Path(out_dir)
    workload_path = output / GPCRMD_WORKLOAD_MANIFEST_NAME
    if workload_path.is_file() or workload_path.is_symlink():
        workload_path.unlink()
    attempt = attempt_gpcrmd_prepared_artifact_import(
        cache_path,
        output,
        target_id=target_id,
        registry_path=registry_path,
        source_manifest_path=source_manifest_path,
    )
    strict_load: dict[str, Any] = {"status": "not_attempted"}
    if attempt.exported:
        try:
            from mlx_atomistic.artifacts import load_prepared_mlx_artifact

            artifact = load_prepared_mlx_artifact(output, require_production=True)
            prepared = load_prepared_system(output)
            target = select_gpcrmd_target(
                attempt.target_id,
                targets=(
                    load_gpcrmd_targets(registry_path)
                    if registry_path is not None
                    else None
                ),
            )
            workload = build_gpcrmd_mlx_workload_manifest(
                prepared,
                target=target,
                artifact_dir=output,
                source_manifest_path=source_manifest_path,
            )
            workload_path.write_text(
                json.dumps(workload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            strict_load = {
                "status": "ready",
                "require_production": True,
                "artifact_version": int(artifact.metadata["artifact_version"]),
                "atom_count": artifact.atom_count,
            }
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            _remove_prepared_artifact_files(output)
            if workload_path.exists():
                workload_path.unlink()
            attempt = replace(
                attempt,
                exported=False,
                prepared_artifact_path=None,
                blockers=(f"strict_production_load:{exc}",),
            )
            strict_load = {
                "status": "blocked",
                "require_production": True,
                "blocker": str(exc),
            }

    payload = {
        "kind": "gpcrmd_preparation",
        "status": "prepared" if attempt.exported else "blocked",
        **attempt.to_json_dict(),
        "source_manifest_path": (
            None if source_manifest_path is None else str(source_manifest_path)
        ),
        "strict_production_load": strict_load,
        "workload_manifest_path": str(workload_path) if attempt.exported else None,
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_gpcrmd_mlx_workload_manifest(
    prepared: PreparedSystem,
    *,
    target: GPCRmdTarget,
    artifact_dir: str | Path,
    source_manifest_path: str | Path | None,
) -> dict[str, Any]:
    """Build the canonical MLX workload manifest for GPCRmd parity and runtime.

    Args:
        prepared: Source-faithful prepared system after protocol start-state and HMR.
        target: GPCRmd target metadata.
        artifact_dir: Directory containing the saved schema-v3 artifact.
        source_manifest_path: Authoritative acquisition manifest path.

    Returns:
        JSON-compatible manifest with source, particle, force, protocol, and PME
        hashes and counts. The final ``manifest_sha256`` excludes itself.
    """

    prepared.validate()
    artifact_root = Path(artifact_dir)
    metadata = prepared.metadata
    report = dict(metadata.compatibility_report)
    source_counts = dict(report.get("source_topology_counts", {}))
    term_details = dict(report.get("term_details", {}))
    nbfix_details = dict(term_details.get("nbfix_pair_overrides", {}))
    exception_details = dict(term_details.get("nonbonded_exception", {}))
    protocol = dict(metadata.protocol_metadata)
    nonbonded = dict(protocol.get("nonbonded", {}))
    pme_source = dict(protocol.get("pme", {}))
    constraints_source = dict(protocol.get("constraints", {}))
    hmr = _gpcrmd_hmr_manifest(protocol.get("hydrogen_mass_repartitioning", {}))
    source_net_charge = float(
        metadata.selections.get(
            "system_charge_source_precision",
            metadata.selections.get("system_charge", np.sum(prepared.charges)),
        )
    )
    stored_net_charge = float(np.sum(prepared.charges, dtype=np.float64))

    particle_hashes = {
        "symbols": gpcrmd_array_hash(prepared.symbols, dtype=str),
        "atom_names": gpcrmd_array_hash(prepared.atom_names, dtype=str),
        "atom_types": gpcrmd_array_hash(prepared.atom_types, dtype=str),
        "residue_names": gpcrmd_array_hash(prepared.residue_names, dtype=str),
        "residue_ids": gpcrmd_array_hash(prepared.residue_ids, dtype=np.int32),
        "chain_ids": gpcrmd_array_hash(prepared.chain_ids, dtype=str),
    }
    force_array_names = (
        "bonds",
        "bond_k",
        "bond_length",
        "angles",
        "angle_k",
        "angle_theta",
        "dihedrals",
        "dihedral_k",
        "dihedral_periodicity",
        "dihedral_phase",
        "impropers",
        "improper_k",
        "improper_periodicity",
        "improper_phase",
        "urey_bradley_terms",
        "urey_bradley_k",
        "urey_bradley_distance",
        "charmm_cmap_terms",
        "charmm_cmap_grid_indices",
        "charmm_cmap_grids",
        "nbfix_pairs",
        "nbfix_sigma",
        "nbfix_epsilon",
        "nbfix_type_pairs",
        "nbfix_type_sigma",
        "nbfix_type_epsilon",
    )
    force_hashes = {
        name: gpcrmd_array_hash(getattr(prepared, name)) for name in force_array_names
    }
    exception_parameter_hash = _gpcrmd_payload_hash(
        {
            "charge_product": gpcrmd_array_hash(
                prepared.nonbonded_exception_charge_product,
                dtype=np.float32,
            ),
            "sigma": gpcrmd_array_hash(
                prepared.nonbonded_exception_sigma,
                dtype=np.float32,
            ),
            "epsilon": gpcrmd_array_hash(
                prepared.nonbonded_exception_epsilon,
                dtype=np.float32,
            ),
        }
    )
    pme_config = dict(metadata.pme_config)
    pme_arrays = {
        "mesh_shape": [int(value) for value in prepared.pme_mesh_shape.tolist()],
        "alpha_per_angstrom": float(prepared.pme_alpha[0]),
        "real_cutoff_angstrom": float(prepared.pme_real_cutoff[0]),
        "assignment_order": int(prepared.pme_assignment_order[0]),
        "charge_tolerance_e": float(prepared.pme_charge_tolerance[0]),
        "deconvolve_assignment": bool(prepared.pme_deconvolve_assignment[0]),
        "background_policy": str(prepared.pme_background_policy[0]),
    }
    artifact_files = {
        name: {
            "path": str(artifact_root / name),
            "size_bytes": (artifact_root / name).stat().st_size,
            "sha256": _sha256_file(artifact_root / name),
        }
        for name in (JSON_NAME, NPZ_NAME, VIEW_PDB_NAME)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpcrmd_mlx_workload",
        "engine": {
            "name": "mlx_atomistic",
            "role": "product_runtime",
        },
        "workload": {
            "name": target.target_id,
            "dynamics_id": target.dynamics_id,
            "operation": "fixed_coordinate_parity_and_fixed_cell_nvt",
            "atom_count": prepared.atom_count,
            "ensemble": protocol.get("ensemble"),
            "fixed_cell": bool(protocol.get("fixed_cell", False)),
        },
        "source": {
            "manifest_path": (
                None if source_manifest_path is None else str(source_manifest_path)
            ),
            "manifest": metadata.source.get("gpcrmd_source_manifest"),
            "target_url": target.source_url,
            "parser": metadata.source.get("parser", "native_charmm_psf"),
            "parameter_source": metadata.parameter_source,
        },
        "artifact": {
            "directory": str(artifact_root),
            "artifact_version": int(metadata.artifact_version),
            "strict_production_load": True,
            "files": artifact_files,
        },
        "particles": {
            "count": prepared.atom_count,
            "atom_order_hash": _gpcrmd_payload_hash(particle_hashes),
            "identity_hashes": particle_hashes,
            "coordinate_hash": gpcrmd_array_hash(prepared.positions, dtype=np.float32),
            "velocity_hash": gpcrmd_array_hash(prepared.velocities, dtype=np.float32),
            "mass_hash": gpcrmd_array_hash(prepared.masses, dtype=np.float32),
            "artifact_mass_hash": gpcrmd_array_hash(prepared.masses),
            "artifact_mass_dtype": str(np.asarray(prepared.masses).dtype),
            "charge_hash": gpcrmd_array_hash(prepared.charges, dtype=np.float32),
            "lj_particle_hash": _gpcrmd_payload_hash(
                {
                    "sigma": gpcrmd_array_hash(prepared.sigma, dtype=np.float32),
                    "epsilon": gpcrmd_array_hash(prepared.epsilon, dtype=np.float32),
                }
            ),
            "source_net_charge_e": source_net_charge,
            "stored_float32_net_charge_e": stored_net_charge,
        },
        "selections": {
            "water_atom_count": int(np.count_nonzero(prepared.water_mask)),
            "ion_atom_count": int(np.count_nonzero(prepared.ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(prepared.lipid_mask)),
            "ligand_atom_count": int(np.count_nonzero(prepared.ligand_mask)),
            "receptor_atom_count": int(np.count_nonzero(prepared.receptor_mask)),
            "water_mask_hash": gpcrmd_array_hash(prepared.water_mask, dtype=bool),
            "ion_mask_hash": gpcrmd_array_hash(prepared.ion_mask, dtype=bool),
            "lipid_mask_hash": gpcrmd_array_hash(prepared.lipid_mask, dtype=bool),
            "ligand_mask_hash": gpcrmd_array_hash(prepared.ligand_mask, dtype=bool),
            "receptor_mask_hash": gpcrmd_array_hash(prepared.receptor_mask, dtype=bool),
        },
        "cell": {
            "lengths_angstrom": [float(value) for value in prepared.cell_lengths.tolist()],
            "matrix_angstrom": [
                [float(value) for value in row] for row in prepared.cell_matrix.tolist()
            ],
            "lengths_hash": gpcrmd_array_hash(
                prepared.cell_lengths,
                dtype=np.float32,
            ),
            "matrix_hash": gpcrmd_array_hash(prepared.cell_matrix, dtype=np.float32),
            "source": protocol.get("box_source_path"),
        },
        "topology": {
            "source_counts": source_counts,
            "runtime_counts": {
                "bonds": int(prepared.bonds.shape[0]),
                "angles": int(prepared.angles.shape[0]),
                "proper_dihedral_terms": int(prepared.dihedrals.shape[0]),
                "harmonic_impropers": int(
                    np.count_nonzero(np.asarray(prepared.improper_periodicity) == 0.0)
                ),
                "urey_bradley_terms": int(prepared.urey_bradley_terms.shape[0]),
                "charmm_cmap_terms": int(prepared.charmm_cmap_terms.shape[0]),
                "constraints": int(prepared.constraints.shape[0]),
                "nonbonded_exclusions": int(
                    exception_details.get("excluded_pair_count", -1)
                ),
                "charmm_14_exceptions": int(
                    exception_details.get("one_four_pair_count", -1)
                ),
                "nonbonded_exceptions": int(
                    prepared.nonbonded_exception_pairs.shape[0]
                ),
                "nbfix_applicable_type_pairs": int(prepared.nbfix_type_pairs.shape[0]),
            },
            "force_array_hashes": force_hashes,
            "exception_pairs_hash": gpcrmd_array_hash(
                prepared.nonbonded_exception_pairs,
                dtype=np.int32,
            ),
            "exception_parameter_hash": exception_parameter_hash,
            "nbfix": {
                "source_parameter_override_count": nbfix_details.get(
                    "source_parameter_override_count"
                ),
                "applicable_override_count": nbfix_details.get(
                    "applicable_override_count",
                    int(prepared.nbfix_type_pairs.shape[0]),
                ),
                "type_pair_hash": gpcrmd_array_hash(prepared.nbfix_type_pairs, dtype=str),
                "parameter_hash": _gpcrmd_payload_hash(
                    {
                        "sigma": gpcrmd_array_hash(
                            prepared.nbfix_type_sigma,
                            dtype=np.float32,
                        ),
                        "epsilon": gpcrmd_array_hash(
                            prepared.nbfix_type_epsilon,
                            dtype=np.float32,
                        ),
                    }
                ),
            },
        },
        "forces": {
            "supported_terms": list(report.get("supported_terms_normalized", [])),
            "required_terms": list(report.get("required_terms_normalized", [])),
            "term_counts": dict(report.get("term_counts_normalized", {})),
            "unsupported_terms": list(report.get("unsupported_terms_normalized", [])),
            "rejected_terms": list(report.get("rejected_terms_normalized", [])),
        },
        "constraints": {
            "count": int(prepared.constraints.shape[0]),
            "pairs_hash": gpcrmd_array_hash(prepared.constraints, dtype=np.int32),
            "distance_hash": gpcrmd_array_hash(
                prepared.constraint_distance,
                dtype=np.float32,
            ),
            "distance_source": constraints_source.get("geometry_source"),
            "tolerance": constraints_source.get("tolerance"),
            "source_constrained_x_h_bonds": constraints_source.get(
                "constrained_x_h_bonds"
            ),
            "source_rigid_water_count": constraints_source.get("rigid_water_count"),
        },
        "hydrogen_mass_repartitioning": hmr,
        "protocol": _gpcrmd_protocol_manifest(protocol),
        "nonbonded": {
            "cutoff_angstrom": nonbonded.get("cutoff"),
            "switching": nonbonded.get("switching"),
            "switch_distance_angstrom": nonbonded.get("switch_distance"),
            "exclusion_count": int(exception_details.get("excluded_pair_count", -1)),
            "one_four_exception_count": int(
                exception_details.get("one_four_pair_count", -1)
            ),
            "exception_count": int(prepared.nonbonded_exception_pairs.shape[0]),
            "nbxmod": exception_details.get("nbxmod"),
            "e14fac": exception_details.get("e14fac"),
            "electrostatic_14_scale": exception_details.get(
                "electrostatic_14_scale"
            ),
        },
        "pme": {
            **pme_arrays,
            "ewald_error_tolerance": pme_source.get("ewald_error_tolerance"),
            "derivation": pme_config.get("derivation"),
            "net_charge_policy": {
                "background_policy": pme_arrays["background_policy"],
                "charge_tolerance_e": pme_arrays["charge_tolerance_e"],
                "source_net_charge_e": source_net_charge,
            },
        },
        "runtime_contract": {
            "topology_pair_policy": "lazy",
            "eager_nonbonded_pair_limit": 0,
            "neighbor_backend": "mlx_cell_blocks",
            "neighbor_representation": "NeighborBlocks",
            "fixed_cell_pme_plan_reuse": True,
            "dense_or_tiled_fallback_allowed": False,
        },
    }
    manifest["manifest_sha256"] = _gpcrmd_payload_hash(manifest)
    return manifest


def gpcrmd_array_hash(values: Any, *, dtype: Any | None = None) -> str:
    """Return the canonical dtype-and-shape-aware hash for a workload array."""

    array = np.asarray(values if dtype is None else np.asarray(values, dtype=dtype))
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _gpcrmd_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gpcrmd_hmr_manifest(value: Any) -> dict[str, Any]:
    hmr = dict(value) if isinstance(value, Mapping) else {}
    policy = dict(hmr.get("policy", {}))
    original = np.asarray(hmr.get("original_masses", []), dtype=np.float32)
    transformed = np.asarray(hmr.get("transformed_masses", []), dtype=np.float32)
    selected = hmr.get("selected_hydrogens", [])
    heavy_atoms = hmr.get("heavy_atoms", [])
    return {
        "status": hmr.get("status"),
        "selection": policy.get("selection"),
        "target_hydrogen_mass_da": policy.get("target_hydrogen_mass"),
        "require_constraints": policy.get("require_constraints"),
        "virtual_sites_supported": policy.get("virtual_sites_supported"),
        "selected_hydrogen_count": len(selected) if isinstance(selected, list) else 0,
        "heavy_atom_count": len(heavy_atoms) if isinstance(heavy_atoms, list) else 0,
        "original_mass_hash": gpcrmd_array_hash(original, dtype=np.float32),
        "transformed_mass_hash": gpcrmd_array_hash(transformed, dtype=np.float32),
        "total_mass_before_da": hmr.get("total_mass_before"),
        "total_mass_after_da": hmr.get("total_mass_after"),
    }


def _gpcrmd_protocol_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": protocol.get("source"),
        "selected_replicate": protocol.get("selected_replicate"),
        "ensemble": protocol.get("ensemble"),
        "fixed_cell": protocol.get("fixed_cell"),
        "temperature_K": protocol.get("temperature_K"),
        "langevin_friction_ps^-1": protocol.get("langevin_friction_ps^-1"),
        "time_step_fs": protocol.get("time_step_fs"),
        "barostat_enabled": protocol.get("barostat_enabled"),
        "thermostat_enabled": protocol.get("thermostat_enabled"),
        "run_steps": protocol.get("run_steps"),
        "trajectory": dict(protocol.get("trajectory", {})),
        "starting_state": dict(protocol.get("starting_state", {})),
    }


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
        "supported_terms_normalized",
        "required_terms",
        "required_terms_normalized",
        "unsupported_terms",
        "unsupported_terms_normalized",
        "rejected_terms",
        "rejected_terms_normalized",
        "rejection_reasons",
        "term_details",
        "term_counts",
        "term_counts_normalized",
        "source_topology_counts",
    )
    extra_fields = {
        key: prepared_report[key]
        for key in keys
        if key in prepared_report
    }
    if not extra_fields:
        return compatibility
    unsupported_physics = list(compatibility.unsupported_physics)
    if (
        prepared_report.get("hydrogen_mass_repartitioning")
        == "represented_by_masses"
    ):
        unsupported_physics = [
            item
            for item in unsupported_physics
            if item != "hmr_or_virtual_site_policy_required"
        ]
    prepared_blockers = [
        *prepared_report.get("unsupported_terms", []),
        *prepared_report.get("rejected_terms", []),
        *prepared_report.get("blockers", []),
    ]
    runnable_now = (
        not compatibility.missing_input
        and not unsupported_physics
        and not prepared_blockers
    )
    supported_now = list(compatibility.supported_now)
    if runnable_now:
        supported_now.extend(
            [
                "native_topology_parameter_import",
                "source_protocol_semantics",
                "hmr_represented_by_masses",
                "strict_artifact_v3",
            ]
        )
    return replace(
        compatibility,
        runnable_now=runnable_now,
        supported_now=tuple(dict.fromkeys(supported_now)),
        unsupported_physics=tuple(unsupported_physics),
        next_engine_slice=(
            "ready_for_bounded_mlx_execution"
            if runnable_now
            else _next_engine_slice(compatibility.missing_input, unsupported_physics)
        ),
        warnings=(
            "GPCRmd reference trajectories are comparison context, not MLX-generated results.",
        ),
        extra_fields={**compatibility.extra_fields, **extra_fields},
    )


def _remove_prepared_artifact_files(out_dir: Path) -> None:
    for name in (JSON_NAME, NPZ_NAME, VIEW_PDB_NAME):
        path = out_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _validated_source_manifest(
    path: str | Path | None,
    *,
    inspection: GPCRmdCacheInspection,
    cache_path: Path,
) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"source_manifest_unreadable:{manifest_path}"
        raise GPCRmdInspectionError(msg) from exc
    if not isinstance(payload, Mapping):
        msg = "source_manifest_malformed"
        raise GPCRmdInspectionError(msg)
    if str(payload.get("status", "")) != "complete" or payload.get("blockers"):
        msg = "source_manifest_incomplete"
        raise GPCRmdInspectionError(msg)
    if str(payload.get("target_id", "")) != inspection.target.target_id:
        msg = "source_manifest_target_mismatch"
        raise GPCRmdInspectionError(msg)
    if int(payload.get("dynamics_id", -1)) != inspection.target.dynamics_id:
        msg = "source_manifest_dynamics_mismatch"
        raise GPCRmdInspectionError(msg)

    files = payload.get("files")
    if not isinstance(files, list):
        msg = "source_manifest_files_missing"
        raise GPCRmdInspectionError(msg)
    required_roles = set(REQUIRED_MLX_IMPORT_FILE_ROLES)
    seen_roles: set[str] = set()
    cache_root = cache_path.resolve() if cache_path.is_dir() else None
    for item in files:
        if not isinstance(item, Mapping):
            msg = "source_manifest_file_malformed"
            raise GPCRmdInspectionError(msg)
        role = str(item.get("role", ""))
        if role not in required_roles:
            continue
        source_path = _manifest_entry_path(manifest_path.parent, item)
        if source_path is None or not source_path.is_file():
            msg = f"source_manifest_file_missing:{role}"
            raise GPCRmdInspectionError(msg)
        if cache_root is not None and not source_path.resolve().is_relative_to(cache_root):
            msg = f"source_manifest_file_outside_cache:{role}"
            raise GPCRmdInspectionError(msg)
        _validate_manifest_file_integrity(source_path, item, role=role)
        seen_roles.add(role)
        for member in item.get("archive_members", []) or []:
            if not isinstance(member, Mapping) or str(member.get("kind", "")) != "file":
                continue
            member_path = _manifest_entry_path(manifest_path.parent, member)
            if member_path is None or not member_path.is_file():
                msg = f"source_manifest_archive_member_missing:{member.get('normalized_name', '')}"
                raise GPCRmdInspectionError(msg)
            if cache_root is not None and not member_path.resolve().is_relative_to(cache_root):
                msg = "source_manifest_archive_member_outside_cache"
                raise GPCRmdInspectionError(msg)
            _validate_manifest_file_integrity(member_path, member, role="protocol_member")
    missing = sorted(required_roles - seen_roles)
    if missing:
        msg = "source_manifest_roles_missing:" + ",".join(missing)
        raise GPCRmdInspectionError(msg)
    result = dict(payload)
    result["_manifest_path"] = str(manifest_path.resolve())
    return result


def _validate_manifest_file_integrity(
    path: Path,
    item: Mapping[str, Any],
    *,
    role: str,
) -> None:
    expected_size = item.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        msg = f"source_manifest_size_mismatch:{role}"
        raise GPCRmdInspectionError(msg)
    expected_hash = item.get("sha256")
    if expected_hash is not None and _sha256_file(path) != str(expected_hash):
        msg = f"source_manifest_hash_mismatch:{role}"
        raise GPCRmdInspectionError(msg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_role_paths(
    source_manifest: Mapping[str, Any],
) -> dict[str, list[Path]]:
    manifest_path = Path(str(source_manifest["_manifest_path"]))
    role_paths: dict[str, list[Path]] = {}
    for item in source_manifest.get("files", []):
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", ""))
        if role not in REQUIRED_MLX_IMPORT_FILE_ROLES:
            continue
        path = _manifest_entry_path(manifest_path.parent, item)
        if path is not None:
            role_paths.setdefault(role, []).append(path)
    return {
        role: sorted(paths, key=lambda value: _natural_sort_key(str(value)))
        for role, paths in role_paths.items()
    }


def _resolve_gpcrmd_protocol_bundle(
    role_paths: Mapping[str, Sequence[Path]],
    *,
    source_manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if source_manifest is not None:
        bundles = _source_manifest_protocol_bundles(source_manifest)
        if not bundles:
            msg = "protocol_archive_members_missing"
            raise GPCRmdInspectionError(msg)
    else:
        bundles = []
        for xsc_path in _gpcrmd_protocol_xsc_paths(role_paths):
            parent = xsc_path.parent
            required = {
                "input_path": parent / "input",
                "coordinates_path": parent / "input.coor",
                "velocities_path": parent / "input.vel",
                "xsc_path": xsc_path,
                "log_path": parent / "log.txt",
            }
            if all(path.is_file() for path in required.values()):
                bundles.append({"replicate": parent.name, **required})
    if not bundles:
        return None

    parsed: list[dict[str, Any]] = []
    for bundle in sorted(bundles, key=lambda item: _natural_sort_key(str(item["replicate"]))):
        settings = _parse_gpcrmd_protocol_settings(
            Path(bundle["input_path"]),
            Path(bundle["log_path"]),
        )
        box = _read_gpcrmd_xsc_box(Path(bundle["xsc_path"]))
        if not box:
            msg = f"protocol_box_unreadable:{bundle['xsc_path']}"
            raise GPCRmdInspectionError(msg)
        parsed.append({**bundle, "settings": settings, "box": box})

    signature = _gpcrmd_protocol_signature(parsed[0])
    for bundle in parsed[1:]:
        if _gpcrmd_protocol_signature(bundle) != signature:
            msg = "protocol_replicate_semantics_mismatch"
            raise GPCRmdInspectionError(msg)
    selected = dict(parsed[0])
    selected["replicate_boxes"] = [dict(bundle["box"]) for bundle in parsed]
    selected["replicates"] = [
        {
            "replicate": str(bundle["replicate"]),
            "input_path": str(bundle["input_path"]),
            "coordinates_path": str(bundle["coordinates_path"]),
            "velocities_path": str(bundle["velocities_path"]),
            "xsc_path": str(bundle["xsc_path"]),
            "log_path": str(bundle["log_path"]),
        }
        for bundle in parsed
    ]
    return selected


def _source_manifest_protocol_bundles(
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = Path(str(source_manifest["_manifest_path"]))
    members: list[Mapping[str, Any]] = []
    for item in source_manifest.get("files", []):
        if isinstance(item, Mapping) and str(item.get("role", "")) == "protocol":
            members.extend(
                member
                for member in item.get("archive_members", []) or []
                if isinstance(member, Mapping) and str(member.get("kind", "")) == "file"
            )
    grouped: dict[str, dict[str, Path]] = {}
    names = {
        "input": "input_path",
        "input.coor": "coordinates_path",
        "input.vel": "velocities_path",
        "input.xsc": "xsc_path",
        "log.txt": "log_path",
    }
    for member in members:
        normalized_name = str(member.get("normalized_name", "")).strip("/")
        relative = Path(normalized_name)
        key = names.get(relative.name)
        if key is None or len(relative.parts) < 2:
            continue
        path = _manifest_entry_path(manifest_path.parent, member)
        if path is None:
            continue
        grouped.setdefault(str(relative.parent), {})[key] = path
    required_keys = set(names.values())
    return [
        {"replicate": parent, **paths}
        for parent, paths in grouped.items()
        if set(paths) == required_keys
    ]


def _parse_gpcrmd_protocol_settings(input_path: Path, log_path: Path) -> dict[str, Any]:
    input_values: dict[str, str] = {}
    for raw_line in input_path.read_text(errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) >= 2:
            input_values[fields[0].lower()] = fields[1]
    log_text = log_path.read_text(errors="replace")

    timestep_fs = _required_protocol_float(input_values, "timestep")
    temperature_k = _required_protocol_float(input_values, "temperature")
    thermostat_temperature_k = _required_protocol_float(
        input_values,
        "thermostattemperature",
    )
    friction_ps = _required_protocol_float(input_values, "thermostatdamping")
    cutoff_a = _required_protocol_float(input_values, "cutoff")
    switch_distance_a = _required_protocol_float(input_values, "switchdistance")
    trajectory_interval_steps = _required_protocol_int(input_values, "trajectoryperiod")
    run_steps = _required_protocol_int(input_values, "run")
    thermostat_enabled = _required_protocol_bool(input_values, "thermostat")
    pme_enabled = _required_protocol_bool(input_values, "pme")
    switching_enabled = _required_protocol_bool(input_values, "switching")
    barostat_values = re.findall(
        r"(?im)^\s*(?:\$\s*)?barostat\s+(on|off)\b",
        log_text,
    )
    if not barostat_values:
        msg = "protocol_barostat_state_missing"
        raise GPCRmdInspectionError(msg)
    barostat_enabled = _parse_protocol_bool(barostat_values[0], key="barostat")
    ewald_tolerance = _required_log_float(log_text, r"Ewald tolerance:\s*([0-9.eE+-]+)")
    constraint_tolerance = _required_log_float(
        log_text,
        r"Constraint tolerance:\s*([0-9.eE+-]+)",
    )
    constrained_bond_count = _required_log_int(
        log_text,
        r"Number of constrained bonds:\s*(\d+)",
    )
    rigid_water_count = _required_log_int(
        log_text,
        r"Number of water molecules:\s*(\d+)",
    )
    constraint_counts = [
        int(value)
        for value in re.findall(r"Number of constraints:\s*(\d+)", log_text)
    ]
    if not constraint_counts or len(set(constraint_counts)) != 1:
        msg = "protocol_constraint_count_missing_or_inconsistent"
        raise GPCRmdInspectionError(msg)
    hmr_target_mass = _required_log_float(
        log_text,
        r"New hydrogen mass:\s*([0-9.eE+-]+)",
    )
    hmr_hydrogen_count = _required_log_int(
        log_text,
        r"Number of hydrogen atoms:\s*(\d+)",
    )
    log_friction = _required_log_float(
        log_text,
        r"Friction coefficient:\s*([0-9.eE+-]+)",
    )
    if not math.isclose(temperature_k, thermostat_temperature_k, rel_tol=0.0, abs_tol=1e-9):
        msg = "protocol_temperature_mismatch"
        raise GPCRmdInspectionError(msg)
    if not math.isclose(friction_ps, log_friction, rel_tol=0.0, abs_tol=1e-9):
        msg = "protocol_friction_mismatch"
        raise GPCRmdInspectionError(msg)
    if not thermostat_enabled or barostat_enabled:
        msg = "protocol_not_fixed_cell_nvt"
        raise GPCRmdInspectionError(msg)
    if not pme_enabled or not switching_enabled:
        msg = "protocol_pme_or_switching_disabled"
        raise GPCRmdInspectionError(msg)
    if switch_distance_a <= 0.0 or switch_distance_a >= cutoff_a:
        msg = "protocol_switch_distance_invalid"
        raise GPCRmdInspectionError(msg)
    return {
        "source": "gpcrmd_acemd_input_and_log",
        "ensemble": "NVT",
        "fixed_cell": True,
        "temperature_K": temperature_k,
        "langevin_friction_ps^-1": friction_ps,
        "time_step_fs": timestep_fs,
        "barostat_enabled": False,
        "thermostat_enabled": True,
        "run_steps": run_steps,
        "trajectory": {
            "file": input_values.get("trajectoryfile"),
            "interval_steps": trajectory_interval_steps,
            "interval_ps": trajectory_interval_steps * timestep_fs / 1000.0,
        },
        "nonbonded": {
            "cutoff": cutoff_a,
            "cutoff_unit": "angstrom",
            "switching": True,
            "switch_distance": switch_distance_a,
            "switch_distance_unit": "angstrom",
        },
        "pme": {
            "enabled": True,
            "ewald_error_tolerance": ewald_tolerance,
        },
        "constraints": {
            "count": constraint_counts[0],
            "constrained_x_h_bonds": constrained_bond_count,
            "rigid_water_count": rigid_water_count,
            "tolerance": constraint_tolerance,
            "geometry_source": "force_field_equilibrium_bonds",
        },
        "hydrogen_mass_repartitioning": {
            "target_hydrogen_mass": hmr_target_mass,
            "hydrogen_count": hmr_hydrogen_count,
        },
    }


def _required_protocol_float(values: Mapping[str, str], key: str) -> float:
    try:
        value = float(values[key])
    except (KeyError, ValueError) as exc:
        msg = f"protocol_field_missing_or_invalid:{key}"
        raise GPCRmdInspectionError(msg) from exc
    if not math.isfinite(value) or value <= 0.0:
        msg = f"protocol_field_missing_or_invalid:{key}"
        raise GPCRmdInspectionError(msg)
    return value


def _required_protocol_int(values: Mapping[str, str], key: str) -> int:
    value = _required_protocol_float(values, key)
    if int(value) != value:
        msg = f"protocol_field_missing_or_invalid:{key}"
        raise GPCRmdInspectionError(msg)
    return int(value)


def _required_protocol_bool(values: Mapping[str, str], key: str) -> bool:
    if key not in values:
        msg = f"protocol_field_missing_or_invalid:{key}"
        raise GPCRmdInspectionError(msg)
    return _parse_protocol_bool(values[key], key=key)


def _parse_protocol_bool(value: str, *, key: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"on", "true", "yes", "1"}:
        return True
    if normalized in {"off", "false", "no", "0"}:
        return False
    msg = f"protocol_field_missing_or_invalid:{key}"
    raise GPCRmdInspectionError(msg)


def _required_log_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        msg = f"protocol_log_field_missing:{pattern}"
        raise GPCRmdInspectionError(msg)
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0:
        msg = f"protocol_log_field_invalid:{pattern}"
        raise GPCRmdInspectionError(msg)
    return value


def _required_log_int(text: str, pattern: str) -> int:
    value = _required_log_float(text, pattern)
    if int(value) != value:
        msg = f"protocol_log_field_invalid:{pattern}"
        raise GPCRmdInspectionError(msg)
    return int(value)


def _gpcrmd_protocol_signature(bundle: Mapping[str, Any]) -> str:
    settings = bundle["settings"]
    box = bundle["box"]
    payload = {
        "settings": settings,
        "box_vectors": box["box_vectors"],
        "cell_lengths": box["cell_lengths"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_acemd_binary_vectors(
    path: Path,
    *,
    atom_count: int,
    role: str,
) -> np.ndarray:
    expected_size = 4 + atom_count * 3 * 8
    if path.stat().st_size != expected_size:
        msg = f"acemd_binary_size_mismatch:{role}"
        raise GPCRmdInspectionError(msg)
    raw = path.read_bytes()
    count_little = int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    count_big = int(np.frombuffer(raw, dtype=">i4", count=1)[0])
    if count_little == atom_count:
        dtype = "<f8"
    elif count_big == atom_count:
        dtype = ">f8"
    else:
        msg = f"acemd_binary_atom_count_mismatch:{role}"
        raise GPCRmdInspectionError(msg)
    values = np.frombuffer(raw, dtype=dtype, offset=4).reshape((atom_count, 3))
    if not np.all(np.isfinite(values)):
        msg = f"acemd_binary_nonfinite:{role}"
        raise GPCRmdInspectionError(msg)
    result = values.astype(np.float32)
    if role == "velocities":
        result *= np.float32(ACEMD_BINARY_VELOCITY_TO_ANGSTROM_PER_PS)
    result.setflags(write=False)
    return result


def _apply_gpcrmd_start_state(
    prepared: PreparedSystem,
    protocol_bundle: Mapping[str, Any] | None,
) -> PreparedSystem:
    if protocol_bundle is None:
        return prepared
    positions = _read_acemd_binary_vectors(
        Path(protocol_bundle["coordinates_path"]),
        atom_count=prepared.atom_count,
        role="coordinates",
    )
    velocities = _read_acemd_binary_vectors(
        Path(protocol_bundle["velocities_path"]),
        atom_count=prepared.atom_count,
        role="velocities",
    )
    return replace(
        prepared,
        positions=np.asarray(positions).copy(),
        velocities=np.asarray(velocities).copy(),
        reference_positions=np.asarray(positions).copy(),
    )


def _apply_gpcrmd_hmr(
    prepared: PreparedSystem,
    protocol_bundle: Mapping[str, Any] | None,
) -> PreparedSystem:
    if protocol_bundle is None:
        return prepared
    settings = dict(protocol_bundle["settings"])
    constraint_settings = dict(settings["constraints"])
    if prepared.constraints.shape[0] != int(constraint_settings["count"]):
        msg = (
            "constraint_count_mismatch:"
            f"parsed={prepared.constraints.shape[0]}:source={constraint_settings['count']}"
        )
        raise GPCRmdInspectionError(msg)
    hmr = dict(settings["hydrogen_mass_repartitioning"])
    hydrogen_indices = np.flatnonzero(
        np.char.upper(np.asarray(prepared.symbols, dtype=str)) == "H"
    )
    if hydrogen_indices.shape[0] != int(hmr["hydrogen_count"]):
        msg = (
            "hmr_hydrogen_count_mismatch:"
            f"parsed={hydrogen_indices.shape[0]}:source={hmr['hydrogen_count']}"
        )
        raise GPCRmdInspectionError(msg)
    return apply_hydrogen_mass_repartitioning(
        prepared,
        target_hydrogen_mass=float(hmr["target_hydrogen_mass"]),
        hydrogen_indices=hydrogen_indices.tolist(),
        selection="all_bonded_hydrogens",
        require_constraints=True,
    )


def _import_prepared_system_from_inspection(
    inspection: GPCRmdCacheInspection,
    *,
    source_manifest: Mapping[str, Any] | None = None,
) -> tuple[PreparedSystem | None, list[str], dict[str, Any]]:
    role_paths = _present_required_role_paths(inspection)
    if source_manifest is not None:
        role_paths = _source_manifest_role_paths(source_manifest)
    import_style = _gpcrmd_import_style(inspection.target, role_paths)
    try:
        protocol_bundle = _resolve_gpcrmd_protocol_bundle(
            role_paths,
            source_manifest=source_manifest,
        )
    except GPCRmdInspectionError as exc:
        import_details = _gpcrmd_import_details(
            inspection,
            role_paths,
            import_style=import_style,
            source_manifest=source_manifest,
        )
        return None, [f"protocol_semantics:{exc}"], import_details
    import_details = _gpcrmd_import_details(
        inspection,
        role_paths,
        import_style=import_style,
        source_manifest=source_manifest,
        protocol_bundle=protocol_bundle,
    )
    try:
        if import_style == "amber":
            prepared = import_amber_prmtop(
                prmtop_path=role_paths["topology"][0],
                coords_path=role_paths["model"][0],
            )
        elif import_style == "gromacs":
            prepared = import_gromacs_top_gro(
                top_path=role_paths["topology"][0],
                gro_path=role_paths["model"][0],
            )
        elif import_style == "charmm":
            prepared = import_charmm_psf(
                psf_path=role_paths["topology"][0],
                params=role_paths["parameters"],
                coords_path=role_paths["model"][0],
                box_path=(None if protocol_bundle is None else protocol_bundle["xsc_path"]),
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

    try:
        prepared = _apply_gpcrmd_start_state(prepared, protocol_bundle)
        prepared = _enrich_gpcrmd_prepared_system(
            prepared,
            inspection=inspection,
            role_paths=role_paths,
            import_style=import_style,
            source_manifest=source_manifest,
            protocol_bundle=protocol_bundle,
        )
        prepared = _apply_gpcrmd_hmr(prepared, protocol_bundle)
    except (GPCRmdInspectionError, TopologyImportError, ValueError) as exc:
        return None, [f"protocol_semantics:{exc}"], import_details
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
    source_manifest: Mapping[str, Any] | None = None,
    protocol_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "import_style": import_style,
        "role_paths": {
            role: [str(path) for path in paths] for role, paths in role_paths.items()
        },
        "source_files": _gpcrmd_source_file_payload(inspection),
    }
    box_metadata = _gpcrmd_protocol_box_metadata(role_paths)
    if protocol_bundle is not None:
        box_metadata = _protocol_bundle_box_metadata(protocol_bundle)
    if box_metadata:
        details["protocol_box"] = box_metadata
    if source_manifest is not None:
        details["source_manifest"] = _source_manifest_provenance(source_manifest)
    if protocol_bundle is not None:
        details["source_protocol"] = _protocol_bundle_provenance(protocol_bundle)
    return details


def _source_manifest_provenance(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(source_manifest["_manifest_path"]),
        "manifest_version": int(source_manifest.get("manifest_version", 0)),
        "status": str(source_manifest.get("status", "")),
        "target_id": str(source_manifest.get("target_id", "")),
        "dynamics_id": int(source_manifest.get("dynamics_id", -1)),
        "files": [
            {
                "file_id": int(item["file_id"]),
                "role": str(item["role"]),
                "path": str(item["path"]),
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
            for item in source_manifest.get("files", [])
            if isinstance(item, Mapping)
            and str(item.get("role", "")) in REQUIRED_MLX_IMPORT_FILE_ROLES
        ],
    }


def _protocol_bundle_provenance(protocol_bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_replicate": str(protocol_bundle["replicate"]),
        "input_path": str(protocol_bundle["input_path"]),
        "coordinates_path": str(protocol_bundle["coordinates_path"]),
        "velocities_path": str(protocol_bundle["velocities_path"]),
        "xsc_path": str(protocol_bundle["xsc_path"]),
        "log_path": str(protocol_bundle["log_path"]),
        "replicates": [dict(item) for item in protocol_bundle.get("replicates", [])],
        "settings": dict(protocol_bundle["settings"]),
    }


def _protocol_bundle_box_metadata(protocol_bundle: Mapping[str, Any]) -> dict[str, Any]:
    active = dict(protocol_bundle["box"])
    replicate_boxes = [
        dict(item) for item in protocol_bundle.get("replicate_boxes", [active])
    ]
    return {
        "source": "gpcrmd_protocol_xsc",
        "active_source_path": str(active["source_path"]),
        "source_files": [str(item["source_path"]) for item in replicate_boxes],
        "box_vectors": active["box_vectors"],
        "cell_lengths": active["cell_lengths"],
        "replicate_boxes": replicate_boxes,
    }


def _gpcrmd_import_style(
    target: GPCRmdTarget,
    role_paths: Mapping[str, Sequence[Path]],
) -> str:
    force_field = target.force_field.lower()
    topology_paths = list(role_paths.get("topology", []))
    topology_suffixes = {path.suffix.lower() for path in topology_paths}
    if "gromacs" in force_field or any(
        _looks_like_gromacs_topology(path) for path in topology_paths
    ):
        return "gromacs"
    if "amber" in force_field or topology_suffixes & {".prmtop", ".parm7"}:
        return "amber"
    if any(
        path.suffix.lower() == ".top" and _looks_like_amber_topology(path)
        for path in topology_paths
    ):
        return "amber"
    if "charmm" in force_field or topology_suffixes & {".psf"}:
        return "charmm"
    return "unknown"


def _looks_like_gromacs_topology(path: Path) -> bool:
    if path.suffix.lower() != ".top":
        return False
    try:
        text = path.read_text(errors="replace")[:65536].lower()
    except OSError:
        return False
    if "%flag" in text or "%version" in text:
        return False
    return (
        "[ defaults ]" in text
        or "[defaults]" in text
        or "[ moleculetype ]" in text
        or "[moleculetype]" in text
    )


def _looks_like_amber_topology(path: Path) -> bool:
    if path.suffix.lower() != ".top":
        return False
    try:
        text = path.read_text(errors="replace")[:65536].lower()
    except OSError:
        return False
    return "%flag" in text or "%version" in text


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
    source_manifest: Mapping[str, Any] | None = None,
    protocol_bundle: Mapping[str, Any] | None = None,
) -> PreparedSystem:
    target = inspection.target
    box_metadata = _gpcrmd_protocol_box_metadata(role_paths)
    if protocol_bundle is not None:
        box_metadata = _protocol_bundle_box_metadata(protocol_bundle)
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
    if source_manifest is not None:
        source["gpcrmd_source_manifest"] = _source_manifest_provenance(source_manifest)
    protocol_metadata = dict(metadata.protocol_metadata)
    protocol_metadata.update(
        {
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
    )
    if protocol_bundle is not None:
        source_settings = dict(protocol_bundle["settings"])
        protocol_metadata.update(source_settings)
        protocol_metadata["source"] = "gpcrmd_acemd_input_log_and_restart"
        protocol_metadata["selected_replicate"] = str(protocol_bundle["replicate"])
        protocol_metadata["source_files"] = _protocol_bundle_provenance(protocol_bundle)
        protocol_metadata["starting_state"] = {
            "coordinates": str(protocol_bundle["coordinates_path"]),
            "velocities": str(protocol_bundle["velocities_path"]),
            "coordinate_format": "acemd_namd_binary_float64",
            "coordinate_unit": "angstrom",
            "velocity_format": "acemd_namd_binary_float64",
            "velocity_conversion_to_angstrom_per_ps": (
                ACEMD_BINARY_VELOCITY_TO_ANGSTROM_PER_PS
            ),
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
            "constraints": int(prepared.constraints.shape[0]),
            "nonbonded_exceptions": int(prepared.nonbonded_exception_pairs.shape[0]),
            "charmm_cmap_terms": int(prepared.charmm_cmap_terms.shape[0]),
            "urey_bradley_terms": int(prepared.urey_bradley_terms.shape[0]),
            "nbfix_pair_overrides": max(
                int(term_counts.get("nbfix_pair_overrides", 0)),
                int(prepared.nbfix_pairs.shape[0] + prepared.nbfix_type_pairs.shape[0]),
            ),
        }
    )
    if prepared.impropers.shape[0]:
        harmonic_count = int(
            np.count_nonzero(np.asarray(prepared.improper_periodicity) == 0.0)
        )
        periodic_count = int(prepared.impropers.shape[0]) - harmonic_count
        term_counts.pop("impropers", None)
        if harmonic_count:
            term_counts["charmm_harmonic_improper"] = harmonic_count
        if periodic_count:
            term_counts["periodic_improper"] = periodic_count
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
            "water_atom_count": int(np.count_nonzero(prepared.water_mask)),
            "ion_atom_count": int(np.count_nonzero(prepared.ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(prepared.lipid_mask)),
            "ligand_atom_count": int(np.count_nonzero(prepared.ligand_mask)),
            "receptor_atom_count": int(np.count_nonzero(prepared.receptor_mask)),
        },
        parameter_source=f"gpcrmd_{import_style}_{metadata.parameter_source}",
        compatibility_report=compatibility,
        protocol_metadata=protocol_metadata,
        pme_config=_gpcrmd_pme_config(
            prepared,
            target,
            protocol_metadata=protocol_metadata,
        ),
        warnings=[
            *metadata.warnings,
            "Imported from GPCRmd cached topology/parameter files. No external MD engine was run.",
        ],
    )
    pme_arrays = _gpcrmd_pme_arrays(
        prepared,
        target,
        protocol_metadata=protocol_metadata,
    )
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
    cell_matrix = np.asarray(box_metadata.get("box_vectors", np.asarray([])), dtype=np.float32)
    if cell_matrix.shape != (3, 3):
        return replace(prepared, cell_lengths=cell_lengths)
    return replace(prepared, cell_lengths=cell_lengths, cell_matrix=cell_matrix)


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
        determinant = float(np.linalg.det(box_vectors.astype(np.float64)))
        if not np.isfinite(determinant) or determinant <= 0.0:
            continue
        return {
            "source_path": str(path),
            "step": step,
            "box_vectors": box_vectors.astype(float).tolist(),
            "cell_lengths": cell_lengths.astype(float).tolist(),
        }
    return {}


def _gpcrmd_pme_config(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
    *,
    protocol_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not target.periodic_box_expected or np.asarray(prepared.cell_lengths).shape != (3,):
        return {}
    nonbonded = dict(protocol_metadata.get("nonbonded", {}))
    pme = dict(protocol_metadata.get("pme", {}))
    cutoff = float(nonbonded.get("cutoff", 10.0))
    ewald_tolerance = float(pme.get("ewald_error_tolerance", 1.0e-3))
    mesh_shape, alpha = _derive_gpcrmd_pme_parameters(
        np.asarray(prepared.cell_lengths, dtype=np.float64),
        cutoff=cutoff,
        ewald_tolerance=ewald_tolerance,
    )
    source_net_charge = float(
        prepared.metadata.selections.get(
            "system_charge_source_precision",
            prepared.metadata.selections.get("system_charge", np.sum(prepared.charges)),
        )
    )
    if abs(source_net_charge) > 1.0e-8:
        msg = f"source_non_neutral_charge:{source_net_charge:.12g}"
        raise GPCRmdInspectionError(msg)
    return {
        "mesh_shape": list(mesh_shape),
        "alpha": alpha,
        "real_cutoff": cutoff,
        "assignment_order": GPCRMD_PME_ASSIGNMENT_ORDER,
        "charge_tolerance": GPCRMD_NET_CHARGE_TOLERANCE_E,
        "deconvolve_assignment": True,
        "background_policy": "reject_non_neutral",
        "ewald_error_tolerance": ewald_tolerance,
        "derivation": {
            "source": "OpenMM_PME_tolerance_mapping",
            "alpha_formula": "sqrt(-ln(2*tolerance))/real_cutoff",
            "mesh_formula": "ceil(2*alpha*box_length/(3*tolerance^0.2))",
            "assignment_order": "OpenMM cardinal B-spline order 5",
            "source_net_charge_e": source_net_charge,
        },
    }


def _gpcrmd_pme_arrays(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
    *,
    protocol_metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not target.periodic_box_expected or np.asarray(prepared.cell_lengths).shape != (3,):
        return {}
    config = _gpcrmd_pme_config(
        prepared,
        target,
        protocol_metadata=protocol_metadata,
    )
    return {
        "pme_mesh_shape": np.asarray(config["mesh_shape"], dtype=np.int32),
        "pme_alpha": np.asarray([config["alpha"]], dtype=np.float32),
        "pme_real_cutoff": np.asarray([config["real_cutoff"]], dtype=np.float32),
        "pme_assignment_order": np.asarray([config["assignment_order"]], dtype=np.int32),
        "pme_charge_tolerance": np.asarray([config["charge_tolerance"]], dtype=np.float32),
        "pme_deconvolve_assignment": np.asarray(
            [config["deconvolve_assignment"]],
            dtype=bool,
        ),
        "pme_background_policy": np.asarray([config["background_policy"]], dtype=str),
    }


def _derive_gpcrmd_pme_parameters(
    cell_lengths: np.ndarray,
    *,
    cutoff: float,
    ewald_tolerance: float,
) -> tuple[tuple[int, int, int], float]:
    lengths = np.asarray(cell_lengths, dtype=np.float64)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "invalid_cell_for_pme_derivation"
        raise GPCRmdInspectionError(msg)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        msg = "invalid_cutoff_for_pme_derivation"
        raise GPCRmdInspectionError(msg)
    if (
        not math.isfinite(ewald_tolerance)
        or ewald_tolerance <= 0.0
        or ewald_tolerance >= 0.5
    ):
        msg = "invalid_tolerance_for_pme_derivation"
        raise GPCRmdInspectionError(msg)
    alpha = math.sqrt(-math.log(2.0 * ewald_tolerance)) / cutoff
    scale = 2.0 * alpha / (3.0 * ewald_tolerance**0.2)
    mesh_shape = tuple(max(4, int(math.ceil(scale * float(length)))) for length in lengths)
    return mesh_shape, alpha


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
    blockers.extend(_official_gpcrmd_729_baseline_blockers(prepared, target))
    return blockers


def _official_gpcrmd_729_baseline_blockers(
    prepared: PreparedSystem,
    target: GPCRmdTarget,
) -> list[str]:
    if target.target_id != "gpcrmd-729-beta1-5f8u-cyanopindolol":
        return []

    expected = GPCRMD_729_SOURCE_BASELINE
    report = dict(prepared.metadata.compatibility_report)
    source_counts = dict(report.get("source_topology_counts", {}))
    term_details = dict(report.get("term_details", {}))
    nbfix_details = dict(term_details.get("nbfix_pair_overrides", {}))
    exception_details = dict(term_details.get("nonbonded_exception", {}))
    protocol = dict(prepared.metadata.protocol_metadata)
    nonbonded = dict(protocol.get("nonbonded", {}))
    source_pme = dict(protocol.get("pme", {}))
    source_constraints = dict(protocol.get("constraints", {}))
    hmr = dict(protocol.get("hydrogen_mass_repartitioning", {}))
    hmr_policy = dict(hmr.get("policy", {}))
    pme_config = dict(prepared.metadata.pme_config)

    counts = {
        "atom_count": prepared.atom_count,
        "water_atom_count": int(np.count_nonzero(prepared.water_mask)),
        "ion_atom_count": int(np.count_nonzero(prepared.ion_mask)),
        "lipid_atom_count": int(np.count_nonzero(prepared.lipid_mask)),
        "ligand_atom_count": int(np.count_nonzero(prepared.ligand_mask)),
        "receptor_atom_count": int(np.count_nonzero(prepared.receptor_mask)),
        "bonds": int(prepared.bonds.shape[0]),
        "angles": int(prepared.angles.shape[0]),
        "source_proper_dihedrals": int(source_counts.get("proper_dihedrals", -1)),
        "runtime_proper_dihedrals": int(prepared.dihedrals.shape[0]),
        "harmonic_impropers": int(
            np.count_nonzero(np.asarray(prepared.improper_periodicity) == 0.0)
        ),
        "urey_bradley_terms": int(prepared.urey_bradley_terms.shape[0]),
        "charmm_cmap_terms": int(prepared.charmm_cmap_terms.shape[0]),
        "constraints": int(prepared.constraints.shape[0]),
        "nonbonded_exclusions": int(
            exception_details.get("excluded_pair_count", -1)
        ),
        "charmm_14_exceptions": int(
            exception_details.get("one_four_pair_count", -1)
        ),
        "nonbonded_exceptions": int(prepared.nonbonded_exception_pairs.shape[0]),
        "source_nbfix_overrides": int(
            nbfix_details.get("source_parameter_override_count", -1)
        ),
        "applicable_nbfix_overrides": int(prepared.nbfix_type_pairs.shape[0]),
        "hydrogen_count": int(
            np.count_nonzero(
                np.char.upper(np.asarray(prepared.symbols, dtype=str)) == "H"
            )
        ),
    }
    blockers = [
        _baseline_mismatch(key, counts[key], expected[key])
        for key in counts
        if counts[key] != expected[key]
    ]

    float_values = {
        "temperature_K": protocol.get("temperature_K"),
        "langevin_friction_ps^-1": protocol.get("langevin_friction_ps^-1"),
        "time_step_fs": protocol.get("time_step_fs"),
        "cutoff_A": nonbonded.get("cutoff"),
        "switch_distance_A": nonbonded.get("switch_distance"),
        "ewald_error_tolerance": source_pme.get("ewald_error_tolerance"),
        "constraint_tolerance": source_constraints.get("tolerance"),
        "hmr_target_hydrogen_mass": hmr_policy.get("target_hydrogen_mass"),
        "pme_alpha_per_A": pme_config.get("alpha"),
    }
    for key, value in float_values.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            blockers.append(_baseline_mismatch(key, value, expected[key]))
            continue
        if not math.isclose(parsed, float(expected[key]), rel_tol=0.0, abs_tol=1.0e-7):
            blockers.append(_baseline_mismatch(key, parsed, expected[key]))

    trajectory = dict(protocol.get("trajectory", {}))
    trajectory_interval = trajectory.get("interval_steps")
    if trajectory_interval != expected["trajectory_interval_steps"]:
        blockers.append(
            _baseline_mismatch(
                "trajectory_interval_steps",
                trajectory_interval,
                expected["trajectory_interval_steps"],
            )
        )
    box_lengths = tuple(float(value) for value in np.asarray(prepared.cell_lengths).tolist())
    if not np.allclose(
        box_lengths,
        expected["box_lengths_A"],
        rtol=0.0,
        atol=1.0e-5,
    ):
        blockers.append(
            _baseline_mismatch("box_lengths_A", box_lengths, expected["box_lengths_A"])
        )
    mesh_shape = tuple(int(value) for value in np.asarray(prepared.pme_mesh_shape).tolist())
    if mesh_shape != expected["pme_mesh_shape"]:
        blockers.append(
            _baseline_mismatch("pme_mesh_shape", mesh_shape, expected["pme_mesh_shape"])
        )
    selected_hydrogens = hmr.get("selected_hydrogens", [])
    if len(selected_hydrogens) != expected["hydrogen_count"]:
        blockers.append(
            _baseline_mismatch(
                "hmr_selected_hydrogens",
                len(selected_hydrogens),
                expected["hydrogen_count"],
            )
        )
    hydrogen_mask = (
        np.char.upper(np.asarray(prepared.symbols, dtype=str)) == "H"
    )
    if not np.allclose(
        np.asarray(prepared.masses)[hydrogen_mask],
        expected["hmr_target_hydrogen_mass"],
        rtol=0.0,
        atol=1.0e-9,
    ):
        blockers.append("source_baseline_mismatch:hmr_transformed_hydrogen_masses")
    return blockers


def _baseline_mismatch(name: str, parsed: object, expected: object) -> str:
    return f"source_baseline_mismatch:{name}:parsed={parsed}:expected={expected}"


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


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline GPCRmd preparation command-line interface.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="prepare and strictly validate a source-backed MLX artifact",
    )
    prepare.add_argument("--target-id", required=True)
    prepare.add_argument("--cache", required=True)
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--report", required=True)
    prepare.add_argument("--registry-path")
    args = parser.parse_args(None if argv is None else list(argv))

    if args.command == "prepare":
        payload = prepare_gpcrmd_artifact(
            cache_path=args.cache,
            out_dir=args.out,
            report_path=args.report,
            target_id=args.target_id,
            registry_path=args.registry_path,
            source_manifest_path=args.source_manifest,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "target_id": payload["target_id"],
                    "atom_count": payload["strict_production_load"].get(
                        "atom_count"
                    ),
                    "out_dir": payload["out_dir"],
                    "report_path": str(args.report),
                    "workload_manifest_path": payload["workload_manifest_path"],
                    "blockers": payload["blockers"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if payload["status"] != "prepared":
            raise SystemExit(1)


__all__ = [
    "ACEMD_BINARY_VELOCITY_TO_ANGSTROM_PER_PS",
    "GPCRMD_API_DOCS_URL",
    "GPCRMD_729_SOURCE_BASELINE",
    "GPCRMD_DATA_DOWNLOAD_DOCS_URL",
    "GPCRMD_DYNAMICS_METADATA_URL_TEMPLATE",
    "GPCRMD_FILE_DOWNLOAD_REQUIRES_ACCOUNT",
    "GPCRMD_IMPORT_REPORT_NAME",
    "GPCRMD_WORKLOAD_MANIFEST_NAME",
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
    "build_gpcrmd_mlx_workload_manifest",
    "default_gpcrmd_targets",
    "gpcrmd_array_hash",
    "gpcrmd_mlx_compatibility_report",
    "gpcrmd_mlx_readiness_inventory",
    "gpcrmd_selection_reports",
    "inspect_gpcrmd_cache",
    "load_gpcrmd_targets",
    "main",
    "prepare_gpcrmd_artifact",
    "select_gpcrmd_target",
    "write_gpcrmd_import_report",
    "write_gpcrmd_targets",
]


if __name__ == "__main__":
    main()
