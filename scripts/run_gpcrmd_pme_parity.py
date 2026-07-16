"""Run independent CHARMM/PME parity for the GPCRmd 729 workload."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform as platform_module
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    artifact_readiness_report,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.benchmarks.pme_validation import (
    array_hash,
    force_error_metrics,
    manifest_hash,
    manifest_mismatches,
)
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.pme import PMEConfig, pme_readiness_report
from mlx_atomistic.prep import import_charmm_psf
from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system
from mlx_atomistic.prep.schema import PreparedSystem
from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.topology import Topology
from mlx_atomistic.units import COULOMB_CONSTANT_KJ_MOL_ANGSTROM

REPORT_NAME = "gpcrmd_pme_parity_report.json"
MLX_MANIFEST_NAME = "mlx_workload_manifest.json"
OPENMM_MANIFEST_NAME = "openmm_workload_manifest.json"
MANIFEST_COMPARISON_NAME = "manifest_comparison.json"
FORCE_ARRAYS_NAME = "complete_force_comparison.npz"
GENERATED_RTF_NAME = "openmm_source_types.rtf"
OPENMM_REFERENCE_ROLE = "reference-only validation; not a product runtime dependency"
OPENMM_OPENCL_FFT_PRIME_FACTORS = (2, 3, 5, 7, 11, 13)

SUPPORTED_OPENMM_FORCE_CLASSES = (
    "HarmonicBondForce",
    "HarmonicAngleForce",
    "PeriodicTorsionForce",
    "CustomTorsionForce",
    "CMAPTorsionForce",
    "NonbondedForce",
    "CustomNonbondedForce",
)
SUPPORTED_MLX_FORCE_NAMES = (
    "bond",
    "angle",
    "dihedral",
    "improper",
    "urey_bradley",
    "charmm_cmap_terms",
    "nonbonded",
)
MANIFEST_FIELDS = (
    "schema_version",
    "workload.name",
    "workload.operation",
    "workload.atom_count",
    "source.file_hashes",
    "particles.identity_hash",
    "particles.coordinate_hash",
    "particles.original_mass_hash",
    "particles.transformed_mass_hash",
    "particles.charge_hash",
    "particles.lj_particle_hash",
    "particles.source_net_charge_e",
    "cell.matrix_angstrom",
    "cell.matrix_hash",
    "forces.class_counts",
    "forces.term_counts",
    "forces.bond",
    "forces.angle",
    "forces.urey_bradley",
    "forces.proper_dihedral",
    "forces.harmonic_improper",
    "forces.cmap",
    "forces.nbfix",
    "nonbonded.exception_count",
    "nonbonded.exclusion_count",
    "nonbonded.active_exception_count",
    "nonbonded.exception_pairs_hash",
    "nonbonded.exception_parameter_hash",
    "nonbonded.cutoff_angstrom",
    "nonbonded.switch_distance_angstrom",
    "nonbonded.switching",
    "nonbonded.nbxmod",
    "nonbonded.e14fac",
    "nonbonded.dispersion_correction",
    "constraints.count",
    "constraints.pairs_hash",
    "constraints.distance_hash",
    "hydrogen_mass_repartitioning.status",
    "hydrogen_mass_repartitioning.selection",
    "hydrogen_mass_repartitioning.target_hydrogen_mass_da",
    "hydrogen_mass_repartitioning.selected_hydrogen_count",
    "protocol.ensemble",
    "protocol.fixed_cell",
    "protocol.time_step_fs",
    "pme.method",
    "pme.real_cutoff_angstrom",
    "pme.alpha_per_angstrom",
    "pme.mesh_shape",
    "pme.assignment_order",
    "pme.deconvolve_assignment",
    "pme.background_policy",
    "pme.charge_tolerance_e",
    "pme.ewald_error_tolerance",
    "pme.coulomb_constant_kj_mol_angstrom",
)


class GPCRmdParityError(RuntimeError):
    """Raised when the requested parity comparison cannot be completed."""


class GPCRmdParityBlocked(GPCRmdParityError):
    """Raised for fail-closed source, manifest, platform, or completeness blockers."""


class UnsupportedOpenMMForceError(GPCRmdParityBlocked):
    """Raised when OpenMM constructs a force class outside the approved contract."""


@dataclass(frozen=True)
class GPCRmdParityTolerances:
    """Acceptance thresholds for GPCRmd fixed-coordinate parity."""

    energy_per_atom_kj_mol: float = 5.0e-3
    relative_energy_error: float = 5.0e-5
    force_rms_kj_mol_nm: float = 3.0
    force_maximum_kj_mol_nm: float = 12.0


@dataclass(frozen=True)
class OpenMMApi:
    """Late-bound OpenMM modules used only by this reference script."""

    mm: Any
    app: Any
    unit: Any


def run_gpcrmd_pme_parity(
    *,
    source_manifest: str | Path,
    cache: str | Path,
    mlx_prepared: str | Path,
    platform_name: str,
    out: str | Path,
    precision: str = "single",
    tolerances: GPCRmdParityTolerances | None = None,
) -> dict[str, Any]:
    """Run independent source-vs-MLX GPCRmd CHARMM/PME parity.

    Args:
        source_manifest: Slice-1 fixture manifest with source hashes.
        cache: Caller-owned GPCRmd cache containing the verified files.
        mlx_prepared: Slice-2 prepared artifact directory.
        platform_name: OpenMM platform, normally ``"OpenCL"``.
        out: Caller-owned output directory.
        precision: Requested OpenMM precision when supported.
        tolerances: Optional parity thresholds.

    Returns:
        JSON-serializable passed, failed, or blocked report.
    """

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    tolerance = GPCRmdParityTolerances() if tolerances is None else tolerances
    base = _base_report(
        fixture="gpcrmd-729-beta1-5f8u-cyanopindolol",
        mlx_prepared=Path(mlx_prepared),
        source_manifest=Path(source_manifest),
        platform_name=platform_name,
        precision=precision,
        tolerances=tolerance,
        out=out_path,
    )
    try:
        api = _load_openmm()
        prepared_dir = Path(mlx_prepared)
        prepared = load_prepared_system(prepared_dir)
        artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
        readiness = artifact_readiness_report(
            artifact.metadata,
            require_production=True,
            arrays=artifact.arrays,
        ).to_dict()
        if readiness["status"] != "ready":
            raise GPCRmdParityBlocked(
                "mlx_artifact_not_ready:" + ",".join(readiness.get("blockers", ()))
            )
        source = _resolve_gpcrmd_source(
            source_manifest_path=Path(source_manifest),
            cache=Path(cache),
            prepared_dir=prepared_dir,
            out=out_path,
        )
        return _execute_parity(
            api=api,
            source=source,
            prepared=prepared,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            precision=precision,
            tolerances=tolerance,
            out=out_path,
            base={**base, "artifact_readiness": readiness},
            require_production=True,
        )
    except GPCRmdParityBlocked as exc:
        return _finish_report(
            {**base, "status": "blocked", "passed": False, "blockers": [str(exc)]},
            out_path,
        )
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        return _finish_report(
            {
                **base,
                "status": "blocked",
                "passed": False,
                "blockers": [f"{type(exc).__name__}:{exc}"],
            },
            out_path,
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
        return _finish_report(
            {
                **base,
                "status": "failed",
                "passed": False,
                "blockers": [f"{type(exc).__name__}:{exc}"],
            },
            out_path,
        )


def evaluate_small_charmm_pme_fixture(
    *,
    out: str | Path,
    platform_name: str = "Reference",
    precision: str = "double",
    tolerances: GPCRmdParityTolerances | None = None,
) -> dict[str, Any]:
    """Run the tracked small CHARMM/PME semantic fixture.

    Args:
        out: Caller-owned output directory.
        platform_name: OpenMM platform used for the small gate.
        precision: Requested OpenMM precision when supported.
        tolerances: Optional parity thresholds.

    Returns:
        JSON-serializable parity report with complete forces.
    """

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    tolerance = GPCRmdParityTolerances() if tolerances is None else tolerances
    fixture_root = Path("tests/fixtures/charmm")
    psf_path = fixture_root / "pme-mini.psf"
    prm_path = fixture_root / "pme-mini.prm"
    coords_path = fixture_root / "pme-mini.pdb"
    config = PMEConfig(
        mesh_shape=(32, 34, 36),
        alpha=0.30,
        real_cutoff=9.0,
        assignment_order=5,
        charge_tolerance=1.0e-4,
        deconvolve_assignment=True,
        background_policy="reject_non_neutral",
    )
    prepared = import_charmm_psf(
        psf_path=psf_path,
        params=[prm_path],
        coords_path=coords_path,
    )
    report = dict(prepared.metadata.compatibility_report)
    for key in ("supported_terms", "required_terms"):
        values = [str(value) for value in report.get(key, ())]
        if "pme" not in values:
            values.append("pme")
        report[key] = values
    report["electrostatics_model"] = "pme"
    report["periodic_box_present"] = True
    protocol = dict(prepared.metadata.protocol_metadata)
    protocol.update(
        {
            "ensemble": "NVT",
            "fixed_cell": True,
            "time_step_fs": 1.0,
            "nonbonded": {
                "cutoff": config.real_cutoff,
                "switching": True,
                "switch_distance": 7.5,
            },
            "pme": {"enabled": True, "ewald_error_tolerance": 5.0e-4},
        }
    )
    metadata = replace(
        prepared.metadata,
        compatibility_report=report,
        protocol_metadata=protocol,
        pme_config=_config_payload(config, ewald_error_tolerance=5.0e-4),
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        cell_matrix=np.diag(np.asarray(prepared.cell_lengths, dtype=np.float32)),
        pme_mesh_shape=np.asarray(config.mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([config.alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([config.real_cutoff], dtype=np.float32),
        pme_assignment_order=np.asarray([config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
        pme_background_policy=np.asarray([config.background_policy], dtype=str),
    )
    prepared_dir = out_path / "small-mlx-prepared"
    save_prepared_system(prepared, prepared_dir)
    source = {
        "fixture": "charmm-pme-mini",
        "psf_path": psf_path,
        "prm_path": prm_path,
        "coordinate_path": coords_path,
        "coordinate_kind": "pdb",
        "positions_angstrom": np.asarray(prepared.positions, dtype=np.float32),
        "cell_matrix_angstrom": np.asarray(prepared.cell_matrix, dtype=np.float64),
        "config": config,
        "ewald_error_tolerance": 5.0e-4,
        "protocol": {
            "ensemble": "NVT",
            "fixed_cell": True,
            "time_step_fs": 1.0,
            "nonbonded": {
                "cutoff_angstrom": config.real_cutoff,
                "switch_distance_angstrom": 7.5,
                "switching": True,
            },
            "pme": {"enabled": True, "ewald_error_tolerance": 5.0e-4},
            "constraints": {
                "enabled": True,
                "count": int(prepared.constraints.shape[0]),
            },
            "hydrogen_mass_repartitioning": {"status": "not_applied"},
        },
        "file_hashes": {
            path.name: _sha256_file(path) for path in (psf_path, prm_path, coords_path)
        },
        "generated_rtf_path": out_path / GENERATED_RTF_NAME,
    }
    api = _load_openmm()
    base = _base_report(
        fixture="charmm-pme-mini",
        mlx_prepared=prepared_dir,
        source_manifest=None,
        platform_name=platform_name,
        precision=precision,
        tolerances=tolerance,
        out=out_path,
    )
    try:
        return _execute_parity(
            api=api,
            source=source,
            prepared=prepared,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            precision=precision,
            tolerances=tolerance,
            out=out_path,
            base=base,
            require_production=False,
        )
    except GPCRmdParityBlocked as exc:
        return _finish_report(
            {**base, "status": "blocked", "passed": False, "blockers": [str(exc)]},
            out_path,
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
        return _finish_report(
            {
                **base,
                "status": "failed",
                "passed": False,
                "blockers": [f"{type(exc).__name__}:{exc}"],
            },
            out_path,
        )


def _execute_parity(
    *,
    api: OpenMMApi,
    source: dict[str, Any],
    prepared: PreparedSystem,
    prepared_dir: Path,
    platform_name: str,
    precision: str,
    tolerances: GPCRmdParityTolerances,
    out: Path,
    base: dict[str, Any],
    require_production: bool,
) -> dict[str, Any]:
    reference = _build_openmm_reference(api, source=source)
    mlx_manifest = _mlx_manifest(prepared, source=source)
    openmm_manifest = _openmm_manifest(api, reference=reference, source=source)
    comparison = _compare_manifests(mlx_manifest, openmm_manifest)
    _write_json(out / MLX_MANIFEST_NAME, mlx_manifest)
    _write_json(out / OPENMM_MANIFEST_NAME, openmm_manifest)
    _write_json(out / MANIFEST_COMPARISON_NAME, comparison)
    if not comparison["matched"]:
        raise GPCRmdParityBlocked(
            "manifest_mismatch:" + ",".join(sorted(comparison["mismatches"]))
        )
    _require_openmm_platform(api, platform_name)

    openmm_result = _evaluate_openmm_reference(
        api,
        reference=reference,
        platform_name=platform_name,
        precision=precision,
        config=source["config"],
    )
    del reference["system"]
    gc.collect()
    mlx_result = _evaluate_mlx_prepared(
        prepared_dir,
        require_production=require_production,
    )
    _require_complete_forces(
        mlx_result["forces_kj_mol_nm"],
        atom_count=prepared.atom_count,
        engine="mlx",
    )
    _require_complete_forces(
        openmm_result["forces_kj_mol_nm"],
        atom_count=prepared.atom_count,
        engine="openmm",
    )
    metrics = force_error_metrics(
        mlx_result["forces_kj_mol_nm"],
        openmm_result["forces_kj_mol_nm"],
        candidate_energy=mlx_result["total_energy_kj_mol"],
        reference_energy=openmm_result["total_energy_kj_mol"],
    )
    component_metrics = _component_energy_metrics(
        mlx_result["component_energy_kj_mol"],
        openmm_result["component_energy_kj_mol"],
        required_components=_required_components(mlx_manifest),
        atom_count=prepared.atom_count,
    )
    force_delta = (
        np.asarray(mlx_result["forces_kj_mol_nm"], dtype=np.float64)
        - np.asarray(openmm_result["forces_kj_mol_nm"], dtype=np.float64)
    )
    force_path = out / FORCE_ARRAYS_NAME
    np.savez(
        force_path,
        mlx_forces_kj_mol_nm=np.asarray(
            mlx_result["forces_kj_mol_nm"], dtype=np.float32
        ),
        openmm_forces_kj_mol_nm=np.asarray(
            openmm_result["forces_kj_mol_nm"], dtype=np.float32
        ),
        force_delta_kj_mol_nm=force_delta.astype(np.float32),
    )
    checks = _parity_checks(
        metrics=asdict(metrics),
        component_metrics=component_metrics,
        comparison=comparison,
        openmm_result=openmm_result,
        mlx_result=mlx_result,
        tolerances=tolerances,
    )
    passed = all(checks.values())
    report = {
        **base,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "blockers": [] if passed else [name for name, value in checks.items() if not value],
        "atom_count": prepared.atom_count,
        "manifest_comparison": comparison,
        "manifests": {
            "mlx": str(out / MLX_MANIFEST_NAME),
            "openmm": str(out / OPENMM_MANIFEST_NAME),
        },
        "energies": {
            "mlx_total_kj_mol": mlx_result["total_energy_kj_mol"],
            "openmm_total_kj_mol": openmm_result["total_energy_kj_mol"],
            "mlx_components_kj_mol": mlx_result["component_energy_kj_mol"],
            "openmm_components_kj_mol": openmm_result["component_energy_kj_mol"],
            "component_metrics": component_metrics,
        },
        "force_metrics": asdict(metrics),
        "force_arrays": {
            "path": str(force_path),
            "shape": list(force_delta.shape),
            "mlx_hash": array_hash(
                np.asarray(mlx_result["forces_kj_mol_nm"], dtype=np.float32)
            ),
            "openmm_hash": array_hash(
                np.asarray(openmm_result["forces_kj_mol_nm"], dtype=np.float32)
            ),
            "delta_hash": array_hash(force_delta.astype(np.float32)),
        },
        "checks": checks,
        "mlx": _without_arrays(mlx_result),
        "openmm": _without_arrays(openmm_result),
    }
    return _finish_report(report, out)


def _resolve_gpcrmd_source(
    *,
    source_manifest_path: Path,
    cache: Path,
    prepared_dir: Path,
    out: Path,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("blockers"):
        raise GPCRmdParityBlocked("artifact_source:" + ",".join(source_manifest["blockers"]))
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for record in source_manifest.get("files", ()):
        role = str(record.get("role", ""))
        filename = str(record.get("resolved_filename", ""))
        if not filename:
            continue
        path = cache / filename
        expected = str(record.get("sha256", ""))
        if not path.is_file() or _sha256_file(path) != expected:
            raise GPCRmdParityBlocked(f"artifact_source_hash_mismatch:{role}:{path}")
        files[role] = path
        hashes[filename] = expected
    missing = sorted({"topology", "parameters", "model"} - set(files))
    if missing:
        raise GPCRmdParityBlocked("artifact_source_missing_roles:" + ",".join(missing))

    workload_path = prepared_dir / "mlx-workload-manifest.json"
    workload = json.loads(workload_path.read_text())
    replicate = str(workload.get("protocol", {}).get("selected_replicate", "rep_1"))
    archive = next(iter(source_manifest.get("archives", ())), None)
    if archive is None:
        raise GPCRmdParityBlocked("artifact_source_missing_protocol_archive")
    protocol_record = next(
        (
            record
            for record in source_manifest.get("files", ())
            if str(record.get("role", "")) == "protocol"
        ),
        None,
    )
    if protocol_record is None:
        raise GPCRmdParityBlocked("artifact_source_missing_protocol_record")
    extraction_name = Path(str(archive.get("extraction_root", ""))).name
    protocol_root = cache / extraction_name / replicate
    protocol_paths = {
        name: protocol_root / name
        for name in ("input", "input.coor", "input.xsc", "log.txt")
    }
    member_records = {
        str(member.get("normalized_name")): member
        for member in protocol_record.get("archive_members", ())
        if member.get("kind") == "file"
    }
    for name, path in protocol_paths.items():
        key = f"{replicate}/{name}"
        record = member_records.get(key)
        if record is None or not path.is_file():
            raise GPCRmdParityBlocked(f"artifact_source_missing_protocol_file:{key}")
        expected = str(record.get("sha256", ""))
        if expected and _sha256_file(path) != expected:
            raise GPCRmdParityBlocked(f"artifact_source_hash_mismatch:{key}")
        hashes[key] = _sha256_file(path)

    positions = _read_acemd_vectors(protocol_paths["input.coor"])
    cell_matrix = _read_xsc_matrix(protocol_paths["input.xsc"])
    protocol = _read_gpcrmd_protocol(protocol_paths["input"], protocol_paths["log.txt"])
    config = _derive_source_pme_config(
        cell_matrix,
        cutoff=float(protocol["nonbonded"]["cutoff_angstrom"]),
        tolerance=float(protocol["pme"]["ewald_error_tolerance"]),
    )
    if positions.shape[0] != int(workload["workload"]["atom_count"]):
        raise GPCRmdParityBlocked("source_coordinate_atom_count_mismatch")
    return {
        "fixture": str(workload["workload"]["name"]),
        "psf_path": files["topology"],
        "prm_path": files["parameters"],
        "coordinate_path": protocol_paths["input.coor"],
        "coordinate_kind": "acemd_binary",
        "positions_angstrom": positions,
        "cell_matrix_angstrom": cell_matrix,
        "config": config,
        "ewald_error_tolerance": float(protocol["pme"]["ewald_error_tolerance"]),
        "protocol": protocol,
        "file_hashes": hashes,
        "generated_rtf_path": out / GENERATED_RTF_NAME,
        "source_manifest_path": source_manifest_path,
        "workload_manifest_path": workload_path,
    }


def _build_openmm_reference(api: OpenMMApi, *, source: dict[str, Any]) -> dict[str, Any]:
    rtf_path = Path(source["generated_rtf_path"])
    source_identity = _read_charmm_psf_identity(Path(source["psf_path"]))
    _generate_openmm_rtf(
        api,
        psf_path=Path(source["psf_path"]),
        prm_path=Path(source["prm_path"]),
        out=rtf_path,
    )
    psf = api.app.CharmmPsfFile(str(source["psf_path"]))
    _validate_openmm_source_order(psf, source_identity)
    parameters = api.app.CharmmParameterSet(str(rtf_path), str(source["prm_path"]))
    cell = np.asarray(source["cell_matrix_angstrom"], dtype=np.float64)
    if cell.shape != (3, 3) or not np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-8):
        raise GPCRmdParityBlocked("source_cell_is_not_orthorhombic")
    psf.setBox(*(float(value) * api.unit.angstrom for value in np.diag(cell)))
    protocol = dict(source["protocol"])
    constraint_protocol = dict(protocol.get("constraints", {}))
    constraints_enabled = bool(constraint_protocol.get("enabled", False))
    config = source["config"]
    system = psf.createSystem(
        parameters,
        nonbondedMethod=api.app.PME,
        nonbondedCutoff=float(config.real_cutoff) * api.unit.angstrom,
        switchDistance=float(protocol["nonbonded"]["switch_distance_angstrom"])
        * api.unit.angstrom,
        constraints=api.app.HBonds if constraints_enabled else None,
        rigidWater=constraints_enabled,
        hydrogenMass=None,
        ewaldErrorTolerance=float(source["ewald_error_tolerance"]),
        removeCMMotion=False,
        flexibleConstraints=True,
    )
    original_masses, transformed_masses, hmr_details = _apply_source_masses_and_hmr(
        api,
        system=system,
        psf=psf,
        protocol=protocol,
    )
    nonbonded = _find_force(api, system, "NonbondedForce", expected=1)[0]
    nonbonded.setPMEParameters(
        float(config.alpha) * 10.0 / api.unit.nanometer,
        *config.mesh_shape,
    )
    nonbonded.setUseDispersionCorrection(False)
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
        if isinstance(force, api.mm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(index)
    box_vectors = tuple(
        api.mm.Vec3(*(row * 0.1).tolist()) for row in cell
    ) * api.unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    positions = np.asarray(source["positions_angstrom"], dtype=np.float32)
    if positions.shape != (system.getNumParticles(), 3):
        raise GPCRmdParityBlocked("openmm_source_coordinate_shape_mismatch")
    expected_constraints = int(constraint_protocol.get("count", 0))
    if system.getNumConstraints() != expected_constraints:
        raise GPCRmdParityBlocked(
            f"openmm_constraint_count_mismatch:{system.getNumConstraints()}:{expected_constraints}"
        )
    _validate_openmm_force_inventory(system)
    return {
        "system": system,
        "psf": psf,
        "parameters": parameters,
        "positions_angstrom": positions,
        "cell_matrix_angstrom": cell,
        "box_vectors": box_vectors,
        "original_masses": original_masses,
        "transformed_masses": transformed_masses,
        "hmr": hmr_details,
        "generated_rtf": rtf_path,
        "source_identity": source_identity,
    }


def _mlx_manifest(prepared: PreparedSystem, *, source: dict[str, Any]) -> dict[str, Any]:
    force_payload = _mlx_force_payload(prepared)
    exception_payload = _exception_payload(
        prepared.nonbonded_exception_pairs,
        prepared.nonbonded_exception_charge_product,
        prepared.nonbonded_exception_sigma,
        prepared.nonbonded_exception_epsilon,
    )
    constraints = _constraint_payload(prepared.constraints, prepared.constraint_distance)
    hmr_source = dict(
        prepared.metadata.protocol_metadata.get("hydrogen_mass_repartitioning", {})
    )
    original_masses = np.asarray(
        hmr_source.get("original_masses", prepared.masses),
        dtype=np.float64,
    )
    hmr = _hmr_manifest(
        original_masses=original_masses,
        transformed_masses=np.asarray(prepared.masses, dtype=np.float64),
        source=hmr_source,
    )
    protocol = dict(prepared.metadata.protocol_metadata)
    nonbonded = dict(protocol.get("nonbonded", {}))
    exception_details = dict(
        prepared.metadata.compatibility_report.get("term_details", {}).get(
            "nonbonded_exception", {}
        )
    )
    config = _prepared_pme_config(prepared)
    return _finalize_manifest(
        {
            "schema_version": 1,
            "workload": {
                "name": source["fixture"],
                "operation": "fixed_coordinate_total_energy_and_complete_forces",
                "atom_count": prepared.atom_count,
            },
            "source": {"file_hashes": dict(sorted(source["file_hashes"].items()))},
            "particles": _particle_payload(
                identity=_prepared_identity(prepared),
                positions=prepared.positions,
                original_masses=original_masses,
                transformed_masses=prepared.masses,
                charges=prepared.charges,
                sigma=prepared.sigma,
                epsilon=prepared.epsilon,
                source_net_charge=float(
                    prepared.metadata.selections.get(
                        "system_charge_source_precision",
                        np.sum(prepared.charges, dtype=np.float64),
                    )
                ),
            ),
            "cell": _cell_payload(prepared.cell_matrix),
            "forces": force_payload,
            "nonbonded": {
                **exception_payload,
                "cutoff_angstrom": _rounded(float(nonbonded.get("cutoff", config.real_cutoff)), 6),
                "switch_distance_angstrom": _rounded(
                    float(nonbonded.get("switch_distance", 0.0)), 6
                ),
                "switching": bool(nonbonded.get("switching", False)),
                "nbxmod": int(exception_details.get("nbxmod", 5)),
                "e14fac": _rounded(float(exception_details.get("e14fac", 1.0)), 6),
                "dispersion_correction": False,
            },
            "constraints": constraints,
            "hydrogen_mass_repartitioning": hmr,
            "protocol": _protocol_manifest(protocol),
            "pme": _pme_manifest(
                config,
                ewald_error_tolerance=float(
                    protocol.get("pme", {}).get("ewald_error_tolerance", 5.0e-4)
                ),
            ),
            "engine": {
                "name": "mlx_atomistic",
                "role": "product_runtime",
                "precision": "float32 forces with source-faithful artifact parameters",
            },
        }
    )


def _openmm_manifest(
    api: OpenMMApi,
    *,
    reference: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    psf = reference["psf"]
    system = reference["system"]
    force_payload = _openmm_force_payload(api, reference=reference)
    nonbonded = _find_force(api, system, "NonbondedForce", expected=1)[0]
    exceptions = _openmm_exception_arrays(api, nonbonded)
    exception_payload = _exception_payload(
        exceptions["pairs"],
        exceptions["charge_product"],
        exceptions["sigma"],
        exceptions["epsilon"],
    )
    constraint_arrays = _openmm_constraint_arrays(api, system)
    constraints = _constraint_payload(
        constraint_arrays["pairs"],
        constraint_arrays["distance"],
    )
    charges = np.asarray([float(atom.charge) for atom in psf.atom_list], dtype=np.float64)
    sigma, epsilon = _openmm_source_lj_arrays(psf)
    protocol = dict(source["protocol"])
    parameters = reference["parameters"]
    return _finalize_manifest(
        {
            "schema_version": 1,
            "workload": {
                "name": source["fixture"],
                "operation": "fixed_coordinate_total_energy_and_complete_forces",
                "atom_count": system.getNumParticles(),
            },
            "source": {"file_hashes": dict(sorted(source["file_hashes"].items()))},
            "particles": _particle_payload(
                identity=reference["source_identity"],
                positions=reference["positions_angstrom"],
                original_masses=reference["original_masses"],
                transformed_masses=reference["transformed_masses"],
                charges=charges,
                sigma=sigma,
                epsilon=epsilon,
                source_net_charge=float(np.sum(charges, dtype=np.float64)),
            ),
            "cell": _cell_payload(reference["cell_matrix_angstrom"]),
            "forces": force_payload,
            "nonbonded": {
                **exception_payload,
                "cutoff_angstrom": _rounded(float(source["config"].real_cutoff), 6),
                "switch_distance_angstrom": _rounded(
                    float(protocol["nonbonded"]["switch_distance_angstrom"]), 6
                ),
                "switching": bool(nonbonded.getUseSwitchingFunction()),
                "nbxmod": int(parameters.nbxmod),
                "e14fac": _rounded(float(parameters.e14fac), 6),
                "dispersion_correction": bool(nonbonded.getUseDispersionCorrection()),
            },
            "constraints": constraints,
            "hydrogen_mass_repartitioning": _hmr_manifest(
                original_masses=reference["original_masses"],
                transformed_masses=reference["transformed_masses"],
                source=reference["hmr"],
            ),
            "protocol": _protocol_manifest(protocol),
            "pme": _pme_manifest(
                source["config"],
                ewald_error_tolerance=float(source["ewald_error_tolerance"]),
            ),
            "engine": {
                "name": "openmm",
                "role": OPENMM_REFERENCE_ROLE,
                "version": _openmm_version(api),
                "source": "CharmmPsfFile/CharmmParameterSet/source restart and XSC",
                "generated_rtf_sha256": _sha256_file(reference["generated_rtf"]),
            },
        }
    )


def _mlx_force_payload(prepared: PreparedSystem) -> dict[str, Any]:
    active_bonds = np.asarray(prepared.bond_k, dtype=np.float64) > 1.0e-12
    bond = _term_payload(
        np.asarray(prepared.bonds)[active_bonds],
        np.column_stack(
            [
                np.asarray(prepared.bond_k)[active_bonds],
                np.asarray(prepared.bond_length)[active_bonds],
            ]
        ),
        reversible=True,
    )
    angle = _term_payload(
        prepared.angles,
        np.column_stack([prepared.angle_k, prepared.angle_theta]),
        reversible=True,
    )
    urey = _term_payload(
        np.asarray(prepared.urey_bradley_terms)[:, [0, 2]],
        np.column_stack([prepared.urey_bradley_k, prepared.urey_bradley_distance]),
        reversible=True,
    )
    proper = _term_payload(
        prepared.dihedrals,
        np.column_stack(
            [prepared.dihedral_k, prepared.dihedral_periodicity, prepared.dihedral_phase]
        ),
        reversible=True,
    )
    improper = _term_payload(
        prepared.impropers,
        np.column_stack([prepared.improper_k, prepared.improper_phase]),
        reversible=False,
    )
    cmap = _mlx_cmap_payload(prepared)
    nbfix = _nbfix_payload(
        prepared.nbfix_type_pairs,
        prepared.nbfix_type_sigma,
        prepared.nbfix_type_epsilon,
    )
    term_counts = {
        "bond": bond["count"],
        "angle": angle["count"],
        "urey_bradley": urey["count"],
        "proper_dihedral": proper["count"],
        "harmonic_improper": improper["count"],
        "cmap": cmap["term_count"],
        "cmap_maps": cmap["map_count"],
        "nbfix": nbfix["count"],
        "nonbonded_particles": prepared.atom_count,
        "nonbonded_exceptions": int(prepared.nonbonded_exception_pairs.shape[0]),
        "custom_nonbonded_particles": prepared.atom_count if nbfix["count"] else 0,
        "custom_nonbonded_exclusions": (
            int(prepared.nonbonded_exception_pairs.shape[0]) if nbfix["count"] else 0
        ),
        "constraints": int(prepared.constraints.shape[0]),
    }
    class_counts = {
        "HarmonicBondForce": int(bond["count"] > 0) + int(urey["count"] > 0),
        "HarmonicAngleForce": int(angle["count"] > 0),
        "PeriodicTorsionForce": int(proper["count"] > 0),
        "CustomTorsionForce": int(improper["count"] > 0),
        "CMAPTorsionForce": int(cmap["term_count"] > 0),
        "NonbondedForce": int(prepared.atom_count > 0),
        "CustomNonbondedForce": int(nbfix["count"] > 0),
    }
    return {
        "class_counts": class_counts,
        "term_counts": term_counts,
        "bond": bond,
        "angle": angle,
        "urey_bradley": urey,
        "proper_dihedral": proper,
        "harmonic_improper": improper,
        "cmap": cmap,
        "nbfix": nbfix,
    }


def _openmm_force_payload(api: OpenMMApi, *, reference: dict[str, Any]) -> dict[str, Any]:
    system = reference["system"]
    psf = reference["psf"]
    forces = {
        name: _find_force(api, system, name)
        for name in SUPPORTED_OPENMM_FORCE_CLASSES
    }
    bond_forces = forces["HarmonicBondForce"]
    if len(bond_forces) not in {1, 2}:
        raise UnsupportedOpenMMForceError(
            f"expected one bond force plus optional Urey-Bradley force, found {len(bond_forces)}"
        )
    bond = _openmm_bond_payload(api, bond_forces[0])
    urey = (
        _openmm_bond_payload(api, bond_forces[1])
        if len(bond_forces) == 2
        else _term_payload(np.empty((0, 2)), np.empty((0, 2)), reversible=True)
    )
    angle = _openmm_angle_payload(api, forces["HarmonicAngleForce"][0])
    proper = _openmm_proper_payload(api, forces["PeriodicTorsionForce"][0])
    improper = _openmm_improper_payload(forces["CustomTorsionForce"][0])
    cmap = _openmm_cmap_payload(api, system)
    nbfix = _openmm_nbfix_payload(psf)
    exception_count = _find_force(api, system, "NonbondedForce", expected=1)[
        0
    ].getNumExceptions()
    custom_nonbonded = forces["CustomNonbondedForce"]
    term_counts = {
        "bond": bond["count"],
        "angle": angle["count"],
        "urey_bradley": urey["count"],
        "proper_dihedral": proper["count"],
        "harmonic_improper": improper["count"],
        "cmap": cmap["term_count"],
        "cmap_maps": cmap["map_count"],
        "nbfix": nbfix["count"],
        "nonbonded_particles": system.getNumParticles(),
        "nonbonded_exceptions": exception_count,
        "custom_nonbonded_particles": (
            custom_nonbonded[0].getNumParticles() if custom_nonbonded else 0
        ),
        "custom_nonbonded_exclusions": (
            custom_nonbonded[0].getNumExclusions() if custom_nonbonded else 0
        ),
        "constraints": system.getNumConstraints(),
    }
    return {
        "class_counts": {name: len(values) for name, values in forces.items()},
        "term_counts": term_counts,
        "bond": bond,
        "angle": angle,
        "urey_bradley": urey,
        "proper_dihedral": proper,
        "harmonic_improper": improper,
        "cmap": cmap,
        "nbfix": nbfix,
    }


def _term_payload(
    indices: Any,
    parameters: Any,
    *,
    reversible: bool,
) -> dict[str, Any]:
    index_array = np.asarray(indices, dtype=np.int32)
    parameter_array = np.asarray(parameters, dtype=np.float64)
    if index_array.size == 0:
        width = index_array.shape[1] if index_array.ndim == 2 else 0
        parameter_width = parameter_array.shape[1] if parameter_array.ndim == 2 else 0
        index_array = np.empty((0, width), dtype=np.int32)
        parameter_array = np.empty((0, parameter_width), dtype=np.float64)
    if index_array.ndim != 2 or parameter_array.ndim != 2:
        raise GPCRmdParityBlocked("term_manifest_arrays_must_be_rank_two")
    if index_array.shape[0] != parameter_array.shape[0]:
        raise GPCRmdParityBlocked("term_manifest_array_length_mismatch")
    records: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    canonical_parameters = _canonical_float(
        np.asarray(parameter_array, dtype=np.float32),
        decimals=5,
    )
    for index_row, parameter_row in zip(
        index_array.tolist(), canonical_parameters.tolist(), strict=True
    ):
        index_tuple = tuple(int(value) for value in index_row)
        if reversible:
            index_tuple = min(index_tuple, tuple(reversed(index_tuple)))
        records.append((index_tuple, tuple(float(value) for value in parameter_row)))
    records.sort(key=lambda record: (*record[0], *record[1]))
    sorted_indices = np.asarray([record[0] for record in records], dtype=np.int32).reshape(
        (-1, index_array.shape[1])
    )
    sorted_parameters = np.asarray(
        [record[1] for record in records], dtype=np.float64
    ).reshape((-1, parameter_array.shape[1]))
    return {
        "count": int(sorted_indices.shape[0]),
        "index_hash": array_hash(sorted_indices),
        "parameter_hash": array_hash(sorted_parameters),
    }


def _mlx_cmap_payload(prepared: PreparedSystem) -> dict[str, Any]:
    terms = np.asarray(prepared.charmm_cmap_terms, dtype=np.int32)
    indices = np.asarray(prepared.charmm_cmap_grid_indices, dtype=np.int32)
    grids = np.asarray(prepared.charmm_cmap_grids, dtype=np.float64)
    return _cmap_payload(terms, indices, grids)


def _openmm_cmap_payload(api: OpenMMApi, system: Any) -> dict[str, Any]:
    forces = _find_force(api, system, "CMAPTorsionForce")
    if len(forces) > 1:
        raise UnsupportedOpenMMForceError(
            f"expected at most one CMAPTorsionForce, found {len(forces)}"
        )
    if not forces:
        return _cmap_payload(
            np.empty((0, 8), dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.empty((0, 0, 0), dtype=np.float64),
        )
    force = forces[0]
    grids: list[np.ndarray] = []
    for map_index in range(force.getNumMaps()):
        resolution, values = force.getMapParameters(map_index)
        grid = np.asarray(
            [
                value.value_in_unit(api.unit.kilojoule_per_mole)
                if hasattr(value, "value_in_unit")
                else float(value)
                for value in values
            ],
            dtype=np.float64,
        )
        grids.append(grid.reshape((int(resolution), int(resolution))).T)
    terms: list[tuple[int, ...]] = []
    map_indices: list[int] = []
    for term_index in range(force.getNumTorsions()):
        parameters = force.getTorsionParameters(term_index)
        map_indices.append(int(parameters[0]))
        terms.append(tuple(int(value) for value in parameters[1:]))
    grid_array = np.stack(grids) if grids else np.empty((0, 0, 0), dtype=np.float64)
    return _cmap_payload(
        np.asarray(terms, dtype=np.int32).reshape((-1, 8)),
        np.asarray(map_indices, dtype=np.int32),
        grid_array,
    )


def _cmap_payload(terms: np.ndarray, indices: np.ndarray, grids: np.ndarray) -> dict[str, Any]:
    if terms.shape[0] != indices.shape[0]:
        raise GPCRmdParityBlocked("cmap_term_index_length_mismatch")
    used: dict[int, int] = {}
    selected: list[np.ndarray] = []
    remapped: list[int] = []
    for value in indices.tolist():
        old = int(value)
        if old not in used:
            used[old] = len(selected)
            selected.append(np.asarray(grids[old], dtype=np.float64))
        remapped.append(used[old])
    term_rows = np.column_stack(
        [terms, np.asarray(remapped, dtype=np.int32)]
    ) if terms.size else np.empty((0, 9), dtype=np.int32)
    grid_hashes = [
        array_hash(_canonical_float(np.asarray(grid, dtype=np.float32), decimals=5))
        for grid in selected
    ]
    return {
        "term_count": int(terms.shape[0]),
        "map_count": len(selected),
        "term_hash": array_hash(np.asarray(term_rows, dtype=np.int32)),
        "map_shapes": [list(grid.shape) for grid in selected],
        "map_hashes": grid_hashes,
    }


def _nbfix_payload(type_pairs: Any, sigma: Any, epsilon: Any) -> dict[str, Any]:
    pairs = np.asarray(type_pairs, dtype=str)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=str)
    values = _canonical_float(
        np.asarray(
            np.column_stack([np.asarray(sigma), np.asarray(epsilon)]),
            dtype=np.float32,
        ),
        decimals=5,
    ) if pairs.shape[0] else np.empty((0, 2), dtype=np.float64)
    records = []
    for pair, parameters in zip(pairs.tolist(), values.tolist(), strict=True):
        key = tuple(sorted((str(pair[0]).upper(), str(pair[1]).upper())))
        records.append((key, tuple(float(value) for value in parameters)))
    records.sort(key=lambda record: (*record[0], *record[1]))
    sorted_pairs = np.asarray([record[0] for record in records], dtype=str).reshape((-1, 2))
    sorted_values = np.asarray([record[1] for record in records], dtype=np.float64).reshape(
        (-1, 2)
    )
    return {
        "count": len(records),
        "type_pair_hash": array_hash(sorted_pairs),
        "parameter_hash": array_hash(sorted_values),
    }


def _openmm_nbfix_payload(psf: Any) -> dict[str, Any]:
    present = {str(atom.type.name).upper() for atom in psf.atom_list}
    records: dict[tuple[str, str], tuple[float, float]] = {}
    type_by_name = {str(atom.type.name).upper(): atom.type for atom in psf.atom_list}
    for left in sorted(present):
        atom_type = type_by_name[left]
        for right, values in atom_type.nbfix.items():
            right_key = str(right).upper()
            if right_key not in present:
                continue
            rmin, epsilon, _rmin14, _epsilon14 = values
            pair = tuple(sorted((left, right_key)))
            records[pair] = (
                float(rmin) * 2 ** (-1.0 / 6.0),
                abs(float(epsilon)) * 4.184,
            )
    pairs = np.asarray(sorted(records), dtype=str).reshape((-1, 2))
    values = np.asarray([records[tuple(pair)] for pair in pairs.tolist()], dtype=np.float64)
    return _nbfix_payload(
        pairs,
        values[:, 0] if values.size else [],
        values[:, 1] if values.size else [],
    )


def _openmm_bond_payload(api: OpenMMApi, force: Any) -> dict[str, Any]:
    indices = []
    parameters = []
    for index in range(force.getNumBonds()):
        left, right, length, stiffness = force.getBondParameters(index)
        indices.append((int(left), int(right)))
        parameters.append(
            (
                stiffness.value_in_unit(
                    api.unit.kilojoule_per_mole / api.unit.nanometer**2
                )
                / 100.0,
                length.value_in_unit(api.unit.angstrom),
            )
        )
    return _term_payload(indices, parameters, reversible=True)


def _openmm_angle_payload(api: OpenMMApi, force: Any) -> dict[str, Any]:
    indices = []
    parameters = []
    for index in range(force.getNumAngles()):
        left, center, right, theta, stiffness = force.getAngleParameters(index)
        indices.append((int(left), int(center), int(right)))
        parameters.append(
            (
                stiffness.value_in_unit(
                    api.unit.kilojoule_per_mole / api.unit.radian**2
                ),
                theta.value_in_unit(api.unit.radian),
            )
        )
    return _term_payload(indices, parameters, reversible=True)


def _openmm_proper_payload(api: OpenMMApi, force: Any) -> dict[str, Any]:
    indices = []
    parameters = []
    for index in range(force.getNumTorsions()):
        left, center_left, center_right, right, periodicity, phase, stiffness = (
            force.getTorsionParameters(index)
        )
        indices.append((int(left), int(center_left), int(center_right), int(right)))
        parameters.append(
            (
                stiffness.value_in_unit(api.unit.kilojoule_per_mole),
                float(periodicity),
                phase.value_in_unit(api.unit.radian),
            )
        )
    return _term_payload(indices, parameters, reversible=True)


def _openmm_improper_payload(force: Any) -> dict[str, Any]:
    indices = []
    parameters = []
    for index in range(force.getNumTorsions()):
        left, center_left, center_right, right, values = force.getTorsionParameters(index)
        indices.append((int(left), int(center_left), int(center_right), int(right)))
        parameters.append((float(values[0]), float(values[1])))
    return _term_payload(indices, parameters, reversible=False)


def _exception_payload(
    pairs: Any,
    charge_product: Any,
    sigma: Any,
    epsilon: Any,
) -> dict[str, Any]:
    pair_array = np.sort(np.asarray(pairs, dtype=np.int32), axis=1)
    charge_array = np.asarray(charge_product, dtype=np.float32).astype(np.float64)
    sigma_array = np.asarray(sigma, dtype=np.float32).astype(np.float64)
    epsilon_array = np.asarray(epsilon, dtype=np.float32).astype(np.float64)
    if pair_array.shape[0] != charge_array.shape[0]:
        raise GPCRmdParityBlocked("exception_array_length_mismatch")
    order = np.lexsort((pair_array[:, 1], pair_array[:, 0]))
    pair_array = pair_array[order]
    charge_array = charge_array[order]
    sigma_array = sigma_array[order]
    epsilon_array = epsilon_array[order]
    excluded = (np.abs(charge_array) <= 1.0e-12) & (np.abs(epsilon_array) <= 1.0e-12)
    sigma_array = np.where(excluded, 0.0, sigma_array)
    parameters = np.column_stack(
        [
            _canonical_float(charge_array, decimals=6),
            _canonical_float(sigma_array, decimals=5),
            _canonical_float(epsilon_array, decimals=5),
        ]
    )
    return {
        "exception_count": int(pair_array.shape[0]),
        "exclusion_count": int(np.count_nonzero(excluded)),
        "active_exception_count": int(np.count_nonzero(~excluded)),
        "exception_pairs_hash": array_hash(pair_array),
        "exception_parameter_hash": array_hash(parameters),
    }


def _constraint_payload(pairs: Any, distances: Any) -> dict[str, Any]:
    pair_array = np.sort(np.asarray(pairs, dtype=np.int32), axis=1)
    distance_array = _canonical_float(
        np.asarray(distances, dtype=np.float32),
        decimals=5,
    )
    if pair_array.size == 0:
        pair_array = np.empty((0, 2), dtype=np.int32)
        distance_array = np.asarray([], dtype=np.float64)
    order = np.lexsort((pair_array[:, 1], pair_array[:, 0])) if pair_array.size else []
    pair_array = pair_array[order]
    distance_array = distance_array[order]
    return {
        "count": int(pair_array.shape[0]),
        "pairs_hash": array_hash(pair_array),
        "distance_hash": array_hash(distance_array),
    }


def _particle_payload(
    *,
    identity: np.ndarray,
    positions: Any,
    original_masses: Any,
    transformed_masses: Any,
    charges: Any,
    sigma: Any,
    epsilon: Any,
    source_net_charge: float,
) -> dict[str, Any]:
    lj = np.column_stack(
        [
            _canonical_float(np.asarray(sigma, dtype=np.float32), decimals=5),
            _canonical_float(np.asarray(epsilon, dtype=np.float32), decimals=5),
        ]
    )
    return {
        "identity_hash": array_hash(np.asarray(identity, dtype="<U32")),
        "coordinate_hash": array_hash(np.asarray(positions, dtype=np.float32)),
        "original_mass_hash": array_hash(np.asarray(original_masses, dtype=np.float64)),
        "transformed_mass_hash": array_hash(
            np.asarray(transformed_masses, dtype=np.float64)
        ),
        "charge_hash": array_hash(
            _canonical_float(np.asarray(charges, dtype=np.float32), decimals=6)
        ),
        "lj_particle_hash": array_hash(lj),
        "source_net_charge_e": _rounded(source_net_charge, 10),
    }


def _cell_payload(matrix: Any) -> dict[str, Any]:
    canonical = _canonical_float(np.asarray(matrix, dtype=np.float64), decimals=5)
    return {
        "matrix_angstrom": canonical.tolist(),
        "matrix_hash": array_hash(canonical),
        "shape": "orthorhombic",
    }


def _hmr_manifest(
    *,
    original_masses: np.ndarray,
    transformed_masses: np.ndarray,
    source: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(source.get("policy", {}))
    status = str(source.get("status", "not_applied"))
    selected = source.get("selected_hydrogens", ())
    selected_count = int(
        source.get("selected_hydrogen_count", len(selected) if selected else 0)
    )
    return {
        "status": status,
        "selection": policy.get("selection", source.get("selection", "none")),
        "target_hydrogen_mass_da": (
            None
            if status == "not_applied"
            else _rounded(
                float(
                    policy.get(
                        "target_hydrogen_mass",
                        source.get("target_hydrogen_mass", 0.0),
                    )
                ),
                6,
            )
        ),
        "selected_hydrogen_count": selected_count,
        "original_mass_hash": array_hash(np.asarray(original_masses, dtype=np.float64)),
        "transformed_mass_hash": array_hash(
            np.asarray(transformed_masses, dtype=np.float64)
        ),
    }


def _protocol_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "ensemble": str(protocol.get("ensemble", "NVT")),
        "fixed_cell": bool(protocol.get("fixed_cell", True)),
        "time_step_fs": _rounded(float(protocol.get("time_step_fs", 1.0)), 6),
    }


def _pme_manifest(config: PMEConfig, *, ewald_error_tolerance: float) -> dict[str, Any]:
    return {
        "method": "PME",
        "real_cutoff_angstrom": _rounded(float(config.real_cutoff), 6),
        "alpha_per_angstrom": _rounded(float(config.alpha), 7),
        "mesh_shape": list(config.mesh_shape),
        "assignment_order": int(config.assignment_order),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
        "charge_tolerance_e": _rounded(float(config.charge_tolerance), 8),
        "ewald_error_tolerance": _rounded(float(ewald_error_tolerance), 8),
        "coulomb_constant_kj_mol_angstrom": _rounded(
            COULOMB_CONSTANT_KJ_MOL_ANGSTROM, 9
        ),
    }


def _prepared_pme_config(prepared: PreparedSystem) -> PMEConfig:
    return PMEConfig(
        mesh_shape=tuple(int(value) for value in prepared.pme_mesh_shape.tolist()),
        alpha=float(prepared.pme_alpha[0]),
        real_cutoff=float(prepared.pme_real_cutoff[0]),
        assignment_order=int(prepared.pme_assignment_order[0]),
        charge_tolerance=float(prepared.pme_charge_tolerance[0]),
        deconvolve_assignment=bool(prepared.pme_deconvolve_assignment[0]),
        background_policy=str(prepared.pme_background_policy[0]),
    )


def _prepared_identity(prepared: PreparedSystem) -> np.ndarray:
    return np.column_stack(
        [
            np.arange(prepared.atom_count).astype(str),
            np.asarray(prepared.atom_names, dtype=str),
            np.asarray(prepared.atom_types, dtype=str),
            np.asarray(prepared.residue_names, dtype=str),
            np.asarray(prepared.residue_ids, dtype=np.int32).astype(str),
            np.asarray(prepared.chain_ids, dtype=str),
        ]
    )


def _read_charmm_psf_identity(path: Path) -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    atom_count = None
    first_atom_line = None
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) >= 2 and fields[1].upper().startswith("!NATOM"):
            try:
                atom_count = int(fields[0])
            except ValueError as exc:
                raise GPCRmdParityBlocked("source_psf_malformed_atom_count") from exc
            first_atom_line = index + 1
            break
    if atom_count is None or first_atom_line is None:
        raise GPCRmdParityBlocked("source_psf_missing_atom_section")
    rows: list[list[str]] = []
    for line in lines[first_atom_line:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 8:
            raise GPCRmdParityBlocked("source_psf_malformed_atom_record")
        try:
            atom_index = int(fields[0]) - 1
        except ValueError as exc:
            raise GPCRmdParityBlocked("source_psf_malformed_atom_record") from exc
        rows.append(
            [
                str(atom_index),
                fields[4],
                fields[5],
                fields[3],
                fields[2],
                fields[1],
            ]
        )
        if len(rows) == atom_count:
            break
    if len(rows) != atom_count:
        raise GPCRmdParityBlocked("source_psf_incomplete_atom_section")
    if any(int(row[0]) != index for index, row in enumerate(rows)):
        raise GPCRmdParityBlocked("source_psf_nonsequential_atom_indices")
    return np.asarray(rows, dtype=str)


def _validate_openmm_source_order(psf: Any, source_identity: np.ndarray) -> None:
    if len(psf.atom_list) != source_identity.shape[0]:
        raise GPCRmdParityBlocked("openmm_source_identity_atom_count_mismatch")
    openmm_stable = np.asarray(
        [
            [
                str(atom.idx),
                str(atom.attype),
                str(atom.residue.idx),
                str(atom.system),
            ]
            for atom in psf.atom_list
        ],
        dtype=str,
    )
    source_stable = source_identity[:, [0, 2, 4, 5]]
    if not np.array_equal(openmm_stable, source_stable):
        raise GPCRmdParityBlocked("openmm_source_particle_order_mismatch")


def _openmm_source_lj_arrays(psf: Any) -> tuple[np.ndarray, np.ndarray]:
    factor = 2.0 * 2 ** (-1.0 / 6.0)
    sigma = np.asarray(
        [factor * float(atom.type.rmin) for atom in psf.atom_list], dtype=np.float64
    )
    epsilon = np.asarray(
        [abs(float(atom.type.epsilon)) * 4.184 for atom in psf.atom_list],
        dtype=np.float64,
    )
    return sigma, epsilon


def _apply_source_masses_and_hmr(
    api: OpenMMApi,
    *,
    system: Any,
    psf: Any,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    original = np.asarray(
        [
            np.float32(atom.mass.value_in_unit(api.unit.dalton))
            for atom in psf.atom_list
        ],
        dtype=np.float64,
    )
    transformed = original.copy()
    hmr_protocol = dict(protocol.get("hydrogen_mass_repartitioning", {}))
    if hmr_protocol.get("status") == "not_applied" or not hmr_protocol:
        for index, mass in enumerate(transformed):
            system.setParticleMass(index, float(mass) * api.unit.dalton)
        return original, transformed, {
            "status": "not_applied",
            "selection": "none",
            "selected_hydrogen_count": 0,
        }
    target = float(hmr_protocol["target_hydrogen_mass"])
    selected: list[dict[str, int]] = []
    for atom in psf.atom_list:
        if int(atom.type.atomic_number) != 1:
            continue
        heavy = sorted(
            {
                bond.atom1.idx if bond.atom2.idx == atom.idx else bond.atom2.idx
                for bond in atom.bonds
                if int(
                    (bond.atom1 if bond.atom2.idx == atom.idx else bond.atom2).type.atomic_number
                )
                != 1
            }
        )
        if not heavy:
            raise GPCRmdParityBlocked(f"openmm_hmr_missing_heavy_partner:{atom.idx}")
        heavy_index = int(heavy[0])
        delta = target - original[atom.idx]
        transformed[atom.idx] += delta
        transformed[heavy_index] -= delta
        selected.append({"hydrogen_index": int(atom.idx), "heavy_atom_index": heavy_index})
    expected = int(hmr_protocol.get("hydrogen_count", len(selected)))
    if len(selected) != expected:
        raise GPCRmdParityBlocked(f"openmm_hmr_count_mismatch:{len(selected)}:{expected}")
    for index, mass in enumerate(transformed):
        system.setParticleMass(index, float(mass) * api.unit.dalton)
    return original, transformed, {
        "status": "represented_by_masses",
        "selection": "all_bonded_hydrogens",
        "target_hydrogen_mass": target,
        "selected_hydrogen_count": len(selected),
        "selected_hydrogens": selected,
        "policy": {
            "selection": "all_bonded_hydrogens",
            "target_hydrogen_mass": target,
        },
    }


def _generate_openmm_rtf(
    api: OpenMMApi,
    *,
    psf_path: Path,
    prm_path: Path,
    out: Path,
) -> None:
    psf = api.app.CharmmPsfFile(str(psf_path))
    present_masses: dict[str, float] = {}
    for atom in psf.atom_list:
        key = str(atom.attype).upper()
        mass = float(atom.mass.value_in_unit(api.unit.dalton))
        existing = present_masses.get(key)
        if existing is not None and not math.isclose(existing, mass, abs_tol=1.0e-6):
            raise GPCRmdParityBlocked(f"source_atom_type_mass_mismatch:{key}")
        present_masses[key] = mass
    parameter_types = _nonbonded_parameter_types(prm_path)
    if not set(present_masses).issubset(parameter_types):
        missing = sorted(set(present_masses) - parameter_types)
        raise GPCRmdParityBlocked("source_nonbonded_types_missing:" + ",".join(missing))
    lines = ["* generated from source PSF and PRM atom types", "*", "36 1", ""]
    for index, atom_type in enumerate(sorted(parameter_types), start=1):
        mass = present_masses.get(atom_type, 12.011)
        lines.append(f"MASS {index:5d} {atom_type:<8s} {mass:.8f}")
    lines.extend(["", "END", ""])
    out.write_text("\n".join(lines))


def _nonbonded_parameter_types(path: Path) -> set[str]:
    section = ""
    output: set[str] = set()
    section_names = {
        "BOND",
        "BONDS",
        "ANGLE",
        "ANGLES",
        "DIHEDRAL",
        "DIHEDRALS",
        "IMPROPER",
        "IMPROPERS",
        "CMAP",
        "NBFIX",
        "HBOND",
        "END",
    }
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("*"):
            continue
        fields = line.split()
        keyword = fields[0].upper()
        if keyword in {"NONBONDED", "NBOND", "NBONDED"}:
            section = "NONBONDED"
            continue
        if keyword in section_names:
            section = keyword
            continue
        if section != "NONBONDED" or len(fields) < 4:
            continue
        try:
            float(fields[2])
            float(fields[3])
        except ValueError:
            continue
        output.add(fields[0].upper())
    return output


def _read_acemd_vectors(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 4:
        raise GPCRmdParityBlocked("acemd_binary_too_short")
    little = int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    big = int(np.frombuffer(raw, dtype=">i4", count=1)[0])
    atom_count = little if len(raw) == 4 + little * 24 else big
    if atom_count <= 0 or len(raw) != 4 + atom_count * 24:
        raise GPCRmdParityBlocked("acemd_binary_size_or_count_mismatch")
    dtype = "<f8" if atom_count == little else ">f8"
    values = np.frombuffer(raw, dtype=dtype, offset=4).reshape((atom_count, 3))
    if not np.all(np.isfinite(values)):
        raise GPCRmdParityBlocked("acemd_binary_nonfinite")
    return values.astype(np.float32)


def _read_xsc_matrix(path: Path) -> np.ndarray:
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            matrix = np.asarray([float(value) for value in fields[1:10]]).reshape((3, 3))
        except ValueError:
            continue
        if np.all(np.isfinite(matrix)) and np.linalg.det(matrix) > 0.0:
            return matrix
    raise GPCRmdParityBlocked("source_xsc_box_unreadable")


def _read_gpcrmd_protocol(input_path: Path, log_path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    for raw in input_path.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0].lower()] = fields[1]
    log = log_path.read_text(errors="replace")
    constraint_count = _required_regex_int(log, r"Number of constraints:\s*(\d+)")
    return {
        "ensemble": "NVT",
        "fixed_cell": True,
        "time_step_fs": float(values["timestep"]),
        "nonbonded": {
            "cutoff_angstrom": float(values["cutoff"]),
            "switch_distance_angstrom": float(values["switchdistance"]),
            "switching": values["switching"].lower() == "on",
        },
        "pme": {
            "enabled": values["pme"].lower() == "on",
            "ewald_error_tolerance": _required_regex_float(
                log, r"Ewald tolerance:\s*([0-9.eE+-]+)"
            ),
        },
        "constraints": {
            "enabled": True,
            "count": constraint_count,
            "tolerance": _required_regex_float(
                log, r"Constraint tolerance:\s*([0-9.eE+-]+)"
            ),
        },
        "hydrogen_mass_repartitioning": {
            "status": "represented_by_masses",
            "target_hydrogen_mass": _required_regex_float(
                log, r"New hydrogen mass:\s*([0-9.eE+-]+)"
            ),
            "hydrogen_count": _required_regex_int(
                log, r"Number of hydrogen atoms:\s*(\d+)"
            ),
        },
    }


def _derive_source_pme_config(
    cell_matrix: np.ndarray,
    *,
    cutoff: float,
    tolerance: float,
) -> PMEConfig:
    lengths = np.linalg.norm(cell_matrix, axis=1)
    alpha = math.sqrt(-math.log(2.0 * tolerance)) / cutoff
    mesh = tuple(
        _openmm_opencl_fft_legal_dimension(
            max(
                6,
                int(
                    math.ceil(
                        2.0 * alpha * float(length) / (3.0 * tolerance**0.2)
                    )
                ),
            )
        )
        for length in lengths
    )
    return PMEConfig(
        mesh_shape=mesh,
        alpha=alpha,
        real_cutoff=cutoff,
        assignment_order=5,
        charge_tolerance=1.0e-4,
        deconvolve_assignment=True,
        background_policy="reject_non_neutral",
    )


def _openmm_opencl_fft_legal_dimension(minimum: int) -> int:
    candidate = max(1, int(minimum))
    while True:
        unfactored = candidate
        for factor in OPENMM_OPENCL_FFT_PRIME_FACTORS:
            while unfactored > 1 and unfactored % factor == 0:
                unfactored //= factor
        if unfactored == 1:
            return candidate
        candidate += 1


def _evaluate_openmm_reference(
    api: OpenMMApi,
    *,
    reference: dict[str, Any],
    platform_name: str,
    precision: str,
    config: PMEConfig,
) -> dict[str, Any]:
    system = reference["system"]
    platform = api.mm.Platform.getPlatformByName(platform_name)
    properties: dict[str, str] = {}
    requested: dict[str, str] = {}
    if "Precision" in list(platform.getPropertyNames()):
        requested["Precision"] = precision
    integrator = api.mm.VerletIntegrator(0.001 * api.unit.picoseconds)
    try:
        context = api.mm.Context(system, integrator, platform, requested)
    except Exception as exc:
        raise GPCRmdParityBlocked(
            f"openmm_context_unavailable:{platform_name}:{requested}:{exc}"
        ) from exc
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific.
            properties[name] = f"<unavailable:{exc}>"
    context.setPeriodicBoxVectors(*reference["box_vectors"])
    context.setPositions(reference["positions_angstrom"] * 0.1 * api.unit.nanometer)
    started = time.perf_counter()
    state = context.getState(getEnergy=True, getForces=True)
    elapsed = time.perf_counter() - started
    components: dict[str, float] = {}
    class_occurrence: dict[str, int] = {}
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        name = type(force).__name__
        occurrence = class_occurrence.get(name, 0)
        class_occurrence[name] = occurrence + 1
        component = _openmm_component_name(name, occurrence)
        group_state = context.getState(getEnergy=True, groups={index})
        value = float(
            group_state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
        )
        components[component] = components.get(component, 0.0) + value
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            api.unit.kilojoule_per_mole / api.unit.nanometer
        ),
        dtype=np.float64,
    )
    nonbonded = _find_force(api, system, "NonbondedForce", expected=1)[0]
    alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
    alpha_per_nanometer = (
        float(alpha.value_in_unit(api.unit.nanometer**-1))
        if hasattr(alpha, "value_in_unit")
        else float(alpha)
    )
    resolved_alpha = alpha_per_nanometer / 10.0
    resolved = {
        "alpha_per_angstrom": resolved_alpha,
        "mesh_shape": [int(nx), int(ny), int(nz)],
    }
    result = {
        "total_energy_kj_mol": float(
            state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
        ),
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": forces,
        "evaluation_seconds": elapsed,
        "platform": context.getPlatform().getName(),
        "platform_properties": properties,
        "available_platforms": _available_openmm_platforms(api),
        "version": _openmm_version(api),
        "resolved_pme": resolved,
        "resolved_pme_matches_manifest": bool(
            math.isclose(resolved_alpha, config.alpha, abs_tol=1.0e-7)
            and tuple(resolved["mesh_shape"]) == config.mesh_shape
        ),
    }
    del context
    del integrator
    return result


def _evaluate_mlx_prepared(
    prepared_dir: Path,
    *,
    require_production: bool,
) -> dict[str, Any]:
    artifact = load_prepared_mlx_artifact(
        prepared_dir,
        require_production=require_production,
    )
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=require_production,
        arrays=artifact.arrays,
    ).to_dict()
    system, force_terms, _constraints = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if system.cell is None:
        raise GPCRmdParityBlocked("mlx_artifact_missing_periodic_cell")
    names = tuple(str(getattr(term, "name", type(term).__name__)) for term in force_terms)
    unknown = sorted(set(names) - set(SUPPORTED_MLX_FORCE_NAMES))
    if unknown:
        raise GPCRmdParityBlocked("unknown_mlx_force_terms:" + ",".join(unknown))
    bound_terms = [
        term.bind_pme_plan(system.cell)
        if getattr(term, "electrostatics", None) == "pme"
        else term
        for term in force_terms
    ]
    nonbonded = next(
        (term for term in bound_terms if getattr(term, "electrostatics", None) == "pme"),
        None,
    )
    if nonbonded is None or nonbonded.pme_config is None:
        raise GPCRmdParityBlocked("mlx_artifact_missing_pme_nonbonded_term")
    topology = Topology.from_sequences(
        n_atoms=artifact.atom_count,
        bonds=np.asarray(artifact.arrays["bonds"], dtype=np.int32),
        angles=np.asarray(artifact.arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(artifact.arrays["dihedrals"], dtype=np.int32),
        impropers=np.asarray(artifact.arrays["impropers"], dtype=np.int32),
        partial_charges=np.asarray(artifact.arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray(
            artifact.arrays["nonbonded_exception_pairs"], dtype=np.int32
        ),
        exclude_bonds=True,
        nonbonded_cutoff=float(nonbonded.cutoff),
        eager_nonbonded_pair_limit=0,
    )
    pme_readiness = pme_readiness_report(
        atom_count=artifact.atom_count,
        charges=artifact.arrays["charges"],
        cell_lengths=artifact.arrays["cell_lengths"],
        config=nonbonded.pme_config,
        nonbonded_cutoff=float(nonbonded.cutoff),
        exclusion_count=len(topology.exclusion_set),
        one_four_count=len(topology.one_four_set),
        explicit_exception_count=int(
            np.asarray(artifact.arrays["nonbonded_exception_pairs"]).shape[0]
        ),
    )
    if pme_readiness["status"] != "ready":
        raise GPCRmdParityBlocked(
            "mlx_pme_not_ready:" + ",".join(pme_readiness.get("blockers", ()))
        )
    neighbor_started = time.perf_counter()
    neighbors = build_neighbor_list(
        system.positions,
        system.cell,
        cutoff=float(nonbonded.cutoff),
        skin=0.0,
        backend="mlx_cell_blocks",
        sort_pairs=False,
    )
    neighbor_seconds = time.perf_counter() - neighbor_started
    total_energy = mx.array(0.0, dtype=mx.float32)
    total_forces = mx.zeros_like(system.positions)
    components: dict[str, float] = {}
    term_timings: dict[str, float] = {}
    evaluation_started = time.perf_counter()
    for term in bound_terms:
        name = str(getattr(term, "name", type(term).__name__))
        started = time.perf_counter()
        pairs = neighbors.interactions if name == "nonbonded" else None
        if hasattr(term, "energy_forces_with_components"):
            energy, forces, _term_components = term.energy_forces_with_components(
                system.positions,
                system.cell,
                pairs=pairs,
            )
        else:
            energy, forces = term.energy_forces(
                system.positions,
                system.cell,
                pairs=pairs,
            )
        mx.eval(energy, forces)
        components[_mlx_component_name(name)] = float(np.asarray(energy))
        total_energy = total_energy + energy
        total_forces = total_forces + forces
        mx.eval(total_energy, total_forces)
        term_timings[name] = time.perf_counter() - started
    result = {
        "total_energy_kj_mol": float(np.asarray(total_energy)),
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": np.asarray(total_forces, dtype=np.float64) * 10.0,
        "artifact_readiness": artifact_readiness,
        "pme_readiness": pme_readiness,
        "topology": {
            "pair_policy": getattr(nonbonded.topology, "nonbonded_pair_policy", None),
            "pair_cache_materialized": (
                getattr(nonbonded.topology, "_nonbonded_pairs", None) is not None
            ),
        },
        "neighbor": {
            "backend": neighbors.backend,
            "representation": neighbors.representation_kind,
            "fallback_reason": neighbors.fallback_reason,
            "pair_count": int(neighbors.pair_count),
        },
        "plan": _jsonable(getattr(nonbonded.pme_plan, "diagnostics", None)),
        "timings": {
            "neighbor_build_seconds": neighbor_seconds,
            "force_evaluation_seconds": time.perf_counter() - evaluation_started,
            "force_terms_seconds": term_timings,
        },
        "runtime": asdict(get_runtime_info()),
    }
    return result


def _component_energy_metrics(
    mlx: dict[str, float],
    openmm: dict[str, float],
    *,
    required_components: tuple[str, ...],
    atom_count: int,
) -> dict[str, dict[str, float]]:
    missing = [name for name in required_components if name not in mlx or name not in openmm]
    if missing:
        raise GPCRmdParityBlocked("partial_component_energies:" + ",".join(missing))
    output = {}
    for name in required_components:
        candidate = float(mlx[name])
        reference = float(openmm[name])
        absolute = abs(candidate - reference)
        denominator = abs(reference)
        relative = 0.0 if denominator <= 1.0e-12 and absolute <= 1.0e-12 else (
            math.inf if denominator <= 1.0e-12 else absolute / denominator
        )
        output[name] = {
            "mlx_kj_mol": candidate,
            "openmm_kj_mol": reference,
            "absolute_error_kj_mol": absolute,
            "energy_error_per_atom_kj_mol": absolute / atom_count,
            "relative_error": relative,
        }
    return output


def _parity_checks(
    *,
    metrics: dict[str, float],
    component_metrics: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    openmm_result: dict[str, Any],
    mlx_result: dict[str, Any],
    tolerances: GPCRmdParityTolerances,
) -> dict[str, bool]:
    return {
        "manifest_match": bool(comparison["matched"]),
        "total_energy_per_atom": (
            metrics["energy_error_per_atom_kj_mol"]
            <= tolerances.energy_per_atom_kj_mol
        ),
        "total_relative_energy": (
            metrics["relative_energy_error"] <= tolerances.relative_energy_error
        ),
        "component_energy_per_atom": all(
            values["energy_error_per_atom_kj_mol"]
            <= tolerances.energy_per_atom_kj_mol
            for values in component_metrics.values()
        ),
        "component_relative_energy": all(
            values["relative_error"] <= tolerances.relative_energy_error
            for values in component_metrics.values()
        ),
        "force_rms": metrics["rms_absolute_kj_mol_nm"] <= tolerances.force_rms_kj_mol_nm,
        "force_maximum": (
            metrics["maximum_absolute_kj_mol_nm"]
            <= tolerances.force_maximum_kj_mol_nm
        ),
        "openmm_resolved_pme": bool(openmm_result["resolved_pme_matches_manifest"]),
        "mlx_lazy_topology": mlx_result["topology"]["pair_policy"] == "lazy",
        "mlx_pair_cache_unmaterialized": not mlx_result["topology"][
            "pair_cache_materialized"
        ],
        "mlx_neighbor_blocks": (
            mlx_result["neighbor"]["backend"] == "mlx_cell_blocks"
            and mlx_result["neighbor"]["representation"] == "blocks"
        ),
        "mlx_no_neighbor_fallback": mlx_result["neighbor"]["fallback_reason"] is None,
    }


def _required_components(manifest: dict[str, Any]) -> tuple[str, ...]:
    counts = manifest["forces"]["term_counts"]
    names = [
        name
        for name in (
            "bond",
            "angle",
            "urey_bradley",
            "proper_dihedral",
            "harmonic_improper",
            "cmap",
        )
        if int(counts[name]) > 0
    ]
    names.append("nonbonded")
    return tuple(names)


def _openmm_component_name(force_class: str, occurrence: int) -> str:
    if force_class == "HarmonicBondForce":
        return "bond" if occurrence == 0 else "urey_bradley"
    mapping = {
        "HarmonicAngleForce": "angle",
        "PeriodicTorsionForce": "proper_dihedral",
        "CustomTorsionForce": "harmonic_improper",
        "CMAPTorsionForce": "cmap",
        "NonbondedForce": "nonbonded",
        "CustomNonbondedForce": "nonbonded",
    }
    try:
        return mapping[force_class]
    except KeyError as exc:
        raise UnsupportedOpenMMForceError(force_class) from exc


def _mlx_component_name(name: str) -> str:
    mapping = {
        "bond": "bond",
        "angle": "angle",
        "urey_bradley": "urey_bradley",
        "dihedral": "proper_dihedral",
        "improper": "harmonic_improper",
        "charmm_cmap_terms": "cmap",
        "nonbonded": "nonbonded",
    }
    return mapping[name]


def _compare_manifests(mlx: dict[str, Any], openmm: dict[str, Any]) -> dict[str, Any]:
    mismatches = manifest_mismatches(mlx, openmm, fields=MANIFEST_FIELDS)
    return {
        "status": "matched" if not mismatches else "mismatched",
        "matched": not mismatches,
        "required_fields": list(MANIFEST_FIELDS),
        "mismatches": mismatches,
        "mlx_manifest_hash": mlx["manifest_hash"],
        "openmm_manifest_hash": openmm["manifest_hash"],
    }


def _finalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    payload["manifest_hash"] = manifest_hash(payload)
    return payload


def _find_force(
    api: OpenMMApi,
    system: Any,
    class_name: str,
    *,
    expected: int | None = None,
) -> list[Any]:
    del api
    forces = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if type(system.getForce(index)).__name__ == class_name
    ]
    if expected is not None and len(forces) != expected:
        raise UnsupportedOpenMMForceError(
            f"expected {expected} {class_name}, found {len(forces)}"
        )
    return forces


def _validate_openmm_force_inventory(system: Any) -> None:
    actual = [type(system.getForce(index)).__name__ for index in range(system.getNumForces())]
    unknown = sorted(set(actual) - set(SUPPORTED_OPENMM_FORCE_CLASSES))
    if unknown:
        raise UnsupportedOpenMMForceError("unknown_force_classes:" + ",".join(unknown))


def _openmm_exception_arrays(api: OpenMMApi, force: Any) -> dict[str, np.ndarray]:
    pairs = []
    charge_product = []
    sigma = []
    epsilon = []
    for index in range(force.getNumExceptions()):
        left, right, charge, sigma_value, epsilon_value = force.getExceptionParameters(index)
        pairs.append((int(left), int(right)))
        charge_product.append(charge.value_in_unit(api.unit.elementary_charge**2))
        sigma.append(sigma_value.value_in_unit(api.unit.angstrom))
        epsilon.append(epsilon_value.value_in_unit(api.unit.kilojoule_per_mole))
    return {
        "pairs": np.asarray(pairs, dtype=np.int32),
        "charge_product": np.asarray(charge_product, dtype=np.float64),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "epsilon": np.asarray(epsilon, dtype=np.float64),
    }


def _openmm_constraint_arrays(api: OpenMMApi, system: Any) -> dict[str, np.ndarray]:
    pairs = []
    distances = []
    for index in range(system.getNumConstraints()):
        left, right, distance = system.getConstraintParameters(index)
        pairs.append((int(left), int(right)))
        distances.append(distance.value_in_unit(api.unit.angstrom))
    return {
        "pairs": np.asarray(pairs, dtype=np.int32).reshape((-1, 2)),
        "distance": np.asarray(distances, dtype=np.float64),
    }


def _require_complete_forces(values: Any, *, atom_count: int, engine: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (atom_count, 3) or not np.all(np.isfinite(array)):
        raise GPCRmdParityBlocked(
            f"partial_or_nonfinite_forces:{engine}:shape={list(array.shape)}"
        )


def _require_openmm_platform(api: OpenMMApi, platform_name: str) -> None:
    available = _available_openmm_platforms(api)
    if platform_name not in available:
        raise GPCRmdParityBlocked(
            f"openmm_platform_unavailable:{platform_name}:available={available}"
        )


def _load_openmm() -> OpenMMApi:
    try:
        import openmm as mm
        from openmm import app, unit
    except Exception as exc:  # pragma: no cover - optional reference dependency.
        raise ImportError(f"OpenMM import unavailable: {exc}") from exc
    return OpenMMApi(mm=mm, app=app, unit=unit)


def _available_openmm_platforms(api: OpenMMApi) -> list[str]:
    return [
        api.mm.Platform.getPlatform(index).getName()
        for index in range(api.mm.Platform.getNumPlatforms())
    ]


def _openmm_version(api: OpenMMApi) -> str:
    return str(
        getattr(api.mm.version, "full_version", None)
        or getattr(api.mm.version, "version", "unknown")
    )


def _required_regex_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise GPCRmdParityBlocked(f"source_protocol_field_missing:{pattern}")
    return float(match.group(1))


def _required_regex_int(text: str, pattern: str) -> int:
    return int(_required_regex_float(text, pattern))


def _config_payload(config: PMEConfig, *, ewald_error_tolerance: float) -> dict[str, Any]:
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha),
        "real_cutoff": float(config.real_cutoff),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
        "ewald_error_tolerance": float(ewald_error_tolerance),
    }


def _canonical_float(values: Any, *, decimals: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64), decimals=decimals)


def _rounded(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_report(
    *,
    fixture: str,
    mlx_prepared: Path,
    source_manifest: Path | None,
    platform_name: str,
    precision: str,
    tolerances: GPCRmdParityTolerances,
    out: Path,
) -> dict[str, Any]:
    return {
        "kind": "mlx_atomistic.gpcrmd_pme_parity",
        "schema_version": 1,
        "fixture": fixture,
        "reference_engine": "openmm",
        "reference_engine_role": OPENMM_REFERENCE_ROLE,
        "mlx_prepared": str(mlx_prepared),
        "source_manifest": None if source_manifest is None else str(source_manifest),
        "requested_openmm_platform": platform_name,
        "requested_openmm_precision": precision,
        "tolerances": asdict(tolerances),
        "out": str(out),
        "runtime": asdict(get_runtime_info()),
        "host": {
            "python": platform_module.python_version(),
            "platform": platform_module.platform(),
        },
        "status": "blocked",
        "passed": False,
        "blockers": [],
    }


def _without_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "forces_kj_mol_nm"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _finish_report(report: dict[str, Any], out: Path) -> dict[str, Any]:
    payload = _jsonable(report)
    _write_json(out / REPORT_NAME, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    """Run the GPCRmd parity command-line interface.

    Args:
        argv: Optional argument list; ``None`` reads process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mlx-prepared", type=Path, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--precision", choices=("single", "mixed", "double"), default="single")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_gpcrmd_pme_parity(
        source_manifest=args.source_manifest,
        cache=args.cache,
        mlx_prepared=args.mlx_prepared,
        platform_name=args.platform,
        precision=args.precision,
        out=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "FORCE_ARRAYS_NAME",
    "MANIFEST_COMPARISON_NAME",
    "MLX_MANIFEST_NAME",
    "OPENMM_MANIFEST_NAME",
    "REPORT_NAME",
    "GPCRmdParityBlocked",
    "GPCRmdParityTolerances",
    "evaluate_small_charmm_pme_fixture",
    "run_gpcrmd_pme_parity",
]
