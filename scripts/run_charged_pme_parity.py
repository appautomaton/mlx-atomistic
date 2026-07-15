"""Run independent charged JAC PME parity between MLX and OpenMM."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform as platform_module
import time
from copy import deepcopy
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
    PMEManifestMismatchError,
    array_hash,
    force_error_metrics,
    manifest_hash,
    manifest_mismatches,
    require_matching_manifest,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces, pme_readiness_report
from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system
from mlx_atomistic.prep.schema import ARTIFACT_VERSION, PreparedSystem
from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.topology import Topology
from mlx_atomistic.units import COULOMB_CONSTANT_KJ_MOL_ANGSTROM

REPORT_NAME = "charged_pme_parity_report.json"
MLX_MANIFEST_NAME = "mlx_workload_manifest.json"
OPENMM_MANIFEST_NAME = "openmm_workload_manifest.json"
MANIFEST_COMPARISON_NAME = "manifest_comparison.json"
FORCE_ARRAYS_NAME = "complete_force_comparison.npz"
NORMALIZED_PREPARED_DIR = "mlx-prepared-normalized"
OPENMM_REFERENCE_ROLE = "reference-only validation; not a product runtime dependency"

DEFAULT_BASE_MESH = (64, 64, 64)
DEFAULT_ALPHA_PER_ANGSTROM = 0.35
DEFAULT_CUTOFF_ANGSTROM = 9.0
DEFAULT_ASSIGNMENT_ORDER = 5
DEFAULT_BACKGROUND_POLICY = "uniform_neutralizing_plasma"
DEFAULT_CHARGE_TOLERANCE = 1.0e-5
SUPPORTED_OPENMM_FORCE_CLASSES = (
    "HarmonicBondForce",
    "HarmonicAngleForce",
    "PeriodicTorsionForce",
    "NonbondedForce",
)
MANIFEST_FIELDS = (
    "schema_version",
    "workload.name",
    "workload.operation",
    "workload.atom_count",
    "workload.replicas",
    "workload.replica_order",
    "topology.atom_order_hash",
    "topology.coordinate_hash",
    "topology.mass_hash",
    "topology.charge_hash",
    "topology.lj_particle_hash",
    "topology.exception_pairs_hash",
    "topology.exception_parameter_hash",
    "topology.exclusion_pairs_hash",
    "topology.particle_count",
    "topology.constraint_count",
    "topology.exception_count",
    "topology.exclusion_count",
    "topology.active_exception_count",
    "topology.net_charge_e",
    "forces.class_counts",
    "forces.term_counts",
    "cell.lengths_angstrom",
    "cell.matrix_angstrom",
    "pme.method",
    "pme.real_cutoff_angstrom",
    "pme.alpha_per_angstrom",
    "pme.mesh_shape",
    "pme.assignment_order",
    "pme.deconvolve_assignment",
    "pme.background_policy",
    "pme.coulomb_constant_kj_mol_angstrom",
    "nonbonded.lj_dispersion_correction",
    "nonbonded.switching_function",
)


class ChargedPMEParityError(RuntimeError):
    """Raised when charged parity cannot preserve the approved workload contract."""


class UnsupportedOpenMMForceError(ChargedPMEParityError):
    """Raised when the AMBER source creates an unapproved OpenMM force class."""


@dataclass(frozen=True)
class ChargedPMETolerances:
    """Acceptance thresholds for JAC charged-PME parity."""

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


def run_charged_pme_parity(
    *,
    mlx_prepared: str | Path,
    amber_prmtop: str | Path,
    amber_coordinates: str | Path,
    replicas: object,
    platform_name: str,
    out: str | Path,
    precision: str = "single",
    tolerances: ChargedPMETolerances | None = None,
) -> dict[str, Any]:
    """Run one independent AMBER/OpenMM-vs-MLX charged PME comparison.

    Args:
        mlx_prepared: Existing MLX prepared artifact for the requested replica shape.
        amber_prmtop: Independent AMBER topology used by OpenMM.
        amber_coordinates: Independent AMBER coordinates used by OpenMM.
        replicas: Three positive replica counts ``(nx, ny, nz)``.
        platform_name: OpenMM platform name, normally ``"OpenCL"``.
        out: Caller-owned output directory.
        precision: Requested OpenMM precision when the platform exposes it.
        tolerances: Optional parity thresholds.

    Returns:
        JSON-serializable passed, failed, or blocked parity report.
    """

    tolerance = ChargedPMETolerances() if tolerances is None else tolerances
    replica_shape = _normalize_replicas(replicas)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    prepared_path = Path(mlx_prepared)
    prmtop_path = Path(amber_prmtop)
    coordinates_path = Path(amber_coordinates)
    base = _base_report(
        prepared_path=prepared_path,
        prmtop_path=prmtop_path,
        coordinates_path=coordinates_path,
        replicas=replica_shape,
        platform_name=platform_name,
        precision=precision,
        tolerances=tolerance,
        out_path=out_path,
    )
    missing = _missing_inputs(prepared_path, prmtop_path, coordinates_path)
    if missing:
        return _finish_report(
            {
                **base,
                "status": "blocked",
                "passed": False,
                "blockers": [f"missing_input:{path}" for path in missing],
            },
            out_path,
        )

    try:
        api = _load_openmm()
        source = _load_openmm_source(
            api,
            prmtop_path=prmtop_path,
            coordinates_path=coordinates_path,
        )
        config = _jac_pme_config(replica_shape)
        expected_cell = source["base_cell_lengths_angstrom"] * np.asarray(
            replica_shape,
            dtype=np.float64,
        )
        prepared = _normalize_mlx_prepared(
            load_prepared_system(prepared_path),
            source_atom_count=source["atom_count"],
            replicas=replica_shape,
            expected_cell_lengths=expected_cell,
            config=config,
        )
        normalized_dir = out_path / NORMALIZED_PREPARED_DIR
        save_prepared_system(prepared, normalized_dir)

        reference = _build_openmm_replicas(
            api,
            source=source,
            replicas=replica_shape,
            config=config,
        )
        mlx_manifest = _mlx_manifest(
            prepared,
            source_atom_count=source["atom_count"],
            replicas=replica_shape,
            config=config,
        )
        openmm_manifest = _openmm_manifest(
            api,
            reference=reference,
            replicas=replica_shape,
            config=config,
            platform_name=platform_name,
            precision=precision,
        )
        comparison = _compare_manifests(mlx_manifest, openmm_manifest)
        _write_json(out_path / MLX_MANIFEST_NAME, mlx_manifest)
        _write_json(out_path / OPENMM_MANIFEST_NAME, openmm_manifest)
        _write_json(out_path / MANIFEST_COMPARISON_NAME, comparison)
        require_matching_manifest(
            mlx_manifest,
            openmm_manifest,
            fields=MANIFEST_FIELDS,
        )

        small_gate = evaluate_small_charged_fixture(
            platform_name=platform_name,
            precision=precision,
            tolerances=tolerance,
        )
        if not small_gate["passed"]:
            raise ChargedPMEParityError("small charged analytic/OpenMM PME gate failed")

        openmm_result = _evaluate_openmm_reference(
            api,
            reference=reference,
            platform_name=platform_name,
            precision=precision,
            config=config,
        )
        del reference["system"]
        gc.collect()

        mlx_result = _evaluate_mlx_prepared(normalized_dir)
        metrics = force_error_metrics(
            mlx_result["forces_kj_mol_nm"],
            openmm_result["forces_kj_mol_nm"],
            candidate_energy=mlx_result["total_energy_kj_mol"],
            reference_energy=openmm_result["total_energy_kj_mol"],
        )
        component_metrics = _component_energy_metrics(
            mlx_result["component_energy_kj_mol"],
            openmm_result["component_energy_kj_mol"],
            atom_count=prepared.atom_count,
        )
        force_delta = (
            np.asarray(mlx_result["forces_kj_mol_nm"], dtype=np.float64)
            - np.asarray(openmm_result["forces_kj_mol_nm"], dtype=np.float64)
        )
        force_path = out_path / FORCE_ARRAYS_NAME
        np.savez(
            force_path,
            mlx_forces_kj_mol_nm=np.asarray(mlx_result["forces_kj_mol_nm"], dtype=np.float32),
            openmm_forces_kj_mol_nm=np.asarray(
                openmm_result["forces_kj_mol_nm"],
                dtype=np.float32,
            ),
            force_delta_kj_mol_nm=force_delta.astype(np.float32),
        )

        checks = _parity_checks(
            metrics=asdict(metrics),
            component_metrics=component_metrics,
            small_gate=small_gate,
            manifest_comparison=comparison,
            openmm_result=openmm_result,
            config=config,
            tolerances=tolerance,
        )
        checks.update(
            {
                "mlx_lazy_topology": mlx_result["topology"]["pair_policy"] == "lazy",
                "mlx_pair_cache_unmaterialized": not mlx_result["topology"][
                    "pair_cache_materialized"
                ],
                "mlx_neighbor_blocks": (
                    mlx_result["neighbor"]["backend"] == "mlx_cell_blocks"
                    and mlx_result["neighbor"]["representation"] == "blocks"
                ),
                "mlx_no_neighbor_fallback": (
                    mlx_result["neighbor"]["fallback_reason"] is None
                ),
            }
        )
        passed = all(checks.values())
        report = {
            **base,
            "status": "passed" if passed else "failed",
            "passed": passed,
            "blockers": [] if passed else [name for name, value in checks.items() if not value],
            "atom_count": prepared.atom_count,
            "small_charged_gate": small_gate,
            "manifest_comparison": comparison,
            "manifests": {
                "mlx": str(out_path / MLX_MANIFEST_NAME),
                "openmm": str(out_path / OPENMM_MANIFEST_NAME),
            },
            "normalized_prepared": str(normalized_dir),
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
            "pme": _config_payload(config),
        }
        return _finish_report(report, out_path)
    except PMEManifestMismatchError as exc:
        return _finish_report(
            {
                **base,
                "status": "failed",
                "passed": False,
                "blockers": ["manifest_mismatch"],
                "manifest_mismatches": exc.mismatches,
            },
            out_path,
        )
    except (ChargedPMEParityError, UnsupportedOpenMMForceError, ValueError) as exc:
        return _finish_report(
            {
                **base,
                "status": "failed",
                "passed": False,
                "blockers": [f"{type(exc).__name__}:{exc}"],
            },
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


def evaluate_small_charged_fixture(
    *,
    platform_name: str = "Reference",
    precision: str = "single",
    tolerances: ChargedPMETolerances | None = None,
) -> dict[str, Any]:
    """Run a four-particle analytic-background and OpenMM PME gate.

    Args:
        platform_name: OpenMM platform used for the fixture.
        precision: Requested platform precision when supported.
        tolerances: Optional parity thresholds.

    Returns:
        JSON-serializable small-fixture metrics and pass/fail checks.
    """

    api = _load_openmm()
    tolerance = ChargedPMETolerances() if tolerances is None else tolerances
    positions = np.asarray(
        [
            [2.1, 3.2, 4.3],
            [7.4, 6.1, 8.2],
            [11.3, 13.5, 9.7],
            [15.6, 10.2, 14.4],
        ],
        dtype=np.float32,
    )
    charges = np.asarray([1.0, -0.7, 0.4, 0.3], dtype=np.float32)
    cell_length = 24.0
    config = PMEConfig(
        mesh_shape=(32, 32, 32),
        alpha=DEFAULT_ALPHA_PER_ANGSTROM,
        real_cutoff=8.0,
        assignment_order=DEFAULT_ASSIGNMENT_ORDER,
        charge_tolerance=DEFAULT_CHARGE_TOLERANCE,
        deconvolve_assignment=True,
        background_policy=DEFAULT_BACKGROUND_POLICY,
    )
    energy, forces, components = pme_coulomb_energy_forces(
        mx.array(positions),
        mx.array(charges),
        Cell.cubic(cell_length),
        coulomb_constant=COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
        config=config,
    )
    mx.eval(energy, forces)
    mlx_energy = float(np.asarray(energy))
    mlx_forces = np.asarray(forces, dtype=np.float64) * 10.0
    background = float(np.asarray(components["coulomb_background"]))
    expected_background = -COULOMB_CONSTANT_KJ_MOL_ANGSTROM * math.pi * float(
        np.sum(charges, dtype=np.float64)
    ) ** 2 / (2.0 * cell_length**3 * config.alpha**2)

    system = api.mm.System()
    nonbonded = api.mm.NonbondedForce()
    nonbonded.setNonbondedMethod(api.mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(config.real_cutoff * 0.1 * api.unit.nanometer)
    nonbonded.setPMEParameters(
        config.alpha * 10.0 / api.unit.nanometer,
        *config.mesh_shape,
    )
    nonbonded.setUseDispersionCorrection(False)
    for charge in charges:
        system.addParticle(1.0 * api.unit.dalton)
        nonbonded.addParticle(
            float(charge) * api.unit.elementary_charge,
            0.1 * api.unit.nanometer,
            0.0 * api.unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    box = _box_vectors(api, np.asarray([cell_length] * 3, dtype=np.float64))
    system.setDefaultPeriodicBoxVectors(*box)
    context, properties = _openmm_context(
        api,
        system=system,
        platform_name=platform_name,
        precision=precision,
    )
    context.setPeriodicBoxVectors(*box)
    context.setPositions(positions * 0.1 * api.unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    openmm_energy = float(
        state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
    )
    openmm_forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            api.unit.kilojoule_per_mole / api.unit.nanometer
        ),
        dtype=np.float64,
    )
    metrics = force_error_metrics(
        mlx_forces,
        openmm_forces,
        candidate_energy=mlx_energy,
        reference_energy=openmm_energy,
    )
    background_abs_error = abs(background - expected_background)
    background_relative_error = background_abs_error / abs(expected_background)
    checks = {
        "analytic_background_absolute": background_abs_error <= 1.0e-4,
        "analytic_background_relative": background_relative_error <= 1.0e-5,
        "energy_per_atom": (
            metrics.energy_error_per_atom_kj_mol <= tolerance.energy_per_atom_kj_mol
        ),
        "relative_energy": metrics.relative_energy_error <= tolerance.relative_energy_error,
        "force_rms": metrics.rms_absolute_kj_mol_nm <= tolerance.force_rms_kj_mol_nm,
        "force_maximum": (
            metrics.maximum_absolute_kj_mol_nm <= tolerance.force_maximum_kj_mol_nm
        ),
    }
    del context
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "atom_count": len(charges),
        "net_charge_e": float(np.sum(charges, dtype=np.float64)),
        "background_energy_mlx_kj_mol": background,
        "background_energy_analytic_kj_mol": expected_background,
        "background_abs_error_kj_mol": background_abs_error,
        "background_relative_error": background_relative_error,
        "total_energy_mlx_kj_mol": mlx_energy,
        "total_energy_openmm_kj_mol": openmm_energy,
        "force_metrics": asdict(metrics),
        "platform": platform_name,
        "platform_properties": properties,
        "pme": _config_payload(config),
    }


def _load_openmm() -> OpenMMApi:
    try:
        import openmm as mm
        from openmm import app, unit
    except Exception as exc:  # pragma: no cover - optional reference package.
        msg = f"OpenMM import unavailable: {exc}"
        raise ImportError(msg) from exc
    return OpenMMApi(mm=mm, app=app, unit=unit)


def _load_openmm_source(
    api: OpenMMApi,
    *,
    prmtop_path: Path,
    coordinates_path: Path,
) -> dict[str, Any]:
    prmtop = api.app.AmberPrmtopFile(str(prmtop_path))
    coordinates = api.app.AmberInpcrdFile(str(coordinates_path))
    if coordinates.boxVectors is None:
        raise ChargedPMEParityError("AMBER coordinates do not contain periodic box vectors")
    base_cell = _orthorhombic_lengths_angstrom(api, coordinates.boxVectors)
    positions = np.asarray(
        coordinates.positions.value_in_unit(api.unit.angstrom),
        dtype=np.float32,
    )
    atom_count = prmtop.topology.getNumAtoms()
    if positions.shape != (atom_count, 3):
        msg = f"AMBER coordinate count {positions.shape[0]} != topology count {atom_count}"
        raise ChargedPMEParityError(msg)

    system_kwargs = {
        "nonbondedMethod": api.app.PME,
        "nonbondedCutoff": DEFAULT_CUTOFF_ANGSTROM * 0.1 * api.unit.nanometer,
        "removeCMMotion": False,
    }
    full_system = prmtop.createSystem(
        **system_kwargs,
        constraints=None,
        rigidWater=False,
    )
    constrained_system = prmtop.createSystem(
        **system_kwargs,
        constraints=api.app.HBonds,
        rigidWater=True,
    )
    force_classes = tuple(
        type(full_system.getForce(index)).__name__
        for index in range(full_system.getNumForces())
    )
    unknown = tuple(
        name for name in force_classes if name not in SUPPORTED_OPENMM_FORCE_CLASSES
    )
    missing = tuple(
        name for name in SUPPORTED_OPENMM_FORCE_CLASSES if name not in force_classes
    )
    if unknown or missing:
        details = {"unknown": unknown, "missing": missing, "actual": force_classes}
        raise UnsupportedOpenMMForceError(json.dumps(details, sort_keys=True))
    if any(full_system.isVirtualSite(index) for index in range(atom_count)):
        raise UnsupportedOpenMMForceError("JAC comparator does not support virtual sites")
    identity = _openmm_source_identity(prmtop)
    return {
        "prmtop": prmtop,
        "coordinates": coordinates,
        "full_system": full_system,
        "constrained_system": constrained_system,
        "positions_angstrom": positions,
        "base_cell_lengths_angstrom": base_cell,
        "atom_count": atom_count,
        "identity": identity,
        "force_classes": force_classes,
    }


def _build_openmm_replicas(
    api: OpenMMApi,
    *,
    source: dict[str, Any],
    replicas: tuple[int, int, int],
    config: PMEConfig,
) -> dict[str, Any]:
    source_system = source["full_system"]
    constraint_system = source["constrained_system"]
    source_atom_count = int(source["atom_count"])
    replica_count = int(np.prod(replicas, dtype=np.int64))
    system = api.mm.System()
    for _replica in range(replica_count):
        for atom_index in range(source_atom_count):
            system.addParticle(source_system.getParticleMass(atom_index))

    for replica_index in range(replica_count):
        offset = replica_index * source_atom_count
        for constraint_index in range(constraint_system.getNumConstraints()):
            left, right, distance = constraint_system.getConstraintParameters(constraint_index)
            system.addConstraint(left + offset, right + offset, distance)

    force_components: dict[int, str] = {}
    for force_index in range(source_system.getNumForces()):
        source_force = source_system.getForce(force_index)
        force = _clone_openmm_force(
            api,
            source_force=source_force,
            source_atom_count=source_atom_count,
            replica_count=replica_count,
            config=config,
        )
        force.setForceGroup(force_index)
        if isinstance(force, api.mm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(force_index)
        system.addForce(force)
        force_components[force_index] = _component_name(type(force).__name__)

    cell_lengths = source["base_cell_lengths_angstrom"] * np.asarray(
        replicas,
        dtype=np.float64,
    )
    box_vectors = _box_vectors(api, cell_lengths)
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    translations = _replica_translations(source["base_cell_lengths_angstrom"], replicas)
    positions = np.concatenate(
        [source["positions_angstrom"] + translation for translation in translations],
        axis=0,
    ).astype(np.float32)
    identity = np.concatenate(
        [
            _identity_with_replica(source["identity"], replica_index)
            for replica_index in range(replica_count)
        ],
        axis=0,
    )
    return {
        "system": system,
        "positions_angstrom": positions,
        "cell_lengths_angstrom": cell_lengths,
        "box_vectors": box_vectors,
        "identity": identity,
        "force_components": force_components,
        "source_force_classes": list(source["force_classes"]),
    }


def _clone_openmm_force(
    api: OpenMMApi,
    *,
    source_force: Any,
    source_atom_count: int,
    replica_count: int,
    config: PMEConfig,
) -> Any:
    force_name = type(source_force).__name__
    if force_name == "HarmonicBondForce":
        force = api.mm.HarmonicBondForce()
        for replica_index in range(replica_count):
            offset = replica_index * source_atom_count
            for index in range(source_force.getNumBonds()):
                left, right, length, stiffness = source_force.getBondParameters(index)
                force.addBond(left + offset, right + offset, length, stiffness)
    elif force_name == "HarmonicAngleForce":
        force = api.mm.HarmonicAngleForce()
        for replica_index in range(replica_count):
            offset = replica_index * source_atom_count
            for index in range(source_force.getNumAngles()):
                left, center, right, angle, stiffness = source_force.getAngleParameters(index)
                force.addAngle(left + offset, center + offset, right + offset, angle, stiffness)
    elif force_name == "PeriodicTorsionForce":
        force = api.mm.PeriodicTorsionForce()
        for replica_index in range(replica_count):
            offset = replica_index * source_atom_count
            for index in range(source_force.getNumTorsions()):
                left, center_left, center_right, right, periodicity, phase, stiffness = (
                    source_force.getTorsionParameters(index)
                )
                force.addTorsion(
                    left + offset,
                    center_left + offset,
                    center_right + offset,
                    right + offset,
                    periodicity,
                    phase,
                    stiffness,
                )
    elif force_name == "NonbondedForce":
        force = _clone_nonbonded_force(
            api,
            source_force=source_force,
            source_atom_count=source_atom_count,
            replica_count=replica_count,
            config=config,
        )
    else:
        raise UnsupportedOpenMMForceError(force_name)
    force.setName(source_force.getName())
    return force


def _clone_nonbonded_force(
    api: OpenMMApi,
    *,
    source_force: Any,
    source_atom_count: int,
    replica_count: int,
    config: PMEConfig,
) -> Any:
    unsupported_counts = {
        "global_parameters": source_force.getNumGlobalParameters(),
        "particle_parameter_offsets": source_force.getNumParticleParameterOffsets(),
        "exception_parameter_offsets": source_force.getNumExceptionParameterOffsets(),
    }
    if any(unsupported_counts.values()):
        raise UnsupportedOpenMMForceError(
            "NonbondedForce parameter offsets are unsupported: "
            + json.dumps(unsupported_counts, sort_keys=True)
        )
    force = api.mm.NonbondedForce()
    force.setNonbondedMethod(api.mm.NonbondedForce.PME)
    force.setCutoffDistance(config.real_cutoff * 0.1 * api.unit.nanometer)
    force.setReactionFieldDielectric(source_force.getReactionFieldDielectric())
    force.setUseSwitchingFunction(False)
    force.setUseDispersionCorrection(False)
    force.setEwaldErrorTolerance(source_force.getEwaldErrorTolerance())
    force.setPMEParameters(config.alpha * 10.0 / api.unit.nanometer, *config.mesh_shape)
    force.setIncludeDirectSpace(source_force.getIncludeDirectSpace())
    if hasattr(force, "setExceptionsUsePeriodicBoundaryConditions"):
        force.setExceptionsUsePeriodicBoundaryConditions(
            source_force.getExceptionsUsePeriodicBoundaryConditions()
        )
    for _replica in range(replica_count):
        for atom_index in range(source_atom_count):
            force.addParticle(*source_force.getParticleParameters(atom_index))
    for replica_index in range(replica_count):
        offset = replica_index * source_atom_count
        for exception_index in range(source_force.getNumExceptions()):
            left, right, charge_product, sigma, epsilon = source_force.getExceptionParameters(
                exception_index
            )
            force.addException(
                left + offset,
                right + offset,
                charge_product,
                sigma,
                epsilon,
            )
    return force


def _normalize_mlx_prepared(
    prepared: PreparedSystem,
    *,
    source_atom_count: int,
    replicas: tuple[int, int, int],
    expected_cell_lengths: np.ndarray,
    config: PMEConfig,
) -> PreparedSystem:
    replica_count = int(np.prod(replicas, dtype=np.int64))
    expected_atom_count = source_atom_count * replica_count
    if prepared.atom_count != expected_atom_count:
        msg = f"MLX atom count {prepared.atom_count} != expected {expected_atom_count}"
        raise ChargedPMEParityError(msg)
    cell_lengths = np.asarray(prepared.cell_lengths, dtype=np.float64)
    if not np.allclose(
        cell_lengths,
        expected_cell_lengths,
        rtol=0.0,
        atol=2.0e-5,
    ):
        msg = (
            "MLX cell does not match independent AMBER replica cell: "
            f"mlx={cell_lengths.tolist()}, amber={expected_cell_lengths.tolist()}"
        )
        raise ChargedPMEParityError(msg)
    expected_mesh = np.asarray(config.mesh_shape, dtype=np.int32)
    actual_mesh = np.asarray(prepared.pme_mesh_shape, dtype=np.int32)
    if actual_mesh.shape != (3,) or not np.array_equal(actual_mesh, expected_mesh):
        msg = f"MLX mesh {actual_mesh.tolist()} != pinned mesh {expected_mesh.tolist()}"
        raise ChargedPMEParityError(msg)

    compatibility = deepcopy(prepared.metadata.compatibility_report)
    compatibility["electrostatics_model"] = "pme"
    compatibility["periodic_box_present"] = True
    hydrogen_masses = np.asarray(prepared.masses)[
        np.char.upper(np.asarray(prepared.symbols, dtype=str)) == "H"
    ]
    if hydrogen_masses.size and np.any(hydrogen_masses > 1.25):
        compatibility["hydrogen_mass_repartitioning"] = "represented_by_masses"
    protocol = deepcopy(prepared.metadata.protocol_metadata)
    nonbonded = dict(protocol.get("nonbonded", {}))
    nonbonded["cutoff"] = float(config.real_cutoff)
    protocol["nonbonded"] = nonbonded
    metadata = replace(
        prepared.metadata,
        artifact_version=ARTIFACT_VERSION,
        compatibility_report=compatibility,
        pme_config=_config_payload(config),
        protocol_metadata=protocol,
    )
    normalized = replace(
        prepared,
        metadata=metadata,
        pme_mesh_shape=expected_mesh,
        pme_alpha=np.asarray([config.alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([config.real_cutoff], dtype=np.float32),
        pme_assignment_order=np.asarray([config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray(
            [config.deconvolve_assignment],
            dtype=bool,
        ),
        pme_background_policy=np.asarray([config.background_policy], dtype=str),
    )
    normalized.validate()
    return normalized


def _mlx_manifest(
    prepared: PreparedSystem,
    *,
    source_atom_count: int,
    replicas: tuple[int, int, int],
    config: PMEConfig,
) -> dict[str, Any]:
    identity = _mlx_identity(prepared, source_atom_count=source_atom_count)
    topology = _semantic_topology_manifest(
        identity=identity,
        positions=np.asarray(prepared.positions, dtype=np.float32),
        masses=np.asarray(prepared.masses, dtype=np.float32),
        charges=np.asarray(prepared.charges, dtype=np.float64),
        sigma=np.asarray(prepared.sigma, dtype=np.float64),
        epsilon=np.asarray(prepared.epsilon, dtype=np.float64),
        exception_pairs=np.asarray(prepared.nonbonded_exception_pairs, dtype=np.int32),
        exception_charge_product=np.asarray(
            prepared.nonbonded_exception_charge_product,
            dtype=np.float64,
        ),
        exception_sigma=np.asarray(prepared.nonbonded_exception_sigma, dtype=np.float64),
        exception_epsilon=np.asarray(
            prepared.nonbonded_exception_epsilon,
            dtype=np.float64,
        ),
        constraint_count=int(np.asarray(prepared.constraints).shape[0]),
    )
    forces = _expected_force_manifest(
        bond_count=int(np.asarray(prepared.bonds).shape[0]),
        angle_count=int(np.asarray(prepared.angles).shape[0]),
        torsion_count=(
            int(np.asarray(prepared.dihedrals).shape[0])
            + int(np.asarray(prepared.impropers).shape[0])
        ),
        particle_count=prepared.atom_count,
        exception_count=int(np.asarray(prepared.nonbonded_exception_pairs).shape[0]),
        constraint_count=int(np.asarray(prepared.constraints).shape[0]),
    )
    common = _common_manifest(
        atom_count=prepared.atom_count,
        replicas=replicas,
        topology=topology,
        forces=forces,
        cell_lengths=np.asarray(prepared.cell_lengths, dtype=np.float64),
        config=config,
    )
    common["engine"] = {
        "name": "mlx_atomistic",
        "precision": "float32",
        "artifact_version": int(prepared.metadata.artifact_version),
        "source": "prepared artifact",
    }
    common["manifest_hash"] = manifest_hash(common)
    return common


def _openmm_manifest(
    api: OpenMMApi,
    *,
    reference: dict[str, Any],
    replicas: tuple[int, int, int],
    config: PMEConfig,
    platform_name: str,
    precision: str,
) -> dict[str, Any]:
    system = reference["system"]
    nonbonded = _find_openmm_nonbonded(api, system)
    particles = _openmm_particle_arrays(api, system, nonbonded)
    exceptions = _openmm_exception_arrays(api, nonbonded)
    topology = _semantic_topology_manifest(
        identity=reference["identity"],
        positions=reference["positions_angstrom"],
        masses=particles["masses"],
        charges=particles["charges"],
        sigma=particles["sigma"],
        epsilon=particles["epsilon"],
        exception_pairs=exceptions["pairs"],
        exception_charge_product=exceptions["charge_product"],
        exception_sigma=exceptions["sigma"],
        exception_epsilon=exceptions["epsilon"],
        constraint_count=system.getNumConstraints(),
    )
    forces = _openmm_force_manifest(api, system)
    common = _common_manifest(
        atom_count=system.getNumParticles(),
        replicas=replicas,
        topology=topology,
        forces=forces,
        cell_lengths=reference["cell_lengths_angstrom"],
        config=config,
    )
    common["engine"] = {
        "name": "openmm",
        "role": OPENMM_REFERENCE_ROLE,
        "version": _openmm_version(api),
        "requested_platform": platform_name,
        "requested_precision": precision,
        "source": "AmberPrmtopFile/AmberInpcrdFile",
        "supported_force_classes": list(SUPPORTED_OPENMM_FORCE_CLASSES),
        "actual_force_classes": [
            type(system.getForce(index)).__name__
            for index in range(system.getNumForces())
        ],
    }
    common["manifest_hash"] = manifest_hash(common)
    return common


def _common_manifest(
    *,
    atom_count: int,
    replicas: tuple[int, int, int],
    topology: dict[str, Any],
    forces: dict[str, Any],
    cell_lengths: np.ndarray,
    config: PMEConfig,
) -> dict[str, Any]:
    canonical_lengths = _rounded_list(cell_lengths, decimals=4)
    matrix = np.diag(np.asarray(cell_lengths, dtype=np.float64))
    return {
        "schema_version": 1,
        "workload": {
            "name": "amber20_jac_charged_pme",
            "operation": "fixed_coordinate_total_energy_and_complete_forces",
            "atom_count": int(atom_count),
            "replicas": list(replicas),
            "replica_order": "x-major,y-middle,z-minor,source-atom-minor",
        },
        "topology": topology,
        "forces": forces,
        "cell": {
            "lengths_angstrom": canonical_lengths,
            "matrix_angstrom": [
                _rounded_list(row, decimals=4) for row in matrix
            ],
            "shape": "orthorhombic",
        },
        "pme": {
            "method": "PME",
            "real_cutoff_angstrom": round(float(config.real_cutoff), 6),
            "alpha_per_angstrom": round(float(config.alpha), 6),
            "mesh_shape": list(config.mesh_shape),
            "assignment_order": int(config.assignment_order),
            "deconvolve_assignment": bool(config.deconvolve_assignment),
            "background_policy": config.background_policy,
            "coulomb_constant_kj_mol_angstrom": round(
                COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
                9,
            ),
        },
        "nonbonded": {
            "lj_dispersion_correction": False,
            "switching_function": False,
            "mixing_rule": "lorentz_berthelot_with_explicit_amber_exceptions",
        },
    }


def _semantic_topology_manifest(
    *,
    identity: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    exception_pairs: np.ndarray,
    exception_charge_product: np.ndarray,
    exception_sigma: np.ndarray,
    exception_epsilon: np.ndarray,
    constraint_count: int,
) -> dict[str, Any]:
    charges_canonical = _canonical_float_array(charges, decimals=6)
    sigma_physical = np.where(np.asarray(epsilon) > 1.0e-12, sigma, 0.0)
    lj_particles = np.column_stack(
        [
            _canonical_float_array(sigma_physical, decimals=5),
            _canonical_float_array(epsilon, decimals=5),
        ]
    )
    exception = _canonical_exception_arrays(
        exception_pairs,
        exception_charge_product,
        exception_sigma,
        exception_epsilon,
    )
    return {
        "atom_order_hash": array_hash(np.asarray(identity, dtype=str)),
        "coordinate_hash": array_hash(np.asarray(positions, dtype=np.float32)),
        "mass_hash": array_hash(np.asarray(masses, dtype=np.float32)),
        "charge_hash": array_hash(charges_canonical),
        "lj_particle_hash": array_hash(lj_particles),
        "exception_pairs_hash": array_hash(exception["pairs"]),
        "exception_parameter_hash": array_hash(exception["parameters"]),
        "exclusion_pairs_hash": array_hash(exception["exclusion_pairs"]),
        "active_exception_pairs_hash": array_hash(exception["active_pairs"]),
        "particle_count": int(np.asarray(charges).shape[0]),
        "constraint_count": int(constraint_count),
        "exception_count": int(exception["pairs"].shape[0]),
        "exclusion_count": int(exception["exclusion_pairs"].shape[0]),
        "active_exception_count": int(exception["active_pairs"].shape[0]),
        "net_charge_e": round(float(np.sum(charges, dtype=np.float64)), 4),
        "raw_hashes": {
            "charges": array_hash(np.asarray(charges, dtype=np.float32)),
            "sigma": array_hash(np.asarray(sigma, dtype=np.float32)),
            "epsilon": array_hash(np.asarray(epsilon, dtype=np.float32)),
            "exception_charge_product": array_hash(
                np.asarray(exception_charge_product, dtype=np.float32)
            ),
            "exception_sigma": array_hash(np.asarray(exception_sigma, dtype=np.float32)),
            "exception_epsilon": array_hash(
                np.asarray(exception_epsilon, dtype=np.float32)
            ),
        },
    }


def _canonical_exception_arrays(
    pairs: np.ndarray,
    charge_product: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
) -> dict[str, np.ndarray]:
    pair_array = np.sort(np.asarray(pairs, dtype=np.int32), axis=1)
    order = np.lexsort((pair_array[:, 1], pair_array[:, 0]))
    pair_array = pair_array[order]
    charge_array = np.asarray(charge_product, dtype=np.float64)[order]
    sigma_array = np.asarray(sigma, dtype=np.float64)[order]
    epsilon_array = np.asarray(epsilon, dtype=np.float64)[order]
    sigma_array = np.where(epsilon_array > 1.0e-12, sigma_array, 0.0)
    parameters = np.column_stack(
        [
            _canonical_float_array(charge_array, decimals=4),
            _canonical_float_array(sigma_array, decimals=4),
            _canonical_float_array(epsilon_array, decimals=4),
        ]
    )
    exclusions = (np.abs(charge_array) <= 1.0e-12) & (epsilon_array <= 1.0e-12)
    return {
        "pairs": pair_array,
        "parameters": parameters,
        "exclusion_pairs": pair_array[exclusions],
        "active_pairs": pair_array[~exclusions],
    }


def _compare_manifests(
    mlx_manifest: dict[str, Any],
    openmm_manifest: dict[str, Any],
) -> dict[str, Any]:
    mismatches = manifest_mismatches(
        mlx_manifest,
        openmm_manifest,
        fields=MANIFEST_FIELDS,
    )
    return {
        "status": "matched" if not mismatches else "mismatched",
        "matched": not mismatches,
        "required_fields": list(MANIFEST_FIELDS),
        "mismatches": mismatches,
        "mlx_manifest_hash": mlx_manifest["manifest_hash"],
        "openmm_manifest_hash": openmm_manifest["manifest_hash"],
    }


def _evaluate_openmm_reference(
    api: OpenMMApi,
    *,
    reference: dict[str, Any],
    platform_name: str,
    precision: str,
    config: PMEConfig,
) -> dict[str, Any]:
    system = reference["system"]
    context, properties = _openmm_context(
        api,
        system=system,
        platform_name=platform_name,
        precision=precision,
    )
    context.setPeriodicBoxVectors(*reference["box_vectors"])
    context.setPositions(reference["positions_angstrom"] * 0.1 * api.unit.nanometer)
    started = time.perf_counter()
    state = context.getState(getEnergy=True, getForces=True)
    evaluation_seconds = time.perf_counter() - started
    total_energy = float(
        state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
    )
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            api.unit.kilojoule_per_mole / api.unit.nanometer
        ),
        dtype=np.float64,
    )
    components: dict[str, float] = {}
    for group, name in reference["force_components"].items():
        group_state = context.getState(getEnergy=True, groups={group})
        value = float(
            group_state.getPotentialEnergy().value_in_unit(api.unit.kilojoule_per_mole)
        )
        components[name] = components.get(name, 0.0) + value
    nonbonded = _find_openmm_nonbonded(api, system)
    alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
    alpha_per_nanometer = (
        float(alpha.value_in_unit(api.unit.nanometer**-1))
        if hasattr(alpha, "value_in_unit")
        else float(alpha)
    )
    resolved = {
        "alpha_per_angstrom": alpha_per_nanometer / 10.0,
        "mesh_shape": [int(nx), int(ny), int(nz)],
    }
    resolved_matches = bool(
        np.isclose(resolved["alpha_per_angstrom"], config.alpha, rtol=0.0, atol=1.0e-7)
        and tuple(resolved["mesh_shape"]) == config.mesh_shape
    )
    result = {
        "total_energy_kj_mol": total_energy,
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": forces,
        "force_shape": list(forces.shape),
        "evaluation_seconds": evaluation_seconds,
        "platform": context.getPlatform().getName(),
        "platform_properties": properties,
        "available_platforms": _available_openmm_platforms(api),
        "version": _openmm_version(api),
        "resolved_pme": resolved,
        "resolved_pme_matches_manifest": resolved_matches,
        "precision": properties.get("Precision", "platform-default"),
    }
    del context
    return result


def _evaluate_mlx_prepared(prepared_dir: Path) -> dict[str, Any]:
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    artifact.metadata["nonbonded_cutoff"] = DEFAULT_CUTOFF_ANGSTROM
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=True,
        arrays=artifact.arrays,
    ).to_dict()
    system, force_terms, _ = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if system.cell is None:
        raise ChargedPMEParityError("normalized MLX artifact is missing its periodic cell")
    plan_started = time.perf_counter()
    bound_terms = []
    for term in force_terms:
        if getattr(term, "electrostatics", None) == "pme":
            term = term.bind_pme_plan(system.cell)
        bound_terms.append(term)
    plan_seconds = time.perf_counter() - plan_started
    nonbonded = next(
        (
            term
            for term in bound_terms
            if getattr(term, "electrostatics", None) == "pme"
        ),
        None,
    )
    if nonbonded is None or nonbonded.pme_config is None:
        raise ChargedPMEParityError("normalized MLX artifact has no PME nonbonded term")
    topology = Topology.from_sequences(
        n_atoms=artifact.atom_count,
        bonds=np.asarray(artifact.arrays["bonds"], dtype=np.int32),
        angles=np.asarray(artifact.arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(artifact.arrays["dihedrals"], dtype=np.int32),
        impropers=np.asarray(artifact.arrays["impropers"], dtype=np.int32),
        partial_charges=np.asarray(artifact.arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray(
            artifact.arrays["nonbonded_exception_pairs"],
            dtype=np.int32,
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
        raise ChargedPMEParityError(
            "MLX PME readiness blocked: " + ", ".join(pme_readiness["blockers"])
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
    pme_diagnostics: dict[str, Any] = {}
    evaluation_started = time.perf_counter()
    for term in bound_terms:
        name = str(getattr(term, "name", type(term).__name__))
        term_started = time.perf_counter()
        pairs = neighbors.interactions if name == "nonbonded" else None
        if hasattr(term, "energy_forces_with_components"):
            energy, forces, term_components = term.energy_forces_with_components(
                system.positions,
                system.cell,
                pairs=pairs,
            )
            for component_name, value in term_components.items():
                if component_name == "pme_diagnostics":
                    diagnostics_value = (
                        value.to_dict() if hasattr(value, "to_dict") else value
                    )
                    pme_diagnostics = _jsonable(diagnostics_value)
                    continue
                try:
                    array_value = np.asarray(value)
                except (TypeError, ValueError):
                    continue
                if array_value.shape == ():
                    components[f"{name}.{component_name}"] = float(array_value)
        else:
            energy, forces = term.energy_forces(
                system.positions,
                system.cell,
                pairs=pairs,
            )
        mx.eval(energy, forces)
        components[name] = float(np.asarray(energy))
        total_energy = total_energy + energy
        total_forces = total_forces + forces
        mx.eval(total_energy, total_forces)
        term_timings[name] = time.perf_counter() - term_started
    evaluation_seconds = time.perf_counter() - evaluation_started
    components["torsion"] = components.get("dihedral", 0.0) + components.get(
        "improper",
        0.0,
    )
    plan = getattr(nonbonded, "pme_plan", None)
    runtime_topology = getattr(nonbonded, "topology", None)
    result = {
        "total_energy_kj_mol": float(np.asarray(total_energy)),
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": np.asarray(total_forces, dtype=np.float64) * 10.0,
        "force_shape": list(total_forces.shape),
        "artifact_readiness": artifact_readiness,
        "pme_readiness": pme_readiness,
        "plan": None if plan is None else _jsonable(plan.diagnostics),
        "pme_diagnostics": pme_diagnostics,
        "topology": {
            "pair_policy": getattr(runtime_topology, "nonbonded_pair_policy", None),
            "pair_cache_materialized": (
                getattr(runtime_topology, "_nonbonded_pairs", None) is not None
            ),
            "nonbonded_pair_count": getattr(
                runtime_topology,
                "nonbonded_pair_count",
                None,
            ),
        },
        "neighbor": {
            "backend": neighbors.backend,
            "representation": neighbors.representation_kind,
            "pair_count": int(neighbors.pair_count),
            "compact_pair_count": int(neighbors.compact_pair_count),
            "candidate_count": neighbors.candidate_count,
            "candidate_waste_count": neighbors.candidate_waste_count,
            "fallback_reason": neighbors.fallback_reason,
            "uses_shared_neighbor_policy": True,
        },
        "timings": {
            "plan_build_seconds": plan_seconds,
            "neighbor_build_seconds": neighbor_seconds,
            "force_evaluation_seconds": evaluation_seconds,
            "force_terms_seconds": term_timings,
        },
        "runtime": asdict(get_runtime_info()),
    }
    return result


def _component_energy_metrics(
    mlx_components: dict[str, float],
    openmm_components: dict[str, float],
    *,
    atom_count: int,
) -> dict[str, dict[str, float | None]]:
    mapped_mlx = {
        "bond": mlx_components.get("bond", 0.0),
        "angle": mlx_components.get("angle", 0.0),
        "torsion": mlx_components.get("torsion", 0.0),
        "nonbonded": mlx_components.get("nonbonded", 0.0),
    }
    output: dict[str, dict[str, float | None]] = {}
    for name in ("bond", "angle", "torsion", "nonbonded"):
        candidate = float(mapped_mlx[name])
        reference = float(openmm_components[name])
        absolute = abs(candidate - reference)
        output[name] = {
            "mlx_kj_mol": candidate,
            "openmm_kj_mol": reference,
            "absolute_error_kj_mol": absolute,
            "energy_error_per_atom_kj_mol": absolute / atom_count,
            "relative_error": None if reference == 0.0 else absolute / abs(reference),
        }
    return output


def _parity_checks(
    *,
    metrics: dict[str, float],
    component_metrics: dict[str, dict[str, float | None]],
    small_gate: dict[str, Any],
    manifest_comparison: dict[str, Any],
    openmm_result: dict[str, Any],
    config: PMEConfig,
    tolerances: ChargedPMETolerances,
) -> dict[str, bool]:
    required_components = ("bond", "angle", "torsion", "nonbonded")
    component_energy = all(
        float(component_metrics[name]["energy_error_per_atom_kj_mol"])
        <= tolerances.energy_per_atom_kj_mol
        for name in required_components
    )
    component_relative = all(
        component_metrics[name]["relative_error"] is not None
        and float(component_metrics[name]["relative_error"])
        <= tolerances.relative_energy_error
        for name in required_components
    )
    return {
        "manifest_match": bool(manifest_comparison["matched"]),
        "small_charged_gate": bool(small_gate["passed"]),
        "total_energy_per_atom": (
            metrics["energy_error_per_atom_kj_mol"]
            <= tolerances.energy_per_atom_kj_mol
        ),
        "total_relative_energy": (
            metrics["relative_energy_error"] <= tolerances.relative_energy_error
        ),
        "component_energy_per_atom": component_energy,
        "component_relative_energy": component_relative,
        "force_rms": (
            metrics["rms_absolute_kj_mol_nm"] <= tolerances.force_rms_kj_mol_nm
        ),
        "force_maximum": (
            metrics["maximum_absolute_kj_mol_nm"]
            <= tolerances.force_maximum_kj_mol_nm
        ),
        "openmm_resolved_pme": bool(openmm_result["resolved_pme_matches_manifest"]),
        "assignment_order_5": config.assignment_order == DEFAULT_ASSIGNMENT_ORDER,
        "background_policy": config.background_policy == DEFAULT_BACKGROUND_POLICY,
    }


def _openmm_context(
    api: OpenMMApi,
    *,
    system: Any,
    platform_name: str,
    precision: str,
) -> tuple[Any, dict[str, str]]:
    available = _available_openmm_platforms(api)
    if platform_name not in available:
        raise ChargedPMEParityError(
            f"OpenMM platform {platform_name!r} unavailable; available={available}"
        )
    platform = api.mm.Platform.getPlatformByName(platform_name)
    requested: dict[str, str] = {}
    if "Precision" in list(platform.getPropertyNames()):
        requested["Precision"] = precision
    integrator = api.mm.VerletIntegrator(0.001 * api.unit.picoseconds)
    try:
        context = api.mm.Context(system, integrator, platform, requested)
    except Exception as exc:
        msg = (
            f"OpenMM {platform_name} context unavailable with properties "
            f"{requested}: {exc}"
        )
        raise ChargedPMEParityError(msg) from exc
    properties = {}
    for name in platform.getPropertyNames():
        try:
            properties[name] = platform.getPropertyValue(context, name)
        except Exception as exc:  # pragma: no cover - platform-specific.
            properties[name] = f"<unavailable:{exc}>"
    return context, properties


def _openmm_source_identity(prmtop: Any) -> np.ndarray:
    loader = getattr(prmtop, "_prmtop", None)
    required = (
        "getAtomNames",
        "getAtomTypes",
        "getResidueLabel",
    )
    if loader is None or any(not hasattr(loader, name) for name in required):
        raise ChargedPMEParityError("OpenMM AMBER loader lacks raw atom identity accessors")
    names = loader.getAtomNames()
    atom_types = loader.getAtomTypes()
    rows = [
        [
            str(index),
            str(names[index]),
            str(atom_types[index]),
            str(loader.getResidueLabel(index)),
        ]
        for index in range(len(names))
    ]
    return np.asarray(rows, dtype=str)


def _identity_with_replica(identity: np.ndarray, replica_index: int) -> np.ndarray:
    replica_column = np.full((identity.shape[0], 1), str(replica_index), dtype=str)
    return np.concatenate([replica_column, np.asarray(identity, dtype=str)], axis=1)


def _mlx_identity(prepared: PreparedSystem, *, source_atom_count: int) -> np.ndarray:
    rows = []
    for atom_index in range(prepared.atom_count):
        replica_index, local_index = divmod(atom_index, source_atom_count)
        rows.append(
            [
                str(replica_index),
                str(local_index),
                str(prepared.atom_names[atom_index]),
                str(prepared.atom_types[atom_index]),
                str(prepared.residue_names[atom_index]),
            ]
        )
    return np.asarray(rows, dtype=str)


def _expected_force_manifest(
    *,
    bond_count: int,
    angle_count: int,
    torsion_count: int,
    particle_count: int,
    exception_count: int,
    constraint_count: int,
) -> dict[str, Any]:
    class_counts = {
        "HarmonicBondForce": int(bond_count > 0),
        "HarmonicAngleForce": int(angle_count > 0),
        "PeriodicTorsionForce": int(torsion_count > 0),
        "NonbondedForce": int(particle_count > 0),
    }
    return {
        "class_counts": class_counts,
        "term_counts": {
            "HarmonicBondForce": bond_count,
            "HarmonicAngleForce": angle_count,
            "PeriodicTorsionForce": torsion_count,
            "NonbondedForce.particles": particle_count,
            "NonbondedForce.exceptions": exception_count,
            "System.constraints": constraint_count,
        },
    }


def _openmm_force_manifest(api: OpenMMApi, system: Any) -> dict[str, Any]:
    class_counts = {name: 0 for name in SUPPORTED_OPENMM_FORCE_CLASSES}
    term_counts = {
        "HarmonicBondForce": 0,
        "HarmonicAngleForce": 0,
        "PeriodicTorsionForce": 0,
        "NonbondedForce.particles": 0,
        "NonbondedForce.exceptions": 0,
        "System.constraints": system.getNumConstraints(),
    }
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        name = type(force).__name__
        if name not in class_counts:
            raise UnsupportedOpenMMForceError(name)
        class_counts[name] += 1
        if isinstance(force, api.mm.HarmonicBondForce):
            term_counts[name] += force.getNumBonds()
        elif isinstance(force, api.mm.HarmonicAngleForce):
            term_counts[name] += force.getNumAngles()
        elif isinstance(force, api.mm.PeriodicTorsionForce):
            term_counts[name] += force.getNumTorsions()
        elif isinstance(force, api.mm.NonbondedForce):
            term_counts["NonbondedForce.particles"] += force.getNumParticles()
            term_counts["NonbondedForce.exceptions"] += force.getNumExceptions()
    return {"class_counts": class_counts, "term_counts": term_counts}


def _openmm_particle_arrays(api: OpenMMApi, system: Any, nonbonded: Any) -> dict[str, Any]:
    masses = np.asarray(
        [
            system.getParticleMass(index).value_in_unit(api.unit.dalton)
            for index in range(system.getNumParticles())
        ],
        dtype=np.float32,
    )
    charges = []
    sigma = []
    epsilon = []
    for index in range(nonbonded.getNumParticles()):
        charge, sigma_value, epsilon_value = nonbonded.getParticleParameters(index)
        charges.append(charge.value_in_unit(api.unit.elementary_charge))
        sigma.append(sigma_value.value_in_unit(api.unit.angstrom))
        epsilon.append(epsilon_value.value_in_unit(api.unit.kilojoule_per_mole))
    return {
        "masses": masses,
        "charges": np.asarray(charges, dtype=np.float64),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "epsilon": np.asarray(epsilon, dtype=np.float64),
    }


def _openmm_exception_arrays(api: OpenMMApi, nonbonded: Any) -> dict[str, Any]:
    pairs = []
    charge_product = []
    sigma = []
    epsilon = []
    for index in range(nonbonded.getNumExceptions()):
        left, right, charge, sigma_value, epsilon_value = nonbonded.getExceptionParameters(index)
        pairs.append((left, right))
        charge_product.append(charge.value_in_unit(api.unit.elementary_charge**2))
        sigma.append(sigma_value.value_in_unit(api.unit.angstrom))
        epsilon.append(epsilon_value.value_in_unit(api.unit.kilojoule_per_mole))
    return {
        "pairs": np.asarray(pairs, dtype=np.int32),
        "charge_product": np.asarray(charge_product, dtype=np.float64),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "epsilon": np.asarray(epsilon, dtype=np.float64),
    }


def _find_openmm_nonbonded(api: OpenMMApi, system: Any) -> Any:
    forces = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), api.mm.NonbondedForce)
    ]
    if len(forces) != 1:
        raise UnsupportedOpenMMForceError(
            f"expected exactly one NonbondedForce, found {len(forces)}"
        )
    return forces[0]


def _component_name(force_class: str) -> str:
    mapping = {
        "HarmonicBondForce": "bond",
        "HarmonicAngleForce": "angle",
        "PeriodicTorsionForce": "torsion",
        "NonbondedForce": "nonbonded",
    }
    if force_class not in mapping:
        raise UnsupportedOpenMMForceError(force_class)
    return mapping[force_class]


def _jac_pme_config(replicas: tuple[int, int, int]) -> PMEConfig:
    mesh = tuple(
        int(base * replica)
        for base, replica in zip(DEFAULT_BASE_MESH, replicas, strict=True)
    )
    return PMEConfig(
        mesh_shape=mesh,
        alpha=DEFAULT_ALPHA_PER_ANGSTROM,
        real_cutoff=DEFAULT_CUTOFF_ANGSTROM,
        assignment_order=DEFAULT_ASSIGNMENT_ORDER,
        charge_tolerance=DEFAULT_CHARGE_TOLERANCE,
        deconvolve_assignment=True,
        background_policy=DEFAULT_BACKGROUND_POLICY,
    )


def _config_payload(config: PMEConfig) -> dict[str, Any]:
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha),
        "real_cutoff": float(config.real_cutoff),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
    }


def _normalize_replicas(replicas: object) -> tuple[int, int, int]:
    try:
        values = tuple(replicas)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("replicas must contain three positive integers") from exc
    if len(values) != 3:
        raise ValueError("replicas must contain three positive integers")
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("replicas must contain three positive integers")
        integer = int(value)
        if integer != value or integer <= 0:
            raise ValueError("replicas must contain three positive integers")
        normalized.append(integer)
    return tuple(normalized)  # type: ignore[return-value]


def _parse_replicas(value: str) -> tuple[int, int, int]:
    try:
        return _normalize_replicas(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--replicas must be three comma-separated positive integers"
        ) from exc


def _replica_translations(
    base_cell_lengths: np.ndarray,
    replicas: tuple[int, int, int],
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(
            [
                ix * base_cell_lengths[0],
                iy * base_cell_lengths[1],
                iz * base_cell_lengths[2],
            ],
            dtype=np.float32,
        )
        for ix in range(replicas[0])
        for iy in range(replicas[1])
        for iz in range(replicas[2])
    )


def _orthorhombic_lengths_angstrom(api: OpenMMApi, vectors: Any) -> np.ndarray:
    matrix = np.asarray(
        [
            [vector.x, vector.y, vector.z]
            for vector in vectors.value_in_unit(api.unit.angstrom)
        ],
        dtype=np.float64,
    )
    if matrix.shape != (3, 3) or not np.allclose(
        matrix,
        np.diag(np.diag(matrix)),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ChargedPMEParityError("AMBER JAC comparator requires an orthorhombic box")
    lengths = np.diag(matrix)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ChargedPMEParityError("AMBER JAC box lengths must be finite and positive")
    return lengths


def _box_vectors(api: OpenMMApi, lengths_angstrom: np.ndarray) -> Any:
    lengths_nm = np.asarray(lengths_angstrom, dtype=np.float64) * 0.1
    return (
        api.mm.Vec3(lengths_nm[0], 0.0, 0.0),
        api.mm.Vec3(0.0, lengths_nm[1], 0.0),
        api.mm.Vec3(0.0, 0.0, lengths_nm[2]),
    ) * api.unit.nanometer


def _canonical_float_array(values: np.ndarray, *, decimals: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64), decimals=decimals)


def _rounded_list(values: np.ndarray, *, decimals: int) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values).tolist()]


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


def _missing_inputs(
    prepared_path: Path,
    prmtop_path: Path,
    coordinates_path: Path,
) -> list[str]:
    required = (
        prepared_path / "prepared_system.json",
        prepared_path / "prepared_system.npz",
        prmtop_path,
        coordinates_path,
    )
    return [str(path) for path in required if not path.is_file()]


def _base_report(
    *,
    prepared_path: Path,
    prmtop_path: Path,
    coordinates_path: Path,
    replicas: tuple[int, int, int],
    platform_name: str,
    precision: str,
    tolerances: ChargedPMETolerances,
    out_path: Path,
) -> dict[str, Any]:
    return {
        "kind": "mlx_atomistic.charged_pme_parity",
        "schema_version": 1,
        "fixture": "amber20_jac",
        "reference_engine": "openmm",
        "reference_engine_role": OPENMM_REFERENCE_ROLE,
        "mlx_prepared": str(prepared_path),
        "amber_prmtop": str(prmtop_path),
        "amber_coordinates": str(coordinates_path),
        "replicas": list(replicas),
        "requested_openmm_platform": platform_name,
        "requested_openmm_precision": precision,
        "tolerances": asdict(tolerances),
        "out": str(out_path),
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
    return {
        key: value
        for key, value in payload.items()
        if key not in {"forces_kj_mol_nm"}
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _finish_report(report: dict[str, Any], out_path: Path) -> dict[str, Any]:
    report = _jsonable(report)
    _write_json(out_path / REPORT_NAME, report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the charged-PME parity command-line interface.

    Args:
        argv: Optional argument list; ``None`` reads process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-prepared", type=Path, required=True)
    parser.add_argument("--amber-prmtop", type=Path, required=True)
    parser.add_argument("--amber-coordinates", type=Path, required=True)
    parser.add_argument("--replicas", type=_parse_replicas, required=True)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--precision", choices=("single", "mixed", "double"), default="single")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_charged_pme_parity(
        mlx_prepared=args.mlx_prepared,
        amber_prmtop=args.amber_prmtop,
        amber_coordinates=args.amber_coordinates,
        replicas=args.replicas,
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
    "ChargedPMEParityError",
    "ChargedPMETolerances",
    "FORCE_ARRAYS_NAME",
    "MANIFEST_COMPARISON_NAME",
    "MLX_MANIFEST_NAME",
    "OPENMM_MANIFEST_NAME",
    "REPORT_NAME",
    "UnsupportedOpenMMForceError",
    "evaluate_small_charged_fixture",
    "main",
    "run_charged_pme_parity",
]
