"""OpenMM-vs-MLX force and energy parity helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.pme import PMEConfig, pme_readiness_report
from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.topology_import import import_amber_prmtop
from mlx_atomistic.topology import Topology

DEFAULT_AMBER_FIXTURE = "amber-alanine-dipeptide-implicit"
REPORT_NAME = "openmm_mlx_parity_report.json"


@dataclass(frozen=True)
class ParityTolerances:
    """Numerical tolerances for fixed-coordinate parity checks."""

    total_energy_abs_kj_mol: float = 2.0e-3
    component_energy_abs_kj_mol: float = 2.0e-3
    force_max_abs_kj_mol_nm: float = 12.0
    force_rms_abs_kj_mol_nm: float = 3.0


@dataclass(frozen=True)
class PMEParityConfig:
    """Periodic PME settings used by the OpenMM-vs-MLX parity harness."""

    mesh_shape: tuple[int, int, int] = (48, 48, 48)
    alpha_per_angstrom: float = 0.35
    real_cutoff_angstrom: float = 10.0
    cell_lengths_angstrom: tuple[float, float, float] = (40.0, 40.0, 40.0)
    assignment_order: int = 2
    charge_tolerance: float = 1.0e-5
    deconvolve_assignment: bool = True

    def to_pme_config(self) -> PMEConfig:
        return PMEConfig(
            mesh_shape=self.mesh_shape,
            alpha=self.alpha_per_angstrom,
            real_cutoff=self.real_cutoff_angstrom,
            assignment_order=self.assignment_order,
            charge_tolerance=self.charge_tolerance,
            deconvolve_assignment=self.deconvolve_assignment,
        )


@dataclass(frozen=True)
class OpenMMMLXParityReport:
    """Machine-readable result for one OpenMM-vs-MLX parity run."""

    status: str
    fixture: str
    source_kind: str
    atom_count: int
    prepared_dir: str
    prmtop_path: str
    coords_path: str
    openmm_platform: str
    openmm_nonbonded_method: str
    mlx_nonbonded_cutoff_angstrom: float
    pme_config: dict[str, Any] | None
    pme_readiness: dict[str, Any] | None
    total_energy_openmm_kj_mol: float | None
    total_energy_mlx_kj_mol: float | None
    total_energy_abs_error_kj_mol: float | None
    component_energy_openmm_kj_mol: dict[str, float]
    component_energy_mlx_kj_mol: dict[str, float]
    component_energy_abs_error_kj_mol: dict[str, float]
    force_shape: tuple[int, int] | None
    force_max_abs_error_kj_mol_nm: float | None
    force_rms_abs_error_kj_mol_nm: float | None
    unsupported_terms: tuple[str, ...]
    unmapped_mlx_components: tuple[str, ...]
    tolerances: ParityTolerances
    passed: bool
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["force_shape"] = list(self.force_shape) if self.force_shape is not None else None
        payload["unsupported_terms"] = list(self.unsupported_terms)
        payload["unmapped_mlx_components"] = list(self.unmapped_mlx_components)
        payload["blockers"] = list(self.blockers)
        return payload


def default_amber_fixture_paths(root: str | Path = ".") -> tuple[Path, Path]:
    """Return the default small real AMBER fixture paths from the reference checkout."""

    base = Path(root) / "vendors" / "openmm" / "wrappers" / "python" / "tests" / "systems"
    return base / "alanine-dipeptide-implicit.prmtop", base / "alanine-dipeptide-implicit.inpcrd"


def run_amber_openmm_mlx_parity(
    *,
    prmtop_path: str | Path,
    coords_path: str | Path,
    out_dir: str | Path,
    fixture: str = DEFAULT_AMBER_FIXTURE,
    platform_name: str = "Reference",
    tolerances: ParityTolerances | None = None,
    mlx_nonbonded_cutoff_angstrom: float = 1.0e6,
    pme_config: PMEParityConfig | None = None,
) -> OpenMMMLXParityReport:
    """Build an AMBER artifact and compare fixed-coordinate OpenMM and MLX energies."""

    tolerances = ParityTolerances() if tolerances is None else tolerances
    prmtop = Path(prmtop_path)
    coords = Path(coords_path)
    out = Path(out_dir)
    prepared_dir = out / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    if not prmtop.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing AMBER prmtop: {prmtop}",
        )
    if not coords.exists():
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=mlx_nonbonded_cutoff_angstrom,
            tolerances=tolerances,
            blocker=f"missing AMBER coordinates: {coords}",
        )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    if pme_config is not None:
        prepared = _with_pme_artifact_settings(prepared, pme_config)
    save_prepared_system(prepared, prepared_dir)
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    effective_cutoff = (
        float(pme_config.real_cutoff_angstrom)
        if pme_config is not None
        else float(mlx_nonbonded_cutoff_angstrom)
    )
    artifact.metadata["nonbonded_cutoff"] = effective_cutoff
    if pme_config is not None:
        artifact.metadata["electrostatics_model"] = "pme"
        artifact.metadata["pme_config"] = _pme_config_payload(pme_config)
    readiness = _pme_readiness(artifact, pme_config)
    if readiness is not None and readiness["status"] != "ready":
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker="PME readiness blocked: " + ", ".join(readiness["blockers"]),
            atom_count=artifact.atom_count,
            pme_config=pme_config,
            pme_readiness=readiness,
        )
    unsupported_terms = _unsupported_terms(artifact.metadata)
    if unsupported_terms:
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker="unsupported force-field terms: " + ", ".join(unsupported_terms),
            atom_count=artifact.atom_count,
            unsupported_terms=unsupported_terms,
            pme_config=pme_config,
            pme_readiness=readiness,
        )

    try:
        system, force_terms, _ = build_mlx_system_from_artifact(artifact)
    except MLXCompatibilityError as exc:
        return _blocked_report(
            fixture=fixture,
            prmtop_path=prmtop,
            coords_path=coords,
            prepared_dir=prepared_dir,
            platform_name=platform_name,
            cutoff=effective_cutoff,
            tolerances=tolerances,
            blocker=str(exc),
            atom_count=artifact.atom_count,
            unsupported_terms=unsupported_terms,
            pme_config=pme_config,
            pme_readiness=readiness,
        )

    mlx_total, mlx_forces, mlx_components = _evaluate_mlx(
        force_terms,
        system.positions,
        system.cell,
    )
    openmm_result = _evaluate_openmm_amber(
        prmtop_path=prmtop,
        coords_path=coords,
        platform_name=platform_name,
        pme_config=pme_config,
    )
    component_errors = _component_errors(
        mlx_components,
        openmm_result["component_energy_kj_mol"],
    )
    mlx_forces_kj_mol_nm = mlx_forces * 10.0
    openmm_forces = openmm_result["forces_kj_mol_nm"]
    force_delta = mlx_forces_kj_mol_nm - openmm_forces
    force_max = float(np.max(np.abs(force_delta)))
    force_rms = float(np.sqrt(np.mean(force_delta * force_delta)))
    total_error = abs(float(mlx_total) - float(openmm_result["total_energy_kj_mol"]))
    passed = bool(
        total_error <= tolerances.total_energy_abs_kj_mol
        and all(
            error <= tolerances.component_energy_abs_kj_mol
            for error in component_errors.values()
        )
        and force_max <= tolerances.force_max_abs_kj_mol_nm
        and force_rms <= tolerances.force_rms_abs_kj_mol_nm
    )
    report = OpenMMMLXParityReport(
        status="passed" if passed else "failed",
        fixture=fixture,
        source_kind="amber",
        atom_count=artifact.atom_count,
        prepared_dir=str(prepared_dir),
        prmtop_path=str(prmtop),
        coords_path=str(coords),
        openmm_platform=str(openmm_result["platform"]),
        openmm_nonbonded_method="PME" if pme_config is not None else "NoCutoff",
        mlx_nonbonded_cutoff_angstrom=effective_cutoff,
        pme_config=None if pme_config is None else _pme_config_payload(pme_config),
        pme_readiness=readiness,
        total_energy_openmm_kj_mol=float(openmm_result["total_energy_kj_mol"]),
        total_energy_mlx_kj_mol=float(mlx_total),
        total_energy_abs_error_kj_mol=float(total_error),
        component_energy_openmm_kj_mol=openmm_result["component_energy_kj_mol"],
        component_energy_mlx_kj_mol=mlx_components,
        component_energy_abs_error_kj_mol=component_errors,
        force_shape=tuple(int(item) for item in openmm_forces.shape),
        force_max_abs_error_kj_mol_nm=force_max,
        force_rms_abs_error_kj_mol_nm=force_rms,
        unsupported_terms=unsupported_terms,
        unmapped_mlx_components=tuple(
            sorted(set(mlx_components) - {"bond", "angle", "torsion", "nonbonded"})
        ),
        tolerances=tolerances,
        passed=passed,
    )
    _write_report(report, out)
    return report


def _blocked_report(
    *,
    fixture: str,
    prmtop_path: Path,
    coords_path: Path,
    prepared_dir: Path,
    platform_name: str,
    cutoff: float,
    tolerances: ParityTolerances,
    blocker: str,
    atom_count: int = 0,
    unsupported_terms: tuple[str, ...] = (),
    pme_config: PMEParityConfig | None = None,
    pme_readiness: dict[str, Any] | None = None,
) -> OpenMMMLXParityReport:
    return OpenMMMLXParityReport(
        status="blocked",
        fixture=fixture,
        source_kind="amber",
        atom_count=atom_count,
        prepared_dir=str(prepared_dir),
        prmtop_path=str(prmtop_path),
        coords_path=str(coords_path),
        openmm_platform=platform_name,
        openmm_nonbonded_method="PME" if pme_config is not None else "NoCutoff",
        mlx_nonbonded_cutoff_angstrom=float(cutoff),
        pme_config=None if pme_config is None else _pme_config_payload(pme_config),
        pme_readiness=pme_readiness,
        total_energy_openmm_kj_mol=None,
        total_energy_mlx_kj_mol=None,
        total_energy_abs_error_kj_mol=None,
        component_energy_openmm_kj_mol={},
        component_energy_mlx_kj_mol={},
        component_energy_abs_error_kj_mol={},
        force_shape=None,
        force_max_abs_error_kj_mol_nm=None,
        force_rms_abs_error_kj_mol_nm=None,
        unsupported_terms=unsupported_terms,
        unmapped_mlx_components=(),
        tolerances=tolerances,
        passed=False,
        blockers=(blocker,),
    )


def _with_pme_artifact_settings(prepared: Any, config: PMEParityConfig) -> Any:
    report = dict(prepared.metadata.compatibility_report)
    required_terms = list(report.get("required_terms", ()))
    supported_terms = list(report.get("supported_terms", ()))
    for terms in (required_terms, supported_terms):
        if "pme" not in terms:
            terms.append("pme")
    report.update(
        {
            "periodic_box_present": True,
            "electrostatics_model": "pme",
            "required_terms": required_terms,
            "supported_terms": supported_terms,
        }
    )
    metadata = replace(
        prepared.metadata,
        compatibility_report=report,
        pme_config=_pme_config_payload(config),
    )
    return replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray(config.cell_lengths_angstrom, dtype=np.float32),
        pme_mesh_shape=np.asarray(config.mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([config.alpha_per_angstrom], dtype=np.float32),
        pme_real_cutoff=np.asarray([config.real_cutoff_angstrom], dtype=np.float32),
        pme_assignment_order=np.asarray([config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([config.deconvolve_assignment], dtype=bool),
    )


def _pme_config_payload(config: PMEParityConfig) -> dict[str, Any]:
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha_per_angstrom),
        "real_cutoff": float(config.real_cutoff_angstrom),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
    }


def _pme_readiness(artifact: Any, config: PMEParityConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    arrays = artifact.arrays
    topology = Topology.from_sequences(
        n_atoms=artifact.atom_count,
        bonds=np.asarray(arrays["bonds"], dtype=np.int32),
        angles=np.asarray(arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(arrays["dihedrals"], dtype=np.int32),
        impropers=np.asarray(arrays.get("impropers", np.empty((0, 4))), dtype=np.int32),
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray(
            arrays.get("nonbonded_exception_pairs", np.empty((0, 2))),
            dtype=np.int32,
        ),
        exclude_bonds=True,
        nonbonded_cutoff=float(config.real_cutoff_angstrom),
    )
    return pme_readiness_report(
        atom_count=artifact.atom_count,
        charges=arrays["charges"],
        cell_lengths=arrays.get("cell_lengths", np.asarray([])),
        config=config.to_pme_config(),
        nonbonded_cutoff=float(config.real_cutoff_angstrom),
        exclusion_count=len(topology.exclusion_set),
        one_four_count=len(topology.one_four_set),
        explicit_exception_count=int(
            np.asarray(arrays.get("nonbonded_exception_pairs", np.empty((0, 2)))).shape[0]
        ),
    )


def _unsupported_terms(metadata: dict[str, Any]) -> tuple[str, ...]:
    report = dict(metadata.get("compatibility_report", {}))
    terms = [*report.get("unsupported_terms", ()), *report.get("rejected_terms", ())]
    return tuple(str(term) for term in terms)


def _evaluate_mlx(
    force_terms: list[Any],
    positions: Any,
    cell: Any,
) -> tuple[float, np.ndarray, dict[str, float]]:
    total = mx.array(0.0, dtype=mx.float32)
    forces = mx.zeros_like(positions)
    components: dict[str, float] = {}
    for term in force_terms:
        term_name = str(getattr(term, "name", type(term).__name__))
        if hasattr(term, "energy_forces_with_components"):
            energy, term_forces, term_components = term.energy_forces_with_components(
                positions,
                cell,
            )
            for name, value in term_components.items():
                if isinstance(value, dict):
                    continue
                try:
                    array_value = np.asarray(value)
                except TypeError:
                    continue
                if array_value.shape != ():
                    continue
                try:
                    components[f"{term_name}.{name}"] = float(array_value)
                except (TypeError, ValueError):
                    continue
        else:
            energy, term_forces = term.energy_forces(positions, cell)
        components[term_name] = float(np.asarray(energy))
        total = total + energy
        forces = forces + term_forces
    mx.eval(total, forces)
    components["torsion"] = components.get("dihedral", 0.0) + components.get("improper", 0.0)
    return float(np.asarray(total)), np.asarray(forces, dtype=np.float64), components


def _evaluate_openmm_amber(
    *,
    prmtop_path: Path,
    coords_path: Path,
    platform_name: str,
    pme_config: PMEParityConfig | None = None,
) -> dict[str, Any]:
    import openmm as mm
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    coords = app.AmberInpcrdFile(str(coords_path))
    box_vectors = None
    if pme_config is not None:
        a, b, c = (float(item) * 0.1 for item in pme_config.cell_lengths_angstrom)
        box_vectors = (
            mm.Vec3(a, 0.0, 0.0),
            mm.Vec3(0.0, b, 0.0),
            mm.Vec3(0.0, 0.0, c),
        ) * unit.nanometer
        prmtop.topology.setPeriodicBoxVectors(box_vectors)
    system_kwargs: dict[str, Any] = {
        "nonbondedMethod": app.PME if pme_config is not None else app.NoCutoff,
        "constraints": None,
        "removeCMMotion": False,
    }
    if pme_config is not None:
        system_kwargs["nonbondedCutoff"] = (
            pme_config.real_cutoff_angstrom * 0.1 * unit.nanometer
        )
    openmm_system = prmtop.createSystem(**system_kwargs)
    if box_vectors is not None:
        openmm_system.setDefaultPeriodicBoxVectors(*box_vectors)
        for force_index in range(openmm_system.getNumForces()):
            force = openmm_system.getForce(force_index)
            if isinstance(force, mm.NonbondedForce):
                force.setPMEParameters(
                    pme_config.alpha_per_angstrom * 10.0 / unit.nanometer,
                    *pme_config.mesh_shape,
                )
    for index in range(openmm_system.getNumForces()):
        openmm_system.getForce(index).setForceGroup(index)
    platform = mm.Platform.getPlatformByName(platform_name)
    context = mm.Context(
        openmm_system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        platform,
    )
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(coords.positions)
    state = context.getState(getEnergy=True, getForces=True)
    components: dict[str, float] = {}
    for index in range(openmm_system.getNumForces()):
        force = openmm_system.getForce(index)
        group_state = context.getState(getEnergy=True, groups={index})
        key = _openmm_component_name(type(force).__name__)
        if key is None:
            continue
        components[key] = components.get(key, 0.0) + float(
            group_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    return {
        "platform": context.getPlatform().getName(),
        "total_energy_kj_mol": float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
        "component_energy_kj_mol": components,
        "forces_kj_mol_nm": forces,
    }


def _openmm_component_name(force_class_name: str) -> str | None:
    return {
        "HarmonicBondForce": "bond",
        "HarmonicAngleForce": "angle",
        "PeriodicTorsionForce": "torsion",
        "NonbondedForce": "nonbonded",
    }.get(force_class_name)


def _component_errors(
    mlx_components: dict[str, float],
    openmm_components: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(openmm_components) & {"bond", "angle", "torsion", "nonbonded"})
    return {
        key: abs(float(mlx_components[key]) - float(openmm_components[key]))
        for key in keys
        if key in mlx_components
    }


def _write_report(report: OpenMMMLXParityReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "DEFAULT_AMBER_FIXTURE",
    "REPORT_NAME",
    "OpenMMMLXParityReport",
    "PMEParityConfig",
    "ParityTolerances",
    "default_amber_fixture_paths",
    "run_amber_openmm_mlx_parity",
]
