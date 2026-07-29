"""Run the frozen, staged OpenMM-versus-MLX 5dfr NPT validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.artifacts import (
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.benchmarks.dhfr_npt import (
    DEFAULT_CONTRACT_PATH,
    FINAL_TARGET_SCOPE,
    DHFRNPTValidationError,
    build_stage_report,
    load_completed_stage,
    load_validation_contract,
    stage_report_path,
    validate_prepared_boundary,
    write_stage_report_atomic,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.md import (
    analytic_configurational_virial_tensor,
    configurational_virial_tensor,
)
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import molecularly_strained_positions
from mlx_atomistic.pme import pme_coulomb_strain_components
from mlx_atomistic.prep.io import load_prepared_system
from mlx_atomistic.prep.runner import run_mlx
from mlx_atomistic.units import ATM_TO_KJ_PER_MOL_ANGSTROM3

try:
    from scripts import openmm_mlx_parity as _parity
    from scripts import prepare_openmm_dhfr_explicit as _preparation
except ImportError:  # pragma: no cover - direct script execution.
    import openmm_mlx_parity as _parity
    import prepare_openmm_dhfr_explicit as _preparation

PRESSURE_KJ_MOL_A3_TO_BAR = 1.0 / (0.9869232667160128 * ATM_TO_KJ_PER_MOL_ANGSTROM3)
GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324


def main(argv: list[str] | None = None) -> int:
    """Run one frozen target stage.

    Args:
        argv: Optional process arguments.

    Returns:
        Zero for a passing or safely reused stage.
    """

    args = _parse_args(argv)
    contract = load_validation_contract(args.contract)
    source_identity = validate_prepared_boundary(args.prepared, contract)
    report_path = stage_report_path(args.out, stage=args.stage, seed=args.seed)
    completed = load_completed_stage(
        report_path,
        contract=contract,
        source_identity=source_identity,
        stage=args.stage,
        seed=args.seed,
    )
    if completed is not None:
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0

    if args.stage == "fixed":
        evidence, checks = _run_fixed_stage(
            prepared_dir=args.prepared,
            stage_dir=report_path.parent,
            contract=contract,
            platform_name=args.platform,
        )
    else:
        declared_seeds = [int(seed) for seed in contract["workload"]["seeds"]]
        if args.seed not in declared_seeds:
            msg = f"NPT seed must be one of the frozen values {declared_seeds}"
            raise DHFRNPTValidationError(msg)
        evidence, checks = _run_npt_stage(
            prepared_dir=args.prepared,
            stage_dir=report_path.parent,
            contract=contract,
            platform_name=args.platform,
            seed=int(args.seed),
        )
    report = build_stage_report(
        contract=contract,
        source_identity=source_identity,
        stage=args.stage,
        seed=args.seed,
        scope=FINAL_TARGET_SCOPE,
        evidence=evidence,
        checks=checks,
    )
    write_stage_report_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def _run_fixed_stage(
    *,
    prepared_dir: Path,
    stage_dir: Path,
    contract: dict[str, Any],
    platform_name: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    started = time.perf_counter()
    mlx_result = _evaluate_mlx_fixed(prepared_dir)
    openmm_result = _evaluate_openmm_fixed(
        prepared_dir,
        contract=contract,
        platform_name=platform_name,
    )
    metrics = _fixed_metrics(
        mlx_result,
        openmm_result,
        atom_count=int(contract["target"]["atom_count"]),
    )
    gates = contract["fixed_state_gates"]
    required_components = ("bond", "angle", "dihedral", "nonbonded")
    checks = {
        "finite": _all_finite(metrics),
        "complete_components": (
            sorted(mlx_result["components"]) == sorted(required_components)
            and sorted(openmm_result["components"]) == sorted(required_components)
        ),
        "total_energy_per_atom": (
            metrics["energy_error_per_atom_kj_mol"]
            <= gates["energy_error_per_atom_kj_mol"]
        ),
        "total_relative_energy": (
            metrics["relative_energy_error"] <= gates["relative_energy_error"]
        ),
        "component_energy_per_atom": all(
            row["error_per_atom_kj_mol"]
            <= gates["energy_error_per_atom_kj_mol"]
            for row in metrics["components"].values()
        ),
        "component_relative_energy": all(
            row["relative_error"] <= gates["relative_energy_error"]
            for row in metrics["components"].values()
        ),
        "force_rms": (
            metrics["force_rms_error_kj_mol_nm"]
            <= gates["force_rms_error_kj_mol_nm"]
        ),
        "force_max": (
            metrics["force_max_error_kj_mol_nm"]
            <= gates["force_max_error_kj_mol_nm"]
        ),
        "virial_per_atom": (
            metrics["virial_error_per_atom_kj_mol"]
            <= gates["virial_error_per_atom_kj_mol"]
        ),
        "virial_relative": (
            metrics["relative_virial_error"]
            <= gates["relative_virial_error"]
        ),
        "pressure_diagonal": (
            metrics["pressure_diagonal_max_error_kj_mol_a3"]
            <= gates["pressure_diagonal_max_error_kj_mol_a3"]
        ),
        "analytic_matches_finite_difference": (
            metrics["mlx_analytic_fd_virial_error_per_atom_kj_mol"]
            <= gates["finite_difference_virial_error_per_atom_kj_mol"]
        ),
        "mlx_lazy_neighbors": (
            mlx_result["neighbor"]["representation"] == "blocks"
            and mlx_result["neighbor"]["fallback_reason"] is None
        ),
        "mlx_pme_plan": bool(mlx_result["pme_plan_fingerprint"]),
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = stage_dir / "fixed_arrays.npz"
    _atomic_savez(
        arrays_path,
        {
            "mlx_forces_kj_mol_nm": mlx_result["forces"],
            "openmm_forces_kj_mol_nm": openmm_result["forces"],
            "mlx_analytic_virial_kj_mol": mlx_result["analytic_virial"],
            "mlx_fd_virial_kj_mol": mlx_result["fd_virial"],
            "openmm_virial_kj_mol": openmm_result["virial"],
        },
    )
    evidence = {
        "kind": "fixed_coordinate_energy_force_virial_pressure_parity",
        "reference_engine": "OpenMM",
        "reference_engine_role": "reference-only validation",
        "openmm_platform": openmm_result["platform"],
        "elapsed_wall_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "mlx": _without_arrays(mlx_result),
        "openmm": _without_arrays(openmm_result),
        "artifacts": {"fixed_arrays.npz": _artifact_record(arrays_path)},
    }
    return evidence, checks


def _evaluate_mlx_fixed(prepared_dir: Path) -> dict[str, Any]:
    artifact = load_prepared_mlx_artifact(prepared_dir, require_production=True)
    system, force_terms, _constraints = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    if system.cell is None or system.molecule_ids is None:
        raise DHFRNPTValidationError("MLX target requires cell and molecule identity")
    bound_terms = tuple(
        term.bind_pme_plan(system.cell)
        if getattr(term, "electrostatics", None) == "pme"
        else term
        for term in force_terms
    )
    nonbonded = next(
        (
            term
            for term in bound_terms
            if getattr(term, "electrostatics", None) == "pme"
        ),
        None,
    )
    if nonbonded is None:
        raise DHFRNPTValidationError("MLX target is missing its PME force term")
    neighbors = build_neighbor_list(
        system.positions,
        system.cell,
        cutoff=float(nonbonded.cutoff),
        skin=0.0,
        backend="mlx_cell_blocks",
        sort_pairs=False,
    )
    total_energy = mx.array(0.0, dtype=mx.float32)
    total_forces = mx.zeros_like(system.positions)
    components: dict[str, float] = {}
    for term in bound_terms:
        name = str(getattr(term, "name", type(term).__name__))
        energy, forces = term.energy_forces(
            system.positions,
            system.cell,
            pairs=neighbors.interactions,
        )
        total_energy = total_energy + energy
        total_forces = total_forces + forces
        mx.eval(energy, forces)
        components[name] = float(np.asarray(energy))
    mx.eval(total_energy, total_forces)
    analytic = analytic_configurational_virial_tensor(
        system.positions,
        total_forces,
        bound_terms,
        cell=system.cell,
        pairs=neighbors.interactions,
        masses=system.masses,
        molecule_ids=system.molecule_ids,
    )
    finite_difference = configurational_virial_tensor(
        system.positions,
        total_forces,
        bound_terms,
        cell=system.cell,
        pairs=neighbors.interactions,
        masses=system.masses,
        molecule_ids=system.molecule_ids,
        virial_mode="finite_difference_oracle",
    )
    mx.eval(analytic, finite_difference)
    analytic_components: dict[str, list[list[float]]] = {}
    finite_difference_components: dict[str, list[list[float]]] = {}
    for term in bound_terms:
        name = str(getattr(term, "name", type(term).__name__))
        term_analytic = analytic_configurational_virial_tensor(
            system.positions,
            total_forces,
            (term,),
            cell=system.cell,
            pairs=neighbors.interactions,
            masses=system.masses,
            molecule_ids=system.molecule_ids,
        )
        term_finite_difference = configurational_virial_tensor(
            system.positions,
            total_forces,
            (term,),
            cell=system.cell,
            pairs=neighbors.interactions,
            masses=system.masses,
            molecule_ids=system.molecule_ids,
            virial_mode="finite_difference_oracle",
        )
        mx.eval(term_analytic, term_finite_difference)
        analytic_components[name] = np.asarray(
            term_analytic,
            dtype=np.float64,
        ).tolist()
        finite_difference_components[name] = np.asarray(
            term_finite_difference,
            dtype=np.float64,
        ).tolist()
    nonbonded_local_analytic_components = (
        _mlx_nonbonded_local_analytic_virial_components(
            nonbonded,
            positions=system.positions,
            cell=system.cell,
            pairs=neighbors.interactions,
            masses=system.masses,
            molecule_ids=system.molecule_ids,
        )
    )
    plan = nonbonded.pme_plan
    return {
        "energy_kj_mol": float(np.asarray(total_energy)),
        "components": components,
        "forces": np.asarray(total_forces, dtype=np.float64) * 10.0,
        "analytic_virial": np.asarray(analytic, dtype=np.float64),
        "fd_virial": np.asarray(finite_difference, dtype=np.float64),
        "analytic_virial_components": analytic_components,
        "fd_virial_components": finite_difference_components,
        "nonbonded_local_analytic_virial_components": (
            nonbonded_local_analytic_components
        ),
        "volume_angstrom3": float(np.asarray(system.cell.volume)),
        "pme_plan_fingerprint": None if plan is None else plan.fingerprint,
        "neighbor": {
            "backend": neighbors.backend,
            "representation": neighbors.representation_kind,
            "fallback_reason": neighbors.fallback_reason,
            "pair_count": int(neighbors.pair_count),
        },
    }


def _evaluate_openmm_fixed(
    prepared_dir: Path,
    *,
    contract: dict[str, Any],
    platform_name: str,
) -> dict[str, Any]:
    prepared = load_prepared_system(prepared_dir)
    api, system = _build_openmm_system(contract)
    mm = api.openmm
    unit = api.unit
    cell_matrix = np.asarray(prepared.cell_matrix, dtype=np.float64)
    lengths = np.diag(cell_matrix)
    box_vectors = _openmm_box(mm, unit, lengths)
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    force_groups: dict[int, str] = {}
    class_names = {
        "HarmonicBondForce": "bond",
        "HarmonicAngleForce": "angle",
        "PeriodicTorsionForce": "dihedral",
        "NonbondedForce": "nonbonded",
    }
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
        name = class_names.get(type(force).__name__)
        if name is not None:
            force_groups[index] = name
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        mm.Platform.getPlatformByName(platform_name),
    )
    positions = np.asarray(prepared.positions, dtype=np.float64)
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions * 0.1 * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    components = {
        name: float(
            context.getState(getEnergy=True, groups={index})
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        for index, name in force_groups.items()
    }
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    virial = _parity._openmm_molecular_strain_virial(
        context,
        positions_angstrom=positions,
        lengths_angstrom=lengths,
        masses=np.asarray(prepared.masses, dtype=np.float64),
        molecule_ids=np.asarray(prepared.molecule_ids, dtype=np.int32),
        epsilon=1.0e-3,
    )
    virial_components = {
        name: np.asarray(
            _parity._openmm_molecular_strain_virial(
                context,
                positions_angstrom=positions,
                lengths_angstrom=lengths,
                masses=np.asarray(prepared.masses, dtype=np.float64),
                molecule_ids=np.asarray(prepared.molecule_ids, dtype=np.int32),
                epsilon=1.0e-3,
                groups={index},
            ),
            dtype=np.float64,
        ).tolist()
        for index, name in force_groups.items()
    }
    result = {
        "energy_kj_mol": float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
        "components": components,
        "forces": forces,
        "virial": np.asarray(virial, dtype=np.float64),
        "virial_components": virial_components,
        "volume_angstrom3": float(np.prod(lengths)),
        "platform": context.getPlatform().getName(),
        "openmm_version": api.openmm.version.version,
    }
    del context
    return result


def _mlx_nonbonded_local_analytic_virial_components(
    term,
    *,
    positions,
    cell,
    pairs,
    masses,
    molecule_ids,
) -> dict[str, list[float]]:
    component_names = (
        "lj_direct",
        "coulomb_direct",
        "coulomb_reciprocal",
    )
    source_lengths = cell.lengths

    def component_energies(strain):
        strained_cell = Cell(source_lengths * (1.0 + strain))
        strained_positions = molecularly_strained_positions(
            positions,
            source_cell=cell,
            target_cell=strained_cell,
            masses=masses,
            molecule_ids=molecule_ids,
        )
        regular_lj, _ = term._regular_lj_components(
            strained_positions,
            strained_cell,
            pairs,
        )
        exception_lj, _ = term._exception_lj_components(
            strained_positions,
            strained_cell,
        )
        pme_components = pme_coulomb_strain_components(
            strained_positions,
            term.charges,
            strained_cell,
            coulomb_constant=term.coulomb_constant,
            config=term.pme_config,
            direct_space_pairs=pairs,
        )
        _, _, corrections = term._periodic_coulomb_corrections(
            strained_positions,
            strained_cell,
        )
        return mx.stack(
            (
                regular_lj
                + exception_lj
                + term._dispersion_correction_energy(strained_cell),
                pme_components["coulomb_real"] + sum(corrections.values()),
                pme_components["coulomb_reciprocal"]
                + pme_components["coulomb_self"]
                + pme_components["coulomb_background"],
            )
        )

    zero = mx.zeros((3,), dtype=positions.dtype)
    diagonal_by_component = np.zeros((len(component_names), 3), dtype=np.float64)
    for component_index in range(len(component_names)):
        cotangent = mx.zeros(
            (len(component_names),),
            dtype=positions.dtype,
        ).at[component_index].add(1.0)
        _, derivatives = mx.vjp(
            component_energies,
            (zero,),
            (cotangent,),
        )
        derivative = derivatives[0]
        mx.eval(derivative)
        diagonal_by_component[component_index] = -np.asarray(
            derivative,
            dtype=np.float64,
        )
    return {
        name: diagonal_by_component[index].tolist()
        for index, name in enumerate(component_names)
    }


def _fixed_metrics(
    mlx_result: dict[str, Any],
    openmm_result: dict[str, Any],
    *,
    atom_count: int,
) -> dict[str, Any]:
    energy_delta = abs(mlx_result["energy_kj_mol"] - openmm_result["energy_kj_mol"])
    energy_denominator = abs(openmm_result["energy_kj_mol"])
    force_delta = mlx_result["forces"] - openmm_result["forces"]
    virial_delta = np.diag(mlx_result["analytic_virial"] - openmm_result["virial"])
    virial_denominator = max(float(np.max(np.abs(np.diag(openmm_result["virial"])))), 1e-12)
    volume = openmm_result["volume_angstrom3"]
    components = {}
    for name in sorted(set(mlx_result["components"]) & set(openmm_result["components"])):
        candidate = mlx_result["components"][name]
        reference = openmm_result["components"][name]
        absolute = abs(candidate - reference)
        components[name] = {
            "mlx_kj_mol": candidate,
            "openmm_kj_mol": reference,
            "absolute_error_kj_mol": absolute,
            "error_per_atom_kj_mol": absolute / atom_count,
            "relative_error": _relative_error(candidate, reference),
        }
    return {
        "energy_mlx_kj_mol": mlx_result["energy_kj_mol"],
        "energy_openmm_kj_mol": openmm_result["energy_kj_mol"],
        "energy_abs_error_kj_mol": energy_delta,
        "energy_error_per_atom_kj_mol": energy_delta / atom_count,
        "relative_energy_error": (
            energy_delta / energy_denominator
            if energy_denominator > 1e-12
            else energy_delta
        ),
        "components": components,
        "force_rms_error_kj_mol_nm": float(
            np.sqrt(np.mean(force_delta * force_delta))
        ),
        "force_max_error_kj_mol_nm": float(np.max(np.abs(force_delta))),
        "virial_diagonal_mlx_kj_mol": np.diag(
            mlx_result["analytic_virial"]
        ).tolist(),
        "virial_diagonal_openmm_kj_mol": np.diag(
            openmm_result["virial"]
        ).tolist(),
        "virial_error_per_atom_kj_mol": (
            float(np.max(np.abs(virial_delta))) / atom_count
        ),
        "relative_virial_error": (
            float(np.max(np.abs(virial_delta))) / virial_denominator
        ),
        "pressure_diagonal_max_error_kj_mol_a3": (
            float(np.max(np.abs(virial_delta))) / volume
        ),
        "mlx_analytic_fd_virial_error_per_atom_kj_mol": (
            float(
                np.max(
                    np.abs(
                        np.diag(
                            mlx_result["analytic_virial"]
                            - mlx_result["fd_virial"]
                        )
                    )
                )
            )
            / atom_count
        ),
    }


def _run_npt_stage(
    *,
    prepared_dir: Path,
    stage_dir: Path,
    contract: dict[str, Any],
    platform_name: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    started = time.perf_counter()
    workload = contract["workload"]
    gates = contract["npt_gates"]
    prepared = _npt_prepared(load_prepared_system(prepared_dir))
    stage_dir.mkdir(parents=True, exist_ok=True)
    mlx_trajectory = stage_dir / "mlx_trajectory.npz"
    mlx_checkpoint = stage_dir / "mlx_checkpoint.npz"
    mlx_result = _run_mlx_npt(
        prepared,
        workload=workload,
        seed=seed,
        out=mlx_trajectory,
        checkpoint_out=mlx_checkpoint,
        steps=int(workload["steps"]),
    )
    mlx_summary = _mlx_npt_summary(
        mlx_result,
        atom_count=prepared.atom_count,
    )
    mlx_pme_dynamic = bool(
        mlx_result.barostat_metadata["final_pme_plan_fingerprints"]
    )
    resume_evidence = None
    resume_artifacts: dict[str, dict[str, Any]] = {}
    declared_seeds = [int(value) for value in workload["seeds"]]
    if seed == declared_seeds[0]:
        continuous_snapshot = _mlx_resume_snapshot(mlx_result)
        del mlx_result
        mx.clear_cache()
        resume_evidence, resume_artifacts = _run_mlx_resume_parity(
            prepared,
            workload=workload,
            seed=seed,
            stage_dir=stage_dir,
            continuous=continuous_snapshot,
        )
    else:
        del mlx_result
        mx.clear_cache()
    openmm_result = _run_openmm_npt(
        prepared,
        contract=contract,
        platform_name=platform_name,
        seed=seed,
    )
    openmm_arrays = stage_dir / "openmm_samples.npz"
    _atomic_savez(openmm_arrays, openmm_result.pop("arrays"))
    engine_delta = {
        "mean_volume_ratio": abs(
            mlx_summary["mean_volume_ratio"]
            - openmm_result["mean_volume_ratio"]
        ),
        "mean_pressure_bar": abs(
            mlx_summary["mean_pressure_bar"]
            - openmm_result["mean_pressure_bar"]
        ),
    }
    checks = {
        "mlx_finite": mlx_summary["finite"],
        "openmm_finite": openmm_result["finite"],
        "mlx_attempt_count": (
            mlx_summary["barostat_attempts"]
            == gates["required_attempts_per_seed"]
        ),
        "openmm_attempt_schedule": (
            openmm_result["configured_barostat_attempts"]
            == gates["required_attempts_per_seed"]
        ),
        "mlx_cell_evolution": (
            mlx_summary["barostat_accepted"]
            >= gates["minimum_accepted_moves_per_seed"]
        ),
        "openmm_cell_evolution": (
            openmm_result["observed_cell_changes"]
            >= gates["minimum_accepted_moves_per_seed"]
        ),
        "mlx_constraints": (
            mlx_summary["maximum_constraint_error_angstrom"]
            <= gates["maximum_constraint_error_angstrom"]
        ),
        "openmm_constraints": (
            openmm_result["maximum_constraint_error_angstrom"]
            <= gates["maximum_constraint_error_angstrom"]
        ),
        "mlx_volume_bounds": _within(
            mlx_summary["minimum_volume_ratio"],
            gates["minimum_volume_ratio"],
            gates["maximum_volume_ratio"],
        )
        and _within(
            mlx_summary["maximum_volume_ratio"],
            gates["minimum_volume_ratio"],
            gates["maximum_volume_ratio"],
        ),
        "openmm_volume_bounds": _within(
            openmm_result["minimum_volume_ratio"],
            gates["minimum_volume_ratio"],
            gates["maximum_volume_ratio"],
        )
        and _within(
            openmm_result["maximum_volume_ratio"],
            gates["minimum_volume_ratio"],
            gates["maximum_volume_ratio"],
        ),
        "orthorhombic_cells": (
            mlx_summary["maximum_cell_off_diagonal_angstrom"]
            <= gates["orthorhombic_off_diagonal_tolerance_angstrom"]
            and openmm_result["maximum_cell_off_diagonal_angstrom"]
            <= gates["orthorhombic_off_diagonal_tolerance_angstrom"]
        ),
        "temperature_bounds": (
            mlx_summary["maximum_temperature_K"] <= gates["maximum_temperature_K"]
            and openmm_result["maximum_temperature_K"]
            <= gates["maximum_temperature_K"]
        ),
        "pressure_bounds": (
            mlx_summary["maximum_abs_pressure_bar"]
            <= gates["maximum_abs_pressure_bar"]
            and openmm_result["maximum_abs_pressure_bar"]
            <= gates["maximum_abs_pressure_bar"]
        ),
        "energy_stability": (
            mlx_summary["maximum_energy_excursion_per_atom_kj_mol"]
            <= gates["maximum_energy_excursion_per_atom_kj_mol"]
            and openmm_result["maximum_energy_excursion_per_atom_kj_mol"]
            <= gates["maximum_energy_excursion_per_atom_kj_mol"]
        ),
        "aggregate_volume_compatibility": (
            engine_delta["mean_volume_ratio"]
            <= gates["maximum_engine_mean_volume_ratio_delta"]
        ),
        "aggregate_pressure_compatibility": (
            engine_delta["mean_pressure_bar"]
            <= gates["maximum_engine_mean_pressure_delta_bar"]
        ),
        "mlx_pme_dynamic": mlx_pme_dynamic,
    }
    if resume_evidence is not None:
        checks["mlx_checkpoint_resume_parity"] = bool(
            resume_evidence["passed"]
        )
    evidence = {
        "kind": "bounded_multi_opportunity_anisotropic_npt",
        "seed": seed,
        "reference_engine": "OpenMM",
        "reference_engine_role": "reference-only validation",
        "openmm_platform": openmm_result["platform"],
        "elapsed_wall_seconds": time.perf_counter() - started,
        "mlx": mlx_summary,
        "openmm": openmm_result,
        "engine_delta": engine_delta,
        "checkpoint_resume": resume_evidence,
        "artifacts": {
            "mlx_trajectory.npz": _artifact_record(mlx_trajectory),
            "mlx_checkpoint.npz": _artifact_record(mlx_checkpoint),
            "openmm_samples.npz": _artifact_record(openmm_arrays),
            **resume_artifacts,
        },
    }
    return evidence, checks


def _run_mlx_npt(
    prepared,
    *,
    workload: dict[str, Any],
    seed: int,
    out: Path,
    checkpoint_out: Path,
    steps: int,
    resume_checkpoint: Path | None = None,
):
    return run_mlx(
        prepared,
        out=out,
        checkpoint_out=checkpoint_out,
        resume_checkpoint=resume_checkpoint,
        steps=steps,
        sample_interval=int(workload["sample_interval"]),
        diagnostic_interval=int(workload["diagnostic_interval"]),
        dt=float(workload["dt_ps"]),
        temperature=float(workload["temperature_K"]),
        friction=float(workload["friction_per_ps"]),
        seed=seed,
        pressure_atm=float(workload["pressure_atm"]),
        barostat_interval=int(workload["barostat"]["interval"]),
        barostat_mode="anisotropic",
        barostat_axes=tuple(workload["barostat"]["axes"]),
        barostat_max_log_volume_scale=float(
            workload["barostat"]["max_log_volume_scale"]
        ),
        restraint_k=0.0,
        require_production=True,
        minimize_steps=0,
        equilibration_steps=0,
        constraint_max_iterations=20,
        eager_nonbonded_pair_limit=0,
    )


def _run_mlx_resume_parity(
    prepared,
    *,
    workload: dict[str, Any],
    seed: int,
    stage_dir: Path,
    continuous,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    total_steps = int(workload["steps"])
    first_steps = total_steps // 2
    second_steps = total_steps - first_steps
    interval = int(workload["barostat"]["interval"])
    if first_steps % interval != 0 or second_steps % interval != 0:
        msg = "target split must preserve complete barostat intervals"
        raise DHFRNPTValidationError(msg)

    first_trajectory = stage_dir / "mlx_resume_first_trajectory.npz"
    split_checkpoint = stage_dir / "mlx_resume_split_checkpoint.npz"
    second_trajectory = stage_dir / "mlx_resume_second_trajectory.npz"
    final_checkpoint = stage_dir / "mlx_resume_final_checkpoint.npz"
    first = _run_mlx_npt(
        prepared,
        workload=workload,
        seed=seed,
        out=first_trajectory,
        checkpoint_out=split_checkpoint,
        steps=first_steps,
    )
    first_snapshot = _mlx_resume_snapshot(first)
    del first
    mx.clear_cache()
    resumed = _run_mlx_npt(
        prepared,
        workload=workload,
        seed=seed,
        out=second_trajectory,
        checkpoint_out=final_checkpoint,
        resume_checkpoint=split_checkpoint,
        steps=second_steps,
    )
    resumed_snapshot = _mlx_resume_snapshot(resumed)
    del resumed
    mx.clear_cache()
    evidence = _mlx_resume_parity_evidence(
        continuous=continuous,
        first=first_snapshot,
        resumed=resumed_snapshot,
        split_step=first_steps,
    )
    artifacts = {
        path.name: _artifact_record(path)
        for path in (
            first_trajectory,
            split_checkpoint,
            second_trajectory,
            final_checkpoint,
        )
    }
    return evidence, artifacts


def _mlx_resume_snapshot(result) -> dict[str, Any]:
    sampled_fields = (
        "sampled_positions",
        "sampled_velocities",
        "sampled_time",
        "diagnostic_time",
        "cell_history",
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "temperature",
        "virial_tensor",
        "pressure_tensor",
        "pressure",
        "constraint_max_error",
        "sampled_steps",
        "diagnostic_steps",
        "pair_count",
    )
    return {
        "final_state": {
            name: np.asarray(getattr(result.final_state, name)).copy()
            for name in ("positions", "velocities", "forces")
        },
        "final_cell": np.asarray(result.final_cell.matrix).copy(),
        "sampled": {
            name: np.asarray(getattr(result, name)).copy()
            for name in sampled_fields
        },
        "potential_energy_by_term": {
            name: np.asarray(values).copy()
            for name, values in result.potential_energy_by_term.items()
        },
        "barostat_attempts": int(result.barostat_attempts),
        "barostat_accepted": int(result.barostat_accepted),
        "proposal_history": copy.deepcopy(
            result.barostat_metadata["proposal_history"]
        ),
        "rng_state": copy.deepcopy(result.barostat_metadata["rng_state"]),
    }


def _mlx_resume_parity_evidence(
    *,
    continuous: dict[str, Any],
    first: dict[str, Any],
    resumed: dict[str, Any],
    split_step: int,
) -> dict[str, Any]:
    tolerance = 1.0e-6

    def max_abs(left, right) -> float:
        delta = np.abs(
            np.asarray(left, dtype=np.float64)
            - np.asarray(right, dtype=np.float64)
        )
        return float(np.max(delta)) if delta.size else 0.0

    final_errors = {
        name: max_abs(
            resumed["final_state"][name],
            continuous["final_state"][name],
        )
        for name in ("positions", "velocities", "forces")
    }
    final_errors["cell"] = max_abs(
        resumed["final_cell"],
        continuous["final_cell"],
    )
    sampled_errors = {}
    for name in (
        "sampled_positions",
        "sampled_velocities",
        "sampled_time",
        "diagnostic_time",
        "cell_history",
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "temperature",
        "virial_tensor",
        "pressure_tensor",
        "pressure",
        "constraint_max_error",
    ):
        combined = np.concatenate(
            (
                np.asarray(first["sampled"][name]),
                np.asarray(resumed["sampled"][name])[1:],
            ),
            axis=0,
        )
        sampled_errors[name] = max_abs(
            combined,
            continuous["sampled"][name],
        )
    index_fields_match = all(
        np.array_equal(
            np.concatenate(
                (
                    np.asarray(first["sampled"][name]),
                    np.asarray(resumed["sampled"][name])[1:],
                ),
                axis=0,
            ),
            np.asarray(continuous["sampled"][name]),
        )
        for name in (
            "sampled_steps",
            "diagnostic_steps",
            "pair_count",
        )
    )
    term_errors = {}
    for name in continuous["potential_energy_by_term"]:
        combined = np.concatenate(
            (
                np.asarray(first["potential_energy_by_term"][name]),
                np.asarray(resumed["potential_energy_by_term"][name])[1:],
            ),
            axis=0,
        )
        term_errors[name] = max_abs(
            combined,
            continuous["potential_energy_by_term"][name],
        )
    metadata_match = (
        resumed["barostat_attempts"] == continuous["barostat_attempts"]
        and resumed["barostat_accepted"] == continuous["barostat_accepted"]
        and resumed["proposal_history"] == continuous["proposal_history"]
        and resumed["rng_state"] == continuous["rng_state"]
    )
    maximum_error = max(
        (
            *final_errors.values(),
            *sampled_errors.values(),
            *term_errors.values(),
        )
    )
    return {
        "passed": (
            maximum_error <= tolerance
            and index_fields_match
            and metadata_match
        ),
        "split_step": split_step,
        "tolerance": tolerance,
        "maximum_abs_error": maximum_error,
        "final_state_max_abs_errors": final_errors,
        "sampled_max_abs_errors": sampled_errors,
        "energy_term_max_abs_errors": term_errors,
        "index_fields_match": index_fields_match,
        "barostat_and_rng_state_match": metadata_match,
    }


def _npt_prepared(prepared):
    protocol = {
        **prepared.metadata.protocol_metadata,
        "ensemble": "NPT",
        "proof_mode": "short_npt",
        "barostat": "anisotropic",
        "npt_barostat": True,
    }
    return replace(
        prepared,
        metadata=replace(prepared.metadata, protocol_metadata=protocol),
    )


def _mlx_npt_summary(result, *, atom_count: int) -> dict[str, Any]:
    volumes = np.asarray(result.volume, dtype=np.float64)
    initial_volume = float(volumes[0])
    volume_ratios = volumes / initial_volume
    temperatures = np.asarray(result.temperature, dtype=np.float64)
    pressures_bar = (
        np.asarray(result.pressure, dtype=np.float64)
        * PRESSURE_KJ_MOL_A3_TO_BAR
    )
    total_energies = np.asarray(result.total_energy, dtype=np.float64)
    cells = np.asarray(result.cell_history, dtype=np.float64)
    finite = _all_finite(
        {
            "volumes": volumes,
            "temperatures": temperatures,
            "pressures": pressures_bar,
            "energies": total_energies,
            "cells": cells,
            "positions": np.asarray(result.sampled_positions),
            "velocities": np.asarray(result.sampled_velocities),
        }
    )
    return {
        "finite": finite,
        "sample_count": int(cells.shape[0]),
        "barostat_attempts": int(result.barostat_attempts),
        "barostat_accepted": int(result.barostat_accepted),
        "minimum_volume_ratio": float(np.min(volume_ratios)),
        "maximum_volume_ratio": float(np.max(volume_ratios)),
        "mean_volume_ratio": float(np.mean(volume_ratios)),
        "maximum_cell_off_diagonal_angstrom": _maximum_off_diagonal(cells),
        "maximum_constraint_error_angstrom": float(
            np.max(np.asarray(result.constraint_max_error, dtype=np.float64))
        ),
        "maximum_temperature_K": float(np.max(temperatures)),
        "mean_pressure_bar": float(np.mean(pressures_bar)),
        "maximum_abs_pressure_bar": float(np.max(np.abs(pressures_bar))),
        "maximum_energy_excursion_per_atom_kj_mol": (
            float(np.max(total_energies) - np.min(total_energies)) / atom_count
        ),
        "final_pme_plan_fingerprints": list(
            result.barostat_metadata["final_pme_plan_fingerprints"]
        ),
    }


def _run_openmm_npt(
    prepared,
    *,
    contract: dict[str, Any],
    platform_name: str,
    seed: int,
) -> dict[str, Any]:
    workload = contract["workload"]
    api, system = _build_openmm_system(contract)
    mm = api.openmm
    unit = api.unit
    interval = int(workload["barostat"]["interval"])
    barostat = mm.MonteCarloAnisotropicBarostat(
        mm.Vec3(
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
            float(workload["pressure_bar"]),
        )
        * unit.bar,
        float(workload["temperature_K"]) * unit.kelvin,
        True,
        True,
        True,
        interval,
    )
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        float(workload["temperature_K"]) * unit.kelvin,
        float(workload["friction_per_ps"]) / unit.picosecond,
        float(workload["dt_ps"]) * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(
        system,
        integrator,
        mm.Platform.getPlatformByName(platform_name),
    )
    positions = np.asarray(prepared.positions, dtype=np.float64)
    initial_lengths = np.diag(np.asarray(prepared.cell_matrix, dtype=np.float64))
    context.setPeriodicBoxVectors(*_openmm_box(mm, unit, initial_lengths))
    context.setPositions(positions * 0.1 * unit.nanometer)
    context.applyConstraints(integrator.getConstraintTolerance())
    context.setVelocitiesToTemperature(
        float(workload["temperature_K"]) * unit.kelvin,
        seed,
    )
    context.applyVelocityConstraints(integrator.getConstraintTolerance())
    sampled = [_openmm_sample(context, prepared, api)]
    for _ in range(int(workload["steps"]) // interval):
        integrator.step(interval)
        sampled.append(_openmm_sample(context, prepared, api))
    arrays = {
        name: np.asarray([sample[name] for sample in sampled])
        for name in (
            "cell_matrix_angstrom",
            "volume_angstrom3",
            "potential_energy_kj_mol",
            "kinetic_energy_kj_mol",
            "total_energy_kj_mol",
            "temperature_K",
            "pressure_bar",
            "constraint_error_angstrom",
        )
    }
    volumes = arrays["volume_angstrom3"]
    ratios = volumes / volumes[0]
    total_energies = arrays["total_energy_kj_mol"]
    cell_changes = int(
        np.count_nonzero(
            np.max(
                np.abs(np.diff(arrays["cell_matrix_angstrom"], axis=0)),
                axis=(1, 2),
            )
            > 1.0e-7
        )
    )
    result = {
        "finite": _all_finite(arrays),
        "platform": context.getPlatform().getName(),
        "openmm_version": api.openmm.version.version,
        "sample_count": len(sampled),
        "configured_barostat_attempts": int(workload["steps"]) // interval,
        "observed_cell_changes": cell_changes,
        "minimum_volume_ratio": float(np.min(ratios)),
        "maximum_volume_ratio": float(np.max(ratios)),
        "mean_volume_ratio": float(np.mean(ratios)),
        "maximum_cell_off_diagonal_angstrom": _maximum_off_diagonal(
            arrays["cell_matrix_angstrom"]
        ),
        "maximum_constraint_error_angstrom": float(
            np.max(arrays["constraint_error_angstrom"])
        ),
        "maximum_temperature_K": float(np.max(arrays["temperature_K"])),
        "mean_pressure_bar": float(np.mean(arrays["pressure_bar"])),
        "maximum_abs_pressure_bar": float(np.max(np.abs(arrays["pressure_bar"]))),
        "maximum_energy_excursion_per_atom_kj_mol": (
            float(np.max(total_energies) - np.min(total_energies))
            / prepared.atom_count
        ),
        "arrays": arrays,
    }
    del context
    del integrator
    return result


def _openmm_sample(context, prepared, api) -> dict[str, Any]:
    unit = api.unit
    state = context.getState(
        getEnergy=True,
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True,
    )
    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.angstrom
    )
    cell = np.asarray(vectors, dtype=np.float64)
    lengths = np.diag(cell)
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
        dtype=np.float64,
    )
    velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(
            unit.angstrom / unit.picosecond
        ),
        dtype=np.float64,
    )
    masses = np.asarray(prepared.masses, dtype=np.float64)
    molecule_ids = np.asarray(prepared.molecule_ids, dtype=np.int32)
    configurational = _parity._openmm_molecular_strain_virial(
        context,
        positions_angstrom=positions,
        lengths_angstrom=lengths,
        masses=masses,
        molecule_ids=molecule_ids,
        epsilon=1.0e-3,
    )
    context.setPeriodicBoxVectors(*_openmm_box(api.openmm, unit, lengths))
    context.setPositions(positions * 0.1 * unit.nanometer)
    context.setVelocities(velocities * 0.1 * unit.nanometer / unit.picosecond)
    kinetic_tensor = _molecular_kinetic_tensor(
        velocities,
        masses,
        molecule_ids,
    )
    volume = float(np.prod(lengths))
    pressure_bar = float(
        np.trace((kinetic_tensor + configurational) / volume)
        / 3.0
        * PRESSURE_KJ_MOL_A3_TO_BAR
    )
    kinetic_energy = float(
        state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    potential_energy = float(
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    degrees_of_freedom = max(
        1,
        3 * prepared.atom_count - int(prepared.constraints.shape[0]) - 3,
    )
    temperature = 2.0 * kinetic_energy / (
        degrees_of_freedom * GAS_CONSTANT_KJ_MOL_K
    )
    return {
        "cell_matrix_angstrom": cell,
        "volume_angstrom3": volume,
        "potential_energy_kj_mol": potential_energy,
        "kinetic_energy_kj_mol": kinetic_energy,
        "total_energy_kj_mol": potential_energy + kinetic_energy,
        "temperature_K": temperature,
        "pressure_bar": pressure_bar,
        "constraint_error_angstrom": _constraint_error(
            positions,
            np.asarray(prepared.constraints, dtype=np.int32),
            np.asarray(prepared.constraint_distance, dtype=np.float64),
        ),
    }


def _build_openmm_system(contract: dict[str, Any]):
    api = _preparation._common._load_openmm()
    app = api.app
    unit = api.unit
    pdb = app.PDBFile(str(contract["target"]["pdb_path"]))
    force_field = app.ForceField("amber99sb.xml", "tip3p.xml")
    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=9.0 * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
        hydrogenMass=1.5 * unit.amu,
    )
    nonbonded = _preparation._common._single_force(
        api.openmm.NonbondedForce,
        system,
        "NonbondedForce",
    )
    pme = contract["workload"]["pme"]
    nonbonded.setEwaldErrorTolerance(float(pme["ewald_error_tolerance"]))
    nonbonded.setUseDispersionCorrection(bool(pme["dispersion_correction"]))
    nonbonded.setPMEParameters(
        float(pme["alpha_per_angstrom"]) * 10.0 / unit.nanometer,
        *(int(value) for value in pme["mesh_shape"]),
    )
    cmm = _preparation._common._single_force(
        api.openmm.CMMotionRemover,
        system,
        "CMMotionRemover",
    )
    cmm.setFrequency(int(contract["workload"]["center_of_mass_motion_interval"]))
    if system.getNumParticles() != int(contract["target"]["atom_count"]):
        raise DHFRNPTValidationError("OpenMM target particle count drifted")
    return api, system


def _molecular_kinetic_tensor(
    velocities_angstrom_ps: np.ndarray,
    masses_dalton: np.ndarray,
    molecule_ids: np.ndarray,
) -> np.ndarray:
    molecule_count = int(np.max(molecule_ids)) + 1
    molecule_masses = np.bincount(
        molecule_ids,
        weights=masses_dalton,
        minlength=molecule_count,
    )
    momenta = np.zeros((molecule_count, 3), dtype=np.float64)
    np.add.at(
        momenta,
        molecule_ids,
        masses_dalton[:, None] * velocities_angstrom_ps,
    )
    molecule_velocities = momenta / molecule_masses[:, None]
    return (
        molecule_velocities.T
        @ (molecule_masses[:, None] * molecule_velocities)
        * 0.01
    )


def _constraint_error(
    positions: np.ndarray,
    pairs: np.ndarray,
    target: np.ndarray,
) -> float:
    if pairs.size == 0:
        return 0.0
    distances = np.linalg.norm(
        positions[pairs[:, 0]] - positions[pairs[:, 1]],
        axis=1,
    )
    return float(np.max(np.abs(distances - target)))


def _openmm_box(mm, unit, lengths_angstrom: np.ndarray):
    a, b, c = np.asarray(lengths_angstrom, dtype=np.float64) * 0.1
    return (
        mm.Vec3(float(a), 0.0, 0.0),
        mm.Vec3(0.0, float(b), 0.0),
        mm.Vec3(0.0, 0.0, float(c)),
    ) * unit.nanometer


def _maximum_off_diagonal(cells: np.ndarray) -> float:
    values = np.asarray(cells, dtype=np.float64)
    diagonal = np.zeros_like(values)
    indices = np.arange(3)
    diagonal[:, indices, indices] = values[:, indices, indices]
    return float(np.max(np.abs(values - diagonal)))


def _relative_error(candidate: float, reference: float) -> float:
    absolute = abs(candidate - reference)
    return absolute / abs(reference) if abs(reference) > 1.0e-12 else absolute


def _within(value: float, minimum: float, maximum: float) -> bool:
    return bool(minimum <= value <= maximum)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_all_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, float | int):
        return bool(math.isfinite(float(value)))
    return True


def _without_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in payload.items()
        if not isinstance(value, np.ndarray)
    }


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


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fixed", "npt"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--platform", default="Reference")
    args = parser.parse_args(argv)
    if args.stage == "fixed" and args.seed is not None:
        parser.error("--seed is only valid for the NPT stage")
    if args.stage == "npt" and args.seed is None:
        parser.error("--seed is required for the NPT stage")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
