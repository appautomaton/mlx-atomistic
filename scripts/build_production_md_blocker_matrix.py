"""Build the Phase 3 production-MD blocker matrix and readiness report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TAXONOMY_CATEGORIES = (
    "artifact_source",
    "preparation",
    "topology_terms",
    "forcefield_terms",
    "constraints_hmr_virtual_sites",
    "electrostatics_pme",
    "npt_barostat",
    "integrator_protocol",
    "stability_finiteness",
    "parity_tolerance",
    "performance_runtime",
    "output_restart",
    "dependency_boundary",
)

GPCRMD_FIXTURE_ID = "gpcrmd-729-beta1-5f8u-cyanopindolol"
GPCRMD_ATOM_COUNT = 92_001
GPCRMD_REQUIRED_SOURCE_IDS = {15_286, 17_686, 15_290, 17_687}
GPCRMD_REQUIRED_SOURCE_ROLES = {"topology", "model", "parameters", "protocol"}
GPCRMD_REQUIRED_PROTOCOL_MEMBERS = {
    "rep_1/input",
    "rep_1/input.coor",
    "rep_1/input.vel",
    "rep_1/input.xsc",
    "rep_1/log.txt",
}
GPCRMD_REQUIRED_FORCE_TERMS = {
    "charmm_cmap",
    "charmm_harmonic_improper",
    "harmonic_angle",
    "harmonic_bond",
    "nbfix_pair_overrides",
    "nonbonded_exception",
    "nonbonded_lj_coulomb",
    "periodic_dihedral",
    "urey_bradley",
}
GPCRMD_REQUIRED_PARITY_CHECKS = {
    "component_energy_per_atom",
    "component_relative_energy",
    "force_maximum",
    "force_rms",
    "manifest_match",
    "mlx_lazy_topology",
    "mlx_neighbor_blocks",
    "mlx_no_neighbor_fallback",
    "mlx_pair_cache_unmaterialized",
    "openmm_resolved_pme",
    "total_energy_per_atom",
    "total_relative_energy",
}


def build_blocker_matrix(
    *,
    candidate: dict[str, Any],
    openmm: dict[str, Any],
    mlx: dict[str, Any],
) -> dict[str, Any]:
    """Normalize fixture, OpenMM, and MLX evidence into the blocker taxonomy."""

    if (
        candidate.get("kind") == "gpcrmd_fixture_acquisition"
        and openmm.get("kind") == "mlx_atomistic.gpcrmd_pme_parity"
        and mlx.get("kind") == "gpcrmd_source_protocol_benchmark"
    ):
        return _build_gpcrmd_closure_matrix(candidate, openmm, mlx)

    fixture_id = (
        candidate.get("fixture", {}).get("id")
        or openmm.get("fixture_id")
        or mlx.get("fixture", {}).get("id")
    )
    entries = {category: _base_entry(category, fixture_id) for category in TAXONOMY_CATEGORIES}

    _apply_candidate(entries, candidate)
    _apply_openmm(entries, openmm)
    _apply_mlx(entries, mlx)
    _finalize_defaults(entries, candidate, openmm, mlx)

    ordered = [entries[category] for category in TAXONOMY_CATEGORIES]
    status = "blocked" if any(item["prevents_bounded_pass"] for item in ordered) else "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "change": "production-md-readiness-fixture-probe",
        "fixture_id": fixture_id,
        "status": status,
        "bounded_pass": status == "passed",
        "summary": {
            "candidate_status": candidate.get("status"),
            "openmm_status": openmm.get("status"),
            "mlx_status": mlx.get("status"),
            "blocking_categories": [
                item["category"] for item in ordered if item["prevents_bounded_pass"]
            ],
        },
        "entries": ordered,
    }


def _build_gpcrmd_closure_matrix(
    candidate: dict[str, Any],
    parity: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    fixture_id = str(
        candidate.get("target_id")
        or parity.get("fixture")
        or runtime.get("config", {}).get("target_id")
    )
    entries = {category: _base_entry(category, fixture_id) for category in TAXONOMY_CATEGORIES}
    acquisition_command = candidate.get("command", {}).get(
        "reproduction_command",
        "scripts/acquire_gpcrmd_fixture.py",
    )
    parity_command = (
        "uv run --with openmm python scripts/run_gpcrmd_pme_parity.py "
        "--source-manifest results/gpcrmd-pme-runtime-closure/source/fixture-manifest.json "
        "--cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 "
        "--mlx-prepared results/gpcrmd-pme-runtime-closure/prepared "
        "--platform OpenCL --out results/gpcrmd-pme-runtime-closure/parity"
    )
    runtime_command = (
        "uv run python -m mlx_atomistic.prep.gpcrmd_benchmark "
        "--target-id gpcrmd-729-beta1-5f8u-cyanopindolol "
        "--prepared results/gpcrmd-pme-runtime-closure/prepared "
        "--protocol-manifest "
        "results/gpcrmd-pme-runtime-closure/prepared/mlx-workload-manifest.json "
        "--warmups 1 --measured-steps 2 --checkpoint-restart "
        "--out results/gpcrmd-pme-runtime-closure/runtime --force --json"
    )

    source_files: dict[int, dict[str, Any]] = {}
    for item in candidate.get("files", []):
        file_id = _integer_value(item.get("file_id"))
        if file_id is not None:
            source_files[file_id] = item
    source_ids = set(source_files)
    source_roles = {str(item.get("role")) for item in source_files.values()}
    required_files = [source_files.get(file_id) for file_id in GPCRMD_REQUIRED_SOURCE_IDS]
    source_file_metadata_ok = all(
        item is not None
        and _number_above(item.get("size_bytes"), 0)
        and _is_sha256(item.get("sha256"))
        and str(item.get("source_url", "")).startswith("https://www.gpcrmd.org/")
        for item in required_files
    )
    protocol_file = source_files.get(17_687, {})
    protocol_members = {
        str(item.get("normalized_name")): item
        for item in protocol_file.get("archive_members", [])
        if item.get("kind") == "file"
    }
    protocol_archive_ok = (
        protocol_file.get("archive_extraction") in {"extracted", "reused"}
        and set(protocol_members) >= GPCRMD_REQUIRED_PROTOCOL_MEMBERS
        and all(
            _number_above(protocol_members[name].get("size_bytes"), 0)
            and _is_sha256(protocol_members[name].get("sha256"))
            for name in GPCRMD_REQUIRED_PROTOCOL_MEMBERS
        )
    )
    source_ok = (
        candidate.get("status") == "complete"
        and not candidate.get("blockers")
        and candidate.get("target_id") == GPCRMD_FIXTURE_ID
        and source_ids >= GPCRMD_REQUIRED_SOURCE_IDS
        and source_roles >= GPCRMD_REQUIRED_SOURCE_ROLES
        and source_file_metadata_ok
        and protocol_archive_ok
    )
    _set_closure_entry(
        entries,
        "artifact_source",
        passed=source_ok,
        command=str(acquisition_command),
        observed=(
            "authenticated GPCRmd acquisition recorded all four required source IDs, "
            "sizes, hashes, and archive extraction"
        ),
        context="source/fixture-manifest.json",
        acceptance=["AC1"],
        next_decision="reacquire or reconcile the first missing source file",
    )

    protocol_validation = runtime.get("protocol_validation", {})
    source_settings = protocol_validation.get("source_settings", {})
    manifest_hashes = {
        runtime.get("config", {}).get("protocol_manifest_sha256"),
        protocol_validation.get("declared_manifest_sha256"),
        protocol_validation.get("rebuilt_manifest_sha256"),
    }
    preparation_ok = (
        protocol_validation.get("status") == "ready"
        and protocol_validation.get("strict_production_load") is True
        and protocol_validation.get("manifest_matches_prepared") is True
        and not protocol_validation.get("blockers")
        and source_settings.get("atom_count") == GPCRMD_ATOM_COUNT
        and runtime.get("config", {}).get("target_id") == fixture_id
        and len(manifest_hashes) == 1
        and _is_sha256(next(iter(manifest_hashes), None))
    )
    _set_closure_entry(
        entries,
        "preparation",
        passed=preparation_ok,
        command=runtime_command,
        observed=(
            "strict production load rebuilt the source workload manifest for "
            f"{_format_integer(source_settings.get('atom_count'))} atoms without drift"
        ),
        context="runtime/gpcrmd_performance.json:protocol_validation",
        acceptance=["AC2"],
        next_decision="fix the first prepared-artifact or manifest mismatch",
    )

    parity_checks = parity.get("checks", {})
    cases = list(runtime.get("cases", []))
    cases_by_phase = {
        str(case.get("phase")): case for case in cases if case.get("phase") is not None
    }
    required_phase_names = {"warmup", "measured", "restart"}
    required_cases = [cases_by_phase.get(name) for name in sorted(required_phase_names)]
    runtime_contract_ok = all(case is not None for case in required_cases) and all(
        case.get("topology_pair_policy") == "lazy"
        and case.get("neighbor_backend") == "mlx_cell_blocks"
        and case.get("neighbor_representation") == "NeighborBlocks"
        and case.get("dense_or_tiled_fallback_used") is False
        and case.get("checks", {}).get("runtime_contract_matches") is True
        for case in required_cases
        if case is not None
    )
    topology_ok = (
        parity_checks.get("mlx_lazy_topology") is True
        and parity_checks.get("mlx_neighbor_blocks") is True
        and parity_checks.get("mlx_no_neighbor_fallback") is True
        and parity_checks.get("mlx_pair_cache_unmaterialized") is True
        and runtime_contract_ok
    )
    _set_closure_entry(
        entries,
        "topology_terms",
        passed=topology_ok,
        command=runtime_command,
        observed=(
            "lazy topology used shared mlx_cell_blocks/NeighborBlocks in parity, "
            "warmup, measured, and restart phases with no dense/tiled fallback"
        ),
        context="parity/gpcrmd_pme_parity_report.json and runtime/gpcrmd_performance.json",
        acceptance=["AC3", "AC4", "AC5"],
        next_decision="fix the first lazy-topology or shared-neighbor contract failure",
    )

    artifact_readiness = parity.get("artifact_readiness", {})
    artifact_required_terms = set(
        artifact_readiness.get("metadata", {}).get("required_terms", [])
    )
    component_metrics = parity.get("energies", {}).get("component_metrics", {})
    tolerances = parity.get("tolerances", {})
    energy_per_atom_tolerance = tolerances.get("energy_per_atom_kj_mol")
    relative_energy_tolerance = tolerances.get("relative_energy_error")
    component_metrics_ok = bool(component_metrics) and all(
        _number_at_most(
            metric.get("energy_error_per_atom_kj_mol"),
            energy_per_atom_tolerance,
        )
        and _number_at_most(metric.get("relative_error"), relative_energy_tolerance)
        for metric in component_metrics.values()
    )
    forcefield_ok = (
        artifact_readiness.get("status") == "ready"
        and not artifact_readiness.get("blockers")
        and artifact_required_terms >= GPCRMD_REQUIRED_FORCE_TERMS
        and parity_checks.get("component_energy_per_atom") is True
        and parity_checks.get("component_relative_energy") is True
        and component_metrics_ok
    )
    _set_closure_entry(
        entries,
        "forcefield_terms",
        passed=forcefield_ok,
        command=parity_command,
        observed=(
            "CHARMM bonds, angles, Urey-Bradley, proper and harmonic-improper "
            "torsions, CMAP, NBFIX, and nonbonded terms passed component parity"
        ),
        context="parity/gpcrmd_pme_parity_report.json:energies.component_metrics",
        acceptance=["AC2", "AC3"],
        next_decision="fix the first failing CHARMM component or unsupported term",
    )

    constraints_count = source_settings.get("constraints_count")
    hmr_hydrogen_count = source_settings.get("hmr_selected_hydrogen_count")
    hmr_target_mass = source_settings.get("hmr_target_hydrogen_mass_da")
    constraints_ok = (
        constraints_count == 78_896
        and source_settings.get("hmr_status") == "represented_by_masses"
        and hmr_hydrogen_count == 58_952
        and _numbers_close(hmr_target_mass, 4.032)
        and all(case is not None for case in required_cases)
        and all(
            case.get("hmr_preserved") is True
            and case.get("constraint_error_finite") is True
            for case in required_cases
            if case is not None
        )
    )
    _set_closure_entry(
        entries,
        "constraints_hmr_virtual_sites",
        passed=constraints_ok,
        command=runtime_command,
        observed=(
            f"{_format_integer(constraints_count)} source constraints and "
            f"{_format_number(hmr_target_mass)} Da HMR across "
            f"{_format_integer(hmr_hydrogen_count)} bonded hydrogens were preserved; "
            "virtual sites are not required by this fixture"
        ),
        context="runtime/gpcrmd_performance.json:protocol_validation.source_settings",
        acceptance=["AC2", "AC4", "AC5"],
        next_decision="fix the first constraint, HMR, or fixture-requirement mismatch",
    )

    pme_readiness = parity.get("mlx", {}).get("pme_readiness", {})
    pme_mesh = pme_readiness.get("mesh_shape")
    openmm_pme = parity.get("openmm", {}).get("resolved_pme", {})
    pme_ok = (
        pme_readiness.get("status") == "ready"
        and pme_readiness.get("production_executable") is True
        and not pme_readiness.get("blockers")
        and pme_readiness.get("atom_count") == GPCRMD_ATOM_COUNT
        and pme_readiness.get("background_policy") == "reject_non_neutral"
        and pme_mesh == [78, 78, 108]
        and pme_readiness.get("assignment_order") == 5
        and _numbers_close(pme_readiness.get("real_cutoff"), 9.0)
        and pme_readiness.get("checks", {}).get("neutrality") is True
        and parity_checks.get("openmm_resolved_pme") is True
        and parity.get("openmm", {}).get("resolved_pme_matches_manifest") is True
        and openmm_pme.get("mesh_shape") == pme_mesh
        and all(case is not None for case in required_cases)
        and all(
            case.get("pme_execution_plan_build_count") == 1
            and _number_above(case.get("pme_execution_plan_reuse_count"), 0)
            for case in required_cases
            if case is not None
        )
    )
    _set_closure_entry(
        entries,
        "electrostatics_pme",
        passed=pme_ok,
        command=parity_command,
        observed=(
            "neutral fixed-cell PME passed at mesh "
            f"{_format_mesh(pme_mesh)} with explicit reject_non_neutral policy, "
            "one plan per phase, and recorded reuse"
        ),
        context="parity/gpcrmd_pme_parity_report.json:mlx.pme_readiness",
        acceptance=["AC2", "AC3", "AC4"],
        next_decision="fix the first PME readiness, resolved-grid, or plan-reuse failure",
    )

    entries["npt_barostat"].update(
        {
            "status": "anti_goal",
            "command": runtime_command,
            "observed_result": (
                "source-faithful bounded row is fixed-cell NVT; production NPT and "
                "cell changes remain outside this closure"
            ),
            "smallest_reproduction_context": "runtime/gpcrmd_performance.json:config",
            "affected_acceptance_criteria": ["AC4", "AC6"],
            "next_implementation_decision": (
                "open a separate NPT/analytic-virial objective when evidence requires it"
            ),
            "prevents_bounded_pass": False,
        }
    )

    config = runtime.get("config", {})
    dt_ps = _finite_number(config.get("dt_ps"))
    dt_fs = None if dt_ps is None else dt_ps * 1000.0
    protocol_ok = (
        runtime.get("status") == "passed"
        and config.get("ensemble") == "NVT"
        and config.get("fixed_cell") is True
        and _numbers_close(config.get("dt_ps"), 0.004)
        and _numbers_close(config.get("temperature_K"), 310.0)
        and _numbers_close(config.get("friction_ps^-1"), 0.1)
        and source_settings.get("ensemble") == "NVT"
        and source_settings.get("fixed_cell") is True
        and _numbers_close(source_settings.get("dt_ps"), config.get("dt_ps"))
        and _numbers_close(
            source_settings.get("temperature_K"),
            config.get("temperature_K"),
        )
        and _numbers_close(
            source_settings.get("friction_ps^-1"),
            config.get("friction_ps^-1"),
        )
    )
    _set_closure_entry(
        entries,
        "integrator_protocol",
        passed=protocol_ok,
        command=runtime_command,
        observed=(
            f"source-derived {_format_number(dt_fs)} "
            f"fs, {_format_number(config.get('temperature_K'))} K, "
            f"{_format_number(config.get('friction_ps^-1'))} ps^-1 fixed-cell "
            "Langevin NVT ran"
        ),
        context="runtime/gpcrmd_performance.json:config",
        acceptance=["AC2", "AC4"],
        next_decision="fix the first source-protocol divergence",
    )

    finite_check_names = {
        "positions_finite",
        "velocities_finite",
        "potential_energy_finite",
        "kinetic_energy_finite",
        "total_energy_finite",
        "forces_finite",
        "temperature_finite",
        "constraint_error_finite",
    }
    finite_ok = all(case is not None for case in required_cases) and all(
        case.get("status") == "ran"
        and all(case.get("checks", {}).get(name) is True for name in finite_check_names)
        for case in required_cases
        if case is not None
    )
    warmup = cases_by_phase.get("warmup", {})
    measured = cases_by_phase.get("measured", {})
    restart = cases_by_phase.get("restart", {})
    expected_step_counts_ok = (
        _number_at_least(warmup.get("steps"), 1)
        and _number_at_least(measured.get("steps"), 2)
        and _number_at_least(restart.get("steps"), 1)
    )
    finite_ok = finite_ok and expected_step_counts_ok
    _set_closure_entry(
        entries,
        "stability_finiteness",
        passed=finite_ok,
        command=runtime_command,
        observed=(
            f"{_format_step_phrase(warmup.get('steps'), 'warmup')}, "
            f"{_format_step_phrase(measured.get('steps'), 'measured')}, and "
            f"{_format_step_phrase(restart.get('steps'), 'restart')} retained "
            "finite state; this is a bounded run, not production-length stability"
        ),
        context="runtime/gpcrmd_performance.json:cases",
        acceptance=["AC4", "AC5"],
        next_decision="reproduce and fix the first non-finite phase",
    )

    force_metrics = parity.get("force_metrics", {})
    force_arrays = parity.get("force_arrays", {})
    parity_metrics_ok = (
        _number_at_most(
            force_metrics.get("energy_error_per_atom_kj_mol"),
            tolerances.get("energy_per_atom_kj_mol"),
        )
        and _number_at_most(
            force_metrics.get("relative_energy_error"),
            tolerances.get("relative_energy_error"),
        )
        and _number_at_most(
            force_metrics.get("rms_absolute_kj_mol_nm"),
            tolerances.get("force_rms_kj_mol_nm"),
        )
        and _number_at_most(
            force_metrics.get("maximum_absolute_kj_mol_nm"),
            tolerances.get("force_maximum_kj_mol_nm"),
        )
        and force_arrays.get("shape") == [GPCRMD_ATOM_COUNT, 3]
        and _is_sha256(force_arrays.get("mlx_hash"))
        and _is_sha256(force_arrays.get("openmm_hash"))
        and _is_sha256(force_arrays.get("delta_hash"))
    )
    parity_ok = (
        parity.get("status") == "passed"
        and parity.get("passed") is True
        and parity.get("atom_count") == GPCRMD_ATOM_COUNT
        and not parity.get("blockers")
        and parity.get("manifest_comparison", {}).get("matched") is True
        and parity_metrics_ok
        and set(parity_checks) >= GPCRMD_REQUIRED_PARITY_CHECKS
        and all(
            parity_checks.get(name) is True for name in GPCRMD_REQUIRED_PARITY_CHECKS
        )
        and all(value is True for value in parity_checks.values())
    )
    _set_closure_entry(
        entries,
        "parity_tolerance",
        passed=parity_ok,
        command=parity_command,
        observed=(
            "matched manifests passed total/component energy and complete-force bounds"
        ),
        context="parity/gpcrmd_pme_parity_report.json",
        acceptance=["AC3"],
        next_decision="fix the first manifest, component-energy, or complete-force bound",
    )

    performance_ok = (
        measured.get("status") == "ran"
        and _number_at_least(measured.get("steps"), 2)
        and _number_above(measured.get("run_wall_s"), 0.0)
        and _number_above(measured.get("integration_steps_per_s"), 0.0)
        and _number_above(measured.get("max_rss_mb"), 0.0)
    )
    _set_closure_entry(
        entries,
        "performance_runtime",
        passed=performance_ok,
        command=runtime_command,
        observed=(
            f"{_format_integer(measured.get('steps'))} measured steps completed in "
            f"{_format_number(measured.get('run_wall_s'), digits=6)} s at "
            f"{_format_number(measured.get('integration_steps_per_s'), digits=6)} "
            "steps/s; no OpenMM throughput ratio is claimed"
        ),
        context="runtime/gpcrmd_performance.json:cases[phase=measured]",
        acceptance=["AC4", "AC6"],
        next_decision="fix the first resource ceiling or incomplete measured phase",
    )

    continuation = runtime.get("continuation", {})
    warmup_start_step = _integer_value(warmup.get("start_step"))
    warmup_final_step = _integer_value(warmup.get("final_step"))
    warmup_steps = _integer_value(warmup.get("steps"))
    measured_start_step = _integer_value(measured.get("start_step"))
    measured_final_step = _integer_value(measured.get("final_step"))
    measured_steps = _integer_value(measured.get("steps"))
    restart_start_step = _integer_value(restart.get("start_step"))
    restart_final_step = _integer_value(restart.get("final_step"))
    restart_steps = _integer_value(restart.get("steps"))
    phase_steps_present = all(
        value is not None
        for value in (
            warmup_start_step,
            warmup_final_step,
            warmup_steps,
            measured_start_step,
            measured_final_step,
            measured_steps,
            restart_start_step,
            restart_final_step,
            restart_steps,
        )
    )
    phase_chain_ok = (
        phase_steps_present
        and warmup_final_step == measured_start_step
        and measured_final_step == restart_start_step
        and _numbers_close(warmup.get("final_time_ps"), measured.get("start_time_ps"))
        and _numbers_close(measured.get("final_time_ps"), restart.get("start_time_ps"))
        and warmup_final_step == warmup_start_step + warmup_steps
        and measured_final_step == measured_start_step + measured_steps
        and restart_final_step == restart_start_step + restart_steps
    )
    output_ok = (
        continuation.get("status") == "passed"
        and continuation.get("warmup_to_measured") is True
        and continuation.get("measured_to_restart") is True
        and continuation.get("monotonic_step_time") is True
        and continuation.get("fixed_cell_preserved") is True
        and phase_chain_ok
        and all(
            case.get("trajectory_loaded") is True
            and case.get("checkpoint_loaded") is True
            for case in required_cases
            if case is not None
        )
    )
    _set_closure_entry(
        entries,
        "output_restart",
        passed=output_ok,
        command=runtime_command,
        observed=(
            "trajectory and checkpoint artifacts reloaded, then restart advanced "
            f"step/time from {measured.get('final_step')}/"
            f"{_format_number(measured.get('final_time_ps'), digits=6)} ps to "
            f"{restart.get('final_step')}/"
            f"{_format_number(restart.get('final_time_ps'), digits=6)} ps"
        ),
        context="runtime/gpcrmd_performance.json:continuation",
        acceptance=["AC5"],
        next_decision="fix the first trajectory, checkpoint, or continuation mismatch",
    )

    boundary_ok = (
        parity.get("reference_engine") == "openmm"
        and str(parity.get("reference_engine_role", "")).startswith("reference-only")
        and runtime.get("kind") == "gpcrmd_source_protocol_benchmark"
    )
    _set_closure_entry(
        entries,
        "dependency_boundary",
        passed=boundary_ok,
        command="runtime-boundary tests and GPCRmd parity report",
        observed=(
            "MLX/Metal generated the runtime trajectory; OpenMM remained reference-only"
        ),
        context="parity/gpcrmd_pme_parity_report.json:reference_engine_role",
        acceptance=["AC3", "AC6"],
        next_decision="remove any reference-engine or vendor import from the product path",
    )

    ordered = [entries[category] for category in TAXONOMY_CATEGORIES]
    status = "blocked" if any(item["prevents_bounded_pass"] for item in ordered) else "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "change": "2026-07-15-gpcrmd-pme-runtime-closure",
        "fixture_id": fixture_id,
        "status": status,
        "bounded_pass": status == "passed",
        "summary": {
            "candidate_status": candidate.get("status"),
            "openmm_status": parity.get("status"),
            "mlx_status": runtime.get("status"),
            "blocking_categories": [
                item["category"] for item in ordered if item["prevents_bounded_pass"]
            ],
        },
        "limitations": [
            "fixed-cell orthorhombic NVT only",
            "no production NPT or cell-changing execution claim",
            "no analytic PME virial claim",
            "no triclinic PME claim",
            "no production-length stability claim",
            "no broad membrane-readiness claim",
            "no OpenMM/MLX throughput ratio without a matching runtime manifest",
        ],
        "entries": ordered,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _number_at_most(value: Any, limit: Any) -> bool:
    number = _finite_number(value)
    maximum = _finite_number(limit)
    return number is not None and maximum is not None and number <= maximum


def _number_above(value: Any, limit: Any) -> bool:
    number = _finite_number(value)
    minimum = _finite_number(limit)
    return number is not None and minimum is not None and number > minimum


def _number_at_least(value: Any, limit: Any) -> bool:
    number = _finite_number(value)
    minimum = _finite_number(limit)
    return number is not None and minimum is not None and number >= minimum


def _numbers_close(left: Any, right: Any) -> bool:
    left_number = _finite_number(left)
    right_number = _finite_number(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-9)
    )


def _integer_value(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _format_integer(value: Any) -> str:
    number = _integer_value(value)
    if number is None:
        return "unknown"
    return f"{number:,}"


def _format_number(value: Any, *, digits: int = 8) -> str:
    number = _finite_number(value)
    return "unknown" if number is None else f"{number:.{digits}g}"


def _format_step_phrase(value: Any, phase: str) -> str:
    count = _integer_value(value)
    if count is None:
        return f"unknown {phase} steps"
    suffix = "" if count == 1 else "s"
    return f"{count:,} {phase} step{suffix}"


def _format_mesh(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "unknown"
    dimensions = [_format_integer(dimension).replace(",", "") for dimension in value]
    return "x".join(dimensions)


def _set_closure_entry(
    entries: dict[str, dict[str, Any]],
    category: str,
    *,
    passed: bool,
    command: str,
    observed: str,
    context: str,
    acceptance: list[str],
    next_decision: str,
) -> None:
    entries[category].update(
        {
            "status": "passed" if passed else "blocked",
            "command": command,
            "observed_result": (
                observed if passed else f"required closure evidence failed: {observed}"
            ),
            "smallest_reproduction_context": context,
            "affected_acceptance_criteria": acceptance,
            "next_implementation_decision": "none" if passed else next_decision,
            "prevents_bounded_pass": not passed,
        }
    )


def write_blocker_matrix(matrix: dict[str, Any], out: Path) -> None:
    """Write blocker matrix as stable JSON."""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")


def build_readiness_report(matrix: dict[str, Any]) -> str:
    """Build a compact human-readable readiness report."""

    lines = [
        "# Production MD Readiness Fixture Probe",
        "",
        f"- fixture: `{matrix['fixture_id']}`",
        f"- status: `{matrix['status']}`",
        f"- bounded pass: `{str(matrix['bounded_pass']).lower()}`",
        "",
        "## Blocking Categories",
        "",
    ]
    blocking = [entry for entry in matrix["entries"] if entry["prevents_bounded_pass"]]
    if blocking:
        for entry in blocking:
            lines.extend(
                [
                    f"- `{entry['category']}`: {entry['observed_result']}",
                    f"  - command: `{entry['command']}`",
                    f"  - next: {entry['next_implementation_decision']}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Category Matrix", ""])
    lines.append("| Category | Status | Prevents Pass | Observed Result |")
    lines.append("| --- | --- | --- | --- |")
    for entry in matrix["entries"]:
        observed = str(entry["observed_result"]).replace("|", "\\|")
        lines.append(
            f"| `{entry['category']}` | `{entry['status']}` | "
            f"`{str(entry['prevents_bounded_pass']).lower()}` | {observed} |"
        )
    lines.extend(
        [
            "",
            "## Production Claim Boundary",
            "",
            "This report is one bounded fixture probe. It is not broad production MD "
            "certification.",
        ]
    )
    limitations = matrix.get("limitations", [])
    if limitations:
        lines.extend(["", "Retained limitations:", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    return "\n".join(lines)


def write_readiness_report(matrix: dict[str, Any], out: Path) -> None:
    """Write the Markdown readiness report."""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_readiness_report(matrix))


def _base_entry(category: str, fixture_id: str | None) -> dict[str, Any]:
    return {
        "category": category,
        "status": "deferred",
        "fixture": fixture_id,
        "command": "not evaluated in this bounded probe",
        "observed_result": "not evaluated in this bounded probe",
        "smallest_reproduction_context": "not applicable",
        "affected_acceptance_criteria": [],
        "next_implementation_decision": "none",
        "prevents_bounded_pass": False,
    }


def _apply_candidate(entries: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> None:
    if candidate.get("selected"):
        entries["artifact_source"].update(
            {
                "status": "passed",
                "command": candidate.get("fixture", {}).get(
                    "source_reproduction_command",
                    "select_production_md_fixture",
                ),
                "observed_result": (
                    "selected local GPCRmd cache fixture with "
                    f"{candidate.get('scale', {}).get('atom_count')} atoms"
                ),
                "smallest_reproduction_context": candidate.get("fixture", {}).get(
                    "source_path", "candidate evidence"
                ),
                "affected_acceptance_criteria": ["AC2", "AC7"],
                "next_implementation_decision": "use selected fixture evidence",
            }
        )
    for blocker in candidate.get("blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC2", "AC7"])
    for blocker in candidate.get("known_pre_execution_blockers", []):
        if blocker.get("status") == "blocked":
            continue
        _merge_blocker(entries, blocker, default_ac=["AC2"], prevents=False)


def _apply_openmm(entries: dict[str, dict[str, Any]], openmm: dict[str, Any]) -> None:
    if openmm.get("status") == "ran":
        entries["parity_tolerance"].update(
            {
                "status": "partial",
                "command": openmm.get("command", {}).get("command", "OpenMM reference probe"),
                "observed_result": (
                    "OpenMM reference ran with finite outputs; comparison is bounded "
                    "by documented protocol divergences"
                ),
                "smallest_reproduction_context": openmm.get("evidence_source", {}).get(
                    "run_report", "openmm-reference.json"
                ),
                "affected_acceptance_criteria": ["AC3", "AC6"],
                "next_implementation_decision": "compare against MLX probe once MLX run passes",
            }
        )
        entries["stability_finiteness"].update(
            {
                "status": "partial",
                "command": openmm.get("command", {}).get("command", "OpenMM reference probe"),
                "observed_result": openmm.get("finite_output_checks", {}).get(
                    "observed_result",
                    "OpenMM finite-output evidence recorded",
                ),
                "smallest_reproduction_context": openmm.get("evidence_source", {}).get(
                    "preview_summary", "openmm-reference.json"
                ),
                "affected_acceptance_criteria": ["AC3", "AC6"],
                "next_implementation_decision": "MLX run must produce energies for parity",
            }
        )
        entries["dependency_boundary"].update(
            {
                "status": "passed",
                "command": "OpenMM reference evidence writer",
                "observed_result": "OpenMM remains reference-only/dev evidence",
                "smallest_reproduction_context": "openmm-reference.json",
                "affected_acceptance_criteria": ["AC3", "AC8"],
                "next_implementation_decision": "preserve reference-only boundary",
            }
        )
    for blocker in openmm.get("blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC3", "AC7"])


def _apply_mlx(entries: dict[str, dict[str, Any]], mlx: dict[str, Any]) -> None:
    stages = mlx.get("stages", {})
    if stages.get("prep", {}).get("status") == "passed":
        entries["preparation"].update(
            {
                "status": "passed",
                "command": stages["prep"].get("command", "MLX prep probe"),
                "observed_result": "prepared artifact exported for selected fixture",
                "smallest_reproduction_context": mlx.get("fixture", {}).get(
                    "prep_reproduction_command", "mlx-probe.json"
                ),
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "use prepared artifact for runtime blocker work",
            }
        )
    compatibility = stages.get("prep", {}).get("compatibility_report", {})
    if compatibility:
        required_terms = compatibility.get("required_terms", [])
        entries["forcefield_terms"].update(
            {
                "status": "passed" if required_terms else "partial",
                "command": stages.get("prep", {}).get("command", "MLX prep probe"),
                "observed_result": (
                    f"prepared terms represented: {len(required_terms)} required term families"
                ),
                "smallest_reproduction_context": "mlx-probe.json:stages.prep",
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "keep term coverage tied to runtime support",
            }
        )
        unsupported = compatibility.get("unsupported_physics", [])
        if unsupported:
            entries["constraints_hmr_virtual_sites"].update(
                {
                    "status": "partial",
                    "command": stages.get("prep", {}).get("command", "MLX prep probe"),
                    "observed_result": ", ".join(str(item) for item in unsupported),
                    "smallest_reproduction_context": "mlx-probe.json:stages.prep",
                    "affected_acceptance_criteria": ["AC4", "AC5", "AC7"],
                    "next_implementation_decision": (
                        "verify HMR/virtual-site policy before production-length runs"
                    ),
                }
            )
    readiness = stages.get("readiness", {}).get("reports", {})
    if readiness:
        entries["integrator_protocol"].update(
            {
                "status": "partial",
                "command": "protocol_readiness_report",
                "observed_result": "NVT short proof protocol is accepted; NPT is not required",
                "smallest_reproduction_context": "mlx-probe.json:stages.readiness",
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "keep NPT/barostat out of this NVT fixture claim",
            }
        )
        entries["npt_barostat"].update(
            {
                "status": "passed",
                "command": "protocol_readiness_report",
                "observed_result": "selected fixture protocol is NVT; no barostat required",
                "smallest_reproduction_context": "mlx-probe.json:stages.readiness",
                "affected_acceptance_criteria": ["AC5"],
                "next_implementation_decision": "do not claim NPT coverage from this fixture",
            }
        )
    run_stage = stages.get("run", {})
    nonbonded_runtime = run_stage.get("nonbonded_runtime", {})
    if run_stage.get("status") == "passed" and nonbonded_runtime.get("backend") in {
        "mlx_dense_pairs",
        "mlx_cell_pairs",
        "mlx_cell_blocks",
    }:
        entries["topology_terms"].update(
            {
                "status": "passed",
                "command": "run_minimize_then_nvt bounded production probe",
                "observed_result": (
                    "lazy topology used runtime compact neighbor pairs via "
                    f"{nonbonded_runtime.get('backend')}"
                ),
                "smallest_reproduction_context": "mlx-probe.json:stages.run.nonbonded_runtime",
                "affected_acceptance_criteria": ["AC4", "AC6", "AC7"],
                "next_implementation_decision": (
                    "keep compact neighbor provisioning while addressing downstream physics"
                ),
                "prevents_bounded_pass": False,
            }
        )
    finite = mlx.get("finite_checks", {})
    if finite.get("positions") and finite.get("velocities"):
        energies_finite = finite.get("energies") is True
        entries["stability_finiteness"].update(
            {
                "status": "passed" if energies_finite else "partial",
                "command": mlx.get("command", "MLX probe"),
                "observed_result": (
                    "MLX bounded run produced finite positions, velocities, and energies"
                    if energies_finite
                    else "MLX prep produced finite positions and velocities; energies "
                    "are unavailable because bounded run blocked"
                ),
                "smallest_reproduction_context": "mlx-probe.json:finite_checks",
                "affected_acceptance_criteria": ["AC6", "AC7"],
                "next_implementation_decision": (
                    "retain finite-state checks in at-scale runs"
                    if energies_finite
                    else "rerun finite energy checks after runtime blocker"
                ),
            }
        )
    runtime = mlx.get("runtime_performance", {})
    if runtime:
        entries["performance_runtime"].update(
            {
                "status": "blocked"
                if not runtime.get("bounded_run_completed")
                else "passed",
                "command": mlx.get("command", "MLX probe"),
                "observed_result": (
                    "bounded run attempted but did not complete"
                    if not runtime.get("bounded_run_completed")
                    else "bounded run completed"
                ),
                "smallest_reproduction_context": "mlx-probe.json:runtime_performance",
                "affected_acceptance_criteria": ["AC4", "AC6", "AC7"],
                "next_implementation_decision": "fix runtime blocker before timing claims",
                "prevents_bounded_pass": not runtime.get("bounded_run_completed"),
            }
        )
    if mlx.get("dependency_boundary", {}).get("status") == "passed":
        entries["dependency_boundary"].update(
            {
                "status": "passed",
                "command": "MLX probe dependency boundary",
                "observed_result": "MLX probe imported no reference engines or vendors",
                "smallest_reproduction_context": "mlx-probe.json:dependency_boundary",
                "affected_acceptance_criteria": ["AC8"],
                "next_implementation_decision": "preserve product-runtime boundary",
            }
        )
    for blocker in mlx.get("taxonomy_blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC4", "AC6", "AC7", "AC8"])


def _finalize_defaults(
    entries: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    openmm: dict[str, Any],
    mlx: dict[str, Any],
) -> None:
    del openmm
    pme_relevance = str(
        candidate.get("protocol_relevance", {}).get("pme_electrostatics_relevance", "")
    ).lower()
    neighbor_axis_passed = entries["topology_terms"]["status"] == "passed"
    if neighbor_axis_passed and "required" in pme_relevance:
        entries["electrostatics_pme"].update(
            {
                "status": "blocked",
                "command": "candidate fixture and successful MLX short-range probe",
                "observed_result": (
                    "neighbor-listed cutoff execution passed, but the selected fixture "
                    "requires PME-scale electrostatics"
                ),
                "smallest_reproduction_context": "candidate-fixture.json and mlx-probe.json",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": (
                    "implement and validate PME beyond the current runtime envelope"
                ),
                "prevents_bounded_pass": True,
            }
        )
    if entries["electrostatics_pme"]["status"] == "deferred":
        entries["electrostatics_pme"].update(
            {
                "status": "partial",
                "command": "candidate fixture and readiness evidence",
                "observed_result": (
                    "selected fixture requires periodic PME-scale electrostatics; "
                    "OpenMM reference used PME while MLX readiness reports cutoff"
                ),
                "smallest_reproduction_context": "candidate-fixture.json and mlx-probe.json",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": (
                    "decide whether next wave implements PME runtime path for this fixture"
                ),
            }
        )
    if entries["output_restart"]["status"] == "deferred":
        entries["output_restart"].update(
            {
                "status": "blocked" if mlx.get("status") == "blocked" else "partial",
                "command": "MLX probe output check",
                "observed_result": (
                    "no trajectory, checkpoint, or restart output because bounded MLX run blocked"
                ),
                "smallest_reproduction_context": "mlx-probe.json:stages.run",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": (
                    "record output/restart behavior after runtime blocker is fixed"
                ),
                "prevents_bounded_pass": mlx.get("status") == "blocked",
            }
        )
    if entries["topology_terms"]["status"] == "deferred" and candidate.get("selected"):
        entries["topology_terms"].update(
            {
                "status": "partial",
                "command": "selected fixture topology inspection",
                "observed_result": (
                    "large CHARMM topology selected; runtime topology path not proven"
                ),
                "smallest_reproduction_context": "candidate-fixture.json",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": "use MLX runtime blocker evidence",
            }
        )


def _merge_blocker(
    entries: dict[str, dict[str, Any]],
    blocker: dict[str, Any],
    *,
    default_ac: list[str],
    prevents: bool | None = None,
) -> None:
    category = str(blocker.get("category", "preparation"))
    if category not in entries:
        category = "preparation"
    entry = entries[category]
    status = str(blocker.get("status", "blocked"))
    prevents_bounded_pass = (
        bool(blocker.get("prevents_bounded_pass", status == "blocked"))
        if prevents is None
        else prevents
    )
    if entry["status"] == "blocked" and status != "blocked":
        return
    entry.update(
        {
            "status": status,
            "command": str(blocker.get("command", entry["command"])),
            "observed_result": str(
                blocker.get(
                    "observed_result",
                    blocker.get("observed", entry["observed_result"]),
                )
            ),
            "smallest_reproduction_context": str(
                blocker.get(
                    "smallest_reproduction_context",
                    blocker.get("context", entry["smallest_reproduction_context"]),
                )
            ),
            "affected_acceptance_criteria": list(
                blocker.get("affected_acceptance_criteria", default_ac)
            ),
            "next_implementation_decision": str(
                blocker.get(
                    "next_implementation_decision",
                    blocker.get("next_decision", entry["next_implementation_decision"]),
                )
            ),
            "prevents_bounded_pass": prevents_bounded_pass,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--openmm", required=True, type=Path)
    parser.add_argument("--mlx", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matrix = build_blocker_matrix(
        candidate=_read_json(args.candidate),
        openmm=_read_json(args.openmm),
        mlx=_read_json(args.mlx),
    )
    write_blocker_matrix(matrix, args.out)
    write_readiness_report(matrix, args.report)


if __name__ == "__main__":
    main()
