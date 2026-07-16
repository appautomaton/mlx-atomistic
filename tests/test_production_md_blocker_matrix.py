from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_production_md_blocker_matrix.py"
SPEC = importlib.util.spec_from_file_location("build_production_md_blocker_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)

build_blocker_matrix = HELPER.build_blocker_matrix
build_readiness_report = HELPER.build_readiness_report
TAXONOMY_CATEGORIES = HELPER.TAXONOMY_CATEGORIES

EVIDENCE_DIR = Path(__file__).resolve().parent / "fixtures" / "production-md-readiness"


def _evidence(name: str) -> dict:
    return json.loads((EVIDENCE_DIR / name).read_text())


def test_blocker_matrix_covers_every_taxonomy_category():
    matrix = build_blocker_matrix(
        candidate=_evidence("candidate-fixture.json"),
        openmm=_evidence("openmm-reference.json"),
        mlx=_evidence("mlx-probe.json"),
    )

    assert [entry["category"] for entry in matrix["entries"]] == list(TAXONOMY_CATEGORIES)
    assert {
        entry["status"] for entry in matrix["entries"]
    } <= {"passed", "partial", "blocked", "deferred", "anti_goal"}
    assert matrix["status"] == "blocked"
    assert matrix["bounded_pass"] is False
    categories = {entry["category"]: entry for entry in matrix["entries"]}
    assert categories["artifact_source"]["status"] == "passed"
    assert categories["preparation"]["status"] == "passed"
    assert categories["topology_terms"]["status"] == "blocked"
    assert categories["topology_terms"]["prevents_bounded_pass"] is True
    assert categories["topology_terms"]["affected_acceptance_criteria"] == [
        "AC4",
        "AC6",
        "AC7",
        "AC8",
    ]
    assert categories["dependency_boundary"]["status"] == "passed"
    assert categories["parity_tolerance"]["status"] == "partial"
    assert categories["output_restart"]["status"] == "blocked"


def test_blocking_entries_have_reproduction_and_next_decision():
    matrix = build_blocker_matrix(
        candidate=_evidence("candidate-fixture.json"),
        openmm=_evidence("openmm-reference.json"),
        mlx=_evidence("mlx-probe.json"),
    )

    for entry in matrix["entries"]:
        if not entry["prevents_bounded_pass"]:
            continue
        assert entry["command"]
        assert entry["observed_result"]
        assert entry["smallest_reproduction_context"]
        assert entry["affected_acceptance_criteria"]
        assert entry["next_implementation_decision"] != "none"


def test_readiness_report_states_bounded_claim_boundary():
    matrix = build_blocker_matrix(
        candidate=_evidence("candidate-fixture.json"),
        openmm=_evidence("openmm-reference.json"),
        mlx=_evidence("mlx-probe.json"),
    )

    report = build_readiness_report(matrix)

    assert "not broad production MD certification" in report
    assert "`topology_terms`" in report
    assert "lazy topology requires a runtime nonbonded pair provider" in report


def test_successful_neighbor_probe_advances_to_pme_blocker():
    candidate = _evidence("candidate-fixture.json")
    mlx = _evidence("mlx-probe.json")
    mlx["status"] = "passed"
    mlx["earliest_blocker"] = None
    mlx["taxonomy_blockers"] = []
    mlx["stages"]["run"] = {
        "status": "passed",
        "duration_seconds": 1.0,
        "production_steps": 2,
        "sample_interval": 1,
        "nonbonded_runtime": {
            "backend": "mlx_cell_pairs",
            "fallback_reason": None,
            "pair_count": 100,
            "candidate_count": 120,
            "candidate_waste_fraction": 1.0 / 6.0,
        },
    }
    mlx["finite_checks"]["energies"] = True
    mlx["finite_checks"]["reason"] = None
    mlx["runtime_performance"].update(
        {
            "bounded_run_attempted": True,
            "bounded_run_completed": True,
            "backend": "mlx_cell_pairs",
        }
    )

    matrix = build_blocker_matrix(
        candidate=candidate,
        openmm=_evidence("openmm-reference.json"),
        mlx=mlx,
    )

    categories = {entry["category"]: entry for entry in matrix["entries"]}
    assert categories["topology_terms"]["status"] == "passed"
    assert categories["topology_terms"]["prevents_bounded_pass"] is False
    assert categories["stability_finiteness"]["status"] == "passed"
    assert categories["performance_runtime"]["status"] == "passed"
    assert categories["electrostatics_pme"]["status"] == "blocked"
    assert categories["electrostatics_pme"]["prevents_bounded_pass"] is True
    assert matrix["status"] == "blocked"


def test_gpcrmd_closure_evidence_replaces_stale_pme_blocker():
    finite_checks = {
        "positions_finite": True,
        "velocities_finite": True,
        "potential_energy_finite": True,
        "kinetic_energy_finite": True,
        "total_energy_finite": True,
        "forces_finite": True,
        "temperature_finite": True,
        "constraint_error_finite": True,
        "runtime_contract_matches": True,
    }

    def runtime_case(phase, *, start_step, start_time_ps, steps=1):
        return {
            "phase": phase,
            "status": "ran",
            "start_step": start_step,
            "final_step": start_step + steps,
            "start_time_ps": start_time_ps,
            "final_time_ps": start_time_ps + (steps * 0.004),
            "steps": steps,
            "checks": finite_checks,
            "topology_pair_policy": "lazy",
            "neighbor_backend": "mlx_cell_blocks",
            "neighbor_representation": "NeighborBlocks",
            "dense_or_tiled_fallback_used": False,
            "pme_execution_plan_build_count": 1,
            "pme_execution_plan_reuse_count": 2,
            "hmr_preserved": True,
            "constraint_error_finite": True,
            "trajectory_loaded": True,
            "checkpoint_loaded": True,
            "run_wall_s": 1.0,
            "integration_steps_per_s": 1.0,
            "max_rss_mb": 1024.0,
        }

    digest = "a" * 64
    source_files = [
        {
            "file_id": 15286,
            "role": "topology",
            "size_bytes": 100,
            "sha256": digest,
            "source_url": "https://www.gpcrmd.org/topology.psf",
        },
        {
            "file_id": 17686,
            "role": "model",
            "size_bytes": 100,
            "sha256": digest,
            "source_url": "https://www.gpcrmd.org/model.pdb",
        },
        {
            "file_id": 15290,
            "role": "parameters",
            "size_bytes": 100,
            "sha256": digest,
            "source_url": "https://www.gpcrmd.org/parameters.prm",
        },
        {
            "file_id": 17687,
            "role": "protocol",
            "size_bytes": 100,
            "sha256": digest,
            "source_url": "https://www.gpcrmd.org/protocol.tar.gz",
            "archive_extraction": "reused",
            "archive_members": [
                {
                    "kind": "file",
                    "normalized_name": name,
                    "size_bytes": 10,
                    "sha256": digest,
                }
                for name in (
                    "rep_1/input",
                    "rep_1/input.coor",
                    "rep_1/input.vel",
                    "rep_1/input.xsc",
                    "rep_1/log.txt",
                )
            ],
        },
    ]
    candidate = {
        "kind": "gpcrmd_fixture_acquisition",
        "status": "complete",
        "target_id": "gpcrmd-729-beta1-5f8u-cyanopindolol",
        "blockers": [],
        "files": source_files,
        "command": {"reproduction_command": "acquire gpcrmd 729"},
    }
    parity = {
        "kind": "mlx_atomistic.gpcrmd_pme_parity",
        "status": "passed",
        "passed": True,
        "atom_count": 92001,
        "fixture": "gpcrmd-729-beta1-5f8u-cyanopindolol",
        "blockers": [],
        "manifest_comparison": {"matched": True},
        "checks": {
            "component_energy_per_atom": True,
            "component_relative_energy": True,
            "manifest_match": True,
            "mlx_lazy_topology": True,
            "mlx_neighbor_blocks": True,
            "mlx_no_neighbor_fallback": True,
            "mlx_pair_cache_unmaterialized": True,
            "openmm_resolved_pme": True,
            "total_energy_per_atom": True,
            "total_relative_energy": True,
            "force_maximum": True,
            "force_rms": True,
        },
        "tolerances": {
            "energy_per_atom_kj_mol": 5e-3,
            "relative_energy_error": 5e-5,
            "force_rms_kj_mol_nm": 3.0,
            "force_maximum_kj_mol_nm": 12.0,
        },
        "force_metrics": {
            "energy_error_per_atom_kj_mol": 1e-7,
            "relative_energy_error": 1e-8,
            "rms_absolute_kj_mol_nm": 0.1,
            "maximum_absolute_kj_mol_nm": 11.0,
        },
        "force_arrays": {
            "shape": [92001, 3],
            "mlx_hash": digest,
            "openmm_hash": digest,
            "delta_hash": digest,
        },
        "energies": {
            "component_metrics": {
                "bond": {
                    "energy_error_per_atom_kj_mol": 1e-7,
                    "relative_error": 1e-8,
                }
            }
        },
        "artifact_readiness": {
            "status": "ready",
            "blockers": [],
            "metadata": {
                "required_terms": [
                    "charmm_cmap",
                    "charmm_harmonic_improper",
                    "harmonic_angle",
                    "harmonic_bond",
                    "nbfix_pair_overrides",
                    "nonbonded_exception",
                    "nonbonded_lj_coulomb",
                    "periodic_dihedral",
                    "urey_bradley",
                ]
            },
        },
        "mlx": {
            "pme_readiness": {
                "status": "ready",
                "production_executable": True,
                "blockers": [],
                "atom_count": 92001,
                "background_policy": "reject_non_neutral",
                "mesh_shape": [78, 78, 108],
                "assignment_order": 5,
                "real_cutoff": 9.0,
                "checks": {"neutrality": True},
            }
        },
        "openmm": {
            "resolved_pme": {"mesh_shape": [78, 78, 108]},
            "resolved_pme_matches_manifest": True,
        },
        "reference_engine": "openmm",
        "reference_engine_role": "reference-only validation",
    }
    manifest_digest = "b" * 64
    runtime = {
        "kind": "gpcrmd_source_protocol_benchmark",
        "status": "passed",
        "config": {
            "target_id": candidate["target_id"],
            "ensemble": "NVT",
            "fixed_cell": True,
            "dt_ps": 0.004,
            "temperature_K": 310.0,
            "friction_ps^-1": 0.1,
            "protocol_manifest_sha256": manifest_digest,
        },
        "protocol_validation": {
            "status": "ready",
            "strict_production_load": True,
            "manifest_matches_prepared": True,
            "blockers": [],
            "declared_manifest_sha256": manifest_digest,
            "rebuilt_manifest_sha256": manifest_digest,
            "source_settings": {
                "atom_count": 92001,
                "constraints_count": 78896,
                "hmr_status": "represented_by_masses",
                "hmr_selected_hydrogen_count": 58952,
                "hmr_target_hydrogen_mass_da": 4.032,
                "ensemble": "NVT",
                "fixed_cell": True,
                "dt_ps": 0.004,
                "temperature_K": 310.0,
                "friction_ps^-1": 0.1,
            },
        },
        "cases": [
            runtime_case("warmup", start_step=0, start_time_ps=0.0),
            runtime_case("measured", start_step=1, start_time_ps=0.004, steps=2),
            runtime_case("restart", start_step=3, start_time_ps=0.012),
        ],
        "continuation": {
            "status": "passed",
            "warmup_to_measured": True,
            "measured_to_restart": True,
            "monotonic_step_time": True,
            "fixed_cell_preserved": True,
        },
    }

    matrix = build_blocker_matrix(candidate=candidate, openmm=parity, mlx=runtime)
    categories = {entry["category"]: entry for entry in matrix["entries"]}
    report = build_readiness_report(matrix)

    assert matrix["status"] == "passed"
    assert matrix["bounded_pass"] is True
    assert matrix["summary"]["blocking_categories"] == []
    assert categories["electrostatics_pme"]["status"] == "passed"
    assert categories["topology_terms"]["status"] == "passed"
    assert categories["output_restart"]["status"] == "passed"
    assert categories["npt_barostat"]["status"] == "anti_goal"
    assert "not broad production MD certification" in report

    parity["force_metrics"]["maximum_absolute_kj_mol_nm"] = 13.0
    failed = build_blocker_matrix(candidate=candidate, openmm=parity, mlx=runtime)
    failed_categories = {entry["category"]: entry for entry in failed["entries"]}
    assert failed["status"] == "blocked"
    assert failed_categories["parity_tolerance"]["status"] == "blocked"

    parity["force_metrics"]["maximum_absolute_kj_mol_nm"] = 11.0
    candidate["files"][0]["sha256"] = "not-a-digest"
    failed = build_blocker_matrix(candidate=candidate, openmm=parity, mlx=runtime)
    failed_categories = {entry["category"]: entry for entry in failed["entries"]}
    assert failed_categories["artifact_source"]["status"] == "blocked"

    candidate["files"][0]["sha256"] = digest
    runtime["protocol_validation"]["rebuilt_manifest_sha256"] = "c" * 64
    failed = build_blocker_matrix(candidate=candidate, openmm=parity, mlx=runtime)
    failed_categories = {entry["category"]: entry for entry in failed["entries"]}
    assert failed_categories["preparation"]["status"] == "blocked"

    runtime["protocol_validation"]["rebuilt_manifest_sha256"] = manifest_digest
    parity["mlx"]["pme_readiness"]["mesh_shape"] = [78, 78, 106]
    failed = build_blocker_matrix(candidate=candidate, openmm=parity, mlx=runtime)
    failed_categories = {entry["category"]: entry for entry in failed["entries"]}
    assert failed_categories["electrostatics_pme"]["status"] == "blocked"
