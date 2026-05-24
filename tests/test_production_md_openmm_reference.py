from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_openmm_production_md_reference.py"
SPEC = importlib.util.spec_from_file_location("run_openmm_production_md_reference", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


def test_existing_gpcrmd_reference_evidence_is_recorded(tmp_path: Path):
    candidate_path = (
        ROOT / ".agent/work/production-md-readiness-fixture-probe/evidence/candidate-fixture.json"
    )
    candidate = json.loads(candidate_path.read_text())

    report = HELPER.build_openmm_reference_evidence(
        candidate=candidate,
        candidate_path=candidate_path,
        argv=[
            "scripts/run_openmm_production_md_reference.py",
            "--candidate",
            str(candidate_path),
            "--out",
            str(tmp_path / "openmm-reference.json"),
        ],
    )

    assert report["status"] == "ran"
    assert report["fixture_id"] == "gpcrmd-729-beta1-5f8u-cyanopindolol"
    assert report["reference_engine"] == "openmm"
    assert report["platform"]["reference_only"] is True
    assert report["platform"]["product_runtime"] == "mlx_atomistic"
    assert report["protocol_settings"]["openmm_reference_protocol"]["nonbonded_method"] == "PME"
    assert report["finite_output_checks"]["finite_outputs"] is True
    assert report["finite_output_checks"]["energy_fields"] == {
        "potential_energy_kj_mol": True,
        "kinetic_energy_kj_mol": True,
        "temperature_K": True,
    }
    assert report["reference_only_dependency_status"]["runtime_role"] == "reference-only"
    assert report["reference_only_dependency_status"]["product_runtime_dependency"] is False
    assert report["product_runtime_boundary"]["status"] == "preserved"
    assert report["product_runtime_boundary"]["src_imports_openmm"] is False
    assert report["evidence_source"]["kind"] == "existing_reference_data"
    assert report["evidence_source"]["new_trajectory_written"] is False
    assert report["evidence_source"]["run_report"] == (
        "notebooks/ligand-receptor-motion/data/openmm-md/"
        "729-50000-opencl-charmm-pme-sample11/openmm_charmm_md_run_report.json"
    )
    assert report["evidence_source"]["preview_summary"] == (
        "notebooks/ligand-receptor-motion/data/openmm-md/"
        "729-50000-opencl-charmm-pme-sample11/preview_summary.json"
    )
    assert not Path(report["evidence_source"]["run_report"]).is_absolute()
    assert not Path(report["evidence_source"]["preview_summary"]).is_absolute()
    assert {item["category"] for item in report["supported_divergences_from_mlx_protocol"]} == {
        "reference_engine",
        "time_step",
        "trajectory_length",
        "platform",
    }


def test_missing_reference_data_records_reference_blocker(tmp_path: Path):
    candidate = {
        "change": "production-md-readiness-fixture-probe",
        "status": "selected",
        "fixture": {"id": "missing", "dynamics_id": 999999},
        "protocol_relevance": {"ensemble": "NVT", "time_step_fs": 4.0},
    }

    report = HELPER.build_openmm_reference_evidence(
        candidate=candidate,
        candidate_path=tmp_path / "candidate-fixture.json",
        reference_root=tmp_path / "missing-openmm-md",
        argv=["script"],
    )

    assert report["status"] == "blocked"
    assert report["finite_output_checks"]["finite_outputs"] is False
    assert report["blockers"] == [
        {
            "category": "reference_evidence",
            "status": "blocked",
            "observed_result": (
                "no existing OpenMM reference report found under "
                f"{tmp_path / 'missing-openmm-md'} for dynamics_id=999999"
            ),
            "product_runtime_boundary_changed": False,
        }
    ]
    assert report["product_runtime_boundary"]["status"] == "preserved"


def test_cli_writes_openmm_reference_json(tmp_path: Path):
    candidate_path = (
        ROOT / ".agent/work/production-md-readiness-fixture-probe/evidence/candidate-fixture.json"
    )
    out_path = tmp_path / "openmm-reference.json"

    old_argv = sys.argv
    try:
        sys.argv = [
            str(SCRIPT_PATH),
            "--candidate",
            str(candidate_path),
            "--out",
            str(out_path),
        ]
        HELPER.main()
    finally:
        sys.argv = old_argv

    report = json.loads(out_path.read_text())
    assert report["status"] == "ran"
    assert report["command"]["candidate"] == str(candidate_path)
    assert report["finite_output_checks"]["finite_outputs"] is True
