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


def _evidence(name: str) -> dict:
    return json.loads(
        (ROOT / ".agent/work/production-md-readiness-fixture-probe/evidence" / name).read_text()
    )


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
