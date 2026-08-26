from __future__ import annotations

import copy

import pytest

from mlx_atomistic.benchmarks.dft_gth_transferability import (
    build_gth_transferability_report,
    load_gth_transferability_contract,
)


def test_gth_transferability_matrix_separates_coverage_from_strict_science():
    report = build_gth_transferability_report()

    assert report["coverage_complete"] is True
    assert report["strict_science_passed"] is False
    assert report["identity_complete"] is False
    assert report["production_envelope_verified"] is False
    assert "case:rocksalt-mgo-q2-q6:metric:bulk_derivative_relative" in report[
        "blockers"
    ]
    assert "case:rocksalt-mgo-q2-q6:metric:force_max_abs_hartree_per_bohr" in report[
        "blockers"
    ]
    assert "case:bcc-iron-q16:method_validation" in report["blockers"]
    iron = next(case for case in report["cases"] if case["case_id"] == "bcc-iron-q16")
    assert iron["method_validation"]["metrics"][0]["passed"] is False


def test_gth_transferability_matrix_retains_failed_q10_candidate():
    report = build_gth_transferability_report()
    candidate = report["candidates"][0]

    assert candidate["candidate_id"] == "rocksalt-mgo-primitive-q10-q6-c40-k4"
    assert candidate["passed"] is False
    assert candidate["method_validation"]["passed"] is False
    assert candidate["elapsed_wall_seconds"] == pytest.approx(40.98934508301318)
    assert candidate["peak_temporary_bytes"] == 57_201_084


def test_gth_transferability_matrix_fails_closed_on_missing_coverage_or_identity():
    contract = load_gth_transferability_contract()
    missing_d = copy.deepcopy(contract)
    missing_d["cases"] = [
        case for case in missing_d["cases"] if case["case_id"] != "bcc-iron-q16"
    ]
    report = build_gth_transferability_report(missing_d)
    assert report["coverage_complete"] is False
    assert report["coverage"]["periodic_blocks"]["missing"] == ["d"]

    invalid = copy.deepcopy(contract)
    invalid["cases"][0]["identity"]["reference_protocol_sha256"] = "invalid"
    with pytest.raises(ValueError, match="reference protocol identity"):
        build_gth_transferability_report(invalid)

    missing_method = copy.deepcopy(contract)
    del missing_method["cases"][0]["method_validation"]
    with pytest.raises(ValueError, match="method validation scope"):
        build_gth_transferability_report(missing_method)
