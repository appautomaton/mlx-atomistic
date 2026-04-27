import json

import numpy as np
import pytest

from mlx_atomistic.forcefields import HarmonicBondPotential
from mlx_atomistic.validation import (
    ForceValidationCase,
    default_force_validation_cases,
    run_force_validation_suite,
    summarize_validation_results,
    validate_force_term,
)


def test_validate_force_term_returns_json_safe_result():
    term = HarmonicBondPotential([(0, 1)], k=5.0, length=1.0)
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)

    result = validate_force_term(
        term,
        positions,
        case_name="bond-check",
        seed=11,
        tolerance=3e-3,
    )

    assert result.passed
    assert result.case_name == "bond-check"
    assert result.term_name == "bond"
    assert result.atom_count == 2
    assert result.coordinate_count == 6
    json.dumps(result.to_dict())


def test_force_validation_suite_is_seed_reproducible():
    first = run_force_validation_suite(seed=13, cases_per_term=1)
    second = run_force_validation_suite(seed=13, cases_per_term=1)

    assert [result.to_dict() for result in first] == [result.to_dict() for result in second]
    assert all(result.passed for result in first)


def test_force_validation_summary_reports_failures():
    term = HarmonicBondPotential([(0, 1)], k=5.0, length=1.0)
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)
    result = validate_force_term(
        term,
        positions,
        case_name="strict-bond",
        seed=19,
        tolerance=0.0,
    )

    summary = summarize_validation_results([result])

    assert not summary["all_passed"]
    assert summary["failed_cases"] == 1
    assert summary["failed_case_names"] == ["strict-bond"]
    assert summary["worst_case"]["case_name"] == "strict-bond"


def test_default_force_validation_cases_validate_inputs():
    with pytest.raises(ValueError, match="cases_per_term"):
        default_force_validation_cases(cases_per_term=0)
    with pytest.raises(ValueError, match="epsilon"):
        ForceValidationCase(
            name="bad",
            term=HarmonicBondPotential([(0, 1)], k=1.0, length=1.0),
            positions=np.zeros((2, 3), dtype=np.float32),
            seed=1,
            epsilon=0.0,
        )


def test_validate_force_term_rejects_empty_positions():
    term = HarmonicBondPotential([(0, 1)], k=1.0, length=1.0)

    with pytest.raises(ValueError, match="at least one atom"):
        validate_force_term(term, np.zeros((0, 3), dtype=np.float32))
