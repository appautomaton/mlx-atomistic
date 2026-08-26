from __future__ import annotations

import math

import pytest

from mlx_atomistic.dft import (
    PeriodicConvergenceCriterion,
    PeriodicConvergencePoint,
    compare_periodic_convergence,
)


def _point(
    parameter_value: float | tuple[float, ...],
    calculation_digit: str,
    observables: dict[str, float],
) -> PeriodicConvergencePoint:
    return PeriodicConvergencePoint(
        parameter_value=parameter_value,
        calculation_fingerprint=calculation_digit * 64,
        source_fingerprint="a" * 64,
        runtime_fingerprint="b" * 64,
        observables=observables,
    )


def test_periodic_convergence_reports_signed_absolute_and_relative_differences():
    report = compare_periodic_convergence(
        "cutoff_hartree",
        _point(100.0, "1", {"energy_hartree_per_atom": -4.0, "force": 0.02}),
        _point(150.0, "2", {"energy_hartree_per_atom": -4.001, "force": 0.019}),
        (
            PeriodicConvergenceCriterion("energy_hartree_per_atom", 0.002),
            PeriodicConvergenceCriterion("force", 0.0, 0.1),
        ),
    )

    assert report.passed is True
    assert report.metrics[0].signed_difference == pytest.approx(-0.001)
    assert report.metrics[0].absolute_difference == pytest.approx(0.001)
    assert report.metrics[1].relative_difference == pytest.approx(0.05)
    assert report.to_dict()["baseline"]["source_fingerprint"] == "a" * 64


def test_periodic_convergence_reuses_one_contract_for_kpoint_and_smearing_axes():
    criterion = (PeriodicConvergenceCriterion("free_energy_hartree", 5.0e-4),)
    kpoint = compare_periodic_convergence(
        "kpoint_mesh",
        _point((4, 4, 4), "3", {"free_energy_hartree": -3.0}),
        _point((6, 6, 6), "4", {"free_energy_hartree": -2.999}),
        criterion,
    )
    smearing = compare_periodic_convergence(
        "smearing_width_hartree",
        _point(0.01, "5", {"free_energy_hartree": -3.0}),
        _point(0.005, "6", {"free_energy_hartree": -3.0002}),
        criterion,
    )

    assert kpoint.passed is False
    assert kpoint.baseline.parameter_value == (4.0, 4.0, 4.0)
    assert smearing.passed is True
    assert smearing.axis == "smearing_width_hartree"


def test_periodic_convergence_fails_closed_on_invalid_or_missing_evidence():
    baseline = _point(100.0, "7", {"energy": -1.0})
    check = _point(150.0, "8", {"other": -1.0})

    with pytest.raises(ValueError, match="missing"):
        compare_periodic_convergence(
            "cutoff_hartree",
            baseline,
            check,
            (PeriodicConvergenceCriterion("energy", 0.1),),
        )
    with pytest.raises(ValueError, match="distinct"):
        compare_periodic_convergence(
            "cutoff_hartree",
            baseline,
            _point(150.0, "7", {"energy": -1.0}),
            (PeriodicConvergenceCriterion("energy", 0.1),),
        )
    with pytest.raises(ValueError, match="finite"):
        _point(100.0, "9", {"energy": math.nan})
    with pytest.raises(ValueError, match="unique"):
        compare_periodic_convergence(
            "cutoff_hartree",
            baseline,
            _point(150.0, "0", {"energy": -1.0}),
            (
                PeriodicConvergenceCriterion("energy", 0.1),
                PeriodicConvergenceCriterion("energy", 0.2),
            ),
        )
