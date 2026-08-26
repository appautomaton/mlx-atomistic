"""Axis-agnostic convergence reports for periodic DFT calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _fingerprint(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parameter_value(value: object) -> float | tuple[float, ...]:
    if isinstance(value, bool):
        raise ValueError("convergence parameter value must be finite numeric data")
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("convergence parameter value must be finite numeric data")
        return result
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("convergence parameter value must be finite numeric data")
    parsed = tuple(_parameter_value(item) for item in value)
    if any(isinstance(item, tuple) for item in parsed):
        raise ValueError("convergence parameter sequences must be one-dimensional")
    return tuple(float(item) for item in parsed)


@dataclass(frozen=True)
class PeriodicConvergenceCriterion:
    """Tolerance for one named scalar observable.

    Args:
        observable: Observable key shared by both calculation points.
        absolute_tolerance: Non-negative tolerance in the observable's unit.
        relative_tolerance: Non-negative fractional tolerance. Defaults to zero.
    """

    observable: str
    absolute_tolerance: float
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.observable, str) or not self.observable:
            raise ValueError("convergence observable must be a non-empty string")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, object]:
        """Return the criterion as JSON-safe metadata."""

        return {
            "observable": self.observable,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True)
class PeriodicConvergencePoint:
    """One source-bound periodic calculation in a convergence comparison.

    Args:
        parameter_value: Scalar or one-dimensional numerical axis value.
        calculation_fingerprint: Exact calculation-contract SHA-256 identity.
        source_fingerprint: Exact source/resource SHA-256 identity.
        runtime_fingerprint: Exact runtime/environment SHA-256 identity.
        observables: Named finite scalar values from the calculation.
    """

    parameter_value: float | tuple[float, ...]
    calculation_fingerprint: str
    source_fingerprint: str
    runtime_fingerprint: str
    observables: Mapping[str, float]

    def __post_init__(self) -> None:
        parameter = _parameter_value(self.parameter_value)
        calculation = _fingerprint(
            self.calculation_fingerprint,
            "calculation_fingerprint",
        )
        source = _fingerprint(self.source_fingerprint, "source_fingerprint")
        runtime = _fingerprint(self.runtime_fingerprint, "runtime_fingerprint")
        if not isinstance(self.observables, Mapping) or not self.observables:
            raise ValueError("convergence observables must be a non-empty mapping")
        observables: dict[str, float] = {}
        for name, value in self.observables.items():
            if not isinstance(name, str) or not name:
                raise ValueError("convergence observable names must be non-empty strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
            ):
                raise ValueError("convergence observables must be finite scalars")
            observables[name] = float(value)
        object.__setattr__(self, "parameter_value", parameter)
        object.__setattr__(self, "calculation_fingerprint", calculation)
        object.__setattr__(self, "source_fingerprint", source)
        object.__setattr__(self, "runtime_fingerprint", runtime)
        object.__setattr__(self, "observables", MappingProxyType(observables))

    def to_dict(self) -> dict[str, object]:
        """Return the source-bound point as JSON-safe metadata."""

        parameter = self.parameter_value
        return {
            "parameter_value": list(parameter) if isinstance(parameter, tuple) else parameter,
            "calculation_fingerprint": self.calculation_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "observables": dict(self.observables),
        }


@dataclass(frozen=True)
class PeriodicConvergenceMetric:
    """Evaluated convergence status for one scalar observable."""

    observable: str
    baseline_value: float
    check_value: float
    signed_difference: float
    absolute_difference: float
    relative_difference: float
    allowed_difference: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return the evaluated metric as JSON-safe metadata."""

        return {
            "observable": self.observable,
            "baseline_value": self.baseline_value,
            "check_value": self.check_value,
            "signed_difference": self.signed_difference,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "allowed_difference": self.allowed_difference,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PeriodicConvergenceReport:
    """Reusable comparison of two exact periodic calculation identities."""

    axis: str
    baseline: PeriodicConvergencePoint
    check: PeriodicConvergencePoint
    metrics: tuple[PeriodicConvergenceMetric, ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return the complete convergence report as JSON-safe metadata."""

        return {
            "axis": self.axis,
            "baseline": self.baseline.to_dict(),
            "check": self.check.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "passed": self.passed,
        }


def compare_periodic_convergence(
    axis: str,
    baseline: PeriodicConvergencePoint,
    check: PeriodicConvergencePoint,
    criteria: Sequence[PeriodicConvergenceCriterion],
) -> PeriodicConvergenceReport:
    """Compare two source-bound periodic calculations on any numerical axis.

    Args:
        axis: Non-empty numerical-axis label such as ``cutoff_hartree``.
        baseline: Selected production or lower-resolution calculation.
        check: Independent refined calculation.
        criteria: Unique observable tolerances to evaluate.

    Returns:
        Per-observable differences and aggregate pass/fail status.

    Raises:
        TypeError: If points or criteria use unsupported types.
        ValueError: If identities, the axis, criteria, or observables are invalid.
    """

    if not isinstance(axis, str) or not axis:
        raise ValueError("convergence axis must be a non-empty string")
    if not isinstance(baseline, PeriodicConvergencePoint) or not isinstance(
        check,
        PeriodicConvergencePoint,
    ):
        raise TypeError("baseline and check must be PeriodicConvergencePoint values")
    if baseline.calculation_fingerprint == check.calculation_fingerprint:
        raise ValueError("convergence points must bind distinct calculations")
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
        raise TypeError("criteria must be a sequence of convergence criteria")
    parsed = tuple(criteria)
    if not parsed or any(
        not isinstance(criterion, PeriodicConvergenceCriterion)
        for criterion in parsed
    ):
        raise ValueError("criteria must contain convergence criterion values")
    names = tuple(criterion.observable for criterion in parsed)
    if len(set(names)) != len(names):
        raise ValueError("convergence criteria must name unique observables")
    missing = [
        name
        for name in names
        if name not in baseline.observables or name not in check.observables
    ]
    if missing:
        raise ValueError(f"convergence observables are missing: {', '.join(missing)}")
    metrics = []
    for criterion in parsed:
        baseline_value = baseline.observables[criterion.observable]
        check_value = check.observables[criterion.observable]
        signed = check_value - baseline_value
        absolute = abs(signed)
        scale = max(abs(baseline_value), abs(check_value))
        relative = 0.0 if scale == 0.0 else absolute / scale
        allowed = (
            criterion.absolute_tolerance
            + criterion.relative_tolerance * scale
        )
        metrics.append(
            PeriodicConvergenceMetric(
                observable=criterion.observable,
                baseline_value=baseline_value,
                check_value=check_value,
                signed_difference=signed,
                absolute_difference=absolute,
                relative_difference=relative,
                allowed_difference=allowed,
                absolute_tolerance=criterion.absolute_tolerance,
                relative_tolerance=criterion.relative_tolerance,
                passed=absolute <= allowed,
            )
        )
    result = tuple(metrics)
    return PeriodicConvergenceReport(
        axis=axis,
        baseline=baseline,
        check=check,
        metrics=result,
        passed=all(metric.passed for metric in result),
    )
