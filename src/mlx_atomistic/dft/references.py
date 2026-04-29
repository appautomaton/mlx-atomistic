"""Reference-case comparison helpers for DFT validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceDFTCase:
    """Static external-reference summary for a tiny DFT case."""

    name: str
    source: str
    expected_energy: float
    energy_tolerance: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe reference case."""

        return {
            "name": self.name,
            "source": self.source,
            "expected_energy": self.expected_energy,
            "energy_tolerance": self.energy_tolerance,
            "metadata": {} if self.metadata is None else dict(self.metadata),
        }


@dataclass(frozen=True)
class ReferenceComparisonResult:
    """Comparison between an observed value and a reference case."""

    case: ReferenceDFTCase
    observed_energy: float
    energy_error: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe comparison result."""

        return {
            "case": self.case.to_dict(),
            "observed_energy": self.observed_energy,
            "energy_error": self.energy_error,
            "passed": self.passed,
        }


def compare_reference_case(
    case: ReferenceDFTCase,
    *,
    observed_energy: float,
) -> ReferenceComparisonResult:
    """Compare an observed energy with a static reference tolerance."""

    error = float(observed_energy - case.expected_energy)
    return ReferenceComparisonResult(
        case=case,
        observed_energy=float(observed_energy),
        energy_error=error,
        passed=abs(error) <= case.energy_tolerance,
    )
