"""Reference-case comparison helpers for DFT validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_atomistic.runtime import ReadinessReport


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


@dataclass(frozen=True)
class DFTQMScopeEntry:
    """One local DFT/QM capability classification against reference suites."""

    feature: str
    status: str
    local_surface: tuple[str, ...]
    reference_families: tuple[str, ...]
    rationale: str
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe DFT/QM scope entry."""

        return {
            "feature": self.feature,
            "status": self.status,
            "local_surface": list(self.local_surface),
            "reference_families": list(self.reference_families),
            "rationale": self.rationale,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class DFTQMScopeReport:
    """Report local DFT/QM scope without claiming CP2K/QE suite parity."""

    product_runtime: str
    reference_policy: dict[str, str]
    entries: tuple[DFTQMScopeEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe DFT/QM scope report."""

        return {
            "product_runtime": self.product_runtime,
            "reference_policy": dict(self.reference_policy),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def entry_for(self, feature: str) -> DFTQMScopeEntry | None:
        """Return the entry for a feature or alias."""

        normalized = _normalize_feature(feature)
        for entry in self.entries:
            if normalized == entry.feature or normalized in _FEATURE_ALIASES.get(
                entry.feature,
                (),
            ):
                return entry
        return None


_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "plane_wave_scf": ("scf", "pw", "pwscf", "quickstep"),
    "pseudopotentials": ("upf", "gth", "nonlocal_pseudopotential"),
    "geometry_and_stress": ("geometry", "relaxation", "stress", "motion"),
    "static_reference_comparison": ("reference_comparison", "fixtures"),
    "qmmm_orchestration": ("qmmm", "qm_mm", "force_environment"),
    "production_suite_modules": ("phonons", "td_dft", "tddft", "epw", "neb", "mpi"),
    "external_runtime_execution": ("run_cp2k", "run_qe", "qe_runtime", "cp2k_runtime"),
}


def _normalize_feature(feature: str) -> str:
    return feature.strip().lower().replace("-", "_").replace(" ", "_")


def get_dft_qm_scope_report() -> DFTQMScopeReport:
    """Classify local DFT/QM scope against CP2K and Quantum ESPRESSO families."""

    return DFTQMScopeReport(
        product_runtime="mlx_atomistic",
        reference_policy={
            "cp2k": "reference family for Quickstep, force environments, and QM/MM scope",
            "quantum_espresso": "reference family for PWscf, UPF, bands, and suite modules",
            "vendors": "static reference source/data only; not runtime dependencies",
        },
        entries=(
            DFTQMScopeEntry(
                feature="plane_wave_scf",
                status="proof-level",
                local_surface=("DFTSystem", "run_scf", "KohnShamOperator", "SCFResult"),
                reference_families=("CP2K Quickstep", "Quantum ESPRESSO PWscf"),
                rationale=(
                    "Local SCF, LDA, Hartree, kinetic, and pseudopotential paths are "
                    "tested on tiny grids but are not chemically certified production DFT."
                ),
            ),
            DFTQMScopeEntry(
                feature="pseudopotentials",
                status="proof-level",
                local_surface=("read_upf", "read_gth", "NonlocalPseudopotentialOperator"),
                reference_families=("Quantum ESPRESSO UPF", "CP2K GTH"),
                rationale=(
                    "UPF and GTH parsing plus nonlocal projector application are "
                    "covered by fixtures, with format-convention limits documented."
                ),
            ),
            DFTQMScopeEntry(
                feature="geometry_and_stress",
                status="proof-level",
                local_surface=("optimize_geometry", "finite_difference_stress"),
                reference_families=("CP2K MOTION/GEO_OPT", "Quantum ESPRESSO relax"),
                rationale=(
                    "Geometry and diagonal stress workflows exist for small local "
                    "fixtures; cell, constraints, and production relaxation breadth "
                    "remain limited."
                ),
            ),
            DFTQMScopeEntry(
                feature="static_reference_comparison",
                status="supported",
                local_surface=("ReferenceDFTCase", "compare_reference_case"),
                reference_families=("static CP2K/QE fixture summaries",),
                rationale=(
                    "Reference comparisons are static JSON-safe summaries and do not "
                    "execute external engines."
                ),
            ),
            DFTQMScopeEntry(
                feature="qmmm_orchestration",
                status="deferred",
                local_surface=("separate MD and DFT modules",),
                reference_families=("CP2K FORCE_EVAL/QMMM",),
                rationale="No coupled QM/MM force-environment orchestration exists locally.",
                blockers=("qmmm_orchestration:deferred",),
            ),
            DFTQMScopeEntry(
                feature="production_suite_modules",
                status="deferred",
                local_surface=("DFT proof modules only",),
                reference_families=("QE PH/EPW/NEB/TDDFT", "CP2K MPI/offload suites"),
                rationale=(
                    "Phonons, EPW, NEB, TDDFT, MPI/offload, and broad production "
                    "suite workflows are outside this platform-maturity slice."
                ),
                blockers=("production_suite_modules:deferred",),
            ),
            DFTQMScopeEntry(
                feature="external_runtime_execution",
                status="anti-goal",
                local_surface=("vendors/ reference-only policy",),
                reference_families=("CP2K executable", "Quantum ESPRESSO executable"),
                rationale="This project does not import, build, wrap, or run CP2K/QE as runtime.",
                blockers=("external_runtime_execution:anti_goal",),
            ),
        ),
    )


def dft_qm_scope_readiness_report(feature: str) -> ReadinessReport:
    """Return a fail-closed readiness report for one DFT/QM feature request."""

    report = get_dft_qm_scope_report()
    entry = report.entry_for(feature)
    if entry is None:
        normalized = _normalize_feature(feature)
        return ReadinessReport(
            name="dft_qm_scope",
            status="fail-closed",
            blockers=(f"unknown_dft_qm_feature:{normalized}",),
            metadata={
                "feature": normalized,
                "known_features": [item.feature for item in report.entries],
            },
        )
    blockers = entry.blockers
    if entry.status in {"deferred", "anti-goal", "fail-closed"} and not blockers:
        blockers = (f"{entry.feature}:{entry.status}",)
    return ReadinessReport(
        name="dft_qm_scope",
        status=entry.status,
        blockers=blockers,
        metadata={
            "product_runtime": report.product_runtime,
            "reference_policy": report.reference_policy,
            "entry": entry.to_dict(),
        },
    )


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


__all__ = [
    "DFTQMScopeEntry",
    "DFTQMScopeReport",
    "ReferenceComparisonResult",
    "ReferenceDFTCase",
    "compare_reference_case",
    "dft_qm_scope_readiness_report",
    "get_dft_qm_scope_report",
]
