"""Runtime probes for the local MLX environment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    """Small, display-friendly summary of the active MLX runtime."""

    mlx_version: str
    default_device: str
    metal_available: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe runtime summary."""

        return {
            "mlx_version": self.mlx_version,
            "default_device": self.default_device,
            "metal_available": self.metal_available,
        }


@dataclass(frozen=True)
class PlatformBoundarySection:
    """One local platform-boundary concept group."""

    name: str
    status: str
    local_concepts: tuple[str, ...]
    responsibility: str
    vendor_references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe section summary."""

        return {
            "name": self.name,
            "status": self.status,
            "local_concepts": list(self.local_concepts),
            "responsibility": self.responsibility,
            "vendor_references": list(self.vendor_references),
        }


@dataclass(frozen=True)
class ReadinessReport:
    """JSON-safe readiness status shared by platform gates."""

    name: str
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe readiness report."""

        return {
            "name": self.name,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": {} if self.metadata is None else dict(self.metadata),
        }


@dataclass(frozen=True)
class PlatformBoundaryReport:
    """Traceable summary of the local mini-platform boundary."""

    product_runtime: str
    runtime: RuntimeInfo
    reference_engine_policy: dict[str, str]
    sections: tuple[PlatformBoundarySection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe platform-boundary report."""

        return {
            "product_runtime": self.product_runtime,
            "runtime": self.runtime.to_dict(),
            "reference_engine_policy": dict(self.reference_engine_policy),
            "sections": [section.to_dict() for section in self.sections],
        }


def get_runtime_info() -> RuntimeInfo:
    """Return basic information about MLX and the default device."""

    import mlx.core as mx

    return RuntimeInfo(
        mlx_version=version("mlx"),
        default_device=str(mx.default_device()),
        metal_available=bool(mx.metal.is_available()),
    )


def get_platform_boundary_report(
    *,
    runtime_info: RuntimeInfo | None = None,
) -> PlatformBoundaryReport:
    """Return the local platform-boundary contract without importing reference engines."""

    runtime = get_runtime_info() if runtime_info is None else runtime_info
    reference_policy = {
        "openmm": "reference and preview engine; dev/test surface only",
        "lammps": "reference engine for neighbor and GPU/OpenCL semantics",
        "gromacs": "reference for biomolecular MD staging and PME/nonbonded performance",
        "cp2k": "reference for force-environment and QM/MM suite boundaries",
        "quantum_espresso": "reference for plane-wave pseudopotential DFT suite boundaries",
        "vendors": "local reference source trees only; not package inputs or runtime deps",
    }
    sections = (
        PlatformBoundarySection(
            name="runtime_backend",
            status="supported",
            local_concepts=("RuntimeInfo", "get_runtime_info", "MLX/Metal"),
            responsibility="Report active MLX runtime, device, and reference-engine policy.",
            vendor_references=("OpenMM Platform", "GROMACS GPU backend"),
        ),
        PlatformBoundarySection(
            name="system_artifact",
            status="supported",
            local_concepts=(
                "MMSystem",
                "PreparedMLXArtifact",
                "validate_mlx_compatibility",
            ),
            responsibility="Represent prepared systems, units, topology terms, and compatibility.",
            vendor_references=("OpenMM System", "GROMACS topology/preprocessing"),
        ),
        PlatformBoundarySection(
            name="protocol",
            status="proof-level",
            local_concepts=(
                "MinimizeThenNVTProtocol",
                "validate_gpcrmd_protocol_request",
                "simulate_nve/simulate_nvt/simulate_npt",
                "NoseHooverThermostat",
            ),
            responsibility=(
                "Classify accepted ensembles, proof modes, thermostats, and protocol blockers."
            ),
            vendor_references=("OpenMM Integrator", "LAMMPS fix"),
        ),
        PlatformBoundarySection(
            name="readiness",
            status="proof-level",
            local_concepts=(
                "MLXCompatibilityError",
                "ProtocolCompatibilityReport",
                "ReadinessReport",
                "pme_readiness_report",
            ),
            responsibility="Expose supported, proof-level, fail-closed, and deferred physics.",
            vendor_references=("GROMACS mdrun checks", "LAMMPS package/style validation"),
        ),
        PlatformBoundarySection(
            name="validation",
            status="proof-level",
            local_concepts=(
                "ForceValidationCase",
                "ForceValidationResult",
                "OpenMM parity scripts",
            ),
            responsibility="Bind fixtures, parity metrics, runtime metadata, and finite checks.",
            vendor_references=("OpenMM validation", "GROMACS regression tests"),
        ),
        PlatformBoundarySection(
            name="dft_qm_scope",
            status="proof-level",
            local_concepts=("DFTSystem", "SCFResult", "ReferenceDFTCase"),
            responsibility="Classify local DFT/QM proofs without production-suite parity claims.",
            vendor_references=("CP2K Quickstep", "Quantum ESPRESSO PW/PH/EPW suites"),
        ),
    )
    return PlatformBoundaryReport(
        product_runtime="mlx_atomistic",
        runtime=runtime,
        reference_engine_policy=reference_policy,
        sections=sections,
    )


__all__ = [
    "PlatformBoundaryReport",
    "PlatformBoundarySection",
    "ReadinessReport",
    "RuntimeInfo",
    "get_platform_boundary_report",
    "get_runtime_info",
]
