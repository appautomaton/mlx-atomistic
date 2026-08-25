"""Scientific equation-of-state validation for periodic silicon DFT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks.dft_eos import (
    CONVERGENCE_THRESHOLDS,
    EV_PER_ANGSTROM3_TO_GPA,
    EXCELLENT_THRESHOLDS,
    HARTREE_TO_EV,
    VERIFIED_THRESHOLDS,
    birch_murnaghan_energy,
    compare_eos_convergence,
    compare_fit_to_reference,
    cubic_validation_lattice_constants,
    delta_factor_mev_per_atom,
    fit_birch_murnaghan,
    fit_cubic_eos,
    load_eos_reference_bundle,
    reference_fit,
)

REFERENCE_SCHEMA = "mlx-atomistic.silicon-eos-references.v1"
REFERENCE_SHA256 = "3cbf727f17d31ab7859acfc32d0bc313b5c02f7e870cd97411aa695c5986d53a"
EOS_REPORT_SCHEMA = "mlx-atomistic.silicon-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "silicon_eos_references.json"


def load_silicon_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed silicon EOS reference bundle."""

    return load_eos_reference_bundle(
        _reference_path(),
        expected_sha256=REFERENCE_SHA256,
        expected_schema=REFERENCE_SCHEMA,
        expected_material={
            "cell": "diamond-silicon",
            "functional": "PBE",
            "spin": "unpolarized",
        },
    )


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_silicon_eos_references() if references is None else references
    return cubic_validation_lattice_constants(payload)


def fit_cubic_silicon_eos(
    lattice_constants_angstrom: Sequence[float],
    total_energies_hartree: Sequence[float],
    *,
    atom_count: int = 8,
) -> dict[str, Any]:
    """Fit a conventional cubic-cell silicon EOS from total energies."""

    return fit_cubic_eos(
        lattice_constants_angstrom,
        total_energies_hartree,
        atom_count=atom_count,
    )


__all__ = [
    "CONVERGENCE_THRESHOLDS",
    "EOS_REPORT_SCHEMA",
    "EV_PER_ANGSTROM3_TO_GPA",
    "EXCELLENT_THRESHOLDS",
    "HARTREE_TO_EV",
    "REFERENCE_SCHEMA",
    "REFERENCE_SHA256",
    "VERIFIED_THRESHOLDS",
    "birch_murnaghan_energy",
    "compare_eos_convergence",
    "compare_fit_to_reference",
    "delta_factor_mev_per_atom",
    "fit_birch_murnaghan",
    "fit_cubic_silicon_eos",
    "load_silicon_eos_references",
    "reference_fit",
    "validation_lattice_constants",
]
