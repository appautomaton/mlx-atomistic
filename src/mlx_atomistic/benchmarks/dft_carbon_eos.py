"""Scientific equation-of-state validation for periodic diamond carbon."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks.dft_eos import (
    CONVERGENCE_THRESHOLDS,
    HARTREE_TO_EV,
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

REFERENCE_SCHEMA = "mlx-atomistic.carbon-eos-references.v1"
REFERENCE_SHA256 = "8414886566bf47ed231d285293a586e0a7a6ee9f45cff44ce5944e1f502f76c6"
EOS_REPORT_SCHEMA = "mlx-atomistic.carbon-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "carbon_eos_references.json"


def load_carbon_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed diamond-carbon reference bundle."""

    return load_eos_reference_bundle(
        _reference_path(),
        expected_sha256=REFERENCE_SHA256,
        expected_schema=REFERENCE_SCHEMA,
        expected_material={
            "cell": "diamond-carbon",
            "functional": "PBE",
            "spin": "unpolarized",
        },
    )


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_carbon_eos_references() if references is None else references
    return cubic_validation_lattice_constants(payload)


def fit_cubic_carbon_eos(
    lattice_constants_angstrom: Sequence[float],
    total_energies_hartree: Sequence[float],
    *,
    atom_count: int = 8,
) -> dict[str, Any]:
    """Fit a conventional cubic-cell diamond-carbon EOS."""

    return fit_cubic_eos(
        lattice_constants_angstrom,
        total_energies_hartree,
        atom_count=atom_count,
    )


__all__ = [
    "CONVERGENCE_THRESHOLDS",
    "EOS_REPORT_SCHEMA",
    "HARTREE_TO_EV",
    "REFERENCE_SCHEMA",
    "REFERENCE_SHA256",
    "birch_murnaghan_energy",
    "compare_eos_convergence",
    "compare_fit_to_reference",
    "delta_factor_mev_per_atom",
    "fit_birch_murnaghan",
    "fit_cubic_carbon_eos",
    "load_carbon_eos_references",
    "reference_fit",
    "validation_lattice_constants",
]
