"""Scientific equation-of-state validation for periodic fcc Aluminum."""

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

REFERENCE_SCHEMA = "mlx-atomistic.aluminum-eos-references.v1"
REFERENCE_SHA256 = "eebfe487557fcff52b1934ced8924844c307e0cfe30c739987e36734890a6788"
EOS_REPORT_SCHEMA = "mlx-atomistic.aluminum-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "aluminum_eos_references.json"


def load_aluminum_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed fcc Aluminum reference bundle."""

    return load_eos_reference_bundle(
        _reference_path(),
        expected_sha256=REFERENCE_SHA256,
        expected_schema=REFERENCE_SCHEMA,
        expected_material={
            "cell": "fcc-aluminum",
            "functional": "PBE",
            "spin": "unpolarized",
        },
    )


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_aluminum_eos_references() if references is None else references
    return cubic_validation_lattice_constants(payload)


def fit_cubic_aluminum_eos(
    lattice_constants_angstrom: Sequence[float],
    free_energies_hartree: Sequence[float],
    *,
    atom_count: int = 4,
) -> dict[str, Any]:
    """Fit a conventional cubic-cell Aluminum EOS from free energies."""

    return fit_cubic_eos(
        lattice_constants_angstrom,
        free_energies_hartree,
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
    "fit_cubic_aluminum_eos",
    "load_aluminum_eos_references",
    "reference_fit",
    "validation_lattice_constants",
]
