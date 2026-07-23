"""Scientific equation-of-state validation for periodic diamond carbon."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_atomistic.benchmarks.dft_eos import (
    CONVERGENCE_THRESHOLDS,
    HARTREE_TO_EV,
    birch_murnaghan_energy,
    compare_eos_convergence,
    compare_fit_to_reference,
    delta_factor_mev_per_atom,
    fit_birch_murnaghan,
    fit_cubic_eos,
    reference_fit,
)

REFERENCE_SCHEMA = "mlx-atomistic.carbon-eos-references.v1"
REFERENCE_SHA256 = "8414886566bf47ed231d285293a586e0a7a6ee9f45cff44ce5944e1f502f76c6"
EOS_REPORT_SCHEMA = "mlx-atomistic.carbon-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "carbon_eos_references.json"


def load_carbon_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed diamond-carbon reference bundle."""

    raw = _reference_path().read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("carbon EOS reference bundle hash mismatch")
    payload = json.loads(raw)
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("unsupported carbon EOS reference schema")
    if payload.get("material") != {
        "cell": "diamond-carbon",
        "functional": "PBE",
        "spin": "unpolarized",
    }:
        raise ValueError("carbon EOS reference material identity mismatch")
    if payload.get("protocol", {}).get("volume_factors") != [
        0.94,
        0.96,
        0.98,
        1.0,
        1.02,
        1.04,
        1.06,
    ]:
        raise ValueError("carbon EOS reference volume protocol mismatch")
    return payload


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_carbon_eos_references() if references is None else references
    protocol = payload["protocol"]
    center = float(protocol["central_conventional_lattice_angstrom"])
    return [
        center * float(volume_factor) ** (1.0 / 3.0) for volume_factor in protocol["volume_factors"]
    ]


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
