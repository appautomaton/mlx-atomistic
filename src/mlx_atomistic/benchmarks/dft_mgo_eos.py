"""Scientific equation-of-state validation for rock-salt magnesium oxide."""

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

REFERENCE_SCHEMA = "mlx-atomistic.mgo-eos-references.v1"
REFERENCE_SHA256 = "ad4e262690181b6f009e5683eadb8402ee47c73acee32ca96873c95cf1c3e5ea"
EOS_REPORT_SCHEMA = "mlx-atomistic.mgo-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "mgo_eos_references.json"


def load_mgo_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed rock-salt MgO reference bundle."""

    raw = _reference_path().read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("MgO EOS reference bundle hash mismatch")
    payload = json.loads(raw)
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError("unsupported MgO EOS reference schema")
    if payload.get("material") != {
        "cell": "rocksalt-mgo",
        "functional": "PBE",
        "spin": "unpolarized",
    }:
        raise ValueError("MgO EOS reference material identity mismatch")
    if payload.get("protocol", {}).get("volume_factors") != [
        0.94,
        0.96,
        0.98,
        1.0,
        1.02,
        1.04,
        1.06,
    ]:
        raise ValueError("MgO EOS reference volume protocol mismatch")
    return payload


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_mgo_eos_references() if references is None else references
    protocol = payload["protocol"]
    center = float(protocol["central_conventional_lattice_angstrom"])
    return [
        center * float(volume_factor) ** (1.0 / 3.0)
        for volume_factor in protocol["volume_factors"]
    ]


def fit_cubic_mgo_eos(
    lattice_constants_angstrom: Sequence[float],
    total_energies_hartree: Sequence[float],
    *,
    atom_count: int = 8,
) -> dict[str, Any]:
    """Fit a conventional cubic-cell rock-salt MgO EOS."""

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
    "fit_cubic_mgo_eos",
    "load_mgo_eos_references",
    "reference_fit",
    "validation_lattice_constants",
]
