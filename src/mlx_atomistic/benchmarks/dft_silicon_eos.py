"""Scientific equation-of-state validation for periodic silicon DFT."""

from __future__ import annotations

import hashlib
import json
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
    delta_factor_mev_per_atom,
    fit_birch_murnaghan,
    fit_cubic_eos,
    reference_fit,
)

REFERENCE_SCHEMA = "mlx-atomistic.silicon-eos-references.v1"
REFERENCE_SHA256 = "3cbf727f17d31ab7859acfc32d0bc313b5c02f7e870cd97411aa695c5986d53a"
EOS_REPORT_SCHEMA = "mlx-atomistic.silicon-eos-report.v1"


def _reference_path() -> Path:
    return Path(__file__).with_name("data") / "silicon_eos_references.json"


def load_silicon_eos_references() -> dict[str, Any]:
    """Load the pinned, source-attributed silicon EOS reference bundle."""

    path = _reference_path()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REFERENCE_SHA256:
        msg = "silicon EOS reference bundle hash mismatch"
        raise ValueError(msg)
    payload = json.loads(raw)
    if payload.get("schema_version") != REFERENCE_SCHEMA:
        msg = "unsupported silicon EOS reference schema"
        raise ValueError(msg)
    if payload.get("material") != {
        "cell": "diamond-silicon",
        "functional": "PBE",
        "spin": "unpolarized",
    }:
        msg = "silicon EOS reference material identity mismatch"
        raise ValueError(msg)
    factors = payload.get("protocol", {}).get("volume_factors")
    if factors != [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]:
        msg = "silicon EOS reference volume protocol mismatch"
        raise ValueError(msg)
    return payload


def validation_lattice_constants(
    references: Mapping[str, Any] | None = None,
) -> list[float]:
    """Return the seven conventional-cell lattice constants for validation."""

    payload = load_silicon_eos_references() if references is None else references
    protocol = payload["protocol"]
    center = float(protocol["central_conventional_lattice_angstrom"])
    return [
        center * float(volume_factor) ** (1.0 / 3.0) for volume_factor in protocol["volume_factors"]
    ]


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
