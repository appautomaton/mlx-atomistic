"""Fail-closed numerical and manifest helpers for PME validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForceErrorMetrics:
    """Absolute and scale-aware force and energy differences."""

    rms_absolute_kj_mol_nm: float
    maximum_absolute_kj_mol_nm: float
    normalized_rms: float
    normalized_maximum: float
    energy_error_per_atom_kj_mol: float
    relative_energy_error: float


class PMEManifestMismatchError(ValueError):
    """Raised when two PME workload manifests do not describe the same work."""

    def __init__(self, mismatches: dict[str, dict[str, Any]]) -> None:
        self.mismatches = mismatches
        fields = ", ".join(sorted(mismatches))
        super().__init__(f"PME workload manifest mismatch: {fields}")


def force_error_metrics(
    candidate_forces: np.ndarray,
    reference_forces: np.ndarray,
    *,
    candidate_energy: float,
    reference_energy: float,
) -> ForceErrorMetrics:
    """Return complete absolute and normalized force/energy error metrics.

    Args:
        candidate_forces: Candidate forces with shape ``(n_atoms, 3)``.
        reference_forces: Non-zero reference forces with matching shape.
        candidate_energy: Candidate scalar energy in kJ/mol.
        reference_energy: Non-zero reference scalar energy in kJ/mol.

    Returns:
        Absolute RMS/maximum force errors, normalized force errors, energy
        error per atom, and relative energy error.

    Raises:
        ValueError: If shapes differ, values are non-finite, the fixture is
            empty, or a normalized reference denominator is zero.
    """

    candidate = np.asarray(candidate_forces, dtype=np.float64)
    reference = np.asarray(reference_forces, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        msg = "candidate and reference forces must have matching shape (n_atoms, 3)"
        raise ValueError(msg)
    if candidate.shape[0] == 0:
        msg = "force metrics require at least one atom"
        raise ValueError(msg)
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        msg = "force metrics require finite candidate and reference forces"
        raise ValueError(msg)
    energies = np.asarray([candidate_energy, reference_energy], dtype=np.float64)
    if not np.all(np.isfinite(energies)):
        msg = "force metrics require finite candidate and reference energies"
        raise ValueError(msg)

    reference_rms_denominator = float(np.sum(reference * reference))
    reference_maximum = float(np.max(np.linalg.norm(reference, axis=1)))
    reference_energy_magnitude = abs(float(reference_energy))
    if reference_rms_denominator <= 0.0 or reference_maximum <= 0.0:
        msg = "force metrics require a non-zero reference force field"
        raise ValueError(msg)
    if reference_energy_magnitude <= 0.0:
        msg = "force metrics require a non-zero reference energy"
        raise ValueError(msg)

    delta = candidate - reference
    energy_error = abs(float(candidate_energy) - float(reference_energy))
    return ForceErrorMetrics(
        rms_absolute_kj_mol_nm=float(np.sqrt(np.mean(delta * delta))),
        maximum_absolute_kj_mol_nm=float(np.max(np.abs(delta))),
        normalized_rms=_sqrt_ratio(
            float(np.sum(delta * delta)),
            reference_rms_denominator,
        ),
        normalized_maximum=float(np.max(np.linalg.norm(delta, axis=1)))
        / reference_maximum,
        energy_error_per_atom_kj_mol=energy_error / candidate.shape[0],
        relative_energy_error=energy_error / reference_energy_magnitude,
    )


def array_hash(values: np.ndarray) -> str:
    """Return a stable SHA-256 hash that includes array dtype and shape.

    Args:
        values: Numerical or string array to hash.

    Returns:
        Hexadecimal SHA-256 digest over dtype, shape, and contiguous bytes.
    """

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable manifest.

    Args:
        manifest: Manifest mapping containing only JSON-compatible values.

    Returns:
        Hexadecimal digest of canonical sorted JSON.
    """

    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest_mismatches(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Return mismatches for required dotted manifest fields.

    Args:
        candidate: Candidate manifest.
        reference: Reference manifest.
        fields: Required dotted paths such as ``"pme.mesh_shape"``.

    Returns:
        Mapping from mismatched field to candidate/reference values. Missing
        fields are mismatches rather than wildcards.
    """

    mismatches: dict[str, dict[str, Any]] = {}
    missing = object()
    for field in fields:
        candidate_value = _manifest_value(candidate, field, missing)
        reference_value = _manifest_value(reference, field, missing)
        if (
            candidate_value is missing
            or reference_value is missing
            or candidate_value != reference_value
        ):
            mismatches[field] = {
                "candidate": None if candidate_value is missing else candidate_value,
                "reference": None if reference_value is missing else reference_value,
                "candidate_present": candidate_value is not missing,
                "reference_present": reference_value is not missing,
            }
    return mismatches


def require_matching_manifest(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> None:
    """Raise when required workload-manifest fields differ or are missing.

    Args:
        candidate: Candidate manifest.
        reference: Reference manifest.
        fields: Required dotted manifest paths.

    Raises:
        PMEManifestMismatchError: If any required field is absent or differs.
    """

    mismatches = manifest_mismatches(candidate, reference, fields=fields)
    if mismatches:
        raise PMEManifestMismatchError(mismatches)


def _sqrt_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        msg = "normalized metric denominator must be finite and positive"
        raise ValueError(msg)
    return float(np.sqrt(max(0.0, numerator) / denominator))


def _manifest_value(manifest: dict[str, Any], field: str, missing: object) -> Any:
    value: Any = manifest
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return missing
        value = value[part]
    return value


__all__ = [
    "ForceErrorMetrics",
    "PMEManifestMismatchError",
    "array_hash",
    "force_error_metrics",
    "manifest_hash",
    "manifest_mismatches",
    "require_matching_manifest",
]
