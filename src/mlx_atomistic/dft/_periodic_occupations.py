"""Weighted electronic occupations for periodic DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _PeriodicOccupationResult:
    """Resolved per-k-point occupations and thermodynamic diagnostics."""

    occupations: tuple[tuple[float, ...], ...]
    electron_count: float
    chemical_potential: float | None
    electronic_entropy: float


def _validated_spectrum(
    eigenvalues: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float],
) -> tuple[tuple[np.ndarray, ...], np.ndarray, int]:
    spectra = tuple(np.asarray(values, dtype=np.float64).reshape(-1) for values in eigenvalues)
    integration_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not spectra or integration_weights.shape != (len(spectra),):
        msg = "periodic occupations require one weight per k-point spectrum"
        raise ValueError(msg)
    band_count = int(spectra[0].size)
    if band_count == 0 or any(values.size != band_count for values in spectra):
        msg = "periodic k-point spectra must have one common positive band count"
        raise ValueError(msg)
    if any(not np.all(np.isfinite(values)) for values in spectra):
        msg = "periodic eigenvalues must be finite"
        raise ValueError(msg)
    if not np.all(np.isfinite(integration_weights)) or np.any(integration_weights <= 0.0):
        msg = "periodic k-point weights must be finite and positive"
        raise ValueError(msg)
    if not np.isclose(float(np.sum(integration_weights)), 1.0, rtol=0.0, atol=1e-12):
        msg = "periodic k-point weights must sum to one"
        raise ValueError(msg)
    return spectra, integration_weights, band_count


def _fermi_probability(energy_offset: np.ndarray, width_hartree: float) -> np.ndarray:
    scaled = np.clip(energy_offset / width_hartree, -80.0, 80.0)
    probability = np.empty_like(scaled)
    positive = scaled >= 0.0
    decaying = np.exp(-scaled[positive])
    probability[positive] = decaying / (1.0 + decaying)
    growing = np.exp(scaled[~positive])
    probability[~positive] = 1.0 / (1.0 + growing)
    return probability


def _electronic_entropy(
    probabilities: Sequence[np.ndarray],
    weights: np.ndarray,
) -> float:
    entropy = 0.0
    for probability, weight in zip(probabilities, weights, strict=True):
        interior = (probability > 0.0) & (probability < 1.0)
        values = probability[interior]
        entropy -= 2.0 * float(weight) * float(
            np.sum(values * np.log(values) + (1.0 - values) * np.log1p(-values))
        )
    return entropy


def _resolve_periodic_occupations(
    eigenvalues: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float],
    *,
    electron_count: float,
    smearing_width_hartree: float | None,
) -> _PeriodicOccupationResult:
    """Resolve fixed or Fermi-Dirac occupations over a weighted k-point mesh."""

    spectra, integration_weights, band_count = _validated_spectrum(eigenvalues, weights)
    count = float(electron_count)
    capacity = 2.0 * band_count
    if not np.isfinite(count) or count <= 0.0:
        msg = "periodic occupation electron_count must be finite and positive"
        raise ValueError(msg)
    if smearing_width_hartree is None:
        if not np.isclose(count, capacity, rtol=0.0, atol=1e-10):
            msg = "fixed periodic occupations require two electrons per computed band"
            raise ValueError(msg)
        occupations = tuple((2.0,) * band_count for _ in spectra)
        return _PeriodicOccupationResult(
            occupations=occupations,
            electron_count=count,
            chemical_potential=None,
            electronic_entropy=0.0,
        )

    width = float(smearing_width_hartree)
    if not np.isfinite(width) or width <= 0.0:
        msg = "smearing_width_hartree must be finite and positive"
        raise ValueError(msg)
    if count >= capacity:
        msg = "Fermi-Dirac occupations require at least one partially empty band"
        raise ValueError(msg)

    minimum = min(float(np.min(values)) for values in spectra)
    maximum = max(float(np.max(values)) for values in spectra)
    margin = max(1.0, 100.0 * width)
    lower = minimum - margin
    upper = maximum + margin

    def resolved(candidate: float) -> tuple[tuple[np.ndarray, ...], float]:
        probabilities = tuple(
            _fermi_probability(values - candidate, width) for values in spectra
        )
        observed = 2.0 * sum(
            float(weight) * float(np.sum(probability))
            for probability, weight in zip(probabilities, integration_weights, strict=True)
        )
        return probabilities, observed

    _, lower_count = resolved(lower)
    _, upper_count = resolved(upper)
    if not lower_count < count < upper_count:
        msg = "Fermi-Dirac chemical-potential bracket does not contain electron_count"
        raise ValueError(msg)

    tolerance = 1e-12 * max(1.0, count)
    chemical_potential = 0.5 * (lower + upper)
    probabilities: tuple[np.ndarray, ...] = ()
    observed_count = float("nan")
    for _ in range(256):
        chemical_potential = 0.5 * (lower + upper)
        probabilities, observed_count = resolved(chemical_potential)
        if abs(observed_count - count) <= tolerance:
            break
        if observed_count > count:
            upper = chemical_potential
        else:
            lower = chemical_potential
    else:
        msg = "Fermi-Dirac chemical-potential solve did not converge"
        raise RuntimeError(msg)

    occupations = tuple(
        tuple(float(value) for value in 2.0 * probability) for probability in probabilities
    )
    return _PeriodicOccupationResult(
        occupations=occupations,
        electron_count=observed_count,
        chemical_potential=chemical_potential,
        electronic_entropy=_electronic_entropy(probabilities, integration_weights),
    )
