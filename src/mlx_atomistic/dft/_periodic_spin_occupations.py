"""Spin-resolved weighted occupations for periodic DFT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlx_atomistic.dft._periodic_occupations import (
    _fermi_probability,
    _validated_spectrum,
)


@dataclass(frozen=True)
class _PeriodicSpinOccupationResult:
    """Two-channel occupations and thermodynamic diagnostics."""

    occupations: tuple[tuple[tuple[float, ...], ...], ...]
    electron_counts: tuple[float, float]
    chemical_potentials: tuple[float | None, float | None]
    shared_chemical_potential: float | None
    electronic_entropy: float


def _spin_entropy(probabilities: tuple[np.ndarray, ...], weights: np.ndarray) -> float:
    entropy = 0.0
    for probability, weight in zip(probabilities, weights, strict=True):
        interior = (probability > 0.0) & (probability < 1.0)
        values = probability[interior]
        entropy -= float(weight) * float(
            np.sum(values * np.log(values) + (1.0 - values) * np.log1p(-values))
        )
    return entropy


def _resolve_fermi_channel(
    spectra: tuple[np.ndarray, ...],
    weights: np.ndarray,
    *,
    electron_count: float,
    width_hartree: float,
) -> tuple[tuple[np.ndarray, ...], float, float]:
    minimum = min(float(np.min(values)) for values in spectra)
    maximum = max(float(np.max(values)) for values in spectra)
    margin = max(1.0, 100.0 * width_hartree)
    lower = minimum - margin
    upper = maximum + margin

    def resolved(candidate: float) -> tuple[tuple[np.ndarray, ...], float]:
        probabilities = tuple(
            _fermi_probability(values - candidate, width_hartree) for values in spectra
        )
        observed = sum(
            float(weight) * float(np.sum(probability))
            for probability, weight in zip(probabilities, weights, strict=True)
        )
        return probabilities, observed

    tolerance = 1.0e-12 * max(1.0, electron_count)
    probabilities: tuple[np.ndarray, ...] = ()
    observed = float("nan")
    chemical_potential = 0.5 * (lower + upper)
    for _ in range(256):
        chemical_potential = 0.5 * (lower + upper)
        probabilities, observed = resolved(chemical_potential)
        if abs(observed - electron_count) <= tolerance:
            break
        if observed > electron_count:
            upper = chemical_potential
        else:
            lower = chemical_potential
    else:
        raise RuntimeError("spin Fermi-Dirac chemical-potential solve did not converge")
    return probabilities, observed, chemical_potential


def _fixed_integer_occupations(
    spectra: tuple[np.ndarray, ...],
    electron_count: float,
) -> tuple[tuple[float, ...], ...]:
    rounded = int(round(electron_count))
    if not np.isclose(electron_count, rounded, atol=1.0e-10, rtol=0.0):
        raise ValueError("fixed spin occupations require integer channel electron counts")
    band_count = int(spectra[0].size)
    if not 0 <= rounded <= band_count:
        raise ValueError("spin channel electron count exceeds its band capacity")
    values = (1.0,) * rounded + (0.0,) * (band_count - rounded)
    return tuple(values for _ in spectra)


def _resolve_periodic_spin_occupations(
    eigenvalues: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    weights: tuple[float, ...],
    *,
    electron_count: float,
    smearing_width_hartree: float | None,
    magnetization: float | None,
) -> _PeriodicSpinOccupationResult:
    """Resolve fixed-magnetization or shared-Fermi-level spin occupations."""

    up_spectra, integration_weights, up_bands = _validated_spectrum(
        eigenvalues[0],
        weights,
    )
    down_spectra, down_weights, down_bands = _validated_spectrum(
        eigenvalues[1],
        weights,
    )
    if up_bands != down_bands or not np.array_equal(integration_weights, down_weights):
        raise ValueError("spin channels must share k-point weights and band capacity")
    total = float(electron_count)
    if not np.isfinite(total) or total <= 0.0 or total > 2.0 * up_bands:
        raise ValueError("spin electron count must fit the two-channel band capacity")

    width = None if smearing_width_hartree is None else float(smearing_width_hartree)
    if width is not None and (not np.isfinite(width) or width <= 0.0):
        raise ValueError("smearing_width_hartree must be finite and positive")

    if magnetization is None:
        if width is None:
            raise ValueError("unconstrained spin occupations require Fermi-Dirac smearing")
        combined = tuple(
            np.concatenate((up, down))
            for up, down in zip(up_spectra, down_spectra, strict=True)
        )
        probabilities, observed, chemical_potential = _resolve_fermi_channel(
            combined,
            integration_weights,
            electron_count=total,
            width_hartree=width,
        )
        up_probabilities = tuple(values[:up_bands] for values in probabilities)
        down_probabilities = tuple(values[up_bands:] for values in probabilities)
        up_count = sum(
            float(weight) * float(np.sum(values))
            for weight, values in zip(
                integration_weights,
                up_probabilities,
                strict=True,
            )
        )
        down_count = observed - up_count
        return _PeriodicSpinOccupationResult(
            occupations=(
                tuple(tuple(float(value) for value in row) for row in up_probabilities),
                tuple(tuple(float(value) for value in row) for row in down_probabilities),
            ),
            electron_counts=(up_count, down_count),
            chemical_potentials=(chemical_potential, chemical_potential),
            shared_chemical_potential=chemical_potential,
            electronic_entropy=_spin_entropy(probabilities, integration_weights),
        )

    moment = float(magnetization)
    if not np.isfinite(moment) or abs(moment) > total:
        raise ValueError("fixed magnetization must be finite and lie within electron count")
    targets = (0.5 * (total + moment), 0.5 * (total - moment))
    if any(target < 0.0 or target > up_bands for target in targets):
        raise ValueError("fixed magnetization exceeds a spin channel band capacity")
    if width is None:
        occupations = (
            _fixed_integer_occupations(up_spectra, targets[0]),
            _fixed_integer_occupations(down_spectra, targets[1]),
        )
        return _PeriodicSpinOccupationResult(
            occupations=occupations,
            electron_counts=targets,
            chemical_potentials=(None, None),
            shared_chemical_potential=None,
            electronic_entropy=0.0,
        )

    channel_results = tuple(
        _resolve_fermi_channel(
            spectra,
            integration_weights,
            electron_count=target,
            width_hartree=width,
        )
        for spectra, target in zip((up_spectra, down_spectra), targets, strict=True)
    )
    probabilities_by_channel = tuple(result[0] for result in channel_results)
    return _PeriodicSpinOccupationResult(
        occupations=tuple(
            tuple(tuple(float(value) for value in row) for row in channel)
            for channel in probabilities_by_channel
        ),
        electron_counts=tuple(result[1] for result in channel_results),
        chemical_potentials=tuple(result[2] for result in channel_results),
        shared_chemical_potential=None,
        electronic_entropy=sum(
            _spin_entropy(channel, integration_weights)
            for channel in probabilities_by_channel
        ),
    )
