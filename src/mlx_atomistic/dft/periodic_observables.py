"""Electronic observables derived from converged periodic SCF states."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._periodic_models import (
    PeriodicKPointResult,
    PeriodicSCFResult,
)


@dataclass(frozen=True)
class PeriodicDOSChannel:
    """One total or physical-spin density-of-states channel."""

    label: str
    state_density: mx.array
    occupied_density: mx.array
    expected_state_count: float
    integrated_state_count: float
    expected_electron_count: float
    integrated_electron_count: float
    fermi_level_hartree: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe channel summary without sampled arrays."""

        return {
            "label": self.label,
            "expected_state_count": self.expected_state_count,
            "integrated_state_count": self.integrated_state_count,
            "expected_electron_count": self.expected_electron_count,
            "integrated_electron_count": self.integrated_electron_count,
            "fermi_level_hartree": self.fermi_level_hartree,
        }


@dataclass(frozen=True)
class PeriodicDOSResult:
    """Total and occupation-weighted periodic density of states."""

    energies_hartree: mx.array
    state_density_per_hartree: mx.array
    occupied_density_per_hartree: mx.array
    channels: tuple[PeriodicDOSChannel, ...]
    broadening_hartree: float
    energy_window_hartree: tuple[float, float]
    fermi_level_hartree: float | None
    fermi_level_convention: str
    expected_state_count: float
    integrated_state_count: float
    expected_electron_count: float
    integrated_electron_count: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe DOS summary without sampled arrays."""

        return {
            "energy_point_count": int(self.energies_hartree.size),
            "energy_window_hartree": list(self.energy_window_hartree),
            "broadening_hartree": self.broadening_hartree,
            "fermi_level_hartree": self.fermi_level_hartree,
            "fermi_level_convention": self.fermi_level_convention,
            "expected_state_count": self.expected_state_count,
            "integrated_state_count": self.integrated_state_count,
            "expected_electron_count": self.expected_electron_count,
            "integrated_electron_count": self.integrated_electron_count,
            "channels": [channel.to_dict() for channel in self.channels],
        }


@dataclass(frozen=True)
class _DOSInputChannel:
    label: str
    kpoints: tuple[PeriodicKPointResult, ...]
    degeneracy: float
    electron_count: float
    chemical_potential: float | None


def _input_channels(source: PeriodicSCFResult) -> tuple[_DOSInputChannel, ...]:
    if source.spin_channels:
        if len(source.spin_channels) != 2:
            raise ValueError("collinear DOS requires exactly two spin channels")
        labels = tuple(channel.label for channel in source.spin_channels)
        if len(set(labels)) != len(labels) or any(not label for label in labels):
            raise ValueError("collinear DOS requires unique non-empty channel labels")
        channel_count = sum(channel.electron_count for channel in source.spin_channels)
        if not np.isclose(channel_count, source.electron_count, atol=2.0e-6, rtol=0.0):
            raise ValueError("spin-channel electron counts differ from SCF total")
        chemical_potentials = tuple(
            channel.chemical_potential for channel in source.spin_channels
        )
        if any(value is None for value in chemical_potentials) and any(
            value is not None for value in chemical_potentials
        ):
            raise ValueError("spin-channel chemical potentials are incomplete")
        return tuple(
            _DOSInputChannel(
                label=channel.label,
                kpoints=channel.kpoints,
                degeneracy=1.0,
                electron_count=channel.electron_count,
                chemical_potential=channel.chemical_potential,
            )
            for channel in source.spin_channels
        )
    return (
        _DOSInputChannel(
            label="total",
            kpoints=source.kpoints,
            degeneracy=2.0,
            electron_count=source.electron_count,
            chemical_potential=source.chemical_potential,
        ),
    )


def _flatten_channel(
    channel: _DOSInputChannel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not channel.kpoints:
        raise ValueError("periodic DOS requires at least one k-point per channel")
    weights = np.asarray([point.weight for point in channel.kpoints], dtype=np.float64)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("periodic DOS k-point weights must be positive and sum to one")
    energies = []
    state_weights = []
    occupied_weights = []
    band_count: int | None = None
    highest_occupied = -np.inf
    for point, point_weight in zip(channel.kpoints, weights, strict=True):
        spectrum = np.asarray(point.eigen.eigenvalues, dtype=np.float64).reshape(-1)
        if point.occupations is None:
            raise ValueError("periodic DOS requires resolved occupations")
        occupations = np.asarray(point.occupations, dtype=np.float64).reshape(-1)
        if (
            spectrum.size == 0
            or occupations.shape != spectrum.shape
            or not np.all(np.isfinite(spectrum))
            or not np.all(np.isfinite(occupations))
            or np.any(occupations < 0.0)
            or np.any(occupations > channel.degeneracy)
        ):
            raise ValueError("periodic DOS spectra and occupations are invalid")
        if band_count is None:
            band_count = int(spectrum.size)
        elif spectrum.size != band_count:
            raise ValueError("periodic DOS channels require a common band count")
        occupied = occupations > 1.0e-10
        if np.any(occupied):
            highest_occupied = max(highest_occupied, float(np.max(spectrum[occupied])))
        energies.extend(spectrum)
        state_weights.extend(np.full(spectrum.shape, point_weight * channel.degeneracy))
        occupied_weights.extend(point_weight * occupations)
    if not np.isfinite(highest_occupied):
        raise ValueError("periodic DOS channel has no occupied state")
    expected_electrons = float(np.sum(occupied_weights, dtype=np.float64))
    if not np.isclose(
        expected_electrons,
        channel.electron_count,
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("periodic DOS occupations differ from channel electron count")
    return (
        np.asarray(energies, dtype=np.float32),
        np.asarray(state_weights, dtype=np.float32),
        np.asarray(occupied_weights, dtype=np.float32),
        highest_occupied,
    )


def _integral(values: mx.array, spacing: float) -> float:
    integral = spacing * (
        0.5 * values[0] + mx.sum(values[1:-1]) + 0.5 * values[-1]
    )
    mx.eval(integral)
    return float(integral)


def periodic_density_of_states(
    source: PeriodicSCFResult,
    *,
    broadening_hartree: float = 0.01,
    energy_points: int = 2001,
    energy_window_hartree: tuple[float, float] | None = None,
) -> PeriodicDOSResult:
    """Build total and occupation-weighted DOS from a converged periodic SCF.

    Args:
        source: Converged scalar or collinear-spin periodic SCF result.
        broadening_hartree: Positive Gaussian standard deviation in Hartree.
        energy_points: Number of uniformly spaced output energies.
        energy_window_hartree: Optional explicit lower and upper energy bounds.
            The default spans every eigenvalue plus eight Gaussian widths.

    Returns:
        Shared-grid total and channel-resolved density of states.

    Raises:
        TypeError: If ``source`` is not a periodic SCF result.
        ValueError: If source state, broadening, grid, spectra, or occupations
            are invalid.
    """

    if not isinstance(source, PeriodicSCFResult):
        raise TypeError("source must be PeriodicSCFResult")
    if not source.converged:
        raise ValueError("periodic DOS requires a converged SCF result")
    if not np.isfinite(broadening_hartree) or broadening_hartree <= 0.0:
        raise ValueError("broadening_hartree must be finite and positive")
    if type(energy_points) is not int or energy_points < 3:
        raise ValueError("energy_points must be a non-bool integer of at least three")
    inputs = _input_channels(source)
    flattened = tuple(_flatten_channel(channel) for channel in inputs)
    minimum = min(float(np.min(values[0])) for values in flattened)
    maximum = max(float(np.max(values[0])) for values in flattened)
    if energy_window_hartree is None:
        lower = minimum - 8.0 * broadening_hartree
        upper = maximum + 8.0 * broadening_hartree
    else:
        if len(energy_window_hartree) != 2:
            raise ValueError("energy_window_hartree must contain two bounds")
        lower, upper = (float(value) for value in energy_window_hartree)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("DOS energy window must be finite and increasing")
    energies = mx.linspace(lower, upper, energy_points, dtype=mx.float32)
    spacing = (upper - lower) / (energy_points - 1)
    normalization = 1.0 / (broadening_hartree * sqrt(2.0 * pi))
    channels = []
    for channel, (eigenvalues, state_weights, occupied_weights, occupied_edge) in zip(
        inputs,
        flattened,
        strict=True,
    ):
        difference = (
            energies[:, None] - mx.array(eigenvalues, dtype=mx.float32)[None, :]
        ) / broadening_hartree
        kernels = normalization * mx.exp(-0.5 * difference * difference)
        state_density = kernels @ mx.array(state_weights)
        occupied_density = kernels @ mx.array(occupied_weights)
        mx.eval(state_density, occupied_density)
        expected_states = float(np.sum(state_weights, dtype=np.float64))
        expected_electrons = float(np.sum(occupied_weights, dtype=np.float64))
        channels.append(
            PeriodicDOSChannel(
                label=channel.label,
                state_density=state_density,
                occupied_density=occupied_density,
                expected_state_count=expected_states,
                integrated_state_count=_integral(state_density, spacing),
                expected_electron_count=expected_electrons,
                integrated_electron_count=_integral(occupied_density, spacing),
                fermi_level_hartree=(
                    occupied_edge
                    if channel.chemical_potential is None
                    else channel.chemical_potential
                ),
            )
        )
    total_state_density = mx.sum(
        mx.stack([channel.state_density for channel in channels]),
        axis=0,
    )
    total_occupied_density = mx.sum(
        mx.stack([channel.occupied_density for channel in channels]),
        axis=0,
    )
    mx.eval(total_state_density, total_occupied_density)
    channel_chemical_potentials = [
        channel.chemical_potential for channel in inputs if channel.chemical_potential is not None
    ]
    if source.chemical_potential is not None:
        fermi_level = source.chemical_potential
        convention = "shared_fermi_dirac_chemical_potential"
    elif channel_chemical_potentials:
        fermi_level = None
        convention = "fixed_spin_channel_chemical_potentials"
    elif source.spin_channels:
        fermi_level = None
        convention = "spin_channel_highest_occupied_computed_states"
    else:
        fermi_level = channels[0].fermi_level_hartree
        convention = "highest_occupied_computed_state"
    return PeriodicDOSResult(
        energies_hartree=energies,
        state_density_per_hartree=total_state_density,
        occupied_density_per_hartree=total_occupied_density,
        channels=tuple(channels),
        broadening_hartree=float(broadening_hartree),
        energy_window_hartree=(lower, upper),
        fermi_level_hartree=fermi_level,
        fermi_level_convention=convention,
        expected_state_count=sum(channel.expected_state_count for channel in channels),
        integrated_state_count=_integral(total_state_density, spacing),
        expected_electron_count=sum(
            channel.expected_electron_count for channel in channels
        ),
        integrated_electron_count=_integral(total_occupied_density, spacing),
    )
