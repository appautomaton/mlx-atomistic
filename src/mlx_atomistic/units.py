"""Internal unit systems for atomistic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class LennardJonesReducedUnits:
    """Lennard-Jones reduced-unit system for v1 molecular dynamics.

    The numerical MD kernels operate on dimensionless values by default:
    sigma = 1, epsilon = 1, particle mass = 1, and k_B = 1.
    """

    sigma: float = 1.0
    epsilon: float = 1.0
    mass: float = 1.0
    boltzmann: float = 1.0

    @property
    def length(self) -> float:
        """Length unit."""

        return self.sigma

    @property
    def energy(self) -> float:
        """Energy unit."""

        return self.epsilon

    @property
    def time(self) -> float:
        """Time unit tau = sigma * sqrt(mass / epsilon)."""

        return self.sigma * sqrt(self.mass / self.epsilon)

    @property
    def force(self) -> float:
        """Force unit."""

        return self.epsilon / self.sigma

    @property
    def velocity(self) -> float:
        """Velocity unit."""

        return self.length / self.time

    @property
    def temperature(self) -> float:
        """Temperature unit."""

        return self.epsilon / self.boltzmann

    def to_reduced_length(self, value: float) -> float:
        """Convert a length into reduced units."""

        return value / self.length

    def from_reduced_length(self, value: float) -> float:
        """Convert a reduced length into the represented unit system."""

        return value * self.length

    def to_reduced_energy(self, value: float) -> float:
        """Convert an energy into reduced units."""

        return value / self.energy

    def from_reduced_energy(self, value: float) -> float:
        """Convert a reduced energy into the represented unit system."""

        return value * self.energy

    def to_reduced_temperature(self, value: float) -> float:
        """Convert a temperature into reduced units."""

        return value / self.temperature

    def from_reduced_temperature(self, value: float) -> float:
        """Convert a reduced temperature into the represented unit system."""

        return value * self.temperature


LJ_REDUCED_UNITS = LennardJonesReducedUnits()

