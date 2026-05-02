"""Internal unit systems for atomistic simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

SUPPORTED_REAL_UNITS = {
    "coordinates": {"angstrom", "nanometer"},
    "length": {"angstrom", "nanometer"},
    "mass": {"dalton", "atomic_mass", "amu"},
    "charge": {"elementary_charge"},
    "energy": {"kilojoule_per_mole", "kJ/mol"},
    "time": {"picosecond", "ps"},
    "temperature": {"kelvin", "K"},
}

COULOMB_CONSTANT_KJ_MOL_NM = 138.93545764438198
COULOMB_CONSTANT_KJ_MOL_ANGSTROM = COULOMB_CONSTANT_KJ_MOL_NM / 10.0
BOLTZMANN_CONSTANT_KJ_MOL_K = 0.00831446261815324
DALTON_ANGSTROM2_PER_PS2_TO_KJ_PER_MOL = 0.01


@dataclass(frozen=True)
class MDUnitSystem:
    """Explicit physical units used by production molecular mechanics kernels.

    The recommended MLX production convention is Angstrom, ps, dalton,
    elementary charge, kJ/mol, and K.  The Coulomb constant below matches those
    units when distances are stored in Angstrom.
    """

    coordinates: str = "angstrom"
    mass: str = "dalton"
    charge: str = "elementary_charge"
    energy: str = "kilojoule_per_mole"
    time: str = "picosecond"
    temperature: str = "kelvin"
    coulomb_constant: float = COULOMB_CONSTANT_KJ_MOL_ANGSTROM
    boltzmann_constant: float = BOLTZMANN_CONSTANT_KJ_MOL_K

    @classmethod
    def from_metadata(cls, units: Mapping[str, str]) -> MDUnitSystem:
        """Create and validate a physical MD unit system from artifact metadata."""

        coordinates = str(units.get("coordinates") or units.get("length") or "")
        system = cls(
            coordinates=coordinates,
            mass=str(units.get("mass", "")),
            charge=str(units.get("charge", "")),
            energy=str(units.get("energy", "")),
            time=str(units.get("time", "")),
            temperature=str(units.get("temperature", "")),
            coulomb_constant=float(
                units.get("coulomb_constant", COULOMB_CONSTANT_KJ_MOL_ANGSTROM)
            ),
        )
        system.validate()
        return system

    def validate(self) -> None:
        """Reject reduced or ambiguous units for production artifacts."""

        values = {
            "coordinates": self.coordinates,
            "mass": self.mass,
            "charge": self.charge,
            "energy": self.energy,
            "time": self.time,
            "temperature": self.temperature,
        }
        for key, value in values.items():
            if not value:
                msg = f"production artifact units must define {key}"
                raise ValueError(msg)
            if "reduced" in value:
                msg = f"production artifact unit {key}={value!r} is reduced, not physical"
                raise ValueError(msg)
            if value not in SUPPORTED_REAL_UNITS[key]:
                msg = f"unsupported production artifact unit {key}={value!r}"
                raise ValueError(msg)
        if self.coulomb_constant <= 0.0:
            msg = "coulomb_constant must be positive"
            raise ValueError(msg)

    @property
    def coordinate_scale_to_angstrom(self) -> float:
        """Scale stored coordinates into Angstrom."""

        if self.coordinates == "angstrom":
            return 1.0
        if self.coordinates == "nanometer":
            return 10.0
        msg = f"unsupported coordinate unit {self.coordinates!r}"
        raise ValueError(msg)

    @property
    def kinetic_energy_scale(self) -> float:
        """Scale `0.5 * dalton * velocity**2` into kJ/mol."""

        if self.coordinates == "angstrom" and self.time in {"picosecond", "ps"}:
            return DALTON_ANGSTROM2_PER_PS2_TO_KJ_PER_MOL
        if self.coordinates == "nanometer" and self.time in {"picosecond", "ps"}:
            return 1.0
        msg = (
            "kinetic energy scaling is only defined for Angstrom/ps or "
            "nanometer/ps velocities"
        )
        raise ValueError(msg)

    @property
    def force_to_acceleration_scale(self) -> float:
        """Scale `(kJ/mol/length) / dalton` into stored-length/ps**2."""

        if self.coordinates == "angstrom" and self.time in {"picosecond", "ps"}:
            return 100.0
        if self.coordinates == "nanometer" and self.time in {"picosecond", "ps"}:
            return 1.0
        msg = (
            "force-to-acceleration scaling is only defined for Angstrom/ps or "
            "nanometer/ps coordinates"
        )
        raise ValueError(msg)


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
PRODUCTION_MD_UNITS = MDUnitSystem()

__all__ = [
    "BOLTZMANN_CONSTANT_KJ_MOL_K",
    "COULOMB_CONSTANT_KJ_MOL_ANGSTROM",
    "COULOMB_CONSTANT_KJ_MOL_NM",
    "DALTON_ANGSTROM2_PER_PS2_TO_KJ_PER_MOL",
    "LJ_REDUCED_UNITS",
    "LennardJonesReducedUnits",
    "MDUnitSystem",
    "PRODUCTION_MD_UNITS",
]
