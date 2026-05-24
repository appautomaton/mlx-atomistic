"""Molecular mechanics systems and force-field parameter assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import (
    HarmonicAnglePotential,
    HarmonicBondPotential,
    ImproperDihedralPotential,
    NonbondedPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.topology import Topology


def _string_tuple(value: Sequence[str], *, count: int, name: str) -> tuple[str, ...]:
    if len(value) != count:
        msg = f"{name} must contain {count} entries"
        raise ValueError(msg)
    return tuple(str(item) for item in value)


def _type_key(types: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item) for item in types)


def _symmetric_key(types: Sequence[str]) -> tuple[str, ...]:
    forward = _type_key(types)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


@dataclass(frozen=True)
class MMSystem:
    """A typed molecular mechanics system."""

    symbols: Sequence[str]
    atom_names: Sequence[str]
    atom_types: Sequence[str]
    masses: object
    charges: object | None
    positions: object
    topology: Topology
    velocities: object | None = None
    cell: Cell | None = None
    virtual_sites: object | None = None

    @classmethod
    def from_sequences(
        cls,
        *,
        symbols: Sequence[str],
        positions: Sequence[Sequence[float]],
        topology: Topology,
        atom_types: Sequence[str],
        atom_names: Sequence[str] | None = None,
        masses: Sequence[float] | None = None,
        charges: Sequence[float] | None = None,
        velocities: Sequence[Sequence[float]] | None = None,
        cell: Cell | None = None,
        atom_type_masses: Mapping[str, float] | None = None,
        virtual_sites: object | None = None,
    ) -> MMSystem:
        """Create an MM system from Python data."""

        atom_count = len(symbols)
        if atom_names is None:
            atom_names = [f"{symbol}{index + 1}" for index, symbol in enumerate(symbols)]
        if masses is None:
            if atom_type_masses is None:
                msg = "masses are required when atom_type_masses is not provided"
                raise ValueError(msg)
            try:
                masses = [float(atom_type_masses[str(atom_type)]) for atom_type in atom_types]
            except KeyError as err:
                msg = f"missing mass for atom type {err.args[0]!r}"
                raise ValueError(msg) from err
        if charges is None:
            charges = [0.0] * atom_count
        if velocities is None:
            velocities = np.zeros((atom_count, 3), dtype=np.float32)
        return cls(
            symbols=symbols,
            atom_names=atom_names,
            atom_types=atom_types,
            masses=masses,
            charges=charges,
            positions=positions,
            topology=topology,
            velocities=velocities,
            cell=cell,
            virtual_sites=virtual_sites,
        )

    def __post_init__(self) -> None:
        atom_count = len(self.symbols)
        if atom_count <= 0:
            msg = "symbols must contain at least one atom"
            raise ValueError(msg)
        if self.topology.n_atoms != atom_count:
            msg = "topology.n_atoms must match the number of symbols"
            raise ValueError(msg)

        positions = as_mx_array(self.positions)
        if positions.shape != (atom_count, 3):
            msg = "positions must have shape (n_atoms, 3)"
            raise ValueError(msg)

        if self.masses is None:
            msg = "masses are required"
            raise ValueError(msg)
        masses = as_mx_array(self.masses)
        if masses.shape != (atom_count,):
            msg = "masses must have shape (n_atoms,)"
            raise ValueError(msg)
        if bool(np.any(np.asarray(masses) <= 0.0)):
            msg = "masses must be positive"
            raise ValueError(msg)

        charges = (
            as_mx_array([0.0] * atom_count)
            if self.charges is None
            else as_mx_array(self.charges)
        )
        if charges.shape != (atom_count,):
            msg = "charges must have shape (n_atoms,)"
            raise ValueError(msg)

        velocities = (
            mx.zeros_like(positions) if self.velocities is None else as_mx_array(self.velocities)
        )
        if velocities.shape != (atom_count, 3):
            msg = "velocities must have shape (n_atoms, 3)"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "symbols",
            _string_tuple(self.symbols, count=atom_count, name="symbols"),
        )
        object.__setattr__(
            self,
            "atom_names",
            _string_tuple(self.atom_names, count=atom_count, name="atom_names"),
        )
        object.__setattr__(
            self,
            "atom_types",
            _string_tuple(self.atom_types, count=atom_count, name="atom_types"),
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "charges", charges)

    @property
    def atom_count(self) -> int:
        """Number of atoms."""

        return len(self.symbols)


@dataclass(frozen=True)
class AtomType:
    """Atom type definition."""

    name: str
    mass: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            msg = "atom type mass must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class NonbondedParameter:
    """Per-type nonbonded parameters."""

    atom_type: str
    sigma: float
    epsilon: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        if self.epsilon < 0.0:
            msg = "epsilon must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class BondParameter:
    """Harmonic bond parameter for an unordered atom-type pair."""

    atom_types: tuple[str, str]
    k: float
    length: float

    def __post_init__(self) -> None:
        if len(self.atom_types) != 2:
            msg = "bond atom_types must contain two entries"
            raise ValueError(msg)
        if self.k < 0.0 or self.length <= 0.0:
            msg = "bond k must be non-negative and length must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class AngleParameter:
    """Harmonic angle parameter for a reversible atom-type triplet."""

    atom_types: tuple[str, str, str]
    k: float
    angle: float

    def __post_init__(self) -> None:
        if len(self.atom_types) != 3:
            msg = "angle atom_types must contain three entries"
            raise ValueError(msg)
        if self.k < 0.0 or self.angle <= 0.0:
            msg = "angle k must be non-negative and angle must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class DihedralParameter:
    """Periodic dihedral parameter for a reversible atom-type quartet."""

    atom_types: tuple[str, str, str, str]
    k: float
    periodicity: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if len(self.atom_types) != 4:
            msg = "dihedral atom_types must contain four entries"
            raise ValueError(msg)
        if self.periodicity <= 0.0:
            msg = "periodicity must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class ImproperParameter(DihedralParameter):
    """Periodic improper torsion parameter for a reversible atom-type quartet."""


@dataclass(frozen=True)
class ForceField:
    """Small programmatic force-field parameter set."""

    atom_types: Sequence[AtomType]
    nonbonded: Sequence[NonbondedParameter]
    bonds: Sequence[BondParameter] = ()
    angles: Sequence[AngleParameter] = ()
    dihedrals: Sequence[DihedralParameter] = ()
    impropers: Sequence[ImproperParameter] = ()
    lj_one_four_scale: float = 1.0
    coulomb_one_four_scale: float = 1.0
    cutoff: float | None = 2.5
    lj_shift: bool = True
    coulomb_shift: bool = False
    switch_distance: float | None = None
    coulomb_constant: float = 1.0

    def __post_init__(self) -> None:
        if self.lj_one_four_scale < 0.0 or self.coulomb_one_four_scale < 0.0:
            msg = "1-4 scaling factors must be non-negative"
            raise ValueError(msg)
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.switch_distance is not None:
            if self.cutoff is None:
                msg = "switch_distance requires a cutoff"
                raise ValueError(msg)
            if self.switch_distance < 0.0 or self.switch_distance >= self.cutoff:
                msg = "switch_distance must be non-negative and smaller than cutoff"
                raise ValueError(msg)
        object.__setattr__(self, "atom_types", tuple(self.atom_types))
        object.__setattr__(self, "nonbonded", tuple(self.nonbonded))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        object.__setattr__(self, "angles", tuple(self.angles))
        object.__setattr__(self, "dihedrals", tuple(self.dihedrals))
        object.__setattr__(self, "impropers", tuple(self.impropers))

    @property
    def atom_type_masses(self) -> dict[str, float]:
        """Atom-type masses by type name."""

        return {parameter.name: float(parameter.mass) for parameter in self.atom_types}

    def _nonbonded_by_type(self) -> dict[str, NonbondedParameter]:
        return {parameter.atom_type: parameter for parameter in self.nonbonded}

    def masses_for(self, atom_types: Sequence[str]) -> tuple[float, ...]:
        """Return masses for atom type names."""

        masses = self.atom_type_masses
        try:
            return tuple(float(masses[str(atom_type)]) for atom_type in atom_types)
        except KeyError as err:
            msg = f"missing atom type mass for {err.args[0]!r}"
            raise ValueError(msg) from err

    def build_force_terms(self, system: MMSystem) -> list:
        """Build force terms for a typed molecular mechanics system."""

        nonbonded_by_type = self._nonbonded_by_type()
        sigmas: list[float] = []
        epsilons: list[float] = []
        for atom_index, atom_type in enumerate(system.atom_types):
            parameter = nonbonded_by_type.get(atom_type)
            if parameter is None:
                msg = f"missing nonbonded parameter for atom {atom_index} type {atom_type!r}"
                raise ValueError(msg)
            sigmas.append(float(parameter.sigma))
            epsilons.append(float(parameter.epsilon))

        terms = []
        bond_terms = self._bond_terms(system)
        if bond_terms is not None:
            terms.append(bond_terms)
        angle_terms = self._angle_terms(system)
        if angle_terms is not None:
            terms.append(angle_terms)
        dihedral_terms = self._dihedral_terms(system)
        if dihedral_terms is not None:
            terms.append(dihedral_terms)
        improper_terms = self._improper_terms(system)
        if improper_terms is not None:
            terms.append(improper_terms)
        terms.append(
            NonbondedPotential(
                sigma=sigmas,
                epsilon=epsilons,
                charges=system.charges,
                topology=system.topology,
                cutoff=self.cutoff,
                lj_shift=self.lj_shift,
                coulomb_shift=self.coulomb_shift,
                switch_distance=self.switch_distance,
                lj_one_four_scale=self.lj_one_four_scale,
                coulomb_one_four_scale=self.coulomb_one_four_scale,
                coulomb_constant=self.coulomb_constant,
            )
        )
        return terms

    def _bond_terms(self, system: MMSystem):
        bonds = np.asarray(system.topology.bonds, dtype=np.int32)
        if bonds.size == 0:
            return None
        parameters = {_symmetric_key(parameter.atom_types): parameter for parameter in self.bonds}
        k_values: list[float] = []
        lengths: list[float] = []
        for atom_i, atom_j in bonds.tolist():
            atom_types = (system.atom_types[atom_i], system.atom_types[atom_j])
            parameter = parameters.get(_symmetric_key(atom_types))
            if parameter is None:
                msg = f"missing bond parameter for bond ({atom_i}, {atom_j}) types {atom_types}"
                raise ValueError(msg)
            k_values.append(float(parameter.k))
            lengths.append(float(parameter.length))
        return HarmonicBondPotential(bonds, k=k_values, length=lengths)

    def _angle_terms(self, system: MMSystem):
        angles = np.asarray(system.topology.angles, dtype=np.int32)
        if angles.size == 0:
            return None
        parameters = {_symmetric_key(parameter.atom_types): parameter for parameter in self.angles}
        k_values: list[float] = []
        angle_values: list[float] = []
        for atom_i, atom_j, atom_k in angles.tolist():
            atom_types = (
                system.atom_types[atom_i],
                system.atom_types[atom_j],
                system.atom_types[atom_k],
            )
            parameter = parameters.get(_symmetric_key(atom_types))
            if parameter is None:
                msg = (
                    f"missing angle parameter for angle ({atom_i}, {atom_j}, {atom_k}) "
                    f"types {atom_types}"
                )
                raise ValueError(msg)
            k_values.append(float(parameter.k))
            angle_values.append(float(parameter.angle))
        return HarmonicAnglePotential(angles, k=k_values, angle=angle_values)

    def _dihedral_terms(self, system: MMSystem):
        dihedrals = np.asarray(system.topology.dihedrals, dtype=np.int32)
        if dihedrals.size == 0:
            return None
        parameters = {
            _symmetric_key(parameter.atom_types): parameter for parameter in self.dihedrals
        }
        k_values: list[float] = []
        periodicities: list[float] = []
        phases: list[float] = []
        for atom_i, atom_j, atom_k, atom_m in dihedrals.tolist():
            atom_types = (
                system.atom_types[atom_i],
                system.atom_types[atom_j],
                system.atom_types[atom_k],
                system.atom_types[atom_m],
            )
            parameter = parameters.get(_symmetric_key(atom_types))
            if parameter is None:
                msg = (
                    "missing dihedral parameter for dihedral "
                    f"({atom_i}, {atom_j}, {atom_k}, {atom_m}) types {atom_types}"
                )
                raise ValueError(msg)
            k_values.append(float(parameter.k))
            periodicities.append(float(parameter.periodicity))
            phases.append(float(parameter.phase))
        return PeriodicDihedralPotential(
            dihedrals,
            k=k_values,
            periodicity=periodicities,
            phase=phases,
        )

    def _improper_terms(self, system: MMSystem):
        impropers = np.asarray(system.topology.impropers, dtype=np.int32)
        if impropers.size == 0:
            return None
        parameters = {
            _symmetric_key(parameter.atom_types): parameter for parameter in self.impropers
        }
        k_values: list[float] = []
        periodicities: list[float] = []
        phases: list[float] = []
        for atom_i, atom_j, atom_k, atom_m in impropers.tolist():
            atom_types = (
                system.atom_types[atom_i],
                system.atom_types[atom_j],
                system.atom_types[atom_k],
                system.atom_types[atom_m],
            )
            parameter = parameters.get(_symmetric_key(atom_types))
            if parameter is None:
                msg = (
                    "missing improper parameter for improper "
                    f"({atom_i}, {atom_j}, {atom_k}, {atom_m}) types {atom_types}"
                )
                raise ValueError(msg)
            k_values.append(float(parameter.k))
            periodicities.append(float(parameter.periodicity))
            phases.append(float(parameter.phase))
        return ImproperDihedralPotential(
            impropers,
            k=k_values,
            periodicity=periodicities,
            phase=phases,
        )


__all__ = [
    "AngleParameter",
    "AtomType",
    "BondParameter",
    "DihedralParameter",
    "ForceField",
    "ImproperParameter",
    "MMSystem",
    "NonbondedParameter",
]
