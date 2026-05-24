"""Native GROMACS topology/coordinate import for the supported subset."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)
from mlx_atomistic.prep.topology_import import (
    TopologyImportError,
    _hydrogen_bond_constraints,
    _infer_symbol,
    _ion_mask_from_residues,
    _ligand_mask_from_residues,
    _lipid_mask_from_residues,
    _normalize_pair,
    _term_counts,
    _water_mask_from_residues,
)

NM_TO_ANGSTROM = 10.0
GROMACS_SUPPORTED_SECTIONS = frozenset(
    {
        "defaults",
        "atomtypes",
        "moleculetype",
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "pairs",
        "exclusions",
        "system",
        "molecules",
    }
)
GROMACS_VIRTUAL_SITE_MARKERS = frozenset(
    {"DUM", "DUMMY", "EP", "LP", "LP1", "LP2", "M", "MW", "VS"}
)


@dataclass(frozen=True)
class GromacsDefaults:
    nbfunc: int
    combination_rule: int
    gen_pairs: bool
    fudge_lj: float
    fudge_qq: float


@dataclass(frozen=True)
class GromacsAtomType:
    name: str
    mass: float
    charge: float
    sigma: float
    epsilon: float


@dataclass(frozen=True)
class GromacsAtom:
    index: int
    atom_type: str
    residue_id: int
    residue_name: str
    atom_name: str
    charge_group: int
    charge: float
    mass: float


@dataclass(frozen=True)
class GromacsBond:
    i: int
    j: int
    k: float
    length: float


@dataclass(frozen=True)
class GromacsAngle:
    i: int
    j: int
    k: int
    force_constant: float
    theta: float


@dataclass(frozen=True)
class GromacsPeriodicDihedral:
    i: int
    j: int
    k: int
    m: int
    force_constant: float
    periodicity: float
    phase: float


@dataclass(frozen=True)
class GromacsRBDihedral:
    i: int
    j: int
    k: int
    m: int
    coefficients: tuple[float, float, float, float, float, float]


@dataclass
class GromacsMoleculeType:
    name: str
    nrexcl: int
    atoms: list[GromacsAtom] = field(default_factory=list)
    bonds: list[GromacsBond] = field(default_factory=list)
    angles: list[GromacsAngle] = field(default_factory=list)
    dihedrals: list[GromacsPeriodicDihedral] = field(default_factory=list)
    rb_dihedrals: list[GromacsRBDihedral] = field(default_factory=list)
    pairs: set[tuple[int, int]] = field(default_factory=set)
    exclusions: set[tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class GromacsTopology:
    defaults: GromacsDefaults
    atomtypes: Mapping[str, GromacsAtomType]
    molecule_types: Mapping[str, GromacsMoleculeType]
    molecules: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GromacsSection:
    name: str
    lines: tuple[tuple[int, str], ...]


def import_gromacs_top_gro(
    *,
    top_path: str | Path,
    gro_path: str | Path,
) -> PreparedSystem:
    """Import a supported GROMACS `.top` plus `.gro` coordinate pair."""

    top_path = Path(top_path)
    gro_path = Path(gro_path)
    topology = _read_gromacs_topology(top_path)
    positions, velocities, cell_lengths, cell_matrix = _read_gromacs_gro(gro_path)
    expanded = _expand_gromacs_topology(topology)
    atom_count = int(expanded["positions_atom_count"])
    if positions.shape != (atom_count, 3):
        msg = (
            "coordinate atom count does not match GROMACS topology: "
            f"gro={positions.shape[0]}, topology={atom_count}"
        )
        raise TopologyImportError(msg)
    if velocities.size == 0:
        velocities = np.zeros_like(positions, dtype=np.float32)

    symbols = np.asarray(
        [
            _infer_symbol(atom_name, atom_type)
            for atom_name, atom_type in zip(
                expanded["atom_names"],
                expanded["atom_types"],
                strict=True,
            )
        ],
        dtype=str,
    )
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols.astype(str)) == "H"))
    if hydrogen_count == 0:
        msg = "production GROMACS import found no hydrogens"
        raise TopologyImportError(msg)

    residue_names = np.asarray(expanded["residue_names"], dtype=str)
    ligand_mask = _ligand_mask_from_residues(residue_names)
    water_mask = _water_mask_from_residues(residue_names)
    ion_mask = _ion_mask_from_residues(residue_names)
    lipid_mask = _lipid_mask_from_residues(residue_names)
    receptor_mask = ~(ligand_mask | water_mask | ion_mask | lipid_mask)
    constraints, constraint_distance = _hydrogen_bond_constraints(
        expanded["bonds"],
        symbols=symbols,
        positions=positions,
    )

    supported_terms = ["nonbonded_lj_coulomb"]
    if expanded["bonds"].shape[0]:
        supported_terms.append("harmonic_bond")
    if expanded["angles"].shape[0]:
        supported_terms.append("harmonic_angle")
    if expanded["dihedrals"].shape[0]:
        supported_terms.append("periodic_dihedral")
    if expanded["rb_dihedrals"].shape[0]:
        supported_terms.append("rb_dihedral")
    if expanded["nonbonded_exception_pairs"].shape[0]:
        supported_terms.append("nonbonded_exception")
    if constraints.shape[0]:
        supported_terms.append("distance_constraint")

    term_counts = _term_counts(
        bonds=expanded["bonds"],
        angles=expanded["angles"],
        dihedrals=expanded["dihedrals"],
        impropers=empty_indices(4),
        constraints=constraints,
        nonbonded_exception_pairs=expanded["nonbonded_exception_pairs"],
    )
    term_counts.update(
        {
            "rb_dihedrals": int(expanded["rb_dihedrals"].shape[0]),
            "gromacs_molecule_types": len(topology.molecule_types),
            "gromacs_molecule_instances": int(expanded["molecule_instance_count"]),
            "gromacs_pair_exceptions": int(expanded["gromacs_pair_exception_count"]),
            "gromacs_exclusions": int(expanded["gromacs_exclusion_count"]),
        }
    )
    term_details = {
        "nonbonded_exception": {
            "source": "GROMACS nrexcl graph exclusions, [ exclusions ], and [ pairs ]",
            "fudge_lj": topology.defaults.fudge_lj,
            "fudge_qq": topology.defaults.fudge_qq,
            "gen_pairs": topology.defaults.gen_pairs,
        },
        "rb_dihedral": {
            "source": "GROMACS [ dihedrals ] function type 3",
            "angle_convention": "cos(phi - pi), direct coefficient mapping",
        },
    }

    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source={
            "kind": "gromacs",
            "parser": "native_gromacs_top_gro",
            "top_path": str(top_path),
            "gro_path": str(gro_path),
        },
        selections={
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
            "water_atom_count": int(np.count_nonzero(water_mask)),
            "ion_atom_count": int(np.count_nonzero(ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(lipid_mask)),
            "system_charge": float(np.sum(expanded["charges"])),
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="gromacs_top_gro_native",
        compatibility_report={
            "engine": "mlx_atomistic",
            "parser": "native_gromacs_top_gro",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "water_present": bool(np.any(water_mask)),
            "ions_present": bool(np.any(ion_mask)),
            "lipids_present": bool(np.any(lipid_mask)),
            "periodic_box_present": bool(cell_lengths.shape == (3,)),
            "supported_terms": supported_terms,
            "required_terms": list(supported_terms),
            "unsupported_terms": [],
            "rejected_terms": [],
            "rejection_reasons": {},
            "rejected_term_details": {},
            "term_details": term_details,
            "parameter_counts_match_topology": True,
            "term_counts": term_counts,
            "force_field_provenance": "native GROMACS .top/.gro import",
        },
        warnings=[
            "Imported with the native GROMACS parser for the declared supported subset. "
            "Preprocessing and unsupported directives are rejected.",
        ],
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=symbols,
        atom_names=np.asarray(expanded["atom_names"], dtype=str),
        atom_types=np.asarray(expanded["atom_types"], dtype=str),
        residue_names=residue_names,
        residue_ids=np.asarray(expanded["residue_ids"], dtype=np.int32),
        chain_ids=np.asarray(expanded["chain_ids"], dtype=str),
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        masses=expanded["masses"],
        charges=expanded["charges"],
        sigma=expanded["sigma"],
        epsilon=expanded["epsilon"],
        bonds=expanded["bonds"],
        bond_k=expanded["bond_k"],
        bond_length=expanded["bond_length"],
        angles=expanded["angles"],
        angle_k=expanded["angle_k"],
        angle_theta=expanded["angle_theta"],
        dihedrals=expanded["dihedrals"],
        dihedral_k=expanded["dihedral_k"],
        dihedral_periodicity=expanded["dihedral_periodicity"],
        dihedral_phase=expanded["dihedral_phase"],
        nonbonded_pairs=empty_indices(2),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=receptor_mask & ~ligand_mask,
        reference_positions=positions.astype(np.float32).copy(),
        cell_lengths=cell_lengths.astype(np.float32),
        cell_matrix=cell_matrix.astype(np.float32),
        rb_dihedrals=expanded["rb_dihedrals"],
        rb_c0=expanded["rb_c0"],
        rb_c1=expanded["rb_c1"],
        rb_c2=expanded["rb_c2"],
        rb_c3=expanded["rb_c3"],
        rb_c4=expanded["rb_c4"],
        rb_c5=expanded["rb_c5"],
        constraints=constraints,
        constraint_distance=constraint_distance,
        water_mask=water_mask,
        ion_mask=ion_mask,
        lipid_mask=lipid_mask,
        impropers=empty_indices(4),
        improper_k=np.asarray([], dtype=np.float32),
        improper_periodicity=np.asarray([], dtype=np.float32),
        improper_phase=np.asarray([], dtype=np.float32),
        nonbonded_exception_pairs=expanded["nonbonded_exception_pairs"],
        nonbonded_exception_charge_product=expanded["nonbonded_exception_charge_product"],
        nonbonded_exception_sigma=expanded["nonbonded_exception_sigma"],
        nonbonded_exception_epsilon=expanded["nonbonded_exception_epsilon"],
    )
    try:
        prepared.validate()
    except ValueError as exc:
        msg = "unsupported_terms:gromacs_malformed_topology"
        raise TopologyImportError(msg) from exc
    return prepared


def _read_gromacs_topology(path: Path) -> GromacsTopology:
    sections = _gromacs_sections(path)
    defaults: GromacsDefaults | None = None
    atomtypes: dict[str, GromacsAtomType] = {}
    molecule_types: dict[str, GromacsMoleculeType] = {}
    molecules: list[tuple[str, int]] = []
    current_molecule: GromacsMoleculeType | None = None

    for section in sections:
        if section.name == "defaults":
            if defaults is not None:
                raise TopologyImportError("unsupported_terms:gromacs_duplicate_defaults")
            defaults = _parse_gromacs_defaults(section)
        elif section.name == "atomtypes":
            atomtypes.update(_parse_gromacs_atomtypes(section))
        elif section.name == "moleculetype":
            current_molecule = _parse_gromacs_moleculetype(section)
            if current_molecule.name in molecule_types:
                msg = f"unsupported_terms:gromacs_duplicate_moleculetype:{current_molecule.name}"
                raise TopologyImportError(msg)
            molecule_types[current_molecule.name] = current_molecule
        elif section.name == "atoms":
            _require_gromacs_molecule_context(current_molecule, section.name)
            current_molecule.atoms.extend(_parse_gromacs_atoms(section, atomtypes))
        elif section.name == "bonds":
            _require_gromacs_molecule_context(current_molecule, section.name)
            current_molecule.bonds.extend(_parse_gromacs_bonds(section, current_molecule))
        elif section.name == "angles":
            _require_gromacs_molecule_context(current_molecule, section.name)
            current_molecule.angles.extend(_parse_gromacs_angles(section, current_molecule))
        elif section.name == "dihedrals":
            _require_gromacs_molecule_context(current_molecule, section.name)
            periodic, rb = _parse_gromacs_dihedrals(section, current_molecule)
            current_molecule.dihedrals.extend(periodic)
            current_molecule.rb_dihedrals.extend(rb)
        elif section.name == "pairs":
            _require_gromacs_molecule_context(current_molecule, section.name)
            current_molecule.pairs.update(_parse_gromacs_pairs(section, current_molecule))
        elif section.name == "exclusions":
            _require_gromacs_molecule_context(current_molecule, section.name)
            current_molecule.exclusions.update(
                _parse_gromacs_exclusions(section, current_molecule)
            )
        elif section.name == "molecules":
            molecules.extend(_parse_gromacs_molecules(section))
        elif section.name == "system":
            continue
        else:  # pragma: no cover - _gromacs_sections rejects unsupported sections.
            raise TopologyImportError(f"unsupported_terms:gromacs_directive_{section.name}")

    if defaults is None:
        raise TopologyImportError("unsupported_terms:gromacs_missing_defaults")
    if not atomtypes:
        raise TopologyImportError("unsupported_terms:gromacs_missing_atomtypes")
    if not molecule_types:
        raise TopologyImportError("unsupported_terms:gromacs_missing_moleculetype")
    if not molecules:
        raise TopologyImportError("unsupported_terms:gromacs_missing_molecules")
    for molecule_name, _count in molecules:
        if molecule_name not in molecule_types:
            msg = f"unsupported_terms:gromacs_missing_moleculetype:{molecule_name}"
            raise TopologyImportError(msg)
    for molecule in molecule_types.values():
        _validate_gromacs_molecule_type(molecule, atomtypes, defaults)
    return GromacsTopology(
        defaults=defaults,
        atomtypes=atomtypes,
        molecule_types=molecule_types,
        molecules=tuple(molecules),
    )


def _gromacs_sections(path: Path) -> tuple[GromacsSection, ...]:
    sections: list[GromacsSection] = []
    current_name: str | None = None
    current_lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = _strip_gromacs_comment(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            directive = line.split(maxsplit=1)[0].lstrip("#").lower() or "preprocessor"
            msg = f"unsupported_terms:gromacs_preprocessor_directive:{directive}"
            raise TopologyImportError(msg)
        if line.startswith("["):
            match = re.fullmatch(r"\[\s*([A-Za-z0-9_]+)\s*\]", line)
            if match is None:
                raise TopologyImportError("unsupported_terms:gromacs_malformed_directive")
            if current_name is not None:
                sections.append(GromacsSection(current_name, tuple(current_lines)))
            current_name = match.group(1).lower()
            current_lines = []
            if current_name not in GROMACS_SUPPORTED_SECTIONS:
                msg = f"unsupported_terms:gromacs_directive_{current_name}"
                raise TopologyImportError(msg)
            continue
        if current_name is None:
            raise TopologyImportError("unsupported_terms:gromacs_record_outside_section")
        current_lines.append((line_number, line))
    if current_name is not None:
        sections.append(GromacsSection(current_name, tuple(current_lines)))
    return tuple(sections)


def _strip_gromacs_comment(line: str) -> str:
    return line.split(";", maxsplit=1)[0].strip()


def _parse_gromacs_defaults(section: GromacsSection) -> GromacsDefaults:
    rows = _gromacs_data_rows(section)
    if len(rows) != 1:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_defaults")
    _line_number, fields = rows[0]
    if len(fields) != 5:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_defaults")
    nbfunc = _gromacs_int(fields[0], "unsupported_terms:gromacs_malformed_defaults")
    combination_rule = _gromacs_int(fields[1], "unsupported_terms:gromacs_malformed_defaults")
    gen_pairs_token = fields[2].lower()
    if nbfunc != 1:
        raise TopologyImportError(f"unsupported_terms:gromacs_nonbonded_function_{nbfunc}")
    if combination_rule != 2:
        msg = f"unsupported_terms:gromacs_combination_rule_{combination_rule}"
        raise TopologyImportError(msg)
    if gen_pairs_token not in {"yes", "no"}:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_defaults")
    fudge_lj = _gromacs_float(fields[3], "unsupported_terms:gromacs_malformed_defaults")
    fudge_qq = _gromacs_float(fields[4], "unsupported_terms:gromacs_malformed_defaults")
    if fudge_lj < 0.0 or fudge_qq < 0.0:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_defaults")
    return GromacsDefaults(
        nbfunc=nbfunc,
        combination_rule=combination_rule,
        gen_pairs=gen_pairs_token == "yes",
        fudge_lj=fudge_lj,
        fudge_qq=fudge_qq,
    )


def _parse_gromacs_atomtypes(section: GromacsSection) -> dict[str, GromacsAtomType]:
    atomtypes: dict[str, GromacsAtomType] = {}
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) < 6:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_atomtypes")
        atom_type = fields[0]
        ptype = fields[-3].upper()
        if ptype != "A":
            raise TopologyImportError(f"unsupported_terms:gromacs_particle_type_{ptype.lower()}")
        mass = _gromacs_float(fields[-5], "unsupported_terms:gromacs_malformed_atomtypes")
        charge = _gromacs_float(fields[-4], "unsupported_terms:gromacs_malformed_atomtypes")
        sigma = (
            _gromacs_float(fields[-2], "unsupported_terms:gromacs_malformed_atomtypes")
            * NM_TO_ANGSTROM
        )
        epsilon = _gromacs_float(fields[-1], "unsupported_terms:gromacs_malformed_atomtypes")
        _validate_gromacs_nonbonded_values(
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon,
            blocker="unsupported_terms:gromacs_malformed_atomtypes",
        )
        if atom_type in atomtypes:
            raise TopologyImportError(f"unsupported_terms:gromacs_duplicate_atomtype:{atom_type}")
        atomtypes[atom_type] = GromacsAtomType(
            name=atom_type,
            mass=mass,
            charge=charge,
            sigma=sigma,
            epsilon=epsilon,
        )
    return atomtypes


def _parse_gromacs_moleculetype(section: GromacsSection) -> GromacsMoleculeType:
    rows = _gromacs_data_rows(section)
    if len(rows) != 1:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_moleculetype")
    _line_number, fields = rows[0]
    if len(fields) < 2:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_moleculetype")
    nrexcl = _gromacs_int(fields[1], "unsupported_terms:gromacs_malformed_moleculetype")
    if nrexcl < 0 or nrexcl > 4:
        raise TopologyImportError(f"unsupported_terms:gromacs_nrexcl_{nrexcl}")
    return GromacsMoleculeType(name=fields[0], nrexcl=nrexcl)


def _parse_gromacs_atoms(
    section: GromacsSection,
    atomtypes: Mapping[str, GromacsAtomType],
) -> list[GromacsAtom]:
    atoms: list[GromacsAtom] = []
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) > 8:
            raise TopologyImportError("unsupported_terms:gromacs_atoms_b_state")
        if len(fields) < 8:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_atoms")
        atom_type = fields[1]
        if atom_type not in atomtypes:
            raise TopologyImportError(f"unsupported_terms:gromacs_missing_atomtype:{atom_type}")
        index = _gromacs_int(fields[0], "unsupported_terms:gromacs_malformed_atoms") - 1
        residue_id = _gromacs_int(fields[2], "unsupported_terms:gromacs_malformed_atoms")
        charge_group = _gromacs_int(fields[5], "unsupported_terms:gromacs_malformed_atoms")
        charge = _gromacs_float(fields[6], "unsupported_terms:gromacs_malformed_atoms")
        mass = _gromacs_float(fields[7], "unsupported_terms:gromacs_malformed_atoms")
        if index != len(atoms) or residue_id <= 0 or charge_group <= 0:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_atoms")
        atom_name = fields[4]
        if mass <= 0.0 or atom_name.upper() in GROMACS_VIRTUAL_SITE_MARKERS:
            raise TopologyImportError("unsupported_terms:gromacs_virtual_sites")
        _validate_gromacs_finite("unsupported_terms:gromacs_malformed_atoms", charge, mass)
        atoms.append(
            GromacsAtom(
                index=index,
                atom_type=atom_type,
                residue_id=residue_id,
                residue_name=fields[3],
                atom_name=atom_name,
                charge_group=charge_group,
                charge=charge,
                mass=mass,
            )
        )
    return atoms


def _parse_gromacs_bonds(
    section: GromacsSection,
    molecule: GromacsMoleculeType,
) -> list[GromacsBond]:
    bonds: list[GromacsBond] = []
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) != 5:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_bonds")
        i = _gromacs_atom_index(fields[0], molecule, "unsupported_terms:gromacs_malformed_bonds")
        j = _gromacs_atom_index(fields[1], molecule, "unsupported_terms:gromacs_malformed_bonds")
        function_type = _gromacs_int(fields[2], "unsupported_terms:gromacs_malformed_bonds")
        if function_type != 1:
            raise TopologyImportError(f"unsupported_terms:gromacs_bond_function_{function_type}")
        length = (
            _gromacs_float(fields[3], "unsupported_terms:gromacs_malformed_bonds")
            * NM_TO_ANGSTROM
        )
        force_constant = (
            _gromacs_float(fields[4], "unsupported_terms:gromacs_malformed_bonds")
            / (NM_TO_ANGSTROM * NM_TO_ANGSTROM)
        )
        if length <= 0.0 or force_constant < 0.0:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_bonds")
        bonds.append(GromacsBond(i=i, j=j, k=force_constant, length=length))
    return bonds


def _parse_gromacs_angles(
    section: GromacsSection,
    molecule: GromacsMoleculeType,
) -> list[GromacsAngle]:
    angles: list[GromacsAngle] = []
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) != 6:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_angles")
        i = _gromacs_atom_index(fields[0], molecule, "unsupported_terms:gromacs_malformed_angles")
        j = _gromacs_atom_index(fields[1], molecule, "unsupported_terms:gromacs_malformed_angles")
        k = _gromacs_atom_index(fields[2], molecule, "unsupported_terms:gromacs_malformed_angles")
        function_type = _gromacs_int(fields[3], "unsupported_terms:gromacs_malformed_angles")
        if function_type != 1:
            raise TopologyImportError(f"unsupported_terms:gromacs_angle_function_{function_type}")
        theta = np.deg2rad(
            _gromacs_float(fields[4], "unsupported_terms:gromacs_malformed_angles")
        )
        force_constant = _gromacs_float(
            fields[5],
            "unsupported_terms:gromacs_malformed_angles",
        )
        if force_constant < 0.0:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_angles")
        angles.append(
            GromacsAngle(
                i=i,
                j=j,
                k=k,
                force_constant=force_constant,
                theta=float(theta),
            )
        )
    return angles


def _parse_gromacs_dihedrals(
    section: GromacsSection,
    molecule: GromacsMoleculeType,
) -> tuple[list[GromacsPeriodicDihedral], list[GromacsRBDihedral]]:
    periodic: list[GromacsPeriodicDihedral] = []
    rb: list[GromacsRBDihedral] = []
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) < 5:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_dihedrals")
        i = _gromacs_atom_index(
            fields[0],
            molecule,
            "unsupported_terms:gromacs_malformed_dihedrals",
        )
        j = _gromacs_atom_index(
            fields[1],
            molecule,
            "unsupported_terms:gromacs_malformed_dihedrals",
        )
        k = _gromacs_atom_index(
            fields[2],
            molecule,
            "unsupported_terms:gromacs_malformed_dihedrals",
        )
        m = _gromacs_atom_index(
            fields[3],
            molecule,
            "unsupported_terms:gromacs_malformed_dihedrals",
        )
        function_type = _gromacs_int(fields[4], "unsupported_terms:gromacs_malformed_dihedrals")
        if function_type == 1:
            if len(fields) != 8:
                raise TopologyImportError("unsupported_terms:gromacs_malformed_dihedrals")
            phase = -np.deg2rad(
                _gromacs_float(fields[5], "unsupported_terms:gromacs_malformed_dihedrals")
            )
            force_constant = _gromacs_float(
                fields[6],
                "unsupported_terms:gromacs_malformed_dihedrals",
            )
            periodicity = _gromacs_float(
                fields[7],
                "unsupported_terms:gromacs_malformed_dihedrals",
            )
            if periodicity <= 0.0:
                raise TopologyImportError("unsupported_terms:gromacs_malformed_dihedrals")
            periodic.append(
                GromacsPeriodicDihedral(
                    i=i,
                    j=j,
                    k=k,
                    m=m,
                    force_constant=force_constant,
                    periodicity=periodicity,
                    phase=float(phase),
                )
            )
            continue
        if function_type == 3:
            if len(fields) != 11:
                raise TopologyImportError("unsupported_terms:gromacs_malformed_rb_dihedrals")
            coefficients = tuple(
                _gromacs_float(value, "unsupported_terms:gromacs_malformed_rb_dihedrals")
                for value in fields[5:11]
            )
            rb.append(
                GromacsRBDihedral(
                    i=i,
                    j=j,
                    k=k,
                    m=m,
                    coefficients=coefficients,
                )
            )
            continue
        raise TopologyImportError(f"unsupported_terms:gromacs_dihedral_function_{function_type}")
    return periodic, rb


def _parse_gromacs_pairs(
    section: GromacsSection,
    molecule: GromacsMoleculeType,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) != 3:
            raise TopologyImportError("unsupported_terms:gromacs_explicit_pair_parameters")
        i = _gromacs_atom_index(fields[0], molecule, "unsupported_terms:gromacs_malformed_pairs")
        j = _gromacs_atom_index(fields[1], molecule, "unsupported_terms:gromacs_malformed_pairs")
        function_type = _gromacs_int(fields[2], "unsupported_terms:gromacs_malformed_pairs")
        if function_type != 1:
            raise TopologyImportError(f"unsupported_terms:gromacs_pair_function_{function_type}")
        pairs.add(_normalize_pair(i, j))
    return pairs


def _parse_gromacs_exclusions(
    section: GromacsSection,
    molecule: GromacsMoleculeType,
) -> set[tuple[int, int]]:
    exclusions: set[tuple[int, int]] = set()
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) < 2:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_exclusions")
        i = _gromacs_atom_index(
            fields[0],
            molecule,
            "unsupported_terms:gromacs_malformed_exclusions",
        )
        for token in fields[1:]:
            j = _gromacs_atom_index(
                token,
                molecule,
                "unsupported_terms:gromacs_malformed_exclusions",
            )
            exclusions.add(_normalize_pair(i, j))
    return exclusions


def _parse_gromacs_molecules(section: GromacsSection) -> list[tuple[str, int]]:
    molecules: list[tuple[str, int]] = []
    for _line_number, fields in _gromacs_data_rows(section):
        if len(fields) != 2:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_molecules")
        count = _gromacs_int(fields[1], "unsupported_terms:gromacs_malformed_molecules")
        if count <= 0:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_molecules")
        molecules.append((fields[0], count))
    return molecules


def _gromacs_data_rows(section: GromacsSection) -> list[tuple[int, list[str]]]:
    return [(line_number, line.split()) for line_number, line in section.lines if line.split()]


def _require_gromacs_molecule_context(
    molecule: GromacsMoleculeType | None,
    section_name: str,
) -> None:
    if molecule is None:
        msg = f"unsupported_terms:gromacs_{section_name}_without_moleculetype"
        raise TopologyImportError(msg)


def _validate_gromacs_molecule_type(
    molecule: GromacsMoleculeType,
    atomtypes: Mapping[str, GromacsAtomType],
    defaults: GromacsDefaults,
) -> None:
    if not molecule.atoms:
        msg = f"unsupported_terms:gromacs_moleculetype_without_atoms:{molecule.name}"
        raise TopologyImportError(msg)
    if molecule.pairs and not defaults.gen_pairs:
        raise TopologyImportError("unsupported_terms:gromacs_pairs_without_generated_parameters")
    for atom in molecule.atoms:
        atomtype = atomtypes.get(atom.atom_type)
        if atomtype is None:
            msg = f"unsupported_terms:gromacs_missing_atomtype:{atom.atom_type}"
            raise TopologyImportError(msg)
        if atom.atom_type.upper() in GROMACS_VIRTUAL_SITE_MARKERS or atomtype.mass <= 0.0:
            raise TopologyImportError("unsupported_terms:gromacs_virtual_sites")
    if defaults.gen_pairs and not molecule.pairs and (molecule.dihedrals or molecule.rb_dihedrals):
        raise TopologyImportError("unsupported_terms:gromacs_generated_pairs")


def _expand_gromacs_topology(topology: GromacsTopology) -> dict[str, Any]:
    atom_names: list[str] = []
    atom_types: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    chain_ids: list[str] = []
    masses: list[float] = []
    charges: list[float] = []
    sigma: list[float] = []
    epsilon: list[float] = []
    bonds: list[tuple[int, int]] = []
    bond_k: list[float] = []
    bond_length: list[float] = []
    angles: list[tuple[int, int, int]] = []
    angle_k: list[float] = []
    angle_theta: list[float] = []
    dihedrals: list[tuple[int, int, int, int]] = []
    dihedral_k: list[float] = []
    dihedral_periodicity: list[float] = []
    dihedral_phase: list[float] = []
    rb_dihedrals: list[tuple[int, int, int, int]] = []
    rb_coefficients: list[tuple[float, float, float, float, float, float]] = []
    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {}
    molecule_instance_count = 0
    gromacs_pair_exception_count = 0
    gromacs_exclusion_count = 0

    atom_offset = 0
    residue_offset = 0
    for molecule_name, count in topology.molecules:
        molecule = topology.molecule_types[molecule_name]
        for _instance_index in range(count):
            molecule_instance_count += 1
            local_residue_ids = [atom.residue_id for atom in molecule.atoms]
            for atom in molecule.atoms:
                atomtype = topology.atomtypes[atom.atom_type]
                atom_names.append(atom.atom_name)
                atom_types.append(atom.atom_type)
                residue_names.append(atom.residue_name)
                residue_ids.append(atom.residue_id + residue_offset)
                chain_ids.append("A")
                masses.append(atom.mass if atom.mass > 0.0 else atomtype.mass)
                charges.append(atom.charge)
                sigma.append(atomtype.sigma)
                epsilon.append(atomtype.epsilon)
            for bond in molecule.bonds:
                bonds.append((atom_offset + bond.i, atom_offset + bond.j))
                bond_k.append(bond.k)
                bond_length.append(bond.length)
            for angle in molecule.angles:
                angles.append((atom_offset + angle.i, atom_offset + angle.j, atom_offset + angle.k))
                angle_k.append(angle.force_constant)
                angle_theta.append(angle.theta)
            for dihedral in molecule.dihedrals:
                dihedrals.append(
                    (
                        atom_offset + dihedral.i,
                        atom_offset + dihedral.j,
                        atom_offset + dihedral.k,
                        atom_offset + dihedral.m,
                    )
                )
                dihedral_k.append(dihedral.force_constant)
                dihedral_periodicity.append(dihedral.periodicity)
                dihedral_phase.append(dihedral.phase)
            for rb_dihedral in molecule.rb_dihedrals:
                rb_dihedrals.append(
                    (
                        atom_offset + rb_dihedral.i,
                        atom_offset + rb_dihedral.j,
                        atom_offset + rb_dihedral.k,
                        atom_offset + rb_dihedral.m,
                    )
                )
                rb_coefficients.append(rb_dihedral.coefficients)
            local_exceptions, local_pair_count, local_exclusion_count = _gromacs_exceptions(
                molecule,
                topology.atomtypes,
                topology.defaults,
            )
            gromacs_pair_exception_count += local_pair_count
            gromacs_exclusion_count += local_exclusion_count
            for (i, j), values in local_exceptions.items():
                exceptions[(atom_offset + i, atom_offset + j)] = values
            atom_offset += len(molecule.atoms)
            residue_offset += max(local_residue_ids, default=0)

    exception_pairs = np.asarray(sorted(exceptions), dtype=np.int32).reshape((-1, 2))
    exception_values = [exceptions[tuple(pair)] for pair in exception_pairs.tolist()]
    coefficients = np.asarray(rb_coefficients, dtype=np.float32).reshape((-1, 6))
    return {
        "positions_atom_count": atom_offset,
        "molecule_instance_count": molecule_instance_count,
        "gromacs_pair_exception_count": gromacs_pair_exception_count,
        "gromacs_exclusion_count": gromacs_exclusion_count,
        "atom_names": atom_names,
        "atom_types": atom_types,
        "residue_names": residue_names,
        "residue_ids": residue_ids,
        "chain_ids": chain_ids,
        "masses": np.asarray(masses, dtype=np.float32),
        "charges": np.asarray(charges, dtype=np.float32),
        "sigma": np.asarray(sigma, dtype=np.float32),
        "epsilon": np.asarray(epsilon, dtype=np.float32),
        "bonds": np.asarray(bonds, dtype=np.int32).reshape((-1, 2)),
        "bond_k": np.asarray(bond_k, dtype=np.float32),
        "bond_length": np.asarray(bond_length, dtype=np.float32),
        "angles": np.asarray(angles, dtype=np.int32).reshape((-1, 3)),
        "angle_k": np.asarray(angle_k, dtype=np.float32),
        "angle_theta": np.asarray(angle_theta, dtype=np.float32),
        "dihedrals": np.asarray(dihedrals, dtype=np.int32).reshape((-1, 4)),
        "dihedral_k": np.asarray(dihedral_k, dtype=np.float32),
        "dihedral_periodicity": np.asarray(dihedral_periodicity, dtype=np.float32),
        "dihedral_phase": np.asarray(dihedral_phase, dtype=np.float32),
        "rb_dihedrals": np.asarray(rb_dihedrals, dtype=np.int32).reshape((-1, 4)),
        "rb_c0": coefficients[:, 0].astype(np.float32),
        "rb_c1": coefficients[:, 1].astype(np.float32),
        "rb_c2": coefficients[:, 2].astype(np.float32),
        "rb_c3": coefficients[:, 3].astype(np.float32),
        "rb_c4": coefficients[:, 4].astype(np.float32),
        "rb_c5": coefficients[:, 5].astype(np.float32),
        "nonbonded_exception_pairs": exception_pairs,
        "nonbonded_exception_charge_product": np.asarray(
            [value[0] for value in exception_values],
            dtype=np.float32,
        ),
        "nonbonded_exception_sigma": np.asarray(
            [value[1] for value in exception_values],
            dtype=np.float32,
        ),
        "nonbonded_exception_epsilon": np.asarray(
            [value[2] for value in exception_values],
            dtype=np.float32,
        ),
    }


def _gromacs_exceptions(
    molecule: GromacsMoleculeType,
    atomtypes: Mapping[str, GromacsAtomType],
    defaults: GromacsDefaults,
) -> tuple[dict[tuple[int, int], tuple[float, float, float]], int, int]:
    excluded = set(molecule.exclusions)
    if molecule.nrexcl > 0:
        excluded |= _bond_graph_pairs_within(molecule, max_depth=molecule.nrexcl)

    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {
        pair: (0.0, 0.0, 0.0) for pair in excluded
    }
    for i, j in sorted(molecule.pairs):
        atom_i = molecule.atoms[i]
        atom_j = molecule.atoms[j]
        type_i = atomtypes[atom_i.atom_type]
        type_j = atomtypes[atom_j.atom_type]
        sigma = 0.5 * (type_i.sigma + type_j.sigma)
        epsilon = (type_i.epsilon * type_j.epsilon) ** 0.5 * defaults.fudge_lj
        qprod = atom_i.charge * atom_j.charge * defaults.fudge_qq
        _validate_gromacs_finite("unsupported_terms:gromacs_malformed_pairs", sigma, epsilon, qprod)
        exceptions[_normalize_pair(i, j)] = (qprod, sigma, epsilon)
    return exceptions, len(molecule.pairs), len(excluded)


def _bond_graph_pairs_within(
    molecule: GromacsMoleculeType,
    *,
    max_depth: int,
) -> set[tuple[int, int]]:
    adjacency: dict[int, set[int]] = {atom.index: set() for atom in molecule.atoms}
    for bond in molecule.bonds:
        adjacency[bond.i].add(bond.j)
        adjacency[bond.j].add(bond.i)
    pairs: set[tuple[int, int]] = set()
    for start in adjacency:
        queue: deque[tuple[int, int]] = deque((neighbor, 1) for neighbor in adjacency[start])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            if node in seen or depth > max_depth:
                continue
            seen.add(node)
            pairs.add(_normalize_pair(start, node))
            if depth == max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    queue.append((neighbor, depth + 1))
    return pairs


def _read_gromacs_gro(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_gro") from exc
    if atom_count <= 0 or len(lines) < atom_count + 3:
        raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    saw_velocity = False
    for raw_line in lines[2 : 2 + atom_count]:
        fields = raw_line[20:].split()
        if len(fields) not in {3, 6}:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
        try:
            position = tuple(float(value) * NM_TO_ANGSTROM for value in fields[:3])
            velocity = tuple(float(value) * NM_TO_ANGSTROM for value in fields[3:6])
        except ValueError as exc:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_gro") from exc
        positions.append(position)
        if len(fields) == 6:
            saw_velocity = True
            velocities.append(velocity)
        elif saw_velocity:
            raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
    position_array = np.asarray(positions, dtype=np.float32)
    if not np.all(np.isfinite(position_array)):
        raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
    velocity_array = (
        np.asarray(velocities, dtype=np.float32)
        if saw_velocity
        else np.asarray([], dtype=np.float32)
    )
    if velocity_array.size and not np.all(np.isfinite(velocity_array)):
        raise TopologyImportError("unsupported_terms:gromacs_malformed_gro")
    cell_lengths, cell_matrix = _parse_gromacs_gro_box(lines[2 + atom_count])
    return position_array, velocity_array, cell_lengths, cell_matrix


def _parse_gromacs_gro_box(line: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.asarray([float(value) for value in line.split()], dtype=np.float32)
    except ValueError as exc:
        raise TopologyImportError("unsupported_terms:gromacs_invalid_periodic_box") from exc
    if values.shape == (3,):
        lengths = values * NM_TO_ANGSTROM
        _validate_gromacs_box_lengths(lengths)
        return lengths.astype(np.float32), np.asarray([], dtype=np.float32)
    if values.shape == (9,):
        scaled = values.astype(np.float32) * NM_TO_ANGSTROM
        matrix = np.asarray(
            [
                [scaled[0], scaled[3], scaled[4]],
                [scaled[5], scaled[1], scaled[6]],
                [scaled[7], scaled[8], scaled[2]],
            ],
            dtype=np.float32,
        )
        lengths = np.linalg.norm(matrix, axis=1).astype(np.float32)
        _validate_gromacs_box_lengths(lengths)
        determinant = float(np.linalg.det(matrix.astype(np.float64)))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise TopologyImportError("unsupported_terms:gromacs_invalid_periodic_box")
        return lengths, matrix
    raise TopologyImportError("unsupported_terms:gromacs_invalid_periodic_box")


def _validate_gromacs_box_lengths(lengths: np.ndarray) -> None:
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise TopologyImportError("unsupported_terms:gromacs_invalid_periodic_box")


def _gromacs_atom_index(token: str, molecule: GromacsMoleculeType, blocker: str) -> int:
    index = _gromacs_int(token, blocker) - 1
    if index < 0 or index >= len(molecule.atoms):
        raise TopologyImportError(blocker)
    return index


def _gromacs_int(token: str, blocker: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise TopologyImportError(blocker) from exc


def _gromacs_float(token: str, blocker: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise TopologyImportError(blocker) from exc
    _validate_gromacs_finite(blocker, value)
    return value


def _validate_gromacs_finite(blocker: str, *values: float) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise TopologyImportError(blocker)


def _validate_gromacs_nonbonded_values(
    *,
    mass: float,
    charge: float,
    sigma: float,
    epsilon: float,
    blocker: str,
) -> None:
    _validate_gromacs_finite(blocker, mass, charge, sigma, epsilon)
    if mass <= 0.0 or sigma <= 0.0 or epsilon < 0.0:
        raise TopologyImportError(blocker)
