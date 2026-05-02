"""Import prepared topology/parameter files into MLX-ready artifacts.

The importers in this module do not run molecular dynamics.  They translate
existing all-atom topology/parameter data into the strict artifact schema that
`mlx_atomistic` can validate and execute.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomistic_prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

AMBER_CHARGE_SCALE = 18.2223
KCAL_TO_KJ = 4.184
RMIN_TO_SIGMA = 2 ** (-1.0 / 6.0)
STANDARD_AMBER_14_ELECTROSTATIC_SCALE = 1.0 / 1.2
STANDARD_AMBER_14_LJ_SCALE = 1.0 / 2.0


class TopologyImportError(ValueError):
    """Raised when topology/parameter files cannot be imported faithfully."""


@dataclass(frozen=True)
class CharmmMassPrelude:
    """PSF-derived CHARMM MASS records used only as parser aid."""

    source_path: str
    missing_atom_types: tuple[str, ...]
    text: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "missing_atom_types": list(self.missing_atom_types),
            "missing_atom_type_count": len(self.missing_atom_types),
        }


@dataclass(frozen=True)
class AmberPrmtop:
    """Parsed AMBER prmtop flags."""

    flags: dict[str, list[str] | list[int] | list[float]]
    formats: dict[str, str]

    def values(self, name: str) -> list[str] | list[int] | list[float]:
        if name not in self.flags:
            msg = f"AMBER prmtop is missing required %FLAG {name}"
            raise TopologyImportError(msg)
        return self.flags[name]

    def optional_values(self, name: str) -> list[str] | list[int] | list[float]:
        return self.flags.get(name, [])


def import_amber_prmtop(
    *,
    prmtop_path: str | Path,
    coords_path: str | Path,
) -> PreparedSystem:
    """Import an AMBER `prmtop` plus `inpcrd`/`rst7` coordinate file."""

    prmtop_path = Path(prmtop_path)
    coords_path = Path(coords_path)
    topology = _read_amber_prmtop(prmtop_path)
    positions, velocities, cell_lengths = _read_amber_restart(coords_path)

    atom_names = np.asarray([str(item).strip() for item in topology.values("ATOM_NAME")], dtype=str)
    atom_count = int(atom_names.shape[0])
    if positions.shape != (atom_count, 3):
        msg = (
            f"coordinate atom count does not match prmtop: positions={positions.shape[0]}, "
            f"prmtop={atom_count}"
        )
        raise TopologyImportError(msg)
    if velocities.size == 0:
        velocities = np.zeros_like(positions, dtype=np.float32)

    atom_types_raw = topology.optional_values("AMBER_ATOM_TYPE")
    if atom_types_raw:
        atom_types = np.asarray([str(item).strip() for item in atom_types_raw], dtype=str)
    else:
        atom_types = np.asarray(
            [str(item) for item in topology.values("ATOM_TYPE_INDEX")],
            dtype=str,
        )
    symbols = np.asarray(
        [
            _infer_symbol(name, atom_type)
            for name, atom_type in zip(atom_names, atom_types, strict=True)
        ],
        dtype=str,
    )
    charges = np.asarray(topology.values("CHARGE"), dtype=np.float32) / AMBER_CHARGE_SCALE
    masses = np.asarray(topology.values("MASS"), dtype=np.float32)
    type_indices = np.asarray(topology.values("ATOM_TYPE_INDEX"), dtype=np.int32)
    sigma, epsilon = _amber_lj_self_parameters(topology, type_indices)

    residue_names, residue_ids, chain_ids = _amber_residue_arrays(topology, atom_count)
    bonds, bond_k, bond_length = _amber_bonds(topology)
    angles, angle_k, angle_theta = _amber_angles(topology)
    dihedrals, dihedral_k, dihedral_periodicity, dihedral_phase, raw_14_pairs = _amber_dihedrals(
        topology,
        improper=False,
    )
    impropers, improper_k, improper_periodicity, improper_phase, _ = _amber_dihedrals(
        topology,
        improper=True,
    )
    exception_pairs, exception_qprod, exception_sigma, exception_epsilon = _amber_exceptions(
        bonds=bonds,
        angles=angles,
        one_four_pairs=raw_14_pairs,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
    )

    ligand_mask = _ligand_mask_from_residues(residue_names)
    water_mask = _water_mask_from_residues(residue_names)
    ion_mask = _ion_mask_from_residues(residue_names)
    lipid_mask = _lipid_mask_from_residues(residue_names)
    receptor_mask = ~(ligand_mask | water_mask | ion_mask | lipid_mask)
    restraint_mask = receptor_mask & ~ligand_mask
    constraints, constraint_distance = _hydrogen_bond_constraints(
        bonds,
        symbols=symbols,
        positions=positions,
    )
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols.astype(str)) == "H"))
    if hydrogen_count == 0:
        msg = (
            "production AMBER import found no hydrogens. Add hydrogens/protonation "
            "before importing; production MLX MD requires an all-atom topology."
        )
        raise TopologyImportError(msg)

    supported_terms = ["nonbonded_lj_coulomb"]
    if bonds.shape[0]:
        supported_terms.append("harmonic_bond")
    if angles.shape[0]:
        supported_terms.append("harmonic_angle")
    if dihedrals.shape[0]:
        supported_terms.append("periodic_dihedral")
    if impropers.shape[0]:
        supported_terms.append("periodic_improper")
    if exception_pairs.shape[0]:
        supported_terms.append("nonbonded_exception")
    if constraints.shape[0]:
        supported_terms.append("distance_constraint")

    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source={
            "kind": "amber",
            "prmtop_path": str(prmtop_path),
            "coords_path": str(coords_path),
        },
        selections={
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
            "water_atom_count": int(np.count_nonzero(water_mask)),
            "ion_atom_count": int(np.count_nonzero(ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(lipid_mask)),
            "system_charge": float(np.sum(charges)),
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="amber_prmtop",
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "water_present": bool(np.any(water_mask)),
            "ions_present": bool(np.any(ion_mask)),
            "lipids_present": bool(np.any(lipid_mask)),
            "periodic_box_present": bool(cell_lengths.shape == (3,)),
            "supported_terms": supported_terms,
            "required_terms": supported_terms,
            "unsupported_terms": [],
            "rejected_terms": [],
            "parameter_counts_match_topology": True,
            "term_counts": _term_counts(
                bonds=bonds,
                angles=angles,
                dihedrals=dihedrals,
                impropers=impropers,
                constraints=constraints,
                nonbonded_exception_pairs=exception_pairs,
            ),
            "force_field_provenance": "AMBER prmtop/inpcrd import",
        },
        warnings=[
            "Imported from fixed-topology AMBER files. MLX generates trajectories; "
            "this importer does not run an external MD engine.",
        ],
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=symbols,
        atom_names=atom_names,
        atom_types=atom_types,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        masses=masses.astype(np.float32),
        charges=charges.astype(np.float32),
        sigma=sigma.astype(np.float32),
        epsilon=epsilon.astype(np.float32),
        bonds=bonds,
        bond_k=bond_k,
        bond_length=bond_length,
        angles=angles,
        angle_k=angle_k,
        angle_theta=angle_theta,
        dihedrals=dihedrals,
        dihedral_k=dihedral_k,
        dihedral_periodicity=dihedral_periodicity,
        dihedral_phase=dihedral_phase,
        nonbonded_pairs=empty_indices(2),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=restraint_mask,
        reference_positions=positions.astype(np.float32).copy(),
        cell_lengths=cell_lengths.astype(np.float32),
        constraints=constraints,
        constraint_distance=constraint_distance,
        water_mask=water_mask,
        ion_mask=ion_mask,
        lipid_mask=lipid_mask,
        impropers=impropers,
        improper_k=improper_k,
        improper_periodicity=improper_periodicity,
        improper_phase=improper_phase,
        nonbonded_exception_pairs=exception_pairs,
        nonbonded_exception_charge_product=exception_qprod,
        nonbonded_exception_sigma=exception_sigma,
        nonbonded_exception_epsilon=exception_epsilon,
    )
    prepared.validate()
    return prepared


def import_charmm_with_parmed(
    *,
    psf_path: str | Path,
    params: Sequence[str | Path],
    coords_path: str | Path,
) -> PreparedSystem:
    """Import a CHARMM PSF/parameter/coordinate bundle through ParmEd parsing."""

    if importlib.util.find_spec("parmed") is None:
        msg = (
            "CHARMM import needs ParmEd as a parser. Install `uv sync --extra prep`; "
            "no external MD engine is used."
        )
        raise TopologyImportError(msg)

    import parmed as pmd

    params = [str(Path(path)) for path in params]
    try:
        parameter_set = pmd.charmm.CharmmParameterSet(*params)
        structure = pmd.charmm.CharmmPsfFile(str(psf_path))
        structure.load_parameters(parameter_set)
        coordinates = pmd.load_file(str(coords_path))
        structure.coordinates = coordinates.coordinates
    except Exception as exc:  # pragma: no cover - depends on external parser details
        msg = f"could not parse CHARMM topology/parameters with ParmEd: {exc}"
        raise TopologyImportError(msg) from exc
    return _prepared_from_parmed_structure(
        structure,
        source={
            "kind": "charmm",
            "psf_path": str(psf_path),
            "params": params,
            "coords_path": str(coords_path),
        },
        parameter_source="charmm_psf_parameters",
    )


def build_charmm_psf_mass_prelude(
    *,
    psf_path: str | Path,
    params: Sequence[str | Path],
) -> CharmmMassPrelude | None:
    """Build missing CHARMM MASS records from atom types and masses in a PSF."""

    psf = Path(psf_path)
    psf_masses = _psf_atom_type_masses(psf)
    if not psf_masses:
        return None
    parameter_mass_types = _charmm_parameter_mass_types(params)
    missing_types = tuple(sorted(set(psf_masses) - parameter_mass_types))
    if not missing_types:
        return None
    lines = [
        "* generated from PSF atom type masses for ParmEd parsing",
        "*",
    ]
    for index, atom_type in enumerate(missing_types, start=1):
        lines.append(f"MASS {index:5d} {atom_type:<10s} {psf_masses[atom_type]:10.5f}")
    return CharmmMassPrelude(
        source_path=str(psf),
        missing_atom_types=missing_types,
        text="\n".join(lines) + "\n",
    )


def _prepared_from_parmed_structure(
    structure: Any,
    *,
    source: dict[str, Any],
    parameter_source: str,
) -> PreparedSystem:
    unexportable_terms = _unexportable_charmm_terms_from_parmed_structure(structure)
    if unexportable_terms:
        msg = "unsupported_terms:" + ",".join(unexportable_terms)
        raise TopologyImportError(msg)

    atoms = list(structure.atoms)
    atom_count = len(atoms)
    if atom_count <= 0:
        msg = "parsed topology contains no atoms"
        raise TopologyImportError(msg)
    if getattr(structure, "coordinates", None) is None:
        msg = "parsed topology does not include coordinates"
        raise TopologyImportError(msg)

    positions = np.asarray(structure.coordinates, dtype=np.float32).reshape((atom_count, 3))
    symbols = np.asarray(
        [_infer_symbol(getattr(atom, "name", ""), getattr(atom, "type", "")) for atom in atoms],
        dtype=str,
    )
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols.astype(str)) == "H"))
    if hydrogen_count == 0:
        msg = "production topology import found no hydrogens"
        raise TopologyImportError(msg)

    atom_names = np.asarray(
        [str(getattr(atom, "name", f"A{idx + 1}")) for idx, atom in enumerate(atoms)],
    )
    atom_types = np.asarray(
        [str(getattr(atom, "type", symbols[idx])) for idx, atom in enumerate(atoms)],
    )
    residue_names = np.asarray(
        [str(getattr(getattr(atom, "residue", None), "name", "SYS")) for atom in atoms],
        dtype=str,
    )
    residue_ids = np.asarray(
        [
            int(getattr(getattr(atom, "residue", None), "number", idx + 1) or idx + 1)
            for idx, atom in enumerate(atoms)
        ],
        dtype=np.int32,
    )
    chain_ids = np.asarray(["A"] * atom_count, dtype=str)
    masses = np.asarray([float(getattr(atom, "mass", 12.0)) for atom in atoms], dtype=np.float32)
    charges = np.asarray([float(getattr(atom, "charge", 0.0)) for atom in atoms], dtype=np.float32)
    sigma = np.asarray(
        [float(getattr(atom, "sigma", 1.0) or 1.0) for atom in atoms],
        dtype=np.float32,
    )
    epsilon = np.asarray(
        [abs(float(getattr(atom, "epsilon", 0.0) or 0.0)) * KCAL_TO_KJ for atom in atoms],
        dtype=np.float32,
    )

    bonds, bond_k, bond_length = _parmed_bonds(structure.bonds)
    angles, angle_k, angle_theta = _parmed_angles(getattr(structure, "angles", []))
    (
        dihedrals,
        dihedral_k,
        dihedral_periodicity,
        dihedral_phase,
        impropers,
        improper_k,
        improper_periodicity,
        improper_phase,
    ) = _parmed_dihedrals(
        getattr(structure, "dihedrals", []),
    )
    urey_bradley_terms, urey_bradley_k, urey_bradley_distance = _parmed_urey_bradleys(
        getattr(structure, "urey_bradleys", []),
        angles=angles,
    )
    charmm_cmap_terms, charmm_cmap_grid_indices, charmm_cmap_grids = _parmed_cmaps(
        getattr(structure, "cmaps", []),
    )
    nbfix_type_pairs, nbfix_type_sigma, nbfix_type_epsilon, nbfix_details = (
        _parmed_nbfix_type_overrides(structure)
    )

    angle_exclusions = _pairs_from_angles(angles)
    bond_pairs = _normalized_pairs(bonds)
    exception_pairs = np.asarray(
        sorted(bond_pairs | angle_exclusions),
        dtype=np.int32,
    ).reshape((-1, 2))
    exception_count = exception_pairs.shape[0]
    ligand_mask = _ligand_mask_from_residues(residue_names)
    water_mask = _water_mask_from_residues(residue_names)
    ion_mask = _ion_mask_from_residues(residue_names)
    lipid_mask = _lipid_mask_from_residues(residue_names)
    receptor_mask = ~(ligand_mask | water_mask | ion_mask | lipid_mask)
    constraints, constraint_distance = _hydrogen_bond_constraints(
        bonds,
        symbols=symbols,
        positions=positions,
    )
    supported_terms = [
        "harmonic_bond",
        "harmonic_angle",
        "periodic_dihedral",
        "periodic_improper",
        "nonbonded_lj_coulomb",
        "nonbonded_exception",
    ]
    if constraints.shape[0]:
        supported_terms.append("distance_constraint")
    if urey_bradley_terms.shape[0]:
        supported_terms.append("urey_bradley")
    if charmm_cmap_terms.shape[0]:
        supported_terms.append("charmm_cmap_terms")
    if nbfix_type_pairs.shape[0]:
        supported_terms.append("nbfix_pair_overrides")
    required_terms = list(supported_terms)
    rejected_terms: list[str] = []
    rejection_reasons: dict[str, str] = {}
    term_counts = _term_counts(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        constraints=constraints,
        nonbonded_exception_pairs=exception_pairs,
    )
    term_counts.update(
        {
            "charmm_cmap_terms": int(charmm_cmap_terms.shape[0]),
            "charmm_cmap_grids": int(charmm_cmap_grids.shape[0]),
            "urey_bradley_terms": int(urey_bradley_terms.shape[0]),
            "nbfix_pair_overrides": int(nbfix_type_pairs.shape[0]),
        }
    )
    term_details = {}
    if nbfix_details:
        term_details["nbfix_pair_overrides"] = nbfix_details
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source=source,
        selections={
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
            "water_atom_count": int(np.count_nonzero(water_mask)),
            "ion_atom_count": int(np.count_nonzero(ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(lipid_mask)),
            "system_charge": float(np.sum(charges)),
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source=parameter_source,
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "water_present": bool(np.any(water_mask)),
            "ions_present": bool(np.any(ion_mask)),
            "lipids_present": bool(np.any(lipid_mask)),
            "periodic_box_present": bool(getattr(structure, "box", None) is not None),
            "supported_terms": supported_terms,
            "required_terms": required_terms,
            "unsupported_terms": [],
            "rejected_terms": rejected_terms,
            "rejection_reasons": rejection_reasons,
            "rejected_term_details": {},
            "term_details": term_details,
            "parameter_counts_match_topology": True,
            "term_counts": term_counts,
        },
        warnings=[
            "Imported with ParmEd as a parser only. MLX generates trajectories.",
        ],
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=symbols,
        atom_names=atom_names,
        atom_types=atom_types,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        positions=positions,
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=masses,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        bonds=bonds,
        bond_k=bond_k,
        bond_length=bond_length,
        angles=angles,
        angle_k=angle_k,
        angle_theta=angle_theta,
        dihedrals=dihedrals,
        dihedral_k=dihedral_k,
        dihedral_periodicity=dihedral_periodicity,
        dihedral_phase=dihedral_phase,
        nonbonded_pairs=empty_indices(2),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=receptor_mask & ~ligand_mask,
        reference_positions=positions.copy(),
        constraints=constraints,
        constraint_distance=constraint_distance,
        water_mask=water_mask,
        ion_mask=ion_mask,
        lipid_mask=lipid_mask,
        impropers=impropers,
        improper_k=improper_k,
        improper_periodicity=improper_periodicity,
        improper_phase=improper_phase,
        nonbonded_exception_pairs=exception_pairs,
        nonbonded_exception_charge_product=np.zeros((exception_count,), dtype=np.float32),
        nonbonded_exception_sigma=np.zeros((exception_count,), dtype=np.float32),
        nonbonded_exception_epsilon=np.zeros((exception_count,), dtype=np.float32),
        charmm_cmap_terms=charmm_cmap_terms,
        charmm_cmap_grid_indices=charmm_cmap_grid_indices,
        charmm_cmap_grids=charmm_cmap_grids,
        urey_bradley_terms=urey_bradley_terms,
        urey_bradley_k=urey_bradley_k,
        urey_bradley_distance=urey_bradley_distance,
        nbfix_type_pairs=nbfix_type_pairs,
        nbfix_type_sigma=nbfix_type_sigma,
        nbfix_type_epsilon=nbfix_type_epsilon,
    )
    prepared.validate()
    return prepared


def _psf_atom_type_masses(path: Path) -> dict[str, float]:
    lines = path.read_text(errors="replace").splitlines()
    masses: dict[str, float] = {}
    atom_count = 0
    atom_start = 0
    for index, line in enumerate(lines):
        if "!NATOM" not in line:
            continue
        fields = line.split()
        if not fields:
            break
        atom_count = int(fields[0])
        atom_start = index + 1
        break
    if atom_count <= 0:
        return masses
    for line in lines[atom_start : atom_start + atom_count]:
        fields = line.split()
        if len(fields) < 8:
            continue
        atom_type = fields[5]
        try:
            mass = float(fields[7])
        except ValueError:
            continue
        previous = masses.get(atom_type)
        if previous is not None and abs(previous - mass) > 1e-3:
            msg = f"PSF atom type {atom_type!r} has inconsistent masses"
            raise TopologyImportError(msg)
        masses[atom_type] = mass
    return masses


def _charmm_parameter_mass_types(params: Sequence[str | Path]) -> set[str]:
    mass_types: set[str] = set()
    for param in params:
        path = Path(param)
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(errors="replace").splitlines():
            line = raw_line.split("!", maxsplit=1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) >= 3 and fields[0].upper() == "MASS":
                mass_types.add(fields[2])
    return mass_types


def _read_amber_prmtop(path: Path) -> AmberPrmtop:
    current_flag: str | None = None
    current_format = ""
    data: dict[str, list[str]] = {}
    formats: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("%FLAG"):
            current_flag = line.split(maxsplit=1)[1].strip()
            data[current_flag] = []
            current_format = ""
            continue
        if line.startswith("%FORMAT"):
            current_format = line
            if current_flag is not None:
                formats[current_flag] = line
            continue
        if current_flag is None or not line:
            continue
        if "a" in current_format.lower():
            width = _format_width(current_format, default=4)
            data[current_flag].extend(
                chunk.strip() for chunk in _fixed_width_chunks(line, width) if chunk.strip()
            )
        else:
            data[current_flag].extend(line.split())

    parsed: dict[str, list[str] | list[int] | list[float]] = {}
    for flag, values in data.items():
        fmt = formats.get(flag, "")
        lowered = fmt.lower()
        if "a" in lowered:
            parsed[flag] = values
        elif "i" in lowered:
            parsed[flag] = [int(value) for value in values]
        else:
            parsed[flag] = [float(value.replace("D", "E").replace("d", "e")) for value in values]
    return AmberPrmtop(flags=parsed, formats=formats)


def _read_amber_restart(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        msg = f"AMBER coordinate file is too short: {path}"
        raise TopologyImportError(msg)
    header = lines[1].split()
    if not header:
        msg = f"AMBER coordinate file has no atom count: {path}"
        raise TopologyImportError(msg)
    atom_count = int(header[0])
    values = [
        float(value.replace("D", "E").replace("d", "e"))
        for line in lines[2:]
        for value in line.split()
    ]
    coordinate_count = 3 * atom_count
    if len(values) < coordinate_count:
        msg = f"AMBER coordinate file has fewer than {coordinate_count} coordinate values"
        raise TopologyImportError(msg)
    positions = np.asarray(values[:coordinate_count], dtype=np.float32).reshape((atom_count, 3))
    remainder = values[coordinate_count:]
    velocities = np.asarray([], dtype=np.float32)
    if len(remainder) >= coordinate_count:
        velocities = np.asarray(
            remainder[:coordinate_count],
            dtype=np.float32,
        ).reshape((atom_count, 3))
        remainder = remainder[coordinate_count:]
    cell_lengths = np.asarray([], dtype=np.float32)
    if len(remainder) >= 3:
        cell_lengths = np.asarray(remainder[:3], dtype=np.float32)
    return positions, velocities, cell_lengths


def _format_width(format_line: str, *, default: int) -> int:
    match = re.search(r"[aifedg](\d+)", format_line, flags=re.IGNORECASE)
    if match is None:
        return default
    return int(match.group(1))


def _fixed_width_chunks(line: str, width: int) -> Iterable[str]:
    for start in range(0, len(line), width):
        yield line[start : start + width]


def _infer_symbol(atom_name: str, atom_type: str = "") -> str:
    text = str(atom_name or atom_type).strip().upper().lstrip("0123456789")
    atom_type_text = str(atom_type).strip().upper().lstrip("0123456789")
    for candidate in (text, atom_type_text):
        if candidate.startswith(("CL", "BR", "NA", "MG", "ZN", "FE", "CA")):
            return candidate[:2].title()
    for character in text or atom_type_text:
        if character.isalpha():
            return character.upper()
    return "C"


def _amber_residue_arrays(
    topology: AmberPrmtop,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = [str(item).strip() or "SYS" for item in topology.optional_values("RESIDUE_LABEL")]
    pointers = [int(item) for item in topology.optional_values("RESIDUE_POINTER")]
    if not labels or not pointers:
        return (
            np.asarray(["SYS"] * atom_count, dtype=str),
            np.ones((atom_count,), dtype=np.int32),
            np.asarray(["A"] * atom_count, dtype=str),
        )
    starts = [pointer - 1 for pointer in pointers] + [atom_count]
    residue_names: list[str] = []
    residue_ids: list[int] = []
    residue_ranges = zip(labels, starts[:-1], starts[1:], strict=True)
    for residue_index, (name, start, stop) in enumerate(residue_ranges, start=1):
        count = max(0, stop - start)
        residue_names.extend([name] * count)
        residue_ids.extend([residue_index] * count)
    if len(residue_names) != atom_count:
        msg = "AMBER residue pointers do not cover all atoms"
        raise TopologyImportError(msg)
    return (
        np.asarray(residue_names, dtype=str),
        np.asarray(residue_ids, dtype=np.int32),
        np.asarray(["A"] * atom_count, dtype=str),
    )


def _amber_bonds(topology: AmberPrmtop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = _amber_index_parameter_records(
        topology.optional_values("BONDS_INC_HYDROGEN"),
        width=3,
    ) + _amber_index_parameter_records(topology.optional_values("BONDS_WITHOUT_HYDROGEN"), width=3)
    force_constants = np.asarray(topology.optional_values("BOND_FORCE_CONSTANT"), dtype=np.float32)
    lengths = np.asarray(topology.optional_values("BOND_EQUIL_VALUE"), dtype=np.float32)
    bonds: list[tuple[int, int]] = []
    k_values: list[float] = []
    length_values: list[float] = []
    for i_raw, j_raw, parameter_index in triples:
        bonds.append((i_raw // 3, j_raw // 3))
        param = parameter_index - 1
        k_values.append(2.0 * float(force_constants[param]) * KCAL_TO_KJ)
        length_values.append(float(lengths[param]))
    return (
        np.asarray(bonds, dtype=np.int32).reshape((-1, 2)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(length_values, dtype=np.float32),
    )


def _amber_angles(topology: AmberPrmtop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = _amber_index_parameter_records(
        topology.optional_values("ANGLES_INC_HYDROGEN"),
        width=4,
    ) + _amber_index_parameter_records(topology.optional_values("ANGLES_WITHOUT_HYDROGEN"), width=4)
    force_constants = np.asarray(topology.optional_values("ANGLE_FORCE_CONSTANT"), dtype=np.float32)
    theta = np.asarray(topology.optional_values("ANGLE_EQUIL_VALUE"), dtype=np.float32)
    angles: list[tuple[int, int, int]] = []
    k_values: list[float] = []
    theta_values: list[float] = []
    for i_raw, j_raw, k_raw, parameter_index in records:
        angles.append((i_raw // 3, j_raw // 3, k_raw // 3))
        param = parameter_index - 1
        k_values.append(2.0 * float(force_constants[param]) * KCAL_TO_KJ)
        theta_values.append(float(theta[param]))
    return (
        np.asarray(angles, dtype=np.int32).reshape((-1, 3)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(theta_values, dtype=np.float32),
    )


def _amber_dihedrals(
    topology: AmberPrmtop,
    *,
    improper: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set[tuple[int, int]]]:
    records = _amber_index_parameter_records(
        topology.optional_values("DIHEDRALS_INC_HYDROGEN"),
        width=5,
    ) + _amber_index_parameter_records(
        topology.optional_values("DIHEDRALS_WITHOUT_HYDROGEN"),
        width=5,
    )
    force_constants = np.asarray(
        topology.optional_values("DIHEDRAL_FORCE_CONSTANT"),
        dtype=np.float32,
    )
    periodicity = np.asarray(topology.optional_values("DIHEDRAL_PERIODICITY"), dtype=np.float32)
    phase = np.asarray(topology.optional_values("DIHEDRAL_PHASE"), dtype=np.float32)
    selected: list[tuple[int, int, int, int]] = []
    k_values: list[float] = []
    periodicity_values: list[float] = []
    phase_values: list[float] = []
    one_four_pairs: set[tuple[int, int]] = set()
    for i_raw, j_raw, k_raw, l_raw, parameter_index in records:
        is_improper = l_raw < 0
        if is_improper != improper:
            if not is_improper and not improper:
                one_four_pairs.add(_normalize_pair(i_raw // 3, abs(l_raw) // 3))
            continue
        atoms = (i_raw // 3, j_raw // 3, abs(k_raw) // 3, abs(l_raw) // 3)
        selected.append(atoms)
        param = parameter_index - 1
        k_values.append(float(force_constants[param]) * KCAL_TO_KJ)
        periodicity_values.append(float(abs(periodicity[param])))
        phase_values.append(float(phase[param]))
        if not improper:
            one_four_pairs.add(_normalize_pair(atoms[0], atoms[3]))
    return (
        np.asarray(selected, dtype=np.int32).reshape((-1, 4)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(periodicity_values, dtype=np.float32),
        np.asarray(phase_values, dtype=np.float32),
        one_four_pairs,
    )


def _amber_index_parameter_records(
    values: Sequence[int] | Sequence[float],
    *,
    width: int,
) -> list[tuple[int, ...]]:
    if not values:
        return []
    ints = [int(value) for value in values]
    if len(ints) % width:
        msg = f"AMBER topology record length {len(ints)} is not divisible by {width}"
        raise TopologyImportError(msg)
    return [tuple(ints[index : index + width]) for index in range(0, len(ints), width)]


def _amber_lj_self_parameters(
    topology: AmberPrmtop,
    atom_type_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pointers = topology.values("POINTERS")
    ntypes = int(pointers[1]) if len(pointers) > 1 else int(np.max(atom_type_indices))
    nb_index = np.asarray(topology.values("NONBONDED_PARM_INDEX"), dtype=np.int32)
    acoef = np.asarray(topology.values("LENNARD_JONES_ACOEF"), dtype=np.float64)
    bcoef = np.asarray(topology.values("LENNARD_JONES_BCOEF"), dtype=np.float64)
    type_sigma = np.zeros((ntypes,), dtype=np.float32)
    type_epsilon = np.zeros((ntypes,), dtype=np.float32)
    for type_id in range(1, ntypes + 1):
        packed_index = int(nb_index[(type_id - 1) * ntypes + (type_id - 1)])
        if packed_index <= 0:
            msg = "negative AMBER nonbonded parameter indices are not supported"
            raise TopologyImportError(msg)
        a = float(acoef[packed_index - 1])
        b = float(bcoef[packed_index - 1])
        if a <= 0.0 or b <= 0.0:
            type_sigma[type_id - 1] = 1.0
            type_epsilon[type_id - 1] = 0.0
            continue
        type_sigma[type_id - 1] = float((a / b) ** (1.0 / 6.0))
        type_epsilon[type_id - 1] = float((b * b / (4.0 * a)) * KCAL_TO_KJ)
    indices = atom_type_indices.astype(np.int32) - 1
    return type_sigma[indices], type_epsilon[indices]


def _amber_exceptions(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    one_four_pairs: set[tuple[int, int]],
    charges: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    excluded_pairs = _normalized_pairs(bonds) | _pairs_from_angles(angles)
    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {
        pair: (0.0, 0.0, 0.0) for pair in excluded_pairs
    }
    for i, j in sorted(one_four_pairs):
        sigma_ij = 0.5 * (float(sigma[i]) + float(sigma[j]))
        epsilon_ij = (float(epsilon[i]) * float(epsilon[j])) ** 0.5
        exceptions[(i, j)] = (
            float(charges[i] * charges[j]) * STANDARD_AMBER_14_ELECTROSTATIC_SCALE,
            sigma_ij,
            epsilon_ij * STANDARD_AMBER_14_LJ_SCALE,
        )
    if not exceptions:
        return empty_indices(2), *(np.asarray([], dtype=np.float32) for _ in range(3))
    pairs = np.asarray(sorted(exceptions), dtype=np.int32)
    values = [exceptions[tuple(pair)] for pair in pairs.tolist()]
    return (
        pairs.reshape((-1, 2)),
        np.asarray([value[0] for value in values], dtype=np.float32),
        np.asarray([value[1] for value in values], dtype=np.float32),
        np.asarray([value[2] for value in values], dtype=np.float32),
    )


def _normalized_pairs(array: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if array.size == 0:
        return pairs
    for row in np.asarray(array, dtype=np.int32):
        pairs.add(_normalize_pair(int(row[0]), int(row[1])))
    return pairs


def _pairs_from_angles(angles: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if angles.size == 0:
        return pairs
    for i, _, k in np.asarray(angles, dtype=np.int32):
        pairs.add(_normalize_pair(int(i), int(k)))
    return pairs


def _normalize_pair(i: int, j: int) -> tuple[int, int]:
    return (min(i, j), max(i, j))


def _ligand_mask_from_residues(residue_names: np.ndarray) -> np.ndarray:
    ligand_resnames = {"ATP", "ADP", "ANP", "LIG", "P32"}
    return np.asarray([str(name).upper() in ligand_resnames for name in residue_names], dtype=bool)


def _water_mask_from_residues(residue_names: np.ndarray) -> np.ndarray:
    water_resnames = {"WAT", "HOH", "TIP3", "TP3", "SOL"}
    return np.asarray([str(name).upper() in water_resnames for name in residue_names], dtype=bool)


def _ion_mask_from_residues(residue_names: np.ndarray) -> np.ndarray:
    ion_resnames = {"NA", "K", "CL", "MG", "CA", "ZN", "SOD", "CLA"}
    return np.asarray([str(name).upper() in ion_resnames for name in residue_names], dtype=bool)


def _unexportable_charmm_terms_from_parmed_structure(structure: Any) -> tuple[str, ...]:
    terms: set[str] = set()
    attr_terms = {
        "out_of_plane_bends": "charmm_out_of_plane_bend_terms",
        "stretch_bends": "charmm_stretch_bend_terms",
        "improper_periodic": "charmm_periodic_improper_terms",
    }
    for attr_name, term_name in attr_terms.items():
        if _parmed_collection_present(getattr(structure, attr_name, None)):
            terms.add(term_name)
    return tuple(sorted(terms))


def _parmed_collection_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _parmed_urey_bradleys(
    urey_bradleys: Iterable[Any],
    *,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int]] = []
    k_values: list[float] = []
    distance_values: list[float] = []
    angle_centers: dict[tuple[int, int], int] = {}
    for i, j, k in np.asarray(angles, dtype=np.int32):
        pair = _normalize_pair(int(i), int(k))
        previous = angle_centers.get(pair)
        if previous is not None and previous != int(j):
            msg = "unsupported_terms:urey_bradley_terms"
            raise TopologyImportError(msg)
        angle_centers[pair] = int(j)
    for term in urey_bradleys:
        i = int(term.atom1.idx)
        k = int(term.atom2.idx)
        pair = _normalize_pair(i, k)
        if pair not in angle_centers:
            msg = "unsupported_terms:urey_bradley_terms"
            raise TopologyImportError(msg)
        term_type = getattr(term, "type", None)
        rows.append((i, angle_centers[pair], k))
        k_values.append(2.0 * float(getattr(term_type, "k", 0.0)) * KCAL_TO_KJ)
        distance_values.append(float(getattr(term_type, "req", 0.0)))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 3)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(distance_values, dtype=np.float32),
    )


def _parmed_cmaps(cmaps: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    grid_indices: list[int] = []
    grids: list[np.ndarray] = []
    grid_ids: dict[int, int] = {}
    for cmap in cmaps:
        cmap_type = getattr(cmap, "type", None)
        if cmap_type is None:
            msg = "unsupported_terms:charmm_cmap_terms"
            raise TopologyImportError(msg)
        grid_id = id(cmap_type)
        grid_index = grid_ids.get(grid_id)
        if grid_index is None:
            resolution = int(getattr(cmap_type, "resolution", 0))
            grid = np.asarray(getattr(cmap_type, "grid", []), dtype=np.float32)
            if resolution < 4 or grid.size != resolution * resolution:
                msg = "unsupported_terms:charmm_cmap_terms"
                raise TopologyImportError(msg)
            grid_index = len(grids)
            grid_ids[grid_id] = grid_index
            grids.append(grid.reshape((resolution, resolution)) * KCAL_TO_KJ)
        atom_ids = [int(getattr(cmap, f"atom{index}").idx) for index in range(1, 6)]
        rows.append(
            (
                atom_ids[0],
                atom_ids[1],
                atom_ids[2],
                atom_ids[3],
                atom_ids[1],
                atom_ids[2],
                atom_ids[3],
                atom_ids[4],
            )
        )
        grid_indices.append(grid_index)
    if grids:
        grid_array = np.stack(grids).astype(np.float32)
    else:
        grid_array = np.empty((0, 0, 0), dtype=np.float32)
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 8)),
        np.asarray(grid_indices, dtype=np.int32),
        grid_array,
    )


def _parmed_nbfix_type_overrides(
    structure: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    overrides: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for atom in getattr(structure, "atoms", []) or []:
        atom_type = getattr(atom, "atom_type", None)
        if atom_type is None:
            continue
        atom_type_name = _parmed_atom_type_name(atom, atom_type)
        for nbfix_attr in ("nbfix", "nbfixes", "nbfix_types"):
            nbfix = getattr(atom_type, nbfix_attr, None)
            if not _parmed_collection_present(nbfix):
                continue
            if not hasattr(nbfix, "items"):
                msg = "unsupported_terms:nbfix_pair_overrides:malformed_entries"
                raise TopologyImportError(msg)
            for partner, raw_values in nbfix.items():
                partner_name = str(partner).strip()
                if not atom_type_name or not partner_name:
                    msg = "unsupported_terms:nbfix_pair_overrides:missing_type_identifier"
                    raise TopologyImportError(msg)
                values = _parmed_nbfix_values(raw_values)
                pair = tuple(sorted((atom_type_name, partner_name)))
                previous = overrides.get(pair)
                if previous is not None and not np.allclose(previous, values, rtol=0.0, atol=1e-7):
                    msg = "unsupported_terms:nbfix_pair_overrides:conflicting_values"
                    raise TopologyImportError(msg)
                overrides[pair] = values
    if not overrides:
        return (
            np.empty((0, 2), dtype=str),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            {},
        )
    pairs: list[tuple[str, str]] = []
    sigma_values: list[float] = []
    epsilon_values: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    for type1, type2 in sorted(overrides):
        rmin, epsilon_kcal, _rmin14, _epsilon14 = overrides[(type1, type2)]
        sigma = rmin * RMIN_TO_SIGMA
        epsilon = abs(epsilon_kcal) * KCAL_TO_KJ
        pairs.append((type1, type2))
        sigma_values.append(sigma)
        epsilon_values.append(epsilon)
        detail_rows.append(
            {
                "type1": type1,
                "type2": type2,
                "sigma": sigma,
                "epsilon": epsilon,
                "source_rmin": rmin,
                "source_epsilon_kcal_per_mol": epsilon_kcal,
            }
        )
    return (
        np.asarray(pairs, dtype=str).reshape((-1, 2)),
        np.asarray(sigma_values, dtype=np.float32),
        np.asarray(epsilon_values, dtype=np.float32),
        {
            "term": "nbfix_pair_overrides",
            "override_count": len(pairs),
            "atom_type_pair_override_count": len(pairs),
            "source": "parmed_atom_type_nbfix",
            "converted_units": {
                "sigma": "angstrom",
                "epsilon": "kilojoule_per_mole",
            },
            "source_units": {
                "rmin": "angstrom",
                "epsilon": "kilocalorie_per_mole",
            },
            "atom_type_pairs": detail_rows,
        },
    )


def _parmed_atom_type_name(atom: Any, atom_type: Any) -> str:
    name = getattr(atom_type, "name", None)
    if name is None or str(name).strip() == "":
        name = getattr(atom, "type", "")
    return str(name).strip()


def _parmed_nbfix_values(raw_values: Any) -> tuple[float, float, float, float]:
    if all(hasattr(raw_values, name) for name in ("rmin", "epsilon")):
        rmin = raw_values.rmin
        epsilon = raw_values.epsilon
        rmin14 = getattr(raw_values, "rmin_14", rmin)
        epsilon14 = getattr(raw_values, "epsilon_14", epsilon)
    else:
        try:
            values = tuple(raw_values)
        except TypeError as err:
            msg = "unsupported_terms:nbfix_pair_overrides:missing_values"
            raise TopologyImportError(msg) from err
        if len(values) != 4:
            msg = "unsupported_terms:nbfix_pair_overrides:missing_values"
            raise TopologyImportError(msg)
        rmin, epsilon, rmin14, epsilon14 = values
    try:
        parsed = tuple(float(value) for value in (rmin, epsilon, rmin14, epsilon14))
    except (TypeError, ValueError) as err:
        msg = "unsupported_terms:nbfix_pair_overrides:missing_values"
        raise TopologyImportError(msg) from err
    if not np.all(np.isfinite(parsed)):
        msg = "unsupported_terms:nbfix_pair_overrides:nonfinite_parameters"
        raise TopologyImportError(msg)
    rmin_value, epsilon_value, rmin14_value, epsilon14_value = parsed
    if rmin_value <= 0.0 or rmin14_value <= 0.0:
        msg = "unsupported_terms:nbfix_pair_overrides:nonpositive_rmin"
        raise TopologyImportError(msg)
    if not np.isclose(rmin_value, rmin14_value, rtol=0.0, atol=1e-7) or not np.isclose(
        epsilon_value,
        epsilon14_value,
        rtol=0.0,
        atol=1e-7,
    ):
        msg = "unsupported_terms:nbfix_pair_overrides:distinct_1_4_values"
        raise TopologyImportError(msg)
    return parsed


def _lipid_mask_from_residues(residue_names: np.ndarray) -> np.ndarray:
    lipid_resnames = {
        "POPC",
        "POPE",
        "POPG",
        "POPS",
        "DOPC",
        "DOPE",
        "DPPC",
        "DMPC",
        "CHL",
        "CHOL",
    }
    return np.asarray([str(name).upper() in lipid_resnames for name in residue_names], dtype=bool)


def _hydrogen_bond_constraints(
    bonds: np.ndarray,
    *,
    symbols: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if bonds.size == 0:
        return empty_indices(2), np.asarray([], dtype=np.float32)
    upper_symbols = np.char.upper(np.asarray(symbols, dtype=str))
    rows = [
        (int(i), int(j))
        for i, j in np.asarray(bonds, dtype=np.int32)
        if upper_symbols[int(i)] == "H" or upper_symbols[int(j)] == "H"
    ]
    if not rows:
        return empty_indices(2), np.asarray([], dtype=np.float32)
    constraints = np.asarray(sorted(set(rows)), dtype=np.int32).reshape((-1, 2))
    distances = _distances(np.asarray(positions, dtype=np.float32), constraints)
    return constraints, distances


def _distances(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.asarray([], dtype=np.float32)
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _term_counts(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    impropers: np.ndarray,
    constraints: np.ndarray,
    nonbonded_exception_pairs: np.ndarray,
) -> dict[str, int]:
    return {
        "bonds": int(np.asarray(bonds).shape[0]),
        "angles": int(np.asarray(angles).shape[0]),
        "dihedrals": int(np.asarray(dihedrals).shape[0]),
        "impropers": int(np.asarray(impropers).shape[0]),
        "constraints": int(np.asarray(constraints).shape[0]),
        "nonbonded_exceptions": int(np.asarray(nonbonded_exception_pairs).shape[0]),
    }


def _parmed_bonds(bonds: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int]] = []
    k_values: list[float] = []
    lengths: list[float] = []
    for bond in bonds:
        rows.append((int(bond.atom1.idx), int(bond.atom2.idx)))
        bond_type = getattr(bond, "type", None)
        k_values.append(2.0 * float(getattr(bond_type, "k", 0.0)) * KCAL_TO_KJ)
        lengths.append(float(getattr(bond_type, "req", 0.0)))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 2)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def _parmed_angles(angles: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int]] = []
    k_values: list[float] = []
    theta_values: list[float] = []
    for angle in angles:
        rows.append((int(angle.atom1.idx), int(angle.atom2.idx), int(angle.atom3.idx)))
        angle_type = getattr(angle, "type", None)
        k_values.append(2.0 * float(getattr(angle_type, "k", 0.0)) * KCAL_TO_KJ)
        theta_values.append(np.deg2rad(float(getattr(angle_type, "theteq", 0.0))))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 3)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(theta_values, dtype=np.float32),
    )


def _parmed_dihedrals(
    dihedrals: Iterable[Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    proper_rows: list[tuple[int, int, int, int]] = []
    proper_k: list[float] = []
    proper_periodicity: list[float] = []
    proper_phase: list[float] = []
    improper_rows: list[tuple[int, int, int, int]] = []
    improper_k: list[float] = []
    improper_periodicity: list[float] = []
    improper_phase: list[float] = []
    for dihedral in dihedrals:
        row = (
            int(dihedral.atom1.idx),
            int(dihedral.atom2.idx),
            int(dihedral.atom3.idx),
            int(dihedral.atom4.idx),
        )
        dtype = getattr(dihedral, "type", None)
        is_improper = bool(getattr(dihedral, "improper", False))
        target_rows = improper_rows if is_improper else proper_rows
        target_k = improper_k if is_improper else proper_k
        target_periodicity = (
            improper_periodicity if is_improper else proper_periodicity
        )
        target_phase = improper_phase if is_improper else proper_phase
        target_rows.append(row)
        target_k.append(float(getattr(dtype, "phi_k", 0.0)) * KCAL_TO_KJ)
        target_periodicity.append(float(getattr(dtype, "per", 1.0)))
        target_phase.append(np.deg2rad(float(getattr(dtype, "phase", 0.0))))
    return (
        np.asarray(proper_rows, dtype=np.int32).reshape((-1, 4)),
        np.asarray(proper_k, dtype=np.float32),
        np.asarray(proper_periodicity, dtype=np.float32),
        np.asarray(proper_phase, dtype=np.float32),
        np.asarray(improper_rows, dtype=np.int32).reshape((-1, 4)),
        np.asarray(improper_k, dtype=np.float32),
        np.asarray(improper_periodicity, dtype=np.float32),
        np.asarray(improper_phase, dtype=np.float32),
    )


__all__ = [
    "CharmmMassPrelude",
    "TopologyImportError",
    "build_charmm_psf_mass_prelude",
    "import_amber_prmtop",
    "import_charmm_with_parmed",
]
