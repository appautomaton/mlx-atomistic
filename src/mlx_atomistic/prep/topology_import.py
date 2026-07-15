"""Import prepared topology/parameter files into MLX-ready artifacts.

The importers in this module do not run molecular dynamics.  They translate
existing all-atom topology/parameter data into the strict artifact schema that
`mlx_atomistic` can validate and execute.
"""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)
from mlx_atomistic.virtual_sites import tip4p_ew_m_site_weights

AMBER_CHARGE_SCALE = 18.2223
KCAL_TO_KJ = 4.184
RMIN_TO_SIGMA = 2 ** (-1.0 / 6.0)
STANDARD_AMBER_14_ELECTROSTATIC_SCALE = 1.0 / 1.2
STANDARD_AMBER_14_LJ_SCALE = 1.0 / 2.0
AMBER_POINTER_IFPERT_INDEX = 20
AMBER_POINTER_IFBOX_INDEX = 27
AMBER_POINTER_IFCAP_INDEX = 29
AMBER_POINTER_NUMEXTRA_INDEX = 30
AMBER_WATER_RESIDUES = {"WAT", "HOH", "TIP3", "TP3", "SOL"}
UNSUPPORTED_AMBER_FLAGS = {
    "LENNARD_JONES_CCOEF": "amber_12_6_4_lj",
    "LENNARD_JONES_14_ACOEF": "amber_chamber_lj14",
    "LENNARD_JONES_14_BCOEF": "amber_chamber_lj14",
    "CHARMM_UREY_BRADLEY": "amber_chamber_urey_bradley",
    "CHARMM_CMAP_COUNT": "amber_chamber_cmap",
    "CMAP_COUNT": "amber_cmap",
    "CMAP_INDEX": "amber_cmap",
}
UNSUPPORTED_AMBER_FLAG_PREFIXES = {
    "CMAP_PARAMETER_": "amber_cmap",
    "CHARMM_CMAP_PARAMETER_": "amber_chamber_cmap",
}


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


@dataclass(frozen=True)
class CharmmPsfAtom:
    """One atom record parsed from a CHARMM PSF."""

    index: int
    segid: str
    residue_id_raw: str
    residue_name: str
    atom_name: str
    atom_type: str
    charge: float
    mass: float


@dataclass(frozen=True)
class CharmmPsfTopology:
    """Native CHARMM PSF topology records for the supported subset."""

    atoms: tuple[CharmmPsfAtom, ...]
    bonds: np.ndarray
    angles: np.ndarray
    dihedrals: np.ndarray
    impropers: np.ndarray
    cmaps: np.ndarray


@dataclass(frozen=True)
class CharmmImproperParameter:
    """One harmonic CHARMM improper parameter."""

    kind: str
    k: float
    periodicity: float
    phase: float


@dataclass(frozen=True)
class CharmmParameterSetNative:
    """Native CHARMM parameter records for the supported subset."""

    bonds: dict[tuple[str, str], tuple[float, float]]
    angles: dict[tuple[str, str, str], tuple[float, float, float | None, float | None]]
    dihedrals: dict[tuple[str, str, str, str], tuple[tuple[float, float, float], ...]]
    impropers: dict[tuple[str, str, str, str], CharmmImproperParameter]
    nonbonded: dict[str, tuple[float, float]]
    nbfix: dict[tuple[str, str], tuple[float, float, float, float]]
    cmap_indices: dict[tuple[str, ...], int]
    cmap_grids: tuple[np.ndarray, ...]


def import_amber_prmtop(
    *,
    prmtop_path: str | Path,
    coords_path: str | Path,
) -> PreparedSystem:
    """Import an AMBER `prmtop` plus `inpcrd`/`rst7` coordinate file."""

    prmtop_path = Path(prmtop_path)
    coords_path = Path(coords_path)
    topology = _read_amber_prmtop(prmtop_path)
    _check_unsupported_amber_records(topology)
    expect_periodic_box = _amber_has_periodic_box(topology)
    positions, velocities, cell_lengths, cell_matrix = _read_amber_restart(
        coords_path,
        expect_periodic_box=expect_periodic_box,
    )

    atom_count = _amber_atom_count(topology)
    atom_names = np.asarray([str(item).strip() for item in topology.values("ATOM_NAME")], dtype=str)
    if positions.shape != (atom_count, 3):
        msg = (
            f"coordinate atom count does not match prmtop: positions={positions.shape[0]}, "
            f"prmtop={atom_count}"
        )
        raise TopologyImportError(msg)
    if velocities.size == 0:
        velocities = np.zeros_like(positions, dtype=np.float32)

    atom_types_raw = topology.optional_values("AMBER_ATOM_TYPE")
    has_amber_atom_types = "AMBER_ATOM_TYPE" in topology.flags
    if has_amber_atom_types:
        atom_types = np.asarray([str(item).strip() for item in atom_types_raw], dtype=str)
    else:
        atom_type_indices_raw = topology.values("ATOM_TYPE_INDEX")
        atom_types = np.asarray(
            [str(item) for item in atom_type_indices_raw],
            dtype=str,
        )
    charges = np.asarray(topology.values("CHARGE"), dtype=np.float32) / AMBER_CHARGE_SCALE
    masses = np.asarray(topology.values("MASS"), dtype=np.float32)
    type_indices = np.asarray(topology.values("ATOM_TYPE_INDEX"), dtype=np.int32)
    _validate_amber_atom_arrays(
        atom_count=atom_count,
        atom_names=atom_names,
        atom_types=atom_types,
        charges=charges,
        masses=masses,
        type_indices=type_indices,
    )
    symbols = np.asarray(
        [
            _infer_symbol(name, atom_type)
            for name, atom_type in zip(atom_names, atom_types, strict=True)
        ],
        dtype=str,
    )
    sigma, epsilon = _amber_lj_self_parameters(topology, type_indices)

    residue_names, residue_ids, chain_ids = _amber_residue_arrays(topology, atom_count)
    virtual_site_parent_atoms, virtual_site_weights, virtual_site_types = (
        _amber_tip4p_ew_virtual_site_arrays(
            atom_names=atom_names,
            residue_names=residue_names,
            residue_ids=residue_ids,
        )
    )
    bonds, bond_k, bond_length = _amber_bonds(topology, atom_count=atom_count)
    angles, angle_k, angle_theta = _amber_angles(topology, atom_count=atom_count)
    dihedrals, dihedral_k, dihedral_periodicity, dihedral_phase, raw_14_scaling = _amber_dihedrals(
        topology,
        atom_count=atom_count,
        improper=False,
    )
    impropers, improper_k, improper_periodicity, improper_phase, _ = _amber_dihedrals(
        topology,
        atom_count=atom_count,
        improper=True,
    )
    exception_pairs, exception_qprod, exception_sigma, exception_epsilon = _amber_exceptions(
        topology=topology,
        bonds=bonds,
        angles=angles,
        excluded_pairs=_amber_excluded_pairs(topology, atom_count=atom_count),
        one_four_scaling=raw_14_scaling,
        charges=charges,
        type_indices=type_indices,
    )
    _validate_amber_lj_outputs(sigma, epsilon, exception_sigma, exception_epsilon)
    if cell_lengths.size == 0:
        cell_lengths, cell_matrix = _amber_box_from_topology(topology)
    if expect_periodic_box:
        _require_valid_amber_periodic_box(cell_lengths, cell_matrix)

    ligand_mask = _ligand_mask_from_residues(residue_names)
    water_mask = _water_mask_from_residues(residue_names)
    ion_mask = _ion_mask_from_residues(residue_names)
    lipid_mask = _lipid_mask_from_residues(residue_names)
    receptor_mask = ~(ligand_mask | water_mask | ion_mask | lipid_mask)
    restraint_mask = receptor_mask & ~ligand_mask
    constraints, constraint_distance = _hydrogen_bond_constraints(
        bonds,
        symbols=symbols,
        bond_lengths=bond_length,
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
    if virtual_site_types.shape[0]:
        supported_terms.append("virtual_site")
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
            "amber_14_exceptions": int(len(raw_14_scaling)),
            "amber_excluded_pairs": int(
                max(0, exception_pairs.shape[0] - len(raw_14_scaling))
            ),
            "virtual_site": int(virtual_site_types.shape[0]),
        }
    )
    negative_lj_pair_policy = _amber_allowed_negative_lj_pair_policy(topology)
    term_details = {
        "nonbonded_exception": {
            "source": "AMBER excluded-atom list and dihedral 1-4 records",
            "amber_14_scaling": _amber_14_scaling_metadata(topology, raw_14_scaling),
        }
    }

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
            "term_counts": term_counts,
            "term_details": term_details,
            "force_field_provenance": "AMBER prmtop/inpcrd import",
            "virtual_sites_present": bool(virtual_site_types.shape[0]),
            **({"water_model": "tip4p_ew"} if virtual_site_types.shape[0] else {}),
            **(
                {"amber_negative_lj_pair_policy": negative_lj_pair_policy}
                if negative_lj_pair_policy
                else {}
            ),
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
        cell_matrix=cell_matrix.astype(np.float32),
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
        virtual_site_parent_atoms=virtual_site_parent_atoms,
        virtual_site_weights=virtual_site_weights,
        virtual_site_types=virtual_site_types,
    )
    try:
        prepared.validate()
    except ValueError as exc:
        msg = "unsupported_terms:amber_malformed_topology"
        raise TopologyImportError(msg) from exc
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


def import_charmm_psf(
    *,
    psf_path: str | Path,
    params: Sequence[str | Path],
    coords_path: str | Path,
    box_path: str | Path | None = None,
) -> PreparedSystem:
    """Import a supported CHARMM PSF/parameter/coordinate bundle natively."""

    psf_path = Path(psf_path)
    coords_path = Path(coords_path)
    parameter_paths = [Path(path) for path in params]
    topology = _read_charmm_psf(psf_path)
    parameters = _read_charmm_parameters(parameter_paths)
    positions, cell_lengths, cell_matrix = _read_charmm_coordinates(coords_path)
    if box_path is not None:
        cell_lengths, cell_matrix = _read_charmm_xsc_box(Path(box_path))

    atom_count = len(topology.atoms)
    if positions.shape != (atom_count, 3):
        msg = (
            f"coordinate atom count does not match PSF: positions={positions.shape[0]}, "
            f"psf={atom_count}"
        )
        raise TopologyImportError(msg)

    atom_names = np.asarray([atom.atom_name for atom in topology.atoms], dtype=str)
    atom_types = np.asarray([atom.atom_type for atom in topology.atoms], dtype=str)
    atom_type_keys = [_charmm_type_key(atom.atom_type) for atom in topology.atoms]
    residue_names = np.asarray([atom.residue_name for atom in topology.atoms], dtype=str)
    residue_ids = _charmm_residue_ids(topology.atoms)
    chain_ids = np.asarray([atom.segid or "A" for atom in topology.atoms], dtype=str)
    masses = np.asarray([atom.mass for atom in topology.atoms], dtype=np.float32)
    charges = np.asarray([atom.charge for atom in topology.atoms], dtype=np.float32)
    source_net_charge = float(math.fsum(atom.charge for atom in topology.atoms))
    stored_net_charge = float(np.sum(charges, dtype=np.float64))
    _check_charmm_unsupported_atom_records(topology.atoms)
    sigma, epsilon = _charmm_nonbonded_atom_parameters(atom_type_keys, parameters)
    symbols = np.asarray(
        [
            _infer_symbol(atom.atom_name, atom.atom_type)
            for atom in topology.atoms
        ],
        dtype=str,
    )
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols.astype(str)) == "H"))
    if hydrogen_count == 0:
        msg = "production CHARMM import found no hydrogens"
        raise TopologyImportError(msg)

    bonds, bond_k, bond_length = _charmm_bond_arrays(topology.bonds, atom_type_keys, parameters)
    angles, angle_k, angle_theta, urey_terms, urey_k, urey_distance = _charmm_angle_arrays(
        topology.angles,
        atom_type_keys,
        parameters,
    )
    dihedrals, dihedral_k, dihedral_periodicity, dihedral_phase = _charmm_dihedral_arrays(
        topology.dihedrals,
        atom_type_keys,
        parameters,
    )
    impropers, improper_k, improper_periodicity, improper_phase = _charmm_improper_arrays(
        topology.impropers,
        atom_type_keys,
        parameters,
    )
    charmm_cmap_terms, charmm_cmap_grid_indices, charmm_cmap_grids = _charmm_cmap_arrays(
        topology.cmaps,
        atom_type_keys,
        parameters,
    )
    nbfix_type_pairs, nbfix_type_sigma, nbfix_type_epsilon, nbfix_details = (
        _charmm_nbfix_type_overrides(parameters, atom_type_keys)
    )

    exception_pairs = np.asarray(
        sorted(_normalized_pairs(bonds) | _pairs_from_angles(angles)),
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
        bond_lengths=bond_length,
    )

    supported_terms = ["nonbonded_lj_coulomb"]
    if bonds.shape[0]:
        supported_terms.append("harmonic_bond")
    if angles.shape[0]:
        supported_terms.append("harmonic_angle")
    if dihedrals.shape[0]:
        supported_terms.append("periodic_dihedral")
    if impropers.shape[0]:
        supported_terms.append("charmm_harmonic_improper")
    if exception_pairs.shape[0]:
        supported_terms.append("nonbonded_exception")
    if constraints.shape[0]:
        supported_terms.append("distance_constraint")
    if urey_terms.shape[0]:
        supported_terms.append("urey_bradley")
    if charmm_cmap_terms.shape[0]:
        supported_terms.append("charmm_cmap_terms")
    if nbfix_type_pairs.shape[0]:
        supported_terms.append("nbfix_pair_overrides")

    term_counts = _term_counts(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        constraints=constraints,
        nonbonded_exception_pairs=exception_pairs,
    )
    term_counts.pop("impropers", None)
    term_counts.update(
        {
            "charmm_harmonic_improper": int(impropers.shape[0]),
            "charmm_cmap_terms": int(charmm_cmap_terms.shape[0]),
            "charmm_cmap_grids": int(charmm_cmap_grids.shape[0]),
            "urey_bradley_terms": int(urey_terms.shape[0]),
            "nbfix_pair_overrides": int(nbfix_type_pairs.shape[0]),
        }
    )
    term_details = {}
    if nbfix_details:
        term_details["nbfix_pair_overrides"] = nbfix_details

    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source={
            "kind": "charmm",
            "parser": "native_charmm_psf",
            "psf_path": str(psf_path),
            "params": [str(path) for path in parameter_paths],
            "coords_path": str(coords_path),
            **({"box_path": str(box_path)} if box_path is not None else {}),
        },
        selections={
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
            "water_atom_count": int(np.count_nonzero(water_mask)),
            "ion_atom_count": int(np.count_nonzero(ion_mask)),
            "lipid_atom_count": int(np.count_nonzero(lipid_mask)),
            "system_charge": source_net_charge,
            "system_charge_source_precision": source_net_charge,
            "system_charge_stored_float32": stored_net_charge,
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="charmm_psf_parameters_native",
        compatibility_report={
            "engine": "mlx_atomistic",
            "parser": "native_charmm_psf",
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
            "source_topology_counts": {
                "atoms": atom_count,
                "bonds": int(topology.bonds.shape[0]),
                "angles": int(topology.angles.shape[0]),
                "proper_dihedrals": int(topology.dihedrals.shape[0]),
                "impropers": int(topology.impropers.shape[0]),
                "cmaps": int(topology.cmaps.shape[0]),
            },
            "force_field_provenance": "native CHARMM PSF/parameter import",
        },
        warnings=[
            "Imported with the native CHARMM parser. MLX generates trajectories; "
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
        reference_positions=positions.astype(np.float32).copy(),
        cell_lengths=cell_lengths.astype(np.float32),
        cell_matrix=cell_matrix.astype(np.float32),
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
        urey_bradley_terms=urey_terms,
        urey_bradley_k=urey_k,
        urey_bradley_distance=urey_distance,
        nbfix_type_pairs=nbfix_type_pairs,
        nbfix_type_sigma=nbfix_type_sigma,
        nbfix_type_epsilon=nbfix_type_epsilon,
    )
    try:
        prepared.validate()
    except ValueError as exc:
        msg = "unsupported_terms:charmm_malformed_topology"
        raise TopologyImportError(msg) from exc
    return prepared


def import_gromacs_top_gro(
    *,
    top_path: str | Path,
    gro_path: str | Path,
) -> PreparedSystem:
    """Import a supported GROMACS `.top` plus `.gro` coordinate pair."""

    from mlx_atomistic.prep.gromacs import import_gromacs_top_gro as _import_gromacs_top_gro

    return _import_gromacs_top_gro(top_path=top_path, gro_path=gro_path)


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


def _read_charmm_psf(path: Path) -> CharmmPsfTopology:
    lines = path.read_text(errors="replace").splitlines()
    if not lines or "PSF" not in lines[0].upper():
        msg = "unsupported_terms:charmm_malformed_psf"
        raise TopologyImportError(msg)
    header = lines[0].upper()
    if any(flag in header for flag in ("DRUDE", "CHEQ")):
        msg = "unsupported_terms:charmm_polarizable_psf"
        raise TopologyImportError(msg)

    section_headers = _charmm_psf_section_headers(lines)
    supported = {"NATOM", "NBOND", "NTHETA", "NPHI", "NIMPHI", "NCRTERM"}
    ignored_metadata = {"NTITLE", "NGRP"}
    tolerated_empty = {"NDON", "NACC", "NNB", "NGRP", "MOLNT", "NUMLP", "NUMLPH"}
    blockers = [
        _charmm_psf_marker_name(marker)
        for marker, count, _line_index in section_headers
        if marker not in supported
        and marker not in ignored_metadata
        and not (marker in tolerated_empty and count == 0)
    ]
    if blockers:
        msg = "unsupported_terms:" + ",".join(
            f"charmm_psf_{blocker.lower()}" for blocker in sorted(set(blockers))
        )
        raise TopologyImportError(msg)

    atoms = _read_charmm_psf_atoms(lines)
    atom_count = len(atoms)
    return CharmmPsfTopology(
        atoms=tuple(atoms),
        bonds=_read_charmm_psf_indices(lines, marker="NBOND", width=2, atom_count=atom_count),
        angles=_read_charmm_psf_indices(lines, marker="NTHETA", width=3, atom_count=atom_count),
        dihedrals=_read_charmm_psf_indices(lines, marker="NPHI", width=4, atom_count=atom_count),
        impropers=_read_charmm_psf_indices(lines, marker="NIMPHI", width=4, atom_count=atom_count),
        cmaps=_read_charmm_psf_indices(lines, marker="NCRTERM", width=8, atom_count=atom_count),
    )


def _charmm_psf_section_headers(lines: Sequence[str]) -> list[tuple[str, int, int]]:
    headers: list[tuple[str, int, int]] = []
    for index, raw_line in enumerate(lines):
        if "!" not in raw_line:
            continue
        left, right = raw_line.split("!", maxsplit=1)
        fields = left.split()
        if not fields:
            continue
        try:
            count = int(fields[0])
        except ValueError:
            continue
        marker = _charmm_psf_marker_name(right.split(":", maxsplit=1)[0].split()[0])
        headers.append((marker, count, index))
    return headers


def _charmm_psf_marker_name(value: str) -> str:
    return value.strip().upper().lstrip("!")


def _read_charmm_psf_atoms(lines: Sequence[str]) -> list[CharmmPsfAtom]:
    for marker, atom_count, line_index in _charmm_psf_section_headers(lines):
        if marker != "NATOM":
            continue
        if atom_count <= 0:
            msg = "unsupported_terms:charmm_malformed_psf"
            raise TopologyImportError(msg)
        atoms: list[CharmmPsfAtom] = []
        for raw_line in lines[line_index + 1 : line_index + 1 + atom_count]:
            fields = raw_line.split()
            if len(fields) < 8:
                msg = "unsupported_terms:charmm_malformed_psf"
                raise TopologyImportError(msg)
            try:
                atom_index = int(fields[0]) - 1
                charge = float(fields[6])
                mass = float(fields[7])
            except ValueError as exc:
                msg = "unsupported_terms:charmm_malformed_psf"
                raise TopologyImportError(msg) from exc
            _validate_charmm_finite(
                "unsupported_terms:charmm_malformed_psf_atom_parameters",
                charge,
                mass,
            )
            if atom_index != len(atoms):
                msg = "unsupported_terms:charmm_malformed_psf"
                raise TopologyImportError(msg)
            atoms.append(
                CharmmPsfAtom(
                    index=atom_index,
                    segid=fields[1],
                    residue_id_raw=fields[2],
                    residue_name=fields[3],
                    atom_name=fields[4],
                    atom_type=fields[5],
                    charge=charge,
                    mass=mass,
                )
            )
        return atoms
    msg = "unsupported_terms:charmm_malformed_psf"
    raise TopologyImportError(msg)


def _read_charmm_psf_indices(
    lines: Sequence[str],
    *,
    marker: str,
    width: int,
    atom_count: int,
) -> np.ndarray:
    for section_marker, count, line_index in _charmm_psf_section_headers(lines):
        if section_marker != marker:
            continue
        if count == 0:
            return empty_indices(width)
        raw_values: list[int] = []
        target = count * width
        for raw_line in lines[line_index + 1 :]:
            if "!" in raw_line and raw_line.split("!", maxsplit=1)[0].strip().split()[:1]:
                left = raw_line.split("!", maxsplit=1)[0].split()[0]
                if left.isdigit() and len(raw_values) >= target:
                    break
            for field in raw_line.split("!", maxsplit=1)[0].split():
                try:
                    raw_values.append(int(field))
                except ValueError as exc:
                    msg = f"unsupported_terms:charmm_malformed_psf_{marker.lower()}"
                    raise TopologyImportError(msg) from exc
                if len(raw_values) == target:
                    break
            if len(raw_values) == target:
                break
        if len(raw_values) != target:
            msg = f"unsupported_terms:charmm_malformed_psf_{marker.lower()}"
            raise TopologyImportError(msg)
        array = np.asarray(raw_values, dtype=np.int32).reshape((count, width)) - 1
        if np.any(array < 0) or np.any(array >= atom_count):
            msg = f"unsupported_terms:charmm_malformed_psf_{marker.lower()}"
            raise TopologyImportError(msg)
        return array
    return empty_indices(width)


def _check_charmm_unsupported_atom_records(atoms: Sequence[CharmmPsfAtom]) -> None:
    unsupported_water_models = {"TIP4", "TIP4P", "TIP5", "TIP5P", "SWM4", "SWM4NDP", "OPC"}
    virtual_markers = {"LP", "LP1", "LP2", "EP", "EPW", "DUM", "DUMMY", "VS"}
    blockers: set[str] = set()
    for atom in atoms:
        residue = atom.residue_name.upper()
        atom_name = atom.atom_name.upper()
        atom_type = atom.atom_type.upper()
        if residue in unsupported_water_models:
            blockers.add("charmm_unsupported_water_model")
        if atom.mass <= 0.0 or atom_name in virtual_markers or atom_type in virtual_markers:
            blockers.add("charmm_virtual_sites")
    if blockers:
        msg = "unsupported_terms:" + ",".join(sorted(blockers))
        raise TopologyImportError(msg)


def _charmm_residue_ids(atoms: Sequence[CharmmPsfAtom]) -> np.ndarray:
    ids: list[int] = []
    fallback_ids: dict[tuple[str, str, str], int] = {}
    for atom in atoms:
        raw = atom.residue_id_raw.strip()
        try:
            ids.append(int(raw))
            continue
        except ValueError:
            key = (atom.segid, raw, atom.residue_name)
            if key not in fallback_ids:
                fallback_ids[key] = len(fallback_ids) + 1
            ids.append(fallback_ids[key])
    return np.asarray(ids, dtype=np.int32)


def _read_charmm_parameters(paths: Sequence[Path]) -> CharmmParameterSetNative:
    bonds: dict[tuple[str, str], tuple[float, float]] = {}
    angles: dict[tuple[str, str, str], tuple[float, float, float | None, float | None]] = {}
    dihedrals: dict[tuple[str, str, str, str], list[tuple[float, float, float]]] = {}
    impropers: dict[tuple[str, str, str, str], CharmmImproperParameter] = {}
    nonbonded: dict[str, tuple[float, float]] = {}
    nbfix: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    cmap_indices: dict[tuple[str, ...], int] = {}
    cmap_grids: list[np.ndarray] = []

    for path in paths:
        lines = [
            _strip_charmm_comment(line)
            for line in path.read_text(errors="replace").splitlines()
        ]
        section = ""
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            index += 1
            if not line or line.startswith("*"):
                continue
            fields = line.split()
            keyword = fields[0].upper()
            maybe_section = _charmm_parameter_section(keyword)
            if maybe_section is not None:
                section = maybe_section
                if section in {"LONEPAIR", "DRUDE", "THOLE", "ANISOTROPY"}:
                    msg = f"unsupported_terms:charmm_parameter_{section.lower()}"
                    raise TopologyImportError(msg)
                if section == "HBOND":
                    if len(fields) == 1:
                        continue
                    msg = "unsupported_terms:charmm_parameter_hbond"
                    raise TopologyImportError(msg)
                if keyword == "MASS" and len(fields) >= 4:
                    continue
                if len(fields) == 1 or section in {"CMAP", "NONBONDED", "NBFIX", "HBOND"}:
                    continue

            if keyword == "MASS":
                continue
            if section == "BONDS":
                key, value = _parse_charmm_bond_parameter(fields)
                bonds[key] = value
            elif section == "ANGLES":
                key, value = _parse_charmm_angle_parameter(fields)
                angles[key] = value
            elif section == "DIHEDRALS":
                key, value = _parse_charmm_dihedral_parameter(fields)
                dihedrals.setdefault(key, []).append(value)
            elif section == "IMPROPERS":
                key, value = _parse_charmm_improper_parameter(fields)
                impropers[key] = value
            elif section == "NONBONDED":
                parsed = _parse_charmm_nonbonded_parameter(fields)
                if parsed is not None:
                    key, value = parsed
                    nonbonded[key] = value
            elif section == "NBFIX":
                key, value = _parse_charmm_nbfix_parameter(fields)
                nbfix[key] = value
            elif section == "CMAP":
                key, grid, next_index = _parse_charmm_cmap_parameter(fields, lines, index)
                index = next_index
                cmap_indices[key] = len(cmap_grids)
                cmap_grids.append(grid)
            elif section == "HBOND":
                msg = "unsupported_terms:charmm_parameter_hbond"
                raise TopologyImportError(msg)
            else:
                msg = f"unsupported_terms:charmm_parameter_record:{keyword.lower()}"
                raise TopologyImportError(msg)

    return CharmmParameterSetNative(
        bonds=bonds,
        angles=angles,
        dihedrals={key: tuple(values) for key, values in dihedrals.items()},
        impropers=impropers,
        nonbonded=nonbonded,
        nbfix=nbfix,
        cmap_indices=cmap_indices,
        cmap_grids=tuple(cmap_grids),
    )


def _strip_charmm_comment(line: str) -> str:
    return line.split("!", maxsplit=1)[0].strip()


def _charmm_parameter_section(keyword: str) -> str | None:
    sections = {
        "BOND": "BONDS",
        "BONDS": "BONDS",
        "ANGL": "ANGLES",
        "ANGLE": "ANGLES",
        "ANGLES": "ANGLES",
        "THETAS": "ANGLES",
        "DIHE": "DIHEDRALS",
        "DIHEDRAL": "DIHEDRALS",
        "DIHEDRALS": "DIHEDRALS",
        "PHI": "DIHEDRALS",
        "IMPHI": "IMPROPERS",
        "IMPROPER": "IMPROPERS",
        "IMPROPERS": "IMPROPERS",
        "CMAP": "CMAP",
        "NONBONDED": "NONBONDED",
        "NBOND": "NONBONDED",
        "NBFIX": "NBFIX",
        "HBOND": "HBOND",
        "END": "END",
        "RETURN": "END",
        "MASS": "MASS",
        "LONEPAIR": "LONEPAIR",
        "DRUDE": "DRUDE",
        "THOLE": "THOLE",
        "ANISOTROPY": "ANISOTROPY",
    }
    return sections.get(keyword)


def _parse_charmm_bond_parameter(
    fields: Sequence[str],
) -> tuple[tuple[str, str], tuple[float, float]]:
    if len(fields) < 4:
        msg = "unsupported_terms:charmm_malformed_bond_parameters"
        raise TopologyImportError(msg)
    try:
        k = 2.0 * float(fields[2]) * KCAL_TO_KJ
        distance = float(fields[3])
    except ValueError as exc:
        msg = "unsupported_terms:charmm_malformed_bond_parameters"
        raise TopologyImportError(msg) from exc
    _validate_charmm_finite("unsupported_terms:charmm_malformed_bond_parameters", k, distance)
    if k < 0.0 or distance <= 0.0:
        msg = "unsupported_terms:charmm_invalid_bond_parameters"
        raise TopologyImportError(msg)
    return (_charmm_type_key(fields[0]), _charmm_type_key(fields[1])), (k, distance)


def _parse_charmm_angle_parameter(
    fields: Sequence[str],
) -> tuple[tuple[str, str, str], tuple[float, float, float | None, float | None]]:
    if len(fields) < 5:
        msg = "unsupported_terms:charmm_malformed_angle_parameters"
        raise TopologyImportError(msg)
    try:
        k = 2.0 * float(fields[3]) * KCAL_TO_KJ
        theta = np.deg2rad(float(fields[4]))
        urey_k = 2.0 * float(fields[5]) * KCAL_TO_KJ if len(fields) >= 7 else None
        urey_distance = float(fields[6]) if len(fields) >= 7 else None
    except ValueError as exc:
        msg = "unsupported_terms:charmm_malformed_angle_parameters"
        raise TopologyImportError(msg) from exc
    values = [k, theta]
    if urey_k is not None and urey_distance is not None:
        values.extend([urey_k, urey_distance])
    _validate_charmm_finite("unsupported_terms:charmm_malformed_angle_parameters", *values)
    if k < 0.0:
        msg = "unsupported_terms:charmm_invalid_angle_parameters"
        raise TopologyImportError(msg)
    return (
        (_charmm_type_key(fields[0]), _charmm_type_key(fields[1]), _charmm_type_key(fields[2])),
        (k, float(theta), urey_k, urey_distance),
    )


def _parse_charmm_dihedral_parameter(
    fields: Sequence[str],
) -> tuple[tuple[str, str, str, str], tuple[float, float, float]]:
    if len(fields) < 7:
        msg = "unsupported_terms:charmm_malformed_dihedral_parameters"
        raise TopologyImportError(msg)
    try:
        k = float(fields[4]) * KCAL_TO_KJ
        periodicity = float(fields[5])
        phase = np.deg2rad(float(fields[6]))
    except ValueError as exc:
        msg = "unsupported_terms:charmm_malformed_dihedral_parameters"
        raise TopologyImportError(msg) from exc
    _validate_charmm_finite(
        "unsupported_terms:charmm_malformed_dihedral_parameters",
        k,
        periodicity,
        phase,
    )
    return tuple(_charmm_type_key(field) for field in fields[:4]), (k, periodicity, float(phase))


def _parse_charmm_improper_parameter(
    fields: Sequence[str],
) -> tuple[tuple[str, str, str, str], CharmmImproperParameter]:
    if len(fields) < 6:
        msg = "unsupported_terms:charmm_malformed_improper_parameters"
        raise TopologyImportError(msg)
    try:
        k = float(fields[4]) * KCAL_TO_KJ
        # CHARMM impropers use a harmonic form.  Seven-column parameter
        # records retain a historical zero field before the equilibrium
        # angle; it is not a periodicity declaration.
        periodicity = 0.0
        phase = np.deg2rad(float(fields[-1]))
        kind = "harmonic"
    except ValueError as exc:
        msg = "unsupported_terms:charmm_malformed_improper_parameters"
        raise TopologyImportError(msg) from exc
    _validate_charmm_finite(
        "unsupported_terms:charmm_malformed_improper_parameters",
        k,
        periodicity,
        phase,
    )
    return (
        tuple(_charmm_type_key(field) for field in fields[:4]),
        CharmmImproperParameter(kind=kind, k=k, periodicity=periodicity, phase=float(phase)),
    )


def _parse_charmm_nonbonded_parameter(
    fields: Sequence[str],
) -> tuple[str, tuple[float, float]] | None:
    if len(fields) < 4:
        return None
    try:
        epsilon_kcal = float(fields[2])
        rmin_half = float(fields[3])
    except ValueError:
        return None
    sigma = 2.0 * rmin_half * RMIN_TO_SIGMA
    epsilon = abs(epsilon_kcal) * KCAL_TO_KJ
    _validate_charmm_finite(
        "unsupported_terms:charmm_malformed_nonbonded_parameters",
        sigma,
        epsilon,
    )
    return _charmm_type_key(fields[0]), (sigma, epsilon)


def _parse_charmm_nbfix_parameter(
    fields: Sequence[str],
) -> tuple[tuple[str, str], tuple[float, float, float, float]]:
    if len(fields) not in {4, 6}:
        msg = "unsupported_terms:nbfix_pair_overrides:malformed_entries"
        raise TopologyImportError(msg)
    try:
        epsilon = float(fields[2])
        rmin = float(fields[3])
        epsilon14 = float(fields[4]) if len(fields) >= 6 else epsilon
        rmin14 = float(fields[5]) if len(fields) >= 6 else rmin
    except ValueError as exc:
        msg = "unsupported_terms:nbfix_pair_overrides:malformed_entries"
        raise TopologyImportError(msg) from exc
    values = (rmin, epsilon, rmin14, epsilon14)
    _validate_charmm_finite("unsupported_terms:nbfix_pair_overrides:missing_values", *values)
    if rmin <= 0.0 or rmin14 <= 0.0:
        msg = "unsupported_terms:nbfix_pair_overrides:nonpositive_rmin"
        raise TopologyImportError(msg)
    if not np.isclose(rmin, rmin14, rtol=0.0, atol=1e-7) or not np.isclose(
        epsilon,
        epsilon14,
        rtol=0.0,
        atol=1e-7,
    ):
        msg = "unsupported_terms:nbfix_pair_overrides:distinct_1_4_values"
        raise TopologyImportError(msg)
    key = tuple(sorted((_charmm_type_key(fields[0]), _charmm_type_key(fields[1]))))
    return key, values


def _parse_charmm_cmap_parameter(
    fields: Sequence[str],
    lines: Sequence[str],
    next_index: int,
) -> tuple[tuple[str, ...], np.ndarray, int]:
    if len(fields) < 9:
        msg = "unsupported_terms:charmm_cmap_terms"
        raise TopologyImportError(msg)
    try:
        resolution = int(fields[8])
    except ValueError as exc:
        msg = "unsupported_terms:charmm_cmap_terms"
        raise TopologyImportError(msg) from exc
    if resolution < 4:
        msg = "unsupported_terms:charmm_cmap_terms"
        raise TopologyImportError(msg)
    values = _charmm_float_fields(fields[9:])
    index = next_index
    target = resolution * resolution
    while len(values) < target and index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("*"):
            continue
        values.extend(_charmm_float_fields(line.split()))
    if len(values) != target:
        msg = "unsupported_terms:charmm_cmap_terms"
        raise TopologyImportError(msg)
    grid_values = np.asarray(values, dtype=np.float64) * KCAL_TO_KJ
    _validate_charmm_finite("unsupported_terms:charmm_cmap_terms", grid_values)
    grid = grid_values.astype(np.float32).reshape((resolution, resolution))
    return tuple(_charmm_type_key(field) for field in fields[:8]), grid, index


def _charmm_float_fields(fields: Sequence[str]) -> list[float]:
    values: list[float] = []
    for field in fields:
        try:
            values.append(float(field.replace("D", "E").replace("d", "e")))
        except ValueError as exc:
            msg = "unsupported_terms:charmm_malformed_parameter_value"
            raise TopologyImportError(msg) from exc
    return values


def _validate_charmm_finite(blocker: str, *values: float) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise TopologyImportError(blocker)


def _charmm_type_key(atom_type: str) -> str:
    return str(atom_type).strip().upper()


def _charmm_nonbonded_atom_parameters(
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray]:
    sigma: list[float] = []
    epsilon: list[float] = []
    for atom_type in atom_type_keys:
        values = parameters.nonbonded.get(atom_type)
        if values is None:
            msg = f"unsupported_terms:charmm_missing_nonbonded_parameter:{atom_type}"
            raise TopologyImportError(msg)
        if values[0] <= 0.0:
            msg = "unsupported_terms:charmm_malformed_nonbonded_parameters"
            raise TopologyImportError(msg)
        sigma.append(values[0])
        epsilon.append(values[1])
    return np.asarray(sigma, dtype=np.float32), np.asarray(epsilon, dtype=np.float32)


def _charmm_bond_arrays(
    bonds: np.ndarray,
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_values: list[float] = []
    lengths: list[float] = []
    lookup_cache: dict[tuple[str, str], tuple[float, float]] = {}
    for i, j in np.asarray(bonds, dtype=np.int32):
        query = (atom_type_keys[int(i)], atom_type_keys[int(j)])
        values = lookup_cache.get(query)
        if values is None:
            values = _charmm_lookup_parameter(
                parameters.bonds,
                query,
                blocker="unsupported_terms:charmm_missing_bond_parameter",
                reversible=True,
            )
            lookup_cache[query] = values
        k, distance = values
        k_values.append(k)
        lengths.append(distance)
    return (
        np.asarray(bonds, dtype=np.int32).reshape((-1, 2)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def _charmm_angle_arrays(
    angles: np.ndarray,
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_values: list[float] = []
    theta_values: list[float] = []
    urey_terms: list[tuple[int, int, int]] = []
    urey_k_values: list[float] = []
    urey_distances: list[float] = []
    lookup_cache: dict[
        tuple[str, str, str],
        tuple[float, float, float | None, float | None],
    ] = {}
    for i, j, k in np.asarray(angles, dtype=np.int32):
        query = (atom_type_keys[int(i)], atom_type_keys[int(j)], atom_type_keys[int(k)])
        values = lookup_cache.get(query)
        if values is None:
            values = _charmm_lookup_parameter(
                parameters.angles,
                query,
                blocker="unsupported_terms:charmm_missing_angle_parameter",
                reversible=True,
            )
            lookup_cache[query] = values
        angle_k, theta, urey_k, urey_distance = values
        k_values.append(angle_k)
        theta_values.append(theta)
        if urey_k is not None and urey_distance is not None:
            if urey_k < 0.0 or urey_distance <= 0.0:
                msg = "unsupported_terms:charmm_invalid_urey_bradley_parameters"
                raise TopologyImportError(msg)
            urey_terms.append((int(i), int(j), int(k)))
            urey_k_values.append(urey_k)
            urey_distances.append(urey_distance)
    return (
        np.asarray(angles, dtype=np.int32).reshape((-1, 3)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(theta_values, dtype=np.float32),
        np.asarray(urey_terms, dtype=np.int32).reshape((-1, 3)),
        np.asarray(urey_k_values, dtype=np.float32),
        np.asarray(urey_distances, dtype=np.float32),
    )


def _charmm_dihedral_arrays(
    dihedrals: np.ndarray,
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int, int]] = []
    k_values: list[float] = []
    periodicities: list[float] = []
    phases: list[float] = []
    lookup_cache: dict[
        tuple[str, str, str, str],
        tuple[tuple[float, float, float], ...],
    ] = {}
    for i, j, k, m in np.asarray(dihedrals, dtype=np.int32):
        query = (
            atom_type_keys[int(i)],
            atom_type_keys[int(j)],
            atom_type_keys[int(k)],
            atom_type_keys[int(m)],
        )
        records = lookup_cache.get(query)
        if records is None:
            records = _charmm_lookup_parameter(
                parameters.dihedrals,
                query,
                blocker="unsupported_terms:charmm_missing_dihedral_parameter",
                reversible=True,
            )
            lookup_cache[query] = records
        for force_constant, periodicity, phase in records:
            rows.append((int(i), int(j), int(k), int(m)))
            k_values.append(force_constant)
            periodicities.append(periodicity)
            phases.append(phase)
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 4)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(periodicities, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
    )


def _charmm_improper_arrays(
    impropers: np.ndarray,
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int, int]] = []
    k_values: list[float] = []
    periodicities: list[float] = []
    phases: list[float] = []
    lookup_cache: dict[tuple[str, str, str, str], CharmmImproperParameter] = {}
    for i, j, k, m in np.asarray(impropers, dtype=np.int32):
        query = (
            atom_type_keys[int(i)],
            atom_type_keys[int(j)],
            atom_type_keys[int(k)],
            atom_type_keys[int(m)],
        )
        parameter = lookup_cache.get(query)
        if parameter is None:
            parameter = _charmm_lookup_parameter(
                parameters.impropers,
                query,
                blocker="unsupported_terms:charmm_missing_improper_parameter",
                reversible=True,
            )
            lookup_cache[query] = parameter
        rows.append((int(i), int(j), int(k), int(m)))
        k_values.append(parameter.k)
        periodicities.append(parameter.periodicity)
        phases.append(parameter.phase)
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 4)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(periodicities, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
    )


def _charmm_cmap_arrays(
    cmaps: np.ndarray,
    atom_type_keys: Sequence[str],
    parameters: CharmmParameterSetNative,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    grid_indices: list[int] = []
    for row in np.asarray(cmaps, dtype=np.int32):
        query = tuple(atom_type_keys[int(index)] for index in row.tolist())
        grid_index = _charmm_lookup_parameter(
            parameters.cmap_indices,
            query,
            blocker="unsupported_terms:charmm_missing_cmap_parameter",
            reversible=True,
        )
        rows.append(tuple(int(index) for index in row.tolist()))
        grid_indices.append(int(grid_index))
    if parameters.cmap_grids:
        grid_shapes = {grid.shape for grid in parameters.cmap_grids}
        if len(grid_shapes) != 1:
            msg = "unsupported_terms:charmm_cmap_terms:mixed_grid_resolution"
            raise TopologyImportError(msg)
        grid_array = np.stack(parameters.cmap_grids).astype(np.float32)
    else:
        grid_array = np.empty((0, 0, 0), dtype=np.float32)
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 8)),
        np.asarray(grid_indices, dtype=np.int32),
        grid_array,
    )


def _charmm_nbfix_type_overrides(
    parameters: CharmmParameterSetNative,
    atom_type_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    present_types = set(atom_type_keys)
    selected = {
        pair: values
        for pair, values in parameters.nbfix.items()
        if pair[0] in present_types and pair[1] in present_types
    }
    if not selected:
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
    for type1, type2 in sorted(selected):
        rmin, epsilon_kcal, _rmin14, _epsilon14 = selected[(type1, type2)]
        sigma = rmin * RMIN_TO_SIGMA
        epsilon = abs(epsilon_kcal) * KCAL_TO_KJ
        _validate_charmm_finite(
            "unsupported_terms:nbfix_pair_overrides:malformed_entries",
            sigma,
            epsilon,
        )
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
            "source": "charmm_parameter_nbfix",
            "source_parameter_override_count": int(len(parameters.nbfix)),
            "applicable_override_count": len(pairs),
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


def _charmm_lookup_parameter(
    table: dict[tuple[str, ...], Any],
    query: tuple[str, ...],
    *,
    blocker: str,
    reversible: bool,
) -> Any:
    candidates = [query]
    if reversible:
        reverse = tuple(reversed(query))
        if reverse != query:
            candidates.append(reverse)
    for candidate in candidates:
        if candidate in table:
            return table[candidate]
    wildcard_matches: list[tuple[int, Any]] = []
    for key, value in table.items():
        for candidate in candidates:
            if _charmm_parameter_key_matches(key, candidate):
                specificity = sum(part != "X" for part in key)
                wildcard_matches.append((specificity, value))
                break
    if wildcard_matches:
        wildcard_matches.sort(key=lambda item: item[0], reverse=True)
        return wildcard_matches[0][1]
    msg = f"{blocker}:{'-'.join(query)}"
    raise TopologyImportError(msg)


def _charmm_parameter_key_matches(key: tuple[str, ...], query: tuple[str, ...]) -> bool:
    return len(key) == len(query) and all(
        key_part == "X" or key_part == query_part
        for key_part, query_part in zip(key, query, strict=True)
    )


def _read_charmm_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    if any(line.startswith(("ATOM", "HETATM", "CRYST1")) for line in lines):
        return _read_charmm_pdb_coordinates(lines)
    return _read_charmm_crd_coordinates(lines)


def _read_charmm_pdb_coordinates(
    lines: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    cell_lengths = np.asarray([], dtype=np.float32)
    cell_matrix = np.asarray([], dtype=np.float32)
    for line in lines:
        if line.startswith("CRYST1"):
            cell_lengths, cell_matrix = _read_charmm_pdb_cryst1(line)
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            if len(line) >= 54:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            else:
                fields = line.split()
                x, y, z = (float(value) for value in fields[6:9])
        except (ValueError, IndexError) as exc:
            msg = "unsupported_terms:charmm_malformed_coordinates"
            raise TopologyImportError(msg) from exc
        positions.append((x, y, z))
    if not positions:
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg)
    positions_array = np.asarray(positions, dtype=np.float32)
    if not np.all(np.isfinite(positions_array)):
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg)
    return positions_array, cell_lengths, cell_matrix


def _read_charmm_pdb_cryst1(line: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        if len(line) >= 54:
            lengths = np.asarray(
                [float(line[6:15]), float(line[15:24]), float(line[24:33])],
                dtype=np.float32,
            )
            angles = np.asarray(
                [float(line[33:40]), float(line[40:47]), float(line[47:54])],
                dtype=np.float32,
            )
        else:
            fields = line.split()
            lengths = np.asarray([float(value) for value in fields[1:4]], dtype=np.float32)
            angles = np.asarray([float(value) for value in fields[4:7]], dtype=np.float32)
    except (ValueError, IndexError) as exc:
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg) from exc
    return _charmm_cell_from_lengths_angles(lengths, angles)


def _read_charmm_crd_coordinates(
    lines: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("*")
    ]
    if not data_lines:
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg)
    try:
        atom_count = int(data_lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg) from exc
    positions: list[tuple[float, float, float]] = []
    for line in data_lines[1:]:
        fields = line.split()
        if len(fields) < 7:
            msg = "unsupported_terms:charmm_malformed_coordinates"
            raise TopologyImportError(msg)
        try:
            positions.append((float(fields[4]), float(fields[5]), float(fields[6])))
        except (ValueError, IndexError) as exc:
            msg = "unsupported_terms:charmm_malformed_coordinates"
            raise TopologyImportError(msg) from exc
    if len(positions) != atom_count:
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg)
    positions_array = np.asarray(positions, dtype=np.float32)
    if not np.all(np.isfinite(positions_array)):
        msg = "unsupported_terms:charmm_malformed_coordinates"
        raise TopologyImportError(msg)
    return positions_array, np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)


def _read_charmm_xsc_box(path: Path) -> tuple[np.ndarray, np.ndarray]:
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            vector_values = [float(value) for value in fields[1:10]]
        except ValueError:
            continue
        matrix = np.asarray(vector_values, dtype=np.float32).reshape((3, 3))
        return _validate_charmm_cell_matrix(matrix)
    msg = "unsupported_terms:charmm_invalid_periodic_box"
    raise TopologyImportError(msg)


def _charmm_cell_from_lengths_angles(
    lengths: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    if lengths.shape != (3,) or angles.shape != (3,):
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(angles)):
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    if np.any(lengths <= 0.0):
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    if np.allclose(angles, np.asarray([90.0, 90.0, 90.0], dtype=np.float32), atol=1e-5):
        return lengths.astype(np.float32), np.asarray([], dtype=np.float32)
    try:
        matrix = _cell_matrix_from_lengths_angles(lengths, angles)
    except TopologyImportError as exc:
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg) from exc
    return _validate_charmm_cell_matrix(matrix)


def _validate_charmm_cell_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    lengths = np.linalg.norm(matrix, axis=1).astype(np.float32)
    determinant = float(np.linalg.det(matrix.astype(np.float64)))
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    if not np.isfinite(determinant) or determinant <= 0.0:
        msg = "unsupported_terms:charmm_invalid_periodic_box"
        raise TopologyImportError(msg)
    return lengths, matrix


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
        bond_lengths=bond_length,
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
        if _format_kind(current_format) == "a":
            width = _format_width(current_format, default=4)
            data[current_flag].extend(
                chunk.strip() for chunk in _fixed_width_chunks(line, width) if chunk.strip()
            )
        else:
            data[current_flag].extend(_numeric_or_split_values(line, current_format))

    parsed: dict[str, list[str] | list[int] | list[float]] = {}
    for flag, values in data.items():
        fmt = formats.get(flag, "")
        kind = _format_kind(fmt)
        try:
            if kind == "a":
                parsed[flag] = values
            elif kind == "i":
                parsed[flag] = [int(value) for value in values]
            else:
                parsed[flag] = [
                    float(value.replace("D", "E").replace("d", "e")) for value in values
                ]
        except ValueError as exc:
            msg = "unsupported_terms:amber_malformed_topology"
            raise TopologyImportError(msg) from exc
    return AmberPrmtop(flags=parsed, formats=formats)


def _check_unsupported_amber_records(topology: AmberPrmtop) -> None:
    blockers: set[str] = set()
    pointers = [int(value) for value in topology.optional_values("POINTERS")]
    has_tip4p_ew_sites = _amber_has_tip4p_ew_virtual_sites(topology)
    if _amber_pointer_value(pointers, AMBER_POINTER_IFPERT_INDEX) > 0:
        blockers.add("amber_perturbation")
    if _amber_pointer_value(pointers, AMBER_POINTER_IFCAP_INDEX) > 0:
        blockers.add("amber_cap")
    if _amber_pointer_value(pointers, AMBER_POINTER_NUMEXTRA_INDEX) > 0 and not has_tip4p_ew_sites:
        blockers.add("amber_extra_points")
    if _amber_pointer_value(pointers, AMBER_POINTER_IFBOX_INDEX) not in {0, 1, 2}:
        blockers.add("amber_unknown_box_type")
    for flag, blocker in UNSUPPORTED_AMBER_FLAGS.items():
        if flag in topology.flags:
            blockers.add(blocker)
    for flag in topology.flags:
        for prefix, blocker in UNSUPPORTED_AMBER_FLAG_PREFIXES.items():
            if flag.startswith(prefix):
                blockers.add(blocker)
    hbond_acoef = np.asarray(topology.optional_values("HBOND_ACOEF"), dtype=np.float64)
    hbond_bcoef = np.asarray(topology.optional_values("HBOND_BCOEF"), dtype=np.float64)
    if (hbond_acoef.size and np.any(np.abs(hbond_acoef) > 0.0)) or (
        hbond_bcoef.size and np.any(np.abs(hbond_bcoef) > 0.0)
    ):
        blockers.add("amber_10_12_hbond")
    if blockers:
        msg = "unsupported_terms:" + ",".join(sorted(blockers))
        raise TopologyImportError(msg)


def _amber_pointer_value(pointers: Sequence[int], index: int) -> int:
    if len(pointers) <= index:
        return 0
    return int(pointers[index])


def _amber_has_tip4p_ew_virtual_sites(topology: AmberPrmtop) -> bool:
    try:
        atom_names = np.asarray(
            [str(item).strip() for item in topology.values("ATOM_NAME")],
            dtype=str,
        )
        atom_count = _amber_atom_count(topology)
        residue_names, residue_ids, _ = _amber_residue_arrays(topology, atom_count)
    except TopologyImportError:
        return False
    parents, _, _ = _amber_tip4p_ew_virtual_site_arrays(
        atom_names=atom_names,
        residue_names=residue_names,
        residue_ids=residue_ids,
    )
    return bool(parents.shape[0])


def _amber_tip4p_ew_virtual_site_arrays(
    *,
    atom_names: np.ndarray,
    residue_names: np.ndarray,
    residue_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect explicit TIP4P-Ew water sites in AMBER-style atom/residue arrays."""

    tip4p_residues = {"TIP4P", "TIP4", "TP4E", "T4E", "WAT", "HOH"}
    oxygen_names = {"O", "OW", "OH2"}
    hydrogen_names = {"H", "H1", "H2", "HW1", "HW2", "H01", "H02"}
    site_names = {"M", "MW", "EP", "EPW", "LP"}
    rows: list[list[int]] = []
    for residue_id in np.unique(np.asarray(residue_ids, dtype=np.int32)):
        indices = np.where(np.asarray(residue_ids, dtype=np.int32) == residue_id)[0]
        if indices.size < 4:
            continue
        residue = str(residue_names[indices[0]]).strip().upper()
        if residue not in tip4p_residues:
            continue
        names = {int(index): str(atom_names[index]).strip().upper() for index in indices}
        oxygen = [index for index, name in names.items() if name in oxygen_names]
        hydrogens = [index for index, name in names.items() if name in hydrogen_names]
        sites = [index for index, name in names.items() if name in site_names]
        if len(oxygen) != 1 or len(hydrogens) != 2 or len(sites) != 1:
            continue
        rows.append([oxygen[0], hydrogens[0], hydrogens[1], sites[0]])
    if not rows:
        return empty_indices(4), np.empty((0, 4), dtype=np.float32), np.asarray([], dtype=str)
    weights = np.zeros((len(rows), 4), dtype=np.float32)
    weights[:, :3] = np.asarray(tip4p_ew_m_site_weights(), dtype=np.float32)
    return (
        np.asarray(rows, dtype=np.int32),
        weights,
        np.asarray(["tip4p_ew"] * len(rows), dtype=str),
    )


def _amber_has_periodic_box(topology: AmberPrmtop) -> bool:
    pointers = [int(value) for value in topology.optional_values("POINTERS")]
    return _amber_pointer_value(pointers, AMBER_POINTER_IFBOX_INDEX) > 0 or (
        "BOX_DIMENSIONS" in topology.flags
    )


def _read_amber_restart(
    path: Path,
    *,
    expect_periodic_box: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.read_bytes()[:3] == b"CDF":
        return _read_amber_netcdf_restart(path, expect_periodic_box=expect_periodic_box)
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        msg = f"AMBER coordinate file is too short: {path}"
        raise TopologyImportError(msg)
    header = lines[1].split()
    if not header:
        msg = f"AMBER coordinate file has no atom count: {path}"
        raise TopologyImportError(msg)
    try:
        atom_count = int(header[0])
        values = [
            float(value.replace("D", "E").replace("d", "e"))
            for line in lines[2:]
            for value in line.split()
        ]
    except ValueError as exc:
        msg = "unsupported_terms:amber_malformed_topology"
        raise TopologyImportError(msg) from exc
    coordinate_count = 3 * atom_count
    if len(values) < coordinate_count:
        msg = f"AMBER coordinate file has fewer than {coordinate_count} coordinate values"
        raise TopologyImportError(msg)
    positions = np.asarray(values[:coordinate_count], dtype=np.float32).reshape((atom_count, 3))
    if not np.all(np.isfinite(positions)):
        msg = "unsupported_terms:amber_malformed_topology"
        raise TopologyImportError(msg)
    remainder = values[coordinate_count:]
    velocities = np.asarray([], dtype=np.float32)
    cell_lengths = np.asarray([], dtype=np.float32)
    cell_matrix = np.asarray([], dtype=np.float32)
    box_value_count = 0
    if expect_periodic_box and len(remainder) in {3, 6, coordinate_count + 3, coordinate_count + 6}:
        box_value_count = 6 if len(remainder) in {6, coordinate_count + 6} else 3
    if len(remainder) - box_value_count == coordinate_count:
        velocities = np.asarray(
            remainder[: coordinate_count],
            dtype=np.float32,
        ).reshape((atom_count, 3))
        if not np.all(np.isfinite(velocities)):
            msg = "unsupported_terms:amber_malformed_topology"
            raise TopologyImportError(msg)
        remainder = remainder[coordinate_count:]
    if box_value_count == 0 and remainder:
        msg = "unsupported_terms:amber_restart_remainder"
        raise TopologyImportError(msg)
    if len(remainder) in {3, 6}:
        cell_lengths = np.asarray(remainder[:3], dtype=np.float32)
        _validate_amber_box_lengths(cell_lengths)
    if len(remainder) == 6:
        cell_angles = np.asarray(remainder[3:6], dtype=np.float32)
        cell_matrix = _cell_matrix_from_lengths_angles(cell_lengths, cell_angles)
        cell_lengths = np.linalg.norm(cell_matrix, axis=1).astype(np.float32)
    return positions, velocities, cell_lengths, cell_matrix


def _read_amber_netcdf_restart(
    path: Path,
    *,
    expect_periodic_box: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency.
        msg = "unsupported_terms:amber_netcdf_restart_requires_scipy"
        raise TopologyImportError(msg) from exc
    try:
        with netcdf_file(path, "r", mmap=False) as restart:
            if "coordinates" not in restart.variables:
                msg = "unsupported_terms:amber_netcdf_restart_missing_coordinates"
                raise TopologyImportError(msg)
            positions = np.asarray(restart.variables["coordinates"].data, dtype=np.float32)
            velocities = (
                np.asarray(restart.variables["velocities"].data, dtype=np.float32)
                if "velocities" in restart.variables
                else np.asarray([], dtype=np.float32)
            )
            cell_lengths = (
                np.asarray(restart.variables["cell_lengths"].data, dtype=np.float32)
                if "cell_lengths" in restart.variables
                else np.asarray([], dtype=np.float32)
            )
            cell_angles = (
                np.asarray(restart.variables["cell_angles"].data, dtype=np.float32)
                if "cell_angles" in restart.variables
                else np.asarray([], dtype=np.float32)
            )
    except OSError as exc:
        msg = "unsupported_terms:amber_malformed_netcdf_restart"
        raise TopologyImportError(msg) from exc
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        msg = "unsupported_terms:amber_malformed_netcdf_restart"
        raise TopologyImportError(msg)
    if velocities.size and (
        velocities.shape != positions.shape or not np.all(np.isfinite(velocities))
    ):
        msg = "unsupported_terms:amber_malformed_netcdf_restart"
        raise TopologyImportError(msg)
    if expect_periodic_box:
        _validate_amber_box_lengths(cell_lengths)
    cell_matrix = np.asarray([], dtype=np.float32)
    if cell_angles.size == 3:
        cell_matrix = _cell_matrix_from_lengths_angles(cell_lengths, cell_angles)
        cell_lengths = np.linalg.norm(cell_matrix, axis=1).astype(np.float32)
    return positions, velocities, cell_lengths, cell_matrix


def _amber_box_from_topology(topology: AmberPrmtop) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(topology.optional_values("BOX_DIMENSIONS"), dtype=np.float32)
    if values.size < 4:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    beta = float(values[0])
    lengths = values[1:4].astype(np.float32)
    if not np.isfinite(beta):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    _validate_amber_box_lengths(lengths)
    if abs(beta - 90.0) <= 1.0e-5:
        return lengths, np.asarray([], dtype=np.float32)
    angles = np.asarray([beta, beta, beta], dtype=np.float32)
    matrix = _cell_matrix_from_lengths_angles(lengths, angles)
    return np.linalg.norm(matrix, axis=1).astype(np.float32), matrix


def _validate_amber_box_lengths(lengths: np.ndarray) -> None:
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)


def _require_valid_amber_periodic_box(cell_lengths: np.ndarray, cell_matrix: np.ndarray) -> None:
    _validate_amber_box_lengths(cell_lengths)
    matrix = np.asarray(cell_matrix, dtype=np.float64)
    if matrix.size == 0:
        return
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or np.any(np.linalg.norm(matrix, axis=1) <= 0.0)
        or abs(float(np.linalg.det(matrix))) <= 1.0e-12
    ):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)


def _cell_matrix_from_lengths_angles(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if lengths.shape != (3,) or angles.shape != (3,):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(angles)):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    if np.any(lengths <= 0.0):
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    alpha, beta, gamma = np.deg2rad(angles)
    sin_gamma = np.sin(gamma)
    if abs(float(sin_gamma)) < 1.0e-7:
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    ax = lengths[0]
    bx = lengths[1] * np.cos(gamma)
    by = lengths[1] * sin_gamma
    cx = lengths[2] * np.cos(beta)
    cy = lengths[2] * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cz2 = lengths[2] * lengths[2] - cx * cx - cy * cy
    if cz2 <= 0.0:
        msg = "unsupported_terms:amber_invalid_periodic_box"
        raise TopologyImportError(msg)
    return np.asarray(
        [
            [ax, 0.0, 0.0],
            [bx, by, 0.0],
            [cx, cy, np.sqrt(cz2)],
        ],
        dtype=np.float32,
    )


def _format_kind(format_line: str) -> str:
    match = re.search(r"\(([^)]*)\)", format_line)
    if match is None:
        return ""
    kind = re.search(r"\d*\s*([AIFEDG])", match.group(1), flags=re.IGNORECASE)
    if kind is None:
        return ""
    return kind.group(1).lower()


def _numeric_or_split_values(line: str, format_line: str) -> list[str]:
    split_values = line.split()
    if _format_kind(format_line) not in {"i", "f", "e", "d", "g"}:
        return split_values
    width = _format_width(format_line, default=0)
    if width <= 0:
        return split_values
    fixed_values = [chunk.strip() for chunk in _fixed_width_chunks(line, width) if chunk.strip()]
    if len(fixed_values) > len(split_values):
        return fixed_values
    return split_values


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


def _amber_atom_count(topology: AmberPrmtop) -> int:
    pointers = topology.values("POINTERS")
    atom_count = int(pointers[0]) if pointers else 0
    if atom_count <= 0:
        msg = "unsupported_terms:amber_malformed_atom_arrays"
        raise TopologyImportError(msg)
    return atom_count


def _validate_amber_atom_arrays(
    *,
    atom_count: int,
    atom_names: np.ndarray,
    atom_types: np.ndarray,
    charges: np.ndarray,
    masses: np.ndarray,
    type_indices: np.ndarray,
) -> None:
    for values in (atom_names, atom_types, charges, masses, type_indices):
        if int(np.asarray(values).shape[0]) != atom_count:
            msg = "unsupported_terms:amber_malformed_atom_arrays"
            raise TopologyImportError(msg)
    if not np.all(np.isfinite(charges)):
        msg = "unsupported_terms:amber_malformed_atom_arrays"
        raise TopologyImportError(msg)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        msg = "unsupported_terms:amber_malformed_atom_arrays"
        raise TopologyImportError(msg)


def _amber_residue_arrays(
    topology: AmberPrmtop,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    has_labels = "RESIDUE_LABEL" in topology.flags
    has_pointers = "RESIDUE_POINTER" in topology.flags
    if not has_labels and not has_pointers:
        return (
            np.asarray(["SYS"] * atom_count, dtype=str),
            np.ones((atom_count,), dtype=np.int32),
            np.asarray(["A"] * atom_count, dtype=str),
        )
    labels = [str(item).strip() or "SYS" for item in topology.optional_values("RESIDUE_LABEL")]
    pointers = [int(item) for item in topology.optional_values("RESIDUE_POINTER")]
    if (
        not labels
        or not pointers
        or len(labels) != len(pointers)
        or pointers[0] != 1
        or any(pointer < 1 or pointer > atom_count for pointer in pointers)
        or any(
            next_pointer <= pointer
            for pointer, next_pointer in zip(pointers, pointers[1:], strict=False)
        )
    ):
        msg = "unsupported_terms:amber_malformed_residues"
        raise TopologyImportError(msg)
    starts = [pointer - 1 for pointer in pointers] + [atom_count]
    residue_names: list[str] = []
    residue_ids: list[int] = []
    residue_ranges = zip(labels, starts[:-1], starts[1:], strict=True)
    for residue_index, (name, start, stop) in enumerate(residue_ranges, start=1):
        count = max(0, stop - start)
        residue_names.extend([name] * count)
        residue_ids.extend([residue_index] * count)
    if len(residue_names) != atom_count:
        msg = "unsupported_terms:amber_malformed_residues"
        raise TopologyImportError(msg)
    return (
        np.asarray(residue_names, dtype=str),
        np.asarray(residue_ids, dtype=np.int32),
        np.asarray(["A"] * atom_count, dtype=str),
    )


def _amber_bonds(
    topology: AmberPrmtop,
    *,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = _amber_index_parameter_records(
        topology.optional_values("BONDS_INC_HYDROGEN"),
        width=3,
        blocker="unsupported_terms:amber_malformed_bond_parameters",
    ) + _amber_index_parameter_records(
        topology.optional_values("BONDS_WITHOUT_HYDROGEN"),
        width=3,
        blocker="unsupported_terms:amber_malformed_bond_parameters",
    )
    force_constants = np.asarray(topology.optional_values("BOND_FORCE_CONSTANT"), dtype=np.float32)
    lengths = np.asarray(topology.optional_values("BOND_EQUIL_VALUE"), dtype=np.float32)
    _validate_amber_finite_parameter_arrays(
        "unsupported_terms:amber_malformed_bond_parameters",
        force_constants,
        lengths,
    )
    bonds: list[tuple[int, int]] = []
    k_values: list[float] = []
    length_values: list[float] = []
    for i_raw, j_raw, parameter_index in triples:
        _validate_amber_raw_atom_indices(
            (i_raw, j_raw),
            atom_count=atom_count,
            blocker="unsupported_terms:amber_malformed_bond_parameters",
        )
        bonds.append((i_raw // 3, j_raw // 3))
        param = parameter_index - 1
        if param < 0 or param >= force_constants.shape[0] or param >= lengths.shape[0]:
            msg = "unsupported_terms:amber_malformed_bond_parameters"
            raise TopologyImportError(msg)
        k_values.append(2.0 * float(force_constants[param]) * KCAL_TO_KJ)
        length_values.append(float(lengths[param]))
    return (
        np.asarray(bonds, dtype=np.int32).reshape((-1, 2)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(length_values, dtype=np.float32),
    )


def _amber_angles(
    topology: AmberPrmtop,
    *,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = _amber_index_parameter_records(
        topology.optional_values("ANGLES_INC_HYDROGEN"),
        width=4,
        blocker="unsupported_terms:amber_malformed_angle_parameters",
    ) + _amber_index_parameter_records(
        topology.optional_values("ANGLES_WITHOUT_HYDROGEN"),
        width=4,
        blocker="unsupported_terms:amber_malformed_angle_parameters",
    )
    force_constants = np.asarray(topology.optional_values("ANGLE_FORCE_CONSTANT"), dtype=np.float32)
    theta = np.asarray(topology.optional_values("ANGLE_EQUIL_VALUE"), dtype=np.float32)
    _validate_amber_finite_parameter_arrays(
        "unsupported_terms:amber_malformed_angle_parameters",
        force_constants,
        theta,
    )
    angles: list[tuple[int, int, int]] = []
    k_values: list[float] = []
    theta_values: list[float] = []
    for i_raw, j_raw, k_raw, parameter_index in records:
        _validate_amber_raw_atom_indices(
            (i_raw, j_raw, k_raw),
            atom_count=atom_count,
            blocker="unsupported_terms:amber_malformed_angle_parameters",
        )
        angles.append((i_raw // 3, j_raw // 3, k_raw // 3))
        param = parameter_index - 1
        if param < 0 or param >= force_constants.shape[0] or param >= theta.shape[0]:
            msg = "unsupported_terms:amber_malformed_angle_parameters"
            raise TopologyImportError(msg)
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
    atom_count: int,
    improper: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], tuple[float, float]],
]:
    records = _amber_index_parameter_records(
        topology.optional_values("DIHEDRALS_INC_HYDROGEN"),
        width=5,
        blocker="unsupported_terms:amber_malformed_dihedral_parameters",
    ) + _amber_index_parameter_records(
        topology.optional_values("DIHEDRALS_WITHOUT_HYDROGEN"),
        width=5,
        blocker="unsupported_terms:amber_malformed_dihedral_parameters",
    )
    force_constants = np.asarray(
        topology.optional_values("DIHEDRAL_FORCE_CONSTANT"),
        dtype=np.float32,
    )
    periodicity = np.asarray(topology.optional_values("DIHEDRAL_PERIODICITY"), dtype=np.float32)
    phase = np.asarray(topology.optional_values("DIHEDRAL_PHASE"), dtype=np.float32)
    _validate_amber_finite_parameter_arrays(
        "unsupported_terms:amber_malformed_dihedral_parameters",
        force_constants,
        periodicity,
        phase,
    )
    selected: list[tuple[int, int, int, int]] = []
    k_values: list[float] = []
    periodicity_values: list[float] = []
    phase_values: list[float] = []
    one_four_scaling: dict[tuple[int, int], tuple[float, float]] = {}
    scee_values = np.asarray(topology.optional_values("SCEE_SCALE_FACTOR"), dtype=np.float64)
    scnb_values = np.asarray(topology.optional_values("SCNB_SCALE_FACTOR"), dtype=np.float64)
    for i_raw, j_raw, k_raw, l_raw, parameter_index in records:
        _validate_amber_raw_atom_indices(
            (i_raw, j_raw),
            atom_count=atom_count,
            blocker="unsupported_terms:amber_malformed_dihedral_parameters",
        )
        _validate_amber_raw_atom_indices(
            (k_raw, l_raw),
            atom_count=atom_count,
            blocker="unsupported_terms:amber_malformed_dihedral_parameters",
            allow_signed=True,
        )
        is_improper = l_raw < 0
        if is_improper != improper:
            continue
        atoms = (i_raw // 3, j_raw // 3, abs(k_raw) // 3, abs(l_raw) // 3)
        selected.append(atoms)
        param = parameter_index - 1
        if (
            param < 0
            or param >= force_constants.shape[0]
            or param >= periodicity.shape[0]
            or param >= phase.shape[0]
        ):
            msg = "unsupported_terms:amber_malformed_dihedral_parameters"
            raise TopologyImportError(msg)
        k_values.append(float(force_constants[param]) * KCAL_TO_KJ)
        periodicity_values.append(float(abs(periodicity[param])))
        phase_values.append(float(phase[param]))
        if not improper and k_raw >= 0 and l_raw >= 0:
            pair = _normalize_pair(atoms[0], atoms[3])
            scale = _amber_14_scale_for_parameter(
                param,
                scee_values=scee_values,
                scnb_values=scnb_values,
            )
            previous = one_four_scaling.get(pair)
            if previous is not None and not np.allclose(previous, scale, rtol=0.0, atol=1.0e-8):
                msg = "unsupported_terms:amber_conflicting_14_scaling"
                raise TopologyImportError(msg)
            one_four_scaling[pair] = scale
    return (
        np.asarray(selected, dtype=np.int32).reshape((-1, 4)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(periodicity_values, dtype=np.float32),
        np.asarray(phase_values, dtype=np.float32),
        one_four_scaling,
    )


def _amber_index_parameter_records(
    values: Sequence[int] | Sequence[float],
    *,
    width: int,
    blocker: str,
) -> list[tuple[int, ...]]:
    if not values:
        return []
    ints = [int(value) for value in values]
    if len(ints) % width:
        raise TopologyImportError(blocker)
    return [tuple(ints[index : index + width]) for index in range(0, len(ints), width)]


def _validate_amber_finite_parameter_arrays(blocker: str, *arrays: np.ndarray) -> None:
    for array in arrays:
        if array.size and not np.all(np.isfinite(array)):
            raise TopologyImportError(blocker)


def _validate_amber_raw_atom_indices(
    raw_indices: Sequence[int],
    *,
    atom_count: int,
    blocker: str,
    allow_signed: bool = False,
) -> None:
    for raw_index in raw_indices:
        if raw_index < 0 and not allow_signed:
            raise TopologyImportError(blocker)
        abs_index = abs(int(raw_index))
        if abs_index % 3 or abs_index // 3 >= atom_count:
            raise TopologyImportError(blocker)


def _amber_lj_self_parameters(
    topology: AmberPrmtop,
    atom_type_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pointers = topology.values("POINTERS")
    ntypes = int(pointers[1]) if len(pointers) > 1 else int(np.max(atom_type_indices))
    _validate_amber_atom_type_indices(atom_type_indices, ntypes)
    nb_index, acoef, bcoef = _amber_lj_arrays(topology, ntypes)
    type_sigma = np.zeros((ntypes,), dtype=np.float32)
    type_epsilon = np.zeros((ntypes,), dtype=np.float32)
    for type_id in range(1, ntypes + 1):
        packed_index = int(nb_index[(type_id - 1) * ntypes + (type_id - 1)])
        if packed_index <= 0:
            msg = "unsupported_terms:amber_10_12_nonbonded"
            raise TopologyImportError(msg)
        a, b = _amber_lj_coefficients(packed_index, acoef, bcoef)
        if a <= 0.0 or b <= 0.0:
            type_sigma[type_id - 1] = 1.0
            type_epsilon[type_id - 1] = 0.0
            continue
        sigma_value = float((a / b) ** (1.0 / 6.0))
        epsilon_value = float((b * b / (4.0 * a)) * KCAL_TO_KJ)
        _validate_amber_lj_outputs(
            np.asarray([sigma_value], dtype=np.float64),
            np.asarray([epsilon_value], dtype=np.float64),
        )
        type_sigma[type_id - 1] = sigma_value
        type_epsilon[type_id - 1] = epsilon_value
    _validate_amber_lj_outputs(type_sigma, type_epsilon)
    _check_amber_lj_combining_rules(topology, type_sigma, type_epsilon, ntypes)
    indices = atom_type_indices.astype(np.int32) - 1
    return type_sigma[indices], type_epsilon[indices]


def _check_amber_lj_combining_rules(
    topology: AmberPrmtop,
    type_sigma: np.ndarray,
    type_epsilon: np.ndarray,
    ntypes: int,
) -> None:
    for type_i in range(1, ntypes + 1):
        for type_j in range(1, ntypes + 1):
            sigma_ij, epsilon_ij = _amber_lj_type_pair_parameters(topology, type_i, type_j)
            expected_sigma = 0.5 * (
                float(type_sigma[type_i - 1]) + float(type_sigma[type_j - 1])
            )
            expected_epsilon = (
                float(type_epsilon[type_i - 1]) * float(type_epsilon[type_j - 1])
            ) ** 0.5
            if epsilon_ij == 0.0 and expected_epsilon == 0.0:
                continue
            if not (
                np.isclose(sigma_ij, expected_sigma, rtol=1.0e-5, atol=1.0e-6)
                and np.isclose(epsilon_ij, expected_epsilon, rtol=1.0e-5, atol=1.0e-6)
            ):
                msg = "unsupported_terms:amber_modified_lj_pair_parameters"
                raise TopologyImportError(msg)


def _amber_lj_pair_parameters(
    topology: AmberPrmtop,
    atom_type_indices: np.ndarray,
    i: int,
    j: int,
) -> tuple[float, float]:
    type_i = int(atom_type_indices[i])
    type_j = int(atom_type_indices[j])
    return _amber_lj_type_pair_parameters(topology, type_i, type_j)


def _amber_lj_type_pair_parameters(
    topology: AmberPrmtop,
    type_i: int,
    type_j: int,
) -> tuple[float, float]:
    pointers = topology.values("POINTERS")
    ntypes = int(pointers[1]) if len(pointers) > 1 else max(type_i, type_j)
    _validate_amber_atom_type_indices(np.asarray([type_i, type_j], dtype=np.int32), ntypes)
    nb_index, acoef, bcoef = _amber_lj_arrays(topology, ntypes)
    packed_index = int(nb_index[(type_i - 1) * ntypes + (type_j - 1)])
    if packed_index <= 0:
        if _amber_zero_lj_negative_pair_detail(
            topology,
            type_i=type_i,
            type_j=type_j,
            ntypes=ntypes,
            nb_index=nb_index,
            acoef=acoef,
            bcoef=bcoef,
        ):
            return 1.0, 0.0
        msg = "unsupported_terms:amber_10_12_nonbonded"
        raise TopologyImportError(msg)
    a, b = _amber_lj_coefficients(packed_index, acoef, bcoef)
    if a == 0.0 and b == 0.0:
        return 1.0, 0.0
    if a <= 0.0 or b <= 0.0:
        msg = "unsupported_terms:amber_nonpositive_lj_pair_parameters"
        raise TopologyImportError(msg)
    sigma = float((a / b) ** (1.0 / 6.0))
    epsilon = float((b * b / (4.0 * a)) * KCAL_TO_KJ)
    _validate_amber_lj_outputs(
        np.asarray([sigma], dtype=np.float64),
        np.asarray([epsilon], dtype=np.float64),
    )
    return sigma, epsilon


def _amber_allowed_negative_lj_pair_policy(topology: AmberPrmtop) -> dict[str, Any]:
    pointers = topology.values("POINTERS")
    ntypes = int(pointers[1]) if len(pointers) > 1 else 0
    if ntypes <= 0:
        return {}
    nb_index, acoef, bcoef = _amber_lj_arrays(topology, ntypes)
    pairs: list[dict[str, Any]] = []
    for type_i in range(1, ntypes + 1):
        for type_j in range(1, ntypes + 1):
            packed_index = int(nb_index[(type_i - 1) * ntypes + (type_j - 1)])
            if packed_index > 0:
                continue
            detail = _amber_zero_lj_negative_pair_detail(
                topology,
                type_i=type_i,
                type_j=type_j,
                ntypes=ntypes,
                nb_index=nb_index,
                acoef=acoef,
                bcoef=bcoef,
            )
            if detail is not None:
                pairs.append(detail)
    if not pairs:
        return {}
    return {
        "status": "allowed_zero_lj_water_pairs",
        "reason": (
            "negative NONBONDED_PARM_INDEX entries are accepted only for water O/H "
            "type pairs with zero HBOND coefficients and zero standard mixed LJ epsilon"
        ),
        "affected_type_pairs": pairs,
    }


def _amber_zero_lj_negative_pair_detail(
    topology: AmberPrmtop,
    *,
    type_i: int,
    type_j: int,
    ntypes: int,
    nb_index: np.ndarray,
    acoef: np.ndarray,
    bcoef: np.ndarray,
) -> dict[str, Any] | None:
    if not _amber_hbond_coefficients_are_zero(topology):
        return None
    context_i = _amber_atom_type_context(topology, type_i)
    context_j = _amber_atom_type_context(topology, type_j)
    symbols = {context_i["symbol"], context_j["symbol"]}
    if symbols != {"O", "H"}:
        return None
    if not (
        _amber_atom_type_context_is_water(context_i)
        and _amber_atom_type_context_is_water(context_j)
    ):
        return None
    epsilon_i = _amber_lj_self_epsilon(type_i, ntypes, nb_index, acoef, bcoef)
    epsilon_j = _amber_lj_self_epsilon(type_j, ntypes, nb_index, acoef, bcoef)
    mixed_epsilon = float((epsilon_i * epsilon_j) ** 0.5)
    if not np.isclose(mixed_epsilon, 0.0, rtol=0.0, atol=1.0e-12):
        return None
    return {
        "type_i": int(type_i),
        "type_j": int(type_j),
        "atom_types_i": context_i["atom_types"],
        "atom_types_j": context_j["atom_types"],
        "residue_names_i": context_i["residue_names"],
        "residue_names_j": context_j["residue_names"],
        "symbols": sorted(symbols),
        "mixed_epsilon_kj_mol": mixed_epsilon,
    }


def _amber_hbond_coefficients_are_zero(topology: AmberPrmtop) -> bool:
    hbond_acoef = np.asarray(topology.optional_values("HBOND_ACOEF"), dtype=np.float64)
    hbond_bcoef = np.asarray(topology.optional_values("HBOND_BCOEF"), dtype=np.float64)
    return not (
        (hbond_acoef.size and np.any(np.abs(hbond_acoef) > 0.0))
        or (hbond_bcoef.size and np.any(np.abs(hbond_bcoef) > 0.0))
    )


def _amber_lj_self_epsilon(
    type_id: int,
    ntypes: int,
    nb_index: np.ndarray,
    acoef: np.ndarray,
    bcoef: np.ndarray,
) -> float:
    packed_index = int(nb_index[(type_id - 1) * ntypes + (type_id - 1)])
    if packed_index <= 0:
        msg = "unsupported_terms:amber_10_12_nonbonded"
        raise TopologyImportError(msg)
    a, b = _amber_lj_coefficients(packed_index, acoef, bcoef)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float((b * b / (4.0 * a)) * KCAL_TO_KJ)


def _amber_atom_type_context(topology: AmberPrmtop, type_id: int) -> dict[str, Any]:
    atom_count = _amber_atom_count(topology)
    type_indices = np.asarray(topology.values("ATOM_TYPE_INDEX"), dtype=np.int32)
    atom_names = [str(item).strip() for item in topology.values("ATOM_NAME")]
    atom_type_values = topology.optional_values("AMBER_ATOM_TYPE")
    if atom_type_values:
        atom_types = [str(item).strip() for item in atom_type_values]
    else:
        atom_types = [str(item) for item in type_indices.tolist()]
    residue_names = _amber_residue_arrays(topology, atom_count)[0]
    selected = np.where(type_indices == int(type_id))[0]
    if selected.size == 0:
        msg = "unsupported_terms:amber_malformed_atom_types"
        raise TopologyImportError(msg)
    symbols = {
        _infer_symbol(atom_names[index], atom_types[index]).upper()
        for index in selected.tolist()
    }
    symbol = symbols.pop() if len(symbols) == 1 else ""
    return {
        "type_id": int(type_id),
        "atom_types": sorted({atom_types[index] for index in selected.tolist()}),
        "atom_names": sorted({atom_names[index] for index in selected.tolist()}),
        "residue_names": sorted(
            {str(residue_names[index]).strip().upper() for index in selected.tolist()}
        ),
        "symbol": symbol,
        "atom_count": int(selected.size),
    }


def _amber_atom_type_context_is_water(context: dict[str, Any]) -> bool:
    residue_names = set(context["residue_names"])
    return bool(residue_names) and residue_names <= AMBER_WATER_RESIDUES


def _amber_14_scale_for_parameter(
    parameter_index: int,
    *,
    scee_values: np.ndarray,
    scnb_values: np.ndarray,
) -> tuple[float, float]:
    scee = _amber_scale_denominator(
        "SCEE_SCALE_FACTOR",
        scee_values,
        parameter_index,
        1.0 / STANDARD_AMBER_14_ELECTROSTATIC_SCALE,
    )
    scnb = _amber_scale_denominator(
        "SCNB_SCALE_FACTOR",
        scnb_values,
        parameter_index,
        1.0 / STANDARD_AMBER_14_LJ_SCALE,
    )
    return 1.0 / scee, 1.0 / scnb


def _amber_scale_denominator(
    flag_name: str,
    values: np.ndarray,
    parameter_index: int,
    default: float,
) -> float:
    if values.size == 0:
        return float(default)
    if parameter_index >= values.shape[0]:
        msg = "unsupported_terms:amber_incomplete_14_scaling"
        raise TopologyImportError(msg)
    value = float(values[parameter_index])
    if not np.isfinite(value) or value <= 0.0:
        msg = "unsupported_terms:amber_invalid_14_scaling"
        raise TopologyImportError(msg)
    return value


def _validate_amber_atom_type_indices(atom_type_indices: np.ndarray, ntypes: int) -> None:
    if ntypes <= 0:
        msg = "unsupported_terms:amber_malformed_atom_types"
        raise TopologyImportError(msg)
    indices = np.asarray(atom_type_indices, dtype=np.int32)
    if indices.size and (np.any(indices <= 0) or np.any(indices > ntypes)):
        msg = "unsupported_terms:amber_malformed_atom_types"
        raise TopologyImportError(msg)


def _amber_lj_arrays(
    topology: AmberPrmtop,
    ntypes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb_index = np.asarray(topology.values("NONBONDED_PARM_INDEX"), dtype=np.int32)
    if nb_index.shape[0] != ntypes * ntypes:
        msg = "unsupported_terms:amber_malformed_lj_parameters"
        raise TopologyImportError(msg)
    acoef = np.asarray(topology.values("LENNARD_JONES_ACOEF"), dtype=np.float64)
    bcoef = np.asarray(topology.values("LENNARD_JONES_BCOEF"), dtype=np.float64)
    if not np.all(np.isfinite(acoef)) or not np.all(np.isfinite(bcoef)):
        msg = "unsupported_terms:amber_malformed_lj_parameters"
        raise TopologyImportError(msg)
    return (
        nb_index,
        acoef,
        bcoef,
    )


def _amber_lj_coefficients(
    packed_index: int,
    acoef: np.ndarray,
    bcoef: np.ndarray,
) -> tuple[float, float]:
    if packed_index > min(acoef.shape[0], bcoef.shape[0]):
        msg = "unsupported_terms:amber_malformed_lj_parameters"
        raise TopologyImportError(msg)
    a = float(acoef[packed_index - 1])
    b = float(bcoef[packed_index - 1])
    if not np.isfinite(a) or not np.isfinite(b):
        msg = "unsupported_terms:amber_malformed_lj_parameters"
        raise TopologyImportError(msg)
    return a, b


def _validate_amber_lj_outputs(*arrays: np.ndarray) -> None:
    float32_max = float(np.finfo(np.float32).max)
    for array in arrays:
        values = np.asarray(array)
        if values.size and (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > float32_max)
        ):
            msg = "unsupported_terms:amber_malformed_lj_parameters"
            raise TopologyImportError(msg)


def _amber_excluded_pairs(
    topology: AmberPrmtop,
    *,
    atom_count: int,
) -> set[tuple[int, int]] | None:
    has_counts = "NUMBER_EXCLUDED_ATOMS" in topology.flags
    has_values = "EXCLUDED_ATOMS_LIST" in topology.flags
    if not has_counts and not has_values:
        return None
    counts = [int(value) for value in topology.optional_values("NUMBER_EXCLUDED_ATOMS")]
    values = [int(value) for value in topology.optional_values("EXCLUDED_ATOMS_LIST")]
    if len(counts) != atom_count:
        msg = "unsupported_terms:amber_malformed_exclusions"
        raise TopologyImportError(msg)
    if any(count < 0 for count in counts):
        msg = "unsupported_terms:amber_malformed_exclusions"
        raise TopologyImportError(msg)
    expected = sum(counts)
    if expected != len(values):
        msg = "unsupported_terms:amber_malformed_exclusions"
        raise TopologyImportError(msg)
    pairs: set[tuple[int, int]] = set()
    offset = 0
    for atom_index, count in enumerate(counts):
        for excluded_atom in values[offset : offset + count]:
            if excluded_atom < 0:
                msg = "unsupported_terms:amber_malformed_exclusions"
                raise TopologyImportError(msg)
            if excluded_atom == 0:
                continue
            if excluded_atom > atom_count or excluded_atom == atom_index + 1:
                msg = "unsupported_terms:amber_malformed_exclusions"
                raise TopologyImportError(msg)
            pairs.add(_normalize_pair(atom_index, excluded_atom - 1))
        offset += count
    return pairs


def _amber_14_scaling_metadata(
    topology: AmberPrmtop,
    one_four_scaling: dict[tuple[int, int], tuple[float, float]],
) -> dict[str, Any]:
    electrostatic = sorted({round(float(scale[0]), 10) for scale in one_four_scaling.values()})
    lj = sorted({round(float(scale[1]), 10) for scale in one_four_scaling.values()})
    return {
        "source": _amber_14_scaling_source(topology),
        "one_four_pair_count": int(len(one_four_scaling)),
        "electrostatic_scale_values": electrostatic,
        "lj_scale_values": lj,
    }


def _amber_14_scaling_source(topology: AmberPrmtop) -> str:
    has_scee = "SCEE_SCALE_FACTOR" in topology.flags
    has_scnb = "SCNB_SCALE_FACTOR" in topology.flags
    if has_scee and has_scnb:
        return "topology_scee_scnb"
    if has_scee:
        return "topology_scee_standard_scnb"
    if has_scnb:
        return "standard_scee_topology_scnb"
    return "standard_amber_fallback"


def _amber_exceptions(
    *,
    topology: AmberPrmtop,
    bonds: np.ndarray,
    angles: np.ndarray,
    excluded_pairs: set[tuple[int, int]] | None,
    one_four_scaling: dict[tuple[int, int], tuple[float, float]],
    charges: np.ndarray,
    type_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if excluded_pairs is None:
        excluded_pairs = _normalized_pairs(bonds) | _pairs_from_angles(angles)
    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {
        pair: (0.0, 0.0, 0.0) for pair in excluded_pairs
    }
    for (i, j), (electrostatic_scale, lj_scale) in sorted(one_four_scaling.items()):
        sigma_ij, epsilon_ij = _amber_lj_pair_parameters(topology, type_indices, i, j)
        exceptions[(i, j)] = (
            float(charges[i] * charges[j]) * electrostatic_scale,
            sigma_ij,
            epsilon_ij * lj_scale,
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
            try:
                raw_grid = np.asarray(getattr(cmap_type, "grid", []), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                msg = "unsupported_terms:charmm_cmap_terms"
                raise TopologyImportError(msg) from exc
            if resolution < 4 or raw_grid.size != resolution * resolution:
                msg = "unsupported_terms:charmm_cmap_terms"
                raise TopologyImportError(msg)
            grid = raw_grid * KCAL_TO_KJ
            _validate_charmm_finite("unsupported_terms:charmm_cmap_terms", grid)
            grid_index = len(grids)
            grid_ids[grid_id] = grid_index
            grids.append(grid.astype(np.float32).reshape((resolution, resolution)))
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
        _validate_charmm_finite(
            "unsupported_terms:nbfix_pair_overrides:malformed_entries",
            sigma,
            epsilon,
        )
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
    bond_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if bonds.size == 0:
        return empty_indices(2), np.asarray([], dtype=np.float32)
    bond_array = np.asarray(bonds, dtype=np.int32).reshape((-1, 2))
    length_array = np.asarray(bond_lengths, dtype=np.float32)
    if length_array.shape != (bond_array.shape[0],):
        msg = "unsupported_terms:constraint_equilibrium_geometry"
        raise TopologyImportError(msg)
    upper_symbols = np.char.upper(np.asarray(symbols, dtype=str))
    distances_by_pair: dict[tuple[int, int], float] = {}
    for (i, j), distance in zip(bond_array.tolist(), length_array.tolist(), strict=True):
        if upper_symbols[int(i)] != "H" and upper_symbols[int(j)] != "H":
            continue
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        previous = distances_by_pair.get(pair)
        if previous is not None and not np.isclose(previous, distance, rtol=0.0, atol=1e-7):
            msg = "unsupported_terms:constraint_equilibrium_geometry"
            raise TopologyImportError(msg)
        distances_by_pair[pair] = float(distance)
    if not distances_by_pair:
        return empty_indices(2), np.asarray([], dtype=np.float32)
    pairs = sorted(distances_by_pair)
    constraints = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
    distances = np.asarray([distances_by_pair[pair] for pair in pairs], dtype=np.float32)
    return constraints, distances


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
    "import_charmm_psf",
    "import_charmm_with_parmed",
    "import_gromacs_top_gro",
]
