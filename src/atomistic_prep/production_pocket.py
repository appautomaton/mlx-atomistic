"""Internal all-atom ATP-pocket preparation for the bundled 4DW1 notebook example."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from atomistic_prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

INTERNAL_FORCE_FIELD_VERSION = "mlx_internal_p2x4_atp_pocket_v1"
SUPPORTED_PRODUCTION_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "GLU",
    "GLY",
    "ILE",
    "LYS",
    "PHE",
    "PRO",
    "TYR",
    "VAL",
}
LIGAND_RESNAMES = {"ATP", "ADP", "ANP"}
LIGAND_TARGET_CHARGE = {"ATP": -4.0, "ADP": -3.0, "ANP": -4.0}

ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.06,
}
VDW_SIGMA = {
    "H": 1.2,
    "C": 3.4,
    "N": 3.25,
    "O": 2.96,
    "P": 3.74,
    "S": 3.55,
}
VDW_EPSILON_KJ = {
    "H": 0.02,
    "C": 0.28,
    "N": 0.60,
    "O": 0.65,
    "P": 0.80,
    "S": 1.00,
}
XH_BOND_LENGTH = {
    "C": 1.09,
    "N": 1.01,
    "O": 0.96,
    "S": 1.34,
}

RESIDUE_BONDS = {
    "ALA": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    "ARG": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ],
    "ASN": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "ND2"),
    ],
    "ASP": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "OD2"),
    ],
    "GLU": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "OE2"),
    ],
    "GLY": [("N", "CA"), ("CA", "C"), ("C", "O")],
    "ILE": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CG1", "CD1"),
    ],
    "LYS": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "CE"),
        ("CE", "NZ"),
    ],
    "PHE": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ],
    "PRO": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "N"),
    ],
    "TYR": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ],
    "VAL": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
    ],
}

PROTEIN_HYDROGEN_COUNTS = {
    "N": 1,
    "CA": 1,
    "CB": 2,
    "CG": 2,
    "CD": 2,
    "CE": 2,
    "NE": 1,
    "NZ": 3,
    "NH1": 2,
    "NH2": 2,
    "ND2": 2,
    "OH": 1,
}
RESIDUE_HYDROGEN_OVERRIDES = {
    ("ALA", "CB"): 3,
    ("ARG", "CZ"): 0,
    ("ASN", "CG"): 0,
    ("ASP", "CG"): 0,
    ("GLU", "CD"): 0,
    ("GLY", "CA"): 2,
    ("ILE", "CB"): 1,
    ("ILE", "CG2"): 3,
    ("ILE", "CD1"): 3,
    ("LYS", "NZ"): 3,
    ("PHE", "CG"): 0,
    ("PHE", "CD1"): 1,
    ("PHE", "CD2"): 1,
    ("PHE", "CE1"): 1,
    ("PHE", "CE2"): 1,
    ("PHE", "CZ"): 1,
    ("PRO", "N"): 0,
    ("PRO", "CA"): 1,
    ("PRO", "CB"): 2,
    ("PRO", "CG"): 2,
    ("PRO", "CD"): 2,
    ("TYR", "CG"): 0,
    ("TYR", "CD1"): 1,
    ("TYR", "CD2"): 1,
    ("TYR", "CE1"): 1,
    ("TYR", "CE2"): 1,
    ("TYR", "CZ"): 0,
    ("VAL", "CB"): 1,
    ("VAL", "CG1"): 3,
    ("VAL", "CG2"): 3,
}

ATP_HYDROGEN_COUNTS = {
    "C5'": 2,
    "C4'": 1,
    "C3'": 1,
    "O3'": 1,
    "C2'": 1,
    "O2'": 1,
    "C1'": 1,
    "C8": 1,
    "N6": 2,
    "C2": 1,
}


@dataclass(frozen=True)
class _BuildAtom:
    name: str
    resname: str
    resid: int
    chain_id: str
    element: str
    position: np.ndarray
    is_ligand: bool


def prepare_p2x4_atp_production(
    *,
    pdb_path: str | Path,
    cutoff_angstrom: float,
) -> PreparedSystem:
    """Build a production-gated all-atom MLX artifact for bundled 4DW1 ATP pocket."""

    from atomistic_prep.prepare import parse_pdb

    atoms, conect_pairs = parse_pdb(pdb_path)
    selected_atoms, selected_residue_keys = _select_pocket_atoms(
        atoms,
        cutoff_angstrom=cutoff_angstrom,
    )
    _validate_supported_residues(selected_residue_keys)
    build_atoms = [
        _BuildAtom(
            name=atom.name,
            resname=atom.resname,
            resid=atom.resid,
            chain_id=atom.chain_id,
            element=atom.element.upper(),
            position=np.asarray(atom.position, dtype=np.float32),
            is_ligand=atom.is_ligand,
        )
        for atom in selected_atoms
    ]
    bonds = _initial_heavy_bonds(selected_atoms, conect_pairs)
    build_atoms, bonds, xh_constraints = _add_template_hydrogens(build_atoms, bonds)
    positions = np.stack([atom.position for atom in build_atoms]).astype(np.float32)
    bonds_array = np.asarray(sorted({tuple(sorted(pair)) for pair in bonds}), dtype=np.int32)
    bonds_array = bonds_array.reshape((-1, 2))
    angles = _infer_angles(bonds_array)
    dihedrals = _filter_dihedrals_by_geometry(
        positions,
        _infer_dihedrals(bonds_array, atom_count=len(build_atoms)),
    )
    impropers = _filter_dihedrals_by_geometry(
        positions,
        _infer_impropers(build_atoms, bonds_array),
    )
    charges = _assign_charges(build_atoms)
    sigma = np.asarray([VDW_SIGMA.get(atom.element, 3.2) for atom in build_atoms], dtype=np.float32)
    epsilon = np.asarray(
        [VDW_EPSILON_KJ.get(atom.element, 0.30) for atom in build_atoms],
        dtype=np.float32,
    )
    exceptions = _nonbonded_exceptions(
        bonds=bonds_array,
        angles=angles,
        dihedrals=dihedrals,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
    )
    bond_lengths = _distances(positions, bonds_array)
    angle_theta = _angle_values(positions, angles)
    dihedral_phase = _dihedral_reference_phases(positions, dihedrals, periodicity=3.0)
    improper_phase = _dihedral_reference_phases(positions, impropers, periodicity=2.0)
    constraints = np.asarray(sorted(set(xh_constraints)), dtype=np.int32).reshape((-1, 2))
    constraint_distance = _distances(positions, constraints)
    symbols = np.asarray([atom.element for atom in build_atoms], dtype=str)
    ligand_mask = np.asarray([atom.is_ligand for atom in build_atoms], dtype=bool)
    receptor_mask = ~ligand_mask
    hydrogen_count = int(np.count_nonzero(np.char.upper(symbols.astype(str)) == "H"))
    supported_terms = [
        "harmonic_bond",
        "harmonic_angle",
        "periodic_dihedral",
        "periodic_improper",
        "nonbonded_lj_coulomb",
        "nonbonded_exception",
        "distance_constraint",
        "positional_restraint",
    ]
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source={
            "kind": "bundled_4dw1_internal_template",
            "pdb_id": "4DW1",
            "source_path": str(Path(pdb_path)),
            "description": "ATP-bound zebrafish P2X4 receptor pocket from bundled 4DW1 data",
        },
        selections={
            "ligand_resnames": sorted(LIGAND_RESNAMES),
            "pocket_cutoff_angstrom": float(cutoff_angstrom),
            "receptor_residue_count": len(selected_residue_keys),
            "atom_count": len(build_atoms),
            "hydrogen_count": hydrogen_count,
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
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
        parameter_source=INTERNAL_FORCE_FIELD_VERSION,
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "supported_terms": supported_terms,
            "required_terms": supported_terms,
            "unsupported_terms": [],
            "rejected_terms": [],
            "force_field_provenance": INTERNAL_FORCE_FIELD_VERSION,
            "fixed_topology": True,
            "reactive_chemistry": False,
            "notes": [
                "Internal fixed-topology MLX pocket force field for bundled 4DW1 example.",
                "Not a full membrane/solvent CHARMM or AMBER production system.",
            ],
        },
        warnings=[
            "Generated by internal templates for notebook-native MLX MD.",
            "Fixed-topology classical MD: no ATP hydrolysis, bond breaking, or docking search.",
            "Pocket system only: no membrane, no solvent, no ions, no PME, no NPT.",
        ],
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=symbols,
        atom_names=np.asarray([atom.name for atom in build_atoms], dtype=str),
        atom_types=np.asarray([_atom_type(atom) for atom in build_atoms], dtype=str),
        residue_names=np.asarray([atom.resname for atom in build_atoms], dtype=str),
        residue_ids=np.asarray([atom.resid for atom in build_atoms], dtype=np.int32),
        chain_ids=np.asarray([atom.chain_id for atom in build_atoms], dtype=str),
        positions=positions,
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=np.asarray([ATOMIC_MASSES.get(atom.element, 12.0) for atom in build_atoms]),
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        bonds=bonds_array,
        bond_k=np.asarray(
            [_bond_k(build_atoms[int(i)], build_atoms[int(j)]) for i, j in bonds_array],
            dtype=np.float32,
        ),
        bond_length=bond_lengths,
        angles=angles,
        angle_k=np.full((angles.shape[0],), 45.0, dtype=np.float32),
        angle_theta=angle_theta,
        dihedrals=dihedrals,
        dihedral_k=np.full((dihedrals.shape[0],), 0.30, dtype=np.float32),
        dihedral_periodicity=np.full((dihedrals.shape[0],), 3.0, dtype=np.float32),
        dihedral_phase=dihedral_phase,
        nonbonded_pairs=empty_indices(2),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=receptor_mask,
        reference_positions=positions.copy(),
        constraints=constraints,
        constraint_distance=constraint_distance,
        impropers=impropers,
        improper_k=np.full((impropers.shape[0],), 0.20, dtype=np.float32),
        improper_periodicity=np.full((impropers.shape[0],), 2.0, dtype=np.float32),
        improper_phase=improper_phase,
        nonbonded_exception_pairs=exceptions[0],
        nonbonded_exception_charge_product=exceptions[1],
        nonbonded_exception_sigma=exceptions[2],
        nonbonded_exception_epsilon=exceptions[3],
    )
    prepared.validate()
    return prepared


def _select_pocket_atoms(
    atoms,
    *,
    cutoff_angstrom: float,
) -> tuple[list, list[tuple[str, int, str, str]]]:
    ligand_atoms = [atom for atom in atoms if atom.is_ligand]
    if not ligand_atoms:
        msg = "no ATP/ADP/ANP ligand atoms found in structure"
        raise ValueError(msg)
    ligand_positions = np.stack([atom.position for atom in ligand_atoms])
    residues: dict[tuple[str, int, str, str], list] = defaultdict(list)
    for atom in atoms:
        if atom.is_protein:
            residues[atom.residue_key].append(atom)
    selected_residue_keys = []
    for key, residue_atoms in residues.items():
        positions = np.stack([atom.position for atom in residue_atoms])
        distances = np.linalg.norm(
            positions[:, None, :] - ligand_positions[None, :, :],
            axis=-1,
        )
        if float(np.min(distances)) <= cutoff_angstrom:
            selected_residue_keys.append(key)
    selected_residue_keys = sorted(selected_residue_keys, key=lambda item: (item[0], item[1]))
    selected_set = set(selected_residue_keys)
    selected_atoms = [
        atom
        for atom in atoms
        if atom.is_ligand or (atom.is_protein and atom.residue_key in selected_set)
    ]
    if len(selected_atoms) == len(ligand_atoms):
        msg = f"no receptor residues found within {cutoff_angstrom:g} A of ATP"
        raise ValueError(msg)
    return selected_atoms, selected_residue_keys


def _validate_supported_residues(keys: Iterable[tuple[str, int, str, str]]) -> None:
    unsupported = sorted({key[3] for key in keys if key[3] not in SUPPORTED_PRODUCTION_RESIDUES})
    if unsupported:
        missing = ", ".join(unsupported)
        msg = f"production 4DW1 builder does not have residue templates for: {missing}"
        raise ValueError(msg)


def _initial_heavy_bonds(
    selected_atoms,
    conect_pairs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    serial_to_index = {atom.serial: index for index, atom in enumerate(selected_atoms)}
    bonds: set[tuple[int, int]] = set()
    for i_serial, j_serial in conect_pairs:
        if i_serial in serial_to_index and j_serial in serial_to_index:
            i = serial_to_index[i_serial]
            j = serial_to_index[j_serial]
            if selected_atoms[i].is_ligand or selected_atoms[j].is_ligand:
                bonds.add(_pair(i, j))
    atom_by_residue: dict[tuple[str, int, str, str], dict[str, int]] = defaultdict(dict)
    for index, atom in enumerate(selected_atoms):
        if atom.is_protein:
            atom_by_residue[atom.residue_key][atom.name] = index
    for residue_key, atom_indices in atom_by_residue.items():
        resname = residue_key[3]
        for left, right in RESIDUE_BONDS[resname]:
            if left in atom_indices and right in atom_indices:
                bonds.add(_pair(atom_indices[left], atom_indices[right]))
    sorted_residues = sorted(atom_by_residue, key=lambda key: (key[0], key[1], key[2]))
    for left_key, right_key in zip(sorted_residues, sorted_residues[1:], strict=False):
        if left_key[0] != right_key[0] or right_key[1] != left_key[1] + 1:
            continue
        left_atoms = atom_by_residue[left_key]
        right_atoms = atom_by_residue[right_key]
        if "C" not in left_atoms or "N" not in right_atoms:
            continue
        i = left_atoms["C"]
        j = right_atoms["N"]
        distance = np.linalg.norm(selected_atoms[i].position - selected_atoms[j].position)
        if float(distance) <= 1.8:
            bonds.add(_pair(i, j))
    return sorted(bonds)


def _add_template_hydrogens(
    atoms: list[_BuildAtom],
    bonds: list[tuple[int, int]],
) -> tuple[list[_BuildAtom], list[tuple[int, int]], list[tuple[int, int]]]:
    built = list(atoms)
    all_bonds = list(bonds)
    heavy_neighbors = _neighbors(len(built), all_bonds)
    constraints: list[tuple[int, int]] = []
    for atom_index, atom in enumerate(atoms):
        count = _hydrogen_count(atom)
        if count <= 0:
            continue
        h_positions = _hydrogen_positions(atom_index, built, heavy_neighbors, count)
        for h_index, h_position in enumerate(h_positions, start=1):
            name = _hydrogen_name(atom.name, h_index, count)
            new_index = len(built)
            built.append(
                _BuildAtom(
                    name=name,
                    resname=atom.resname,
                    resid=atom.resid,
                    chain_id=atom.chain_id,
                    element="H",
                    position=h_position.astype(np.float32),
                    is_ligand=atom.is_ligand,
                )
            )
            all_bonds.append(_pair(atom_index, new_index))
            constraints.append(_pair(atom_index, new_index))
    return built, sorted(set(all_bonds)), constraints


def _hydrogen_count(atom: _BuildAtom) -> int:
    if atom.element == "H":
        return 0
    if atom.resname in LIGAND_RESNAMES:
        return ATP_HYDROGEN_COUNTS.get(atom.name, 0)
    if atom.name in {"C", "O"}:
        return 0
    return RESIDUE_HYDROGEN_OVERRIDES.get(
        (atom.resname, atom.name),
        PROTEIN_HYDROGEN_COUNTS.get(atom.name, 0),
    )


def _hydrogen_positions(
    atom_index: int,
    atoms: list[_BuildAtom],
    neighbors: dict[int, set[int]],
    count: int,
) -> list[np.ndarray]:
    atom = atoms[atom_index]
    origin = atom.position
    length = XH_BOND_LENGTH.get(atom.element, 1.0)
    neighbor_vectors = []
    for neighbor in sorted(neighbors.get(atom_index, set())):
        vector = atoms[neighbor].position - origin
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            neighbor_vectors.append(vector / norm)
    if neighbor_vectors:
        direction = -np.sum(neighbor_vectors, axis=0)
        if np.linalg.norm(direction) < 1e-6:
            direction = _perpendicular(neighbor_vectors[0])
    else:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    direction = _unit(direction)
    perpendicular = _perpendicular(direction)
    second = _unit(np.cross(direction, perpendicular))
    if count == 1:
        offsets = [_unit(direction + 0.35 * perpendicular)] if neighbor_vectors else [direction]
    elif count == 2:
        offsets = [_unit(direction + 0.75 * perpendicular), _unit(direction - 0.75 * perpendicular)]
    else:
        offsets = [
            _unit(direction + 0.85 * perpendicular),
            _unit(direction - 0.45 * perpendicular + 0.74 * second),
            _unit(direction - 0.45 * perpendicular - 0.74 * second),
        ][:count]
    return [origin + length * offset for offset in offsets]


def _hydrogen_name(parent: str, index: int, count: int) -> str:
    parent = parent.replace("'", "")
    if count == 1:
        return f"H{parent}"[:4]
    return f"H{parent}{index}"[:4]


def _assign_charges(atoms: list[_BuildAtom]) -> np.ndarray:
    charges = np.asarray([_base_charge(atom) for atom in atoms], dtype=np.float32)
    ligand_resnames = {atom.resname for atom in atoms if atom.is_ligand}
    for resname in ligand_resnames:
        target = LIGAND_TARGET_CHARGE.get(resname)
        if target is None:
            continue
        indices = [index for index, atom in enumerate(atoms) if atom.resname == resname]
        delta = target - float(np.sum(charges[indices]))
        charges[indices] += delta / len(indices)
    residue_indices: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        if not atom.is_ligand:
            residue_indices[(atom.chain_id, atom.resid, atom.resname)].append(index)
    for (_, _, resname), indices in residue_indices.items():
        target = _residue_target_charge(resname)
        delta = target - float(np.sum(charges[indices]))
        charges[indices] += delta / len(indices)
    return charges.astype(np.float32)


def _residue_target_charge(resname: str) -> float:
    if resname in {"ARG", "LYS"}:
        return 1.0
    if resname in {"ASP", "GLU"}:
        return -1.0
    return 0.0


def _base_charge(atom: _BuildAtom) -> float:
    if atom.element == "H":
        if atom.resname in LIGAND_RESNAMES:
            return 0.12
        return 0.18
    if atom.resname in LIGAND_RESNAMES:
        if atom.element == "P":
            return 1.20
        if atom.element == "O":
            return -0.55
        if atom.element == "N":
            return -0.25
        if atom.element == "C":
            return 0.12
        return 0.0
    if atom.resname in {"ASP", "GLU"} and atom.name.startswith(("OD", "OE")):
        return -0.50
    if atom.resname == "ARG" and atom.name in {"NE", "NH1", "NH2"}:
        return 0.33
    if atom.resname == "LYS" and atom.name == "NZ":
        return 1.00
    if atom.name == "N":
        return -0.25
    if atom.name == "O":
        return -0.40
    if atom.name == "C":
        return 0.35
    return 0.0


def _nonbonded_exceptions(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    charges: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    excluded = {_pair(int(i), int(j)) for i, j in bonds.tolist()}
    excluded |= {_pair(int(i), int(k)) for i, _, k in angles.tolist()}
    one_four = {_pair(int(i), int(last)) for i, _, _, last in dihedrals.tolist()}
    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {
        pair: (0.0, 0.0, 0.0) for pair in excluded
    }
    for i, j in sorted(one_four - excluded):
        sigma_ij = 0.5 * (float(sigma[i]) + float(sigma[j]))
        epsilon_ij = (float(epsilon[i]) * float(epsilon[j])) ** 0.5
        exceptions[(i, j)] = (
            float(charges[i] * charges[j]) * 0.8333333,
            sigma_ij,
            epsilon_ij * 0.5,
        )
    if not exceptions:
        return empty_indices(2), *(np.asarray([], dtype=np.float32) for _ in range(3))
    pairs = np.asarray(sorted(exceptions), dtype=np.int32).reshape((-1, 2))
    values = [exceptions[tuple(pair)] for pair in pairs.tolist()]
    return (
        pairs,
        np.asarray([value[0] for value in values], dtype=np.float32),
        np.asarray([value[1] for value in values], dtype=np.float32),
        np.asarray([value[2] for value in values], dtype=np.float32),
    )


def _infer_angles(bonds: np.ndarray) -> np.ndarray:
    neighbors = _neighbors(int(np.max(bonds)) + 1 if bonds.size else 0, bonds.tolist())
    angles = set()
    for center, bonded in neighbors.items():
        for left, right in combinations(sorted(bonded), 2):
            angles.add((left, center, right))
    if not angles:
        return empty_indices(3)
    return np.asarray(sorted(angles), dtype=np.int32)


def _infer_dihedrals(bonds: np.ndarray, *, atom_count: int) -> np.ndarray:
    neighbors = _neighbors(atom_count, bonds.tolist())
    dihedrals = set()
    for j, k in bonds.tolist():
        for i in neighbors[j] - {k}:
            for last in neighbors[k] - {j}:
                candidate = (i, j, k, last)
                reverse = (last, k, j, i)
                dihedrals.add(min(candidate, reverse))
    if not dihedrals:
        return empty_indices(4)
    return np.asarray(sorted(dihedrals), dtype=np.int32)


def _infer_impropers(atoms: list[_BuildAtom], bonds: np.ndarray) -> np.ndarray:
    neighbors = _neighbors(len(atoms), bonds.tolist())
    impropers = []
    for center, bonded in neighbors.items():
        if len(bonded) < 3 or atoms[center].element not in {"C", "N", "P"}:
            continue
        if atoms[center].element == "C" and atoms[center].name in {"CA", "CB", "CG", "CD", "CE"}:
            continue
        first_three = sorted(bonded)[:3]
        impropers.append((first_three[0], center, first_three[1], first_three[2]))
    if not impropers:
        return empty_indices(4)
    return np.asarray(sorted(set(impropers)), dtype=np.int32)


def _filter_dihedrals_by_geometry(
    positions: np.ndarray,
    dihedrals: np.ndarray,
    *,
    min_sine: float = 0.05,
) -> np.ndarray:
    if dihedrals.size == 0:
        return empty_indices(4)
    kept = [
        row
        for row in dihedrals.tolist()
        if _dihedral_is_well_conditioned(positions[list(row)], min_sine=min_sine)
    ]
    if not kept:
        return empty_indices(4)
    return np.asarray(kept, dtype=np.int32)


def _dihedral_is_well_conditioned(coords: np.ndarray, *, min_sine: float) -> bool:
    p0, p1, p2, p3 = coords
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = float(np.linalg.norm(b1))
    b0_norm = float(np.linalg.norm(b0))
    b2_norm = float(np.linalg.norm(b2))
    if min(b0_norm, b1_norm, b2_norm) < 1e-6:
        return False
    b1_unit = b1 / b1_norm
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    sine_left = float(np.linalg.norm(v) / b0_norm)
    sine_right = float(np.linalg.norm(w) / b2_norm)
    return min(sine_left, sine_right) >= min_sine


def _neighbors(atom_count: int, bonds: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    neighbors = {index: set() for index in range(atom_count)}
    for i, j in bonds:
        neighbors[int(i)].add(int(j))
        neighbors[int(j)].add(int(i))
    return neighbors


def _distances(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.asarray([], dtype=np.float32)
    distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    return distances.astype(np.float32)


def _angle_values(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    if angles.size == 0:
        return np.asarray([], dtype=np.float32)
    left = positions[angles[:, 0]] - positions[angles[:, 1]]
    right = positions[angles[:, 2]] - positions[angles[:, 1]]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-8)
    return np.arccos(np.clip(cosine, -0.999999, 0.999999)).astype(np.float32)


def _dihedral_reference_phases(
    positions: np.ndarray,
    dihedrals: np.ndarray,
    *,
    periodicity: float,
) -> np.ndarray:
    if dihedrals.size == 0:
        return np.asarray([], dtype=np.float32)
    phi = np.asarray([_dihedral_angle(positions[list(row)]) for row in dihedrals], dtype=np.float32)
    return (periodicity * phi - np.pi).astype(np.float32)


def _dihedral_angle(coords: np.ndarray) -> float:
    p0, p1, p2, p3 = coords
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = _unit(b1)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return float(np.arctan2(y, x))


def _bond_k(first: _BuildAtom, second: _BuildAtom) -> float:
    if first.element == "H" or second.element == "H":
        return 2500.0
    return 450.0


def _atom_type(atom: _BuildAtom) -> str:
    return f"{atom.resname}:{atom.name}:{atom.element}"


def _pair(i: int, j: int) -> tuple[int, int]:
    return (min(int(i), int(j)), max(int(i), int(j)))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    vector = _unit(vector)
    trial = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(vector, trial))) > 0.8:
        trial = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return _unit(np.cross(vector, trial))


__all__ = ["INTERNAL_FORCE_FIELD_VERSION", "prepare_p2x4_atp_production"]
