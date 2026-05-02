"""Bundled T4 lysozyme L99A / benzene steering fixture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations

import numpy as np

from atomistic_prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

T4L_BENZENE_PARAMETER_SOURCE = "mlx_internal_t4l_benzene_forced_smd_demo_v2"

_POCKET_PDB = """
ATOM    635  N   ILE A  78     -36.183   6.986  11.886  1.00 10.94           N
ATOM    636  CA  ILE A  78     -36.582   6.222  10.734  1.00 10.92           C
ATOM    637  C   ILE A  78     -37.774   5.324  11.043  1.00 13.98           C
ATOM    638  O   ILE A  78     -38.724   5.283  10.265  1.00 13.18           O
ATOM    639  CB  ILE A  78     -35.401   5.380  10.249  1.00 10.52           C
ATOM    640  CG1 ILE A  78     -34.371   6.290   9.581  1.00 10.56           C
ATOM    641  CG2 ILE A  78     -35.879   4.318   9.244  1.00 12.09           C
ATOM    642  CD1 ILE A  78     -33.038   5.601   9.375  1.00 13.01           C
ATOM    678  N   LEU A  84     -40.212   6.766   5.797  1.00 11.62           N
ATOM    679  CA  LEU A  84     -38.883   6.177   5.848  1.00 11.69           C
ATOM    680  C   LEU A  84     -38.878   4.709   6.237  1.00 10.57           C
ATOM    681  O   LEU A  84     -38.112   3.921   5.688  1.00 11.67           O
ATOM    682  CB  LEU A  84     -37.993   6.936   6.846  1.00  9.65           C
ATOM    683  CG  LEU A  84     -37.745   8.409   6.463  1.00 12.18           C
ATOM    684  CD1 LEU A  84     -36.910   9.087   7.566  1.00 12.42           C
ATOM    685  CD2 LEU A  84     -37.084   8.614   5.093  1.00 15.80           C
ATOM    702  N   VAL A  87     -39.121   1.963   3.433  1.00  9.74           N
ATOM    703  CA  VAL A  87     -37.930   1.885   2.597  1.00 11.11           C
ATOM    704  C   VAL A  87     -36.791   1.237   3.383  1.00 10.32           C
ATOM    705  O   VAL A  87     -36.100   0.331   2.895  1.00 10.16           O
ATOM    706  CB  VAL A  87     -37.509   3.272   2.127  1.00 11.98           C
ATOM    707  CG1 VAL A  87     -36.333   3.139   1.142  1.00 12.58           C
ATOM    708  CG2 VAL A  87     -38.699   3.987   1.446  1.00 15.36           C
ATOM    709  N   TYR A  88     -36.578   1.680   4.621  1.00  9.24           N
ATOM    710  CA  TYR A  88     -35.568   1.080   5.450  1.00  9.49           C
ATOM    711  C   TYR A  88     -35.744  -0.418   5.621  1.00 10.08           C
ATOM    712  O   TYR A  88     -34.789  -1.173   5.455  1.00 10.93           O
ATOM    713  CB  TYR A  88     -35.580   1.814   6.810  1.00  9.85           C
ATOM    714  CG  TYR A  88     -34.530   1.370   7.804  1.00 11.01           C
ATOM    715  CD1 TYR A  88     -33.233   1.889   7.762  1.00 10.83           C
ATOM    716  CD2 TYR A  88     -34.837   0.430   8.768  1.00 14.94           C
ATOM    717  CE1 TYR A  88     -32.281   1.491   8.659  1.00 12.39           C
ATOM    718  CE2 TYR A  88     -33.891   0.039   9.684  1.00 17.74           C
ATOM    719  CZ  TYR A  88     -32.609   0.578   9.613  1.00 16.57           C
ATOM    720  OH  TYR A  88     -31.644   0.173  10.516  1.00 20.05           O
ATOM    735  N   LEU A  91     -34.526  -2.607   2.808  1.00  9.62           N
ATOM    736  CA  LEU A  91     -33.102  -2.614   2.457  1.00  8.41           C
ATOM    737  C   LEU A  91     -32.346  -3.751   3.149  1.00  8.40           C
ATOM    738  O   LEU A  91     -32.765  -4.229   4.211  1.00 11.20           O
ATOM    739  CB  LEU A  91     -32.449  -1.283   2.891  1.00  9.23           C
ATOM    740  CG  LEU A  91     -32.958  -0.037   2.149  1.00  9.80           C
ATOM    741  CD1 LEU A  91     -32.397   1.224   2.837  1.00 10.87           C
ATOM    742  CD2 LEU A  91     -32.538  -0.052   0.699  1.00 12.15           C
ATOM    796  N   ALA A  99     -28.726   4.633   4.982  1.00  7.80           N
ATOM    797  CA  ALA A  99     -29.914   5.373   5.383  1.00  8.04           C
ATOM    798  C   ALA A  99     -29.740   6.056   6.723  1.00  8.65           C
ATOM    799  O   ALA A  99     -30.160   7.200   6.895  1.00  9.35           O
ATOM    800  CB  ALA A  99     -31.122   4.413   5.448  1.00  9.95           C
ATOM    825  N   VAL A 103     -29.888  10.174   7.681  1.00  8.57           N
ATOM    826  CA  VAL A 103     -30.761  10.714   8.732  1.00  9.34           C
ATOM    827  C   VAL A 103     -29.935  11.359   9.837  1.00  9.13           C
ATOM    828  O   VAL A 103     -30.322  12.409  10.378  1.00 10.67           O
ATOM    829  CB  VAL A 103     -31.698   9.625   9.267  1.00  9.11           C
ATOM    830  CG1 VAL A 103     -32.446  10.099  10.534  1.00 12.24           C
ATOM    831  CG2 VAL A 103     -32.706   9.260   8.175  1.00 11.17           C
ATOM    893  N   VAL A 111     -34.641  13.912   5.211  0.90 14.46           N
ATOM    894  CA  VAL A 111     -34.518  12.732   4.375  0.90 11.88           C
ATOM    895  C   VAL A 111     -35.876  12.282   3.831  0.90 12.90           C
ATOM    896  O   VAL A 111     -35.977  11.789   2.695  0.90 13.74           O
ATOM    897  CB  VAL A 111     -33.861  11.568   5.125  0.90 12.73           C
ATOM    898  CG1 VAL A 111     -33.796  10.333   4.227  0.90 13.29           C
ATOM    899  CG2 VAL A 111     -32.451  11.955   5.559  0.90 14.91           C
ATOM    941  N   LEU A 118     -36.253   7.839  -2.476  1.00 10.63           N
ATOM    942  CA  LEU A 118     -36.776   6.792  -1.583  1.00 10.30           C
ATOM    943  C   LEU A 118     -37.526   5.743  -2.386  1.00 11.22           C
ATOM    944  O   LEU A 118     -37.380   4.543  -2.114  1.00 12.32           O
ATOM    945  CB  LEU A 118     -37.708   7.397  -0.539  1.00 11.60           C
ATOM    946  CG  LEU A 118     -36.983   8.232   0.523  1.00 12.09           C
ATOM    947  CD1 LEU A 118     -37.979   9.063   1.302  1.00 14.33           C
ATOM    948  CD2 LEU A 118     -36.180   7.340   1.488  1.00 14.24           C
ATOM    981  N   LEU A 121     -35.039   3.601  -4.222  1.00  9.47           N
ATOM    982  CA  LEU A 121     -34.400   2.675  -3.295  1.00 10.23           C
ATOM    983  C   LEU A 121     -35.322   1.514  -2.940  1.00 10.86           C
ATOM    984  O   LEU A 121     -34.865   0.361  -2.926  1.00 10.75           O
ATOM    985  CB  LEU A 121     -33.922   3.420  -2.015  1.00  8.64           C
ATOM    986  CG  LEU A 121     -32.748   4.385  -2.211  1.00  9.72           C
ATOM    987  CD1 LEU A 121     -32.564   5.183  -0.897  1.00 10.86           C
ATOM    988  CD2 LEU A 121     -31.429   3.682  -2.587  1.00 11.47           C
ATOM   1253  N   PHE A 153     -25.264   3.061  -1.106  1.00 10.02           N
ATOM   1254  CA  PHE A 153     -26.583   2.701  -1.657  1.00 10.71           C
ATOM   1255  C   PHE A 153     -26.493   2.029  -3.022  1.00 12.61           C
ATOM   1256  O   PHE A 153     -27.307   1.136  -3.343  1.00 13.17           O
ATOM   1257  CB  PHE A 153     -27.466   3.956  -1.817  1.00  9.94           C
ATOM   1258  CG  PHE A 153     -28.119   4.443  -0.554  1.00  9.62           C
ATOM   1259  CD1 PHE A 153     -28.957   3.598   0.192  1.00  9.76           C
ATOM   1260  CD2 PHE A 153     -27.987   5.768  -0.153  1.00 10.75           C
ATOM   1261  CE1 PHE A 153     -29.633   4.080   1.317  1.00 10.31           C
ATOM   1262  CE2 PHE A 153     -28.647   6.258   0.992  1.00 10.93           C
ATOM   1263  CZ  PHE A 153     -29.471   5.388   1.729  1.00  8.86           C
HETATM 1360  C1  BNZ A 200     -32.969   6.196   2.877  0.70 15.06           C
HETATM 1361  C2  BNZ A 200     -32.945   7.046   3.973  0.70 12.84           C
HETATM 1362  C3  BNZ A 200     -33.719   6.798   5.113  0.70 12.24           C
HETATM 1363  C4  BNZ A 200     -34.540   5.680   5.143  0.70 13.09           C
HETATM 1364  C5  BNZ A 200     -34.545   4.825   4.044  0.70 12.54           C
HETATM 1365  C6  BNZ A 200     -33.787   5.069   2.915  0.70 14.23           C
"""

ATOMIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
SIGMA = {"H": 1.2, "C": 3.4, "N": 3.25, "O": 2.96, "S": 3.55}
EPSILON = {"H": 0.002, "C": 0.010, "N": 0.012, "O": 0.012, "S": 0.012}


@dataclass(frozen=True)
class _Atom:
    name: str
    resname: str
    resid: int
    chain_id: str
    element: str
    position: np.ndarray
    ligand: bool


def prepare_t4l_benzene() -> PreparedSystem:
    """Build the bundled T4L L99A / benzene pocket SMD fixture."""

    atoms = _parse_fixture_atoms()
    atoms, bonds = _add_fixture_hydrogens_and_bonds(atoms)
    positions = np.stack([atom.position for atom in atoms]).astype(np.float32)
    ligand_mask = np.asarray([atom.ligand for atom in atoms], dtype=bool)
    receptor_mask = ~ligand_mask
    masses = np.asarray(
        [ATOMIC_MASSES.get(atom.element, 12.011) for atom in atoms],
        dtype=np.float32,
    )
    charges = np.asarray([_charge(atom) for atom in atoms], dtype=np.float32)
    sigma = np.asarray([SIGMA.get(atom.element, 3.2) for atom in atoms], dtype=np.float32)
    epsilon = np.asarray([EPSILON.get(atom.element, 0.008) for atom in atoms], dtype=np.float32)

    bonds_array = np.asarray(sorted(bonds), dtype=np.int32).reshape((-1, 2))
    bond_length = _distances(positions, bonds_array)
    angles = _infer_angles(bonds_array)
    angle_theta = _angle_values(positions, angles)
    dihedrals = _infer_dihedrals(bonds_array)
    dihedral_periodicity = np.full((dihedrals.shape[0],), 3.0, dtype=np.float32)
    dihedral_phase = _dihedral_reference_phases(positions, dihedrals, dihedral_periodicity)
    impropers = _infer_impropers(bonds_array)
    improper_periodicity = np.full((impropers.shape[0],), 2.0, dtype=np.float32)
    improper_phase = _dihedral_reference_phases(positions, impropers, improper_periodicity)
    constrained_pairs = [
        pair
        for pair in bonds_array.tolist()
        if atoms[pair[0]].element == "H" or atoms[pair[1]].element == "H"
    ]
    constraints = np.asarray(constrained_pairs, dtype=np.int32).reshape((-1, 2))
    one_four_pairs = _one_four_pairs(dihedrals, set(map(tuple, bonds_array.tolist())))
    exception_pairs = np.asarray(one_four_pairs, dtype=np.int32).reshape((-1, 2))
    exception_qprod, exception_sigma, exception_epsilon = _exception_parameters(
        charges,
        sigma,
        epsilon,
        exception_pairs,
    )
    ligand_center = positions[ligand_mask].mean(axis=0)
    receptor_center = positions[receptor_mask].mean(axis=0)
    exit_vector = ligand_center - receptor_center
    exit_vector = exit_vector / np.linalg.norm(exit_vector)
    hydrogen_count = int(np.count_nonzero(np.asarray([atom.element for atom in atoms]) == "H"))

    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source={
            "kind": "bundled_fixture",
            "pdb_id": "4W52",
            "pdb_title": "T4 Lysozyme L99A with Benzene Bound",
            "source_url": "https://www.rcsb.org/structure/4W52",
            "pdb_doi": "10.2210/pdb4W52/pdb",
        },
        selections={
            "ligand_resname": "BNZ",
            "ligand_resid": 200,
            "pocket_residue_count": len(
                {(a.chain_id, a.resid, a.resname) for a in atoms if not a.ligand}
            ),
            "atom_count": len(atoms),
            "ligand_atom_count": int(np.count_nonzero(ligand_mask)),
            "hydrogen_count": hydrogen_count,
            "steering_bias_direction": exit_vector.astype(float).tolist(),
            "steering_direction_basis": (
                "Heuristic radial vector from receptor-pocket atom center to benzene center. "
                "Use only as a forced-SMD method demo, not as an inferred egress pathway."
            ),
            "recommended_steering_velocity_A_per_ps": 0.5,
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source=T4L_BENZENE_PARAMETER_SOURCE,
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": False,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "supported_terms": [
                "harmonic_bond",
                "harmonic_angle",
                "periodic_dihedral",
                "periodic_improper",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
                "positional_restraint",
            ],
            "required_terms": [
                "harmonic_bond",
                "harmonic_angle",
                "periodic_dihedral",
                "periodic_improper",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
                "positional_restraint",
            ],
            "unsupported_terms": [],
            "rejected_terms": [],
        },
        warnings=[
            (
                "Bundled soluble SMD benchmark fixture. It is complete enough to run MLX "
                "fixed-topology steering, but it is not a validated CHARMM/AMBER "
                "production force field."
            ),
            (
                "Visible ligand translation is generated by an explicit steered-COM bias "
                "along a heuristic radial direction. This is not natural diffusion or a "
                "scientifically inferred egress pathway."
            ),
        ],
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=np.asarray([atom.element for atom in atoms], dtype=str),
        atom_names=np.asarray([atom.name for atom in atoms], dtype=str),
        atom_types=np.asarray([atom.element for atom in atoms], dtype=str),
        residue_names=np.asarray([atom.resname for atom in atoms], dtype=str),
        residue_ids=np.asarray([atom.resid for atom in atoms], dtype=np.int32),
        chain_ids=np.asarray([atom.chain_id for atom in atoms], dtype=str),
        positions=positions,
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=masses,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        bonds=bonds_array,
        bond_k=np.full((bonds_array.shape[0],), 250.0, dtype=np.float32),
        bond_length=bond_length,
        angles=angles,
        angle_k=np.full((angles.shape[0],), 35.0, dtype=np.float32),
        angle_theta=angle_theta,
        dihedrals=dihedrals,
        dihedral_k=np.full((dihedrals.shape[0],), 0.4, dtype=np.float32),
        dihedral_periodicity=dihedral_periodicity,
        dihedral_phase=dihedral_phase,
        nonbonded_pairs=empty_indices(2),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=receptor_mask,
        reference_positions=positions.copy(),
        constraints=constraints,
        constraint_distance=_distances(positions, constraints),
        impropers=impropers,
        improper_k=np.full((impropers.shape[0],), 0.2, dtype=np.float32),
        improper_periodicity=improper_periodicity,
        improper_phase=improper_phase,
        nonbonded_exception_pairs=exception_pairs,
        nonbonded_exception_charge_product=exception_qprod,
        nonbonded_exception_sigma=exception_sigma,
        nonbonded_exception_epsilon=exception_epsilon,
    )
    prepared.validate()
    return prepared


def _parse_fixture_atoms() -> list[_Atom]:
    atoms: list[_Atom] = []
    for line in _POCKET_PDB.splitlines():
        if not line.strip():
            continue
        record = line[:6].strip()
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain_id = line[21:22].strip() or "A"
        resid = int(line[22:26])
        element = line[76:78].strip().upper() or name[0].upper()
        position = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        )
        atoms.append(
            _Atom(
                name=name,
                resname=resname,
                resid=resid,
                chain_id=chain_id,
                element=element,
                position=position,
                ligand=record == "HETATM" and resname == "BNZ",
            )
        )
    return atoms


def _add_fixture_hydrogens_and_bonds(
    atoms: list[_Atom],
) -> tuple[list[_Atom], set[tuple[int, int]]]:
    built = list(atoms)
    bonds = _infer_heavy_bonds(built)
    ligand_indices = [i for i, atom in enumerate(built) if atom.ligand]
    ligand_center = np.stack([built[i].position for i in ligand_indices]).mean(axis=0)
    for i in ligand_indices:
        atom = built[i]
        direction = atom.position - ligand_center
        direction = direction / np.linalg.norm(direction)
        h_index = len(built)
        built.append(
            _Atom(
                name=f"H{atom.name[1:]}",
                resname=atom.resname,
                resid=atom.resid,
                chain_id=atom.chain_id,
                element="H",
                position=(atom.position + 1.09 * direction).astype(np.float32),
                ligand=True,
            )
        )
        bonds.add(tuple(sorted((i, h_index))))
    residue_centers = _residue_centers(built)
    for i, atom in list(enumerate(built)):
        if atom.ligand or atom.element == "H":
            continue
        should_add = atom.name == "N" or atom.name == "OH"
        if not should_add:
            continue
        center = residue_centers[(atom.chain_id, atom.resid, atom.resname)]
        direction = atom.position - center
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        h_index = len(built)
        built.append(
            _Atom(
                name=f"H{atom.name[:2]}",
                resname=atom.resname,
                resid=atom.resid,
                chain_id=atom.chain_id,
                element="H",
                position=(atom.position + 1.0 * direction).astype(np.float32),
                ligand=False,
            )
        )
        bonds.add(tuple(sorted((i, h_index))))
    return built, bonds


def _infer_heavy_bonds(atoms: list[_Atom]) -> set[tuple[int, int]]:
    bonds: set[tuple[int, int]] = set()
    ligand = [i for i, atom in enumerate(atoms) if atom.ligand]
    ligand_by_name = {atoms[i].name: i for i in ligand}
    benzene_ring = [
        ("C1", "C2"),
        ("C2", "C3"),
        ("C3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "C1"),
    ]
    for left, right in benzene_ring:
        bonds.add(tuple(sorted((ligand_by_name[left], ligand_by_name[right]))))
    for i, j in combinations(range(len(atoms)), 2):
        if atoms[i].ligand or atoms[j].ligand:
            continue
        if (atoms[i].chain_id, atoms[i].resid, atoms[i].resname) != (
            atoms[j].chain_id,
            atoms[j].resid,
            atoms[j].resname,
        ):
            continue
        max_distance = COVALENT_RADII[atoms[i].element] + COVALENT_RADII[atoms[j].element] + 0.45
        distance = float(np.linalg.norm(atoms[i].position - atoms[j].position))
        if 0.45 <= distance <= max_distance:
            bonds.add((i, j))
    return bonds


def _residue_centers(atoms: list[_Atom]) -> dict[tuple[str, int, str], np.ndarray]:
    grouped: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for atom in atoms:
        if atom.ligand or atom.element == "H":
            continue
        grouped.setdefault((atom.chain_id, atom.resid, atom.resname), []).append(atom.position)
    return {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}


def _charge(atom: _Atom) -> float:
    if atom.ligand:
        return 0.0
    if atom.element == "N":
        return 0.10
    if atom.element == "O":
        return -0.10
    if atom.element == "H":
        return 0.04
    return 0.0


def _neighbors(bonds: np.ndarray) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = {}
    for i, j in np.asarray(bonds, dtype=np.int32).tolist():
        neighbors.setdefault(int(i), set()).add(int(j))
        neighbors.setdefault(int(j), set()).add(int(i))
    return neighbors


def _infer_angles(bonds: np.ndarray) -> np.ndarray:
    neighbors = _neighbors(bonds)
    angles = set()
    for center, bonded in neighbors.items():
        for i, k in combinations(sorted(bonded), 2):
            angles.add((i, center, k))
    return np.asarray(sorted(angles), dtype=np.int32).reshape((-1, 3))


def _infer_dihedrals(bonds: np.ndarray) -> np.ndarray:
    neighbors = _neighbors(bonds)
    dihedrals = set()
    for j, k in np.asarray(bonds, dtype=np.int32).tolist():
        for i in neighbors.get(int(j), set()) - {int(k)}:
            for m in neighbors.get(int(k), set()) - {int(j)}:
                if len({i, int(j), int(k), m}) == 4:
                    dihedrals.add((i, int(j), int(k), m))
    return np.asarray(sorted(dihedrals), dtype=np.int32).reshape((-1, 4))


def _infer_impropers(bonds: np.ndarray) -> np.ndarray:
    neighbors = _neighbors(bonds)
    impropers = []
    for center, bonded in sorted(neighbors.items()):
        if len(bonded) >= 3:
            a, b, c = sorted(bonded)[:3]
            impropers.append((a, center, b, c))
    return np.asarray(impropers, dtype=np.int32).reshape((-1, 4))


def _distances(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _angle_values(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    if angles.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    values = []
    for i, j, k in angles:
        left = positions[i] - positions[j]
        right = positions[k] - positions[j]
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        values.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def _dihedral_reference_phases(
    positions: np.ndarray,
    dihedrals: np.ndarray,
    periodicity: np.ndarray,
) -> np.ndarray:
    if dihedrals.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    phases = []
    for row, n in zip(dihedrals, periodicity, strict=True):
        phi = _dihedral_angle(*(positions[index] for index in row))
        phases.append(np.pi - float(n) * phi)
    return np.asarray(phases, dtype=np.float32)


def _dihedral_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = b - a
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _one_four_pairs(
    dihedrals: np.ndarray,
    bond_pairs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    pairs = set()
    for i, _, _, j in dihedrals.tolist():
        pair = tuple(sorted((int(i), int(j))))
        if pair not in bond_pairs:
            pairs.add(pair)
    return sorted(pairs)


def _exception_parameters(
    charges: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairs.shape[0] == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty
    qprod = charges[pairs[:, 0]] * charges[pairs[:, 1]] * 0.8333333
    sigma_ij = 0.5 * (sigma[pairs[:, 0]] + sigma[pairs[:, 1]])
    epsilon_ij = np.sqrt(epsilon[pairs[:, 0]] * epsilon[pairs[:, 1]]) * 0.5
    return qprod.astype(np.float32), sigma_ij.astype(np.float32), epsilon_ij.astype(np.float32)


__all__ = ["T4L_BENZENE_PARAMETER_SOURCE", "prepare_t4l_benzene"]
