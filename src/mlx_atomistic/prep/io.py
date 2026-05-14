"""Read and write prepared-system artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
    empty_string_pairs,
)

JSON_NAME = "prepared_system.json"
NPZ_NAME = "prepared_system.npz"
VIEW_PDB_NAME = "view.pdb"

NPZ_ARRAY_NAMES = (
    "symbols",
    "atom_names",
    "atom_types",
    "residue_names",
    "residue_ids",
    "chain_ids",
    "positions",
    "velocities",
    "masses",
    "charges",
    "sigma",
    "epsilon",
    "bonds",
    "bond_k",
    "bond_length",
    "angles",
    "angle_k",
    "angle_theta",
    "dihedrals",
    "dihedral_k",
    "dihedral_periodicity",
    "dihedral_phase",
    "nonbonded_pairs",
    "ligand_mask",
    "receptor_mask",
    "restraint_mask",
    "reference_positions",
)

OPTIONAL_NPZ_ARRAY_DEFAULTS = {
    "cell_lengths": lambda: np.asarray([], dtype=np.float32),
    "constraints": lambda: empty_indices(2),
    "constraint_distance": lambda: np.asarray([], dtype=np.float32),
    "impropers": lambda: empty_indices(4),
    "improper_k": lambda: np.asarray([], dtype=np.float32),
    "improper_periodicity": lambda: np.asarray([], dtype=np.float32),
    "improper_phase": lambda: np.asarray([], dtype=np.float32),
    "nonbonded_exception_pairs": lambda: empty_indices(2),
    "nonbonded_exception_charge_product": lambda: np.asarray([], dtype=np.float32),
    "nonbonded_exception_sigma": lambda: np.asarray([], dtype=np.float32),
    "nonbonded_exception_epsilon": lambda: np.asarray([], dtype=np.float32),
    "water_mask": lambda: np.asarray([], dtype=bool),
    "ion_mask": lambda: np.asarray([], dtype=bool),
    "lipid_mask": lambda: np.asarray([], dtype=bool),
    "pme_mesh_shape": lambda: np.asarray([], dtype=np.int32),
    "pme_alpha": lambda: np.asarray([], dtype=np.float32),
    "pme_real_cutoff": lambda: np.asarray([], dtype=np.float32),
    "pme_assignment_order": lambda: np.asarray([], dtype=np.int32),
    "pme_charge_tolerance": lambda: np.asarray([], dtype=np.float32),
    "pme_deconvolve_assignment": lambda: np.asarray([], dtype=bool),
    "charmm_cmap_terms": lambda: empty_indices(8),
    "charmm_cmap_grid_indices": lambda: np.asarray([], dtype=np.int32),
    "charmm_cmap_grids": lambda: np.empty((0, 0, 0), dtype=np.float32),
    "urey_bradley_terms": lambda: empty_indices(3),
    "urey_bradley_k": lambda: np.asarray([], dtype=np.float32),
    "urey_bradley_distance": lambda: np.asarray([], dtype=np.float32),
    "nbfix_pairs": lambda: empty_indices(2),
    "nbfix_sigma": lambda: np.asarray([], dtype=np.float32),
    "nbfix_epsilon": lambda: np.asarray([], dtype=np.float32),
    "nbfix_type_pairs": empty_string_pairs,
    "nbfix_type_sigma": lambda: np.asarray([], dtype=np.float32),
    "nbfix_type_epsilon": lambda: np.asarray([], dtype=np.float32),
}


def _as_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [convert(v) for v in value]
        return value

    return {str(k): convert(v) for k, v in payload.items()}


def save_prepared_system(prepared: PreparedSystem, out_dir: str | Path) -> None:
    """Write `prepared_system.json`, `prepared_system.npz`, and `view.pdb`."""

    prepared.validate()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metadata = _as_jsonable(prepared.metadata.to_json_dict())
    (out_path / JSON_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        out_path / NPZ_NAME,
        **{
            name: np.asarray(getattr(prepared, name))
            for name in (*NPZ_ARRAY_NAMES, *OPTIONAL_NPZ_ARRAY_DEFAULTS)
        },
    )
    write_view_pdb(out_path / VIEW_PDB_NAME, prepared)


def load_prepared_system(path: str | Path) -> PreparedSystem:
    """Load a prepared system from a directory or `prepared_system.json` path."""

    input_path = Path(path)
    if input_path.is_file():
        base_dir = input_path.parent
        json_path = input_path
    else:
        base_dir = input_path
        json_path = base_dir / JSON_NAME
    npz_path = base_dir / NPZ_NAME
    if not json_path.exists():
        msg = f"missing prepared-system metadata: {json_path}"
        raise FileNotFoundError(msg)
    if not npz_path.exists():
        msg = f"missing prepared-system arrays: {npz_path}"
        raise FileNotFoundError(msg)

    metadata = PreparedSystemMetadata.from_json_dict(json.loads(json_path.read_text()))
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in NPZ_ARRAY_NAMES}
        for name, default_factory in OPTIONAL_NPZ_ARRAY_DEFAULTS.items():
            payload[name] = np.asarray(data[name]) if name in data else default_factory()
    prepared = PreparedSystem(metadata=metadata, **payload)
    prepared.validate()
    return prepared


def write_view_pdb(path: str | Path, prepared: PreparedSystem) -> None:
    """Write a compact PDB for notebook/debug visualization."""

    prepared.validate()
    lines = [
        "REMARK Generated by mlx_atomistic.prep. Coordinates are prepared-system positions.",
        f"REMARK Parameter source: {prepared.metadata.parameter_source}",
    ]
    for index in range(prepared.atom_count):
        record = "HETATM" if bool(prepared.ligand_mask[index]) else "ATOM  "
        serial = index + 1
        atom_name = str(prepared.atom_names[index])[:4]
        resname = str(prepared.residue_names[index])[:3]
        chain = (str(prepared.chain_ids[index]) or "A")[:1]
        resid = int(prepared.residue_ids[index])
        x, y, z = np.asarray(prepared.positions[index], dtype=np.float32)
        element = str(prepared.symbols[index]).strip().upper()[:2].rjust(2)
        lines.append(
            f"{record}{serial:5d} {atom_name:^4s} {resname:>3s} {chain:1s}"
            f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )
    for bond in np.asarray(prepared.bonds, dtype=np.int32):
        i, j = int(bond[0]) + 1, int(bond[1]) + 1
        lines.append(f"CONECT{i:5d}{j:5d}")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def synthetic_prepared_system() -> PreparedSystem:
    """Small two-atom fixture used by tests and notebook smoke paths."""

    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at="fixture",
        source={"kind": "synthetic"},
        selections={"description": "two-atom harmonic fixture"},
        units={
            "length": "angstrom_like_reduced",
            "mass": "atomic_mass",
            "charge": "elementary_charge_like_reduced",
            "energy": "mlx_reduced",
        },
        parameter_source="synthetic_test",
        compatibility_report={"supported_terms": ["bonds"], "unsupported_terms": []},
        warnings=[],
    )
    return PreparedSystem(
        metadata=metadata,
        symbols=np.asarray(["C", "O"], dtype=str),
        atom_names=np.asarray(["C1", "O1"], dtype=str),
        atom_types=np.asarray(["C", "O"], dtype=str),
        residue_names=np.asarray(["LIG", "LIG"], dtype=str),
        residue_ids=np.asarray([1, 1], dtype=np.int32),
        chain_ids=np.asarray(["A", "A"], dtype=str),
        positions=np.asarray([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=np.float32),
        velocities=np.zeros((2, 3), dtype=np.float32),
        masses=np.asarray([12.011, 15.999], dtype=np.float32),
        charges=np.asarray([0.1, -0.1], dtype=np.float32),
        sigma=np.asarray([3.4, 3.0], dtype=np.float32),
        epsilon=np.asarray([0.001, 0.001], dtype=np.float32),
        bonds=np.asarray([[0, 1]], dtype=np.int32),
        bond_k=np.asarray([2.0], dtype=np.float32),
        bond_length=np.asarray([1.25], dtype=np.float32),
        angles=empty_indices(3),
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=empty_indices(4),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=empty_indices(2),
        ligand_mask=np.asarray([True, True]),
        receptor_mask=np.asarray([False, False]),
        restraint_mask=np.asarray([False, False]),
        reference_positions=np.asarray([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=np.float32),
    )


__all__ = [
    "JSON_NAME",
    "NPZ_NAME",
    "VIEW_PDB_NAME",
    "load_prepared_system",
    "save_prepared_system",
    "synthetic_prepared_system",
    "write_view_pdb",
]
