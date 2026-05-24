"""Prepare the OpenMM implicit DHFR benchmark as an MLX artifact.

This is a reference-prep script. OpenMM stays outside the product runtime; the
saved artifact is the boundary consumed by `mlx_atomistic.benchmarks.dhfr`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

OPENMM_DHFR_MINIMIZED = Path("vendors/openmm/examples/benchmarks/5dfr_minimized.pdb")
ARTIFACT_DIR = Path("results/dhfr-artifacts/dhfr-implicit")


@dataclass(frozen=True)
class OpenMMApi:
    app: Any
    openmm: Any
    unit: Any


def main() -> None:
    args = _parse_args()
    artifact = prepare_dhfr_implicit(
        repo_root=args.repo_root,
        pdb_path=args.pdb,
        out_dir=args.out,
    )
    if args.json:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    else:
        print(
            "prepared OpenMM implicit DHFR artifact: "
            f"{artifact['artifact_path']} ({artifact['atom_count']} atoms)"
        )


def prepare_dhfr_implicit(
    *,
    repo_root: Path,
    pdb_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build the OpenMM implicit DHFR system and save an MLX artifact."""

    api = _load_openmm()
    app = api.app
    unit = api.unit
    source_pdb = repo_root / pdb_path
    artifact_dir = repo_root / out_dir
    pdb = app.PDBFile(str(source_pdb))
    force_field = app.ForceField("amber99sb.xml", "amber99_obc.xml")
    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )
    prepared, summary = _prepared_from_openmm(
        api=api,
        system=system,
        topology=pdb.topology,
        positions=pdb.positions,
        source_pdb=pdb_path,
    )
    save_prepared_system(prepared, artifact_dir)
    return {
        "status": "ok",
        "artifact_path": str(out_dir),
        "artifact_files": [
            str(out_dir / "prepared_system.json"),
            str(out_dir / "prepared_system.npz"),
            str(out_dir / "view.pdb"),
        ],
        "atom_count": prepared.atom_count,
        **summary,
    }


def _prepared_from_openmm(
    *,
    api: OpenMMApi,
    system: Any,
    topology: Any,
    positions: Any,
    source_pdb: Path,
) -> tuple[PreparedSystem, dict[str, Any]]:
    unit = api.unit
    atoms = list(topology.atoms())
    atom_count = len(atoms)
    positions_a = _quantity_array(positions, unit.angstrom)
    masses = np.asarray(
        [
            system.getParticleMass(index).value_in_unit(unit.dalton)
            for index in range(atom_count)
        ],
        dtype=np.float32,
    )
    atom_payload = _atom_payload(atoms)
    charges, sigma, epsilon, nonbonded_exceptions, nonbonded_setup = _nonbonded_arrays(
        api,
        system,
    )
    (
        gbsa_radius,
        gbsa_scale,
        gbsa_setup,
    ) = _gbsa_arrays(api, system, charges)
    bonds, bond_k, bond_length = _bond_arrays(api, system)
    angles, angle_k, angle_theta = _angle_arrays(api, system)
    dihedrals, dihedral_k, dihedral_periodicity, dihedral_phase = _dihedral_arrays(
        api,
        system,
    )
    constraints, constraint_distance = _constraint_arrays(api, system)
    terms = _required_terms(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        exceptions=nonbonded_exceptions[0],
        constraints=constraints,
    )
    hydrogen_count = int(np.count_nonzero(atom_payload["symbols"] == "H"))
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source={
            "kind": "openmm_forcefield",
            "parser": "scripts/prepare_openmm_dhfr_implicit.py",
            "pdb_path": str(source_pdb),
            "forcefield_files": ["amber99sb.xml", "amber99_obc.xml"],
            "openmm_version": api.openmm.version.version,
        },
        selections={
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "system_charge": float(np.sum(charges)),
            "solvent_model": "implicit",
            "electrostatics_model": "gbsa_obc",
        },
        units={
            "coordinates": "angstrom",
            "length": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
            "force": "kilojoule_per_mole_per_angstrom",
        },
        parameter_source="openmm_amber99sb_obc",
        compatibility_report={
            "production_force_field": True,
            "physical_units": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "hydrogen_mass_repartitioning": "represented_by_masses",
            "electrostatics_model": "cutoff",
            "solvent_model": "implicit",
            "supported_terms": terms,
            "required_terms": terms,
            "unsupported_terms": [],
            "rejected_terms": [],
            "term_counts": {
                "harmonic_bond": int(bonds.shape[0]),
                "harmonic_angle": int(angles.shape[0]),
                "periodic_dihedral": int(dihedrals.shape[0]),
                "nonbonded_exception": int(nonbonded_exceptions[0].shape[0]),
                "distance_constraint": int(constraints.shape[0]),
                "gbsa": 1,
            },
            "force_field_provenance": {
                "source": "OpenMM ForceField",
                "files": ["amber99sb.xml", "amber99_obc.xml"],
                "constraints": "HBonds",
                "hydrogen_mass_amu": 1.5,
            },
        },
        protocol_metadata={
            "gbsa": gbsa_setup,
            "nonbonded": nonbonded_setup,
            "hydrogen_mass_repartitioning": {
                "source": "OpenMM ForceField.createSystem(hydrogenMass=1.5 amu)",
                "status": "represented_by_masses",
                "provenance_available": False,
                "policy": {"virtual_sites_supported": False},
            },
        },
    )
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=atom_payload["symbols"],
        atom_names=atom_payload["atom_names"],
        atom_types=atom_payload["atom_types"],
        residue_names=atom_payload["residue_names"],
        residue_ids=atom_payload["residue_ids"],
        chain_ids=atom_payload["chain_ids"],
        positions=positions_a,
        velocities=np.zeros_like(positions_a, dtype=np.float32),
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
        ligand_mask=np.zeros(atom_count, dtype=bool),
        receptor_mask=np.ones(atom_count, dtype=bool),
        restraint_mask=np.ones(atom_count, dtype=bool),
        reference_positions=positions_a.copy(),
        constraints=constraints,
        constraint_distance=constraint_distance,
        nonbonded_exception_pairs=nonbonded_exceptions[0],
        nonbonded_exception_charge_product=nonbonded_exceptions[1],
        nonbonded_exception_sigma=nonbonded_exceptions[2],
        nonbonded_exception_epsilon=nonbonded_exceptions[3],
        water_mask=np.zeros(atom_count, dtype=bool),
        ion_mask=np.zeros(atom_count, dtype=bool),
        lipid_mask=np.zeros(atom_count, dtype=bool),
        gbsa_radius=gbsa_radius,
        gbsa_scale=gbsa_scale,
    )
    return prepared, {
        "gbsa": gbsa_setup,
        "nonbonded": nonbonded_setup,
        "term_counts": metadata.compatibility_report["term_counts"],
    }


def _load_openmm() -> OpenMMApi:
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except Exception as exc:  # pragma: no cover - optional reference package.
        msg = f"OpenMM import unavailable: {exc}"
        raise RuntimeError(msg) from exc
    return OpenMMApi(app=app, openmm=openmm, unit=unit)


def _atom_payload(atoms: list[Any]) -> dict[str, np.ndarray]:
    symbols: list[str] = []
    atom_names: list[str] = []
    atom_types: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    chain_ids: list[str] = []
    residue_fallback: dict[int, int] = {}
    for atom in atoms:
        element = getattr(atom, "element", None)
        residue = atom.residue
        chain = residue.chain
        residue_key = id(residue)
        residue_fallback.setdefault(residue_key, len(residue_fallback) + 1)
        symbols.append(str(getattr(element, "symbol", "") or atom.name[:1] or "X"))
        atom_names.append(str(atom.name))
        atom_types.append(f"{symbols[-1]}:{atom.name}")
        residue_names.append(str(residue.name))
        residue_ids.append(_residue_id(residue.id, residue_fallback[residue_key]))
        chain_ids.append(str(chain.id or "A")[:1])
    return {
        "symbols": np.asarray(symbols, dtype=str),
        "atom_names": np.asarray(atom_names, dtype=str),
        "atom_types": np.asarray(atom_types, dtype=str),
        "residue_names": np.asarray(residue_names, dtype=str),
        "residue_ids": np.asarray(residue_ids, dtype=np.int32),
        "chain_ids": np.asarray(chain_ids, dtype=str),
    }


def _residue_id(value: str, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        return int(fallback)


def _nonbonded_arrays(
    api: OpenMMApi,
    system: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], dict[str, Any]]:
    force = _single_force(api.openmm.NonbondedForce, system, "NonbondedForce")
    unit = api.unit
    charges: list[float] = []
    sigma: list[float] = []
    epsilon: list[float] = []
    for index in range(force.getNumParticles()):
        charge, sigma_nm, epsilon_kj = force.getParticleParameters(index)
        charges.append(charge.value_in_unit(unit.elementary_charge))
        sigma.append(sigma_nm.value_in_unit(unit.angstrom))
        epsilon.append(epsilon_kj.value_in_unit(unit.kilojoule_per_mole))
    exception_rows: list[tuple[int, int]] = []
    exception_charge_product: list[float] = []
    exception_sigma: list[float] = []
    exception_epsilon: list[float] = []
    for index in range(force.getNumExceptions()):
        i, j, charge_product, sigma_nm, epsilon_kj = force.getExceptionParameters(index)
        exception_rows.append((int(i), int(j)))
        exception_charge_product.append(
            charge_product.value_in_unit(unit.elementary_charge**2)
        )
        exception_sigma.append(sigma_nm.value_in_unit(unit.angstrom))
        exception_epsilon.append(epsilon_kj.value_in_unit(unit.kilojoule_per_mole))
    sigma_array = np.asarray(sigma, dtype=np.float32)
    epsilon_array = np.asarray(epsilon, dtype=np.float32)
    zero_sigma = sigma_array <= 0.0
    if np.any(zero_sigma & (epsilon_array != 0.0)):
        msg = "OpenMM nonbonded export has nonzero-epsilon particles with nonpositive sigma"
        raise ValueError(msg)
    zero_lj_sigma_replacements = int(np.count_nonzero(zero_sigma))
    sigma_array = sigma_array.copy()
    sigma_array[zero_sigma] = 1.0
    return (
        np.asarray(charges, dtype=np.float32),
        sigma_array,
        epsilon_array,
        (
            np.asarray(exception_rows, dtype=np.int32).reshape((-1, 2)),
            np.asarray(exception_charge_product, dtype=np.float32),
            np.asarray(exception_sigma, dtype=np.float32),
            np.asarray(exception_epsilon, dtype=np.float32),
        ),
        {
            "force": "NonbondedForce",
            "method": _nonbonded_method_name(api, force.getNonbondedMethod()),
            "cutoff": force.getCutoffDistance().value_in_unit(unit.angstrom),
            "cutoff_unit": "angstrom",
            "exception_count": force.getNumExceptions(),
            "zero_lj_sigma_replacements": zero_lj_sigma_replacements,
            "zero_lj_sigma_replacement_value": 1.0,
            "zero_lj_sigma_replacement_unit": "angstrom",
        },
    )


def _gbsa_arrays(
    api: OpenMMApi,
    system: Any,
    nonbonded_charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    force = _single_force(api.openmm.GBSAOBCForce, system, "GBSAOBCForce")
    unit = api.unit
    charges: list[float] = []
    radii: list[float] = []
    scales: list[float] = []
    for index in range(force.getNumParticles()):
        charge, radius_nm, scale = force.getParticleParameters(index)
        charges.append(charge.value_in_unit(unit.elementary_charge))
        radii.append(radius_nm.value_in_unit(unit.angstrom))
        scales.append(float(scale))
    gbsa_charges = np.asarray(charges, dtype=np.float32)
    if not np.allclose(gbsa_charges, nonbonded_charges, atol=1.0e-6, rtol=0.0):
        msg = "GBSAOBCForce charges do not match NonbondedForce particle charges"
        raise ValueError(msg)
    surface_area = force.getSurfaceAreaEnergy().value_in_unit(
        unit.kilojoule_per_mole / unit.angstrom**2
    )
    return (
        np.asarray(radii, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        {
            "model": "OBC",
            "force": "GBSAOBCForce",
            "parameter_source": "OpenMM amber99_obc.xml",
            "solvent_dielectric": float(force.getSolventDielectric()),
            "solute_dielectric": float(force.getSoluteDielectric()),
            "surface_area_energy": float(surface_area),
            "surface_area_energy_unit": "kilojoule_per_mole_per_angstrom_squared",
            "probe_radius": 1.4,
            "radius_offset": 0.09,
            "nonbonded_method": _gbsa_method_name(api, force.getNonbondedMethod()),
            "cutoff": force.getCutoffDistance().value_in_unit(unit.angstrom),
            "cutoff_unit": "angstrom",
            "particle_count": force.getNumParticles(),
        },
    )


def _bond_arrays(api: OpenMMApi, system: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        force = _single_force(api.openmm.HarmonicBondForce, system, "HarmonicBondForce")
    except ValueError:
        return empty_indices(2), np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    unit = api.unit
    rows: list[tuple[int, int]] = []
    k_values: list[float] = []
    lengths: list[float] = []
    for index in range(force.getNumBonds()):
        i, j, length_nm, k = force.getBondParameters(index)
        rows.append((int(i), int(j)))
        lengths.append(length_nm.value_in_unit(unit.angstrom))
        k_values.append(k.value_in_unit(unit.kilojoule_per_mole / unit.angstrom**2))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 2)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
    )


def _angle_arrays(api: OpenMMApi, system: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        force = _single_force(api.openmm.HarmonicAngleForce, system, "HarmonicAngleForce")
    except ValueError:
        return empty_indices(3), np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    unit = api.unit
    rows: list[tuple[int, int, int]] = []
    k_values: list[float] = []
    angles: list[float] = []
    for index in range(force.getNumAngles()):
        i, j, k_atom, theta, k = force.getAngleParameters(index)
        rows.append((int(i), int(j), int(k_atom)))
        angles.append(theta.value_in_unit(unit.radian))
        k_values.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 3)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(angles, dtype=np.float32),
    )


def _dihedral_arrays(api: OpenMMApi, system: Any) -> tuple[np.ndarray, ...]:
    try:
        force = _single_force(api.openmm.PeriodicTorsionForce, system, "PeriodicTorsionForce")
    except ValueError:
        return (
            empty_indices(4),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
    unit = api.unit
    rows: list[tuple[int, int, int, int]] = []
    k_values: list[float] = []
    periodicity: list[float] = []
    phase: list[float] = []
    for index in range(force.getNumTorsions()):
        i, j, k_atom, l_atom, n, phase_value, k = force.getTorsionParameters(index)
        rows.append((int(i), int(j), int(k_atom), int(l_atom)))
        periodicity.append(float(n))
        phase.append(phase_value.value_in_unit(unit.radian))
        k_values.append(k.value_in_unit(unit.kilojoule_per_mole))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 4)),
        np.asarray(k_values, dtype=np.float32),
        np.asarray(periodicity, dtype=np.float32),
        np.asarray(phase, dtype=np.float32),
    )


def _constraint_arrays(api: OpenMMApi, system: Any) -> tuple[np.ndarray, np.ndarray]:
    unit = api.unit
    rows: list[tuple[int, int]] = []
    distances: list[float] = []
    for index in range(system.getNumConstraints()):
        i, j, distance = system.getConstraintParameters(index)
        rows.append((int(i), int(j)))
        distances.append(distance.value_in_unit(unit.angstrom))
    return (
        np.asarray(rows, dtype=np.int32).reshape((-1, 2)),
        np.asarray(distances, dtype=np.float32),
    )


def _single_force(force_type: type, system: Any, label: str) -> Any:
    matches = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), force_type)
    ]
    if len(matches) != 1:
        msg = f"expected exactly one {label}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def _quantity_array(values: Any, target_unit: Any) -> np.ndarray:
    return np.asarray(values.value_in_unit(target_unit), dtype=np.float32)


def _required_terms(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    exceptions: np.ndarray,
    constraints: np.ndarray,
) -> list[str]:
    terms = ["nonbonded_lj_coulomb", "gbsa"]
    if bonds.shape[0]:
        terms.append("harmonic_bond")
    if angles.shape[0]:
        terms.append("harmonic_angle")
    if dihedrals.shape[0]:
        terms.append("periodic_dihedral")
    if exceptions.shape[0]:
        terms.append("nonbonded_exception")
    if constraints.shape[0]:
        terms.append("distance_constraint")
    return terms


def _nonbonded_method_name(api: OpenMMApi, method: int) -> str:
    force = api.openmm.NonbondedForce
    names = {
        force.NoCutoff: "NoCutoff",
        force.CutoffNonPeriodic: "CutoffNonPeriodic",
        force.CutoffPeriodic: "CutoffPeriodic",
        force.Ewald: "Ewald",
        force.PME: "PME",
        force.LJPME: "LJPME",
    }
    return names.get(method, f"unknown:{method}")


def _gbsa_method_name(api: OpenMMApi, method: int) -> str:
    force = api.openmm.GBSAOBCForce
    names = {
        force.NoCutoff: "NoCutoff",
        force.CutoffNonPeriodic: "CutoffNonPeriodic",
        force.CutoffPeriodic: "CutoffPeriodic",
    }
    return names.get(method, f"unknown:{method}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pdb", type=Path, default=OPENMM_DHFR_MINIMIZED)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
