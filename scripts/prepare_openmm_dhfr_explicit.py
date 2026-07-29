"""Prepare the exact OpenMM ``5dfr`` solvated PME benchmark for MLX.

OpenMM is a reference-only construction surface. The resulting prepared
artifact and source manifest are the boundary consumed by the MLX runtime.
No dynamics are run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.artifacts import (
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.mm import molecule_identity_sha256, normalize_molecule_ids
from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

try:
    from scripts import prepare_openmm_dhfr_implicit as _common
except ImportError:  # pragma: no cover - direct script execution.
    import prepare_openmm_dhfr_implicit as _common

CASE_ID = "dhfr-5dfr-pme"
MANIFEST_SCHEMA = "mlx-atomistic.openmm-5dfr-preparation.v1"
MANIFEST_NAME = "source_manifest.json"
OPENMM_DHFR_SOLVATED = Path(
    "vendors/openmm/examples/benchmarks/5dfr_solv-cube_equil.pdb"
)
OPENMM_BENCHMARK_SOURCE = Path("vendors/openmm/examples/benchmarks/benchmark.py")
DEFAULT_OUT = Path("results/dhfr-npt-closure/prepared")
FORCE_FIELD_FILES = ("amber99sb.xml", "tip3p.xml")
EXPECTED_FORCE_CLASSES = {
    "CMMotionRemover": 1,
    "HarmonicAngleForce": 1,
    "HarmonicBondForce": 1,
    "NonbondedForce": 1,
    "PeriodicTorsionForce": 1,
}
PARAMETER_ARRAY_NAMES = (
    "masses",
    "charges",
    "sigma",
    "epsilon",
    "bond_k",
    "bond_length",
    "angle_k",
    "angle_theta",
    "dihedral_k",
    "dihedral_periodicity",
    "dihedral_phase",
    "constraint_distance",
    "nonbonded_exception_charge_product",
    "nonbonded_exception_sigma",
    "nonbonded_exception_epsilon",
)
INDEX_ARRAY_NAMES = (
    "bonds",
    "angles",
    "dihedrals",
    "constraints",
    "nonbonded_exception_pairs",
)


def main() -> None:
    args = _parse_args()
    result = prepare_openmm_5dfr(
        repo_root=args.repo_root,
        pdb_path=args.pdb,
        out_dir=args.out,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "prepared exact OpenMM 5dfr artifact: "
            f"{result['artifact_path']} ({result['atom_count']} atoms, "
            f"{result['molecule_count']} molecules)"
        )


def prepare_openmm_5dfr(
    *,
    repo_root: Path,
    pdb_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build and export the selected Amber99SB/TIP3P PME ``5dfr`` system."""

    root = Path(repo_root).resolve()
    source_pdb = _require_exact_source(root, pdb_path)
    benchmark_source = (root / OPENMM_BENCHMARK_SOURCE).resolve()
    if not benchmark_source.is_file():
        msg = f"missing OpenMM benchmark construction source: {OPENMM_BENCHMARK_SOURCE}"
        raise FileNotFoundError(msg)
    artifact_dir = _results_output_path(root, out_dir)

    api = _common._load_openmm()
    app = api.app
    unit = api.unit
    pdb = app.PDBFile(str(source_pdb))
    force_field = app.ForceField(*FORCE_FIELD_FILES)
    system = force_field.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
        hydrogenMass=1.5 * unit.amu,
    )
    nonbonded = _common._single_force(
        api.openmm.NonbondedForce,
        system,
        "NonbondedForce",
    )
    nonbonded.setEwaldErrorTolerance(5.0e-4)
    nonbonded.setUseDispersionCorrection(True)
    force_class_counts = _validated_force_classes(system)
    context_payload = _resolved_context_payload(
        api=api,
        system=system,
        positions=pdb.positions,
        nonbonded=nonbonded,
    )
    prepared = _prepared_from_openmm(
        api=api,
        system=system,
        topology=pdb.topology,
        positions=pdb.positions,
        source_pdb=OPENMM_DHFR_SOLVATED,
        force_class_counts=force_class_counts,
        context_payload=context_payload,
    )
    save_prepared_system(prepared, artifact_dir)

    artifact = load_prepared_mlx_artifact(artifact_dir, require_production=True)
    runtime_system, _, _ = build_mlx_system_from_artifact(
        artifact,
        eager_nonbonded_pair_limit=0,
    )
    artifact_hash = artifact.molecule_identity_sha256
    runtime_hash = molecule_identity_sha256(
        runtime_system.molecule_ids,
        atom_count=runtime_system.atom_count,
    )
    expected_hash = str(context_payload["molecule_identity_sha256"])
    expected_count = int(context_payload["molecule_count"])
    if (
        artifact_hash != expected_hash
        or runtime_hash != expected_hash
        or runtime_system.molecule_count != expected_count
    ):
        msg = "OpenMM, artifact, and MLX runtime molecule identities do not match"
        raise ValueError(msg)

    manifest = _source_manifest(
        api=api,
        prepared=prepared,
        artifact_dir=artifact_dir,
        source_pdb=source_pdb,
        benchmark_source=benchmark_source,
        force_class_counts=force_class_counts,
        context_payload=context_payload,
    )
    manifest_path = artifact_dir / MANIFEST_NAME
    manifest_path.write_text(_canonical_json(manifest) + "\n")
    return {
        "status": "ok",
        "case_id": CASE_ID,
        "artifact_path": str(artifact_dir.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "artifact_files": [
            str((artifact_dir / name).relative_to(root))
            for name in (
                "prepared_system.json",
                "prepared_system.npz",
                "view.pdb",
                MANIFEST_NAME,
            )
        ],
        "atom_count": prepared.atom_count,
        "molecule_count": expected_count,
        "molecule_identity_sha256": expected_hash,
        "pme": dict(context_payload["pme"]),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
    }


def _prepared_from_openmm(
    *,
    api: _common.OpenMMApi,
    system: Any,
    topology: Any,
    positions: Any,
    source_pdb: Path,
    force_class_counts: dict[str, int],
    context_payload: dict[str, Any],
) -> PreparedSystem:
    unit = api.unit
    atoms = list(topology.atoms())
    atom_count = len(atoms)
    atom_payload = _common._atom_payload(atoms)
    positions_a = _common._quantity_array(positions, unit.angstrom)
    masses = np.asarray(
        [
            system.getParticleMass(index).value_in_unit(unit.dalton)
            for index in range(atom_count)
        ],
        dtype=np.float32,
    )
    charges, sigma, epsilon, exceptions, nonbonded_setup = (
        _common._nonbonded_arrays(api, system)
    )
    bonds, bond_k, bond_length = _common._bond_arrays(api, system)
    angles, angle_k, angle_theta = _common._angle_arrays(api, system)
    dihedrals, dihedral_k, dihedral_periodicity, dihedral_phase = (
        _common._dihedral_arrays(api, system)
    )
    constraints, constraint_distance = _common._constraint_arrays(api, system)
    molecule_ids = normalize_molecule_ids(
        context_payload["molecule_ids"],
        atom_count=atom_count,
        required=True,
    )
    cell_matrix = np.asarray(context_payload["cell_matrix_angstrom"], dtype=np.float32)
    diagonal = np.diag(cell_matrix)
    if not np.allclose(cell_matrix, np.diag(diagonal), rtol=0.0, atol=1.0e-6):
        msg = "selected 5dfr preparation requires an orthorhombic periodic cell"
        raise ValueError(msg)

    pme = dict(context_payload["pme"])
    net_charge = float(np.sum(charges, dtype=np.float64))
    background_policy = (
        "reject_non_neutral"
        if abs(net_charge) <= 1.0e-5
        else "uniform_neutralizing_plasma"
    )
    terms = _required_terms(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        exceptions=exceptions[0],
        constraints=constraints,
    )
    hydrogen_count = int(np.count_nonzero(atom_payload["symbols"] == "H"))
    residue_names = np.char.upper(np.asarray(atom_payload["residue_names"], dtype=str))
    water_mask = residue_names == "HOH"
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source={
            "kind": "openmm_forcefield",
            "parser": "scripts/prepare_openmm_dhfr_explicit.py",
            "case_id": CASE_ID,
            "pdb_path": str(source_pdb),
            "forcefield_files": list(FORCE_FIELD_FILES),
            "openmm_version": api.openmm.version.version,
        },
        selections={
            "case_id": CASE_ID,
            "atom_count": atom_count,
            "hydrogen_count": hydrogen_count,
            "molecule_count": int(context_payload["molecule_count"]),
            "molecule_identity_sha256": context_payload[
                "molecule_identity_sha256"
            ],
            "system_charge": net_charge,
            "solvent_model": "explicit",
            "water_model": "tip3p",
            "electrostatics_model": "pme",
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
        parameter_source="openmm_amber99sb_tip3p",
        compatibility_report={
            "production_force_field": True,
            "physical_units": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "hydrogen_mass_repartitioning": "represented_by_masses",
            "periodic_box_present": True,
            "electrostatics_model": "pme",
            "solvent_model": "explicit",
            "water_model": "tip3p",
            "virtual_sites_present": False,
            "supported_terms": terms,
            "required_terms": terms,
            "unsupported_terms": [],
            "rejected_terms": [],
            "term_counts": {
                "harmonic_bond": int(bonds.shape[0]),
                "harmonic_angle": int(angles.shape[0]),
                "periodic_dihedral": int(dihedrals.shape[0]),
                "nonbonded_exception": int(exceptions[0].shape[0]),
                "distance_constraint": int(constraints.shape[0]),
                "pme": 1,
            },
            "force_field_provenance": {
                "source": "OpenMM ForceField",
                "files": list(FORCE_FIELD_FILES),
                "constraints": "HBonds",
                "rigid_water": True,
                "hydrogen_mass_amu": 1.5,
            },
        },
        pme_config={
            "mesh_shape": list(pme["mesh_shape"]),
            "alpha": pme["alpha_per_angstrom"],
            "real_cutoff": 9.0,
            "assignment_order": 5,
            "charge_tolerance": 1.0e-5,
            "deconvolve_assignment": True,
            "background_policy": background_policy,
        },
        protocol_metadata={
            "case_id": CASE_ID,
            "construction": {
                "forcefield_files": list(FORCE_FIELD_FILES),
                "nonbonded_method": "PME",
                "cutoff_angstrom": 9.0,
                "constraints": "HBonds",
                "rigid_water": True,
                "hydrogen_mass_amu": 1.5,
            },
            "nonbonded": {
                **nonbonded_setup,
                "ewald_error_tolerance": pme["ewald_error_tolerance"],
                "dispersion_correction": pme["dispersion_correction"],
                "switching_function": pme["switching_function"],
            },
            "pme": pme,
            "center_of_mass_motion": context_payload["center_of_mass_motion"],
            "force_classes": force_class_counts,
            "molecules": {
                "count": int(context_payload["molecule_count"]),
                "identity_sha256": context_payload[
                    "molecule_identity_sha256"
                ],
                "table_sha256": context_payload["molecule_table_sha256"],
            },
            "hydrogen_mass_repartitioning": {
                "source": "OpenMM ForceField.createSystem(hydrogenMass=1.5 amu)",
                "status": "represented_by_masses",
                "provenance_available": False,
                "policy": {"virtual_sites_supported": False},
            },
        },
    )
    return PreparedSystem(
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
        receptor_mask=~water_mask,
        restraint_mask=np.zeros(atom_count, dtype=bool),
        reference_positions=positions_a.copy(),
        cell_lengths=diagonal,
        cell_matrix=cell_matrix,
        constraints=constraints,
        constraint_distance=constraint_distance,
        nonbonded_exception_pairs=exceptions[0],
        nonbonded_exception_charge_product=exceptions[1],
        nonbonded_exception_sigma=exceptions[2],
        nonbonded_exception_epsilon=exceptions[3],
        water_mask=water_mask,
        ion_mask=np.zeros(atom_count, dtype=bool),
        lipid_mask=np.zeros(atom_count, dtype=bool),
        pme_mesh_shape=np.asarray(pme["mesh_shape"], dtype=np.int32),
        pme_alpha=np.asarray([pme["alpha_per_angstrom"]], dtype=np.float32),
        pme_real_cutoff=np.asarray([9.0], dtype=np.float32),
        pme_assignment_order=np.asarray([5], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1.0e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
        pme_background_policy=np.asarray([background_policy], dtype=str),
        molecule_ids=molecule_ids,
    )


def _resolved_context_payload(
    *,
    api: _common.OpenMMApi,
    system: Any,
    positions: Any,
    nonbonded: Any,
) -> dict[str, Any]:
    unit = api.unit
    integrator = api.openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = api.openmm.Platform.getPlatformByName("Reference")
    context = api.openmm.Context(system, integrator, platform)
    try:
        context.setPositions(positions)
        molecules = tuple(tuple(int(atom) for atom in row) for row in context.getMolecules())
        molecule_ids = _molecule_ids_from_table(
            molecules,
            atom_count=system.getNumParticles(),
        )
        alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
        alpha_per_angstrom = _inverse_nanometer_to_inverse_angstrom(
            alpha,
            unit=unit,
        )
    finally:
        del context
        del integrator
    cell_matrix = _box_vectors_angstrom(
        system.getDefaultPeriodicBoxVectors(),
        unit=unit,
    )
    cmm = _common._single_force(
        api.openmm.CMMotionRemover,
        system,
        "CMMotionRemover",
    )
    molecule_hash = molecule_identity_sha256(
        molecule_ids,
        atom_count=system.getNumParticles(),
    )
    return {
        "cell_matrix_angstrom": cell_matrix,
        "molecule_ids": molecule_ids,
        "molecule_count": len(molecules),
        "molecule_identity_sha256": molecule_hash,
        "molecule_table_sha256": _json_sha256([list(row) for row in molecules]),
        "center_of_mass_motion": {
            "force": "CMMotionRemover",
            "frequency_steps": int(cmm.getFrequency()),
            "enabled": True,
        },
        "pme": {
            "mesh_shape": [int(nx), int(ny), int(nz)],
            "alpha_per_angstrom": alpha_per_angstrom,
            "assignment_order": 5,
            "real_cutoff_angstrom": float(
                nonbonded.getCutoffDistance().value_in_unit(unit.angstrom)
            ),
            "ewald_error_tolerance": float(nonbonded.getEwaldErrorTolerance()),
            "dispersion_correction": bool(nonbonded.getUseDispersionCorrection()),
            "switching_function": bool(nonbonded.getUseSwitchingFunction()),
            "method": _common._nonbonded_method_name(
                api,
                nonbonded.getNonbondedMethod(),
            ),
            "platform": "Reference",
        },
    }


def _validated_force_classes(system: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in range(system.getNumForces()):
        name = type(system.getForce(index)).__name__
        counts[name] = counts.get(name, 0) + 1
    if counts != EXPECTED_FORCE_CLASSES:
        msg = (
            "selected 5dfr force classes do not match the exact supported set: "
            f"expected={EXPECTED_FORCE_CLASSES}, actual={counts}"
        )
        raise ValueError(msg)
    return dict(sorted(counts.items()))


def _molecule_ids_from_table(
    molecules: tuple[tuple[int, ...], ...],
    *,
    atom_count: int,
) -> np.ndarray:
    ids = np.full(atom_count, -1, dtype=np.int32)
    for molecule_index, atoms in enumerate(molecules):
        if not atoms:
            msg = "OpenMM molecule table contains an empty molecule"
            raise ValueError(msg)
        for atom_index in atoms:
            if atom_index < 0 or atom_index >= atom_count:
                msg = "OpenMM molecule table contains an out-of-range atom"
                raise ValueError(msg)
            if ids[atom_index] != -1:
                msg = "OpenMM molecule table assigns an atom more than once"
                raise ValueError(msg)
            ids[atom_index] = molecule_index
    if np.any(ids < 0):
        msg = "OpenMM molecule table does not cover every atom"
        raise ValueError(msg)
    return normalize_molecule_ids(ids, atom_count=atom_count, required=True)


def _required_terms(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    exceptions: np.ndarray,
    constraints: np.ndarray,
) -> list[str]:
    terms = ["nonbonded_lj_coulomb", "pme"]
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


def _inverse_nanometer_to_inverse_angstrom(value: Any, *, unit: Any) -> float:
    if hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(unit.angstrom**-1))
    return float(value) / 10.0


def _box_vectors_angstrom(vectors: Any, *, unit: Any) -> np.ndarray:
    return np.asarray(
        [vector.value_in_unit(unit.angstrom) for vector in vectors],
        dtype=np.float32,
    )


def _source_manifest(
    *,
    api: _common.OpenMMApi,
    prepared: PreparedSystem,
    artifact_dir: Path,
    source_pdb: Path,
    benchmark_source: Path,
    force_class_counts: dict[str, int],
    context_payload: dict[str, Any],
) -> dict[str, Any]:
    forcefield_resources = _forcefield_resources(api)
    array_descriptors = {
        name: _array_descriptor(np.asarray(getattr(prepared, name)))
        for name in (*PARAMETER_ARRAY_NAMES, *INDEX_ARRAY_NAMES)
    }
    atom_order = {
        name: _array_descriptor(np.asarray(getattr(prepared, name)))
        for name in (
            "symbols",
            "atom_names",
            "atom_types",
            "residue_names",
            "residue_ids",
            "chain_ids",
        )
    }
    artifact_files = {
        name: {
            "byte_size": (artifact_dir / name).stat().st_size,
            "sha256": _file_sha256(artifact_dir / name),
        }
        for name in ("prepared_system.json", "prepared_system.npz", "view.pdb")
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "case_id": CASE_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "pdb": _file_record(
                source_pdb,
                role="coordinates_and_atom_order",
                display_path=OPENMM_DHFR_SOLVATED,
            ),
            "vendor_benchmark": _file_record(
                benchmark_source,
                role="construction_contract",
                display_path=OPENMM_BENCHMARK_SOURCE,
            ),
            "forcefield_resources": forcefield_resources,
            "openmm_version": api.openmm.version.version,
        },
        "construction": {
            "forcefield_files": list(FORCE_FIELD_FILES),
            "nonbonded_method": "PME",
            "cutoff_angstrom": 9.0,
            "constraints": "HBonds",
            "rigid_water": True,
            "hydrogen_mass_amu": 1.5,
            "dispersion_correction": True,
            "center_of_mass_motion": context_payload["center_of_mass_motion"],
        },
        "identity": {
            "atom_count": prepared.atom_count,
            "atom_order": atom_order,
            "atom_order_sha256": _json_sha256(atom_order),
            "positions": _array_descriptor(prepared.positions),
            "cell": _array_descriptor(prepared.cell_matrix),
            "molecule_count": context_payload["molecule_count"],
            "molecule_identity_sha256": context_payload[
                "molecule_identity_sha256"
            ],
            "molecule_table_sha256": context_payload["molecule_table_sha256"],
        },
        "units": {
            "values": dict(prepared.metadata.units),
            "sha256": _json_sha256(prepared.metadata.units),
        },
        "forces": {
            "classes": force_class_counts,
            "classes_sha256": _json_sha256(force_class_counts),
            "parameter_arrays": array_descriptors,
            "parameters_sha256": _json_sha256(array_descriptors),
            "constraints_sha256": _json_sha256(
                {
                    name: array_descriptors[name]
                    for name in ("constraints", "constraint_distance")
                }
            ),
        },
        "pme": context_payload["pme"],
        "artifact_files": artifact_files,
    }
    manifest["manifest_fingerprint"] = _json_sha256(manifest)
    return manifest


def _forcefield_resources(api: _common.OpenMMApi) -> list[dict[str, Any]]:
    data_dir = Path(api.app.__file__).resolve().parent / "data"
    records = []
    for name in FORCE_FIELD_FILES:
        path = data_dir / name
        if not path.is_file():
            msg = f"missing installed OpenMM force-field resource: {name}"
            raise FileNotFoundError(msg)
        records.append(
            {
                "resource": f"openmm.app/data/{name}",
                "byte_size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return records


def _require_exact_source(repo_root: Path, pdb_path: Path) -> Path:
    requested = (repo_root / pdb_path).resolve()
    expected = (repo_root / OPENMM_DHFR_SOLVATED).resolve()
    if requested != expected:
        msg = (
            f"{CASE_ID} requires exactly {OPENMM_DHFR_SOLVATED}; "
            "JAC and synthetic solvent inputs cannot be substituted"
        )
        raise ValueError(msg)
    if not expected.is_file():
        msg = f"missing exact OpenMM 5dfr source: {OPENMM_DHFR_SOLVATED}"
        raise FileNotFoundError(msg)
    return expected


def _results_output_path(repo_root: Path, out_dir: Path) -> Path:
    requested = out_dir if out_dir.is_absolute() else repo_root / out_dir
    resolved = requested.resolve()
    results_root = (repo_root / "results").resolve()
    try:
        relative = resolved.relative_to(results_root)
    except ValueError as exc:
        msg = "5dfr preparation output must stay below the repository results/ directory"
        raise ValueError(msg) from exc
    if not relative.parts:
        msg = "5dfr preparation output must name a directory below results/"
        raise ValueError(msg)
    return resolved


def _array_descriptor(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _file_record(
    path: Path,
    *,
    role: str,
    display_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "path": str(path if display_path is None else display_path),
        "role": role,
        "byte_size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pdb", type=Path, default=OPENMM_DHFR_SOLVATED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
