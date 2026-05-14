"""Bundled solvated ligand-receptor MLX NVT example.

This module builds a complete small all-atom system for the active notebook:
T4 lysozyme L99A pocket atoms plus benzene from PDB 4W52, explicit hydrogens,
TIP3P-style waters, sodium/chloride ions, a periodic box, fixed topology, and
MLX-supported force terms.  It does not run any external MD engine.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.prep.io import JSON_NAME, load_prepared_system, save_prepared_system
from mlx_atomistic.prep.schema import ARTIFACT_VERSION, PreparedSystem, PreparedSystemMetadata
from mlx_atomistic.prep.t4l_benzene import prepare_t4l_benzene

SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE = (
    "mlx_internal_t4l_benzene_solvated_short_range_v1"
)
SOLVATED_LIGAND_RECEPTOR_WORKFLOW = "mlx_ligand_receptor_solvated_nvt_v1"
ELECTROSTATICS_MODEL = "short_range_electrostatics_prototype"

WATER_OH_DISTANCE_A = 0.9572
WATER_HOH_ANGLE_RAD = np.deg2rad(104.52)
WATER_HH_DISTANCE_A = float(
    2.0 * WATER_OH_DISTANCE_A * np.sin(0.5 * WATER_HOH_ANGLE_RAD)
)
DEFAULT_WATER_COUNT = 48
DEFAULT_CONSTRAINT_MAX_ITERATIONS = 40


class SolvatedExampleError(ValueError):
    """Raised when the bundled solvated example cannot be built or run."""


def prepare_solvated_ligand_receptor_example(
    *,
    water_count: int = DEFAULT_WATER_COUNT,
) -> PreparedSystem:
    """Build the bundled complete solvated ligand-receptor example."""

    if water_count <= 0:
        msg = "water_count must be positive for the solvated example"
        raise ValueError(msg)

    base = prepare_t4l_benzene()
    shifted_positions, cell_lengths = _center_positions_in_periodic_box(base.positions)
    base = replace(
        base,
        positions=shifted_positions,
        reference_positions=shifted_positions.copy(),
        cell_lengths=cell_lengths,
    )
    water_records = _place_waters(
        solute_positions=shifted_positions,
        ligand_positions=shifted_positions[np.asarray(base.ligand_mask, dtype=bool)],
        cell_lengths=cell_lengths,
        water_count=water_count,
    )
    ion_records = _place_ions(
        occupied_positions=np.vstack([shifted_positions, water_records.positions]),
        cell_lengths=cell_lengths,
    )

    appended = _append_solvent_and_ions(base, water_records, ion_records)
    prepared = _with_solvated_metadata(appended, water_count=water_count)
    validate_complete_solvated_ligand_receptor_system(prepared)
    return prepared


def ensure_solvated_ligand_receptor_example(
    out_dir: str | Path,
    *,
    steps: int = 5000,
    dt: float = 0.001,
    sample_interval: int = 25,
    temperature: float = 300.0,
    friction: float = 1.0,
    force: bool = False,
    water_count: int = DEFAULT_WATER_COUNT,
    minimize_steps: int = 100,
    equilibration_steps: int = 250,
    restraint_k: float = 10.0,
    constraint_max_iterations: int = DEFAULT_CONSTRAINT_MAX_ITERATIONS,
    diagnostic_interval: int | None = None,
) -> dict[str, Any]:
    """Create the prepared artifact and run the MLX trajectory if needed."""

    from mlx_atomistic.prep.runner import TRAJECTORY_NAME, run_mlx

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_path / TRAJECTORY_NAME

    generated_artifact = False
    if force or _prepared_artifact_is_missing_or_stale(out_path, water_count=water_count):
        prepared = prepare_solvated_ligand_receptor_example(water_count=water_count)
        save_prepared_system(prepared, out_path)
        generated_artifact = True
    else:
        prepared = load_prepared_system(out_path)
        validate_complete_solvated_ligand_receptor_system(prepared)

    generated_trajectory = False
    if force or _trajectory_is_missing_or_stale(
        trajectory_path,
        steps=steps,
        dt=dt,
        sample_interval=sample_interval,
        minimize_steps=minimize_steps,
        equilibration_steps=equilibration_steps,
        restraint_k=restraint_k,
        constraint_max_iterations=constraint_max_iterations,
        diagnostic_interval=diagnostic_interval,
    ):
        if diagnostic_interval is None:
            diagnostic_interval = sample_interval
        run_mlx(
            out_path,
            out=trajectory_path,
            steps=steps,
            sample_interval=sample_interval,
            dt=dt,
            temperature=temperature,
            friction=friction,
            restraint_k=restraint_k,
            require_production=False,
            minimize_steps=minimize_steps,
            equilibration_steps=equilibration_steps,
            constraint_max_iterations=constraint_max_iterations,
            diagnostic_interval=diagnostic_interval,
            metadata_overrides={
                "source": "mlx_atomistic",
                "workflow": SOLVATED_LIGAND_RECEPTOR_WORKFLOW,
                "dataset_id": "t4l-benzene-solvated-short-range-mlx",
                "electrostatics_model": ELECTROSTATICS_MODEL,
                "pme": False,
                "npt_barostat": False,
                "water_count": water_count,
                "ion_count": int(np.count_nonzero(prepared.ion_mask)),
                "runtime_note": (
                    "MLX-generated solvated NVT trajectory. Electrostatics are "
                    "periodic short-range cutoff only; PME is not implemented yet."
                ),
            },
        )
        generated_trajectory = True

    return {
        "prepared_dir": out_path,
        "trajectory_path": trajectory_path,
        "prepared": prepared,
        "generated_artifact": generated_artifact,
        "generated_trajectory": generated_trajectory,
    }


def ensure_solvated_ligand_receptor_prepared(
    out_dir: str | Path,
    *,
    water_count: int = DEFAULT_WATER_COUNT,
    force: bool = False,
) -> dict[str, Any]:
    """Create or load only the prepared solvated artifact, without running MD."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_artifact = False
    if force or _prepared_artifact_is_missing_or_stale(out_path, water_count=water_count):
        prepared = prepare_solvated_ligand_receptor_example(water_count=water_count)
        save_prepared_system(prepared, out_path)
        generated_artifact = True
    else:
        prepared = load_prepared_system(out_path)
        validate_complete_solvated_ligand_receptor_system(prepared)
    return {
        "prepared_dir": out_path,
        "prepared": prepared,
        "generated_artifact": generated_artifact,
    }


def validate_complete_solvated_ligand_receptor_system(prepared: PreparedSystem) -> None:
    """Fail closed for the active real-MD notebook runtime contract."""

    prepared.validate()
    report = prepared.metadata.compatibility_report
    blockers = []
    if prepared.cell_lengths.shape != (3,):
        blockers.append("missing periodic box vectors")
    if int(np.count_nonzero(prepared.ligand_mask)) <= 0:
        blockers.append("missing ligand mask/atoms")
    if int(np.count_nonzero(prepared.receptor_mask)) <= 0:
        blockers.append("missing receptor mask/atoms")
    if prepared.water_mask.shape != (prepared.atom_count,) or not np.any(prepared.water_mask):
        blockers.append("missing explicit water mask/atoms")
    if prepared.ion_mask.shape != (prepared.atom_count,) or not np.any(prepared.ion_mask):
        blockers.append("missing explicit ion mask/atoms")
    if int(np.count_nonzero(np.char.upper(prepared.symbols.astype(str)) == "H")) <= 0:
        blockers.append("missing explicit hydrogens")
    if prepared.constraints.shape[0] <= 0:
        blockers.append("missing distance constraints")
    if prepared.nonbonded_exception_pairs.shape[0] <= 0:
        blockers.append("missing nonbonded exclusions/exceptions")
    if not bool(report.get("physical_units", False)):
        blockers.append("metadata must declare physical_units=true")
    if report.get("electrostatics_model") != ELECTROSTATICS_MODEL:
        blockers.append(f"electrostatics_model must be {ELECTROSTATICS_MODEL}")
    if blockers:
        msg = "incomplete solvated ligand-receptor MLX system: " + "; ".join(blockers)
        raise SolvatedExampleError(msg)


def _center_positions_in_periodic_box(
    positions: np.ndarray,
    *,
    margin_A: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    extents = maximum - minimum
    cell_lengths = np.maximum(extents + 2.0 * margin_A, 36.0).astype(np.float32)
    shifted = positions - minimum + 0.5 * (cell_lengths - extents)
    return shifted.astype(np.float32), cell_lengths


class _SolventRecords:
    def __init__(
        self,
        *,
        symbols: list[str],
        atom_names: list[str],
        residue_names: list[str],
        residue_ids: list[int],
        chain_ids: list[str],
        positions: np.ndarray,
        masses: list[float],
        charges: list[float],
        sigma: list[float],
        epsilon: list[float],
        bonds: list[tuple[int, int]],
        angles: list[tuple[int, int, int]],
        constraints: list[tuple[int, int]],
    ) -> None:
        self.symbols = symbols
        self.atom_names = atom_names
        self.residue_names = residue_names
        self.residue_ids = residue_ids
        self.chain_ids = chain_ids
        self.positions = positions
        self.masses = masses
        self.charges = charges
        self.sigma = sigma
        self.epsilon = epsilon
        self.bonds = bonds
        self.angles = angles
        self.constraints = constraints


def _place_waters(
    *,
    solute_positions: np.ndarray,
    ligand_positions: np.ndarray,
    cell_lengths: np.ndarray,
    water_count: int,
) -> _SolventRecords:
    rng = np.random.default_rng(20260501)
    ligand_center = ligand_positions.mean(axis=0)
    candidates = _water_oxygen_candidates(
        ligand_center=ligand_center,
        solute_positions=solute_positions,
        cell_lengths=cell_lengths,
    )
    if len(candidates) < water_count:
        msg = f"could place only {len(candidates)} non-overlapping waters, need {water_count}"
        raise SolvatedExampleError(msg)

    symbols: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    residue_ids: list[int] = []
    chain_ids: list[str] = []
    positions: list[np.ndarray] = []
    masses: list[float] = []
    charges: list[float] = []
    sigma: list[float] = []
    epsilon: list[float] = []
    bonds: list[tuple[int, int]] = []
    angles: list[tuple[int, int, int]] = []
    constraints: list[tuple[int, int]] = []

    for water_index, oxygen in enumerate(candidates[:water_count], start=1):
        h1, h2 = _water_hydrogens(oxygen, rng)
        start = len(symbols)
        symbols.extend(["O", "H", "H"])
        atom_names.extend(["O", "H1", "H2"])
        residue_names.extend(["WAT", "WAT", "WAT"])
        residue_ids.extend([1000 + water_index] * 3)
        chain_ids.extend(["W"] * 3)
        positions.extend([oxygen, h1, h2])
        masses.extend([15.999, 1.008, 1.008])
        charges.extend([-0.834, 0.417, 0.417])
        sigma.extend([3.1507, 1.0, 1.0])
        epsilon.extend([0.6364, 0.0, 0.0])
        bonds.extend([(start, start + 1), (start, start + 2)])
        angles.append((start + 1, start, start + 2))
        constraints.extend([(start, start + 1), (start, start + 2), (start + 1, start + 2)])

    return _SolventRecords(
        symbols=symbols,
        atom_names=atom_names,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        positions=np.asarray(positions, dtype=np.float32),
        masses=masses,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        bonds=bonds,
        angles=angles,
        constraints=constraints,
    )


def _water_oxygen_candidates(
    *,
    ligand_center: np.ndarray,
    solute_positions: np.ndarray,
    cell_lengths: np.ndarray,
) -> list[np.ndarray]:
    margin = 3.0
    spacing = 3.05
    axes = [
        np.arange(margin, float(length) - margin + 1e-6, spacing, dtype=np.float32)
        for length in cell_lengths
    ]
    candidates = []
    occupied = [np.asarray(position, dtype=np.float32) for position in solute_positions]
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                oxygen = np.asarray([x, y, z], dtype=np.float32)
                if _minimum_distance(oxygen, np.asarray(occupied, dtype=np.float32)) < 2.55:
                    continue
                occupied.append(oxygen)
                candidates.append(oxygen)
    candidates.sort(key=lambda position: float(np.linalg.norm(position - ligand_center)))
    return candidates


def _water_hydrogens(
    oxygen: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bisector = _unit_vector(rng.normal(size=3))
    perpendicular = _unit_vector(np.cross(bisector, rng.normal(size=3)))
    if np.linalg.norm(perpendicular) < 1e-6:
        perpendicular = _unit_vector(np.cross(bisector, np.asarray([1.0, 0.0, 0.0])))
    half_angle = 0.5 * WATER_HOH_ANGLE_RAD
    center_component = np.cos(half_angle) * bisector
    side_component = np.sin(half_angle) * perpendicular
    h1 = oxygen + WATER_OH_DISTANCE_A * (center_component + side_component)
    h2 = oxygen + WATER_OH_DISTANCE_A * (center_component - side_component)
    return h1.astype(np.float32), h2.astype(np.float32)


def _place_ions(
    *,
    occupied_positions: np.ndarray,
    cell_lengths: np.ndarray,
) -> _SolventRecords:
    candidates = [
        np.asarray([4.0, 4.0, 4.0], dtype=np.float32),
        cell_lengths - np.asarray([4.5, 4.5, 4.5], dtype=np.float32),
        np.asarray([4.0, cell_lengths[1] - 4.0, 4.0], dtype=np.float32),
        np.asarray([cell_lengths[0] - 4.0, 4.0, cell_lengths[2] - 4.0], dtype=np.float32),
    ]
    positions: list[np.ndarray] = []
    for candidate in candidates:
        if _minimum_distance(candidate, occupied_positions) >= 3.0:
            positions.append(candidate)
        if len(positions) == 2:
            break
    if len(positions) < 2:
        msg = "could not place sodium/chloride ions without close contacts"
        raise SolvatedExampleError(msg)
    return _SolventRecords(
        symbols=["Na", "Cl"],
        atom_names=["NA", "CL"],
        residue_names=["NA", "CL"],
        residue_ids=[2001, 2002],
        chain_ids=["I", "I"],
        positions=np.asarray(positions, dtype=np.float32),
        masses=[22.99, 35.45],
        charges=[1.0, -1.0],
        sigma=[2.6, 4.4],
        epsilon=[0.13, 0.42],
        bonds=[],
        angles=[],
        constraints=[],
    )


def _append_solvent_and_ions(
    base: PreparedSystem,
    water: _SolventRecords,
    ions: _SolventRecords,
) -> PreparedSystem:
    solvent = _concat_records(water, ions)
    atom_offset = base.atom_count
    positions = np.vstack([base.positions, solvent.positions]).astype(np.float32)
    velocities = np.zeros_like(positions, dtype=np.float32)
    bonds = _append_index_rows(base.bonds, solvent.bonds, atom_offset, width=2)
    angles = _append_index_rows(base.angles, solvent.angles, atom_offset, width=3)
    dihedrals = np.asarray(base.dihedrals, dtype=np.int32)
    impropers = np.asarray(base.impropers, dtype=np.int32)
    constraints = _append_index_rows(base.constraints, solvent.constraints, atom_offset, width=2)

    water_mask = np.concatenate(
        [
            np.zeros((base.atom_count,), dtype=bool),
            np.asarray(solvent.residue_names) == "WAT",
        ]
    )
    ion_mask = np.concatenate(
        [
            np.zeros((base.atom_count,), dtype=bool),
            np.isin(np.asarray(solvent.residue_names), ["NA", "CL"]),
        ]
    )
    ligand_mask = np.concatenate([base.ligand_mask, np.zeros((len(solvent.symbols),), dtype=bool)])
    receptor_mask = np.concatenate(
        [base.receptor_mask, np.zeros((len(solvent.symbols),), dtype=bool)]
    )
    restraint_mask = np.concatenate(
        [base.restraint_mask, np.zeros((len(solvent.symbols),), dtype=bool)]
    )
    masses = np.concatenate([base.masses, np.asarray(solvent.masses, dtype=np.float32)])
    charges = np.concatenate([base.charges, np.asarray(solvent.charges, dtype=np.float32)])
    sigma = np.concatenate([base.sigma, np.asarray(solvent.sigma, dtype=np.float32)])
    epsilon = np.concatenate([base.epsilon, np.asarray(solvent.epsilon, dtype=np.float32)])

    (
        exception_pairs,
        exception_qprod,
        exception_sigma,
        exception_epsilon,
    ) = _nonbonded_exceptions_for_topology(
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
    )
    bond_count_delta = bonds.shape[0] - base.bonds.shape[0]
    angle_count_delta = angles.shape[0] - base.angles.shape[0]

    return replace(
        base,
        symbols=np.concatenate([base.symbols, np.asarray(solvent.symbols, dtype=str)]),
        atom_names=np.concatenate([base.atom_names, np.asarray(solvent.atom_names, dtype=str)]),
        atom_types=np.concatenate([base.atom_types, np.asarray(solvent.symbols, dtype=str)]),
        residue_names=np.concatenate(
            [base.residue_names, np.asarray(solvent.residue_names, dtype=str)]
        ),
        residue_ids=np.concatenate(
            [base.residue_ids, np.asarray(solvent.residue_ids, dtype=np.int32)]
        ),
        chain_ids=np.concatenate([base.chain_ids, np.asarray(solvent.chain_ids, dtype=str)]),
        positions=positions,
        velocities=velocities,
        masses=masses.astype(np.float32),
        charges=charges.astype(np.float32),
        sigma=sigma.astype(np.float32),
        epsilon=epsilon.astype(np.float32),
        bonds=bonds,
        bond_k=np.concatenate(
            [base.bond_k, np.full((bond_count_delta,), 450.0, dtype=np.float32)]
        ),
        bond_length=_distances(positions, bonds),
        angles=angles,
        angle_k=np.concatenate(
            [base.angle_k, np.full((angle_count_delta,), 55.0, dtype=np.float32)]
        ),
        angle_theta=_angle_values(positions, angles),
        dihedrals=dihedrals,
        impropers=impropers,
        nonbonded_exception_pairs=exception_pairs,
        nonbonded_exception_charge_product=exception_qprod,
        nonbonded_exception_sigma=exception_sigma,
        nonbonded_exception_epsilon=exception_epsilon,
        constraints=constraints,
        constraint_distance=_distances(positions, constraints),
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        restraint_mask=restraint_mask,
        water_mask=water_mask,
        ion_mask=ion_mask,
        reference_positions=positions.copy(),
    )


def _with_solvated_metadata(prepared: PreparedSystem, *, water_count: int) -> PreparedSystem:
    symbols = np.char.upper(prepared.symbols.astype(str))
    hydrogen_count = int(np.count_nonzero(symbols == "H"))
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
            "kind": "bundled_solvated_ligand_receptor",
            "pdb_id": "4W52",
            "pdb_title": "T4 lysozyme L99A with benzene bound",
            "source_url": "https://www.rcsb.org/structure/4W52",
            "workflow": SOLVATED_LIGAND_RECEPTOR_WORKFLOW,
        },
        selections={
            "ligand_resname": "BNZ",
            "ligand_resid": 200,
            "atom_count": prepared.atom_count,
            "ligand_atom_count": int(np.count_nonzero(prepared.ligand_mask)),
            "receptor_atom_count": int(np.count_nonzero(prepared.receptor_mask)),
            "water_atom_count": int(np.count_nonzero(prepared.water_mask)),
            "water_count": water_count,
            "ion_atom_count": int(np.count_nonzero(prepared.ion_mask)),
            "hydrogen_count": hydrogen_count,
            "box_lengths_A": prepared.cell_lengths.astype(float).tolist(),
            "system_charge": float(np.sum(prepared.charges)),
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source=SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE,
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": False,
            "physical_units": True,
            "hydrogens_present": True,
            "hydrogen_count": hydrogen_count,
            "water_present": True,
            "water_count": water_count,
            "ions_present": True,
            "ion_count": int(np.count_nonzero(prepared.ion_mask)),
            "periodic_box_present": True,
            "electrostatics_model": ELECTROSTATICS_MODEL,
            "pme": False,
            "npt_barostat": False,
            "supported_terms": supported_terms,
            "required_terms": supported_terms,
            "unsupported_terms": [],
            "rejected_terms": [],
        },
        warnings=[
            (
                "MLX runs this trajectory directly. No OpenMM, LAMMPS, GROMACS, "
                "or other external MD engine is used."
            ),
            (
                "This is an unbiased short solvated NVT benchmark. It is not a "
                "ligand egress, docking, binding, or free-energy calculation."
            ),
            (
                "PME/Ewald electrostatics are not implemented yet, so this artifact "
                f"is labeled {ELECTROSTATICS_MODEL}."
            ),
            (
                "Internal parameters are sufficient for MLX engine/notebook workflow "
                "development, but are not a published CHARMM/AMBER force field."
            ),
        ],
    )
    return replace(prepared, metadata=metadata)


def _prepared_artifact_is_missing_or_stale(path: Path, *, water_count: int) -> bool:
    if not (path / JSON_NAME).exists():
        return True
    try:
        prepared = load_prepared_system(path)
        validate_complete_solvated_ligand_receptor_system(prepared)
    except Exception:
        return True
    return not (
        prepared.metadata.parameter_source == SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE
        and int(prepared.metadata.selections.get("water_count", -1)) == int(water_count)
    )


def _trajectory_is_missing_or_stale(
    path: Path,
    *,
    steps: int,
    dt: float,
    sample_interval: int,
    minimize_steps: int,
    equilibration_steps: int,
    restraint_k: float,
    constraint_max_iterations: int,
    diagnostic_interval: int | None,
) -> bool:
    if not path.exists():
        return True
    try:
        from mlx_atomistic.io import load_npz_trajectory

        record = load_npz_trajectory(path)
    except Exception:
        return True
    metadata = record.metadata
    return not (
        metadata.get("source") == "mlx_atomistic"
        and metadata.get("workflow") == SOLVATED_LIGAND_RECEPTOR_WORKFLOW
        and int(metadata.get("steps", -1)) == int(steps)
        and int(metadata.get("sample_interval", -1)) == int(sample_interval)
        and abs(float(metadata.get("dt", -1.0)) - float(dt)) < 1e-12
        and int(metadata.get("minimize_steps", -1)) == int(minimize_steps)
        and int(metadata.get("equilibration_steps", -1)) == int(equilibration_steps)
        and abs(float(metadata.get("restraint_k", -1.0)) - float(restraint_k)) < 1e-12
        and int(metadata.get("constraint_max_iterations", -1)) == int(constraint_max_iterations)
        and int(metadata.get("diagnostic_interval", -1))
        == int(sample_interval if diagnostic_interval is None else diagnostic_interval)
    )


def _concat_records(left: _SolventRecords, right: _SolventRecords) -> _SolventRecords:
    offset = len(left.symbols)
    return _SolventRecords(
        symbols=[*left.symbols, *right.symbols],
        atom_names=[*left.atom_names, *right.atom_names],
        residue_names=[*left.residue_names, *right.residue_names],
        residue_ids=[*left.residue_ids, *right.residue_ids],
        chain_ids=[*left.chain_ids, *right.chain_ids],
        positions=np.vstack([left.positions, right.positions]).astype(np.float32),
        masses=[*left.masses, *right.masses],
        charges=[*left.charges, *right.charges],
        sigma=[*left.sigma, *right.sigma],
        epsilon=[*left.epsilon, *right.epsilon],
        bonds=[*left.bonds, *[(i + offset, j + offset) for i, j in right.bonds]],
        angles=[
            *left.angles,
            *[(i + offset, j + offset, k + offset) for i, j, k in right.angles],
        ],
        constraints=[
            *left.constraints,
            *[(i + offset, j + offset) for i, j in right.constraints],
        ],
    )


def _append_index_rows(
    base: np.ndarray,
    rows: list[tuple[int, ...]],
    offset: int,
    *,
    width: int,
) -> np.ndarray:
    if rows:
        appended = np.asarray(
            [tuple(int(value) + offset for value in row) for row in rows],
            dtype=np.int32,
        ).reshape((-1, width))
        return np.vstack([np.asarray(base, dtype=np.int32), appended]).astype(np.int32)
    return np.asarray(base, dtype=np.int32).reshape((-1, width))


def _nonbonded_exceptions_for_topology(
    *,
    bonds: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    charges: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero_pairs = _pairs_from_bonds(bonds) | _pairs_from_angles(angles)
    one_four_pairs = _pairs_from_dihedrals(dihedrals) - zero_pairs
    exception_values: dict[tuple[int, int], tuple[float, float, float]] = {
        pair: (0.0, 0.0, 0.0) for pair in zero_pairs
    }
    for i, j in sorted(one_four_pairs):
        sigma_ij = 0.5 * (float(sigma[i]) + float(sigma[j]))
        epsilon_ij = float(np.sqrt(float(epsilon[i]) * float(epsilon[j]))) * 0.5
        qprod = float(charges[i] * charges[j]) * (1.0 / 1.2)
        exception_values[(i, j)] = (qprod, sigma_ij, epsilon_ij)
    if not exception_values:
        empty = np.asarray([], dtype=np.float32)
        return np.empty((0, 2), dtype=np.int32), empty, empty, empty
    pairs = np.asarray(sorted(exception_values), dtype=np.int32).reshape((-1, 2))
    values = [exception_values[tuple(pair)] for pair in pairs.tolist()]
    return (
        pairs,
        np.asarray([value[0] for value in values], dtype=np.float32),
        np.asarray([value[1] for value in values], dtype=np.float32),
        np.asarray([value[2] for value in values], dtype=np.float32),
    )


def _pairs_from_bonds(bonds: np.ndarray) -> set[tuple[int, int]]:
    return {_pair(int(i), int(j)) for i, j in np.asarray(bonds, dtype=np.int32).tolist()}


def _pairs_from_angles(angles: np.ndarray) -> set[tuple[int, int]]:
    return {_pair(int(i), int(k)) for i, _, k in np.asarray(angles, dtype=np.int32).tolist()}


def _pairs_from_dihedrals(dihedrals: np.ndarray) -> set[tuple[int, int]]:
    return {
        _pair(int(i), int(j))
        for i, _, _, j in np.asarray(dihedrals, dtype=np.int32).tolist()
    }


def _pair(i: int, j: int) -> tuple[int, int]:
    return (min(i, j), max(i, j))


def _minimum_distance(position: np.ndarray, positions: np.ndarray) -> float:
    if positions.size == 0:
        return float("inf")
    delta = np.asarray(positions, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    return float(np.sqrt(np.sum(delta * delta, axis=1)).min())


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return vector / norm


def _distances(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _angle_values(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    if angles.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    values = []
    for i, j, k in np.asarray(angles, dtype=np.int32):
        left = positions[i] - positions[j]
        right = positions[k] - positions[j]
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        values.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


__all__ = [
    "ELECTROSTATICS_MODEL",
    "DEFAULT_CONSTRAINT_MAX_ITERATIONS",
    "SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE",
    "SOLVATED_LIGAND_RECEPTOR_WORKFLOW",
    "SolvatedExampleError",
    "ensure_solvated_ligand_receptor_example",
    "prepare_solvated_ligand_receptor_example",
    "validate_complete_solvated_ligand_receptor_system",
]
