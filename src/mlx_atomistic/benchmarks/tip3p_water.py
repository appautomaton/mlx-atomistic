"""Create deterministic production-compatible TIP3P water benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.schema import ARTIFACT_VERSION, PreparedSystem, PreparedSystemMetadata

TIP3P_OH_DISTANCE_ANGSTROM = 0.9572
TIP3P_HOH_ANGLE_DEGREES = 104.52
TIP3P_MOLECULAR_MASS_DALTON = 18.01528
TIP3P_DENSITY_G_CM3 = 0.997
TIP3P_ALPHA_PER_ANGSTROM = 0.29202898720871845


@dataclass(frozen=True)
class WaterBoxPreset:
    """Shape and PME mesh for one deterministic water benchmark."""

    name: str
    grid_shape: tuple[int, int, int]
    mesh_shape: tuple[int, int, int]

    @property
    def water_count(self) -> int:
        """Return the number of water molecules."""

        return int(np.prod(self.grid_shape, dtype=np.int64))

    @property
    def atom_count(self) -> int:
        """Return the number of atoms."""

        return 3 * self.water_count


WATER_BOX_PRESETS = {
    "30k": WaterBoxPreset("30k", (20, 20, 25), (64, 64, 80)),
    "90k": WaterBoxPreset("90k", (30, 25, 40), (96, 80, 128)),
}


def build_tip3p_water_box(
    *,
    grid_shape: tuple[int, int, int],
    mesh_shape: tuple[int, int, int],
    seed: int = 20260814,
) -> PreparedSystem:
    """Build a deterministic orthorhombic TIP3P water box."""

    if len(grid_shape) != 3 or any(int(value) <= 0 for value in grid_shape):
        raise ValueError("grid_shape must contain three positive integers")
    if len(mesh_shape) != 3 or any(int(value) <= 0 for value in mesh_shape):
        raise ValueError("mesh_shape must contain three positive integers")
    water_count = int(np.prod(grid_shape, dtype=np.int64))
    atom_count = 3 * water_count
    spacing = _water_spacing_angstrom()
    cell_lengths = spacing * np.asarray(grid_shape, dtype=np.float32)
    oxygen_positions = _oxygen_grid(grid_shape, spacing=spacing)
    hydrogen_one, hydrogen_two = _oriented_hydrogens(oxygen_positions, seed=seed)
    positions = np.stack((oxygen_positions, hydrogen_one, hydrogen_two), axis=1).reshape(
        atom_count, 3
    )

    atoms = np.arange(atom_count, dtype=np.int32).reshape((water_count, 3))
    constraint_left = atoms[:, (0, 0, 1)].reshape(-1)
    constraint_right = atoms[:, (1, 2, 2)].reshape(-1)
    constraints = np.stack((constraint_left, constraint_right), axis=1)
    hh_distance = float(
        2.0
        * TIP3P_OH_DISTANCE_ANGSTROM
        * np.sin(np.deg2rad(TIP3P_HOH_ANGLE_DEGREES) / 2.0)
    )
    constraint_distance = np.tile(
        np.asarray(
            [TIP3P_OH_DISTANCE_ANGSTROM, TIP3P_OH_DISTANCE_ANGSTROM, hh_distance],
            dtype=np.float32,
        ),
        water_count,
    )
    empty_pairs = np.empty((0, 2), dtype=np.int32)
    empty_triples = np.empty((0, 3), dtype=np.int32)
    empty_quads = np.empty((0, 4), dtype=np.int32)
    exceptions = constraints.copy()
    terms = [
        "nonbonded_lj_coulomb",
        "nonbonded_exception",
        "distance_constraint",
        "pme",
    ]
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source={
            "kind": "deterministic_tip3p_water_box",
            "generator": "mlx_atomistic.benchmarks.tip3p_water",
            "seed": seed,
            "grid_shape": list(grid_shape),
            "density_g_cm3": TIP3P_DENSITY_G_CM3,
        },
        selections={
            "atom_count": atom_count,
            "hydrogen_count": 2 * water_count,
            "molecule_count": water_count,
            "water_atom_count": atom_count,
            "system_charge": 0.0,
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
            "force": "kilojoule_per_mole_per_angstrom",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="tip3p_reference_parameters",
        compatibility_report={
            "production_force_field": True,
            "physical_units": True,
            "hydrogens_present": True,
            "hydrogen_count": 2 * water_count,
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
                "nonbonded_exception": int(exceptions.shape[0]),
                "distance_constraint": int(constraints.shape[0]),
                "pme": 1,
            },
            "force_field_provenance": {
                "source": "TIP3P reference parameters",
                "constraints": "rigid water",
                "rigid_water": True,
            },
        },
        pme_config={
            "mesh_shape": list(mesh_shape),
            "alpha": TIP3P_ALPHA_PER_ANGSTROM,
            "real_cutoff": 9.0,
            "assignment_order": 5,
            "charge_tolerance": 1.0e-5,
            "deconvolve_assignment": True,
            "background_policy": "reject_non_neutral",
        },
        protocol_metadata={
            "case_id": f"tip3p-water-{atom_count}",
            "ensemble": "NVT",
            "construction": {
                "water_model": "TIP3P",
                "nonbonded_method": "PME",
                "cutoff_angstrom": 9.0,
                "constraints": "rigid water",
            },
            "nonbonded": {
                "force": "NonbondedForce",
                "method": "PME",
                "cutoff": 9.0,
                "cutoff_unit": "angstrom",
                "ewald_error_tolerance": 5.0e-4,
                "dispersion_correction": True,
                "switching_function": False,
                "exception_count": int(exceptions.shape[0]),
            },
            "pme": {
                "mesh_shape": list(mesh_shape),
                "alpha_per_angstrom": TIP3P_ALPHA_PER_ANGSTROM,
                "assignment_order": 5,
                "real_cutoff_angstrom": 9.0,
                "method": "PME",
                "ewald_error_tolerance": 5.0e-4,
                "dispersion_correction": True,
                "switching_function": False,
            },
            "center_of_mass_motion": {
                "enabled": True,
                "force": "CMMotionRemover",
                "frequency_steps": 1,
            },
        },
    )
    symbols = np.tile(np.asarray(["O", "H", "H"], dtype=str), water_count)
    prepared = PreparedSystem(
        metadata=metadata,
        symbols=symbols,
        atom_names=np.tile(np.asarray(["O", "H1", "H2"], dtype=str), water_count),
        atom_types=np.tile(np.asarray(["OW", "HW", "HW"], dtype=str), water_count),
        residue_names=np.full(atom_count, "HOH", dtype=str),
        residue_ids=np.repeat(np.arange(1, water_count + 1, dtype=np.int32), 3),
        chain_ids=np.full(atom_count, "W", dtype=str),
        positions=positions.astype(np.float32),
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=np.tile(np.asarray([15.9994, 1.00794, 1.00794], dtype=np.float32), water_count),
        charges=np.tile(np.asarray([-0.834, 0.417, 0.417], dtype=np.float32), water_count),
        sigma=np.tile(np.asarray([3.1507, 1.0, 1.0], dtype=np.float32), water_count),
        epsilon=np.tile(np.asarray([0.636386, 0.0, 0.0], dtype=np.float32), water_count),
        bonds=empty_pairs,
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
        angles=empty_triples,
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=empty_quads,
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=empty_pairs,
        ligand_mask=np.zeros(atom_count, dtype=bool),
        receptor_mask=np.zeros(atom_count, dtype=bool),
        restraint_mask=np.zeros(atom_count, dtype=bool),
        reference_positions=positions.astype(np.float32),
        cell_lengths=cell_lengths,
        cell_matrix=np.diag(cell_lengths).astype(np.float32),
        constraints=constraints,
        constraint_distance=constraint_distance,
        nonbonded_exception_pairs=exceptions,
        nonbonded_exception_charge_product=np.zeros(exceptions.shape[0], dtype=np.float32),
        nonbonded_exception_sigma=np.ones(exceptions.shape[0], dtype=np.float32),
        nonbonded_exception_epsilon=np.zeros(exceptions.shape[0], dtype=np.float32),
        water_mask=np.ones(atom_count, dtype=bool),
        ion_mask=np.zeros(atom_count, dtype=bool),
        lipid_mask=np.zeros(atom_count, dtype=bool),
        pme_mesh_shape=np.asarray(mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([TIP3P_ALPHA_PER_ANGSTROM], dtype=np.float32),
        pme_real_cutoff=np.asarray([9.0], dtype=np.float32),
        pme_assignment_order=np.asarray([5], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1.0e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
        pme_background_policy=np.asarray(["reject_non_neutral"], dtype=str),
        molecule_ids=np.repeat(np.arange(water_count, dtype=np.int32), 3),
    )
    prepared.validate()
    return prepared


def prepare_preset(*, preset: str, out: str | Path, seed: int = 20260814) -> dict:
    """Build and save one named water benchmark preset."""

    try:
        spec = WATER_BOX_PRESETS[preset]
    except KeyError as exc:
        choices = ", ".join(sorted(WATER_BOX_PRESETS))
        raise ValueError(f"unknown TIP3P water preset {preset!r}; choose from {choices}") from exc
    prepared = build_tip3p_water_box(
        grid_shape=spec.grid_shape,
        mesh_shape=spec.mesh_shape,
        seed=seed,
    )
    out_path = Path(out)
    save_prepared_system(prepared, out_path)
    return {
        "schema": "mlx_atomistic.tip3p_water_preparation.v1",
        "status": "ok",
        "preset": spec.name,
        "out": str(out_path),
        "water_count": spec.water_count,
        "atom_count": spec.atom_count,
        "grid_shape": list(spec.grid_shape),
        "mesh_shape": list(spec.mesh_shape),
        "cell_lengths_angstrom": prepared.cell_lengths.tolist(),
        "density_g_cm3": TIP3P_DENSITY_G_CM3,
        "seed": seed,
    }


def _water_spacing_angstrom() -> float:
    volume_per_molecule = (
        TIP3P_MOLECULAR_MASS_DALTON * 1.66053906660 / TIP3P_DENSITY_G_CM3
    )
    return float(np.cbrt(volume_per_molecule))


def _oxygen_grid(grid_shape: tuple[int, int, int], *, spacing: float) -> np.ndarray:
    axes = [(np.arange(size, dtype=np.float32) + 0.5) * spacing for size in grid_shape]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape((-1, 3)).astype(np.float32)


def _oriented_hydrogens(
    oxygen_positions: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = oxygen_positions.shape[0]
    bisectors = _normalize_rows(rng.normal(size=(count, 3)))
    auxiliaries = _normalize_rows(rng.normal(size=(count, 3)))
    perpendicular = np.cross(bisectors, auxiliaries)
    degenerate = np.linalg.norm(perpendicular, axis=1) < 1.0e-8
    if np.any(degenerate):
        fallback = np.tile(np.asarray([1.0, 0.0, 0.0]), (int(np.count_nonzero(degenerate)), 1))
        parallel = np.abs(bisectors[degenerate, 0]) > 0.9
        fallback[parallel] = np.asarray([0.0, 1.0, 0.0])
        perpendicular[degenerate] = np.cross(bisectors[degenerate], fallback)
    perpendicular = _normalize_rows(perpendicular)
    half_angle = np.deg2rad(TIP3P_HOH_ANGLE_DEGREES) / 2.0
    center = np.cos(half_angle) * bisectors
    side = np.sin(half_angle) * perpendicular
    h1 = oxygen_positions + TIP3P_OH_DISTANCE_ANGSTROM * (center + side)
    h2 = oxygen_positions + TIP3P_OH_DISTANCE_ANGSTROM * (center - side)
    return h1.astype(np.float32), h2.astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-12)


def main(argv: list[str] | None = None) -> int:
    """Prepare one deterministic TIP3P water benchmark artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(WATER_BOX_PRESETS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)
    payload = prepare_preset(preset=args.preset, out=args.out, seed=args.seed)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TIP3P_DENSITY_G_CM3",
    "WATER_BOX_PRESETS",
    "WaterBoxPreset",
    "build_tip3p_water_box",
    "main",
    "prepare_preset",
]
