"""Deterministic TIP3P/NaCl fixtures for PME validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import ceil, cos, pi, sin, sqrt
from pathlib import Path

import numpy as np

from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)

AVOGADRO = 6.02214076e23
WATER_MOLARITY = 55.4
PME_ERROR_TOLERANCE = 5.0e-4
PME_REAL_CUTOFF_ANGSTROM = 9.0
PME_ASSIGNMENT_ORDER = 5

OH_DISTANCE_ANGSTROM = 0.9572
HOH_ANGLE_RAD = 1.82421813418
HH_DISTANCE_ANGSTROM = 2.0 * OH_DISTANCE_ANGSTROM * sin(0.5 * HOH_ANGLE_RAD)

TIP3P_O_CHARGE = -0.834
TIP3P_H_CHARGE = 0.417
TIP3P_O_MASS = 15.99943
TIP3P_H_MASS = 1.007947
TIP3P_O_SIGMA_ANGSTROM = 3.150752406575124
TIP3P_O_EPSILON_KJ_MOL = 0.635968

NA_MASS = 22.99
NA_SIGMA_ANGSTROM = 2.439280690268249
NA_EPSILON_KJ_MOL = 0.3658460312
CL_MASS = 35.45
CL_SIGMA_ANGSTROM = 4.477656957373345
CL_EPSILON_KJ_MOL = 0.148912744

WATER_BOND_K_KJ_MOL_ANGSTROM2 = 4627.504
WATER_ANGLE_K_KJ_MOL_RAD2 = 836.8


@dataclass(frozen=True)
class PMEFixtureSpec:
    """Configuration for one deterministic PME fixture."""

    name: str
    bcc_cells_per_axis: int
    ion_pairs: int
    seed: int = 20260713

    @property
    def site_count(self) -> int:
        """Return the number of solvent sites in the cubic BCC lattice."""

        return 2 * self.bcc_cells_per_axis**3

    @property
    def water_count(self) -> int:
        """Return the water count after ion replacement."""

        return self.site_count - 2 * self.ion_pairs

    @property
    def atom_count(self) -> int:
        """Return the final atom count."""

        return 3 * self.water_count + 2 * self.ion_pairs

    @property
    def ionic_strength_molar(self) -> float:
        """Return the NaCl pair concentration implied by the site count."""

        return self.ion_pairs * WATER_MOLARITY / self.site_count


FIXTURE_SPECS = {
    "water-small": PMEFixtureSpec("water-small", bcc_cells_per_axis=5, ion_pairs=0),
    "salt-small": PMEFixtureSpec("salt-small", bcc_cells_per_axis=6, ion_pairs=1),
    "target": PMEFixtureSpec("target", bcc_cells_per_axis=16, ion_pairs=22),
}


def fixture_spec(name: str) -> PMEFixtureSpec:
    """Return a named fixture specification."""

    try:
        return FIXTURE_SPECS[name]
    except KeyError as err:
        msg = f"unknown PME fixture {name!r}; expected one of {sorted(FIXTURE_SPECS)}"
        raise ValueError(msg) from err


def build_pme_fixture(
    spec: PMEFixtureSpec | str = "target",
    *,
    minimum_clearance_angstrom: float = 1.0,
) -> PreparedSystem:
    """Build a neutral periodic TIP3P/NaCl prepared system."""

    if isinstance(spec, str):
        spec = fixture_spec(spec)
    if spec.bcc_cells_per_axis <= 0:
        msg = "bcc_cells_per_axis must be positive"
        raise ValueError(msg)
    if spec.ion_pairs < 0 or 2 * spec.ion_pairs > spec.site_count:
        msg = "ion_pairs must fit within the solvent-site count"
        raise ValueError(msg)
    if minimum_clearance_angstrom <= 0.0:
        msg = "minimum_clearance_angstrom must be positive"
        raise ValueError(msg)

    box_length = _box_length_angstrom(spec.site_count)
    site_positions = _bcc_site_positions(spec.bcc_cells_per_axis, box_length)
    clearance_lower_bound = _clearance_lower_bound(spec.bcc_cells_per_axis, box_length)
    if clearance_lower_bound < minimum_clearance_angstrom:
        msg = (
            "fixture geometry cannot guarantee the requested intermolecular clearance: "
            f"lower_bound={clearance_lower_bound:.6g} A, "
            f"required={minimum_clearance_angstrom:.6g} A"
        )
        raise ValueError(msg)

    ion_sites = _select_separated_sites(site_positions, box_length, 2 * spec.ion_pairs, spec.seed)
    sodium_sites = ion_sites[::2]
    chloride_sites = ion_sites[1::2]
    ion_mask_by_site = np.zeros((spec.site_count,), dtype=bool)
    ion_mask_by_site[ion_sites] = True
    water_sites = np.flatnonzero(~ion_mask_by_site)

    water_positions = _water_positions(
        site_positions[water_sites],
        box_length,
        seed=spec.seed,
    )
    positions = np.concatenate(
        [
            water_positions.reshape(-1, 3),
            site_positions[sodium_sites],
            site_positions[chloride_sites],
        ],
        axis=0,
    ).astype(np.float32)
    water_count = water_sites.size
    sodium_count = sodium_sites.size
    chloride_count = chloride_sites.size
    atom_count = positions.shape[0]

    water_starts = 3 * np.arange(water_count, dtype=np.int32)
    bonds = np.stack(
        [
            np.concatenate([water_starts, water_starts]),
            np.concatenate([water_starts + 1, water_starts + 2]),
        ],
        axis=1,
    )
    angles = np.stack(
        [water_starts + 1, water_starts, water_starts + 2],
        axis=1,
    )
    constraints = np.stack(
        [
            np.concatenate([water_starts, water_starts, water_starts + 1]),
            np.concatenate([water_starts + 1, water_starts + 2, water_starts + 2]),
        ],
        axis=1,
    )
    constraint_distance = np.tile(
        np.asarray(
            [OH_DISTANCE_ANGSTROM, OH_DISTANCE_ANGSTROM, HH_DISTANCE_ANGSTROM],
            dtype=np.float32,
        ),
        water_count,
    )

    symbols = np.concatenate(
        [
            np.tile(np.asarray(["O", "H", "H"], dtype=str), water_count),
            np.full((sodium_count,), "Na", dtype=str),
            np.full((chloride_count,), "Cl", dtype=str),
        ]
    )
    atom_names = np.concatenate(
        [
            np.tile(np.asarray(["O", "H1", "H2"], dtype=str), water_count),
            np.full((sodium_count,), "NA", dtype=str),
            np.full((chloride_count,), "CL", dtype=str),
        ]
    )
    atom_types = np.concatenate(
        [
            np.tile(np.asarray(["tip3p-O", "tip3p-H", "tip3p-H"], dtype=str), water_count),
            np.full((sodium_count,), "tip3p_standard-Na+", dtype=str),
            np.full((chloride_count,), "tip3p_standard-Cl-", dtype=str),
        ]
    )
    residue_names = np.concatenate(
        [
            np.full((3 * water_count,), "WAT", dtype=str),
            np.full((sodium_count,), "NA", dtype=str),
            np.full((chloride_count,), "CL", dtype=str),
        ]
    )
    residue_ids = np.concatenate(
        [
            np.repeat(np.arange(1, water_count + 1, dtype=np.int32), 3),
            np.arange(water_count + 1, water_count + sodium_count + 1, dtype=np.int32),
            np.arange(
                water_count + sodium_count + 1,
                water_count + sodium_count + chloride_count + 1,
                dtype=np.int32,
            ),
        ]
    )
    chain_ids = np.concatenate(
        [
            np.full((3 * water_count,), "W", dtype=str),
            np.full((sodium_count + chloride_count,), "I", dtype=str),
        ]
    )
    masses = np.concatenate(
        [
            np.tile(np.asarray([TIP3P_O_MASS, TIP3P_H_MASS, TIP3P_H_MASS]), water_count),
            np.full((sodium_count,), NA_MASS),
            np.full((chloride_count,), CL_MASS),
        ]
    ).astype(np.float32)
    charges = np.concatenate(
        [
            np.tile(np.asarray([TIP3P_O_CHARGE, TIP3P_H_CHARGE, TIP3P_H_CHARGE]), water_count),
            np.ones((sodium_count,)),
            -np.ones((chloride_count,)),
        ]
    ).astype(np.float32)
    sigma = np.concatenate(
        [
            np.tile(np.asarray([TIP3P_O_SIGMA_ANGSTROM, 10.0, 10.0]), water_count),
            np.full((sodium_count,), NA_SIGMA_ANGSTROM),
            np.full((chloride_count,), CL_SIGMA_ANGSTROM),
        ]
    ).astype(np.float32)
    epsilon = np.concatenate(
        [
            np.tile(np.asarray([TIP3P_O_EPSILON_KJ_MOL, 0.0, 0.0]), water_count),
            np.full((sodium_count,), NA_EPSILON_KJ_MOL),
            np.full((chloride_count,), CL_EPSILON_KJ_MOL),
        ]
    ).astype(np.float32)
    water_mask = np.zeros((atom_count,), dtype=bool)
    water_mask[: 3 * water_count] = True
    ion_mask = ~water_mask

    pme_alpha = sqrt(-np.log(2.0 * PME_ERROR_TOLERANCE)) / PME_REAL_CUTOFF_ANGSTROM
    requested_mesh = _next_smooth_235(
        ceil(
            2.0
            * pme_alpha
            * box_length
            / (3.0 * PME_ERROR_TOLERANCE) ** (1.0 / PME_ASSIGNMENT_ORDER)
        )
    )
    arrays_for_hash = {
        "positions": positions,
        "masses": masses,
        "charges": charges,
        "sigma": sigma,
        "epsilon": epsilon,
        "bonds": bonds,
        "angles": angles,
        "constraints": constraints,
    }
    content_hash = _fixture_hash(spec, box_length, arrays_for_hash)
    supported_terms = [
        "harmonic_bond",
        "harmonic_angle",
        "nonbonded_lj_coulomb",
        "nonbonded_exception",
        "distance_constraint",
        "pme",
    ]
    pme_config = {
        "mesh_shape": [requested_mesh, requested_mesh, requested_mesh],
        "alpha": pme_alpha,
        "real_cutoff": PME_REAL_CUTOFF_ANGSTROM,
        "assignment_order": PME_ASSIGNMENT_ORDER,
        "charge_tolerance": 1.0e-5,
        "deconvolve_assignment": True,
        "error_tolerance_request": PME_ERROR_TOLERANCE,
        "parameter_authority": "openmm_context_pending",
    }
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        created_at=None,
        source={
            "kind": "deterministic_pme_electrolyte",
            "fixture": spec.name,
            "seed": spec.seed,
            "lattice": "body_centered_cubic",
            "parameter_reference": "OpenMM amber14/tip3p.xml",
        },
        selections={
            "content_hash": content_hash,
            "site_count": spec.site_count,
            "water_count": int(water_count),
            "sodium_count": int(sodium_count),
            "chloride_count": int(chloride_count),
            "atom_count": int(atom_count),
            "ionic_strength_molar": spec.ionic_strength_molar,
            "water_molarity": WATER_MOLARITY,
            "box_lengths_A": [box_length, box_length, box_length],
            "minimum_clearance_lower_bound_A": clearance_lower_bound,
            "net_charge_e": float(np.sum(charges, dtype=np.float64)),
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="amber14_tip3p_joung_cheatham_ions",
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "physical_units": True,
            "hydrogens_present": True,
            "water_present": bool(water_count),
            "water_count": int(water_count),
            "water_model": "TIP3P",
            "ions_present": bool(spec.ion_pairs),
            "ion_count": int(sodium_count + chloride_count),
            "periodic_box_present": True,
            "electrostatics_model": "pme",
            "pme": True,
            "supported_terms": supported_terms,
            "required_terms": supported_terms,
            "unsupported_terms": [],
            "rejected_terms": [],
            "parameter_counts_match_topology": True,
            "fixture_content_hash": content_hash,
        },
        pme_config=pme_config,
        warnings=[
            (
                "The deterministic lattice is a controlled PME validation fixture, "
                "not an equilibrated liquid-water ensemble."
            ),
            (
                "OpenMM is reference-only and must resolve the final alpha and mesh "
                "before matched parity is claimed."
            ),
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
        velocities=np.zeros_like(positions),
        masses=masses,
        charges=charges,
        sigma=sigma,
        epsilon=epsilon,
        bonds=bonds.astype(np.int32),
        bond_k=np.full((bonds.shape[0],), WATER_BOND_K_KJ_MOL_ANGSTROM2, dtype=np.float32),
        bond_length=np.full((bonds.shape[0],), OH_DISTANCE_ANGSTROM, dtype=np.float32),
        angles=angles.astype(np.int32),
        angle_k=np.full((angles.shape[0],), WATER_ANGLE_K_KJ_MOL_RAD2, dtype=np.float32),
        angle_theta=np.full((angles.shape[0],), HOH_ANGLE_RAD, dtype=np.float32),
        dihedrals=empty_indices(4),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=empty_indices(2),
        ligand_mask=np.zeros((atom_count,), dtype=bool),
        receptor_mask=np.zeros((atom_count,), dtype=bool),
        restraint_mask=np.zeros((atom_count,), dtype=bool),
        reference_positions=positions.copy(),
        cell_lengths=np.full((3,), box_length, dtype=np.float32),
        constraints=constraints.astype(np.int32),
        constraint_distance=constraint_distance,
        nonbonded_exception_pairs=constraints.astype(np.int32),
        nonbonded_exception_charge_product=np.zeros((constraints.shape[0],), dtype=np.float32),
        nonbonded_exception_sigma=np.zeros((constraints.shape[0],), dtype=np.float32),
        nonbonded_exception_epsilon=np.zeros((constraints.shape[0],), dtype=np.float32),
        water_mask=water_mask,
        ion_mask=ion_mask,
        lipid_mask=np.zeros((atom_count,), dtype=bool),
        pme_mesh_shape=np.asarray([requested_mesh] * 3, dtype=np.int32),
        pme_alpha=np.asarray([pme_alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([PME_REAL_CUTOFF_ANGSTROM], dtype=np.float32),
        pme_assignment_order=np.asarray([PME_ASSIGNMENT_ORDER], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1.0e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
    )
    prepared.validate()
    return prepared


def fixture_summary(prepared: PreparedSystem) -> dict[str, object]:
    """Return normalized fixture metadata."""

    selections = prepared.metadata.selections
    return {
        "fixture": prepared.metadata.source["fixture"],
        "content_hash": selections["content_hash"],
        "site_count": selections["site_count"],
        "water_count": selections["water_count"],
        "sodium_count": selections["sodium_count"],
        "chloride_count": selections["chloride_count"],
        "atom_count": prepared.atom_count,
        "ionic_strength_molar": selections["ionic_strength_molar"],
        "net_charge_e": float(np.sum(prepared.charges, dtype=np.float64)),
        "box_lengths_A": prepared.cell_lengths.astype(float).tolist(),
        "minimum_clearance_lower_bound_A": selections[
            "minimum_clearance_lower_bound_A"
        ],
        "pme_config": dict(prepared.metadata.pme_config),
    }


def write_pme_fixture(spec: PMEFixtureSpec | str, out_dir: str | Path) -> dict[str, object]:
    """Build and write one prepared fixture."""

    prepared = build_pme_fixture(spec)
    save_prepared_system(prepared, out_dir)
    summary = fixture_summary(prepared)
    summary_path = Path(out_dir) / "pme_fixture.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _box_length_angstrom(site_count: int) -> float:
    volume_liter = site_count / (WATER_MOLARITY * AVOGADRO)
    return float((volume_liter * 1.0e27) ** (1.0 / 3.0))


def _bcc_site_positions(cells: int, box_length: float) -> np.ndarray:
    spacing = box_length / cells
    grid = np.stack(
        np.meshgrid(np.arange(cells), np.arange(cells), np.arange(cells), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    sites = np.concatenate([grid + 0.25, grid + 0.75], axis=0)
    return (sites * spacing).astype(np.float64)


def _clearance_lower_bound(cells: int, box_length: float) -> float:
    nearest_site_distance = sqrt(3.0) * box_length / (2.0 * cells)
    return nearest_site_distance - 2.0 * OH_DISTANCE_ANGSTROM


def _select_separated_sites(
    positions: np.ndarray,
    box_length: float,
    count: int,
    seed: int,
) -> np.ndarray:
    if count == 0:
        return np.asarray([], dtype=np.int32)
    selected: list[int] = [seed % positions.shape[0]]
    min_distance2 = np.full((positions.shape[0],), np.inf, dtype=np.float64)
    while len(selected) < count:
        displacement = positions - positions[selected[-1]]
        displacement -= box_length * np.round(displacement / box_length)
        min_distance2 = np.minimum(min_distance2, np.sum(displacement * displacement, axis=1))
        min_distance2[np.asarray(selected, dtype=np.int32)] = -1.0
        selected.append(int(np.argmax(min_distance2)))
    return np.asarray(selected, dtype=np.int32)


def _water_positions(centers: np.ndarray, box_length: float, *, seed: int) -> np.ndarray:
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [OH_DISTANCE_ANGSTROM, 0.0, 0.0],
            [
                OH_DISTANCE_ANGSTROM * cos(HOH_ANGLE_RAD),
                OH_DISTANCE_ANGSTROM * sin(HOH_ANGLE_RAD),
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random((3, centers.shape[0]))
    quaternions = np.stack(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * pi * u2),
            np.sqrt(u1) * np.sin(2.0 * pi * u3),
            np.sqrt(u1) * np.cos(2.0 * pi * u3),
        ],
        axis=1,
    )
    rotations = _quaternion_rotation_matrices(quaternions)
    positions = centers[:, None, :] + np.einsum("wij,aj->wai", rotations, base)
    return np.mod(positions, box_length)


def _quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternions.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _next_smooth_235(value: int) -> int:
    candidate = max(4, int(value))
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def _fixture_hash(
    spec: PMEFixtureSpec,
    box_length: float,
    arrays: dict[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "name": spec.name,
                "bcc_cells_per_axis": spec.bcc_cells_per_axis,
                "ion_pairs": spec.ion_pairs,
                "seed": spec.seed,
                "box_length": box_length,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(FIXTURE_SPECS), default="target")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.out or Path("results/dhfr-scale-neutral-pme-validation/fixtures") / args.case
    summary = write_pme_fixture(args.case, out)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"fixture={summary['fixture']} atoms={summary['atom_count']} "
            f"hash={summary['content_hash']} out={out}"
        )


if __name__ == "__main__":
    main()
