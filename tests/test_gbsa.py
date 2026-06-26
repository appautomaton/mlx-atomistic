import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.artifacts import (
    PREPARED_JSON_NAME,
    PREPARED_NPZ_NAME,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
    validate_mlx_compatibility,
)
from mlx_atomistic.core import Cell
from mlx_atomistic.gbsa import (
    DEFAULT_PROBE_RADIUS_A,
    DEFAULT_RADIUS_OFFSET_A,
    DEFAULT_SURFACE_AREA_ENERGY_KJ_MOL_A2,
    GBSAForcePotential,
)
from mlx_atomistic.prep.io import (
    load_prepared_system,
    save_prepared_system,
    synthetic_prepared_system,
)


def _finite_difference_force(term, positions, *, cell=None, epsilon=1e-3):
    positions = np.asarray(positions, dtype=np.float32)
    forces = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus = float(np.asarray(term.energy_forces(plus, cell=cell)[0]))
            e_minus = float(np.asarray(term.energy_forces(minus, cell=cell)[0]))
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * epsilon)
    return forces


def _core_arrays() -> dict[str, np.ndarray]:
    return {
        "symbols": np.asarray(["C", "N", "O", "H"], dtype=str),
        "atom_names": np.asarray(["C", "N", "O", "H"], dtype=str),
        "atom_types": np.asarray(["C", "N", "O", "H"], dtype=str),
        "positions": np.asarray(
            [[0.0, 0.0, 0.0], [1.6, 0.1, 0.0], [0.2, 1.7, 0.3], [2.2, 0.2, 0.8]],
            dtype=np.float32,
        ),
        "velocities": np.zeros((4, 3), dtype=np.float32),
        "masses": np.asarray([12.0, 14.0, 16.0, 1.008], dtype=np.float32),
        "charges": np.asarray([0.2, -0.3, 0.1, 0.0], dtype=np.float32),
        "sigma": np.ones((4,), dtype=np.float32),
        "epsilon": np.zeros((4,), dtype=np.float32),
        "bonds": np.empty((0, 2), dtype=np.int32),
        "bond_k": np.asarray([], dtype=np.float32),
        "bond_length": np.asarray([], dtype=np.float32),
        "angles": np.empty((0, 3), dtype=np.int32),
        "angle_k": np.asarray([], dtype=np.float32),
        "angle_theta": np.asarray([], dtype=np.float32),
        "dihedrals": np.empty((0, 4), dtype=np.int32),
        "dihedral_k": np.asarray([], dtype=np.float32),
        "dihedral_periodicity": np.asarray([], dtype=np.float32),
        "dihedral_phase": np.asarray([], dtype=np.float32),
        "gbsa_radius": np.asarray([1.7, 1.55, 1.5, 1.2], dtype=np.float32),
        "gbsa_scale": np.asarray([0.72, 0.79, 0.85, 0.85], dtype=np.float32),
    }


def test_gbsa_surface_area_matches_single_atom_analytical_reference():
    radius = 1.7
    term = GBSAForcePotential(charges=[0.0], radius=[radius], scale=[0.8])
    positions = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    energy = term.ace_surface_area_energy(positions)
    born = radius - DEFAULT_RADIUS_OFFSET_A
    expected = (
        4.0
        * np.pi
        * DEFAULT_SURFACE_AREA_ENERGY_KJ_MOL_A2
        * (radius + DEFAULT_PROBE_RADIUS_A) ** 2
        * (radius / born) ** 6
    )

    np.testing.assert_allclose(np.asarray(energy), expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize(
    "charges",
    [
        [0.0, 0.0, 0.0],
        [0.4, -0.7, 0.3],
    ],
)
def test_gbsa_energy_forces_are_finite_and_match_finite_difference_periodic(charges):
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.8, 0.2, 0.1], [0.4, 2.0, -0.2]],
        dtype=np.float32,
    )
    cell = Cell.orthorhombic([8.0, 8.0, 8.0])
    term = GBSAForcePotential(
        charges=charges,
        radius=[1.7, 1.55, 1.5],
        scale=[0.72, 0.79, 0.85],
    )

    energy, forces = term.energy_forces(positions, cell=cell)

    assert np.isfinite(float(np.asarray(energy)))
    assert np.all(np.isfinite(np.asarray(forces)))
    np.testing.assert_allclose(
        np.asarray(forces),
        _finite_difference_force(term, positions, cell=cell),
        atol=2e-2,
        rtol=2e-2,
    )


def test_gbsa_ignores_runtime_pairs_for_nocutoff_obc_semantics():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.8, 0.2, 0.1], [0.4, 2.0, -0.2], [2.5, 2.1, 0.3]],
        dtype=np.float32,
    )
    truncated_pairs = np.asarray([[0, 1]], dtype=np.int32)
    term = GBSAForcePotential(
        charges=[0.4, -0.7, 0.3, 0.2],
        radius=[1.7, 1.55, 1.5, 1.2],
        scale=[0.72, 0.79, 0.85, 0.85],
    )

    full_energy, full_forces = term.energy_forces(positions)
    pair_energy, pair_forces = term.energy_forces(positions, pairs=truncated_pairs)
    potential_energy = term.potential_energy(positions, pairs=truncated_pairs)

    np.testing.assert_allclose(np.asarray(pair_energy), np.asarray(full_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_forces), np.asarray(full_forces), atol=1e-6)
    np.testing.assert_allclose(np.asarray(potential_energy), np.asarray(full_energy), atol=1e-6)


def test_gbsa_md_term_path_ignores_neighbor_pairs():
    from mlx_atomistic.md import _energy_forces_from_terms

    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.8, 0.2, 0.1], [0.4, 2.0, -0.2], [2.5, 2.1, 0.3]],
        dtype=np.float32,
    )
    truncated_pairs = np.asarray([[0, 1]], dtype=np.int32)
    term = GBSAForcePotential(
        charges=[0.4, -0.7, 0.3, 0.2],
        radius=[1.7, 1.55, 1.5, 1.2],
        scale=[0.72, 0.79, 0.85, 0.85],
    )

    full_energy, full_forces = _energy_forces_from_terms(
        positions,
        (term,),
        cell=None,
        pairs=None,
    )
    pair_energy, pair_forces = _energy_forces_from_terms(
        positions,
        (term,),
        cell=None,
        pairs=truncated_pairs,
    )

    np.testing.assert_allclose(np.asarray(pair_energy), np.asarray(full_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(pair_forces), np.asarray(full_forces), atol=1e-6)


def _native_mini_protein_positions() -> np.ndarray:
    pdb_path = Path("tests/fixtures/charmm/native-mini.pdb")
    positions = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM"):
            positions.append(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
    return np.asarray(positions, dtype=np.float32)


@pytest.mark.reference
def test_gbsa_openmm_obc_reference_energy_for_protein_fixture():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    positions_a = _native_mini_protein_positions()
    charges = np.asarray([-0.3, 0.3, 0.1, 0.09, 0.5, -0.5, -0.1, 0.1], dtype=np.float32)
    radii_a = np.asarray([1.55, 1.2, 1.7, 1.2, 1.7, 1.5, 1.7, 1.2], dtype=np.float32)
    scales = np.asarray([0.79, 0.85, 0.72, 0.85, 0.72, 0.85, 0.72, 0.85], dtype=np.float32)
    term = GBSAForcePotential(charges=charges, radius=radii_a, scale=scales)
    mlx_energy = float(np.asarray(term.energy_forces(positions_a)[0]))

    system = openmm.System()
    for _ in range(positions_a.shape[0]):
        system.addParticle(12.0)
    gbsa = openmm.GBSAOBCForce()
    gbsa.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    gbsa.setSolventDielectric(78.5)
    gbsa.setSoluteDielectric(1.0)
    gbsa.setSurfaceAreaEnergy(2.25936 * unit.kilojoule_per_mole / unit.nanometer**2)
    for charge, radius, scale in zip(charges, radii_a, scales, strict=True):
        gbsa.addParticle(float(charge), float(radius / 10.0), float(scale))
    system.addForce(gbsa)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator)
    context.setPositions((positions_a / 10.0) * unit.nanometer)
    state = context.getState(getEnergy=True)
    openmm_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    np.testing.assert_allclose(mlx_energy, openmm_energy, atol=3e-3, rtol=3e-4)


def test_gbsa_artifact_load_build_preserves_parameters(tmp_path: Path):
    arrays = _core_arrays()
    metadata = {
        "artifact_version": 2,
        "source": {"kind": "synthetic"},
        "units": {
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        "parameter_source": "gbsa_test",
        "compatibility_report": {
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": 1,
            "supported_terms": ["nonbonded_lj_coulomb", "gbsa"],
            "required_terms": ["nonbonded_lj_coulomb", "gbsa"],
            "unsupported_terms": [],
        },
        "gbsa": {
            "solvent_dielectric": 70.0,
            "solute_dielectric": 1.2,
            "surface_area_energy": 0.03,
            "probe_radius": 1.3,
            "radius_offset": 0.08,
        },
    }
    (tmp_path / PREPARED_JSON_NAME).write_text(json.dumps(metadata))
    np.savez_compressed(tmp_path / PREPARED_NPZ_NAME, **arrays)

    assert validate_mlx_compatibility(metadata, require_production=True, arrays=arrays) is not None
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    _, terms, _ = build_mlx_system_from_artifact(artifact)

    gbsa_terms = [term for term in terms if term.name == "gbsa"]
    assert len(gbsa_terms) == 1
    gbsa = gbsa_terms[0]
    np.testing.assert_allclose(np.asarray(artifact.arrays["gbsa_radius"]), arrays["gbsa_radius"])
    np.testing.assert_allclose(np.asarray(gbsa.radius), arrays["gbsa_radius"])
    np.testing.assert_allclose(np.asarray(gbsa.scale), arrays["gbsa_scale"])
    assert gbsa.solvent_dielectric == 70.0
    assert gbsa.solute_dielectric == 1.2


def test_gbsa_prepared_system_save_load_round_trips_parameters(tmp_path: Path):
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            "supported_terms": ["harmonic_bond", "nonbonded_lj_coulomb", "gbsa"],
            "required_terms": ["harmonic_bond", "nonbonded_lj_coulomb", "gbsa"],
            "unsupported_terms": [],
        },
    )
    gbsa_radius = np.asarray([1.7, 1.5], dtype=np.float32)
    gbsa_scale = np.asarray([0.72, 0.85], dtype=np.float32)
    prepared = replace(
        prepared,
        metadata=metadata,
        gbsa_radius=gbsa_radius,
        gbsa_scale=gbsa_scale,
    )

    save_prepared_system(prepared, tmp_path)
    reloaded = load_prepared_system(tmp_path)

    np.testing.assert_allclose(reloaded.gbsa_radius, gbsa_radius)
    np.testing.assert_allclose(reloaded.gbsa_scale, gbsa_scale)
