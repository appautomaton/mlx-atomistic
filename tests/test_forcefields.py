from math import pi

import numpy as np
import pytest

from mlx_atomistic import PMEConfig as ExportedPMEConfig
from mlx_atomistic import RBDihedralPotential as ExportedRBDihedralPotential
from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    ImproperDihedralPotential,
    NonbondedPotential,
    PairRestrictedNonbondedPotential,
    PeriodicDihedralPotential,
    RBDihedralPotential,
)
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.neighbors import build_neighbor_list
from mlx_atomistic.nonbonded import EwaldReferenceConfig, ewald_reference_coulomb_energy_forces
from mlx_atomistic.pme import PMEConfig, pme_coulomb_energy_forces
from mlx_atomistic.topology import Topology


def finite_difference_force(term, positions, *, epsilon=1e-3):
    positions = np.array(positions, dtype=np.float32)
    forces = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus = float(np.array(term.energy_forces(plus)[0]))
            e_minus = float(np.array(term.energy_forces(minus)[0]))
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * epsilon)
    return forces


def assert_force_matches_finite_difference(term, positions, *, atol=2e-3):
    _, forces = term.energy_forces(positions)
    np.testing.assert_allclose(
        np.array(forces),
        finite_difference_force(term, positions),
        atol=atol,
    )


def reference_periodic_dihedral_angle(points):
    delta_ab = points[1] - points[0]
    delta_bc = points[1] - points[2]
    delta_cd = points[3] - points[2]
    cross_ab_bc = np.cross(delta_ab, delta_bc)
    cross_bc_cd = np.cross(delta_bc, delta_cd)
    cosine = np.dot(cross_ab_bc, cross_bc_cd) / (
        np.linalg.norm(cross_ab_bc) * np.linalg.norm(cross_bc_cd)
    )
    angle = np.arccos(np.clip(cosine, -0.999999, 0.999999))
    sign = -1.0 if np.dot(delta_ab, cross_bc_cd) < 0.0 else 1.0
    return angle * sign


def test_harmonic_bond_force_matches_finite_difference():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)
    term = HarmonicBondPotential([(0, 1)], k=5.0, length=1.0)

    assert_force_matches_finite_difference(term, positions)


def test_harmonic_angle_force_matches_finite_difference():
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.9, 0.0]], dtype=np.float32)
    term = HarmonicAnglePotential([(0, 1, 2)], k=2.0, angle=pi / 2.0)

    assert_force_matches_finite_difference(term, positions, atol=3e-3)


def test_periodic_dihedral_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 1.0, 0.0], [1.4, 1.1, 0.8]],
        dtype=np.float32,
    )
    term = PeriodicDihedralPotential([(0, 1, 2, 3)], k=0.4, periodicity=3.0, phase=0.1)

    assert_force_matches_finite_difference(term, positions, atol=5e-3)


def test_rb_dihedral_is_package_export():
    assert ExportedRBDihedralPotential is RBDihedralPotential


def test_pme_config_is_package_export():
    assert ExportedPMEConfig is PMEConfig


def test_rb_dihedral_reference_expression_uses_periodic_angle_minus_pi():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.3, 1.1, 0.1], [1.6, 1.3, 0.9]],
        dtype=np.float32,
    )
    coefficients = np.array([0.17, -0.23, 0.11, 0.07, -0.03, 0.02], dtype=np.float32)
    term = RBDihedralPotential(
        [(0, 1, 2, 3)],
        c0=coefficients[0],
        c1=coefficients[1],
        c2=coefficients[2],
        c3=coefficients[3],
        c4=coefficients[4],
        c5=coefficients[5],
    )

    energy, forces = term.energy_forces(positions)
    periodic_phi = reference_periodic_dihedral_angle(positions)
    rb_phi = periodic_phi - pi
    rb_cosine = np.cos(rb_phi)
    expected = sum(
        coefficient * rb_cosine**power
        for power, coefficient in enumerate(coefficients)
    )

    assert np.isfinite(np.asarray(energy))
    assert np.all(np.isfinite(np.asarray(forces)))
    np.testing.assert_allclose(np.asarray(energy), expected, atol=1e-6)


def test_rb_dihedral_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.3, 1.1, 0.1], [1.6, 1.3, 0.9]],
        dtype=np.float32,
    )
    term = RBDihedralPotential(
        [(0, 1, 2, 3)],
        c0=0.17,
        c1=-0.23,
        c2=0.11,
        c3=0.07,
        c4=-0.03,
        c5=0.02,
    )

    assert_force_matches_finite_difference(term, positions, atol=5e-3)


def test_improper_dihedral_uses_periodic_form():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 1.0, 0.0], [1.4, 1.1, 0.8]],
        dtype=np.float32,
    )
    term = ImproperDihedralPotential([(0, 1, 2, 3)], k=0.4, periodicity=2.0, phase=0.1)

    assert term.name == "improper"
    assert_force_matches_finite_difference(term, positions, atol=5e-3)


def test_coulomb_force_matches_finite_difference():
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.5, 1.2, 0.0]], dtype=np.float32)
    term = CoulombPotential(charges=[1.0, -0.5, 0.25], cutoff=None)

    assert_force_matches_finite_difference(term, positions, atol=3e-3)


def test_lj_pair_list_matches_all_pairs():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.4, 0.0]],
        dtype=np.float32,
    )
    term = LennardJonesPotential(cutoff=None, shift=False)
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)

    dense_energy, dense_forces = term.energy_forces(positions)
    pair_energy, pair_forces = term.energy_forces(positions, pairs=pairs)

    np.testing.assert_allclose(np.array(pair_energy), np.array(dense_energy), atol=1e-6)
    np.testing.assert_allclose(np.array(pair_forces), np.array(dense_forces), atol=1e-6)


def test_lj_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.5, 0.0]],
        dtype=np.float32,
    )
    term = LennardJonesPotential(cutoff=None, shift=False)

    assert_force_matches_finite_difference(term, positions, atol=4e-3)


def test_lj_topology_exclusions_and_one_four_scaling():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0], [4.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        dihedrals=[(0, 1, 2, 3)],
        exclude_bonds=True,
    )
    scaled = LennardJonesPotential(
        cutoff=None,
        shift=False,
        topology=topology,
        one_four_scale=0.5,
    )
    reference = LennardJonesPotential(cutoff=None, shift=False)

    energy, _ = scaled.energy_forces(positions)
    expected = 0.0
    for pair, scale in [((0, 2), 1.0), ((0, 3), 0.5), ((1, 2), 1.0), ((1, 3), 1.0), ((2, 3), 1.0)]:
        pair_energy, _ = reference.energy_forces(positions[list(pair)])
        expected += scale * float(np.array(pair_energy))

    np.testing.assert_allclose(np.array(energy), expected, atol=1e-6)


def test_coulomb_topology_exclusions_and_one_four_scaling():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        dihedrals=[(0, 1, 2, 3)],
        partial_charges=[1.0, -1.0, 1.0, -1.0],
    )
    term = CoulombPotential(topology=topology, one_four_scale=0.5)
    energy, _ = term.energy_forces(positions)

    expected = (1.0 / 2.0) + (-0.5 / 3.0) + (-1.0 / 1.0) + (1.0 / 2.0) + (-1.0 / 1.0)
    np.testing.assert_allclose(np.array(energy), expected, atol=1e-6)


def test_nonbonded_explicit_exception_overrides_regular_pair():
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    excluded = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[1.0, 1.0],
        charges=[1.0, -1.0],
        cutoff=None,
        lj_shift=False,
        exception_pairs=[(0, 1)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
    )
    overridden = NonbondedPotential(
        sigma=[1.0, 1.0],
        epsilon=[1.0, 1.0],
        charges=[1.0, -1.0],
        cutoff=None,
        lj_shift=False,
        exception_pairs=[(0, 1)],
        exception_charge_products=[-0.5],
        exception_sigma=[1.0],
        exception_epsilon=[0.25],
    )

    zero_energy, zero_forces = excluded.energy_forces(positions)
    override_energy, override_forces = overridden.energy_forces(positions)

    np.testing.assert_allclose(np.array(zero_energy), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.array(zero_forces), np.zeros_like(positions), atol=1e-7)
    assert float(np.array(override_energy)) != 0.0
    assert np.linalg.norm(np.array(override_forces)) > 0.0


def test_pair_restricted_nonbonded_matches_explicit_pair_argument():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.6, 0.0]],
        dtype=np.float32,
    )
    base = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.2, 0.1, 0.3],
        charges=[1.0, -0.5, 0.25],
        cutoff=None,
        lj_shift=False,
    )
    pairs = np.asarray([[0, 2]], dtype=np.int32)
    restricted = PairRestrictedNonbondedPotential(base, pairs)

    expected_energy, expected_forces = base.energy_forces(positions, pairs=pairs)
    actual_energy, actual_forces = restricted.energy_forces(positions)

    np.testing.assert_allclose(np.array(actual_energy), np.array(expected_energy), atol=1e-7)
    np.testing.assert_allclose(np.array(actual_forces), np.array(expected_forces), atol=1e-7)


def _bare_coulomb_energy(positions, charges_or_products, pairs):
    total = 0.0
    for index, (i, j) in enumerate(pairs):
        distance = np.linalg.norm(positions[i] - positions[j])
        if np.asarray(charges_or_products).ndim == 1 and len(charges_or_products) == len(pairs):
            charge_product = charges_or_products[index]
        else:
            charge_product = charges_or_products[i] * charges_or_products[j]
        total += charge_product / distance
    return total


def _bare_coulomb_energy_forces(positions, charges_or_products, pairs):
    total = 0.0
    forces = np.zeros_like(positions, dtype=np.float32)
    for index, (i, j) in enumerate(pairs):
        displacement = positions[i] - positions[j]
        r2 = float(np.sum(displacement * displacement))
        distance = float(np.sqrt(r2))
        if np.asarray(charges_or_products).ndim == 1 and len(charges_or_products) == len(pairs):
            charge_product = float(charges_or_products[index])
        else:
            charge_product = float(charges_or_products[i] * charges_or_products[j])
        total += charge_product / distance
        pair_force = charge_product / (r2 * distance) * displacement
        forces[i] += pair_force
        forces[j] -= pair_force
    return total, forces


def _bare_lj_energy(positions, sigma, epsilon, pairs):
    total = 0.0
    for index, (i, j) in enumerate(pairs):
        distance = np.linalg.norm(positions[i] - positions[j])
        inv_r = sigma[index] / distance
        inv_r6 = inv_r**6
        total += 4.0 * epsilon[index] * (inv_r6 * inv_r6 - inv_r6)
    return total


def test_nonbonded_nbfix_type_pairs_substitute_lj_parameters():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.5, 0.0]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.2, 1.0],
        epsilon=[0.2, 0.3, 0.2],
        charges=[0.0, 0.0, 0.0],
        atom_types=["H", "O", "H"],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.1],
        nbfix_type_epsilon=[0.5],
        cutoff=None,
        lj_shift=False,
    )

    energy, forces = term.energy_forces(positions)
    expected = _bare_lj_energy(
        positions,
        sigma=[1.1, 1.0, 1.1],
        epsilon=[0.5, 0.2, 0.5],
        pairs=[(0, 1), (0, 2), (1, 2)],
    )

    np.testing.assert_allclose(np.array(energy), expected, atol=1e-6)
    assert np.all(np.isfinite(np.array(forces)))
    assert_force_matches_finite_difference(term, positions, atol=5e-3)


def test_nonbonded_nbfix_explicit_pairs_substitute_lj_parameters():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.5, 0.0]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.2, 1.0],
        epsilon=[0.2, 0.3, 0.2],
        charges=[0.0, 0.0, 0.0],
        nbfix_pairs=[(0, 1)],
        nbfix_sigma=[1.1],
        nbfix_epsilon=[0.5],
        cutoff=None,
        lj_shift=False,
    )

    energy, forces = term.energy_forces(positions)
    expected = _bare_lj_energy(
        positions,
        sigma=[1.1, 1.0, 1.1],
        epsilon=[0.5, 0.2, np.sqrt(0.3 * 0.2)],
        pairs=[(0, 1), (0, 2), (1, 2)],
    )

    np.testing.assert_allclose(np.array(energy), expected, atol=1e-6)
    assert np.all(np.isfinite(np.array(forces)))
    assert_force_matches_finite_difference(term, positions, atol=5e-3)


def test_nonbonded_nbfix_type_pairs_reject_unknown_atom_types():
    try:
        NonbondedPotential(
            sigma=[1.0, 1.2],
            epsilon=[0.2, 0.3],
            charges=[0.0, 0.0],
            atom_types=["H", "O"],
            nbfix_type_pairs=[("H", "CLGR1")],
            nbfix_type_sigma=[1.1],
            nbfix_type_epsilon=[0.5],
            cutoff=None,
            lj_shift=False,
        )
    except ValueError as err:
        assert "absent from atom_types" in str(err)
        assert "CLGR1" in str(err)
    else:
        raise AssertionError("NBFIX accepted an unknown atom type")


def test_nonbonded_nbfix_respects_topology_exclusions():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.6, 0.0]],
        dtype=np.float32,
    )
    topology = Topology.from_sequences(n_atoms=3, bonds=[(0, 1)], exclude_bonds=True)
    term = NonbondedPotential(
        sigma=[1.0, 1.2, 1.0],
        epsilon=[0.2, 0.3, 0.2],
        charges=[0.0, 0.0, 0.0],
        topology=topology,
        atom_types=["H", "O", "H"],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.1],
        nbfix_type_epsilon=[0.5],
        cutoff=None,
        lj_shift=False,
    )

    energy, _ = term.energy_forces(positions)
    expected = _bare_lj_energy(
        positions,
        sigma=[1.0, 1.1],
        epsilon=[0.2, 0.5],
        pairs=[(0, 2), (1, 2)],
    )

    np.testing.assert_allclose(np.array(energy), expected, atol=1e-6)


def test_nonbonded_nbfix_does_not_override_explicit_exceptions():
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float32)
    term = NonbondedPotential(
        sigma=[1.0, 1.2],
        epsilon=[0.2, 0.3],
        charges=[0.0, 0.0],
        atom_types=["H", "O"],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.1],
        nbfix_type_epsilon=[0.5],
        exception_pairs=[(0, 1)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
        cutoff=None,
        lj_shift=False,
    )

    energy, forces = term.energy_forces(positions)

    np.testing.assert_allclose(np.array(energy), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.array(forces), np.zeros_like(positions), atol=1e-7)


def test_nonbonded_ewald_matches_reference_without_lj():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -0.5, -0.5], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="ewald_reference",
        ewald_config=config,
    )

    energy, forces, components = term.energy_forces_with_components(positions, cell)
    (
        reference_energy,
        reference_forces,
        reference_components,
    ) = ewald_reference_coulomb_energy_forces(positions, charges, cell, config=config)

    np.testing.assert_allclose(np.array(energy), np.array(reference_energy), atol=1e-6)
    np.testing.assert_allclose(np.array(forces), np.array(reference_forces), atol=1e-6)
    for name in ["coulomb_real", "coulomb_reciprocal", "coulomb_self"]:
        np.testing.assert_allclose(
            np.array(components[name]),
            np.array(reference_components[name]),
            atol=1e-6,
        )


def test_nonbonded_ewald_topology_exclusions_and_exceptions_are_corrections():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [2.0, 3.0, 5.0], [7.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    cell = Cell.cubic(12.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        nonbonded_exception_pairs=[(0, 2)],
    )
    config = EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="ewald_reference",
        ewald_config=config,
        topology=topology,
        exception_pairs=[(0, 2)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
    )

    energy, _, components = term.energy_forces_with_components(positions, cell)
    full_ewald, _, _ = ewald_reference_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    expected_correction = -_bare_coulomb_energy(positions, charges, [(0, 1), (0, 2)])

    np.testing.assert_allclose(
        np.array(components["coulomb_exclusion_correction"]),
        expected_correction,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.array(components["coulomb_exception"]), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        np.array(energy),
        np.array(full_ewald) + expected_correction,
        atol=1e-6,
    )


def test_nonbonded_ewald_one_four_scaling_is_correction_not_double_count():
    positions = np.array(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0], [4.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    cell = Cell.cubic(12.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[],
        dihedrals=[(0, 1, 2, 3)],
        exclude_bonds=False,
    )
    config = EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="ewald_reference",
        ewald_config=config,
        topology=topology,
        coulomb_one_four_scale=0.5,
    )

    energy, _, components = term.energy_forces_with_components(positions, cell)
    full_ewald, _, _ = ewald_reference_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    expected_correction = -0.5 * _bare_coulomb_energy(positions, charges, [(0, 3)])

    np.testing.assert_allclose(
        np.array(components["coulomb_one_four_correction"]),
        expected_correction,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(energy),
        np.array(full_ewald) + expected_correction,
        atol=1e-6,
    )


def test_nonbonded_pme_requires_mesh_config_cell_and_full_system():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -0.5, -0.5], dtype=np.float32)

    try:
        NonbondedPotential(
            sigma=[1.0, 1.0, 1.0],
            epsilon=[0.0, 0.0, 0.0],
            charges=charges,
            cutoff=5.0,
            electrostatics="pme",
        )
    except ValueError as err:
        assert "pme_config" in str(err)
    else:
        raise AssertionError("PME nonbonded accepted missing pme_config")

    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0),
    )
    for kwargs, expected in [
        ({}, "periodic cell"),
        (
            {"cell": Cell.cubic(12.0), "pairs": np.array([[0, 1]], dtype=np.int32)},
            "pme_production_direct_space_requires_neighbor_blocks",
        ),
    ]:
        try:
            term.energy_forces(positions, **kwargs)
        except ValueError as err:
            assert expected in str(err)
        else:
            raise AssertionError(f"PME nonbonded accepted invalid input: {kwargs}")

    with pytest.raises(ValueError, match="positive orthorhombic"):
        Cell.orthorhombic([12.0, 0.0, 12.0])


def test_nonbonded_pme_refuses_non_neutral_background_policy():
    try:
        NonbondedPotential(
            sigma=[1.0, 1.0, 1.0],
            epsilon=[0.0, 0.0, 0.0],
            charges=[1.0, -0.5, -0.4],
            cutoff=5.0,
            electrostatics="pme",
            pme_config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0),
        )
    except ValueError as err:
        assert "non-neutral background policy is not implemented" in str(err)
    else:
        raise AssertionError("PME nonbonded accepted a non-neutral system")


def test_nonbonded_pme_refuses_non_finite_coulomb_constant_and_config_fields():
    valid_kwargs = {
        "sigma": [1.0, 1.0],
        "epsilon": [0.0, 0.0],
        "charges": [1.0, -1.0],
        "cutoff": 5.0,
        "electrostatics": "pme",
    }

    for value in [np.nan, np.inf, -np.inf]:
        try:
            NonbondedPotential(
                **valid_kwargs,
                coulomb_constant=value,
                pme_config=PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35, real_cutoff=5.0),
            )
        except ValueError as err:
            assert "coulomb_constant must be finite" in str(err)
        else:
            raise AssertionError(f"PME nonbonded accepted coulomb_constant={value}")

    negative_tolerance = PMEConfig(mesh_shape=(16, 16, 16), alpha=0.35)
    object.__setattr__(negative_tolerance, "charge_tolerance", -1.0)

    try:
        NonbondedPotential(**valid_kwargs, pme_config=negative_tolerance)
    except ValueError as err:
        assert "pme_config.charge_tolerance" in str(err)
    else:
        raise AssertionError("PME nonbonded accepted invalid pme_config.charge_tolerance")


def test_nonbonded_pme_matches_standalone_pme_without_lj():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -0.5, -0.5], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )

    energy, forces, components = term.energy_forces_with_components(positions, cell)
    reference_energy, reference_forces, reference_components = pme_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )

    np.testing.assert_allclose(np.array(energy), np.array(reference_energy), atol=1e-6)
    np.testing.assert_allclose(np.array(forces), np.array(reference_forces), atol=1e-6)
    for name in ["coulomb_real", "coulomb_reciprocal", "coulomb_self"]:
        np.testing.assert_allclose(
            np.array(components[name]),
            np.array(reference_components[name]),
            atol=1e-6,
        )
    assert components["pme_diagnostics"].mesh_shape == (24, 24, 24)
    assert components["pme_diagnostics"].assignment_order == 2


@pytest.mark.parametrize("assignment_order", [4, 5])
def test_nonbonded_pme_diagnostics_report_assignment_order(assignment_order):
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(
        mesh_shape=(24, 24, 24),
        alpha=0.35,
        real_cutoff=5.0,
        assignment_order=assignment_order,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )

    energy, forces, components = term.energy_forces_with_components(positions, cell)

    assert components["pme_diagnostics"].assignment_order == assignment_order
    assert np.isfinite(np.asarray(energy))
    assert np.all(np.isfinite(np.asarray(forces)))


def test_nonbonded_pme_uses_shared_direct_space_blocks_for_production_total():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=5.0,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=2,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )

    dense_energy, dense_forces, _ = term.energy_forces_with_components(positions, cell)
    block_energy, block_forces, components = term.energy_forces_with_components(
        positions,
        cell,
        pairs=neighbors.interactions,
    )

    diagnostics = components["pme_diagnostics"]
    assert diagnostics.direct_space_policy == "block_candidate"
    assert diagnostics.direct_space_representation == "blocks"
    assert diagnostics.direct_space_candidate_count == neighbors.candidate_count
    np.testing.assert_allclose(np.asarray(block_energy), np.asarray(dense_energy), atol=1e-6)
    np.testing.assert_allclose(np.asarray(block_forces), np.asarray(dense_forces), atol=1e-6)


def test_nonbonded_pme_fails_closed_for_unsafe_shared_direct_space_policy():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=7.0)
    neighbors = build_neighbor_list(
        positions,
        cell,
        cutoff=5.0,
        skin=0.0,
        backend="mlx_cell_blocks",
        block_size=2,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
    )

    with pytest.raises(
        ValueError,
        match="pme_direct_space_pair_policy_requires_cutoff_at_or_below_half_min_box",
    ):
        term.energy_forces(positions, cell, pairs=neighbors.interactions)


def test_nonbonded_pme_allows_nbfix_lj_without_changing_coulomb():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0]],
        dtype=np.float32,
    )
    charges = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    cell = Cell.cubic(12.0)
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.2, 1.0],
        epsilon=[0.2, 0.3, 0.2],
        charges=charges,
        atom_types=["H", "O", "H"],
        nbfix_type_pairs=[("H", "O")],
        nbfix_type_sigma=[1.1],
        nbfix_type_epsilon=[0.5],
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
        lj_shift=False,
    )

    energy, _, components = term.energy_forces_with_components(positions, cell)
    reference_coulomb, _, _ = pme_coulomb_energy_forces(positions, charges, cell, config=config)
    expected_lj = _bare_lj_energy(
        positions,
        sigma=[1.1, 1.0, 1.1],
        epsilon=[0.5, 0.2, 0.5],
        pairs=[(0, 1), (0, 2), (1, 2)],
    )

    np.testing.assert_allclose(
        np.array(components["coulomb"]),
        np.array(reference_coulomb),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.array(components["lj"]), expected_lj, atol=1e-6)
    np.testing.assert_allclose(
        np.array(energy),
        np.array(reference_coulomb) + expected_lj,
        atol=1e-6,
    )


def test_nonbonded_pme_nonzero_exception_override_corrects_energy_and_forces():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.2, 1.1], [2.0, 3.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    charges = np.array([0.7, -0.2, -0.3, -0.2], dtype=np.float32)
    cell = Cell.cubic(12.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        nonbonded_exception_pairs=[(0, 2)],
    )
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
        topology=topology,
        exception_pairs=[(0, 2)],
        exception_charge_products=[0.125],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
    )

    energy, forces, components = term.energy_forces_with_components(positions, cell)
    full_pme, full_pme_forces, _ = pme_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )
    original_energy, original_forces = _bare_coulomb_energy_forces(
        positions,
        charges,
        [(0, 2)],
    )
    override_energy, override_forces = _bare_coulomb_energy_forces(
        positions,
        [0.125],
        [(0, 2)],
    )

    np.testing.assert_allclose(
        np.array(components["coulomb_exclusion_correction"]),
        -original_energy,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(components["coulomb_exception"]),
        override_energy,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(components["coulomb"]),
        np.array(full_pme) - original_energy + override_energy,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(energy),
        np.array(full_pme) - original_energy + override_energy,
        atol=1e-6,
    )
    expected_forces = np.array(full_pme_forces) - original_forces + override_forces
    assert np.all(np.isfinite(np.array(forces)))
    np.testing.assert_allclose(np.array(forces), expected_forces, atol=1e-6)


def test_nonbonded_pme_exclusions_exceptions_and_one_four_are_corrections():
    positions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [2.0, 3.0, 5.0], [7.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    cell = Cell.cubic(12.0)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        dihedrals=[(0, 1, 2, 3)],
        nonbonded_exception_pairs=[(0, 2)],
        exclude_bonds=True,
    )
    config = PMEConfig(mesh_shape=(24, 24, 24), alpha=0.35, real_cutoff=5.0)
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0, 1.0],
        epsilon=[0.0, 0.0, 0.0, 0.0],
        charges=charges,
        cutoff=5.0,
        electrostatics="pme",
        pme_config=config,
        topology=topology,
        coulomb_one_four_scale=0.5,
        exception_pairs=[(0, 2)],
        exception_charge_products=[0.0],
        exception_sigma=[0.0],
        exception_epsilon=[0.0],
    )

    energy, _, components = term.energy_forces_with_components(positions, cell)
    full_pme, _, _ = pme_coulomb_energy_forces(positions, charges, cell, config=config)
    excluded_and_exception = _bare_coulomb_energy(positions, charges, [(0, 1), (0, 2)])
    one_four_delta = -0.5 * _bare_coulomb_energy(positions, charges, [(0, 3)])

    np.testing.assert_allclose(
        np.array(components["coulomb_exclusion_correction"]),
        -excluded_and_exception,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.array(components["coulomb_exception"]), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        np.array(components["coulomb_one_four_correction"]),
        one_four_delta,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(energy),
        np.array(full_pme) - excluded_and_exception + one_four_delta,
        atol=1e-6,
    )


def test_nonbonded_switch_force_matches_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.6, 0.0]],
        dtype=np.float32,
    )
    term = NonbondedPotential(
        sigma=[1.0, 1.0, 1.0],
        epsilon=[1.0, 1.0, 1.0],
        charges=[0.0, 0.0, 0.0],
        cutoff=2.0,
        switch_distance=1.5,
        lj_shift=False,
    )

    assert_force_matches_finite_difference(term, positions, atol=7e-3)
