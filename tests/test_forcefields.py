from math import pi

import numpy as np

from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.md import LennardJonesPotential
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
