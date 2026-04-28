"""Small programmatic examples for molecular mechanics and DFT workflows."""

from __future__ import annotations

from math import pi

import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.dft import DFTSystem, SCFConfig
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.mm import (
    AngleParameter,
    AtomType,
    BondParameter,
    DihedralParameter,
    ForceField,
    MMSystem,
    NonbondedParameter,
)
from mlx_atomistic.topology import Topology


def bonded_chain_example():
    """Return a four-particle bonded chain with bond, angle, and torsion terms."""

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [3.0, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        angles=[(0, 1, 2), (1, 2, 3)],
        dihedrals=[(0, 1, 2, 3)],
        partial_charges=[0.0, 0.0, 0.0, 0.0],
    )
    force_terms = [
        HarmonicBondPotential(topology.bonds, k=100.0, length=1.0),
        HarmonicAnglePotential(topology.angles, k=10.0, angle=pi),
        PeriodicDihedralPotential(topology.dihedrals, k=0.2, periodicity=3, phase=0.0),
    ]
    return positions, velocities, topology, force_terms


def charged_dimer_example():
    """Return a two-particle charged system with direct Coulomb interactions."""

    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    topology = Topology.from_sequences(n_atoms=2, partial_charges=[1.0, -1.0])
    force_terms = [CoulombPotential(topology=topology)]
    return positions, velocities, topology, force_terms


def lj_liquid_example(
    *,
    particles: int = 32,
    density: float = 0.8,
    temperature: float = 1.0,
    seed: int = 7,
):
    """Return an LJ liquid starting point."""

    positions, cell = fcc_lattice(particles, density=density)
    velocities = thermal_velocities(particles, temperature=temperature, seed=seed)
    return positions, velocities, cell, [LennardJonesPotential(cutoff=2.5)]


def periodic_cell_example() -> Cell:
    """Return the default reduced-unit demonstration cell."""

    return Cell.cubic(8.0)


def water_like_constrained_example():
    """Return a small constrained water-like molecular mechanics system."""

    topology = Topology.from_sequences(
        n_atoms=3,
        bonds=[(0, 1), (0, 2)],
        angles=[(1, 0, 2)],
        partial_charges=[-0.8, 0.4, 0.4],
    )
    force_field = ForceField(
        atom_types=[AtomType("OW", 16.0), AtomType("HW", 1.0)],
        nonbonded=[
            NonbondedParameter("OW", sigma=1.0, epsilon=0.15),
            NonbondedParameter("HW", sigma=0.4, epsilon=0.02),
        ],
        bonds=[BondParameter(("OW", "HW"), k=100.0, length=1.0)],
        angles=[AngleParameter(("HW", "OW", "HW"), k=20.0, angle=1.824)],
        cutoff=None,
        lj_shift=False,
    )
    system = MMSystem.from_sequences(
        symbols=["O", "H", "H"],
        atom_names=["O", "H1", "H2"],
        atom_types=["OW", "HW", "HW"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.25, 0.9682458, 0.0],
        ],
        velocities=[
            [0.0, 0.02, 0.0],
            [0.01, -0.01, 0.0],
            [-0.01, -0.01, 0.0],
        ],
        topology=topology,
        charges=[-0.8, 0.4, 0.4],
        atom_type_masses=force_field.atom_type_masses,
    )
    constraints = DistanceConstraints([(0, 1), (0, 2)], distances=[1.0, 1.0])
    return system, force_field, constraints


def butane_like_torsion_example():
    """Return a four-atom butane-like torsion system."""

    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        angles=[(0, 1, 2), (1, 2, 3)],
        dihedrals=[(0, 1, 2, 3)],
        partial_charges=[0.0, 0.0, 0.0, 0.0],
    )
    force_field = ForceField(
        atom_types=[AtomType("CT", 12.0)],
        nonbonded=[NonbondedParameter("CT", sigma=1.0, epsilon=0.4)],
        bonds=[BondParameter(("CT", "CT"), k=80.0, length=1.0)],
        angles=[AngleParameter(("CT", "CT", "CT"), k=10.0, angle=1.91)],
        dihedrals=[DihedralParameter(("CT", "CT", "CT", "CT"), k=0.4, periodicity=3.0)],
        cutoff=None,
        lj_shift=False,
        lj_one_four_scale=0.5,
        coulomb_one_four_scale=0.5,
    )
    system = MMSystem.from_sequences(
        symbols=["C", "C", "C", "C"],
        atom_names=["C1", "C2", "C3", "C4"],
        atom_types=["CT", "CT", "CT", "CT"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.45, 0.9, 0.0],
            [2.0, 1.1, 0.75],
        ],
        topology=topology,
        atom_type_masses=force_field.atom_type_masses,
    )
    return system, force_field


def ionic_cluster_example():
    """Return a small typed ionic cluster."""

    topology = Topology.from_sequences(
        n_atoms=4,
        partial_charges=[1.0, -1.0, 1.0, -1.0],
        exclude_bonds=False,
    )
    force_field = ForceField(
        atom_types=[AtomType("IP", 22.0), AtomType("IN", 35.0)],
        nonbonded=[
            NonbondedParameter("IP", sigma=0.9, epsilon=0.2),
            NonbondedParameter("IN", sigma=1.1, epsilon=0.25),
        ],
        cutoff=None,
        lj_shift=False,
    )
    system = MMSystem.from_sequences(
        symbols=["Na", "Cl", "Na", "Cl"],
        atom_names=["Na1", "Cl1", "Na2", "Cl2"],
        atom_types=["IP", "IN", "IP", "IN"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 1.4, 0.0],
            [1.4, 1.4, 0.0],
        ],
        topology=topology,
        charges=[1.0, -1.0, 1.0, -1.0],
        atom_type_masses=force_field.atom_type_masses,
    )
    return system, force_field


def mixed_lj_fluid_example(
    *,
    particles: int = 32,
    density: float = 0.8,
    temperature: float = 1.0,
    seed: int = 7,
):
    """Return a typed mixed LJ fluid."""

    positions, cell = fcc_lattice(particles, density=density)
    velocities = thermal_velocities(particles, temperature=temperature, seed=seed)
    atom_types = ["A" if index % 2 == 0 else "B" for index in range(particles)]
    topology = Topology.from_sequences(n_atoms=particles, exclude_bonds=False)
    force_field = ForceField(
        atom_types=[AtomType("A", 1.0), AtomType("B", 1.5)],
        nonbonded=[
            NonbondedParameter("A", sigma=1.0, epsilon=1.0),
            NonbondedParameter("B", sigma=0.88, epsilon=0.6),
        ],
    )
    system = MMSystem.from_sequences(
        symbols=["Ar"] * particles,
        atom_names=[f"Ar{index + 1}" for index in range(particles)],
        atom_types=atom_types,
        positions=positions,
        velocities=velocities,
        topology=topology,
        cell=cell,
        atom_type_masses=force_field.atom_type_masses,
    )
    return system, force_field


def toy_one_electron_dft_example():
    """Return a compact one-electron Gaussian-well DFT toy system."""

    system = DFTSystem(
        cell=[8.0, 8.0, 8.0],
        grid_shape=(6, 6, 6),
        electron_count=1.0,
        centers=[[4.0, 4.0, 4.0]],
        amplitudes=-2.5,
        widths=0.9,
    )
    config = SCFConfig(max_iterations=8, mixing=0.4, solver="auto", seed=13)
    return system, config


def toy_closed_shell_dft_example():
    """Return a compact two-electron closed-shell Gaussian-well DFT toy system."""

    system = DFTSystem(
        cell=[8.0, 8.0, 8.0],
        grid_shape=(8, 8, 8),
        electron_count=2.0,
        centers=[[4.0, 4.0, 4.0]],
        amplitudes=-3.0,
        widths=0.9,
    )
    config = SCFConfig(
        max_iterations=12,
        mixing=0.4,
        solver="auto",
        max_dense_grid_points=256,
        seed=11,
    )
    return system, config


def toy_two_center_dft_example():
    """Return a two-center closed-shell Gaussian DFT toy system."""

    system = DFTSystem(
        cell=[10.0, 8.0, 8.0],
        grid_shape=(8, 6, 6),
        electron_count=2.0,
        centers=[[4.0, 4.0, 4.0], [6.0, 4.0, 4.0]],
        amplitudes=[-2.2, -2.2],
        widths=[0.8, 0.8],
    )
    config = SCFConfig(max_iterations=10, mixing=0.35, mixer="diis", solver="auto", seed=19)
    return system, config


def toy_periodic_cluster_dft_example():
    """Return a small periodic three-center Gaussian DFT toy system."""

    system = DFTSystem(
        cell=[10.0, 10.0, 10.0],
        grid_shape=(8, 8, 8),
        electron_count=4.0,
        centers=[[4.0, 4.0, 5.0], [6.0, 4.0, 5.0], [5.0, 6.0, 5.0]],
        amplitudes=[-2.0, -2.0, -1.5],
        widths=[0.85, 0.85, 0.95],
    )
    config = SCFConfig(max_iterations=12, mixing=0.3, mixer="diis", solver="auto", seed=23)
    return system, config
