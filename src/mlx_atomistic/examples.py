"""Small programmatic examples for molecular mechanics workflows."""

from __future__ import annotations

from math import pi

import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.forcefields import (
    CoulombPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    PeriodicDihedralPotential,
)
from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.md import LennardJonesPotential
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
