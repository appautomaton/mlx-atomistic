import numpy as np

from mlx_atomistic.forcefields import CoulombPotential, HarmonicBondPotential
from mlx_atomistic.md import SimulationConfig, simulate_nve
from mlx_atomistic.topology import Topology


def test_simulate_nve_reports_per_term_energy_decomposition():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    topology = Topology.from_sequences(n_atoms=2, partial_charges=[1.0, -1.0])
    result = simulate_nve(
        positions,
        velocities,
        force_terms=[
            HarmonicBondPotential([(0, 1)], k=10.0, length=1.0),
            CoulombPotential(topology=topology),
        ],
        config=SimulationConfig(dt=0.0001, steps=2, sample_interval=2),
    )

    assert set(result.potential_energy_by_term) == {"bond", "coulomb"}
    reconstructed = sum(np.array(series) for series in result.potential_energy_by_term.values())
    np.testing.assert_allclose(reconstructed, np.array(result.potential_energy), atol=1e-6)


def test_bonded_toy_nve_keeps_total_energy_bounded():
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]], dtype=np.float32)
    result = simulate_nve(
        positions,
        velocities,
        force_terms=[HarmonicBondPotential([(0, 1)], k=10.0, length=1.0)],
        config=SimulationConfig(dt=0.0005, steps=50, sample_interval=25),
    )

    total_energy = np.array(result.total_energy)
    assert np.max(np.abs(total_energy - total_energy[0])) < 1e-5
