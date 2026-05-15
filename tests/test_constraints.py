import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate_nve


def test_distance_constraints_remain_stable_in_short_long_run():
    positions = np.array([[1.0, 1.0, 1.0], [2.25, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]], dtype=np.float32)
    constraints = DistanceConstraints(
        np.asarray([[0, 1]], dtype=np.int32),
        distances=np.asarray([1.25], dtype=np.float32),
        max_iterations=8,
    )

    result = simulate_nve(
        positions,
        velocities,
        cell=Cell.cubic(8.0),
        force_terms=LennardJonesPotential(cutoff=3.0),
        constraints=constraints,
        config=SimulationConfig(
            dt=0.001,
            steps=100,
            sample_interval=25,
            diagnostic_interval=10,
        ),
    )

    assert float(np.max(np.asarray(result.constraint_max_error))) < 1e-4
    assert result.final_state.step == 100
