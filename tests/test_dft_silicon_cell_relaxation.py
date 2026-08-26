from __future__ import annotations

import numpy as np

from mlx_atomistic.benchmarks.dft_hexagonal_silicon import (
    hexagonal_silicon_geometry,
)
from mlx_atomistic.benchmarks.dft_silicon_cell_relaxation import (
    INITIAL_LINEAR_SCALE,
    STRESS_TOLERANCE_HARTREE_PER_BOHR3,
    _cell_config,
    _initial_geometry,
)


def test_silicon_cell_protocol_starts_from_a_scaled_source_cell():
    accepted_cell, accepted_positions = hexagonal_silicon_geometry()
    initial_cell, initial_positions = _initial_geometry()

    np.testing.assert_allclose(initial_cell, INITIAL_LINEAR_SCALE * accepted_cell)
    np.testing.assert_allclose(
        initial_positions @ np.linalg.inv(initial_cell),
        accepted_positions @ np.linalg.inv(accepted_cell),
        atol=1.0e-12,
    )


def test_silicon_cell_protocol_locks_topology_and_convergence_controls():
    config = _cell_config()

    assert config.relaxation_mode == "cell"
    assert config.stress_tolerance == STRESS_TOLERANCE_HARTREE_PER_BOHR3
    assert config.stress_config.mode == "isotropic"
    assert config.stress_config.require_fixed_basis_topology
    assert config.stress_config.strain_step == 1.0e-3
    assert config.stress_config.electronic_response == "frozen_variational"
