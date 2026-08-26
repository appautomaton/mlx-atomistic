from __future__ import annotations

from math import pi, sqrt

import numpy as np

from mlx_atomistic.benchmarks.dft_hexagonal_silicon import (
    CUBIC_LATTICE_ANGSTROM,
    INTERNAL_COORDINATE,
    hexagonal_silicon_geometry,
)
from mlx_atomistic.benchmarks.dft_silicon import ANGSTROM_TO_BOHR


def test_hexagonal_silicon_geometry_locks_ideal_2h_contract():
    cell, positions = hexagonal_silicon_geometry()
    a = CUBIC_LATTICE_ANGSTROM / sqrt(2.0) * ANGSTROM_TO_BOHR
    c = sqrt(8.0 / 3.0) * a
    reciprocal = 2.0 * pi * np.linalg.inv(cell).T
    fractional = positions @ np.linalg.inv(cell)

    assert positions.shape == (4, 3)
    assert np.isclose(np.linalg.norm(cell[0]), a)
    assert np.isclose(np.linalg.norm(cell[1]), a)
    assert np.isclose(np.linalg.norm(cell[2]), c)
    assert np.isclose(np.dot(cell[0], cell[1]) / (a * a), -0.5)
    assert np.isclose(fractional[0, 2], INTERNAL_COORDINATE)
    np.testing.assert_allclose(cell @ reciprocal.T, 2.0 * pi * np.eye(3), atol=1e-12)
