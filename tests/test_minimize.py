import numpy as np
import pytest

from mlx_atomistic.core import Cell
from mlx_atomistic.md import LennardJonesPotential
from mlx_atomistic.minimize import minimize_energy
from mlx_atomistic.neighbors import NeighborListManager


class HarmonicWell:
    name = "harmonic_well"
    supports_virial = True

    def __init__(self, target):
        self.target = np.asarray(target, dtype=np.float32)

    def energy_forces(self, positions, cell=None, pairs=None):
        import mlx.core as mx

        target = mx.array(self.target, dtype=positions.dtype)
        displacement = positions - target
        return 0.5 * mx.sum(displacement * displacement), -displacement


@pytest.mark.parametrize("method", ["l-bfgs", "conjugate_gradient"])
def test_selectable_minimizers_reduce_energy_and_force_norm(method):
    positions = np.array([[2.0, -1.0, 0.5], [-0.5, 1.5, -2.0]], dtype=np.float32)
    target = np.zeros_like(positions)

    result = minimize_energy(
        positions,
        HarmonicWell(target),
        method=method,
        max_steps=50,
        force_tolerance=1e-5,
    )

    assert result.method in {"l-bfgs", "conjugate-gradient"}
    assert result.steps <= 50
    assert np.asarray(result.energy_history)[-1] <= np.asarray(result.energy_history)[0]
    assert np.asarray(result.max_force_history)[-1] <= np.asarray(result.max_force_history)[0]
    assert np.asarray(result.energy) < 1e-8
    np.testing.assert_allclose(np.asarray(result.positions), target, atol=1e-4)


def test_l_bfgs_handles_neighbor_list_backed_force_terms():
    positions = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32)
    cell = Cell.cubic(8.0)
    term = LennardJonesPotential(cutoff=3.0)
    neighbor_manager = NeighborListManager(cell, cutoff=3.0, skin=0.5)

    result = minimize_energy(
        positions,
        term,
        cell=cell,
        method="l-bfgs",
        max_steps=50,
        force_tolerance=1e-4,
        neighbor_manager=neighbor_manager,
    )

    assert np.asarray(result.energy) < 0.0
    assert result.steps <= 50
    assert np.isfinite(np.asarray(result.positions)).all()


def test_minimizer_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown minimization method"):
        minimize_energy([[0.0, 0.0, 0.0]], HarmonicWell([[0.0, 0.0, 0.0]]), method="magic")
