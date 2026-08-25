from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.dft import (
    GammaCenteredGrid,
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
    build_time_reversal_ownership,
    cubic_reciprocal_symmetry_operations,
    reduce_kpoint_mesh_by_symmetry,
)


def test_gamma_centered_grid_is_distinct_from_even_half_shifted_mesh():
    gamma = GammaCenteredGrid((4, 2, 1))
    shifted = MonkhorstPackGrid((4, 2, 1))

    assert len(gamma.points) == 8
    assert sum(point.weight for point in gamma.points) == pytest.approx(1.0)
    assert gamma.points[0].vector == (0.0, 0.0, 0.0)
    assert shifted.points[0].vector != gamma.points[0].vector
    assert {point.vector[0] for point in gamma.points} == {0.0, 0.25, 0.5, -0.25}


def test_cubic_reduction_preserves_invariant_quadrature_and_time_reversal():
    full = GammaCenteredGrid((4, 4, 4))
    operations = cubic_reciprocal_symmetry_operations()
    reduced = reduce_kpoint_mesh_by_symmetry(full, operations)

    def invariant(point):
        values = np.asarray(point.vector, dtype=np.float64)
        return float(np.sum(np.cos(2.0 * np.pi * values) ** 2))

    full_value = sum(point.weight * invariant(point) for point in full.points)
    reduced_value = sum(point.weight * invariant(point) for point in reduced.points)

    assert len(operations) == 48
    assert len(reduced.points) == 10
    assert reduced_value == pytest.approx(full_value, abs=2.0e-15)
    assert sum(point.weight for point in reduced.points) == pytest.approx(1.0)
    assert len(build_time_reversal_ownership(full).owned_indices) == 36


def test_symmetry_reduction_fails_closed_on_invalid_operations_and_meshes():
    mesh = GammaCenteredGrid((4, 2, 2))
    with pytest.raises(ValueError, match="integers"):
        reduce_kpoint_mesh_by_symmetry(mesh, (((1.0, 0.1, 0.0), (0, 1, 0), (0, 0, 1)),))
    with pytest.raises(ValueError, match="unimodular"):
        reduce_kpoint_mesh_by_symmetry(mesh, (((2, 0, 0), (0, 1, 0), (0, 0, 1)),))
    with pytest.raises(ValueError, match="not closed"):
        reduce_kpoint_mesh_by_symmetry(mesh, (((0, 1, 0), (1, 0, 0), (0, 0, 1)),))

    unequal = KPointMesh(
        (
            KPoint((0.25, 0.0, 0.0), weight=0.4, coordinate_system="reduced"),
            KPoint((-0.25, 0.0, 0.0), weight=0.6, coordinate_system="reduced"),
        )
    )
    inversion = (((-1, 0, 0), (0, -1, 0), (0, 0, -1)),)
    with pytest.raises(ValueError, match="equal input weights"):
        reduce_kpoint_mesh_by_symmetry(unequal, inversion)
