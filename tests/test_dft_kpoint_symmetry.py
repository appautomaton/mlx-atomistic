from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from mlx_atomistic.dft import (
    GammaCenteredGrid,
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
    build_time_reversal_ownership,
    cubic_reciprocal_symmetry_operations,
    reciprocal_symmetry_operations_for_cell,
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
    assert reduced.point_group_symmetry_reduced is True
    assert reduced_value == pytest.approx(full_value, abs=2.0e-15)
    assert sum(point.weight for point in reduced.points) == pytest.approx(1.0)
    assert len(build_time_reversal_ownership(full).owned_indices) == 36


def test_symmetry_reduction_metadata_round_trips_and_partitions_full_mesh():
    full = GammaCenteredGrid((4, 4, 4))
    reduced = reduce_kpoint_mesh_by_symmetry(
        full,
        cubic_reciprocal_symmetry_operations(),
    )
    payload = reduced.to_dict()
    restored = KPointMesh.from_dict(payload)

    assert KPointMesh.from_dict(full.to_dict()).point_group_symmetry_reduced is False
    assert restored.point_group_symmetry_reduced is True
    assert restored.to_dict() == payload
    symmetry = payload["point_group_symmetry"]
    assert symmetry["full_point_count"] == len(full.points)
    members = [member for orbit in symmetry["orbits"] for member in orbit["members"]]
    assert sorted(member["full_index"] for member in members) == list(range(len(full.points)))
    for representative, orbit in zip(restored.points, symmetry["orbits"], strict=True):
        for member in orbit["members"]:
            operation = np.asarray(member["reciprocal_operation"])
            difference = operation @ representative.vector - member["reduced_kpoint"]
            difference -= np.rint(difference)
            np.testing.assert_allclose(difference, 0.0, atol=1.0e-12)


def test_symmetry_reduction_metadata_fails_closed_when_corrupted():
    payload = reduce_kpoint_mesh_by_symmetry(
        GammaCenteredGrid((2, 2, 2)),
        cubic_reciprocal_symmetry_operations(),
    ).to_dict()

    duplicated_index = deepcopy(payload)
    orbits = duplicated_index["point_group_symmetry"]["orbits"]
    orbit = next(value for value in orbits if len(value["members"]) > 1)
    duplicate = next(
        member
        for member in orbit["members"]
        if member["full_index"] != orbit["representative_full_index"]
    )
    duplicate["full_index"] = orbit["representative_full_index"]
    with pytest.raises(ValueError, match="partition"):
        KPointMesh.from_dict(duplicated_index)

    invalid_operation = deepcopy(payload)
    invalid_operation["point_group_symmetry"]["orbits"][0]["members"][0][
        "reciprocal_operation"
    ] = ((2, 0, 0), (0, 1, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="unimodular"):
        KPointMesh.from_dict(invalid_operation)

    invalid_count = deepcopy(payload)
    invalid_count["point_group_symmetry"]["full_point_count"] = 8.0
    with pytest.raises(ValueError, match="full point count"):
        KPointMesh.from_dict(invalid_count)


def test_cubic_symmetries_transform_to_fcc_primitive_reciprocal_basis():
    cell = np.asarray(
        (
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        )
    )
    operations = reciprocal_symmetry_operations_for_cell(
        cell,
        cubic_reciprocal_symmetry_operations(),
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    metric = reciprocal @ reciprocal.T
    reduced = reduce_kpoint_mesh_by_symmetry(
        GammaCenteredGrid((4, 4, 4)),
        operations,
    )

    assert len(operations) == 48
    assert len(reduced.points) < 32
    assert sum(point.weight for point in reduced.points) == pytest.approx(1.0)
    for operation in operations:
        matrix = np.asarray(operation)
        np.testing.assert_allclose(matrix.T @ metric @ matrix, metric, atol=2.0e-13)


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
