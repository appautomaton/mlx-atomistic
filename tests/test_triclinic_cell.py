import numpy as np
import pytest

from mlx_atomistic.artifacts import PreparedMLXArtifact
from mlx_atomistic.core import Cell


def test_triclinic_cell_wraps_via_fractional_coordinates():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    fractional = np.array([[1.2, -0.25, 0.5]], dtype=np.float32)
    positions = fractional @ matrix

    wrapped = cell.wrap(positions)

    np.testing.assert_allclose(
        np.asarray(wrapped),
        np.array([[0.2, 0.75, 0.5]], dtype=np.float32) @ matrix,
        atol=1e-6,
    )


def test_triclinic_cell_minimum_image_uses_nearest_fractional_image():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    displacement = np.array([[0.6, -0.55, 0.49]], dtype=np.float32) @ matrix

    minimum = cell.minimum_image(displacement)

    np.testing.assert_allclose(
        np.asarray(minimum),
        np.array([[-0.4, 0.45, 0.49]], dtype=np.float32) @ matrix,
        atol=1e-6,
    )


def test_triclinic_cell_round_trips_fractional_and_cartesian_coordinates():
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    cell = Cell.triclinic(matrix)
    fractional = np.array([[0.2, 0.75, 0.5]], dtype=np.float32)

    cartesian = cell.cartesian_coordinates(fractional)

    np.testing.assert_allclose(
        np.asarray(cell.fractional_coordinates(cartesian)),
        fractional,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(cell.volume), 24.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(cell.lengths), np.linalg.norm(matrix, axis=1))


def test_cell_rejects_invalid_shapes_and_singular_matrices():
    with pytest.raises(ValueError, match=r"shape \(3,\) or \(3, 3\)"):
        Cell([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        Cell.orthorhombic([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="positive non-singular"):
        Cell.triclinic(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )


def test_prepared_artifact_cell_preserves_triclinic_matrix(tmp_path):
    matrix = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    artifact = PreparedMLXArtifact(
        base_dir=tmp_path,
        metadata={},
        arrays={
            "positions": np.zeros((2, 3), dtype=np.float32),
            "cell_matrix": matrix,
        },
        unit_system=None,
    )

    assert artifact.cell is not None
    np.testing.assert_allclose(np.asarray(artifact.cell.matrix), matrix)
