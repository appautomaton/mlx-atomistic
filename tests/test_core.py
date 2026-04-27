import numpy as np

from mlx_atomistic.core import Atoms, Cell


def test_minimum_image_orthorhombic_cell():
    cell = Cell.orthorhombic([10.0, 20.0, 30.0])
    displacement = cell.minimum_image(np.array([[6.0, -11.0, 14.0]], dtype=np.float32))

    np.testing.assert_allclose(np.array(displacement), [[-4.0, 9.0, 14.0]], atol=1e-6)


def test_cell_wrap():
    cell = Cell.cubic(5.0)
    wrapped = cell.wrap(np.array([[5.5, -0.5, 2.0]], dtype=np.float32))

    np.testing.assert_allclose(np.array(wrapped), [[0.5, 4.5, 2.0]], atol=1e-6)


def test_atoms_from_sequences_defaults_masses():
    atoms = Atoms.from_sequences(["Ar", "Ar"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert atoms.count == 2
    np.testing.assert_allclose(np.array(atoms.masses), [1.0, 1.0])
