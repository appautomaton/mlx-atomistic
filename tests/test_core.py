import numpy as np

from mlx_atomistic.core import Atoms, Cell


def test_wrap_is_exact_lattice_translation():
    # Wrapping must return the input modulo the box -- x minus an integer number
    # of box lengths -- so it performs no spurious work in periodic MD. The old
    # fractional round-trip ((x/L - floor(x/L)) * L) nudged boundary atoms by
    # ~1e-2 in float32, injecting energy every step (only visible over long runs).
    length = 8.5499
    cell = Cell.cubic(length)
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2.0 * length, 3.0 * length, size=(200, 3)).astype(np.float32)
    wrapped = np.array(cell.wrap(positions), dtype=np.float64)

    assert wrapped.min() >= -1e-4
    assert wrapped.max() <= length + 1e-4
    multiples = (positions.astype(np.float64) - wrapped) / length
    np.testing.assert_allclose(multiples, np.round(multiples), atol=1e-4)
    # Wrapping an already-wrapped position is a no-op.
    rewrapped = np.array(cell.wrap(wrapped.astype(np.float32)), dtype=np.float64)
    np.testing.assert_allclose(rewrapped, wrapped, atol=1e-5)


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
