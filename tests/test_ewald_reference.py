import numpy as np
import pytest

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.nonbonded import (
    EwaldReferenceConfig,
    ewald_reference_coulomb_energy,
    ewald_reference_coulomb_energy_forces,
)


def _neutral_positions():
    return as_mx_array(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.2, 1.1],
            [2.0, 3.0, 5.0],
        ]
    )


def _neutral_charges():
    return as_mx_array([1.0, -0.5, -0.5])


def test_ewald_zero_charges_produce_zero_energy():
    positions = _neutral_positions()
    charges = as_mx_array([0.0, 0.0, 0.0])
    energy, components = ewald_reference_coulomb_energy(positions, charges, Cell.cubic(12.0))

    np.testing.assert_allclose(np.asarray(energy), 0.0, atol=1e-7)
    for value in components.values():
        np.testing.assert_allclose(np.asarray(value), 0.0, atol=1e-7)


def test_ewald_rejects_non_neutral_system_by_default():
    positions = _neutral_positions()
    charges = as_mx_array([1.0, -0.25, -0.5])

    with pytest.raises(ValueError, match="requires a neutral system"):
        ewald_reference_coulomb_energy(positions, charges, Cell.cubic(12.0))


def test_ewald_energy_is_translation_and_wrapping_invariant():
    cell = Cell.cubic(12.0)
    positions = _neutral_positions()
    charges = _neutral_charges()
    config = EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4)

    reference, _ = ewald_reference_coulomb_energy(positions, charges, cell, config=config)
    translated = cell.wrap(positions + as_mx_array([13.0, -11.5, 24.0]))
    shifted, _ = ewald_reference_coulomb_energy(translated, charges, cell, config=config)

    np.testing.assert_allclose(np.asarray(shifted), np.asarray(reference), atol=2e-6)


def test_ewald_energy_converges_as_cutoffs_tighten():
    cell = Cell.cubic(12.0)
    positions = _neutral_positions()
    charges = _neutral_charges()
    loose, _ = ewald_reference_coulomb_energy(
        positions,
        charges,
        cell,
        config=EwaldReferenceConfig(alpha=0.25, real_cutoff=4.0, reciprocal_cutoff=2),
    )
    medium, _ = ewald_reference_coulomb_energy(
        positions,
        charges,
        cell,
        config=EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4),
    )
    tight, _ = ewald_reference_coulomb_energy(
        positions,
        charges,
        cell,
        config=EwaldReferenceConfig(alpha=0.25, real_cutoff=6.0, reciprocal_cutoff=5),
    )

    loose_error = abs(float(np.asarray(loose - tight)))
    medium_error = abs(float(np.asarray(medium - tight)))
    assert medium_error < loose_error
    assert medium_error < 1e-5


def test_ewald_forces_match_finite_difference():
    cell = Cell.cubic(12.0)
    positions = np.asarray(_neutral_positions(), dtype=np.float32)
    charges = _neutral_charges()
    config = EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4)

    _, forces, _ = ewald_reference_coulomb_energy_forces(
        positions,
        charges,
        cell,
        config=config,
    )

    finite_difference = np.zeros_like(positions)
    epsilon = 1e-3
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus, _ = ewald_reference_coulomb_energy(plus, charges, cell, config=config)
            e_minus, _ = ewald_reference_coulomb_energy(minus, charges, cell, config=config)
            finite_difference[atom, axis] = -float(np.asarray(e_plus - e_minus)) / (
                2.0 * epsilon
            )

    np.testing.assert_allclose(np.asarray(forces), finite_difference, atol=2e-3)


def test_ewald_forces_have_near_zero_net_force():
    energy, forces, components = ewald_reference_coulomb_energy_forces(
        _neutral_positions(),
        _neutral_charges(),
        Cell.cubic(12.0),
        config=EwaldReferenceConfig(alpha=0.25, real_cutoff=5.0, reciprocal_cutoff=4),
    )

    assert np.isfinite(float(np.asarray(energy)))
    assert set(components) == {"coulomb_real", "coulomb_reciprocal", "coulomb_self"}
    np.testing.assert_allclose(np.asarray(forces).sum(axis=0), np.zeros(3), atol=1e-6)


def test_ewald_single_charge_has_no_self_force_when_neutrality_override_is_explicit():
    _, forces, _ = ewald_reference_coulomb_energy_forces(
        as_mx_array([[1.2, 2.3, 3.4]]),
        as_mx_array([1.0]),
        Cell.cubic(12.0),
        config=EwaldReferenceConfig(
            alpha=0.25,
            real_cutoff=5.0,
            reciprocal_cutoff=4,
            charge_tolerance=2.0,
        ),
    )

    np.testing.assert_allclose(np.asarray(forces), np.zeros((1, 3)), atol=1e-6)
