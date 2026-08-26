from __future__ import annotations

from math import pi

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicGTHNonlocalOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
)
from mlx_atomistic.dft.periodic_gth import _real_spherical_harmonics


def _d_channel_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="Fe",
        format=PseudopotentialFormat.GTH,
        valence_charge=8.0,
        gth_rloc=0.6,
        gth_coefficients=(0.16,),
        gth_channels=(GTHProjectorChannel(2, 0.31, ((-9.15,),)),),
    )


def test_real_d_harmonics_follow_qe_order_and_addition_theorem():
    vectors = mx.array(
        np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, -2.0, 3.0),
            ),
            dtype=np.float32,
        )
    )
    radii = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
    harmonics = np.asarray(mx.stack(_real_spherical_harmonics(2, vectors, radii), axis=-1))

    np.testing.assert_allclose(
        np.sum(harmonics * harmonics, axis=-1),
        5.0 / (4.0 * pi),
        atol=2.0e-7,
    )
    assert harmonics[2, 0] == pytest.approx(np.sqrt(5.0 / (4.0 * pi)), abs=2.0e-7)
    assert harmonics[0, 3] == pytest.approx(np.sqrt(15.0 / (16.0 * pi)), abs=2.0e-7)
    assert harmonics[1, 3] == pytest.approx(-np.sqrt(15.0 / (16.0 * pi)), abs=2.0e-7)


def test_periodic_d_projector_is_hermitian_and_cell_translation_invariant():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, -0.25, 0.25))
    rng = np.random.default_rng(91)
    left = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(
                np.complex64
            )
        )
    )
    right = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(
                np.complex64
            )
        )
    )
    pseudo = _d_channel_gth()
    first = PeriodicGTHNonlocalOperator(pseudo, basis, ((1.0, 2.0, 3.0),))
    shifted = PeriodicGTHNonlocalOperator(pseudo, basis, ((9.0, 2.0, 3.0),))

    left_right = np.asarray(mx.sum(mx.conjugate(left) * first.apply(right))).item()
    right_left = np.asarray(mx.sum(mx.conjugate(right) * first.apply(left))).item()
    first_energy = float(first.energy(left, occupations=[1.0]))
    shifted_energy = float(shifted.energy(left, occupations=[1.0]))

    assert left_right == pytest.approx(np.conjugate(right_left), abs=2.0e-5)
    assert shifted_energy == pytest.approx(first_energy, abs=2.0e-5)
    assert first.to_dict()["angular_projector_count_per_ion"] == 5


def test_periodic_d_projector_force_matches_fixed_orbital_energy_derivative():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, 0.125, -0.25))
    position = np.asarray(((1.0, 2.0, 3.0),))
    rng = np.random.default_rng(92)
    orbital = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(
                np.complex64
            )
        )
    )
    operator = PeriodicGTHNonlocalOperator(_d_channel_gth(), basis, position)
    observed = np.asarray(operator.forces(orbital, occupations=[1.0]))
    reference = np.zeros_like(position)
    displacement = 2.0e-3
    for axis in range(3):
        plus = position.copy()
        minus = position.copy()
        plus[0, axis] += displacement
        minus[0, axis] -= displacement
        energy_plus = float(
            PeriodicGTHNonlocalOperator(
                _d_channel_gth(),
                basis,
                plus,
            ).energy(orbital, occupations=[1.0])
        )
        energy_minus = float(
            PeriodicGTHNonlocalOperator(
                _d_channel_gth(),
                basis,
                minus,
            ).energy(orbital, occupations=[1.0])
        )
        reference[0, axis] = -(energy_plus - energy_minus) / (2.0 * displacement)

    np.testing.assert_allclose(observed, reference, atol=8.0e-5, rtol=5.0e-4)
