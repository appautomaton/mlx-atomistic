from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicGTHNonlocalOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    gth_local_reciprocal_coefficients,
)
from mlx_atomistic.dft.periodic_gth import _gth_radial

# These values are independent numerical oracles evaluated from Quantum
# ESPRESSO's upflib/gth.f90 vloc_gth and mk_ffnl_gth formulas. Keeping the
# expected values literal prevents this test from reproducing the product
# implementation as its own reference.
_QE_RADIAL_ORACLES = (
    (0, 0, 0.42, (1.0, 0.9879980252214315, 0.8712620380637395)),
    (0, 1, 0.42, (1.5491933384829666, 1.518279058004542, 1.225744756829006)),
    (0, 2, 0.42, (1.9518001458970664, 1.8974039100424047, 1.3966696177222717)),
    (1, 0, 0.51, (0.0, 0.20985000556619063, 0.5889784296309902)),
    (1, 1, 0.51, (0.0, 0.352185160471394, 0.9146352762065771)),
    (1, 2, 0.51, (0.0, 0.4920069043499521, 1.1796919782667894)),
    (2, 0, 0.63, (0.0, 0.03440004342534713, 0.29587561785430677)),
    (2, 1, 0.63, (0.0, 0.0602049924684082, 0.47564062514140676)),
)


def test_gth_radial_projectors_match_quantum_espresso_source_oracles():
    q = mx.array((0.0, 0.37, 1.25), dtype=mx.float32)

    for angular_momentum, projector_index, radius, expected in _QE_RADIAL_ORACLES:
        count = projector_index + 1
        channel = GTHProjectorChannel(
            angular_momentum,
            radius,
            np.eye(count, dtype=np.float64),
        )
        observed = np.asarray(_gth_radial(channel, projector_index, q))

        np.testing.assert_allclose(observed, expected, rtol=2.0e-7, atol=2.0e-7)


def test_gth_local_transform_matches_quantum_espresso_source_oracles():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    oxygen = PseudopotentialData(
        element="O",
        format=PseudopotentialFormat.GTH,
        valence_charge=6.0,
        gth_rloc=0.24455430,
        gth_coefficients=(-16.66721480, 2.48731132),
    )

    observed = np.asarray(
        gth_local_reciprocal_coefficients(
            oxygen,
            basis,
            ((1.1, 2.2, 3.3),),
        )
    )

    np.testing.assert_allclose(
        observed[0, 0, 0],
        0.00026209065053374534 + 0.0j,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        observed[1, 0, 0],
        -0.1548774645904494 + 0.18133821221633792j,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        observed[1, -1, 0],
        -0.0773583086997834 - 0.0905749421763413j,
        rtol=2.0e-7,
        atol=2.0e-8,
    )


def test_gth_nonlocal_projectors_match_quantum_espresso_source_oracle():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    coupling = (
        (2.1, -0.3, 0.2),
        (-0.3, 1.7, -0.4),
        (0.2, -0.4, 1.2),
    )
    channel = GTHProjectorChannel(1, 0.51, coupling)
    pseudo = PseudopotentialData(
        element="X",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.4,
        gth_coefficients=(-1.0,),
        gth_channels=(channel,),
    )
    position = np.asarray((0.4, -0.2, 0.7), dtype=np.float64)
    operator = PeriodicGTHNonlocalOperator(pseudo, basis, (position,))
    vectors = mx.array(((0.0, 0.0, 0.0), (0.3, -0.7, 1.1)))
    q = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
    # The second value is QE's normalized real p_z harmonic for this vector.
    harmonic = mx.array((0.0, 0.4017185301832866))

    try:
        observed = np.asarray(
            operator._projector_group(position, channel, harmonic, vectors, q)
        )
        expected = np.asarray(
            (
                (0.0j, -0.0578946799103185 - 0.0347664847587711j),
                (0.0j, -0.08874757001200705 - 0.05329403401106808j),
                (0.0j, -0.11290381447094572 - 0.06780016317719822j),
            ),
            dtype=np.complex128,
        )
        np.testing.assert_allclose(observed, expected, rtol=3.0e-7, atol=3.0e-8)

        expected_coupling = np.kron(np.eye(3), np.asarray(coupling))
        np.testing.assert_allclose(
            np.asarray(operator._flattened_coupling),
            expected_coupling,
            rtol=0.0,
            atol=1.0e-7,
        )
    finally:
        operator.close()
