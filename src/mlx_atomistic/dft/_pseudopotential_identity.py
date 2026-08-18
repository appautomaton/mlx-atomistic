"""Stable identities for DFT pseudopotential data."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from mlx_atomistic.dft.pseudopotentials import PseudopotentialData


def _pseudopotential_fingerprint(pseudopotential: PseudopotentialData) -> str:
    """Return the canonical fingerprint for one GTH pseudopotential."""

    digest = sha256()
    digest.update(b"mlx-atomistic.gth-nonlocal.v1\0")
    digest.update(pseudopotential.element.encode("utf-8"))
    digest.update(str(pseudopotential.format).encode("utf-8"))
    digest.update(
        np.asarray(
            [
                pseudopotential.valence_charge,
                float(pseudopotential.gth_rloc),
                *pseudopotential.gth_coefficients,
            ],
            dtype=np.float64,
        ).tobytes()
    )
    for channel in pseudopotential.gth_channels:
        digest.update(np.asarray([channel.angular_momentum], dtype=np.int64).tobytes())
        digest.update(np.asarray([channel.radius], dtype=np.float64).tobytes())
        digest.update(np.asarray(channel.coupling_matrix, dtype=np.float64).tobytes())
    return digest.hexdigest()
