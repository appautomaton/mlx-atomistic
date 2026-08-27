"""Stable identities for DFT pseudopotential data."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from mlx_atomistic.dft.pseudopotentials import PseudopotentialData


def _pseudopotential_fingerprint(pseudopotential: PseudopotentialData) -> str:
    """Return the canonical fingerprint for one parsed pseudopotential."""

    digest = sha256()
    if str(pseudopotential.format) == "upf":
        digest.update(b"mlx-atomistic.upf-nonlocal.v1\0")
        digest.update(pseudopotential.element.encode("utf-8"))
        digest.update(str(pseudopotential.format).encode("utf-8"))
        digest.update(
            np.asarray([pseudopotential.valence_charge], dtype=np.float64).tobytes()
        )
        local = pseudopotential.local_grid
        if local is None:
            raise ValueError("UPF pseudopotential identity requires a local radial grid")
        digest.update(np.asarray(local.radii, dtype=np.float64).tobytes())
        digest.update(np.asarray(local.values, dtype=np.float64).tobytes())
        weights = local.integration_weights
        if weights is None:
            raise ValueError("UPF pseudopotential identity requires radial weights")
        digest.update(np.asarray(weights, dtype=np.float64).tobytes())
        for projector in pseudopotential.nonlocal_projectors:
            digest.update(
                np.asarray([projector.angular_momentum], dtype=np.int64).tobytes()
            )
            digest.update(np.asarray(projector.values, dtype=np.float64).tobytes())
            has_cutoff = projector.cutoff_radius is not None
            digest.update(np.asarray([has_cutoff], dtype=np.uint8).tobytes())
            digest.update(
                np.asarray(
                    [
                        0.0 if not has_cutoff else projector.cutoff_radius,
                        projector.coupling,
                    ],
                    dtype=np.float64,
                ).tobytes()
            )
            digest.update(np.asarray(projector.coefficients, dtype=np.float64).tobytes())
        digest.update(
            np.asarray(
                pseudopotential.nonlocal_coupling_matrix,
                dtype=np.float64,
            ).tobytes()
        )
        metadata = pseudopotential.metadata or {}
        for name in (
            "version",
            "pseudo_type",
            "relativistic",
            "functional",
            "is_ultrasoft",
            "is_paw",
            "has_so",
            "core_correction",
        ):
            digest.update(name.encode("ascii"))
            digest.update(repr(metadata.get(name)).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

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
