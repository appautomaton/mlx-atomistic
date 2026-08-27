"""Periodic local operators for numerical UPF pseudopotentials."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import mlx.core as mx
import numpy as np
from scipy.special import erf

from mlx_atomistic.dft._periodic_pseudopotential import (
    _periodic_positions,
    _periodic_pseudopotentials,
    _periodic_structure_factor,
)
from mlx_atomistic.dft._pseudopotential_identity import _pseudopotential_fingerprint
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import (
    PseudopotentialData,
    PseudopotentialFormat,
    RadialGrid,
)

_RADIAL_TRANSFORM_CHUNK_SIZE = 1024


def _simpson_radial(values: np.ndarray, grid: RadialGrid) -> np.ndarray:
    """Integrate along the final axis with Quantum ESPRESSO's Simpson rule."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[-1] != grid.size:
        raise ValueError("radial integrand size must match its grid")
    if grid.size < 3:
        raise ValueError("periodic UPF transforms require at least three radial samples")
    weights = grid.integration_weights
    if weights is None:
        raise ValueError("periodic UPF transforms require PP_RAB weights")
    coefficients = np.full(grid.size, 2.0 / 3.0, dtype=np.float64)
    coefficients[1:-1:2] = 4.0 / 3.0
    coefficients[0] = 1.0 / 3.0
    coefficients[-1] = 1.0 / 3.0
    if grid.size % 2 == 0:
        coefficients[-3] -= 1.0 / 12.0
        coefficients[-2] += 1.0 / 3.0
        coefficients[-1] += 1.0 / 12.0
    return np.sum(samples * (coefficients * weights), axis=-1, dtype=np.float64)


def _upf_local_radial_transform(
    pseudopotential: PseudopotentialData,
    q: np.ndarray,
    *,
    volume: float,
) -> np.ndarray:
    grid = pseudopotential.local_grid
    if grid is None:
        raise ValueError("UPF local transform requires a radial local potential")
    q_values = np.asarray(q, dtype=np.float64)
    if q_values.ndim != 1 or not np.isfinite(q_values).all() or np.any(q_values < 0.0):
        raise ValueError("UPF local transform magnitudes must be finite and non-negative")
    radius = grid.radii
    local = grid.values
    charge = float(pseudopotential.valence_charge)
    compensated = radius * local + charge * erf(radius)
    transformed = np.empty_like(q_values)
    zero = q_values <= 1.0e-14
    if np.any(zero):
        transformed[zero] = (
            4.0
            * pi
            / volume
            * _simpson_radial(radius * (radius * local + charge), grid)
        )
    positive_indices = np.flatnonzero(~zero)
    for start in range(0, positive_indices.size, _RADIAL_TRANSFORM_CHUNK_SIZE):
        indices = positive_indices[start : start + _RADIAL_TRANSFORM_CHUNK_SIZE]
        selected = q_values[indices]
        qr = selected[:, None] * radius[None, :]
        short_range = _simpson_radial(
            compensated[None, :] * np.sin(qr) / selected[:, None],
            grid,
        )
        coulomb = charge * np.exp(-(selected**2) / 4.0) / selected**2
        transformed[indices] = 4.0 * pi / volume * (short_range - coulomb)
    return transformed


def _single_upf_local_coefficients(
    pseudopotential: PseudopotentialData,
    basis: PlaneWaveBasis,
    vectors: np.ndarray,
) -> np.ndarray:
    q = np.sqrt(np.sum(vectors * vectors, axis=-1))
    return _upf_local_radial_transform(
        pseudopotential,
        q.reshape(-1),
        volume=basis.volume,
    ).reshape(basis.grid.shape)


def upf_local_reciprocal_coefficients(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return periodic UPF local-potential Fourier coefficients.

    The numerical transform follows Quantum ESPRESSO's compensated ``vloc``
    convention. It removes ``erf(r) / r`` before radial integration, restores
    the analytic reciprocal-space Coulomb tail, and uses the finite ``G=0``
    alpha term.

    Args:
        pseudopotential: One shared or one-per-ion parsed UPF pseudopotential.
        basis: Plane-wave basis supplying reciprocal vectors and volume.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Complex local-potential coefficients with shape ``basis.grid.shape``.
    """

    centers = _periodic_positions(positions)
    per_ion = _periodic_pseudopotentials(
        pseudopotential,
        int(centers.shape[0]),
        expected_format=PseudopotentialFormat.UPF,
    )
    vectors = np.asarray(basis.reciprocal_vectors, dtype=np.float64)
    grouped: dict[str, tuple[PseudopotentialData, list[np.ndarray]]] = {}
    for pseudo, center in zip(per_ion, centers, strict=True):
        fingerprint = _pseudopotential_fingerprint(pseudo)
        if fingerprint not in grouped:
            grouped[fingerprint] = (pseudo, [])
        grouped[fingerprint][1].append(center)
    values = np.zeros(basis.grid.shape, dtype=np.complex128)
    for pseudo, species_centers in grouped.values():
        single = _single_upf_local_coefficients(
            pseudo,
            basis,
            vectors,
        )
        values += single * _periodic_structure_factor(
            vectors,
            np.asarray(species_centers, dtype=np.float64),
        )
    return mx.array(values.astype(np.complex64))


def upf_local_potential_grid(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return the real periodic UPF local potential on the FFT grid.

    Args:
        pseudopotential: One shared or one-per-ion parsed UPF pseudopotential.
        basis: Plane-wave basis supplying the FFT grid.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Real local potential with shape ``basis.grid.shape``.
    """

    coefficients = upf_local_reciprocal_coefficients(
        pseudopotential,
        basis,
        positions,
    )
    return mx.real(mx.fft.ifftn(coefficients) * basis.grid.size)


def periodic_upf_local_forces(
    density: mx.array,
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return analytic fixed-cell local-UPF Hellmann--Feynman forces.

    Args:
        density: Positive electron density on ``basis.grid``.
        pseudopotential: One shared or one-per-ion parsed UPF pseudopotential.
        basis: Plane-wave basis supplying reciprocal vectors and volume.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Local electron-ion forces with shape ``(n_ions, 3)`` in Hartree/bohr.
    """

    centers = _periodic_positions(positions)
    per_ion = _periodic_pseudopotentials(
        pseudopotential,
        int(centers.shape[0]),
        expected_format=PseudopotentialFormat.UPF,
    )
    density_array = mx.real(mx.array(density)).astype(mx.float32)
    if density_array.shape != basis.grid.shape:
        raise ValueError("density shape must match the periodic FFT grid")
    finite = mx.all(mx.isfinite(density_array))
    mx.eval(finite)
    if not bool(finite):
        raise ValueError("density must contain only finite values")
    density_reciprocal = mx.conjugate(mx.fft.fftn(density_array))
    vectors_np = np.asarray(basis.reciprocal_vectors, dtype=np.float64)
    vectors = mx.array(vectors_np.astype(np.float32))
    imaginary = mx.array(1j, dtype=mx.complex64)
    single_by_species: dict[str, np.ndarray] = {}
    forces = []
    for pseudo, center in zip(per_ion, centers, strict=True):
        fingerprint = _pseudopotential_fingerprint(pseudo)
        single = single_by_species.get(fingerprint)
        if single is None:
            single = _single_upf_local_coefficients(
                pseudo,
                basis,
                vectors_np,
            )
            single_by_species[fingerprint] = single
        phase = np.exp(
            -1j
            * np.einsum(
                "...d,d->...",
                vectors_np,
                center,
                optimize=True,
            )
        )
        coefficients = mx.array((single * phase).astype(np.complex64))
        forces.append(
            mx.real(
                mx.sum(
                    density_reciprocal[..., None]
                    * imaginary
                    * vectors
                    * coefficients[..., None],
                    axis=(0, 1, 2),
                )
                * basis.grid.dv
            )
        )
    result = mx.stack(forces, axis=0).astype(mx.float32)
    mx.eval(result)
    return result
