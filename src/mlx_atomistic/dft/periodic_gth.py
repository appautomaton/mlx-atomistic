"""Analytical periodic GTH operators for cutoff-projected plane waves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erfc, pi, sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import GTHProjectorChannel, PseudopotentialData


def _validated_gth(pseudopotential: PseudopotentialData) -> None:
    if str(pseudopotential.format) != "gth":
        msg = "periodic GTH operators require a GTH pseudopotential"
        raise ValueError(msg)
    if pseudopotential.gth_rloc is None:
        msg = "GTH local radius is missing"
        raise ValueError(msg)


def _positions(positions: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        msg = "positions must have shape (n_ions, 3)"
        raise ValueError(msg)
    return values


def _structure_factor(vectors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    phase = np.einsum("...d,id->i...", vectors, positions, optimize=True)
    return np.sum(np.exp(-1j * phase), axis=0)


def gth_local_reciprocal_coefficients(
    pseudopotential: PseudopotentialData,
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return periodic local GTH Fourier-series coefficients.

    The formula follows Quantum ESPRESSO's analytical GTH transform in Hartree
    units, including the finite ``G=0`` limit and ionic structure factor.

    Args:
        pseudopotential: Parsed GTH pseudopotential shared with the reference.
        basis: Plane-wave basis supplying reciprocal vectors and volume.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Complex local-potential coefficients with shape ``basis.grid.shape``.
    """

    _validated_gth(pseudopotential)
    centers = _positions(positions)
    vectors = np.asarray(basis.reciprocal_vectors, dtype=np.float64)
    g2 = np.sum(vectors * vectors, axis=-1)
    rloc = float(pseudopotential.gth_rloc)
    coefficients = list(pseudopotential.gth_coefficients) + [0.0] * 4
    c1, c2, c3, c4 = coefficients[:4]
    zion = float(pseudopotential.valence_charge)
    rq2 = g2 * rloc * rloc
    gaussian = np.exp(-0.5 * rq2)
    polynomial = (
        c1
        + c2 * (3.0 - rq2)
        + c3 * (15.0 - 10.0 * rq2 + rq2 * rq2)
        + c4 * (105.0 - rq2 * (105.0 - rq2 * (21.0 - rq2)))
    )
    single = np.empty_like(g2)
    nonzero = g2 > 1e-14
    single[nonzero] = (
        4.0
        * pi
        * gaussian[nonzero]
        * (
            -zion / g2[nonzero]
            + sqrt(pi / 2.0) * rloc**3 * polynomial[nonzero]
        )
        / basis.volume
    )
    epsatm = 2.0 * pi * rloc * rloc * zion + (2.0 * pi) ** 1.5 * rloc**3 * (
        c1 + 3.0 * c2 + 15.0 * c3 + 105.0 * c4
    )
    single[~nonzero] = epsatm / basis.volume
    values = single * _structure_factor(vectors, centers)
    return mx.array(values.astype(np.complex64))


def gth_local_potential_grid(
    pseudopotential: PseudopotentialData,
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return the real periodic local GTH potential on the FFT grid.

    Args:
        pseudopotential: Parsed GTH pseudopotential.
        basis: Plane-wave basis supplying the FFT grid.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Real local potential with shape ``basis.grid.shape``.
    """

    coefficients = gth_local_reciprocal_coefficients(pseudopotential, basis, positions)
    return mx.real(mx.fft.ifftn(coefficients) * basis.grid.size)


def _gth_radial(channel: GTHProjectorChannel, projector_index: int, q: np.ndarray) -> np.ndarray:
    radius = channel.radius
    qr2 = (q * radius) ** 2
    gaussian = np.exp(-0.5 * qr2)
    l_value = channel.angular_momentum
    index = projector_index + 1
    if l_value == 0 and index == 1:
        return gaussian
    if l_value == 0 and index == 2:
        return 2.0 / sqrt(15.0) * gaussian * (3.0 - qr2)
    if l_value == 0 and index == 3:
        return 4.0 / (3.0 * sqrt(105.0)) * gaussian * (15.0 - 10.0 * qr2 + qr2**2)
    if l_value == 1 and index == 1:
        return gaussian * q / sqrt(3.0)
    if l_value == 1 and index == 2:
        return 2.0 / sqrt(105.0) * gaussian * q * (5.0 - qr2)
    if l_value == 1 and index == 3:
        return (
            4.0
            / (3.0 * sqrt(1155.0))
            * gaussian
            * q
            * (35.0 - 14.0 * qr2 + qr2**2)
        )
    if l_value == 2 and index == 1:
        return gaussian * q**2 / sqrt(15.0)
    if l_value == 2 and index == 2:
        return 2.0 / (3.0 * sqrt(105.0)) * gaussian * q**2 * (7.0 - qr2)
    if l_value == 3 and index == 1:
        return gaussian * q**3 / sqrt(105.0)
    msg = f"unsupported GTH projector l={l_value} index={index}"
    raise ValueError(msg)


def _real_spherical_harmonics(l_value: int, vectors: np.ndarray) -> tuple[np.ndarray, ...]:
    q = np.linalg.norm(vectors, axis=-1)
    if l_value == 0:
        return (np.full(q.shape, 1.0 / sqrt(4.0 * pi), dtype=np.float64),)
    safe = np.where(q > 1e-14, q, 1.0)
    coefficient = sqrt(3.0 / (4.0 * pi))
    if l_value == 1:
        values = (
            coefficient * vectors[..., 2] / safe,
            -coefficient * vectors[..., 0] / safe,
            -coefficient * vectors[..., 1] / safe,
        )
        return tuple(np.where(q > 1e-14, value, 0.0) for value in values)
    msg = f"periodic GTH spherical harmonics currently support l<=1, received {l_value}"
    raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicGTHNonlocalOperator:
    """Complete separable GTH nonlocal operator at one Bloch k-point."""

    pseudopotential: PseudopotentialData
    basis: PlaneWaveBasis
    positions: np.ndarray

    def __init__(
        self,
        pseudopotential: PseudopotentialData,
        basis: PlaneWaveBasis,
        positions: Sequence[Sequence[float]],
    ):
        _validated_gth(pseudopotential)
        if not pseudopotential.gth_channels:
            msg = "GTH pseudopotential has no complete nonlocal channels"
            raise ValueError(msg)
        object.__setattr__(self, "pseudopotential", pseudopotential)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "positions", _positions(positions))

    def _projector_group(
        self,
        position: np.ndarray,
        channel: GTHProjectorChannel,
        harmonic: np.ndarray,
    ) -> mx.array:
        vectors = np.asarray(self.basis.shifted_vectors, dtype=np.float64)
        q = np.linalg.norm(vectors, axis=-1)
        phase = np.exp(-1j * np.einsum("...d,d->...", vectors, position, optimize=True))
        angular_phase = (-1j) ** channel.angular_momentum
        prefactor = (
            4.0
            * pi
            * pi**0.25
            * sqrt(
                2.0 ** (channel.angular_momentum + 1)
                * channel.radius ** (2 * channel.angular_momentum + 3)
                / self.basis.volume
            )
        )
        projectors = []
        mask = np.asarray(self.basis.mask)
        for projector_index in range(channel.projector_count):
            radial = _gth_radial(channel, projector_index, q)
            values = prefactor * radial * harmonic * phase * angular_phase
            projectors.append(np.where(mask, values, 0.0).astype(np.complex64))
        return mx.array(np.stack(projectors))

    def apply(self, coefficients: mx.array) -> mx.array:
        """Apply the nonlocal operator to one orbital or an orbital stack.

        Args:
            coefficients: One admitted coefficient grid or a stack.

        Returns:
            Nonlocal operator action with the same shape.
        """

        values = mx.array(coefficients)
        was_single = values.shape == self.basis.grid.shape
        stack = mx.reshape(values, (1, *values.shape)) if was_single else values
        if len(stack.shape) != 4 or stack.shape[1:] != self.basis.grid.shape:
            msg = "coefficients must have shape grid.shape or (n, *grid.shape)"
            raise ValueError(msg)
        stack = self.basis.project(stack)
        vectors = np.asarray(self.basis.shifted_vectors, dtype=np.float64)
        outputs = []
        for orbital_index in range(int(stack.shape[0])):
            orbital = mx.reshape(stack[orbital_index], (self.basis.grid.size,))
            output = mx.zeros((self.basis.grid.size,), dtype=stack.dtype)
            for position in self.positions:
                for channel in self.pseudopotential.gth_channels:
                    coupling = mx.array(
                        np.asarray(channel.coupling_matrix, dtype=np.float32)
                    )
                    for harmonic in _real_spherical_harmonics(
                        channel.angular_momentum,
                        vectors,
                    ):
                        beta = self._projector_group(position, channel, harmonic)
                        beta_flat = mx.reshape(
                            beta,
                            (channel.projector_count, self.basis.grid.size),
                        )
                        overlaps = mx.sum(mx.conjugate(beta_flat) * orbital[None, :], axis=1)
                        mixed = coupling @ overlaps
                        applied = mx.sum(beta_flat * mixed[:, None], axis=0)
                        output = output + applied
            outputs.append(mx.reshape(output, self.basis.grid.shape))
        projected = self.basis.project(mx.stack(outputs, axis=0))
        return projected[0] if was_single else projected

    def energy(
        self,
        coefficients: mx.array,
        *,
        occupations: Sequence[float],
    ) -> mx.array:
        """Return occupied nonlocal energy in Hartree.

        Args:
            coefficients: Orbital stack in the admitted basis.
            occupations: One occupation per orbital.

        Returns:
            Real occupied nonlocal energy.
        """

        stack = mx.array(coefficients)
        if stack.shape == self.basis.grid.shape:
            stack = mx.reshape(stack, (1, *self.basis.grid.shape))
        if len(occupations) != int(stack.shape[0]):
            msg = "occupations length must match the orbital count"
            raise ValueError(msg)
        applied = self.apply(stack)
        energy = mx.array(0.0, dtype=mx.float32)
        for index, occupation in enumerate(occupations):
            expectation = mx.sum(mx.conjugate(stack[index]) * applied[index])
            energy = energy + float(occupation) * mx.real(expectation)
        return energy

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe nonlocal operator metadata.

        Returns:
            Channel, projector, angular, ion, and k-point metadata.
        """

        return {
            "ion_count": int(self.positions.shape[0]),
            "channel_count": len(self.pseudopotential.gth_channels),
            "radial_projector_count_per_ion": sum(
                channel.projector_count for channel in self.pseudopotential.gth_channels
            ),
            "angular_projector_count_per_ion": sum(
                channel.projector_count * (2 * channel.angular_momentum + 1)
                for channel in self.pseudopotential.gth_channels
            ),
            "kpoint_cartesian_bohr_inverse": list(self.basis.kpoint_cartesian),
        }


def periodic_ewald_energy(
    charges: Sequence[float],
    positions: Sequence[Sequence[float]],
    cell_lengths: Sequence[float],
    *,
    eta: float | None = None,
    tolerance: float = 1e-10,
) -> float:
    """Return periodic point-charge Ewald energy with neutralizing background.

    Args:
        charges: Point charges in atomic units.
        positions: Cartesian positions in bohr.
        cell_lengths: Orthorhombic cell lengths in bohr.
        eta: Optional Ewald splitting parameter in inverse bohr. Defaults to a
            cell-scaled value.
        tolerance: Real/reciprocal truncation target. Defaults to ``1e-10``.

    Returns:
        Ewald energy in Hartree.
    """

    charge = np.asarray(charges, dtype=np.float64)
    centers = _positions(positions)
    lengths = np.asarray(cell_lengths, dtype=np.float64)
    if charge.shape != (centers.shape[0],):
        msg = "charges length must match positions"
        raise ValueError(msg)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        msg = "cell_lengths must contain three positive values"
        raise ValueError(msg)
    if tolerance <= 0.0 or tolerance >= 1.0:
        msg = "tolerance must lie in (0, 1)"
        raise ValueError(msg)
    eta_value = float(eta) if eta is not None else 5.0 / float(np.min(lengths))
    if eta_value <= 0.0:
        msg = "eta must be positive"
        raise ValueError(msg)
    cutoff_factor = sqrt(-np.log(tolerance))
    real_cutoff = cutoff_factor / eta_value
    real_ranges = [
        range(
            -int(np.ceil(real_cutoff / length)) - 1,
            int(np.ceil(real_cutoff / length)) + 2,
        )
        for length in lengths
    ]
    real_energy = 0.0
    for ion_index, first in enumerate(centers):
        for other_index, second in enumerate(centers):
            for image in np.ndindex(*(len(values) for values in real_ranges)):
                translation = np.array(
                    [real_ranges[axis][image[axis]] * lengths[axis] for axis in range(3)]
                )
                displacement = first - second + translation
                distance = float(np.linalg.norm(displacement))
                if distance <= 1e-14 or distance > real_cutoff:
                    continue
                real_energy += charge[ion_index] * charge[other_index] * erfc(
                    eta_value * distance
                ) / distance
    real_energy *= 0.5

    reciprocal_cutoff = 2.0 * eta_value * cutoff_factor
    max_indices = np.ceil(reciprocal_cutoff * lengths / (2.0 * pi)).astype(int)
    reciprocal_energy = 0.0
    for h in range(-int(max_indices[0]), int(max_indices[0]) + 1):
        for k in range(-int(max_indices[1]), int(max_indices[1]) + 1):
            for l_value in range(-int(max_indices[2]), int(max_indices[2]) + 1):
                if h == 0 and k == 0 and l_value == 0:
                    continue
                vector = 2.0 * pi * np.array([h, k, l_value], dtype=np.float64) / lengths
                g2 = float(np.dot(vector, vector))
                if sqrt(g2) > reciprocal_cutoff:
                    continue
                structure = np.sum(charge * np.exp(-1j * (centers @ vector)))
                reciprocal_energy += (
                    np.exp(-g2 / (4.0 * eta_value * eta_value))
                    * float(abs(structure) ** 2)
                    / g2
                )
    volume = float(np.prod(lengths))
    reciprocal_energy *= 2.0 * pi / volume
    self_energy = -eta_value / sqrt(pi) * float(np.sum(charge * charge))
    total_charge = float(np.sum(charge))
    background = -pi * total_charge * total_charge / (2.0 * eta_value**2 * volume)
    return float(real_energy + reciprocal_energy + self_energy + background)


def periodic_ewald_forces(
    charges: Sequence[float],
    positions: Sequence[Sequence[float]],
    cell_lengths: Sequence[float],
    *,
    displacement: float = 1e-4,
    eta: float | None = None,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return central-difference forces for the periodic Ewald energy.

    Args:
        charges: Point charges in atomic units.
        positions: Cartesian positions in bohr.
        cell_lengths: Orthorhombic cell lengths in bohr.
        displacement: Central-difference step in bohr. Defaults to ``1e-4``.
        eta: Optional Ewald splitting parameter. Defaults to a cell-scaled value.
        tolerance: Ewald truncation target. Defaults to ``1e-10``.

    Returns:
        Force array with shape ``(n_ions, 3)`` in Hartree/bohr.
    """

    if displacement <= 0.0:
        msg = "displacement must be positive"
        raise ValueError(msg)
    centers = _positions(positions)
    forces = np.zeros_like(centers)
    for ion_index in range(centers.shape[0]):
        for axis in range(3):
            plus = centers.copy()
            minus = centers.copy()
            plus[ion_index, axis] += displacement
            minus[ion_index, axis] -= displacement
            e_plus = periodic_ewald_energy(
                charges,
                plus,
                cell_lengths,
                eta=eta,
                tolerance=tolerance,
            )
            e_minus = periodic_ewald_energy(
                charges,
                minus,
                cell_lengths,
                eta=eta,
                tolerance=tolerance,
            )
            forces[ion_index, axis] = -(e_plus - e_minus) / (2.0 * displacement)
    return forces
