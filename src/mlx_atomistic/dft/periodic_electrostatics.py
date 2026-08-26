"""Periodic point-charge electrostatics for full-rank cells."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import erfc, pi, sqrt

import numpy as np

from mlx_atomistic.core import Cell


def _positions(positions: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.array(positions, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        msg = "positions must have shape (n_ions, 3)"
        raise ValueError(msg)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class _EwaldParameters:
    charge: np.ndarray
    centers: np.ndarray
    direct_matrix: np.ndarray
    reciprocal_matrix: np.ndarray
    direct_lengths: np.ndarray
    eta: float
    real_cutoff: float
    real_ranges: tuple[range, ...]
    reciprocal_cutoff: float
    max_indices: np.ndarray
    volume: float


def _validated_ewald_parameters(
    charge: np.ndarray,
    centers: np.ndarray,
    cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
    *,
    eta: float | None,
    tolerance: float,
) -> _EwaldParameters:
    cell = cell_lengths if isinstance(cell_lengths, Cell) else Cell(cell_lengths)
    direct = np.asarray(cell.matrix, dtype=np.float64)
    reciprocal = 2.0 * pi * np.linalg.inv(direct).T
    lengths = np.linalg.norm(direct, axis=1)
    volume = float(np.linalg.det(direct))
    if charge.shape != (centers.shape[0],):
        msg = "charges length must match positions"
        raise ValueError(msg)
    if tolerance <= 0.0 or tolerance >= 1.0:
        msg = "tolerance must lie in (0, 1)"
        raise ValueError(msg)
    if cell.is_orthorhombic:
        minimum_height = float(np.min(lengths))
    else:
        face_areas = np.asarray(
            [
                np.linalg.norm(np.cross(direct[1], direct[2])),
                np.linalg.norm(np.cross(direct[2], direct[0])),
                np.linalg.norm(np.cross(direct[0], direct[1])),
            ]
        )
        minimum_height = float(np.min(volume / face_areas))
    eta_value = float(eta) if eta is not None else 5.0 / minimum_height
    if eta_value <= 0.0:
        msg = "eta must be positive"
        raise ValueError(msg)
    cutoff_factor = sqrt(-np.log(tolerance))
    real_cutoff = cutoff_factor / eta_value
    if cell.is_orthorhombic:
        real_max_indices = np.ceil(real_cutoff / lengths).astype(int) + 1
    else:
        reciprocal_norms = np.linalg.norm(reciprocal, axis=1)
        real_max_indices = np.ceil(real_cutoff * reciprocal_norms / (2.0 * pi)).astype(int) + 1
    real_ranges = tuple(range(-int(value), int(value) + 1) for value in real_max_indices)
    reciprocal_cutoff = 2.0 * eta_value * cutoff_factor
    return _EwaldParameters(
        charge=charge,
        centers=centers,
        direct_matrix=direct,
        reciprocal_matrix=reciprocal,
        direct_lengths=lengths,
        eta=eta_value,
        real_cutoff=real_cutoff,
        real_ranges=real_ranges,
        reciprocal_cutoff=reciprocal_cutoff,
        max_indices=np.ceil(reciprocal_cutoff * lengths / (2.0 * pi)).astype(int),
        volume=volume,
    )


def _ewald_translation(
    parameters: _EwaldParameters,
    image: tuple[int, ...],
) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) @ parameters.direct_matrix


def _ewald_reciprocal_vectors(
    parameters: _EwaldParameters,
):
    ranges = tuple(
        range(-int(value), int(value) + 1) for value in parameters.max_indices
    )
    for indices in product(*ranges):
        if indices == (0, 0, 0):
            continue
        vector = np.asarray(indices, dtype=np.float64) @ parameters.reciprocal_matrix
        g2 = float(np.dot(vector, vector))
        if sqrt(g2) <= parameters.reciprocal_cutoff:
            yield vector, g2


def _ewald_real_energy(parameters: _EwaldParameters) -> float:
    energy = 0.0
    for ion_index, first in enumerate(parameters.centers):
        for other_index, second in enumerate(parameters.centers):
            for image in product(*parameters.real_ranges):
                displacement = first - second + _ewald_translation(parameters, image)
                distance = float(np.linalg.norm(displacement))
                if distance <= 1e-14 or distance > parameters.real_cutoff:
                    continue
                energy += (
                    parameters.charge[ion_index]
                    * parameters.charge[other_index]
                    * erfc(parameters.eta * distance)
                    / distance
                )
    return 0.5 * energy


def _ewald_reciprocal_energy(parameters: _EwaldParameters) -> float:
    energy = 0.0
    for vector, g2 in _ewald_reciprocal_vectors(parameters):
        structure = np.sum(parameters.charge * np.exp(-1j * (parameters.centers @ vector)))
        energy += (
            np.exp(-g2 / (4.0 * parameters.eta * parameters.eta))
            * float(abs(structure) ** 2)
            / g2
        )
    return energy * 2.0 * pi / parameters.volume


def _ewald_real_forces(parameters: _EwaldParameters) -> np.ndarray:
    forces = np.zeros_like(parameters.centers)
    for ion_index, first in enumerate(parameters.centers):
        for other_index, second in enumerate(parameters.centers):
            for image in product(*parameters.real_ranges):
                displacement = first - second + _ewald_translation(parameters, image)
                distance = float(np.linalg.norm(displacement))
                if distance <= 1e-14 or distance > parameters.real_cutoff:
                    continue
                coefficient = (
                    erfc(parameters.eta * distance) / distance**3
                    + 2.0
                    * parameters.eta
                    / sqrt(pi)
                    * np.exp(-((parameters.eta * distance) ** 2))
                    / distance**2
                )
                forces[ion_index] += (
                    parameters.charge[ion_index]
                    * parameters.charge[other_index]
                    * coefficient
                    * displacement
                )
    return forces


def _add_ewald_reciprocal_forces(
    parameters: _EwaldParameters,
    forces: np.ndarray,
) -> np.ndarray:
    for vector, g2 in _ewald_reciprocal_vectors(parameters):
        damping = np.exp(-g2 / (4.0 * parameters.eta * parameters.eta)) / g2
        structure = np.sum(parameters.charge * np.exp(-1j * (parameters.centers @ vector)))
        phase_imaginary = np.imag(np.exp(1j * (parameters.centers @ vector)) * structure)
        forces += (
            4.0
            * pi
            / parameters.volume
            * damping
            * parameters.charge[:, None]
            * phase_imaginary[:, None]
            * vector[None, :]
        )
    return forces


def _ewald_analytic_forces(parameters: _EwaldParameters) -> np.ndarray:
    forces = _ewald_real_forces(parameters)
    return _add_ewald_reciprocal_forces(parameters, forces)


def _ewald_analytic_stress(parameters: _EwaldParameters) -> np.ndarray:
    stress = np.zeros((3, 3), dtype=np.float64)
    for ion_index, first in enumerate(parameters.centers):
        for other_index, second in enumerate(parameters.centers):
            for image in product(*parameters.real_ranges):
                displacement = first - second + _ewald_translation(parameters, image)
                distance = float(np.linalg.norm(displacement))
                if distance <= 1e-14 or distance > parameters.real_cutoff:
                    continue
                coefficient = (
                    erfc(parameters.eta * distance) / distance**3
                    + 2.0
                    * parameters.eta
                    / sqrt(pi)
                    * np.exp(-((parameters.eta * distance) ** 2))
                    / distance**2
                )
                stress += (
                    0.5
                    * parameters.charge[ion_index]
                    * parameters.charge[other_index]
                    * coefficient
                    * np.outer(displacement, displacement)
                    / parameters.volume
                )
    reciprocal_diagonal = 0.0
    for vector, g2 in _ewald_reciprocal_vectors(parameters):
        structure = np.sum(
            parameters.charge * np.exp(-1j * (parameters.centers @ vector))
        )
        energy = (
            2.0
            * pi
            / parameters.volume
            * np.exp(-g2 / (4.0 * parameters.eta * parameters.eta))
            * float(abs(structure) ** 2)
            / g2
        )
        reciprocal_diagonal += energy / parameters.volume
        stress -= (
            2.0
            * energy
            / parameters.volume
            * (1.0 / (4.0 * parameters.eta * parameters.eta) + 1.0 / g2)
            * np.outer(vector, vector)
        )
    stress += reciprocal_diagonal * np.eye(3)
    background = -(
        pi
        * float(np.sum(parameters.charge)) ** 2
        / (2.0 * parameters.eta**2 * parameters.volume)
    )
    stress += background / parameters.volume * np.eye(3)
    return 0.5 * (stress + stress.T)


def _ewald_finite_difference_forces(
    charges: Sequence[float],
    centers: np.ndarray,
    cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
    *,
    displacement: float,
    eta: float | None,
    tolerance: float,
) -> np.ndarray:
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


def periodic_ewald_energy(
    charges: Sequence[float],
    positions: Sequence[Sequence[float]],
    cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
    *,
    eta: float | None = None,
    tolerance: float = 1e-10,
) -> float:
    """Return periodic point-charge Ewald energy with neutralizing background.

    Args:
        charges: Point charges in atomic units.
        positions: Cartesian positions in bohr.
        cell_lengths: A periodic `Cell`, three orthorhombic lengths, or a full
            row-vector cell matrix in bohr.
        eta: Optional Ewald splitting parameter in inverse bohr. Defaults to a
            cell-scaled value.
        tolerance: Real/reciprocal truncation target. Defaults to ``1e-10``.

    Returns:
        Ewald energy in Hartree.
    """

    charge = np.asarray(charges, dtype=np.float64)
    parameters = _validated_ewald_parameters(
        charge,
        _positions(positions),
        cell_lengths,
        eta=eta,
        tolerance=tolerance,
    )
    real_energy = _ewald_real_energy(parameters)
    reciprocal_energy = _ewald_reciprocal_energy(parameters)
    self_energy = -parameters.eta / sqrt(pi) * float(np.sum(charge * charge))
    total_charge = float(np.sum(charge))
    background = -pi * total_charge * total_charge / (2.0 * parameters.eta**2 * parameters.volume)
    return float(real_energy + reciprocal_energy + self_energy + background)


def periodic_ewald_forces(
    charges: Sequence[float],
    positions: Sequence[Sequence[float]],
    cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
    *,
    displacement: float = 1e-4,
    eta: float | None = None,
    tolerance: float = 1e-10,
    method: str = "analytic",
) -> np.ndarray:
    """Return forces for the periodic Ewald ion-ion energy.

    Args:
        charges: Point charges in atomic units.
        positions: Cartesian positions in bohr.
        cell_lengths: A periodic `Cell`, three orthorhombic lengths, or a full
            row-vector cell matrix in bohr.
        displacement: Central-difference step used only when
            ``method="finite_difference"``. Defaults to ``1e-4``.
        eta: Optional Ewald splitting parameter. Defaults to a cell-scaled value.
        tolerance: Ewald truncation target. Defaults to ``1e-10``.
        method: ``"analytic"`` or the validation-only
            ``"finite_difference"``. Defaults to ``"analytic"``.

    Returns:
        Force array with shape ``(n_ions, 3)`` in Hartree/bohr.
    """

    if displacement <= 0.0:
        msg = "displacement must be positive"
        raise ValueError(msg)
    if method not in {"analytic", "finite_difference"}:
        msg = "method must be 'analytic' or 'finite_difference'"
        raise ValueError(msg)
    centers = _positions(positions)
    if method == "finite_difference":
        return _ewald_finite_difference_forces(
            charges,
            centers,
            cell_lengths,
            displacement=displacement,
            eta=eta,
            tolerance=tolerance,
        )
    parameters = _validated_ewald_parameters(
        np.asarray(charges, dtype=np.float64),
        centers,
        cell_lengths,
        eta=eta,
        tolerance=tolerance,
    )
    return _ewald_analytic_forces(parameters)


def periodic_ewald_stress(
    charges: Sequence[float],
    positions: Sequence[Sequence[float]],
    cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
    *,
    eta: float | None = None,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return compression-positive periodic Ewald stress.

    Args:
        charges: Point charges in atomic units.
        positions: Cartesian positions in bohr.
        cell_lengths: Periodic cell in bohr.
        eta: Optional Ewald splitting parameter in inverse bohr.
        tolerance: Real/reciprocal truncation target.

    Returns:
        Symmetric stress tensor in Hartree/bohr cubed.
    """

    parameters = _validated_ewald_parameters(
        np.asarray(charges, dtype=np.float64),
        _positions(positions),
        cell_lengths,
        eta=eta,
        tolerance=tolerance,
    )
    return _ewald_analytic_stress(parameters)
