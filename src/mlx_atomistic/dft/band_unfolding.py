"""Primitive-cell interpretation of commensurate supercell band structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState
from mlx_atomistic.dft.kpoints import BandPath, KPoint
from mlx_atomistic.dft.periodic_scf import PeriodicBandStructureResult


def _cell_matrix(cell: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    values = np.asarray(cell, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        msg = f"{name} must be a finite 3x3 direct-lattice matrix"
        raise ValueError(msg)
    determinant = float(np.linalg.det(values))
    if determinant <= 0.0:
        msg = f"{name} must be right-handed with positive volume"
        raise ValueError(msg)
    return values


def _reciprocal_rows(direct: np.ndarray) -> np.ndarray:
    return 2.0 * pi * np.linalg.inv(direct).T


@dataclass(frozen=True)
class FoldedBandPath:
    """Mapping from a primitive reciprocal path into a commensurate supercell.

    Args:
        primitive_path: Original reduced-coordinate primitive-cell path.
        supercell_path: Equivalent reduced-coordinate supercell path.
        primitive_cell: Primitive direct-lattice row vectors in bohr.
        supercell_cell: Supercell direct-lattice row vectors in bohr.
        supercell_transform: Integer direct-lattice transform from primitive
            cell to supercell.
        volume_ratio: Number of primitive cells contained in the supercell.
        reciprocal_shifts: Integer supercell reciprocal shifts removed while
            folding every path point into the canonical first zone.
        path_distances: Cumulative primitive reciprocal-space distance.
    """

    primitive_path: BandPath
    supercell_path: BandPath
    primitive_cell: np.ndarray
    supercell_cell: np.ndarray
    supercell_transform: np.ndarray
    volume_ratio: int
    reciprocal_shifts: np.ndarray
    path_distances: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping summary."""

        return {
            "primitive_path": {
                "points": [point.to_dict() for point in self.primitive_path.points]
            },
            "supercell_path": {
                "points": [point.to_dict() for point in self.supercell_path.points]
            },
            "primitive_cell_bohr": self.primitive_cell.tolist(),
            "supercell_cell_bohr": self.supercell_cell.tolist(),
            "supercell_transform": self.supercell_transform.tolist(),
            "volume_ratio": self.volume_ratio,
            "reciprocal_shifts": self.reciprocal_shifts.tolist(),
            "path_distances_inverse_bohr": self.path_distances.tolist(),
        }


@dataclass(frozen=True)
class PeriodicUnfoldedBandStructureResult:
    """Supercell eigenvalues decorated with primitive-cell spectral weights.

    Args:
        folded_path: Primitive-to-supercell path mapping.
        bands: Underlying production supercell band result.
        spectral_weights: Primitive Bloch-character weights with shape
            ``(n_kpoints, n_bands)``.
        primitive_occupied_band_count: Doubly occupied primitive-cell bands.
    """

    folded_path: FoldedBandPath
    bands: PeriodicBandStructureResult
    spectral_weights: mx.array
    primitive_occupied_band_count: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe unfolded energies and spectral weights."""

        return {
            "folded_path": self.folded_path.to_dict(),
            "eigenvalues_hartree": np.asarray(self.bands.eigenvalues).tolist(),
            "spectral_weights": np.asarray(self.spectral_weights).tolist(),
            "primitive_occupied_band_count": self.primitive_occupied_band_count,
            "supercell_occupied_band_count": self.bands.occupied_band_count,
            "n_bands": self.bands.n_bands,
        }


def fold_band_path_to_supercell(
    primitive_cell: Sequence[Sequence[float]],
    supercell_cell: Sequence[Sequence[float]],
    band_path: BandPath,
    *,
    commensurability_tolerance: float = 1e-8,
) -> FoldedBandPath:
    """Fold a primitive reduced-coordinate path into a commensurate supercell.

    Direct lattice vectors are rows. The returned supercell points are reduced
    modulo integer reciprocal vectors into the half-open interval
    ``[-0.5, 0.5)``.

    Args:
        primitive_cell: Primitive direct-lattice row vectors in bohr.
        supercell_cell: Supercell direct-lattice row vectors in bohr.
        band_path: Primitive reduced-coordinate path.
        commensurability_tolerance: Absolute tolerance for the integer
            direct-lattice transform.

    Returns:
        Validated path mapping and primitive reciprocal-space distances.
    """

    primitive = _cell_matrix(primitive_cell, name="primitive_cell")
    supercell = _cell_matrix(supercell_cell, name="supercell_cell")
    transform_float = supercell @ np.linalg.inv(primitive)
    transform = np.rint(transform_float).astype(np.int64)
    if not np.allclose(
        transform_float,
        transform,
        rtol=0.0,
        atol=commensurability_tolerance,
    ):
        msg = "supercell_cell is not an integer transform of primitive_cell"
        raise ValueError(msg)
    volume_ratio_float = float(np.linalg.det(transform_float))
    volume_ratio = int(round(volume_ratio_float))
    if volume_ratio <= 0 or not np.isclose(
        volume_ratio_float,
        volume_ratio,
        rtol=0.0,
        atol=commensurability_tolerance,
    ):
        msg = "primitive-to-supercell volume ratio must be a positive integer"
        raise ValueError(msg)
    for point in band_path.points:
        if point.coordinate_system != "reduced":
            msg = "primitive band path must use reduced coordinates"
            raise ValueError(msg)

    primitive_reciprocal = _reciprocal_rows(primitive)
    supercell_reciprocal = _reciprocal_rows(supercell)
    primitive_reduced = np.asarray(
        [point.vector for point in band_path.points],
        dtype=np.float64,
    )
    cartesian = primitive_reduced @ primitive_reciprocal
    supercell_unwrapped = cartesian @ np.linalg.inv(supercell_reciprocal)
    shifts = np.floor(supercell_unwrapped + 0.5).astype(np.int64)
    supercell_reduced = supercell_unwrapped - shifts
    points = tuple(
        KPoint(
            vector,
            label=source.label,
            coordinate_system="reduced",
        )
        for vector, source in zip(
            supercell_reduced,
            band_path.points,
            strict=True,
        )
    )
    segment_lengths = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    primitive.setflags(write=False)
    supercell.setflags(write=False)
    transform.setflags(write=False)
    shifts.setflags(write=False)
    distances.setflags(write=False)
    return FoldedBandPath(
        primitive_path=band_path,
        supercell_path=BandPath(points),
        primitive_cell=primitive,
        supercell_cell=supercell,
        supercell_transform=transform,
        volume_ratio=volume_ratio,
        reciprocal_shifts=shifts,
        path_distances=distances,
    )


def unfold_periodic_band_structure(
    bands: PeriodicBandStructureResult,
    folded_path: FoldedBandPath,
    *,
    reciprocal_tolerance: float = 2e-5,
) -> PeriodicUnfoldedBandStructureResult:
    """Compute primitive Bloch-character weights from compact plane waves.

    A supercell plane wave contributes to primitive point ``k`` when its
    Cartesian wavevector differs from ``k`` by a primitive reciprocal-lattice
    vector. Summing squared coefficients over that exact reciprocal coset gives
    the unfolding spectral weight without materializing full FFT grids.

    Args:
        bands: Production bands evaluated along ``folded_path.supercell_path``.
        folded_path: Commensurate primitive-to-supercell path mapping.
        reciprocal_tolerance: Integer-coordinate admission tolerance.

    Returns:
        Underlying eigenvalues and primitive spectral weights.
    """

    if len(bands.points) != len(folded_path.supercell_path.points):
        msg = "band result and folded path must contain the same number of points"
        raise ValueError(msg)
    if bands.occupied_band_count % folded_path.volume_ratio != 0:
        msg = "supercell occupied bands must divide by the primitive-cell volume ratio"
        raise ValueError(msg)

    primitive_reciprocal = _reciprocal_rows(folded_path.primitive_cell)
    supercell_reciprocal = _reciprocal_rows(folded_path.supercell_cell)
    weights: list[mx.array] = []
    for point_index, (result, expected, primitive_point) in enumerate(
        zip(
            bands.points,
            folded_path.supercell_path.points,
            folded_path.primitive_path.points,
            strict=True,
        )
    ):
        if not np.allclose(
            result.requested_kpoint.vector,
            expected.vector,
            rtol=0.0,
            atol=1e-10,
        ):
            msg = f"band k-point {point_index} does not match folded path"
            raise ValueError(msg)
        compact = result.eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "band unfolding requires owned compact plane-wave coefficients"
            raise ValueError(msg)
        integer_g = np.asarray(result.basis.active_integer_g, dtype=np.float64)
        total_vectors = (
            np.asarray(result.basis.kpoint_cartesian, dtype=np.float64)
            + integer_g @ supercell_reciprocal
        )
        primitive_cartesian = (
            np.asarray(primitive_point.vector, dtype=np.float64)
            @ primitive_reciprocal
        )
        primitive_coordinates = (
            total_vectors - primitive_cartesian
        ) @ np.linalg.inv(primitive_reciprocal)
        selected = np.flatnonzero(
            np.max(
                np.abs(primitive_coordinates - np.rint(primitive_coordinates)),
                axis=1,
            )
            <= reciprocal_tolerance
        ).astype(np.int32)
        if selected.size == 0:
            msg = f"no primitive reciprocal coset found at path point {point_index}"
            raise ValueError(msg)
        selected_values = mx.take(compact.values, mx.array(selected), axis=1)
        weights.append(mx.sum(mx.abs(selected_values) ** 2, axis=1))

    spectral_weights = mx.stack(weights)
    mx.eval(spectral_weights)
    if not bool(mx.all(mx.isfinite(spectral_weights))):
        msg = "unfolded spectral weights are non-finite"
        raise ValueError(msg)
    if float(mx.min(spectral_weights)) < -1e-6 or float(mx.max(spectral_weights)) > 1.0001:
        msg = "unfolded spectral weights fall outside the normalized range"
        raise ValueError(msg)
    return PeriodicUnfoldedBandStructureResult(
        folded_path=folded_path,
        bands=bands,
        spectral_weights=spectral_weights,
        primitive_occupied_band_count=(
            bands.occupied_band_count // folded_path.volume_ratio
        ),
    )
