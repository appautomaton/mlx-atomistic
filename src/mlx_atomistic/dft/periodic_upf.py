"""Periodic local and nonlocal operators for numerical UPF data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import pi, sqrt

import mlx.core as mx
import numpy as np
from scipy.special import erf, spherical_jn

from mlx_atomistic.dft._compact import _CompactBatch, _CompactLaneState
from mlx_atomistic.dft._periodic_pseudopotential import (
    _periodic_positions,
    _periodic_pseudopotentials,
    _periodic_structure_factor,
)
from mlx_atomistic.dft._pseudopotential_identity import _pseudopotential_fingerprint
from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _ProjectorCache,
    _real_spherical_harmonics,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import (
    NonlocalProjectorData,
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


def _upf_projector_radial_transforms(
    projectors: Sequence[NonlocalProjectorData],
    q: np.ndarray,
    *,
    volume: float,
) -> tuple[np.ndarray, ...]:
    """Transform radial projectors while sharing angular Bessel work."""

    q_values = np.asarray(q, dtype=np.float64)
    if q_values.ndim != 1 or not np.isfinite(q_values).all() or np.any(q_values < 0.0):
        raise ValueError("UPF projector magnitudes must be finite and non-negative")
    validated = []
    groups: dict[tuple[int, int], list[int]] = {}
    for index, projector in enumerate(projectors):
        radial_grid = projector.radial_grid
        if not isinstance(radial_grid, RadialGrid):
            raise ValueError("UPF projector transform requires a radial grid")
        values = np.asarray(projector.values, dtype=np.float64)
        if values.shape != radial_grid.radii.shape:
            raise ValueError("UPF projector samples must match their radial grid")
        validated.append((projector.angular_momentum, radial_grid, values))
        groups.setdefault(
            (projector.angular_momentum, id(radial_grid)),
            [],
        ).append(index)
    transformed = [np.empty_like(q_values) for _projector in projectors]
    prefactor = 4.0 * pi / sqrt(volume)
    for indices in groups.values():
        angular_momentum, radial_grid, _values = validated[indices[0]]
        radius = radial_grid.radii
        values = np.stack([validated[index][2] for index in indices], axis=0)
        for start in range(0, q_values.size, _RADIAL_TRANSFORM_CHUNK_SIZE):
            selected = q_values[start : start + _RADIAL_TRANSFORM_CHUNK_SIZE]
            bessel = spherical_jn(
                angular_momentum,
                selected[:, None] * radius[None, :],
            )
            observed = prefactor * _simpson_radial(
                values[:, None, :] * bessel[None, :, :] * radius[None, None, :],
                radial_grid,
            )
            for local_index, projector_index in enumerate(indices):
                transformed[projector_index][start : start + selected.size] = (
                    observed[local_index]
                )
    return tuple(transformed)


def _upf_projector_radial_transform(
    projector: NonlocalProjectorData,
    q: np.ndarray,
    *,
    volume: float,
) -> np.ndarray:
    """Transform one UPF radial projector for source-oracle diagnostics."""

    return _upf_projector_radial_transforms(
        (projector,),
        q,
        volume=volume,
    )[0]


def _upf_angular_blocks(
    pseudopotential: PseudopotentialData,
) -> tuple[tuple[int, tuple[int, ...], np.ndarray], ...]:
    projectors = pseudopotential.nonlocal_projectors
    coupling = np.asarray(
        pseudopotential.nonlocal_coupling_matrix,
        dtype=np.float64,
    )
    count = len(projectors)
    if coupling.shape != (count, count):
        raise ValueError("UPF coupling matrix must match its projector count")
    angular = np.asarray(
        [projector.angular_momentum for projector in projectors],
        dtype=np.int64,
    )
    if np.any((angular < 0) | (angular > 2)):
        raise ValueError("periodic UPF nonlocal projectors require 0 <= l <= 2")
    cross_channel = angular[:, None] != angular[None, :]
    if np.any(np.abs(coupling[cross_channel]) > 1.0e-12):
        raise ValueError("scalar UPF PP_DIJ cannot couple different angular channels")
    blocks = []
    for l_value in sorted(set(int(value) for value in angular)):
        indices = np.flatnonzero(angular == l_value)
        blocks.append(
            (
                l_value,
                tuple(int(index) for index in indices),
                coupling[np.ix_(indices, indices)],
            )
        )
    return tuple(blocks)


def _flattened_upf_coupling(
    pseudopotentials: Sequence[PseudopotentialData],
) -> tuple[mx.array, int, int]:
    blocks = [
        np.asarray(block, dtype=np.float32)
        for pseudo in pseudopotentials
        for l_value, _indices, block in _upf_angular_blocks(pseudo)
        for _harmonic in range(2 * l_value + 1)
    ]
    projector_count = sum(int(block.shape[0]) for block in blocks)
    coupling = np.zeros((projector_count, projector_count), dtype=np.float32)
    offset = 0
    for block in blocks:
        width = int(block.shape[0])
        coupling[offset : offset + width, offset : offset + width] = block
        offset += width
    return mx.array(coupling), projector_count, len(blocks)


def _upf_projector_context_identity(
    pseudopotentials: Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(b"mlx-atomistic.upf-projector-context.v1\0")
    for pseudo in pseudopotentials:
        digest.update(_pseudopotential_fingerprint(pseudo).encode("ascii"))
        digest.update(b"\0")
    digest.update(basis.reciprocal_grid.fingerprint.encode("ascii"))
    digest.update(np.asarray(positions, dtype=np.float64).tobytes())
    digest.update(b"complex64-float32\0")
    return digest.hexdigest()


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


@dataclass(frozen=True)
class PeriodicUPFNonlocalOperator:
    """Compact separable scalar norm-conserving UPF operator."""

    pseudopotentials: tuple[PseudopotentialData, ...]
    basis: PlaneWaveBasis
    positions: np.ndarray
    _cache: _ProjectorCache
    _context_identity: str
    _owns_cache: bool
    _flattened_coupling: mx.array
    _projector_count: int
    _projector_group_count: int
    _harmonic_width: int
    _radial_projectors_by_ion: tuple[tuple[mx.array, ...], ...]
    _angular_blocks_by_ion: tuple[
        tuple[tuple[int, tuple[int, ...], np.ndarray], ...],
        ...,
    ]
    _coupling_by_ion: tuple[mx.array, ...]

    def __init__(
        self,
        pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
        basis: PlaneWaveBasis,
        positions: Sequence[Sequence[float]],
        *,
        cache: _ProjectorCache | None = None,
        cache_budget_bytes: int = _ProjectorCache.DEFAULT_BUDGET_BYTES,
    ):
        centers = _periodic_positions(positions)
        per_ion = _periodic_pseudopotentials(
            pseudopotential,
            int(centers.shape[0]),
            expected_format=PseudopotentialFormat.UPF,
        )
        if any(not pseudo.periodic_upf_compatible for pseudo in per_ion):
            raise ValueError(
                "periodic UPF nonlocal execution requires scalar norm-conserving "
                "input without augmentation, SOC, or nonlinear core correction"
            )
        flattened_coupling, projector_count, group_count = (
            _flattened_upf_coupling(per_ion)
        )
        projector_cache = (
            _ProjectorCache(cache_budget_bytes) if cache is None else cache
        )
        context_identity = _upf_projector_context_identity(
            per_ion,
            basis,
            centers,
        )
        projector_cache.bind(context_identity)
        vectors = np.asarray(
            basis._layout._active_shifted_vectors,
            dtype=np.float64,
        )
        q = np.sqrt(np.sum(vectors * vectors, axis=-1))
        unique_q, q_inverse = np.unique(q, return_inverse=True)
        radial_by_species: dict[str, tuple[mx.array, ...]] = {}
        blocks_by_species: dict[
            str,
            tuple[tuple[int, tuple[int, ...], np.ndarray], ...],
        ] = {}
        coupling_by_species: dict[str, mx.array] = {}
        radial_by_ion = []
        angular_blocks_by_ion = []
        coupling_by_ion = []
        for pseudo in per_ion:
            fingerprint = _pseudopotential_fingerprint(pseudo)
            radials = radial_by_species.get(fingerprint)
            if radials is None:
                transformed = _upf_projector_radial_transforms(
                    pseudo.nonlocal_projectors,
                    unique_q,
                    volume=basis.volume,
                )
                radials = tuple(
                    mx.array(values[q_inverse].astype(np.float32))
                    for values in transformed
                )
                radial_by_species[fingerprint] = radials
            blocks = blocks_by_species.get(fingerprint)
            coupling = coupling_by_species.get(fingerprint)
            if blocks is None or coupling is None:
                blocks = _upf_angular_blocks(pseudo)
                coupling, _projector_count, _group_count = (
                    _flattened_upf_coupling((pseudo,))
                )
                blocks_by_species[fingerprint] = blocks
                coupling_by_species[fingerprint] = coupling
            radial_by_ion.append(radials)
            angular_blocks_by_ion.append(blocks)
            coupling_by_ion.append(coupling)
        mx.eval(
            flattened_coupling,
            *(radial for radials in radial_by_species.values() for radial in radials),
            *coupling_by_species.values(),
        )
        object.__setattr__(self, "pseudopotentials", per_ion)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "positions", centers)
        object.__setattr__(self, "_cache", projector_cache)
        object.__setattr__(self, "_context_identity", context_identity)
        object.__setattr__(self, "_owns_cache", cache is None)
        object.__setattr__(self, "_flattened_coupling", flattened_coupling)
        object.__setattr__(self, "_projector_count", projector_count)
        object.__setattr__(self, "_projector_group_count", group_count)
        object.__setattr__(
            self,
            "_harmonic_width",
            sum(
                2 * l_value + 1
                for l_value in {
                    projector.angular_momentum
                    for pseudo in per_ion
                    for projector in pseudo.nonlocal_projectors
                }
            ),
        )
        object.__setattr__(self, "_radial_projectors_by_ion", tuple(radial_by_ion))
        object.__setattr__(self, "_angular_blocks_by_ion", tuple(angular_blocks_by_ion))
        object.__setattr__(self, "_coupling_by_ion", tuple(coupling_by_ion))

    def _flattened_cache_key(self) -> tuple[object, ...]:
        return (
            self._context_identity,
            self.basis.basis_fingerprint,
            self.basis.order_fingerprint,
            "flattened-projectors",
            "complex64",
        )

    @staticmethod
    def _estimated_batch_transient_bytes(
        operators: Sequence[PeriodicUPFNonlocalOperator],
        batch: _CompactBatch,
    ) -> int:
        """Return a conservative logical workspace bound for batched UPF."""

        return PeriodicGTHNonlocalOperator._estimated_batch_transient_bytes(
            operators,
            batch,
        )

    @staticmethod
    def _apply_compact_batch(
        operators: Sequence[PeriodicUPFNonlocalOperator],
        coefficients: Sequence[_CompactLaneState],
        *,
        batch: _CompactBatch,
        evaluate: bool = True,
    ) -> tuple[tuple[_CompactLaneState, ...], tuple[dict[str, int], ...]]:
        """Apply UPF projectors through the shared compensated batch backend."""

        return PeriodicGTHNonlocalOperator._apply_compact_batch(
            operators,
            coefficients,
            batch=batch,
            evaluate=evaluate,
        )

    def _ion_projectors(
        self,
        position: np.ndarray,
        radials: tuple[mx.array, ...],
        angular_blocks: tuple[tuple[int, tuple[int, ...], np.ndarray], ...],
        harmonics: dict[int, tuple[mx.array, ...]],
        vectors: mx.array,
    ) -> mx.array:
        center = mx.array(np.asarray(position, dtype=np.float32))
        phase = mx.exp(
            mx.array(-1j, dtype=mx.complex64)
            * mx.sum(vectors * center[None, :], axis=-1)
        )
        rows = []
        for l_value, indices, _block in angular_blocks:
            radial = mx.stack([radials[index] for index in indices], axis=0)
            angular_phase = (-1j) ** l_value
            for harmonic in harmonics[l_value]:
                rows.append(
                    (
                        radial
                        * harmonic[None, :]
                        * phase[None, :]
                        * angular_phase
                    ).astype(mx.complex64)
                )
        return mx.concatenate(rows, axis=0)

    def _generate_flattened_projectors(self, vectors: mx.array) -> mx.array:
        q = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
        harmonics = {
            l_value: _real_spherical_harmonics(l_value, vectors, q)
            for angular_blocks in self._angular_blocks_by_ion
            for l_value, _indices, _block in angular_blocks
        }
        rows = [
            self._ion_projectors(
                position,
                radials,
                angular_blocks,
                harmonics,
                vectors,
            )
            for position, radials, angular_blocks in zip(
                self.positions,
                self._radial_projectors_by_ion,
                self._angular_blocks_by_ion,
                strict=True,
            )
        ]
        beta = mx.concatenate(rows, axis=0)
        if int(beta.shape[0]) != self._projector_count:
            raise RuntimeError("flattened UPF projector count is inconsistent")
        return beta

    def _apply_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        evaluate: bool = True,
    ) -> tuple[_CompactLaneState, dict[str, int]]:
        batch = _CompactBatch.from_states((coefficients,))
        actions, metrics = self._apply_compact_batch(
            (self,),
            (coefficients,),
            batch=batch,
            evaluate=evaluate,
        )
        return actions[0], metrics[0]

    def apply(self, coefficients: mx.array) -> mx.array:
        """Apply the nonlocal UPF operator to one orbital or a stack.

        Args:
            coefficients: One admitted coefficient grid or a stack.

        Returns:
            Nonlocal operator action with the same shape.
        """

        state, was_single = self.basis._state_from_full(coefficients)
        applied, _metrics = self._apply_compact(state)
        return self.basis._layout.unpack_fresh(applied.values, single=was_single)

    def energy(
        self,
        coefficients: mx.array,
        *,
        occupations: Sequence[float],
    ) -> mx.array:
        """Return occupied nonlocal UPF energy in Hartree.

        Args:
            coefficients: Orbital stack in the admitted basis.
            occupations: One occupation per orbital.

        Returns:
            Real occupied nonlocal energy.
        """

        state, _was_single = self.basis._state_from_full(coefficients)
        if len(occupations) != state.vector_count:
            raise ValueError("occupations length must match the orbital count")
        applied, _metrics = self._apply_compact(state)
        expectations = mx.real(
            mx.sum(mx.conjugate(state.values) * applied.values, axis=1)
        )
        return mx.sum(
            expectations * mx.array(np.asarray(occupations, dtype=np.float32))
        )

    def _forces_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        occupations: Sequence[float],
        evaluate: bool = True,
    ) -> mx.array:
        self.basis._validate_state(coefficients)
        if coefficients.kind != "coefficients":
            raise ValueError("UPF force input must be coefficient state")
        occupation_values = np.asarray(occupations, dtype=np.float32)
        if occupation_values.shape != (coefficients.vector_count,):
            raise ValueError("occupations length must match the orbital count")
        if not np.all(np.isfinite(occupation_values)):
            raise ValueError("occupations must contain only finite values")
        vectors = self.basis._layout._active_shifted_vectors
        q = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
        harmonics = {
            l_value: _real_spherical_harmonics(l_value, vectors, q)
            for angular_blocks in self._angular_blocks_by_ion
            for l_value, _indices, _block in angular_blocks
        }
        coefficient_matrix = mx.transpose(coefficients.values)
        occupation_array = mx.array(occupation_values)
        imaginary = mx.array(1j, dtype=mx.complex64)
        forces = []
        for position, radials, angular_blocks, coupling in zip(
            self.positions,
            self._radial_projectors_by_ion,
            self._angular_blocks_by_ion,
            self._coupling_by_ion,
            strict=True,
        ):
            beta = self._ion_projectors(
                position,
                radials,
                angular_blocks,
                harmonics,
                vectors,
            )
            overlaps = mx.matmul(mx.conjugate(beta), coefficient_matrix)
            mixed = mx.matmul(coupling, overlaps)
            derivative_bras = (
                imaginary
                * mx.transpose(vectors)[:, None, :]
                * mx.conjugate(beta)[None, :, :]
            )
            derivative_overlaps = mx.matmul(derivative_bras, coefficient_matrix)
            derivative_energy = 2.0 * mx.real(
                mx.sum(
                    mx.conjugate(derivative_overlaps)
                    * (mixed * occupation_array[None, :])[None, :, :],
                    axis=(1, 2),
                )
            )
            forces.append(-derivative_energy)
        result = mx.stack(forces, axis=0).astype(mx.float32)
        if evaluate:
            mx.eval(result)
        return result

    def forces(
        self,
        coefficients: mx.array,
        *,
        occupations: Sequence[float],
    ) -> mx.array:
        """Return analytic nonlocal-UPF Hellmann--Feynman forces.

        Args:
            coefficients: Orbital stack in the admitted basis.
            occupations: One occupation per orbital.

        Returns:
            Nonlocal forces with shape ``(n_ions, 3)`` in Hartree/bohr.
        """

        state, _was_single = self.basis._state_from_full(coefficients)
        return self._forces_compact(state, occupations=occupations)

    def cache_info(self) -> dict[str, int]:
        """Return bounded projector-cache accounting."""

        return {
            "byte_budget": self._cache.byte_budget,
            "current_bytes": self._cache.current_bytes,
            "peak_bytes": self._cache.peak_bytes,
            "entry_count": self._cache.entry_count,
            "evictions": self._cache.evictions,
            "invalidations": self._cache.invalidations,
        }

    def close(self) -> None:
        """Release an operator-owned projector cache context."""

        if self._owns_cache:
            self._cache.close()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe nonlocal UPF metadata."""

        fingerprints = [
            _pseudopotential_fingerprint(pseudo)
            for pseudo in self.pseudopotentials
        ]
        radial_by_ion = [
            len(pseudo.nonlocal_projectors)
            for pseudo in self.pseudopotentials
        ]
        angular_by_ion = [
            sum(
                2 * projector.angular_momentum + 1
                for projector in pseudo.nonlocal_projectors
            )
            for pseudo in self.pseudopotentials
        ]
        homogeneous = len(set(fingerprints)) == 1
        return {
            "format": "upf",
            "ion_count": int(self.positions.shape[0]),
            "species_count": len(set(fingerprints)),
            "radial_projector_count_per_ion": (
                radial_by_ion[0] if homogeneous else radial_by_ion
            ),
            "angular_projector_count_per_ion": (
                angular_by_ion[0] if homogeneous else angular_by_ion
            ),
            "angular_projector_count_total": sum(angular_by_ion),
            "kpoint_cartesian_bohr_inverse": list(self.basis.kpoint_cartesian),
        }
