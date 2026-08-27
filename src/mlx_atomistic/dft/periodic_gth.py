"""Analytical periodic GTH operators for cutoff-projected plane waves."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence, Set
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, pi, sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactBatch, _CompactLaneState
from mlx_atomistic.dft._periodic_pseudopotential import (
    _periodic_positions,
    _periodic_pseudopotentials,
    _periodic_structure_factor,
)
from mlx_atomistic.dft._pseudopotential_identity import _pseudopotential_fingerprint
from mlx_atomistic.dft.periodic_electrostatics import (
    periodic_ewald_energy as periodic_ewald_energy,
)
from mlx_atomistic.dft.periodic_electrostatics import (
    periodic_ewald_forces as periodic_ewald_forces,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import (
    GTHProjectorChannel,
    PseudopotentialData,
    PseudopotentialFormat,
)

_GTH_OVERLAP_CHUNK_SIZE = 1024


def _validated_gth(pseudopotential: PseudopotentialData) -> None:
    if str(pseudopotential.format) != "gth":
        msg = "periodic GTH operators require a GTH pseudopotential"
        raise ValueError(msg)
    if pseudopotential.gth_rloc is None:
        msg = "GTH local radius is missing"
        raise ValueError(msg)


def _per_ion_pseudopotentials(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    ion_count: int,
) -> tuple[PseudopotentialData, ...]:
    """Return one validated GTH pseudopotential per ion."""

    values = _periodic_pseudopotentials(
        pseudopotential,
        ion_count,
        expected_format=PseudopotentialFormat.GTH,
    )
    for value in values:
        _validated_gth(value)
    return values


_positions = _periodic_positions
_structure_factor = _periodic_structure_factor


def gth_local_reciprocal_coefficients(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return periodic local GTH Fourier-series coefficients.

    The formula follows Quantum ESPRESSO's analytical GTH transform in Hartree
    units, including the finite ``G=0`` limit and ionic structure factor.

    Args:
        pseudopotential: One shared or one-per-ion parsed GTH pseudopotential.
        basis: Plane-wave basis supplying reciprocal vectors and volume.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Complex local-potential coefficients with shape ``basis.grid.shape``.
    """

    centers = _positions(positions)
    per_ion = _per_ion_pseudopotentials(pseudopotential, int(centers.shape[0]))
    vectors = np.asarray(basis.reciprocal_vectors, dtype=np.float64)
    g2 = np.sum(vectors * vectors, axis=-1)
    grouped: dict[str, tuple[PseudopotentialData, list[np.ndarray]]] = {}
    for pseudo, center in zip(per_ion, centers, strict=True):
        fingerprint = _pseudopotential_fingerprint(pseudo)
        if fingerprint not in grouped:
            grouped[fingerprint] = (pseudo, [])
        grouped[fingerprint][1].append(center)
    values = np.zeros(basis.grid.shape, dtype=np.complex128)
    for pseudo, species_centers in grouped.values():
        rloc = float(pseudo.gth_rloc)
        coefficients = list(pseudo.gth_coefficients) + [0.0] * 4
        c1, c2, c3, c4 = coefficients[:4]
        zion = float(pseudo.valence_charge)
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
            * (-zion / g2[nonzero] + sqrt(pi / 2.0) * rloc**3 * polynomial[nonzero])
            / basis.volume
        )
        epsatm = 2.0 * pi * rloc * rloc * zion + (2.0 * pi) ** 1.5 * rloc**3 * (
            c1 + 3.0 * c2 + 15.0 * c3 + 105.0 * c4
        )
        single[~nonzero] = epsatm / basis.volume
        values += single * _structure_factor(
            vectors,
            np.asarray(species_centers, dtype=np.float64),
        )
    return mx.array(values.astype(np.complex64))


def gth_local_potential_grid(
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return the real periodic local GTH potential on the FFT grid.

    Args:
        pseudopotential: One shared or one-per-ion parsed GTH pseudopotential.
        basis: Plane-wave basis supplying the FFT grid.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Real local potential with shape ``basis.grid.shape``.
    """

    coefficients = gth_local_reciprocal_coefficients(pseudopotential, basis, positions)
    return mx.real(mx.fft.ifftn(coefficients) * basis.grid.size)


def periodic_gth_local_forces(
    density: mx.array,
    pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Return analytic local-GTH Hellmann--Feynman forces.

    The derivative is evaluated in reciprocal space from the converged
    electron density and the phase derivative of each ionic local potential.

    Args:
        density: Positive electron density on ``basis.grid``.
        pseudopotential: One shared or one-per-ion parsed GTH pseudopotential.
        basis: Plane-wave basis supplying the FFT grid and reciprocal vectors.
        positions: Ionic Cartesian positions in bohr.

    Returns:
        Local electron-ion forces with shape ``(n_ions, 3)`` in Hartree/bohr.
    """

    centers = _positions(positions)
    per_ion = _per_ion_pseudopotentials(pseudopotential, int(centers.shape[0]))
    density_array = mx.real(mx.array(density)).astype(mx.float32)
    if density_array.shape != basis.grid.shape:
        msg = "density shape must match the periodic FFT grid"
        raise ValueError(msg)
    density_finite = mx.all(mx.isfinite(density_array))
    mx.eval(density_finite)
    if not bool(density_finite):
        msg = "density must contain only finite values"
        raise ValueError(msg)

    density_reciprocal = mx.conjugate(mx.fft.fftn(density_array))
    vectors = mx.array(
        np.asarray(basis.reciprocal_vectors, dtype=np.float32),
    )
    imaginary = mx.array(1j, dtype=mx.complex64)
    forces = []
    for pseudo, center in zip(per_ion, centers, strict=True):
        coefficients = gth_local_reciprocal_coefficients(
            pseudo,
            basis,
            (center,),
        )
        force = mx.real(
            mx.sum(
                density_reciprocal[..., None] * imaginary * vectors * coefficients[..., None],
                axis=(0, 1, 2),
            )
            * basis.grid.dv
        )
        forces.append(force)
    result = mx.stack(forces, axis=0).astype(mx.float32)
    mx.eval(result)
    return result


def _gth_radial(
    channel: GTHProjectorChannel,
    projector_index: int,
    q: mx.array,
) -> mx.array:
    radius = channel.radius
    qr2 = (q * radius) ** 2
    gaussian = mx.exp(-0.5 * qr2)
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
        return 4.0 / (3.0 * sqrt(1155.0)) * gaussian * q * (35.0 - 14.0 * qr2 + qr2**2)
    if l_value == 2 and index == 1:
        return gaussian * q**2 / sqrt(15.0)
    if l_value == 2 and index == 2:
        return 2.0 / (3.0 * sqrt(105.0)) * gaussian * q**2 * (7.0 - qr2)
    if l_value == 3 and index == 1:
        return gaussian * q**3 / sqrt(105.0)
    msg = f"unsupported GTH projector l={l_value} index={index}"
    raise ValueError(msg)


def _real_spherical_harmonics(
    l_value: int,
    vectors: mx.array,
    q: mx.array,
) -> tuple[mx.array, ...]:
    if l_value == 0:
        return (mx.full(q.shape, 1.0 / sqrt(4.0 * pi), dtype=mx.float32),)
    safe = mx.where(q > 1e-14, q, 1.0)
    coefficient = sqrt(3.0 / (4.0 * pi))
    if l_value == 1:
        values = (
            coefficient * vectors[..., 2] / safe,
            -coefficient * vectors[..., 0] / safe,
            -coefficient * vectors[..., 1] / safe,
        )
        return tuple(mx.where(q > 1e-14, value, 0.0) for value in values)
    if l_value == 2:
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]
        radius_squared = safe * safe
        values = (
            sqrt(5.0 / (16.0 * pi))
            * (3.0 * z * z - radius_squared)
            / radius_squared,
            -sqrt(15.0 / (4.0 * pi)) * x * z / radius_squared,
            -sqrt(15.0 / (4.0 * pi)) * y * z / radius_squared,
            sqrt(15.0 / (16.0 * pi)) * (x * x - y * y) / radius_squared,
            sqrt(15.0 / (4.0 * pi)) * x * y / radius_squared,
        )
        return tuple(mx.where(q > 1e-14, value, 0.0) for value in values)
    msg = f"periodic GTH spherical harmonics currently support l<=2, received {l_value}"
    raise ValueError(msg)


@dataclass(frozen=True)
class _ProjectorCacheEntry:
    values: mx.array
    byte_count: int


class _ProjectorCache:
    """Bounded context-owned LRU cache for compact nonlocal projectors."""

    DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self, byte_budget: int = DEFAULT_BUDGET_BYTES):
        if byte_budget <= 0:
            msg = "projector cache byte budget must be positive"
            raise ValueError(msg)
        self.byte_budget = int(byte_budget)
        self._entries: OrderedDict[tuple[object, ...], _ProjectorCacheEntry] = OrderedDict()
        self._context_identity: str | None = None
        self._current_bytes = 0
        self._peak_bytes = 0
        self._evictions = 0
        self._invalidations = 0
        self._closed = False

    @property
    def current_bytes(self) -> int:
        """Return bytes currently retained by cache entries."""

        return self._current_bytes

    @property
    def peak_bytes(self) -> int:
        """Return the largest retained cache payload."""

        return self._peak_bytes

    @property
    def entry_count(self) -> int:
        """Return the current cache entry count."""

        return len(self._entries)

    @property
    def evictions(self) -> int:
        """Return the cumulative deterministic eviction count."""

        return self._evictions

    @property
    def invalidations(self) -> int:
        """Return the cumulative context invalidation count."""

        return self._invalidations

    def bind(self, context_identity: str) -> None:
        """Bind the cache to one geometry/cell/pseudopotential context."""

        if self._closed:
            msg = "closed projector cache cannot be rebound"
            raise RuntimeError(msg)
        if self._context_identity is None:
            self._context_identity = context_identity
        elif self._context_identity != context_identity:
            self.clear()
            self._context_identity = context_identity
            self._invalidations += 1

    def __enter__(self) -> _ProjectorCache:
        """Enter this cache's deterministic lifetime boundary."""

        if self._closed:
            msg = "closed projector cache cannot be entered"
            raise RuntimeError(msg)
        return self

    def __exit__(self, *_: object) -> None:
        """Close the cache when its owning runtime context exits."""

        self.close()

    def get(self, key: tuple[object, ...]) -> mx.array | None:
        """Return and refresh one cached projector group."""

        if self._closed:
            msg = "closed projector cache cannot be read"
            raise RuntimeError(msg)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry.values

    def put(
        self,
        key: tuple[object, ...],
        values: mx.array,
        *,
        protected_keys: Set[tuple[object, ...]] = frozenset(),
    ) -> tuple[int, bool]:
        """Insert one group without evicting inputs of the active lazy action."""

        if self._closed:
            msg = "closed projector cache cannot be written"
            raise RuntimeError(msg)
        payload = mx.array(values)
        byte_count = int(np.prod(payload.shape)) * 8
        if byte_count > self.byte_budget:
            return 0, False
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._current_bytes -= existing.byte_count
        removable_bytes = sum(
            entry.byte_count
            for candidate, entry in self._entries.items()
            if candidate not in protected_keys
        )
        if self._current_bytes + byte_count - removable_bytes > self.byte_budget:
            if existing is not None:
                self._entries[key] = existing
                self._current_bytes += existing.byte_count
            return 0, False
        evicted = 0
        while self._entries and self._current_bytes + byte_count > self.byte_budget:
            candidate = next(
                candidate for candidate in self._entries if candidate not in protected_keys
            )
            entry = self._entries.pop(candidate)
            self._current_bytes -= entry.byte_count
            self._evictions += 1
            evicted += 1
        self._entries[key] = _ProjectorCacheEntry(payload, byte_count)
        self._current_bytes += byte_count
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        return evicted, True

    def clear(self) -> None:
        """Release every cached MLX projector buffer."""

        self._entries.clear()
        self._current_bytes = 0

    def close(self) -> None:
        """Clear and permanently close this runtime cache context."""

        self.clear()
        self._context_identity = None
        self._closed = True


# Preserve the old private name for existing internal callers and downstream
# diagnostics while the shared periodic runtime adopts the format-neutral name.
_GTHProjectorCache = _ProjectorCache


def _projector_context_identity(
    pseudopotentials: Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(b"mlx-atomistic.gth-projector-context.v2\0")
    for pseudopotential in pseudopotentials:
        digest.update(_pseudopotential_fingerprint(pseudopotential).encode("ascii"))
        digest.update(b"\0")
    digest.update(basis.reciprocal_grid.fingerprint.encode("ascii"))
    digest.update(np.asarray(positions, dtype=np.float64).tobytes())
    digest.update(b"complex64-float32\0")
    return digest.hexdigest()


def _flattened_gth_coupling(
    pseudopotentials: Sequence[PseudopotentialData],
) -> tuple[mx.array, int]:
    """Return the block coupling for the canonical flattened projector order."""

    blocks = [
        np.asarray(channel.coupling_matrix, dtype=np.float32)
        for pseudopotential in pseudopotentials
        for channel in pseudopotential.gth_channels
        for _harmonic_index in range(2 * channel.angular_momentum + 1)
    ]
    projector_count = sum(int(block.shape[0]) for block in blocks)
    coupling = np.zeros((projector_count, projector_count), dtype=np.float32)
    offset = 0
    for block in blocks:
        width = int(block.shape[0])
        coupling[offset : offset + width, offset : offset + width] = block
        offset += width
    return mx.array(coupling), projector_count


@dataclass(frozen=True)
class PeriodicGTHNonlocalOperator:
    """Complete compact separable GTH operator at one Bloch k-point."""

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

    def __init__(
        self,
        pseudopotential: PseudopotentialData | Sequence[PseudopotentialData],
        basis: PlaneWaveBasis,
        positions: Sequence[Sequence[float]],
        *,
        cache: _ProjectorCache | None = None,
        cache_budget_bytes: int = _ProjectorCache.DEFAULT_BUDGET_BYTES,
    ):
        centers = _positions(positions)
        per_ion = _per_ion_pseudopotentials(pseudopotential, int(centers.shape[0]))
        if any(not value.gth_channels for value in per_ion):
            msg = "every GTH pseudopotential must have complete nonlocal channels"
            raise ValueError(msg)
        projector_cache = _ProjectorCache(cache_budget_bytes) if cache is None else cache
        context_identity = _projector_context_identity(
            per_ion,
            basis,
            centers,
        )
        flattened_coupling, projector_count = _flattened_gth_coupling(per_ion)
        projector_cache.bind(context_identity)
        object.__setattr__(self, "pseudopotentials", per_ion)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "positions", centers)
        object.__setattr__(self, "_cache", projector_cache)
        object.__setattr__(self, "_context_identity", context_identity)
        object.__setattr__(self, "_owns_cache", cache is None)
        object.__setattr__(self, "_flattened_coupling", flattened_coupling)
        object.__setattr__(self, "_projector_count", projector_count)
        object.__setattr__(
            self,
            "_projector_group_count",
            sum(
                2 * channel.angular_momentum + 1
                for pseudo in per_ion
                for channel in pseudo.gth_channels
            ),
        )
        object.__setattr__(
            self,
            "_harmonic_width",
            sum(
                2 * angular_momentum + 1
                for angular_momentum in {
                    channel.angular_momentum
                    for pseudo in per_ion
                    for channel in pseudo.gth_channels
                }
            ),
        )

    def _projector_group(
        self,
        position: np.ndarray,
        channel: GTHProjectorChannel,
        harmonic: mx.array,
        vectors: mx.array,
        q: mx.array,
    ) -> mx.array:
        center = mx.array(np.asarray(position, dtype=np.float32))
        phase = mx.exp(
            mx.array(-1j, dtype=mx.complex64) * mx.sum(vectors * center[None, :], axis=-1)
        )
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
        for projector_index in range(channel.projector_count):
            radial = _gth_radial(channel, projector_index, q)
            values = prefactor * radial * harmonic * phase * angular_phase
            projectors.append(values.astype(mx.complex64))
        return mx.stack(projectors, axis=0)

    def _flattened_cache_key(self) -> tuple[object, ...]:
        return (
            self._context_identity,
            self.basis.basis_fingerprint,
            self.basis.order_fingerprint,
            "flattened-projectors",
            "complex64",
        )

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

    @staticmethod
    def _estimated_batch_transient_bytes(
        operators: Sequence[PeriodicGTHNonlocalOperator],
        batch: _CompactBatch,
    ) -> int:
        """Return a conservative workspace bound for separable projectors."""

        if not operators:
            return 0
        logical_lane_count = len(operators)
        if logical_lane_count > batch.lane_capacity:
            msg = "nonlocal operator count exceeds the compact capacity"
            raise ValueError(msg)
        lane_count = batch.lane_capacity
        vector_count = batch.vector_count
        bucket_size = batch.bucket_size
        output_bytes = lane_count * vector_count * bucket_size * 8
        reference = operators[0]
        harmonic_width = reference._harmonic_width
        estimate = lane_count * bucket_size * (1 + harmonic_width) * 4
        projector_bytes = lane_count * reference._projector_count * bucket_size * 8
        overlap_bytes = lane_count * reference._projector_count * vector_count * 8
        overlap_chunks = ceil(bucket_size / _GTH_OVERLAP_CHUNK_SIZE)
        estimate += (
            2 * projector_bytes
            + (4 * overlap_chunks + 3) * overlap_bytes
            + 2 * output_bytes
            + reference._projector_count**2 * 4
        )
        return estimate

    @staticmethod
    def _initial_batch_metrics(state: _CompactLaneState) -> dict[str, int]:
        return {
            "projector_payload_elements": 0,
            "projector_elements_generated": 0,
            "projector_elements_loaded": 0,
            "projector_traffic_elements": 0,
            "projector_cache_hits": 0,
            "projector_cache_misses": 0,
            "projector_cache_evictions": 0,
            "projector_cache_bytes": 0,
            "projector_peak_workspace_bytes": (state.vector_count * state.layout.active_count * 8),
        }

    @staticmethod
    def _validated_batch_lanes(
        operators: Sequence[PeriodicGTHNonlocalOperator],
        coefficients: Sequence[_CompactLaneState],
        batch: _CompactBatch,
    ) -> tuple[
        PeriodicGTHNonlocalOperator,
        list[tuple[PeriodicGTHNonlocalOperator, mx.array]],
        list[dict[str, int]],
        dict[int, tuple[_ProjectorCache, set[tuple[object, ...]]]],
    ]:
        if not operators or len(operators) != len(coefficients):
            msg = "nonlocal batches require matching non-empty operator lanes"
            raise ValueError(msg)
        if batch.lane_count != len(operators):
            msg = "nonlocal batch lane count does not match its operators"
            raise ValueError(msg)
        reference = operators[0]
        lane_data: list[tuple[PeriodicGTHNonlocalOperator, mx.array]] = []
        metrics: list[dict[str, int]] = []
        protected_by_cache: dict[
            int,
            tuple[_ProjectorCache, set[tuple[object, ...]]],
        ] = {}
        for operator, state, layout in zip(
            operators,
            coefficients,
            batch.layouts,
            strict=True,
        ):
            if operator._context_identity != reference._context_identity:
                msg = "nonlocal batch operators must share one physical context"
                raise ValueError(msg)
            if layout is not state.layout:
                msg = "nonlocal batch layout does not match its coefficient lane"
                raise ValueError(msg)
            operator._cache.bind(operator._context_identity)
            operator.basis._validate_state(state)
            if state.kind != "coefficients":
                msg = "nonlocal input must be coefficient state"
                raise ValueError(msg)
            if state.vector_count > batch.vector_count:
                msg = "nonlocal batch coefficient width exceeds its capacity"
                raise ValueError(msg)
            lane_data.append((operator, operator.basis._layout._active_shifted_vectors))
            metrics.append(PeriodicGTHNonlocalOperator._initial_batch_metrics(state))
            protected_by_cache.setdefault(
                id(operator._cache),
                (operator._cache, set()),
            )
        return reference, lane_data, metrics, protected_by_cache

    def _generate_flattened_projectors(
        self,
        vectors: mx.array,
    ) -> mx.array:
        q = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
        harmonics = {
            channel.angular_momentum: _real_spherical_harmonics(
                channel.angular_momentum,
                vectors,
                q,
            )
            for pseudopotential in self.pseudopotentials
            for channel in pseudopotential.gth_channels
        }
        rows = []
        for position, pseudopotential in zip(
            self.positions,
            self.pseudopotentials,
            strict=True,
        ):
            for channel in pseudopotential.gth_channels:
                for harmonic in harmonics[channel.angular_momentum]:
                    rows.append(
                        self._projector_group(
                            position,
                            channel,
                            harmonic,
                            vectors,
                            q,
                        )
                    )
        return mx.concatenate(rows, axis=0)

    @staticmethod
    def _projectors_for_batch_lane(
        operator: PeriodicGTHNonlocalOperator,
        state: _CompactLaneState,
        vectors: mx.array,
        *,
        batch: _CompactBatch,
        projector_count: int,
        group_count: int,
        protected_by_cache: dict[
            int,
            tuple[_ProjectorCache, set[tuple[object, ...]]],
        ],
        metrics: dict[str, int],
    ) -> mx.array:
        cache, protected_keys = protected_by_cache[id(operator._cache)]
        key = operator._flattened_cache_key()
        beta = cache.get(key)
        payload_elements = operator._projector_count * operator.basis.active_count
        metrics["projector_payload_elements"] = payload_elements
        metrics["projector_elements_loaded"] = 2 * state.vector_count * payload_elements
        if beta is None:
            metrics["projector_cache_misses"] = group_count
            beta = operator._generate_flattened_projectors(vectors)
            metrics["projector_elements_generated"] = payload_elements
            evictions, inserted = cache.put(
                key,
                beta,
                protected_keys=protected_keys,
            )
            metrics["projector_cache_evictions"] = evictions
            if inserted:
                protected_keys.add(key)
        else:
            metrics["projector_cache_hits"] = group_count
            protected_keys.add(key)
        if int(beta.shape[0]) != projector_count:
            msg = "flattened nonlocal projector count is inconsistent"
            raise RuntimeError(msg)
        padding = batch.bucket_size - operator.basis.active_count
        if padding:
            beta = mx.concatenate(
                [
                    beta,
                    mx.zeros((operator._projector_count, padding), dtype=mx.complex64),
                ],
                axis=1,
            )
        metrics["projector_peak_workspace_bytes"] = max(
            metrics["projector_peak_workspace_bytes"],
            (
                projector_count * batch.bucket_size
                + 4 * projector_count * batch.vector_count
                + batch.vector_count * batch.bucket_size
            )
            * 8,
        )
        return beta

    @staticmethod
    def _padded_projector_batch(
        projectors: Sequence[mx.array],
        *,
        batch: _CompactBatch,
        projector_count: int,
    ) -> mx.array:
        beta_batch = mx.stack(projectors, axis=0)
        lane_padding = batch.lane_capacity - batch.lane_count
        if not lane_padding:
            return beta_batch
        return mx.concatenate(
            [
                beta_batch,
                mx.zeros(
                    (lane_padding, projector_count, batch.bucket_size),
                    dtype=mx.complex64,
                ),
            ],
            axis=0,
        )

    @staticmethod
    def _compensated_projector_overlaps(
        beta_batch: mx.array,
        batch: _CompactBatch,
        *,
        projector_count: int,
    ) -> mx.array:
        overlap_shape = (
            batch.lane_capacity,
            projector_count,
            batch.vector_count,
        )
        overlaps = mx.zeros(overlap_shape, dtype=mx.complex64)
        compensation = mx.zeros_like(overlaps)
        for start in range(0, batch.bucket_size, _GTH_OVERLAP_CHUNK_SIZE):
            stop = min(start + _GTH_OVERLAP_CHUNK_SIZE, batch.bucket_size)
            partial = mx.matmul(
                mx.conjugate(beta_batch[:, :, start:stop]),
                mx.transpose(batch.values[:, :, start:stop], (0, 2, 1)),
            )
            adjusted = partial - compensation
            updated = overlaps + adjusted
            compensation = (updated - overlaps) - adjusted
            overlaps = updated
        return overlaps

    @staticmethod
    def _finalize_batch_metrics(
        lane_data: Sequence[tuple[PeriodicGTHNonlocalOperator, mx.array]],
        metrics: Sequence[dict[str, int]],
    ) -> None:
        for lane_index, (operator, _) in enumerate(lane_data):
            lane_metrics = metrics[lane_index]
            lane_metrics["projector_traffic_elements"] = (
                lane_metrics["projector_elements_generated"]
                + lane_metrics["projector_elements_loaded"]
            )
            lane_metrics["projector_cache_bytes"] = operator._cache.current_bytes

    @staticmethod
    def _apply_compact_batch(
        operators: Sequence[PeriodicGTHNonlocalOperator],
        coefficients: Sequence[_CompactLaneState],
        *,
        batch: _CompactBatch,
        evaluate: bool = True,
    ) -> tuple[tuple[_CompactLaneState, ...], tuple[dict[str, int], ...]]:
        """Apply separable projectors with one padded k-lane matrix path."""

        (
            reference,
            lane_data,
            metrics,
            protected_by_cache,
        ) = PeriodicGTHNonlocalOperator._validated_batch_lanes(
            operators,
            coefficients,
            batch,
        )
        group_count = reference._projector_group_count
        flattened_projectors = [
            PeriodicGTHNonlocalOperator._projectors_for_batch_lane(
                operator,
                coefficients[lane_index],
                vectors,
                batch=batch,
                projector_count=reference._projector_count,
                group_count=group_count,
                protected_by_cache=protected_by_cache,
                metrics=metrics[lane_index],
            )
            for lane_index, (operator, vectors) in enumerate(lane_data)
        ]
        beta_batch = PeriodicGTHNonlocalOperator._padded_projector_batch(
            flattened_projectors,
            batch=batch,
            projector_count=reference._projector_count,
        )
        overlaps = PeriodicGTHNonlocalOperator._compensated_projector_overlaps(
            beta_batch,
            batch,
            projector_count=reference._projector_count,
        )
        mixed = mx.matmul(reference._flattened_coupling[None, :, :], overlaps)
        output = mx.matmul(mx.transpose(mixed, (0, 2, 1)), beta_batch)

        actions = batch.unpad(output, kind="hamiltonian_action")
        if evaluate:
            mx.eval(*(action.values for action in actions))
        PeriodicGTHNonlocalOperator._finalize_batch_metrics(lane_data, metrics)
        return actions, tuple(metrics)

    def apply(self, coefficients: mx.array) -> mx.array:
        """Apply the nonlocal operator to one orbital or an orbital stack.

        Args:
            coefficients: One admitted coefficient grid or a stack.

        Returns:
            Nonlocal operator action with the same shape.
        """

        state, was_single = self.basis._state_from_full(coefficients)
        applied, _ = self._apply_compact(state)
        return self.basis._layout.unpack_fresh(applied.values, single=was_single)

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

        state, _ = self.basis._state_from_full(coefficients)
        if len(occupations) != state.vector_count:
            msg = "occupations length must match the orbital count"
            raise ValueError(msg)
        applied, _ = self._apply_compact(state)
        expectations = mx.real(mx.sum(mx.conjugate(state.values) * applied.values, axis=1))
        return mx.sum(expectations * mx.array(np.asarray(occupations, dtype=np.float32)))

    def _forces_compact(
        self,
        coefficients: _CompactLaneState,
        *,
        occupations: Sequence[float],
        evaluate: bool = True,
    ) -> mx.array:
        """Return analytic per-ion nonlocal forces for compact orbitals."""

        self.basis._validate_state(coefficients)
        if coefficients.kind != "coefficients":
            msg = "GTH force input must be coefficient state"
            raise ValueError(msg)
        occupation_values = np.asarray(occupations, dtype=np.float32)
        if occupation_values.shape != (coefficients.vector_count,):
            msg = "occupations length must match the orbital count"
            raise ValueError(msg)
        if not np.all(np.isfinite(occupation_values)):
            msg = "occupations must contain only finite values"
            raise ValueError(msg)

        vectors = self.basis._layout._active_shifted_vectors
        q = mx.sqrt(mx.sum(vectors * vectors, axis=-1))
        coefficient_matrix = mx.transpose(coefficients.values)
        occupation_array = mx.array(occupation_values)
        imaginary = mx.array(1j, dtype=mx.complex64)
        forces = []
        for position, pseudo in zip(
            self.positions,
            self.pseudopotentials,
            strict=True,
        ):
            harmonics = {
                channel.angular_momentum: _real_spherical_harmonics(
                    channel.angular_momentum,
                    vectors,
                    q,
                )
                for channel in pseudo.gth_channels
            }
            rows = [
                self._projector_group(
                    position,
                    channel,
                    harmonic,
                    vectors,
                    q,
                )
                for channel in pseudo.gth_channels
                for harmonic in harmonics[channel.angular_momentum]
            ]
            beta = mx.concatenate(rows, axis=0)
            coupling, projector_count = _flattened_gth_coupling((pseudo,))
            if int(beta.shape[0]) != projector_count:
                msg = "per-ion GTH projector count is inconsistent"
                raise RuntimeError(msg)
            overlaps = mx.matmul(mx.conjugate(beta), coefficient_matrix)
            mixed = mx.matmul(coupling, overlaps)
            derivative_bras = (
                imaginary * mx.transpose(vectors)[:, None, :] * mx.conjugate(beta)[None, :, :]
            )
            derivative_overlaps = mx.matmul(
                derivative_bras,
                coefficient_matrix,
            )
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
        """Return analytic nonlocal-GTH Hellmann--Feynman forces.

        Args:
            coefficients: Orbital stack in the admitted basis.
            occupations: One occupation per orbital.

        Returns:
            Nonlocal forces with shape ``(n_ions, 3)`` in Hartree/bohr.
        """

        state, _ = self.basis._state_from_full(coefficients)
        return self._forces_compact(state, occupations=occupations)

    def cache_info(self) -> dict[str, int]:
        """Return bounded projector-cache accounting.

        Returns:
            Budget, retained/peak bytes, entries, evictions, and invalidations.
        """

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
        """Return JSON-safe nonlocal operator metadata.

        Returns:
            Channel, projector, angular, ion, and k-point metadata.
        """

        fingerprints = [_pseudopotential_fingerprint(value) for value in self.pseudopotentials]
        radial_by_ion = [
            sum(channel.projector_count for channel in value.gth_channels)
            for value in self.pseudopotentials
        ]
        angular_by_ion = [
            sum(
                channel.projector_count * (2 * channel.angular_momentum + 1)
                for channel in value.gth_channels
            )
            for value in self.pseudopotentials
        ]
        homogeneous = len(set(fingerprints)) == 1
        return {
            "ion_count": int(self.positions.shape[0]),
            "species_count": len(set(fingerprints)),
            "channel_count": (len(self.pseudopotentials[0].gth_channels) if homogeneous else None),
            "channel_count_total": sum(len(value.gth_channels) for value in self.pseudopotentials),
            "radial_projector_count_per_ion": (radial_by_ion[0] if homogeneous else radial_by_ion),
            "angular_projector_count_per_ion": (
                angular_by_ion[0] if homogeneous else angular_by_ion
            ),
            "angular_projector_count_total": sum(angular_by_ion),
            "kpoint_cartesian_bohr_inverse": list(self.basis.kpoint_cartesian),
        }
