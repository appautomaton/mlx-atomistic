"""Private compact plane-wave layouts and batched FFT boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.grids import ReciprocalGrid


def _update_array_digest(digest: object, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _scatter_complex_zeros(
    shape: tuple[int, ...],
    indices: mx.array,
    values: mx.array,
    *,
    axis: int,
) -> mx.array:
    """Scatter complex64 values through Metal-supported float32 stores."""

    real = mx.put_along_axis(
        mx.zeros(shape, dtype=mx.float32),
        indices,
        mx.real(values),
        axis=axis,
    )
    imaginary = mx.put_along_axis(
        mx.zeros(shape, dtype=mx.float32),
        indices,
        mx.imag(values),
        axis=axis,
    )
    return real.astype(mx.complex64) + mx.array(
        1j,
        dtype=mx.complex64,
    ) * imaginary.astype(mx.complex64)


def _ascending_true_indices(mask: mx.array) -> tuple[mx.array, int]:
    """Select true positions on MLX and return them in ascending order."""

    values = mx.reshape(mx.array(mask), (-1,))
    count = int(mx.sum(values))
    if count == 0:
        return mx.zeros((0,), dtype=mx.int32), 0
    keys = mx.where(values, 0, 1)
    partitioned = mx.argpartition(keys, count - 1)[:count]
    return mx.sort(partitioned).astype(mx.int32), count


@dataclass(frozen=True)
class _CompactBasisLayout:
    """Immutable active-plane-wave ordering for one k-point lane."""

    reciprocal: ReciprocalGrid
    cutoff_hartree: float
    kpoint_cartesian: tuple[float, float, float]
    active_flat_indices: mx.array
    active_integer_g: mx.array
    active_shifted_vectors: mx.array
    active_kinetic_energies: mx.array
    active_count: int
    order_fingerprint: str
    basis_fingerprint: str
    lane_id: str
    _active_flat_indices_np: np.ndarray
    _active_integer_g_np: np.ndarray

    @classmethod
    def build(
        cls,
        reciprocal: ReciprocalGrid,
        cutoff_hartree: float,
        kpoint_cartesian: Sequence[float],
        *,
        lane_label: str,
    ) -> _CompactBasisLayout:
        """Build a canonical compact layout from shared reciprocal metadata."""

        if cutoff_hartree <= 0.0:
            msg = "cutoff_hartree must be positive"
            raise ValueError(msg)
        if len(kpoint_cartesian) != 3:
            msg = "kpoint_cartesian must have three components"
            raise ValueError(msg)
        kpoint = np.asarray(kpoint_cartesian, dtype=np.float64)
        reciprocal_flat = mx.reshape(reciprocal.vectors, (-1, 3))
        shifted = reciprocal_flat + mx.array(kpoint.astype(np.float32))[None, :]
        kinetic = 0.5 * mx.sum(shifted * shifted, axis=1)
        active_indices, active_count = _ascending_true_indices(
            kinetic <= float(cutoff_hartree) + 1e-12
        )
        if active_count == 0:
            msg = "plane-wave cutoff admits no reciprocal coefficients"
            raise ValueError(msg)
        active = np.asarray(active_indices, dtype=np.int32)
        active_integer_g = mx.take(
            mx.reshape(reciprocal.integer_g, (-1, 3)),
            active_indices,
            axis=0,
        )
        ny, nz = reciprocal.real_grid.shape[1:]
        coordinate_indices = np.stack(
            [
                active // (ny * nz),
                (active // nz) % ny,
                active % nz,
            ],
            axis=1,
        ).astype(np.int32)
        grid_shape = np.asarray(reciprocal.real_grid.shape, dtype=np.int32)
        integer_g_np = np.where(
            coordinate_indices <= (grid_shape[None, :] - 1) // 2,
            coordinate_indices,
            coordinate_indices - grid_shape[None, :],
        ).astype(np.int32)
        active_shifted = mx.take(shifted, active_indices, axis=0)
        active_kinetic = mx.take(kinetic, active_indices, axis=0)
        active.setflags(write=False)
        integer_g_np.setflags(write=False)

        order_digest = sha256()
        order_digest.update(b"mlx-atomistic.compact-plane-wave-order.v1\0")
        order_digest.update(reciprocal.fingerprint.encode("ascii"))
        order_digest.update(b"ascending-flat-fft-index\0")
        _update_array_digest(order_digest, active)
        _update_array_digest(order_digest, integer_g_np)
        order_fingerprint = order_digest.hexdigest()

        basis_digest = sha256()
        basis_digest.update(b"mlx-atomistic.compact-plane-wave-basis.v1\0")
        basis_digest.update(order_fingerprint.encode("ascii"))
        basis_digest.update(np.asarray([cutoff_hartree], dtype=np.float64).tobytes())
        basis_digest.update(kpoint.tobytes())
        basis_digest.update(b"complex64-float32\0")
        basis_fingerprint = basis_digest.hexdigest()

        lane_digest = sha256()
        lane_digest.update(b"mlx-atomistic.compact-plane-wave-lane.v1\0")
        lane_digest.update(basis_fingerprint.encode("ascii"))
        lane_digest.update(lane_label.encode("utf-8"))
        return cls(
            reciprocal=reciprocal,
            cutoff_hartree=float(cutoff_hartree),
            kpoint_cartesian=tuple(float(value) for value in kpoint),
            active_flat_indices=active_indices,
            active_integer_g=active_integer_g,
            active_shifted_vectors=active_shifted,
            active_kinetic_energies=active_kinetic,
            active_count=active_count,
            order_fingerprint=order_fingerprint,
            basis_fingerprint=basis_fingerprint,
            lane_id=lane_digest.hexdigest(),
            _active_flat_indices_np=active,
            _active_integer_g_np=integer_g_np,
        )

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the shared FFT shape."""

        return self.reciprocal.real_grid.shape

    @property
    def grid_size(self) -> int:
        """Return the shared FFT entry count."""

        return self.reciprocal.real_grid.size

    @property
    def volume(self) -> float:
        """Return the shared cell volume."""

        return self.reciprocal.real_grid.volume

    def mask_fresh(self) -> mx.array:
        """Materialize a fresh full-grid cutoff mask."""

        flat = mx.put_along_axis(
            mx.zeros((self.grid_size,), dtype=mx.float32),
            self.active_flat_indices,
            mx.ones((self.active_count,), dtype=mx.float32),
            axis=0,
        )
        return mx.reshape(flat > 0.0, self.grid_shape)

    def shifted_vectors_fresh(self) -> mx.array:
        """Materialize fresh full-grid shifted reciprocal vectors."""

        kpoint = mx.array(np.asarray(self.kpoint_cartesian, dtype=np.float32))
        return self.reciprocal.vectors + kpoint

    def kinetic_energies_fresh(self) -> mx.array:
        """Materialize fresh full-grid kinetic energies."""

        shifted = self.shifted_vectors_fresh()
        return 0.5 * mx.sum(shifted * shifted, axis=-1)

    def pack_full(self, coefficients: mx.array) -> tuple[mx.array, bool]:
        """Gather a public full-grid value into canonical compact order."""

        values = mx.array(coefficients).astype(mx.complex64)
        was_single = values.shape == self.grid_shape
        if was_single:
            stack = mx.reshape(values, (1, self.grid_size))
        elif len(values.shape) == 4 and values.shape[1:] == self.grid_shape:
            stack = mx.reshape(values, (values.shape[0], self.grid_size))
        else:
            msg = "coefficients must have shape grid.shape or (n, *grid.shape)"
            raise ValueError(msg)
        return mx.take(stack, self.active_flat_indices, axis=1), was_single

    def unpack_fresh(self, coefficients: mx.array, *, single: bool = False) -> mx.array:
        """Scatter compact coefficients into a fresh exact-zero full grid."""

        values = mx.array(coefficients).astype(mx.complex64)
        if len(values.shape) == 1:
            values = mx.reshape(values, (1, values.shape[0]))
            single = True
        if len(values.shape) != 2 or values.shape[1] != self.active_count:
            msg = "compact coefficients must have shape (vectors, active_count)"
            raise ValueError(msg)
        indices = mx.broadcast_to(
            self.active_flat_indices[None, :],
            values.shape,
        )
        flat = _scatter_complex_zeros(
            (values.shape[0], self.grid_size),
            indices,
            values,
            axis=1,
        )
        full = mx.reshape(flat, (values.shape[0], *self.grid_shape))
        return full[0] if single else full


@dataclass(frozen=True)
class _CompactLaneState:
    """Unpadded persistent coefficients bound to one immutable layout."""

    values: mx.array
    layout: _CompactBasisLayout
    kind: str = "coefficients"

    def __post_init__(self) -> None:
        values = mx.array(self.values).astype(mx.complex64)
        if len(values.shape) == 1:
            values = mx.reshape(values, (1, values.shape[0]))
        if len(values.shape) != 2 or values.shape[1] != self.layout.active_count:
            msg = "compact lane values must have shape (vectors, active_count)"
            raise ValueError(msg)
        if self.kind not in {"coefficients", "hamiltonian_action"}:
            msg = "compact lane kind must be coefficients or hamiltonian_action"
            raise ValueError(msg)
        object.__setattr__(self, "values", values)

    @property
    def vector_count(self) -> int:
        """Return the number of coefficient vectors."""

        return int(self.values.shape[0])

    def full_grid_fresh(self) -> mx.array:
        """Return a fresh public full-grid coefficient stack."""

        return self.layout.unpack_fresh(self.values)


@dataclass(frozen=True)
class _CompatibilityCoefficientState:
    """Sparse exact round-trip state for the legacy public result constructor."""

    values: mx.array
    flat_indices: mx.array
    grid_shape: tuple[int, int, int]

    @classmethod
    def from_full(cls, coefficients: mx.array) -> _CompatibilityCoefficientState:
        """Pack exact nonzero support when no physical basis was supplied."""

        full = mx.array(coefficients).astype(mx.complex64)
        if len(full.shape) != 4:
            msg = "public result coefficients must have shape (vectors, *grid_shape)"
            raise ValueError(msg)
        grid_shape = tuple(int(value) for value in full.shape[1:])
        if len(grid_shape) != 3 or any(value <= 0 for value in grid_shape):
            msg = "public result coefficient grids must have three positive axes"
            raise ValueError(msg)
        flat = mx.reshape(full, (full.shape[0], int(np.prod(grid_shape))))
        support, _ = _ascending_true_indices(mx.any(flat != 0.0, axis=0))
        values = mx.take(flat, support, axis=1)
        return cls(
            values=values,
            flat_indices=support,
            grid_shape=grid_shape,
        )

    @property
    def vector_count(self) -> int:
        """Return the number of stored vectors."""

        return int(self.values.shape[0])

    def full_grid_fresh(self) -> mx.array:
        """Materialize a fresh exact full-grid coefficient stack."""

        grid_size = int(np.prod(self.grid_shape))
        if int(self.flat_indices.size) == 0:
            return mx.zeros(
                (self.vector_count, *self.grid_shape),
                dtype=mx.complex64,
            )
        indices = mx.broadcast_to(
            self.flat_indices[None, :],
            self.values.shape,
        )
        flat = _scatter_complex_zeros(
            (self.vector_count, grid_size),
            indices,
            self.values,
            axis=1,
        )
        return mx.reshape(flat, (self.vector_count, *self.grid_shape))


def _require_layout(state: _CompactLaneState, layout: _CompactBasisLayout) -> None:
    if (
        state.layout.basis_fingerprint != layout.basis_fingerprint
        or state.layout.order_fingerprint != layout.order_fingerprint
        or state.layout.lane_id != layout.lane_id
    ):
        msg = "compact coefficient state does not match the lane basis identity"
        raise ValueError(msg)


def _remap_initial_coefficients(
    state: _CompactLaneState,
    target: _CompactBasisLayout,
) -> _CompactLaneState:
    """Explicitly remap an initial state by exact integer ``G`` identity."""

    if state.kind == "hamiltonian_action":
        msg = "cached Hamiltonian actions cannot be remapped across basis identities"
        raise ValueError(msg)
    source_slots = {
        tuple(int(value) for value in integer_g): index
        for index, integer_g in enumerate(state.layout._active_integer_g_np)
    }
    slots = np.asarray(
        [
            source_slots.get(tuple(int(value) for value in integer_g), -1)
            for integer_g in target._active_integer_g_np
        ],
        dtype=np.int32,
    )
    present = slots >= 0
    safe_slots = np.where(present, slots, 0).astype(np.int32)
    gathered = mx.take(state.values, mx.array(safe_slots), axis=1)
    values = mx.where(
        mx.array(present)[None, :],
        gathered,
        mx.zeros_like(gathered),
    )
    return _CompactLaneState(values, target)


@dataclass(frozen=True)
class _CompactBatch:
    """Transient batch-first payload for compact plane-wave operators."""

    values: mx.array
    layouts: tuple[_CompactBasisLayout, ...]
    active_counts: tuple[int, ...]
    fft_indices: mx.array
    valid_mask: mx.array
    kinds: tuple[str, ...]
    padding_elements: int
    estimated_transient_bytes: int

    _DEFAULT_MAX_PADDING_FRACTION = 0.25
    _DEFAULT_MAX_TRANSIENT_BYTES = 512 * 1024 * 1024

    @classmethod
    def from_states(
        cls,
        states: Sequence[_CompactLaneState],
        *,
        max_padding_fraction: float = _DEFAULT_MAX_PADDING_FRACTION,
        max_transient_bytes: int = _DEFAULT_MAX_TRANSIENT_BYTES,
    ) -> _CompactBatch:
        """Bucket compatible unpadded lane states into one transient batch."""

        if not states:
            msg = "a compact batch requires at least one lane"
            raise ValueError(msg)
        if not 0.0 <= max_padding_fraction < 1.0:
            msg = "max_padding_fraction must lie in [0, 1)"
            raise ValueError(msg)
        if max_transient_bytes <= 0:
            msg = "max_transient_bytes must be positive"
            raise ValueError(msg)
        first = states[0]
        vector_count = first.vector_count
        reciprocal = first.layout.reciprocal
        for state in states:
            if state.vector_count != vector_count:
                msg = "compact batch lanes must have the same vector count"
                raise ValueError(msg)
            if state.layout.reciprocal is not reciprocal:
                msg = "compact batch lanes must share one reciprocal descriptor"
                raise ValueError(msg)
        bucket = max(state.layout.active_count for state in states)
        padding_elements = sum(bucket - state.layout.active_count for state in states)
        worst_padding_fraction = max(
            (bucket - state.layout.active_count) / bucket for state in states
        )
        if worst_padding_fraction > max_padding_fraction:
            msg = "compact batch active counts exceed the padding-fraction cap"
            raise ValueError(msg)
        lane_count = len(states)
        compact_payload_bytes = lane_count * vector_count * bucket * 8
        index_and_mask_bytes = lane_count * bucket * (4 + 1)
        fft_workspace_bytes = (
            2 * lane_count * vector_count * first.layout.grid_size * 8
        )
        estimated_transient_bytes = (
            compact_payload_bytes + index_and_mask_bytes + fft_workspace_bytes
        )
        if estimated_transient_bytes > max_transient_bytes:
            msg = "compact batch exceeds the transient byte budget"
            raise ValueError(msg)
        padded_values = []
        padded_indices = []
        masks = []
        all_grid_indices = np.arange(first.layout.grid_size, dtype=np.int32)
        for state in states:
            count = state.layout.active_count
            padding = bucket - count
            if padding:
                zeros = mx.zeros((vector_count, padding), dtype=mx.complex64)
                padded_values.append(mx.concatenate([state.values, zeros], axis=1))
                inactive = np.setdiff1d(
                    all_grid_indices,
                    state.layout._active_flat_indices_np,
                    assume_unique=True,
                )
                if inactive.size < padding:
                    msg = "compact FFT bucket has insufficient distinct padding indices"
                    raise ValueError(msg)
                padded_indices.append(
                    np.concatenate(
                        [state.layout._active_flat_indices_np, inactive[:padding]]
                    )
                )
            else:
                padded_values.append(state.values)
                padded_indices.append(state.layout._active_flat_indices_np)
            masks.append(np.arange(bucket) < count)
        return cls(
            values=mx.stack(padded_values, axis=0),
            layouts=tuple(state.layout for state in states),
            active_counts=tuple(state.layout.active_count for state in states),
            fft_indices=mx.array(np.stack(padded_indices).astype(np.int32)),
            valid_mask=mx.array(np.stack(masks)),
            kinds=tuple(state.kind for state in states),
            padding_elements=padding_elements,
            estimated_transient_bytes=estimated_transient_bytes,
        )

    @property
    def lane_count(self) -> int:
        """Return the number of batched lanes."""

        return int(self.values.shape[0])

    @property
    def vector_count(self) -> int:
        """Return the shared vector width."""

        return int(self.values.shape[1])

    @property
    def bucket_size(self) -> int:
        """Return the transient active-slot width."""

        return int(self.values.shape[2])

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the shared spatial FFT shape."""

        return self.layouts[0].grid_shape

    @property
    def grid_size(self) -> int:
        """Return the shared spatial FFT entry count."""

        return self.layouts[0].grid_size

    @property
    def volume(self) -> float:
        """Return the shared cell volume."""

        return self.layouts[0].volume

    def _expanded_indices(self) -> mx.array:
        return mx.broadcast_to(self.fft_indices[:, None, :], self.values.shape)

    def scatter(self) -> mx.array:
        """Scatter into one bounded batch-first full-grid FFT workspace."""

        masked = mx.where(
            self.valid_mask[:, None, :],
            self.values,
            mx.zeros_like(self.values),
        )
        flat = _scatter_complex_zeros(
            (self.lane_count, self.vector_count, self.grid_size),
            self._expanded_indices(),
            masked,
            axis=2,
        )
        return mx.reshape(
            flat,
            (self.lane_count, self.vector_count, *self.grid_shape),
        )

    def gather(self, full_grid: mx.array) -> mx.array:
        """Gather a batch-first full grid into padded compact slots."""

        values = mx.array(full_grid).astype(mx.complex64)
        expected = (self.lane_count, self.vector_count, *self.grid_shape)
        if values.shape != expected:
            msg = "full-grid batch shape does not match the compact batch"
            raise ValueError(msg)
        flat = mx.reshape(values, (self.lane_count, self.vector_count, self.grid_size))
        gathered = mx.take_along_axis(flat, self._expanded_indices(), axis=2)
        return mx.where(
            self.valid_mask[:, None, :],
            gathered,
            mx.zeros_like(gathered),
        )

    def to_real(self, *, scattered: mx.array | None = None) -> mx.array:
        """Run one inverse FFT across the final three spatial axes."""

        full_grid = self.scatter() if scattered is None else mx.array(scattered)
        expected = (self.lane_count, self.vector_count, *self.grid_shape)
        if full_grid.shape != expected:
            msg = "scattered coefficient workspace has the wrong shape"
            raise ValueError(msg)
        scale = self.grid_size / sqrt(self.volume)
        return mx.fft.ifftn(
            full_grid,
            s=self.grid_shape,
            axes=(-3, -2, -1),
        ) * scale

    def from_real(self, orbitals: mx.array) -> mx.array:
        """Run one forward FFT and gather the canonical compact slots."""

        values = mx.array(orbitals).astype(mx.complex64)
        expected = (self.lane_count, self.vector_count, *self.grid_shape)
        if values.shape != expected:
            msg = "real-space orbital batch has the wrong shape"
            raise ValueError(msg)
        scale = sqrt(self.volume) / self.grid_size
        reciprocal = mx.fft.fftn(
            values,
            s=self.grid_shape,
            axes=(-3, -2, -1),
        ) * scale
        return self.gather(reciprocal)

    def apply_local(
        self,
        potential: mx.array,
        *,
        scattered: mx.array | None = None,
    ) -> mx.array:
        """Apply lane-local or shared potentials with one FFT pair."""

        field = mx.array(potential)
        if field.shape == self.grid_shape:
            batched_field = field[None, None, ...]
        elif field.shape == (self.lane_count, *self.grid_shape):
            batched_field = field[:, None, ...]
        else:
            msg = "local potential must be shared grid-shaped or lane-batched"
            raise ValueError(msg)
        real = self.to_real(scattered=scattered)
        return self.from_real(real * batched_field)

    def unpad(self, values: mx.array, *, kind: str | None = None) -> tuple[_CompactLaneState, ...]:
        """Return unpadded persistent states from a padded operator payload."""

        payload = mx.array(values).astype(mx.complex64)
        if payload.shape != self.values.shape:
            msg = "compact batch payload shape does not match the bucket"
            raise ValueError(msg)
        states = []
        for index, (layout, count) in enumerate(
            zip(self.layouts, self.active_counts, strict=True)
        ):
            state_kind = self.kinds[index] if kind is None else kind
            states.append(_CompactLaneState(payload[index, :, :count], layout, state_kind))
        return tuple(states)
