"""Molecular mechanics topology primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array

DEFAULT_EAGER_NONBONDED_PAIR_LIMIT = 2_000_000


def _index_array(value, *, width: int, name: str) -> mx.array:
    array = np.asarray(value, dtype=np.int32)
    if array.size == 0:
        array = np.empty((0, width), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != width:
        msg = f"{name} must have shape (n, {width})"
        raise ValueError(msg)
    return mx.array(array, dtype=mx.int32)


def _unique_sorted_pairs(pairs: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    normalized = set()
    for pair in pairs:
        if len(pair) != 2:
            msg = "pair entries must contain exactly two atom indices"
            raise ValueError(msg)
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            msg = "pair entries cannot reference the same atom twice"
            raise ValueError(msg)
        normalized.add((min(i, j), max(i, j)))
    return tuple(sorted(normalized))


def _pairs_from_array(pairs: mx.array) -> tuple[tuple[int, int], ...]:
    array = np.asarray(pairs, dtype=np.int32)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 2:
        msg = "pairs must have shape (n, 2)"
        raise ValueError(msg)
    return _unique_sorted_pairs(array.tolist())


def _pair_codes(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    if pairs.size == 0:
        return np.empty((0,), dtype=np.int64)
    left = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64, copy=False)
    right = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64, copy=False)
    return left * np.int64(n_atoms) + right


def _sorted_pair_codes(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    return np.sort(_pair_codes(pairs, n_atoms))


def _isin_sorted_codes(codes: np.ndarray, sorted_codes: np.ndarray) -> np.ndarray:
    if codes.size == 0 or sorted_codes.size == 0:
        return np.zeros(codes.shape, dtype=bool)
    indices = np.searchsorted(sorted_codes, codes)
    matched = indices < sorted_codes.size
    result = np.zeros(codes.shape, dtype=bool)
    result[matched] = sorted_codes[indices[matched]] == codes[matched]
    return result


@dataclass(frozen=True)
class Topology:
    """Programmatic molecular mechanics topology."""

    n_atoms: int
    bonds: object = ()
    angles: object = ()
    dihedrals: object = ()
    impropers: object = ()
    exclusions: object = ()
    partial_charges: object | None = None
    one_four_pairs: object = ()
    nonbonded_exception_pairs: object = ()
    exclude_bonds: bool = True
    nonbonded_cutoff: float | None = None
    eager_nonbonded_pair_limit: int | None = DEFAULT_EAGER_NONBONDED_PAIR_LIMIT
    virtual_sites: object = ()
    virtual_site_types: object = ()

    @classmethod
    def from_sequences(
        cls,
        *,
        n_atoms: int,
        bonds: Sequence[Sequence[int]] = (),
        angles: Sequence[Sequence[int]] = (),
        dihedrals: Sequence[Sequence[int]] = (),
        impropers: Sequence[Sequence[int]] = (),
        exclusions: Sequence[Sequence[int]] = (),
        partial_charges: Sequence[float] | None = None,
        one_four_pairs: Sequence[Sequence[int]] | None = None,
        nonbonded_exception_pairs: Sequence[Sequence[int]] = (),
        exclude_bonds: bool = True,
        nonbonded_cutoff: float | None = None,
        eager_nonbonded_pair_limit: int | None = DEFAULT_EAGER_NONBONDED_PAIR_LIMIT,
        virtual_sites: Sequence[object] = (),
        virtual_site_types: Sequence[str] = (),
    ) -> Topology:
        """Create a topology from Python sequences."""

        if one_four_pairs is None:
            one_four_pairs = [(int(d[0]), int(d[3])) for d in dihedrals if len(d) == 4]
        return cls(
            n_atoms=n_atoms,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            impropers=impropers,
            exclusions=exclusions,
            partial_charges=partial_charges,
            one_four_pairs=one_four_pairs,
            nonbonded_exception_pairs=nonbonded_exception_pairs,
            exclude_bonds=exclude_bonds,
            nonbonded_cutoff=nonbonded_cutoff,
            eager_nonbonded_pair_limit=eager_nonbonded_pair_limit,
            virtual_sites=virtual_sites,
            virtual_site_types=virtual_site_types,
        )

    def __post_init__(self) -> None:
        if self.n_atoms <= 0:
            msg = "n_atoms must be positive"
            raise ValueError(msg)
        if self.eager_nonbonded_pair_limit is not None and self.eager_nonbonded_pair_limit < 0:
            msg = "eager_nonbonded_pair_limit must be non-negative when provided"
            raise ValueError(msg)

        bonds = _index_array(self.bonds, width=2, name="bonds")
        angles = _index_array(self.angles, width=3, name="angles")
        dihedrals = _index_array(self.dihedrals, width=4, name="dihedrals")
        impropers = _index_array(self.impropers, width=4, name="impropers")
        one_four_pairs = _unique_sorted_pairs(self.one_four_pairs)
        one_four_pair_set = frozenset(one_four_pairs)
        nonbonded_exception_pairs = _unique_sorted_pairs(self.nonbonded_exception_pairs)
        exclusion_pairs = list(_unique_sorted_pairs(self.exclusions))
        if self.exclude_bonds:
            exclusion_pairs.extend(_pairs_from_array(bonds))
        exclusion_pairs.extend(nonbonded_exception_pairs)
        exclusion_pair_set = frozenset(_unique_sorted_pairs(exclusion_pairs))
        exclusions = mx.array(tuple(sorted(exclusion_pair_set)), dtype=mx.int32)
        if exclusions.size == 0:
            exclusions = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        one_four = mx.array(one_four_pairs, dtype=mx.int32)
        if one_four.size == 0:
            one_four = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        exception_pairs = mx.array(nonbonded_exception_pairs, dtype=mx.int32)
        if exception_pairs.size == 0:
            exception_pairs = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)

        self._validate_indices(bonds, "bonds")
        self._validate_indices(angles, "angles")
        self._validate_indices(dihedrals, "dihedrals")
        self._validate_indices(impropers, "impropers")
        self._validate_indices(exclusions, "exclusions")
        self._validate_indices(one_four, "one_four_pairs")
        self._validate_indices(exception_pairs, "nonbonded_exception_pairs")

        charges = None
        if self.partial_charges is not None:
            charges = as_mx_array(self.partial_charges)
            if charges.shape != (self.n_atoms,):
                msg = "partial_charges must have shape (n_atoms,)"
                raise ValueError(msg)

        total_pair_count = self.n_atoms * (self.n_atoms - 1) // 2
        nonbonded_pair_count = total_pair_count - len(exclusion_pair_set)
        pair_limit = self.eager_nonbonded_pair_limit
        should_materialize = pair_limit is None or nonbonded_pair_count <= pair_limit
        nonbonded_pair_array = None
        nonbonded_one_four_mask = None
        if should_materialize:
            nonbonded_pair_array, nonbonded_one_four_mask = self._materialize_nonbonded_pairs(
                exclusion_pair_set,
                one_four_pair_set,
            )

        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "dihedrals", dihedrals)
        object.__setattr__(self, "impropers", impropers)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "partial_charges", charges)
        object.__setattr__(self, "one_four_pairs", one_four)
        object.__setattr__(self, "nonbonded_exception_pairs", exception_pairs)
        object.__setattr__(self, "_exclusion_set", exclusion_pair_set)
        object.__setattr__(self, "_one_four_set", one_four_pair_set)
        object.__setattr__(
            self,
            "_exclusion_codes",
            _sorted_pair_codes(
                np.asarray(tuple(exclusion_pair_set), dtype=np.int32),
                self.n_atoms,
            ),
        )
        object.__setattr__(
            self,
            "_one_four_codes",
            _sorted_pair_codes(
                np.asarray(tuple(one_four_pair_set), dtype=np.int32),
                self.n_atoms,
            ),
        )
        object.__setattr__(self, "_nonbonded_pairs", nonbonded_pair_array)
        object.__setattr__(self, "_nonbonded_one_four_mask", nonbonded_one_four_mask)
        object.__setattr__(self, "_nonbonded_pair_count", nonbonded_pair_count)
        object.__setattr__(self, "_nonbonded_pair_policy",
            "eager" if should_materialize else "lazy",
        )
        virtual_sites_tuple = tuple(self.virtual_sites)
        virtual_site_types_tuple = tuple(
            str(t) for t in self.virtual_site_types
        )
        if virtual_site_types_tuple and len(virtual_site_types_tuple) != len(
            virtual_sites_tuple
        ):
            msg = "virtual_site_types must have same length as virtual_sites"
            raise ValueError(msg)
        object.__setattr__(self, "virtual_sites", virtual_sites_tuple)
        object.__setattr__(self, "virtual_site_types", virtual_site_types_tuple)

    def _materialize_nonbonded_pairs(
        self,
        exclusion_pair_set: frozenset[tuple[int, int]],
        one_four_pair_set: frozenset[tuple[int, int]],
    ) -> tuple[mx.array, mx.array]:
        nonbonded_pairs = [
            (i, j)
            for i in range(self.n_atoms)
            for j in range(i + 1, self.n_atoms)
            if (i, j) not in exclusion_pair_set
        ]
        nonbonded_pair_array = mx.array(nonbonded_pairs, dtype=mx.int32)
        if nonbonded_pair_array.size == 0:
            nonbonded_pair_array = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        nonbonded_one_four_mask = as_mx_array(
            np.asarray([pair in one_four_pair_set for pair in nonbonded_pairs], dtype=np.float32)
        )
        return nonbonded_pair_array, nonbonded_one_four_mask

    def _validate_indices(self, indices: mx.array, name: str) -> None:
        array = np.asarray(indices)
        if array.size == 0:
            return
        if np.any(array < 0) or np.any(array >= self.n_atoms):
            msg = f"{name} contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)

    @property
    def exclusion_set(self) -> frozenset[tuple[int, int]]:
        """Excluded nonbonded pairs as normalized Python pairs."""

        return self._exclusion_set

    @property
    def one_four_set(self) -> frozenset[tuple[int, int]]:
        """1-4 scaled nonbonded pairs as normalized Python pairs."""

        return self._one_four_set

    @property
    def nonbonded_pair_policy(self) -> str:
        """Whether full nonbonded pairs were eagerly materialized or deferred."""

        return self._nonbonded_pair_policy

    @property
    def nonbonded_pair_count(self) -> int:
        """Number of non-excluded full-system nonbonded pairs."""

        return self._nonbonded_pair_count

    @property
    def nonbonded_build_report(self) -> dict[str, int | float | str | None]:
        """Compact report for topology nonbonded pair handling."""

        return {
            "pair_policy": self.nonbonded_pair_policy,
            "atom_count": self.n_atoms,
            "cutoff": self.nonbonded_cutoff,
            "exclusions": len(self.exclusion_set),
            "exceptions": int(self.nonbonded_exception_pairs.shape[0]),
            "one_four_pairs": int(self.one_four_pairs.shape[0]),
            "nonbonded_pairs": self.nonbonded_pair_count,
        }

    def nonbonded_pairs(self, pairs=None) -> mx.array:
        """Return nonbonded pairs with topology exclusions removed."""

        if pairs is None:
            if self._nonbonded_pairs is None:
                nonbonded_pair_array, nonbonded_one_four_mask = self._materialize_nonbonded_pairs(
                    self.exclusion_set,
                    self.one_four_set,
                )
                object.__setattr__(self, "_nonbonded_pairs", nonbonded_pair_array)
                object.__setattr__(self, "_nonbonded_one_four_mask", nonbonded_one_four_mask)
            return self._nonbonded_pairs
        pair_array = np.asarray(pairs, dtype=np.int32)
        if pair_array.size == 0:
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            msg = "pairs must have shape (n, 2)"
            raise ValueError(msg)
        if np.any(pair_array < 0) or np.any(pair_array >= self.n_atoms):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        left = np.minimum(pair_array[:, 0], pair_array[:, 1]).astype(np.int64, copy=False)
        right = np.maximum(pair_array[:, 0], pair_array[:, 1]).astype(np.int64, copy=False)
        codes = left * np.int64(self.n_atoms) + right
        keep = ~_isin_sorted_codes(codes, self._exclusion_codes)
        return mx.array(pair_array[keep], dtype=mx.int32)

    def pair_scales(self, pairs, *, one_four_scale: float = 1.0) -> mx.array:
        """Return per-pair 1-4 scaling factors."""

        pair_array = np.asarray(pairs, dtype=np.int32)
        if pair_array.size == 0:
            return as_mx_array([])
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            msg = "pairs must have shape (n, 2)"
            raise ValueError(msg)
        if float(one_four_scale) == 1.0 or self._one_four_codes.size == 0:
            return mx.ones((pair_array.shape[0],), dtype=mx.float32)
        codes = _pair_codes(pair_array, self.n_atoms)
        one_four = _isin_sorted_codes(codes, self._one_four_codes)
        scales = np.where(one_four, float(one_four_scale), 1.0).astype(np.float32)
        return as_mx_array(scales)

    def nonbonded_pair_scales(self, *, one_four_scale: float = 1.0) -> mx.array:
        """Return cached per-pair 1-4 scaling factors for all nonbonded pairs."""

        pairs = self.nonbonded_pairs()
        if pairs.shape[0] == 0:
            return as_mx_array([])
        if float(one_four_scale) == 1.0 or self._one_four_codes.size == 0:
            return mx.ones((pairs.shape[0],), dtype=mx.float32)
        return 1.0 + self._nonbonded_one_four_mask * (one_four_scale - 1.0)
