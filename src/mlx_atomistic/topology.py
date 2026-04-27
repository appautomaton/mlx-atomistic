"""Molecular mechanics topology primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array


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


@dataclass(frozen=True)
class Topology:
    """Programmatic molecular mechanics topology."""

    n_atoms: int
    bonds: object = ()
    angles: object = ()
    dihedrals: object = ()
    exclusions: object = ()
    partial_charges: object | None = None
    one_four_pairs: object = ()
    exclude_bonds: bool = True

    @classmethod
    def from_sequences(
        cls,
        *,
        n_atoms: int,
        bonds: Sequence[Sequence[int]] = (),
        angles: Sequence[Sequence[int]] = (),
        dihedrals: Sequence[Sequence[int]] = (),
        exclusions: Sequence[Sequence[int]] = (),
        partial_charges: Sequence[float] | None = None,
        one_four_pairs: Sequence[Sequence[int]] | None = None,
        exclude_bonds: bool = True,
    ) -> Topology:
        """Create a topology from Python sequences."""

        if one_four_pairs is None:
            one_four_pairs = [(int(d[0]), int(d[3])) for d in dihedrals if len(d) == 4]
        return cls(
            n_atoms=n_atoms,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            exclusions=exclusions,
            partial_charges=partial_charges,
            one_four_pairs=one_four_pairs,
            exclude_bonds=exclude_bonds,
        )

    def __post_init__(self) -> None:
        if self.n_atoms <= 0:
            msg = "n_atoms must be positive"
            raise ValueError(msg)

        bonds = _index_array(self.bonds, width=2, name="bonds")
        angles = _index_array(self.angles, width=3, name="angles")
        dihedrals = _index_array(self.dihedrals, width=4, name="dihedrals")
        one_four_pairs = _unique_sorted_pairs(self.one_four_pairs)
        exclusion_pairs = list(_unique_sorted_pairs(self.exclusions))
        if self.exclude_bonds:
            exclusion_pairs.extend(_pairs_from_array(bonds))
        exclusions = mx.array(_unique_sorted_pairs(exclusion_pairs), dtype=mx.int32)
        if exclusions.size == 0:
            exclusions = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        one_four = mx.array(one_four_pairs, dtype=mx.int32)
        if one_four.size == 0:
            one_four = mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)

        self._validate_indices(bonds, "bonds")
        self._validate_indices(angles, "angles")
        self._validate_indices(dihedrals, "dihedrals")
        self._validate_indices(exclusions, "exclusions")
        self._validate_indices(one_four, "one_four_pairs")

        charges = None
        if self.partial_charges is not None:
            charges = as_mx_array(self.partial_charges)
            if charges.shape != (self.n_atoms,):
                msg = "partial_charges must have shape (n_atoms,)"
                raise ValueError(msg)

        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "dihedrals", dihedrals)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "partial_charges", charges)
        object.__setattr__(self, "one_four_pairs", one_four)

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

        return frozenset(_pairs_from_array(self.exclusions))

    @property
    def one_four_set(self) -> frozenset[tuple[int, int]]:
        """1-4 scaled nonbonded pairs as normalized Python pairs."""

        return frozenset(_pairs_from_array(self.one_four_pairs))

    def nonbonded_pairs(self, pairs=None) -> mx.array:
        """Return nonbonded pairs with topology exclusions removed."""

        if pairs is None:
            candidate_pairs = [
                (i, j) for i in range(self.n_atoms) for j in range(i + 1, self.n_atoms)
            ]
        else:
            candidate_pairs = _pairs_from_array(pairs)

        excluded = self.exclusion_set
        filtered = [pair for pair in candidate_pairs if pair not in excluded]
        if not filtered:
            return mx.array(np.empty((0, 2), dtype=np.int32), dtype=mx.int32)
        return mx.array(filtered, dtype=mx.int32)

    def pair_scales(self, pairs, *, one_four_scale: float = 1.0) -> mx.array:
        """Return per-pair 1-4 scaling factors."""

        one_four = self.one_four_set
        scales = [one_four_scale if pair in one_four else 1.0 for pair in _pairs_from_array(pairs)]
        return as_mx_array(scales)
