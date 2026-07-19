"""k-point meshes and band-structure diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft.nonlocal_pseudopotential import NonlocalPseudopotentialOperator
from mlx_atomistic.dft.operators import DenseHamiltonianReference, KohnShamOperator
from mlx_atomistic.dft.scf import SCFResult
from mlx_atomistic.dft.system import DFTSystem
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

_TIME_REVERSAL_COORDINATE_ATOL = 1e-10
_TIME_REVERSAL_WEIGHT_RTOL = 1e-12
_TIME_REVERSAL_WEIGHT_ATOL = 1e-15


def _reciprocal_shift_if_close(
    values: Sequence[float],
    *,
    atol: float = _TIME_REVERSAL_COORDINATE_ATOL,
) -> tuple[int, int, int] | None:
    shifts = tuple(int(round(float(value))) for value in values)
    if all(
        abs(float(value) - shift) <= atol
        for value, shift in zip(values, shifts, strict=True)
    ):
        return shifts
    return None


def _time_reversal_shift(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[int, int, int] | None:
    return _reciprocal_shift_if_close(
        tuple(
            float(left) + float(right)
            for left, right in zip(first, second, strict=True)
        )
    )


@dataclass(frozen=True, eq=False)
class TimeReversalOwnershipEntry:
    """Ownership metadata for one explicit reduced-coordinate k-point.

    The time-reversal permutation maps each active compact coefficient index at
    this explicit point to its signed-``G`` index at ``partner_index``. It is
    populated only after active-basis admission succeeds.

    Args:
        explicit_index: Original index in the caller's k-point mesh.
        reduced_kpoint: Original reduced-coordinate point.
        original_weight: Original normalized integration weight.
        owner_index: Explicit point whose compact eigenstate is retained.
        partner_index: Time-reversed explicit point, or ``None`` when absent.
        role: ``"owner"``, ``"partner"``, or ``"independent"``.
        aggregated_weight: Integration weight consumed by the owner lane.
        reciprocal_shift: Integer vector satisfying
            ``k + k_partner = reciprocal_shift``.
        _time_reversal_permutation: Private source-to-partner compact-index
            permutation snapshot, or ``None`` when reuse is not admitted.
        fallback_reason: Stable independent-lane reason, or ``None``.
    """

    explicit_index: int
    reduced_kpoint: tuple[float, float, float]
    original_weight: float
    owner_index: int
    partner_index: int | None
    role: str
    aggregated_weight: float
    reciprocal_shift: tuple[int, int, int] | None
    _time_reversal_permutation: np.ndarray | None = None
    fallback_reason: str | None = None

    @property
    def time_reversal_permutation(self) -> np.ndarray | None:
        """Return a caller-owned copy of the signed-``G`` permutation."""

        if self._time_reversal_permutation is None:
            return None
        return np.array(self._time_reversal_permutation, copy=True)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe ownership record.

        Returns:
            Explicit point, weight, owner, partner, role, permutation, and
            fallback diagnostics.
        """

        return {
            "explicit_index": self.explicit_index,
            "reduced_kpoint": list(self.reduced_kpoint),
            "original_weight": self.original_weight,
            "owner_index": self.owner_index,
            "partner_index": self.partner_index,
            "role": self.role,
            "aggregated_weight": self.aggregated_weight,
            "reciprocal_shift": (
                None if self.reciprocal_shift is None else list(self.reciprocal_shift)
            ),
            "time_reversal_permutation": (
                None
                if self._time_reversal_permutation is None
                else self._time_reversal_permutation.tolist()
            ),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, eq=False)
class TimeReversalOwnership:
    """Deterministic owner/partner topology for an explicit k-point mesh.

    Args:
        entries: One ownership entry per explicit mesh point, in original order.
        active_bases_admitted: Whether exact active-basis permutations have been
            checked and attached.
    """

    entries: tuple[TimeReversalOwnershipEntry, ...]
    active_bases_admitted: bool = False

    @property
    def owned_indices(self) -> tuple[int, ...]:
        """Return explicit indices whose compact states are retained."""

        return tuple(
            entry.explicit_index
            for entry in self.entries
            if entry.owner_index == entry.explicit_index
        )

    @property
    def representative_indices(self) -> tuple[int, ...]:
        """Return admitted time-reversal representative indices."""

        return tuple(
            entry.explicit_index for entry in self.entries if entry.role == "owner"
        )

    @property
    def partner_indices(self) -> tuple[int, ...]:
        """Return explicit indices published from owner time-reversal views."""

        return tuple(
            entry.explicit_index for entry in self.entries if entry.role == "partner"
        )

    @property
    def fallback_reasons(self) -> dict[int, str]:
        """Return independent-lane fallback reasons keyed by explicit index."""

        return {
            entry.explicit_index: entry.fallback_reason
            for entry in self.entries
            if entry.fallback_reason is not None
        }

    def entry_for(self, explicit_index: int) -> TimeReversalOwnershipEntry:
        """Return ownership metadata for one explicit index.

        Args:
            explicit_index: Original mesh index.

        Returns:
            Matching ownership entry.

        Raises:
            IndexError: If the index is outside the explicit mesh.
        """

        if explicit_index < 0 or explicit_index >= len(self.entries):
            raise IndexError(explicit_index)
        return self.entries[explicit_index]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe topology metadata.

        Returns:
            Admission state, counts, and ordered ownership entries.
        """

        return {
            "active_bases_admitted": self.active_bases_admitted,
            "explicit_count": len(self.entries),
            "owned_count": len(self.owned_indices),
            "representative_count": len(self.representative_indices),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _independent_pair(
    ownership: TimeReversalOwnership,
    explicit_index: int,
    reason: str,
) -> TimeReversalOwnership:
    entry = ownership.entry_for(explicit_index)
    affected = {explicit_index}
    if entry.partner_index is not None:
        affected.add(entry.partner_index)
    updated = list(ownership.entries)
    for index in sorted(affected):
        current = updated[index]
        updated[index] = replace(
            current,
            owner_index=index,
            role="independent",
            aggregated_weight=current.original_weight,
            _time_reversal_permutation=None,
            fallback_reason=reason,
        )
    return TimeReversalOwnership(
        tuple(updated),
        active_bases_admitted=ownership.active_bases_admitted,
    )


def build_time_reversal_ownership(kpoint_mesh: KPointMesh) -> TimeReversalOwnership:
    """Build deterministic reduced-coordinate owner/partner topology.

    Geometry is matched modulo integer reciprocal-lattice shifts. Missing or
    unequal-weight partners become independent lanes, while duplicate or
    multiply claimed maps fail closed. Active-basis permutations are admitted
    separately after bases exist.

    Args:
        kpoint_mesh: Explicit weighted reduced-coordinate mesh.

    Returns:
        Ordered topology without active-basis permutations.

    Raises:
        ValueError: If input is non-finite, non-reduced, duplicated, or has an
            ambiguous/multiply-claimed time-reversal map.
    """

    points = kpoint_mesh.points
    vectors: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for index, point in enumerate(points):
        vector = tuple(float(value) for value in point.vector)
        weight = float(point.weight)
        if point.coordinate_system != "reduced":
            msg = "time-reversal ownership requires reduced-coordinate k-points"
            raise ValueError(msg)
        if not np.isfinite(np.asarray(vector, dtype=np.float64)).all():
            msg = f"k-point {index} has non-finite reduced coordinates"
            raise ValueError(msg)
        if not np.isfinite(weight) or weight <= 0.0:
            msg = f"k-point {index} has a non-finite or non-positive weight"
            raise ValueError(msg)
        vectors.append(vector)
        weights.append(weight)

    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            difference = tuple(
                left - right
                for left, right in zip(vectors[first], vectors[second], strict=True)
            )
            if _reciprocal_shift_if_close(difference) is not None:
                msg = (
                    "ambiguous duplicate k-points modulo the reciprocal lattice: "
                    f"{first} and {second}"
                )
                raise ValueError(msg)

    partner_candidates: list[list[tuple[int, tuple[int, int, int]]]] = []
    for index, vector in enumerate(vectors):
        candidates = []
        for candidate_index, candidate in enumerate(vectors):
            shift = _time_reversal_shift(vector, candidate)
            if shift is not None:
                candidates.append((candidate_index, shift))
        if len(candidates) > 1:
            msg = f"k-point {index} has multiple time-reversal partners"
            raise ValueError(msg)
        partner_candidates.append(candidates)

    for index, candidates in enumerate(partner_candidates):
        if not candidates:
            continue
        partner_index, _ = candidates[0]
        reverse = partner_candidates[partner_index]
        if len(reverse) != 1 or reverse[0][0] != index:
            msg = f"k-point {index} is multiply claimed by the partner map"
            raise ValueError(msg)

    entries: list[TimeReversalOwnershipEntry | None] = [None] * len(points)
    for index in range(len(points)):
        if entries[index] is not None:
            continue
        candidates = partner_candidates[index]
        if not candidates:
            entries[index] = TimeReversalOwnershipEntry(
                explicit_index=index,
                reduced_kpoint=vectors[index],
                original_weight=weights[index],
                owner_index=index,
                partner_index=None,
                role="independent",
                aggregated_weight=weights[index],
                reciprocal_shift=None,
                fallback_reason="missing_time_reversal_partner",
            )
            continue
        partner_index, shift = candidates[0]
        if partner_index == index:
            entries[index] = TimeReversalOwnershipEntry(
                explicit_index=index,
                reduced_kpoint=vectors[index],
                original_weight=weights[index],
                owner_index=index,
                partner_index=index,
                role="owner",
                aggregated_weight=weights[index],
                reciprocal_shift=shift,
            )
            continue
        if not np.isclose(
            weights[index],
            weights[partner_index],
            rtol=_TIME_REVERSAL_WEIGHT_RTOL,
            atol=_TIME_REVERSAL_WEIGHT_ATOL,
        ):
            for affected, counterpart in (
                (index, partner_index),
                (partner_index, index),
            ):
                entries[affected] = TimeReversalOwnershipEntry(
                    explicit_index=affected,
                    reduced_kpoint=vectors[affected],
                    original_weight=weights[affected],
                    owner_index=affected,
                    partner_index=counterpart,
                    role="independent",
                    aggregated_weight=weights[affected],
                    reciprocal_shift=shift,
                    fallback_reason="unequal_time_reversal_weight",
                )
            continue
        owner_index = min(index, partner_index)
        aggregated_weight = weights[index] + weights[partner_index]
        for affected, counterpart in (
            (index, partner_index),
            (partner_index, index),
        ):
            entries[affected] = TimeReversalOwnershipEntry(
                explicit_index=affected,
                reduced_kpoint=vectors[affected],
                original_weight=weights[affected],
                owner_index=owner_index,
                partner_index=counterpart,
                role="owner" if affected == owner_index else "partner",
                aggregated_weight=aggregated_weight,
                reciprocal_shift=shift,
            )

    return TimeReversalOwnership(tuple(entry for entry in entries if entry is not None))


def _active_time_reversal_permutation(
    source_basis: Any,
    target_basis: Any,
    shift: tuple[int, int, int],
) -> np.ndarray | None:
    if (
        source_basis.reciprocal_grid.fingerprint
        != target_basis.reciprocal_grid.fingerprint
        or source_basis.cutoff_hartree != target_basis.cutoff_hartree
    ):
        return None
    source = source_basis._layout._active_integer_g_np
    target = target_basis._layout._active_integer_g_np
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return None
    target_by_g = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(target)
    }
    if len(target_by_g) != target.shape[0]:
        return None
    permutation = []
    shift_array = np.asarray(shift, dtype=np.int64)
    for row in source:
        signed = tuple(int(value) for value in (-row - shift_array))
        mapped = target_by_g.get(signed)
        if mapped is None:
            return None
        permutation.append(mapped)
    values = np.asarray(permutation, dtype=np.int32)
    if not np.array_equal(np.sort(values), np.arange(values.size, dtype=np.int32)):
        return None
    mapped_target = mx.take(
        target_basis._layout._active_shifted_vectors,
        mx.array(values),
        axis=0,
    )
    maximum_vector_error = mx.max(
        mx.abs(mapped_target + source_basis._layout._active_shifted_vectors)
    )
    if float(maximum_vector_error) > 2e-6:
        return None
    return np.frombuffer(values.tobytes(), dtype=np.int32)


def admit_time_reversal_bases(
    ownership: TimeReversalOwnership,
    bases: Sequence[Any],
) -> TimeReversalOwnership:
    """Admit exact signed-``G`` permutations for active compact bases.

    Args:
        ownership: Geometry/weight topology built before basis construction.
        bases: One compact plane-wave basis per explicit point.

    Returns:
        Topology with read-only source-to-partner permutations. Only pairs whose
        active bases are exact one-to-one time reversals remain reused.

    Raises:
        ValueError: If the basis count differs or admission was already run.
    """

    if ownership.active_bases_admitted:
        msg = "time-reversal active bases were already admitted"
        raise ValueError(msg)
    if len(bases) != len(ownership.entries):
        msg = "active basis count must match the explicit k-point topology"
        raise ValueError(msg)
    visited: set[int] = set()
    updated = list(ownership.entries)
    for entry in ownership.entries:
        index = entry.explicit_index
        if index in visited or entry.role == "independent":
            continue
        partner_index = entry.partner_index
        if partner_index is None:
            continue
        partner_entry = ownership.entry_for(partner_index)
        shift = entry.reciprocal_shift
        if shift is None:
            fallback = _independent_pair(
                TimeReversalOwnership(tuple(updated)),
                index,
                "active_basis_time_reversal_mismatch",
            )
            updated = list(fallback.entries)
            visited.update({index, partner_index})
            continue
        forward = _active_time_reversal_permutation(
            bases[index],
            bases[partner_index],
            shift,
        )
        reverse_shift = partner_entry.reciprocal_shift
        reverse = (
            None
            if reverse_shift is None
            else _active_time_reversal_permutation(
                bases[partner_index],
                bases[index],
                reverse_shift,
            )
        )
        inverse_ok = (
            forward is not None
            and reverse is not None
            and np.array_equal(reverse[forward], np.arange(forward.size))
        )
        if not inverse_ok:
            fallback = _independent_pair(
                TimeReversalOwnership(tuple(updated)),
                index,
                "active_basis_time_reversal_mismatch",
            )
            updated = list(fallback.entries)
        else:
            updated[index] = replace(
                updated[index],
                _time_reversal_permutation=forward,
            )
            if partner_index != index:
                updated[partner_index] = replace(
                    updated[partner_index],
                    _time_reversal_permutation=reverse,
                )
        visited.update({index, partner_index})
    return TimeReversalOwnership(tuple(updated), active_bases_admitted=True)


@dataclass(frozen=True)
class KPoint:
    """One reciprocal-space k point.

    Args:
        vector: Three-component k-point vector. Cartesian vectors are in the
            same reciprocal units as `ReciprocalGrid.vectors`; reduced vectors
            are fractional diagnostic coordinates and are not accepted by
            Hamiltonian evaluation.
        weight: Positive integration weight. Defaults to ``1.0``.
        label: Optional display label such as ``"Γ"``. Defaults to ``None``.
        coordinate_system: Either ``"cartesian"`` or ``"reduced"``. Defaults
            to ``"cartesian"``.
    """

    vector: tuple[float, float, float]
    weight: float = 1.0
    label: str | None = None
    coordinate_system: str = "cartesian"

    def __init__(
        self,
        vector: Sequence[float],
        *,
        weight: float = 1.0,
        label: str | None = None,
        coordinate_system: str = "cartesian",
    ):
        if len(vector) != 3:
            msg = "k-point vector must have three components"
            raise ValueError(msg)
        parsed_vector = tuple(float(value) for value in vector)
        parsed_weight = float(weight)
        if not np.isfinite(np.asarray(parsed_vector, dtype=np.float64)).all():
            msg = "k-point vector components must be finite"
            raise ValueError(msg)
        if not np.isfinite(parsed_weight) or parsed_weight <= 0.0:
            msg = "k-point weight must be finite and positive"
            raise ValueError(msg)
        if coordinate_system not in {"cartesian", "reduced"}:
            msg = "coordinate_system must be 'cartesian' or 'reduced'"
            raise ValueError(msg)
        object.__setattr__(self, "vector", parsed_vector)
        object.__setattr__(self, "weight", parsed_weight)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coordinate_system", coordinate_system)

    @classmethod
    def gamma(cls) -> KPoint:
        """Return the Γ point."""

        return cls((0.0, 0.0, 0.0), label="Γ")

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""

        return {
            "vector": list(self.vector),
            "weight": self.weight,
            "label": self.label,
            "coordinate_system": self.coordinate_system,
        }


@dataclass(frozen=True)
class KPointMesh:
    """Weighted k-point mesh."""

    points: tuple[KPoint, ...]

    def __init__(self, points: Sequence[KPoint]):
        if not points:
            msg = "KPointMesh requires at least one point"
            raise ValueError(msg)
        weight_sum = sum(point.weight for point in points)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            msg = "KPointMesh weights must have a finite positive sum"
            raise ValueError(msg)
        normalized = tuple(
            KPoint(
                point.vector,
                weight=point.weight / weight_sum,
                label=point.label,
                coordinate_system=point.coordinate_system,
            )
            for point in points
        )
        object.__setattr__(self, "points", normalized)

    @classmethod
    def gamma(cls) -> KPointMesh:
        """Return a one-point Γ mesh."""

        return cls([KPoint.gamma()])

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""

        return {"points": [point.to_dict() for point in self.points]}


@dataclass(frozen=True)
class MonkhorstPackGrid(KPointMesh):
    """Simple Γ-centered Monkhorst-Pack-style mesh."""

    size: tuple[int, int, int] = (1, 1, 1)

    def __init__(self, size: Sequence[int]):
        parsed = tuple(int(value) for value in size)
        if len(parsed) != 3 or any(value <= 0 for value in parsed):
            msg = "MonkhorstPackGrid size must contain three positive integers"
            raise ValueError(msg)
        points = []
        total = int(np.prod(parsed))
        for indices in np.ndindex(parsed):
            vector = tuple(
                (index - (count - 1) / 2.0) / count
                for index, count in zip(indices, parsed, strict=True)
            )
            points.append(KPoint(vector, weight=1.0 / total, coordinate_system="reduced"))
        KPointMesh.__init__(self, points)
        object.__setattr__(self, "size", parsed)


@dataclass(frozen=True)
class BandPath:
    """Explicit k-point path for non-SCF band diagnostics."""

    points: tuple[KPoint, ...]

    def __init__(self, points: Sequence[KPoint]):
        if not points:
            msg = "BandPath requires at least one point"
            raise ValueError(msg)
        object.__setattr__(self, "points", tuple(points))

    @classmethod
    def line(
        cls,
        start: Sequence[float],
        end: Sequence[float],
        *,
        count: int,
        start_label: str | None = None,
        end_label: str | None = None,
    ) -> BandPath:
        """Build a linear path between two k points."""

        if count <= 0:
            msg = "count must be positive"
            raise ValueError(msg)
        start_np = np.asarray(start, dtype=np.float64)
        end_np = np.asarray(end, dtype=np.float64)
        points = []
        for index in range(count):
            fraction = 0.0 if count == 1 else index / (count - 1)
            label = start_label if index == 0 else end_label if index == count - 1 else None
            points.append(KPoint((1.0 - fraction) * start_np + fraction * end_np, label=label))
        return cls(points)


@dataclass(frozen=True)
class BandStructureResult:
    """Non-SCF band energies along a path.

    Args:
        kpoints: Cartesian k-points evaluated in path order.
        eigenvalues: Eigenvalue array with shape ``(n_kpoints, n_bands)``.
        reused_density: Whether the calculation reused the SCF density without
            another SCF cycle.
        nonlocal_available: Whether ion-backed nonlocal projector metadata was
            available on the system.
        nonlocal_applied: Whether nonlocal projectors were applied to the band
            Hamiltonian.
        nonlocal_projector_count: Number of projector channels included when
            nonlocal projectors were applied.
    """

    kpoints: tuple[KPoint, ...]
    eigenvalues: mx.array
    reused_density: bool
    nonlocal_available: bool = False
    nonlocal_applied: bool = False
    nonlocal_projector_count: int = 0

    def to_dict(self) -> dict:
        """Return JSON-safe band data."""

        return {
            "kpoints": [point.to_dict() for point in self.kpoints],
            "eigenvalues": np.array(self.eigenvalues).tolist(),
            "reused_density": self.reused_density,
            "nonlocal_available": self.nonlocal_available,
            "nonlocal_applied": self.nonlocal_applied,
            "nonlocal_projector_count": self.nonlocal_projector_count,
        }


def _is_gamma(point: KPoint, *, atol: float = 1e-12) -> bool:
    return all(abs(component) <= atol for component in point.vector)


def _validate_cartesian_band_path(band_path: BandPath) -> None:
    for point in band_path.points:
        if point.coordinate_system != "cartesian":
            msg = "run_band_structure requires cartesian k-points"
            raise ValueError(msg)


def run_band_structure(
    system: DFTSystem,
    scf_result: SCFResult,
    band_path: BandPath,
    *,
    n_bands: int = 1,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    apply_nonlocal: bool | None = None,
) -> BandStructureResult:
    """Evaluate non-SCF bands on top of a converged density.

    Args:
        system: DFT system that supplied the SCF density.
        scf_result: Converged or diagnostic SCF result whose density is reused.
        band_path: Explicit k-point path.
        n_bands: Number of eigenvalues to report at each k-point. Defaults to ``1``.
        xc_functional: Exchange-correlation functional for the fixed-density operator;
            ``None`` uses LDA. Defaults to ``None``.
        apply_nonlocal: Whether to include ion-backed nonlocal pseudopotential
            projectors. ``None`` mirrors ``scf_result.nonlocal_applied``. Defaults to
            ``None``.

    Returns:
        Non-SCF band energies and pseudopotential diagnostics.
    """

    if n_bands <= 0:
        msg = "n_bands must be positive"
        raise ValueError(msg)
    if n_bands > system.grid.size:
        msg = "n_bands cannot exceed the real-space grid size"
        raise ValueError(msg)
    _validate_cartesian_band_path(band_path)
    v_local = system.pseudopotential.field(system.grid)
    nonlocal_available = bool(system.ions is not None and system.ions.nonlocal_available)
    should_apply_nonlocal = (
        scf_result.nonlocal_applied if apply_nonlocal is None else bool(apply_nonlocal)
    )
    if should_apply_nonlocal and nonlocal_available and any(
        not _is_gamma(point) for point in band_path.points
    ):
        msg = "nonlocal band diagnostics are currently limited to Γ-point paths"
        raise ValueError(msg)
    nonlocal_operator = None
    nonlocal_projector_count = 0
    if should_apply_nonlocal and nonlocal_available and system.ions is not None:
        nonlocal_operator = NonlocalPseudopotentialOperator.from_ions(system.ions, system.grid)
        nonlocal_projector_count = nonlocal_operator.projectors.count
    nonlocal_applied = bool(nonlocal_operator is not None and nonlocal_operator.available)
    values = []
    for point in band_path.points:
        operator = KohnShamOperator.from_density(
            system.grid,
            v_local,
            scf_result.density,
            xc_functional=xc_functional,
            nonlocal_operator=nonlocal_operator,
            kpoint=point.vector,
        )
        diagonalized = DenseHamiltonianReference(operator).diagonalize(n_bands)
        values.append(np.array(diagonalized.eigenvalues, dtype=np.float32))
    return BandStructureResult(
        kpoints=band_path.points,
        eigenvalues=mx.array(np.stack(values)),
        reused_density=True,
        nonlocal_available=nonlocal_available,
        nonlocal_applied=nonlocal_applied,
        nonlocal_projector_count=nonlocal_projector_count,
    )
