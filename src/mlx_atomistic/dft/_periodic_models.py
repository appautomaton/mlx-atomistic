"""Data contracts for periodic plane-wave DFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.dft._compact import (
    _CompactBatch,
    _CompactBatchPolicy,
    _CompactLaneState,
    _CompatibilityCoefficientState,
)
from mlx_atomistic.dft._pseudopotential_identity import _pseudopotential_fingerprint
from mlx_atomistic.dft._runtime_observer import RuntimeObserver, add_observed_work
from mlx_atomistic.dft.grids import RealSpaceGrid
from mlx_atomistic.dft.kpoints import KPoint, TimeReversalOwnership
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import PseudopotentialData

if TYPE_CHECKING:
    from mlx_atomistic.dft._periodic_state import _PeriodicSCFContinuationState


def _eigensolve_provenance() -> dict[str, str]:
    return {
        "full_grid_precision": "complex64/float32",
        "projected_eigensolve_device": "cpu",
        "projected_eigensolve_backend": "numpy-lapack-cpu-complex128",
        "projected_eigensolve_precision": "complex128",
        "projected_eigensolve_output_precision": "float32/complex64",
    }


def _is_finite_positive_control(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _time_reversed_compact_values(
    values: mx.array,
    permutation: np.ndarray,
) -> mx.array:
    """Map source compact coefficients into target time-reversal order."""

    mapping = np.asarray(permutation, dtype=np.int32)
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(mapping.size, dtype=np.int32)
    return mx.take(
        mx.conjugate(values),
        mx.array(inverse),
        axis=1,
    ).astype(mx.complex64)


@dataclass(frozen=True)
class PeriodicDFTSystem:
    """Periodic DFT system with ordered per-ion pseudopotentials.

    Args:
        cell_lengths: A periodic `Cell`, three orthorhombic lengths, or a full
            row-vector cell matrix in bohr.
        grid_shape: FFT grid shape.
        positions: Ionic Cartesian positions in bohr.
        pseudopotential: Shared pseudopotential for every ion. Mutually
            exclusive with ``pseudopotentials``.
        electron_count: Total valence electron count. Defaults to the neutral
            pseudopotential charge sum.
        pseudopotentials: Ordered one-per-ion pseudopotentials.
    """

    grid: RealSpaceGrid
    positions: np.ndarray
    pseudopotentials: tuple[PseudopotentialData, ...]
    electron_count: float

    def __init__(
        self,
        cell_lengths: Cell | Sequence[float] | Sequence[Sequence[float]],
        grid_shape: Sequence[int],
        positions: Sequence[Sequence[float]],
        pseudopotential: PseudopotentialData | None = None,
        electron_count: float | None = None,
        *,
        pseudopotentials: Sequence[PseudopotentialData] | None = None,
    ):
        positions_np = np.asarray(positions, dtype=np.float64)
        if positions_np.ndim != 2 or positions_np.shape[1] != 3 or positions_np.shape[0] == 0:
            msg = "positions must have shape (n_ions, 3)"
            raise ValueError(msg)
        if not np.isfinite(positions_np).all():
            raise ValueError("positions must contain only finite values")
        if (pseudopotential is None) == (pseudopotentials is None):
            msg = "provide exactly one of pseudopotential or pseudopotentials"
            raise ValueError(msg)
        if pseudopotentials is None:
            if not isinstance(pseudopotential, PseudopotentialData):
                msg = "pseudopotential must be PseudopotentialData"
                raise TypeError(msg)
            per_ion = (pseudopotential,) * int(positions_np.shape[0])
        else:
            per_ion = tuple(pseudopotentials)
            if len(per_ion) != int(positions_np.shape[0]):
                msg = "pseudopotentials length must match the ion count"
                raise ValueError(msg)
            if any(not isinstance(value, PseudopotentialData) for value in per_ion):
                msg = "pseudopotentials must contain PseudopotentialData values"
                raise TypeError(msg)
        count = (
            float(sum(value.valence_charge for value in per_ion))
            if electron_count is None
            else float(electron_count)
        )
        if count <= 0.0:
            msg = "electron_count must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "grid", RealSpaceGrid(grid_shape, cell_lengths))
        object.__setattr__(self, "positions", positions_np)
        object.__setattr__(self, "pseudopotentials", per_ion)
        object.__setattr__(self, "electron_count", count)

    @property
    def pseudopotential(self) -> PseudopotentialData:
        """Return the shared pseudopotential of a homogeneous system.

        Raises:
            ValueError: If the system contains more than one pseudopotential.
        """

        if not self.is_homogeneous:
            msg = "multi-element systems do not have one shared pseudopotential"
            raise ValueError(msg)
        return self.pseudopotentials[0]

    @property
    def ion_count(self) -> int:
        """Number of ions in the periodic cell."""

        return int(self.positions.shape[0])

    @property
    def charges(self) -> tuple[float, ...]:
        """Valence point charges used by the periodic Ewald term."""

        return tuple(float(value.valence_charge) for value in self.pseudopotentials)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Element symbol assigned to every ion in position order."""

        return tuple(value.element for value in self.pseudopotentials)

    @property
    def is_homogeneous(self) -> bool:
        """Whether every ion uses an identical pseudopotential."""

        fingerprints = {_pseudopotential_fingerprint(value) for value in self.pseudopotentials}
        return len(fingerprints) == 1

    @property
    def fingerprint(self) -> str:
        """Stable identity of the periodic cell and per-ion Hamiltonian."""

        digest = sha256()
        digest.update(b"mlx-atomistic.periodic-system.v1\0")
        digest.update(np.asarray(self.grid.cell.matrix, dtype=np.float64).tobytes())
        digest.update(np.asarray(self.grid.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(self.positions, dtype=np.float64).tobytes())
        digest.update(np.asarray([self.electron_count], dtype=np.float64).tobytes())
        for value in self.pseudopotentials:
            digest.update(_pseudopotential_fingerprint(value).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def with_positions(
        self,
        positions: Sequence[Sequence[float]],
    ) -> PeriodicDFTSystem:
        """Return the same fixed-cell Hamiltonian identity at new ion positions.

        Args:
            positions: Replacement Cartesian ion positions in bohr.

        Returns:
            A new periodic system preserving cell, grid, pseudopotentials, and
            electron count.
        """

        return PeriodicDFTSystem(
            self.grid.cell,
            self.grid.shape,
            positions,
            electron_count=self.electron_count,
            pseudopotentials=self.pseudopotentials,
        )

    def with_cell(
        self,
        cell: Cell | Sequence[float] | Sequence[Sequence[float]],
        *,
        scale_positions: bool = True,
    ) -> PeriodicDFTSystem:
        """Return the system in a replacement periodic cell.

        Args:
            cell: Replacement right-handed periodic cell in bohr.
            scale_positions: Preserve fractional positions when true; otherwise
                preserve Cartesian positions.

        Returns:
            A new periodic system with fixed FFT shape, pseudopotentials, and
            electron count.

        Raises:
            ValueError: If ``scale_positions`` is not boolean.
        """

        if type(scale_positions) is not bool:
            raise ValueError("scale_positions must be bool")
        replacement = cell if isinstance(cell, Cell) else Cell(cell)
        positions = np.asarray(self.positions, dtype=np.float64)
        if scale_positions:
            direct = np.asarray(self.grid.cell.matrix, dtype=np.float64)
            fractional = positions @ np.linalg.inv(direct)
            positions = fractional @ np.asarray(replacement.matrix, dtype=np.float64)
        return PeriodicDFTSystem(
            replacement,
            self.grid.shape,
            positions,
            electron_count=self.electron_count,
            pseudopotentials=self.pseudopotentials,
        )


@dataclass(frozen=True)
class PeriodicDavidsonConfig:
    """Controls for the incremental block Davidson/Rayleigh-Ritz eigensolver."""

    max_iterations: int = 30
    tolerance: float = 1e-5
    max_subspace_size: int = 64
    preconditioner_floor: float = 0.25

    def __post_init__(self) -> None:
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            msg = "max_iterations must be a positive non-bool integer"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.tolerance):
            msg = "tolerance must be finite and positive"
            raise ValueError(msg)
        if type(self.max_subspace_size) is not int or self.max_subspace_size <= 1:
            msg = "max_subspace_size must be a non-bool integer exceeding one"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.preconditioner_floor):
            msg = "preconditioner_floor must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicFermiDiracSmearing:
    """Finite-temperature occupations for periodic metallic SCF.

    Args:
        width_hartree: Electronic temperature ``k_B T`` in Hartree.
    """

    width_hartree: float

    def __post_init__(self) -> None:
        if not _is_finite_positive_control(self.width_hartree):
            msg = "width_hartree must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicCollinearSpinConfig:
    """Controls for collinear spin-polarized periodic SCF.

    Args:
        mode: Electron-allocation mode. ``fixed_magnetization`` preserves a
            caller-supplied total moment, while ``unconstrained`` resolves one
            shared Fermi level across both spin channels.
        magnetization: Fixed electron-count difference ``N_up - N_down``.
            Required only for ``fixed_magnetization``.
        initial_magnetization: Optional initial electron-count difference for
            unconstrained SCF. It seeds symmetry breaking without constraining
            the converged moment.
        magnetization_mixing_beta: Linear damping applied to the signed
            magnetization density after charge-density mixing.
    """

    mode: str = "fixed_magnetization"
    magnetization: float | None = 0.0
    initial_magnetization: float | None = None
    magnetization_mixing_beta: float = 0.2

    def __post_init__(self) -> None:
        if self.mode not in {"fixed_magnetization", "unconstrained"}:
            msg = "spin mode must be 'fixed_magnetization' or 'unconstrained'"
            raise ValueError(msg)
        if self.mode == "fixed_magnetization":
            if self.magnetization is None or not np.isfinite(float(self.magnetization)):
                msg = "fixed_magnetization mode requires a finite magnetization"
                raise ValueError(msg)
            if self.initial_magnetization is not None:
                msg = "fixed_magnetization mode does not accept a separate initial seed"
                raise ValueError(msg)
        elif self.magnetization is not None:
            msg = "unconstrained spin mode does not accept a fixed magnetization"
            raise ValueError(msg)
        if self.initial_magnetization is not None and not np.isfinite(
            float(self.initial_magnetization)
        ):
            msg = "initial_magnetization must be finite when supplied"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.magnetization_mixing_beta) or (
            float(self.magnetization_mixing_beta) > 1.0
        ):
            msg = "magnetization_mixing_beta must lie in (0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicSCFConfig:
    """Controls for weighted k-point self-consistent field iteration."""

    max_iterations: int = 40
    density_tolerance: float = 1e-5
    energy_tolerance: float = 1e-6
    orbital_tolerance: float = 1e-5
    min_iterations: int = 2
    mixing_beta: float = 0.35
    mixer: str = "diis"
    davidson: PeriodicDavidsonConfig = field(default_factory=PeriodicDavidsonConfig)
    kpoint_batch_size: int = 8
    max_batch_padding_fraction: float = _CompactBatch._DEFAULT_MAX_PADDING_FRACTION
    max_batch_transient_bytes: int = _CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES
    hpsi_shape_policy: str = "finite-buckets"
    adaptive_eigensolver_tolerance: bool = False
    initial_eigensolver_tolerance: float = 1e-2
    eigensolver_tolerance_scale: float = 0.1
    smearing: PeriodicFermiDiracSmearing | None = None
    spin: PeriodicCollinearSpinConfig | None = None

    def __post_init__(self) -> None:
        self._validate_iteration_controls()
        self._validate_eigensolver_controls()
        if self.smearing is not None and not isinstance(
            self.smearing,
            PeriodicFermiDiracSmearing,
        ):
            msg = "smearing must be PeriodicFermiDiracSmearing or None"
            raise TypeError(msg)
        if self.spin is not None and not isinstance(
            self.spin,
            PeriodicCollinearSpinConfig,
        ):
            msg = "spin must be PeriodicCollinearSpinConfig or None"
            raise TypeError(msg)
        if (
            self.spin is not None
            and self.spin.mode == "unconstrained"
            and self.smearing is None
        ):
            msg = "unconstrained collinear spin requires Fermi-Dirac smearing"
            raise ValueError(msg)
        self._compact_batch_policy()

    def _validate_iteration_controls(self) -> None:
        if self.max_iterations <= 0:
            msg = "max_iterations must be positive"
            raise ValueError(msg)
        if self.density_tolerance <= 0.0 or self.energy_tolerance <= 0.0:
            msg = "SCF tolerances must be positive"
            raise ValueError(msg)
        if self.orbital_tolerance <= 0.0:
            msg = "orbital_tolerance must be positive"
            raise ValueError(msg)
        if self.min_iterations <= 0:
            msg = "min_iterations must be positive"
            raise ValueError(msg)
        if not 0.0 < self.mixing_beta <= 1.0:
            msg = "mixing_beta must lie in (0, 1]"
            raise ValueError(msg)
        if self.mixer not in {"linear", "diis"}:
            msg = "mixer must be 'linear' or 'diis'"
            raise ValueError(msg)

    def _validate_eigensolver_controls(self) -> None:
        if type(self.adaptive_eigensolver_tolerance) is not bool:
            msg = "adaptive_eigensolver_tolerance must be a bool"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.initial_eigensolver_tolerance):
            msg = "initial_eigensolver_tolerance must be finite and positive"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.eigensolver_tolerance_scale):
            msg = "eigensolver_tolerance_scale must be finite and positive"
            raise ValueError(msg)
        if (
            self.adaptive_eigensolver_tolerance
            and self.initial_eigensolver_tolerance < self.davidson.tolerance
        ):
            msg = (
                "initial_eigensolver_tolerance must not be tighter than the "
                "final Davidson tolerance"
            )
            raise ValueError(msg)

    def _compact_batch_policy(self) -> _CompactBatchPolicy:
        """Return the validated compact execution policy for this SCF run."""

        return _CompactBatchPolicy(
            batch_cap=self.kpoint_batch_size,
            max_padding_fraction=self.max_batch_padding_fraction,
            max_transient_bytes=self.max_batch_transient_bytes,
            shape_policy=self.hpsi_shape_policy,
        )

    def batch_policy(self) -> dict[str, int | float | str | list[int]]:
        """Return the exact bounded compact-batch policy."""

        policy = self._compact_batch_policy()
        return {
            "kpoint_batch_size": policy.batch_cap,
            "max_batch_padding_fraction": float(policy.max_padding_fraction),
            "max_batch_transient_bytes": policy.max_transient_bytes,
            "hpsi_shape_policy": policy.shape_policy,
            "hpsi_lane_capacity_buckets": list(policy.LANE_CAPACITY_BUCKETS),
            "hpsi_vector_capacity_buckets": list(policy.VECTOR_CAPACITY_BUCKETS),
        }


@dataclass(frozen=True, init=False)
class PeriodicEigenResult:
    """Lowest eigenspace result with compact runtime-owned coefficients.

    Public construction accepts full-grid coefficients only with an explicit
    basis. Runtime code uses the private compact factory, so no dense fallback
    is retained.

    Args:
        eigenvalues: Lowest eigenvalues in Hartree.
        coefficients: Public full-grid coefficient stack.
        residuals: Direct ``H(X) - epsilon X`` norm per returned eigenpair.
        orthonormality_error: Maximum overlap error.
        iterations: Davidson iteration count.
        converged: Whether the requested tolerance was reached.
        subspace_size: Final Davidson subspace width.
        restart_count: Number of Davidson restarts.
        basis: Optional basis used to pack the public full-grid coefficient
            input. When omitted, the legacy eight-argument constructor stores
            only the input's exact nonzero support for round-trip compatibility.
    """

    eigenvalues: mx.array
    _compact_coefficients: _CompactLaneState | _CompatibilityCoefficientState | None
    _basis: PlaneWaveBasis | None
    _time_reversal_owner: PeriodicEigenResult | None
    _time_reversal_permutation: np.ndarray | None
    _time_reversal_observer: RuntimeObserver | None
    residuals: mx.array
    orthonormality_error: float
    iterations: int
    converged: bool
    subspace_size: int
    restart_count: int

    def __init__(
        self,
        eigenvalues: mx.array,
        coefficients: mx.array,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
        *,
        basis: PlaneWaveBasis | None = None,
    ) -> None:
        compact: _CompactLaneState | _CompatibilityCoefficientState
        if basis is None:
            compact = _CompatibilityCoefficientState.from_full(coefficients)
        else:
            compact, _ = basis._state_from_full(coefficients)
        self._set_fields(
            eigenvalues=eigenvalues,
            compact_coefficients=compact,
            basis=basis,
            residuals=residuals,
            orthonormality_error=orthonormality_error,
            iterations=iterations,
            converged=converged,
            subspace_size=subspace_size,
            restart_count=restart_count,
        )

    def _set_fields(
        self,
        *,
        eigenvalues: mx.array,
        compact_coefficients: _CompactLaneState | _CompatibilityCoefficientState,
        basis: PlaneWaveBasis | None,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
    ) -> None:
        if basis is None:
            if not isinstance(compact_coefficients, _CompatibilityCoefficientState):
                msg = "basis-free public results require compatibility coefficient state"
                raise ValueError(msg)
        else:
            if not isinstance(compact_coefficients, _CompactLaneState):
                msg = "basis-bound results require compact lane state"
                raise ValueError(msg)
            basis._validate_state(compact_coefficients)
            if compact_coefficients.kind != "coefficients":
                msg = "periodic eigen results must own coefficient state"
                raise ValueError(msg)
        object.__setattr__(self, "eigenvalues", mx.array(eigenvalues))
        object.__setattr__(self, "_compact_coefficients", compact_coefficients)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "residuals", mx.array(residuals))
        object.__setattr__(self, "orthonormality_error", float(orthonormality_error))
        object.__setattr__(self, "iterations", int(iterations))
        object.__setattr__(self, "converged", bool(converged))
        object.__setattr__(self, "subspace_size", int(subspace_size))
        object.__setattr__(self, "restart_count", int(restart_count))
        object.__setattr__(self, "_time_reversal_owner", None)
        object.__setattr__(self, "_time_reversal_permutation", None)
        object.__setattr__(self, "_time_reversal_observer", None)

    @classmethod
    def _from_compact(
        cls,
        *,
        eigenvalues: mx.array,
        compact_coefficients: _CompactLaneState,
        basis: PlaneWaveBasis,
        residuals: mx.array,
        orthonormality_error: float,
        iterations: int,
        converged: bool,
        subspace_size: int,
        restart_count: int,
    ) -> PeriodicEigenResult:
        result = object.__new__(cls)
        result._set_fields(
            eigenvalues=eigenvalues,
            compact_coefficients=compact_coefficients,
            basis=basis,
            residuals=residuals,
            orthonormality_error=orthonormality_error,
            iterations=iterations,
            converged=converged,
            subspace_size=subspace_size,
            restart_count=restart_count,
        )
        return result

    @classmethod
    def _from_time_reversal_owner(
        cls,
        *,
        owner: PeriodicEigenResult,
        partner_basis: PlaneWaveBasis,
        permutation: np.ndarray,
        observer: RuntimeObserver | None,
    ) -> PeriodicEigenResult:
        owner_state = owner._compact_coefficients
        if not isinstance(owner_state, _CompactLaneState) or owner._basis is None:
            msg = "time-reversal views require a compact basis-bound owner"
            raise ValueError(msg)
        mapping = np.array(permutation, dtype=np.int32, copy=True)
        if (
            mapping.shape != (owner_state.layout.active_count,)
            or partner_basis.active_count != mapping.size
            or not np.array_equal(
                np.sort(mapping),
                np.arange(mapping.size, dtype=np.int32),
            )
        ):
            msg = "time-reversal permutation must be a complete active-basis bijection"
            raise ValueError(msg)
        mapping.setflags(write=False)
        result = object.__new__(cls)
        eigenvalues = mx.array(owner.eigenvalues)
        residuals = mx.array(owner.residuals)
        mx.eval(eigenvalues, residuals)
        object.__setattr__(result, "eigenvalues", eigenvalues)
        object.__setattr__(result, "_compact_coefficients", None)
        object.__setattr__(result, "_basis", partner_basis)
        object.__setattr__(result, "residuals", residuals)
        object.__setattr__(
            result,
            "orthonormality_error",
            owner.orthonormality_error,
        )
        object.__setattr__(result, "iterations", owner.iterations)
        object.__setattr__(result, "converged", owner.converged)
        object.__setattr__(result, "subspace_size", owner.subspace_size)
        object.__setattr__(result, "restart_count", owner.restart_count)
        object.__setattr__(result, "_time_reversal_owner", owner)
        object.__setattr__(result, "_time_reversal_permutation", mapping)
        object.__setattr__(result, "_time_reversal_observer", observer)
        return result

    @property
    def is_time_reversal_view(self) -> bool:
        """Whether coefficients are an uncached time-reversed owner view."""

        return self._time_reversal_owner is not None

    @property
    def coefficients(self) -> mx.array:
        """Materialize a fresh full-grid coefficient stack.

        Returns:
            Caller-owned ``complex64`` coefficients with exact inactive zeros.
        """

        if self._time_reversal_owner is None:
            if self._compact_coefficients is None:
                msg = "periodic eigen result has no coefficient state"
                raise RuntimeError(msg)
            return self._compact_coefficients.full_grid_fresh()
        owner_state = self._time_reversal_owner._compact_coefficients
        if (
            not isinstance(owner_state, _CompactLaneState)
            or self._time_reversal_permutation is None
            or self._basis is None
        ):
            msg = "time-reversal owner state is unavailable"
            raise RuntimeError(msg)
        values = _time_reversed_compact_values(
            owner_state.values,
            self._time_reversal_permutation,
        )
        add_observed_work(
            self._time_reversal_observer,
            {"partner_reconstructions": 1},
        )
        return self._basis._layout.unpack_fresh(values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe eigensolver summary.

        Returns:
            Eigenvalues, residuals, convergence, and subspace diagnostics.
        """

        return {
            "eigenvalues_hartree": np.asarray(self.eigenvalues).tolist(),
            "residuals": np.asarray(self.residuals).tolist(),
            "orthonormality_error": self.orthonormality_error,
            "iterations": self.iterations,
            "converged": self.converged,
            "subspace_size": self.subspace_size,
            "restart_count": self.restart_count,
            "solver": "block-davidson-rayleigh-ritz",
            "dense_full_hamiltonian": False,
            **_eigensolve_provenance(),
            "full_grid_device": "default-mlx-device",
        }


@dataclass(frozen=True)
class PeriodicKPointResult:
    """One weighted k-point result in a periodic SCF calculation."""

    reduced_kpoint: tuple[float, float, float]
    weight: float
    basis: PlaneWaveBasis
    eigen: PeriodicEigenResult
    explicit_index: int | None = None
    aggregated_weight: float | None = None
    ownership_role: str = "independent"
    fallback_reason: str | None = None
    occupations: tuple[float, ...] | None = None

    @property
    def integration_weight(self) -> float:
        """Return the owner-aggregated or original integration weight."""

        return self.weight if self.aggregated_weight is None else self.aggregated_weight

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe k-point summary.

        Returns:
            Reduced k-point, weight, basis metadata, and eigensolver summary.
        """

        payload = {
            "reduced_kpoint": list(self.reduced_kpoint),
            "weight": self.weight,
            "basis": self.basis.to_dict(),
            "eigensolver": self.eigen.to_dict(),
        }
        if self.occupations is not None:
            payload["occupations"] = list(self.occupations)
        return payload


@dataclass(frozen=True)
class _TimeReversalContinuationSeed:
    owner_index: int


@dataclass(frozen=True)
class PeriodicSpinChannelResult:
    """One physical channel of a collinear periodic SCF result."""

    label: str
    electron_count: float
    density: mx.array
    kpoints: tuple[PeriodicKPointResult, ...]
    chemical_potential: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe spin-channel summary."""

        return {
            "label": self.label,
            "electron_count": self.electron_count,
            "chemical_potential_hartree": self.chemical_potential,
            "kpoints": [result.to_dict() for result in self.kpoints],
        }


@dataclass(frozen=True)
class PeriodicSCFResult:
    """Result bundle for a weighted periodic plane-wave SCF calculation.

    ``total_energy`` is the variational free energy when smearing is active and
    the internal energy otherwise. ``internal_energy`` always excludes the
    electronic entropy correction.
    """

    converged: bool
    status: str
    iterations: int
    total_energy: float
    electron_count: float
    density_residual: float
    energy_delta: float | None
    density: mx.array
    kpoints: tuple[PeriodicKPointResult, ...]
    energy_by_term: dict[str, float]
    history: tuple[dict[str, float | int | str | None], ...]
    timings: dict[str, float]
    time_reversal_ownership: TimeReversalOwnership | None = None
    point_group_symmetry_reduced: bool = False
    batch_policy: dict[str, int | float | str | list[int]] = field(default_factory=dict)
    numerical_status: str = "not_evaluated"
    resume_integrity_status: str = "fresh"
    timing_admission_status: str = "fresh"
    lineage: tuple[str, ...] = ()
    system_fingerprint: str | None = None
    internal_energy: float | None = None
    chemical_potential: float | None = None
    electronic_entropy: float = 0.0
    smearing_width_hartree: float | None = None
    spin_channels: tuple[PeriodicSpinChannelResult, ...] = ()
    integrated_magnetization: float | None = None
    magnetization_density: mx.array | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _owned_kpoints: tuple[PeriodicKPointResult, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _checkpoint_state: _PeriodicSCFContinuationState | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _artifact_execution_contract_fingerprint: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _artifact_calculation_fingerprint: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def owned_kpoints(self) -> tuple[PeriodicKPointResult, ...]:
        """Return the compact-state-owning k-point results.

        Legacy manually constructed results without ownership metadata return
        their explicit k-point tuple unchanged.
        """

        return self.kpoints if self._owned_kpoints is None else self._owned_kpoints

    @property
    def continuation_coefficients(self) -> tuple[object, ...]:
        """Return an explicit owner-aware initial-coefficient sequence.

        Owner and independent entries reference their compact state. Admitted
        partners use lightweight time-reversal descriptors, so constructing the
        sequence neither materializes nor retains partner coefficients.
        """

        if self.time_reversal_ownership is None:
            return tuple(item.eigen._compact_coefficients for item in self.kpoints)
        owned = {
            item.explicit_index: item.eigen._compact_coefficients for item in self.owned_kpoints
        }
        return tuple(
            owned[entry.explicit_index]
            if entry.owner_index == entry.explicit_index
            else _TimeReversalContinuationSeed(entry.owner_index)
            for entry in self.time_reversal_ownership.entries
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe periodic SCF summary.

        Returns:
            Convergence, energy, k-point, history, and timing diagnostics without
            dense orbital or density payloads.
        """

        return {
            "converged": self.converged,
            "status": self.status,
            "iterations": self.iterations,
            "total_energy_hartree": self.total_energy,
            "internal_energy_hartree": (
                self.total_energy if self.internal_energy is None else self.internal_energy
            ),
            "electron_count": self.electron_count,
            "chemical_potential_hartree": self.chemical_potential,
            "electronic_entropy": self.electronic_entropy,
            "smearing_width_hartree": self.smearing_width_hartree,
            "spin_channels": [channel.to_dict() for channel in self.spin_channels],
            "integrated_magnetization": self.integrated_magnetization,
            "density_residual": self.density_residual,
            "energy_delta_hartree": self.energy_delta,
            "kpoints": [result.to_dict() for result in self.kpoints],
            "energy_by_term_hartree": dict(self.energy_by_term),
            "history": [dict(row) for row in self.history],
            "timings_ms": dict(self.timings),
            "batch_policy": dict(self.batch_policy),
            "numerical_status": self.numerical_status,
            "resume_integrity_status": self.resume_integrity_status,
            "timing_admission_status": self.timing_admission_status,
            "point_group_symmetry_reduced": self.point_group_symmetry_reduced,
            "lineage": list(self.lineage),
            "system_fingerprint": self.system_fingerprint,
            "dense_full_hamiltonian": False,
        }


@dataclass(frozen=True)
class PeriodicFrozenDensity:
    """Portable fixed-density input for a periodic non-SCF calculation.

    Args:
        density: Converged electron density on ``system.grid``.
        cutoff_hartree: Plane-wave kinetic cutoff used by the source SCF.
        electron_count: Electron count represented by the density.
        system_fingerprint: Optional source-system identity. Required when the
            density is reused for a multi-element calculation.
    """

    density: mx.array
    cutoff_hartree: float
    electron_count: float
    system_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not _is_finite_positive_control(self.cutoff_hartree):
            msg = "cutoff_hartree must be finite and positive"
            raise ValueError(msg)
        if not _is_finite_positive_control(self.electron_count):
            msg = "electron_count must be finite and positive"
            raise ValueError(msg)
        object.__setattr__(self, "density", mx.real(mx.array(self.density)))
        object.__setattr__(self, "cutoff_hartree", float(self.cutoff_hartree))
        object.__setattr__(self, "electron_count", float(self.electron_count))
        if self.system_fingerprint is not None and (
            len(self.system_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.system_fingerprint)
        ):
            msg = "system_fingerprint must be a lowercase SHA-256 digest"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicBandPointResult:
    """One fixed-density periodic eigensolve along a band path.

    Args:
        requested_kpoint: Reduced-coordinate path point requested by the caller.
        basis: Point-specific cutoff-projected plane-wave basis.
        eigen: Lowest periodic eigenpairs returned by Davidson.
    """

    requested_kpoint: KPoint
    basis: PlaneWaveBasis
    eigen: PeriodicEigenResult

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe point summary."""

        return {
            "kpoint": self.requested_kpoint.to_dict(),
            "basis": self.basis.to_dict(),
            "eigensolver": self.eigen.to_dict(),
        }


@dataclass(frozen=True)
class PeriodicBandStructureResult:
    """Production non-self-consistent bands from one frozen periodic density.

    Args:
        kpoints: Evaluated reduced-coordinate path points.
        eigenvalues: Eigenvalues in Hartree with shape ``(n_kpoints, n_bands)``.
        residuals: Direct eigensolver residuals with the same shape.
        points: Basis and compact eigenspace retained for each path point.
        occupied_band_count: Number of doubly occupied bands in the source SCF.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        density_source: ``"scf_result"`` or ``"frozen_density"``.
        timings: Wall-clock phase timings in milliseconds.
        guard_band_count: Additional unpublished states used to stabilize the
            requested eigenspace boundary.
    """

    kpoints: tuple[KPoint, ...]
    eigenvalues: mx.array
    residuals: mx.array
    points: tuple[PeriodicBandPointResult, ...]
    occupied_band_count: int
    cutoff_hartree: float
    density_source: str
    timings: dict[str, float]
    guard_band_count: int = 0

    @property
    def n_bands(self) -> int:
        """Return the number of bands solved at every k-point."""

        return int(self.eigenvalues.shape[1])

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe bands without materializing orbital coefficients."""

        return {
            "kpoints": [point.to_dict() for point in self.kpoints],
            "eigenvalues_hartree": np.asarray(self.eigenvalues).tolist(),
            "residuals": np.asarray(self.residuals).tolist(),
            "points": [point.to_dict() for point in self.points],
            "occupied_band_count": self.occupied_band_count,
            "n_bands": self.n_bands,
            "cutoff_hartree": self.cutoff_hartree,
            "density_source": self.density_source,
            "guard_band_count": self.guard_band_count,
            "timings_ms": dict(self.timings),
            "reused_density": True,
            "self_consistency_iterations": 0,
            "dense_full_hamiltonian": False,
        }
