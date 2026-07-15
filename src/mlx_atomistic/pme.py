"""Standalone particle-mesh Ewald electrostatics for small periodic fixtures."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from math import ceil
from typing import Literal, cast

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell
from mlx_atomistic.neighbors import NeighborBlocks
from mlx_atomistic.nonbonded import (
    FORCE_EVALUATION_SCOPES,
    ForceScopeReport,
    normalize_force_scope,
)
from mlx_atomistic.runtime import ReadinessReport

PME_EXECUTION_BACKEND = "mlx_fft_cic"
PME_PRODUCTION_EXECUTABLE = True
PME_PRODUCTION_MAX_ATOMS = 4096
PME_SUPPORTED_ASSIGNMENT_ORDERS = (2, 4, 5)
PME_BACKGROUND_POLICIES = (
    "reject_non_neutral",
    "uniform_neutralizing_plasma",
)
PME_EXECUTION_PLAN_SCHEMA_VERSION = 1
PMEBackgroundPolicy = Literal[
    "reject_non_neutral",
    "uniform_neutralizing_plasma",
]


def normalize_pme_background_policy(value: object) -> PMEBackgroundPolicy:
    """Return a supported PME background-charge policy.

    Args:
        value: Policy name to normalize.

    Returns:
        A supported normalized policy string.

    Raises:
        ValueError: If ``value`` is not a supported policy.
    """

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in PME_BACKGROUND_POLICIES:
        choices = ", ".join(PME_BACKGROUND_POLICIES)
        msg = f"PME background_policy must be one of: {choices}"
        raise ValueError(msg)
    return cast(PMEBackgroundPolicy, normalized)


@dataclass(frozen=True)
class PMEConfig:
    """Controls for the standalone PME mesh backend."""

    mesh_shape: tuple[int, int, int] = (32, 32, 32)
    alpha: float = 0.35
    real_cutoff: float | None = None
    assignment_order: int = 2
    charge_tolerance: float = 1e-5
    deconvolve_assignment: bool = True
    background_policy: PMEBackgroundPolicy = "reject_non_neutral"

    def __post_init__(self) -> None:
        if len(self.mesh_shape) != 3:
            msg = "mesh_shape must contain exactly three dimensions"
            raise ValueError(msg)
        if any(int(size) != size or size < 4 for size in self.mesh_shape):
            msg = "mesh_shape dimensions must be integers >= 4"
            raise ValueError(msg)
        object.__setattr__(self, "mesh_shape", tuple(int(size) for size in self.mesh_shape))
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha <= 0.0:
            msg = "alpha must be finite and positive"
            raise ValueError(msg)
        object.__setattr__(self, "alpha", alpha)
        if self.real_cutoff is not None:
            real_cutoff = float(self.real_cutoff)
            if not np.isfinite(real_cutoff) or real_cutoff <= 0.0:
                msg = "real_cutoff must be finite and positive when provided"
                raise ValueError(msg)
            object.__setattr__(self, "real_cutoff", real_cutoff)
        charge_tolerance = float(self.charge_tolerance)
        if not np.isfinite(charge_tolerance) or charge_tolerance < 0.0:
            msg = "charge_tolerance must be finite and non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "charge_tolerance", charge_tolerance)
        object.__setattr__(
            self,
            "background_policy",
            normalize_pme_background_policy(self.background_policy),
        )
        object.__setattr__(
            self,
            "assignment_order",
            _validate_assignment_order(self.assignment_order),
        )


class PMEPlanMismatchError(ValueError):
    """Raised when a PME execution plan does not match an evaluation request."""

    def __init__(
        self,
        mismatches: tuple[str, ...],
        *,
        expected_fingerprint: str,
        actual_fingerprint: str,
    ) -> None:
        self.mismatches = mismatches
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        fields = ",".join(mismatches)
        super().__init__(
            "pme_execution_plan_mismatch:"
            f"fields={fields}:expected={expected_fingerprint}:actual={actual_fingerprint}"
        )


class PMEExecutionPlan:
    """Reusable cell-invariant reciprocal state for fixed-cell PME evaluation.

    Args:
        cell: Periodic orthorhombic cell whose reciprocal state is planned.
        config: PME parameters; ``None`` uses `PMEConfig` defaults.
        coulomb_constant: Coulomb prefactor in the configured unit system.
        dtype: Reciprocal-array dtype. The current backend supports ``float32``.
        backend: PME execution backend identifier. The current backend is
            `PME_EXECUTION_BACKEND`.

    Raises:
        ValueError: If the cell, Coulomb constant, dtype, or backend is unsupported.
    """

    def __init__(
        self,
        cell: Cell,
        *,
        config: PMEConfig | None = None,
        coulomb_constant: float = 1.0,
        dtype: object = mx.float32,
        backend: str = PME_EXECUTION_BACKEND,
    ) -> None:
        config = _resolve_plan_config(config)
        coulomb_constant = _validate_coulomb_constant(coulomb_constant)
        dtype_name = _normalize_plan_dtype(dtype)
        backend_name = _normalize_plan_backend(backend)
        if dtype_name != "float32":
            msg = "PME execution plans currently support dtype='float32' only"
            raise ValueError(msg)
        if backend_name != PME_EXECUTION_BACKEND:
            msg = (
                "PME execution plan backend is unsupported: "
                f"backend={backend_name!r},supported={PME_EXECUTION_BACKEND!r}"
            )
            raise ValueError(msg)
        cell_matrix, cell_lengths = _validated_plan_cell(cell)
        real_cutoff = _resolve_real_cutoff(config, cell_lengths)
        device = str(mx.default_device())
        signature = _plan_signature(
            cell_matrix=cell_matrix,
            config=config,
            real_cutoff=real_cutoff,
            coulomb_constant=coulomb_constant,
            dtype=dtype_name,
            backend=backend_name,
            device=device,
        )

        setup_started = time.perf_counter()
        influence, wavevectors, reciprocal_modes = _influence_function_mx(
            cell_lengths,
            config.mesh_shape,
            alpha=config.alpha,
            coulomb_constant=coulomb_constant,
            deconvolve_assignment=config.deconvolve_assignment,
            assignment_order=config.assignment_order,
        )
        mx.eval(influence, *wavevectors)
        setup_seconds = time.perf_counter() - setup_started

        self.cell = cell
        self.cell_matrix = tuple(tuple(float(value) for value in row) for row in cell_matrix)
        self.cell_lengths = tuple(float(value) for value in cell_lengths)
        self.config = config
        self.real_cutoff = float(real_cutoff)
        self.coulomb_constant = coulomb_constant
        self.dtype = dtype_name
        self.backend = backend_name
        self.device = device
        self.influence = influence
        self.wavevectors = wavevectors
        self.grid_size = int(np.prod(config.mesh_shape, dtype=np.int64))
        self.reciprocal_modes = reciprocal_modes
        self.fingerprint = _plan_fingerprint(signature)
        self.build_count = 1
        self.setup_seconds = float(setup_seconds)
        self.estimated_resident_bytes = _estimated_plan_resident_bytes(
            self.grid_size,
            dtype=dtype_name,
        )
        self.reuse_count = 0
        self.reuse_validation_seconds = 0.0
        self.last_reuse_validation_seconds = 0.0
        self._signature = signature

    @classmethod
    def build(
        cls,
        cell: Cell,
        *,
        config: PMEConfig | None = None,
        coulomb_constant: float = 1.0,
        dtype: object = mx.float32,
        backend: str = PME_EXECUTION_BACKEND,
    ) -> PMEExecutionPlan:
        """Build a reusable PME execution plan.

        Args:
            cell: Periodic orthorhombic cell whose reciprocal state is planned.
            config: PME parameters; ``None`` uses `PMEConfig` defaults.
            coulomb_constant: Coulomb prefactor in the configured unit system.
            dtype: Reciprocal-array dtype. The current backend supports ``float32``.
            backend: PME execution backend identifier.

        Returns:
            A materialized reusable execution plan.
        """

        return cls(
            cell,
            config=config,
            coulomb_constant=coulomb_constant,
            dtype=dtype,
            backend=backend,
        )

    def validate(
        self,
        cell: Cell,
        *,
        config: PMEConfig | None = None,
        coulomb_constant: float | None = None,
        dtype: object = mx.float32,
        backend: str = PME_EXECUTION_BACKEND,
    ) -> None:
        """Validate that requested PME inputs exactly match this plan.

        Args:
            cell: Cell requested by the evaluation.
            config: Requested PME configuration; ``None`` uses this plan's config.
            coulomb_constant: Requested Coulomb prefactor; ``None`` uses this plan's value.
            dtype: Requested reciprocal-array dtype.
            backend: Requested PME execution backend.

        Raises:
            PMEPlanMismatchError: If any plan-defining input differs.
            ValueError: If an input is invalid independently of the stored plan.
        """

        config = self.config if config is None else _resolve_plan_config(config)
        coulomb_constant = (
            self.coulomb_constant
            if coulomb_constant is None
            else _validate_coulomb_constant(coulomb_constant)
        )
        dtype_name = _normalize_plan_dtype(dtype)
        backend_name = _normalize_plan_backend(backend)
        cell_matrix, cell_lengths = _validated_plan_cell(cell)
        actual_signature = _plan_signature(
            cell_matrix=cell_matrix,
            config=config,
            real_cutoff=_resolve_real_cutoff(config, cell_lengths),
            coulomb_constant=coulomb_constant,
            dtype=dtype_name,
            backend=backend_name,
            device=str(mx.default_device()),
        )
        mismatches = tuple(
            key
            for key in self._signature
            if self._signature[key] != actual_signature.get(key)
        )
        if mismatches:
            raise PMEPlanMismatchError(
                mismatches,
                expected_fingerprint=self.fingerprint,
                actual_fingerprint=_plan_fingerprint(actual_signature),
            )

    def rebuild(
        self,
        *,
        cell: Cell | None = None,
        config: PMEConfig | None = None,
        coulomb_constant: float | None = None,
        dtype: object | None = None,
        backend: str | None = None,
    ) -> PMEExecutionPlan:
        """Build a distinct plan after an explicit defining-input change.

        Args:
            cell: Replacement cell; ``None`` retains the current cell.
            config: Replacement configuration; ``None`` retains the current config.
            coulomb_constant: Replacement Coulomb prefactor; ``None`` retains the current value.
            dtype: Replacement reciprocal dtype; ``None`` retains the current dtype.
            backend: Replacement execution backend; ``None`` retains the current backend.

        Returns:
            A new materialized execution plan with counters independent of this plan.
        """

        return PMEExecutionPlan(
            self.cell if cell is None else cell,
            config=self.config if config is None else config,
            coulomb_constant=(
                self.coulomb_constant if coulomb_constant is None else coulomb_constant
            ),
            dtype=self.dtype if dtype is None else dtype,
            backend=self.backend if backend is None else backend,
        )

    @property
    def diagnostics(self) -> dict[str, object]:
        """Return JSON-serializable setup and reuse diagnostics for this plan."""

        return {
            "schema_version": PME_EXECUTION_PLAN_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "build_count": self.build_count,
            "setup_seconds": self.setup_seconds,
            "reuse_count": self.reuse_count,
            "reuse_validation_seconds": self.reuse_validation_seconds,
            "last_reuse_validation_seconds": self.last_reuse_validation_seconds,
            "mesh_shape": self.config.mesh_shape,
            "grid_size": self.grid_size,
            "reciprocal_modes": self.reciprocal_modes,
            "estimated_resident_bytes": self.estimated_resident_bytes,
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "cell_lengths": self.cell_lengths,
            "alpha": self.config.alpha,
            "real_cutoff": self.real_cutoff,
            "assignment_order": self.config.assignment_order,
            "deconvolve_assignment": self.config.deconvolve_assignment,
            "coulomb_constant": self.coulomb_constant,
            "background_policy": self.config.background_policy,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the plan diagnostics as a plain JSON-serializable dict.

        Returns:
            Setup, fingerprint, resident-size, backend, and reuse metadata.
        """

        return self.diagnostics

    def _record_reuse(self, validation_seconds: float) -> None:
        self.reuse_count += 1
        self.last_reuse_validation_seconds = float(validation_seconds)
        self.reuse_validation_seconds += float(validation_seconds)


@dataclass(frozen=True)
class PMEDiagnostics:
    """Diagnostics emitted by one standalone PME evaluation."""

    mesh_shape: tuple[int, int, int]
    assignment_order: int
    alpha: float
    real_cutoff: float
    net_charge: float
    volume: float
    charge_grid_sum: float
    reciprocal_modes: int
    max_charge_grid_abs: float
    background_policy: PMEBackgroundPolicy = "reject_non_neutral"
    background_energy: float = 0.0
    direct_space_policy: str = "dense"
    direct_space_representation: str = "dense"
    direct_space_pair_count: int | None = None
    direct_space_candidate_count: int | None = None
    direct_space_fallback_reason: str | None = None
    plan_fingerprint: str | None = None
    plan_build_count: int = 0
    plan_reuse_count: int = 0
    plan_setup_seconds: float = 0.0
    plan_estimated_resident_bytes: int = 0
    plan_backend: str | None = None
    plan_device: str | None = None
    plan_dtype: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the diagnostics as a plain JSON-serializable dict.

        Returns:
            The diagnostics fields (mesh shape, alpha, cutoff, net charge, grid sums,
                direct-space policy, …) as a dict.
        """

        return {
            "mesh_shape": self.mesh_shape,
            "assignment_order": self.assignment_order,
            "alpha": self.alpha,
            "real_cutoff": self.real_cutoff,
            "net_charge": self.net_charge,
            "volume": self.volume,
            "charge_grid_sum": self.charge_grid_sum,
            "reciprocal_modes": self.reciprocal_modes,
            "max_charge_grid_abs": self.max_charge_grid_abs,
            "background_policy": self.background_policy,
            "background_energy": self.background_energy,
            "direct_space_policy": self.direct_space_policy,
            "direct_space_representation": self.direct_space_representation,
            "direct_space_pair_count": self.direct_space_pair_count,
            "direct_space_candidate_count": self.direct_space_candidate_count,
            "direct_space_fallback_reason": self.direct_space_fallback_reason,
            "plan_fingerprint": self.plan_fingerprint,
            "plan_build_count": self.plan_build_count,
            "plan_reuse_count": self.plan_reuse_count,
            "plan_setup_seconds": self.plan_setup_seconds,
            "plan_estimated_resident_bytes": self.plan_estimated_resident_bytes,
            "plan_backend": self.plan_backend,
            "plan_device": self.plan_device,
            "plan_dtype": self.plan_dtype,
        }


@dataclass(frozen=True)
class PMEDirectSpacePolicyReport:
    """Execution policy selected for PME real/direct-space evaluation."""

    policy: str
    representation: str
    uses_shared_neighbor_policy: bool
    supported: bool
    real_cutoff: float
    minimum_image_safe: bool
    pair_count: int | None = None
    compact_pair_count: int | None = None
    candidate_count: int | None = None
    candidate_waste_count: int | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the policy report as a plain JSON-serializable dict.

        Returns:
            The policy report fields (policy, representation, support, cutoff, pair
                counts, fallback reason) as a dict.
        """

        return {
            "policy": self.policy,
            "representation": self.representation,
            "uses_shared_neighbor_policy": self.uses_shared_neighbor_policy,
            "supported": self.supported,
            "real_cutoff": self.real_cutoff,
            "minimum_image_safe": self.minimum_image_safe,
            "pair_count": self.pair_count,
            "compact_pair_count": self.compact_pair_count,
            "candidate_count": self.candidate_count,
            "candidate_waste_count": self.candidate_waste_count,
            "fallback_reason": self.fallback_reason,
        }


def pme_force_scope_report(scope: str) -> dict[str, object]:
    """Return PME support metadata for a requested force-evaluation scope.

    Args:
        scope: Force-evaluation scope to query (``"total"``, ``"components"``,
            ``"direct_space"``, ``"reciprocal_space"``).

    Returns:
        A `ForceScopeReport` dict; every PME scope is supported and marked
            as requiring full-system evaluation.
    """

    normalized = normalize_force_scope(scope)
    if normalized == "total":
        return ForceScopeReport(
            scope=normalized,
            supported=True,
            execution_path="pme_total",
            backend=PME_EXECUTION_BACKEND,
            electrostatics="pme",
            production_total_only=True,
            requires_full_system=True,
        ).to_dict()
    if normalized == "components":
        return ForceScopeReport(
            scope=normalized,
            supported=True,
            execution_path="pme_components",
            backend=PME_EXECUTION_BACKEND,
            electrostatics="pme",
            diagnostic_components=True,
            component_work=True,
            requires_full_system=True,
        ).to_dict()
    if normalized == "direct_space":
        return ForceScopeReport(
            scope=normalized,
            supported=True,
            execution_path="pme_direct_space",
            backend=PME_EXECUTION_BACKEND,
            electrostatics="pme",
            direct_space=True,
            requires_full_system=True,
        ).to_dict()
    return ForceScopeReport(
        scope=normalized,
        supported=True,
        execution_path="pme_reciprocal_space",
        backend=PME_EXECUTION_BACKEND,
        electrostatics="pme",
        reciprocal_space=True,
        requires_full_system=True,
    ).to_dict()


def pme_readiness_report(
    *,
    atom_count: int,
    charges: object,
    cell_lengths: object,
    config: PMEConfig | None,
    nonbonded_cutoff: float | None,
    exclusion_count: int,
    one_four_count: int,
    explicit_exception_count: int,
) -> dict[str, object]:
    """Return fail-closed PME readiness metadata for production run gates.

    Args:
        atom_count: Number of atoms in the system.
        charges: Per-atom partial charges, shape ``(atom_count,)``.
        cell_lengths: Orthorhombic box lengths, shape ``(3,)``.
        config: PME parameters (mesh, alpha, cutoffs); ``None`` is reported as a blocker.
        nonbonded_cutoff: Real-space nonbonded cutoff, cross-checked against the PME cutoff.
        exclusion_count: Number of excluded pairs.
        one_four_count: Number of 1-4 corrected pairs.
        explicit_exception_count: Number of explicit nonbonded exceptions.

    Returns:
        A readiness dict with a ``"status"``, a ``"blockers"`` list, and the
            individual boolean checks (neutrality, box, mesh, alpha, cutoff, …).
    """

    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["production_executable_backend"] = PME_PRODUCTION_EXECUTABLE
    if not PME_PRODUCTION_EXECUTABLE:
        blockers.append(
            f"pme_backend_not_production_executable:current_backend={PME_EXECUTION_BACKEND}"
        )
    checks["atom_count"] = 0 <= int(atom_count) <= PME_PRODUCTION_MAX_ATOMS
    if not checks["atom_count"]:
        blockers.append(
            "atom_count:outside_pme_runtime_envelope:"
            f"atom_count={int(atom_count)},max_atoms={PME_PRODUCTION_MAX_ATOMS}"
        )

    background_policy: PMEBackgroundPolicy = "reject_non_neutral"
    if config is None:
        checks["config"] = False
        blockers.append("pme_config:missing")
        charge_tolerance = 1e-5
    else:
        checks["config"] = True
        charge_tolerance = float(config.charge_tolerance)
        background_policy = config.background_policy
        checks["mesh_shape"] = (
            len(config.mesh_shape) == 3
            and all(isinstance(size, int) and size >= 4 for size in config.mesh_shape)
        )
        checks["alpha"] = np.isfinite(float(config.alpha)) and float(config.alpha) > 0.0
        checks["cutoff"] = (
            config.real_cutoff is not None
            and np.isfinite(float(config.real_cutoff))
            and float(config.real_cutoff) > 0.0
            and nonbonded_cutoff is not None
            and np.isfinite(float(nonbonded_cutoff))
            and float(nonbonded_cutoff) > 0.0
        )
        for name in ("mesh_shape", "alpha", "cutoff"):
            if not checks[name]:
                blockers.append(f"pme_{name}:invalid")

    charge_values = np.asarray(charges, dtype=np.float64)
    net_charge = float(np.sum(charge_values, dtype=np.float64)) if charge_values.size else 0.0
    checks["neutrality"] = bool(
        charge_values.shape == (int(atom_count),)
        and np.all(np.isfinite(charge_values))
        and abs(net_charge) <= charge_tolerance
    )
    checks["charge_policy"] = bool(
        charge_values.shape == (int(atom_count),)
        and np.all(np.isfinite(charge_values))
        and (
            checks["neutrality"]
            or background_policy == "uniform_neutralizing_plasma"
        )
    )
    if not checks["charge_policy"]:
        blockers.append(
            "charge_policy:non_neutral_requires_uniform_neutralizing_plasma:"
            f"net_charge={net_charge:g}"
        )

    box = np.asarray(cell_lengths, dtype=np.float64)
    checks["box"] = bool(box.shape == (3,) and np.all(np.isfinite(box)) and np.all(box > 0.0))
    if not checks["box"]:
        blockers.append("box:missing_or_invalid")

    checks["exclusions"] = int(exclusion_count) >= 0
    checks["one_four_corrections"] = int(one_four_count) >= 0
    checks["explicit_exceptions"] = int(explicit_exception_count) >= 0
    for name in ("exclusions", "one_four_corrections", "explicit_exceptions"):
        if not checks[name]:
            blockers.append(f"{name}:invalid")

    return {
        "status": "ready" if not blockers else "blocked",
        "backend": PME_EXECUTION_BACKEND,
        "production_executable": PME_PRODUCTION_EXECUTABLE,
        "atom_count": int(atom_count),
        "net_charge": net_charge,
        "background_policy": background_policy,
        "mesh_shape": None if config is None else config.mesh_shape,
        "alpha": None if config is None else float(config.alpha),
        "real_cutoff": None if config is None else config.real_cutoff,
        "nonbonded_cutoff": nonbonded_cutoff,
        "assignment_order": None if config is None else config.assignment_order,
        "exclusion_count": int(exclusion_count),
        "one_four_count": int(one_four_count),
        "explicit_exception_count": int(explicit_exception_count),
        "runtime_envelope": {
            "max_atoms": PME_PRODUCTION_MAX_ATOMS,
            "cell": "orthorhombic",
            "assignment": None
            if config is None
            else f"cardinal_b_spline_order_{config.assignment_order}",
            "supported_assignment_orders": PME_SUPPORTED_ASSIGNMENT_ORDERS,
        },
        "virial": {
            "status": "finite_difference_cell_strain",
            "analytic_supported": False,
        },
        "force_scopes": {
            scope: pme_force_scope_report(scope) for scope in FORCE_EVALUATION_SCOPES
        },
        "checks": checks,
        "blockers": tuple(blockers),
    }


def pme_platform_readiness_report(
    *,
    atom_count: int,
    charges: object,
    cell_lengths: object,
    config: PMEConfig | None,
    nonbonded_cutoff: float | None,
    exclusion_count: int,
    one_four_count: int,
    explicit_exception_count: int,
) -> ReadinessReport:
    """Return PME readiness using the shared platform readiness schema.

    Args:
        atom_count: Number of atoms in the system.
        charges: Per-atom partial charges, shape ``(atom_count,)``.
        cell_lengths: Orthorhombic box lengths, shape ``(3,)``.
        config: PME parameters; ``None`` is reported as a blocker.
        nonbonded_cutoff: Real-space nonbonded cutoff, cross-checked against the PME cutoff.
        exclusion_count: Number of excluded pairs.
        one_four_count: Number of 1-4 corrected pairs.
        explicit_exception_count: Number of explicit nonbonded exceptions.

    Returns:
        A `ReadinessReport` (name ``"pme"``, status, blockers, and the
            remaining checks as metadata).
    """

    report = pme_readiness_report(
        atom_count=atom_count,
        charges=charges,
        cell_lengths=cell_lengths,
        config=config,
        nonbonded_cutoff=nonbonded_cutoff,
        exclusion_count=exclusion_count,
        one_four_count=one_four_count,
        explicit_exception_count=explicit_exception_count,
    )
    return ReadinessReport(
        name="pme",
        status=str(report["status"]),
        blockers=tuple(str(item) for item in report["blockers"]),
        metadata={key: value for key, value in report.items() if key != "blockers"},
    )


def pme_coulomb_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: PMEConfig | None = None,
    direct_space_pairs: mx.array | NeighborBlocks | None = None,
    plan: PMEExecutionPlan | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array | PMEDiagnostics]]:
    """Evaluate neutral orthorhombic Coulomb energy and forces with PME.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom partial charges, shape ``(n_atoms,)``; charged systems
            require the explicit uniform-plasma background policy.
        cell: Periodic orthorhombic `Cell` (triclinic is unsupported).
        coulomb_constant: Coulomb prefactor in the configured unit system. Defaults to ``1.0``.
        config: PME parameters; ``None`` uses defaults. Defaults to ``None``.
        direct_space_pairs: Optional precomputed real-space pairs or neighbor blocks.
            Defaults to ``None``.
        plan: Optional reusable fixed-cell execution plan. A one-shot plan is
            built when omitted. Defaults to ``None``.

    Returns:
        An ``(energy, forces, components)`` tuple: scalar total Coulomb energy,
            ``(n_atoms, 3)`` forces, and a components dict including a
            `PMEDiagnostics` entry.

    Raises:
        ValueError: If shapes are wrong, the cell is non-orthorhombic, or charge
            is incompatible with the configured background policy.
    """

    total_energy, forces, components = _pme_coulomb_energy_forces_impl(
        positions,
        charges,
        cell,
        coulomb_constant=coulomb_constant,
        config=config,
        include_components=True,
        direct_space_pairs=direct_space_pairs,
        plan=plan,
    )
    if components is None:
        msg = "PME component evaluation did not produce components"
        raise RuntimeError(msg)
    return total_energy, forces, components


def pme_coulomb_total_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: PMEConfig | None = None,
    direct_space_pairs: mx.array | NeighborBlocks | None = None,
    plan: PMEExecutionPlan | None = None,
) -> tuple[mx.array, mx.array]:
    """Evaluate neutral orthorhombic Coulomb total energy and forces with PME.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom partial charges, shape ``(n_atoms,)``; charged systems
            require the explicit uniform-plasma background policy.
        cell: Periodic orthorhombic `Cell` (triclinic is unsupported).
        coulomb_constant: Coulomb prefactor in the configured unit system. Defaults to ``1.0``.
        config: PME parameters; ``None`` uses defaults. Defaults to ``None``.
        direct_space_pairs: Optional precomputed real-space pairs or neighbor blocks.
            Defaults to ``None``.
        plan: Optional reusable fixed-cell execution plan. A one-shot plan is
            built when omitted. Defaults to ``None``.

    Returns:
        An ``(energy, forces)`` tuple: scalar total Coulomb energy and ``(n_atoms, 3)`` forces.

    Raises:
        ValueError: If shapes/cell or the configured charge policy are invalid.
    """

    total_energy, forces, _ = _pme_coulomb_energy_forces_impl(
        positions,
        charges,
        cell,
        coulomb_constant=coulomb_constant,
        config=config,
        include_components=False,
        direct_space_pairs=direct_space_pairs,
        plan=plan,
    )
    return total_energy, forces


def pme_direct_space_policy_report(
    cell: Cell,
    *,
    config: PMEConfig | None = None,
    pairs: mx.array | NeighborBlocks | None = None,
    plan: PMEExecutionPlan | None = None,
) -> dict[str, object]:
    """Report whether PME direct-space can use dense, pair, block, or fallback policy.

    Args:
        cell: Periodic orthorhombic cell.
        config: PME parameters; ``None`` uses defaults. Defaults to ``None``.
        pairs: Optional candidate real-space pairs or neighbor blocks to evaluate the
            policy against. Defaults to ``None``.
        plan: Optional plan whose configuration and fixed cell must match. The
            policy query does not advance its reuse counter. Defaults to ``None``.

    Returns:
        A `PMEDirectSpacePolicyReport` dict (selected policy/representation,
            support, cutoff, and pair-count provenance).

    Raises:
        ValueError: If the cell is missing, non-orthorhombic, or has non-positive lengths.
    """

    config = _resolve_evaluation_config(config, plan)
    if plan is not None:
        plan.validate(cell, config=config)
    if not isinstance(cell, Cell):
        msg = "PME requires an orthorhombic Cell"
        raise ValueError(msg)
    if not cell.is_orthorhombic:
        msg = (
            "PME currently supports orthorhombic cells only; "
            "triclinic cell matrices are not supported"
        )
        raise ValueError(msg)
    cell_lengths = np.asarray(cell.lengths, dtype=np.float64)
    if cell_lengths.shape != (3,) or not np.all(np.isfinite(cell_lengths)):
        msg = "PME requires finite orthorhombic cell lengths with shape (3,)"
        raise ValueError(msg)
    if np.any(cell_lengths <= 0.0):
        msg = "PME requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    real_cutoff = _resolve_real_cutoff(config, cell_lengths)
    return _direct_space_policy_report(pairs, real_cutoff, cell_lengths).to_dict()


def pme_coulomb_direct_space_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: PMEConfig | None = None,
    pairs: mx.array | NeighborBlocks | None = None,
    plan: PMEExecutionPlan | None = None,
) -> tuple[mx.array, mx.array]:
    """Evaluate only the PME real/direct-space Coulomb energy and forces.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom partial charges, shape ``(n_atoms,)``; charged systems
            require the explicit uniform-plasma background policy.
        cell: Periodic orthorhombic `Cell` (triclinic is unsupported).
        coulomb_constant: Coulomb prefactor in the configured unit system. Defaults to ``1.0``.
        config: PME parameters; ``None`` uses defaults. Defaults to ``None``.
        pairs: Optional real-space pairs or neighbor blocks. Defaults to ``None``.
        plan: Optional reusable fixed-cell execution plan. A one-shot plan is
            built when omitted. Defaults to ``None``.

    Returns:
        An ``(energy, forces)`` tuple: scalar real-space energy and ``(n_atoms, 3)`` forces.

    Raises:
        ValueError: If shapes/cell or the configured charge policy are invalid.
    """

    config = _resolve_evaluation_config(config, plan)
    positions_mx, charges_mx, cell_lengths_mx, cell_lengths_np = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=config.charge_tolerance,
        background_policy=config.background_policy,
    )
    _acquire_execution_plan(
        cell,
        config=config,
        coulomb_constant=coulomb_constant,
        plan=plan,
    )
    real_cutoff = _resolve_real_cutoff(config, cell_lengths_np)
    energy, forces, _ = _real_space_energy_forces_with_policy_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
        pairs=pairs,
    )
    return energy.astype(mx.float32), forces.astype(mx.float32)


def pme_coulomb_reciprocal_space_energy_forces(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float = 1.0,
    config: PMEConfig | None = None,
    include_self_correction: bool = True,
    plan: PMEExecutionPlan | None = None,
) -> tuple[mx.array, mx.array]:
    """Evaluate PME mesh/reciprocal-space Coulomb forces.

    By default the scalar self correction is included so direct-space plus
    reciprocal-space equals the total PME Coulomb energy.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom partial charges, shape ``(n_atoms,)``; charged systems
            require the explicit uniform-plasma background policy.
        cell: Periodic orthorhombic `Cell` (triclinic is unsupported).
        coulomb_constant: Coulomb prefactor in the configured unit system. Defaults to ``1.0``.
        config: PME parameters; ``None`` uses defaults. Defaults to ``None``.
        include_self_correction: Whether to add the scalar self-energy correction so
            direct + reciprocal equals the total. Defaults to ``True``.
        plan: Optional reusable fixed-cell execution plan. A one-shot plan is
            built when omitted. Defaults to ``None``.

    Returns:
        An ``(energy, forces)`` tuple: scalar reciprocal-space energy and ``(n_atoms, 3)`` forces.

    Raises:
        ValueError: If shapes/cell or the configured charge policy are invalid.
    """

    config = _resolve_evaluation_config(config, plan)
    positions_mx, charges_mx, cell_lengths_mx, cell_lengths_np = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=config.charge_tolerance,
        background_policy=config.background_policy,
    )
    execution_plan = _acquire_execution_plan(
        cell,
        config=config,
        coulomb_constant=coulomb_constant,
        plan=plan,
    )
    energy, forces, _ = _mesh_reciprocal_energy_forces_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        config=config,
        coulomb_constant=coulomb_constant,
        plan=execution_plan,
        return_mesh_info=False,
    )
    if include_self_correction:
        energy = energy - float(coulomb_constant) * config.alpha / float(
            np.sqrt(np.pi)
        ) * mx.sum(charges_mx * charges_mx)
        energy = energy + _neutralizing_plasma_energy_mx(
            charges_mx,
            volume=float(np.prod(cell_lengths_np, dtype=np.float64)),
            alpha=config.alpha,
            coulomb_constant=coulomb_constant,
            background_policy=config.background_policy,
        )
    return energy.astype(mx.float32), forces.astype(mx.float32)


def _pme_coulomb_energy_forces_impl(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    coulomb_constant: float,
    config: PMEConfig | None,
    include_components: bool,
    direct_space_pairs: mx.array | NeighborBlocks | None,
    plan: PMEExecutionPlan | None,
) -> tuple[mx.array, mx.array, dict[str, mx.array | PMEDiagnostics] | None]:
    config = _resolve_evaluation_config(config, plan)
    positions_mx, charges_mx, cell_lengths_mx, cell_lengths_np = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=config.charge_tolerance,
        background_policy=config.background_policy,
    )
    execution_plan = _acquire_execution_plan(
        cell,
        config=config,
        coulomb_constant=coulomb_constant,
        plan=plan,
    )
    real_cutoff = _resolve_real_cutoff(config, cell_lengths_np)

    real_energy, real_forces, direct_space_report = _real_space_energy_forces_with_policy_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        alpha=config.alpha,
        cutoff=real_cutoff,
        coulomb_constant=coulomb_constant,
        pairs=direct_space_pairs,
    )
    reciprocal_energy, reciprocal_forces, mesh_info = _mesh_reciprocal_energy_forces_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        cell_lengths_np,
        config=config,
        coulomb_constant=coulomb_constant,
        plan=execution_plan,
        return_mesh_info=include_components,
    )
    self_energy = (
        -float(coulomb_constant)
        * config.alpha
        / float(np.sqrt(np.pi))
        * mx.sum(charges_mx * charges_mx)
    )
    background_energy = _neutralizing_plasma_energy_mx(
        charges_mx,
        volume=float(np.prod(cell_lengths_np, dtype=np.float64)),
        alpha=config.alpha,
        coulomb_constant=coulomb_constant,
        background_policy=config.background_policy,
    )

    total_energy = real_energy + reciprocal_energy + self_energy + background_energy
    forces = real_forces + reciprocal_forces
    if not include_components:
        return total_energy.astype(mx.float32), forces.astype(mx.float32), None

    mx.eval(
        total_energy,
        forces,
        real_energy,
        reciprocal_energy,
        self_energy,
        background_energy,
    )
    if mesh_info is None:
        msg = "PME component evaluation requires mesh diagnostics"
        raise RuntimeError(msg)
    diagnostics = PMEDiagnostics(
        mesh_shape=config.mesh_shape,
        assignment_order=config.assignment_order,
        alpha=config.alpha,
        real_cutoff=real_cutoff,
        net_charge=float(np.asarray(mx.sum(charges_mx))),
        volume=float(np.asarray(cell.volume)),
        charge_grid_sum=mesh_info["charge_grid_sum"],
        reciprocal_modes=int(mesh_info["reciprocal_modes"]),
        max_charge_grid_abs=mesh_info["max_charge_grid_abs"],
        background_policy=config.background_policy,
        background_energy=float(np.asarray(background_energy)),
        direct_space_policy=direct_space_report.policy,
        direct_space_representation=direct_space_report.representation,
        direct_space_pair_count=direct_space_report.compact_pair_count,
        direct_space_candidate_count=direct_space_report.candidate_count,
        direct_space_fallback_reason=direct_space_report.fallback_reason,
        plan_fingerprint=execution_plan.fingerprint,
        plan_build_count=execution_plan.build_count,
        plan_reuse_count=execution_plan.reuse_count,
        plan_setup_seconds=execution_plan.setup_seconds,
        plan_estimated_resident_bytes=execution_plan.estimated_resident_bytes,
        plan_backend=execution_plan.backend,
        plan_device=execution_plan.device,
        plan_dtype=execution_plan.dtype,
    )
    components = {
        "coulomb_real": real_energy.astype(mx.float32),
        "coulomb_reciprocal": reciprocal_energy.astype(mx.float32),
        "coulomb_self": self_energy.astype(mx.float32),
        "coulomb_background": background_energy.astype(mx.float32),
        "diagnostics": diagnostics,
    }
    return total_energy.astype(mx.float32), forces.astype(mx.float32), components


def assign_charges_cic(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    mesh_shape: tuple[int, int, int],
) -> mx.array:
    """Assign charges to a periodic mesh with cloud-in-cell weights.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom charges, shape ``(n_atoms,)``.
        cell: Periodic orthorhombic cell.
        mesh_shape: Reciprocal-space mesh dimensions ``(nx, ny, nz)``.

    Returns:
        The charge density on the mesh, shape ``mesh_shape``.

    Raises:
        ValueError: If the cell is non-orthorhombic or the shapes are invalid.
    """

    return assign_charges_bspline(
        positions,
        charges,
        cell,
        mesh_shape,
        assignment_order=2,
    )


def assign_charges_bspline(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    mesh_shape: tuple[int, int, int],
    *,
    assignment_order: int = 2,
) -> mx.array:
    """Assign charges to a periodic mesh with cardinal B-spline weights.

    Args:
        positions: Atomic coordinates, shape ``(n_atoms, 3)``.
        charges: Per-atom charges, shape ``(n_atoms,)``.
        cell: Periodic orthorhombic cell.
        mesh_shape: Reciprocal-space mesh dimensions ``(nx, ny, nz)``.
        assignment_order: Cardinal B-spline order (2 = cloud-in-cell). Defaults to ``2``.

    Returns:
        The charge density on the mesh, shape ``mesh_shape``.

    Raises:
        ValueError: If the cell is non-orthorhombic or the shapes are invalid.
    """

    positions_mx, charges_mx, cell_lengths_mx, _ = _validate_inputs_mx(
        positions,
        charges,
        cell,
        charge_tolerance=np.inf,
        background_policy="reject_non_neutral",
    )
    mesh_shape = _validate_mesh_shape(mesh_shape)
    assignment_order = _validate_assignment_order(assignment_order)
    return _assign_charges_bspline_mx(
        positions_mx,
        charges_mx,
        cell_lengths_mx,
        mesh_shape,
        assignment_order=assignment_order,
    )


def _resolve_plan_config(config: PMEConfig | None) -> PMEConfig:
    if config is None:
        return PMEConfig()
    if not isinstance(config, PMEConfig):
        msg = "PME config must be a PMEConfig instance"
        raise TypeError(msg)
    return config


def _resolve_evaluation_config(
    config: PMEConfig | None,
    plan: PMEExecutionPlan | None,
) -> PMEConfig:
    if plan is not None and not isinstance(plan, PMEExecutionPlan):
        msg = "PME plan must be a PMEExecutionPlan instance"
        raise TypeError(msg)
    if config is None and plan is not None:
        return plan.config
    return _resolve_plan_config(config)


def _validate_coulomb_constant(value: float) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        msg = "coulomb_constant must be finite"
        raise ValueError(msg)
    return normalized


def _normalize_plan_dtype(dtype: object) -> str:
    normalized = str(dtype).strip().lower()
    aliases = {
        "<class 'numpy.float16'>": "float16",
        "<class 'numpy.float32'>": "float32",
        "<class 'numpy.float64'>": "float64",
        "mlx.core.float16": "float16",
        "mlx.core.float32": "float32",
        "mlx.core.float64": "float64",
        "numpy.float16": "float16",
        "numpy.float32": "float32",
        "numpy.float64": "float64",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"float16", "float32", "float64"}:
        msg = f"unsupported PME plan dtype: {dtype!r}"
        raise ValueError(msg)
    return normalized


def _normalize_plan_backend(backend: str) -> str:
    normalized = str(backend).strip()
    if not normalized:
        msg = "PME execution plan backend must be non-empty"
        raise ValueError(msg)
    return normalized


def _validated_plan_cell(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(cell, Cell):
        msg = "PME requires an orthorhombic Cell"
        raise ValueError(msg)
    if not cell.is_orthorhombic:
        msg = (
            "PME currently supports orthorhombic cells only; "
            "triclinic cell matrices are not supported"
        )
        raise ValueError(msg)
    cell_matrix = np.asarray(cell.matrix, dtype=np.float64)
    cell_lengths = np.asarray(cell.lengths, dtype=np.float64)
    if cell_matrix.shape != (3, 3) or not np.all(np.isfinite(cell_matrix)):
        msg = "PME requires a finite orthorhombic cell matrix with shape (3, 3)"
        raise ValueError(msg)
    if cell_lengths.shape != (3,) or not np.all(np.isfinite(cell_lengths)):
        msg = "PME requires finite orthorhombic cell lengths with shape (3,)"
        raise ValueError(msg)
    if np.any(cell_lengths <= 0.0):
        msg = "PME requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    return cell_matrix, cell_lengths


def _fingerprint_float(value: float) -> str:
    return float(value).hex()


def _plan_signature(
    *,
    cell_matrix: np.ndarray,
    config: PMEConfig,
    real_cutoff: float,
    coulomb_constant: float,
    dtype: str,
    backend: str,
    device: str,
) -> dict[str, object]:
    return {
        "schema_version": PME_EXECUTION_PLAN_SCHEMA_VERSION,
        "cell_matrix": [
            [_fingerprint_float(value) for value in row]
            for row in np.asarray(cell_matrix, dtype=np.float64)
        ],
        "mesh_shape": list(config.mesh_shape),
        "alpha": _fingerprint_float(config.alpha),
        "configured_real_cutoff": (
            None
            if config.real_cutoff is None
            else _fingerprint_float(config.real_cutoff)
        ),
        "real_cutoff": _fingerprint_float(real_cutoff),
        "assignment_order": config.assignment_order,
        "charge_tolerance": _fingerprint_float(config.charge_tolerance),
        "deconvolve_assignment": config.deconvolve_assignment,
        "coulomb_constant": _fingerprint_float(coulomb_constant),
        "background_policy": config.background_policy,
        "dtype": dtype,
        "backend": backend,
        "device": device,
    }


def _plan_fingerprint(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _estimated_plan_resident_bytes(grid_size: int, *, dtype: str) -> int:
    dtype_bytes = {"float16": 2, "float32": 4, "float64": 8}[dtype]
    reciprocal_arrays = 4  # influence plus kx, ky, and kz
    return int(grid_size) * dtype_bytes * reciprocal_arrays


def _acquire_execution_plan(
    cell: Cell,
    *,
    config: PMEConfig,
    coulomb_constant: float,
    plan: PMEExecutionPlan | None,
) -> PMEExecutionPlan:
    execution_plan = (
        PMEExecutionPlan(
            cell,
            config=config,
            coulomb_constant=coulomb_constant,
        )
        if plan is None
        else plan
    )
    validation_started = time.perf_counter()
    execution_plan.validate(
        cell,
        config=config,
        coulomb_constant=coulomb_constant,
        dtype=mx.float32,
        backend=PME_EXECUTION_BACKEND,
    )
    execution_plan._record_reuse(time.perf_counter() - validation_started)
    return execution_plan


def _validate_inputs_mx(
    positions: mx.array,
    charges: mx.array,
    cell: Cell,
    *,
    charge_tolerance: float,
    background_policy: PMEBackgroundPolicy,
) -> tuple[mx.array, mx.array, mx.array, np.ndarray]:
    if not isinstance(cell, Cell):
        msg = "PME requires an orthorhombic Cell"
        raise ValueError(msg)
    if not cell.is_orthorhombic:
        msg = (
            "PME currently supports orthorhombic cells only; "
            "triclinic cell matrices are not supported"
        )
        raise ValueError(msg)
    positions_mx = mx.array(positions, dtype=mx.float32)
    charges_mx = mx.array(charges, dtype=mx.float32)
    if positions_mx.ndim != 2 or positions_mx.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if charges_mx.shape != (positions_mx.shape[0],):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if not bool(np.asarray(mx.all(mx.isfinite(positions_mx)))):
        msg = "positions must be finite"
        raise ValueError(msg)
    if not bool(np.asarray(mx.all(mx.isfinite(charges_mx)))):
        msg = "charges must be finite"
        raise ValueError(msg)
    cell_lengths_mx = mx.array(cell.lengths, dtype=mx.float32)
    cell_lengths_np = np.asarray(cell_lengths_mx, dtype=np.float64)
    if cell_lengths_np.shape != (3,) or not np.all(np.isfinite(cell_lengths_np)):
        msg = "PME requires finite orthorhombic cell lengths with shape (3,)"
        raise ValueError(msg)
    if np.any(cell_lengths_np <= 0.0):
        msg = "PME requires positive orthorhombic cell lengths"
        raise ValueError(msg)
    net_charge = float(np.asarray(mx.sum(charges_mx)))
    if (
        abs(net_charge) > charge_tolerance
        and background_policy != "uniform_neutralizing_plasma"
    ):
        msg = (
            "PME requires a neutral system unless background_policy="
            "'uniform_neutralizing_plasma'; "
            f"net_charge={net_charge:g}"
        )
        raise ValueError(msg)
    wrapped_positions = positions_mx - mx.floor(positions_mx / cell_lengths_mx) * cell_lengths_mx
    return wrapped_positions, charges_mx, cell_lengths_mx, cell_lengths_np


def _neutralizing_plasma_energy_mx(
    charges: mx.array,
    *,
    volume: float,
    alpha: float,
    coulomb_constant: float,
    background_policy: PMEBackgroundPolicy,
) -> mx.array:
    if background_policy != "uniform_neutralizing_plasma":
        return mx.sum(charges * 0.0)
    net_charge = mx.sum(charges)
    scale = -float(coulomb_constant) * float(np.pi) / (
        2.0 * float(volume) * float(alpha) * float(alpha)
    )
    return scale * net_charge * net_charge


def _validate_mesh_shape(mesh_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(mesh_shape) != 3:
        msg = "mesh_shape must contain exactly three dimensions"
        raise ValueError(msg)
    if any(int(size) != size or size < 4 for size in mesh_shape):
        msg = "mesh_shape dimensions must be integers >= 4"
        raise ValueError(msg)
    normalized = tuple(int(size) for size in mesh_shape)
    return normalized


def _validate_assignment_order(assignment_order: int) -> int:
    try:
        normalized = int(assignment_order)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "assignment_order must be one of 2, 4, or 5"
        raise ValueError(msg) from exc
    if normalized != assignment_order or normalized not in PME_SUPPORTED_ASSIGNMENT_ORDERS:
        msg = "assignment_order must be one of 2, 4, or 5"
        raise ValueError(msg)
    return normalized


def _resolve_real_cutoff(config: PMEConfig, cell_lengths: np.ndarray) -> float:
    if config.real_cutoff is not None:
        return float(config.real_cutoff)
    return 0.5 * float(np.min(cell_lengths))


def _minimum_image_pair_policy_supported(cutoff: float, cell_lengths: np.ndarray) -> bool:
    return float(cutoff) <= 0.5 * float(np.min(cell_lengths)) + 1e-7


def _direct_space_policy_report(
    pairs: mx.array | NeighborBlocks | None,
    cutoff: float,
    cell_lengths: np.ndarray,
) -> PMEDirectSpacePolicyReport:
    minimum_image_safe = _minimum_image_pair_policy_supported(cutoff, cell_lengths)
    if pairs is None:
        return PMEDirectSpacePolicyReport(
            policy="dense",
            representation="dense",
            uses_shared_neighbor_policy=False,
            supported=True,
            real_cutoff=float(cutoff),
            minimum_image_safe=minimum_image_safe,
        )
    if isinstance(pairs, NeighborBlocks):
        if not minimum_image_safe:
            return PMEDirectSpacePolicyReport(
                policy="fallback",
                representation="dense",
                uses_shared_neighbor_policy=False,
                supported=True,
                real_cutoff=float(cutoff),
                minimum_image_safe=False,
                compact_pair_count=int(pairs.compact_pair_count),
                candidate_count=int(pairs.candidate_count),
                candidate_waste_count=int(pairs.candidate_waste_count),
                fallback_reason=(
                    "pme_direct_space_pair_policy_requires_cutoff_at_or_below_half_min_box"
                ),
            )
        return PMEDirectSpacePolicyReport(
            policy="block_candidate",
            representation="blocks",
            uses_shared_neighbor_policy=True,
            supported=True,
            real_cutoff=float(cutoff),
            minimum_image_safe=True,
            pair_count=int(pairs.candidate_count),
            compact_pair_count=int(pairs.compact_pair_count),
            candidate_count=int(pairs.candidate_count),
            candidate_waste_count=int(pairs.candidate_waste_count),
        )

    pair_array = mx.array(pairs, dtype=mx.int32)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        msg = "PME direct-space pairs must have shape (n_pairs, 2)"
        raise ValueError(msg)
    pair_count = int(pair_array.shape[0])
    if not minimum_image_safe:
        return PMEDirectSpacePolicyReport(
            policy="fallback",
            representation="dense",
            uses_shared_neighbor_policy=False,
            supported=True,
            real_cutoff=float(cutoff),
            minimum_image_safe=False,
            pair_count=pair_count,
            compact_pair_count=pair_count,
            candidate_count=pair_count,
            candidate_waste_count=0,
            fallback_reason=(
                "pme_direct_space_pair_policy_requires_cutoff_at_or_below_half_min_box"
            ),
        )
    return PMEDirectSpacePolicyReport(
        policy="compact_pair",
        representation="pairs",
        uses_shared_neighbor_policy=True,
        supported=True,
        real_cutoff=float(cutoff),
        minimum_image_safe=True,
        pair_count=pair_count,
        compact_pair_count=pair_count,
        candidate_count=pair_count,
        candidate_waste_count=0,
    )


def _validate_compact_pairs(pairs: mx.array, n_atoms: int) -> mx.array:
    pair_array = mx.array(pairs, dtype=mx.int32)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        msg = "PME direct-space pairs must have shape (n_pairs, 2)"
        raise ValueError(msg)
    pair_np = np.asarray(pair_array, dtype=np.int32)
    if pair_np.size and (np.any(pair_np < 0) or np.any(pair_np >= n_atoms)):
        msg = "PME direct-space pairs contain atom indices outside [0, n_atoms)"
        raise ValueError(msg)
    return pair_array


def _validate_neighbor_blocks(blocks: NeighborBlocks, n_atoms: int) -> NeighborBlocks:
    left = np.asarray(blocks.left, dtype=np.int32)
    right = np.asarray(blocks.right, dtype=np.int32)
    valid = np.asarray(blocks.valid_mask, dtype=bool)
    if np.any(left[valid] < 0) or np.any(right[valid] < 0):
        msg = "PME direct-space blocks contain atom indices outside [0, n_atoms)"
        raise ValueError(msg)
    if np.any(left[valid] >= n_atoms) or np.any(right[valid] >= n_atoms):
        msg = "PME direct-space blocks contain atom indices outside [0, n_atoms)"
        raise ValueError(msg)
    return blocks


def _minimum_image_displacement(displacement: mx.array, cell_lengths: mx.array) -> mx.array:
    image = mx.floor(displacement / cell_lengths + 0.5)
    return displacement - image * cell_lengths


def _real_space_shifts(cell_lengths: np.ndarray, cutoff: float) -> np.ndarray:
    ranges = [
        range(
            -int(ceil(float(cutoff) / float(length))) - 1,
            int(ceil(float(cutoff) / float(length))) + 2,
        )
        for length in cell_lengths
    ]
    return np.asarray(
        [
            (
                nx * float(cell_lengths[0]),
                ny * float(cell_lengths[1]),
                nz * float(cell_lengths[2]),
            )
            for nx in ranges[0]
            for ny in ranges[1]
            for nz in ranges[2]
        ],
        dtype=np.float64,
    )


def _real_space_energy_forces_with_policy_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    cell_lengths_np: np.ndarray,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
    pairs: mx.array | NeighborBlocks | None,
) -> tuple[mx.array, mx.array, PMEDirectSpacePolicyReport]:
    report = _direct_space_policy_report(pairs, cutoff, cell_lengths_np)
    if report.policy == "compact_pair":
        pair_array = _validate_compact_pairs(mx.array(pairs, dtype=mx.int32), positions.shape[0])
        energy, forces = _real_space_pair_energy_forces_mx(
            positions,
            charges,
            cell_lengths,
            pair_array,
            alpha=alpha,
            cutoff=cutoff,
            coulomb_constant=coulomb_constant,
        )
        return energy, forces, report
    if report.policy == "block_candidate":
        if not isinstance(pairs, NeighborBlocks):
            msg = "PME direct-space block policy requires NeighborBlocks"
            raise TypeError(msg)
        blocks = _validate_neighbor_blocks(pairs, positions.shape[0])
        energy, forces = _real_space_block_energy_forces_mx(
            positions,
            charges,
            cell_lengths,
            blocks,
            alpha=alpha,
            cutoff=cutoff,
            coulomb_constant=coulomb_constant,
        )
        return energy, forces, report
    energy, forces = _real_space_energy_forces_mx(
        positions,
        charges,
        cell_lengths,
        cell_lengths_np,
        alpha=alpha,
        cutoff=cutoff,
        coulomb_constant=coulomb_constant,
    )
    return energy, forces, report


def _real_space_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    cell_lengths_np: np.ndarray,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    shifts = _real_space_shifts(cell_lengths_np, cutoff)
    n_atoms = int(positions.shape[0])
    atom_index = mx.arange(n_atoms)
    cutoff2 = float(cutoff) * float(cutoff)
    total_energy = mx.array(0.0, dtype=mx.float32)
    forces = mx.zeros_like(positions)
    qij = charges[:, None] * charges[None, :]
    for shift in shifts:
        shift_value = mx.array(shift, dtype=mx.float32)
        displacement = positions[:, None, :] - positions[None, :, :] + shift_value
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = (r2 > 0.0) & (r2 < cutoff2)
        if bool(np.sum(shift * shift) == 0.0):
            pair_mask = pair_mask & (atom_index[:, None] != atom_index[None, :])
        safe_r2 = mx.where(pair_mask, r2, 1.0)
        distance = mx.sqrt(safe_r2)
        erfc_term = 1.0 - mx.erf(float(alpha) * distance)
        pair_energy = float(coulomb_constant) * qij * erfc_term / distance
        pair_energy = mx.where(pair_mask, pair_energy, 0.0)
        scalar = float(coulomb_constant) * qij * (
            erfc_term / (safe_r2 * distance)
            + (2.0 * float(alpha) / float(np.sqrt(np.pi)))
            * mx.exp(-(float(alpha) * float(alpha)) * safe_r2)
            / safe_r2
        )
        scalar = mx.where(pair_mask, scalar, 0.0)
        forces = forces + mx.sum(scalar[:, :, None] * displacement, axis=1)
        total_energy = total_energy + 0.5 * mx.sum(pair_energy)
    return total_energy, forces


def _real_space_pair_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    pairs: mx.array,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    if pairs.shape[0] == 0:
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)
    i = pairs[:, 0]
    j = pairs[:, 1]
    displacement = _minimum_image_displacement(positions[i] - positions[j], cell_lengths)
    r2 = mx.sum(displacement * displacement, axis=-1)
    pair_mask = (r2 > 0.0) & (r2 < float(cutoff) * float(cutoff))
    safe_r2 = mx.where(pair_mask, r2, 1.0)
    distance = mx.sqrt(safe_r2)
    qij = charges[i] * charges[j]
    erfc_term = 1.0 - mx.erf(float(alpha) * distance)
    pair_energy = float(coulomb_constant) * qij * erfc_term / distance
    pair_energy = mx.where(pair_mask, pair_energy, 0.0)
    scalar = float(coulomb_constant) * qij * (
        erfc_term / (safe_r2 * distance)
        + (2.0 * float(alpha) / float(np.sqrt(np.pi)))
        * mx.exp(-(float(alpha) * float(alpha)) * safe_r2)
        / safe_r2
    )
    scalar = mx.where(pair_mask, scalar, 0.0)
    pair_forces = scalar[:, None] * displacement
    forces = mx.zeros_like(positions).at[i].add(pair_forces).at[j].add(-pair_forces)
    return mx.sum(pair_energy), forces


def _real_space_block_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    blocks: NeighborBlocks,
    *,
    alpha: float,
    cutoff: float,
    coulomb_constant: float,
) -> tuple[mx.array, mx.array]:
    if blocks.candidate_count == 0:
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)
    i = blocks.left
    j = blocks.right
    displacement = _minimum_image_displacement(positions[i] - positions[j], cell_lengths)
    r2 = mx.sum(displacement * displacement, axis=-1)
    pair_mask = blocks.valid_mask & (r2 > 0.0) & (r2 < float(cutoff) * float(cutoff))
    safe_r2 = mx.where(pair_mask, r2, 1.0)
    distance = mx.sqrt(safe_r2)
    qij = charges[i] * charges[j]
    erfc_term = 1.0 - mx.erf(float(alpha) * distance)
    pair_energy = float(coulomb_constant) * qij * erfc_term / distance
    pair_energy = mx.where(pair_mask, pair_energy, 0.0)
    scalar = float(coulomb_constant) * qij * (
        erfc_term / (safe_r2 * distance)
        + (2.0 * float(alpha) / float(np.sqrt(np.pi)))
        * mx.exp(-(float(alpha) * float(alpha)) * safe_r2)
        / safe_r2
    )
    scalar = mx.where(pair_mask, scalar, 0.0)
    pair_forces = scalar[..., None] * displacement
    flat_i = mx.reshape(i, (-1,))
    flat_j = mx.reshape(j, (-1,))
    flat_forces = mx.reshape(pair_forces, (-1, 3))
    forces = mx.zeros_like(positions).at[flat_i].add(flat_forces).at[flat_j].add(
        -flat_forces
    )
    return mx.sum(pair_energy), forces


def _mesh_reciprocal_energy_forces_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    cell_lengths_np: np.ndarray,
    *,
    config: PMEConfig,
    coulomb_constant: float,
    plan: PMEExecutionPlan,
    return_mesh_info: bool = True,
) -> tuple[mx.array, mx.array, dict[str, float | int] | None]:
    charge_grid = _assign_charges_bspline_mx(
        positions,
        charges,
        cell_lengths,
        config.mesh_shape,
        assignment_order=config.assignment_order,
    )
    rho_hat = mx.fft.fftn(charge_grid)
    del cell_lengths_np, coulomb_constant
    phi_hat = plan.influence * rho_hat
    grid_size = plan.grid_size
    potential_grid = mx.real(mx.fft.ifftn(phi_hat)) * float(grid_size)
    field_grids = [
        mx.real(mx.fft.ifftn((-1j * k_axis) * phi_hat)) * float(grid_size)
        for k_axis in plan.wavevectors
    ]
    field_grid = mx.stack(field_grids, axis=-1)
    potential_at_atoms = _interpolate_bspline_mx(
        positions,
        potential_grid,
        cell_lengths,
        assignment_order=config.assignment_order,
    )
    field_at_atoms = _interpolate_bspline_mx(
        positions,
        field_grid,
        cell_lengths,
        assignment_order=config.assignment_order,
    )
    energy = 0.5 * mx.sum(charges * potential_at_atoms)
    forces = charges[:, None] * field_at_atoms
    if not return_mesh_info:
        return energy, forces, None
    mx.eval(energy, forces, charge_grid)
    return (
        energy,
        forces,
        {
            "charge_grid_sum": float(np.asarray(mx.sum(charge_grid))),
            "max_charge_grid_abs": float(np.asarray(mx.max(mx.abs(charge_grid))))
            if int(np.prod(config.mesh_shape)) > 0
            else 0.0,
            "reciprocal_modes": plan.reciprocal_modes,
        },
    )


def _influence_function_mx(
    cell_lengths: np.ndarray,
    mesh_shape: tuple[int, int, int],
    *,
    alpha: float,
    coulomb_constant: float,
    deconvolve_assignment: bool,
    assignment_order: int = 2,
) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array], int]:
    assignment_order = _validate_assignment_order(assignment_order)
    k_components = []
    window = mx.ones(mesh_shape, dtype=mx.float32)
    for axis, (length, size) in enumerate(zip(cell_lengths, mesh_shape, strict=True)):
        frequencies = mx.fft.fftfreq(size, d=float(length) / float(size))
        k_axis = 2.0 * float(np.pi) * frequencies
        shape = [1, 1, 1]
        shape[axis] = int(size)
        k_grid = mx.reshape(k_axis, tuple(shape))
        k_components.append(mx.broadcast_to(k_grid, mesh_shape))
        if deconvolve_assignment:
            window_axis = _sinc_mx(
                k_axis * float(length) / (2.0 * float(np.pi) * float(size))
            ) ** assignment_order
            window = window * mx.broadcast_to(mx.reshape(window_axis, tuple(shape)), mesh_shape)

    kx, ky, kz = k_components
    k2 = kx * kx + ky * ky + kz * kz
    mask = k2 > 0.0
    denominator = k2
    if deconvolve_assignment:
        denominator = denominator * mx.maximum(window * window, mx.array(1e-12))
    safe_denominator = mx.where(mask, denominator, 1.0)
    volume = float(np.prod(cell_lengths, dtype=np.float64))
    influence = (
        float(coulomb_constant)
        * 4.0
        * float(np.pi)
        / volume
        * mx.exp(-k2 / (4.0 * float(alpha) * float(alpha)))
        / safe_denominator
    )
    influence = mx.where(mask, influence, 0.0)
    return influence, (kx, ky, kz), int(np.prod(mesh_shape) - 1)


def _sinc_mx(values: mx.array) -> mx.array:
    argument = float(np.pi) * values
    near_zero = mx.abs(argument) < 1e-7
    safe_argument = mx.where(near_zero, 1.0, argument)
    return mx.where(near_zero, 1.0, mx.sin(argument) / safe_argument)


def _assign_charges_cic_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    mesh_shape: tuple[int, int, int],
) -> mx.array:
    return _assign_charges_bspline_mx(
        positions,
        charges,
        cell_lengths,
        mesh_shape,
        assignment_order=2,
    )


def _assign_charges_bspline_mx(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    mesh_shape: tuple[int, int, int],
    *,
    assignment_order: int,
) -> mx.array:
    assignment_order = _validate_assignment_order(assignment_order)
    mesh = mx.array(mesh_shape, dtype=mx.float32)
    scaled = (positions - mx.floor(positions / cell_lengths) * cell_lengths) / cell_lengths * mesh
    base = mx.floor(scaled).astype(mx.int32) - ((assignment_order - 1) // 2)
    fractions = [scaled[:, axis] - mx.floor(scaled[:, axis]) for axis in range(3)]
    nx, ny, nz = mesh_shape
    grid = mx.zeros((nx * ny * nz,), dtype=mx.float32)
    weights = [
        [
            _bspline_weight_mx(fractions[axis], offset, assignment_order)
            for offset in range(assignment_order)
        ]
        for axis in range(3)
    ]
    for dx in range(assignment_order):
        wx = weights[0][dx]
        ix = (base[:, 0] + dx) % nx
        for dy in range(assignment_order):
            wy = weights[1][dy]
            iy = (base[:, 1] + dy) % ny
            for dz in range(assignment_order):
                wz = weights[2][dz]
                iz = (base[:, 2] + dz) % nz
                flat_index = (ix * ny + iy) * nz + iz
                grid = grid.at[flat_index].add(charges * wx * wy * wz)
    return mx.reshape(grid, mesh_shape)


def _interpolate_cic_mx(
    positions: mx.array,
    grid: mx.array,
    cell_lengths: mx.array,
) -> mx.array:
    return _interpolate_bspline_mx(
        positions,
        grid,
        cell_lengths,
        assignment_order=2,
    )


def _interpolate_bspline_mx(
    positions: mx.array,
    grid: mx.array,
    cell_lengths: mx.array,
    *,
    assignment_order: int,
) -> mx.array:
    assignment_order = _validate_assignment_order(assignment_order)
    mesh_shape = grid.shape[:3]
    trailing_shape = grid.shape[3:]
    n_atoms = int(positions.shape[0])
    mesh = mx.array(mesh_shape, dtype=mx.float32)
    scaled = (positions - mx.floor(positions / cell_lengths) * cell_lengths) / cell_lengths * mesh
    base = mx.floor(scaled).astype(mx.int32) - ((assignment_order - 1) // 2)
    fractions = [scaled[:, axis] - mx.floor(scaled[:, axis]) for axis in range(3)]
    values = mx.zeros((n_atoms, *trailing_shape), dtype=grid.dtype)
    nx, ny, nz = mesh_shape
    weights = [
        [
            _bspline_weight_mx(fractions[axis], offset, assignment_order)
            for offset in range(assignment_order)
        ]
        for axis in range(3)
    ]
    for dx in range(assignment_order):
        wx = weights[0][dx]
        ix = (base[:, 0] + dx) % nx
        for dy in range(assignment_order):
            wy = weights[1][dy]
            iy = (base[:, 1] + dy) % ny
            for dz in range(assignment_order):
                wz = weights[2][dz]
                iz = (base[:, 2] + dz) % nz
                weight = wx * wy * wz
                corner_values = grid[ix, iy, iz]
                if trailing_shape:
                    weight = mx.reshape(weight, (n_atoms, *([1] * len(trailing_shape))))
                values = values + weight * corner_values
    return values


def _bspline_weight_mx(fraction: mx.array, offset: int, assignment_order: int) -> mx.array:
    values = float(offset + 1) - fraction
    return _cardinal_bspline_mx(values, assignment_order)


def _cardinal_bspline_mx(values: mx.array, order: int) -> mx.array:
    if order == 1:
        return mx.where((values >= 0.0) & (values < 1.0), 1.0, 0.0)
    previous = _cardinal_bspline_mx(values, order - 1)
    shifted_previous = _cardinal_bspline_mx(values - 1.0, order - 1)
    degree = float(order - 1)
    return (values / degree) * previous + ((float(order) - values) / degree) * shifted_previous
