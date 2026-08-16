"""Molecular dynamics primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import partial
from math import exp, sqrt
from time import perf_counter
from types import NotImplementedType
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic._md_langevin import (
    _LangevinBlockExecutor,
    _LangevinBlockState,
    _next_recording_local_step,
)
from mlx_atomistic._md_state import (
    BarostatProposal,
    LangevinThermostat,
    MonteCarloBarostat,
    NoseHooverThermostat,
    NPTResult,
    NVEResult,
    NVTResult,
    ReporterEvent,
    RuntimeReporter,
    SimulationConfig,
    SimulationResult,
    SimulationState,
    StepState,
    Thermostat,
    _barostat_axis_index,
    _barostat_plane_axes,
    _NVTBoundaryDiagnostics,
    _validate_barostat_cell_support,
)
from mlx_atomistic._md_thermodynamics import (
    _pressure_diagnostics_from_virial,
    _remove_center_of_mass_velocity,
    _temperature_degrees_of_freedom,
    instantaneous_temperature,
    kinetic_energy,
    kinetic_pressure_tensor,
    virial_tensor,
)
from mlx_atomistic.constraints import (
    CompositeConstraints,
    DistanceConstraints,
    SettleWaterConstraints,
    _project_constraint_positions_unchecked,
)
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.force_evaluation import (
    ForceTerm,
    _dense_pair_count,
    _energy_forces_by_term,
    _energy_forces_from_terms,
    _energy_from_term_demands,
    _forces_from_term_demands,
    _groupable_potential_terms,
    _is_energy_component,
    _named_force_terms,
    _neighbor_evaluation_positions,
)
from mlx_atomistic.force_runtime import (
    _ExclusiveRouteProfiler,
    _PreparedForcePipeline,
)
from mlx_atomistic.lj import LennardJonesPotential
from mlx_atomistic.metal_kernels import _fused_langevin_baoab_drift
from mlx_atomistic.neighbors import (
    NeighborBlocks,
    NeighborList,
    NeighborListManager,
    _bounded_metal_md_cache,
)
from mlx_atomistic.nonbonded import (
    molecularly_strained_positions,
    normalize_molecule_ids,
)
from mlx_atomistic.runtime import (
    VIRIAL_SUPPORT_ANALYTIC,
    VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE,
    VIRIAL_SUPPORT_UNSUPPORTED,
    ReadinessReport,
    normalize_virial_support,
)
from mlx_atomistic.virtual_sites import VirtualSiteManager

RUNTIME_SYNC_REASONS = (
    "reporter",
    "diagnostic",
    "checkpoint",
    "final_state",
    "failure_check",
    "explicit_user_output",
)


_ForceEvaluationMode = Literal[
    "forces",
    "energy_forces",
    "energy",
    "diagnostic",
]


@dataclass(frozen=True)
class _ForceEvaluationRequest:
    """Describe one exact internal force-evaluation demand."""

    mode: _ForceEvaluationMode
    virial_mode: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {
            "forces",
            "energy_forces",
            "energy",
            "diagnostic",
        }:
            msg = f"unsupported force-evaluation mode: {self.mode!r}"
            raise ValueError(msg)
        if self.virial_mode is not None:
            if self.mode != "diagnostic":
                msg = "virial output is available only in diagnostic mode"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "virial_mode",
                normalize_virial_support(self.virial_mode),
            )


@dataclass(frozen=True)
class _ForceEvaluationResult:
    """Hold only the outputs requested by an internal force evaluation."""

    energy: mx.array | None = None
    forces: mx.array | None = None
    components: dict[str, mx.array] = field(default_factory=dict)
    virial: mx.array | None = None
    optimized_terms: int = 0
    fallback_terms: int = 0


_FORCES_REQUEST = _ForceEvaluationRequest("forces")
_ENERGY_FORCES_REQUEST = _ForceEvaluationRequest("energy_forces")
_ENERGY_REQUEST = _ForceEvaluationRequest("energy")
_DIAGNOSTIC_REQUEST = _ForceEvaluationRequest("diagnostic")


def configurational_virial_tensor(
    positions: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
    strain_epsilon: float = 1e-3,
    masses: mx.array | None = None,
    molecule_ids: object | None = None,
    virial_mode: str = VIRIAL_SUPPORT_ANALYTIC,
) -> mx.array:
    """Return an analytic production virial or the named validation oracle.

    Args:
        positions: Particle coordinates, shape ``(n_particles, 3)``.
        forces: Forces used for the non-periodic fallback.
        force_terms: Force terms contributing to the configurational virial.
        cell: Periodic cell; ``None`` uses the non-periodic force virial.
        pairs: Optional neighbor/pair structure forwarded to analytic terms.
        virtual_sites: Optional virtual-site manager applied before energy
            evaluation. Defaults to ``None``.
        strain_epsilon: Half-width used only by the finite-difference oracle.
        masses: Optional particle masses for molecular center construction.
        molecule_ids: Optional contiguous per-particle molecule identifiers;
            absent identifiers treat every particle as its own molecule.
        virial_mode: ``analytic`` for production or
            ``finite_difference_oracle`` for validation.

    Returns:
        The diagonal ``(3, 3)`` configurational virial tensor.

    Raises:
        ValueError: If the requested support level or cell is invalid.
    """

    mode = normalize_virial_support(virial_mode)
    if mode == VIRIAL_SUPPORT_ANALYTIC:
        return analytic_configurational_virial_tensor(
            positions,
            forces,
            force_terms,
            cell=cell,
            pairs=pairs,
            masses=masses,
            molecule_ids=molecule_ids,
        )
    if mode == VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE:
        return finite_difference_configurational_virial_oracle(
            positions,
            forces,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
            strain_epsilon=strain_epsilon,
            masses=masses,
            molecule_ids=molecule_ids,
        )
    msg = "unsupported virial mode cannot produce pressure"
    raise ValueError(msg)


def analytic_configurational_virial_tensor(
    positions: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    masses: mx.array | None = None,
    molecule_ids: object | None = None,
) -> mx.array:
    """Return the production analytic diagonal configurational virial."""

    validate_analytic_virial_support(force_terms)
    positions = as_mx_array(positions)
    if cell is None:
        return virial_tensor(positions, forces)
    _validate_orthorhombic_pressure_cell(cell)
    contributions = []
    for _, term in _named_force_terms(force_terms):
        method = getattr(term, "analytic_virial_tensor", None)
        if not callable(method):
            msg = "analytic virial support requires analytic_virial_tensor"
            raise ValueError(msg)
        contribution = as_mx_array(
            method(
                positions,
                cell=cell,
                pairs=pairs,
                masses=masses,
                molecule_ids=molecule_ids,
            )
        )
        if contribution.shape != (3, 3):
            msg = "analytic virial contributions must have shape (3, 3)"
            raise ValueError(msg)
        contributions.append(mx.diag(mx.diag(contribution)))
    if not contributions:
        return mx.zeros((3, 3), dtype=positions.dtype)
    return sum(contributions[1:], start=contributions[0])


def finite_difference_configurational_virial_oracle(
    positions: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
    strain_epsilon: float = 1e-3,
    masses: mx.array | None = None,
    molecule_ids: object | None = None,
) -> mx.array:
    """Return the validation-only molecular cell-strain virial oracle."""

    validate_virial_support(force_terms)
    positions = as_mx_array(positions)
    if cell is None:
        return virial_tensor(positions, forces)
    if strain_epsilon <= 0.0:
        msg = "strain_epsilon must be positive"
        raise ValueError(msg)
    if float(np.asarray(cell.volume)) <= 0.0:
        msg = "virial diagnostics require positive cell volume"
        raise ValueError(msg)

    specialized = []
    remaining_terms = []
    if virtual_sites is None:
        for term in force_terms:
            method = getattr(term, "finite_difference_virial_tensor", None)
            if (
                callable(method)
                and getattr(term, "electrostatics", None) == "pme"
                and getattr(term, "pme_config", None) is not None
            ):
                specialized.append(
                    method(
                        positions,
                        cell=cell,
                        pairs=pairs,
                        masses=masses,
                        molecule_ids=molecule_ids,
                        strain_epsilon=strain_epsilon,
                    )
                )
            else:
                remaining_terms.append(term)
    else:
        remaining_terms.extend(force_terms)
    if not remaining_terms:
        return sum(
            specialized[1:],
            start=specialized[0],
        )

    diagonal = []
    for axis in range(3):
        plus_cell = Cell(_strained_cell_matrix(cell.matrix, axis, strain_epsilon))
        minus_cell = Cell(_strained_cell_matrix(cell.matrix, axis, -strain_epsilon))
        plus_energy = _potential_energy_for_virial(
            _molecularly_strained_positions(
                positions,
                source_cell=cell,
                target_cell=plus_cell,
                masses=masses,
                molecule_ids=molecule_ids,
            ),
            tuple(remaining_terms),
            cell=plus_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        minus_energy = _potential_energy_for_virial(
            _molecularly_strained_positions(
                positions,
                source_cell=cell,
                target_cell=minus_cell,
                masses=masses,
                molecule_ids=molecule_ids,
            ),
            tuple(remaining_terms),
            cell=minus_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        diagonal.append(-(plus_energy - minus_energy) / (2.0 * strain_epsilon))
    generic = mx.diag(mx.stack(diagonal))
    return sum(specialized, start=generic)


def _molecularly_strained_positions(
    positions: mx.array,
    *,
    source_cell: Cell,
    target_cell: Cell,
    masses: mx.array | None,
    molecule_ids: object | None,
) -> mx.array:
    return molecularly_strained_positions(
        positions,
        source_cell=source_cell,
        target_cell=target_cell,
        masses=masses,
        molecule_ids=molecule_ids,
    )


def _normalize_pressure_molecule_ids(
    molecule_ids: object | None,
    *,
    particle_count: int,
) -> np.ndarray:
    return normalize_molecule_ids(
        molecule_ids,
        particle_count=particle_count,
    )


def _validate_orthorhombic_pressure_cell(cell: Cell) -> None:
    matrix = np.asarray(cell.matrix, dtype=np.float64)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    if not np.allclose(off_diagonal, 0.0, rtol=0.0, atol=1.0e-7):
        msg = "analytic pressure currently requires an orthorhombic cell"
        raise ValueError(msg)


def _strained_cell_matrix(matrix: mx.array, axis: int, strain: float) -> mx.array:
    factors = as_mx_array([1.0 + strain if item == axis else 1.0 for item in range(3)])
    return matrix * factors[:, None]


def _potential_energy_for_virial(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> mx.array:
    candidate_terms = force_terms
    candidate_pairs = pairs
    if any(getattr(term, "electrostatics", None) == "pme" for term in force_terms):
        candidate_terms = _cell_bound_force_terms(
            force_terms,
            cell,
            rebuild_plans=True,
        )
        if isinstance(pairs, NeighborBlocks):
            cutoffs = [
                float(term.cutoff)
                for term in candidate_terms
                if getattr(term, "cutoff", None) is not None
            ]
            if not cutoffs:
                msg = "PME virial oracle requires an explicit direct-space cutoff"
                raise ValueError(msg)
            evaluation_positions = _neighbor_evaluation_positions(
                positions,
                virtual_sites,
            )
            candidate_manager = NeighborListManager(
                cell,
                cutoff=max(cutoffs),
                skin=0.0,
                backend="mlx_cell_blocks",
            )
            candidate_pairs = candidate_manager.update(
                evaluation_positions,
            ).diagnostic_pairs
    energy, _ = _energy_forces_from_terms(
        positions,
        candidate_terms,
        cell=cell,
        pairs=candidate_pairs,
        virtual_sites=virtual_sites,
    )
    return energy


def pressure_tensor(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    kinetic_energy_scale: float = 1.0,
    virtual_sites: VirtualSiteManager | None = None,
    molecule_ids: object | None = None,
    virial_mode: str = VIRIAL_SUPPORT_ANALYTIC,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return virial tensor, pressure tensor, and scalar pressure diagnostics.

    The pressure tensor uses the reduced-unit convention
    ``P = (kinetic tensor + configurational virial) / V``. Periodic virials
    are diagonal-only orthorhombic cell-strain diagnostics; non-periodic runs
    report finite zero pressure diagnostics because no volume is defined.

    Args:
        positions: Particle coordinates, shape ``(n_particles, 3)``.
        velocities: Per-particle velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``.
        forces: Forces on each particle, shape ``(n_particles, 3)``.
        force_terms: Force terms supplying the configurational virial; each must
            support virial diagnostics.
        cell: Periodic cell; ``None`` reports zero pressure (no volume defined).
        pairs: Optional neighbor/pair structure forwarded to the force terms.
        kinetic_energy_scale: Energy-unit factor for the kinetic tensor.
            Defaults to ``1.0``.
        virtual_sites: Optional virtual-site manager applied before energy
            evaluation. Defaults to ``None``.
        molecule_ids: Optional exact molecule membership for molecular
            configurational and kinetic pressure.
        virial_mode: ``analytic`` for production or
            ``finite_difference_oracle`` for validation.

    Returns:
        A ``(virial, pressure, scalar)`` tuple: the ``(3, 3)`` configurational
            virial, the ``(3, 3)`` pressure tensor ``(kinetic + virial) / V``,
            and the scalar pressure ``tr(P) / 3``.

    Raises:
        ValueError: If a periodic cell has non-positive volume.
    """

    virial = configurational_virial_tensor(
        positions,
        forces,
        force_terms,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
        masses=masses,
        molecule_ids=molecule_ids,
        virial_mode=virial_mode,
    )
    if cell is None:
        zeros = mx.zeros((3, 3), dtype=virial.dtype)
        return virial, zeros, mx.sum(virial * 0.0)
    volume = cell.volume
    if float(np.asarray(volume)) <= 0.0:
        msg = "pressure diagnostics require positive cell volume"
        raise ValueError(msg)
    kinetic_tensor = kinetic_pressure_tensor(
        velocities,
        masses,
        kinetic_energy_scale=kinetic_energy_scale,
        molecule_ids=molecule_ids,
    )
    tensor = (kinetic_tensor + virial) / volume
    scalar = mx.trace(tensor) / 3.0
    return virial, tensor, scalar


def _pressure_diagnostics(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    forces: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    kinetic_energy_scale: float,
    enabled: bool,
    virtual_sites: VirtualSiteManager | None = None,
    virial_mode: str = VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE,
    molecule_ids: object | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    if enabled:
        return pressure_tensor(
            positions,
            velocities,
            masses,
            forces,
            force_terms,
            cell=cell,
            pairs=pairs,
            kinetic_energy_scale=kinetic_energy_scale,
            virtual_sites=virtual_sites,
            molecule_ids=molecule_ids,
            virial_mode=virial_mode,
        )
    zeros = mx.zeros((3, 3), dtype=positions.dtype)
    return zeros, zeros, mx.sum(positions[:, 0] * 0.0)


def virial_support_state(term: ForceTerm) -> str:
    """Return one force term's truthful virial capability."""

    if bool(getattr(term, "analytic_virial_supported", False)) and callable(
        getattr(term, "analytic_virial_tensor", None)
    ):
        return VIRIAL_SUPPORT_ANALYTIC
    if _term_supports_virial(term):
        return VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE
    return VIRIAL_SUPPORT_UNSUPPORTED


def virial_readiness_report(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
    *,
    require_analytic: bool = False,
) -> ReadinessReport:
    """Return per-term analytic, oracle-only, or unsupported virial readiness."""

    states = {name: virial_support_state(term) for name, term in _named_force_terms(force_terms)}
    unsupported = tuple(
        name for name, state in states.items() if state == VIRIAL_SUPPORT_UNSUPPORTED
    )
    oracle_only = tuple(
        name for name, state in states.items() if state == VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE
    )
    blockers = tuple(f"unsupported:{name}" for name in unsupported)
    if require_analytic:
        blockers += tuple(f"analytic_required:{name}" for name in oracle_only)
    if blockers:
        status = "blocked"
    elif oracle_only:
        status = "oracle_only"
    else:
        status = "ready"
    return ReadinessReport(
        name="virial",
        status=status,
        blockers=blockers,
        warnings=(
            tuple(f"finite_difference_oracle_only:{name}" for name in oracle_only)
            if not require_analytic
            else ()
        ),
        metadata={
            "require_analytic": require_analytic,
            "term_support": states,
        },
    )


def missing_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> tuple[str, ...]:
    """Return exact force-term names without a supported virial diagnostics path.

    Args:
        force_terms: A single force term or a list/tuple of force terms to inspect.

    Returns:
        The names of the terms lacking virial support, in input order
            (empty if all are supported).
    """

    missing = []
    for name, term in _named_force_terms(force_terms):
        if not _term_supports_virial(term):
            missing.append(name)
    return tuple(missing)


def missing_analytic_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> tuple[str, ...]:
    """Return exact force-term names lacking production analytic virial support."""

    return tuple(
        name
        for name, term in _named_force_terms(force_terms)
        if virial_support_state(term) != VIRIAL_SUPPORT_ANALYTIC
    )


def validate_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> None:
    """Fail closed when future pressure-coupled runtimes see unsupported terms.

    Args:
        force_terms: A single force term or a list/tuple of force terms to validate.

    Raises:
        ValueError: If any term lacks a supported virial diagnostics path.
    """

    missing = missing_virial_support(force_terms)
    if missing:
        msg = "missing virial support for force terms: " + ", ".join(missing)
        raise ValueError(msg)


def validate_analytic_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> None:
    """Fail closed when production pressure sees an oracle-only force term."""

    missing = missing_analytic_virial_support(force_terms)
    if missing:
        msg = "missing analytic virial support for force terms: " + ", ".join(missing)
        raise ValueError(msg)


def _validate_pressure_virial_support(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
    *,
    virial_mode: str,
) -> None:
    mode = normalize_virial_support(virial_mode)
    if mode == VIRIAL_SUPPORT_ANALYTIC:
        validate_analytic_virial_support(force_terms)
        return
    if mode == VIRIAL_SUPPORT_FINITE_DIFFERENCE_ORACLE:
        validate_virial_support(force_terms)
        return
    msg = "unsupported virial mode cannot produce pressure"
    raise ValueError(msg)


def _term_supports_virial(term: ForceTerm) -> bool:
    declared = getattr(term, "supports_virial", None)
    if declared is not None:
        return bool(declared)
    return callable(getattr(term, "virial_tensor", None)) or callable(
        getattr(term, "virial_diagnostics", None)
    )


def _evaluate_force_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    request: _ForceEvaluationRequest,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
    masses: mx.array | None = None,
    molecule_ids: object | None = None,
    named_force_terms: tuple[tuple[str, ForceTerm], ...] | None = None,
    cutoff_strain_pairs: mx.array | None = None,
) -> _ForceEvaluationResult:
    """Evaluate force terms for one exact internal output demand."""

    if request.mode == "diagnostic":
        named_terms = (
            _named_force_terms(force_terms) if named_force_terms is None else named_force_terms
        )
        if request.virial_mode == VIRIAL_SUPPORT_ANALYTIC:
            combined = _analytic_diagnostic_by_term(
                positions,
                named_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
                masses=masses,
                molecule_ids=molecule_ids,
                cutoff_strain_pairs=cutoff_strain_pairs,
            )
            if combined is not NotImplemented:
                (
                    energy,
                    forces,
                    components,
                    virial,
                    optimized_terms,
                    fallback_terms,
                ) = combined
                return _ForceEvaluationResult(
                    energy=energy,
                    forces=forces,
                    components=components,
                    virial=virial,
                    optimized_terms=optimized_terms,
                    fallback_terms=fallback_terms,
                )
        energy, forces, components = _energy_forces_by_term(
            positions,
            named_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        virial = None
        if request.virial_mode is not None:
            virial = configurational_virial_tensor(
                positions,
                forces,
                force_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
                masses=masses,
                molecule_ids=molecule_ids,
                virial_mode=request.virial_mode,
            )
        return _ForceEvaluationResult(
            energy=energy,
            forces=forces,
            components=components,
            virial=virial,
            fallback_terms=len(force_terms),
        )
    if request.mode == "energy_forces":
        energy, forces = _energy_forces_from_terms(
            positions,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        return _ForceEvaluationResult(
            energy=energy,
            forces=forces,
            fallback_terms=len(force_terms),
        )
    if request.mode == "forces":
        forces, optimized_terms, fallback_terms = _forces_from_term_demands(
            positions,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        return _ForceEvaluationResult(
            forces=forces,
            optimized_terms=optimized_terms,
            fallback_terms=fallback_terms,
        )
    energy, optimized_terms, fallback_terms = _energy_from_term_demands(
        positions,
        force_terms,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
    )
    return _ForceEvaluationResult(
        energy=energy,
        optimized_terms=optimized_terms,
        fallback_terms=fallback_terms,
    )


def _analytic_diagnostic_by_term(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None,
    masses: mx.array | None,
    molecule_ids: object | None,
    cutoff_strain_pairs: mx.array | None = None,
) -> (
    tuple[
        mx.array,
        mx.array,
        dict[str, mx.array],
        mx.array,
        int,
        int,
    ]
    | NotImplementedType
):
    """Evaluate analytic diagnostics through one owning term pass."""

    unnamed_terms = tuple(term for _, term in force_terms)
    if (
        cell is None
        or (virtual_sites is not None and virtual_sites.n_virtual_sites > 0)
        or _groupable_potential_terms(unnamed_terms, pairs)
    ):
        return NotImplemented
    validate_analytic_virial_support(unnamed_terms)
    _validate_orthorhombic_pressure_cell(cell)
    eval_positions = as_mx_array(positions)
    total_energy = None
    total_forces = mx.zeros_like(eval_positions)
    total_virial = mx.zeros((3, 3), dtype=eval_positions.dtype)
    energy_by_term: dict[str, mx.array] = {}
    optimized_terms = 0
    fallback_terms = 0

    for name, term in force_terms:
        combined_diagnostic = getattr(
            term,
            "_runtime_energy_forces_with_components_virial",
            None,
        )
        reuse_diagnostic = None
        reuses_cutoff_strain_pairs = False
        if cutoff_strain_pairs is not None:
            reuse_diagnostic = getattr(
                term,
                "_runtime_energy_forces_with_components_virial_reusing_pairs",
                None,
            )
            if callable(reuse_diagnostic):
                combined_diagnostic = reuse_diagnostic
                reuses_cutoff_strain_pairs = True
        result = (
            combined_diagnostic(
                eval_positions,
                cell,
                pairs,
                masses=masses,
                molecule_ids=molecule_ids,
                **(
                    {"cutoff_strain_pairs": cutoff_strain_pairs}
                    if reuses_cutoff_strain_pairs
                    else {}
                ),
            )
            if callable(combined_diagnostic)
            else NotImplemented
        )
        if result is NotImplemented:
            combined_components = getattr(
                term,
                "_runtime_energy_forces_with_components",
                None,
            )
            if not callable(combined_components):
                combined_components = getattr(
                    term,
                    "energy_forces_with_components",
                    None,
                )
            if callable(combined_components):
                energy, forces, components = combined_components(
                    eval_positions,
                    cell,
                    pairs=pairs,
                )
            else:
                energy, forces = term.energy_forces(
                    eval_positions,
                    cell,
                    pairs=pairs,
                )
                component_energies = getattr(
                    term,
                    "component_energies",
                    None,
                )
                components = (
                    component_energies(
                        eval_positions,
                        cell=cell,
                        pairs=pairs,
                    )
                    if callable(component_energies)
                    else {}
                )
            virial_method = getattr(
                term,
                "analytic_virial_tensor",
                None,
            )
            if not callable(virial_method):
                msg = "analytic virial support requires analytic_virial_tensor"
                raise ValueError(msg)
            term_virial = virial_method(
                eval_positions,
                cell=cell,
                pairs=pairs,
                masses=masses,
                molecule_ids=molecule_ids,
            )
            fallback_terms += 1
        else:
            energy, forces, components, term_virial = result
            optimized_terms += 1

        component_count = 0
        for component_name, component_energy in components.items():
            if _is_energy_component(component_energy):
                energy_by_term[f"{name}.{component_name}"] = component_energy
                component_count += 1
        if component_count == 0:
            energy_by_term[name] = energy
        term_virial = as_mx_array(term_virial)
        if term_virial.shape != (3, 3):
            msg = "analytic virial contributions must have shape (3, 3)"
            raise ValueError(msg)
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces
        total_virial = total_virial + mx.diag(mx.diag(term_virial))

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return (
        total_energy,
        total_forces,
        energy_by_term,
        total_virial,
        optimized_terms,
        fallback_terms,
    )


def _diagnostic_cutoff_strain_pairs(
    neighbor_manager: NeighborListManager | None,
    neighbor_list: NeighborList | None,
    cell: Cell | None,
    *,
    strain_epsilon: float = 1.0e-3,
) -> mx.array | None:
    """Return current Verlet pairs when they safely cover cell-strain work."""

    if (
        neighbor_manager is None
        or neighbor_list is None
        or cell is None
        or not cell.is_orthorhombic
        or neighbor_list is not neighbor_manager.neighbor_list
        or neighbor_manager.updates_since_check != 0
        or strain_epsilon <= 0.0
        or not np.isclose(
            float(neighbor_list.cutoff),
            float(neighbor_manager.cutoff),
            rtol=1.0e-7,
            atol=1.0e-8,
        )
        or not np.isclose(
            float(neighbor_list.skin),
            float(neighbor_manager.skin),
            rtol=1.0e-7,
            atol=1.0e-8,
        )
        or not isinstance(neighbor_list.diagnostic_pairs, mx.array)
        or not np.array_equal(
            np.asarray(neighbor_manager.cell.matrix),
            np.asarray(cell.matrix),
        )
    ):
        return None
    required_shell = (
        float(np.max(np.asarray(cell.lengths, dtype=np.float64))) * strain_epsilon * 1.1
    )
    remaining_pair_margin = float(neighbor_list.skin) - 2.0 * float(
        neighbor_manager.last_max_displacement
    )
    if remaining_pair_margin + 1.0e-7 < required_shell:
        return None
    return neighbor_list.diagnostic_pairs


def _forces_from_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> mx.array:
    if not any(callable(getattr(term, "_runtime_forces", None)) for term in force_terms):
        _, forces = _energy_forces_from_terms(
            positions,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        return forces
    result = _evaluate_force_terms(
        positions,
        force_terms,
        request=_FORCES_REQUEST,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
    )
    if result.forces is None:
        msg = "forces evaluation did not return forces"
        raise RuntimeError(msg)
    return result.forces


def _energy_from_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> mx.array:
    if pairs is not None and not any(
        callable(getattr(term, "_runtime_energy", None)) for term in force_terms
    ):
        energy, _ = _energy_forces_from_terms(
            positions,
            force_terms,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        return energy
    result = _evaluate_force_terms(
        positions,
        force_terms,
        request=_ENERGY_REQUEST,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
    )
    if result.energy is None:
        msg = "energy evaluation did not return energy"
        raise RuntimeError(msg)
    return result.energy


def _diagnostic_from_terms(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
    virial_mode: str | None = None,
    masses: mx.array | None = None,
    cutoff_strain_pairs: mx.array | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array], mx.array | None]:
    request = (
        _DIAGNOSTIC_REQUEST
        if virial_mode is None
        else _ForceEvaluationRequest("diagnostic", virial_mode=virial_mode)
    )
    result = _evaluate_force_terms(
        positions,
        tuple(term for _, term in force_terms),
        request=request,
        cell=cell,
        pairs=pairs,
        virtual_sites=virtual_sites,
        masses=masses,
        named_force_terms=force_terms,
        cutoff_strain_pairs=cutoff_strain_pairs,
    )
    if result.energy is None or result.forces is None:
        msg = "diagnostic evaluation did not return energy and forces"
        raise RuntimeError(msg)
    return result.energy, result.forces, result.components, result.virial


def _make_energy_forces_by_term_evaluator(
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    compile_evaluator: bool,
    virtual_sites: VirtualSiteManager | None = None,
    virial_mode: str | None = None,
    masses: mx.array | None = None,
    cutoff_strain_pairs: mx.array | None = None,
):
    unnamed_terms = tuple(term for _, term in force_terms)
    request = (
        _DIAGNOSTIC_REQUEST
        if virial_mode is None
        else _ForceEvaluationRequest("diagnostic", virial_mode=virial_mode)
    )

    def evaluate(pos: mx.array):
        result = _evaluate_force_terms(
            pos,
            unnamed_terms,
            request=request,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
            masses=masses,
            named_force_terms=force_terms,
            cutoff_strain_pairs=cutoff_strain_pairs,
        )
        if result.energy is None or result.forces is None:
            msg = "diagnostic evaluation did not return energy and forces"
            raise RuntimeError(msg)
        return result.energy, result.forces, result.components, result.virial

    if compile_evaluator and pairs is None and virtual_sites is None:
        return mx.compile(evaluate)
    return evaluate


def _make_forces_evaluator(
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    compile_evaluator: bool,
    virtual_sites: VirtualSiteManager | None = None,
):
    runtime_force_hooks = any(
        callable(getattr(term, "_runtime_forces", None)) for term in force_terms
    )

    def evaluate(pos: mx.array):
        if not runtime_force_hooks:
            _, forces = _energy_forces_from_terms(
                pos,
                force_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
            )
            return forces
        result = _evaluate_force_terms(
            pos,
            force_terms,
            request=_FORCES_REQUEST,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        if result.forces is None:
            msg = "forces evaluation did not return forces"
            raise RuntimeError(msg)
        return result.forces

    if compile_evaluator and pairs is None and virtual_sites is None:
        return mx.compile(evaluate)
    return evaluate


def _make_energy_forces_evaluator(
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    compile_evaluator: bool,
    virtual_sites: VirtualSiteManager | None = None,
):
    def evaluate(pos: mx.array):
        result = _evaluate_force_terms(
            pos,
            force_terms,
            request=_ENERGY_FORCES_REQUEST,
            cell=cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        if result.energy is None or result.forces is None:
            msg = "energy-forces evaluation did not return energy and forces"
            raise RuntimeError(msg)
        return result.energy, result.forces

    if compile_evaluator and pairs is None and virtual_sites is None:
        return mx.compile(evaluate)
    return evaluate


def _zero_reason_ints() -> dict[str, int]:
    return {reason: 0 for reason in RUNTIME_SYNC_REASONS}


def _zero_reason_floats() -> dict[str, float]:
    return {reason: 0.0 for reason in RUNTIME_SYNC_REASONS}


@dataclass
class _RuntimeSyncRecorder:
    sync_counts: dict[str, int] = field(default_factory=_zero_reason_ints)
    sync_wall_seconds: dict[str, float] = field(default_factory=_zero_reason_floats)
    materialization_counts: dict[str, int] = field(default_factory=_zero_reason_ints)
    materialization_wall_seconds: dict[str, float] = field(default_factory=_zero_reason_floats)

    def record_sync(self, reason: str, *values: mx.array) -> float:
        _validate_runtime_sync_reason(reason)
        start = perf_counter()
        mx.eval(*values)
        elapsed = perf_counter() - start
        self.sync_counts[reason] += 1
        self.sync_wall_seconds[reason] += elapsed
        return elapsed

    def record_materialization(self, reason: str, elapsed: float = 0.0) -> None:
        _validate_runtime_sync_reason(reason)
        self.materialization_counts[reason] += 1
        self.materialization_wall_seconds[reason] += elapsed

    def record_callback(self, reason: str, callback) -> None:
        _validate_runtime_sync_reason(reason)
        start = perf_counter()
        callback()
        self.record_materialization(reason, perf_counter() - start)

    def to_report(self) -> dict[str, int | float]:
        report: dict[str, int | float] = {
            "runtime_sync_total_count": sum(self.sync_counts.values()),
            "runtime_sync_total_wall_seconds": sum(self.sync_wall_seconds.values()),
            "runtime_materialization_total_count": sum(self.materialization_counts.values()),
            "runtime_materialization_total_wall_seconds": sum(
                self.materialization_wall_seconds.values()
            ),
        }
        for reason in RUNTIME_SYNC_REASONS:
            report[f"runtime_sync_{reason}_count"] = self.sync_counts[reason]
            report[f"runtime_sync_{reason}_wall_seconds"] = self.sync_wall_seconds[reason]
            report[f"runtime_materialization_{reason}_count"] = self.materialization_counts[reason]
            report[f"runtime_materialization_{reason}_wall_seconds"] = (
                self.materialization_wall_seconds[reason]
            )
        return report


def _validate_runtime_sync_reason(reason: str) -> None:
    if reason not in RUNTIME_SYNC_REASONS:
        msg = f"unknown runtime sync reason: {reason}"
        raise ValueError(msg)


def _constraint_profile_name(constraint: object, operation: str) -> str:
    family = getattr(constraint, "_profile_family", None)
    if family is None:
        family = (
            "settle"
            if isinstance(constraint, SettleWaterConstraints)
            else (
                "composite_constraints"
                if isinstance(constraint, CompositeConstraints)
                else "generic_constraints"
            )
        )
    return f"{family}_{operation}"


def _profile_constraint_positions(
    constraints: object,
    predicted_positions: mx.array,
    masses: mx.array,
    cell: Cell | None,
    profiler: _ExclusiveRouteProfiler,
    *,
    reference_positions: mx.array | None = None,
    validate: bool = True,
) -> tuple[mx.array, mx.array]:
    """Apply the production position-constraint route behind one profile barrier."""

    started = profiler.start()
    if validate:
        step_projector = getattr(constraints, "apply_position_step", None)
        if reference_positions is None or step_projector is None:
            constrained, error = constraints.apply_positions(
                predicted_positions,
                masses,
                cell,
            )
        else:
            constrained, error = step_projector(
                reference_positions,
                predicted_positions,
                masses,
                cell,
            )
    else:
        constrained = _project_constraint_positions_unchecked(
            constraints,
            predicted_positions,
            masses,
            cell,
            reference_positions=reference_positions,
        )
        error = _zero_constraint_error(constrained)
    profiler.finish(
        _constraint_profile_name(constraints, "position"),
        started,
        constrained,
        error,
    )
    return constrained, error


def _profile_constraint_pre_force_velocities(
    constraints: object,
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    cell: Cell | None,
    profiler: _ExclusiveRouteProfiler,
) -> mx.array:
    """Apply the production pre-force projection behind one profile barrier."""

    projector = getattr(
        constraints,
        "_apply_pre_force_velocities",
        constraints.apply_velocities,
    )
    started = profiler.start()
    constrained = projector(positions, velocities, masses, cell)
    profiler.finish(
        _constraint_profile_name(constraints, "pre_force_velocity"),
        started,
        constrained,
    )
    return constrained


def _profile_constraint_velocities(
    constraints: object,
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    cell: Cell | None,
    profiler: _ExclusiveRouteProfiler,
    *,
    operation: str = "velocity",
) -> mx.array:
    """Apply the production final velocity route behind one profile barrier."""

    started = profiler.start()
    constrained = constraints.apply_velocities(
        positions,
        velocities,
        masses,
        cell,
    )
    profiler.finish(
        _constraint_profile_name(constraints, operation),
        started,
        constrained,
    )
    return constrained


def _neighbor_profile_values(neighbor_list: NeighborList | None) -> tuple[mx.array, ...]:
    if neighbor_list is None:
        return ()
    values = []
    if neighbor_list.materialized_diagnostic_pairs is not None:
        values.append(neighbor_list.materialized_diagnostic_pairs)
    if neighbor_list.blocks is not None:
        values.extend(
            (
                neighbor_list.blocks.left,
                neighbor_list.blocks.right,
                neighbor_list.blocks.valid_mask,
            )
        )
    if neighbor_list.tiles is not None:
        values.extend(
            (
                neighbor_list.tiles.atom_blocks,
                neighbor_list.tiles.tile_blocks,
                neighbor_list.tiles.member_mask,
            )
        )
        if neighbor_list.tiles.force_columns is not None:
            values.extend(
                (
                    neighbor_list.tiles.force_columns,
                    neighbor_list.tiles.force_group_starts,
                    neighbor_list.tiles.force_group_counts,
                )
            )
    if neighbor_list.interaction32 is not None:
        schedule = neighbor_list.interaction32
        values.extend(
            (
                schedule.atom_order,
                schedule.ordinary_left_blocks,
                schedule.ordinary_right_atoms,
                schedule.ordinary_half_modes,
                schedule.ordinary_group_starts,
                schedule.ordinary_group_counts,
                schedule.special_work_left_blocks,
                schedule.special_work_left_slices,
                schedule.special_work_right_atoms,
                schedule.special_work_lj_enabled,
                schedule.special_work_lj_one_four,
                schedule.special_work_diagonal,
            )
        )
    if neighbor_list._diagnostic_tiles is not None:
        tiles = neighbor_list._diagnostic_tiles
        values.extend(
            (
                tiles.atom_blocks,
                tiles.tile_blocks,
                tiles.member_mask,
            )
        )
        if tiles.force_columns is not None:
            values.extend(
                (
                    tiles.force_columns,
                    tiles.force_group_starts,
                    tiles.force_group_counts,
                )
            )
    return tuple(values)


def _force_terms_support_tile_diagnostics(
    force_terms: tuple[ForceTerm, ...],
) -> bool:
    """Return whether every term declares exact-tile diagnostic ownership."""

    policies = tuple(
        getattr(term, "_neighbor_tile_diagnostic_policy", None) for term in force_terms
    )
    return "consume" in policies and all(policy in {"consume", "ignore"} for policy in policies)


def _validate_compact_nonbonded_backend(
    force_terms: tuple[ForceTerm, ...],
    *,
    neighbor_manager: NeighborListManager | None,
) -> None:
    for term in force_terms:
        topology = getattr(term, "topology", None)
        if topology is None or getattr(topology, "nonbonded_pair_policy", None) != "lazy":
            continue
        if neighbor_manager is not None:
            continue
        msg = (
            "large lazy topology requires compact periodic neighbor pairs; "
            "dense/tiled all-pairs fallback is refused"
        )
        raise ValueError(msg)


def _nonbonded_runtime_report(
    positions: mx.array,
    *,
    neighbor_manager: NeighborListManager | None,
    neighbor_list: NeighborList | None,
    force_evaluation_wall_seconds: float = 0.0,
    runtime_sync_report: dict[str, int | float] | None = None,
) -> dict[str, int | float | str | None]:
    if neighbor_list is None:
        dense_pair_count = _dense_pair_count(positions)
        report = {
            "backend": "dense_all_pairs",
            "pair_count": dense_pair_count,
            "compact_pair_count": dense_pair_count,
            "cutoff": None,
            "skin": None,
            "rebuild_count": 0,
            "estimated_pair_memory_bytes": dense_pair_count * 2 * 4,
            "estimated_compact_pair_memory_bytes": dense_pair_count * 2 * 4,
            "diagnostic_pairs_materialized": True,
            "estimated_cell_list_memory_bytes": 0,
            "representation_kind": "pairs",
            "candidate_count": dense_pair_count,
            "candidate_waste_count": 0,
            "candidate_waste_fraction": 0.0,
            "estimated_candidate_memory_bytes": 0,
            "compaction_backend": None,
            "fallback_reason": None,
            "adaptation_reason": None,
            "two_level_schedule_admitted": None,
            "two_level_observed_generations": 0,
            "two_level_mean_generation_updates": None,
            "neighbor_update_wall_seconds": 0.0,
            "neighbor_rebuild_wall_seconds": 0.0,
            "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
        }
        return _with_runtime_sync_report(report, runtime_sync_report)
    observed_generations = (
        0
        if neighbor_manager is None
        else neighbor_manager._interaction32_two_level_observed_generations
    )
    observed_updates = (
        0
        if neighbor_manager is None
        else neighbor_manager._interaction32_two_level_observed_updates
    )
    report = {
        "backend": neighbor_list.backend,
        "pair_count": neighbor_list.pair_count,
        "compact_pair_count": neighbor_list.compact_pair_count,
        "cutoff": neighbor_list.cutoff,
        "skin": neighbor_list.skin,
        "rebuild_count": 0 if neighbor_manager is None else neighbor_manager.rebuild_count,
        "estimated_pair_memory_bytes": neighbor_list.estimated_pair_bytes,
        "estimated_compact_pair_memory_bytes": neighbor_list.estimated_compact_pair_bytes,
        "diagnostic_pairs_materialized": (neighbor_list.diagnostic_pairs_materialized),
        "estimated_cell_list_memory_bytes": neighbor_list.estimated_cell_list_bytes,
        "representation_kind": neighbor_list.representation_kind,
        "candidate_count": neighbor_list.candidate_count,
        "candidate_waste_count": neighbor_list.candidate_waste_count,
        "candidate_waste_fraction": neighbor_list.candidate_waste_fraction,
        "estimated_candidate_memory_bytes": neighbor_list.estimated_candidate_bytes,
        "compaction_backend": neighbor_list.compaction_backend,
        "fallback_reason": neighbor_list.fallback_reason,
        "adaptation_reason": neighbor_list.adaptation_reason,
        "two_level_schedule_admitted": (
            None if neighbor_manager is None else neighbor_manager._interaction32_two_level_admitted
        ),
        "two_level_observed_generations": observed_generations,
        "two_level_mean_generation_updates": (
            None if observed_generations == 0 else observed_updates / observed_generations
        ),
        "neighbor_update_wall_seconds": (
            0.0 if neighbor_manager is None else neighbor_manager.update_wall_seconds
        ),
        "neighbor_rebuild_wall_seconds": (
            0.0 if neighbor_manager is None else neighbor_manager.rebuild_wall_seconds
        ),
        "force_evaluation_wall_seconds": force_evaluation_wall_seconds,
    }
    return _with_runtime_sync_report(report, runtime_sync_report)


def _with_runtime_sync_report(
    report: dict[str, int | float | str | None],
    runtime_sync_report: dict[str, int | float] | None,
) -> dict[str, int | float | str | None]:
    if runtime_sync_report is None:
        return report
    report.update(runtime_sync_report)
    return report


def _eval_step_state(
    state: SimulationState,
    potential_energy: mx.array,
    kinetic_energy_value: mx.array,
    temperature_value: mx.array,
    virial_value: mx.array,
    pressure_tensor_value: mx.array,
    pressure_value: mx.array,
    constraint_error: mx.array,
    energy_by_term: dict[str, mx.array],
    *,
    evaluate_sampled_state: bool = True,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    sync_reason: str = "diagnostic",
) -> float:
    state_values = (state.positions, state.velocities) if evaluate_sampled_state else ()
    values = (
        *state_values,
        state.forces,
        potential_energy,
        kinetic_energy_value,
        temperature_value,
        virial_value,
        pressure_tensor_value,
        pressure_value,
        constraint_error,
        *energy_by_term.values(),
    )
    if runtime_sync is not None:
        return runtime_sync.record_sync(sync_reason, *values)
    start = perf_counter()
    mx.eval(*values)
    return perf_counter() - start


def _eval_runtime_state(
    state: SimulationState,
    potential_energy: mx.array,
    constraint_error: mx.array,
    *,
    evaluate_sampled_state: bool = True,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    sync_reason: str = "failure_check",
) -> float:
    state_values = (state.positions, state.velocities) if evaluate_sampled_state else ()
    values = (
        *state_values,
        state.forces,
        potential_energy,
        constraint_error,
    )
    if runtime_sync is not None:
        return runtime_sync.record_sync(sync_reason, *values)
    start = perf_counter()
    mx.eval(*values)
    return perf_counter() - start


def _is_diagnostic_step(step: int, config: SimulationConfig, *, final: bool = False) -> bool:
    return step % config.diagnostic_interval == 0 or final


def _langevin_block_execution_enabled(
    config: SimulationConfig,
    *,
    thermostat: object,
    neighbor_manager: NeighborListManager | None,
    constraints: object | None,
    virtual_sites: VirtualSiteManager | None,
) -> bool:
    """Whether the compiled batched-block fast path applies to this NVT run.

    The fast path runs `block_size` velocity-Verlet/Langevin substeps as one
    compiled block between neighbor rebuilds, syncing to the host once per block
    instead of every step. It is only safe when nothing in the loop needs
    per-step host interaction or per-step force bookkeeping: a Langevin
    thermostat (deterministic threaded PRNG, compiles cleanly), a managed
    neighbor list, no constraints, and no virtual sites. Block length is capped
    at the next sampling/diagnostic boundary at run time, so recording cadences
    need not divide `block_size` and no recorded step is ever skipped.
    """

    return (
        config.block_size > 1
        and isinstance(thermostat, LangevinThermostat)
        and neighbor_manager is not None
        and constraints is None
        and virtual_sites is None
        and config.center_of_mass_motion_interval is None
    )


def _async_force_submission_enabled(
    config: SimulationConfig,
    *,
    thermostat: object,
    constraints: object | None,
    prepared_force_pipeline: _PreparedForcePipeline | None,
    neighbor_list: NeighborList | None,
) -> bool:
    """Whether one NVT step may submit its force graph without blocking."""

    return (
        not config.runtime_profile
        and isinstance(thermostat, LangevinThermostat)
        and constraints is not None
        and prepared_force_pipeline is not None
        and neighbor_list is not None
        and neighbor_list.supports_async_force_submission
        and "gpu" in str(mx.default_device()).lower()
        and callable(getattr(mx, "async_eval", None))
    )


def _normalize_reporters(
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None,
) -> tuple[RuntimeReporter, ...]:
    if reporters is None:
        return ()
    if isinstance(reporters, (list, tuple)):
        return tuple(reporters)
    return (reporters,)


def _notify_reporters(
    reporters: tuple[RuntimeReporter, ...],
    event: ReporterEvent,
    *,
    runtime_sync: _RuntimeSyncRecorder | None = None,
) -> None:
    for reporter in reporters:
        if runtime_sync is None:
            reporter(event)
        else:
            runtime_sync.record_callback(
                "reporter",
                lambda reporter=reporter, event=event: reporter(event),
            )


def _zero_constraint_error(positions: mx.array) -> mx.array:
    return mx.array(0.0, dtype=positions.dtype)


def _materialize_sampled_state(
    state: SimulationState,
    *,
    runtime_sync: _RuntimeSyncRecorder | None = None,
    reason: str = "explicit_user_output",
) -> None:
    # Sampled frames may be retained until trajectory serialization; force
    # evaluation so long sampled runs do not retain unevaluated step graphs.
    if runtime_sync is None:
        mx.eval(state.positions, state.velocities)
        return
    runtime_sync.record_sync(reason, state.positions, state.velocities)
    runtime_sync.record_materialization(reason)


def _local_prng_key(seed: int | None) -> mx.array:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return mx.random.key(seed)


def _advance_prng_key(key: mx.array, steps: int) -> mx.array:
    for _ in range(int(steps)):
        key = mx.random.split(key, 2)[0]
    return key


def _nose_hoover_thermal_mass(
    thermostat: NoseHooverThermostat,
    *,
    dof: int,
    boltzmann_constant: float,
) -> float:
    if thermostat.thermal_mass is not None:
        return float(thermostat.thermal_mass)
    return (
        float(dof)
        * boltzmann_constant
        * thermostat.temperature
        * thermostat.relaxation_time
        * thermostat.relaxation_time
    )


def _thermostat_metadata(
    thermostat: Thermostat,
    *,
    dof: int,
    boltzmann_constant: float,
    chain_position: float | None = None,
    chain_velocity: float | None = None,
    rng_step_offset: int | None = None,
) -> dict[str, Any]:
    if isinstance(thermostat, NoseHooverThermostat):
        return {
            "family": "nose_hoover",
            "integrator": "nose_hoover_velocity_verlet",
            "deterministic_state": True,
            "temperature": float(thermostat.temperature),
            "relaxation_time": float(thermostat.relaxation_time),
            "thermal_mass": _nose_hoover_thermal_mass(
                thermostat,
                dof=dof,
                boltzmann_constant=boltzmann_constant,
            ),
            "chain_position": float(
                thermostat.chain_position if chain_position is None else chain_position
            ),
            "chain_velocity": float(
                thermostat.chain_velocity if chain_velocity is None else chain_velocity
            ),
        }
    return {
        "family": "langevin_baoab",
        "integrator": "baoab",
        "temperature": float(thermostat.temperature),
        "friction": float(thermostat.friction),
        "seed": thermostat.seed,
        "rng_step_offset": (
            thermostat.rng_step_offset if rng_step_offset is None else int(rng_step_offset)
        ),
    }


@dataclass(frozen=True)
class VelocityVerlet:
    """Velocity Verlet integrator."""

    dt: float

    def step(
        self,
        positions: mx.array,
        velocities: mx.array,
        masses: mx.array,
        potential: LennardJonesPotential,
        *,
        cell: Cell | None = None,
        forces: mx.array | None = None,
        pairs: object | None = None,
    ) -> StepState:
        """Advance one MD step.

        Args:
            positions: Current coordinates, shape ``(n_particles, 3)``.
            velocities: Current velocities, shape ``(n_particles, 3)``.
            masses: Per-particle masses, shape ``(n_particles,)``.
            potential: Force model providing ``energy_forces``.
            cell: Optional periodic cell; positions are wrapped into it when given.
                Defaults to ``None``.
            forces: Optional forces at the current positions to skip a recompute;
                ``None`` evaluates them. Defaults to ``None``.
            pairs: Optional neighbor/pair structure passed to the potential.
                Defaults to ``None``.

        Returns:
            The `StepState` after one Velocity Verlet step (new positions,
                velocities, forces, and energies).
        """

        if forces is None:
            _, forces = potential.energy_forces(positions, cell, pairs=pairs)

        acceleration = forces / masses[:, None]
        velocities_half = velocities + 0.5 * self.dt * acceleration
        next_positions = positions + self.dt * velocities_half
        if cell is not None:
            next_positions = cell.wrap(next_positions)

        potential_energy, next_forces = potential.energy_forces(next_positions, cell, pairs=pairs)
        next_acceleration = next_forces / masses[:, None]
        next_velocities = velocities_half + 0.5 * self.dt * next_acceleration

        return StepState(
            positions=next_positions,
            velocities=next_velocities,
            forces=next_forces,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energy(next_velocities, masses),
        )


def simulate(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    potential: LennardJonesPotential | None = None,
    pairs: object | None = None,
    dt: float = 0.005,
    steps: int = 100,
) -> SimulationResult:
    """Run a short NVE MD simulation in reduced units.

    Args:
        positions: Initial coordinates, shape ``(n_particles, 3)``.
        velocities: Initial velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``; ``None`` uses unit
            masses. Defaults to ``None``.
        cell: Optional periodic cell for minimum-image distances and wrapping.
            Defaults to ``None``.
        potential: Force model to integrate; ``None`` uses a default
            `LennardJonesPotential`. Defaults to ``None``.
        pairs: Optional precomputed neighbor/pair structure passed to the
            potential. Defaults to ``None``.
        dt: Integration time step. Defaults to ``0.005``.
        steps: Number of Velocity Verlet steps. Defaults to ``100``.

    Returns:
        A `SimulationResult` with stacked per-frame positions,
            velocities, and energy/temperature series (``steps + 1`` frames).
    """

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    if potential is None:
        potential = LennardJonesPotential()

    potential_energy, forces = potential.energy_forces(positions, cell, pairs=pairs)
    state = StepState(
        positions=positions,
        velocities=velocities,
        forces=forces,
        potential_energy=potential_energy,
        kinetic_energy=kinetic_energy(velocities, masses),
    )

    position_frames = [state.positions]
    velocity_frames = [state.velocities]
    potential_energies = [state.potential_energy]
    kinetic_energies = [state.kinetic_energy]
    temperatures = [instantaneous_temperature(state.velocities, masses)]

    integrator = VelocityVerlet(dt)
    for _ in range(steps):
        state = integrator.step(
            state.positions,
            state.velocities,
            masses,
            potential,
            cell=cell,
            forces=state.forces,
            pairs=pairs,
        )
        position_frames.append(state.positions)
        velocity_frames.append(state.velocities)
        potential_energies.append(state.potential_energy)
        kinetic_energies.append(state.kinetic_energy)
        temperatures.append(instantaneous_temperature(state.velocities, masses))

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    return SimulationResult(
        positions=mx.stack(position_frames),
        velocities=mx.stack(velocity_frames),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        temperature=mx.stack(temperatures),
    )


@_bounded_metal_md_cache()
def simulate_nve(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NVEResult:
    """Run NVE molecular dynamics with sparse trajectory and configurable diagnostics.

    ``sample_interval`` controls trajectory storage. ``diagnostic_interval``
    controls energy, temperature, pair-count, and constraint diagnostics.

    Args:
        positions: Initial coordinates, shape ``(n_particles, 3)``.
        velocities: Initial velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``; ``None`` uses unit
            masses. Defaults to ``None``.
        cell: Optional periodic cell. Defaults to ``None``.
        force_terms: One or more force terms; ``None`` uses a default
            `LennardJonesPotential`. Defaults to ``None``.
        neighbor_manager: Optional neighbor-list manager for compact nonbonded
            backends. Defaults to ``None``.
        config: Run configuration (step count, sampling/diagnostic intervals,
            virtual sites); ``None`` uses defaults. Defaults to ``None``.
        constraints: Optional distance constraints applied each step.
            Defaults to ``None``.
        reporters: Optional runtime reporter(s) invoked on diagnostic events.
            Defaults to ``None``.

    Returns:
        An `NVEResult` with the sparse trajectory, diagnostics, and
            energy-drift metrics.
    """

    if config is None:
        config = SimulationConfig()
    virtual_sites = config.virtual_sites
    reporters_tuple = _normalize_reporters(reporters)
    runtime_sync = _RuntimeSyncRecorder()
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _named_force_terms(force_terms)
    unnamed_terms = tuple(term for _, term in terms)
    if config.pressure_diagnostics:
        _validate_pressure_virial_support(
            unnamed_terms,
            virial_mode=config.pressure_virial_mode,
        )
    _validate_compact_nonbonded_backend(
        unnamed_terms,
        neighbor_manager=neighbor_manager,
    )

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    constraint_error = _zero_constraint_error(positions)
    if constraints is not None:
        positions, constraint_error = constraints.apply_positions(positions, masses, cell)
        velocities = constraints.apply_velocities(positions, velocities, masses, cell)
    temperature_dof = _temperature_degrees_of_freedom(positions, constraints)

    eval_positions = _neighbor_evaluation_positions(positions, virtual_sites)
    neighbor_list = (
        neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
    )
    prepared_force_pipeline = (
        None
        if neighbor_manager is None or neighbor_manager.check_interval != 1
        else _PreparedForcePipeline.prepare(
            unnamed_terms,
            cell=cell,
            virtual_sites=virtual_sites,
        )
    )
    force_binding = (
        None if prepared_force_pipeline is None else prepared_force_pipeline.bind(neighbor_list)
    )
    bound_neighbor_list = neighbor_list if force_binding is not None else None
    if neighbor_list is None:
        pairs = None
    elif (
        not config.pressure_diagnostics
        and force_binding is not None
        and force_binding.interactions is None
        and neighbor_list.supports_tile_diagnostics
        and _force_terms_support_tile_diagnostics(unnamed_terms)
    ):
        pairs = neighbor_list.diagnostic_force_candidates(prefer_tiles=True)
    else:
        pairs = neighbor_list.diagnostic_pairs
    pair_count = (
        _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    cutoff_strain_pairs = (
        _diagnostic_cutoff_strain_pairs(
            neighbor_manager,
            neighbor_list,
            cell,
        )
        if config.pressure_diagnostics
        else None
    )
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
        virial_mode=(config.pressure_virial_mode if config.pressure_diagnostics else None),
        masses=masses,
        cutoff_strain_pairs=cutoff_strain_pairs,
    )
    forces_evaluator = _make_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
    )
    force_start = perf_counter()
    potential_energy, forces, energy_by_term, diagnostic_virial = energy_forces_by_term(positions)
    force_evaluation_wall_seconds += perf_counter() - force_start
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
        step=config.initial_step,
        time=config.initial_time,
    )

    _materialize_sampled_state(
        state,
        runtime_sync=runtime_sync,
        reason="explicit_user_output",
    )
    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [config.initial_step]
    sampled_times = [config.initial_time]
    diagnostic_steps = [config.initial_step]
    diagnostic_times = [config.initial_time]
    potential_energies = [potential_energy]
    potential_energy_by_term = {name: [energy] for name, energy in energy_by_term.items()}
    kinetic_energies = [
        kinetic_energy(
            state.velocities,
            masses,
            kinetic_energy_scale=config.kinetic_energy_scale,
        )
    ]
    temperatures = [
        instantaneous_temperature(
            state.velocities,
            masses,
            dof=temperature_dof,
            kinetic_energy_scale=config.kinetic_energy_scale,
            boltzmann_constant=config.boltzmann_constant,
        )
    ]
    virial, pressure_tensor_value, pressure_value = _pressure_diagnostics_from_virial(
        diagnostic_virial,
        state.positions,
        state.velocities,
        masses,
        cell=cell,
        kinetic_energy_scale=config.kinetic_energy_scale,
        enabled=config.pressure_diagnostics,
    )
    virials = [virial]
    pressure_tensors = [pressure_tensor_value]
    pressures = [pressure_value]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]
    constraint_errors = [constraint_error]
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nve",
            event_type="sample",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
        ),
        runtime_sync=runtime_sync,
    )
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nve",
            event_type="diagnostic",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energies[-1],
            total_energy=potential_energy + kinetic_energies[-1],
            temperature=temperatures[-1],
            energy_by_term=energy_by_term,
            virial_tensor=virial,
            pressure_tensor=pressure_tensor_value,
            pressure=pressure_value,
            pair_count=pair_count,
            rebuild_count=rebuild_count,
            constraint_max_error=constraint_error,
        ),
        runtime_sync=runtime_sync,
    )

    for local_step in range(1, config.steps + 1):
        current_step = config.initial_step + local_step
        current_time = config.initial_time + local_step * config.dt
        acceleration = config.force_to_acceleration_scale * state.forces / masses[:, None]
        velocities_half = state.velocities + 0.5 * config.dt * acceleration
        next_positions = state.positions + config.dt * velocities_half
        if cell is not None and config.wrap_positions:
            next_positions = cell.wrap(next_positions)
        constraint_error = _zero_constraint_error(next_positions)
        if constraints is not None:
            next_positions, constraint_error = constraints.apply_positions(
                next_positions,
                masses,
                cell,
            )

        eval_positions = _neighbor_evaluation_positions(next_positions, virtual_sites)
        neighbor_list = (
            neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
        )
        if prepared_force_pipeline is not None and neighbor_list is not bound_neighbor_list:
            force_binding = prepared_force_pipeline.bind(neighbor_list)
            bound_neighbor_list = neighbor_list
        pairs = (
            None if neighbor_list is None else neighbor_list.force_candidates(prefer_tiles=False)
        )
        pair_count = (
            _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
        )
        rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

        diagnostic_step = _is_diagnostic_step(
            current_step,
            config,
            final=local_step == config.steps,
        )
        force_start = perf_counter()
        diagnostic_virial = None
        if neighbor_manager is None and diagnostic_step:
            (
                potential_energy,
                next_forces,
                energy_by_term,
                diagnostic_virial,
            ) = energy_forces_by_term(next_positions)
        elif neighbor_manager is None:
            potential_energy = None
            next_forces = forces_evaluator(next_positions)
            energy_by_term = None
        elif diagnostic_step:
            cutoff_strain_pairs = _diagnostic_cutoff_strain_pairs(
                neighbor_manager,
                neighbor_list,
                cell,
            )
            (
                potential_energy,
                next_forces,
                energy_by_term,
                diagnostic_virial,
            ) = _diagnostic_from_terms(
                next_positions,
                terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
                virial_mode=(config.pressure_virial_mode if config.pressure_diagnostics else None),
                masses=masses,
                cutoff_strain_pairs=cutoff_strain_pairs,
            )
        else:
            potential_energy = None
            next_forces = (
                _forces_from_terms(
                    next_positions,
                    unnamed_terms,
                    cell=cell,
                    pairs=pairs,
                    virtual_sites=virtual_sites,
                )
                if force_binding is None
                else force_binding.forces(
                    next_positions,
                    evaluation_positions=eval_positions,
                )
            )
            energy_by_term = None
        force_evaluation_wall_seconds += perf_counter() - force_start
        next_acceleration = config.force_to_acceleration_scale * next_forces / masses[:, None]
        next_velocities = velocities_half + 0.5 * config.dt * next_acceleration
        if constraints is not None:
            next_velocities = constraints.apply_velocities(
                next_positions,
                next_velocities,
                masses,
                cell,
            )
        state = SimulationState(
            positions=next_positions,
            velocities=next_velocities,
            masses=masses,
            forces=next_forces,
            step=current_step,
            time=current_time,
        )

        sampled_state_evaluated = False
        if current_step % config.sample_interval == 0 or local_step == config.steps:
            _materialize_sampled_state(
                state,
                runtime_sync=runtime_sync,
                reason="final_state" if local_step == config.steps else "explicit_user_output",
            )
            sampled_state_evaluated = True
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(current_step)
            sampled_times.append(state.time)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nve",
                    event_type="sample",
                    step=current_step,
                    time=state.time,
                    state=state,
                ),
                runtime_sync=runtime_sync,
            )
        if diagnostic_step:
            diagnostic_steps.append(current_step)
            diagnostic_times.append(state.time)
            potential_energies.append(potential_energy)
            if energy_by_term is not None:
                for name, energy in energy_by_term.items():
                    potential_energy_by_term[name].append(energy)
            kinetic_energies.append(
                kinetic_energy(
                    state.velocities,
                    masses,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                )
            )
            temperatures.append(
                instantaneous_temperature(
                    state.velocities,
                    masses,
                    dof=temperature_dof,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                    boltzmann_constant=config.boltzmann_constant,
                )
            )
            virial, pressure_tensor_value, pressure_value = _pressure_diagnostics_from_virial(
                diagnostic_virial,
                state.positions,
                state.velocities,
                masses,
                cell=cell,
                kinetic_energy_scale=config.kinetic_energy_scale,
                enabled=config.pressure_diagnostics,
            )
            virials.append(virial)
            pressure_tensors.append(pressure_tensor_value)
            pressures.append(pressure_value)
            pair_counts.append(pair_count)
            rebuild_counts.append(rebuild_count)
            constraint_errors.append(constraint_error)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nve",
                    event_type="diagnostic",
                    step=current_step,
                    time=state.time,
                    state=state,
                    potential_energy=potential_energy,
                    kinetic_energy=kinetic_energies[-1],
                    total_energy=potential_energy + kinetic_energies[-1],
                    temperature=temperatures[-1],
                    energy_by_term={} if energy_by_term is None else energy_by_term,
                    virial_tensor=virial,
                    pressure_tensor=pressure_tensor_value,
                    pressure=pressure_value,
                    pair_count=pair_count,
                    rebuild_count=rebuild_count,
                    constraint_max_error=constraint_error,
                ),
                runtime_sync=runtime_sync,
            )
        if current_step % config.evaluation_interval == 0 or local_step == config.steps:
            if diagnostic_step and energy_by_term is not None:
                _eval_step_state(
                    state,
                    potential_energy,
                    kinetic_energies[-1],
                    temperatures[-1],
                    virials[-1],
                    pressure_tensors[-1],
                    pressures[-1],
                    constraint_error,
                    energy_by_term,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="diagnostic",
                )
            else:
                _eval_runtime_state(
                    state,
                    potential_energy,
                    constraint_error,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="failure_check",
                )
    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    runtime_sync_report = runtime_sync.to_report()
    return NVEResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        diagnostic_steps=mx.array(diagnostic_steps, dtype=mx.int32),
        diagnostic_time=mx.array(diagnostic_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        potential_energy_by_term={
            name: mx.stack(energies) for name, energies in potential_energy_by_term.items()
        },
        temperature=mx.stack(temperatures),
        virial_tensor=mx.stack(virials),
        pressure_tensor=mx.stack(pressure_tensors),
        pressure=mx.stack(pressures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        constraint_max_error=mx.stack(constraint_errors),
        final_state=state,
        nonbonded_report=_nonbonded_runtime_report(
            _neighbor_evaluation_positions(state.positions, virtual_sites),
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
        runtime_sync_report=runtime_sync_report,
    )


def simulate_nvt(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: Thermostat | None = None,
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NVTResult:
    """Run NVT molecular dynamics with Langevin BAOAB or Nose-Hoover dynamics.

    Args:
        positions: Initial coordinates, shape ``(n_particles, 3)``.
        velocities: Initial velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``; ``None`` uses unit
            masses. Defaults to ``None``.
        cell: Optional periodic cell. Defaults to ``None``.
        force_terms: One or more force terms; ``None`` uses a default
            `LennardJonesPotential`. Defaults to ``None``.
        neighbor_manager: Optional neighbor-list manager for compact nonbonded
            backends. Defaults to ``None``.
        config: Run configuration; ``None`` uses defaults. Defaults to ``None``.
        thermostat: Langevin (BAOAB) or Nose-Hoover thermostat; ``None`` uses a
            default `LangevinThermostat`. Defaults to ``None``.
        constraints: Optional distance constraints applied each step.
            Defaults to ``None``.
        reporters: Optional runtime reporter(s). Defaults to ``None``.

    Returns:
        An `NVTResult` with the trajectory, diagnostics, and
            temperature-control metrics.
    """

    return _simulate_nvt(
        positions,
        velocities,
        masses=masses,
        cell=cell,
        force_terms=force_terms,
        neighbor_manager=neighbor_manager,
        config=config,
        thermostat=thermostat,
        constraints=constraints,
        reporters=reporters,
    )


@_bounded_metal_md_cache()
def _simulate_nvt(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: Thermostat | None = None,
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
    initial_diagnostics: _NVTBoundaryDiagnostics | None = None,
    defer_final_diagnostics: bool = False,
) -> NVTResult:
    """Run the NVT core with optional NPT boundary-diagnostic reuse."""

    if config is None:
        config = SimulationConfig()
    route_profiler = _ExclusiveRouteProfiler() if config.runtime_profile else None
    virtual_sites = config.virtual_sites
    reporters_tuple = _normalize_reporters(reporters)
    runtime_sync = _RuntimeSyncRecorder()
    if thermostat is None:
        thermostat = LangevinThermostat()
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = _named_force_terms(force_terms)
    unnamed_terms = tuple(term for _, term in terms)
    if config.pressure_diagnostics:
        _validate_pressure_virial_support(
            unnamed_terms,
            virial_mode=config.pressure_virial_mode,
        )
    _validate_compact_nonbonded_backend(
        unnamed_terms,
        neighbor_manager=neighbor_manager,
    )

    positions = as_mx_array(positions)
    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * positions.shape[0]) if masses is None else as_mx_array(masses)
    masses_col = masses[:, None]
    zero_constraint_error = _zero_constraint_error(positions)
    constraint_error = (
        zero_constraint_error
        if initial_diagnostics is None
        else initial_diagnostics.constraint_error
    )
    if constraints is not None and initial_diagnostics is None:
        if route_profiler is None:
            positions, constraint_error = constraints.apply_positions(
                positions,
                masses,
                cell,
            )
            velocities = constraints.apply_velocities(
                positions,
                velocities,
                masses,
                cell,
            )
        else:
            positions, constraint_error = _profile_constraint_positions(
                constraints,
                positions,
                masses,
                cell,
                route_profiler,
            )
            velocities = _profile_constraint_velocities(
                constraints,
                positions,
                velocities,
                masses,
                cell,
                route_profiler,
            )
    temperature_dof = _temperature_degrees_of_freedom(positions, constraints)

    eval_positions = _neighbor_evaluation_positions(positions, virtual_sites)
    neighbor_started = None if route_profiler is None else route_profiler.start()
    if neighbor_manager is None:
        neighbor_list = None
    elif initial_diagnostics is None:
        neighbor_list = neighbor_manager.update(eval_positions)
    else:
        _validate_neighbor_manager_cell(neighbor_manager, cell)
        neighbor_list = neighbor_manager.neighbor_list
        if neighbor_list is None:
            msg = "NPT boundary reuse requires an initialized neighbor list"
            raise RuntimeError(msg)
    if neighbor_started is not None:
        route_profiler.finish(
            "neighbor_update_rebuild",
            neighbor_started,
            _neighbor_profile_values(neighbor_list),
        )
    prepared_force_pipeline = (
        None
        if neighbor_manager is None or neighbor_manager.check_interval != 1
        else _PreparedForcePipeline.prepare(
            unnamed_terms,
            cell=cell,
            virtual_sites=virtual_sites,
            route_profiler=route_profiler,
        )
    )
    binding_started = None if route_profiler is None else route_profiler.start()
    force_binding = (
        None if prepared_force_pipeline is None else prepared_force_pipeline.bind(neighbor_list)
    )
    bound_neighbor_list = neighbor_list if force_binding is not None else None
    if binding_started is not None:
        route_profiler.finish(
            "neighbor_force_binding",
            binding_started,
        )
    if neighbor_list is None:
        pairs = None
    elif (
        not config.pressure_diagnostics
        and force_binding is not None
        and force_binding.interactions is None
        and neighbor_list.supports_tile_diagnostics
        and _force_terms_support_tile_diagnostics(unnamed_terms)
    ):
        pairs = neighbor_list.diagnostic_force_candidates(prefer_tiles=True)
    else:
        pairs = neighbor_list.diagnostic_pairs
    pair_count = (
        _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    cutoff_strain_pairs = (
        _diagnostic_cutoff_strain_pairs(
            neighbor_manager,
            neighbor_list,
            cell,
        )
        if config.pressure_diagnostics
        else None
    )
    force_evaluation_wall_seconds = 0.0
    energy_forces_by_term = _make_energy_forces_by_term_evaluator(
        terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
        virial_mode=(config.pressure_virial_mode if config.pressure_diagnostics else None),
        masses=masses,
        cutoff_strain_pairs=cutoff_strain_pairs,
    )
    energy_forces = _make_energy_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
    )
    forces_evaluator = _make_forces_evaluator(
        unnamed_terms,
        cell=cell,
        pairs=pairs,
        compile_evaluator=config.compile_force_evaluator and neighbor_manager is None,
        virtual_sites=virtual_sites,
    )
    force_start = perf_counter()
    diagnostic_profile_started = None if route_profiler is None else route_profiler.start()
    if initial_diagnostics is None:
        potential_energy, forces, energy_by_term, diagnostic_virial = energy_forces_by_term(
            positions
        )
    else:
        potential_energy = initial_diagnostics.potential_energy
        forces = initial_diagnostics.forces
        energy_by_term = dict(initial_diagnostics.energy_by_term)
        diagnostic_virial = initial_diagnostics.virial_tensor
    if diagnostic_profile_started is not None:
        route_profiler.finish(
            "diagnostics_reporting",
            diagnostic_profile_started,
            potential_energy,
            forces,
            energy_by_term,
            diagnostic_virial,
        )
    force_evaluation_wall_seconds += perf_counter() - force_start
    state = SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        forces=forces,
        step=config.initial_step,
        time=config.initial_time,
    )

    _materialize_sampled_state(
        state,
        runtime_sync=runtime_sync,
        reason="explicit_user_output",
    )
    sampled_positions = [state.positions]
    sampled_velocities = [state.velocities]
    sampled_steps = [config.initial_step]
    sampled_times = [config.initial_time]
    diagnostic_steps = [config.initial_step]
    diagnostic_times = [config.initial_time]
    potential_energies = [potential_energy]
    potential_energy_by_term = {name: [energy] for name, energy in energy_by_term.items()}
    kinetic_energies = [
        kinetic_energy(
            state.velocities,
            masses,
            kinetic_energy_scale=config.kinetic_energy_scale,
        )
    ]
    temperatures = [
        instantaneous_temperature(
            state.velocities,
            masses,
            dof=temperature_dof,
            kinetic_energy_scale=config.kinetic_energy_scale,
            boltzmann_constant=config.boltzmann_constant,
        )
    ]
    rng_step_offset = None
    key = None
    velocity_decay = None
    noise_scale = None
    thermal_scale = None
    metal_langevin_drift = None
    nh_chain_position = None
    nh_chain_velocity = None
    nh_thermal_mass = None
    nh_target_kinetic = None
    if isinstance(thermostat, LangevinThermostat):
        key = _local_prng_key(thermostat.seed)
        rng_step_offset = (
            config.initial_step
            if thermostat.rng_step_offset is None
            else thermostat.rng_step_offset
        )
        key = _advance_prng_key(key, rng_step_offset)
        velocity_decay = exp(-thermostat.friction * config.dt)
        noise_scale = sqrt(
            (1.0 - velocity_decay * velocity_decay)
            * thermostat.temperature
            * config.boltzmann_constant
            / config.kinetic_energy_scale
        )
        thermal_scale = noise_scale / mx.sqrt(masses)[:, None]
        if (
            mx.metal.is_available()
            and "gpu" in str(mx.default_device()).lower()
            and state.positions.dtype == mx.float32
            and (cell is None or cell.is_orthorhombic)
        ):
            metal_langevin_drift = (
                config.force_to_acceleration_scale / masses,
                mx.ones((3,), dtype=mx.float32) if cell is None else mx.diag(cell.matrix),
                mx.array(
                    [
                        0.5 * config.dt,
                        velocity_decay,
                        float(cell is not None and config.wrap_positions),
                    ],
                    dtype=mx.float32,
                ),
                mx.array([state.positions.shape[0]], dtype=mx.int32),
            )
        thermostat_metadata = _thermostat_metadata(
            thermostat,
            dof=temperature_dof,
            boltzmann_constant=config.boltzmann_constant,
            rng_step_offset=rng_step_offset,
        )
    else:
        nh_chain_position = mx.array(float(thermostat.chain_position))
        nh_chain_velocity = mx.array(float(thermostat.chain_velocity))
        nh_thermal_mass = mx.array(
            _nose_hoover_thermal_mass(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
            )
        )
        nh_target_kinetic = mx.array(
            float(temperature_dof) * config.boltzmann_constant * thermostat.temperature
        )
        thermostat_metadata = _thermostat_metadata(
            thermostat,
            dof=temperature_dof,
            boltzmann_constant=config.boltzmann_constant,
            chain_position=float(np.asarray(nh_chain_position)),
            chain_velocity=float(np.asarray(nh_chain_velocity)),
        )
    if initial_diagnostics is None:
        virial, pressure_tensor_value, pressure_value = _pressure_diagnostics_from_virial(
            diagnostic_virial,
            state.positions,
            state.velocities,
            masses,
            cell=cell,
            kinetic_energy_scale=config.kinetic_energy_scale,
            enabled=config.pressure_diagnostics,
        )
    else:
        virial = initial_diagnostics.virial_tensor
        pressure_tensor_value = initial_diagnostics.pressure_tensor
        pressure_value = initial_diagnostics.pressure
    virials = [virial]
    pressure_tensors = [pressure_tensor_value]
    pressures = [pressure_value]
    pair_counts = [pair_count]
    rebuild_counts = [rebuild_count]
    constraint_errors = [constraint_error]
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nvt",
            event_type="sample",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
            thermostat=thermostat_metadata,
        ),
        runtime_sync=runtime_sync,
    )
    _notify_reporters(
        reporters_tuple,
        ReporterEvent(
            ensemble="nvt",
            event_type="diagnostic",
            step=config.initial_step,
            time=config.initial_time,
            state=state,
            potential_energy=potential_energy,
            kinetic_energy=kinetic_energies[-1],
            total_energy=potential_energy + kinetic_energies[-1],
            temperature=temperatures[-1],
            energy_by_term=energy_by_term,
            virial_tensor=virial,
            pressure_tensor=pressure_tensor_value,
            pressure=pressure_value,
            pair_count=pair_count,
            rebuild_count=rebuild_count,
            constraint_max_error=constraint_error,
            thermostat=thermostat_metadata,
        ),
        runtime_sync=runtime_sync,
    )

    _batched = route_profiler is None and _langevin_block_execution_enabled(
        config,
        thermostat=thermostat,
        neighbor_manager=neighbor_manager,
        constraints=constraints,
        virtual_sites=virtual_sites,
    )
    async_force_submission = _async_force_submission_enabled(
        config,
        thermostat=thermostat,
        constraints=constraints,
        prepared_force_pipeline=prepared_force_pipeline,
        neighbor_list=neighbor_list,
    )
    if _batched:
        if (
            cell is None
            or neighbor_manager is None
            or neighbor_list is None
            or key is None
            or velocity_decay is None
            or noise_scale is None
        ):
            msg = "compiled Langevin blocks require initialized periodic runtime state"
            raise RuntimeError(msg)
        block_force_evaluator = partial(
            _forces_from_terms,
            force_terms=unnamed_terms,
            cell=cell,
            virtual_sites=None,
        )
        block_executor = _LangevinBlockExecutor(
            cell=cell,
            wrap_positions=config.wrap_positions,
            dt=config.dt,
            force_to_acceleration_scale=config.force_to_acceleration_scale,
            masses=masses,
            velocity_decay=velocity_decay,
            noise_scale=noise_scale,
            neighbor_manager=neighbor_manager,
            force_evaluator=block_force_evaluator,
            prepared_force_pipeline=prepared_force_pipeline,
        )

        def _run_langevin_batched(
            state,
            key,
            thermostat_metadata,
            neighbor_list,
            pairs,
            pair_count,
            rebuild_count,
            fe_wall,
        ):
            block_state = _LangevinBlockState(
                positions=state.positions,
                velocities=state.velocities,
                forces=state.forces,
                key=key,
                pairs=pairs,
                pair_count=pair_count,
                rebuild_count=rebuild_count,
                neighbor_list=neighbor_list,
            )
            local_step = 0
            while local_step < config.steps:
                next_recording_step = _next_recording_local_step(config, local_step)
                n = min(config.block_size, next_recording_step - local_step)
                block_state = block_executor.advance(block_state, n)
                local_step += n
                current_step = config.initial_step + local_step
                current_time = config.initial_time + local_step * config.dt
                pos = block_state.positions
                vel = block_state.velocities
                forces = block_state.forces
                pairs = block_state.pairs
                pair_count = block_state.pair_count
                rebuild_count = block_state.rebuild_count
                neighbor_list = block_state.neighbor_list

                state = SimulationState(
                    positions=pos,
                    velocities=vel,
                    masses=masses,
                    forces=forces,
                    step=current_step,
                    time=current_time,
                )
                thermostat_metadata = _thermostat_metadata(
                    thermostat,
                    dof=temperature_dof,
                    boltzmann_constant=config.boltzmann_constant,
                    rng_step_offset=rng_step_offset + local_step,
                )

                diagnostic_step = _is_diagnostic_step(
                    current_step, config, final=local_step == config.steps
                )
                deferred_final = defer_final_diagnostics and local_step == config.steps
                energy_by_term = None
                potential_energy = None
                diagnostic_virial = None
                if diagnostic_step:
                    force_start = perf_counter()
                    if deferred_final:
                        potential_energy = _energy_from_terms(
                            pos,
                            unnamed_terms,
                            cell=cell,
                            pairs=pairs,
                            virtual_sites=None,
                        )
                        energy_by_term = {
                            name: energies[-1]
                            for name, energies in potential_energy_by_term.items()
                        }
                    else:
                        cutoff_strain_pairs = _diagnostic_cutoff_strain_pairs(
                            neighbor_manager,
                            neighbor_list,
                            cell,
                        )
                        (
                            potential_energy,
                            _,
                            energy_by_term,
                            diagnostic_virial,
                        ) = _diagnostic_from_terms(
                            pos,
                            terms,
                            cell=cell,
                            pairs=pairs,
                            virtual_sites=None,
                            virial_mode=(
                                config.pressure_virial_mode if config.pressure_diagnostics else None
                            ),
                            masses=masses,
                            cutoff_strain_pairs=cutoff_strain_pairs,
                        )
                    fe_wall += perf_counter() - force_start

                sampled_state_evaluated = False
                if current_step % config.sample_interval == 0 or local_step == config.steps:
                    _materialize_sampled_state(
                        state,
                        runtime_sync=runtime_sync,
                        reason="final_state"
                        if local_step == config.steps
                        else "explicit_user_output",
                    )
                    sampled_state_evaluated = True
                    sampled_positions.append(state.positions)
                    sampled_velocities.append(state.velocities)
                    sampled_steps.append(current_step)
                    sampled_times.append(state.time)
                    _notify_reporters(
                        reporters_tuple,
                        ReporterEvent(
                            ensemble="nvt",
                            event_type="sample",
                            step=current_step,
                            time=state.time,
                            state=state,
                            thermostat=thermostat_metadata,
                        ),
                        runtime_sync=runtime_sync,
                    )
                if diagnostic_step:
                    diagnostic_steps.append(current_step)
                    diagnostic_times.append(state.time)
                    potential_energies.append(potential_energy)
                    for name, energy in energy_by_term.items():
                        potential_energy_by_term[name].append(energy)
                    kinetic_energies.append(
                        kinetic_energy(
                            state.velocities,
                            masses,
                            kinetic_energy_scale=config.kinetic_energy_scale,
                        )
                    )
                    temperatures.append(
                        instantaneous_temperature(
                            state.velocities,
                            masses,
                            dof=temperature_dof,
                            kinetic_energy_scale=config.kinetic_energy_scale,
                            boltzmann_constant=config.boltzmann_constant,
                        )
                    )
                    if deferred_final:
                        virial = virials[-1]
                        pressure_tensor_value = pressure_tensors[-1]
                        pressure_value = pressures[-1]
                    else:
                        virial, pressure_tensor_value, pressure_value = (
                            _pressure_diagnostics_from_virial(
                                diagnostic_virial,
                                state.positions,
                                state.velocities,
                                masses,
                                cell=cell,
                                kinetic_energy_scale=config.kinetic_energy_scale,
                                enabled=config.pressure_diagnostics,
                            )
                        )
                    virials.append(virial)
                    pressure_tensors.append(pressure_tensor_value)
                    pressures.append(pressure_value)
                    pair_counts.append(pair_count)
                    rebuild_counts.append(rebuild_count)
                    constraint_errors.append(constraint_error)
                    _notify_reporters(
                        reporters_tuple,
                        ReporterEvent(
                            ensemble="nvt",
                            event_type="diagnostic",
                            step=current_step,
                            time=state.time,
                            state=state,
                            potential_energy=potential_energy,
                            kinetic_energy=kinetic_energies[-1],
                            total_energy=potential_energy + kinetic_energies[-1],
                            temperature=temperatures[-1],
                            energy_by_term=energy_by_term,
                            virial_tensor=virial,
                            pressure_tensor=pressure_tensor_value,
                            pressure=pressure_value,
                            pair_count=pair_count,
                            rebuild_count=rebuild_count,
                            constraint_max_error=constraint_error,
                            thermostat=thermostat_metadata,
                        ),
                        runtime_sync=runtime_sync,
                    )
                if (
                    (current_step % config.evaluation_interval == 0 or local_step == config.steps)
                    and diagnostic_step
                    and energy_by_term is not None
                ):
                    _eval_step_state(
                        state,
                        potential_energy,
                        kinetic_energies[-1],
                        temperatures[-1],
                        virials[-1],
                        pressure_tensors[-1],
                        pressures[-1],
                        constraint_error,
                        energy_by_term,
                        evaluate_sampled_state=not sampled_state_evaluated,
                        runtime_sync=runtime_sync,
                        sync_reason="diagnostic",
                    )
                # Bound the lazy graph and catch divergence: one sync per block
                # (every block_size steps) instead of per step. The manager
                # update above already materialized positions; this covers the
                # velocity/force/PRNG state carried into the next block.
                runtime_sync.record_sync("failure_check", vel, forces, block_state.key)
            return (
                state,
                block_state.key,
                thermostat_metadata,
                pairs,
                pair_count,
                rebuild_count,
                fe_wall,
            )

        (
            state,
            key,
            thermostat_metadata,
            pairs,
            pair_count,
            rebuild_count,
            force_evaluation_wall_seconds,
        ) = _run_langevin_batched(
            state,
            key,
            thermostat_metadata,
            neighbor_list,
            pairs,
            pair_count,
            rebuild_count,
            force_evaluation_wall_seconds,
        )

    step_range = range(0) if _batched else range(1, config.steps + 1)
    pre_force_velocity_projector = (
        None
        if constraints is None
        else getattr(
            constraints,
            "_apply_pre_force_velocities",
            constraints.apply_velocities,
        )
    )
    center_of_mass_total_mass = (
        None if config.center_of_mass_motion_interval is None else mx.sum(masses)
    )
    for local_step in step_range:
        integration_started = None if route_profiler is None else route_profiler.start()
        current_step = config.initial_step + local_step
        current_time = config.initial_time + local_step * config.dt
        diagnostic_step = _is_diagnostic_step(
            current_step,
            config,
            final=local_step == config.steps,
        )
        sample_step = current_step % config.sample_interval == 0 or local_step == config.steps
        validate_constraint_step = diagnostic_step or sample_step
        if isinstance(thermostat, LangevinThermostat):
            keys = mx.random.split(key, 2)
            key = keys[0]
            noise = mx.random.normal(state.velocities.shape, key=keys[1])
            if metal_langevin_drift is None:
                acceleration = config.force_to_acceleration_scale * state.forces / masses_col
                velocities_half = state.velocities + 0.5 * config.dt * acceleration
                next_positions = state.positions + 0.5 * config.dt * velocities_half
                if cell is not None and config.wrap_positions:
                    next_positions = cell.wrap(next_positions)
                middle_velocities = velocity_decay * velocities_half + thermal_scale * noise
                next_positions = next_positions + 0.5 * config.dt * middle_velocities
            else:
                force_scale_over_mass, metal_box, metal_params, metal_counts = metal_langevin_drift
                next_positions, middle_velocities = _fused_langevin_baoab_drift(
                    state.positions,
                    state.velocities,
                    state.forces,
                    force_scale_over_mass,
                    thermal_scale[:, 0],
                    noise,
                    metal_box,
                    metal_params,
                    metal_counts,
                )
        else:
            acceleration = config.force_to_acceleration_scale * state.forces / masses_col
            current_kinetic = kinetic_energy(
                state.velocities,
                masses,
                kinetic_energy_scale=config.kinetic_energy_scale,
            )
            nh_chain_velocity = nh_chain_velocity + 0.5 * config.dt * (
                (2.0 * current_kinetic - nh_target_kinetic) / nh_thermal_mass
            )
            thermostat_scale = mx.exp(-0.5 * config.dt * nh_chain_velocity)
            scaled_velocities = state.velocities * thermostat_scale
            velocities_half = scaled_velocities + 0.5 * config.dt * acceleration
            next_positions = state.positions + config.dt * velocities_half
        if cell is not None and config.wrap_positions and metal_langevin_drift is None:
            next_positions = cell.wrap(next_positions)
        constraint_error = zero_constraint_error
        velocity_before_final_kick = (
            middle_velocities if isinstance(thermostat, LangevinThermostat) else velocities_half
        )
        if integration_started is not None:
            route_profiler.finish(
                "integration_thermostat",
                integration_started,
                next_positions,
                velocity_before_final_kick,
                key,
            )
        if constraints is not None:
            unconstrained_positions = next_positions
            step_projector = getattr(constraints, "apply_position_step", None)
            if route_profiler is not None:
                next_positions, constraint_error = _profile_constraint_positions(
                    constraints,
                    next_positions,
                    masses,
                    cell,
                    route_profiler,
                    reference_positions=state.positions,
                    validate=validate_constraint_step,
                )
            elif not validate_constraint_step:
                next_positions = _project_constraint_positions_unchecked(
                    constraints,
                    next_positions,
                    masses,
                    cell,
                    reference_positions=state.positions,
                )
            elif step_projector is None:
                next_positions, constraint_error = constraints.apply_positions(
                    next_positions,
                    masses,
                    cell,
                )
            else:
                next_positions, constraint_error = step_projector(
                    state.positions,
                    next_positions,
                    masses,
                    cell,
                )
            position_correction = next_positions - unconstrained_positions
            if cell is not None:
                position_correction = cell.minimum_image(position_correction)
            velocity_before_final_kick = (
                velocity_before_final_kick + position_correction / config.dt
            )
            if route_profiler is None:
                velocity_before_final_kick = pre_force_velocity_projector(
                    next_positions,
                    velocity_before_final_kick,
                    masses,
                    cell,
                )
            else:
                velocity_before_final_kick = _profile_constraint_pre_force_velocities(
                    constraints,
                    next_positions,
                    velocity_before_final_kick,
                    masses,
                    cell,
                    route_profiler,
                )

        neighbor_started = None if route_profiler is None else route_profiler.start()
        eval_positions = _neighbor_evaluation_positions(next_positions, virtual_sites)
        neighbor_list = (
            neighbor_manager.update(eval_positions) if neighbor_manager is not None else None
        )
        if neighbor_started is not None:
            route_profiler.finish(
                "neighbor_update_rebuild",
                neighbor_started,
                eval_positions,
                _neighbor_profile_values(neighbor_list),
            )
        if prepared_force_pipeline is not None and neighbor_list is not bound_neighbor_list:
            binding_started = None if route_profiler is None else route_profiler.start()
            force_binding = prepared_force_pipeline.bind(neighbor_list)
            bound_neighbor_list = neighbor_list
            if binding_started is not None:
                route_profiler.finish(
                    "neighbor_force_binding",
                    binding_started,
                )
        deferred_final = defer_final_diagnostics and local_step == config.steps
        full_diagnostic_step = diagnostic_step and not deferred_final
        if neighbor_list is None:
            pairs = None
        elif (
            full_diagnostic_step
            and not config.pressure_diagnostics
            and force_binding is not None
            and force_binding.interactions is None
            and neighbor_list.supports_tile_diagnostics
            and _force_terms_support_tile_diagnostics(unnamed_terms)
        ):
            pairs = neighbor_list.diagnostic_force_candidates(prefer_tiles=True)
        elif full_diagnostic_step or deferred_final or force_binding is None:
            pairs = neighbor_list.diagnostic_pairs
        else:
            pairs = force_binding.interactions
        pair_count = (
            _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
        )
        rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count

        force_start = perf_counter()
        force_profile_started = (
            None
            if route_profiler is None
            or (force_binding is not None and not full_diagnostic_step and not deferred_final)
            else route_profiler.start()
        )
        diagnostic_virial = None
        if neighbor_manager is None and full_diagnostic_step:
            (
                potential_energy,
                next_forces,
                energy_by_term,
                diagnostic_virial,
            ) = energy_forces_by_term(next_positions)
        elif neighbor_manager is None and deferred_final:
            potential_energy, next_forces = energy_forces(next_positions)
            energy_by_term = None
        elif neighbor_manager is None:
            potential_energy = None
            next_forces = forces_evaluator(next_positions)
            energy_by_term = None
        elif full_diagnostic_step:
            cutoff_strain_pairs = (
                _diagnostic_cutoff_strain_pairs(
                    neighbor_manager,
                    neighbor_list,
                    cell,
                )
                if config.pressure_diagnostics
                else None
            )
            (
                potential_energy,
                next_forces,
                energy_by_term,
                diagnostic_virial,
            ) = _diagnostic_from_terms(
                next_positions,
                terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
                virial_mode=(config.pressure_virial_mode if config.pressure_diagnostics else None),
                masses=masses,
                cutoff_strain_pairs=cutoff_strain_pairs,
            )
        elif deferred_final:
            potential_energy, next_forces = _energy_forces_from_terms(
                next_positions,
                unnamed_terms,
                cell=cell,
                pairs=pairs,
                virtual_sites=virtual_sites,
            )
            energy_by_term = None
        else:
            potential_energy = None
            next_forces = (
                _forces_from_terms(
                    next_positions,
                    unnamed_terms,
                    cell=cell,
                    pairs=pairs,
                    virtual_sites=virtual_sites,
                )
                if force_binding is None
                else force_binding.forces(
                    next_positions,
                    evaluation_positions=eval_positions,
                )
            )
            energy_by_term = None
        if deferred_final:
            energy_by_term = {
                name: energies[-1] for name, energies in potential_energy_by_term.items()
            }
        if force_profile_started is not None:
            force_profile_name = (
                "diagnostics_reporting"
                if full_diagnostic_step or deferred_final
                else "other_force_terms"
            )
            route_profiler.finish(
                force_profile_name,
                force_profile_started,
                potential_energy,
                next_forces,
                energy_by_term,
                diagnostic_virial,
            )
        force_evaluation_wall_seconds += perf_counter() - force_start
        if (
            async_force_submission
            and force_binding is not None
            and neighbor_list is not None
            and neighbor_list.supports_async_force_submission
            and not full_diagnostic_step
            and not deferred_final
        ):
            mx.async_eval(next_forces)
        final_integration_started = None if route_profiler is None else route_profiler.start()
        next_acceleration = config.force_to_acceleration_scale * next_forces / masses_col
        next_velocities = velocity_before_final_kick + 0.5 * config.dt * next_acceleration
        if final_integration_started is not None:
            route_profiler.finish(
                "integration_thermostat",
                final_integration_started,
                next_acceleration,
                next_velocities,
            )
        if constraints is not None:
            if route_profiler is None:
                next_velocities = constraints.apply_velocities(
                    next_positions,
                    next_velocities,
                    masses,
                    cell,
                )
            else:
                next_velocities = _profile_constraint_velocities(
                    constraints,
                    next_positions,
                    next_velocities,
                    masses,
                    cell,
                    route_profiler,
                )
        post_integration_started = None if route_profiler is None else route_profiler.start()
        if (
            config.center_of_mass_motion_interval is not None
            and current_step % config.center_of_mass_motion_interval == 0
        ):
            next_velocities = _remove_center_of_mass_velocity(
                next_velocities,
                masses,
                total_mass=center_of_mass_total_mass,
            )
        if isinstance(thermostat, NoseHooverThermostat):
            next_velocities = next_velocities * mx.exp(-0.5 * config.dt * nh_chain_velocity)
            next_kinetic = kinetic_energy(
                next_velocities,
                masses,
                kinetic_energy_scale=config.kinetic_energy_scale,
            )
            nh_chain_velocity = nh_chain_velocity + 0.5 * config.dt * (
                (2.0 * next_kinetic - nh_target_kinetic) / nh_thermal_mass
            )
            nh_chain_position = nh_chain_position + config.dt * nh_chain_velocity
        state = SimulationState(
            positions=next_positions,
            velocities=next_velocities,
            masses=masses,
            forces=next_forces,
            step=current_step,
            time=current_time,
        )
        if isinstance(thermostat, LangevinThermostat):
            thermostat_metadata = _thermostat_metadata(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
                rng_step_offset=rng_step_offset + local_step,
            )
        else:
            thermostat_metadata = _thermostat_metadata(
                thermostat,
                dof=temperature_dof,
                boltzmann_constant=config.boltzmann_constant,
                chain_position=float(np.asarray(nh_chain_position)),
                chain_velocity=float(np.asarray(nh_chain_velocity)),
            )
        if post_integration_started is not None:
            route_profiler.finish(
                "integration_thermostat",
                post_integration_started,
                state.positions,
                state.velocities,
                state.forces,
                nh_chain_position,
                nh_chain_velocity,
            )

        sampled_state_evaluated = False
        if sample_step:
            _materialize_sampled_state(
                state,
                runtime_sync=runtime_sync,
                reason="final_state" if local_step == config.steps else "explicit_user_output",
            )
            sampled_state_evaluated = True
            sampled_positions.append(state.positions)
            sampled_velocities.append(state.velocities)
            sampled_steps.append(current_step)
            sampled_times.append(state.time)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nvt",
                    event_type="sample",
                    step=current_step,
                    time=state.time,
                    state=state,
                    thermostat=thermostat_metadata,
                ),
                runtime_sync=runtime_sync,
            )
        if diagnostic_step:
            diagnostic_steps.append(current_step)
            diagnostic_times.append(state.time)
            potential_energies.append(potential_energy)
            if energy_by_term is not None:
                for name, energy in energy_by_term.items():
                    potential_energy_by_term[name].append(energy)
            kinetic_energies.append(
                kinetic_energy(
                    state.velocities,
                    masses,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                )
            )
            temperatures.append(
                instantaneous_temperature(
                    state.velocities,
                    masses,
                    dof=temperature_dof,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                    boltzmann_constant=config.boltzmann_constant,
                )
            )
            if deferred_final:
                virial = virials[-1]
                pressure_tensor_value = pressure_tensors[-1]
                pressure_value = pressures[-1]
            else:
                virial, pressure_tensor_value, pressure_value = _pressure_diagnostics_from_virial(
                    diagnostic_virial,
                    state.positions,
                    state.velocities,
                    masses,
                    cell=cell,
                    kinetic_energy_scale=config.kinetic_energy_scale,
                    enabled=config.pressure_diagnostics,
                )
            virials.append(virial)
            pressure_tensors.append(pressure_tensor_value)
            pressures.append(pressure_value)
            pair_counts.append(pair_count)
            rebuild_counts.append(rebuild_count)
            constraint_errors.append(constraint_error)
            _notify_reporters(
                reporters_tuple,
                ReporterEvent(
                    ensemble="nvt",
                    event_type="diagnostic",
                    step=current_step,
                    time=state.time,
                    state=state,
                    potential_energy=potential_energy,
                    kinetic_energy=kinetic_energies[-1],
                    total_energy=potential_energy + kinetic_energies[-1],
                    temperature=temperatures[-1],
                    energy_by_term={} if energy_by_term is None else energy_by_term,
                    virial_tensor=virial,
                    pressure_tensor=pressure_tensor_value,
                    pressure=pressure_value,
                    pair_count=pair_count,
                    rebuild_count=rebuild_count,
                    constraint_max_error=constraint_error,
                    thermostat=thermostat_metadata,
                ),
                runtime_sync=runtime_sync,
            )
        if current_step % config.evaluation_interval == 0 or local_step == config.steps:
            if diagnostic_step and energy_by_term is not None:
                _eval_step_state(
                    state,
                    potential_energy,
                    kinetic_energies[-1],
                    temperatures[-1],
                    virials[-1],
                    pressure_tensors[-1],
                    pressures[-1],
                    constraint_error,
                    energy_by_term,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="diagnostic",
                )
            else:
                _eval_runtime_state(
                    state,
                    potential_energy,
                    constraint_error,
                    evaluate_sampled_state=not sampled_state_evaluated,
                    runtime_sync=runtime_sync,
                    sync_reason="failure_check",
                )

    potential_energy_series = mx.stack(potential_energies)
    kinetic_energy_series = mx.stack(kinetic_energies)
    runtime_sync_report = runtime_sync.to_report()
    route_profile = {} if route_profiler is None else route_profiler.report()
    return NVTResult(
        sampled_positions=mx.stack(sampled_positions),
        sampled_velocities=mx.stack(sampled_velocities),
        sampled_steps=mx.array(sampled_steps, dtype=mx.int32),
        sampled_time=mx.array(sampled_times),
        diagnostic_steps=mx.array(diagnostic_steps, dtype=mx.int32),
        diagnostic_time=mx.array(diagnostic_times),
        potential_energy=potential_energy_series,
        kinetic_energy=kinetic_energy_series,
        total_energy=potential_energy_series + kinetic_energy_series,
        potential_energy_by_term={
            name: mx.stack(energies) for name, energies in potential_energy_by_term.items()
        },
        temperature=mx.stack(temperatures),
        virial_tensor=mx.stack(virials),
        pressure_tensor=mx.stack(pressure_tensors),
        pressure=mx.stack(pressures),
        pair_count=mx.array(pair_counts, dtype=mx.int32),
        rebuild_count=mx.array(rebuild_counts, dtype=mx.int32),
        constraint_max_error=mx.stack(constraint_errors),
        final_state=state,
        target_temperature=thermostat.temperature,
        thermostat_metadata=thermostat_metadata,
        nonbonded_report=_nonbonded_runtime_report(
            _neighbor_evaluation_positions(state.positions, virtual_sites),
            neighbor_manager=neighbor_manager,
            neighbor_list=None if neighbor_manager is None else neighbor_manager.neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
        runtime_sync_report=runtime_sync_report,
        route_profile=route_profile,
    )


@_bounded_metal_md_cache()
def simulate_npt(
    positions,
    velocities,
    *,
    masses=None,
    cell: Cell | None = None,
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...] | None = None,
    neighbor_manager: NeighborListManager | None = None,
    config: SimulationConfig | None = None,
    thermostat: Thermostat | None = None,
    barostat: MonteCarloBarostat | None = None,
    barostat_state: dict[str, Any] | None = None,
    constraints: DistanceConstraints | None = None,
    molecule_ids: object | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
) -> NPTResult:
    """Run molecular Monte Carlo pressure coupling at exact in-loop intervals.

    Args:
        positions: Initial coordinates, shape ``(n_particles, 3)``.
        velocities: Initial velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``; ``None`` uses unit
            masses. Defaults to ``None``.
        cell: Periodic cell (required for NPT). Defaults to ``None``.
        force_terms: One or more force terms; ``None`` uses a default
            `LennardJonesPotential`. Defaults to ``None``.
        neighbor_manager: Optional neighbor-list manager. Defaults to ``None``.
        config: Run configuration; ``None`` uses defaults. Defaults to ``None``.
        thermostat: Thermostat for the NVT stage; ``None`` uses a default
            `LangevinThermostat`. Defaults to ``None``.
        barostat: Monte Carlo barostat; ``None`` uses one matched to the
            thermostat temperature. Defaults to ``None``.
        barostat_state: Optional serialized persistent barostat state from a
            prior committed NPT boundary. Defaults to ``None``.
        constraints: Optional distance constraints applied each step.
            Defaults to ``None``.
        molecule_ids: Optional contiguous per-particle molecule identifiers.
            When omitted, each particle is treated as a separate molecule.
        reporters: Optional runtime reporter(s). Defaults to ``None``.

    Returns:
        An `NPTResult` with the integrated trajectory, sampled cell history,
            and persistent barostat counters.

    Raises:
        ValueError: If ``cell`` is ``None`` (NPT requires a periodic cell).
    """

    if cell is None:
        msg = "NPT simulation requires a periodic cell"
        raise ValueError(msg)
    if config is None:
        config = SimulationConfig()
    if thermostat is None:
        thermostat = LangevinThermostat()
    if barostat is None:
        barostat = MonteCarloBarostat(temperature=thermostat.temperature)
    if force_terms is None:
        force_terms = LennardJonesPotential()
    terms = tuple(force_terms) if isinstance(force_terms, (list, tuple)) else (force_terms,)
    if config.pressure_diagnostics:
        _validate_pressure_virial_support(
            terms,
            virial_mode=config.pressure_virial_mode,
        )
    _validate_barostat_cell_support(cell, barostat)
    initial_positions = as_mx_array(positions)
    molecule_labels = normalize_molecule_ids(
        molecule_ids,
        particle_count=initial_positions.shape[0],
    )
    molecule_count = int(np.max(molecule_labels)) + 1
    reporters_tuple = _normalize_reporters(reporters)
    barostat_rng = np.random.default_rng(barostat.seed)
    end_step = config.initial_step + config.steps
    current_step = config.initial_step
    current_time = config.initial_time
    current_positions = initial_positions
    current_velocities = as_mx_array(velocities)
    current_masses = masses
    current_cell = cell
    current_terms = _cell_bound_force_terms(
        terms,
        current_cell,
        rebuild_plans=False,
    )
    _validate_dynamic_cell_cutoffs(
        current_terms,
        current_cell,
        neighbor_manager=neighbor_manager,
    )
    base_rng_offset = (
        config.initial_step
        if not isinstance(thermostat, LangevinThermostat) or thermostat.rng_step_offset is None
        else thermostat.rng_step_offset
    )
    active_thermostat: Thermostat = thermostat
    next_barostat_step = ((current_step // barostat.interval) + 1) * barostat.interval
    segments: list[NVTResult] = []
    cell_history_chunks: list[mx.array] = []
    seen_reporter_events: set[tuple[str, int]] = set()
    (
        attempts,
        accepted_count,
        proposal_volume_step,
        proposal_volume_steps,
        axis_attempts,
        axis_accepted,
        adaptation_attempts,
        adaptation_accepted,
        proposal_history,
    ) = _restore_barostat_state(
        barostat,
        barostat_rng,
        barostat_state,
        current_volume=float(np.asarray(cell.volume)),
        molecule_count=molecule_count,
        center_of_mass_motion_interval=config.center_of_mass_motion_interval,
    )
    boundary_diagnostics: _NVTBoundaryDiagnostics | None = None

    while not segments or current_step < end_step:
        segment_end = min(end_step, next_barostat_step)
        segment_steps = segment_end - current_step
        should_attempt = (
            segment_steps > 0 and segment_end == next_barostat_step and segment_end <= end_step
        )
        segment_config = replace(
            config,
            steps=segment_steps,
            initial_step=current_step,
            initial_time=current_time,
            wrap_positions=False,
        )
        if isinstance(thermostat, LangevinThermostat):
            active_thermostat = replace(
                thermostat,
                rng_step_offset=base_rng_offset + (current_step - config.initial_step),
            )
        buffered_events: list[ReporterEvent] = []
        segment = _simulate_nvt(
            current_positions,
            current_velocities,
            masses=current_masses,
            cell=current_cell,
            force_terms=current_terms,
            neighbor_manager=neighbor_manager,
            config=segment_config,
            thermostat=active_thermostat,
            constraints=constraints,
            reporters=buffered_events.append,
            initial_diagnostics=boundary_diagnostics,
            defer_final_diagnostics=should_attempt,
        )
        _materialize_npt_segment(segment)
        mx.clear_cache()
        segment_source_cell = current_cell
        proposal: BarostatProposal | None = None
        accepted = False
        old_volume = float(np.asarray(current_cell.volume))
        if should_attempt:
            attempts += 1
            (
                final_state,
                final_cell,
                current_terms,
                accepted,
                proposal,
            ) = _attempt_barostat_move(
                segment.final_state,
                current_terms,
                current_cell,
                current_energy=segment.potential_energy[-1],
                barostat=barostat,
                rng=barostat_rng,
                volume_step=proposal_volume_step,
                axis_volume_steps=proposal_volume_steps,
                constraints=constraints,
                boltzmann_constant=config.boltzmann_constant,
                neighbor_manager=neighbor_manager,
                virtual_sites=config.virtual_sites,
                molecule_ids=molecule_labels,
            )
            segment = _npt_production_with_final_barostat_state(
                segment,
                final_state,
                current_terms,
                final_cell,
                constraints=constraints,
                config=segment_config,
                neighbor_manager=neighbor_manager,
            )
            current_cell = final_cell
            accepted_count += int(accepted)
            axis_name = None if proposal.axis is None else "xyz"[proposal.axis]
            if axis_name is not None:
                axis_attempts[axis_name] += 1
                axis_accepted[axis_name] += int(accepted)
                adaptation_attempts[axis_name] += 1
                adaptation_accepted[axis_name] += int(accepted)
                proposal_volume_steps[axis_name] = _adapt_anisotropic_volume_step(
                    volume_step=proposal_volume_steps[axis_name],
                    attempted=adaptation_attempts[axis_name],
                    accepted=adaptation_accepted[axis_name],
                    current_volume=old_volume,
                )
                if adaptation_attempts[axis_name] >= 10 and (
                    adaptation_accepted[axis_name] < 0.25 * adaptation_attempts[axis_name]
                    or adaptation_accepted[axis_name] > 0.75 * adaptation_attempts[axis_name]
                ):
                    adaptation_attempts[axis_name] = 0
                    adaptation_accepted[axis_name] = 0
            proposal_record = {
                "attempt": attempts,
                "step": final_state.step,
                "axis": axis_name,
                "accepted": int(accepted),
                "kernel": proposal.kernel,
                "scale_factors": list(proposal.scale_factors),
                "log_reverse_over_forward": proposal.log_reverse_over_forward,
                "source_pme_plan_fingerprints": list(proposal.source_pme_plan_fingerprints),
                "candidate_pme_plan_fingerprints": list(proposal.candidate_pme_plan_fingerprints),
                "delta_energy": proposal.delta_energy,
                "log_acceptance": proposal.log_acceptance,
                "log_uniform_draw": proposal.log_uniform_draw,
                "volume_step": proposal.volume_step,
                "old_volume": old_volume,
                "new_volume": float(np.asarray(final_cell.volume)),
            }
            proposal_history.append(proposal_record)
        else:
            final_state = segment.final_state
        segment_cells = mx.broadcast_to(
            segment_source_cell.matrix,
            (segment.sampled_positions.shape[0], 3, 3),
        )
        if should_attempt:
            segment_cells = _replace_last_frame(
                segment_cells,
                current_cell.matrix,
            )
        if segments:
            segment_cells = segment_cells[1:]
        _materialize_npt_segment(segment, segment_cells)
        cell_history_chunks.append(segment_cells)
        segments.append(segment)
        boundary_diagnostics = _nvt_boundary_diagnostics(segment)
        mx.clear_cache()

        _forward_npt_segment_events(
            buffered_events,
            reporters_tuple,
            segment=segment,
            seen=seen_reporter_events,
        )
        if should_attempt and proposal is not None:
            _notify_barostat_reporters(
                reporters_tuple,
                final_state=final_state,
                final_cell=current_cell,
                metadata={
                    **_barostat_metadata(barostat),
                    **proposal_history[-1],
                },
            )
            next_barostat_step += barostat.interval

        current_positions = final_state.positions
        current_velocities = final_state.velocities
        current_masses = final_state.masses
        current_step = final_state.step
        current_time = final_state.time
        if isinstance(active_thermostat, NoseHooverThermostat):
            active_thermostat = replace(
                active_thermostat,
                chain_position=float(segment.thermostat_metadata["chain_position"]),
                chain_velocity=float(segment.thermostat_metadata["chain_velocity"]),
            )
        if current_step >= end_step:
            break

    production = _concatenate_nvt_segments(segments)
    cell_matrix = mx.concatenate(cell_history_chunks, axis=0)
    cell_lengths = mx.sqrt(mx.sum(cell_matrix * cell_matrix, axis=2))
    volumes = (
        cell_matrix[:, 0, 0]
        * (
            cell_matrix[:, 1, 1] * cell_matrix[:, 2, 2]
            - cell_matrix[:, 1, 2] * cell_matrix[:, 2, 1]
        )
        - cell_matrix[:, 0, 1]
        * (
            cell_matrix[:, 1, 0] * cell_matrix[:, 2, 2]
            - cell_matrix[:, 1, 2] * cell_matrix[:, 2, 0]
        )
        + cell_matrix[:, 0, 2]
        * (
            cell_matrix[:, 1, 0] * cell_matrix[:, 2, 1]
            - cell_matrix[:, 1, 1] * cell_matrix[:, 2, 0]
        )
    )
    barostat_metadata = _barostat_metadata(barostat)
    barostat_metadata.update(
        {
            "attempts": attempts,
            "accepted": accepted_count,
            "target_pressure": barostat.pressure,
            "initial_volume": float(np.asarray(cell.volume)),
            "final_volume": float(np.asarray(current_cell.volume)),
            "molecule_count": molecule_count,
            "proposal_volume_step": proposal_volume_step,
            "proposal_volume_steps": proposal_volume_steps,
            "axis_attempts": axis_attempts,
            "axis_accepted": axis_accepted,
            "adaptation_attempts": adaptation_attempts,
            "adaptation_accepted": adaptation_accepted,
            "proposal_history": proposal_history,
            "rng_state": dict(barostat_rng.bit_generator.state),
            "center_of_mass_motion_interval": config.center_of_mass_motion_interval,
            "final_pme_plan_fingerprints": list(_pme_plan_fingerprints(current_terms)),
        }
    )
    return NPTResult(
        production=production,
        final_state=production.final_state,
        final_cell=current_cell,
        final_force_terms=current_terms,
        cell_lengths=cell_lengths,
        cell_matrix=cell_matrix,
        volume=volumes,
        target_pressure=barostat.pressure,
        barostat_attempts=attempts,
        barostat_accepted=accepted_count,
        barostat_metadata=barostat_metadata,
    )


def _materialize_npt_segment(
    segment: NVTResult,
    cell_history: mx.array | None = None,
) -> None:
    arrays = [
        segment.sampled_positions,
        segment.sampled_velocities,
        segment.sampled_steps,
        segment.sampled_time,
        segment.diagnostic_steps,
        segment.diagnostic_time,
        segment.potential_energy,
        segment.kinetic_energy,
        segment.total_energy,
        segment.temperature,
        segment.virial_tensor,
        segment.pressure_tensor,
        segment.pressure,
        segment.pair_count,
        segment.rebuild_count,
        segment.constraint_max_error,
        segment.final_state.positions,
        segment.final_state.velocities,
        segment.final_state.forces,
        *segment.potential_energy_by_term.values(),
    ]
    if cell_history is not None:
        arrays.append(cell_history)
    mx.eval(*arrays)


def _nvt_boundary_diagnostics(
    segment: NVTResult,
) -> _NVTBoundaryDiagnostics:
    return _NVTBoundaryDiagnostics(
        potential_energy=segment.potential_energy[-1],
        forces=segment.final_state.forces,
        energy_by_term={
            name: values[-1] for name, values in segment.potential_energy_by_term.items()
        },
        virial_tensor=segment.virial_tensor[-1],
        pressure_tensor=segment.pressure_tensor[-1],
        pressure=segment.pressure[-1],
        constraint_error=segment.constraint_max_error[-1],
    )


def _concatenate_nvt_segments(segments: list[NVTResult]) -> NVTResult:
    if not segments:
        msg = "NPT integration requires at least one NVT segment"
        raise ValueError(msg)

    def concatenate(field_name: str) -> mx.array:
        chunks = []
        for index, segment in enumerate(segments):
            values = getattr(segment, field_name)
            chunks.append(values if index == 0 else values[1:])
        return mx.concatenate(chunks, axis=0)

    final = segments[-1]
    runtime_sync_report: dict[str, int | float] = {}
    for key in final.runtime_sync_report:
        runtime_sync_report[key] = sum(
            float(segment.runtime_sync_report.get(key, 0.0))
            if "seconds" in key
            else int(segment.runtime_sync_report.get(key, 0))
            for segment in segments
        )
    nonbonded_report = dict(final.nonbonded_report)
    nonbonded_report["force_evaluation_wall_seconds"] = sum(
        float(segment.nonbonded_report.get("force_evaluation_wall_seconds", 0.0))
        for segment in segments
    )
    potential_energy = concatenate("potential_energy")
    kinetic_energy_values = concatenate("kinetic_energy")
    return replace(
        final,
        sampled_positions=concatenate("sampled_positions"),
        sampled_velocities=concatenate("sampled_velocities"),
        sampled_steps=concatenate("sampled_steps"),
        sampled_time=concatenate("sampled_time"),
        diagnostic_steps=concatenate("diagnostic_steps"),
        diagnostic_time=concatenate("diagnostic_time"),
        potential_energy=potential_energy,
        kinetic_energy=kinetic_energy_values,
        total_energy=potential_energy + kinetic_energy_values,
        potential_energy_by_term={
            name: mx.concatenate(
                [
                    segment.potential_energy_by_term[name]
                    if index == 0
                    else segment.potential_energy_by_term[name][1:]
                    for index, segment in enumerate(segments)
                ],
                axis=0,
            )
            for name in final.potential_energy_by_term
        },
        temperature=concatenate("temperature"),
        virial_tensor=concatenate("virial_tensor"),
        pressure_tensor=concatenate("pressure_tensor"),
        pressure=concatenate("pressure"),
        pair_count=concatenate("pair_count"),
        rebuild_count=concatenate("rebuild_count"),
        constraint_max_error=concatenate("constraint_max_error"),
        nonbonded_report=nonbonded_report,
        runtime_sync_report=runtime_sync_report,
    )


def _forward_npt_segment_events(
    events: list[ReporterEvent],
    reporters: tuple[RuntimeReporter, ...],
    *,
    segment: NVTResult,
    seen: set[tuple[str, int]],
) -> None:
    for event in events:
        key = (event.event_type, event.step)
        if key in seen:
            continue
        seen.add(key)
        forwarded = replace(event, ensemble="npt")
        if event.step == segment.final_state.step:
            if event.event_type == "sample":
                forwarded = replace(forwarded, state=segment.final_state)
            elif event.event_type == "diagnostic":
                forwarded = replace(
                    forwarded,
                    state=segment.final_state,
                    potential_energy=segment.potential_energy[-1],
                    kinetic_energy=segment.kinetic_energy[-1],
                    total_energy=segment.total_energy[-1],
                    temperature=segment.temperature[-1],
                    energy_by_term={
                        name: values[-1]
                        for name, values in segment.potential_energy_by_term.items()
                    },
                    virial_tensor=segment.virial_tensor[-1],
                    pressure_tensor=segment.pressure_tensor[-1],
                    pressure=segment.pressure[-1],
                    pair_count=segment.pair_count[-1],
                    rebuild_count=segment.rebuild_count[-1],
                    constraint_max_error=segment.constraint_max_error[-1],
                    thermostat=segment.thermostat_metadata,
                )
        for reporter in reporters:
            reporter(forwarded)


def _npt_production_with_final_barostat_state(
    production: NVTResult,
    final_state: SimulationState,
    force_terms: tuple[ForceTerm, ...],
    final_cell: Cell,
    *,
    constraints: DistanceConstraints | None,
    config: SimulationConfig,
    neighbor_manager: NeighborListManager | None,
) -> NVTResult:
    virtual_sites = config.virtual_sites
    named_terms = _named_force_terms(force_terms)
    eval_positions = _neighbor_evaluation_positions(final_state.positions, virtual_sites)
    if neighbor_manager is not None:
        _validate_neighbor_manager_cell(neighbor_manager, final_cell)
        neighbor_list = neighbor_manager.neighbor_list
        if neighbor_list is None:
            msg = "NPT committed state requires an initialized neighbor list"
            raise RuntimeError(msg)
    else:
        neighbor_list = None
    pairs = None if neighbor_list is None else neighbor_list.diagnostic_pairs
    if config.pressure_diagnostics:
        cutoff_strain_pairs = _diagnostic_cutoff_strain_pairs(
            neighbor_manager,
            neighbor_list,
            final_cell,
        )
        (
            potential_energy,
            _,
            energy_by_term,
            diagnostic_virial,
        ) = _diagnostic_from_terms(
            final_state.positions,
            named_terms,
            cell=final_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
            virial_mode=config.pressure_virial_mode,
            masses=final_state.masses,
            cutoff_strain_pairs=cutoff_strain_pairs,
        )
    else:
        potential_energy, _, energy_by_term = _energy_forces_by_term(
            final_state.positions,
            named_terms,
            cell=final_cell,
            pairs=pairs,
            virtual_sites=virtual_sites,
        )
        diagnostic_virial = None
    kinetic_energy_value = kinetic_energy(
        final_state.velocities,
        final_state.masses,
        kinetic_energy_scale=config.kinetic_energy_scale,
    )
    temperature_dof = _temperature_degrees_of_freedom(final_state.positions, constraints)
    temperature_value = instantaneous_temperature(
        final_state.velocities,
        final_state.masses,
        dof=temperature_dof,
        kinetic_energy_scale=config.kinetic_energy_scale,
        boltzmann_constant=config.boltzmann_constant,
    )
    virial, pressure_tensor_value, pressure_value = _pressure_diagnostics_from_virial(
        diagnostic_virial,
        final_state.positions,
        final_state.velocities,
        final_state.masses,
        cell=final_cell,
        kinetic_energy_scale=config.kinetic_energy_scale,
        enabled=config.pressure_diagnostics,
    )
    constraint_error = (
        _zero_constraint_error(final_state.positions)
        if constraints is None
        else constraints.max_error(final_state.positions, final_cell)
    )
    updated_terms = {
        name: _replace_last_frame(values, energy_by_term[name])
        for name, values in production.potential_energy_by_term.items()
        if name in energy_by_term
    }
    pair_count = (
        _dense_pair_count(eval_positions) if neighbor_list is None else neighbor_list.pair_count
    )
    rebuild_count = 0 if neighbor_manager is None else neighbor_manager.rebuild_count
    force_evaluation_wall_seconds = float(
        production.nonbonded_report.get("force_evaluation_wall_seconds", 0.0)
    )
    runtime_sync_report = production.runtime_sync_report
    return replace(
        production,
        sampled_positions=_replace_last_frame(production.sampled_positions, final_state.positions),
        sampled_velocities=_replace_last_frame(
            production.sampled_velocities,
            final_state.velocities,
        ),
        potential_energy=_replace_last_frame(production.potential_energy, potential_energy),
        kinetic_energy=_replace_last_frame(production.kinetic_energy, kinetic_energy_value),
        total_energy=_replace_last_frame(
            production.total_energy,
            potential_energy + kinetic_energy_value,
        ),
        potential_energy_by_term=updated_terms,
        temperature=_replace_last_frame(production.temperature, temperature_value),
        virial_tensor=_replace_last_frame(production.virial_tensor, virial),
        pressure_tensor=_replace_last_frame(production.pressure_tensor, pressure_tensor_value),
        pressure=_replace_last_frame(production.pressure, pressure_value),
        pair_count=_replace_last_frame(
            production.pair_count,
            mx.array(pair_count, dtype=mx.int32),
        ),
        rebuild_count=_replace_last_frame(
            production.rebuild_count,
            mx.array(rebuild_count, dtype=mx.int32),
        ),
        constraint_max_error=_replace_last_frame(
            production.constraint_max_error,
            constraint_error,
        ),
        final_state=final_state,
        nonbonded_report=_nonbonded_runtime_report(
            eval_positions,
            neighbor_manager=neighbor_manager,
            neighbor_list=neighbor_list,
            force_evaluation_wall_seconds=force_evaluation_wall_seconds,
            runtime_sync_report=runtime_sync_report,
        ),
    )


def _replace_last_frame(frames: mx.array, frame: mx.array) -> mx.array:
    frame = as_mx_array(frame)
    if frames.shape[0] <= 1:
        return frame[None, ...]
    return mx.concatenate([frames[:-1], frame[None, ...]], axis=0)


def _attempt_barostat_move(
    state: SimulationState,
    force_terms: tuple[ForceTerm, ...],
    cell: Cell,
    *,
    current_energy: mx.array | None = None,
    barostat: MonteCarloBarostat,
    rng: np.random.Generator,
    volume_step: float,
    axis_volume_steps: dict[str, float] | None = None,
    constraints: DistanceConstraints | None,
    boltzmann_constant: float,
    neighbor_manager: NeighborListManager | None = None,
    virtual_sites: VirtualSiteManager | None = None,
    molecule_ids: object | None = None,
) -> tuple[
    SimulationState,
    Cell,
    tuple[ForceTerm, ...],
    bool,
    BarostatProposal,
]:
    proposal = _barostat_proposal(
        barostat,
        rng,
        volume=float(np.asarray(cell.volume)),
        volume_step=volume_step,
        axis_volume_steps=axis_volume_steps,
    )
    proposed_cell = _scaled_cell(cell, np.asarray(proposal.scale_factors))
    proposed_positions = molecularly_strained_positions(
        state.positions,
        source_cell=cell,
        target_cell=proposed_cell,
        masses=state.masses,
        molecule_ids=molecule_ids,
    )
    proposed_velocities = state.velocities
    constraint_error = _zero_constraint_error(proposed_positions)
    if constraints is not None:
        if _barostat_constraint_projection_required(
            constraints,
            molecule_ids=molecule_ids,
            particle_count=state.positions.shape[0],
        ):
            proposed_positions, constraint_error = constraints.apply_positions(
                proposed_positions,
                state.masses,
                proposed_cell,
            )
        else:
            constraint_error = constraints.max_error(
                proposed_positions,
                proposed_cell,
            )

    proposed_eval_positions = _neighbor_evaluation_positions(proposed_positions, virtual_sites)
    current_force_terms = _cell_bound_force_terms(
        force_terms,
        cell,
        rebuild_plans=False,
    )
    _validate_dynamic_cell_cutoffs(
        current_force_terms,
        proposed_cell,
        neighbor_manager=neighbor_manager,
    )
    proposed_force_terms = _cell_bound_force_terms(
        current_force_terms,
        proposed_cell,
        rebuild_plans=True,
    )
    proposal = replace(
        proposal,
        source_pme_plan_fingerprints=_pme_plan_fingerprints(current_force_terms),
        candidate_pme_plan_fingerprints=_pme_plan_fingerprints(proposed_force_terms),
    )
    candidate_neighbor_manager = None
    if neighbor_manager is not None:
        _validate_neighbor_manager_cell(neighbor_manager, cell)
        old_neighbor_list = neighbor_manager.neighbor_list
        if old_neighbor_list is None:
            msg = "barostat requires an initialized current neighbor list"
            raise RuntimeError(msg)
        candidate_neighbor_manager = neighbor_manager.build_cell_candidate(
            proposed_eval_positions,
            proposed_cell,
        )
        proposed_neighbor_list = candidate_neighbor_manager.neighbor_list
    else:
        old_neighbor_list = None
        proposed_neighbor_list = None
    old_pairs = None if old_neighbor_list is None else old_neighbor_list.diagnostic_pairs
    proposed_pairs = (
        None if proposed_neighbor_list is None else proposed_neighbor_list.diagnostic_pairs
    )
    old_energy = (
        _energy_forces_from_terms(
            state.positions,
            current_force_terms,
            cell=cell,
            pairs=old_pairs,
            virtual_sites=virtual_sites,
        )[0]
        if current_energy is None
        else as_mx_array(current_energy)
    )
    new_energy, new_forces = _energy_forces_from_terms(
        proposed_positions,
        proposed_force_terms,
        cell=proposed_cell,
        pairs=proposed_pairs,
        virtual_sites=virtual_sites,
    )
    old_volume = float(np.asarray(cell.volume))
    new_volume = float(np.asarray(proposed_cell.volume))
    beta = 1.0 / (boltzmann_constant * barostat.temperature)
    molecule_count = int(
        np.unique(
            normalize_molecule_ids(
                molecule_ids,
                particle_count=state.positions.shape[0],
            )
        ).size
    )
    delta_energy = float(np.asarray(new_energy - old_energy))
    log_acceptance = _barostat_log_acceptance_probability(
        delta_energy=delta_energy,
        pressure=barostat.pressure,
        old_volume=old_volume,
        new_volume=new_volume,
        molecule_count=molecule_count,
        beta=beta,
        log_reverse_over_forward=proposal.log_reverse_over_forward,
    )
    log_uniform_draw = None if log_acceptance >= 0.0 else float(np.log(rng.random()))
    accepted = (
        log_acceptance >= 0.0 or log_uniform_draw is not None and log_uniform_draw < log_acceptance
    )
    proposal = replace(
        proposal,
        delta_energy=delta_energy,
        log_acceptance=log_acceptance,
        log_uniform_draw=log_uniform_draw,
    )
    if not accepted:
        return state, cell, current_force_terms, False, proposal
    if neighbor_manager is not None and candidate_neighbor_manager is not None:
        neighbor_manager.commit_cell_candidate(candidate_neighbor_manager)
    mx.eval(proposed_positions, proposed_velocities, new_forces, constraint_error)
    return (
        SimulationState(
            positions=proposed_positions,
            velocities=proposed_velocities,
            masses=state.masses,
            forces=new_forces,
            step=state.step,
            time=state.time,
        ),
        proposed_cell,
        proposed_force_terms,
        True,
        proposal,
    )


def _barostat_log_acceptance_probability(
    *,
    delta_energy: float,
    pressure: float,
    old_volume: float,
    new_volume: float,
    molecule_count: int,
    beta: float,
    log_reverse_over_forward: float,
) -> float:
    acceptance_work = (
        delta_energy
        + pressure * (new_volume - old_volume)
        - molecule_count / beta * float(np.log(new_volume / old_volume))
    )
    return -beta * acceptance_work + log_reverse_over_forward


def _barostat_constraint_projection_required(
    constraints: DistanceConstraints,
    *,
    molecule_ids: object | None,
    particle_count: int,
) -> bool:
    pairs = np.asarray(constraints.pairs, dtype=np.int32).reshape((-1, 2))
    if pairs.shape[0] == 0:
        return False
    labels = normalize_molecule_ids(
        molecule_ids,
        particle_count=particle_count,
    )
    return bool(np.any(labels[pairs[:, 0]] != labels[pairs[:, 1]]))


def _cell_bound_force_terms(
    force_terms: tuple[ForceTerm, ...],
    cell: Cell,
    *,
    rebuild_plans: bool,
) -> tuple[ForceTerm, ...]:
    bound_terms = []
    for term in force_terms:
        if getattr(term, "electrostatics", None) != "pme":
            bound_terms.append(term)
            continue
        binder = getattr(term, "bind_pme_plan", None)
        if not callable(binder):
            msg = "PME force term does not expose bind_pme_plan"
            raise TypeError(msg)
        plan = getattr(term, "pme_plan", None)
        if plan is None:
            bound_terms.append(binder(cell))
            continue
        if rebuild_plans:
            rebuild = getattr(plan, "rebuild", None)
            if not callable(rebuild):
                msg = "PME execution plan does not expose rebuild"
                raise TypeError(msg)
            bound_terms.append(binder(rebuild(cell=cell)))
            continue
        plan.validate(
            cell,
            config=getattr(term, "pme_config", None),
            coulomb_constant=float(getattr(term, "coulomb_constant", 1.0)),
        )
        bound_terms.append(term)
    return tuple(bound_terms)


def _pme_plan_fingerprints(
    force_terms: tuple[ForceTerm, ...],
) -> tuple[str, ...]:
    return tuple(
        str(plan.fingerprint)
        for term in force_terms
        if (plan := getattr(term, "pme_plan", None)) is not None
    )


def _validate_neighbor_manager_cell(
    neighbor_manager: NeighborListManager,
    cell: Cell,
) -> None:
    if not np.array_equal(
        np.asarray(neighbor_manager.cell.matrix),
        np.asarray(cell.matrix),
    ):
        msg = "neighbor manager cell does not match authoritative NPT cell"
        raise ValueError(msg)


def _validate_dynamic_cell_cutoffs(
    force_terms: tuple[ForceTerm, ...],
    cell: Cell,
    *,
    neighbor_manager: NeighborListManager | None,
) -> None:
    if neighbor_manager is None:
        return
    half_minimum_length = 0.5 * float(np.min(np.asarray(cell.lengths, dtype=np.float64)))
    for term in force_terms:
        if getattr(term, "electrostatics", None) != "pme":
            continue
        config = getattr(term, "pme_config", None)
        cutoff = None if config is None else getattr(config, "real_cutoff", None)
        if cutoff is None:
            continue
        if float(cutoff) > half_minimum_length + 1.0e-7:
            msg = "dynamic-cell PME real_cutoff must not exceed half the minimum box length"
            raise ValueError(msg)


def _barostat_proposal(
    barostat: MonteCarloBarostat,
    rng: np.random.Generator,
    *,
    volume: float,
    volume_step: float,
    axis_volume_steps: dict[str, float] | None = None,
) -> BarostatProposal:
    max_scale = barostat.max_log_volume_scale
    if volume_step <= 0.0 or volume <= volume_step:
        msg = "barostat volume proposal step must be positive and smaller than volume"
        raise ValueError(msg)
    if barostat.mode == "isotropic":
        relative_volume_change = rng.uniform(-volume_step, volume_step) / volume
        length_scale = (1.0 + relative_volume_change) ** (1.0 / 3.0)
        return BarostatProposal(
            scale_factors=(length_scale, length_scale, length_scale),
            axis=None,
            log_reverse_over_forward=0.0,
            kernel="symmetric_volume_delta",
            volume_step=volume_step,
        )
    if barostat.mode == "anisotropic":
        enabled_axes = np.flatnonzero(np.asarray(barostat.axes, dtype=bool))
        axis = int(rng.choice(enabled_axes))
        selected_volume_step = (
            volume_step if axis_volume_steps is None else float(axis_volume_steps["xyz"[axis]])
        )
        relative_volume_change = rng.uniform(-selected_volume_step, selected_volume_step) / volume
        scale_factors = np.ones((3,), dtype=np.float64)
        scale_factors[axis] = 1.0 + relative_volume_change
        return BarostatProposal(
            scale_factors=tuple(float(value) for value in scale_factors),
            axis=axis,
            log_reverse_over_forward=0.0,
            kernel="symmetric_axis_volume_delta",
            volume_step=selected_volume_step,
        )

    plane_axes = _barostat_plane_axes(barostat.membrane_plane)
    normal_axis = _barostat_axis_index(barostat.normal_axis)
    log_area_scale = rng.uniform(-max_scale, max_scale)
    log_normal_scale = rng.uniform(-max_scale, max_scale)
    log_axis_scale = np.zeros(3, dtype=np.float64)
    for axis in plane_axes:
        log_axis_scale[axis] = log_area_scale / 2.0
    log_axis_scale[normal_axis] = log_normal_scale
    scale_factors = np.exp(log_axis_scale)
    return BarostatProposal(
        scale_factors=tuple(float(value) for value in scale_factors),
        axis=None,
        log_reverse_over_forward=float(np.sum(log_axis_scale)),
        kernel="symmetric_log_area_and_length",
        volume_step=volume_step,
    )


def _scaled_cell(cell: Cell, scale_factors: np.ndarray) -> Cell:
    matrix = np.asarray(cell.matrix, dtype=np.float64).copy()
    matrix *= np.asarray(scale_factors, dtype=np.float64)[:, None]
    return Cell(matrix)


def _adapt_anisotropic_volume_step(
    *,
    volume_step: float,
    attempted: int,
    accepted: int,
    current_volume: float,
) -> float:
    if attempted < 10:
        return volume_step
    if accepted < 0.25 * attempted:
        return volume_step / 1.1
    if accepted > 0.75 * attempted:
        return min(volume_step * 1.1, current_volume * 0.3)
    return volume_step


def _restore_barostat_state(
    barostat: MonteCarloBarostat,
    rng: np.random.Generator,
    state: dict[str, Any] | None,
    *,
    current_volume: float,
    molecule_count: int,
    center_of_mass_motion_interval: int | None,
) -> tuple[
    int,
    int,
    float,
    dict[str, float],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    list[dict[str, Any]],
]:
    if state is None:
        initial_volume_step = current_volume * float(np.expm1(barostat.max_log_volume_scale))
        return (
            0,
            0,
            initial_volume_step,
            {axis: initial_volume_step for axis in "xyz"},
            {axis: 0 for axis in "xyz"},
            {axis: 0 for axis in "xyz"},
            {axis: 0 for axis in "xyz"},
            {axis: 0 for axis in "xyz"},
            [],
        )
    restored = dict(state)
    expected = _barostat_metadata(barostat)
    for name in ("family", "mode", "interval"):
        if restored.get(name) != expected[name]:
            msg = f"barostat checkpoint {name} does not match requested configuration"
            raise ValueError(msg)
    for name in ("pressure", "temperature", "max_log_volume_scale"):
        if not np.isclose(
            float(restored.get(name, np.nan)),
            float(expected[name]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            msg = f"barostat checkpoint {name} does not match requested configuration"
            raise ValueError(msg)
    if barostat.mode == "anisotropic" and restored.get("axes") != expected["axes"]:
        msg = "barostat checkpoint axes do not match requested configuration"
        raise ValueError(msg)
    if barostat.mode == "membrane":
        for name in ("membrane_plane", "normal_axis"):
            if restored.get(name) != expected[name]:
                msg = f"barostat checkpoint {name} does not match requested configuration"
                raise ValueError(msg)
    if int(restored.get("molecule_count", -1)) != molecule_count:
        msg = "barostat checkpoint molecule count does not match runtime topology"
        raise ValueError(msg)
    if restored.get("center_of_mass_motion_interval") != center_of_mass_motion_interval:
        msg = "barostat checkpoint center-of-mass cadence does not match runtime"
        raise ValueError(msg)

    attempts = int(restored.get("attempts", -1))
    accepted = int(restored.get("accepted", -1))
    if attempts < 0 or accepted < 0 or accepted > attempts:
        msg = "barostat checkpoint counters are invalid"
        raise ValueError(msg)
    proposal_volume_step = float(restored.get("proposal_volume_step", np.nan))
    if not np.isfinite(proposal_volume_step) or proposal_volume_step <= 0.0:
        msg = "barostat checkpoint proposal_volume_step must be finite and positive"
        raise ValueError(msg)
    proposal_volume_steps = _restore_barostat_axis_values(
        restored.get(
            "proposal_volume_steps",
            {axis: proposal_volume_step for axis in "xyz"},
        ),
        name="proposal_volume_steps",
    )
    axis_attempts = _restore_barostat_axis_counts(
        restored.get("axis_attempts"),
        name="axis_attempts",
    )
    axis_accepted = _restore_barostat_axis_counts(
        restored.get("axis_accepted"),
        name="axis_accepted",
    )
    if any(axis_accepted[axis] > axis_attempts[axis] for axis in "xyz"):
        msg = "barostat checkpoint per-axis counters are invalid"
        raise ValueError(msg)
    adaptation_attempts = _restore_barostat_axis_counts(
        restored.get("adaptation_attempts", axis_attempts),
        name="adaptation_attempts",
    )
    adaptation_accepted = _restore_barostat_axis_counts(
        restored.get("adaptation_accepted", axis_accepted),
        name="adaptation_accepted",
    )
    if any(adaptation_accepted[axis] > adaptation_attempts[axis] for axis in "xyz"):
        msg = "barostat checkpoint adaptation counters are invalid"
        raise ValueError(msg)
    history = restored.get("proposal_history")
    if not isinstance(history, list) or len(history) != attempts:
        msg = "barostat checkpoint proposal history does not match attempts"
        raise ValueError(msg)
    rng_state = restored.get("rng_state")
    if not isinstance(rng_state, dict):
        msg = "barostat checkpoint RNG state is missing"
        raise ValueError(msg)
    try:
        rng.bit_generator.state = rng_state
    except (TypeError, ValueError) as error:
        msg = "barostat checkpoint RNG state is invalid"
        raise ValueError(msg) from error
    return (
        attempts,
        accepted,
        proposal_volume_step,
        proposal_volume_steps,
        axis_attempts,
        axis_accepted,
        adaptation_attempts,
        adaptation_accepted,
        [dict(record) for record in history],
    )


def _restore_barostat_axis_counts(value: Any, *, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set("xyz"):
        msg = f"barostat checkpoint {name} must contain x, y, and z"
        raise ValueError(msg)
    counts = {axis: int(value[axis]) for axis in "xyz"}
    if any(count < 0 for count in counts.values()):
        msg = f"barostat checkpoint {name} must be non-negative"
        raise ValueError(msg)
    return counts


def _restore_barostat_axis_values(
    value: Any,
    *,
    name: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set("xyz"):
        msg = f"barostat checkpoint {name} must contain x, y, and z"
        raise ValueError(msg)
    values = {axis: float(value[axis]) for axis in "xyz"}
    if any(not np.isfinite(item) or item <= 0.0 for item in values.values()):
        msg = f"barostat checkpoint {name} must be finite and positive"
        raise ValueError(msg)
    return values


def _barostat_metadata(barostat: MonteCarloBarostat) -> dict[str, Any]:
    metadata = {
        "family": "monte_carlo",
        "mode": barostat.mode,
        "pressure": barostat.pressure,
        "temperature": barostat.temperature,
        "interval": barostat.interval,
        "max_log_volume_scale": barostat.max_log_volume_scale,
    }
    if barostat.mode == "anisotropic":
        metadata["axes"] = {
            axis: enabled for axis, enabled in zip(("x", "y", "z"), barostat.axes, strict=True)
        }
    elif barostat.mode == "membrane":
        metadata["membrane_plane"] = barostat.membrane_plane
        metadata["normal_axis"] = barostat.normal_axis
        metadata["plane_policy"] = "coupled_area"
        metadata["normal_policy"] = "independent_length"
    return metadata


def _notify_barostat_reporters(
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None,
    *,
    final_state: SimulationState,
    final_cell: Cell,
    metadata: dict[str, Any],
) -> None:
    if reporters is None:
        return
    event_metadata = dict(metadata)
    event_metadata["final_cell"] = np.asarray(final_cell.matrix, dtype=np.float32).tolist()
    event = ReporterEvent(
        ensemble="npt",
        event_type="barostat",
        step=final_state.step,
        time=final_state.time,
        state=final_state,
        barostat=event_metadata,
    )
    for reporter in _normalize_reporters(reporters):
        reporter(event)
