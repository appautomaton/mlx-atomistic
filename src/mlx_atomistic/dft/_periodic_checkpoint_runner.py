"""Checkpoint-aware periodic SCF runtime orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from mlx_atomistic.dft._periodic_artifact_contracts import (
    PeriodicSCFExecutionIdentity,
    _calculation_fingerprint,
    _coerce_identity,
    _validate_execution_calculation_binding,
    _validate_initialization_binding,
    periodic_scf_calculation_contract,
)
from mlx_atomistic.dft._periodic_checkpoint_codec import (
    PeriodicSCFCheckpoint,
    _publish_checkpoint_state,
    load_periodic_scf_checkpoint,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._periodic_scf_engine import _run_periodic_scf_controlled
from mlx_atomistic.dft._periodic_state import _PeriodicSCFContinuationState
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


def run_periodic_scf_checkpointed(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    execution_context: PeriodicSCFExecutionIdentity | Mapping[str, object],
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    initial_density: mx.array | None = None,
    initial_coefficients: Sequence[mx.array] | None = None,
    observer: RuntimeObserver | None = None,
    checkpoint_to: str | Path | None = None,
    checkpoint_iteration: int | None = None,
    resume_from: str | Path | None = None,
    provenance: Mapping[str, object] | None = None,
    fault_hook: Callable[[str], None] | None = None,
) -> PeriodicSCFResult:
    """Run periodic SCF with opt-in atomic checkpointing or explicit resume.

    Args:
        system: Periodic GTH system.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        execution_context: Complete current execution context or validated identity.
        n_bands: Computed band count. Fixed occupations default to half the
            electron count; smearing requires additional empty bands.
        config: Exact SCF controls. Defaults to ``PeriodicSCFConfig``.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        initial_density: Optional fresh-run starting density.
        initial_coefficients: Optional fresh-run orbital stacks.
        observer: Optional runtime observer.
        checkpoint_to: Previously absent output generation, or ``None``.
        checkpoint_iteration: Accepted iteration after which to publish and stop.
        resume_from: Explicit checkpoint generation to validate and load.
        provenance: Optional non-identity Git or caller provenance.
        fault_hook: Optional deterministic publication-stage test hook.

    Returns:
        Periodic SCF result. Resumed results retain numerical lineage and mark
        timing as ineligible for fresh evidence.

    Raises:
        ValueError: If checkpoint controls are incomplete or conflict with resume.
        ArtifactIntegrityError: If explicit resume validation fails.
    """

    identity = _coerce_identity(execution_context)
    scf_config = PeriodicSCFConfig() if config is None else config
    calculation = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=scf_config,
        xc_functional=xc_functional,
    )
    _validate_execution_calculation_binding(identity, calculation)
    if (checkpoint_to is None) != (checkpoint_iteration is None):
        msg = "checkpoint_to and checkpoint_iteration must be supplied together"
        raise ValueError(msg)
    if checkpoint_iteration is not None and (
        type(checkpoint_iteration) is not int
        or checkpoint_iteration <= 0
        or checkpoint_iteration >= scf_config.max_iterations
    ):
        msg = "checkpoint_iteration must precede the configured SCF iteration limit"
        raise ValueError(msg)
    if resume_from is not None and (
        initial_density is not None or initial_coefficients is not None
    ):
        msg = "explicit periodic resume cannot also use fresh initial guesses"
        raise ValueError(msg)
    if resume_from is None:
        _validate_initialization_binding(
            identity,
            initial_density=initial_density,
            initial_coefficients=initial_coefficients,
        )

    loaded: PeriodicSCFCheckpoint | None = None
    if resume_from is not None:
        loaded = load_periodic_scf_checkpoint(
            resume_from,
            system=system,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            execution_context=identity,
            n_bands=n_bands,
            config=scf_config,
            xc_functional=xc_functional,
        )
        if (
            checkpoint_iteration is not None
            and checkpoint_iteration <= loaded._state.completed_iteration
        ):
            msg = "new checkpoint iteration must follow the resumed iteration"
            raise ValueError(msg)

    published: dict[str, object] | None = None

    def checkpoint_callback(state: _PeriodicSCFContinuationState) -> bool:
        nonlocal published
        if checkpoint_iteration is None or state.completed_iteration != checkpoint_iteration:
            return False
        published = _publish_checkpoint_state(
            checkpoint_to,
            state=state,
            identity=identity,
            calculation_contract=calculation,
            provenance=provenance,
            fault_hook=fault_hook,
        )
        return True

    result = _run_periodic_scf_controlled(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=scf_config,
        xc_functional=xc_functional,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        observer=observer,
        resume_state=None if loaded is None else loaded._state,
        checkpoint_callback=None if checkpoint_to is None else checkpoint_callback,
        checkpoint_iteration=checkpoint_iteration,
    )
    if checkpoint_to is not None and published is None:
        msg = "SCF completed before the requested checkpoint boundary was published"
        raise ValueError(msg)
    return replace(
        result,
        _artifact_execution_contract_fingerprint=(identity.execution_contract_fingerprint),
        _artifact_calculation_fingerprint=_calculation_fingerprint(calculation),
    )

