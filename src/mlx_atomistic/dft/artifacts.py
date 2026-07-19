"""Atomic compact checkpoint publication and explicit periodic-SCF resume."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    AtomicGeneration,
    canonical_json_bytes,
    confined_path,
    generation_root,
    inspect_generation,
    read_generation_json,
    sha256_bytes,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.mixing import _MixerCheckpointState
from mlx_atomistic.dft.periodic_scf import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
    _eigensolve_provenance,
    _PeriodicSCFContinuationState,
    _run_periodic_scf_controlled,
)
from mlx_atomistic.dft.pseudopotentials import PseudopotentialFormat
from mlx_atomistic.dft.runtime_state import _npy_bytes
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

PERIODIC_SCF_CHECKPOINT_KIND = "periodic-scf-checkpoint"
PERIODIC_SCF_CHECKPOINT_SCHEMA = "mlx-atomistic.periodic-scf-checkpoint.v1"
PERIODIC_SCF_CHECKPOINT_PAYLOAD = "checkpoint.json"
PERIODIC_SCF_COMMAND_KIND = "periodic-scf"
_CALCULATION_CONTRACT_SCHEMA = "mlx-atomistic.periodic-scf-calculation.v1"
_SHA256_FIELDS = (
    "workload_fingerprint",
    "protocol_fingerprint",
    "runtime_fingerprint",
    "execution_contract_fingerprint",
)


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_complete_execution_contract(contract: Mapping[str, object]) -> None:
    mapping_fields = (
        "solver",
        "initialization",
        "settings_override",
        "lock",
        "environment",
        "host_protocol",
    )
    if contract.get("schema_version") != (
        "mlx-atomistic.dft-runtime-execution-contract.v1"
    ) or not isinstance(contract.get("command_kind"), str):
        msg = "periodic checkpoint execution contract schema or command is incomplete"
        raise ValueError(msg)
    if any(not isinstance(contract.get(field_name), Mapping) for field_name in mapping_fields):
        msg = "periodic checkpoint execution contract is missing a required object"
        raise ValueError(msg)
    if not isinstance(contract.get("synchronization"), str) or not contract.get(
        "synchronization"
    ):
        msg = "periodic checkpoint execution synchronization identity is missing"
        raise ValueError(msg)

    solver = contract["solver"]
    if not isinstance(solver.get("scf"), Mapping) or not isinstance(
        solver.get("davidson"),
        Mapping,
    ):
        msg = "periodic checkpoint execution solver identity is incomplete"
        raise ValueError(msg)
    lock = contract["lock"]
    if (
        lock.get("path") != "uv.lock"
        or type(lock.get("byte_size")) is not int
        or lock["byte_size"] < 0
        or not _is_sha256(lock.get("sha256"))
    ):
        msg = "periodic checkpoint lock identity is incomplete"
        raise ValueError(msg)
    environment = contract["environment"]
    required_environment = {
        "python_version",
        "python_implementation",
        "mlx_version",
        "default_device",
        "metal_available",
        "selected_device",
        "precision",
        "full_grid_precision",
        "projected_eigensolve_device",
        "projected_eigensolve_backend",
        "projected_eigensolve_precision",
        "projected_eigensolve_output_precision",
    }
    if not required_environment.issubset(environment):
        msg = "periodic checkpoint runtime environment identity is incomplete"
        raise ValueError(msg)
    host_protocol = contract["host_protocol"]
    required_host = {
        "model",
        "model_identifier",
        "chip",
        "machine",
        "macos",
        "power_source",
        "low_power_mode",
    }
    if not required_host.issubset(host_protocol) or not isinstance(
        host_protocol.get("macos"),
        Mapping,
    ):
        msg = "periodic checkpoint host identity is incomplete"
        raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicSCFExecutionIdentity:
    """Path-independent pre-execution identity required for checkpoint reuse.

    Args:
        workload_fingerprint: Canonical workload and GTH-resource fingerprint.
        protocol_fingerprint: Frozen measurement-protocol source fingerprint.
        runtime_fingerprint: Complete executing periodic-DFT source fingerprint.
        execution_contract_fingerprint: Hash of ``execution_contract``.
        execution_contract: Full canonical pre-run execution contract.
    """

    workload_fingerprint: str
    protocol_fingerprint: str
    runtime_fingerprint: str
    execution_contract_fingerprint: str
    execution_contract: Mapping[str, object]

    def __post_init__(self) -> None:
        values = {
            field_name: getattr(self, field_name) for field_name in _SHA256_FIELDS
        }
        if not all(_is_sha256(value) for value in values.values()):
            msg = "periodic checkpoint identity fields must be lowercase SHA-256 values"
            raise ValueError(msg)
        try:
            contract = json.loads(canonical_json_bytes(dict(self.execution_contract)))
        except (TypeError, ValueError) as error:
            msg = "periodic checkpoint execution contract must be finite canonical JSON"
            raise ValueError(msg) from error
        _validate_complete_execution_contract(contract)
        if (
            contract.get("workload_fingerprint") != self.workload_fingerprint
            or contract.get("protocol_fingerprint") != self.protocol_fingerprint
            or contract.get("runtime_fingerprint") != self.runtime_fingerprint
        ):
            msg = "periodic checkpoint execution contract decomposition is inconsistent"
            raise ValueError(msg)
        observed = sha256_bytes(canonical_json_bytes(contract))
        if observed != self.execution_contract_fingerprint:
            msg = "periodic checkpoint execution contract fingerprint is inconsistent"
            raise ValueError(msg)
        object.__setattr__(self, "execution_contract", contract)

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, object],
    ) -> PeriodicSCFExecutionIdentity:
        """Construct identity from the existing DFT execution-context mapping.

        Args:
            context: Mapping returned by the frozen ``build_execution_context``
                function, or an equivalent path-independent mapping.

        Returns:
            Validated checkpoint execution identity.
        """

        if not isinstance(context, Mapping):
            msg = "periodic checkpoint execution context must be an object"
            raise ValueError(msg)
        contract = context.get("execution_contract")
        if not isinstance(contract, Mapping):
            msg = "periodic checkpoint context is missing its execution contract"
            raise ValueError(msg)
        return cls(
            workload_fingerprint=str(contract.get("workload_fingerprint", "")),
            protocol_fingerprint=str(context.get("protocol_fingerprint", "")),
            runtime_fingerprint=str(context.get("runtime_fingerprint", "")),
            execution_contract_fingerprint=str(
                context.get("execution_contract_fingerprint", "")
            ),
            execution_contract=dict(contract),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the four non-circular artifact-manifest identity fields."""

        return {field_name: getattr(self, field_name) for field_name in _SHA256_FIELDS}


@dataclass(frozen=True)
class PeriodicSCFCheckpoint:
    """Validated checkpoint envelope and private continuation payload.

    Args:
        root: Completed generation root.
        manifest: Validated shared atomic-generation manifest.
        metadata: Validated checkpoint payload metadata.
    """

    root: Path
    manifest: Mapping[str, object]
    metadata: Mapping[str, object]
    _state: _PeriodicSCFContinuationState

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe checkpoint summary without numerical arrays."""

        return {
            "artifact_kind": self.manifest["artifact_kind"],
            "artifact_schema_version": self.manifest["artifact_schema_version"],
            "manifest_sha256": self.manifest["manifest_sha256"],
            "identity": dict(self.manifest["identity"]),
            "completed_iteration": self.metadata["completed_iteration"],
            "next_iteration": self.metadata["next_iteration"],
            "owned_lane_count": len(self.metadata["owned_lanes"]),
            "resume_eligible": self.metadata["resume_eligible"],
            "statuses": dict(self.metadata["statuses"]),
            "lineage": list(self.metadata["lineage"]),
        }


def _coerce_identity(
    value: PeriodicSCFExecutionIdentity | Mapping[str, object],
) -> PeriodicSCFExecutionIdentity:
    if isinstance(value, PeriodicSCFExecutionIdentity):
        return PeriodicSCFExecutionIdentity(
            workload_fingerprint=value.workload_fingerprint,
            protocol_fingerprint=value.protocol_fingerprint,
            runtime_fingerprint=value.runtime_fingerprint,
            execution_contract_fingerprint=value.execution_contract_fingerprint,
            execution_contract=value.execution_contract,
        )
    if not isinstance(value, Mapping):
        msg = "periodic checkpoint execution context must be an object"
        raise ValueError(msg)
    return PeriodicSCFExecutionIdentity.from_context(value)


def _config_payload(config: PeriodicSCFConfig) -> dict[str, object]:
    return {
        "max_iterations": config.max_iterations,
        "density_tolerance": config.density_tolerance,
        "energy_tolerance_hartree": config.energy_tolerance,
        "orbital_tolerance": config.orbital_tolerance,
        "min_iterations": config.min_iterations,
        "mixing_beta": config.mixing_beta,
        "mixer": config.mixer,
        "davidson": {
            "max_iterations": config.davidson.max_iterations,
            "tolerance": config.davidson.tolerance,
            "max_subspace_size": config.davidson.max_subspace_size,
            "preconditioner_floor": config.davidson.preconditioner_floor,
        },
        "batch_policy": config.batch_policy(),
    }


def periodic_scf_execution_settings(
    config: PeriodicSCFConfig | None = None,
) -> dict[str, object]:
    """Return settings that bind a frozen execution context to periodic SCF.

    Args:
        config: Exact SCF controls. Defaults to ``PeriodicSCFConfig``.

    Returns:
        Mapping suitable for ``build_execution_context(settings_override=...)``.
    """

    scf_config = PeriodicSCFConfig() if config is None else config
    return {"periodic_scf": _config_payload(scf_config)}


def _solver_identity(config_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "davidson": dict(config_payload["davidson"]),
        "scf": {
            "density_tolerance": config_payload["density_tolerance"],
            "energy_tolerance_hartree": config_payload[
                "energy_tolerance_hartree"
            ],
            "max_iterations": config_payload["max_iterations"],
            "min_iterations": config_payload["min_iterations"],
            "mixer": config_payload["mixer"],
            "mixing_beta": config_payload["mixing_beta"],
            "orbital_tolerance": config_payload["orbital_tolerance"],
        },
    }


def _array_input_identity(values: object) -> dict[str, object]:
    try:
        array = np.ascontiguousarray(np.asarray(values))
    except (TypeError, ValueError) as error:
        msg = "periodic checkpoint initial state must contain concrete arrays"
        raise ValueError(msg) from error
    if (
        array.dtype.hasobject
        or array.dtype.kind not in "biufc"
        or not np.all(np.isfinite(array))
    ):
        msg = "periodic checkpoint initial arrays must be finite numeric values"
        raise ValueError(msg)
    header = {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
    }
    digest = sha256_bytes(canonical_json_bytes(header) + b"\0" + array.tobytes())
    return {**header, "sha256": digest}


def periodic_scf_initialization_identity(
    *,
    initial_density: object | None = None,
    initial_coefficients: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build a path-independent identity for fresh periodic-SCF initialization.

    Args:
        initial_density: Optional caller-supplied density grid.
        initial_coefficients: Optional caller-supplied coefficient stack per
            explicit k-point.

    Returns:
        Canonical initialization identity for an execution contract.
    """

    density_identity: object = "uniform-electron-count-over-cell-volume"
    if initial_density is not None:
        density_identity = {
            "kind": "caller-supplied-density",
            **_array_input_identity(initial_density),
        }
    coefficient_identity: object = "lowest-kinetic-active-plane-waves"
    if initial_coefficients is not None:
        coefficient_identity = {
            "kind": "caller-supplied-explicit-kpoint-coefficients",
            "lanes": [
                _array_input_identity(values) for values in initial_coefficients
            ],
        }
    return {
        "density": density_identity,
        "orbitals": coefficient_identity,
        "random_seed": None,
    }


def _validate_execution_calculation_binding(
    identity: PeriodicSCFExecutionIdentity,
    calculation: Mapping[str, object],
) -> None:
    contract = identity.execution_contract
    settings = contract.get("settings_override")
    calculation_config = calculation.get("config")
    environment = contract.get("environment")
    eigensolve = calculation.get("eigensolve")
    environment_matches = isinstance(environment, Mapping) and (
        environment.get("selected_device") == calculation.get("selected_device")
        and environment.get("precision") == calculation.get("precision")
        and isinstance(eigensolve, Mapping)
        and all(environment.get(key) == value for key, value in eigensolve.items())
    )
    if (
        contract.get("command_kind") != PERIODIC_SCF_COMMAND_KIND
        or not isinstance(settings, Mapping)
        or settings.get("periodic_scf") != calculation_config
        or not isinstance(calculation_config, Mapping)
        or contract.get("solver") != _solver_identity(calculation_config)
        or not environment_matches
    ):
        msg = (
            "periodic checkpoint execution contract is not bound to the exact "
            "SCF and batch settings"
        )
        raise ArtifactIntegrityError(msg)


def _validate_initialization_binding(
    identity: PeriodicSCFExecutionIdentity,
    *,
    initial_density: object | None,
    initial_coefficients: Sequence[object] | None,
) -> None:
    observed = periodic_scf_initialization_identity(
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
    )
    if identity.execution_contract.get("initialization") != observed:
        msg = "periodic checkpoint initialization does not match its execution contract"
        raise ArtifactIntegrityError(msg)


def _pseudopotential_payload(system: PeriodicDFTSystem) -> dict[str, object]:
    pseudo = system.pseudopotential
    if pseudo.format != PseudopotentialFormat.GTH:
        msg = "periodic SCF checkpoints require a GTH pseudopotential"
        raise ValueError(msg)
    return {
        "element": pseudo.element,
        "format": pseudo.format.value,
        "valence_charge": pseudo.valence_charge,
        "gth_rloc": pseudo.gth_rloc,
        "gth_coefficients": list(pseudo.gth_coefficients),
        "gth_channels": [
            {
                "angular_momentum": channel.angular_momentum,
                "radius": channel.radius,
                "coupling_matrix": [list(row) for row in channel.coupling_matrix],
            }
            for channel in pseudo.gth_channels
        ],
    }


def periodic_scf_calculation_contract(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> dict[str, object]:
    """Build the path-independent calculation contract used by checkpoints.

    Args:
        system: Periodic GTH system.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        n_bands: Occupied band count. Defaults to half the electron count.
        config: SCF controls. Defaults to ``PeriodicSCFConfig``.
        xc_functional: Exchange-correlation functional. Only the deterministic
            production PBE path is checkpointable.

    Returns:
        Canonical JSON-compatible calculation settings and physics identity.

    Raises:
        ValueError: If a custom exchange-correlation implementation lacks the
            stable production-PBE identity.
    """

    from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation

    scf_config = PeriodicSCFConfig() if config is None else config
    bands = int(round(system.electron_count / 2.0)) if n_bands is None else n_bands
    if xc_functional is not None and type(xc_functional) is not (
        ProductionPBEExchangeCorrelation
    ):
        msg = "periodic checkpointing supports only the stable production PBE path"
        raise ValueError(msg)
    return {
        "schema_version": _CALCULATION_CONTRACT_SCHEMA,
        "system": {
            "cell_matrix_bohr": np.asarray(
                system.grid.cell.matrix,
                dtype=np.float64,
            ).tolist(),
            "grid_shape": list(system.grid.shape),
            "positions_bohr": np.asarray(system.positions, dtype=np.float64).tolist(),
            "electron_count": system.electron_count,
            "pseudopotential": _pseudopotential_payload(system),
        },
        "cutoff_hartree": float(cutoff_hartree),
        "n_bands": int(bands),
        "kpoints": [
            {
                "reduced_kpoint": list(point.vector),
                "weight": point.weight,
                "coordinate_system": point.coordinate_system,
            }
            for point in kpoint_mesh.points
        ],
        "xc_functional": "production-pbe-v1",
        "config": _config_payload(scf_config),
        "precision": "complex64/float32",
        "selected_device": str(mx.default_device()),
        "eigensolve": _eigensolve_provenance(),
    }


def _calculation_fingerprint(contract: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(contract)))


def _mixer_payload(
    state: _MixerCheckpointState,
    payloads: dict[str, bytes],
    payload_roles: dict[str, str],
) -> dict[str, object]:
    density_files = []
    residual_files = []
    for index, values in enumerate(state.densities):
        path = f"mixer/density-{index:04d}.npy"
        payloads[path] = _npy_bytes(values)
        payload_roles[path] = "diis_density_history"
        density_files.append(path)
    for index, values in enumerate(state.residuals):
        path = f"mixer/residual-{index:04d}.npy"
        payloads[path] = _npy_bytes(values)
        payload_roles[path] = "diis_residual_history"
        residual_files.append(path)
    return {
        "name": state.name,
        "beta": state.beta,
        "history_size": state.history_size,
        "regularization": state.regularization,
        "stored": len(state.densities),
        "last_coefficients": list(state.last_coefficients),
        "density_files": density_files,
        "residual_files": residual_files,
    }


def _publish_checkpoint_state(
    destination: str | Path,
    *,
    state: _PeriodicSCFContinuationState,
    identity: PeriodicSCFExecutionIdentity,
    calculation_contract: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, object]:
    payloads: dict[str, bytes] = {}
    payload_roles: dict[str, str] = {}
    density_file = "density.npy"
    payloads[density_file] = _npy_bytes(state.density)
    payload_roles[density_file] = "next_iteration_density"

    coefficient_map = state.coefficient_map
    owned_lanes = []
    for lane in state.owned_lanes:
        owner_index = int(lane["owner_index"])
        path = f"owned/{owner_index:04d}-coefficients.npy"
        payloads[path] = _npy_bytes(coefficient_map[owner_index])
        payload_roles[path] = "owned_compact_coefficients"
        owned_lanes.append(
            {
                **dict(lane),
                "coefficient_file": path,
                "coefficient_dtype": "complex64",
                "coefficient_shape": list(coefficient_map[owner_index].shape),
            }
        )

    mixer = _mixer_payload(state.mixer_state, payloads, payload_roles)
    metadata: dict[str, object] = {
        "schema_version": PERIODIC_SCF_CHECKPOINT_SCHEMA,
        "status": "accepted_iteration",
        "resume_eligible": True,
        "completed_iteration": state.completed_iteration,
        "next_iteration": state.completed_iteration + 1,
        "previous_energy_hartree": state.previous_energy,
        "energy_by_term_hartree": dict(state.energy_by_term),
        "history": [dict(row) for row in state.history],
        "density_file": density_file,
        "owned_lanes": owned_lanes,
        "ownership": dict(state.ownership),
        "mixer": mixer,
        "execution_identity": identity.to_dict(),
        "execution_contract": dict(identity.execution_contract),
        "calculation_contract": dict(calculation_contract),
        "calculation_fingerprint": _calculation_fingerprint(calculation_contract),
        "payload_roles": payload_roles,
        "lineage": list(state.lineage),
        "statuses": {
            "numerical_status": "accepted_iteration",
            "resume_integrity_status": (
                "validated_parent" if state.lineage else "fresh"
            ),
            "timing_admission_status": "not_a_timing_sample",
        },
        "provenance": dict(provenance or {}),
    }
    with AtomicGeneration(
        Path(destination),
        PERIODIC_SCF_CHECKPOINT_KIND,
        PERIODIC_SCF_CHECKPOINT_SCHEMA,
        identity=identity.to_dict(),
        metadata={
            "status": metadata["status"],
            "resume_eligible": True,
            "completed_iteration": state.completed_iteration,
            "lineage": list(state.lineage),
            "statuses": metadata["statuses"],
        },
        fault_hook=fault_hook,
    ) as generation:
        for relative_path, payload in sorted(payloads.items()):
            generation.write_bytes(relative_path, payload)
        generation.write_json(PERIODIC_SCF_CHECKPOINT_PAYLOAD, metadata)
        return generation.publish()


def publish_periodic_scf_checkpoint(
    destination: str | Path,
    result: PeriodicSCFResult,
    *,
    system: PeriodicDFTSystem,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    execution_context: PeriodicSCFExecutionIdentity | Mapping[str, object],
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    provenance: Mapping[str, object] | None = None,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Atomically publish an eligible periodic-SCF next-iteration checkpoint.

    Args:
        destination: Previously absent final generation directory.
        result: Non-converged SCF result ending at an accepted iteration.
        system: Periodic GTH system used by ``result``.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        execution_context: Existing complete execution context or validated identity.
        n_bands: Occupied band count. Defaults to half the electron count.
        config: Exact SCF controls. Defaults to ``PeriodicSCFConfig``.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        provenance: Optional non-identity Git or caller provenance.
        fault_hook: Optional deterministic publication-stage test hook.

    Returns:
        Completed shared atomic-generation manifest.

    Raises:
        ValueError: If ``result`` has no eligible next-iteration state.
        FileExistsError: If ``destination`` already exists.
    """

    state = result._checkpoint_state
    if state is None or result.converged or result.status != "checkpointed":
        msg = "periodic SCF result is not eligible for next-iteration resume"
        raise ValueError(msg)
    scf_config = PeriodicSCFConfig() if config is None else config
    if state.completed_iteration >= scf_config.max_iterations:
        msg = "periodic SCF checkpoint has no configured next iteration"
        raise ValueError(msg)
    identity = _coerce_identity(execution_context)
    calculation = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=scf_config,
        xc_functional=xc_functional,
    )
    _validate_execution_calculation_binding(identity, calculation)
    calculation_fingerprint = _calculation_fingerprint(calculation)
    if (
        result._artifact_execution_contract_fingerprint
        != identity.execution_contract_fingerprint
        or result._artifact_calculation_fingerprint != calculation_fingerprint
    ):
        msg = "periodic SCF result is not bound to the supplied artifact identity"
        raise ValueError(msg)
    return _publish_checkpoint_state(
        destination,
        state=state,
        identity=identity,
        calculation_contract=calculation,
        provenance=provenance,
        fault_hook=fault_hook,
    )


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        msg = f"periodic checkpoint {field_name} must be an object"
        raise ArtifactIntegrityError(msg)
    return dict(value)


def _require_sequence(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"periodic checkpoint {field_name} must be an array"
        raise ArtifactIntegrityError(msg)
    return list(value)


def _declared_payload_paths(manifest: Mapping[str, object]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        msg = "periodic checkpoint manifest file inventory is missing"
        raise ArtifactIntegrityError(msg)
    return {
        str(record["path"])
        for record in files
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def _read_npy(
    root: Path,
    relative_path: object,
    *,
    declared_paths: set[str],
) -> np.ndarray:
    if not isinstance(relative_path, str) or relative_path not in declared_paths:
        msg = "periodic checkpoint references an undeclared numerical payload"
        raise ArtifactIntegrityError(msg)
    try:
        path = confined_path(root, relative_path, must_exist=True)
    except (ValueError, FileNotFoundError) as error:
        msg = "periodic checkpoint numerical payload path is not confined"
        raise ArtifactIntegrityError(msg) from error
    if path.is_symlink() or not path.is_file():
        msg = "periodic checkpoint numerical payload is not a regular file"
        raise ArtifactIntegrityError(msg)
    try:
        loaded = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    except (OSError, ValueError) as error:
        msg = "periodic checkpoint numerical payload is not a safe NPY array"
        raise ArtifactIntegrityError(msg) from error
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        msg = "periodic checkpoint numerical payload must be one NPY array"
        raise ArtifactIntegrityError(msg)
    return np.array(loaded, copy=True)


def _validate_metadata_identity(
    manifest: Mapping[str, object],
    metadata: Mapping[str, object],
    expected_identity: PeriodicSCFExecutionIdentity | None,
) -> PeriodicSCFExecutionIdentity:
    stored_identity = _require_mapping(
        metadata.get("execution_identity"),
        "execution identity",
    )
    contract = _require_mapping(metadata.get("execution_contract"), "execution contract")
    try:
        identity = PeriodicSCFExecutionIdentity(
            workload_fingerprint=str(stored_identity.get("workload_fingerprint", "")),
            protocol_fingerprint=str(stored_identity.get("protocol_fingerprint", "")),
            runtime_fingerprint=str(stored_identity.get("runtime_fingerprint", "")),
            execution_contract_fingerprint=str(
                stored_identity.get("execution_contract_fingerprint", "")
            ),
            execution_contract=contract,
        )
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error
    if dict(manifest.get("identity", {})) != identity.to_dict():
        msg = "periodic checkpoint envelope and payload identities differ"
        raise ArtifactIntegrityError(msg)
    if expected_identity is not None and identity != expected_identity:
        msg = "periodic checkpoint execution identity does not match the current run"
        raise ArtifactIntegrityError(msg)
    return identity


def _validated_checkpoint_metadata(
    artifact: str | Path,
    *,
    expected_identity: PeriodicSCFExecutionIdentity | None = None,
    expected_calculation: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    manifest = inspect_generation(artifact)
    if (
        manifest.get("artifact_kind") != PERIODIC_SCF_CHECKPOINT_KIND
        or manifest.get("artifact_schema_version") != PERIODIC_SCF_CHECKPOINT_SCHEMA
    ):
        msg = "artifact is not a supported periodic SCF checkpoint"
        raise ArtifactIntegrityError(msg)
    root = generation_root(artifact)
    metadata = read_generation_json(root, PERIODIC_SCF_CHECKPOINT_PAYLOAD)
    if not isinstance(metadata, dict):
        msg = "periodic checkpoint metadata must be an object"
        raise ArtifactIntegrityError(msg)
    if metadata.get("schema_version") != PERIODIC_SCF_CHECKPOINT_SCHEMA:
        msg = "unsupported periodic SCF checkpoint payload schema"
        raise ArtifactIntegrityError(msg)
    if metadata.get("status") != "accepted_iteration" or metadata.get(
        "resume_eligible"
    ) is not True:
        msg = "periodic SCF checkpoint is not an accepted resume boundary"
        raise ArtifactIntegrityError(msg)
    expected_envelope_metadata = {
        "status": metadata["status"],
        "resume_eligible": metadata["resume_eligible"],
        "completed_iteration": metadata.get("completed_iteration"),
        "lineage": metadata.get("lineage"),
        "statuses": metadata.get("statuses"),
    }
    if manifest.get("metadata") != expected_envelope_metadata:
        msg = "periodic checkpoint envelope and payload metadata differ"
        raise ArtifactIntegrityError(msg)
    stored_execution_identity = _validate_metadata_identity(
        manifest,
        metadata,
        expected_identity,
    )
    calculation = _require_mapping(
        metadata.get("calculation_contract"),
        "calculation contract",
    )
    if calculation.get("schema_version") != _CALCULATION_CONTRACT_SCHEMA:
        msg = "unsupported periodic SCF calculation contract"
        raise ArtifactIntegrityError(msg)
    observed_calculation = _calculation_fingerprint(calculation)
    if metadata.get("calculation_fingerprint") != observed_calculation:
        msg = "periodic checkpoint calculation fingerprint is inconsistent"
        raise ArtifactIntegrityError(msg)
    if expected_calculation is not None and calculation != dict(expected_calculation):
        msg = "periodic checkpoint calculation settings do not match the current run"
        raise ArtifactIntegrityError(msg)
    try:
        _validate_execution_calculation_binding(stored_execution_identity, calculation)
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error
    try:
        completed_iteration = int(metadata["completed_iteration"])
        next_iteration = int(metadata["next_iteration"])
        previous_energy = float(metadata["previous_energy_hartree"])
    except (KeyError, TypeError, ValueError) as error:
        msg = "periodic checkpoint iteration or energy cursor is invalid"
        raise ArtifactIntegrityError(msg) from error
    if (
        completed_iteration <= 0
        or next_iteration != completed_iteration + 1
        or not np.isfinite(previous_energy)
    ):
        msg = "periodic checkpoint iteration or energy cursor is inconsistent"
        raise ArtifactIntegrityError(msg)
    history = _require_sequence(metadata.get("history"), "history")
    if len(history) != completed_iteration or any(
        not isinstance(row, Mapping) or row.get("iteration") != index
        for index, row in enumerate(history, start=1)
    ):
        msg = "periodic checkpoint history does not match its iteration cursor"
        raise ArtifactIntegrityError(msg)
    owned_lanes = _require_sequence(metadata.get("owned_lanes"), "owned lanes")
    ownership = _require_mapping(metadata.get("ownership"), "ownership")
    if ownership.get("owned_count") != len(owned_lanes):
        msg = "periodic checkpoint ownership and owner payload counts differ"
        raise ArtifactIntegrityError(msg)
    mixer = _require_mapping(metadata.get("mixer"), "mixer")
    density_files = _require_sequence(mixer.get("density_files"), "mixer density files")
    residual_files = _require_sequence(
        mixer.get("residual_files"),
        "mixer residual files",
    )
    try:
        stored = int(mixer["stored"])
        history_size = int(mixer["history_size"])
        beta = float(mixer["beta"])
        regularization = float(mixer["regularization"])
        last_coefficients = [float(value) for value in mixer["last_coefficients"]]
    except (KeyError, TypeError, ValueError) as error:
        msg = "periodic checkpoint mixer metadata is invalid"
        raise ArtifactIntegrityError(msg) from error
    if (
        stored != len(density_files)
        or stored != len(residual_files)
        or stored < 0
        or stored > history_size
        or not np.isfinite(beta)
        or not np.isfinite(regularization)
        or not np.all(np.isfinite(np.asarray(last_coefficients, dtype=np.float64)))
    ):
        msg = "periodic checkpoint mixer metadata is inconsistent"
        raise ArtifactIntegrityError(msg)
    configured = _require_mapping(calculation.get("config"), "configured SCF controls")
    configured_mixer = configured.get("mixer")
    configured_beta = configured.get("mixing_beta")
    if configured_mixer == "linear":
        mixer_matches_config = (
            mixer.get("name") == "linear"
            and beta == configured_beta
            and history_size == 0
            and regularization == 0.0
            and stored == 0
            and not last_coefficients
        )
    elif configured_mixer == "diis":
        allowed_coefficient_counts = {1, stored} if stored else {0}
        mixer_matches_config = (
            mixer.get("name") == "pulay-diis"
            and beta == configured_beta
            and history_size == 6
            and regularization == 1e-10
            and len(last_coefficients) in allowed_coefficient_counts
        )
    else:
        mixer_matches_config = False
    if not mixer_matches_config:
        msg = "periodic checkpoint mixer state does not match configured SCF controls"
        raise ArtifactIntegrityError(msg)
    statuses = _require_mapping(metadata.get("statuses"), "statuses")
    if (
        statuses.get("numerical_status") != "accepted_iteration"
        or statuses.get("resume_integrity_status") not in {"fresh", "validated_parent"}
        or statuses.get("timing_admission_status") != "not_a_timing_sample"
    ):
        msg = "periodic checkpoint status fields are inconsistent"
        raise ArtifactIntegrityError(msg)
    lineage = _require_sequence(metadata.get("lineage"), "lineage")
    if not all(_is_sha256(value) for value in lineage):
        msg = "periodic checkpoint lineage entries must be SHA-256 values"
        raise ArtifactIntegrityError(msg)
    if bool(lineage) != (statuses["resume_integrity_status"] == "validated_parent"):
        msg = "periodic checkpoint lineage and resume status differ"
        raise ArtifactIntegrityError(msg)

    declared_paths = _declared_payload_paths(manifest)
    if PERIODIC_SCF_CHECKPOINT_PAYLOAD not in declared_paths:
        msg = "periodic checkpoint metadata payload is absent from the manifest"
        raise ArtifactIntegrityError(msg)
    references: list[object] = [metadata.get("density_file")]
    references.extend(
        _require_mapping(lane, "owned lane").get("coefficient_file")
        for lane in owned_lanes
    )
    references.extend(density_files)
    references.extend(residual_files)
    for reference in references:
        if not isinstance(reference, str) or reference not in declared_paths:
            msg = "periodic checkpoint references an undeclared payload"
            raise ArtifactIntegrityError(msg)
        try:
            confined_path(root, reference, must_exist=True)
        except (ValueError, FileNotFoundError) as error:
            msg = "periodic checkpoint payload reference is not confined"
            raise ArtifactIntegrityError(msg) from error
    payload_roles = _require_mapping(metadata.get("payload_roles"), "payload roles")
    if set(payload_roles) != declared_paths - {PERIODIC_SCF_CHECKPOINT_PAYLOAD}:
        msg = "periodic checkpoint semantic payload roles are incomplete"
        raise ArtifactIntegrityError(msg)
    return root, dict(manifest), metadata


def inspect_periodic_scf_checkpoint(
    artifact: str | Path,
    *,
    expected_execution_context: (
        PeriodicSCFExecutionIdentity | Mapping[str, object] | None
    ) = None,
) -> dict[str, object]:
    """Validate checkpoint integrity and identity metadata without array loading.

    Args:
        artifact: Explicit completed checkpoint generation or nested payload.
        expected_execution_context: Optional current context requiring an exact
            workload/protocol/runtime/execution-contract identity match.

    Returns:
        JSON-safe checkpoint summary.

    Raises:
        ArtifactIntegrityError: If integrity, schema, or identity validation fails.
    """

    expected = (
        None
        if expected_execution_context is None
        else _coerce_identity(expected_execution_context)
    )
    _, manifest, metadata = _validated_checkpoint_metadata(
        artifact,
        expected_identity=expected,
    )
    return {
        "status": "ok",
        "artifact_kind": manifest["artifact_kind"],
        "artifact_schema_version": manifest["artifact_schema_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "identity": dict(manifest["identity"]),
        "completed_iteration": metadata["completed_iteration"],
        "next_iteration": metadata["next_iteration"],
        "owned_lane_count": len(metadata["owned_lanes"]),
        "resume_eligible": metadata["resume_eligible"],
        "statuses": dict(metadata["statuses"]),
        "lineage": list(metadata["lineage"]),
    }


def load_periodic_scf_checkpoint(
    artifact: str | Path,
    *,
    system: PeriodicDFTSystem,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    execution_context: PeriodicSCFExecutionIdentity | Mapping[str, object],
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> PeriodicSCFCheckpoint:
    """Load one explicit checkpoint after complete identity and payload validation.

    Args:
        artifact: Explicit completed checkpoint generation or nested payload.
        system: Current periodic GTH system.
        cutoff_hartree: Current plane-wave cutoff in Hartree.
        kpoint_mesh: Current weighted reduced-coordinate k-point mesh.
        execution_context: Current complete execution context or validated identity.
        n_bands: Current occupied band count. Defaults to half the electron count.
        config: Current exact SCF controls. Defaults to ``PeriodicSCFConfig``.
        xc_functional: Current exchange-correlation functional.

    Returns:
        Validated checkpoint containing a private continuation state.

    Raises:
        ArtifactIntegrityError: If integrity, identity, settings, or arrays differ.
    """

    identity = _coerce_identity(execution_context)
    calculation = periodic_scf_calculation_contract(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        xc_functional=xc_functional,
    )
    _validate_execution_calculation_binding(identity, calculation)
    root, manifest, metadata = _validated_checkpoint_metadata(
        artifact,
        expected_identity=identity,
        expected_calculation=calculation,
    )
    declared_paths = _declared_payload_paths(manifest)
    density = _read_npy(
        root,
        metadata.get("density_file"),
        declared_paths=declared_paths,
    )
    if density.dtype != np.float32 or density.shape != system.grid.shape or not np.all(
        np.isfinite(density)
    ):
        msg = "periodic checkpoint density has invalid dtype, shape, or values"
        raise ArtifactIntegrityError(msg)

    owned_lane_payloads = _require_sequence(metadata.get("owned_lanes"), "owned lanes")
    owned_lanes: list[dict[str, object]] = []
    owned_coefficients: list[tuple[int, mx.array]] = []
    seen_owners: set[int] = set()
    for raw_lane in owned_lane_payloads:
        lane = _require_mapping(raw_lane, "owned lane")
        try:
            owner_index = int(lane["owner_index"])
        except (KeyError, TypeError, ValueError) as error:
            msg = "periodic checkpoint owner index is invalid"
            raise ArtifactIntegrityError(msg) from error
        if owner_index in seen_owners:
            msg = "periodic checkpoint owner indices must be unique"
            raise ArtifactIntegrityError(msg)
        seen_owners.add(owner_index)
        values = _read_npy(
            root,
            lane.get("coefficient_file"),
            declared_paths=declared_paths,
        )
        expected_shape = tuple(lane.get("coefficient_shape", ()))
        if (
            values.dtype != np.complex64
            or values.shape != expected_shape
            or lane.get("coefficient_dtype") != "complex64"
            or not np.all(np.isfinite(values))
        ):
            msg = "periodic checkpoint coefficients have invalid dtype, shape, or values"
            raise ArtifactIntegrityError(msg)
        owned_lanes.append(
            {
                key: value
                for key, value in lane.items()
                if key
                not in {
                    "coefficient_file",
                    "coefficient_dtype",
                    "coefficient_shape",
                }
            }
        )
        owned_coefficients.append((owner_index, mx.array(values)))

    mixer_payload = _require_mapping(metadata.get("mixer"), "mixer")
    density_files = _require_sequence(
        mixer_payload.get("density_files"),
        "mixer density files",
    )
    residual_files = _require_sequence(
        mixer_payload.get("residual_files"),
        "mixer residual files",
    )
    mixer_densities = tuple(
        _read_npy(root, path, declared_paths=declared_paths) for path in density_files
    )
    mixer_residuals = tuple(
        _read_npy(root, path, declared_paths=declared_paths) for path in residual_files
    )
    if any(
        values.dtype != np.float32
        or values.shape != system.grid.shape
        or not np.all(np.isfinite(values))
        for values in (*mixer_densities, *mixer_residuals)
    ):
        msg = "periodic checkpoint mixer arrays have invalid dtype, shape, or values"
        raise ArtifactIntegrityError(msg)
    try:
        mixer_state = _MixerCheckpointState(
            name=str(mixer_payload["name"]),
            beta=float(mixer_payload["beta"]),
            history_size=int(mixer_payload["history_size"]),
            regularization=float(mixer_payload["regularization"]),
            densities=tuple(mx.array(values) for values in mixer_densities),
            residuals=tuple(mx.array(values) for values in mixer_residuals),
            last_coefficients=tuple(
                float(value) for value in mixer_payload["last_coefficients"]
            ),
        )
        completed_iteration = int(metadata["completed_iteration"])
        next_iteration = int(metadata["next_iteration"])
        previous_energy = float(metadata["previous_energy_hartree"])
    except (KeyError, TypeError, ValueError) as error:
        msg = "periodic checkpoint scalar state is invalid"
        raise ArtifactIntegrityError(msg) from error
    if completed_iteration <= 0 or next_iteration != completed_iteration + 1:
        msg = "periodic checkpoint iteration cursor is inconsistent"
        raise ArtifactIntegrityError(msg)
    history = _require_sequence(metadata.get("history"), "history")
    history_rows = tuple(_require_mapping(row, "history row") for row in history)
    ownership = _require_mapping(metadata.get("ownership"), "ownership")
    energy_terms_raw = _require_mapping(
        metadata.get("energy_by_term_hartree"),
        "energy terms",
    )
    try:
        energy_terms = {key: float(value) for key, value in energy_terms_raw.items()}
    except (TypeError, ValueError) as error:
        msg = "periodic checkpoint energy terms must be numeric"
        raise ArtifactIntegrityError(msg) from error
    if not np.isfinite(previous_energy) or not np.all(
        np.isfinite(np.asarray(list(energy_terms.values()), dtype=np.float64))
    ):
        msg = "periodic checkpoint energy state must be finite"
        raise ArtifactIntegrityError(msg)
    lineage_raw = _require_sequence(metadata.get("lineage"), "lineage")
    lineage = tuple(str(item) for item in lineage_raw)
    if not all(_is_sha256(item) for item in lineage):
        msg = "periodic checkpoint lineage entries must be SHA-256 values"
        raise ArtifactIntegrityError(msg)
    state = _PeriodicSCFContinuationState(
        completed_iteration=completed_iteration,
        density=mx.array(density),
        owned_coefficients=tuple(owned_coefficients),
        owned_lanes=tuple(owned_lanes),
        previous_energy=previous_energy,
        energy_by_term=energy_terms,
        history=history_rows,
        mixer_state=mixer_state,
        ownership=ownership,
        lineage=(*lineage, str(manifest["manifest_sha256"])),
    )
    return PeriodicSCFCheckpoint(
        root=root,
        manifest=manifest,
        metadata=metadata,
        _state=state,
    )


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
        n_bands: Occupied band count. Defaults to half the electron count.
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
        _artifact_execution_contract_fingerprint=(
            identity.execution_contract_fingerprint
        ),
        _artifact_calculation_fingerprint=_calculation_fingerprint(calculation),
    )
