"""Atomic publication and validated decoding for periodic SCF checkpoints."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    AtomicGeneration,
    confined_path,
    generation_root,
    inspect_generation,
    read_generation_json,
)
from mlx_atomistic.dft._periodic_artifact_contracts import (
    _CALCULATION_CONTRACT_SCHEMA,
    _LEGACY_CALCULATION_CONTRACT_SCHEMA,
    PERIODIC_SCF_CHECKPOINT_KIND,
    PERIODIC_SCF_CHECKPOINT_PAYLOAD,
    PERIODIC_SCF_CHECKPOINT_SCHEMA,
    PeriodicSCFExecutionIdentity,
    _calculation_fingerprint,
    _coerce_identity,
    _is_sha256,
    _upgrade_legacy_calculation_contract,
    _validate_execution_calculation_binding,
    periodic_scf_calculation_contract,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    PeriodicSCFResult,
)
from mlx_atomistic.dft._periodic_state import _PeriodicSCFContinuationState
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.mixing import _MixerCheckpointState
from mlx_atomistic.dft.runtime_state import _npy_bytes
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


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


def _mixer_payload(
    state: _MixerCheckpointState,
    payloads: dict[str, bytes],
    payload_roles: dict[str, str],
    *,
    prefix: str = "mixer",
) -> dict[str, object]:
    density_files = []
    residual_files = []
    for index, values in enumerate(state.densities):
        path = f"{prefix}/density-{index:04d}.npy"
        payloads[path] = _npy_bytes(values)
        payload_roles[path] = "diis_density_history"
        density_files.append(path)
    for index, values in enumerate(state.residuals):
        path = f"{prefix}/residual-{index:04d}.npy"
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
    spin_payload: dict[str, object] | None = None
    if state.spin_densities is not None:
        if state.magnetization_mixer_state is None:
            raise ValueError("spin checkpoint state has no magnetization mixer")
        spin_density_files = ("spin/up-density.npy", "spin/down-density.npy")
        for label, path, values in zip(
            ("up", "down"),
            spin_density_files,
            state.spin_densities,
            strict=True,
        ):
            payloads[path] = _npy_bytes(values)
            payload_roles[path] = f"next_iteration_spin_{label}_density"
        down_coefficient_map = state.down_coefficient_map
        down_owned_lanes = []
        for lane in state.down_owned_lanes:
            owner_index = int(lane["owner_index"])
            path = f"owned/down/{owner_index:04d}-coefficients.npy"
            payloads[path] = _npy_bytes(down_coefficient_map[owner_index])
            payload_roles[path] = "owned_spin_down_compact_coefficients"
            down_owned_lanes.append(
                {
                    **dict(lane),
                    "coefficient_file": path,
                    "coefficient_dtype": "complex64",
                    "coefficient_shape": list(down_coefficient_map[owner_index].shape),
                }
            )
        spin_payload = {
            "density_files": list(spin_density_files),
            "down_owned_lanes": down_owned_lanes,
            "magnetization_mixer": _mixer_payload(
                state.magnetization_mixer_state,
                payloads,
                payload_roles,
                prefix="spin/magnetization-mixer",
            ),
        }
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
        "spin": spin_payload,
        "execution_identity": identity.to_dict(),
        "execution_contract": dict(identity.execution_contract),
        "calculation_contract": dict(calculation_contract),
        "calculation_fingerprint": _calculation_fingerprint(calculation_contract),
        "payload_roles": payload_roles,
        "lineage": list(state.lineage),
        "statuses": {
            "numerical_status": "accepted_iteration",
            "resume_integrity_status": ("validated_parent" if state.lineage else "fresh"),
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
        n_bands: Computed band count. Fixed occupations default to half the
            electron count; smearing requires additional empty bands.
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
        result._artifact_execution_contract_fingerprint != identity.execution_contract_fingerprint
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


def _load_checkpoint_metadata(
    artifact: str | Path,
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
    if (
        metadata.get("status") != "accepted_iteration"
        or metadata.get("resume_eligible") is not True
    ):
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
    return root, dict(manifest), metadata


def _validate_checkpoint_calculation(
    manifest: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    expected_identity: PeriodicSCFExecutionIdentity | None,
    expected_calculation: Mapping[str, object] | None,
) -> dict[str, object]:
    stored_execution_identity = _validate_metadata_identity(
        manifest,
        metadata,
        expected_identity,
    )
    calculation = _require_mapping(
        metadata.get("calculation_contract"),
        "calculation contract",
    )
    if calculation.get("schema_version") not in {
        _CALCULATION_CONTRACT_SCHEMA,
        _LEGACY_CALCULATION_CONTRACT_SCHEMA,
    }:
        msg = "unsupported periodic SCF calculation contract"
        raise ArtifactIntegrityError(msg)
    observed_calculation = _calculation_fingerprint(calculation)
    if metadata.get("calculation_fingerprint") != observed_calculation:
        msg = "periodic checkpoint calculation fingerprint is inconsistent"
        raise ArtifactIntegrityError(msg)
    comparison = _upgrade_legacy_calculation_contract(calculation)
    if expected_calculation is not None and comparison != dict(expected_calculation):
        msg = "periodic checkpoint calculation settings do not match the current run"
        raise ArtifactIntegrityError(msg)
    try:
        _validate_execution_calculation_binding(stored_execution_identity, comparison)
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error
    return calculation


def _validate_checkpoint_cursor(metadata: Mapping[str, object]) -> None:
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


def _validate_checkpoint_owner_inventory(
    metadata: Mapping[str, object],
) -> list[object]:
    owned_lanes = _require_sequence(metadata.get("owned_lanes"), "owned lanes")
    ownership = _require_mapping(metadata.get("ownership"), "ownership")
    if ownership.get("owned_count") != len(owned_lanes):
        msg = "periodic checkpoint ownership and owner payload counts differ"
        raise ArtifactIntegrityError(msg)
    return owned_lanes


def _validate_checkpoint_mixer(
    metadata: Mapping[str, object],
    calculation: Mapping[str, object],
) -> dict[str, object]:
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
    if not _mixer_matches_config(
        mixer,
        configured,
        stored=stored,
        history_size=history_size,
        beta=beta,
        regularization=regularization,
        last_coefficients=last_coefficients,
    ):
        msg = "periodic checkpoint mixer state does not match configured SCF controls"
        raise ArtifactIntegrityError(msg)
    return mixer


def _validate_checkpoint_spin(
    metadata: Mapping[str, object],
    calculation: Mapping[str, object],
) -> dict[str, object] | None:
    configured = _require_mapping(calculation.get("config"), "configured SCF controls")
    configured_spin = configured.get("spin")
    raw_spin = metadata.get("spin")
    if configured_spin is None:
        if raw_spin is not None:
            raise ArtifactIntegrityError("unpolarized checkpoint contains spin state")
        return None
    if not isinstance(configured_spin, Mapping):
        raise ArtifactIntegrityError("configured spin controls are malformed")
    spin = _require_mapping(raw_spin, "spin checkpoint state")
    density_files = _require_sequence(spin.get("density_files"), "spin density files")
    down_owned_lanes = _require_sequence(
        spin.get("down_owned_lanes"),
        "spin-down owned lanes",
    )
    ownership = _require_mapping(metadata.get("ownership"), "ownership")
    if len(density_files) != 2 or ownership.get("owned_count") != len(down_owned_lanes):
        raise ArtifactIntegrityError("spin checkpoint channel inventory is inconsistent")
    magnetization_mixer = _require_mapping(
        spin.get("magnetization_mixer"),
        "magnetization mixer",
    )
    density_history = _require_sequence(
        magnetization_mixer.get("density_files"),
        "magnetization mixer density files",
    )
    residual_history = _require_sequence(
        magnetization_mixer.get("residual_files"),
        "magnetization mixer residual files",
    )
    try:
        valid_mixer = (
            magnetization_mixer.get("name") == "linear"
            and float(magnetization_mixer["beta"])
            == float(configured_spin["magnetization_mixing_beta"])
            and int(magnetization_mixer["history_size"]) == 0
            and float(magnetization_mixer["regularization"]) == 0.0
            and int(magnetization_mixer["stored"]) == 0
            and not magnetization_mixer["last_coefficients"]
            and not density_history
            and not residual_history
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("magnetization mixer metadata is invalid") from error
    if not valid_mixer:
        raise ArtifactIntegrityError("magnetization mixer differs from configured spin controls")
    return spin


def _mixer_matches_config(
    mixer: Mapping[str, object],
    configured: Mapping[str, object],
    *,
    stored: int,
    history_size: int,
    beta: float,
    regularization: float,
    last_coefficients: Sequence[float],
) -> bool:
    configured_mixer = configured.get("mixer")
    configured_beta = configured.get("mixing_beta")
    if configured_mixer == "linear":
        return (
            mixer.get("name") == "linear"
            and beta == configured_beta
            and history_size == 0
            and regularization == 0.0
            and stored == 0
            and not last_coefficients
        )
    if configured_mixer != "diis":
        return False
    allowed_coefficient_counts = {1, stored} if stored else {0}
    return (
        mixer.get("name") == "pulay-diis"
        and beta == configured_beta
        and history_size == 6
        and regularization == 1e-10
        and len(last_coefficients) in allowed_coefficient_counts
    )


def _validate_checkpoint_status(metadata: Mapping[str, object]) -> None:
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


def _validate_checkpoint_payload_references(
    root: Path,
    manifest: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    owned_lanes: Sequence[object],
    mixer: Mapping[str, object],
    spin: Mapping[str, object] | None,
) -> None:
    declared_paths = _declared_payload_paths(manifest)
    if PERIODIC_SCF_CHECKPOINT_PAYLOAD not in declared_paths:
        msg = "periodic checkpoint metadata payload is absent from the manifest"
        raise ArtifactIntegrityError(msg)
    references: list[object] = [metadata.get("density_file")]
    references.extend(
        _require_mapping(lane, "owned lane").get("coefficient_file") for lane in owned_lanes
    )
    references.extend(_require_sequence(mixer.get("density_files"), "mixer density files"))
    references.extend(_require_sequence(mixer.get("residual_files"), "mixer residual files"))
    if spin is not None:
        references.extend(_require_sequence(spin.get("density_files"), "spin density files"))
        references.extend(
            _require_mapping(lane, "spin-down owned lane").get("coefficient_file")
            for lane in _require_sequence(
                spin.get("down_owned_lanes"),
                "spin-down owned lanes",
            )
        )
        magnetization_mixer = _require_mapping(
            spin.get("magnetization_mixer"),
            "magnetization mixer",
        )
        references.extend(
            _require_sequence(
                magnetization_mixer.get("density_files"),
                "magnetization mixer density files",
            )
        )
        references.extend(
            _require_sequence(
                magnetization_mixer.get("residual_files"),
                "magnetization mixer residual files",
            )
        )
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


def _validated_checkpoint_metadata(
    artifact: str | Path,
    *,
    expected_identity: PeriodicSCFExecutionIdentity | None = None,
    expected_calculation: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    root, manifest, metadata = _load_checkpoint_metadata(artifact)
    calculation = _validate_checkpoint_calculation(
        manifest,
        metadata,
        expected_identity=expected_identity,
        expected_calculation=expected_calculation,
    )
    _validate_checkpoint_cursor(metadata)
    owned_lanes = _validate_checkpoint_owner_inventory(metadata)
    mixer = _validate_checkpoint_mixer(metadata, calculation)
    spin = _validate_checkpoint_spin(metadata, calculation)
    _validate_checkpoint_status(metadata)
    _validate_checkpoint_payload_references(
        root,
        manifest,
        metadata,
        owned_lanes=owned_lanes,
        mixer=mixer,
        spin=spin,
    )
    return root, dict(manifest), metadata


def inspect_periodic_scf_checkpoint(
    artifact: str | Path,
    *,
    expected_execution_context: (PeriodicSCFExecutionIdentity | Mapping[str, object] | None) = None,
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
        None if expected_execution_context is None else _coerce_identity(expected_execution_context)
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


def _load_checkpoint_density(
    root: Path,
    metadata: Mapping[str, object],
    *,
    declared_paths: set[str],
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    density = _read_npy(
        root,
        metadata.get("density_file"),
        declared_paths=declared_paths,
    )
    if (
        density.dtype != np.float32
        or density.shape != grid_shape
        or not np.all(np.isfinite(density))
    ):
        msg = "periodic checkpoint density has invalid dtype, shape, or values"
        raise ArtifactIntegrityError(msg)
    return density


def _load_checkpoint_owners(
    root: Path,
    metadata: Mapping[str, object],
    *,
    declared_paths: set[str],
) -> tuple[tuple[dict[str, object], ...], tuple[tuple[int, mx.array], ...]]:
    payloads = _require_sequence(metadata.get("owned_lanes"), "owned lanes")
    owned_lanes: list[dict[str, object]] = []
    owned_coefficients: list[tuple[int, mx.array]] = []
    seen_owners: set[int] = set()
    for raw_lane in payloads:
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
    return tuple(owned_lanes), tuple(owned_coefficients)


def _load_checkpoint_mixer_state(
    root: Path,
    metadata: Mapping[str, object],
    *,
    declared_paths: set[str],
    grid_shape: tuple[int, int, int],
) -> _MixerCheckpointState:
    payload = _require_mapping(metadata.get("mixer"), "mixer")
    density_files = _require_sequence(
        payload.get("density_files"),
        "mixer density files",
    )
    residual_files = _require_sequence(
        payload.get("residual_files"),
        "mixer residual files",
    )
    densities = tuple(
        _read_npy(root, path, declared_paths=declared_paths) for path in density_files
    )
    residuals = tuple(
        _read_npy(root, path, declared_paths=declared_paths) for path in residual_files
    )
    if any(
        values.dtype != np.float32 or values.shape != grid_shape or not np.all(np.isfinite(values))
        for values in (*densities, *residuals)
    ):
        msg = "periodic checkpoint mixer arrays have invalid dtype, shape, or values"
        raise ArtifactIntegrityError(msg)
    try:
        return _MixerCheckpointState(
            name=str(payload["name"]),
            beta=float(payload["beta"]),
            history_size=int(payload["history_size"]),
            regularization=float(payload["regularization"]),
            densities=tuple(mx.array(values) for values in densities),
            residuals=tuple(mx.array(values) for values in residuals),
            last_coefficients=tuple(float(value) for value in payload["last_coefficients"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = "periodic checkpoint scalar state is invalid"
        raise ArtifactIntegrityError(msg) from error


def _load_checkpoint_spin_state(
    root: Path,
    metadata: Mapping[str, object],
    *,
    declared_paths: set[str],
    grid_shape: tuple[int, int, int],
) -> tuple[
    tuple[mx.array, mx.array] | None,
    tuple[dict[str, object], ...],
    tuple[tuple[int, mx.array], ...],
    _MixerCheckpointState | None,
]:
    raw_spin = metadata.get("spin")
    if raw_spin is None:
        return None, (), (), None
    spin = _require_mapping(raw_spin, "spin checkpoint state")
    density_files = _require_sequence(spin.get("density_files"), "spin density files")
    densities = tuple(
        _read_npy(root, path, declared_paths=declared_paths) for path in density_files
    )
    if len(densities) != 2 or any(
        values.dtype != np.float32
        or values.shape != grid_shape
        or not np.all(np.isfinite(values))
        or np.min(values) < 0.0
        for values in densities
    ):
        raise ArtifactIntegrityError("spin checkpoint densities are invalid")
    raw_lanes = _require_sequence(
        spin.get("down_owned_lanes"),
        "spin-down owned lanes",
    )
    down_lanes: list[dict[str, object]] = []
    down_coefficients: list[tuple[int, mx.array]] = []
    seen: set[int] = set()
    for raw_lane in raw_lanes:
        lane = _require_mapping(raw_lane, "spin-down owned lane")
        try:
            owner_index = int(lane["owner_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("spin-down owner index is invalid") from error
        if owner_index in seen:
            raise ArtifactIntegrityError("spin-down owner indices must be unique")
        seen.add(owner_index)
        values = _read_npy(
            root,
            lane.get("coefficient_file"),
            declared_paths=declared_paths,
        )
        if (
            values.dtype != np.complex64
            or values.shape != tuple(lane.get("coefficient_shape", ()))
            or lane.get("coefficient_dtype") != "complex64"
            or not np.all(np.isfinite(values))
        ):
            raise ArtifactIntegrityError("spin-down coefficients are invalid")
        down_lanes.append(
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
        down_coefficients.append((owner_index, mx.array(values)))
    magnetization_mixer = _load_checkpoint_mixer_state(
        root,
        {"mixer": spin["magnetization_mixer"]},
        declared_paths=declared_paths,
        grid_shape=grid_shape,
    )
    return (
        (mx.array(densities[0]), mx.array(densities[1])),
        tuple(down_lanes),
        tuple(down_coefficients),
        magnetization_mixer,
    )


@dataclass(frozen=True)
class _LoadedCheckpointScalars:
    completed_iteration: int
    previous_energy: float
    energy_terms: dict[str, float]
    history: tuple[dict[str, object], ...]
    ownership: dict[str, object]
    lineage: tuple[str, ...]


def _load_checkpoint_scalars(
    metadata: Mapping[str, object],
) -> _LoadedCheckpointScalars:
    try:
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
    lineage = tuple(str(item) for item in _require_sequence(metadata.get("lineage"), "lineage"))
    if not all(_is_sha256(item) for item in lineage):
        msg = "periodic checkpoint lineage entries must be SHA-256 values"
        raise ArtifactIntegrityError(msg)
    return _LoadedCheckpointScalars(
        completed_iteration=completed_iteration,
        previous_energy=previous_energy,
        energy_terms=energy_terms,
        history=history_rows,
        ownership=ownership,
        lineage=lineage,
    )


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
        n_bands: Current computed band count. Fixed occupations default to half
            the electron count; smearing requires additional empty bands.
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
    density = _load_checkpoint_density(
        root,
        metadata,
        declared_paths=declared_paths,
        grid_shape=system.grid.shape,
    )
    owned_lanes, owned_coefficients = _load_checkpoint_owners(
        root,
        metadata,
        declared_paths=declared_paths,
    )
    mixer_state = _load_checkpoint_mixer_state(
        root,
        metadata,
        declared_paths=declared_paths,
        grid_shape=system.grid.shape,
    )
    (
        spin_densities,
        down_owned_lanes,
        down_owned_coefficients,
        magnetization_mixer_state,
    ) = _load_checkpoint_spin_state(
        root,
        metadata,
        declared_paths=declared_paths,
        grid_shape=system.grid.shape,
    )
    scalars = _load_checkpoint_scalars(metadata)
    state = _PeriodicSCFContinuationState(
        completed_iteration=scalars.completed_iteration,
        density=mx.array(density),
        owned_coefficients=owned_coefficients,
        owned_lanes=owned_lanes,
        previous_energy=scalars.previous_energy,
        energy_by_term=scalars.energy_terms,
        history=scalars.history,
        mixer_state=mixer_state,
        ownership=scalars.ownership,
        lineage=(*scalars.lineage, str(manifest["manifest_sha256"])),
        spin_densities=spin_densities,
        down_owned_coefficients=down_owned_coefficients,
        down_owned_lanes=down_owned_lanes,
        magnetization_mixer_state=magnetization_mixer_state,
    )
    return PeriodicSCFCheckpoint(
        root=root,
        manifest=manifest,
        metadata=metadata,
        _state=state,
    )
