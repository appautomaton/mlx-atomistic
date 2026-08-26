"""Atomic outer checkpoints for periodic geometry optimization."""

from __future__ import annotations

import io
from collections.abc import Mapping
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
from mlx_atomistic.dft._periodic_optimization_state import (
    _PeriodicGeometryContinuationState,
)

PERIODIC_GEOMETRY_CHECKPOINT_KIND = "periodic_geometry_optimization_checkpoint"
PERIODIC_GEOMETRY_CHECKPOINT_SCHEMA = "mlx-atomistic.periodic-geometry-checkpoint.v1"
PERIODIC_GEOMETRY_CHECKPOINT_PAYLOAD = "checkpoint.json"


def _npy_bytes(values: object) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def _calculation_fingerprint(contract: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(contract)))


def _publish_periodic_geometry_checkpoint(
    destination: str | Path,
    *,
    state: _PeriodicGeometryContinuationState,
    calculation_contract: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish one accepted outer-step checkpoint without replacement."""

    calculation = dict(calculation_contract)
    fingerprint = _calculation_fingerprint(calculation)
    payload_roles = {
        "positions.npy": "accepted_positions",
        "density.npy": "next_scf_density_seed",
        "forces.npy": "accepted_analytic_forces",
    }
    history = []
    for index, (_s_vector, _y_vector) in enumerate(
        zip(state.s_history, state.y_history, strict=True)
    ):
        s_path = f"optimizer/s-{index:04d}.npy"
        y_path = f"optimizer/y-{index:04d}.npy"
        payload_roles[s_path] = "lbfgs_position_difference"
        payload_roles[y_path] = "lbfgs_gradient_difference"
        history.append({"s_file": s_path, "y_file": y_path})
    metadata: dict[str, object] = {
        "schema_version": PERIODIC_GEOMETRY_CHECKPOINT_SCHEMA,
        "status": "accepted_step",
        "resume_eligible": True,
        "completed_step": state.completed_step,
        "next_step": state.completed_step + 1,
        "energy_hartree": state.energy,
        "positions_file": "positions.npy",
        "density_file": "density.npy",
        "forces_file": "forces.npy",
        "optimizer_history": history,
        "steps": [dict(step) for step in state.steps],
        "scf_evaluations": state.scf_evaluations,
        "line_search_evaluations": state.line_search_evaluations,
        "calculation_contract": calculation,
        "calculation_fingerprint": fingerprint,
        "payload_roles": payload_roles,
        "lineage": list(state.lineage),
        "provenance": dict(provenance or {}),
    }
    with AtomicGeneration(
        Path(destination),
        PERIODIC_GEOMETRY_CHECKPOINT_KIND,
        PERIODIC_GEOMETRY_CHECKPOINT_SCHEMA,
        identity={"calculation_fingerprint": fingerprint},
        metadata={
            "status": "accepted_step",
            "resume_eligible": True,
            "completed_step": state.completed_step,
            "lineage": list(state.lineage),
        },
    ) as generation:
        generation.write_bytes("positions.npy", _npy_bytes(state.positions))
        generation.write_bytes("density.npy", _npy_bytes(state.density))
        generation.write_bytes("forces.npy", _npy_bytes(state.forces))
        for record, s_vector, y_vector in zip(
            history,
            state.s_history,
            state.y_history,
            strict=True,
        ):
            generation.write_bytes(str(record["s_file"]), _npy_bytes(s_vector))
            generation.write_bytes(str(record["y_file"]), _npy_bytes(y_vector))
        generation.write_json(PERIODIC_GEOMETRY_CHECKPOINT_PAYLOAD, metadata)
        return generation.publish()


def _declared_payloads(manifest: Mapping[str, object]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ArtifactIntegrityError("periodic geometry checkpoint inventory is missing")
    return {
        str(record["path"])
        for record in files
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def _load_npy(
    root: Path,
    relative_path: object,
    *,
    declared: set[str],
) -> np.ndarray:
    if not isinstance(relative_path, str) or relative_path not in declared:
        raise ArtifactIntegrityError("periodic geometry checkpoint references an undeclared array")
    try:
        path = confined_path(root, relative_path, must_exist=True)
    except (FileNotFoundError, ValueError) as error:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint array path is not confined"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError("periodic geometry checkpoint array must be a regular file")
    try:
        values = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint array is not safe NPY"
        ) from error
    if not isinstance(values, np.ndarray):
        raise ArtifactIntegrityError("periodic geometry checkpoint payload must be one NPY array")
    return np.array(values, copy=True)


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"periodic geometry checkpoint {name} is invalid")
    return dict(value)


def _require_steps(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ArtifactIntegrityError("periodic geometry checkpoint steps are invalid")
    return tuple(dict(item) for item in value)


def _load_periodic_geometry_checkpoint(
    artifact: str | Path,
    *,
    expected_calculation_contract: Mapping[str, object],
) -> _PeriodicGeometryContinuationState:
    """Load and validate one accepted outer-step checkpoint."""

    manifest = inspect_generation(artifact)
    if (
        manifest.get("artifact_kind") != PERIODIC_GEOMETRY_CHECKPOINT_KIND
        or manifest.get("artifact_schema_version") != PERIODIC_GEOMETRY_CHECKPOINT_SCHEMA
    ):
        raise ArtifactIntegrityError("artifact is not a periodic geometry optimization checkpoint")
    root = generation_root(artifact)
    metadata = read_generation_json(root, PERIODIC_GEOMETRY_CHECKPOINT_PAYLOAD)
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError("periodic geometry checkpoint metadata must be an object")
    if (
        metadata.get("schema_version") != PERIODIC_GEOMETRY_CHECKPOINT_SCHEMA
        or metadata.get("status") != "accepted_step"
        or metadata.get("resume_eligible") is not True
    ):
        raise ArtifactIntegrityError("periodic geometry checkpoint is not resumable")
    calculation = _require_mapping(
        metadata.get("calculation_contract"),
        "calculation contract",
    )
    fingerprint = _calculation_fingerprint(calculation)
    if metadata.get("calculation_fingerprint") != fingerprint or manifest.get("identity") != {
        "calculation_fingerprint": fingerprint
    }:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint calculation fingerprint is inconsistent"
        )
    if calculation != dict(expected_calculation_contract):
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint calculation settings do not match"
        )
    try:
        completed_step = int(metadata["completed_step"])
        next_step = int(metadata["next_step"])
        energy = float(metadata["energy_hartree"])
        scf_evaluations = int(metadata["scf_evaluations"])
        line_search_evaluations = int(metadata["line_search_evaluations"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("periodic geometry checkpoint cursor is invalid") from error
    steps = _require_steps(metadata.get("steps"))
    if (
        completed_step <= 0
        or next_step != completed_step + 1
        or len(steps) != completed_step
        or not np.isfinite(energy)
        or scf_evaluations <= 0
        or line_search_evaluations < completed_step
    ):
        raise ArtifactIntegrityError("periodic geometry checkpoint cursor is inconsistent")
    lineage = metadata.get("lineage")
    if not isinstance(lineage, list) or any(not isinstance(item, str) for item in lineage):
        raise ArtifactIntegrityError("periodic geometry checkpoint lineage is invalid")
    if manifest.get("metadata") != {
        "status": "accepted_step",
        "resume_eligible": True,
        "completed_step": completed_step,
        "lineage": lineage,
    }:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint envelope and payload metadata differ"
        )
    declared = _declared_payloads(manifest)
    history_value = metadata.get("optimizer_history")
    if not isinstance(history_value, list):
        raise ArtifactIntegrityError("periodic geometry checkpoint optimizer history is invalid")
    s_history = []
    y_history = []
    for record in history_value:
        item = _require_mapping(record, "optimizer history entry")
        s_history.append(_load_npy(root, item.get("s_file"), declared=declared))
        y_history.append(_load_npy(root, item.get("y_file"), declared=declared))
    if len(s_history) != len(y_history):
        raise ArtifactIntegrityError("periodic geometry checkpoint optimizer history is incomplete")

    electronic = _require_mapping(
        calculation.get("electronic_calculation"),
        "electronic calculation",
    )
    system = _require_mapping(electronic.get("system"), "system contract")
    optimizer = _require_mapping(calculation.get("optimizer"), "optimizer contract")
    try:
        initial_positions = np.asarray(system["positions_bohr"], dtype=np.float64)
        grid_shape = tuple(int(value) for value in system["grid_shape"])
        history_size = int(optimizer["history_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint calculation shapes are invalid"
        ) from error
    positions = _load_npy(root, metadata.get("positions_file"), declared=declared)
    density = _load_npy(root, metadata.get("density_file"), declared=declared)
    forces = _load_npy(root, metadata.get("forces_file"), declared=declared)
    arrays = [positions, density, forces, *s_history, *y_history]
    if any(values.dtype.hasobject or not np.isfinite(values).all() for values in arrays):
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint arrays must be finite numeric values"
        )
    if (
        positions.shape != initial_positions.shape
        or forces.shape != initial_positions.shape
        or density.shape != grid_shape
        or history_size <= 0
        or len(s_history) > history_size
        or any(values.shape != (positions.size,) for values in [*s_history, *y_history])
    ):
        raise ArtifactIntegrityError("periodic geometry checkpoint array shapes are inconsistent")
    try:
        step_indices = [int(step["index"]) for step in steps]
        final_step_energy = float(steps[-1]["energy_hartree"])
        final_step_positions = np.asarray(steps[-1]["positions_bohr"], dtype=np.float64)
        final_step_forces = np.asarray(
            steps[-1]["forces_hartree_per_bohr"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "periodic geometry checkpoint accepted-step history is invalid"
        ) from error
    if (
        step_indices != list(range(1, completed_step + 1))
        or final_step_energy != energy
        or not np.array_equal(final_step_positions, positions)
        or not np.array_equal(final_step_forces, forces)
    ):
        raise ArtifactIntegrityError("periodic geometry checkpoint final state and history differ")
    payload_roles = _require_mapping(metadata.get("payload_roles"), "payload roles")
    expected_roles = {
        "positions.npy": "accepted_positions",
        "density.npy": "next_scf_density_seed",
        "forces.npy": "accepted_analytic_forces",
    }
    for index in range(len(s_history)):
        expected_roles[f"optimizer/s-{index:04d}.npy"] = "lbfgs_position_difference"
        expected_roles[f"optimizer/y-{index:04d}.npy"] = "lbfgs_gradient_difference"
    if payload_roles != expected_roles:
        raise ArtifactIntegrityError("periodic geometry checkpoint payload roles are inconsistent")
    return _PeriodicGeometryContinuationState(
        completed_step=completed_step,
        positions=positions,
        density=mx.array(density),
        energy=energy,
        forces=forces,
        steps=steps,
        s_history=tuple(s_history),
        y_history=tuple(y_history),
        scf_evaluations=scf_evaluations,
        line_search_evaluations=line_search_evaluations,
        lineage=tuple(lineage) + (str(manifest["manifest_sha256"]),),
    )
