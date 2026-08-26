"""Atomic accepted-step checkpoints for periodic cell optimization."""

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
from mlx_atomistic.dft._periodic_cell_optimization_state import (
    _PeriodicCellContinuationState,
)

PERIODIC_CELL_CHECKPOINT_KIND = "periodic_cell_optimization_checkpoint"
PERIODIC_CELL_CHECKPOINT_SCHEMA = "mlx-atomistic.periodic-cell-checkpoint.v1"
PERIODIC_CELL_CHECKPOINT_PAYLOAD = "checkpoint.json"
PERIODIC_CELL_CHECKPOINT_ARRAYS = "state.npz"


def _calculation_fingerprint(contract: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(contract)))


def _npz_bytes(state: _PeriodicCellContinuationState) -> bytes:
    arrays = {
        "cell": np.asarray(state.cell),
        "positions": np.asarray(state.positions),
        "fractional_positions": np.asarray(state.fractional_positions),
        "density": np.asarray(state.density),
        "stress": np.asarray(state.stress),
    }
    if state.forces is not None:
        arrays["forces"] = np.asarray(state.forces)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _publish_periodic_cell_checkpoint(
    destination: str | Path,
    *,
    state: _PeriodicCellContinuationState,
    calculation_contract: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish one accepted cell state without replacing existing evidence."""

    calculation = dict(calculation_contract)
    fingerprint = _calculation_fingerprint(calculation)
    array_keys = [
        "cell",
        "positions",
        "fractional_positions",
        "density",
        "stress",
    ]
    if state.forces is not None:
        array_keys.append("forces")
    metadata: dict[str, object] = {
        "schema_version": PERIODIC_CELL_CHECKPOINT_SCHEMA,
        "status": "accepted_step",
        "resume_eligible": True,
        "completed_step": state.completed_step,
        "next_step": state.completed_step + 1,
        "energy_hartree": state.energy,
        "arrays_file": PERIODIC_CELL_CHECKPOINT_ARRAYS,
        "array_keys": array_keys,
        "steps": [dict(step) for step in state.steps],
        "scf_evaluations": state.scf_evaluations,
        "stress_evaluations": state.stress_evaluations,
        "line_search_evaluations": state.line_search_evaluations,
        "ionic_scf_evaluations": state.ionic_scf_evaluations,
        "calculation_contract": calculation,
        "calculation_fingerprint": fingerprint,
        "lineage": list(state.lineage),
        "provenance": dict(provenance or {}),
    }
    with AtomicGeneration(
        Path(destination),
        PERIODIC_CELL_CHECKPOINT_KIND,
        PERIODIC_CELL_CHECKPOINT_SCHEMA,
        identity={"calculation_fingerprint": fingerprint},
        metadata={
            "status": "accepted_step",
            "resume_eligible": True,
            "completed_step": state.completed_step,
            "lineage": list(state.lineage),
        },
    ) as generation:
        generation.write_bytes(PERIODIC_CELL_CHECKPOINT_ARRAYS, _npz_bytes(state))
        generation.write_json(PERIODIC_CELL_CHECKPOINT_PAYLOAD, metadata)
        return generation.publish()


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"periodic cell checkpoint {label} is invalid")
    return dict(value)


def _declared_paths(manifest: Mapping[str, object]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ArtifactIntegrityError("periodic cell checkpoint inventory is missing")
    return {
        str(record["path"])
        for record in files
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def _load_arrays(root: Path, manifest: Mapping[str, object]) -> dict[str, np.ndarray]:
    if PERIODIC_CELL_CHECKPOINT_ARRAYS not in _declared_paths(manifest):
        raise ArtifactIntegrityError("periodic cell checkpoint arrays are undeclared")
    try:
        path = confined_path(root, PERIODIC_CELL_CHECKPOINT_ARRAYS, must_exist=True)
    except (FileNotFoundError, ValueError) as error:
        raise ArtifactIntegrityError("periodic cell checkpoint arrays are not confined") from error
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError("periodic cell checkpoint arrays must be a regular file")
    try:
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError("periodic cell checkpoint arrays are not safe NPZ") from error


def _load_periodic_cell_checkpoint(
    artifact: str | Path,
    *,
    expected_calculation_contract: Mapping[str, object],
) -> _PeriodicCellContinuationState:
    """Load and strictly validate one accepted cell checkpoint."""

    manifest = inspect_generation(artifact)
    if (
        manifest.get("artifact_kind") != PERIODIC_CELL_CHECKPOINT_KIND
        or manifest.get("artifact_schema_version") != PERIODIC_CELL_CHECKPOINT_SCHEMA
    ):
        raise ArtifactIntegrityError("artifact is not a periodic cell checkpoint")
    root = generation_root(artifact)
    metadata = read_generation_json(root, PERIODIC_CELL_CHECKPOINT_PAYLOAD)
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError("periodic cell checkpoint metadata must be an object")
    if (
        metadata.get("schema_version") != PERIODIC_CELL_CHECKPOINT_SCHEMA
        or metadata.get("status") != "accepted_step"
        or metadata.get("resume_eligible") is not True
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint is not resumable")
    calculation = _require_mapping(metadata.get("calculation_contract"), "contract")
    fingerprint = _calculation_fingerprint(calculation)
    if metadata.get("calculation_fingerprint") != fingerprint or manifest.get(
        "identity"
    ) != {"calculation_fingerprint": fingerprint}:
        raise ArtifactIntegrityError("periodic cell checkpoint identity is inconsistent")
    if calculation != dict(expected_calculation_contract):
        raise ArtifactIntegrityError("periodic cell checkpoint settings do not match")
    try:
        completed_step = int(metadata["completed_step"])
        next_step = int(metadata["next_step"])
        energy = float(metadata["energy_hartree"])
        counters = {
            name: int(metadata[name])
            for name in (
                "scf_evaluations",
                "stress_evaluations",
                "line_search_evaluations",
                "ionic_scf_evaluations",
            )
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("periodic cell checkpoint cursor is invalid") from error
    steps_value = metadata.get("steps")
    if not isinstance(steps_value, list) or any(
        not isinstance(item, Mapping) for item in steps_value
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint steps are invalid")
    steps = tuple(dict(item) for item in steps_value)
    lineage = metadata.get("lineage")
    if not isinstance(lineage, list) or any(not isinstance(item, str) for item in lineage):
        raise ArtifactIntegrityError("periodic cell checkpoint lineage is invalid")
    if (
        completed_step <= 0
        or next_step != completed_step + 1
        or len(steps) != completed_step
        or not np.isfinite(energy)
        or counters["scf_evaluations"] <= 0
        or counters["stress_evaluations"] <= 0
        or counters["line_search_evaluations"] < completed_step
        or counters["ionic_scf_evaluations"] < 0
        or manifest.get("metadata")
        != {
            "status": "accepted_step",
            "resume_eligible": True,
            "completed_step": completed_step,
            "lineage": lineage,
        }
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint cursor is inconsistent")
    arrays = _load_arrays(root, manifest)
    expected_keys = metadata.get("array_keys")
    if not isinstance(expected_keys, list) or set(expected_keys) != set(arrays):
        raise ArtifactIntegrityError("periodic cell checkpoint array inventory differs")
    electronic = _require_mapping(calculation.get("electronic_calculation"), "electronic contract")
    source_system = _require_mapping(electronic.get("system"), "system contract")
    try:
        source_positions = np.asarray(source_system["positions_bohr"], dtype=np.float64)
        grid_shape = tuple(int(value) for value in source_system["grid_shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("periodic cell checkpoint shapes are invalid") from error
    required = {"cell", "positions", "fractional_positions", "density", "stress"}
    if not required.issubset(arrays) or any(
        values.dtype.hasobject or not np.isfinite(values).all()
        for values in arrays.values()
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint arrays are incomplete or nonfinite")
    cell = arrays["cell"]
    positions = arrays["positions"]
    fractional = arrays["fractional_positions"]
    density = arrays["density"]
    stress = arrays["stress"]
    forces = arrays.get("forces")
    if (
        cell.shape != (3, 3)
        or positions.shape != source_positions.shape
        or fractional.shape != source_positions.shape
        or density.shape != grid_shape
        or stress.shape != (3, 3)
        or (forces is not None and forces.shape != source_positions.shape)
        or float(np.linalg.det(cell)) <= 0.0
        or not np.allclose(stress, stress.T, atol=1.0e-12, rtol=0.0)
        or not np.allclose(positions @ np.linalg.inv(cell), fractional, atol=2.0e-7)
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint array shapes are inconsistent")
    try:
        step_indices = [int(step["index"]) for step in steps]
        final_energy = float(steps[-1]["energy_hartree"])
        final_cell = np.asarray(steps[-1]["cell_matrix_bohr"], dtype=np.float64)
        final_positions = np.asarray(steps[-1]["positions_bohr"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("periodic cell checkpoint history is invalid") from error
    if (
        step_indices != list(range(1, completed_step + 1))
        or final_energy != energy
        or not np.array_equal(final_cell, cell)
        or not np.array_equal(final_positions, positions)
    ):
        raise ArtifactIntegrityError("periodic cell checkpoint final state and history differ")
    return _PeriodicCellContinuationState(
        completed_step=completed_step,
        cell=cell,
        positions=positions,
        fractional_positions=fractional,
        density=mx.array(density),
        energy=energy,
        stress=stress,
        forces=forces,
        steps=steps,
        scf_evaluations=counters["scf_evaluations"],
        stress_evaluations=counters["stress_evaluations"],
        line_search_evaluations=counters["line_search_evaluations"],
        ionic_scf_evaluations=counters["ionic_scf_evaluations"],
        lineage=tuple(lineage) + (str(manifest["manifest_sha256"]),),
    )
