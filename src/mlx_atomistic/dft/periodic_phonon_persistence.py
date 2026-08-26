"""Portable no-pickle persistence for periodic phonon force samples."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes
from mlx_atomistic.dft._periodic_phonon_models import (
    PeriodicDisplacementPlan,
    PeriodicPhononSample,
    PeriodicPhononSampleSet,
)

PERIODIC_PHONON_SAMPLES_SCHEMA = "mlx-atomistic.periodic-phonon-samples.v1"


def _metadata(
    plan: PeriodicDisplacementPlan,
    samples: PeriodicPhononSampleSet,
) -> dict[str, object]:
    samples.missing_representatives(plan)
    return {
        "schema_version": PERIODIC_PHONON_SAMPLES_SCHEMA,
        "plan_fingerprint": plan.fingerprint,
        "system_fingerprint": plan.system_fingerprint,
        "atom_count": plan.atom_count,
        "force_unit": "hartree/bohr",
        "samples": [
            {
                "representative_dof": sample.representative_dof,
                "minus_calculation_fingerprint": (
                    sample.minus_calculation_fingerprint
                ),
                "plus_calculation_fingerprint": sample.plus_calculation_fingerprint,
            }
            for sample in samples.samples
        ],
    }


def write_periodic_phonon_samples(
    path: str | Path,
    plan: PeriodicDisplacementPlan,
    samples: PeriodicPhononSampleSet,
) -> Path:
    """Atomically publish a partial or complete phonon sample collection.

    Args:
        path: Previously absent destination NPZ path.
        plan: Exact displacement plan bound by ``samples``.
        samples: Immutable central-force sample collection.

    Returns:
        Resolved published path.

    Raises:
        FileExistsError: If the destination already exists.
        TypeError: If plan or samples use unsupported types.
        ValueError: If sample and plan identities differ.
    """

    if not isinstance(plan, PeriodicDisplacementPlan):
        raise TypeError("plan must be PeriodicDisplacementPlan")
    if not isinstance(samples, PeriodicPhononSampleSet):
        raise TypeError("samples must be PeriodicPhononSampleSet")
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(plan, samples)
    representatives = np.asarray(
        [sample.representative_dof for sample in samples.samples],
        dtype=np.int64,
    )
    shape = (len(samples.samples), plan.atom_count, 3)
    minus = np.empty(shape, dtype=np.float64)
    plus = np.empty(shape, dtype=np.float64)
    for index, sample in enumerate(samples.samples):
        minus[index] = sample.minus_forces_hartree_per_bohr
        plus[index] = sample.plus_forces_hartree_per_bohr
    payloads = {
        "metadata_json": np.frombuffer(canonical_json_bytes(metadata), dtype=np.uint8),
        "representative_dofs": representatives,
        "minus_forces_hartree_per_bohr": minus,
        "plus_forces_hartree_per_bohr": plus,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payloads)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_periodic_phonon_samples(
    path: str | Path,
    plan: PeriodicDisplacementPlan,
) -> PeriodicPhononSampleSet:
    """Load and validate a no-pickle periodic phonon sample collection.

    Args:
        path: Existing sample NPZ path.
        plan: Exact displacement plan expected by the artifact.

    Returns:
        Immutable partial or complete central-force sample collection.

    Raises:
        TypeError: If ``plan`` has an unsupported type.
        ValueError: If schema, inventory, dtypes, shapes, values, or identity
            are invalid.
    """

    if not isinstance(plan, PeriodicDisplacementPlan):
        raise TypeError("plan must be PeriodicDisplacementPlan")
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            expected_names = {
                "metadata_json",
                "representative_dofs",
                "minus_forces_hartree_per_bohr",
                "plus_forces_hartree_per_bohr",
            }
            if set(archive.files) != expected_names:
                raise ValueError("phonon sample array inventory is inconsistent")
            metadata_array = archive["metadata_json"]
            representatives = archive["representative_dofs"]
            minus = archive["minus_forces_hartree_per_bohr"]
            plus = archive["plus_forces_hartree_per_bohr"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise ValueError("phonon sample metadata array is invalid")
            if representatives.dtype != np.int64 or representatives.ndim != 1:
                raise ValueError("phonon sample representative array is invalid")
            if minus.dtype != np.float64 or plus.dtype != np.float64:
                raise ValueError("phonon sample force arrays must use float64")
            if (
                minus.shape != (representatives.size, plan.atom_count, 3)
                or plus.shape != minus.shape
                or not np.isfinite(minus).all()
                or not np.isfinite(plus).all()
            ):
                raise ValueError("phonon sample force array shapes or values are invalid")
            metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("phonon sample metadata must be a JSON object")
            if metadata.get("schema_version") != PERIODIC_PHONON_SAMPLES_SCHEMA:
                raise ValueError("unsupported periodic phonon sample schema")
            if (
                metadata.get("plan_fingerprint") != plan.fingerprint
                or metadata.get("system_fingerprint") != plan.system_fingerprint
                or metadata.get("atom_count") != plan.atom_count
                or metadata.get("force_unit") != "hartree/bohr"
            ):
                raise ValueError("phonon sample metadata does not match the plan")
            records = metadata.get("samples")
            if not isinstance(records, list) or len(records) != representatives.size:
                raise ValueError("phonon sample metadata records are inconsistent")
            samples = []
            for index, record in enumerate(records):
                if (
                    not isinstance(record, dict)
                    or record.get("representative_dof")
                    != int(representatives[index])
                ):
                    raise ValueError("phonon sample metadata order is inconsistent")
                samples.append(
                    PeriodicPhononSample(
                        representative_dof=int(representatives[index]),
                        minus_forces_hartree_per_bohr=minus[index],
                        plus_forces_hartree_per_bohr=plus[index],
                        minus_calculation_fingerprint=str(
                            record.get("minus_calculation_fingerprint", "")
                        ),
                        plus_calculation_fingerprint=str(
                            record.get("plus_calculation_fingerprint", "")
                        ),
                    )
                )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("periodic phonon sample metadata is invalid") from error
    result = PeriodicPhononSampleSet(
        plan_fingerprint=plan.fingerprint,
        atom_count=plan.atom_count,
        samples=tuple(samples),
    )
    if _metadata(plan, result) != metadata:
        raise ValueError("phonon sample metadata differs from decoded arrays")
    return result


__all__ = [
    "PERIODIC_PHONON_SAMPLES_SCHEMA",
    "read_periodic_phonon_samples",
    "write_periodic_phonon_samples",
]
