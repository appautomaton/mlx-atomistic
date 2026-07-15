from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.benchmarks.pme_validation import (
    PMEManifestMismatchError,
    array_hash,
    force_error_metrics,
    manifest_hash,
    manifest_mismatches,
    require_matching_manifest,
)


def test_charged_pme_force_metrics_include_absolute_and_normalized_errors():
    reference = np.asarray([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    candidate = reference + np.asarray([[0.2, 0.0, 0.0], [0.0, -0.1, 0.0]])

    metrics = force_error_metrics(
        candidate,
        reference,
        candidate_energy=-9.0,
        reference_energy=-10.0,
    )

    assert metrics.rms_absolute_kj_mol_nm == pytest.approx(
        np.sqrt((0.2**2 + 0.1**2) / 6.0)
    )
    assert metrics.maximum_absolute_kj_mol_nm == pytest.approx(0.2)
    assert metrics.normalized_rms == pytest.approx(0.1)
    assert metrics.normalized_maximum == pytest.approx(0.1)
    assert metrics.energy_error_per_atom_kj_mol == pytest.approx(0.5)
    assert metrics.relative_energy_error == pytest.approx(0.1)


def test_charged_pme_force_metrics_fail_closed_for_zero_references():
    zeros = np.zeros((2, 3))

    with pytest.raises(ValueError, match="non-zero reference force"):
        force_error_metrics(
            zeros,
            zeros,
            candidate_energy=1.0,
            reference_energy=1.0,
        )
    with pytest.raises(ValueError, match="non-zero reference energy"):
        force_error_metrics(
            np.ones((2, 3)),
            np.ones((2, 3)),
            candidate_energy=0.0,
            reference_energy=0.0,
        )


def test_charged_pme_array_hash_tracks_dtype_shape_and_values():
    values = np.asarray([[1.0, 2.0]], dtype=np.float32)

    assert array_hash(values) == array_hash(values.copy())
    assert array_hash(values) != array_hash(values.astype(np.float64))
    assert array_hash(values) != array_hash(values.reshape(2, 1))


def test_charged_pme_manifest_matching_is_fail_closed():
    candidate = {
        "workload": {"atom_count": 4},
        "pme": {"mesh_shape": [16, 16, 16], "background_policy": "plasma"},
    }
    reference = {
        "workload": {"atom_count": 4},
        "pme": {"mesh_shape": [16, 16, 16], "background_policy": "plasma"},
    }
    fields = (
        "workload.atom_count",
        "pme.mesh_shape",
        "pme.background_policy",
    )

    require_matching_manifest(candidate, reference, fields=fields)
    mismatches = manifest_mismatches(
        candidate,
        {**reference, "pme": {**reference["pme"], "mesh_shape": [32, 16, 16]}},
        fields=fields,
    )

    assert list(mismatches) == ["pme.mesh_shape"]
    with pytest.raises(PMEManifestMismatchError, match="pme.mesh_shape") as exc_info:
        require_matching_manifest(
            candidate,
            {**reference, "pme": {"background_policy": "plasma"}},
            fields=fields,
        )
    assert exc_info.value.mismatches["pme.mesh_shape"]["reference_present"] is False


def test_charged_pme_manifest_hash_is_order_independent():
    first = {"pme": {"mesh": [8, 8, 8], "alpha": 0.35}, "atoms": 4}
    second = {"atoms": 4, "pme": {"alpha": 0.35, "mesh": [8, 8, 8]}}

    assert manifest_hash(first) == manifest_hash(second)
