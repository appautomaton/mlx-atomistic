from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks.pme_fixture import (
    PMEFixtureSpec,
    build_pme_fixture,
    fixture_summary,
    write_pme_fixture,
)
from mlx_atomistic.benchmarks.pme_validation import (
    deterministic_configurations,
    force_error_metrics,
    run_ewald_convergence,
)
from mlx_atomistic.prep.io import load_prepared_system


def test_pme_fixture_is_deterministic_and_neutral():
    spec = PMEFixtureSpec("test", bcc_cells_per_axis=2, ion_pairs=1, seed=17)

    first = build_pme_fixture(spec)
    second = build_pme_fixture(spec)

    assert first.metadata.selections["content_hash"] == second.metadata.selections[
        "content_hash"
    ]
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.charges, second.charges)
    np.testing.assert_array_equal(first.nonbonded_exception_pairs, first.constraints)
    assert float(np.sum(first.charges, dtype=np.float64)) == pytest.approx(0.0, abs=1e-6)
    assert first.metadata.pme_config["assignment_order"] == 5


def test_target_pme_fixture_has_approved_composition_and_clearance():
    prepared = build_pme_fixture("target")
    summary = fixture_summary(prepared)

    assert summary["site_count"] == 8192
    assert summary["water_count"] == 8148
    assert summary["sodium_count"] == 22
    assert summary["chloride_count"] == 22
    assert summary["atom_count"] == 24488
    assert 0.145 <= summary["ionic_strength_molar"] <= 0.155
    assert summary["net_charge_e"] == pytest.approx(0.0, abs=1e-5)
    assert summary["minimum_clearance_lower_bound_A"] >= 1.0
    assert prepared.water_mask.sum() == 3 * 8148
    assert prepared.ion_mask.sum() == 44
    assert prepared.constraints.shape == (3 * 8148, 2)
    assert prepared.nonbonded_pairs.shape == (0, 2)


def test_pme_fixture_rejects_clearance_it_cannot_guarantee():
    spec = PMEFixtureSpec("test", bcc_cells_per_axis=2, ion_pairs=0)

    with pytest.raises(ValueError, match="cannot guarantee"):
        build_pme_fixture(spec, minimum_clearance_angstrom=2.0)


def test_pme_fixture_prepared_round_trip(tmp_path: Path):
    spec = PMEFixtureSpec("test", bcc_cells_per_axis=2, ion_pairs=1, seed=19)

    summary = write_pme_fixture(spec, tmp_path)
    loaded = load_prepared_system(tmp_path)

    assert summary["content_hash"] == loaded.metadata.selections["content_hash"]
    assert (tmp_path / "pme_fixture.json").exists()
    assert loaded.metadata.parameter_source == "amber14_tip3p_joung_cheatham_ions"
    assert loaded.metadata.compatibility_report["electrostatics_model"] == "pme"
    assert loaded.pme_assignment_order.tolist() == [5]
    np.testing.assert_array_equal(loaded.positions, loaded.reference_positions)


def test_normalized_force_metrics_and_zero_reference_fail_closed():
    reference = np.asarray([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    candidate = reference + np.asarray([[0.2, 0.0, 0.0], [0.0, -0.1, 0.0]])

    metrics = force_error_metrics(
        candidate,
        reference,
        candidate_energy=-9.0,
        reference_energy=-10.0,
    )

    assert metrics.normalized_rms == pytest.approx(0.1)
    assert metrics.normalized_maximum == pytest.approx(0.1)
    assert metrics.energy_error_per_atom_kj_mol == pytest.approx(0.5)
    with pytest.raises(ValueError, match="non-zero reference"):
        force_error_metrics(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            candidate_energy=0.0,
            reference_energy=0.0,
        )


def test_pme_validation_configurations_are_deterministic_wrapped_and_rigid():
    prepared = build_pme_fixture(PMEFixtureSpec("test", 2, 1, seed=23))

    first = deterministic_configurations(prepared, count=3)
    second = deterministic_configurations(prepared, count=3)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.all(left >= 0.0)
        assert np.all(left < prepared.cell_lengths)
        displacement = left[1 : 3 * 14 : 3] - left[0 : 3 * 14 : 3]
        displacement -= prepared.cell_lengths * np.round(displacement / prepared.cell_lengths)
        distances = np.linalg.norm(displacement, axis=1)
        np.testing.assert_allclose(distances, 0.9572, atol=2e-5)


@pytest.mark.gpu
def test_ewald_convergence_payload_reports_finite_metrics():
    prepared = build_pme_fixture(PMEFixtureSpec("test", 2, 0, seed=29))
    prepared = replace(
        prepared,
        pme_real_cutoff=np.asarray([3.0], dtype=np.float32),
        pme_alpha=np.asarray([0.5], dtype=np.float32),
    )

    payload = run_ewald_convergence(
        prepared,
        reciprocal_cutoffs=(2, 3),
        configurations=1,
        convergence_tolerance=1.0,
    )

    assert payload["status"] == "passed"
    row = payload["rows"][0]
    assert row["finite"] is True
    assert row["converged"] is True
    assert np.isfinite(row["pme_vs_ewald"]["normalized_rms"])
