from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks.pme_fixture import (
    PMEFixtureSpec,
    build_pme_fixture,
    fixture_summary,
    write_pme_fixture,
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
