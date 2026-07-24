from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_mgo import (
    MGO_FRACTIONAL_POSITIONS,
    MGO_SYMBOLS,
    load_mgo_workload,
    prepare_mgo_workload,
)
from mlx_atomistic.benchmarks.dft_mgo_eos import (
    fit_birch_murnaghan,
    load_mgo_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_mgo_eos_runner import (
    MEMORY_LIMIT_BYTES,
    POINT_TIMEOUT_SECONDS,
    PROFILE_SPECS,
    _completion_assessment,
    _shape_comparison,
    run_mgo_eos_validation,
)


def _gth_source(path):
    path.write_text(
        """Mg GTH-PBE-q10 GTH-PBE
4 6
0.19275787 2 -20.57539077 3.04016732
2
0.14140682 1 41.04729209
0.10293187 1 -9.98562566
#
Mg GTH-PBE-q2
2
0.57696017 1 -2.69040744
2
0.59392350 2 3.50321099 -0.71677167
0.92534825
0.70715728 1 0.83115848
#
O GTH-PBE-q6 GTH-PBE
2 4
0.24455430 2 -16.66721480 2.48731132
2
0.22095592 1 18.33745811
0.21133247 0
"""
    )
    return path


def test_mgo_reference_bundle_is_pinned_to_acwf_oxide_protocol():
    references = load_mgo_eos_references()
    lattice = validation_lattice_constants(references)
    primary = reference_fit(references["references"]["all_electron_average"])

    assert lattice[3] == pytest.approx(4.254250040100746)
    assert primary["bulk_modulus_gpa"] == pytest.approx(148.9809824)
    np.testing.assert_allclose(
        (np.asarray(lattice) / lattice[3]) ** 3,
        [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
        rtol=0.0,
        atol=2e-15,
    )


def test_shared_eos_fit_recovers_published_mgo_cp2k_curve():
    references = load_mgo_eos_references()
    cp2k = references["references"]["cp2k_gth"]
    rows = cp2k["eos_volume_energy_ev"]

    fit = fit_birch_murnaghan(
        [volume / 2.0 for volume, _energy in rows],
        [energy / 2.0 for _volume, energy in rows],
    )
    published = reference_fit(cp2k)

    assert fit["status"] == "ok"
    assert fit["equilibrium_volume_angstrom3_per_atom"] == pytest.approx(
        published["equilibrium_volume_angstrom3_per_atom"],
        rel=2e-7,
    )
    assert fit["bulk_modulus_ev_angstrom3"] == pytest.approx(
        published["bulk_modulus_ev_angstrom3"],
        rel=3e-6,
    )


def test_mgo_workload_extracts_all_species_and_is_hash_guarded(tmp_path):
    from mlx_atomistic.dft import read_gth

    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_mgo_workload(gth_source=source, out=tmp_path / "workload")
    manifest, resources = load_mgo_workload(prepared["manifest"])

    mg_q2 = read_gth(resources["mg_q2"], element="Mg", name="GTH-PBE-q2")
    mg_q10 = read_gth(resources["mg_q10"], element="Mg", name="GTH-PBE-q10")
    oxygen = read_gth(resources["o_q6"], element="O", name="GTH-PBE-q6")

    assert (mg_q2.valence_charge, mg_q10.valence_charge, oxygen.valence_charge) == (
        2.0,
        10.0,
        6.0,
    )
    assert manifest["system"]["symbols"] == list(MGO_SYMBOLS)
    assert manifest["system"]["fractional_positions"] == [
        list(row) for row in MGO_FRACTIONAL_POSITIONS
    ]
    assert manifest["system"]["q2_electron_count"] == 32
    assert manifest["system"]["q2_occupied_band_count"] == 16
    assert manifest["system"]["q10_electron_count"] == 64
    assert manifest["system"]["q10_occupied_band_count"] == 32

    resources["o_q6"].write_text(resources["o_q6"].read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mgo_workload(prepared["manifest"])


def test_mgo_validation_dry_run_exposes_bounded_decision_ladder(tmp_path):
    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_mgo_workload(gth_source=source, out=tmp_path / "workload")

    plan = run_mgo_eos_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
        dry_run=True,
    )

    assert plan["status"] == "planned"
    assert plan["memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert plan["point_timeout_seconds"] == POINT_TIMEOUT_SECONDS
    assert plan["maximum_point_count"] == 35
    assert plan["initial_smoke_point"]["profile"] == "smoke-q2"
    assert [point["cutoff_hartree"] for point in plan["cutoff_screen_points"]] == [
        25.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
    ]
    assert len(PROFILE_SPECS) == 22


def test_mgo_kpoint_shape_comparison_removes_total_energy_offset():
    def rows(profile, values):
        return [
            {
                "numerical_passed": True,
                "point": {"profile": profile, "volume_index": index},
                "result": {"total_energy_hartree": value},
            }
            for index, value in zip((2, 3, 4), values, strict=True)
        ]

    comparison = _shape_comparison(
        rows("q2-c40-k4", (-10.0, -10.1, -10.02)),
        rows("q2-c40-k6", (-20.0, -20.1, -20.02)),
    )

    assert comparison["passed"] is True
    assert comparison["metrics"]["curve_max_mev_per_atom"] < 1e-10


def test_mgo_completion_records_bprime_deviation_without_weakening_gate():
    fit = {
        "status": "ok",
        "bulk_derivative": 3.3999463985004303,
    }
    scientific = {
        "verified": False,
        "metrics": {
            "delta_mev_per_atom": 1.06037963128625,
            "lattice_relative": 0.0012346761562464165,
            "bulk_modulus_relative": 0.013872114100056362,
            "bulk_derivative_relative": 0.16890536631389652,
        },
        "verified_thresholds": {
            "delta_mev_per_atom": 3.0,
            "lattice_relative": 0.005,
            "bulk_modulus_relative": 0.1,
            "bulk_derivative_relative": 0.15,
        },
    }

    completion = _completion_assessment(fit, scientific)

    assert completion["status"] == "complete_with_known_deviation"
    assert completion["validation_complete"] is True
    assert completion["core_properties_validated"] is True
    assert completion["strict_reference_gate_passed"] is False
    assert completion["scientifically_verified"] is False
    assert completion["admitted"] is False
    assert completion["blockers"] == []
    assert completion["failed_strict_metrics"] == [
        "bulk_derivative_relative"
    ]
    deviation = completion["known_residual_deviations"][0]
    assert deviation["relative_error"] == pytest.approx(0.16890536631389652)
    assert deviation["strict_threshold"] == pytest.approx(0.15)
