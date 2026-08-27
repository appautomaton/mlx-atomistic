from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_mgo import (
    MGO_FRACTIONAL_POSITIONS,
    MGO_PRIMITIVE_CELL_MATRIX,
    MGO_PRIMITIVE_FRACTIONAL_POSITIONS,
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
    _kpoint_mesh,
    _point_group_energy_oracle,
    _shape_comparison,
    _system_geometry,
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
    assert manifest["primitive_system"]["atom_count"] == 2
    assert manifest["primitive_system"]["q10_electron_count"] == 16
    assert manifest["primitive_system"]["q10_occupied_band_count"] == 8
    assert manifest["primitive_system"]["fractional_cell_matrix"] == [
        list(row) for row in MGO_PRIMITIVE_CELL_MATRIX
    ]

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
    assert plan["maximum_point_count"] == 36
    assert plan["initial_smoke_point"]["profile"] == "smoke-q2"
    assert len(plan["initial_smoke_point"]["runtime_fingerprint"]) == 64
    assert [point["cutoff_hartree"] for point in plan["cutoff_screen_points"]] == [
        25.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
    ]
    assert len(PROFILE_SPECS) == 25


def test_mgo_q10_primitive_profiles_preserve_geometry_and_oracle_pair():
    lattice = 8.0
    system = {
        "cell_representation": "primitive",
        "fractional_cell_matrix": [list(row) for row in MGO_PRIMITIVE_CELL_MATRIX],
        "fractional_positions": [
            list(row) for row in MGO_PRIMITIVE_FRACTIONAL_POSITIONS
        ],
    }
    cell, positions = _system_geometry(system, lattice)
    reduced_settings = PROFILE_SPECS["q10-primitive-c40-k4"]
    full_settings = PROFILE_SPECS["q10-primitive-c40-k4-full"]
    reduced = _kpoint_mesh(reduced_settings, cell)
    full = _kpoint_mesh(full_settings, cell)

    assert np.linalg.det(cell) == pytest.approx(lattice**3 / 4.0)
    np.testing.assert_allclose(positions[1], (lattice / 2.0,) * 3)
    assert reduced_settings["symmetry_reduction"] == "full_cubic_point_group"
    assert full_settings.get("symmetry_reduction") is None
    assert reduced.point_group_symmetry_reduced is True
    assert full.point_group_symmetry_reduced is False
    assert len(reduced.points) == 8
    assert len(full.points) == 64
    assert sum(point.weight for point in reduced.points) == pytest.approx(1.0)
    assert sum(point.weight for point in full.points) == pytest.approx(1.0)


def test_mgo_conventional_point_group_profile_changes_only_symmetry():
    full = PROFILE_SPECS["q2-c70-k6"]
    reduced = PROFILE_SPECS["q2-c70-k6-point-group"]

    assert reduced == {
        **full,
        "symmetry_reduction": "full_cubic_point_group",
    }
    mesh = _kpoint_mesh(
        {**reduced, "kpoint_mesh": [2, 2, 2]},
        (8.0, 8.0, 8.0),
    )
    assert mesh.point_group_symmetry_reduced is True
    assert len(mesh.points) < 2**3
    assert sum(point.weight for point in mesh.points) == pytest.approx(1.0)


def test_mgo_point_group_oracle_uses_the_established_energy_gate():
    full = {
        "numerical_passed": True,
        "result": {"total_energy_hartree": -68.0},
    }
    passing = {
        "numerical_passed": True,
        "result": {"total_energy_hartree": -68.0 + 8 * 4.0e-5},
    }
    failing = {
        "numerical_passed": True,
        "result": {"total_energy_hartree": -68.0 + 8 * 6.0e-5},
    }

    assert _point_group_energy_oracle(full, passing, atom_count=8)["passed"] is True
    assert _point_group_energy_oracle(full, failing, atom_count=8)["passed"] is False


def test_mgo_point_group_oracle_fails_closed():
    valid = {
        "numerical_passed": True,
        "result": {"total_energy_hartree": -68.0},
    }
    numerical_failure = {
        "numerical_passed": False,
        "result": {"total_energy_hartree": -68.0},
    }
    nonfinite = {
        "numerical_passed": True,
        "result": {"total_energy_hartree": np.nan},
    }

    blocked = _point_group_energy_oracle(valid, numerical_failure, atom_count=8)
    assert blocked["status"] == "blocked"
    assert blocked["blocker"] == "point_group_oracle_numerical_failure"
    assert blocked["passed"] is False

    blocked = _point_group_energy_oracle(valid, nonfinite, atom_count=8)
    assert blocked["status"] == "blocked"
    assert blocked["blocker"] == "point_group_oracle_energy_not_finite"
    assert blocked["passed"] is False

    with pytest.raises(ValueError, match="atom_count must be positive"):
        _point_group_energy_oracle(valid, valid, atom_count=0)


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
