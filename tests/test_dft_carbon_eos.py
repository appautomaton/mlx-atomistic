from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_carbon import (
    CARBON_FRACTIONAL_POSITIONS,
    load_carbon_workload,
    prepare_carbon_workload,
)
from mlx_atomistic.benchmarks.dft_carbon_eos import (
    HARTREE_TO_EV,
    birch_murnaghan_energy,
    compare_fit_to_reference,
    fit_birch_murnaghan,
    fit_cubic_carbon_eos,
    load_carbon_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_carbon_eos_runner import (
    MEMORY_LIMIT_BYTES,
    POINT_TIMEOUT_SECONDS,
    PROFILE_SPECS,
    _final_report,
    _shape_comparison,
    run_carbon_eos_point,
    run_carbon_eos_validation,
)


def _gth_source(path):
    path.write_text(
        """C GTH-PBE-q4 GTH-PBE
2 2
0.33847124 2 -8.80367398 1.33921085
2
0.30257575 1 9.62248665
0.29150694 0
"""
    )
    return path


def test_carbon_reference_bundle_is_pinned_and_matches_acwf_protocol():
    references = load_carbon_eos_references()
    lattice = validation_lattice_constants(references)
    primary = reference_fit(references["references"]["all_electron_average"])

    assert lattice[3] == pytest.approx(3.571746218068632)
    assert primary["bulk_modulus_gpa"] == pytest.approx(433.4194999846373)
    np.testing.assert_allclose(
        (np.asarray(lattice) / lattice[3]) ** 3,
        [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_shared_birch_murnaghan_fit_recovers_published_carbon_cp2k_curve():
    references = load_carbon_eos_references()
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
        rel=2.0e-8,
    )
    assert fit["bulk_modulus_ev_angstrom3"] == pytest.approx(
        published["bulk_modulus_ev_angstrom3"],
        rel=1.0e-6,
    )
    assert compare_fit_to_reference(
        {**published, "status": "ok"},
        reference_fit(references["references"]["all_electron_average"]),
    )["verified"]


def test_carbon_workload_extracts_only_selected_entry_and_is_hash_guarded(tmp_path):
    from mlx_atomistic.dft import read_gth

    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_carbon_workload(gth_source=source, out=tmp_path / "workload")

    manifest, resource = load_carbon_workload(prepared["manifest"])
    pseudopotential = read_gth(resource, element="C", name="GTH-PBE-q4")

    assert resource.read_text().startswith("C GTH-PBE-q4 GTH-PBE\n")
    assert "Si " not in resource.read_text()
    assert [channel.angular_momentum for channel in pseudopotential.gth_channels] == [0]
    assert pseudopotential.gth_channels[0].projector_count == 1
    assert manifest["system"]["fractional_positions"] == [
        list(row) for row in CARBON_FRACTIONAL_POSITIONS
    ]
    assert manifest["system"]["electron_count"] == 32
    assert manifest["system"]["occupied_band_count"] == 16

    resource.write_text(resource.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_carbon_workload(prepared["manifest"])


def test_carbon_validation_dry_run_exposes_fail_early_bounds(tmp_path):
    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_carbon_workload(gth_source=source, out=tmp_path / "workload")

    plan = run_carbon_eos_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
        dry_run=True,
    )

    assert plan["status"] == "planned"
    assert plan["initial_point_count"] == 6
    assert plan["maximum_point_count_after_escalation"] == 16
    assert plan["memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert plan["point_timeout_seconds"] == POINT_TIMEOUT_SECONDS
    assert {point["profile"] for point in plan["initial_screen_points"]} == {
        "cutoff30",
        "cutoff40",
    }
    assert max(point["timeout_seconds"] for point in plan["initial_screen_points"]) <= 180
    assert set(PROFILE_SPECS) == {
        "cutoff30",
        "cutoff40",
        "cutoff50",
        "kpoint30",
        "kpoint40",
        "kpoint50",
    }

    report_path = tmp_path / "validation" / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "mlx-atomistic.carbon-eos-report.v1",
                "status": "passed",
                "admitted": True,
                "scientifically_verified": True,
            }
        )
    )
    summary = run_carbon_eos_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
        summarize_only=True,
    )
    assert summary["status"] == "passed"
    assert summary["evidence_status"] == "complete"


def test_central_shape_comparison_removes_irrelevant_energy_offset():
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
        rows("cutoff30", (-10.0, -10.1, -10.02)),
        rows("cutoff40", (-20.0, -20.1, -20.02)),
        atom_count=8,
    )

    assert comparison["passed"] is True
    assert comparison["metrics"]["curve_max_mev_per_atom"] < 1.0e-10


def test_single_carbon_point_persists_compact_numerical_evidence(tmp_path, monkeypatch):
    import mlx.core as mx

    import mlx_atomistic.dft as dft

    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_carbon_workload(gth_source=source, out=tmp_path / "workload")
    eigen = SimpleNamespace(
        orthonormality_error=2.0e-7,
        residuals=np.full(16, 8.0e-7),
    )
    result = SimpleNamespace(
        converged=True,
        total_energy=-45.0,
        electron_count=32.0,
        owned_kpoints=[SimpleNamespace(eigen=eigen)],
        kpoints=[SimpleNamespace(eigen=eigen), SimpleNamespace(eigen=eigen)],
        iterations=10,
        density_residual=2.0e-7,
        energy_delta=3.0e-7,
        timings={"total": 1.0},
        density=np.ones((4, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(dft, "PeriodicDFTSystem", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "MonkhorstPackGrid", lambda value: tuple(value))
    monkeypatch.setattr(dft, "read_gth", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "run_periodic_scf", lambda *args, **kwargs: result)
    monkeypatch.setattr(mx, "synchronize", lambda: None)

    payload = run_carbon_eos_point(
        manifest_path=prepared["manifest"],
        profile="cutoff30",
        volume_index=3,
        out=tmp_path / "point.json",
    )

    assert payload["status"] == "ok"
    assert payload["numerical_passed"] is True
    assert payload["result"]["maximum_orbital_residual"] == pytest.approx(8.0e-7)
    assert payload["result"]["representative_kpoint_count"] == 1
    assert "events" not in payload["result"]["observation"]
    assert (tmp_path / "density.npy").is_file()


def test_cubic_carbon_fit_uses_total_cell_hartree_and_eight_atoms():
    references = load_carbon_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    lattice = np.asarray(validation_lattice_constants(references))
    energies = birch_murnaghan_energy(
        lattice**3 / 8.0,
        -150.0,
        primary["equilibrium_volume_angstrom3_per_atom"],
        primary["bulk_modulus_ev_angstrom3"],
        primary["bulk_derivative"],
    )

    fit = fit_cubic_carbon_eos(lattice, energies * 8.0 / HARTREE_TO_EV)

    assert fit["status"] == "ok"
    assert fit["atom_count"] == 8
    assert fit["equilibrium_lattice_constant_angstrom"] == pytest.approx(
        primary["equilibrium_lattice_constant_angstrom"],
        rel=1.0e-8,
    )


def test_report_accepts_central_cutoff_evidence_without_full_upper_curve():
    references = load_carbon_eos_references()
    candidate = {
        **reference_fit(references["references"]["all_electron_average"]),
        "status": "ok",
    }
    passed_shape = {
        "status": "ok",
        "passed": True,
        "metrics": {"curve_max_mev_per_atom": 0.4},
    }

    report = _final_report(
        manifest={"workload_fingerprint": "workload"},
        rows=[],
        cutoff_pair=("cutoff40", "cutoff50"),
        selected_profile="cutoff40",
        cutoff_shape=passed_shape,
        cutoff_convergence={
            "status": "ok",
            "passed": True,
            "full_upper_cutoff_curve_required": False,
        },
        selected_fit=candidate,
        kpoint_comparison=passed_shape,
    )

    assert report["status"] == "passed"
    assert report["scientifically_verified"] is True
    assert report["admitted"] is True
    assert report["selected_cutoff_profile"] == "cutoff40"
    assert report["accepted_workload"]["full_upper_cutoff_curve_required"] is False
