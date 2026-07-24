from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_runtime_contract import prepare_workload
from mlx_atomistic.benchmarks.dft_silicon_eos import (
    HARTREE_TO_EV,
    birch_murnaghan_energy,
    compare_eos_convergence,
    compare_fit_to_reference,
    delta_factor_mev_per_atom,
    fit_birch_murnaghan,
    fit_cubic_silicon_eos,
    load_silicon_eos_references,
    reference_fit,
    validation_lattice_constants,
)
from mlx_atomistic.benchmarks.dft_silicon_eos_runner import (
    _LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS,
    PROFILE_SPECS,
    _load_matching_point,
    _point_spec,
    build_silicon_eos_report,
    run_silicon_eos_point,
    run_silicon_eos_validation,
)


def _gth_source(path: Path) -> Path:
    path.write_text(
        """Si GTH-PBE-q4 GTH-PBE
2 2
0.44000000 1 -6.26928833
2
0.43563383 2 8.95174150 -2.70627082
3.49378060
0.49794218 1 2.43127673
"""
    )
    return path


def test_reference_bundle_is_pinned_and_generates_exact_protocol_lattices():
    references = load_silicon_eos_references()
    lattice = validation_lattice_constants(references)

    assert references["references"]["all_electron_average"]["role"] == "primary"
    assert len(lattice) == 7
    assert lattice[3] == pytest.approx(5.470205139257224)
    np.testing.assert_allclose(
        (np.asarray(lattice) / lattice[3]) ** 3,
        [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
        rtol=0.0,
        atol=2e-15,
    )


def test_birch_murnaghan_fit_recovers_published_cp2k_curve():
    references = load_silicon_eos_references()
    cp2k = references["references"]["cp2k_gth"]
    rows = cp2k["eos_volume_energy_ev"]

    fit = fit_birch_murnaghan(
        [volume / 2.0 for volume, _energy in rows],
        [energy / 2.0 for _volume, energy in rows],
    )
    published = reference_fit(cp2k)

    assert fit["status"] == "ok"
    assert fit["equilibrium_volume_angstrom3_per_atom"] == pytest.approx(
        published["equilibrium_volume_angstrom3_per_atom"], rel=2e-8
    )
    assert fit["bulk_modulus_ev_angstrom3"] == pytest.approx(
        published["bulk_modulus_ev_angstrom3"], rel=1e-6
    )
    assert fit["bulk_derivative"] == pytest.approx(published["bulk_derivative"], rel=2e-6)
    assert fit["rmse_mev_per_atom"] < 0.01


def test_same_gth_family_reference_meets_verified_but_not_excellent_gate():
    references = load_silicon_eos_references()
    candidate = {
        **reference_fit(references["references"]["cp2k_gth"]),
        "status": "ok",
    }
    primary = reference_fit(references["references"]["all_electron_average"])

    comparison = compare_fit_to_reference(candidate, primary)

    assert delta_factor_mev_per_atom(candidate, primary) == pytest.approx(2.3169867, rel=1e-6)
    assert comparison["verified"] is True
    assert comparison["excellent"] is False


def test_cubic_fit_and_convergence_are_invariant_to_total_energy_offset():
    references = load_silicon_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    lattice = np.asarray(validation_lattice_constants(references))
    volumes = lattice**3 / 8.0
    energy = birch_murnaghan_energy(
        volumes,
        -100.0,
        primary["equilibrium_volume_angstrom3_per_atom"],
        primary["bulk_modulus_ev_angstrom3"],
        primary["bulk_derivative"],
    )

    first = fit_cubic_silicon_eos(lattice, energy * 8.0 / HARTREE_TO_EV)
    second = fit_cubic_silicon_eos(
        lattice,
        (energy + 20.0) * 8.0 / HARTREE_TO_EV,
    )

    assert first["status"] == "ok"
    assert compare_eos_convergence(first, second)["passed"] is True


def test_admission_report_requires_science_cutoff_convergence_and_kpoint_spot_check():
    references = load_silicon_eos_references()
    primary = reference_fit(references["references"]["all_electron_average"])
    lattice = validation_lattice_constants(references)
    rows = []
    for profile in ("baseline", "cutoff"):
        for index, length in enumerate(lattice):
            volume = length**3 / 8.0
            energy = float(
                birch_murnaghan_energy(
                    volume,
                    -100.0,
                    primary["equilibrium_volume_angstrom3_per_atom"],
                    primary["bulk_modulus_ev_angstrom3"],
                    primary["bulk_derivative"],
                )
            )
            rows.append(
                {
                    "numerical_passed": True,
                    "point": {
                        "profile": profile,
                        "volume_index": index,
                        "lattice_constant_angstrom": length,
                        "point_fingerprint": f"{profile}-{index}",
                    },
                    "result": {"total_energy_hartree": energy * 8 / HARTREE_TO_EV},
                }
            )
    for index in (2, 3, 4):
        source = next(
            row
            for row in rows
            if row["point"]["profile"] == "baseline" and row["point"]["volume_index"] == index
        )
        rows.append(
            {
                **source,
                "point": {
                    **source["point"],
                    "profile": "kpoint",
                    "point_fingerprint": f"kpoint-{index}",
                },
            }
        )
    manifest = {
        "workload_fingerprint": "test-workload",
        "system": {"atom_count": 8},
    }

    report = build_silicon_eos_report(
        manifest=manifest,
        level="admission",
        point_reports=rows,
    )

    assert report["status"] == "passed"
    assert report["admitted"] is True
    assert report["scientific_comparison"]["verified"] is True
    assert report["numerical_convergence"]["cutoff"]["passed"] is True
    assert report["numerical_convergence"]["kpoint_spot_check"]["passed"] is True
    assert report["profiles"]["kpoint"]["point_count"] == 3
    assert report["profiles"]["kpoint"]["fit"] is None
    assert report["profiles"]["combined"]["point_count"] == 0


def test_validation_dry_run_is_bounded_and_does_not_execute_scf(tmp_path):
    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")

    screen = run_silicon_eos_validation(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "screen",
        level="screen",
        dry_run=True,
    )
    admission = run_silicon_eos_validation(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "admission",
        level="admission",
        dry_run=True,
    )
    combined = run_silicon_eos_validation(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "combined",
        level="admission",
        dry_run=True,
        include_combined=True,
    )
    summary = run_silicon_eos_validation(
        manifest_path=prepared["manifest"],
        gth_source=source,
        out=tmp_path / "summary",
        level="admission",
        summarize_only=True,
    )

    assert screen["status"] == "planned"
    assert screen["point_count"] == 3
    assert admission["point_count"] == 17
    assert admission["include_combined"] is False
    assert combined["point_count"] == 20
    assert combined["include_combined"] is True
    assert summary["evidence_status"] == "partial"
    assert summary["completed_point_count"] == 0
    assert len(summary["missing_points"]) == 17
    assert [
        point["volume_index"] for point in admission["points"] if point["profile"] == "kpoint"
    ] == [2, 3, 4]
    assert set(PROFILE_SPECS) == {"baseline", "cutoff", "kpoint", "combined"}
    assert all(point["timeout_seconds"] <= 240 for point in admission["points"])
    assert PROFILE_SPECS["baseline"]["max_batch_transient_bytes"] == 512 * 1024**2
    assert PROFILE_SPECS["cutoff"]["max_batch_transient_bytes"] == 768 * 1024**2
    assert PROFILE_SPECS["combined"]["max_batch_transient_bytes"] == 768 * 1024**2


def test_orchestration_change_preserves_exact_legacy_point_science(tmp_path):
    values = {
        "workload_fingerprint": "workload",
        "runtime_fingerprint": "runtime",
        "profile": "baseline",
        "volume_index": 3,
        "lattice_angstrom": 5.470205139257224,
    }
    expected = _point_spec(**values)
    legacy_implementation = next(iter(_LEGACY_POINT_IMPLEMENTATION_FINGERPRINTS))
    legacy = _point_spec(
        **values,
        implementation_fingerprint=legacy_implementation,
    )
    path = tmp_path / "point.json"
    path.write_text(
        json.dumps({"schema_version": "mlx-atomistic.silicon-eos-point.v1", "point": legacy})
    )

    loaded = _load_matching_point(path, expected)

    assert loaded is not None
    assert loaded["point"]["point_fingerprint"] == legacy["point_fingerprint"]
    mismatched = {**expected, "cutoff_hartree": 26.0}
    with pytest.raises(ValueError, match="mismatched"):
        _load_matching_point(path, mismatched)


def test_single_point_persists_compact_numerical_evidence(
    tmp_path,
    monkeypatch,
):
    import mlx.core as mx

    import mlx_atomistic.dft as dft

    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "workload")
    eigen = SimpleNamespace(
        orthonormality_error=2.0e-7,
        residuals=np.full(16, 8.0e-7),
    )
    result = SimpleNamespace(
        converged=True,
        total_energy=-31.0,
        electron_count=32.0,
        owned_kpoints=[SimpleNamespace(eigen=eigen)],
        kpoints=[SimpleNamespace(eigen=eigen), SimpleNamespace(eigen=eigen)],
        iterations=12,
        density_residual=2.0e-7,
        energy_delta=3.0e-7,
        timings={"total": 1.0},
    )
    monkeypatch.setattr(dft, "PeriodicDFTSystem", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "MonkhorstPackGrid", lambda value: tuple(value))
    monkeypatch.setattr(dft, "read_gth", lambda *args, **kwargs: object())
    monkeypatch.setattr(dft, "run_periodic_scf", lambda *args, **kwargs: result)
    monkeypatch.setattr(mx, "synchronize", lambda: None)

    payload = run_silicon_eos_point(
        manifest_path=prepared["manifest"],
        gth_source=source,
        profile="baseline",
        volume_index=3,
        out=tmp_path / "point.json",
    )

    assert payload["status"] == "ok"
    assert payload["numerical_passed"] is True
    assert payload["result"]["maximum_orbital_residual"] == pytest.approx(8.0e-7)
    assert payload["result"]["representative_kpoint_count"] == 1
    assert "events" not in payload["result"]["observation"]
    assert (tmp_path / "point.json").is_file()
