from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_silicon import (
    GTH_NAME,
    SOURCE_SCHEMA,
    TARGET_ID,
    WORKLOAD_SCHEMA,
    _parse_pmset_power_mode,
    finite_difference_force_array,
    fit_lattice_curve,
    inspect_workload,
    isotropic_stress_tensor,
    main,
    parse_gth_entry,
    prepare_workload,
    render_qe_gth,
    run_mlx_smoke,
    run_mlx_workload,
)


def _gth_database(path: Path) -> Path:
    path.write_text(
        """# compact fixture
H GTH-PBE-q1 GTH-PBE
    1
    0.2 1 -1.0
    0
#
Si GTH-PBE-q4 GTH-PBE
    2 2
    0.44000000 1 -6.26928833
    2
    0.43563383 2 8.95174150 -2.70627082
                   3.49378060
    0.49794218 1 2.43127673
#
P GTH-PBE-q5 GTH-PBE
    2 3
    0.43 1 -5.8
    0
"""
    )
    return path


def test_parse_gth_entry_preserves_full_channel_matrices(tmp_path):
    entry = parse_gth_entry(_gth_database(tmp_path / "GTH_POTENTIALS"))

    assert entry.element == "Si"
    assert GTH_NAME in entry.names
    assert entry.valence_charge == 4.0
    assert entry.local_coefficients == (-6.26928833,)
    assert len(entry.channels) == 2
    assert entry.channels[0].coupling_matrix == (
        (8.9517415, -2.70627082),
        (-2.70627082, 3.4937806),
    )
    assert entry.channels[1].coupling_matrix == ((2.43127673,),)


def test_parse_gth_entry_rejects_incomplete_triangular_matrix(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    source.write_text(source.read_text().replace("                   3.49378060\n", ""))

    with pytest.raises(ValueError, match="coupling"):
        parse_gth_entry(source)


def test_render_qe_gth_is_deterministic_and_includes_zero_spin_orbit(tmp_path):
    entry = parse_gth_entry(_gth_database(tmp_path / "GTH_POTENTIALS"))
    rendered = render_qe_gth(entry)

    assert rendered.startswith("Goedecker pseudopotential for Si\n14 4")
    assert "10 11 1 2" in rendered
    assert "0.43563383 2 8.9517415 -2.70627082" in rendered
    assert rendered.endswith("2.43127673\n0\n")
    assert render_qe_gth(entry) == rendered


def test_prepare_workload_writes_strict_idempotent_manifests(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"
    command = ["uv", "run", "python", "-m", "mlx_atomistic.benchmarks.dft_silicon"]

    first = prepare_workload(gth_source=source, out=out, command=command)
    second = prepare_workload(gth_source=source, out=out, command=command)

    assert first == second
    source_manifest = json.loads(Path(first["source_manifest"]).read_text())
    workload = json.loads(Path(first["workload_manifest"]).read_text())
    assert source_manifest["schema_version"] == SOURCE_SCHEMA
    assert source_manifest["target_id"] == TARGET_ID
    assert source_manifest["extracted_sha256"] == first["gth_sha256"]
    assert workload["schema_version"] == WORKLOAD_SCHEMA
    assert workload["target_id"] == TARGET_ID
    assert workload["system"]["atom_count"] == 8
    assert workload["system"]["electron_count"] == 32.0
    assert workload["target_host"]["chip"] == "Apple M5 Max"
    assert workload["target_host"]["pmset_expected_lowpowermode"] == 1
    assert sorted(workload["cases"]) == [
        "displaced_atom",
        "equilibrium",
        "strain_minus",
        "strain_plus",
        "volume_scan",
    ]
    assert inspect_workload(first["workload_manifest"])["status"] == "ready"


def test_prepare_workload_refuses_mismatched_existing_output(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"
    prepared = prepare_workload(gth_source=source, out=out, command=["prepare"])
    Path(prepared["gth_path"]).write_text("tampered\n")

    with pytest.raises(ValueError, match="refusing to replace mismatched"):
        prepare_workload(gth_source=source, out=out, command=["prepare"])


def test_inspect_workload_rejects_manifest_and_gth_tampering(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "prepared", command=["prepare"])
    manifest_path = Path(prepared["workload_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["system"]["electron_count"] = 31.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fingerprint"):
        inspect_workload(manifest_path)


def test_dft_silicon_prepare_and_inspect_cli_json(tmp_path, capsys):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"

    main(["prepare", "--gth-database", str(source), "--out", str(out), "--json"])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["status"] == "prepared"

    main(["inspect", "--manifest", prepared["workload_manifest"], "--json"])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "ready"
    assert inspected["target_chip"] == "Apple M5 Max"


def test_dft_silicon_mlx_smoke_uses_full_periodic_path(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(
        gth_source=source,
        out=tmp_path / "prepared",
        command=["prepare"],
    )

    payload = run_mlx_smoke(
        manifest_path=prepared["workload_manifest"],
        out=tmp_path / "smoke",
    )

    assert payload["converged"]
    assert payload["status"] == "converged"
    assert payload["electron_count"] == pytest.approx(32.0, abs=2e-4)
    assert payload["dense_full_hamiltonian"] is False
    report = json.loads(Path(payload["report"]).read_text())
    assert report["result"]["dense_full_hamiltonian"] is False
    assert report["result"]["kpoints"][0]["eigensolver"]["converged"]


def test_finite_difference_force_array_matches_quadratic_gradient():
    positions = np.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])

    def energy(values):
        return 0.5 * float(np.sum(values * values))

    forces = finite_difference_force_array(energy, positions, displacement_bohr=1e-4)

    np.testing.assert_allclose(forces, -positions, atol=1e-10)


def test_isotropic_stress_tensor_has_cubic_symmetry_and_expected_derivative():
    lattice = 5.0

    def energy(length):
        return 2.0 * length**3

    stress = isotropic_stress_tensor(energy, lattice, strain_step=1e-4)
    expected = 2.0 * 29421.02648438959

    np.testing.assert_allclose(np.diag(stress), expected, rtol=2e-8)
    np.testing.assert_allclose(stress - np.diag(np.diag(stress)), 0.0, atol=0.0)


def test_lattice_fit_requires_seven_points_and_interior_convex_minimum():
    lattice = np.linspace(5.25, 5.61, 7)
    energies = -10.0 + 3.0 * (lattice - 5.43) ** 2

    fit = fit_lattice_curve(lattice, energies)

    assert fit["status"] == "ok"
    assert fit["equilibrium_lattice_constant_angstrom"] == pytest.approx(5.43, abs=1e-10)
    with pytest.raises(ValueError, match="seven"):
        fit_lattice_curve(lattice[:-1], energies[:-1])


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Currently in use:\n lowpowermode 1\n", ("lowpowermode", 1)),
        ("Currently in use:\n powermode 0\n", ("powermode", 0)),
        ("Currently in use:\n sleep 0\n", (None, None)),
    ],
)
def test_parse_pmset_power_mode_supports_old_and_new_macos_keys(payload, expected):
    assert _parse_pmset_power_mode(payload) == expected


def test_dft_silicon_mlx_equilibrium_persists_reproducibility_and_profile(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(
        gth_source=source,
        out=tmp_path / "prepared",
        command=["prepare"],
    )

    payload = run_mlx_workload(
        manifest_path=prepared["workload_manifest"],
        out=tmp_path / "mlx",
        case="equilibrium",
        repetitions=2,
    )

    assert payload["status"] == "ok"
    report = json.loads(Path(payload["report"]).read_text())
    assert report["cases"]["equilibrium"]["rerun_energy_delta_hartree_per_atom"] <= 1e-6
    assert len(report["cases"]["equilibrium"]["repetitions"]) == 2
    assert report["internal_gates"]["energy_accounting_consistent"] is True
    assert report["run_protocol"]["synchronization"].startswith("mx.eval")
    assert report["profile_rows"]
    for row in report["cases"]["equilibrium"]["repetitions"]:
        arrays = np.load(row["arrays"], allow_pickle=False)
        assert arrays["density"].shape == (8, 8, 8)


def test_dft_silicon_paired_continuation_persists_owner_arrays_only(
    tmp_path,
    monkeypatch,
):
    import mlx_atomistic.benchmarks.dft_silicon as silicon

    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(
        gth_source=source,
        out=tmp_path / "prepared",
        command=["prepare"],
    )
    original_settings = silicon._periodic_settings

    def paired_settings(profile):
        settings = original_settings(profile)
        return {**settings, "kpoint_mesh": (2, 1, 1)}

    monkeypatch.setattr(silicon, "_periodic_settings", paired_settings)
    payload = run_mlx_workload(
        manifest_path=prepared["workload_manifest"],
        out=tmp_path / "mlx-paired",
        case="strain_minus",
        repetitions=2,
    )

    report = json.loads(Path(payload["report"]).read_text())
    case = report["cases"]["strain_minus"]
    rows = [case["base"], *case["branches"]]
    assert len(rows) > 1
    for row in rows:
        assert row["explicit_kpoint_count"] == 2
        assert row["owned_kpoint_indices"] == [0]
        arrays = np.load(row["arrays"], allow_pickle=False)
        assert "coefficients_0" in arrays
        assert "eigenvalues_0" in arrays
        assert "residuals_0" in arrays
        assert "coefficients_1" not in arrays
        assert "eigenvalues_1" not in arrays
        assert "residuals_1" not in arrays
