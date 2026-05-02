import json

from mlx_atomistic.benchmarks import (
    dft_geometry,
    dft_nonlocal,
    dft_operator,
    dft_pseudopotential,
    dft_relaxation,
    dft_scf,
    dft_solver,
    dft_spin_kpoints,
    ewald_reference,
    lj_md,
    mm_force_terms,
    stability,
    validation_gauntlet,
)


def test_validation_gauntlet_cli_json_and_csv(tmp_path, capsys):
    csv_path = tmp_path / "validation.csv"

    validation_gauntlet.main(
        [
            "--cases-per-term",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total_cases"] == 6
    assert payload["summary"]["all_passed"]
    assert len(payload["cases"]) == 6
    assert csv_path.read_text().startswith("case_name,term_name")


def test_stability_cli_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "stability.csv"

    stability.main(
        [
            "--sizes",
            "16",
            "--steps",
            "2",
            "--bonded-steps",
            "2",
            "--dt-values",
            "0.001",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["case_count"] == 3
    assert payload["summary"]["nonfinite_cases"] == 0
    assert {case["ensemble"] for case in payload["cases"]} == {"nve", "nvt"}
    assert csv_path.read_text().startswith("case,ensemble")


def test_lj_benchmark_csv_smoke(tmp_path):
    csv_path = tmp_path / "lj.csv"

    lj_md.main(["--particles", "16", "--steps", "1", "--csv", str(csv_path)])

    text = csv_path.read_text()
    assert text.startswith("mode,particles")
    assert "all-pairs" in text
    assert "nvt-dynamic-neighbor" in text


def test_force_term_benchmark_includes_profile_rows():
    results = mm_force_terms.run_benchmark(evaluations=1, particles=16)

    categories = {result.category for result in results}
    assert "bonded-autodiff" in categories
    assert "neighbor-list" in categories
    assert "lj-pair-eval" in categories
    assert "coulomb-direct" in categories
    assert "combined-nonbonded" in categories
    assert "constraints" in categories


def test_ewald_reference_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "ewald.csv"

    ewald_reference.main(
        [
            "--atoms",
            "4",
            "--evaluations",
            "1",
            "--reciprocal-cutoff",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 1
    assert "correctness backend" in payload["scope_note"]
    assert "not GPCRmd-scale PME" in payload["scope_note"]
    row = payload["cases"][0]
    assert row["atoms"] == 4
    assert row["evaluations"] == 1
    assert row["k_vector_count"] == 26
    assert row["real_shift_count"] == 125
    assert row["finite"]
    assert "coulomb_real" in row
    assert csv_path.read_text().startswith("case,atoms")


def test_dft_scf_benchmark_json_smoke(capsys):
    dft_scf.main(["--grid", "4,4,4", "--iterations", "2", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_shape"] == [4, 4, 4]
    assert payload["iterations_requested"] == 2
    assert payload["iterations_completed"] == 2
    assert payload["solver"] == "dense"
    assert payload["fft_backend"] in {"mlx", "numpy"}
    assert "runtime" in payload
    assert "energy_by_term" in payload
    assert "timings" in payload


def test_dft_scf_benchmark_csv_and_mixer_matrix(tmp_path, capsys):
    csv_path = tmp_path / "dft.csv"

    dft_scf.main(
        [
            "--sizes",
            "4",
            "--iterations",
            "1",
            "--mixer",
            "both",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 2
    assert {case["mixer"] for case in payload["cases"]} == {"linear", "diis"}
    assert "fft_probe_ms" in payload["cases"][0]
    assert csv_path.read_text().startswith("grid_shape,grid_points")


def test_dft_operator_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_operator.csv"

    dft_operator.main(
        [
            "--grid",
            "2,2,2",
            "--iterations",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_shape"] == [2, 2, 2]
    assert payload["case_count"] == 1
    assert payload["dense_vs_operator_max_error"] < 1e-5
    assert "operator_apply_ms" in payload
    assert csv_path.read_text().startswith("grid_shape,grid_points")


def test_dft_pseudopotential_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_pseudo.csv"

    dft_pseudopotential.main(
        [
            "--grid",
            "2,2,2",
            "--iterations",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_shape"] == [2, 2, 2]
    assert payload["case_count"] == 3
    assert {case["case"] for case in payload["cases"]} == {
        "gaussian",
        "gth-local",
        "upf-local",
    }
    assert csv_path.read_text().startswith("case,grid_shape")


def test_dft_geometry_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_geometry.csv"

    dft_geometry.main(
        [
            "--grid",
            "4,4,4",
            "--steps",
            "1",
            "--systems",
            "gaussian-dimer,gth-h2",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_shape"] == [4, 4, 4]
    assert payload["case_count"] == 2
    assert {case["case"] for case in payload["cases"]} == {"gaussian-dimer", "gth-h2"}
    assert all(case["steps_completed"] == 1 for case in payload["cases"])
    assert csv_path.read_text().startswith("case,grid_shape")


def test_dft_nonlocal_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_nonlocal.csv"

    dft_nonlocal.main(["--grid", "4,4,4", "--iterations", "1", "--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["projector_count"] > 0
    assert payload["nonlocal_applied"]
    assert payload["dense_vs_operator_max_error"] < 1e-5
    assert csv_path.read_text().startswith("case,grid_shape")


def test_dft_solver_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_solver.csv"

    dft_solver.main(["--grid", "4,4,4", "--iterations", "1", "--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["eigenvalue_error"] < 1e-6
    assert "davidson_metadata" in payload
    assert csv_path.read_text().startswith("grid_shape,grid_points")


def test_dft_spin_kpoints_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_spin_kpoints.csv"

    dft_spin_kpoints.main(["--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["band_reused_density"]
    assert payload["band_shape"] == [3, 1]
    assert csv_path.read_text().startswith("kpoint_count,occupation_count")


def test_dft_relaxation_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_relaxation.csv"

    dft_relaxation.main(["--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["steps_completed"] == 1
    assert "stress" in payload
    assert csv_path.read_text().startswith("status,steps_completed")
