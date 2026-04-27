import json

from mlx_atomistic.benchmarks import lj_md, mm_force_terms, stability, validation_gauntlet


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
    assert payload["summary"]["total_cases"] == 5
    assert payload["summary"]["all_passed"]
    assert len(payload["cases"]) == 5
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
