import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_atomistic.benchmarks import (
    NORMALIZED_BENCHMARK_FIELDS,
    cadence_sensitivity,
    dft_geometry,
    dft_nonlocal,
    dft_operator,
    dft_pseudopotential,
    dft_relaxation,
    dft_scf,
    dft_solver,
    dft_spin_kpoints,
    dhfr,
    ewald_reference,
    lj_md,
    md_acceleration,
    md_performance,
    mm_force_terms,
    neighbor_nonbonded_parity,
    normalize_benchmark_row,
    phase3_physics,
    pme_performance,
    same_workload_compare,
    stability,
    validation_gauntlet,
)


def _assert_normalized_payload(payload, *, timing_metric, status="ok"):
    for field in NORMALIZED_BENCHMARK_FIELDS:
        assert field in payload
    assert payload["engine"] == "mlx_atomistic"
    assert payload["timing_metric"] == timing_metric
    assert payload["status"] == status
    assert payload["command"].startswith("uv run python -m mlx_atomistic.benchmarks.")
    assert "runtime" in payload
    assert "hardware" in payload
    assert "commit" in payload


def _assert_normalized_row(row, *, benchmark_name, timing_metric):
    assert row["engine"] == "mlx_atomistic"
    assert row["benchmark_name"] == benchmark_name
    assert row["timing_metric"] == timing_metric
    assert row["timing_value"] == row[timing_metric]
    assert row["status"] in {"ok", "failed", "blocked"}
    assert "system" in row
    assert "atom_count" in row


def _assert_reference_payload(payload, *, engine, benchmark_name, timing_metric):
    for field in NORMALIZED_BENCHMARK_FIELDS:
        assert field in payload
    assert payload["engine"] == engine
    assert payload["benchmark_name"] == benchmark_name
    assert payload["timing_metric"] == timing_metric
    assert payload["status"] in {"ok", "failed", "blocked", "diagnostic"}
    assert payload["command"].startswith("uv run python scripts/")
    assert "runtime" in payload
    assert "hardware" in payload
    assert "commit" in payload


def _assert_mlx_comparison_fields(row, *, pair_id, metric_family):
    assert row["comparison_pair_id"] == pair_id
    assert row["comparison_role"] == "mlx"
    assert row["comparison_metric_family"] == metric_family
    assert row["comparison_command"].startswith("uv run python -m mlx_atomistic.benchmarks.")
    assert row["comparison_raw_output_path"].startswith(
        "outputs/benchmarks/same-workload-openmm-comparison/mlx-"
    )


def test_dhfr_readiness_requires_explicit_inputs_for_both_cases():
    implicit = dhfr.readiness_payload(case_spec=dhfr.CASE_SPECS["dhfr-implicit"])
    explicit = dhfr.readiness_payload(case_spec=dhfr.CASE_SPECS["dhfr-explicit-pme"])

    _assert_normalized_payload(implicit, timing_metric="ns_per_day", status="blocked")
    _assert_normalized_payload(explicit, timing_metric="ns_per_day", status="blocked")
    assert implicit["case"] == "dhfr-implicit"
    assert explicit["case"] == "dhfr-explicit-pme"
    assert implicit["comparison_pair_id"] == "dhfr-implicit"
    assert explicit["comparison_pair_id"] == "dhfr-explicit-pme"
    assert implicit["solvent_model"] == "implicit"
    assert explicit["solvent_model"] == "explicit"
    assert implicit["electrostatics_model"] == "gbsa_obc"
    assert explicit["electrostatics_model"] == "pme"
    assert implicit["input_status"]["downloads_attempted"] is False
    assert explicit["input_status"]["downloads_attempted"] is False
    assert implicit["atom_count"] is None
    assert explicit["atom_count"] is None
    assert explicit["cell_metadata_available"] is False
    assert "caller-provided DHFR input path" in implicit["blocker"]
    assert "caller-provided DHFR input path" in explicit["blocker"]


def test_dhfr_readiness_missing_inputs_fail_closed(tmp_path):
    payload = dhfr.readiness_payload(
        case_spec=dhfr.CASE_SPECS["dhfr-explicit-pme"],
        repo_root=tmp_path,
    )

    _assert_normalized_payload(payload, timing_metric="ns_per_day", status="blocked")
    assert payload["input_status"]["all_inputs_present"] is False
    assert payload["input_status"]["downloads_attempted"] is False
    assert "caller-provided DHFR input path" in payload["blocker"]


@pytest.mark.slow
def test_dhfr_implicit_prepare_blocks_without_explicit_inputs():
    payload = dhfr.prepare_payload(case_spec=dhfr.CASE_SPECS["dhfr-implicit"])

    _assert_normalized_payload(payload, timing_metric="ns_per_day", status="blocked")
    assert payload["prepare"] is True
    assert payload["artifact_status"] == "not_attempted"
    assert payload["electrostatics_model"] == "gbsa_obc"
    assert payload["artifact_path"].startswith("outputs/benchmarks/dhfr-artifacts/")
    assert "caller-provided DHFR input path" in payload["blocker"]


@pytest.mark.slow
def test_dhfr_explicit_prepare_reports_amber_or_pme_gate():
    payload = dhfr.prepare_payload(case_spec=dhfr.CASE_SPECS["dhfr-explicit-pme"])

    _assert_normalized_payload(payload, timing_metric="ns_per_day", status="blocked")
    assert payload["prepare"] is True
    assert payload["electrostatics_model"] == "pme"
    assert payload["artifact_path"].startswith("outputs/benchmarks/dhfr-artifacts/")
    assert payload["force_term_required_arrays"] == [
        "pme_mesh_shape",
        "pme_alpha",
        "pme_real_cutoff",
        "pme_assignment_order",
        "pme_charge_tolerance",
        "pme_deconvolve_assignment",
    ]
    assert payload["artifact_status"] == "not_attempted"
    assert "caller-provided DHFR input path" in payload["blocker"]


@pytest.mark.slow
def test_dhfr_implicit_runtime_blocks_without_explicit_inputs():
    payload = dhfr.runtime_payload(
        case_spec=dhfr.CASE_SPECS["dhfr-implicit"],
        steps=1,
    )

    _assert_normalized_payload(payload, timing_metric="ns_per_day", status="blocked")
    assert payload["comparison_pair_id"] == "dhfr-implicit"
    assert payload["step_count"] == 1
    assert payload["runtime_attempted"] is False
    assert payload["runtime_stage"] == "blocked"
    assert payload["runtime_blocker_category"] == "input_absence"
    assert "caller-provided DHFR input path" in payload["blocker"]


@pytest.mark.slow
def test_dhfr_explicit_runtime_blocks_without_explicit_inputs():
    payload = dhfr.runtime_payload(
        case_spec=dhfr.CASE_SPECS["dhfr-explicit-pme"],
        steps=1,
    )

    _assert_normalized_payload(payload, timing_metric="ns_per_day", status="blocked")
    assert payload["comparison_pair_id"] == "dhfr-explicit-pme"
    assert payload["step_count"] == 1
    assert payload["runtime_attempted"] is False
    assert payload["runtime_stage"] == "blocked"
    assert payload["runtime_blocker_category"] == "input_absence"
    assert "caller-provided DHFR input path" in payload["blocker"]


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
    assert "virtual-sites" in categories


def test_shared_benchmark_schema_helper_contract():
    required = set(NORMALIZED_BENCHMARK_FIELDS)
    assert {
        "engine",
        "benchmark_name",
        "fixture",
        "system",
        "atom_count",
        "step_count",
        "evaluation_count",
        "timing_metric",
        "hardware",
        "runtime",
        "finite",
        "status",
        "blocker",
        "command",
        "commit",
        "raw_output_path",
    } <= required


def test_shared_benchmark_schema_maps_legacy_local_fields():
    row = normalize_benchmark_row(
        {
            "case": "legacy-case",
            "test": "fallback-test",
            "steps": 3,
            "evaluations": 4,
            "median_s": 0.25,
        },
        benchmark_name="legacy_benchmark",
        timing_metric="median_s",
    )

    assert row["system"] == "legacy-case"
    assert row["step_count"] == 3
    assert row["evaluation_count"] == 4
    assert row["timing_value"] == 0.25
    assert row["timing_unit"] == "s"
    assert row["status"] == "ok"


def test_force_term_payload_uses_normalized_schema():
    payload = mm_force_terms.build_payload(evaluations=1, particles=16)

    _assert_normalized_payload(payload, timing_metric="ms_per_eval")
    row = payload["cases"][0]
    _assert_normalized_row(
        row,
        benchmark_name="mm_force_terms",
        timing_metric="ms_per_eval",
    )
    assert row["evaluation_count"] == 1
    assert row["atom_count"] > 0
    assert row["category"]


def test_phase3_physics_benchmark_covers_required_feature_rows():
    payload = phase3_physics.build_payload(
        evaluations=1,
        waters=1,
        atoms=4,
        replica_steps=1,
    )

    _assert_normalized_payload(payload, timing_metric="ms_per_eval")
    features = {row["feature"] for row in payload["cases"]}
    assert {
        "virtual_sites",
        "tip4p_ew",
        "gbsa_obc",
        "soft_core_lambda",
        "replica_exchange",
    } <= features
    for row in payload["cases"]:
        _assert_normalized_row(
            row,
            benchmark_name="phase3_physics",
            timing_metric="ms_per_eval",
        )
        assert row["status"] == "ok"
        assert row["blocker"] is None

    virtual_rows = [row for row in payload["cases"] if row["feature"] == "virtual_sites"]
    assert {row["operation"] for row in virtual_rows} == {
        "reconstruct_positions",
        "redistribute_forces",
    }
    tip4p_row = next(row for row in payload["cases"] if row["feature"] == "tip4p_ew")
    assert tip4p_row["json_csv_status"] == "available"
    _assert_mlx_comparison_fields(
        tip4p_row,
        pair_id="tip4p-ew-water",
        metric_family="ms/eval",
    )
    gbsa_ops = {row["operation"] for row in payload["cases"] if row["feature"] == "gbsa_obc"}
    assert {"obc_pair_accumulation_and_force", "surface_area_term"} <= gbsa_ops
    for gbsa_row in (row for row in payload["cases"] if row["feature"] == "gbsa_obc"):
        _assert_mlx_comparison_fields(
            gbsa_row,
            pair_id="gbsa-obc-small",
            metric_family="ms/eval",
        )
    soft_core = next(row for row in payload["cases"] if row["feature"] == "soft_core_lambda")
    assert soft_core["lambda_evaluation_count"] == 3
    replica = next(row for row in payload["cases"] if row["feature"] == "replica_exchange")
    assert replica["swap_attempts"] == 1
    assert 0.0 <= replica["acceptance_rate"] <= 1.0
    assert replica["history_materialization_count"] > 0


def test_phase3_physics_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "phase3.csv"

    phase3_physics.main(
        [
            "--evaluations",
            "1",
            "--waters",
            "1",
            "--atoms",
            "4",
            "--replica-steps",
            "1",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark_name"] == "phase3_physics"
    assert payload["case_count"] >= 7
    assert "replica_exchange" in {row["feature"] for row in payload["cases"]}
    csv_text = csv_path.read_text()
    assert csv_text.startswith("feature,case")
    assert "tip4p_ew" in csv_text


def test_md_acceleration_baseline_schema_reports_neighbor_split():
    payload = md_acceleration.build_payload(
        sizes=(16,),
        backends=("mlx_pairs",),
        evaluations=1,
    )

    row = payload["cases"][0]
    _assert_normalized_payload(payload, timing_metric="ms_per_eval")
    _assert_normalized_row(
        row,
        benchmark_name="md_acceleration",
        timing_metric="ms_per_eval",
    )
    assert payload["benchmark_name"] == "md_acceleration"
    assert "hardware" in payload
    assert row["atom_count"] == 16
    assert row["evaluation_count"] == 1
    assert row["neighbor_build_ms_per_eval"] >= 0.0
    assert row["force_eval_ms_per_eval"] >= 0.0
    assert row["candidate_count"] >= row["compact_pair_count"]
    assert row["candidate_waste_count"] is not None
    assert row["selected_policy"] == "requested:mlx_pairs"
    assert row["representation_kind"] == "pairs"


def test_md_performance_baseline_schema_reports_cadence_and_neighbor_fields():
    payload = md_performance.build_payload(
        sizes=(32,),
        steps=1,
        dt=0.002,
        mode="dynamic-neighbor",
        dense_threshold=1536,
        sample_interval=1,
        diagnostic_interval=1,
        evaluation_interval=1,
        neighbor_check_interval=1,
    )

    row = payload["cases"][0]
    _assert_normalized_payload(payload, timing_metric="steps_per_s")
    _assert_normalized_row(
        row,
        benchmark_name="md_performance",
        timing_metric="steps_per_s",
    )
    assert payload["benchmark_name"] == "md_performance"
    assert "hardware" in payload
    assert payload["platform_evidence"]["product_runtime"] == "mlx_atomistic"
    assert payload["platform_evidence"]["finite_outputs"] is True
    assert "AC5" in payload["platform_evidence"]["acceptance_criteria"]
    assert "G11" in payload["platform_evidence"]["gap_ids"]
    assert row["atom_count"] == 32
    assert row["step_count"] == 1
    _assert_mlx_comparison_fields(
        row,
        pair_id="lj-synthetic-loop",
        metric_family="steps/s",
    )
    assert row["neighbor_candidate_count"] >= row["compact_pair_count"]
    assert row["force_eval_ms_per_step"] >= 0.0
    assert row["sample_interval"] == 1
    assert row["diagnostic_interval"] == 1
    assert row["evaluation_interval"] == 1
    assert row["materialized_frame_count"] >= 1
    assert "diagnostics:1" in row["sync_cadence"]


def test_md_performance_json_output(tmp_path, capsys):
    out = tmp_path / "md-performance.json"

    md_performance.main(
        [
            "--sizes",
            "16",
            "--steps",
            "1",
            "--mode",
            "dynamic-neighbor",
            "--sample-interval",
            "1",
            "--diagnostic-interval",
            "1",
            "--evaluation-interval",
            "1",
            "--json-out",
            str(out),
            "--json",
        ]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(out.read_text())
    assert persisted == stdout_payload
    assert persisted["cases"][0]["neighbor_candidate_count"] is not None


def test_neighbor_nonbonded_parity_command_writes_validated_row(tmp_path, capsys):
    out = tmp_path / "neighbor-parity.json"

    neighbor_nonbonded_parity.main(
        [
            "--sizes",
            "32",
            "--density",
            "0.1",
            "--cutoff",
            "1.5",
            "--tile-size",
            "8",
            "--out",
            str(out),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(out.read_text())
    row = payload["cases"][0]
    assert persisted == payload
    assert row["status"] == "validated"
    assert row["backend"] == "mlx_cell_pairs"
    assert row["pair_policy"] == "lazy"
    assert row["dense_pair_cache_materialized"] is False
    assert row["semantic_topology"] is True
    assert row["exclusion_count"] >= 2
    assert row["one_four_count"] == 1
    assert row["energy_abs_delta"] <= (
        row["energy_tolerance"]["atol"]
        + row["energy_tolerance"]["rtol"] * max(abs(row["reference_energy"]), 1.0)
    )
    assert row["force_max_abs_delta"] <= (
        row["force_tolerance"]["atol"]
        + row["force_tolerance"]["rtol"] * max(row["reference_force_max_abs"], 1.0)
    )
    assert payload["largest_validated_size"] == 32
    assert payload["triclinic_status"] == "deferred_fail_closed"


def test_cadence_sensitivity_reports_timing_and_sync_counts():
    payload = cadence_sensitivity.build_payload(
        sizes=(16,),
        steps=2,
        cadences=(
            cadence_sensitivity.CadenceConfig("sparse", 2, 2, 2),
            cadence_sensitivity.CadenceConfig("frequent", 1, 1, 1),
        ),
    )

    assert payload["benchmark_name"] == "cadence_sensitivity"
    _assert_normalized_payload(payload, timing_metric="steps_per_s")
    assert payload["case_count"] == 2
    sparse, frequent = payload["cases"]
    assert sparse["steps_per_s"] > 0.0
    assert frequent["steps_per_s"] > 0.0
    assert sparse["sync_materialization_counts"]["materialized_frame_count"] == 2
    assert frequent["sync_materialization_counts"]["materialized_frame_count"] == 3
    assert sparse["sync_materialization_counts"]["diagnostic_sync_count"] == 2
    assert frequent["sync_materialization_counts"]["diagnostic_sync_count"] == 3
    assert sparse["sync_materialization_counts"]["evaluation_sync_count"] == 1
    assert frequent["sync_materialization_counts"]["evaluation_sync_count"] == 2
    assert payload["comparisons"][0]["cadence_name"] == "frequent"


def test_cadence_sensitivity_auto_uses_s5_default_neighbor_policy():
    payload = cadence_sensitivity.build_payload(
        sizes=(512,),
        steps=1,
        mode="auto",
        cadences=(
            cadence_sensitivity.CadenceConfig("sparse", 1, 1, 1),
            cadence_sensitivity.CadenceConfig("frequent", 1, 1, 1),
        ),
    )

    assert payload["config"]["dense_threshold"] == 1536
    for row in payload["cases"]:
        assert row["backend"] == "mlx_dense"
        assert row["selected_policy"] == (
            "auto:evidence_dense_below_threshold; dense_threshold:1536"
        )


def test_cadence_sensitivity_compares_each_size_to_own_baseline():
    payload = cadence_sensitivity.build_payload(
        sizes=(16, 32),
        steps=2,
        cadences=(
            cadence_sensitivity.CadenceConfig("sparse", 2, 2, 2),
            cadence_sensitivity.CadenceConfig("frequent", 1, 1, 1),
        ),
    )

    comparisons = payload["comparisons"]
    assert [item["particles"] for item in comparisons] == [16, 32]
    assert all(item["baseline_cadence_name"] == "sparse" for item in comparisons)
    assert all(item["cadence_name"] == "frequent" for item in comparisons)
    assert all(item["materialized_frame_delta"] == 1 for item in comparisons)
    assert all(item["diagnostic_sync_delta"] == 1 for item in comparisons)
    assert all(item["evaluation_sync_delta"] == 1 for item in comparisons)


def test_openmm_opencl_unavailable_platform_non_json_does_not_crash(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"
    csv_path = tmp_path / "blocked-openmm.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--platform",
            "DefinitelyMissing",
            "--particles",
            "16",
            "--steps",
            "1",
            "--spacing-nm",
            "1.0",
            "--csv",
            str(csv_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "blocked" in result.stdout
    assert "DefinitelyMissing" in result.stdout
    assert "available_platforms=" in result.stdout
    csv_text = csv_path.read_text()
    assert "status" in csv_text
    assert "blocked" in csv_text


def test_openmm_opencl_unavailable_platform_json_uses_normalized_schema():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--platform",
            "DefinitelyMissing",
            "--particles",
            "16",
            "--steps",
            "1",
            "--spacing-nm",
            "1.0",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_opencl_reference",
        timing_metric="steps_per_s",
    )
    assert payload["status"] == "blocked"
    assert payload["blocker"]
    assert payload["atom_count"] == 16
    assert payload["step_count"] == 1
    assert payload["evaluation_count"] == 1


def test_openmm_controlled_gbsa_case_reports_normalized_blocker_or_ok():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "gbsa-obc-small",
            "--platform",
            "Reference",
            "--particles",
            "4",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_opencl_reference",
        timing_metric="ms_per_eval",
    )
    assert payload["fixture"] == "gbsa_obc_small"
    assert payload["case"] == "gbsa-obc-small"
    assert payload["status"] in {"ok", "blocked"}
    assert payload["atom_count"] == 4
    assert payload["step_count"] == 1
    assert payload["evaluation_count"] == 1
    if payload["status"] == "blocked":
        assert payload["blocker"]
        assert "not implemented yet" not in payload["blocker"]
    else:
        assert payload["blocker"] is None
        assert payload["finite"] is True
        assert payload["ms_per_eval"] >= 0.0
        assert payload["timing_value"] == payload["ms_per_eval"]
        assert payload["potential_energy_kj_mol"] is not None
        assert payload["force_norm_kj_mol_nm"] >= 0.0
        setup = payload["obc_force_setup"]
        assert setup["force"] == "GBSAOBCForce"
        assert setup["nonbonded_method"] == "NoCutoff"
        assert setup["solvent_dielectric"] == 78.5
        assert setup["solute_dielectric"] == 1.0
        assert setup["surface_area_energy_kj_mol_nm2"] == 2.25936
        assert [round(value, 6) for value in setup["charge_e"]] == [0.4, -0.35, 0.4, -0.35]
        assert [round(value, 6) for value in setup["radius_angstrom"]] == [
            1.45,
            1.55,
            1.65,
            1.75,
        ]
        assert [round(value, 6) for value in setup["scale"]] == [
            0.72,
            0.763333,
            0.806667,
            0.85,
        ]


def test_openmm_controlled_gbsa_case_non_json_reports_latency_or_blocker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "gbsa-obc-small",
            "--platform",
            "Reference",
            "--particles",
            "4",
            "--steps",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OpenMM Reference 4 atoms:" in result.stdout
    assert "ms/eval" in result.stdout or "blocked" in result.stdout
    assert "KeyError" not in result.stderr


def test_openmm_controlled_tip4p_case_reports_normalized_blocker_or_ok():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "tip4p-ew-water",
            "--platform",
            "Reference",
            "--particles",
            "4",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_opencl_reference",
        timing_metric="ms_per_eval",
    )
    assert payload["fixture"] == "tip4p_ew_water"
    assert payload["case"] == "tip4p-ew-water"
    assert payload["status"] in {"ok", "blocked"}
    assert payload["atom_count"] == 4
    assert payload["step_count"] == 1
    assert payload["evaluation_count"] == 1
    if payload["status"] == "blocked":
        assert payload["blocker"]
        assert "not implemented yet" not in payload["blocker"]
    else:
        assert payload["blocker"] is None
        assert payload["finite"] is True
        assert payload["ms_per_eval"] >= 0.0
        assert payload["timing_value"] == payload["ms_per_eval"]
        assert payload["operation_semantics"] == "virtual_site_reconstruction"
        assert payload["openmm_operation"] == "Context.computeVirtualSites"
        assert payload["water_model"] == "TIP4P-Ew"
        assert payload["virtual_site_type"] == "ThreeParticleAverageSite"
        assert payload["virtual_site_count"] == 1
        assert payload["virtual_site_indices"] == [3]
        assert len(payload["virtual_site_positions_nm"]) == 1
        assert payload["virtual_site_position_norm_nm"] > 0.0
        geometry = payload["tip4p_ew_geometry"]
        assert geometry["oh_distance_angstrom"] == 0.9572
        assert geometry["hoh_angle_degrees"] == 104.52
        assert geometry["om_distance_angstrom"] == 0.125
        assert set(geometry["m_site_weights"]) == {"oxygen", "hydrogen1", "hydrogen2"}


def test_openmm_import_failure_reports_blocked_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "openmm.py").write_text("raise ImportError('synthetic missing openmm')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo_root)))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--platform",
            "OpenCL",
            "--particles",
            "16",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_opencl_reference",
        timing_metric="steps_per_s",
    )
    assert payload["status"] == "blocked"
    assert "OpenMM import unavailable" in payload["blocker"]


def test_openmm_invalid_input_exits_nonzero():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--platform",
            "DefinitelyMissing",
            "--particles",
            "-1",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "particles must be positive" in result.stderr


def test_openmm_import_failure_does_not_mask_invalid_input(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "openmm.py").write_text("raise ImportError('synthetic missing openmm')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo_root)))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--platform",
            "OpenCL",
            "--particles",
            "-1",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode != 0
    assert "particles must be positive" in result.stderr
    assert "synthetic missing openmm" not in result.stderr


@pytest.mark.slow
def test_openmm_dhfr_reference_cases_report_normalized_blocker_or_ok():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_dhfr.py"

    for case, fixture in (
        ("dhfr-implicit", "dhfr_implicit"),
        ("dhfr-explicit-pme", "dhfr_explicit_pme"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--case",
                case,
                "--platform",
                "Reference",
                "--steps",
                "1",
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        _assert_reference_payload(
            payload,
            engine="openmm-reference",
            benchmark_name="openmm_dhfr_reference",
            timing_metric="ns_per_day",
        )
        assert payload["case"] == case
        assert payload["fixture"] == fixture
        assert payload["status"] in {"ok", "blocked"}
        assert payload["atom_count"] and payload["atom_count"] > 0
        assert payload["step_count"] == 1
        assert payload["evaluation_count"] == 1
        assert payload["input_status"]["downloads_attempted"] is False
        assert payload["raw_input_paths"]
        assert payload["raw_input_metadata"]
        if payload["status"] == "ok":
            assert payload["blocker"] is None
            assert payload["finite"] is True
            assert payload["ns_per_day"] >= 0.0
            assert payload["timing_value"] == payload["ns_per_day"]
        else:
            assert payload["blocker"]


def test_openmm_dhfr_missing_inputs_report_normalized_blocker(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_dhfr.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "dhfr-explicit-pme",
            "--platform",
            "Reference",
            "--steps",
            "1",
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_dhfr_reference",
        timing_metric="ns_per_day",
    )
    assert payload["status"] == "blocked"
    assert "missing OpenMM input path" in payload["blocker"]
    assert payload["input_status"]["all_inputs_present"] is False
    assert payload["input_status"]["downloads_attempted"] is False
    assert payload["ns_per_day"] is None


@pytest.mark.data  # needs gitignored vendors/ data; skipped on CI fast lane
def test_openmm_dhfr_unavailable_platform_reports_blocked_json():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_dhfr.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "dhfr-implicit",
            "--platform",
            "DefinitelyMissing",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="openmm_dhfr_reference",
        timing_metric="ns_per_day",
    )
    assert payload["status"] == "blocked"
    assert "DefinitelyMissing" in payload["blocker"]
    assert payload["atom_count"] and payload["atom_count"] > 0


def test_openmm_dhfr_invalid_input_exits_nonzero_before_import(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_dhfr.py"
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "openmm.py").write_text("raise ImportError('synthetic missing openmm')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo_root)))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "dhfr-implicit",
            "--platform",
            "Reference",
            "--steps",
            "-1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode != 0
    assert "steps must be positive" in result.stderr
    assert "synthetic missing openmm" not in result.stderr


def test_openmm_controlled_case_invalid_input_exits_nonzero():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "gbsa-obc-small",
            "--platform",
            "Reference",
            "--particles",
            "-1",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "particles must be positive" in result.stderr


def test_openmm_tip4p_invalid_particle_count_exits_nonzero_before_import(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openmm_opencl.py"
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "openmm.py").write_text("raise ImportError('synthetic missing openmm')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo_root)))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "tip4p-ew-water",
            "--platform",
            "Reference",
            "--particles",
            "5",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode != 0
    assert "particles must be a multiple of 4 for tip4p-ew-water" in result.stderr
    assert "synthetic missing openmm" not in result.stderr


def test_same_workload_comparison_summary_computes_only_comparable_ratios():
    mlx_payloads = [
        {
            "cases": [
                {
                    "comparison_pair_id": "lj-synthetic-loop",
                    "status": "ok",
                    "timing_metric": "steps_per_s",
                    "timing_value": 50.0,
                    "atom_count": 32,
                    "step_count": 1,
                    "command": "uv run python -m mlx_atomistic.benchmarks.md_performance",
                    "hardware": {"device": "mlx"},
                    "runtime": {"engine": "mlx"},
                },
                {
                    "comparison_pair_id": "gbsa-obc-small",
                    "status": "ok",
                    "timing_metric": "ms_per_eval",
                    "timing_value": 1.0,
                    "atom_count": 4,
                    "step_count": None,
                    "command": "uv run python -m mlx_atomistic.benchmarks.phase3_physics",
                    "hardware": {"device": "mlx"},
                    "runtime": {"engine": "mlx"},
                },
            ],
        }
    ]
    openmm_payloads = [
        {
            "case": "synthetic-lj-periodic",
            "fixture": "synthetic_lj_periodic",
            "status": "ok",
            "timing_metric": "steps_per_s",
            "timing_value": 100.0,
            "atom_count": 32,
            "step_count": 1,
            "command": "uv run python scripts/benchmark_openmm_opencl.py",
            "hardware": {"device": "openmm"},
            "runtime": {"engine": "openmm"},
        },
        {
            "case": "gbsa-obc-small",
            "fixture": "gbsa_obc_small",
            "status": "blocked",
            "blocker": "controlled reference not implemented",
            "timing_metric": "ms_per_eval",
            "timing_value": None,
            "atom_count": 4,
            "step_count": 1,
            "command": "uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small",
            "hardware": {"device": "openmm"},
            "runtime": {"engine": "openmm"},
        },
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    for field in NORMALIZED_BENCHMARK_FIELDS:
        assert field in payload
    assert payload["engine"] == "mlx-openmm-comparison"
    assert payload["benchmark_name"] == "same_workload_openmm_comparison"
    assert payload["timing_metric"] == "openmm_to_mlx_ratio"
    assert payload["command"].startswith("uv run python -m mlx_atomistic.benchmarks.")
    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert rows["lj-synthetic-loop"]["comparison_status"] == "comparable"
    assert rows["lj-synthetic-loop"]["openmm_to_mlx_ratio"] == 2.0
    assert rows["gbsa-obc-small"]["comparison_status"] == "blocked"
    assert "OpenMM status is blocked" in rows["gbsa-obc-small"]["blocker"]
    assert rows["gbsa-obc-small"]["openmm_to_mlx_ratio"] is None
    assert rows["tip4p-ew-water"]["comparison_status"] == "blocked"
    assert rows["tip4p-ew-water"]["blocker"] == "missing MLX normalized row for pair"


def _mlx_lj_case(*, atom_count, step_count, steps_per_s, status="ok"):
    return {
        "case": "synthetic_lj",
        "status": status,
        "timing_metric": "steps_per_s",
        "timing_value": steps_per_s,
        "atom_count": atom_count,
        "step_count": step_count,
        "command": "uv run python -m mlx_atomistic.benchmarks.md_performance",
    }


def _reference_lj_row(*, fixture, atom_count, step_count, steps_per_s, status="ok", blocker=None):
    return {
        "fixture": fixture,
        "case": fixture,
        "status": status,
        "blocker": blocker,
        "timing_metric": "steps_per_s",
        "timing_value": steps_per_s,
        "atom_count": atom_count,
        "step_count": step_count,
        "command": f"uv run python scripts/benchmark_{fixture}.py",
    }


def test_scaling_summary_pairs_ladder_by_size_and_computes_ratios():
    mlx_payloads = [
        {
            "cases": [
                _mlx_lj_case(atom_count=1000, step_count=500, steps_per_s=50.0),
                _mlx_lj_case(atom_count=4000, step_count=500, steps_per_s=12.0),
            ]
        }
    ]
    openmm_payloads = [
        _reference_lj_row(
            fixture="synthetic_lj_periodic", atom_count=1000, step_count=500, steps_per_s=100.0
        ),
        _reference_lj_row(
            fixture="synthetic_lj_periodic", atom_count=4000, step_count=500, steps_per_s=18.0
        ),
    ]
    lammps_payloads = [
        _reference_lj_row(
            fixture="synthetic_lj_periodic", atom_count=1000, step_count=500, steps_per_s=200.0
        ),
        _reference_lj_row(
            fixture="synthetic_lj_periodic",
            atom_count=4000,
            step_count=500,
            steps_per_s=None,
            status="diagnostic",
            blocker="GPU pair style not confirmed active",
        ),
    ]

    payload = same_workload_compare.build_scaling_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
        lammps_payloads=lammps_payloads,
    )

    assert payload["benchmark_name"] == "same_workload_lj_scaling"
    # The ladder must NOT collide on a single comparison_pair_id.
    assert payload["size_count"] == 2
    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert set(rows) == {"lj-synthetic@N=1000", "lj-synthetic@N=4000"}

    small = rows["lj-synthetic@N=1000"]
    assert small["comparison_status"] == "comparable"
    assert small["openmm_to_mlx_ratio"] == 2.0
    assert small["lammps_to_mlx_ratio"] == 4.0
    assert small["mlx_steps_per_s"] == 50.0

    large = rows["lj-synthetic@N=4000"]
    assert large["comparison_status"] == "comparable"  # OpenMM is comparable
    assert large["openmm_to_mlx_ratio"] == 1.5
    # LAMMPS diagnostic at this size suppresses its ratio but does not block the row.
    assert large["lammps_status"] == "diagnostic"
    assert large["lammps_to_mlx_ratio"] is None


def test_scaling_summary_mismatched_size_does_not_pair():
    mlx_payloads = [{"cases": [_mlx_lj_case(atom_count=1000, step_count=500, steps_per_s=50.0)]}]
    openmm_payloads = [
        _reference_lj_row(
            fixture="synthetic_lj_periodic", atom_count=2000, step_count=500, steps_per_s=99.0
        )
    ]

    payload = same_workload_compare.build_scaling_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert set(rows) == {"lj-synthetic@N=1000", "lj-synthetic@N=2000"}
    # MLX-only size: reference missing, no spurious ratio.
    assert rows["lj-synthetic@N=1000"]["comparison_status"] == "blocked"
    assert rows["lj-synthetic@N=1000"]["openmm_to_mlx_ratio"] is None
    # OpenMM-only size: no MLX row, blocked.
    assert rows["lj-synthetic@N=2000"]["comparison_status"] == "blocked"


def test_scaling_summary_nonpositive_reference_is_diagnostic():
    mlx_payloads = [{"cases": [_mlx_lj_case(atom_count=1000, step_count=500, steps_per_s=50.0)]}]
    openmm_payloads = [
        _reference_lj_row(
            fixture="synthetic_lj_periodic", atom_count=1000, step_count=500, steps_per_s=0.0
        )
    ]

    payload = same_workload_compare.build_scaling_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    row = {r["pair_id"]: r for r in payload["cases"]}["lj-synthetic@N=1000"]
    assert row["openmm_status"] == "diagnostic"
    assert row["openmm_to_mlx_ratio"] is None
    assert row["comparison_status"] == "diagnostic"


def test_same_workload_comparison_controlled_pairs_require_matching_operations():
    mlx_payloads = [
        {
            "cases": [
                {
                    "comparison_pair_id": "gbsa-obc-small",
                    "feature": "gbsa_obc",
                    "operation": "obc_pair_accumulation_and_force",
                    "status": "ok",
                    "timing_metric": "ms_per_eval",
                    "timing_value": 2.0,
                    "atom_count": 4,
                    "command": "uv run python -m mlx_atomistic.benchmarks.phase3_physics",
                },
                {
                    "comparison_pair_id": "tip4p-ew-water",
                    "feature": "tip4p_ew",
                    "operation": "m_site_reconstruction",
                    "status": "ok",
                    "timing_metric": "ms_per_eval",
                    "timing_value": 4.0,
                    "atom_count": 4,
                    "command": "uv run python -m mlx_atomistic.benchmarks.phase3_physics",
                },
            ]
        }
    ]
    openmm_payloads = [
        {
            "case": "gbsa-obc-small",
            "fixture": "gbsa_obc_small",
            "status": "ok",
            "timing_metric": "ms_per_eval",
            "timing_value": 6.0,
            "atom_count": 4,
            "command": "uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small",
            "obc_force_setup": {"force": "GBSAOBCForce"},
        },
        {
            "case": "tip4p-ew-water",
            "fixture": "tip4p_ew_water",
            "status": "ok",
            "timing_metric": "ms_per_eval",
            "timing_value": 8.0,
            "atom_count": 4,
            "command": "uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water",
            "operation_semantics": "virtual_site_reconstruction",
            "openmm_operation": "Context.computeVirtualSites",
        },
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert rows["gbsa-obc-small"]["comparison_status"] == "comparable"
    assert rows["gbsa-obc-small"]["openmm_to_mlx_ratio"] == 3.0
    assert rows["tip4p-ew-water"]["comparison_status"] == "comparable"
    assert rows["tip4p-ew-water"]["openmm_to_mlx_ratio"] == 2.0


def test_same_workload_comparison_controlled_pair_operation_mismatch_is_diagnostic():
    mlx_payloads = [
        {
            "comparison_pair_id": "tip4p-ew-water",
            "feature": "tip4p_ew",
            "operation": "m_site_reconstruction",
            "status": "ok",
            "timing_metric": "ms_per_eval",
            "timing_value": 4.0,
            "atom_count": 4,
        }
    ]
    openmm_payloads = [
        {
            "case": "tip4p-ew-water",
            "fixture": "tip4p_ew_water",
            "status": "ok",
            "timing_metric": "ms_per_eval",
            "timing_value": 8.0,
            "atom_count": 4,
            "operation_semantics": "full_water_force_evaluation",
            "openmm_operation": "Context.getState",
        }
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    row = {row["pair_id"]: row for row in payload["cases"]}["tip4p-ew-water"]
    assert row["comparison_status"] == "diagnostic"
    assert row["status"] == "diagnostic"
    assert "full_water_force_evaluation" in row["blocker"]
    assert row["openmm_to_mlx_ratio"] is None


def test_same_workload_comparison_controlled_pair_status_gates_retain_reasons():
    mlx_payloads = [
        {
            "comparison_pair_id": "gbsa-obc-small",
            "feature": "gbsa_obc",
            "operation": "obc_pair_accumulation_and_force",
            "status": "diagnostic",
            "blocker": "MLX row is an instrumentation-only GBSA probe",
            "timing_metric": "ms_per_eval",
            "timing_value": 2.0,
            "atom_count": 4,
        },
        {
            "comparison_pair_id": "tip4p-ew-water",
            "feature": "tip4p_ew",
            "operation": "m_site_reconstruction",
            "status": "ok",
            "timing_metric": "ms_per_eval",
            "timing_value": 4.0,
            "atom_count": 4,
        },
    ]
    openmm_payloads = [
        {
            "case": "gbsa-obc-small",
            "fixture": "gbsa_obc_small",
            "status": "blocked",
            "blocker": "OpenMM GBSA control unavailable",
            "timing_metric": "ms_per_eval",
            "timing_value": None,
            "atom_count": 4,
            "obc_force_setup": {"force": "GBSAOBCForce"},
        },
        {
            "case": "tip4p-ew-water",
            "fixture": "tip4p_ew_water",
            "status": "blocked",
            "blocker": "OpenMM platform lacks virtual site reconstruction support",
            "timing_metric": "ms_per_eval",
            "timing_value": None,
            "atom_count": 4,
        },
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert rows["gbsa-obc-small"]["comparison_status"] == "diagnostic"
    assert "instrumentation-only GBSA probe" in rows["gbsa-obc-small"]["blocker"]
    assert rows["gbsa-obc-small"]["openmm_to_mlx_ratio"] is None
    assert rows["tip4p-ew-water"]["comparison_status"] == "blocked"
    assert "virtual site reconstruction support" in rows["tip4p-ew-water"]["blocker"]
    assert rows["tip4p-ew-water"]["openmm_to_mlx_ratio"] is None


def test_same_workload_comparison_dhfr_blocked_rows_suppress_ratio():
    mlx_payloads = [
        {
            "comparison_pair_id": "dhfr-implicit",
            "status": "blocked",
            "blocker": "missing GBSA/OBC artifact capability",
            "timing_metric": "ns_per_day",
            "timing_value": None,
            "atom_count": 2489,
            "step_count": 1,
            "solvent_model": "implicit",
            "electrostatics_model": "gbsa_obc",
        },
        {
            "comparison_pair_id": "dhfr-explicit-pme",
            "status": "blocked",
            "blocker": "AMBER explicit PME artifact import blocked",
            "timing_metric": "ns_per_day",
            "timing_value": None,
            "atom_count": 23558,
            "step_count": 1,
            "solvent_model": "explicit",
            "electrostatics_model": "pme",
        },
    ]
    openmm_payloads = [
        {
            "case": "dhfr-implicit",
            "fixture": "dhfr_implicit",
            "status": "ok",
            "timing_metric": "ns_per_day",
            "timing_value": 2.0,
            "atom_count": 2489,
            "step_count": 1,
            "solvent_model": "implicit",
            "electrostatics_model": "gbsa_obc",
        },
        {
            "case": "dhfr-explicit-pme",
            "fixture": "dhfr_explicit_pme",
            "status": "ok",
            "timing_metric": "ns_per_day",
            "timing_value": 1.0,
            "atom_count": 23558,
            "step_count": 1,
            "solvent_model": "explicit",
            "electrostatics_model": "pme",
        },
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    rows = {row["pair_id"]: row for row in payload["cases"]}
    assert rows["dhfr-implicit"]["comparison_status"] == "blocked"
    assert "missing GBSA/OBC" in rows["dhfr-implicit"]["blocker"]
    assert rows["dhfr-implicit"]["openmm_to_mlx_ratio"] is None
    assert rows["dhfr-explicit-pme"]["comparison_status"] == "blocked"
    assert "AMBER explicit PME" in rows["dhfr-explicit-pme"]["blocker"]
    assert rows["dhfr-explicit-pme"]["openmm_to_mlx_ratio"] is None


def test_same_workload_comparison_dhfr_semantic_mismatch_is_diagnostic():
    mlx_payloads = [
        {
            "comparison_pair_id": "dhfr-implicit",
            "status": "ok",
            "timing_metric": "ns_per_day",
            "timing_value": 1.0,
            "atom_count": 2489,
            "step_count": 1,
            "solvent_model": "explicit",
            "electrostatics_model": "pme",
        }
    ]
    openmm_payloads = [
        {
            "case": "dhfr-implicit",
            "status": "ok",
            "timing_metric": "ns_per_day",
            "timing_value": 2.0,
            "atom_count": 2489,
            "step_count": 1,
            "solvent_model": "implicit",
            "electrostatics_model": "gbsa_obc",
        }
    ]

    payload = same_workload_compare.build_summary(
        mlx_payloads=mlx_payloads,
        openmm_payloads=openmm_payloads,
    )

    row = {row["pair_id"]: row for row in payload["cases"]}["dhfr-implicit"]
    assert row["comparison_status"] == "diagnostic"
    assert "solvent/electrostatics semantics differ" in row["blocker"]
    assert row["openmm_to_mlx_ratio"] is None


def test_lammps_opencl_unsupported_fixture_reports_normalized_blocker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_lammps_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fixture",
            "missing-fixture",
            "--particles",
            "16",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="lammps-reference",
        benchmark_name="lammps_opencl_reference",
        timing_metric="steps_per_s",
    )
    assert payload["status"] == "blocked"
    assert "unsupported fixture" in payload["blocker"]
    assert payload["atom_count"] == 16
    assert payload["step_count"] == 1
    assert payload["opencl_device"] is None
    assert payload["requested_opencl_device"] == "0"
    assert payload["opencl_device_applied"] is False


def test_lammps_invalid_input_exits_nonzero():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_lammps_opencl.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--particles",
            "-1",
            "--steps",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "particles must be positive" in result.stderr


def test_reference_payload_allows_diagnostic_status():
    payload = {field: None for field in NORMALIZED_BENCHMARK_FIELDS}
    payload.update(
        {
            "engine": "lammps-reference",
            "benchmark_name": "diagnostic_reference",
            "fixture": "diagnostic",
            "system": "diagnostic",
            "timing_metric": "steps_per_s",
            "timing_value": None,
            "hardware": {},
            "runtime": {},
            "finite": False,
            "status": "diagnostic",
            "blocker": "style has no GPU/OpenCL equivalent",
            "command": "uv run python scripts/benchmark_lammps_opencl.py",
            "commit": "test",
            "raw_output_path": "results/diagnostic.json",
        }
    )

    _assert_reference_payload(
        payload,
        engine="lammps-reference",
        benchmark_name="diagnostic_reference",
        timing_metric="steps_per_s",
    )


def test_m5max_reference_environment_probe_reports_reference_engines():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_m5max_reference.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "environment",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="reference-environment",
        benchmark_name="m5max_reference_environment",
        timing_metric="environment_probe",
    )
    assert payload["status"] in {"ok", "diagnostic"}
    assert payload["raw_output_path"] == "results/m5max-reference/environment.json"
    cases = {case["engine"]: case for case in payload["cases"]}
    assert {"openmm-reference", "lammps-reference"} == set(cases)
    assert cases["openmm-reference"]["status"] in {"ok", "diagnostic"}
    assert cases["lammps-reference"]["status"] in {"ok", "diagnostic"}
    assert payload["command_surface"]["harness"].startswith(
        "uv run python scripts/benchmark_m5max_reference.py"
    )
    assert payload["command_surface"]["lammps_console_script"] == ".venv/bin/lmp"
    assert "packaged executable" in payload["command_surface"]["lammps_console_script_policy"]


@pytest.mark.data  # needs gitignored vendors/ data; skipped on CI fast lane
def test_m5max_reference_lammps_classify_only_covers_official_cases():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_m5max_reference.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "lammps",
            "--classify-only",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="lammps-reference",
        benchmark_name="m5max_lammps_official",
        timing_metric="loop_time_s",
    )
    cases = {case["case"]: case for case in payload["cases"]}
    assert set(cases) == {"lj", "chain", "eam", "chute", "rhodo"}
    assert payload["status"] == "diagnostic"
    assert {case["status"] for case in cases.values()} == {"diagnostic"}
    assert cases["lj"]["acceleration_classification"] == "full_gpu_opencl"
    assert cases["eam"]["acceleration_classification"] == "full_gpu_opencl"
    assert cases["chain"]["acceleration_classification"] == "partial_gpu_opencl"
    assert cases["rhodo"]["acceleration_classification"] == "partial_gpu_opencl"
    assert cases["chute"]["acceleration_classification"] == "cpu_only_diagnostic"
    for case in cases.values():
        assert case["input_script"].startswith("vendors/lammps/bench/")
        assert case["work_dir"].startswith("results/m5max-reference/lammps/")
        assert case["command"].startswith("uv run python scripts/benchmark_m5max_reference.py")
        assert case["blocker"] == "classification only; benchmark not executed"


@pytest.mark.data  # needs gitignored vendors/ data; skipped on CI fast lane
def test_m5max_reference_openmm_dry_run_covers_named_systems():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_m5max_reference.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "openmm",
            "--dry-run",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    _assert_reference_payload(
        payload,
        engine="openmm-reference",
        benchmark_name="m5max_openmm_official",
        timing_metric="ns_per_day",
    )
    cases = {case["case"]: case for case in payload["cases"]}
    assert set(cases) == {"dhfr", "apoa1", "amber20"}
    assert [case["case"] for case in payload["cases"]] == ["dhfr", "apoa1", "amber20"]
    assert payload["status"] == "diagnostic"
    assert {case["status"] for case in cases.values()} == {"diagnostic"}
    assert cases["dhfr"]["tests"] == "gbsa,rf,pme"
    assert cases["apoa1"]["tests"] == "apoa1rf,apoa1pme,apoa1ljpme"
    assert cases["amber20"]["tests"] == "amber20-cellulose,amber20-stmv"
    assert cases["amber20"]["external_inputs"] == [
        "https://ambermd.org/Amber20_Benchmark_Suite.tar.gz"
    ]
    for case in cases.values():
        assert "benchmark.py" in case["command"]
        assert "--platform OpenCL" in case["command"]
        assert "--precision single" in case["command"]
        assert case["timeout_s"] == 1200
        assert case["raw_output_path"].startswith("results/m5max-reference/openmm/")
        assert case["blocker"] == "dry run; benchmark not executed"


@pytest.mark.data  # needs gitignored vendors/ data; skipped on CI fast lane
def test_m5max_reference_openmm_run_path_invokes_executor(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_m5max_reference.py"
    spec = importlib.util.spec_from_file_location("benchmark_m5max_reference_under_test", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    calls = []

    def fake_execute(record):
        calls.append(record["case"])
        updated = dict(record)
        updated.update(
            {
                "status": "ok",
                "blocker": None,
                "benchmarks": [{"ns_per_day": 1.0}],
                "benchmark_count": 1,
                "ns_per_day": 1.0,
                "timing_value": 1.0,
                "finite": True,
            }
        )
        return updated

    monkeypatch.setattr(module, "_execute_openmm_case", fake_execute)

    payload = module.build_openmm_payload(
        cases=["dhfr"],
        dry_run=False,
        output_root=tmp_path / "m5max-reference",
        seconds=0.01,
    )

    assert calls == ["dhfr"]
    assert payload["status"] == "ok"
    case = payload["cases"][0]
    assert case["case"] == "dhfr"
    assert case["benchmark_count"] == 1
    assert case["timing_value"] == 1.0


def test_pme_performance_stage_summary_schema_without_fixture(tmp_path):
    payload = pme_performance.build_payload(
        fixture_dir=tmp_path / "missing-pme-fixture",
        iterations=1,
        warmups=0,
    )

    assert payload["benchmark_name"] == "pme_performance"
    _assert_normalized_payload(payload, timing_metric="median_s", status="blocked")
    assert payload["status"] == "blocked"
    assert payload["blocker"].startswith("missing PME parity report")
    assert payload["evaluation_count"] == 1
    assert "hardware" in payload
    assert payload["stage_timings"]["direct_space"]["available"] is False
    assert payload["unsupported_timing_split_blockers"][0]["name"] == "pme_fixture"


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
    assert payload["case_count"] == 1
    assert {case["case"] for case in payload["cases"]} == {"gaussian"}
    assert payload["external_pseudopotentials"] == {
        "upf_path": None,
        "gth_path": None,
        "gth_element": None,
        "gth_name": None,
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
            "gaussian-dimer",
            "--csv",
            str(csv_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_shape"] == [4, 4, 4]
    assert payload["case_count"] == 1
    assert {case["case"] for case in payload["cases"]} == {"gaussian-dimer"}
    assert all(case["steps_completed"] == 1 for case in payload["cases"])
    assert csv_path.read_text().startswith("case,grid_shape")


def test_dft_nonlocal_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_nonlocal.csv"

    dft_nonlocal.main(["--grid", "4,4,4", "--iterations", "1", "--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["projector_count"] == 0
    assert payload["nonlocal_applied"] is False
    assert "explicit --upf" in payload["blocker"]
    assert csv_path.read_text().startswith("case,status")


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


@pytest.mark.slow
def test_dft_relaxation_benchmark_json_and_csv_smoke(tmp_path, capsys):
    csv_path = tmp_path / "dft_relaxation.csv"

    dft_relaxation.main(["--csv", str(csv_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["steps_completed"] == 1
    assert "stress" in payload
    assert csv_path.read_text().startswith("status,steps_completed")
