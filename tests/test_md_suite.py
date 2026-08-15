from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.artifacts import load_prepared_mlx_artifact
from mlx_atomistic.benchmarks.md_suite import (
    STAGE_PROFILE_SCHEMA,
    SUITE_SCHEMA,
    _route_stage,
    case_inventory,
    compare_suites,
    load_case_registry,
    profile_suite,
    resolve_cases,
    run_suite,
)
from mlx_atomistic.benchmarks.tip3p_water import (
    TIP3P_DENSITY_G_CM3,
    build_tip3p_water_box,
)
from mlx_atomistic.prep.io import save_prepared_system
from scripts import prepare_openmm_dhfr_explicit


def test_md_suite_registry_defines_local_acceptance_pair_and_release_cases():
    registry = load_case_registry()

    local = resolve_cases(registry=registry, suite="local")
    release = resolve_cases(registry=registry, suite="release")

    assert [case.case_id for case in local] == ["dhfr-5dfr-pme", "jac-94k-pme"]
    assert [case.role for case in local] == ["non_regression", "improvement"]
    assert {case.case_id for case in release} >= {
        "tip3p-water-30k",
        "tip3p-water-90k",
        "apoa1-pme",
        "gpcrmd-729-pme",
    }
    assert len({case.fingerprint for case in release}) == len(release)
    gpcrmd = next(case for case in release if case.case_id == "gpcrmd-729-pme")
    assert gpcrmd.neighbor_backend == "mlx_interaction32"


def test_md_suite_inventory_reports_prepared_artifact_availability(tmp_path: Path):
    registry_path = tmp_path / "cases.json"
    prepared = tmp_path / "results" / "case" / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "prepared_system.json").write_text("{}")
    (prepared / "prepared_system.npz").write_bytes(b"npz")
    registry_path.write_text(
        json.dumps(
            {
                "schema": "mlx_atomistic.md_suite_cases.v1",
                "suites": {"local": ["case"]},
                "cases": [
                    {
                        "id": "case",
                        "description": "test",
                        "prepared_path": "results/case/prepared",
                        "expected_atom_count": 12,
                        "tier": "local",
                        "role": "improvement",
                        "neighbor_backend": "mlx_cell_tiles",
                        "features": ["pme"],
                        "preparation_command": "uv run prepare",
                    }
                ],
            }
        )
    )

    inventory = case_inventory(repo_root=tmp_path, registry_path=registry_path)

    assert inventory["cases"][0]["available"] is True


def test_md_suite_runner_persists_medians_and_contracts(tmp_path: Path):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        prepared = str(kwargs["prepared"])
        atom_count = 23558 if "dhfr-npt-closure" in prepared else 94232
        repeat = len(calls)
        seconds_per_step = 0.1 + 0.001 * (repeat % 3)
        return {
            "passed": True,
            "blockers": [],
            "atom_count": atom_count,
            "hardware": {"machine": "arm64"},
            "runtime": {"mlx_version": "test", "default_device": "gpu"},
            "neighbor": {"measured_rebuild_wall_seconds": 0.01},
            "timings": {"seconds_per_measured_step": seconds_per_step},
            "throughput": {"ns_per_day": 100.0 / seconds_per_step},
        }

    out = tmp_path / "suite.json"
    payload = run_suite(
        repo_root=Path.cwd(),
        out=out,
        repeats=3,
        warmup_steps=2,
        measured_steps=4,
        runner=fake_runner,
    )

    assert payload["schema"] == SUITE_SCHEMA
    assert payload["passed"] is True
    assert [row["case_id"] for row in payload["cases"]] == [
        "dhfr-5dfr-pme",
        "jac-94k-pme",
    ]
    assert all(row["sample_count"] == 3 for row in payload["cases"])
    assert all(row["rehearsal_passed"] for row in payload["cases"])
    assert all(row["relative_timing_spread"] < 0.10 for row in payload["cases"])
    assert all(row["contract_fingerprint"] for row in payload["cases"])
    assert len(calls) == 8
    assert [call["steps"] for call in calls if call["out"].name == "rehearsal.json"] == [
        75,
        75,
    ]
    assert {call["neighbor_backend"] for call in calls} == {"mlx_interaction32"}
    assert json.loads(out.read_text())["passed"] is True


def test_md_suite_runner_blocks_unstable_timing_samples(tmp_path: Path):
    timings = iter((0.1, 0.1, 0.14, 0.1))

    def fake_runner(**kwargs):
        seconds_per_step = next(timings)
        return {
            "passed": True,
            "blockers": [],
            "atom_count": 23558,
            "hardware": {"machine": "arm64"},
            "runtime": {"mlx_version": "test", "default_device": "gpu"},
            "timings": {"seconds_per_measured_step": seconds_per_step},
            "throughput": {"ns_per_day": 100.0 / seconds_per_step},
        }

    payload = run_suite(
        repo_root=Path.cwd(),
        out=tmp_path / "unstable.json",
        case_ids=("dhfr-5dfr-pme",),
        repeats=3,
        warmup_steps=2,
        measured_steps=4,
        runner=fake_runner,
    )

    assert payload["passed"] is False
    assert payload["cases"][0]["relative_timing_spread"] == pytest.approx(0.4)
    assert any(
        blocker.startswith("timing_spread_exceeded:")
        for blocker in payload["cases"][0]["blockers"]
    )


def test_md_stage_profile_keeps_clean_throughput_separate_from_stage_attribution(
    tmp_path: Path,
):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        instrumented = kwargs["runtime_profile"]
        route_profile = (
            {
                "instrumented_wall_seconds": 10.0,
                "accounted_route_seconds": 9.0,
                "residual_seconds": 1.0,
                "reconciled": True,
                "routes": {
                    "direct_spatial_tiles": {
                        "wall_seconds": 4.0,
                        "completion_seconds": 3.8,
                        "graph_and_host_seconds": 0.2,
                    },
                    "neighbor_update_rebuild": {
                        "wall_seconds": 3.0,
                        "completion_seconds": 2.0,
                        "graph_and_host_seconds": 1.0,
                    },
                    "composite_constraints_position": {
                        "wall_seconds": 2.0,
                        "completion_seconds": 1.8,
                        "graph_and_host_seconds": 0.2,
                    },
                },
            }
            if instrumented
            else {}
        )
        return {
            "passed": True,
            "blockers": [],
            "atom_count": 23558,
            "hardware": {"machine": "arm64"},
            "runtime": {"mlx_version": "test", "default_device": "gpu"},
            "timings": {
                "seconds_per_measured_step": 0.2 if instrumented else 0.1,
            },
            "throughput": {"ns_per_day": 50.0 if instrumented else 100.0},
            "state": {
                "potential_energy_kj_mol": -1.0,
                "kinetic_energy_kj_mol": 1.0,
                "total_energy_kj_mol": 0.0,
                "temperature_k": 300.0,
                "constraint_max_error_angstrom": 1.0e-7,
            },
            "route_profile": route_profile,
        }

    out = tmp_path / "profile.json"
    payload = profile_suite(
        repo_root=Path.cwd(),
        out=out,
        case_ids=("dhfr-5dfr-pme",),
        warmup_steps=2,
        measured_steps=4,
        runner=fake_runner,
    )

    assert payload["schema"] == STAGE_PROFILE_SCHEMA
    assert payload["passed"] is True
    assert [call["runtime_profile"] for call in calls] == [False, True]
    row = payload["cases"][0]
    assert row["clean_seconds_per_step"] == 0.1
    assert row["instrumentation_slowdown_ratio"] == 2.0
    assert row["dominant_stage"] == "direct_nonbonded"
    assert row["stages"][0]["instrumented_wall_fraction"] == pytest.approx(0.4)
    assert payload["cross_case_stage_ranking"][0]["stage"] == "direct_nonbonded"
    assert (
        json.loads(out.read_text())["profile_semantics"][
            "instrumented_preserves_lazy_force_schedule"
        ]
        is False
    )


@pytest.mark.parametrize(
    "route_name",
    ("direct_spatial_tiles", "direct_lj_screened_coulomb", "direct_interaction32"),
)
def test_md_stage_profile_classifies_every_direct_nonbonded_route(route_name):
    assert _route_stage(route_name) == "direct_nonbonded"


def test_md_suite_comparison_requires_5dfr_non_regression_and_jac_improvement():
    baseline = _suite_payload(five_seconds=1.0, jac_seconds=2.0, commit="baseline")
    candidate = _suite_payload(five_seconds=1.01, jac_seconds=1.8, commit="candidate")

    comparison = compare_suites(baseline, candidate)

    assert comparison["passed"] is True
    speedups = {row["case_id"]: row["speedup_fraction"] for row in comparison["cases"]}
    assert speedups["dhfr-5dfr-pme"] == pytest.approx(1.0 / 1.01 - 1.0)
    assert speedups["jac-94k-pme"] == pytest.approx(2.0 / 1.8 - 1.0)

    regressed = _suite_payload(five_seconds=1.1, jac_seconds=1.8, commit="candidate")
    failed = compare_suites(baseline, regressed)
    assert failed["passed"] is False
    assert "dhfr-5dfr-pme:regression" in failed["blockers"]


def test_tip3p_water_builder_is_deterministic_neutral_and_production_admissible(
    tmp_path: Path,
):
    first = build_tip3p_water_box(grid_shape=(2, 2, 2), mesh_shape=(8, 8, 8), seed=7)
    second = build_tip3p_water_box(grid_shape=(2, 2, 2), mesh_shape=(8, 8, 8), seed=7)

    assert first.atom_count == 24
    assert first.constraints.shape == (24, 2)
    assert first.molecule_ids.shape == (24,)
    assert np.array_equal(first.positions, second.positions)
    assert float(np.sum(first.charges, dtype=np.float64)) == pytest.approx(0.0, abs=1e-6)
    assert first.metadata.source["density_g_cm3"] == TIP3P_DENSITY_G_CM3
    assert first.metadata.pme_config["assignment_order"] == 5
    assert first.metadata.pme_config["background_policy"] == "reject_non_neutral"

    artifact_dir = tmp_path / "prepared"
    save_prepared_system(first, artifact_dir)
    artifact = load_prepared_mlx_artifact(artifact_dir, require_production=True)
    assert artifact.atom_count == 24
    assert artifact.molecule_ids.shape == (24,)


def test_apoa1_preparation_contract_uses_official_openmm_fixture():
    spec = prepare_openmm_dhfr_explicit.APOA1_SPEC

    assert spec.case_id == "apoa1-pme"
    assert spec.source_pdb == Path("vendors/openmm/examples/benchmarks/apoa1.pdb")
    assert spec.default_out == Path("results/md-benchmarks/apoa1/prepared")
    assert spec.force_field_files == (
        "amber14/protein.ff14SB.xml",
        "amber14/lipid17.xml",
        "amber14/tip3p.xml",
    )
    assert "TIP" in spec.water_residue_names
    assert "POP" in spec.lipid_residue_names


def _suite_payload(*, five_seconds: float, jac_seconds: float, commit: str) -> dict:
    return {
        "schema": SUITE_SCHEMA,
        "commit": commit,
        "repeats": 3,
        "warmup_steps": 10,
        "measured_steps": 75,
        "rehearsal_steps": 75,
        "maximum_relative_spread": 0.10,
        "neighbor_backend": "mlx_cell_tiles",
        "cases": [
            {
                "case_id": "dhfr-5dfr-pme",
                "passed": True,
                "contract_fingerprint": "five",
                "prepared_fingerprint": "five-input",
                "hardware": {"machine": "arm64"},
                "runtime": {"mlx_version": "test", "default_device": "gpu"},
                "median_seconds_per_step": five_seconds,
            },
            {
                "case_id": "jac-94k-pme",
                "passed": True,
                "contract_fingerprint": "jac",
                "prepared_fingerprint": "jac-input",
                "hardware": {"machine": "arm64"},
                "runtime": {"mlx_version": "test", "default_device": "gpu"},
                "median_seconds_per_step": jac_seconds,
            },
        ],
    }
