import hashlib
import json
from pathlib import Path

import pytest

from mlx_atomistic.benchmarks.dhfr_npt import (
    FINAL_TARGET_SCOPE,
    DHFRNPTValidationError,
    build_final_report,
    build_stage_report,
    contract_fingerprint,
    load_completed_stage,
    load_validation_contract,
    stage_report_path,
    validate_source_manifest,
    write_stage_report_atomic,
)


def _source_manifest(contract):
    target = contract["target"]
    workload = contract["workload"]
    return {
        "schema": "mlx-atomistic.openmm-5dfr-preparation.v1",
        "case_id": target["case_id"],
        "source": {
            "pdb": {
                "path": target["pdb_path"],
                "sha256": target["pdb_sha256"],
            },
            "vendor_benchmark": {
                "path": target["vendor_benchmark_path"],
                "sha256": target["vendor_benchmark_sha256"],
            },
            "forcefield_resources": [
                {
                    "resource": f"openmm.app/data/{name}",
                    "sha256": digest,
                }
                for name, digest in target["forcefield_resources"].items()
            ],
        },
        "identity": {
            "atom_count": target["atom_count"],
            "molecule_count": target["molecule_count"],
            "molecule_identity_sha256": target["molecule_identity_sha256"],
            "atom_order_sha256": target["atom_order_sha256"],
        },
        "forces": {
            "classes_sha256": target["force_classes_sha256"],
            "parameters_sha256": target["parameters_sha256"],
            "constraints_sha256": target["constraints_sha256"],
        },
        "construction": {
            "forcefield_files": ["amber99sb.xml", "tip3p.xml"],
            "nonbonded_method": "PME",
            "constraints": "HBonds",
            "rigid_water": True,
            "hydrogen_mass_amu": workload["hydrogen_mass_amu"],
        },
        "pme": dict(workload["pme"]),
    }


def _source_identity(contract):
    return {
        "case_id": contract["target"]["case_id"],
        "manifest_fingerprint": "a" * 64,
        "contract_fingerprint": contract_fingerprint(contract),
        "atom_count": contract["target"]["atom_count"],
        "molecule_count": contract["target"]["molecule_count"],
        "molecule_identity_sha256": contract["target"][
            "molecule_identity_sha256"
        ],
        "source_checks": {"exact_source": True},
    }


def _artifact_record(path):
    return {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _stage_report(tmp_path, contract, *, stage, seed=None, scope=FINAL_TARGET_SCOPE):
    path = stage_report_path(tmp_path, stage=stage, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = path.parent / "evidence.npz"
    artifact.write_bytes(b"bounded evidence")
    report = build_stage_report(
        contract=contract,
        source_identity=_source_identity(contract),
        stage=stage,
        seed=seed,
        scope=scope,
        evidence={"artifacts": {"evidence.npz": _artifact_record(artifact)}},
        checks={"finite": True, "scientific_gate": True},
    )
    return path, report


def test_frozen_contract_declares_exact_target_and_resource_bound():
    contract = load_validation_contract()

    assert contract["schema"] == "mlx-atomistic.dhfr-npt-validation-contract.v1"
    assert contract["target"]["atom_count"] == 23558
    assert contract["target"]["molecule_count"] == 7024
    assert contract["workload"]["seeds"] == [7, 19]
    assert contract["workload"]["barostat"] == {
        "family": "monte_carlo",
        "mode": "anisotropic",
        "interval": 25,
        "axes": [True, True, True],
        "max_log_volume_scale": 0.02,
        "expected_attempts": 10,
    }
    assert contract["resource_limits"]["process_tree_max_bytes"] == 40_000_000_000
    assert len(contract_fingerprint(contract)) == 64


def test_exact_source_manifest_accepts_5dfr_and_rejects_jac_substitution():
    contract = load_validation_contract()
    manifest = _source_manifest(contract)

    assert all(validate_source_manifest(manifest, contract).values())

    manifest["source"]["pdb"]["path"] = "results/inputs/JAC.inpcrd"
    with pytest.raises(DHFRNPTValidationError, match="pdb_path"):
        validate_source_manifest(manifest, contract)


def test_completed_stage_is_reused_only_with_matching_artifacts(tmp_path):
    contract = load_validation_contract()
    identity = _source_identity(contract)
    path, report = _stage_report(tmp_path, contract, stage="fixed")
    write_stage_report_atomic(path, report)

    loaded = load_completed_stage(
        path,
        contract=contract,
        source_identity=identity,
        stage="fixed",
        seed=None,
    )
    assert loaded == report

    (path.parent / "evidence.npz").write_bytes(b"tampered")
    with pytest.raises(DHFRNPTValidationError, match="byte size mismatch"):
        load_completed_stage(
            path,
            contract=contract,
            source_identity=identity,
            stage="fixed",
            seed=None,
        )


def test_incomplete_or_cross_contract_stage_fails_closed(tmp_path):
    contract = load_validation_contract()
    identity = _source_identity(contract)
    path = stage_report_path(tmp_path, stage="npt", seed=7)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema":')

    with pytest.raises(DHFRNPTValidationError, match="incomplete or corrupt"):
        load_completed_stage(
            path,
            contract=contract,
            source_identity=identity,
            stage="npt",
            seed=7,
        )

    path, report = _stage_report(tmp_path / "other", contract, stage="npt", seed=7)
    report["contract_fingerprint"] = "b" * 64
    report["source_identity"]["contract_fingerprint"] = "b" * 64
    write_stage_report_atomic(path, report)
    with pytest.raises(DHFRNPTValidationError, match="contract_fingerprint"):
        load_completed_stage(
            path,
            contract=contract,
            source_identity=identity,
            stage="npt",
            seed=7,
        )


def test_final_report_refuses_fixture_or_mixed_source_evidence(tmp_path):
    contract = load_validation_contract()
    identity = _source_identity(contract)
    _, fixed = _stage_report(tmp_path / "fixed", contract, stage="fixed")
    _, seed7 = _stage_report(tmp_path / "seed7", contract, stage="npt", seed=7)
    _, seed19 = _stage_report(tmp_path / "seed19", contract, stage="npt", seed=19)

    final = build_final_report(
        contract=contract,
        source_identity=identity,
        fixed_report=fixed,
        npt_reports=[seed7, seed19],
        checks={"all_stages": True},
        evidence={"bounded": True},
    )
    assert final["status"] == "passed"

    fixture = dict(seed19)
    fixture["scope"] = "small-fixture-not-final-target"
    fixture["final_target"] = False
    with pytest.raises(DHFRNPTValidationError, match="final-target"):
        build_final_report(
            contract=contract,
            source_identity=identity,
            fixed_report=fixed,
            npt_reports=[seed7, fixture],
            checks={"all_stages": True},
            evidence={"bounded": True},
        )


def test_staged_runner_cli_requires_declared_stage_seed_pair():
    from scripts import run_openmm_mlx_dhfr_npt as runner

    fixed = runner._parse_args(
        ["--stage", "fixed", "--prepared", "prepared", "--out", "out"]
    )
    npt = runner._parse_args(
        [
            "--stage",
            "npt",
            "--seed",
            "7",
            "--prepared",
            "prepared",
            "--out",
            "out",
        ]
    )

    assert fixed.seed is None
    assert npt.seed == 7
    with pytest.raises(SystemExit):
        runner._parse_args(
            ["--stage", "npt", "--prepared", "prepared", "--out", "out"]
        )


@pytest.mark.reference
def test_small_periodic_fixture_exercises_report_path_without_target_claim():
    from scripts import openmm_mlx_parity as module

    observed = module.evaluate_periodic_nonbonded_virial_parity()
    contract = load_validation_contract()
    report = build_stage_report(
        contract=contract,
        source_identity=_source_identity(contract),
        stage="fixed",
        seed=None,
        scope="small-periodic-pme-fixture-not-final-target",
        evidence={"fixture": observed},
        checks={
            "energy": observed["total_energy_abs_error_kj_mol"] < 5.0e-3,
            "force": observed["force_max_abs_error_kj_mol_nm"] < 2.0e-2,
            "virial": observed["virial_max_abs_error_kj_mol"] < 5.0e-3,
            "pressure": (
                observed["pressure_max_abs_error_kj_mol_a3"] < 1.0e-5
            ),
        },
    )
    assert report["status"] == "passed"
    assert report["final_target"] is False


@pytest.mark.reference
def test_jac_evidence_is_admitted_only_as_non_target_smell_case():
    evidence_path = Path(
        "results/scalable-charged-pme-runtime/jac-1x/"
        "charged_pme_parity_report.json"
    )
    if not evidence_path.is_file():
        pytest.skip("local JAC parity evidence is unavailable")
    observed = json.loads(evidence_path.read_text())
    contract = load_validation_contract()
    report = build_stage_report(
        contract=contract,
        source_identity=_source_identity(contract),
        stage="fixed",
        seed=None,
        scope="amber20-jac-smell-evidence-not-final-target",
        evidence={
            "fixture": observed["fixture"],
            "atom_count": observed["atom_count"],
            "checks": observed["checks"],
        },
        checks={
            "prior_parity_passed": observed["passed"] is True,
            "not_5dfr_source": observed["fixture"] != contract["target"]["case_id"],
        },
    )
    assert report["status"] == "passed"
    assert report["final_target"] is False
