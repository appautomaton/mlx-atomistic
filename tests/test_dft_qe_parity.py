from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from mlx_atomistic.benchmarks.dft_silicon import prepare_workload
from mlx_atomistic.benchmarks.dft_silicon_parity import (
    NORMALIZED_UNITS,
    QE_REPORT_SCHEMA,
    compare_silicon_reports,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_qe_silicon_reference.py"
SPEC = importlib.util.spec_from_file_location("run_qe_silicon_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QE_HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QE_HELPER
SPEC.loader.exec_module(QE_HELPER)


def _gth_database(path: Path) -> Path:
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


def _prepared(tmp_path: Path) -> tuple[dict, dict]:
    prepared = prepare_workload(
        gth_source=_gth_database(tmp_path / "GTH_POTENTIALS"),
        out=tmp_path / "workload",
        command=["prepare"],
    )
    manifest = json.loads(Path(prepared["workload_manifest"]).read_text())
    return prepared, manifest


def _qe_output(atom_count: int = 8) -> str:
    forces = "\n".join(
        f"atom {index:4d} type  1   force =  0.00100000  -0.00200000  0.00300000"
        for index in range(1, atom_count + 1)
    )
    return f"""
Program PWSCF v.7.4.1 starts on 16Jul2026
!    total energy              =     -60.00000000 Ry
estimated scf accuracy    <       0.00000001 Ry
convergence has been achieved in  9 iterations
Forces acting on atoms (cartesian axes, Ry/au):
{forces}
total   stress  (Ry/bohr**3)                   (kbar)     P=  14.71
 0.00100000 0.00000000 0.00000000 14.71 0.00 0.00
 0.00000000 0.00200000 0.00000000 0.00 29.42 0.00
 0.00000000 0.00000000 0.00300000 0.00 0.00 44.13
JOB DONE.
"""


def test_qe_input_uses_explicit_matching_kpoints_fft_and_gth(tmp_path):
    prepared, manifest = _prepared(tmp_path)
    settings = QE_HELPER.qe_settings_from_manifest(manifest)
    lattice = manifest["system"]["lattice_constant_bohr"]
    positions = lattice * np.asarray(manifest["system"]["fractional_positions"])

    payload = QE_HELPER.render_qe_input(
        manifest=manifest,
        settings=settings,
        lattice_bohr=lattice,
        positions_bohr=positions,
        pseudopotential=Path(prepared["gth_path"]),
        scratch=tmp_path / "scratch",
        prefix="silicon_equilibrium",
    )

    assert "nosym = .true." in payload
    assert "noinv = .true." in payload
    assert "ecutwfc = 20" in payload
    assert "nr1 = 32, nr2 = 32, nr3 = 32" in payload
    assert "K_POINTS crystal\n8\n" in payload
    assert "-0.25 -0.25 -0.25 0.125" in payload
    assert Path(prepared["gth_path"]).name in payload


def test_qe_output_parser_requires_complete_energy_force_stress_and_convergence():
    parsed = QE_HELPER.parse_qe_output(_qe_output(), atom_count=8)

    assert parsed["qe_version"] == "7.4.1"
    assert parsed["converged"] is True
    assert parsed["complete"] is True
    assert parsed["total_energy_hartree"] == -30.0
    np.testing.assert_allclose(
        parsed["forces_hartree_per_bohr"][0],
        [0.0005, -0.001, 0.0015],
    )
    np.testing.assert_allclose(
        np.diag(parsed["stress_gpa"]),
        [14.710513242194795, 29.42102648438959, 44.131539726584385],
    )


def test_missing_pw_x_writes_concrete_blocked_reference_report(tmp_path):
    prepared, _ = _prepared(tmp_path)
    out = tmp_path / "qe-blocked"

    report = QE_HELPER.run_qe_reference(
        pw_x=tmp_path / "missing-pw.x",
        manifest_path=prepared["workload_manifest"],
        gth_path=prepared["gth_path"],
        out=out,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["pw_x_not_found"]
    assert report["product_runtime_boundary"]["package_dependency"] is False
    persisted = json.loads((out / "reference-report.json").read_text())
    assert persisted == report


def _matching_reports(manifest: dict) -> tuple[dict, dict]:
    settings = QE_HELPER.qe_settings_from_manifest(manifest)
    energy = -30.0
    forces = np.zeros((8, 3)).tolist()
    stress_minus = np.diag([-1.0, -1.0, -1.0]).tolist()
    stress_plus = np.diag([1.0, 1.0, 1.0]).tolist()
    fit = {
        "status": "ok",
        "equilibrium_lattice_constant_angstrom": 5.43,
        "observed_minimum_index": 3,
    }
    mlx = {
        "target_id": manifest["target_id"],
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "status": "ok",
        "comparison_status": "comparable",
        "settings": settings,
        "cases": {
            "equilibrium": {
                "complete": True,
                "repetitions": [{"result": {"total_energy_hartree": energy}}],
            },
            "displaced_atom": {
                "complete": True,
                "base": {"result": {"total_energy_hartree": energy}},
                "forces_hartree_per_bohr": forces,
            },
            "strain_minus": {
                "complete": True,
                "base": {"result": {"total_energy_hartree": energy}},
                "stress_gpa": stress_minus,
            },
            "strain_plus": {
                "complete": True,
                "base": {"result": {"total_energy_hartree": energy}},
                "stress_gpa": stress_plus,
            },
            "volume_scan": {"complete": True, "fit": fit},
        },
    }
    qe = {
        "schema_version": QE_REPORT_SCHEMA,
        "target_id": manifest["target_id"],
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "pseudopotential_sha256": manifest["pseudopotential"]["sha256"],
        "status": "ran",
        "complete": True,
        "normalized_units": NORMALIZED_UNITS,
        "settings": json.loads(json.dumps(settings)),
        "cases": {
            "equilibrium": {
                "complete": True,
                "converged": True,
                "total_energy_hartree": energy,
            },
            "displaced_atom": {
                "complete": True,
                "converged": True,
                "total_energy_hartree": energy,
                "forces_hartree_per_bohr": forces,
            },
            "strain_minus": {
                "complete": True,
                "converged": True,
                "total_energy_hartree": energy,
                "stress_gpa": stress_minus,
            },
            "strain_plus": {
                "complete": True,
                "converged": True,
                "total_energy_hartree": energy,
                "stress_gpa": stress_plus,
            },
            "volume_scan": {"complete": True, "fit": fit},
        },
    }
    return mlx, qe


def test_strict_comparator_passes_complete_matching_normalized_reports(tmp_path):
    prepared, manifest = _prepared(tmp_path)
    mlx, qe = _matching_reports(manifest)
    mlx_path = tmp_path / "mlx.json"
    qe_path = tmp_path / "qe.json"
    out = tmp_path / "comparison.json"
    mlx_path.write_text(json.dumps(mlx))
    qe_path.write_text(json.dumps(qe))

    report = compare_silicon_reports(
        manifest_path=prepared["workload_manifest"],
        mlx_report_path=mlx_path,
        qe_report_path=qe_path,
        out=out,
    )

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["metrics"]["force"]["component_count"] == 24
    assert json.loads(out.read_text()) == report


def test_strict_comparator_blocks_settings_mismatch_before_metrics(tmp_path):
    prepared, manifest = _prepared(tmp_path)
    mlx, qe = _matching_reports(manifest)
    mlx["settings"]["cutoff_hartree"] = 15.0
    mlx_path = tmp_path / "mlx.json"
    qe_path = tmp_path / "qe.json"
    mlx_path.write_text(json.dumps(mlx))
    qe_path.write_text(json.dumps(qe))

    report = compare_silicon_reports(
        manifest_path=prepared["workload_manifest"],
        mlx_report_path=mlx_path,
        qe_report_path=qe_path,
        out=tmp_path / "comparison.json",
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["numerical_settings_mismatch"]
    assert report["metrics"] == {}


def test_strict_comparator_rejects_partial_force_arrays(tmp_path):
    prepared, manifest = _prepared(tmp_path)
    mlx, qe = _matching_reports(manifest)
    qe["cases"]["displaced_atom"]["forces_hartree_per_bohr"] = [[0.0, 0.0, 0.0]]
    mlx_path = tmp_path / "mlx.json"
    qe_path = tmp_path / "qe.json"
    mlx_path.write_text(json.dumps(mlx))
    qe_path.write_text(json.dumps(qe))

    report = compare_silicon_reports(
        manifest_path=prepared["workload_manifest"],
        mlx_report_path=mlx_path,
        qe_report_path=qe_path,
        out=tmp_path / "comparison.json",
    )

    assert report["status"] == "blocked"
    assert report["blockers"][0].startswith("normalized_payload_invalid:")
    assert report["metrics"] == {}
