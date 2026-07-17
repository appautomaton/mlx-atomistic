"""Strict normalized-report comparison for the bounded silicon DFT workload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

QE_REPORT_SCHEMA = "mlx-atomistic.dft-silicon-qe-reference.v1"
COMPARISON_SCHEMA = "mlx-atomistic.dft-silicon-comparison.v1"
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.422067476325886
NORMALIZED_UNITS = {
    "energy": "hartree",
    "force": "hartree/bohr",
    "stress": "gigapascal",
    "lattice_constant": "angstrom",
}


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        msg = f"expected a JSON object: {path}"
        raise ValueError(msg)
    return payload


def _settings_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "cutoff_hartree": float(payload["cutoff_hartree"]),
        "fft_shape": [int(value) for value in payload["fft_shape"]],
        "kpoint_mesh": [int(value) for value in payload["kpoint_mesh"]],
    }


def _case_energy(case: dict[str, Any], case_id: str) -> float:
    if case_id == "equilibrium":
        repetitions = case.get("repetitions", [])
        if not repetitions:
            raise ValueError("MLX equilibrium case has no repetitions")
        return float(repetitions[0]["result"]["total_energy_hartree"])
    return float(case["base"]["result"]["total_energy_hartree"])


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        msg = f"{name} has shape {array.shape}; expected {shape}"
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)
    return array


def _preflight_blockers(
    manifest: dict[str, Any],
    mlx: dict[str, Any],
    qe: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    fingerprint_payload = {
        key: value for key, value in manifest.items() if key != "fingerprint_sha256"
    }
    if _sha256_bytes(_canonical_json(fingerprint_payload)) != manifest.get("fingerprint_sha256"):
        blockers.append("manifest_fingerprint_invalid")
    pseudo_path = Path(manifest["pseudopotential"]["path"])
    if not pseudo_path.is_file():
        blockers.append("manifest_pseudopotential_missing")
    elif _sha256_bytes(pseudo_path.read_bytes()) != manifest["pseudopotential"]["sha256"]:
        blockers.append("manifest_pseudopotential_hash_invalid")
    expected_cases = set(manifest["cases"])
    if mlx.get("target_id") != manifest["target_id"]:
        blockers.append("mlx_target_id_mismatch")
    if qe.get("target_id") != manifest["target_id"]:
        blockers.append("qe_target_id_mismatch")
    fingerprint = manifest["fingerprint_sha256"]
    if mlx.get("manifest_fingerprint") != fingerprint:
        blockers.append("mlx_manifest_fingerprint_mismatch")
    if qe.get("manifest_fingerprint") != fingerprint:
        blockers.append("qe_manifest_fingerprint_mismatch")
    pseudo_hash = manifest["pseudopotential"]["sha256"]
    if qe.get("pseudopotential_sha256") != pseudo_hash:
        blockers.append("qe_pseudopotential_hash_mismatch")
    if mlx.get("status") != "ok":
        blockers.append("mlx_report_not_ok")
    if mlx.get("comparison_status") != "comparable":
        blockers.append("mlx_report_not_comparable")
    if qe.get("schema_version") != QE_REPORT_SCHEMA:
        blockers.append("qe_report_schema_mismatch")
    if qe.get("status") != "ran":
        blockers.append("qe_reference_not_ran")
    if qe.get("complete") is not True:
        blockers.append("qe_reference_incomplete")
    if qe.get("normalized_units") != NORMALIZED_UNITS:
        blockers.append("qe_normalized_units_mismatch")
    if set(mlx.get("cases", {})) != expected_cases:
        blockers.append("mlx_case_set_mismatch")
    if set(qe.get("cases", {})) != expected_cases:
        blockers.append("qe_case_set_mismatch")
    try:
        if _settings_identity(mlx["settings"]) != _settings_identity(qe["settings"]):
            blockers.append("numerical_settings_mismatch")
    except (KeyError, TypeError, ValueError):
        blockers.append("numerical_settings_incomplete")
    for case_id in sorted(expected_cases):
        mlx_case = mlx.get("cases", {}).get(case_id, {})
        qe_case = qe.get("cases", {}).get(case_id, {})
        if mlx_case.get("complete") is not True:
            blockers.append(f"mlx_case_incomplete:{case_id}")
        if qe_case.get("complete") is not True:
            blockers.append(f"qe_case_incomplete:{case_id}")
        if case_id == "volume_scan":
            if qe_case.get("fit", {}).get("status") != "ok":
                blockers.append("qe_lattice_fit_not_ok")
            continue
        if qe_case.get("converged") is not True:
            blockers.append(f"qe_case_unconverged:{case_id}")
    return sorted(set(blockers))


def compare_silicon_reports(
    *,
    manifest_path: str | Path,
    mlx_report_path: str | Path,
    qe_report_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Compare complete normalized MLX and QE silicon reports.

    Args:
        manifest_path: Canonical prepared workload manifest.
        mlx_report_path: Complete MLX workload report.
        qe_report_path: Fresh normalized QE reference report.
        out: Comparison JSON output path.

    Returns:
        Strict blocked, failed, or passed comparison payload.
    """

    manifest = _read_json(manifest_path)
    mlx = _read_json(mlx_report_path)
    qe = _read_json(qe_report_path)
    blockers = _preflight_blockers(manifest, mlx, qe)
    base = {
        "schema_version": COMPARISON_SCHEMA,
        "target_id": manifest["target_id"],
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "pseudopotential_sha256": manifest["pseudopotential"]["sha256"],
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_bytes(Path(manifest_path).read_bytes()),
            "mlx_report": str(mlx_report_path),
            "mlx_report_sha256": _sha256_bytes(Path(mlx_report_path).read_bytes()),
            "qe_report": str(qe_report_path),
            "qe_report_sha256": _sha256_bytes(Path(qe_report_path).read_bytes()),
        },
        "tolerances": dict(manifest["comparison_tolerances"]),
        "normalized_units": dict(NORMALIZED_UNITS),
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if blockers:
        report = {**base, "status": "blocked", "blockers": blockers, "metrics": {}}
        output.write_bytes(_canonical_json(report))
        return report

    try:
        atom_count = int(manifest["system"]["atom_count"])
        energy_rows = []
        for case_id in ("equilibrium", "displaced_atom", "strain_minus", "strain_plus"):
            mlx_energy = _case_energy(mlx["cases"][case_id], case_id)
            qe_energy = float(qe["cases"][case_id]["total_energy_hartree"])
            delta = abs(mlx_energy - qe_energy) * HARTREE_TO_EV * 1000.0 / atom_count
            energy_rows.append(
                {
                    "case_id": case_id,
                    "mlx_total_energy_hartree": mlx_energy,
                    "qe_total_energy_hartree": qe_energy,
                    "absolute_error_mev_per_atom": delta,
                }
            )

        force_shape = (atom_count, 3)
        mlx_forces = _finite_array(
            mlx["cases"]["displaced_atom"]["forces_hartree_per_bohr"],
            force_shape,
            "MLX displaced force array",
        )
        qe_forces = _finite_array(
            qe["cases"]["displaced_atom"]["forces_hartree_per_bohr"],
            force_shape,
            "QE displaced force array",
        )
        force_delta = (mlx_forces - qe_forces) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        force_rms = float(np.sqrt(np.mean(force_delta * force_delta)))
        force_max = float(np.max(np.abs(force_delta)))

        stress_rows = []
        for case_id in ("strain_minus", "strain_plus"):
            mlx_stress = _finite_array(
                mlx["cases"][case_id]["stress_gpa"],
                (3, 3),
                f"MLX {case_id} stress",
            )
            qe_stress = _finite_array(
                qe["cases"][case_id]["stress_gpa"],
                (3, 3),
                f"QE {case_id} stress",
            )
            stress_rows.append(
                {
                    "case_id": case_id,
                    "maximum_component_error_gpa": float(np.max(np.abs(mlx_stress - qe_stress))),
                }
            )

        mlx_fit = mlx["cases"]["volume_scan"]["fit"]
        qe_fit = qe["cases"]["volume_scan"]["fit"]
        mlx_lattice = float(mlx_fit["equilibrium_lattice_constant_angstrom"])
        qe_lattice = float(qe_fit["equilibrium_lattice_constant_angstrom"])
        lattice_relative = abs(mlx_lattice - qe_lattice) / abs(qe_lattice)
        minimum_ordering_match = int(mlx_fit["observed_minimum_index"]) == int(
            qe_fit["observed_minimum_index"]
        )
    except (KeyError, TypeError, ValueError) as error:
        report = {
            **base,
            "status": "blocked",
            "blockers": [f"normalized_payload_invalid:{error}"],
            "metrics": {},
        }
        output.write_bytes(_canonical_json(report))
        return report

    tolerances = manifest["comparison_tolerances"]
    max_energy = max(row["absolute_error_mev_per_atom"] for row in energy_rows)
    max_stress = max(row["maximum_component_error_gpa"] for row in stress_rows)
    numerical_blockers = []
    if max_energy > float(tolerances["energy_mev_per_atom"]):
        numerical_blockers.append("energy_tolerance_exceeded")
    if force_rms > float(tolerances["force_rms_ev_per_angstrom"]):
        numerical_blockers.append("force_rms_tolerance_exceeded")
    if force_max > float(tolerances["force_max_component_ev_per_angstrom"]):
        numerical_blockers.append("force_max_component_tolerance_exceeded")
    if max_stress > float(tolerances["stress_max_component_gpa"]):
        numerical_blockers.append("stress_tolerance_exceeded")
    if lattice_relative > float(tolerances["lattice_constant_relative"]):
        numerical_blockers.append("lattice_constant_tolerance_exceeded")
    if not minimum_ordering_match:
        numerical_blockers.append("lattice_minimum_ordering_mismatch")

    metrics = {
        "energy": {
            "cases": energy_rows,
            "maximum_absolute_error_mev_per_atom": max_energy,
        },
        "force": {
            "rms_error_ev_per_angstrom": force_rms,
            "maximum_component_error_ev_per_angstrom": force_max,
            "component_count": int(force_delta.size),
        },
        "stress": {
            "cases": stress_rows,
            "maximum_component_error_gpa": max_stress,
        },
        "lattice": {
            "mlx_equilibrium_lattice_constant_angstrom": mlx_lattice,
            "qe_equilibrium_lattice_constant_angstrom": qe_lattice,
            "relative_error": lattice_relative,
            "minimum_ordering_match": minimum_ordering_match,
        },
    }
    report = {
        **base,
        "status": "passed" if not numerical_blockers else "failed",
        "blockers": numerical_blockers,
        "metrics": metrics,
    }
    output.write_bytes(_canonical_json(report))
    return report
