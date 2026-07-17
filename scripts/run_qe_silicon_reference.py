"""Run the bounded bulk-silicon workload with reference-only Quantum ESPRESSO."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.benchmarks.dft_silicon import (
    ANGSTROM_TO_BOHR,
    fit_lattice_curve,
    inspect_workload,
)
from mlx_atomistic.benchmarks.dft_silicon_parity import (
    NORMALIZED_UNITS,
    QE_REPORT_SCHEMA,
)

RYDBERG_TO_HARTREE = 0.5
RYDBERG_PER_BOHR3_TO_GPA = 14710.513242194795
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fortran_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def qe_settings_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic QE settings selected from the workload manifest.

    Args:
        manifest: Prepared silicon workload manifest.

    Returns:
        JSON-safe cutoff, FFT, k-point, SCF, and QE-unit settings.
    """

    numerics = manifest["numerics"]
    selected = numerics.get("selected", {})
    cutoff = float(selected.get("cutoff_hartree", numerics["kinetic_cutoff_candidates_hartree"][0]))
    fft_shape = [
        int(value) for value in selected.get("fft_shape", numerics["fft_shape_candidates"][0])
    ]
    kpoint_mesh = [
        int(value) for value in selected.get("kpoint_mesh", numerics["kpoint_mesh_candidates"][0])
    ]
    ecutwfc = 2.0 * cutoff
    return {
        "selection_status": "selected" if selected else "first_manifest_candidate",
        "cutoff_hartree": cutoff,
        "fft_shape": fft_shape,
        "kpoint_mesh": kpoint_mesh,
        "ecutwfc_rydberg": ecutwfc,
        "ecutrho_rydberg": 4.0 * ecutwfc,
        "conv_thr_rydberg": min(
            1e-10,
            2.0
            * float(numerics["scf"]["energy_tolerance_hartree_per_atom"])
            * int(manifest["system"]["atom_count"]),
        ),
        "electron_maxstep": int(numerics["scf"]["max_iterations"]),
        "mixing_beta": float(numerics["scf"]["mixing_beta"]),
        "occupations": "fixed",
        "input_dft": "PBE",
        "nosym": True,
        "noinv": True,
    }


def _kpoints(mesh: list[int]) -> list[tuple[float, float, float, float]]:
    total = math.prod(mesh)
    points = []
    for indices in itertools.product(*(range(count) for count in mesh)):
        vector = tuple(
            (index - (count - 1) / 2.0) / count for index, count in zip(indices, mesh, strict=True)
        )
        points.append((*vector, 1.0 / total))
    return points


def render_qe_input(
    *,
    manifest: dict[str, Any],
    settings: dict[str, Any],
    lattice_bohr: float,
    positions_bohr: np.ndarray,
    pseudopotential: Path,
    scratch: Path,
    prefix: str,
) -> str:
    """Render one deterministic QE PWscf input.

    Args:
        manifest: Prepared silicon workload manifest.
        settings: Normalized settings from :func:`qe_settings_from_manifest`.
        lattice_bohr: Cubic lattice constant in bohr.
        positions_bohr: Complete Cartesian atom positions in bohr.
        pseudopotential: Exact extracted GTH path.
        scratch: Caller-owned QE scratch directory.
        prefix: Filesystem-safe QE calculation prefix.

    Returns:
        Complete PWscf input text.
    """

    atom_count = int(manifest["system"]["atom_count"])
    positions = np.asarray(positions_bohr, dtype=np.float64)
    if positions.shape != (atom_count, 3) or not np.isfinite(positions).all():
        msg = f"positions must have finite shape ({atom_count}, 3)"
        raise ValueError(msg)
    fft = settings["fft_shape"]
    lines = [
        "&CONTROL",
        "  calculation = 'scf',",
        f"  prefix = '{_fortran_string(prefix)}',",
        f"  pseudo_dir = '{_fortran_string(pseudopotential.parent.resolve())}',",
        f"  outdir = '{_fortran_string(scratch.resolve())}',",
        "  restart_mode = 'from_scratch',",
        "  disk_io = 'none',",
        "  verbosity = 'high',",
        "  tprnfor = .true.,",
        "  tstress = .true.,",
        "/",
        "&SYSTEM",
        "  ibrav = 0,",
        f"  nat = {atom_count},",
        "  ntyp = 1,",
        f"  nbnd = {int(manifest['system']['occupied_band_count'])},",
        f"  ecutwfc = {settings['ecutwfc_rydberg']:.12g},",
        f"  ecutrho = {settings['ecutrho_rydberg']:.12g},",
        f"  nr1 = {fft[0]}, nr2 = {fft[1]}, nr3 = {fft[2]},",
        f"  nr1s = {fft[0]}, nr2s = {fft[1]}, nr3s = {fft[2]},",
        "  nspin = 1,",
        f"  occupations = '{settings['occupations']}',",
        f"  input_dft = '{settings['input_dft']}',",
        "  nosym = .true.,",
        "  noinv = .true.,",
        "/",
        "&ELECTRONS",
        f"  conv_thr = {settings['conv_thr_rydberg']:.12g},",
        f"  electron_maxstep = {settings['electron_maxstep']},",
        "  diagonalization = 'david',",
        "  mixing_mode = 'plain',",
        f"  mixing_beta = {settings['mixing_beta']:.12g},",
        "  startingpot = 'atomic',",
        "  startingwfc = 'random',",
        "/",
        "ATOMIC_SPECIES",
        f"Si 28.0855 {pseudopotential.name}",
        "CELL_PARAMETERS bohr",
        f"{lattice_bohr:.15g} 0.0 0.0",
        f"0.0 {lattice_bohr:.15g} 0.0",
        f"0.0 0.0 {lattice_bohr:.15g}",
        "ATOMIC_POSITIONS bohr",
    ]
    lines.extend(f"Si {row[0]:.15g} {row[1]:.15g} {row[2]:.15g}" for row in positions)
    points = _kpoints(settings["kpoint_mesh"])
    lines.extend(["K_POINTS crystal", str(len(points))])
    lines.extend(f"{x:.15g} {y:.15g} {z:.15g} {weight:.15g}" for x, y, z, weight in points)
    return "\n".join(lines) + "\n"


def parse_qe_output(output: str, *, atom_count: int) -> dict[str, Any]:
    """Parse one complete PWscf SCF output into normalized units.

    Args:
        output: PWscf standard output text.
        atom_count: Exact expected atom count.

    Returns:
        Normalized version, convergence, energy, force, and stress payload.

    Raises:
        ValueError: If required complete evidence is absent or non-finite.
    """

    version_matches = re.findall(r"Program\s+PWSCF\s+v\.([^\s]+)", output)
    energy_matches = re.findall(
        rf"^\s*!\s+total energy\s*=\s*({_NUMBER})\s+Ry",
        output,
        flags=re.MULTILINE,
    )
    force_matches = re.findall(
        rf"atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*"
        rf"({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})",
        output,
    )
    accuracy_matches = re.findall(
        rf"estimated scf accuracy\s*<\s*({_NUMBER})\s+Ry",
        output,
    )
    iteration_matches = re.findall(
        r"convergence has been achieved in\s+(\d+)\s+iterations?",
        output,
    )
    stress_header = list(re.finditer(r"total\s+stress\s+\(Ry/bohr\*\*3\)", output))
    stress_rows: list[list[float]] = []
    if stress_header:
        tail = output[stress_header[-1].end() :]
        for line in tail.splitlines():
            values = re.findall(_NUMBER, line)
            if len(values) >= 6:
                stress_rows.append([_float(value) for value in values[:3]])
                if len(stress_rows) == 3:
                    break
            elif stress_rows:
                break
    if not version_matches:
        raise ValueError("QE version is missing")
    if not energy_matches:
        raise ValueError("QE total energy is missing")
    if "convergence has been achieved" not in output or "convergence NOT achieved" in output:
        raise ValueError("QE SCF did not converge")
    if "JOB DONE." not in output:
        raise ValueError("QE job completion marker is missing")
    if len(force_matches) < atom_count:
        raise ValueError("QE complete force array is missing")
    forces = np.asarray(
        [[_float(value) for value in row[1:]] for row in force_matches[-atom_count:]],
        dtype=np.float64,
    )
    atom_indices = [int(row[0]) for row in force_matches[-atom_count:]]
    if atom_indices != list(range(1, atom_count + 1)):
        raise ValueError("QE force atom ordering is incomplete")
    stress = np.asarray(stress_rows, dtype=np.float64)
    if stress.shape != (3, 3):
        raise ValueError("QE complete stress tensor is missing")
    energy_hartree = _float(energy_matches[-1]) * RYDBERG_TO_HARTREE
    forces_hartree_per_bohr = forces * RYDBERG_TO_HARTREE
    stress_gpa = stress * RYDBERG_PER_BOHR3_TO_GPA
    if not (
        np.isfinite(energy_hartree)
        and np.isfinite(forces_hartree_per_bohr).all()
        and np.isfinite(stress_gpa).all()
    ):
        raise ValueError("QE output contains non-finite normalized values")
    return {
        "qe_version": version_matches[-1],
        "converged": True,
        "complete": True,
        "iterations": int(iteration_matches[-1]) if iteration_matches else None,
        "estimated_scf_accuracy_rydberg": (
            _float(accuracy_matches[-1]) if accuracy_matches else None
        ),
        "total_energy_hartree": energy_hartree,
        "forces_hartree_per_bohr": forces_hartree_per_bohr.tolist(),
        "stress_gpa": stress_gpa.tolist(),
    }


def _geometry_cases(manifest: dict[str, Any]) -> dict[str, Any]:
    fractional = np.asarray(manifest["system"]["fractional_positions"], dtype=np.float64)
    equilibrium_angstrom = float(manifest["system"]["lattice_constant_angstrom"])
    equilibrium_bohr = equilibrium_angstrom * ANGSTROM_TO_BOHR
    equilibrium_positions = equilibrium_bohr * fractional
    displaced = equilibrium_positions.copy()
    displaced_case = manifest["cases"]["displaced_atom"]
    displaced[int(displaced_case["atom_index"]), int(displaced_case["axis"])] += (
        float(displaced_case["offset_angstrom"]) * ANGSTROM_TO_BOHR
    )
    cases: dict[str, Any] = {
        "equilibrium": {
            "lattice_bohr": equilibrium_bohr,
            "positions_bohr": equilibrium_positions,
        },
        "displaced_atom": {
            "lattice_bohr": equilibrium_bohr,
            "positions_bohr": displaced,
        },
    }
    for case_id in ("strain_minus", "strain_plus"):
        strain = float(manifest["cases"][case_id]["isotropic_strain"])
        lattice = equilibrium_bohr * (1.0 + strain)
        cases[case_id] = {
            "lattice_bohr": lattice,
            "positions_bohr": lattice * fractional,
        }
    cases["volume_scan"] = [
        {
            "lattice_constant_angstrom": float(lattice_angstrom),
            "lattice_bohr": float(lattice_angstrom) * ANGSTROM_TO_BOHR,
            "positions_bohr": float(lattice_angstrom) * ANGSTROM_TO_BOHR * fractional,
        }
        for lattice_angstrom in manifest["cases"]["volume_scan"]["lattice_constants_angstrom"]
    ]
    return cases


def _resolve_executable(value: str | Path) -> Path | None:
    raw = str(value)
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(raw)
    if resolved is None:
        return None
    return Path(resolved).resolve()


def _run_case(
    *,
    executable: Path,
    manifest: dict[str, Any],
    settings: dict[str, Any],
    pseudopotential: Path,
    case_root: Path,
    case_id: str,
    lattice_bohr: float,
    positions_bohr: np.ndarray,
    timeout_seconds: float,
) -> dict[str, Any]:
    case_root.mkdir(parents=True, exist_ok=True)
    scratch = case_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    input_path = case_root / "input.in"
    output_path = case_root / "output.out"
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", f"si_{case_id}")
    input_text = render_qe_input(
        manifest=manifest,
        settings=settings,
        lattice_bohr=lattice_bohr,
        positions_bohr=positions_bohr,
        pseudopotential=pseudopotential,
        scratch=scratch,
        prefix=prefix,
    )
    input_path.write_text(input_text)
    command = [str(executable), "-in", str(input_path.resolve())]
    try:
        completed = subprocess.run(
            command,
            cwd=case_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "case_id": case_id,
            "status": "blocked",
            "complete": False,
            "blocker": f"qe_execution_error:{error}",
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_path),
            "command": command,
        }
    output_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        return {
            "case_id": case_id,
            "status": "blocked",
            "complete": False,
            "blocker": f"qe_returncode:{completed.returncode}",
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "command": command,
            "returncode": completed.returncode,
        }
    try:
        parsed = parse_qe_output(completed.stdout, atom_count=manifest["system"]["atom_count"])
    except ValueError as error:
        return {
            "case_id": case_id,
            "status": "blocked",
            "complete": False,
            "blocker": f"qe_parse_error:{error}",
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "command": command,
            "returncode": completed.returncode,
        }
    return {
        **parsed,
        "case_id": case_id,
        "status": "converged",
        "lattice_bohr": lattice_bohr,
        "positions_bohr": np.asarray(positions_bohr).tolist(),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "command": command,
        "returncode": completed.returncode,
    }


def run_qe_reference(
    *,
    pw_x: str | Path,
    manifest_path: str | Path,
    out: str | Path,
    gth_path: str | Path | None = None,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Run every canonical silicon case with caller-provided PWscf.

    Args:
        pw_x: Explicit PWscf executable path or command name.
        manifest_path: Canonical prepared silicon workload manifest.
        out: Caller-owned QE output directory.
        gth_path: Explicit extracted GTH path. Defaults to the manifest path.
        timeout_seconds: Per-case execution timeout. Defaults to one hour.

    Returns:
        Normalized fresh QE report, including concrete blockers when unavailable.
    """

    inspect_workload(manifest_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    output_root = Path(out)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "reference-report.json"
    pseudopotential = Path(
        manifest["pseudopotential"]["path"] if gth_path is None else gth_path
    ).expanduser()
    executable = _resolve_executable(pw_x)
    base = {
        "schema_version": QE_REPORT_SCHEMA,
        "target_id": manifest["target_id"],
        "manifest": str(manifest_path),
        "manifest_fingerprint": manifest["fingerprint_sha256"],
        "pseudopotential": str(pseudopotential),
        "pseudopotential_sha256": (_sha256(pseudopotential) if pseudopotential.is_file() else None),
        "normalized_units": dict(NORMALIZED_UNITS),
        "settings": qe_settings_from_manifest(manifest),
        "reference_engine": "quantum_espresso_pwscf",
        "reference_engine_role": "reference-only; never a product runtime dependency",
        "requested_pw_x": str(pw_x),
        "resolved_pw_x": str(executable) if executable is not None else None,
        "command": shlex.join(str(value) for value in sys.argv),
        "product_runtime_boundary": {
            "product_runtime": "mlx_atomistic",
            "qe_scope": "caller-run script only",
            "package_dependency": False,
        },
    }
    blockers = []
    if executable is None:
        blockers.append("pw_x_not_found")
    if not pseudopotential.is_file():
        blockers.append("gth_not_found")
    elif _sha256(pseudopotential) != manifest["pseudopotential"]["sha256"]:
        blockers.append("gth_hash_mismatch")
    if blockers:
        report = {
            **base,
            "status": "blocked",
            "complete": False,
            "blockers": blockers,
            "cases": {},
        }
        report_path.write_bytes(_canonical_json(report))
        return report

    assert executable is not None
    settings = base["settings"]
    geometries = _geometry_cases(manifest)
    cases = {}
    for case_id in ("equilibrium", "displaced_atom", "strain_minus", "strain_plus"):
        geometry = geometries[case_id]
        cases[case_id] = _run_case(
            executable=executable,
            manifest=manifest,
            settings=settings,
            pseudopotential=pseudopotential,
            case_root=output_root / case_id,
            case_id=case_id,
            lattice_bohr=geometry["lattice_bohr"],
            positions_bohr=geometry["positions_bohr"],
            timeout_seconds=timeout_seconds,
        )
        if cases[case_id]["complete"] is not True:
            blockers.append(f"case_blocked:{case_id}:{cases[case_id]['blocker']}")
            break

    if not blockers:
        volume_rows = []
        for index, geometry in enumerate(geometries["volume_scan"]):
            row = _run_case(
                executable=executable,
                manifest=manifest,
                settings=settings,
                pseudopotential=pseudopotential,
                case_root=output_root / "volume_scan" / f"point-{index:02d}",
                case_id=f"volume_scan_{index}",
                lattice_bohr=geometry["lattice_bohr"],
                positions_bohr=geometry["positions_bohr"],
                timeout_seconds=timeout_seconds,
            )
            row["lattice_constant_angstrom"] = geometry["lattice_constant_angstrom"]
            volume_rows.append(row)
            if row["complete"] is not True:
                blockers.append(f"case_blocked:volume_scan_{index}:{row['blocker']}")
                break
        if not blockers:
            lattice = [row["lattice_constant_angstrom"] for row in volume_rows]
            energies = [row["total_energy_hartree"] for row in volume_rows]
            fit = fit_lattice_curve(lattice, energies)
            cases["volume_scan"] = {
                "status": "complete" if fit["status"] == "ok" else "blocked",
                "complete": fit["status"] == "ok" and len(volume_rows) == 7,
                "rows": volume_rows,
                "lattice_constants_angstrom": lattice,
                "energies_hartree": energies,
                "fit": fit,
            }
            if cases["volume_scan"]["complete"] is not True:
                blockers.append("qe_lattice_fit_blocked")
        else:
            cases["volume_scan"] = {
                "status": "blocked",
                "complete": False,
                "rows": volume_rows,
            }

    versions = {
        case["qe_version"]
        for case in cases.values()
        if isinstance(case, dict) and "qe_version" in case
    }
    if "volume_scan" in cases:
        versions.update(
            row["qe_version"] for row in cases["volume_scan"].get("rows", []) if "qe_version" in row
        )
    if len(versions) > 1:
        blockers.append("qe_version_changed_within_run")
    report = {
        **base,
        "status": "ran" if not blockers else "blocked",
        "blockers": blockers,
        "qe_version": next(iter(versions)) if len(versions) == 1 else None,
        "pw_x_sha256": _sha256(executable),
        "cases": cases,
        "complete": not blockers and set(cases) == set(manifest["cases"]),
    }
    report_path.write_bytes(_canonical_json(report))
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the Quantum ESPRESSO silicon reference CLI.

    Args:
        argv: Optional argument vector. Defaults to process arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pw-x", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gth", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_qe_reference(
        pw_x=args.pw_x,
        manifest_path=args.manifest,
        gth_path=args.gth,
        out=args.out,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"status={payload['status']} report={args.out / 'reference-report.json'}")


if __name__ == "__main__":
    main()
