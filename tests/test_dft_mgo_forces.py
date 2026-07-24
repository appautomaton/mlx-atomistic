from __future__ import annotations

import json
import signal

import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_mgo import prepare_mgo_workload
from mlx_atomistic.benchmarks.dft_mgo_eos_runner import (
    MEMORY_LIMIT_BYTES,
    POINT_TIMEOUT_SECONDS,
)
from mlx_atomistic.benchmarks.dft_mgo_forces import (
    COMPARISON_COUNT,
    DISPLACEMENT_BOHR,
    DISPLACEMENT_SCF_COUNT,
    FORCE_POINT_SCHEMA,
    FORCE_THRESHOLD_HARTREE_PER_BOHR,
    _force_comparisons,
    _point_root,
    _point_spec,
    _run_bounded_point,
    _validation_outcome,
    run_mgo_force_validation,
)


def _gth_source(path):
    path.write_text(
        """Mg GTH-PBE-q10 GTH-PBE
4 6
0.19275787 2 -20.57539077 3.04016732
2
0.14140682 1 41.04729209
0.10293187 1 -9.98562566
#
Mg GTH-PBE-q2
2
0.57696017 1 -2.69040744
2
0.59392350 2 3.50321099 -0.71677167
0.92534825
0.70715728 1 0.83115848
#
O GTH-PBE-q6 GTH-PBE
2 4
0.24455430 2 -16.66721480 2.48731132
2
0.22095592 1 18.33745811
0.21133247 0
"""
    )
    return path


def _accepted_eos_report(path):
    path.write_text(
        json.dumps(
            {
                "validation_complete": True,
                "accepted_workload": {
                    "profile": "q2-c70-k6",
                    "cutoff_hartree": 70,
                    "fft_shape": [68, 68, 68],
                    "kpoint_mesh": [6, 6, 6],
                },
                "selected_fit": {
                    "status": "ok",
                    "equilibrium_lattice_constant_angstrom": 4.259502661194667,
                },
            }
        )
    )
    return path


def test_mgo_force_validation_plan_has_exact_displacement_inventory(tmp_path):
    source = _gth_source(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_mgo_workload(
        gth_source=source,
        out=tmp_path / "workload",
    )
    eos_report = _accepted_eos_report(tmp_path / "eos-report.json")
    density = tmp_path / "density.npy"
    np.save(density, np.zeros((1,), dtype=np.float32), allow_pickle=False)

    plan = run_mgo_force_validation(
        manifest_path=prepared["manifest"],
        eos_report_path=eos_report,
        initial_density_path=density,
        out=tmp_path / "validation",
        dry_run=True,
    )

    assert plan["status"] == "planned"
    assert plan["equilibrium_seed_scf_count"] == 1
    assert plan["displacement_scf_count"] == DISPLACEMENT_SCF_COUNT == 48
    assert plan["central_difference_comparison_count"] == COMPARISON_COUNT == 24
    assert plan["memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert plan["point_timeout_seconds"] == POINT_TIMEOUT_SECONDS
    points = plan["displacement_points"]
    assert {
        (point["atom_index"], point["axis"], point["direction"])
        for point in points
    } == {
        (atom_index, axis, direction)
        for atom_index in range(8)
        for axis in range(3)
        for direction in ("minus", "plus")
    }
    assert len({point["point_fingerprint"] for point in points}) == 48


def _synthetic_force_rows(analytic):
    rows = []
    baseline = -100.0
    for atom_index in range(8):
        for axis in range(3):
            force = float(analytic[atom_index, axis])
            for direction, sign in (("minus", -1.0), ("plus", 1.0)):
                rows.append(
                    {
                        "point": {
                            "atom_index": atom_index,
                            "axis": axis,
                            "direction": direction,
                            "point_fingerprint": (
                                f"{atom_index}-{axis}-{direction}"
                            ),
                        },
                        "result": {
                            "total_energy_hartree": (
                                baseline - force * sign * DISPLACEMENT_BOHR
                            )
                        },
                    }
                )
    return rows


def test_mgo_force_comparison_uses_central_energy_difference():
    analytic = np.arange(24, dtype=np.float64).reshape(8, 3) * 1.0e-5
    equilibrium = {
        "result": {
            "analytic_forces": {
                "forces_hartree_per_bohr": analytic.tolist(),
            }
        }
    }
    rows = _synthetic_force_rows(analytic)

    comparisons, maximum = _force_comparisons(equilibrium, rows)

    assert len(comparisons) == COMPARISON_COUNT
    assert maximum < 1.0e-12
    assert all(row["passed"] for row in comparisons)

    rows[-1]["result"]["total_energy_hartree"] += (
        4.0 * FORCE_THRESHOLD_HARTREE_PER_BOHR * DISPLACEMENT_BOHR
    )
    comparisons, maximum = _force_comparisons(equilibrium, rows)
    assert maximum == pytest.approx(2.0 * FORCE_THRESHOLD_HARTREE_PER_BOHR)
    assert comparisons[-1]["passed"] is False


def test_mgo_force_runner_reuses_exact_completed_point(tmp_path, monkeypatch):
    spec = _point_spec(
        workload_fingerprint="workload",
        eos_report_sha256="eos",
        equilibrium_lattice_angstrom=4.25,
        kind="displacement",
        atom_index=3,
        axis=1,
        direction="plus",
        initial_seed_fingerprint="seed",
    )
    output = tmp_path / "validation"
    root = _point_root(
        output,
        kind="displacement",
        atom_index=3,
        axis=1,
        direction="plus",
    )
    root.mkdir(parents=True)
    (root / "report.json").write_text(
        json.dumps(
            {
                "schema_version": FORCE_POINT_SCHEMA,
                "numerical_passed": True,
                "point": spec,
            }
        )
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("a matching completed point must not be rerun")

    monkeypatch.setattr(
        "mlx_atomistic.benchmarks.dft_mgo_forces.subprocess.Popen",
        unexpected_run,
    )

    report, failure, reused = _run_bounded_point(
        manifest_path=tmp_path / "manifest.json",
        eos_report_path=tmp_path / "eos.json",
        output=output,
        spec=spec,
        equilibrium_seed_path=tmp_path / "seed.json",
    )

    assert reused is True
    assert failure is None
    assert report is not None
    assert report["point"]["point_fingerprint"] == spec["point_fingerprint"]


def test_mgo_force_runner_terminates_point_process_group_on_interrupt(
    tmp_path,
    monkeypatch,
):
    spec = _point_spec(
        workload_fingerprint="workload",
        eos_report_sha256="eos",
        equilibrium_lattice_angstrom=4.25,
        kind="displacement",
        atom_index=7,
        axis=2,
        direction="plus",
        initial_seed_fingerprint="seed",
    )

    class InterruptedProcess:
        pid = 43210
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

    process = InterruptedProcess()
    signals = []
    monkeypatch.setattr(
        "mlx_atomistic.benchmarks.dft_mgo_forces.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "mlx_atomistic.benchmarks.dft_mgo_forces.os.killpg",
        lambda pid, value: signals.append((pid, value)),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_bounded_point(
            manifest_path=tmp_path / "manifest.json",
            eos_report_path=tmp_path / "eos.json",
            output=tmp_path / "validation",
            spec=spec,
            equilibrium_seed_path=tmp_path / "seed.json",
        )

    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.calls == 2


def test_mgo_force_precision_limit_is_accepted_without_weakening_gate():
    comparisons = [
        {
            "atom_index": atom_index,
            "symbol": "Mg" if atom_index < 4 else "O",
            "axis_label": ("x", "y", "z")[axis],
            "absolute_deviation_hartree_per_bohr": 5.0e-5,
            "passed": True,
        }
        for atom_index in range(8)
        for axis in range(3)
    ]
    for atom_index, axis in ((6, 0), (7, 1), (7, 2)):
        row = comparisons[atom_index * 3 + axis]
        row["absolute_deviation_hartree_per_bohr"] = 2.0e-4
        row["passed"] = False

    outcome = _validation_outcome(
        comparisons,
        accept_precision_limit=True,
    )

    assert outcome["status"] == "complete_with_known_precision_limit"
    assert outcome["accepted"] is True
    assert outcome["strict_gate_passed"] is False
    assert outcome["strict_pass_count"] == 21
    assert outcome["strict_fail_count"] == 3
    assert outcome["blockers"] == []
    limitation = outcome["known_precision_limit"]
    assert limitation["force_threshold_hartree_per_bohr"] == 1.0e-4
    assert limitation["threshold_weakened"] is False
