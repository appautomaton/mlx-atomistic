from __future__ import annotations

import numpy as np
import pytest

import mlx_atomistic.benchmarks.dft_iron_spin as iron_spin
from mlx_atomistic.benchmarks.dft_iron_spin import (
    load_iron_spin_workload,
    prepare_iron_spin_workload,
    run_iron_spin_validation,
)


def _gth_source(path):
    path.write_text(
        """Fe GTH-PBE-q8 GTH-GGA-q8
    2    0    6    0
    0.59578850974080       1    0.16043266015033
       3
    0.45708850300757       3    3.28795514212277   -1.01066080718626    0.78740540871188
                                                    2.67805610338415   -2.03777744650135
                                                                        3.28627319626836
    0.63922887323798       2    1.54380725058909   -0.09985698097411
                                                    0.33708086116139
    0.30621323095055       1   -9.14944579471234
"""
    )
    return path


def _q16_gth_source(path):
    path.write_text(
        """Fe GTH-PBE-q16 GTH-GGA-q16
4    6    6    0
0.35995327161394       2    6.77089340472311   -0.19302363970878
3
0.27176718262559       2    0.57493731608079    7.91313536575066
-10.00782096686501
0.25503930918596       2   -7.89264568445203    7.69707832352137
-9.14331139080110
0.22321053924521       1  -12.41029471698018
"""
    )
    return path


def test_iron_spin_workload_is_hash_guarded_and_has_a_bounded_plan(tmp_path):
    prepared = prepare_iron_spin_workload(
        gth_source=_gth_source(tmp_path / "POTENTIAL_UZH"),
        out=tmp_path / "workload",
    )
    workload, resource = load_iron_spin_workload(prepared["manifest"])
    plan = run_iron_spin_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
        dry_run=True,
    )

    assert workload["resource"]["sha256"] == iron_spin.GTH_RESOURCE_SHA256
    assert workload["system"]["electron_count"] == 8
    assert workload["physics"]["initial_magnetization_per_cell"] == 2.0
    assert workload["kpoint_meshes"]["2"]["full_point_count"] == 16
    assert iron_spin._kpoint_mesh(workload, 2).point_group_symmetry_reduced is True
    assert workload["representation"]["radial_tail_ratio_at_cutoff"]["selected"] < 0.02
    assert workload["representation"]["radial_tail_ratio_at_cutoff"]["cutoff-check"] < 0.003
    assert plan["points"] == workload["validation"]["required_points"]
    resource.write_text(resource.read_text() + "\n")
    with pytest.raises(ValueError, match="missing or mismatched"):
        load_iron_spin_workload(prepared["manifest"])


def test_iron_spin_validation_aggregates_locked_gates_without_extra_points(
    tmp_path,
    monkeypatch,
):
    prepared = prepare_iron_spin_workload(
        gth_source=_gth_source(tmp_path / "POTENTIAL_UZH"),
        out=tmp_path / "workload",
    )

    def fake_point(*, profile, polarized, kpoint_mode, out, manifest_path):
        workload, _ = load_iron_spin_workload(manifest_path)
        moments = {
            "smoke": 2.2,
            "selected": 2.31,
            "cutoff-check": 2.34,
            "kpoint-check": 2.28,
        }
        energy = -20.0 if polarized else -19.8
        if kpoint_mode == "full":
            energy += 1.0e-5
        payload = {
            "schema_version": iron_spin.POINT_SCHEMA,
            "workload_fingerprint": workload["workload_fingerprint"],
            "profile": profile,
            "polarized": polarized,
            "kpoint_mode": kpoint_mode,
            "settings": workload["profiles"][profile],
            "source_fingerprints": {
                "material_protocol": iron_spin._material_protocol_record(),
            },
            "result": {
                "converged": True,
                "electron_count": 8.0,
                "free_energy_hartree": energy,
                "moment_per_atom": moments[profile] if polarized else None,
            }
        }
        iron_spin._write_json(out, payload)
        return payload

    monkeypatch.setattr(iron_spin, "run_iron_spin_point", fake_point)
    report = run_iron_spin_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
    )

    assert report["verified"] is True
    assert report["status"] == "verified"
    assert all(report["gates"].values())
    assert len(report["points"]) == 6
    assert report["symmetry_oracle"]["free_energy_abs_hartree_per_atom"] == (
        pytest.approx(1.0e-5)
    )

    monkeypatch.setattr(
        iron_spin,
        "run_iron_spin_point",
        lambda **_kwargs: pytest.fail("matching point reports must be reused"),
    )
    reused = run_iron_spin_validation(
        manifest_path=prepared["manifest"],
        out=tmp_path / "validation",
    )
    assert reused["verified"] is True


def test_q16_workload_locks_its_separate_electron_and_cutoff_contract(tmp_path):
    prepared = prepare_iron_spin_workload(
        gth_source=_q16_gth_source(tmp_path / "POTENTIAL_UZH"),
        gth_name="GTH-PBE-q16",
        out=tmp_path / "workload",
    )
    workload, _ = load_iron_spin_workload(prepared["manifest"])

    assert workload["resource"]["name"] == "GTH-PBE-q16"
    assert workload["system"]["electron_count"] == 16
    assert workload["physics"]["initial_magnetization_per_cell"] == 2.2
    assert workload["profiles"]["selected"]["cutoff_hartree"] == 150.0
    assert workload["validation"]["symmetry_moment_abs_per_atom"] == 0.02
    assert workload["representation"]["radial_tail_ratio_at_cutoff"]["selected"] < 0.02


def test_bcc_primitive_symmetry_preserves_metric_and_exact_unfolded_weights():
    operations = iron_spin._primitive_reciprocal_operations()
    reciprocal = 2.0 * np.pi * np.linalg.inv(iron_spin.PRIMITIVE_CELL_MATRIX).T
    metric = reciprocal @ reciprocal.T

    assert len(operations) == 48
    assert np.linalg.det(iron_spin.PRIMITIVE_CELL_MATRIX) == pytest.approx(
        iron_spin.LATTICE_BOHR**3 / 2.0,
        rel=1.0e-12,
    )
    for operation in operations:
        matrix = np.asarray(operation)
        assert round(np.linalg.det(matrix)) in {-1, 1}
        np.testing.assert_allclose(matrix.T @ metric @ matrix, metric, atol=2.0e-13)
    mesh = iron_spin._primitive_unfolded_mesh(3)
    assert len(mesh.points) == 54
    assert sum(point.weight for point in mesh.points) == pytest.approx(1.0, abs=1.0e-14)

    prepared_mesh = {
        "kpoint_meshes": {"3": iron_spin._mesh_payload(3)},
    }
    assert len(iron_spin._kpoint_mesh(prepared_mesh, 3, mode="full").points) == 54
    assert len(iron_spin._kpoint_mesh(prepared_mesh, 3, mode="reduced").points) < 54
