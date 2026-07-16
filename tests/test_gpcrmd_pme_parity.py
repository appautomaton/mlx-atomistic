from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_gpcrmd_pme_parity.py"
)
_SPEC = importlib.util.spec_from_file_location("run_gpcrmd_pme_parity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)


def test_source_pme_config_uses_opencl_fft_legal_mesh_dimensions():
    config = _HELPER._derive_source_pme_config(
        np.diag([87.17031860351562, 87.15242004394531, 118.58049774169922]),
        cutoff=9.0,
        tolerance=5.0e-4,
    )

    assert config.mesh_shape == (78, 78, 108)
    assert config.alpha == pytest.approx(0.2920289872)


@pytest.fixture
def _metal_device(monkeypatch):
    monkeypatch.setenv("MLX_ATOMISTIC_DEVICE", "gpu")
    previous = mx.default_device()
    try:
        device = mx.Device(mx.gpu, 0)
        mx.set_default_device(device)
        mx.set_default_stream(mx.new_stream(device))
        mx.eval(mx.array([1.0], dtype=mx.float32))
    except Exception:  # noqa: BLE001 - any Metal load failure means skip.
        mx.set_default_device(previous)
        mx.set_default_stream(mx.new_stream(previous))
        pytest.skip("Metal GPU unavailable")
    yield
    mx.set_default_device(previous)
    mx.set_default_stream(mx.new_stream(previous))


@pytest.mark.reference
@pytest.mark.gpu
def test_small_charmm_pme_fixture_matches_components_and_complete_forces(
    tmp_path: Path,
    _metal_device,
):
    pytest.importorskip("openmm")

    report = _HELPER.evaluate_small_charmm_pme_fixture(
        out=tmp_path,
        platform_name="Reference",
        precision="double",
    )

    assert report["status"] == "passed"
    assert report["passed"] is True
    assert report["blockers"] == []
    assert report["manifest_comparison"]["matched"] is True
    assert all(report["checks"].values())
    expected_components = {
        "bond",
        "angle",
        "urey_bradley",
        "proper_dihedral",
        "harmonic_improper",
        "cmap",
        "nonbonded",
    }
    assert set(report["energies"]["component_metrics"]) == expected_components
    assert report["force_arrays"]["shape"] == [8, 3]
    with np.load(tmp_path / _HELPER.FORCE_ARRAYS_NAME) as arrays:
        assert set(arrays.files) == {
            "mlx_forces_kj_mol_nm",
            "openmm_forces_kj_mol_nm",
            "force_delta_kj_mol_nm",
        }
        for values in arrays.values():
            assert values.shape == (8, 3)
            assert np.all(np.isfinite(values))


def test_manifest_mismatch_blocks_before_force_metrics(monkeypatch, tmp_path: Path):
    mismatch = {
        "matched": False,
        "mismatches": {"particles.charge_hash": {"mlx": "a", "openmm": "b"}},
        "mlx_manifest_hash": "mlx",
        "openmm_manifest_hash": "openmm",
        "required_fields": list(_HELPER.MANIFEST_FIELDS),
        "status": "mismatched",
    }
    monkeypatch.setattr(_HELPER, "_build_openmm_reference", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(_HELPER, "_mlx_manifest", lambda *_args, **_kwargs: {"engine": "mlx"})
    monkeypatch.setattr(
        _HELPER,
        "_openmm_manifest",
        lambda *_args, **_kwargs: {"engine": "openmm"},
    )
    monkeypatch.setattr(_HELPER, "_compare_manifests", lambda *_args, **_kwargs: mismatch)
    monkeypatch.setattr(
        _HELPER,
        "_evaluate_openmm_reference",
        lambda *_args, **_kwargs: pytest.fail("metrics evaluated after manifest mismatch"),
    )

    with pytest.raises(_HELPER.GPCRmdParityBlocked, match="manifest_mismatch"):
        _HELPER._execute_parity(
            api=object(),
            source={},
            prepared=object(),
            prepared_dir=tmp_path,
            platform_name="Reference",
            precision="double",
            tolerances=_HELPER.GPCRmdParityTolerances(),
            out=tmp_path,
            base={},
            require_production=False,
        )

    assert (tmp_path / _HELPER.MLX_MANIFEST_NAME).is_file()
    assert (tmp_path / _HELPER.OPENMM_MANIFEST_NAME).is_file()
    assert (tmp_path / _HELPER.MANIFEST_COMPARISON_NAME).is_file()
    assert not (tmp_path / _HELPER.FORCE_ARRAYS_NAME).exists()


@pytest.mark.reference
def test_missing_openmm_platform_blocks_after_manifest_construction(tmp_path: Path):
    pytest.importorskip("openmm")

    report = _HELPER.evaluate_small_charmm_pme_fixture(
        out=tmp_path,
        platform_name="definitely-unavailable-platform",
    )

    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert report["blockers"][0].startswith("openmm_platform_unavailable:")
    comparison = json.loads((tmp_path / _HELPER.MANIFEST_COMPARISON_NAME).read_text())
    assert comparison["matched"] is True
    assert "force_metrics" not in report
    assert not (tmp_path / _HELPER.FORCE_ARRAYS_NAME).exists()


def test_unknown_openmm_force_class_fails_closed():
    unknown_force = type("UnregisteredForce", (), {})()

    class FakeSystem:
        def getNumForces(self):
            return 1

        def getForce(self, index):
            assert index == 0
            return unknown_force

    with pytest.raises(
        _HELPER.UnsupportedOpenMMForceError,
        match="unknown_force_classes:UnregisteredForce",
    ):
        _HELPER._validate_openmm_force_inventory(FakeSystem())


@pytest.mark.parametrize(
    "forces",
    [
        np.zeros((7, 3), dtype=np.float64),
        np.full((8, 3), np.nan, dtype=np.float64),
    ],
)
def test_partial_or_nonfinite_force_arrays_fail_closed(forces: np.ndarray):
    with pytest.raises(_HELPER.GPCRmdParityBlocked, match="partial_or_nonfinite_forces"):
        _HELPER._require_complete_forces(forces, atom_count=8, engine="test")


def test_protocol_member_hashes_come_from_authoritative_file_record(
    monkeypatch,
    tmp_path: Path,
):
    cache = tmp_path / "cache"
    prepared = tmp_path / "prepared"
    cache.mkdir()
    prepared.mkdir()
    records = []
    for role, name in (
        ("topology", "system.psf"),
        ("parameters", "system.prm"),
        ("model", "system.pdb"),
        ("protocol", "protocol.tar.gz"),
    ):
        path = cache / name
        path.write_bytes(role.encode())
        records.append(
            {
                "role": role,
                "resolved_filename": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "archive_members": [],
            }
        )
    protocol_root = cache / "protocol-root" / "rep_1"
    protocol_root.mkdir(parents=True)
    member_records = []
    for name in ("input", "input.coor", "input.xsc", "log.txt"):
        path = protocol_root / name
        path.write_bytes(name.encode())
        member_records.append(
            {
                "kind": "file",
                "normalized_name": f"rep_1/{name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records[-1]["archive_members"] = member_records
    source_manifest = {
        "blockers": [],
        "files": records,
        "archives": [
            {
                "extraction_root": str(cache / "protocol-root"),
                "members": [
                    {
                        "kind": "file",
                        "normalized_name": "rep_1/input",
                        "sha256": "not-authoritative",
                    }
                ],
            }
        ],
    }
    source_manifest_path = tmp_path / "source.json"
    source_manifest_path.write_text(json.dumps(source_manifest))
    (prepared / "mlx-workload-manifest.json").write_text(
        json.dumps(
            {
                "workload": {"name": "fixture", "atom_count": 1},
                "protocol": {"selected_replicate": "rep_1"},
            }
        )
    )
    protocol = {
        "nonbonded": {"cutoff_angstrom": 9.0},
        "pme": {"ewald_error_tolerance": 5.0e-4},
    }
    config = _HELPER.PMEConfig(
        mesh_shape=(8, 8, 8),
        alpha=0.3,
        real_cutoff=9.0,
        assignment_order=5,
        charge_tolerance=1.0e-4,
        deconvolve_assignment=True,
        background_policy="reject_non_neutral",
    )
    monkeypatch.setattr(
        _HELPER,
        "_read_acemd_vectors",
        lambda _path: np.zeros((1, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        _HELPER,
        "_read_xsc_matrix",
        lambda _path: np.diag([20.0, 20.0, 20.0]),
    )
    monkeypatch.setattr(
        _HELPER,
        "_read_gpcrmd_protocol",
        lambda _input, _log: protocol,
    )
    monkeypatch.setattr(
        _HELPER,
        "_derive_source_pme_config",
        lambda *_args, **_kwargs: config,
    )

    resolved = _HELPER._resolve_gpcrmd_source(
        source_manifest_path=source_manifest_path,
        cache=cache,
        prepared_dir=prepared,
        out=tmp_path / "out",
    )

    assert resolved["fixture"] == "fixture"
    assert resolved["file_hashes"]["rep_1/input"] == member_records[0]["sha256"]
