from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_atomistic.benchmarks.dft_silicon import (
    GTH_NAME,
    SOURCE_SCHEMA,
    TARGET_ID,
    WORKLOAD_SCHEMA,
    inspect_workload,
    main,
    parse_gth_entry,
    prepare_workload,
    render_qe_gth,
)


def _gth_database(path: Path) -> Path:
    path.write_text(
        """# compact fixture
H GTH-PBE-q1 GTH-PBE
    1
    0.2 1 -1.0
    0
#
Si GTH-PBE-q4 GTH-PBE
    2 2
    0.44000000 1 -6.26928833
    2
    0.43563383 2 8.95174150 -2.70627082
                   3.49378060
    0.49794218 1 2.43127673
#
P GTH-PBE-q5 GTH-PBE
    2 3
    0.43 1 -5.8
    0
"""
    )
    return path


def test_parse_gth_entry_preserves_full_channel_matrices(tmp_path):
    entry = parse_gth_entry(_gth_database(tmp_path / "GTH_POTENTIALS"))

    assert entry.element == "Si"
    assert GTH_NAME in entry.names
    assert entry.valence_charge == 4.0
    assert entry.local_coefficients == (-6.26928833,)
    assert len(entry.channels) == 2
    assert entry.channels[0].coupling_matrix == (
        (8.9517415, -2.70627082),
        (-2.70627082, 3.4937806),
    )
    assert entry.channels[1].coupling_matrix == ((2.43127673,),)


def test_parse_gth_entry_rejects_incomplete_triangular_matrix(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    source.write_text(source.read_text().replace("                   3.49378060\n", ""))

    with pytest.raises(ValueError, match="coupling"):
        parse_gth_entry(source)


def test_render_qe_gth_is_deterministic_and_includes_zero_spin_orbit(tmp_path):
    entry = parse_gth_entry(_gth_database(tmp_path / "GTH_POTENTIALS"))
    rendered = render_qe_gth(entry)

    assert rendered.startswith("Goedecker pseudopotential for Si\n14 4")
    assert "10 11 1 2" in rendered
    assert "0.43563383 2 8.9517415 -2.70627082" in rendered
    assert rendered.endswith("2.43127673\n0\n")
    assert render_qe_gth(entry) == rendered


def test_prepare_workload_writes_strict_idempotent_manifests(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"
    command = ["uv", "run", "python", "-m", "mlx_atomistic.benchmarks.dft_silicon"]

    first = prepare_workload(gth_source=source, out=out, command=command)
    second = prepare_workload(gth_source=source, out=out, command=command)

    assert first == second
    source_manifest = json.loads(Path(first["source_manifest"]).read_text())
    workload = json.loads(Path(first["workload_manifest"]).read_text())
    assert source_manifest["schema_version"] == SOURCE_SCHEMA
    assert source_manifest["target_id"] == TARGET_ID
    assert source_manifest["extracted_sha256"] == first["gth_sha256"]
    assert workload["schema_version"] == WORKLOAD_SCHEMA
    assert workload["target_id"] == TARGET_ID
    assert workload["system"]["atom_count"] == 8
    assert workload["system"]["electron_count"] == 32.0
    assert workload["target_host"]["chip"] == "Apple M5 Max"
    assert workload["target_host"]["pmset_expected_lowpowermode"] == 1
    assert sorted(workload["cases"]) == [
        "displaced_atom",
        "equilibrium",
        "strain_minus",
        "strain_plus",
        "volume_scan",
    ]
    assert inspect_workload(first["workload_manifest"])["status"] == "ready"


def test_prepare_workload_refuses_mismatched_existing_output(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"
    prepared = prepare_workload(gth_source=source, out=out, command=["prepare"])
    Path(prepared["gth_path"]).write_text("tampered\n")

    with pytest.raises(ValueError, match="refusing to replace mismatched"):
        prepare_workload(gth_source=source, out=out, command=["prepare"])


def test_inspect_workload_rejects_manifest_and_gth_tampering(tmp_path):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    prepared = prepare_workload(gth_source=source, out=tmp_path / "prepared", command=["prepare"])
    manifest_path = Path(prepared["workload_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["system"]["electron_count"] = 31.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fingerprint"):
        inspect_workload(manifest_path)


def test_dft_silicon_prepare_and_inspect_cli_json(tmp_path, capsys):
    source = _gth_database(tmp_path / "GTH_POTENTIALS")
    out = tmp_path / "prepared"

    main(["prepare", "--gth-database", str(source), "--out", str(out), "--json"])
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["status"] == "prepared"

    main(["inspect", "--manifest", prepared["workload_manifest"], "--json"])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "ready"
    assert inspected["target_chip"] == "Apple M5 Max"
