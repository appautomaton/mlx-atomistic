from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK_DIR = Path("notebooks/ligand-receptor-motion").resolve()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

# The notebook helpers import pandas (it ships only in the `viz` extra), which the
# CI lanes do not install. Skip the whole module when pandas is absent so pytest
# collection never fails on the helper imports below.
pytest.importorskip("pandas")

import helpers.mlx_real_md as mlx_real_md  # noqa: E402
from helpers.mlx_real_md import (  # noqa: E402
    ensure_gpcrmd_mlx_bundle,
    load_gpcrmd_mlx_artifact,
)
from helpers.motion_analysis import (  # noqa: E402
    ProcessedTrajectory,
    align_trajectory_to_reference,
    analysis_tables,
    build_synthetic_motion_fixture,
    build_synthetic_pbc_wrap_fixture,
    contact_counts,
    hydrogen_bond_counts,
    ligand_com,
    motion_gate_report,
    raw_ligand_com,
    save_processed_trajectory,
    trajectory_quality_report,
    water_counts_around_ligand,
)
from helpers.visualization import make_ligand_motion_figure  # noqa: E402

pytestmark = pytest.mark.integration


def _write_gpcrmd_prepared_fixture(out_dir: Path) -> str:
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system

    target_id = "tiny-gpcrmd-notebook-fixture"
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        source={
            "kind": "gpcrmd_fixture",
            "gpcrmd_target_id": target_id,
            "gpcrmd_dynamics_id": 17,
            "pdb_id": "5F8U",
        },
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="gpcrmd_notebook_prmtop_fixture",
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": 1,
            "supported_terms": [
                "harmonic_bond",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
                "ligand",
                "receptor",
            ],
            "required_terms": [
                "harmonic_bond",
                "nonbonded_lj_coulomb",
                "nonbonded_exception",
                "distance_constraint",
                "ligand",
                "receptor",
            ],
            "unsupported_terms": [],
            "rejected_terms": [],
            "electrostatics_model": "cutoff",
        },
        protocol_metadata={
            "ensemble": "NVT",
            "barostat": "none",
        },
    )
    prepared = replace(
        prepared,
        metadata=metadata,
        symbols=np.asarray(["H", "O"], dtype=str),
        atom_names=np.asarray(["H1", "O1"], dtype=str),
        atom_types=np.asarray(["H", "O"], dtype=str),
        residue_names=np.asarray(["LIG", "REC"], dtype=str),
        residue_ids=np.asarray([1, 2], dtype=np.int32),
        chain_ids=np.asarray(["L", "R"], dtype=str),
        masses=np.asarray([1.008, 15.999], dtype=np.float32),
        ligand_mask=np.asarray([True, False]),
        receptor_mask=np.asarray([False, True]),
        restraint_mask=np.asarray([False, True]),
        constraints=np.asarray([[0, 1]], dtype=np.int32),
        constraint_distance=np.asarray([1.25], dtype=np.float32),
        nonbonded_exception_pairs=np.asarray([[0, 1]], dtype=np.int32),
        nonbonded_exception_charge_product=np.asarray([0.0], dtype=np.float32),
        nonbonded_exception_sigma=np.asarray([0.0], dtype=np.float32),
        nonbonded_exception_epsilon=np.asarray([0.0], dtype=np.float32),
        water_mask=np.asarray([False, False]),
        ion_mask=np.asarray([False, False]),
        lipid_mask=np.asarray([False, False]),
    )
    save_prepared_system(prepared, out_dir)
    return target_id


def test_synthetic_fixture_passes_visible_motion_gate(tmp_path: Path):
    trajectory = build_synthetic_motion_fixture(tmp_path / "fixture.npz")
    report = motion_gate_report(trajectory)
    tables = analysis_tables(trajectory)

    assert report["passes_motion_gate"] is True
    assert report["max_ligand_com_displacement_A"] >= 12.0
    assert report["contact_count_delta"] >= 10
    assert set(tables) == {
        "summary",
        "quality",
        "quality_warnings",
        "frames",
        "closest_residues",
        "contact_occupancy",
        "hydrogen_bonds",
    }
    assert tables["frames"].shape[0] == trajectory.frame_count


def test_pbc_wrapped_ligand_com_is_unwrapped_and_reported(tmp_path: Path):
    trajectory = build_synthetic_pbc_wrap_fixture(tmp_path / "pbc.npz")
    loaded = trajectory.load(tmp_path / "pbc.npz")
    raw_centers = raw_ligand_com(loaded)
    corrected_centers = ligand_com(loaded)
    raw_steps = np.linalg.norm(np.diff(raw_centers, axis=0), axis=1)
    corrected_steps = np.linalg.norm(np.diff(corrected_centers, axis=0), axis=1)
    report = trajectory_quality_report(loaded)

    np.testing.assert_allclose(loaded.cell_lengths_A, [10.0, 10.0, 10.0])
    assert float(raw_steps.max()) > 9.0
    assert float(corrected_steps.max()) < 1.0
    assert report["pbc_corrected_display"] is True
    assert report["raw_pbc_jump_count"] >= 1
    assert report["pbc_corrected_jump_count"] >= 1
    assert any("PBC" in warning for warning in report["warnings"])


def test_processed_trajectory_round_trips_cell_lengths(tmp_path: Path):
    trajectory = build_synthetic_pbc_wrap_fixture(tmp_path / "source.npz")
    out = tmp_path / "roundtrip.npz"
    save_processed_trajectory(out, trajectory)
    loaded = trajectory.load(out)

    np.testing.assert_allclose(loaded.cell_lengths_A, trajectory.cell_lengths_A)


def test_contact_counts_use_minimum_image_when_cell_is_present(tmp_path: Path):
    trajectory = build_synthetic_pbc_wrap_fixture(tmp_path / "pbc.npz")
    counts = contact_counts(trajectory, cutoff_A=0.8)
    without_cell = replace(trajectory, cell_lengths_A=None)
    raw_counts = contact_counts(without_cell, cutoff_A=0.8)

    assert counts[1] > raw_counts[1]
    assert counts[1] > 0


def test_ligand_motion_figure_hides_future_path_and_extends_current_trail(tmp_path: Path):
    trajectory = build_synthetic_pbc_wrap_fixture(tmp_path / "pbc.npz")
    figure = make_ligand_motion_figure(trajectory)
    traces = {trace.name: trace for trace in figure.data}

    assert traces["ligand COM path"].visible == "legendonly"
    assert traces["net displacement axis"].visible == "legendonly"
    assert traces["pocket residue labels"].visible == "legendonly"
    assert len(traces["current progress"].x) == 1
    assert len(figure.frames[2].data[2].x) == 3


def test_notebook_viewer_defaults_bound_large_system_context():
    ligand = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    receptor = np.vstack(
        [
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[40.0 + i, 0.0, 0.0] for i in range(20)], dtype=np.float32),
        ]
    )
    waters = np.asarray([[0.0, 1.0 + 0.001 * i, 0.0] for i in range(520)], dtype=np.float32)
    ions = np.asarray([[0.0, -1.0 - 0.001 * i, 0.0] for i in range(220)], dtype=np.float32)
    lipids = np.asarray([[0.0, 0.0, 1.0 + 0.001 * i] for i in range(930)], dtype=np.float32)
    frame0 = np.vstack([receptor, ligand, waters, ions, lipids]).astype(np.float32)
    positions = np.stack([frame0, frame0 + np.asarray([0.1, 0.0, 0.0], dtype=np.float32)])
    receptor_indices = np.arange(receptor.shape[0], dtype=np.int32)
    ligand_start = receptor.shape[0]
    water_start = ligand_start + ligand.shape[0]
    ion_start = water_start + waters.shape[0]
    lipid_start = ion_start + ions.shape[0]
    trajectory = ProcessedTrajectory(
        positions=positions,
        time_ps=np.asarray([0.0, 1.0], dtype=np.float32),
        symbols=np.asarray(
            ["C"] * receptor.shape[0]
            + ["C"] * 2
            + ["O"] * 520
            + ["NA"] * 220
            + ["P"] * 930
        ),
        atom_names=np.asarray(["A"] * positions.shape[1]),
        residue_names=np.asarray(
            ["REC"] * receptor.shape[0]
            + ["LIG"] * 2
            + ["WAT"] * 520
            + ["SOD"] * 220
            + ["POP"] * 930
        ),
        residue_ids=np.arange(positions.shape[1], dtype=np.int32),
        segment_ids=np.asarray(["A"] * positions.shape[1]),
        ligand_indices=np.arange(ligand_start, ligand_start + 2, dtype=np.int32),
        receptor_indices=receptor_indices,
        water_indices=np.arange(water_start, ion_start, dtype=np.int32),
        ion_indices=np.arange(ion_start, lipid_start, dtype=np.int32),
        lipid_indices=np.arange(lipid_start, positions.shape[1], dtype=np.int32),
        cell_lengths_A=np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
        source={"kind": "test_fixture"},
    )
    figure = make_ligand_motion_figure(trajectory)
    traces = {trace.name: trace for trace in figure.data}

    assert len(traces["receptor pocket atoms"].x) == 2
    assert len(traces["nearby waters"].x) == 500
    assert len(traces["ions"].x) == 200
    assert len(traces["selected membrane context"].x) == 900


def test_notebook_viewer_reimages_receptor_pocket_around_ligand():
    positions = np.asarray(
        [
            [2.0, 50.0, 50.0],
            [98.0, 50.0, 50.0],
        ],
        dtype=np.float32,
    )
    trajectory = ProcessedTrajectory(
        positions=positions[None, :, :],
        time_ps=np.asarray([0.0], dtype=np.float32),
        symbols=np.asarray(["C", "C"]),
        atom_names=np.asarray(["CA", "L1"]),
        residue_names=np.asarray(["REC", "LIG"]),
        residue_ids=np.asarray([1, 2], dtype=np.int32),
        segment_ids=np.asarray(["A", "L"]),
        ligand_indices=np.asarray([1], dtype=np.int32),
        receptor_indices=np.asarray([0], dtype=np.int32),
        water_indices=np.asarray([], dtype=np.int32),
        ion_indices=np.asarray([], dtype=np.int32),
        lipid_indices=np.asarray([], dtype=np.int32),
        cell_lengths_A=np.asarray([100.0, 100.0, 100.0], dtype=np.float32),
        source={"kind": "test_fixture"},
    )

    figure = make_ligand_motion_figure(trajectory)
    receptor_trace = {trace.name: trace for trace in figure.data}["receptor pocket atoms"]

    np.testing.assert_allclose(receptor_trace.x, [102.0])


def test_receptor_alignment_removes_global_rotation_translation():
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    theta = np.deg2rad(35.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translated = reference @ rotation.T + np.asarray([5.0, -3.0, 2.0], dtype=np.float32)
    positions = np.stack([reference, translated], axis=0)

    aligned = align_trajectory_to_reference(positions, align_indices=np.arange(4, dtype=np.int32))

    np.testing.assert_allclose(aligned[0, :4], reference[:4], atol=1e-5)
    np.testing.assert_allclose(aligned[1, :4], reference[:4], atol=1e-5)


def test_gpcrmd_mlx_helper_generates_processed_trajectory(tmp_path: Path):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    bundle = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=4,
        sample_interval=2,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )

    trajectory = bundle.processed_trajectory
    assert trajectory is not None
    assert bundle.generated_trajectory is True
    assert bundle.run_report["status"] == "ran"
    assert trajectory.source["kind"] == "gpcrmd_mlx_nvt"
    assert trajectory.source["workflow"] == "run_gpcrmd_mlx"
    assert trajectory.source["target_id"] == target_id
    assert trajectory.source["dynamics_id"] == 17
    assert trajectory.source["source"] == "mlx_atomistic"
    assert trajectory.source["reference_role"] == "comparison_only"
    assert trajectory.source["trajectory_public"] is False
    assert trajectory.frame_count == 3
    assert trajectory.ligand_indices.size == 1
    assert trajectory.receptor_indices.size == 1
    assert trajectory.water_indices is not None
    assert trajectory.water_indices.size == 0
    assert trajectory.ion_indices is not None
    assert trajectory.ion_indices.size == 0
    assert bundle.metadata["source"] == "mlx_atomistic"
    assert bundle.metadata["kind"] == "gpcrmd_mlx_nvt"
    assert bundle.metadata["workflow"] == "run_gpcrmd_mlx"
    assert bundle.metadata["gpcrmd_target_id"] == target_id
    assert {"temperature_K", "pressure", "constraint_max_error_A"} <= set(
        bundle.diagnostics.columns
    )
    assert np.isfinite(bundle.diagnostics["total_energy_kJ_mol"]).all()
    assert np.isfinite(bundle.diagnostics["pressure"]).all()


def test_gpcrmd_mlx_helper_returns_blockers_without_processed_trajectory(tmp_path: Path):
    bundle = ensure_gpcrmd_mlx_bundle(
        out_dir=tmp_path / "missing-gpcrmd-mlx",
        target_id="missing-target",
        steps=2,
        sample_interval=1,
    )

    assert bundle.processed_trajectory is None
    assert bundle.diagnostics.empty
    assert bundle.run_report["status"] == "blocked"
    assert bundle.run_report["trajectory_path"] is None
    assert any(item.startswith("missing_prepared_artifact:") for item in bundle.blockers)
    assert "missing_prepared_artifact" in bundle.blocker_json()
    assert not bundle.trajectory_path.exists()


def test_gpcrmd_mlx_notebook_loader_blocks_missing_artifact(tmp_path: Path):
    bundle = load_gpcrmd_mlx_artifact(out_dir=tmp_path / "missing-gpcrmd-mlx")

    assert bundle.processed_trajectory is None
    assert bundle.diagnostics.empty
    assert bundle.run_report["status"] == "blocked"
    assert "missing_mlx_artifact" in bundle.blocker_json()


def test_gpcrmd_mlx_helper_loads_existing_verified_bundle(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory

    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    second = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=8,
        sample_interval=2,
        dt=0.0005,
        temperature=0.0,
        force=False,
        minimize_steps=0,
        equilibration_steps=0,
    )

    trajectory = second.processed_trajectory
    assert trajectory is not None
    record = load_npz_trajectory(second.trajectory_path)

    assert first.generated_trajectory is True
    assert second.generated_trajectory is False
    assert trajectory.source["kind"] == "gpcrmd_mlx_nvt"
    assert trajectory.source["engine"] == "mlx_atomistic"
    assert record.sampled_positions.shape == (3, trajectory.atom_count, 3)
    assert record.metadata["engine"] == "mlx_atomistic"
    assert record.metadata["workflow"] == "run_gpcrmd_mlx"


def test_gpcrmd_mlx_notebook_loader_loads_existing_verified_artifact(tmp_path: Path):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )

    loaded = load_gpcrmd_mlx_artifact(out_dir=out_dir, target_id=target_id)

    assert loaded.processed_trajectory is not None
    assert loaded.generated_trajectory is False
    assert loaded.processed_trajectory.source["engine"] == "mlx_atomistic"
    assert loaded.processed_trajectory.source["workflow"] == "run_gpcrmd_mlx"


def test_gpcrmd_mlx_helper_blocks_existing_bundle_for_different_requested_target(
    tmp_path: Path,
):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )

    blocked = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id="different-requested-target",
        force=False,
    )

    assert first.processed_trajectory is not None
    assert blocked.processed_trajectory is None
    assert blocked.diagnostics.empty
    assert blocked.run_report["status"] == "blocked"
    assert blocked.run_report["trajectory_path"] is None
    assert blocked.trajectory_path.exists()
    assert blocked.blockers == (
        "existing_gpcrmd_target_mismatch:"
        f"requested=different-requested-target:existing={target_id}",
    )
    assert "different-requested-target" in blocked.blocker_json()


def test_gpcrmd_mlx_helper_blocks_when_existing_prepared_artifact_is_missing(
    tmp_path: Path,
):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    (out_dir / "prepared_system.json").unlink()

    blocked = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        force=False,
    )

    assert first.processed_trajectory is not None
    assert first.trajectory_path.exists()
    assert first.report_path.exists()
    assert blocked.processed_trajectory is None
    assert blocked.diagnostics.empty
    assert blocked.run_report["status"] == "blocked"
    assert blocked.run_report["trajectory_path"] is None
    assert blocked.trajectory_path.exists()
    assert any(item.startswith("missing_prepared_artifact:") for item in blocked.blockers)
    assert "prepared_system.json" in blocked.blocker_json()


def test_gpcrmd_mlx_helper_blocks_when_existing_trajectory_is_corrupt(
    tmp_path: Path,
):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    first.trajectory_path.write_text("not a valid npz trajectory\n")

    blocked = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        force=False,
    )

    assert first.processed_trajectory is not None
    assert first.report_path.exists()
    assert blocked.processed_trajectory is None
    assert blocked.diagnostics.empty
    assert blocked.run_report["status"] == "blocked"
    assert blocked.run_report["trajectory_path"] is None
    assert blocked.trajectory_path.exists()
    assert any(
        item.startswith("trajectory_artifact_load_failed:")
        for item in blocked.blockers
    )
    assert "trajectory_artifact_load_failed" in blocked.blocker_json()


def test_gpcrmd_mlx_helper_blocks_mismatched_existing_prepared_artifact(
    tmp_path: Path,
):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    metadata_path = out_dir / "prepared_system.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source"]["gpcrmd_target_id"] = "different-gpcrmd-target"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    blocked = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        force=False,
    )

    assert first.processed_trajectory is not None
    assert first.trajectory_path.exists()
    assert first.report_path.exists()
    assert blocked.processed_trajectory is None
    assert blocked.diagnostics.empty
    assert blocked.run_report["status"] == "blocked"
    assert blocked.run_report["trajectory_path"] is None
    assert blocked.trajectory_path.exists()
    assert any(
        item.startswith("prepared_artifact_target_mismatch:")
        for item in blocked.blockers
    )
    assert "different-gpcrmd-target" in blocked.blocker_json()


def test_gpcrmd_mlx_helper_blocks_stale_large_runtime_route(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    first = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    monkeypatch.setattr(mlx_real_md, "GPCRMD_FAST_RUNTIME_ATOM_LIMIT", 0)

    blocked = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        force=False,
    )

    assert first.processed_trajectory is not None
    assert blocked.processed_trajectory is None
    assert blocked.run_report["status"] == "blocked"
    assert any(item.startswith("runtime_nonbonded_backend:") for item in blocked.blockers)


def test_solvent_and_hbond_analysis_have_frame_aligned_axes(tmp_path: Path):
    out_dir = tmp_path / "gpcrmd-mlx"
    target_id = _write_gpcrmd_prepared_fixture(out_dir)
    bundle = ensure_gpcrmd_mlx_bundle(
        out_dir=out_dir,
        target_id=target_id,
        steps=4,
        sample_interval=2,
        dt=0.0005,
        temperature=0.0,
        force=True,
        minimize_steps=0,
        equilibration_steps=0,
        restraint_k=0.0,
        diagnostic_interval=1,
        constraint_max_iterations=8,
    )
    trajectory = bundle.processed_trajectory
    assert trajectory is not None
    water_counts = water_counts_around_ligand(trajectory)
    hbonds = hydrogen_bond_counts(trajectory)
    tables = analysis_tables(trajectory)

    assert water_counts.shape == (trajectory.frame_count,)
    assert hbonds.shape[0] == trajectory.frame_count
    assert tables["frames"].shape[0] == trajectory.frame_count
    assert {"ligand_receptor_hbond_count", "water_involved_hbond_count"} <= set(hbonds.columns)
