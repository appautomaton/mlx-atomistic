from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _rb_prepared_system():
    from mlx_atomistic.prep.io import synthetic_prepared_system

    prepared = synthetic_prepared_system()
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = replace(
        prepared.metadata,
        compatibility_report={
            "supported_terms": ["rb_dihedral"],
            "unsupported_terms": [],
        },
    )
    return replace(
        prepared,
        metadata=metadata,
        symbols=np.asarray(["H", "C", "C", "H"], dtype=str),
        atom_names=np.asarray(["H1", "C1", "C2", "H2"], dtype=str),
        atom_types=np.asarray(["H", "CT", "CT", "H"], dtype=str),
        residue_names=np.asarray(["LIG"] * 4, dtype=str),
        residue_ids=np.ones((4,), dtype=np.int32),
        chain_ids=np.asarray(["A"] * 4, dtype=str),
        positions=positions,
        velocities=np.zeros((4, 3), dtype=np.float32),
        masses=np.asarray([1.008, 12.011, 12.011, 1.008], dtype=np.float32),
        charges=np.zeros((4,), dtype=np.float32),
        sigma=np.ones((4,), dtype=np.float32),
        epsilon=np.zeros((4,), dtype=np.float32),
        bonds=np.empty((0, 2), dtype=np.int32),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
        angles=np.empty((0, 3), dtype=np.int32),
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=np.empty((0, 4), dtype=np.int32),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=np.empty((0, 2), dtype=np.int32),
        ligand_mask=np.ones((4,), dtype=bool),
        receptor_mask=np.zeros((4,), dtype=bool),
        restraint_mask=np.zeros((4,), dtype=bool),
        reference_positions=positions.copy(),
        rb_dihedrals=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        rb_c0=np.asarray([0.1], dtype=np.float32),
        rb_c1=np.asarray([0.2], dtype=np.float32),
        rb_c2=np.asarray([0.3], dtype=np.float32),
        rb_c3=np.asarray([0.4], dtype=np.float32),
        rb_c4=np.asarray([0.5], dtype=np.float32),
        rb_c5=np.asarray([0.6], dtype=np.float32),
    )


def _pme_prepared_system(assignment_order: int):
    from mlx_atomistic.prep.io import synthetic_prepared_system

    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        pme_config={
            "mesh_shape": [8, 8, 8],
            "alpha": 0.35,
            "real_cutoff": 5.0,
            "assignment_order": assignment_order,
            "charge_tolerance": 1e-5,
        },
        compatibility_report={
            **prepared.metadata.compatibility_report,
            "electrostatics_model": "pme",
            "supported_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "pme",
            ],
            "required_terms": [
                *prepared.metadata.compatibility_report["supported_terms"],
                "pme",
            ],
        },
    )
    return replace(
        prepared,
        metadata=metadata,
        cell_lengths=np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
        pme_mesh_shape=np.asarray([8, 8, 8], dtype=np.int32),
        pme_alpha=np.asarray([0.35], dtype=np.float32),
        pme_real_cutoff=np.asarray([5.0], dtype=np.float32),
        pme_assignment_order=np.asarray([assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([1e-5], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([True], dtype=bool),
    )


def test_core_import_does_not_require_prep_dependencies():
    import mlx_atomistic

    assert mlx_atomistic.__version__


def test_core_source_does_not_import_external_md_engines():
    source_root = Path("src/mlx_atomistic")
    source = "\n".join(path.read_text() for path in source_root.rglob("*.py"))

    assert "import openmm" not in source.lower()
    assert "pdbfixer" not in source.lower()


def test_mlx_prep_import_and_dependency_report():
    import mlx_atomistic.prep
    from mlx_atomistic.prep.prepare import MissingPrepDependencyError

    status = mlx_atomistic.prep.optional_prep_dependency_status()
    assert {"gemmi", "parmed", "rdkit"} <= set(status)
    assert "openmm" not in status
    assert "pdbfixer" not in status
    if not all(status.values()):
        with pytest.raises(
            MissingPrepDependencyError,
            match="Production biomolecular preparation needs optional parsing",
        ):
            mlx_atomistic.prep.require_production_prep_dependencies()


def test_prepared_artifact_round_trip(tmp_path: Path):
    from mlx_atomistic.prep.io import (
        load_prepared_system,
        save_prepared_system,
        synthetic_prepared_system,
    )

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    loaded = load_prepared_system(tmp_path)

    assert loaded.atom_count == prepared.atom_count
    np.testing.assert_allclose(loaded.positions, prepared.positions)
    np.testing.assert_array_equal(loaded.bonds, prepared.bonds)
    assert loaded.metadata.parameter_source == "synthetic_test"


def test_prepared_artifact_round_trips_rb_arrays(tmp_path: Path):
    from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system

    prepared = _rb_prepared_system()
    save_prepared_system(prepared, tmp_path)
    loaded = load_prepared_system(tmp_path)

    np.testing.assert_array_equal(loaded.rb_dihedrals, prepared.rb_dihedrals)
    for name in ("rb_c0", "rb_c1", "rb_c2", "rb_c3", "rb_c4", "rb_c5"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(prepared, name))


@pytest.mark.parametrize("assignment_order", [4, 5])
def test_prepared_artifact_round_trips_pme_assignment_order(
    tmp_path: Path,
    assignment_order: int,
):
    from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system

    prepared = _pme_prepared_system(assignment_order)
    save_prepared_system(prepared, tmp_path)
    loaded = load_prepared_system(tmp_path)

    assert loaded.metadata.pme_config["assignment_order"] == assignment_order
    assert int(loaded.pme_assignment_order[0]) == assignment_order


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("rb_dihedrals", np.asarray([0, 1, 2, 3], dtype=np.int32), "rb_dihedrals"),
        ("rb_dihedrals", np.asarray([[0, 1, 2, 8]], dtype=np.int32), "rb_dihedrals"),
        ("rb_c3", np.asarray([0.4, 0.5], dtype=np.float32), "rb_c3"),
        ("rb_c2", np.asarray([np.nan], dtype=np.float32), "rb_c2"),
    ],
)
def test_prepared_system_validate_rejects_invalid_rb_arrays(field_name, value, match):
    prepared = replace(_rb_prepared_system(), **{field_name: value})

    with pytest.raises(ValueError, match=match):
        prepared.validate()


def test_prepared_artifact_round_trips_cell_matrix(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import (
        load_prepared_system,
        save_prepared_system,
        synthetic_prepared_system,
    )

    matrix = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 0.25, 2.0],
        ],
        dtype=np.float32,
    )
    prepared = replace(
        synthetic_prepared_system(),
        cell_lengths=np.linalg.norm(matrix, axis=1).astype(np.float32),
        cell_matrix=matrix,
    )

    save_prepared_system(prepared, tmp_path)
    loaded = load_prepared_system(tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path)

    np.testing.assert_allclose(loaded.cell_matrix, matrix)
    np.testing.assert_allclose(artifact.arrays["cell_matrix"], matrix)
    assert artifact.cell is not None
    np.testing.assert_allclose(np.asarray(artifact.cell.matrix), matrix)


def test_gpcrmd_protocol_box_preserves_cell_matrix():
    from mlx_atomistic.prep.gpcrmd import _apply_gpcrmd_protocol_box
    from mlx_atomistic.prep.io import synthetic_prepared_system

    matrix = np.asarray(
        [
            [20.0, 0.0, 0.0],
            [2.0, 19.0, 0.0],
            [0.5, 1.0, 18.0],
        ],
        dtype=np.float32,
    )
    prepared = _apply_gpcrmd_protocol_box(
        synthetic_prepared_system(),
        {
            "box_vectors": matrix.astype(float).tolist(),
            "cell_lengths": np.linalg.norm(matrix, axis=1).astype(float).tolist(),
        },
    )

    np.testing.assert_allclose(prepared.cell_matrix, matrix)
    np.testing.assert_allclose(prepared.cell_lengths, np.linalg.norm(matrix, axis=1))


def test_build_mlx_system_matches_artifact_counts():
    from mlx_atomistic.prep.io import synthetic_prepared_system
    from mlx_atomistic.prep.runner import build_mlx_system

    prepared = synthetic_prepared_system()
    system, terms = build_mlx_system(prepared, receptor_mass_scale=1.0)

    assert system.atom_count == prepared.atom_count
    assert system.topology.n_atoms == prepared.atom_count
    assert len(terms) >= 1


@pytest.mark.slow
def test_tiny_prepared_system_runs_mlx_nvt(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    trajectory_path = tmp_path / "trajectory.npz"
    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    result = run_mlx(
        tmp_path,
        out=trajectory_path,
        steps=4,
        sample_interval=2,
        dt=0.0005,
        temperature=0.0,
        receptor_mass_scale=1.0,
    )
    record = load_npz_trajectory(trajectory_path)

    assert np.asarray(result.sampled_positions).shape == (3, 2, 3)
    assert record.sampled_positions.shape == (3, 2, 3)
    assert record.metadata["kind"] == "mlx_atomistic.prep_nvt"


def test_run_mlx_rejects_npt_without_supported_barostat_before_system_build(
    tmp_path: Path,
    monkeypatch,
):
    from mlx_atomistic.prep import runner
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.protocols import ProtocolCompatibilityError

    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        protocol_metadata={
            "ensemble": "NPT",
        },
    )
    prepared = replace(prepared, metadata=metadata)
    save_prepared_system(prepared, tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("system build started before protocol gate")

    monkeypatch.setattr(runner, "build_mlx_system_from_artifact", fail_if_called)

    with pytest.raises(ProtocolCompatibilityError) as exc_info:
        runner.run_mlx(
            tmp_path,
            out=tmp_path / "trajectory.npz",
            steps=1,
            sample_interval=1,
            minimize_steps=0,
            equilibration_steps=0,
        )

    assert exc_info.value.blockers == ("barostat",)
    assert not (tmp_path / "trajectory.npz").exists()


def test_run_mlx_persists_normalized_nvt_protocol_metadata(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    trajectory_path = tmp_path / "trajectory.npz"
    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        protocol_metadata={
            "ensemble": "nvt",
            "barostat": "none",
        },
    )
    save_prepared_system(replace(prepared, metadata=metadata), tmp_path)
    run_mlx(
        tmp_path,
        out=trajectory_path,
        steps=2,
        sample_interval=1,
        temperature=0.0,
        minimize_steps=0,
        equilibration_steps=0,
        metadata_overrides={"ensemble": "overridden"},
    )
    record = load_npz_trajectory(trajectory_path)

    assert record.metadata["ensemble"] == "NVT"
    assert record.metadata["proof_mode"] == "short_nvt"
    assert record.metadata["barostat"] == "none"
    assert record.metadata["barostat_status"] == "not_required_for_nvt_proof"
    assert record.metadata["unsupported_protocol_blockers"] == []
    readiness = record.metadata["platform_readiness"]
    assert readiness["artifact"]["name"] == "artifact"
    assert readiness["artifact"]["status"] == "proof-level"
    assert readiness["protocol"]["name"] == "protocol"
    assert readiness["protocol"]["status"] == "proof-level"
    assert readiness["protocol"]["blockers"] == []


def test_artifact_readiness_report_blocks_unsupported_terms():
    from mlx_atomistic.artifacts import artifact_readiness_report

    report = artifact_readiness_report(
        {
            "compatibility_report": {
                "supported_terms": [],
                "unsupported_terms": ["barostat"],
            }
        },
        require_production=True,
    )

    assert report.name == "artifact"
    assert report.status == "blocked"
    assert any("barostat" in blocker for blocker in report.blockers)


@pytest.mark.slow
def test_solvated_ligand_receptor_replicas_write_selected_and_all_outputs(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.replicas import run_ligand_receptor_replicas

    summary = run_ligand_receptor_replicas(
        tmp_path / "replicas",
        replicas=2,
        selected_replica=0,
        steps=4,
        sample_interval=2,
        dt=0.001,
        water_count=4,
        minimize_steps=0,
        equilibration_steps=0,
        save_all_replicas=True,
        force=True,
    )
    record = load_npz_trajectory(summary.selected_trajectory_path)

    assert summary.all_replicas_trajectory_path is not None
    assert summary.all_replicas_trajectory_path.exists()
    assert summary.metadata["source"] == "mlx_atomistic"
    assert summary.metadata["workflow"].endswith("_replicas")
    assert summary.metadata["replicas"] == 2
    assert summary.metadata["gpu_visible_atoms"] == summary.metadata["atoms_per_replica"] * 2
    assert record.sampled_positions.shape[0] == 3
    assert record.sampled_positions.shape[1] == summary.metadata["atoms_per_replica"]
    assert np.isfinite(record.total_energy).all()
    assert float(np.max(record.constraint_max_error)) < 5e-4


@pytest.mark.slow
def test_ligand_receptor_performance_profile_emits_aggregate_rows(tmp_path: Path):
    from mlx_atomistic.prep.replicas import profile_ligand_receptor_performance

    rows = profile_ligand_receptor_performance(
        tmp_path / "profile",
        durations_ps=[0.002],
        replica_counts=[1, 2],
        dt=0.001,
        sample_interval=1,
        water_count=4,
        minimize_steps=0,
        equilibration_steps=0,
        force=True,
        write_json=True,
        write_csv=True,
    )

    assert len(rows) == 2
    assert (tmp_path / "profile/performance_profile.json").exists()
    assert (tmp_path / "profile/performance_profile.csv").exists()
    assert rows[0]["replicas"] == 1
    assert rows[1]["replicas"] == 2
    assert rows[1]["gpu_visible_atoms"] == rows[1]["atoms_per_replica"] * 2
    assert rows[1]["aggregate_steps_per_s"] > 0
    assert rows[1]["force_total_ms"] >= 0
    assert rows[0]["constraint_max_iterations"] == 40


@pytest.mark.slow
def test_ligand_receptor_replicas_rerun_when_constraint_iterations_change(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.replicas import run_ligand_receptor_replicas

    out_dir = tmp_path / "replicas"
    run_ligand_receptor_replicas(
        out_dir,
        replicas=2,
        steps=2,
        sample_interval=1,
        water_count=4,
        minimize_steps=0,
        equilibration_steps=0,
        constraint_max_iterations=20,
        force=True,
    )
    summary = run_ligand_receptor_replicas(
        out_dir,
        replicas=2,
        steps=2,
        sample_interval=1,
        water_count=4,
        minimize_steps=0,
        equilibration_steps=0,
        constraint_max_iterations=12,
        force=False,
    )
    record = load_npz_trajectory(summary.selected_trajectory_path)

    assert summary.generated_trajectory is True
    assert record.metadata["constraint_max_iterations"] == 12


@pytest.mark.slow
def test_notebook_bundle_loader_uses_full_prepared_artifact(tmp_path: Path):
    pytest.importorskip("MDAnalysis")
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.notebook import (
        load_prepared_trajectory_bundle,
        make_mdanalysis_universe,
    )
    from mlx_atomistic.prep.runner import run_mlx

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    run_mlx(
        tmp_path,
        steps=2,
        sample_interval=1,
        temperature=0.0,
        receptor_mass_scale=1.0,
    )
    bundle = load_prepared_trajectory_bundle(tmp_path)

    assert bundle.prepared.atom_count == prepared.atom_count
    assert len(bundle.universe.atoms) == prepared.atom_count
    assert bundle.trajectory.sampled_positions.shape[1] == prepared.atom_count
    assert bundle.trajectory.symbols == tuple(prepared.symbols.tolist())

    viewer_universe = make_mdanalysis_universe(
        bundle.view_path,
        bundle.trajectory.sampled_positions,
        dt=0.25,
    )
    assert viewer_universe is not bundle.universe
    assert len(viewer_universe.atoms) == prepared.atom_count
    assert len(viewer_universe.trajectory) == len(bundle.universe.trajectory)


def test_notebook_multimodel_pdb_export_has_ordered_frames():
    from mlx_atomistic.prep.io import synthetic_prepared_system
    from mlx_atomistic.prep.notebook import PreparedTrajectoryRecord, trajectory_to_multimodel_pdb

    prepared = synthetic_prepared_system()
    positions = np.stack(
        [
            prepared.positions,
            prepared.positions + np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
        ]
    )
    record = PreparedTrajectoryRecord(
        sampled_positions=positions,
        sampled_steps=np.asarray([0, 10]),
        sampled_time=np.asarray([0.0, 0.005]),
        symbols=tuple(prepared.symbols.tolist()),
        metadata={},
    )

    pdb = trajectory_to_multimodel_pdb(prepared, record)

    assert pdb.count("MODEL") == 2
    assert pdb.index("MODEL        1") < pdb.index("MODEL        2")
    assert pdb.count("ENDMDL") == 2
    assert "CONECT    1    2" in pdb


def test_py3dmol_frame_player_html_has_visible_controls():
    from mlx_atomistic.prep.notebook import py3dmol_frame_player_html

    class FakeView:
        uniqueid = "test123"

        def write_html(self):
            return '<div id="3dmolviewer_test123"></div><script>var viewer_test123 = {};</script>'

    html = py3dmol_frame_player_html(
        FakeView(),
        sampled_steps=np.asarray([0, 10, 20]),
        sampled_time=np.asarray([0.0, 0.005, 0.01]),
    )

    assert 'data-role="play"' in html
    assert 'data-role="slider"' in html
    assert "viewer_test123" in html
    assert ".setFrame(frame)" in html
    assert 'label.textContent = "frame "' in html
    assert "steps.length - 1" in html


def test_4dw1_pocket_preparation_smoke():
    from mlx_atomistic.prep.prepare import prepare_p2x4_atp

    pdb_path = Path("notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb")
    if not pdb_path.exists():
        pytest.skip("4DW1 notebook data is not present")
    prepared = prepare_p2x4_atp(
        pdb_path=pdb_path,
        cutoff_angstrom=5.0,
        backend="generic_mlx",
    )

    assert prepared.atom_count > int(np.count_nonzero(prepared.ligand_mask))
    assert prepared.bonds.shape[0] > 0
    assert prepared.nonbonded_pairs.shape[1] == 2
    assert not prepared.metadata.compatibility_report["production_force_field"]


def test_4dw1_production_builder_exports_explicit_h_mlx_artifact(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.prepare import prepare_p2x4_atp

    pdb_path = Path("notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb")
    if not pdb_path.exists():
        pytest.skip("4DW1 notebook data is not present")
    prepared = prepare_p2x4_atp(
        pdb_path=pdb_path,
        cutoff_angstrom=5.0,
        backend="production_mlx",
    )
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
    report = prepared.metadata.compatibility_report

    assert artifact.atom_count == prepared.atom_count
    assert report["production_force_field"] is True
    assert report["hydrogens_present"] is True
    assert report["hydrogen_count"] > 0
    assert prepared.bonds.shape[0] > 0
    assert prepared.angles.shape[0] > 0
    assert prepared.dihedrals.shape[0] > 0
    assert prepared.impropers.shape[0] > 0
    assert prepared.constraints.shape[0] > 0
    assert prepared.nonbonded_exception_pairs.shape[0] > 0
    assert prepared.metadata.parameter_source.startswith("mlx_internal_p2x4_atp_pocket")


def test_4dw1_production_builder_runs_short_mlx_nvt(tmp_path: Path):
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.prepare import prepare_p2x4_atp
    from mlx_atomistic.prep.runner import run_mlx

    pdb_path = Path("notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb")
    if not pdb_path.exists():
        pytest.skip("4DW1 notebook data is not present")
    prepared = prepare_p2x4_atp(
        pdb_path=pdb_path,
        cutoff_angstrom=5.0,
        backend="production_mlx",
    )
    save_prepared_system(prepared, tmp_path)
    result = run_mlx(
        tmp_path,
        steps=4,
        sample_interval=2,
        dt=0.0005,
        temperature=0.0,
        restraint_k=5.0,
        require_production=True,
        minimize_steps=2,
        equilibration_steps=0,
    )

    assert np.asarray(result.sampled_positions).shape == (3, prepared.atom_count, 3)
    assert np.isfinite(np.asarray(result.sampled_positions)).all()
    assert np.isfinite(np.asarray(result.total_energy)).all()


def test_t4l_benzene_fixture_exports_complete_internal_smd_artifact(tmp_path: Path):
    from mlx_atomistic.artifacts import MLXCompatibilityError, load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.t4l_benzene import (
        T4L_BENZENE_PARAMETER_SOURCE,
        prepare_t4l_benzene,
    )

    prepared = prepare_t4l_benzene()
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=False)

    assert artifact.atom_count == prepared.atom_count
    assert prepared.metadata.source["pdb_id"] == "4W52"
    assert prepared.metadata.parameter_source == T4L_BENZENE_PARAMETER_SOURCE
    assert prepared.metadata.compatibility_report["hydrogens_present"] is True
    assert prepared.metadata.compatibility_report["production_force_field"] is False
    assert int(np.count_nonzero(prepared.symbols == "H")) > 0
    assert int(np.count_nonzero(prepared.ligand_mask)) == 12
    assert int(np.count_nonzero(prepared.receptor_mask)) > 0
    assert prepared.constraints.shape[0] > 0
    assert prepared.nonbonded_exception_pairs.shape[0] > 0

    with pytest.raises(MLXCompatibilityError, match="not marked as a production"):
        load_prepared_mlx_artifact(tmp_path, require_production=True)


@pytest.mark.slow
def test_t4l_benzene_steered_run_writes_cv_trace(tmp_path: Path):
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.runner import run_steered_mlx
    from mlx_atomistic.prep.t4l_benzene import prepare_t4l_benzene

    save_prepared_system(prepare_t4l_benzene(), tmp_path)
    out = tmp_path / "steered_trajectory.npz"
    result = run_steered_mlx(
        tmp_path,
        out=out,
        steps=20,
        sample_interval=5,
        dt=0.001,
        temperature=300.0,
        minimize_steps=2,
        equilibration_steps=2,
        bias_k=200.0,
        target_velocity=0.5,
    )

    with np.load(out, allow_pickle=False) as data:
        assert "sampled_cv" in data.files
        assert "sampled_target" in data.files
        assert "sampled_bias_energy" in data.files
        assert "sampled_work" in data.files
        np.testing.assert_allclose(data["sampled_cv"], np.asarray(result.sampled_cv))
        assert float(data["sampled_target"][-1]) > float(data["sampled_target"][0])
        assert np.isfinite(data["sampled_positions"]).all()


def test_ligand_receptor_motion_notebook_uses_gpcrmd_mlx_main_path():
    notebook_path = Path(
        "notebooks/ligand-receptor-motion/01-ligand-receptor-translational-motion.ipynb"
    )
    cells = json.loads(notebook_path.read_text())["cells"]
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    helper_dir = Path("notebooks/ligand-receptor-motion/helpers")
    helper_source = "\n".join(path.read_text() for path in helper_dir.glob("*.py"))
    readme = Path("notebooks/ligand-receptor-motion/README.md").read_text()
    gitignore = Path(".gitignore").read_text()

    assert not Path("notebooks/macromolecule-viz").exists()
    assert Path("notebooks/archive/atp-pocket-mlx-demo").exists()
    assert "generic_mlx" not in source
    assert "mlx_atomistic_demo" not in source
    assert "VISUALIZATION_MOTION_SCALE" not in source
    assert "ATP center-of-mass translation" not in source
    assert "public_md" not in source
    assert "mlx_steered_md" not in source
    assert "ensure_gpcrmd_mlx_bundle" in source
    assert "run_gpcrmd_mlx" in source
    assert "run_gpcrmd_mlx" in readme
    assert "run-ligand-receptor-example" not in readme
    assert "run-steered-mlx" not in readme
    assert "make_ligand_motion_figure" in source
    assert "active ligand pose" in helper_source
    assert "initial ligand pose" in helper_source
    assert "ligand COM path" in helper_source
    assert "analysis_tables" in source
    assert "GPCRMD_MLX_STEPS" in source
    assert "short_range_electrostatics_prototype" not in source
    assert "notebooks/ligand-receptor-motion/data/gpcrmd-mlx/" in gitignore
    assert "notebooks/ligand-receptor-motion/data/cache/" in gitignore


def test_gpcrmd_production_neighbor_manager_uses_auto_backend_policy():
    from mlx_atomistic.core import Cell
    from mlx_atomistic.prep.runner import GPCRMD_NEIGHBOR_SKIN, _production_neighbor_manager
    from mlx_atomistic.topology import Topology

    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        eager_nonbonded_pair_limit=0,
    )
    system = SimpleNamespace(cell=Cell.cubic(5.0))
    term = SimpleNamespace(topology=topology, cutoff=1.6, electrostatics="cutoff")

    manager = _production_neighbor_manager(system, (term,), require_production=True)

    assert manager is not None
    assert manager.backend == "auto"
    assert manager.skin == GPCRMD_NEIGHBOR_SKIN

    tuned_manager = _production_neighbor_manager(
        system,
        (term,),
        require_production=True,
        neighbor_skin=1.25,
        neighbor_check_interval=4,
    )
    assert tuned_manager is not None
    assert tuned_manager.skin == 1.25
    assert tuned_manager.check_interval == 4


def test_solvated_ligand_receptor_builder_exports_complete_runtime_artifact(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.solvated_example import (
        ELECTROSTATICS_MODEL,
        SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE,
        prepare_solvated_ligand_receptor_example,
        validate_complete_solvated_ligand_receptor_system,
    )

    prepared = prepare_solvated_ligand_receptor_example(water_count=4)
    validate_complete_solvated_ligand_receptor_system(prepared)
    save_prepared_system(prepared, tmp_path)
    artifact = load_prepared_mlx_artifact(tmp_path, require_production=False)

    assert artifact.atom_count == prepared.atom_count
    assert prepared.metadata.parameter_source == SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE
    assert prepared.metadata.compatibility_report["electrostatics_model"] == ELECTROSTATICS_MODEL
    assert prepared.metadata.compatibility_report["physical_units"] is True
    assert int(np.count_nonzero(prepared.ligand_mask)) == 12
    assert int(np.count_nonzero(prepared.receptor_mask)) > 0
    assert int(np.count_nonzero(prepared.water_mask)) == 12
    assert int(np.count_nonzero(prepared.ion_mask)) == 2
    assert prepared.cell_lengths.shape == (3,)
    assert prepared.constraints.shape[0] > 0
    assert prepared.nonbonded_exception_pairs.shape[0] > 0


@pytest.mark.slow
def test_run_ligand_receptor_example_writes_mlx_trajectory(tmp_path: Path):
    from mlx_atomistic.io import load_npz_trajectory
    from mlx_atomistic.prep.solvated_example import ensure_solvated_ligand_receptor_example

    status = ensure_solvated_ligand_receptor_example(
        tmp_path / "example",
        steps=10,
        dt=0.001,
        sample_interval=5,
        water_count=4,
        minimize_steps=2,
        equilibration_steps=2,
        force=True,
    )
    record = load_npz_trajectory(status["trajectory_path"])

    assert record.metadata["source"] == "mlx_atomistic"
    assert record.metadata["workflow"] == "mlx_ligand_receptor_solvated_nvt_v1"
    assert record.metadata["electrostatics_model"] == "short_range_electrostatics_prototype"
    assert record.sampled_positions.shape[0] == 3
    assert np.isfinite(record.sampled_positions).all()


@pytest.mark.slow
def test_import_amber_tiny_topology_runs_mlx(tmp_path: Path):
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact
    from mlx_atomistic.prep.io import save_prepared_system
    from mlx_atomistic.prep.runner import run_mlx
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "tiny.prmtop"
    coords = tmp_path / "tiny.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  O1
%FLAG CHARGE
%FORMAT(5E16.8)
  7.28892000E+00 -7.28892000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.59990000E+01
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H   O
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
LIG
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       3       3       2
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  9.60000000E-01
"""
    )
    coords.write_text(
        """tiny
    2
  0.0000000  0.0000000  0.0000000  0.9600000  0.0000000  0.0000000
"""
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    save_prepared_system(prepared, tmp_path / "prepared")
    artifact = load_prepared_mlx_artifact(tmp_path / "prepared", require_production=True)
    result = run_mlx(
        tmp_path / "prepared",
        steps=2,
        sample_interval=1,
        temperature=0.0,
        require_production=True,
    )

    assert artifact.atom_count == 2
    assert artifact.metadata["compatibility_report"]["hydrogen_count"] == 1
    assert np.asarray(result.sampled_positions).shape == (3, 2, 3)


def test_import_amber_parity_fixture_preserves_phase2_terms():
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = Path("tests/fixtures/amber/alanine-dipeptide-implicit.prmtop")
    coords = Path("tests/fixtures/amber/alanine-dipeptide-implicit.inpcrd")

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    report = prepared.metadata.compatibility_report
    term_counts = report["term_counts"]
    scaling = report["term_details"]["nonbonded_exception"]["amber_14_scaling"]

    assert prepared.atom_count == 22
    assert prepared.bonds.shape == (21, 2)
    assert prepared.angles.shape == (36, 3)
    assert prepared.dihedrals.shape == (48, 4)
    assert prepared.impropers.shape == (4, 4)
    assert prepared.constraints.shape == (12, 2)
    assert prepared.nonbonded_exception_pairs.shape == (98, 2)
    assert prepared.charges.shape == (22,)
    assert prepared.sigma.shape == (22,)
    assert prepared.epsilon.shape == (22,)
    assert report["unsupported_terms"] == []
    assert report["periodic_box_present"] is False
    assert term_counts["amber_14_exceptions"] == 41
    assert term_counts["amber_excluded_pairs"] == 57
    assert scaling["source"] == "standard_amber_fallback"
    assert scaling["electrostatic_scale_values"] == [round(1.0 / 1.2, 10)]
    assert scaling["lj_scale_values"] == [0.5]


def test_import_amber_uses_topology_scee_scnb_for_14_exceptions(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "scaled.prmtop"
    coords = tmp_path / "scaled.inpcrd"
    charge_scale = 18.2223
    charge_line = "".join(
        f"{charge * charge_scale:16.8E}" for charge in (0.25, -0.10, 0.20, -0.40)
    )
    prmtop.write_text(
        f"""%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       4       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  C1  C2  H2
%FLAG CHARGE
%FORMAT(5E16.8)
{charge_line}
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.20110000E+01  1.20110000E+01  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H   C   C   H
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
LIG
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       3       2       1       0
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       3       4       4
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1       6       9       1
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       3       6       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.00000000E+00
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       3       6       9       1
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  5.00000000E+01
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  1.91000000E+00
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
       3       6       0       9       1
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  2.00000000E-01
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.60000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  4.00000000E+00
"""
    )
    coords.write_text(
        """scaled
    4
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
  2.0000000  0.0000000  0.0000000  3.0000000  0.0000000  0.0000000
"""
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)
    assert prepared.dihedrals.tolist() == [[1, 2, 0, 3]]
    pairs = [tuple(pair) for pair in prepared.nonbonded_exception_pairs.tolist()]
    pair_index = pairs.index((1, 3))
    scaling = prepared.metadata.compatibility_report["term_details"]["nonbonded_exception"][
        "amber_14_scaling"
    ]

    assert scaling["source"] == "topology_scee_scnb"
    assert scaling["electrostatic_scale_values"] == [0.625]
    assert scaling["lj_scale_values"] == [0.25]
    assert prepared.nonbonded_exception_charge_product[pair_index] == pytest.approx(
        -0.10 * -0.40 / 1.6
    )
    assert prepared.nonbonded_exception_sigma[pair_index] == pytest.approx(1.0)
    assert prepared.nonbonded_exception_epsilon[pair_index] == pytest.approx(4.184 / 4.0)


def test_import_amber_periodic_restart_box_metadata(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "tiny.prmtop"
    coords = tmp_path / "tiny.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2       0       0       0       0       0       0       0       0
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0       0       1       0       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  O1
%FLAG CHARGE
%FORMAT(5E16.8)
  7.28892000E+00 -7.28892000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.59990000E+01
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H   O
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
LIG
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       3       3       2
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  9.60000000E-01
"""
    )
    coords.write_text(
        """tiny periodic
    2
  0.0000000  0.0000000  0.0000000  0.9600000  0.0000000  0.0000000
 12.0000000 13.0000000 14.0000000 90.0000000 90.0000000 90.0000000
"""
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)

    assert prepared.metadata.compatibility_report["periodic_box_present"] is True
    np.testing.assert_allclose(prepared.cell_lengths, [12.0, 13.0, 14.0])
    np.testing.assert_allclose(prepared.cell_matrix, np.diag([12.0, 13.0, 14.0]), atol=1e-6)


def test_import_amber_periodic_topology_requires_box_metadata(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "periodic-missing-box.prmtop"
    coords = tmp_path / "periodic-missing-box.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2       0       0       0       0       0       0       0       0
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0       0       1       0       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  O1
%FLAG CHARGE
%FORMAT(5E16.8)
  7.28892000E+00 -7.28892000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.59990000E+01
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H   O
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       3       3       2
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  9.60000000E-01
"""
    )
    coords.write_text(
        """periodic missing box
    2
  0.0000000  0.0000000  0.0000000  0.9600000  0.0000000  0.0000000
"""
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_invalid_periodic_box"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_nonperiodic_velocity_only_restart_is_not_box(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "tiny.prmtop"
    coords = tmp_path / "tiny.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       2
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  O1
%FLAG CHARGE
%FORMAT(5E16.8)
  7.28892000E+00 -7.28892000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.59990000E+01
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       2
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
H   O
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
LIG
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       3       3       2
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00  8.00000000E+00  5.65685425E+00
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  9.60000000E-01
"""
    )
    coords.write_text(
        """tiny velocity
    2
  0.0000000  0.0000000  0.0000000  0.9600000  0.0000000  0.0000000
  0.1000000  0.2000000  0.3000000  0.4000000  0.5000000  0.6000000
"""
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)

    np.testing.assert_allclose(
        prepared.velocities,
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    assert prepared.cell_lengths.size == 0
    assert prepared.cell_matrix.size == 0
    assert prepared.metadata.compatibility_report["periodic_box_present"] is False


def test_import_amber_fails_closed_for_unsupported_record(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "unsupported.prmtop"
    coords = tmp_path / "unsupported.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_CCOEF
%FORMAT(5E16.8)
  1.00000000E+00
"""
    )
    coords.write_text(
        """unsupported
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_12_6_4_lj"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def _write_malformed_amber_prmtop(
    path: Path,
    *,
    ntypes: int = 1,
    atom_type_indices: tuple[int, ...] = (1, 1, 1),
    amber_atom_types: tuple[str, ...] | None = None,
    charges: tuple[float, ...] | None = None,
    masses: tuple[float, ...] | None = None,
    residue_labels: tuple[str, ...] | None = None,
    residue_pointers: tuple[int, ...] | None = None,
    nonbonded_indices: tuple[int, ...] = (1,),
    lj_acoef: tuple[float, ...] = (4.0,),
    lj_bcoef: tuple[float, ...] = (4.0,),
    bond_record: tuple[int, int, int] | None = None,
    bond_force_constants: tuple[float, ...] = (100.0,),
    bond_lengths: tuple[float, ...] = (1.0,),
    angle_record: tuple[int, int, int, int] | None = None,
    angle_force_constants: tuple[float, ...] = (50.0,),
    angle_theta: tuple[float, ...] = (1.91,),
    dihedral_record: tuple[int, int, int, int, int] | None = None,
    dihedral_force_constants: tuple[float, ...] = (0.2,),
    dihedral_periodicity: tuple[float, ...] = (3.0,),
    dihedral_phase: tuple[float, ...] = (0.0,),
    scee_scale_factors: tuple[float, ...] | None = None,
    scnb_scale_factors: tuple[float, ...] | None = None,
    exclusion_counts: tuple[int, ...] | None = None,
    exclusion_values: tuple[int, ...] = (),
) -> None:
    atom_count = len(atom_type_indices)
    atom_names = "".join(f"H{index + 1:<3}" for index in range(atom_count))
    charge_values = (0.0,) * atom_count if charges is None else charges
    mass_values = (1.008,) * atom_count if masses is None else masses
    charge_text = "".join(f"{value:16.8E}" for value in charge_values)
    mass_text = "".join(f"{value:16.8E}" for value in mass_values)
    type_indices = "".join(f"{index:8d}" for index in atom_type_indices)
    sections = [
        "%VERSION  VERSION_STAMP = V0001.000",
        "%FLAG POINTERS",
        "%FORMAT(10I8)",
        f"{atom_count:8d}{ntypes:8d}",
        "%FLAG ATOM_NAME",
        "%FORMAT(20a4)",
        atom_names,
        "%FLAG CHARGE",
        "%FORMAT(5E16.8)",
        charge_text,
        "%FLAG MASS",
        "%FORMAT(5E16.8)",
        mass_text,
        "%FLAG ATOM_TYPE_INDEX",
        "%FORMAT(10I8)",
        type_indices,
        "%FLAG NONBONDED_PARM_INDEX",
        "%FORMAT(10I8)",
        "".join(f"{value:8d}" for value in nonbonded_indices),
        "%FLAG LENNARD_JONES_ACOEF",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in lj_acoef),
        "%FLAG LENNARD_JONES_BCOEF",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in lj_bcoef),
        "%FLAG BOND_FORCE_CONSTANT",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in bond_force_constants),
        "%FLAG BOND_EQUIL_VALUE",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in bond_lengths),
        "%FLAG ANGLE_FORCE_CONSTANT",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in angle_force_constants),
        "%FLAG ANGLE_EQUIL_VALUE",
        "%FORMAT(5E16.8)",
        "".join(f"{value:16.8E}" for value in angle_theta),
    ]
    if amber_atom_types is not None:
        sections.extend(
            [
                "%FLAG AMBER_ATOM_TYPE",
                "%FORMAT(20a4)",
                "".join(f"{atom_type:<4}" for atom_type in amber_atom_types),
            ]
        )
    if residue_labels is not None:
        sections.extend(
            [
                "%FLAG RESIDUE_LABEL",
                "%FORMAT(20a4)",
                "".join(f"{label:<4}" for label in residue_labels),
            ]
        )
    if residue_pointers is not None:
        sections.extend(
            [
                "%FLAG RESIDUE_POINTER",
                "%FORMAT(10I8)",
                "".join(f"{pointer:8d}" for pointer in residue_pointers),
            ]
        )
    if bond_record is not None:
        sections.extend(
            [
                "%FLAG BONDS_INC_HYDROGEN",
                "%FORMAT(10I8)",
                "".join(f"{value:8d}" for value in bond_record),
            ]
        )
    if angle_record is not None:
        sections.extend(
            [
                "%FLAG ANGLES_INC_HYDROGEN",
                "%FORMAT(10I8)",
                "".join(f"{value:8d}" for value in angle_record),
            ]
        )
    if dihedral_record is not None:
        sections.extend(
            [
                "%FLAG DIHEDRALS_INC_HYDROGEN",
                "%FORMAT(10I8)",
                "".join(f"{value:8d}" for value in dihedral_record),
                "%FLAG DIHEDRAL_FORCE_CONSTANT",
                "%FORMAT(5E16.8)",
                "".join(f"{value:16.8E}" for value in dihedral_force_constants),
                "%FLAG DIHEDRAL_PERIODICITY",
                "%FORMAT(5E16.8)",
                "".join(f"{value:16.8E}" for value in dihedral_periodicity),
                "%FLAG DIHEDRAL_PHASE",
                "%FORMAT(5E16.8)",
                "".join(f"{value:16.8E}" for value in dihedral_phase),
            ]
        )
    if scee_scale_factors is not None:
        sections.extend(
            [
                "%FLAG SCEE_SCALE_FACTOR",
                "%FORMAT(5E16.8)",
                "".join(f"{value:16.8E}" for value in scee_scale_factors),
            ]
        )
    if scnb_scale_factors is not None:
        sections.extend(
            [
                "%FLAG SCNB_SCALE_FACTOR",
                "%FORMAT(5E16.8)",
                "".join(f"{value:16.8E}" for value in scnb_scale_factors),
            ]
        )
    if exclusion_counts is not None:
        sections.extend(
            [
                "%FLAG NUMBER_EXCLUDED_ATOMS",
                "%FORMAT(10I8)",
                "".join(f"{value:8d}" for value in exclusion_counts),
                "%FLAG EXCLUDED_ATOMS_LIST",
                "%FORMAT(10I8)",
                "".join(f"{value:8d}" for value in exclusion_values),
            ]
        )
    path.write_text("\n".join(sections) + "\n")


def _write_malformed_amber_coords(path: Path, *, atom_count: int) -> None:
    values = [
        coordinate
        for atom_index in range(atom_count)
        for coordinate in (float(atom_index), 0.0, 0.0)
    ]
    path.write_text(
        "malformed\n"
        f"{atom_count:5d}\n"
        + "".join(f"{value:12.7f}" for value in values)
        + "\n"
    )


def test_import_amber_parses_adjacent_fixed_width_numeric_fields(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "fixed-width.prmtop"
    coords = tmp_path / "fixed-width.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(2I1)
21
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(2E14.8)
0.00000000E+000.00000000E+00
%FLAG MASS
%FORMAT(2E14.8)
1.00800000E+001.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(2I1)
11
%FLAG NONBONDED_PARM_INDEX
%FORMAT(1I1)
1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(1E14.8)
4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(1E14.8)
4.00000000E+00
"""
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)

    assert prepared.atom_count == 2
    np.testing.assert_allclose(prepared.masses, [1.008, 1.008])


@pytest.mark.parametrize(
    "residue_pointers",
    [(1, 1), (1, 4)],
    ids=["nonmonotonic", "out_of_bounds"],
)
def test_import_amber_fails_closed_for_malformed_residue_pointers(
    tmp_path: Path,
    residue_pointers: tuple[int, ...],
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-residues.prmtop"
    coords = tmp_path / "bad-residues.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        residue_labels=("R1", "R2"),
        residue_pointers=residue_pointers,
    )
    _write_malformed_amber_coords(coords, atom_count=3)

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_residues"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_nonfinite_derived_lj_values(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-derived-lj.prmtop"
    coords = tmp_path / "bad-derived-lj.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        lj_acoef=(1.0e-20,),
        lj_bcoef=(1.0e20,),
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_lj_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("restart_values", "atom_count"),
    [
        ((float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0), 2),
        ((0.0, 0.0, 0.0, 0.1, float("inf"), 0.3), 1),
    ],
    ids=["nonfinite_position", "nonfinite_velocity"],
)
def test_import_amber_fails_closed_for_nonfinite_restart_values(
    tmp_path: Path,
    restart_values: tuple[float, ...],
    atom_count: int,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-restart-values.prmtop"
    coords = tmp_path / "bad-restart-values.inpcrd"
    _write_malformed_amber_prmtop(prmtop, atom_type_indices=(1,) * atom_count)
    coords.write_text(
        "bad restart values\n"
        f"{atom_count:5d}\n"
        + "".join(f"{value:12.7f}" for value in restart_values)
        + "\n"
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_topology"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_amber_atom_type_length_mismatch(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-atom-types.prmtop"
    coords = tmp_path / "bad-atom-types.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        amber_atom_types=("H",),
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_atom_arrays",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("charges", "masses"),
    [
        ((float("nan"), 0.0), None),
        (None, (0.0, 1.008)),
    ],
    ids=["nan_charge", "nonpositive_mass"],
)
def test_import_amber_fails_closed_for_invalid_per_atom_numeric_arrays(
    tmp_path: Path,
    charges: tuple[float, ...] | None,
    masses: tuple[float, ...] | None,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-atom-numeric.prmtop"
    coords = tmp_path / "bad-atom-numeric.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        charges=charges,
        masses=masses,
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_atom_arrays",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_wraps_malformed_prmtop_numeric_token(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-token.prmtop"
    coords = tmp_path / "bad-token.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       X
"""
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_topology"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("kwargs", "match", "atom_count"),
    [
        (
            {
                "atom_type_indices": (1, 1),
                "bond_record": (0, 3, 1),
                "bond_force_constants": (float("nan"),),
            },
            "unsupported_terms:amber_malformed_bond_parameters",
            2,
        ),
        (
            {
                "angle_record": (0, 3, 6, 1),
                "angle_theta": (float("inf"),),
            },
            "unsupported_terms:amber_malformed_angle_parameters",
            3,
        ),
        (
            {
                "atom_type_indices": (1, 1, 1, 1),
                "dihedral_record": (0, 3, 6, 9, 1),
                "dihedral_phase": (float("nan"),),
            },
            "unsupported_terms:amber_malformed_dihedral_parameters",
            4,
        ),
    ],
    ids=["bond", "angle", "dihedral"],
)
def test_import_amber_fails_closed_for_nonfinite_bonded_parameter_arrays(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
    atom_count: int,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-bonded-parameters.prmtop"
    coords = tmp_path / "bad-bonded-parameters.inpcrd"
    _write_malformed_amber_prmtop(prmtop, **kwargs)
    _write_malformed_amber_coords(coords, atom_count=atom_count)

    with pytest.raises(TopologyImportError, match=match):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("scee_scale_factors", "scnb_scale_factors"),
    [
        ((-1.0,), (2.0,)),
        ((1.2,), (-2.0,)),
    ],
    ids=["negative_scee", "negative_scnb"],
)
def test_import_amber_fails_closed_for_negative_14_scale_denominators(
    tmp_path: Path,
    scee_scale_factors: tuple[float, ...],
    scnb_scale_factors: tuple[float, ...],
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-14-scale.prmtop"
    coords = tmp_path / "bad-14-scale.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1, 1, 1),
        dihedral_record=(0, 3, 6, 9, 1),
        scee_scale_factors=scee_scale_factors,
        scnb_scale_factors=scnb_scale_factors,
    )
    _write_malformed_amber_coords(coords, atom_count=4)

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_invalid_14_scaling"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("bond_parameter_index", "match"),
    [
        (0, "unsupported_terms:amber_malformed_bond_parameters"),
        (2, "unsupported_terms:amber_malformed_bond_parameters"),
    ],
    ids=["zero", "out_of_range"],
)
def test_import_amber_fails_closed_for_malformed_bond_parameter_indices(
    tmp_path: Path,
    bond_parameter_index: int,
    match: str,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-bond-parameter.prmtop"
    coords = tmp_path / "bad-bond-parameter.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        bond_record=(0, 3, bond_parameter_index),
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(TopologyImportError, match=match):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_malformed_bond_record_width(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-bond-width.prmtop"
    coords = tmp_path / "bad-bond-width.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        bond_record=(0, 3),
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_bond_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_invalid_bond_atom_index(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-bond-atom.prmtop"
    coords = tmp_path / "bad-bond-atom.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1),
        bond_record=(0, 6, 1),
    )
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_bond_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("angle_parameter_index", "match"),
    [
        (0, "unsupported_terms:amber_malformed_angle_parameters"),
        (2, "unsupported_terms:amber_malformed_angle_parameters"),
    ],
    ids=["zero", "out_of_range"],
)
def test_import_amber_fails_closed_for_malformed_angle_parameter_indices(
    tmp_path: Path,
    angle_parameter_index: int,
    match: str,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-angle-parameter.prmtop"
    coords = tmp_path / "bad-angle-parameter.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        angle_record=(0, 3, 6, angle_parameter_index),
    )
    _write_malformed_amber_coords(coords, atom_count=3)

    with pytest.raises(TopologyImportError, match=match):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_malformed_angle_record_width(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-angle-width.prmtop"
    coords = tmp_path / "bad-angle-width.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        angle_record=(0, 3, 6),
    )
    _write_malformed_amber_coords(coords, atom_count=3)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_angle_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("dihedral_periodicity", "dihedral_phase"),
    [
        ((), (0.0,)),
        ((3.0,), ()),
    ],
    ids=["short_periodicity", "short_phase"],
)
def test_import_amber_fails_closed_for_short_dihedral_parameter_arrays(
    tmp_path: Path,
    dihedral_periodicity: tuple[float, ...],
    dihedral_phase: tuple[float, ...],
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-dihedral-parameter.prmtop"
    coords = tmp_path / "bad-dihedral-parameter.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1, 1, 1),
        dihedral_record=(0, 3, 6, 9, 1),
        dihedral_periodicity=dihedral_periodicity,
        dihedral_phase=dihedral_phase,
    )
    _write_malformed_amber_coords(coords, atom_count=4)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_dihedral_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_malformed_dihedral_record_width(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-dihedral-width.prmtop"
    coords = tmp_path / "bad-dihedral-width.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        atom_type_indices=(1, 1, 1, 1),
        dihedral_record=(0, 3, 6, 9),
    )
    _write_malformed_amber_coords(coords, atom_count=4)

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_dihedral_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize("atom_type_index", [0, 2], ids=["zero", "out_of_range"])
def test_import_amber_fails_closed_for_malformed_atom_type_indices(
    tmp_path: Path,
    atom_type_index: int,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-atom-type.prmtop"
    coords = tmp_path / "bad-atom-type.inpcrd"
    _write_malformed_amber_prmtop(prmtop, atom_type_indices=(atom_type_index, 1))
    _write_malformed_amber_coords(coords, atom_count=2)

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_atom_types"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


@pytest.mark.parametrize(
    ("ntypes", "atom_type_indices", "nonbonded_indices", "lj_acoef", "lj_bcoef"),
    [
        (2, (1, 2), (1, 2, 2), (4.0, 4.0), (4.0, 4.0)),
        (1, (1, 1), (1, 1), (4.0,), (4.0,)),
        (1, (1, 1), (2,), (4.0,), (4.0,)),
        (1, (1, 1), (1,), (float("nan"),), (4.0,)),
    ],
    ids=[
        "short_nonbonded_table",
        "extra_nonbonded_table_entry",
        "coefficient_reference_out_of_range",
        "nonfinite_coefficient",
    ],
)
def test_import_amber_fails_closed_for_malformed_lj_tables(
    tmp_path: Path,
    ntypes: int,
    atom_type_indices: tuple[int, ...],
    nonbonded_indices: tuple[int, ...],
    lj_acoef: tuple[float, ...],
    lj_bcoef: tuple[float, ...],
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-lj.prmtop"
    coords = tmp_path / "bad-lj.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        ntypes=ntypes,
        atom_type_indices=atom_type_indices,
        nonbonded_indices=nonbonded_indices,
        lj_acoef=lj_acoef,
        lj_bcoef=lj_bcoef,
    )
    _write_malformed_amber_coords(coords, atom_count=len(atom_type_indices))

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:amber_malformed_lj_parameters",
    ):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_explicit_empty_exclusions_do_not_fallback_to_bonds_or_angles(
    tmp_path: Path,
):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "explicit-empty-exclusions.prmtop"
    coords = tmp_path / "explicit-empty-exclusions.inpcrd"
    _write_malformed_amber_prmtop(
        prmtop,
        bond_record=(0, 3, 1),
        angle_record=(0, 3, 6, 1),
        exclusion_counts=(1, 0, 0),
        exclusion_values=(0,),
    )
    _write_malformed_amber_coords(coords, atom_count=3)

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)

    assert prepared.bonds.tolist() == [[0, 1]]
    assert prepared.angles.tolist() == [[0, 1, 2]]
    assert prepared.nonbonded_exception_pairs.shape == (0, 2)


@pytest.mark.parametrize("excluded_atom", [3, 1, -1], ids=["out_of_range", "self", "negative"])
def test_import_amber_fails_closed_for_malformed_exclusion_atom_ids(
    tmp_path: Path,
    excluded_atom: int,
):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "bad-exclusions.prmtop"
    coords = tmp_path / "bad-exclusions.inpcrd"
    prmtop.write_text(
        f"""%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       1       0
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
{excluded_atom:8d}
"""
    )
    coords.write_text(
        """bad exclusions
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_exclusions"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_fails_closed_for_negative_exclusion_count(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop

    prmtop = tmp_path / "negative-exclusion-count.prmtop"
    coords = tmp_path / "negative-exclusion-count.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
      -1       0
"""
    )
    coords.write_text(
        """bad exclusion count
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:amber_malformed_exclusions"):
        import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)


def test_import_amber_preserves_zero_exclusion_sentinel(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import import_amber_prmtop

    prmtop = tmp_path / "zero-exclusion.prmtop"
    coords = tmp_path / "zero-exclusion.inpcrd"
    prmtop.write_text(
        """%VERSION  VERSION_STAMP = V0001.000
%FLAG POINTERS
%FORMAT(10I8)
       2       1
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.00800000E+00
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  4.00000000E+00
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       1       0
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       0
"""
    )
    coords.write_text(
        """zero exclusion
    2
  0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000
"""
    )

    prepared = import_amber_prmtop(prmtop_path=prmtop, coords_path=coords)

    assert prepared.atom_count == 2
    assert prepared.nonbonded_exception_pairs.shape == (0, 2)


def test_charmm_psf_mass_prelude_is_derived_from_psf_atom_masses(tmp_path: Path):
    from mlx_atomistic.prep.topology_import import build_charmm_psf_mass_prelude

    psf = tmp_path / "tiny.psf"
    prm = tmp_path / "tiny.prm"
    psf.write_text(
        "PSF EXT\n\n"
        "       2 !NATOM\n"
        "       1 SYS      1        LIG      C1       CT3     -0.270000       12.0110           0\n"
        "       2 SYS      1        LIG      H1       HA3      0.090000        1.0080           0\n"
    )
    prm.write_text("MASS     1 HA3        1.00800\n")

    prelude = build_charmm_psf_mass_prelude(psf_path=psf, params=[prm])

    assert prelude is not None
    assert prelude.source_path == str(psf)
    assert prelude.missing_atom_types == ("CT3",)
    assert "MASS" in prelude.text
    assert "CT3" in prelude.text
    assert "12.01100" in prelude.text


def test_charmm_parmed_import_exports_cmap_urey_and_nbfix_type_overrides(tmp_path: Path):
    from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system
    from mlx_atomistic.prep.topology_import import (
        _prepared_from_parmed_structure,
    )

    class FakeResidue:
        name = "LIG"
        number = 1

    class FakeAtomType:
        nbfix = {"CLGR1": (3.0, 0.1, 3.0, 0.1)}

    class FakeAtom:
        def __init__(self, idx: int):
            self.idx = idx
            self.name = f"H{idx + 1}"
            self.type = "H"
            self.atom_type = FakeAtomType() if idx == 0 else object()
            self.residue = FakeResidue()
            self.mass = 1.008
            self.charge = 0.0
            self.sigma = 1.0
            self.epsilon = 0.1

    atoms = [FakeAtom(index) for index in range(8)]

    class FakeAngle:
        atom1 = atoms[0]
        atom2 = atoms[1]
        atom3 = atoms[2]
        type = type("AngleType", (), {"k": 2.0, "theteq": 109.5})()

    class FakeUreyBradley:
        atom1 = atoms[0]
        atom2 = atoms[2]
        type = type("UreyType", (), {"k": 5.0, "req": 1.8})()

    class FakeCmapType:
        resolution = 4
        grid = np.arange(16, dtype=np.float32).tolist()

    class FakeCmap:
        atom1, atom2, atom3, atom4, atom5 = atoms[:5]
        type = FakeCmapType()

    prepared = _prepared_from_parmed_structure(
        type(
            "FakeStructure",
            (),
            {
                "atoms": atoms,
                "coordinates": np.zeros((8, 3), dtype=np.float32),
                "bonds": [],
                "angles": [FakeAngle()],
                "dihedrals": [],
                "cmaps": [FakeCmap()],
                "urey_bradleys": [FakeUreyBradley()],
            },
        )(),
        source={"kind": "charmm"},
        parameter_source="charmm_psf_parameters",
    )

    report = prepared.metadata.compatibility_report
    assert prepared.urey_bradley_terms.tolist() == [[0, 1, 2]]
    assert prepared.charmm_cmap_terms.tolist() == [[0, 1, 2, 3, 1, 2, 3, 4]]
    assert prepared.charmm_cmap_grid_indices.tolist() == [0]
    assert prepared.charmm_cmap_grids.shape == (1, 4, 4)
    assert prepared.nbfix_type_pairs.tolist() == [["CLGR1", "H"]]
    np.testing.assert_allclose(prepared.nbfix_type_sigma, [3.0 * 2 ** (-1.0 / 6.0)])
    np.testing.assert_allclose(prepared.nbfix_type_epsilon, [0.1 * 4.184])
    assert "urey_bradley" in report["supported_terms"]
    assert "charmm_cmap_terms" in report["supported_terms"]
    assert "nbfix_pair_overrides" in report["supported_terms"]
    assert "nbfix_pair_overrides" in report["required_terms"]
    assert report["rejected_terms"] == []
    nbfix_details = report["term_details"]["nbfix_pair_overrides"]
    assert nbfix_details["override_count"] == 1
    assert nbfix_details["atom_type_pairs"][0]["type1"] == "CLGR1"
    assert nbfix_details["atom_type_pairs"][0]["type2"] == "H"

    save_prepared_system(prepared, tmp_path)
    reloaded = load_prepared_system(tmp_path)
    assert reloaded.nbfix_type_pairs.tolist() == [["CLGR1", "H"]]
    np.testing.assert_allclose(reloaded.nbfix_type_sigma, prepared.nbfix_type_sigma)
    np.testing.assert_allclose(reloaded.nbfix_type_epsilon, prepared.nbfix_type_epsilon)


def test_charmm_parmed_nbfix_distinct_14_values_fail_closed():
    from mlx_atomistic.prep.topology_import import (
        TopologyImportError,
        _prepared_from_parmed_structure,
    )

    class FakeResidue:
        name = "LIG"
        number = 1

    class FakeAtomType:
        nbfix = {"B": (3.0, 0.1, 3.1, 0.1)}

    class FakeAtom:
        idx = 0
        name = "H1"
        type = "A"
        atom_type = FakeAtomType()
        residue = FakeResidue()
        mass = 1.008
        charge = 0.0
        sigma = 1.0
        epsilon = 0.1

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:nbfix_pair_overrides:distinct_1_4_values",
    ):
        _prepared_from_parmed_structure(
            type(
                "FakeStructure",
                (),
                {
                    "atoms": [FakeAtom()],
                    "coordinates": np.zeros((1, 3), dtype=np.float32),
                    "bonds": [],
                    "angles": [],
                    "dihedrals": [],
                },
            )(),
            source={"kind": "charmm"},
            parameter_source="charmm_psf_parameters",
        )


def test_charmm_parmed_cmap_float32_overflow_fails_closed():
    from mlx_atomistic.prep.topology_import import (
        TopologyImportError,
        _prepared_from_parmed_structure,
    )

    class FakeResidue:
        name = "LIG"
        number = 1

    class FakeAtom:
        def __init__(self, idx: int):
            self.idx = idx
            self.name = f"H{idx + 1}"
            self.type = "H"
            self.atom_type = object()
            self.residue = FakeResidue()
            self.mass = 1.008
            self.charge = 0.0
            self.sigma = 1.0
            self.epsilon = 0.1

    atoms = [FakeAtom(index) for index in range(5)]

    class FakeCmapType:
        resolution = 4
        grid = [1.0e39] * 16

    class FakeCmap:
        atom1, atom2, atom3, atom4, atom5 = atoms
        type = FakeCmapType()

    with pytest.raises(TopologyImportError, match="unsupported_terms:charmm_cmap_terms"):
        _prepared_from_parmed_structure(
            type(
                "FakeStructure",
                (),
                {
                    "atoms": atoms,
                    "coordinates": np.zeros((5, 3), dtype=np.float32),
                    "bonds": [],
                    "angles": [],
                    "dihedrals": [],
                    "cmaps": [FakeCmap()],
                },
            )(),
            source={"kind": "charmm"},
            parameter_source="charmm_psf_parameters",
        )


def test_charmm_parmed_nbfix_float32_overflow_fails_closed():
    from mlx_atomistic.prep.topology_import import (
        TopologyImportError,
        _prepared_from_parmed_structure,
    )

    class FakeResidue:
        name = "LIG"
        number = 1

    class FakeAtomType:
        nbfix = {"B": (3.0, 1.0e38, 3.0, 1.0e38)}

    class FakeAtom:
        idx = 0
        name = "H1"
        type = "A"
        atom_type = FakeAtomType()
        residue = FakeResidue()
        mass = 1.008
        charge = 0.0
        sigma = 1.0
        epsilon = 0.1

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:nbfix_pair_overrides:malformed_entries",
    ):
        _prepared_from_parmed_structure(
            type(
                "FakeStructure",
                (),
                {
                    "atoms": [FakeAtom()],
                    "coordinates": np.zeros((1, 3), dtype=np.float32),
                    "bonds": [],
                    "angles": [],
                    "dihedrals": [],
                },
            )(),
            source={"kind": "charmm"},
            parameter_source="charmm_psf_parameters",
        )


def test_import_charmm_psf_native_fixture_maps_supported_terms(tmp_path: Path):
    from mlx_atomistic.prep import import_charmm_psf
    from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system

    fixture_root = Path("tests/fixtures/charmm")
    prepared = import_charmm_psf(
        psf_path=fixture_root / "native-mini.psf",
        params=[fixture_root / "native-mini.prm"],
        coords_path=fixture_root / "native-mini.pdb",
    )
    report = prepared.metadata.compatibility_report

    assert prepared.metadata.source["parser"] == "native_charmm_psf"
    assert prepared.metadata.parameter_source == "charmm_psf_parameters_native"
    assert prepared.atom_count == 8
    assert prepared.bonds.shape == (7, 2)
    assert prepared.angles.shape == (3, 3)
    assert prepared.dihedrals.tolist() == [[1, 0, 2, 4]]
    assert prepared.urey_bradley_terms.tolist() == [[0, 2, 4]]
    assert prepared.charmm_cmap_terms.tolist() == [[0, 1, 2, 4, 1, 2, 4, 5]]
    assert prepared.charmm_cmap_grid_indices.tolist() == [0]
    assert prepared.charmm_cmap_grids.shape == (1, 4, 4)
    assert prepared.nbfix_type_pairs.tolist() == [["NH1", "O"]]
    np.testing.assert_allclose(prepared.cell_lengths, [24.0, 25.0, 26.0])
    np.testing.assert_allclose(prepared.charges.sum(), 0.0, atol=1e-6)
    assert "harmonic_bond" in report["supported_terms"]
    assert "harmonic_angle" in report["supported_terms"]
    assert "periodic_dihedral" in report["supported_terms"]
    assert "urey_bradley" in report["supported_terms"]
    assert "charmm_cmap_terms" in report["supported_terms"]
    assert "nbfix_pair_overrides" in report["supported_terms"]
    assert report["term_counts"]["charmm_cmap_terms"] == 1
    assert report["term_counts"]["urey_bradley_terms"] == 1
    assert report["term_counts"]["nbfix_pair_overrides"] == 1
    assert report["unsupported_terms"] == []

    save_prepared_system(prepared, tmp_path)
    reloaded = load_prepared_system(tmp_path)
    assert reloaded.metadata.source["parser"] == "native_charmm_psf"
    assert reloaded.charmm_cmap_terms.tolist() == prepared.charmm_cmap_terms.tolist()
    assert reloaded.nbfix_type_pairs.tolist() == [["NH1", "O"]]


def test_import_charmm_psf_blocks_unsupported_virtual_site_records(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    psf = tmp_path / "virtual-site.psf"
    psf.write_text(
        (fixture_root / "native-mini.psf").read_text()
        + "\n       1 !NUMLP: lone pairs\n       1       2\n"
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:charmm_psf_numlp"):
        import_charmm_psf(
            psf_path=psf,
            params=[fixture_root / "native-mini.prm"],
            coords_path=fixture_root / "native-mini.pdb",
        )


def test_import_charmm_psf_blocks_unsupported_water_model(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    psf = tmp_path / "tip4.psf"
    psf.write_text(
        (fixture_root / "native-mini.psf")
        .read_text()
        .replace("ALA      N        NH1", "TIP4     N        NH1", 1)
    )

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:charmm_unsupported_water_model",
    ):
        import_charmm_psf(
            psf_path=psf,
            params=[fixture_root / "native-mini.prm"],
            coords_path=fixture_root / "native-mini.pdb",
        )


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("-0.300000", "NaN", "unsupported_terms:charmm_malformed_psf_atom_parameters"),
        ("14.0070", "Inf", "unsupported_terms:charmm_malformed_psf_atom_parameters"),
        ("-0.300000", "1e39", "unsupported_terms:charmm_malformed_psf_atom_parameters"),
        ("14.0070", "1e39", "unsupported_terms:charmm_malformed_psf_atom_parameters"),
        ("14.0070", "0.0000", "unsupported_terms:charmm_virtual_sites"),
    ],
)
def test_import_charmm_psf_blocks_invalid_atom_numeric_values(
    tmp_path: Path,
    old: str,
    new: str,
    match: str,
):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    psf = tmp_path / "bad-atom-numeric.psf"
    psf.write_text((fixture_root / "native-mini.psf").read_text().replace(old, new, 1))

    with pytest.raises(TopologyImportError, match=match):
        import_charmm_psf(
            psf_path=psf,
            params=[fixture_root / "native-mini.prm"],
            coords_path=fixture_root / "native-mini.pdb",
        )


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            "NH1  H      440.0  1.010",
            "NH1  H      -1.0   1.010",
            "unsupported_terms:charmm_invalid_bond_parameters",
        ),
        (
            "NH1  H      440.0  1.010",
            "NH1  H      440.0  0.000",
            "unsupported_terms:charmm_invalid_bond_parameters",
        ),
        (
            "NH1  H      440.0  1.010",
            "NH1  H      1e39   1.010",
            "unsupported_terms:charmm_malformed_bond_parameters",
        ),
        (
            "NH1  H      440.0  1.010",
            "NH1  H      440.0  1e39",
            "unsupported_terms:charmm_malformed_bond_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       -1.0  110.0  5.0  2.300",
            "unsupported_terms:charmm_invalid_angle_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       1e39  110.0  5.0  2.300",
            "unsupported_terms:charmm_malformed_angle_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       50.0  110.0  -1.0  2.300",
            "unsupported_terms:charmm_invalid_urey_bradley_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       50.0  110.0  1e39  2.300",
            "unsupported_terms:charmm_malformed_angle_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       50.0  110.0  5.0  0.000",
            "unsupported_terms:charmm_invalid_urey_bradley_parameters",
        ),
        (
            "NH1  CT1  C       50.0  110.0  5.0  2.300",
            "NH1  CT1  C       50.0  110.0  5.0  1e39",
            "unsupported_terms:charmm_malformed_angle_parameters",
        ),
    ],
)
def test_import_charmm_psf_blocks_invalid_bonded_parameters(
    tmp_path: Path,
    old: str,
    new: str,
    match: str,
):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    prm = tmp_path / "bad-bonded.prm"
    prm.write_text((fixture_root / "native-mini.prm").read_text().replace(old, new, 1))

    with pytest.raises(TopologyImportError, match=match):
        import_charmm_psf(
            psf_path=fixture_root / "native-mini.psf",
            params=[prm],
            coords_path=fixture_root / "native-mini.pdb",
        )


def test_import_charmm_psf_blocks_nonempty_hbond_parameters(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    prm = tmp_path / "hbond.prm"
    prm.write_text(
        (fixture_root / "native-mini.prm")
        .read_text()
        .replace("END\n", "HBOND\nNH1 O 1.0 2.0\nEND\n")
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:charmm_parameter_hbond"):
        import_charmm_psf(
            psf_path=fixture_root / "native-mini.psf",
            params=[prm],
            coords_path=fixture_root / "native-mini.pdb",
        )


def test_import_charmm_psf_blocks_cmap_float32_overflow(tmp_path: Path):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    prm = tmp_path / "bad-cmap.prm"
    prm.write_text(
        (fixture_root / "native-mini.prm")
        .read_text()
        .replace("1.2 1.3 1.4 1.5", "1.2 1.3 1.4 1e39")
    )

    with pytest.raises(TopologyImportError, match="unsupported_terms:charmm_cmap_terms"):
        import_charmm_psf(
            psf_path=fixture_root / "native-mini.psf",
            params=[prm],
            coords_path=fixture_root / "native-mini.pdb",
        )


@pytest.mark.parametrize(
    "nbfix_line",
    [
        "NH1 O -0.1000",
        "NH1 O not-a-number 3.0000",
        "NH1 O 1e38 3.0000",
    ],
)
def test_import_charmm_psf_blocks_malformed_nbfix_parameters(
    tmp_path: Path,
    nbfix_line: str,
):
    from mlx_atomistic.prep import TopologyImportError, import_charmm_psf

    fixture_root = Path("tests/fixtures/charmm")
    prm = tmp_path / "bad-nbfix.prm"
    prm.write_text(
        (fixture_root / "native-mini.prm")
        .read_text()
        .replace("NH1 O -0.1000 3.0000", nbfix_line)
    )

    with pytest.raises(
        TopologyImportError,
        match="unsupported_terms:nbfix_pair_overrides:malformed_entries",
    ):
        import_charmm_psf(
            psf_path=fixture_root / "native-mini.psf",
            params=[prm],
            coords_path=fixture_root / "native-mini.pdb",
        )


def test_gpcrmd_729_ligand_residue_is_masked_as_ligand():
    from mlx_atomistic.prep.topology_import import _ligand_mask_from_residues

    mask = _ligand_mask_from_residues(np.asarray(["SER", "P32", "TIP3"], dtype=str))

    assert mask.tolist() == [False, True, False]
