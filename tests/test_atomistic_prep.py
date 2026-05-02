from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_core_import_does_not_require_prep_dependencies():
    import mlx_atomistic

    assert mlx_atomistic.__version__


def test_core_source_does_not_import_external_md_engines():
    source_root = Path("src/mlx_atomistic")
    source = "\n".join(path.read_text() for path in source_root.rglob("*.py"))

    assert "import openmm" not in source.lower()
    assert "pdbfixer" not in source.lower()


def test_atomistic_prep_import_and_dependency_report():
    import atomistic_prep
    from atomistic_prep.prepare import MissingPrepDependencyError

    status = atomistic_prep.optional_prep_dependency_status()
    assert {"gemmi", "parmed", "rdkit"} <= set(status)
    assert "openmm" not in status
    assert "pdbfixer" not in status
    if not all(status.values()):
        with pytest.raises(
            MissingPrepDependencyError,
            match="Production biomolecular preparation needs optional parsing",
        ):
            atomistic_prep.require_production_prep_dependencies()


def test_prepared_artifact_round_trip(tmp_path: Path):
    from atomistic_prep.io import (
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


def test_build_mlx_system_matches_artifact_counts():
    from atomistic_prep.io import synthetic_prepared_system
    from atomistic_prep.runner import build_mlx_system

    prepared = synthetic_prepared_system()
    system, terms = build_mlx_system(prepared, receptor_mass_scale=1.0)

    assert system.atom_count == prepared.atom_count
    assert system.topology.n_atoms == prepared.atom_count
    assert len(terms) >= 1


def test_tiny_prepared_system_runs_mlx_nvt(tmp_path: Path):
    from atomistic_prep.io import save_prepared_system, synthetic_prepared_system
    from atomistic_prep.runner import run_mlx
    from mlx_atomistic.io import load_npz_trajectory

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
    assert record.metadata["kind"] == "atomistic_prep_mlx_nvt"


def test_run_mlx_rejects_npt_barostat_protocol_metadata_before_system_build(
    tmp_path: Path,
    monkeypatch,
):
    from atomistic_prep import runner
    from atomistic_prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.protocols import ProtocolCompatibilityError

    prepared = synthetic_prepared_system()
    metadata = replace(
        prepared.metadata,
        protocol_metadata={
            "ensemble": "NPT",
            "barostat": "monte_carlo",
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

    assert exc_info.value.blockers == ("npt_barostat", "barostat")
    assert not (tmp_path / "trajectory.npz").exists()


def test_run_mlx_persists_normalized_nvt_protocol_metadata(tmp_path: Path):
    from atomistic_prep.io import save_prepared_system, synthetic_prepared_system
    from atomistic_prep.runner import run_mlx
    from mlx_atomistic.io import load_npz_trajectory

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


def test_solvated_ligand_receptor_replicas_write_selected_and_all_outputs(tmp_path: Path):
    from atomistic_prep.replicas import run_ligand_receptor_replicas
    from mlx_atomistic.io import load_npz_trajectory

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


def test_ligand_receptor_performance_profile_emits_aggregate_rows(tmp_path: Path):
    from atomistic_prep.replicas import profile_ligand_receptor_performance

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


def test_ligand_receptor_replicas_rerun_when_constraint_iterations_change(tmp_path: Path):
    from atomistic_prep.replicas import run_ligand_receptor_replicas
    from mlx_atomistic.io import load_npz_trajectory

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


def test_notebook_bundle_loader_uses_full_prepared_artifact(tmp_path: Path):
    pytest.importorskip("MDAnalysis")
    from atomistic_prep.io import save_prepared_system, synthetic_prepared_system
    from atomistic_prep.notebook import (
        load_prepared_trajectory_bundle,
        make_mdanalysis_universe,
    )
    from atomistic_prep.runner import run_mlx

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
    from atomistic_prep.io import synthetic_prepared_system
    from atomistic_prep.notebook import PreparedTrajectoryRecord, trajectory_to_multimodel_pdb

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
    from atomistic_prep.notebook import py3dmol_frame_player_html

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
    from atomistic_prep.prepare import prepare_p2x4_atp

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
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.prepare import prepare_p2x4_atp
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact

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
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.prepare import prepare_p2x4_atp
    from atomistic_prep.runner import run_mlx

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
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.t4l_benzene import (
        T4L_BENZENE_PARAMETER_SOURCE,
        prepare_t4l_benzene,
    )
    from mlx_atomistic.artifacts import MLXCompatibilityError, load_prepared_mlx_artifact

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


def test_t4l_benzene_steered_run_writes_cv_trace(tmp_path: Path):
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.runner import run_steered_mlx
    from atomistic_prep.t4l_benzene import prepare_t4l_benzene

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
    assert "run-gpcrmd-mlx" in readme
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


def test_solvated_ligand_receptor_builder_exports_complete_runtime_artifact(tmp_path: Path):
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.solvated_example import (
        ELECTROSTATICS_MODEL,
        SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE,
        prepare_solvated_ligand_receptor_example,
        validate_complete_solvated_ligand_receptor_system,
    )
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact

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


def test_run_ligand_receptor_example_writes_mlx_trajectory(tmp_path: Path):
    from atomistic_prep.solvated_example import ensure_solvated_ligand_receptor_example
    from mlx_atomistic.io import load_npz_trajectory

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


def test_import_amber_tiny_topology_runs_mlx(tmp_path: Path):
    from atomistic_prep.io import save_prepared_system
    from atomistic_prep.runner import run_mlx
    from atomistic_prep.topology_import import import_amber_prmtop
    from mlx_atomistic.artifacts import load_prepared_mlx_artifact

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


def test_charmm_psf_mass_prelude_is_derived_from_psf_atom_masses(tmp_path: Path):
    from atomistic_prep.topology_import import build_charmm_psf_mass_prelude

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
    from atomistic_prep.io import load_prepared_system, save_prepared_system
    from atomistic_prep.topology_import import (
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
    from atomistic_prep.topology_import import (
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


def test_gpcrmd_729_ligand_residue_is_masked_as_ligand():
    from atomistic_prep.topology_import import _ligand_mask_from_residues

    mask = _ligand_mask_from_residues(np.asarray(["SER", "P32", "TIP3"], dtype=str))

    assert mask.tolist() == [False, True, False]
