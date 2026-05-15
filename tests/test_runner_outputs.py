from __future__ import annotations

import builtins
from pathlib import Path

import pytest


def test_run_mlx_writes_dcd_and_xtc_outputs(tmp_path: Path):
    md = pytest.importorskip("mdtraj")
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    npz_path = tmp_path / "trajectory.npz"
    dcd_path = tmp_path / "trajectory.dcd"
    xtc_path = tmp_path / "trajectory.xtc"
    run_mlx(
        tmp_path,
        out=npz_path,
        dcd_out=dcd_path,
        xtc_out=xtc_path,
        steps=2,
        sample_interval=1,
        dt=0.0005,
        temperature=0.0,
        receptor_mass_scale=1.0,
        minimize_steps=0,
        equilibration_steps=0,
    )

    assert npz_path.exists()
    assert dcd_path.exists()
    assert xtc_path.exists()
    topology = tmp_path / "view.pdb"
    dcd = md.load(str(dcd_path), top=str(topology))
    xtc = md.load(str(xtc_path), top=str(topology))
    assert dcd.n_atoms == prepared.atom_count
    assert dcd.n_frames == 3
    assert xtc.n_atoms == prepared.atom_count
    assert xtc.n_frames == 3


def test_run_mlx_optional_writer_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from mlx_atomistic.prep.io import save_prepared_system, synthetic_prepared_system
    from mlx_atomistic.prep.runner import run_mlx
    from mlx_atomistic.trajectory_adapters import OptionalTrajectoryDependencyError

    prepared = synthetic_prepared_system()
    save_prepared_system(prepared, tmp_path)
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "mdtraj":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(OptionalTrajectoryDependencyError, match="uv sync --extra viz"):
        run_mlx(
            tmp_path,
            out=tmp_path / "trajectory.npz",
            dcd_out=tmp_path / "trajectory.dcd",
            steps=1,
            sample_interval=1,
            temperature=0.0,
            minimize_steps=0,
            equilibration_steps=0,
        )
