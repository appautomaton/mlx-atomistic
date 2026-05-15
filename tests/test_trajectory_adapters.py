from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from mlx_atomistic.io import TrajectoryRecord


def _tiny_record() -> TrajectoryRecord:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    zeros = np.zeros((2, 2, 3), dtype=np.float32)
    axis = np.asarray([0, 1], dtype=np.int32)
    time = np.asarray([0.0, 0.5], dtype=np.float32)
    values = np.zeros(2, dtype=np.float32)
    return TrajectoryRecord(
        sampled_positions=positions,
        sampled_velocities=zeros,
        sampled_steps=axis,
        sampled_time=time,
        diagnostic_steps=axis,
        diagnostic_time=time,
        potential_energy=values,
        kinetic_energy=values,
        total_energy=values,
        potential_energy_by_term={},
        temperature=values,
        pair_count=values.astype(np.int32),
        rebuild_count=values.astype(np.int32),
        constraint_max_error=values,
        symbols=("H", "O"),
        cell=None,
        metadata={},
    )


def _tiny_pdb(path: Path) -> None:
    path.write_text(
        """ATOM      1  H1  HOH A   1       0.000   0.000   0.000  1.00  0.00           H
ATOM      2  O   HOH A   1       1.000   0.000   0.000  1.00  0.00           O
END
"""
    )


def test_mdanalysis_missing_dependency_error(monkeypatch: pytest.MonkeyPatch):
    from mlx_atomistic.trajectory_adapters import (
        OptionalTrajectoryDependencyError,
        load_mdanalysis_universe,
    )

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "MDAnalysis":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(OptionalTrajectoryDependencyError, match="uv sync --extra viz"):
        load_mdanalysis_universe("missing.pdb")


def test_mdtraj_writer_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from mlx_atomistic.trajectory_adapters import (
        OptionalTrajectoryDependencyError,
        write_mdtraj_trajectory,
    )

    topology = tmp_path / "tiny.pdb"
    _tiny_pdb(topology)
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "mdtraj":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(OptionalTrajectoryDependencyError, match="uv sync --extra viz"):
        write_mdtraj_trajectory(topology, _tiny_record(), tmp_path / "tiny.dcd")


def test_trajectory_record_to_mdanalysis_preserves_shape(tmp_path: Path):
    pytest.importorskip("MDAnalysis")
    from mlx_atomistic.trajectory_adapters import trajectory_record_to_mdanalysis

    topology = tmp_path / "tiny.pdb"
    _tiny_pdb(topology)
    universe = trajectory_record_to_mdanalysis(topology, _tiny_record())

    assert len(universe.atoms) == 2
    assert len(universe.trajectory) == 2
    np.testing.assert_allclose(universe.trajectory[1].positions[1], [1.2, 0.0, 0.0])


def test_trajectory_record_to_mdtraj_converts_angstrom_to_nm(tmp_path: Path):
    pytest.importorskip("mdtraj")
    from mlx_atomistic.trajectory_adapters import trajectory_record_to_mdtraj

    topology = tmp_path / "tiny.pdb"
    _tiny_pdb(topology)
    trajectory = trajectory_record_to_mdtraj(topology, _tiny_record())

    assert trajectory.n_atoms == 2
    assert trajectory.n_frames == 2
    np.testing.assert_allclose(trajectory.xyz[1, 1], [0.12, 0.0, 0.0])


def test_write_mdtraj_trajectory_writes_dcd_and_xtc(tmp_path: Path):
    md = pytest.importorskip("mdtraj")
    from mlx_atomistic.trajectory_adapters import write_mdtraj_trajectory

    topology = tmp_path / "tiny.pdb"
    _tiny_pdb(topology)
    dcd_path = write_mdtraj_trajectory(topology, _tiny_record(), tmp_path / "tiny.dcd")
    xtc_path = write_mdtraj_trajectory(topology, _tiny_record(), tmp_path / "tiny.xtc")

    dcd = md.load(str(dcd_path), top=str(topology))
    xtc = md.load(str(xtc_path), top=str(topology))

    assert dcd.n_atoms == 2
    assert dcd.n_frames == 2
    assert xtc.n_atoms == 2
    assert xtc.n_frames == 2
