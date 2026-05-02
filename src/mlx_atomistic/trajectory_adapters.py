"""Optional trajectory adapters for analysis libraries.

These helpers intentionally import MDAnalysis and MDTraj lazily so the core
MLX engine remains dependency-light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlx_atomistic.io import TrajectoryRecord


class OptionalTrajectoryDependencyError(RuntimeError):
    """Raised when an optional trajectory analysis dependency is unavailable."""


def _import_mdanalysis():
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        msg = "MDAnalysis support requires the optional viz extra: uv sync --extra viz"
        raise OptionalTrajectoryDependencyError(msg) from exc
    return mda


def _import_mdtraj():
    try:
        import mdtraj as md
    except ImportError as exc:
        msg = "MDTraj support requires the optional viz extra: uv sync --extra viz"
        raise OptionalTrajectoryDependencyError(msg) from exc
    return md


def load_mdanalysis_universe(
    topology_path: str | Path,
    trajectory_path: str | Path | None = None,
    **kwargs,
):
    """Load a topology/trajectory pair as an MDAnalysis Universe."""

    mda = _import_mdanalysis()
    if trajectory_path is None:
        return mda.Universe(str(topology_path), **kwargs)
    return mda.Universe(str(topology_path), str(trajectory_path), **kwargs)


def mdanalysis_universe_from_arrays(
    topology_path: str | Path,
    positions_angstrom: np.ndarray,
    *,
    time_ps: np.ndarray | None = None,
    dt_ps: float | None = None,
):
    """Attach in-memory Angstrom coordinates to an MDAnalysis topology."""

    mda = _import_mdanalysis()
    coordinates = np.asarray(positions_angstrom, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        msg = "positions_angstrom must have shape (n_frames, n_atoms, 3)"
        raise ValueError(msg)
    if dt_ps is None:
        if time_ps is not None and len(time_ps) > 1:
            dt_ps = float(np.asarray(time_ps, dtype=np.float32)[1] - time_ps[0])
        else:
            dt_ps = 1.0
    universe = mda.Universe(str(topology_path))
    universe.load_new(coordinates.copy(), order="fac", dt=float(dt_ps))
    if len(universe.atoms) != coordinates.shape[1]:
        msg = (
            "topology atom count does not match trajectory atom count: "
            f"{len(universe.atoms)} != {coordinates.shape[1]}"
        )
        raise ValueError(msg)
    return universe


def trajectory_record_to_mdanalysis(
    topology_path: str | Path,
    record: TrajectoryRecord,
):
    """Convert a native MLX trajectory record into an MDAnalysis Universe."""

    return mdanalysis_universe_from_arrays(
        topology_path,
        record.sampled_positions,
        time_ps=record.sampled_time,
    )


def trajectory_record_to_mdtraj(
    topology_path: str | Path,
    record: TrajectoryRecord,
):
    """Convert a native MLX trajectory record into an MDTraj Trajectory.

    `mlx_atomistic` stores coordinates in Angstrom. MDTraj stores coordinates in
    nanometers, so coordinates are converted by dividing by 10.
    """

    md = _import_mdtraj()
    topology = md.load(str(topology_path)).topology
    xyz_nm = np.asarray(record.sampled_positions, dtype=np.float32) / 10.0
    trajectory = md.Trajectory(
        xyz=xyz_nm,
        topology=topology,
        time=np.asarray(record.sampled_time, dtype=np.float32),
    )
    if trajectory.n_atoms != xyz_nm.shape[1]:
        msg = (
            "topology atom count does not match trajectory atom count: "
            f"{trajectory.n_atoms} != {xyz_nm.shape[1]}"
        )
        raise ValueError(msg)
    return trajectory


__all__ = [
    "OptionalTrajectoryDependencyError",
    "load_mdanalysis_universe",
    "mdanalysis_universe_from_arrays",
    "trajectory_record_to_mdanalysis",
    "trajectory_record_to_mdtraj",
]
