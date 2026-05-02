"""Trajectory analysis helpers for the macromolecule visualization notebook."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from MDAnalysis.lib.distances import distance_array

from helpers.config import LIGAND_SELECTION, RECEPTOR_SELECTION


def atom_chain_id(atom):
    """Return a chain identifier from an MDAnalysis atom."""

    return getattr(atom, "chainID", "") or getattr(atom, "segid", "")


def diagnostics_dataframe(trajectory_record) -> pd.DataFrame:
    """Return sampled energy/temperature diagnostics for the trajectory."""

    diagnostic_steps = np.asarray(
        trajectory_record.diagnostic_steps
        if trajectory_record.diagnostic_steps is not None
        else np.arange(len(trajectory_record.total_energy), dtype=np.int32)
    )
    diagnostic_time = np.asarray(
        trajectory_record.diagnostic_time
        if trajectory_record.diagnostic_time is not None
        else diagnostic_steps * float(trajectory_record.metadata.get("dt", 1.0))
    )
    diagnostics_df = pd.DataFrame(
        {
            "step": diagnostic_steps,
            "time_ps": diagnostic_time,
            "potential_energy_kJ_mol": np.asarray(trajectory_record.potential_energy),
            "kinetic_energy_kJ_mol": np.asarray(trajectory_record.kinetic_energy),
            "total_energy_kJ_mol": np.asarray(trajectory_record.total_energy),
            "temperature_K": np.asarray(trajectory_record.temperature),
        }
    )
    diagnostics_df["total_energy_drift_kJ_mol"] = (
        diagnostics_df["total_energy_kJ_mol"] - diagnostics_df["total_energy_kJ_mol"].iloc[0]
    )
    return diagnostics_df


def atp_receptor_interaction_trace(
    universe,
    ligand_selection=LIGAND_SELECTION,
    receptor_selection=RECEPTOR_SELECTION,
    cutoff_angstrom=4.0,
    stride=1,
):
    """Return ATP-pocket minimum distance and contact count across frames."""

    ligand = universe.select_atoms(ligand_selection)
    receptor = universe.select_atoms(receptor_selection)
    if len(ligand) == 0:
        msg = f"No ligand atoms matched: {ligand_selection}"
        raise ValueError(msg)
    if len(receptor) == 0:
        msg = f"No receptor atoms matched: {receptor_selection}"
        raise ValueError(msg)

    rows = []
    for ts in universe.trajectory[::stride]:
        dimensions = universe.dimensions
        box = dimensions if dimensions is not None and np.all(dimensions[:3] > 0) else None
        distances = distance_array(ligand.positions, receptor.positions, box=box)
        rows.append(
            {
                "frame": int(ts.frame),
                "time_ps": float(ts.time),
                "min_atp_receptor_distance_A": float(distances.min()),
                "contacts_within_4A": int(np.count_nonzero(distances <= cutoff_angstrom)),
            }
        )
    return pd.DataFrame(rows)


def atp_rmsd(universe, ligand_selection=LIGAND_SELECTION):
    """Return ATP RMSD to frame zero."""

    ligand = universe.select_atoms(ligand_selection)
    if len(ligand) == 0:
        return pd.DataFrame()
    reference = None
    rows = []
    for ts in universe.trajectory:
        coords = ligand.positions.copy()
        if reference is None:
            reference = coords.copy()
        diff = coords - reference
        rows.append(
            {
                "frame": int(ts.frame),
                "time_ps": float(ts.time),
                "atp_rmsd_A": float(np.sqrt(np.mean(diff * diff))),
            }
        )
    return pd.DataFrame(rows)


def pocket_rmsf(universe, receptor_selection=RECEPTOR_SELECTION):
    """Return per-atom RMSF for the selected receptor pocket."""

    receptor = universe.select_atoms(receptor_selection)
    if len(receptor) == 0 or len(universe.trajectory) < 2:
        return pd.DataFrame()
    frames = []
    for _ in universe.trajectory:
        frames.append(receptor.positions.copy())
    coords = np.stack(frames)
    mean = coords.mean(axis=0)
    rmsf = np.sqrt(np.mean(np.sum((coords - mean) ** 2, axis=2), axis=0))
    return pd.DataFrame(
        {
            "resid": [int(atom.resid) for atom in receptor],
            "resname": [atom.resname for atom in receptor],
            "atom": [atom.name for atom in receptor],
            "rmsf_A": rmsf,
        }
    )


def diagnostics_figure(diagnostics_df: pd.DataFrame):
    """Return energy/temperature diagnostic figure."""

    return px.line(
        diagnostics_df,
        x="time_ps",
        y=["temperature_K", "total_energy_drift_kJ_mol"],
        title="MLX trajectory diagnostics",
    )


def interaction_figure(interaction_df: pd.DataFrame):
    """Return ATP-receptor contact trace figure."""

    return px.line(
        interaction_df,
        x="time_ps",
        y=["min_atp_receptor_distance_A", "contacts_within_4A"],
        title="ATP-receptor contact trace across MLX frames",
    )


def rmsd_figure(rmsd_df: pd.DataFrame):
    """Return ATP RMSD figure."""

    return px.line(rmsd_df, x="time_ps", y="atp_rmsd_A", title="ATP RMSD")


def run_hydrogen_bond_analysis(universe) -> tuple[pd.DataFrame | None, str | None]:
    """Run MDAnalysis hydrogen-bond analysis, returning a dataframe or error."""

    try:
        from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

        hbonds = HydrogenBondAnalysis(
            universe=universe,
            donors_sel="protein or resname ATP or resname ADP or resname ANP",
            hydrogens_sel="name H* or type H*",
            acceptors_sel="protein or resname ATP or resname ADP or resname ANP",
        )
        hbonds.run(step=max(1, len(universe.trajectory) // 100))
        hbond_df = pd.DataFrame(
            hbonds.results.hbonds,
            columns=[
                "frame",
                "donor_index",
                "hydrogen_index",
                "acceptor_index",
                "distance_A",
                "angle_deg",
            ],
        )
        return hbond_df, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_prolif_fingerprint(universe, ligand, protein) -> tuple[pd.DataFrame | None, str | None]:
    """Run ProLIF fingerprints, returning a dataframe or error."""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*MDAnalysis.topology.tables has been moved.*",
                category=DeprecationWarning,
            )
            import prolif as plf

        fp = plf.Fingerprint()
        fp.run(universe.trajectory[:: max(1, len(universe.trajectory) // 100)], ligand, protein)
        return fp.to_dataframe(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
