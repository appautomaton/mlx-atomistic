"""Dense SCF restart persistence for small DFT systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.dft.scf import SCFResult


@dataclass(frozen=True)
class DenseSCFRestart:
    """Loaded dense SCF restart data."""

    density: np.ndarray
    orbitals: np.ndarray
    positions: np.ndarray
    cell_lengths: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe restart summary without dense fields."""

        return {
            "density_shape": list(self.density.shape),
            "orbitals_shape": list(self.orbitals.shape),
            "positions": self.positions.tolist(),
            "cell_lengths": self.cell_lengths.tolist(),
            "metadata": dict(self.metadata),
        }


def save_dense_scf_restart(
    path: str | Path,
    result: SCFResult,
    *,
    positions,
    cell_lengths,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save density, orbitals, and ion/cell state to compressed NPZ."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mlx-atomistic.dft.dense-scf-restart.v1",
        "electron_count": result.electron_count,
        "energy": result.total_energy,
        "spin_mode": "unpolarized",
        "kpoints": [{"vector": [0.0, 0.0, 0.0], "weight": 1.0, "label": "Γ"}],
        "user": {} if metadata is None else dict(metadata),
    }
    np.savez_compressed(
        path,
        density=np.array(result.density),
        orbitals=np.array(result.orbitals),
        positions=np.asarray(positions, dtype=np.float64),
        cell_lengths=np.asarray(cell_lengths, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(payload)),
    )


def load_dense_scf_restart(path: str | Path) -> DenseSCFRestart:
    """Load a dense SCF restart file."""

    with np.load(path, allow_pickle=False) as data:
        return DenseSCFRestart(
            density=np.array(data["density"]),
            orbitals=np.array(data["orbitals"]),
            positions=np.array(data["positions"], dtype=np.float64),
            cell_lengths=np.array(data["cell_lengths"], dtype=np.float64),
            metadata=json.loads(str(data["metadata_json"])),
        )
