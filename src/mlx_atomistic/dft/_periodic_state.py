"""Internal continuation contracts for periodic DFT execution."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_atomistic.dft.mixing import _MixerCheckpointState


@dataclass(frozen=True)
class _PeriodicSCFContinuationState:
    """Serializable state captured at an accepted periodic SCF boundary."""

    completed_iteration: int
    density: mx.array
    owned_coefficients: tuple[tuple[int, mx.array], ...]
    owned_lanes: tuple[dict[str, object], ...]
    previous_energy: float
    energy_by_term: dict[str, float]
    history: tuple[dict[str, float | int | str | None], ...]
    mixer_state: _MixerCheckpointState
    ownership: dict[str, object]
    lineage: tuple[str, ...] = ()
    spin_densities: tuple[mx.array, mx.array] | None = None
    down_owned_coefficients: tuple[tuple[int, mx.array], ...] = ()
    down_owned_lanes: tuple[dict[str, object], ...] = ()
    magnetization_mixer_state: _MixerCheckpointState | None = None

    @property
    def coefficient_map(self) -> dict[int, mx.array]:
        """Return the owned coefficient snapshots keyed by explicit k-point index."""

        return dict(self.owned_coefficients)

    @property
    def down_coefficient_map(self) -> dict[int, mx.array]:
        """Return spin-down coefficient snapshots keyed by k-point index."""

        return dict(self.down_owned_coefficients)
