"""Runtime-owned DFT state accounting and serialization adapters."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes
from mlx_atomistic.dft._compact import _CompactLaneState


def _npy_bytes(values: object) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def _state_for_basis(result: Any, basis: Any) -> _CompactLaneState:
    state = result._compact_coefficients
    if isinstance(state, _CompactLaneState):
        basis._validate_state(state)
        return state
    packed, _ = basis._state_from_full(state.full_grid_fresh())
    return packed


def fixed_density_state_metrics(*, result: Any, basis: Any) -> dict[str, int]:
    """Return logical runtime-owned bytes for one fixed-density eigenstate.

    Args:
        result: Periodic eigensolver result.
        basis: Basis owning the result coefficients.

    Returns:
        Logical coefficient payload and full-grid byte counts.
    """

    coefficients = _state_for_basis(result, basis).values
    return {
        "coefficient_payload_bytes": int(np.prod(coefficients.shape)) * 8,
        "full_grid_coefficient_bytes": int(result.eigenvalues.shape[0]) * basis.grid.size * 8,
    }


def serialize_fixed_density_state(state: Mapping[str, Any]) -> dict[str, bytes]:
    """Serialize one runtime-owned fixed-density state without report coupling.

    Args:
        state: Mapping containing result, basis, density, and effective potential.

    Returns:
        Relative payload names mapped to deterministic bytes.
    """

    result = state["result"]
    basis = state["basis"]
    metadata = {
        "schema_version": "mlx-atomistic.dft-fixed-density-state.v1",
        "grid_shape": list(basis.grid.shape),
        "cutoff_hartree": basis.cutoff_hartree,
        "kpoint_cartesian_bohr_inverse": list(basis.kpoint_cartesian),
        "active_count": basis.active_count,
        "coefficient_dtype": "complex64",
        "basis_fingerprint": basis.basis_fingerprint,
        "basis_order_fingerprint": basis.order_fingerprint,
        "lane_id": basis.lane_id,
    }
    compact = _state_for_basis(result, basis)
    return {
        "metadata.json": canonical_json_bytes(metadata) + b"\n",
        # The frozen v1 oracle requires a dense file. Materialize it directly
        # from private compact state without touching the public adapter or
        # retaining it in the runtime result.
        "coefficients.npy": _npy_bytes(
            compact.layout.unpack_fresh(compact.values)
        ),
        "eigenvalues.npy": _npy_bytes(result.eigenvalues),
        "density.npy": _npy_bytes(state["density"]),
        "effective-local-potential.npy": _npy_bytes(state["effective_local_potential"]),
        "basis-mask.npy": _npy_bytes(basis.mask),
    }


def serialize_periodic_scf_state(result: Any) -> dict[str, bytes]:
    """Serialize a periodic SCF result through the runtime-owned state adapter.

    Args:
        result: Periodic SCF result to serialize.

    Returns:
        Relative payload names mapped to deterministic bytes.
    """

    owned_lanes = []
    owned_states = []
    topology = result.time_reversal_ownership
    for owned_position, item in enumerate(result.owned_kpoints):
        owner_index = (
            owned_position if item.explicit_index is None else item.explicit_index
        )
        compact = _state_for_basis(item.eigen, item.basis)
        explicit_indices = (
            [owner_index]
            if topology is None
            else [
                entry.explicit_index
                for entry in topology.entries
                if entry.owner_index == owner_index
            ]
        )
        prefix = f"owned/{owner_index:04d}"
        lane = {
            "owner_index": owner_index,
            "explicit_indices": explicit_indices,
            "reduced_kpoint": list(item.reduced_kpoint),
            "aggregate_weight": item.integration_weight,
            "grid_shape": list(item.basis.grid.shape),
            "active_count": item.basis.active_count,
            "basis_fingerprint": item.basis.basis_fingerprint,
            "basis_order_fingerprint": item.basis.order_fingerprint,
            "lane_id": item.basis.lane_id,
            "compact_coefficient_file": f"{prefix}-coefficients.npy",
            "compact_index_file": f"{prefix}-indices.npy",
            "eigenvalue_file": f"{prefix}-eigenvalues.npy",
        }
        owned_lanes.append(lane)
        owned_states.append((owner_index, compact, item))

    metadata = {
        "schema_version": "mlx-atomistic.periodic-scf-compact-state.v2",
        "grid_shape": list(result.density.shape),
        "status": result.status,
        "converged": result.converged,
        "iterations": result.iterations,
        "total_energy_hartree": result.total_energy,
        "electron_count": result.electron_count,
        "batch_policy": dict(result.batch_policy),
        "kpoint_count": len(result.kpoints),
        "owned_lane_count": len(owned_lanes),
        "owned_lanes": owned_lanes,
        "kpoints": [
            {
                "index": index,
                "reduced_kpoint": list(item.reduced_kpoint),
                "weight": item.weight,
                "grid_shape": list(item.basis.grid.shape),
                "active_count": item.basis.active_count,
                "basis_fingerprint": item.basis.basis_fingerprint,
                "basis_order_fingerprint": item.basis.order_fingerprint,
                "lane_id": item.basis.lane_id,
            }
            for index, item in enumerate(result.kpoints)
        ],
    }
    payloads = {
        "metadata.json": canonical_json_bytes(metadata) + b"\n",
        "density.npy": _npy_bytes(result.density),
    }
    for lane, (owner_index, compact, item) in zip(
        owned_lanes,
        owned_states,
        strict=True,
    ):
        payloads[lane["compact_coefficient_file"]] = _npy_bytes(compact.values)
        payloads[lane["compact_index_file"]] = _npy_bytes(
            compact.layout._active_flat_indices_np
        )
        payloads[lane["eigenvalue_file"]] = _npy_bytes(item.eigen.eigenvalues)
        if topology is None:
            # Retain the legacy dense compatibility filename only for manually
            # constructed results with no ownership topology. The v2 lane
            # contract and official oracle consume the compact files above.
            payloads[f"kpoints/{owner_index:04d}-coefficients.npy"] = _npy_bytes(
                compact.layout.unpack_fresh(compact.values)
            )
    return payloads
