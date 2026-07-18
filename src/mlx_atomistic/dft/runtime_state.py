"""Runtime-owned DFT state accounting and serialization adapters."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes


def _npy_bytes(values: object) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def fixed_density_state_metrics(*, result: Any, basis: Any) -> dict[str, int]:
    """Return logical runtime-owned bytes for one fixed-density eigenstate.

    Args:
        result: Periodic eigensolver result.
        basis: Basis owning the result coefficients.

    Returns:
        Logical coefficient payload and full-grid byte counts.
    """

    coefficients = result.coefficients
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
    }
    return {
        "metadata.json": canonical_json_bytes(metadata) + b"\n",
        "coefficients.npy": _npy_bytes(result.coefficients),
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

    metadata = {
        "schema_version": "mlx-atomistic.periodic-scf-state.v1",
        "grid_shape": list(result.density.shape),
        "status": result.status,
        "converged": result.converged,
        "iterations": result.iterations,
        "total_energy_hartree": result.total_energy,
        "electron_count": result.electron_count,
        "kpoint_count": len(result.kpoints),
        "kpoints": [
            {
                "index": index,
                "reduced_kpoint": list(item.reduced_kpoint),
                "weight": item.weight,
                "grid_shape": list(item.basis.grid.shape),
                "active_count": item.basis.active_count,
            }
            for index, item in enumerate(result.kpoints)
        ],
    }
    payloads = {
        "metadata.json": canonical_json_bytes(metadata) + b"\n",
        "density.npy": _npy_bytes(result.density),
    }
    for index, item in enumerate(result.kpoints):
        payloads[f"kpoints/{index:04d}-coefficients.npy"] = _npy_bytes(
            item.eigen.coefficients
        )
        payloads[f"kpoints/{index:04d}-eigenvalues.npy"] = _npy_bytes(
            item.eigen.eigenvalues
        )
    return payloads
