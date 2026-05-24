"""Hydrogen mass repartitioning helpers for prepared systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from mlx_atomistic.prep.schema import PreparedSystem

DEFAULT_HMR_TARGET_HYDROGEN_MASS = 3.024


def apply_hydrogen_mass_repartitioning(
    prepared: PreparedSystem,
    *,
    target_hydrogen_mass: float = DEFAULT_HMR_TARGET_HYDROGEN_MASS,
    hydrogen_indices: Sequence[int] | None = None,
    selection: str = "all_bonded_hydrogens",
    min_heavy_atom_mass: float = 0.0,
    require_constraints: bool = True,
) -> PreparedSystem:
    """Return a copy of `prepared` with deterministic HMR masses and provenance."""

    prepared.validate()
    _validate_no_virtual_site_claim(prepared.metadata.compatibility_report)
    input_masses = np.asarray(prepared.masses)
    if not np.issubdtype(input_masses.dtype, np.floating):
        msg = "HMR requires prepared.masses to use a floating dtype"
        raise TypeError(msg)
    original = input_masses.astype(np.float64)
    transformed = original.copy()
    symbols = np.char.upper(np.asarray(prepared.symbols, dtype=str))
    bonds = np.asarray(prepared.bonds, dtype=np.int32)
    selected = _selected_hydrogen_indices(
        symbols,
        hydrogen_indices=hydrogen_indices,
        selection=selection,
    )
    selection_label = _hmr_selection_label(
        symbols,
        selected=selected,
        hydrogen_indices=hydrogen_indices,
        selection=selection,
    )
    if target_hydrogen_mass <= 0.0 or not np.isfinite(target_hydrogen_mass):
        msg = "target_hydrogen_mass must be finite and positive"
        raise ValueError(msg)
    if min_heavy_atom_mass < 0.0 or not np.isfinite(min_heavy_atom_mass):
        msg = "min_heavy_atom_mass must be finite and non-negative"
        raise ValueError(msg)
    constraint_pairs = _constraint_pair_set(prepared.constraints)
    hydrogen_records: list[dict[str, Any]] = []
    heavy_atoms: set[int] = set()
    for hydrogen_index in selected:
        heavy_index = _bonded_heavy_atom_index(
            hydrogen_index,
            symbols=symbols,
            bonds=bonds,
        )
        if require_constraints and (
            min(hydrogen_index, heavy_index),
            max(hydrogen_index, heavy_index),
        ) not in constraint_pairs:
            msg = (
                "HMR requires selected hydrogen bonds to be distance-constrained; "
                f"missing constraint for hydrogen {hydrogen_index} and heavy atom {heavy_index}"
            )
            raise ValueError(msg)
        delta = float(target_hydrogen_mass - original[hydrogen_index])
        if delta <= 0.0:
            msg = (
                "target_hydrogen_mass must increase selected hydrogen masses; "
                f"hydrogen {hydrogen_index} has mass {original[hydrogen_index]:g}"
            )
            raise ValueError(msg)
        transformed[hydrogen_index] += delta
        transformed[heavy_index] -= delta
        if transformed[heavy_index] <= min_heavy_atom_mass:
            msg = (
                "HMR would reduce bonded heavy-atom mass below policy minimum; "
                f"heavy atom {heavy_index} mass={transformed[heavy_index]:g}"
            )
            raise ValueError(msg)
        heavy_atoms.add(heavy_index)
        hydrogen_records.append(
            {
                "hydrogen_index": int(hydrogen_index),
                "heavy_atom_index": int(heavy_index),
                "original_hydrogen_mass": float(original[hydrogen_index]),
                "transformed_hydrogen_mass": float(transformed[hydrogen_index]),
                "mass_delta": delta,
            }
        )

    total_before = float(np.sum(original, dtype=np.float64))
    total_after = float(np.sum(transformed, dtype=np.float64))
    if not np.isclose(total_before, total_after, rtol=0.0, atol=1e-10):
        msg = "HMR failed to preserve total mass"
        raise ValueError(msg)
    final_masses = transformed.astype(input_masses.dtype, copy=False)
    final_total = float(np.sum(final_masses, dtype=np.float64))
    if not np.isclose(total_before, final_total, rtol=0.0, atol=1e-6):
        msg = (
            "HMR final mass dtype conversion failed to preserve total mass; "
            "use a floating mass dtype with sufficient precision"
        )
        raise ValueError(msg)
    provenance = {
        "status": "represented_by_masses",
        "policy": {
            "kind": "hydrogen_mass_repartitioning",
            "selection": selection_label,
            "target_hydrogen_mass": float(target_hydrogen_mass),
            "min_heavy_atom_mass": float(min_heavy_atom_mass),
            "heavy_mass_policy": "subtract_hydrogen_mass_delta_from_bonded_heavy_atom",
            "require_constraints": bool(require_constraints),
            "virtual_sites_supported": False,
        },
        "original_masses": [float(item) for item in original.tolist()],
        "transformed_masses": [float(item) for item in final_masses.tolist()],
        "selected_hydrogens": hydrogen_records,
        "heavy_atoms": [
            {
                "heavy_atom_index": int(index),
                "original_mass": float(original[index]),
                "transformed_mass": float(final_masses[index]),
                "mass_delta": float(final_masses[index] - original[index]),
            }
            for index in sorted(heavy_atoms)
        ],
        "total_mass_before": total_before,
        "total_mass_after": final_total,
    }
    if hydrogen_indices is not None:
        provenance["policy"]["hydrogen_indices"] = [int(index) for index in selected]
    metadata = prepared.metadata
    compatibility_report = dict(metadata.compatibility_report)
    compatibility_report["hydrogen_mass_repartitioning"] = "represented_by_masses"
    compatibility_report.setdefault("virtual_sites_present", False)
    for key in ("supported_terms", "required_terms"):
        terms = [str(item) for item in compatibility_report.get(key, [])]
        if require_constraints and "distance_constraint" not in {
            item.strip().lower().replace("-", "_").replace(" ", "_") for item in terms
        }:
            terms.append("distance_constraint")
        compatibility_report[key] = terms
    protocol_metadata = dict(metadata.protocol_metadata)
    protocol_metadata["hydrogen_mass_repartitioning"] = provenance
    return replace(
        prepared,
        metadata=replace(
            metadata,
            compatibility_report=compatibility_report,
            protocol_metadata=protocol_metadata,
        ),
        masses=final_masses,
    )


def _validate_no_virtual_site_claim(report: dict[str, Any]) -> None:
    water_model = str(report.get("water_model") or report.get("solvent_model") or "").lower()
    if bool(report.get("virtual_sites_present", False)) or any(
        model in water_model for model in ("tip4p", "tip5p", "opc")
    ):
        msg = "HMR does not support virtual-site artifacts"
        raise ValueError(msg)


def _selected_hydrogen_indices(
    symbols: np.ndarray,
    *,
    hydrogen_indices: Sequence[int] | None,
    selection: str,
) -> tuple[int, ...]:
    if hydrogen_indices is None:
        if selection != "all_bonded_hydrogens":
            msg = "explicit hydrogen_indices are required for non-default HMR selection"
            raise ValueError(msg)
        selected = tuple(int(index) for index in np.flatnonzero(symbols == "H").tolist())
    else:
        selected = tuple(sorted({int(index) for index in hydrogen_indices}))
    if not selected:
        msg = "HMR selection did not include any hydrogens"
        raise ValueError(msg)
    atom_count = int(symbols.shape[0])
    for index in selected:
        if index < 0 or index >= atom_count:
            msg = f"HMR hydrogen index {index} is outside [0, atom_count)"
            raise ValueError(msg)
        if symbols[index] != "H":
            msg = f"HMR selected atom {index} is not hydrogen"
            raise ValueError(msg)
    return selected


def _hmr_selection_label(
    symbols: np.ndarray,
    *,
    selected: tuple[int, ...],
    hydrogen_indices: Sequence[int] | None,
    selection: str,
) -> str:
    if hydrogen_indices is None or selection != "all_bonded_hydrogens":
        return selection
    all_hydrogens = tuple(int(index) for index in np.flatnonzero(symbols == "H").tolist())
    if selected == all_hydrogens:
        return selection
    return "explicit_hydrogen_indices"


def _bonded_heavy_atom_index(
    hydrogen_index: int,
    *,
    symbols: np.ndarray,
    bonds: np.ndarray,
) -> int:
    candidates: list[int] = []
    for left, right in bonds.tolist():
        if int(left) == hydrogen_index and symbols[int(right)] != "H":
            candidates.append(int(right))
        elif int(right) == hydrogen_index and symbols[int(left)] != "H":
            candidates.append(int(left))
    if not candidates:
        msg = f"HMR selected hydrogen {hydrogen_index} has no bonded heavy atom"
        raise ValueError(msg)
    return min(candidates)


def _constraint_pair_set(constraints: np.ndarray) -> set[tuple[int, int]]:
    values = np.asarray(constraints, dtype=np.int32)
    if values.size == 0:
        return set()
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in values.tolist()
    }


__all__ = [
    "DEFAULT_HMR_TARGET_HYDROGEN_MASS",
    "apply_hydrogen_mass_repartitioning",
]
