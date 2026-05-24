"""Shared compatibility-report normalization for prepared artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

COMPATIBILITY_TERM_ALIASES = {
    "bonds": "harmonic_bond",
    "angles": "harmonic_angle",
    "dihedrals": "periodic_dihedral",
    "impropers": "periodic_improper",
    "constraints": "distance_constraint",
    "exceptions": "nonbonded_exception",
    "nonbonded_exceptions": "nonbonded_exception",
    "pme_ewald": "pme",
    "pme_ewald_periodic_electrostatics": "pme",
    "pme_mesh_periodic_electrostatics": "pme",
    "particle_mesh_ewald": "pme",
    "cmap": "charmm_cmap",
    "charmm_cmap_terms": "charmm_cmap",
    "urey": "urey_bradley",
    "urey_bradley_terms": "urey_bradley",
    "charmm_urey_bradley": "urey_bradley",
    "rb_dihedrals": "rb_dihedral",
    "rb_torsion": "rb_dihedral",
    "rb_torsions": "rb_dihedral",
    "ryckaert_bellemans": "rb_dihedral",
    "ryckaert_bellemans_dihedral": "rb_dihedral",
    "ryckaert_bellemans_torsion": "rb_dihedral",
    "nbfix": "nbfix_pair_overrides",
    "nbfix_pair_override": "nbfix_pair_overrides",
    "pair_overrides": "nbfix_pair_overrides",
    "charmm_nbfix": "nbfix_pair_overrides",
    "force_switch": "charmm_force_switch_nonbonded",
    "force_switching": "charmm_force_switch_nonbonded",
    "charmm_force_switch": "charmm_force_switch_nonbonded",
    "virtual_sites": "virtual_site",
    "tip4p": "virtual_site",
    "tip5p": "virtual_site",
    "opc": "virtual_site",
    "advanced_water": "virtual_site",
    "advanced_water_model": "virtual_site",
}

TERM_COUNT_ALIASES = {
    "bonds": "harmonic_bond",
    "bond": "harmonic_bond",
    "harmonic_bonds": "harmonic_bond",
    "angles": "harmonic_angle",
    "angle": "harmonic_angle",
    "harmonic_angles": "harmonic_angle",
    "dihedrals": "periodic_dihedral",
    "periodic_dihedrals": "periodic_dihedral",
    "impropers": "periodic_improper",
    "periodic_impropers": "periodic_improper",
    "constraints": "distance_constraint",
    "distance_constraints": "distance_constraint",
    "nonbonded_exceptions": "nonbonded_exception",
    "nonbonded_exception_pairs": "nonbonded_exception",
    "exceptions": "nonbonded_exception",
    "rb_dihedrals": "rb_dihedral",
    "rb_torsions": "rb_dihedral",
    "charmm_cmap_terms": "charmm_cmap",
    "cmap_terms": "charmm_cmap",
    "urey_bradley_terms": "urey_bradley",
    "nbfix_pairs": "nbfix_pair_overrides",
    "nbfix_type_pairs": "nbfix_pair_overrides",
    "nbfix_pair_overrides": "nbfix_pair_overrides",
}

_PME_ARRAY_NAMES = (
    "pme_mesh_shape",
    "pme_alpha",
    "pme_real_cutoff",
    "pme_assignment_order",
    "pme_charge_tolerance",
    "pme_deconvolve_assignment",
)


def normalize_compatibility_report(
    report: dict[str, Any] | None,
    *,
    source: dict[str, Any] | None = None,
    parameter_source: str = "",
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Add canonical compatibility metadata while preserving parser-specific fields."""

    normalized = dict(report or {})
    source_payload = dict(source or {})
    supported = _normalized_terms(normalized.get("supported_terms"))
    required = _normalized_terms(normalized.get("required_terms"))
    if not required and supported:
        required = supported
    unsupported = _normalized_terms(normalized.get("unsupported_terms"))
    rejected = _normalized_terms(normalized.get("rejected_terms"))

    normalized.setdefault("unsupported_terms", [])
    normalized.setdefault("rejected_terms", [])
    normalized["supported_terms_normalized"] = supported
    normalized["required_terms_normalized"] = required
    normalized["unsupported_terms_normalized"] = unsupported
    normalized["rejected_terms_normalized"] = rejected

    term_counts = _normalized_term_counts(normalized.get("term_counts", {}))
    array_counts = _array_term_counts(arrays)
    if term_counts or array_counts:
        merged_counts = dict(term_counts)
        for key, value in array_counts.items():
            merged_counts.setdefault(key, value)
        normalized["term_counts_normalized"] = merged_counts
    if array_counts:
        normalized["array_term_counts"] = array_counts

    blockers = set(_string_list(normalized.get("blockers")))
    blockers.update(f"unsupported_terms:{term}" for term in unsupported)
    blockers.update(f"rejected_terms:{term}" for term in rejected)
    normalized["blockers"] = sorted(blockers)

    provenance = dict(normalized.get("parser_provenance", {}))
    source_kind = str(source_payload.get("kind", "") or "")
    parser = (
        source_payload.get("parser")
        or normalized.get("parser")
        or _default_parser_for_source(source_kind, parameter_source)
    )
    if source_kind or parser or parameter_source:
        if source_kind:
            provenance.setdefault("kind", source_kind)
        if parser:
            provenance.setdefault("parser", str(parser))
        if parameter_source:
            provenance.setdefault("parameter_source", str(parameter_source))
        normalized["parser_provenance"] = provenance
    return normalized


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_compatibility_term(value: str) -> str:
    term = value.strip().lower().replace("-", "_").replace(" ", "_")
    return COMPATIBILITY_TERM_ALIASES.get(term, term)


def _normalized_terms(value: Any) -> list[str]:
    return sorted(
        {
            term
            for term in (_normalize_compatibility_term(item) for item in _string_list(value))
            if term
        }
    )


def _normalize_term_count_key(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return TERM_COUNT_ALIASES.get(key, key)


def _normalized_term_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        normalized_key = _normalize_term_count_key(str(key))
        counts[normalized_key] = max(counts.get(normalized_key, 0), count)
    return counts


def _array_term_counts(arrays: dict[str, np.ndarray] | None) -> dict[str, int]:
    if arrays is None:
        return {}
    counts = {
        "harmonic_bond": _row_count(arrays, "bonds"),
        "harmonic_angle": _row_count(arrays, "angles"),
        "periodic_dihedral": _row_count(arrays, "dihedrals"),
        "periodic_improper": _row_count(arrays, "impropers"),
        "rb_dihedral": _row_count(arrays, "rb_dihedrals"),
        "distance_constraint": _row_count(arrays, "constraints"),
        "nonbonded_exception": _row_count(arrays, "nonbonded_exception_pairs"),
        "charmm_cmap": _row_count(arrays, "charmm_cmap_terms"),
        "urey_bradley": _row_count(arrays, "urey_bradley_terms"),
        "nbfix_pair_overrides": _row_count(arrays, "nbfix_pairs")
        + _row_count(arrays, "nbfix_type_pairs"),
    }
    if any(np.asarray(arrays.get(name, np.asarray([]))).size for name in _PME_ARRAY_NAMES):
        counts["pme"] = 1
    return {key: value for key, value in counts.items() if value > 0}


def _row_count(arrays: dict[str, np.ndarray], name: str) -> int:
    if name not in arrays:
        return 0
    array = np.asarray(arrays[name])
    if array.size == 0:
        return 0
    if array.ndim == 0:
        return int(array.size)
    return int(array.shape[0])


def _default_parser_for_source(source_kind: str, parameter_source: str) -> str | None:
    kind = source_kind.lower()
    parameter = parameter_source.lower()
    if kind == "amber" or parameter == "amber_prmtop":
        return "native_amber_prmtop"
    if kind == "charmm" and "native" in parameter:
        return "native_charmm_psf"
    if kind == "gromacs" or "gromacs" in parameter:
        return "native_gromacs_top_gro"
    return None


__all__ = [
    "COMPATIBILITY_TERM_ALIASES",
    "TERM_COUNT_ALIASES",
    "normalize_compatibility_report",
]
