"""Deterministic orthorhombic replication of prepared atomistic systems."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

import numpy as np

from mlx_atomistic.pme import normalize_pme_background_policy
from mlx_atomistic.prep.schema import ARTIFACT_VERSION, PreparedSystem

_TRANSLATED_ATOM_FIELDS = ("positions", "reference_positions")
_REPEATED_ATOM_FIELDS = (
    "symbols",
    "atom_names",
    "atom_types",
    "residue_names",
    "velocities",
    "masses",
    "charges",
    "sigma",
    "epsilon",
    "ligand_mask",
    "receptor_mask",
    "restraint_mask",
    "water_mask",
    "ion_mask",
    "lipid_mask",
    "gbsa_radius",
    "gbsa_scale",
)
_INDEX_PARAMETER_GROUPS = {
    "bonds": ("bond_k", "bond_length"),
    "angles": ("angle_k", "angle_theta"),
    "dihedrals": ("dihedral_k", "dihedral_periodicity", "dihedral_phase"),
    "nonbonded_pairs": (),
    "rb_dihedrals": ("rb_c0", "rb_c1", "rb_c2", "rb_c3", "rb_c4", "rb_c5"),
    "constraints": ("constraint_distance",),
    "impropers": ("improper_k", "improper_periodicity", "improper_phase"),
    "nonbonded_exception_pairs": (
        "nonbonded_exception_charge_product",
        "nonbonded_exception_sigma",
        "nonbonded_exception_epsilon",
    ),
    "charmm_cmap_terms": ("charmm_cmap_grid_indices",),
    "urey_bradley_terms": ("urey_bradley_k", "urey_bradley_distance"),
    "nbfix_pairs": ("nbfix_sigma", "nbfix_epsilon"),
    "virtual_site_parent_atoms": ("virtual_site_weights", "virtual_site_types"),
}
_GLOBAL_ARRAY_FIELDS = (
    "charmm_cmap_grids",
    "nbfix_type_pairs",
    "nbfix_type_sigma",
    "nbfix_type_epsilon",
)
_PME_ARRAY_FIELDS = (
    "pme_mesh_shape",
    "pme_alpha",
    "pme_real_cutoff",
    "pme_assignment_order",
    "pme_charge_tolerance",
    "pme_deconvolve_assignment",
    "pme_background_policy",
)
_SPECIAL_FIELDS = {
    "metadata",
    "residue_ids",
    "chain_ids",
    "cell_lengths",
    "cell_matrix",
    *_TRANSLATED_ATOM_FIELDS,
    *_REPEATED_ATOM_FIELDS,
    *_INDEX_PARAMETER_GROUPS,
    *(name for names in _INDEX_PARAMETER_GROUPS.values() for name in names),
    *_GLOBAL_ARRAY_FIELDS,
    *_PME_ARRAY_FIELDS,
}


class PreparedSupercellError(ValueError):
    """Raised when a prepared system cannot be replicated without data loss."""


def normalize_supercell_replicas(replicas: object) -> tuple[int, int, int]:
    """Validate and normalize three positive integer replica counts.

    Args:
        replicas: Three integer counts ``(nx, ny, nz)``.

    Returns:
        A normalized ``(nx, ny, nz)`` tuple.

    Raises:
        PreparedSupercellError: If the value does not contain three positive integers.
    """

    try:
        values = tuple(replicas)  # type: ignore[arg-type]
    except TypeError as exc:
        msg = "supercell replicas must contain three positive integers"
        raise PreparedSupercellError(msg) from exc
    if len(values) != 3:
        msg = "supercell replicas must contain exactly three values"
        raise PreparedSupercellError(msg)
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            msg = "supercell replicas must be positive integers"
            raise PreparedSupercellError(msg)
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            msg = "supercell replicas must be positive integers"
            raise PreparedSupercellError(msg) from exc
        if integer != value or integer <= 0:
            msg = "supercell replicas must be positive integers"
            raise PreparedSupercellError(msg)
        normalized.append(integer)
    return tuple(normalized)  # type: ignore[return-value]


def replicate_prepared_system(
    prepared: PreparedSystem,
    replicas: object,
    *,
    assignment_order: int | None = None,
    background_policy: str | None = None,
) -> PreparedSystem:
    """Replicate a prepared system over an orthorhombic integer supercell.

    Args:
        prepared: Valid source prepared system. It is never mutated.
        replicas: Three positive integer counts ``(nx, ny, nz)``.
        assignment_order: Optional PME assignment-order override.
        background_policy: Optional PME background-charge policy override.

    Returns:
        A deterministic replicated `PreparedSystem` using replica-major atom
        order and scaled cell/PME mesh dimensions.

    Raises:
        PreparedSupercellError: If the source is non-periodic, non-orthorhombic,
            contains an unsupported populated field, or receives an invalid PME override.
    """

    if not isinstance(prepared, PreparedSystem):
        msg = "prepared must be a PreparedSystem"
        raise TypeError(msg)
    prepared.validate()
    replica_shape = normalize_supercell_replicas(replicas)
    replica_offsets, translations, cell_lengths = _replica_layout(prepared, replica_shape)
    replica_count = len(replica_offsets)
    atom_count = prepared.atom_count

    payload: dict[str, Any] = {}
    for name in _REPEATED_ATOM_FIELDS:
        payload[name] = _repeat_optional_atom_array(
            getattr(prepared, name),
            atom_count=atom_count,
            replica_count=replica_count,
            name=name,
        )
    for name in _TRANSLATED_ATOM_FIELDS:
        payload[name] = _replicate_translated_coordinates(
            getattr(prepared, name),
            translations,
            atom_count=atom_count,
            name=name,
        )
    payload["residue_ids"] = _replicate_residue_ids(prepared.residue_ids, replica_count)
    payload["chain_ids"] = _replicate_chain_ids(prepared.chain_ids, replica_count)

    for index_name, parameter_names in _INDEX_PARAMETER_GROUPS.items():
        allow_negative = index_name == "virtual_site_parent_atoms"
        payload[index_name] = _replicate_index_array(
            getattr(prepared, index_name),
            replica_offsets,
            allow_negative=allow_negative,
        )
        for parameter_name in parameter_names:
            payload[parameter_name] = _repeat_term_parameter(
                getattr(prepared, parameter_name),
                replica_count,
            )

    for name in _GLOBAL_ARRAY_FIELDS:
        payload[name] = np.asarray(getattr(prepared, name)).copy()

    scaled_lengths = cell_lengths * np.asarray(replica_shape, dtype=np.float64)
    payload["cell_lengths"] = scaled_lengths.astype(np.float32)
    payload["cell_matrix"] = np.diag(scaled_lengths).astype(np.float32)
    pme_arrays, pme_config = _replicated_pme_state(
        prepared,
        replica_shape,
        assignment_order=assignment_order,
        background_policy=background_policy,
    )
    payload.update(pme_arrays)

    _copy_unclassified_empty_fields(prepared, payload)
    payload["metadata"] = _replicated_metadata(
        prepared,
        replica_shape,
        pme_config=pme_config,
        atom_count=atom_count * replica_count,
        net_charge=float(np.sum(payload["charges"], dtype=np.float64)),
    )
    replicated = PreparedSystem(**payload)
    replicated.validate()
    return replicated


def prepared_supercell_summary(
    prepared: PreparedSystem,
    *,
    source_atom_count: int | None = None,
    replicas: object | None = None,
) -> dict[str, object]:
    """Return a structural summary for a replicated prepared system.

    Args:
        prepared: Prepared system to summarize.
        source_atom_count: Optional source atom count used for provenance checks.
        replicas: Optional three-axis replica counts.

    Returns:
        JSON-serializable atom, cell, PME, charge, and indexed-term counts.
    """

    prepared.validate()
    normalized_replicas = None if replicas is None else normalize_supercell_replicas(replicas)
    indexed_counts = {
        name: int(np.asarray(getattr(prepared, name)).shape[0])
        for name in _INDEX_PARAMETER_GROUPS
    }
    pme_config = dict(prepared.metadata.pme_config)
    return {
        "artifact_version": prepared.metadata.artifact_version,
        "atom_count": prepared.atom_count,
        "source_atom_count": source_atom_count,
        "replicas": normalized_replicas,
        "net_charge": float(np.sum(prepared.charges, dtype=np.float64)),
        "cell_lengths": np.asarray(prepared.cell_lengths, dtype=np.float64).tolist(),
        "cell_matrix": np.asarray(prepared.cell_matrix, dtype=np.float64).tolist(),
        "mesh_shape": np.asarray(prepared.pme_mesh_shape, dtype=np.int64).tolist(),
        "assignment_order": (
            None
            if np.asarray(prepared.pme_assignment_order).size == 0
            else int(np.asarray(prepared.pme_assignment_order)[0])
        ),
        "background_policy": (
            None
            if np.asarray(prepared.pme_background_policy).size == 0
            else str(np.asarray(prepared.pme_background_policy)[0])
        ),
        "pme_config": pme_config,
        "indexed_term_counts": indexed_counts,
        "replica_order": "x-major,y-middle,z-minor,source-atom-minor",
    }


def _replica_layout(
    prepared: PreparedSystem,
    replicas: tuple[int, int, int],
) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    lengths = np.asarray(prepared.cell_lengths, dtype=np.float64)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        msg = "prepared supercells require finite positive cell_lengths"
        raise PreparedSupercellError(msg)
    matrix = np.asarray(prepared.cell_matrix, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.diag(lengths)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        msg = "prepared supercells require a finite cell_matrix with shape (3, 3)"
        raise PreparedSupercellError(msg)
    diagonal = np.diag(np.diag(matrix))
    if not np.allclose(matrix, diagonal, rtol=0.0, atol=1e-6):
        msg = "prepared supercells currently support orthorhombic cells only"
        raise PreparedSupercellError(msg)
    if not np.allclose(np.abs(np.diag(matrix)), lengths, rtol=1e-6, atol=1e-6):
        msg = "prepared cell_lengths must match the orthorhombic cell_matrix"
        raise PreparedSupercellError(msg)

    replica_offsets = []
    translations = []
    replica_index = 0
    for ix in range(replicas[0]):
        for iy in range(replicas[1]):
            for iz in range(replicas[2]):
                replica_offsets.append(replica_index * prepared.atom_count)
                translations.append(
                    np.asarray(
                        [ix * lengths[0], iy * lengths[1], iz * lengths[2]],
                        dtype=np.float64,
                    )
                )
                replica_index += 1
    return replica_offsets, translations, lengths


def _repeat_optional_atom_array(
    values: object,
    *,
    atom_count: int,
    replica_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return array.copy()
    if array.shape[:1] != (atom_count,):
        msg = f"{name} must be empty or have atom-leading shape ({atom_count}, ...)"
        raise PreparedSupercellError(msg)
    return np.concatenate([array.copy() for _ in range(replica_count)], axis=0)


def _replicate_translated_coordinates(
    values: object,
    translations: list[np.ndarray],
    *,
    atom_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (atom_count, 3):
        msg = f"{name} must have shape ({atom_count}, 3)"
        raise PreparedSupercellError(msg)
    translated = [
        (array.astype(np.float64) + translation).astype(array.dtype)
        for translation in translations
    ]
    return np.concatenate(
        translated,
        axis=0,
    )


def _replicate_residue_ids(values: object, replica_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        msg = "residue_ids must be a non-empty one-dimensional array"
        raise PreparedSupercellError(msg)
    minimum = int(np.min(array))
    span = int(np.max(array)) - minimum + 1
    output = np.concatenate(
        [array.astype(np.int64) + replica_index * span for replica_index in range(replica_count)]
    )
    dtype = array.dtype
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if np.any(output < limits.min) or np.any(output > limits.max):
            msg = "replicated residue_ids exceed the source integer dtype"
            raise PreparedSupercellError(msg)
    return output.astype(dtype)


def _replicate_chain_ids(values: object, replica_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=str)
    if array.ndim != 1 or array.size == 0:
        msg = "chain_ids must be a non-empty one-dimensional array"
        raise PreparedSupercellError(msg)
    if replica_count == 1:
        return array.copy()
    return np.concatenate(
        [
            np.asarray([f"{chain or '_'}:{replica_index}" for chain in array], dtype=str)
            for replica_index in range(replica_count)
        ]
    )


def _replicate_index_array(
    values: object,
    offsets: list[int],
    *,
    allow_negative: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2:
        msg = "prepared indexed-term arrays must be two-dimensional"
        raise PreparedSupercellError(msg)
    if array.shape[0] == 0:
        return array.copy()
    replicas = []
    for offset in offsets:
        if allow_negative:
            replicas.append(np.where(array >= 0, array + offset, array).astype(np.int32))
        else:
            replicas.append((array + offset).astype(np.int32))
    return np.concatenate(replicas, axis=0)


def _repeat_term_parameter(values: object, replica_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] == 0:
        return array.copy()
    return np.concatenate([array.copy() for _ in range(replica_count)], axis=0)


def _replicated_pme_state(
    prepared: PreparedSystem,
    replicas: tuple[int, int, int],
    *,
    assignment_order: int | None,
    background_policy: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mesh = np.asarray(prepared.pme_mesh_shape)
    has_pme = mesh.size != 0 or bool(prepared.metadata.pme_config)
    if not has_pme:
        if assignment_order is not None or background_policy is not None:
            msg = "PME overrides require a source prepared system with PME configuration"
            raise PreparedSupercellError(msg)
        return (
            {name: np.asarray(getattr(prepared, name)).copy() for name in _PME_ARRAY_FIELDS},
            {},
        )
    if mesh.shape != (3,):
        msg = "PME supercell replication requires pme_mesh_shape with shape (3,)"
        raise PreparedSupercellError(msg)
    scaled_mesh = mesh.astype(np.int64) * np.asarray(replicas, dtype=np.int64)
    if np.any(scaled_mesh > np.iinfo(np.int32).max):
        msg = "replicated PME mesh exceeds int32 dimensions"
        raise PreparedSupercellError(msg)

    source_assignment = int(np.asarray(prepared.pme_assignment_order)[0])
    if assignment_order is None:
        selected_assignment = source_assignment
    else:
        if isinstance(assignment_order, (bool, np.bool_)):
            msg = "assignment_order must be one of 2, 4, or 5"
            raise PreparedSupercellError(msg)
        try:
            selected_assignment = int(assignment_order)
        except (TypeError, ValueError, OverflowError) as exc:
            msg = "assignment_order must be one of 2, 4, or 5"
            raise PreparedSupercellError(msg) from exc
        if selected_assignment != assignment_order:
            msg = "assignment_order must be one of 2, 4, or 5"
            raise PreparedSupercellError(msg)
    if selected_assignment not in {2, 4, 5}:
        msg = "assignment_order must be one of 2, 4, or 5"
        raise PreparedSupercellError(msg)
    source_policy_array = np.asarray(prepared.pme_background_policy, dtype=str)
    source_policy = (
        str(source_policy_array[0])
        if source_policy_array.size
        else str(prepared.metadata.pme_config.get("background_policy", "reject_non_neutral"))
    )
    try:
        selected_policy = normalize_pme_background_policy(
            source_policy if background_policy is None else background_policy
        )
    except ValueError as exc:
        raise PreparedSupercellError(str(exc)) from exc
    alpha = float(np.asarray(prepared.pme_alpha)[0])
    real_cutoff = float(np.asarray(prepared.pme_real_cutoff)[0])
    charge_tolerance = float(np.asarray(prepared.pme_charge_tolerance)[0])
    deconvolve = bool(np.asarray(prepared.pme_deconvolve_assignment)[0])
    arrays = {
        "pme_mesh_shape": scaled_mesh.astype(np.int32),
        "pme_alpha": np.asarray([alpha], dtype=np.float32),
        "pme_real_cutoff": np.asarray([real_cutoff], dtype=np.float32),
        "pme_assignment_order": np.asarray([selected_assignment], dtype=np.int32),
        "pme_charge_tolerance": np.asarray([charge_tolerance], dtype=np.float32),
        "pme_deconvolve_assignment": np.asarray([deconvolve], dtype=bool),
        "pme_background_policy": np.asarray([selected_policy], dtype=str),
    }
    config = {
        "mesh_shape": scaled_mesh.astype(int).tolist(),
        "alpha": alpha,
        "real_cutoff": real_cutoff,
        "assignment_order": selected_assignment,
        "charge_tolerance": charge_tolerance,
        "deconvolve_assignment": deconvolve,
        "background_policy": selected_policy,
    }
    return arrays, config


def _copy_unclassified_empty_fields(
    prepared: PreparedSystem,
    payload: dict[str, Any],
) -> None:
    field_names = {field.name for field in fields(PreparedSystem)}
    unknown = sorted(field_names - _SPECIAL_FIELDS)
    populated = [name for name in unknown if np.asarray(getattr(prepared, name)).size]
    if populated:
        msg = "unsupported populated prepared-system fields: " + ", ".join(populated)
        raise PreparedSupercellError(msg)
    for name in unknown:
        payload[name] = np.asarray(getattr(prepared, name)).copy()


def _replicated_metadata(
    prepared: PreparedSystem,
    replicas: tuple[int, int, int],
    *,
    pme_config: dict[str, Any],
    atom_count: int,
    net_charge: float,
):
    replica_count = int(np.prod(replicas, dtype=np.int64))
    source = deepcopy(prepared.metadata.source)
    source["prepared_supercell"] = {
        "replicas": list(replicas),
        "source_atom_count": prepared.atom_count,
        "replica_order": "x-major,y-middle,z-minor,source-atom-minor",
    }
    selections = deepcopy(prepared.metadata.selections)
    for key, value in tuple(selections.items()):
        if key.endswith("_count") and isinstance(value, (int, np.integer)):
            selections[key] = int(value) * replica_count
    selections.update(
        {
            "atom_count": atom_count,
            "system_charge": net_charge,
            "supercell_replicas": list(replicas),
            "source_atom_count": prepared.atom_count,
        }
    )
    compatibility = _replicated_compatibility_report(
        prepared.metadata.compatibility_report,
        replicas,
    )
    hydrogen_masses = np.asarray(prepared.masses)[np.asarray(prepared.symbols, dtype=str) == "H"]
    if hydrogen_masses.size and np.any(hydrogen_masses > 1.25):
        compatibility.setdefault(
            "hydrogen_mass_repartitioning",
            "represented_by_masses",
        )
    protocol = deepcopy(prepared.metadata.protocol_metadata)
    if pme_config:
        nonbonded = dict(protocol.get("nonbonded", {}))
        nonbonded["cutoff"] = float(pme_config["real_cutoff"])
        protocol["nonbonded"] = nonbonded
    warnings = list(prepared.metadata.warnings)
    warnings.append(
        "prepared system replicated deterministically over supercell "
        + "x".join(str(value) for value in replicas)
    )
    return replace(
        prepared.metadata,
        artifact_version=ARTIFACT_VERSION,
        source=source,
        selections=selections,
        compatibility_report=compatibility,
        warnings=warnings,
        pme_config=pme_config,
        protocol_metadata=protocol,
    )


def _replicated_compatibility_report(
    report: dict[str, Any],
    replicas: tuple[int, int, int],
) -> dict[str, Any]:
    replicated = deepcopy(report)
    replica_count = int(np.prod(replicas, dtype=np.int64))
    if isinstance(replicated.get("hydrogen_count"), (int, np.integer)):
        replicated["hydrogen_count"] = int(replicated["hydrogen_count"]) * replica_count
    for container_name in ("array_term_counts", "term_counts", "term_counts_normalized"):
        counts = replicated.get(container_name)
        if not isinstance(counts, dict):
            continue
        replicated[container_name] = {
            key: (
                int(value)
                if str(key).lower().startswith("pme")
                else int(value) * replica_count
            )
            if isinstance(value, (int, np.integer))
            else value
            for key, value in counts.items()
        }
    replicated["supercell_replicas"] = list(replicas)
    return replicated


__all__ = [
    "PreparedSupercellError",
    "normalize_supercell_replicas",
    "prepared_supercell_summary",
    "replicate_prepared_system",
]
