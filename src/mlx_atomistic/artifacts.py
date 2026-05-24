"""Strict loaders for prepared MLX molecular mechanics artifacts."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.compatibility import normalize_compatibility_report
from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import (
    CHARMMCMAPPotential,
    CHARMMForceSwitchNonbondedPotential,
    CHARMMUreyBradleyPotential,
    CustomForcePotential,
    GBSAForcePotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    NonbondedPotential,
    PeriodicDihedralPotential,
    PMEConfig,
    RBDihedralPotential,
    SoftCoreNonbondedPotential,
)
from mlx_atomistic.mm import MMSystem
from mlx_atomistic.nonbonded import normalize_nonbonded_electrostatics
from mlx_atomistic.pme import PME_SUPPORTED_ASSIGNMENT_ORDERS
from mlx_atomistic.replica_exchange import _validate_replica_exchange_metadata_payload
from mlx_atomistic.runtime import ReadinessReport
from mlx_atomistic.topology import DEFAULT_EAGER_NONBONDED_PAIR_LIMIT, Topology
from mlx_atomistic.virtual_sites import (
    ThreeParticleAverage,
    VirtualSiteManager,
    tip4p_ew_virtual_site,
)

if TYPE_CHECKING:
    from mlx_atomistic.units import MDUnitSystem

PREPARED_JSON_NAME = "prepared_system.json"
PREPARED_NPZ_NAME = "prepared_system.npz"
DEFAULT_COULOMB_CONSTANT_KJ_MOL_ANGSTROM = 1389.3545764438198

SUPPORTED_FORCE_TERMS = frozenset(
    {
        "harmonic_bond",
        "harmonic_angle",
        "periodic_dihedral",
        "periodic_torsion",
        "periodic_improper",
        "rb_dihedral",
        "nonbonded_lj_coulomb",
        "pair_restricted_lj_coulomb",
        "nonbonded_exception",
        "soft_core_lj",
        "lambda_scaled_nonbonded",
        "ewald_reference_electrostatics",
        "distance_constraint",
        "positional_restraint",
        "pme",
        "charmm_cmap",
        "urey_bradley",
        "nbfix_pair_overrides",
        "charmm_force_switch_nonbonded",
        "custom_force",
        "gbsa",
        "virtual_site",
        "lipid",
        "water",
        "ion",
        "receptor",
        "ligand",
        "replica_exchange",
    }
)
FAIL_CLOSED_TERMS = frozenset(
    {
        "ljpme",
        "barostat",
        "npt",
        "npt_barostat",
        "virtual_sites",
        "advanced_water",
        "advanced_water_model",
        "tip5p",
        "opc",
        "hmr_or_virtual_site_policy_required",
        "hydrogen_mass_repartitioning",
        "virtual_sites_or_hydrogen_mass_repartitioning_not_checked",
        "charmm_virtual_sites",
        "charmm_unsupported_water_model",
        "gromacs_virtual_sites",
        "gromacs_directive_virtual_sites2",
        "drude",
        "polarizable",
        "reactive",
        "bond_breaking",
        "qm_mm",
    }
)
TERM_ALIASES = {
    "bonds": "harmonic_bond",
    "angles": "harmonic_angle",
    "dihedrals": "periodic_dihedral",
    "rb_dihedrals": "rb_dihedral",
    "rb_torsion": "rb_dihedral",
    "rb_torsions": "rb_dihedral",
    "ryckaert_bellemans": "rb_dihedral",
    "ryckaert_bellemans_dihedral": "rb_dihedral",
    "ryckaert_bellemans_torsion": "rb_dihedral",
    "impropers": "periodic_improper",
    "ewald": "ewald_reference_electrostatics",
    "ewald_reference": "ewald_reference_electrostatics",
    "pme_ewald": "pme",
    "pme_ewald_periodic_electrostatics": "pme",
    "pme_mesh_periodic_electrostatics": "pme",
    "particle_mesh_ewald": "pme",
    "cmap": "charmm_cmap",
    "charmm_cmap_terms": "charmm_cmap",
    "urey": "urey_bradley",
    "urey_bradley_terms": "urey_bradley",
    "charmm_urey_bradley": "urey_bradley",
    "nbfix": "nbfix_pair_overrides",
    "nbfix_pair_override": "nbfix_pair_overrides",
    "nbfix_pair_overrides": "nbfix_pair_overrides",
    "pair_overrides": "nbfix_pair_overrides",
    "charmm_nbfix": "nbfix_pair_overrides",
    "force_switch": "charmm_force_switch_nonbonded",
    "force_switching": "charmm_force_switch_nonbonded",
    "charmm_force_switch": "charmm_force_switch_nonbonded",
    "lipids": "lipid",
    "lipid_mask": "lipid",
    "waters": "water",
    "water_mask": "water",
    "ions": "ion",
    "ion_mask": "ion",
    "receptor_mask": "receptor",
    "ligand_mask": "ligand",
    "constraints": "distance_constraint",
    "hmr": "hydrogen_mass_repartitioning",
    "hydrogen_mass_repartitioning": "hydrogen_mass_repartitioning",
    "exceptions": "nonbonded_exception",
    "nonbonded_exceptions": "nonbonded_exception",
    "soft_core": "soft_core_lj",
    "softcore_lj": "soft_core_lj",
    "lambda_nonbonded": "lambda_scaled_nonbonded",
    "lambda_scaled_lj_coulomb": "lambda_scaled_nonbonded",
    "tip4p": "virtual_site",
    "tip5p": "virtual_site",
    "opc": "virtual_site",
    "advanced_water": "virtual_site",
    "advanced_water_model": "virtual_site",
}

REQUIRED_ARRAYS = (
    "symbols",
    "atom_names",
    "atom_types",
    "positions",
    "velocities",
    "masses",
    "charges",
    "sigma",
    "epsilon",
    "bonds",
    "bond_k",
    "bond_length",
    "angles",
    "angle_k",
    "angle_theta",
    "dihedrals",
    "dihedral_k",
    "dihedral_periodicity",
    "dihedral_phase",
)

RB_COEFFICIENT_ARRAYS = ("rb_c0", "rb_c1", "rb_c2", "rb_c3", "rb_c4", "rb_c5")
PME_CONFIG_ARRAYS = (
    "pme_mesh_shape",
    "pme_alpha",
    "pme_real_cutoff",
    "pme_assignment_order",
    "pme_charge_tolerance",
    "pme_deconvolve_assignment",
)


class MLXCompatibilityError(ValueError):
    """Raised when a prepared artifact cannot be run faithfully by MLX."""


@dataclass(frozen=True)
class _ArtifactPositionalRestraintPotential:
    """Import-safe fallback for notebooks with an older loaded forcefields module."""

    reference_positions: object
    mask: object
    k: float
    name: str = "positional_restraint"

    def __post_init__(self) -> None:
        reference = as_mx_array(self.reference_positions)
        mask = np.asarray(self.mask, dtype=bool)
        if reference.ndim != 2 or reference.shape[1] != 3:
            msg = "reference_positions must have shape (n_atoms, 3)"
            raise ValueError(msg)
        if mask.shape != (reference.shape[0],):
            msg = "mask must have shape (n_atoms,)"
            raise ValueError(msg)
        if self.k < 0.0:
            msg = "restraint k must be non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "reference_positions", reference)
        object.__setattr__(self, "mask", as_mx_array(mask.astype(np.float32)))

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        del cell
        positions = as_mx_array(positions)
        displacement = positions - self.reference_positions
        squared = mx.sum(displacement * displacement, axis=-1)
        return 0.5 * self.k * mx.sum(squared * self.mask)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        positions = as_mx_array(positions)
        energy = self.potential_energy(positions, cell)
        forces = -self.k * self.mask[:, None] * (positions - self.reference_positions)
        return energy, forces


@dataclass(frozen=True)
class _ArtifactPairRestrictedNonbondedPotential:
    """Import-safe fallback for explicit-pair nonbonded evaluation."""

    potential: NonbondedPotential
    pairs: object
    name: str = "pair_restricted_nonbonded"

    def __post_init__(self) -> None:
        pairs = np.asarray(self.pairs, dtype=np.int32)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            msg = "pairs must have shape (n, 2)"
            raise ValueError(msg)
        object.__setattr__(self, "pairs", mx.array(pairs, dtype=mx.int32))

    def component_energies(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> dict[str, mx.array]:
        del pairs
        return self.potential.component_energies(positions, cell=cell, pairs=self.pairs)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        del pairs
        return self.potential.energy_forces(positions, cell=cell, pairs=self.pairs)

    def energy_forces_with_components(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
        del pairs
        return self.potential.energy_forces_with_components(
            positions,
            cell=cell,
            pairs=self.pairs,
        )


@dataclass(frozen=True)
class PreparedMLXArtifact:
    """Loaded prepared-system arrays plus validated production metadata."""

    base_dir: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    unit_system: MDUnitSystem | None

    @property
    def atom_count(self) -> int:
        return int(self.arrays["positions"].shape[0])

    @property
    def cell(self) -> Cell | None:
        cell_matrix = np.asarray(self.arrays.get("cell_matrix", np.asarray([])))
        if cell_matrix.size != 0:
            if cell_matrix.shape != (3, 3):
                msg = "cell_matrix must have shape (3, 3)"
                raise ValueError(msg)
            return Cell.triclinic(cell_matrix.astype(np.float32).tolist())
        cell_lengths = np.asarray(self.arrays.get("cell_lengths", np.asarray([])))
        if cell_lengths.size == 0:
            return None
        if cell_lengths.shape != (3,):
            msg = "cell_lengths must have shape (3,)"
            raise ValueError(msg)
        return Cell.orthorhombic(cell_lengths.astype(np.float32).tolist())

    @property
    def hmr_state(self) -> dict[str, Any]:
        """Return serialized HMR state without treating it as a force term."""

        return hmr_state_from_metadata(self.metadata)


def hmr_state_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the read-only HMR state from artifact or checkpoint metadata."""

    report = dict(metadata.get("compatibility_report", {}))
    protocol_metadata = dict(metadata.get("protocol_metadata", {}))
    raw_state = (
        protocol_metadata.get("hydrogen_mass_repartitioning")
        or metadata.get("hydrogen_mass_repartitioning")
        or metadata.get("hmr")
    )
    status = str(report.get("hydrogen_mass_repartitioning") or "").lower()
    if isinstance(raw_state, dict):
        state = dict(raw_state)
        state.setdefault("status", status or "represented_by_masses")
        state.setdefault("provenance_available", True)
        policy = dict(state.get("policy", {}))
        policy.setdefault("virtual_sites_supported", False)
        state["policy"] = policy
        return state
    if status:
        return {
            "status": status,
            "provenance_available": False,
            "policy": {"virtual_sites_supported": False},
        }
    return {
        "status": "absent",
        "provenance_available": False,
        "policy": {"virtual_sites_supported": False},
    }


def _jsonable_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_term(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return TERM_ALIASES.get(normalized, normalized)


def _metadata_with_normalized_report(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized["compatibility_report"] = normalize_compatibility_report(
        dict(metadata.get("compatibility_report", {})),
        source=dict(metadata.get("source", {})),
        parameter_source=str(metadata.get("parameter_source", "")),
        arrays=arrays,
    )
    return normalized


def _metadata_terms(metadata: dict[str, Any]) -> set[str]:
    report = dict(metadata.get("compatibility_report", {}))
    required = {
        _normalize_term(term)
        for term in _jsonable_list(
            report.get("required_terms_normalized", report.get("required_terms"))
        )
    }
    supported = {
        _normalize_term(term)
        for term in _jsonable_list(
            report.get("supported_terms_normalized", report.get("supported_terms"))
        )
    }
    return required or supported


def _metadata_electrostatics_mode(metadata: dict[str, Any]) -> str:
    report = dict(metadata.get("compatibility_report", {}))
    requested_terms = _metadata_terms(metadata)
    raw_mode = (
        metadata.get("electrostatics")
        or metadata.get("electrostatics_model")
        or report.get("electrostatics")
        or report.get("electrostatics_model")
        or (
            "ewald_reference"
            if "ewald_reference_electrostatics" in requested_terms
            else "cutoff"
        )
    )
    try:
        mode = normalize_nonbonded_electrostatics(str(raw_mode))
    except ValueError as err:
        raise MLXCompatibilityError(str(err)) from err
    return mode


def _metadata_pme_config_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    report = dict(metadata.get("compatibility_report", {}))
    payload = metadata.get("pme_config") or report.get("pme_config") or {}
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = "pme_config metadata must be an object"
        raise MLXCompatibilityError(msg)
    return dict(payload)


def _metadata_replica_exchange_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("replica_exchange", {})
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = "replica_exchange metadata must be an object"
        raise MLXCompatibilityError(msg)
    _validate_replica_exchange_metadata_payload(payload, error_cls=MLXCompatibilityError)
    return dict(payload)


def _metadata_lambda_value(
    metadata: dict[str, Any],
    name: str,
    *,
    term: str | None = None,
) -> float:
    protocol = dict(metadata.get("protocol_metadata", {}))
    term_metadata = dict(metadata.get(term, {})) if term is not None else {}
    protocol_term_metadata = dict(protocol.get(term, {})) if term is not None else {}
    lambda_metadata = dict(metadata.get("lambda_scaled_nonbonded", {}))
    soft_core_metadata = dict(metadata.get("soft_core_lj", {}))
    protocol_lambda_metadata = dict(protocol.get("lambda_scaled_nonbonded", {}))
    protocol_soft_core_metadata = dict(protocol.get("soft_core_lj", {}))
    value = (
        metadata.get(name)
        if name in metadata
        else term_metadata.get(
            name,
            protocol_term_metadata.get(
                name,
                lambda_metadata.get(
                    name,
                    protocol_lambda_metadata.get(
                        name,
                        soft_core_metadata.get(name, protocol_soft_core_metadata.get(name, 1.0)),
                    ),
                ),
            ),
        )
    )
    try:
        lambda_value = float(value)
    except (TypeError, ValueError) as err:
        msg = f"{name} metadata must be a finite value in [0, 1]"
        raise MLXCompatibilityError(msg) from err
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        msg = f"{name} metadata must be a finite value in [0, 1]"
        raise MLXCompatibilityError(msg)
    return lambda_value


def _array_present(arrays: dict[str, np.ndarray] | None, name: str) -> bool:
    return arrays is not None and name in arrays and np.asarray(arrays[name]).size != 0


def _pme_arrays_present(arrays: dict[str, np.ndarray] | None) -> bool:
    return any(_array_present(arrays, name) for name in PME_CONFIG_ARRAYS)


def _pme_scalar_from_arrays(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    dtype: type = float,
) -> Any:
    values = np.asarray(arrays.get(name, np.asarray([])))
    if values.shape != (1,):
        msg = f"{name} must have shape (1,) for pme artifacts"
        raise MLXCompatibilityError(msg)
    return dtype(values[0])


def _validate_pme_config_values(
    *,
    mesh_shape: object,
    alpha: float,
    real_cutoff: float | None,
    assignment_order: object,
    charge_tolerance: float,
) -> tuple[tuple[int, int, int], int]:
    try:
        mesh_values = np.asarray(mesh_shape, dtype=np.float64)
    except (TypeError, ValueError) as err:
        msg = "pme artifacts require pme_mesh_shape dimensions to be finite integers >= 4"
        raise MLXCompatibilityError(msg) from err
    if (
        mesh_values.shape != (3,)
        or not np.all(np.isfinite(mesh_values))
        or not np.all(mesh_values == np.floor(mesh_values))
        or np.any(mesh_values < 4)
    ):
        msg = "pme artifacts require pme_mesh_shape dimensions to be integers >= 4"
        raise MLXCompatibilityError(msg)
    if not np.isfinite(alpha) or alpha <= 0.0:
        msg = "pme artifacts require finite positive pme_alpha"
        raise MLXCompatibilityError(msg)
    if real_cutoff is not None and (not np.isfinite(real_cutoff) or real_cutoff <= 0.0):
        msg = "pme artifacts require finite positive pme_real_cutoff when provided"
        raise MLXCompatibilityError(msg)
    try:
        assignment_order_value = float(assignment_order)
    except (TypeError, ValueError) as err:
        msg = "pme artifacts require pme_assignment_order to be one of 2, 4, or 5"
        raise MLXCompatibilityError(msg) from err
    if (
        not np.isfinite(assignment_order_value)
        or assignment_order_value != np.floor(assignment_order_value)
        or int(assignment_order_value) not in PME_SUPPORTED_ASSIGNMENT_ORDERS
    ):
        msg = "pme artifacts require pme_assignment_order to be one of 2, 4, or 5"
        raise MLXCompatibilityError(msg)
    if not np.isfinite(charge_tolerance) or charge_tolerance < 0.0:
        msg = "pme artifacts require finite non-negative pme_charge_tolerance"
        raise MLXCompatibilityError(msg)
    return tuple(int(item) for item in mesh_values.tolist()), int(assignment_order_value)


def _pme_config_from_artifact(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
    *,
    required: bool = False,
) -> PMEConfig | None:
    has_array_config = _pme_arrays_present(arrays)
    if arrays is not None and has_array_config:
        mesh_shape = np.asarray(arrays.get("pme_mesh_shape", np.asarray([])))
        if mesh_shape.shape != (3,):
            msg = "pme artifacts must include pme_mesh_shape with shape (3,)"
            raise MLXCompatibilityError(msg)
        try:
            alpha = float(_pme_scalar_from_arrays(arrays, "pme_alpha"))
            real_cutoff = float(_pme_scalar_from_arrays(arrays, "pme_real_cutoff"))
            assignment_order = float(_pme_scalar_from_arrays(arrays, "pme_assignment_order"))
            charge_tolerance = float(_pme_scalar_from_arrays(arrays, "pme_charge_tolerance"))
            mesh_shape_tuple, assignment_order = _validate_pme_config_values(
                mesh_shape=mesh_shape,
                alpha=alpha,
                real_cutoff=real_cutoff,
                assignment_order=assignment_order,
                charge_tolerance=charge_tolerance,
            )
            return PMEConfig(
                mesh_shape=mesh_shape_tuple,
                alpha=alpha,
                real_cutoff=real_cutoff,
                assignment_order=assignment_order,
                charge_tolerance=charge_tolerance,
                deconvolve_assignment=bool(
                    _pme_scalar_from_arrays(
                        arrays,
                        "pme_deconvolve_assignment",
                        dtype=bool,
                    )
                ),
            )
        except ValueError as err:
            raise MLXCompatibilityError(str(err)) from err

    payload = _metadata_pme_config_payload(metadata)
    if payload:
        try:
            alpha = float(payload.get("alpha", 0.35))
            real_cutoff = (
                None
                if payload.get("real_cutoff") is None
                else float(payload.get("real_cutoff"))
            )
            assignment_order = float(payload.get("assignment_order", 2))
            charge_tolerance = float(payload.get("charge_tolerance", 1e-5))
            mesh_shape_tuple, assignment_order = _validate_pme_config_values(
                mesh_shape=payload.get("mesh_shape", (32, 32, 32)),
                alpha=alpha,
                real_cutoff=real_cutoff,
                assignment_order=assignment_order,
                charge_tolerance=charge_tolerance,
            )
            return PMEConfig(
                mesh_shape=mesh_shape_tuple,
                alpha=alpha,
                real_cutoff=real_cutoff,
                assignment_order=assignment_order,
                charge_tolerance=charge_tolerance,
                deconvolve_assignment=bool(payload.get("deconvolve_assignment", True)),
            )
        except (TypeError, ValueError) as err:
            raise MLXCompatibilityError(f"invalid pme_config: {err}") from err

    if required:
        msg = "pme artifacts must include pme_config or pme_* arrays"
        raise MLXCompatibilityError(msg)
    return None


def _improper_dihedral_potential_cls():
    from mlx_atomistic import forcefields

    return getattr(forcefields, "ImproperDihedralPotential", PeriodicDihedralPotential)


def _positional_restraint_potential_cls():
    from mlx_atomistic import forcefields

    return getattr(
        forcefields,
        "PositionalRestraintPotential",
        _ArtifactPositionalRestraintPotential,
    )


def _pair_restricted_nonbonded_potential_cls():
    from mlx_atomistic import forcefields

    return getattr(
        forcefields,
        "PairRestrictedNonbondedPotential",
        _ArtifactPairRestrictedNonbondedPotential,
    )


def _md_unit_system_cls():
    units = importlib.import_module("mlx_atomistic.units")
    cls = getattr(units, "MDUnitSystem", None)
    if cls is None:
        msg = (
            "mlx_atomistic.units is loaded without MDUnitSystem; reload the module "
            "or restart the notebook kernel"
        )
        raise MLXCompatibilityError(msg)
    return cls


def _coulomb_constant_kj_mol_angstrom() -> float:
    units = importlib.import_module("mlx_atomistic.units")
    return float(
        getattr(
            units,
            "COULOMB_CONSTANT_KJ_MOL_ANGSTROM",
            DEFAULT_COULOMB_CONSTANT_KJ_MOL_ANGSTROM,
        )
    )


def validate_mlx_compatibility(
    metadata: dict[str, Any],
    *,
    require_production: bool = False,
    arrays: dict[str, np.ndarray] | None = None,
) -> MDUnitSystem | None:
    """Validate fail-closed compatibility metadata for MLX execution."""

    metadata = _metadata_with_normalized_report(metadata, arrays)
    report = dict(metadata.get("compatibility_report", {}))
    unsupported = {
        _normalize_term(term)
        for term in _jsonable_list(
            report.get("unsupported_terms_normalized", report.get("unsupported_terms"))
        )
    }
    unsupported |= {
        _normalize_term(term)
        for term in _jsonable_list(
            report.get("rejected_terms_normalized", report.get("rejected_terms"))
        )
    }
    requested_terms = _metadata_terms(metadata)
    _validate_declared_term_arrays(metadata, requested_terms=requested_terms, arrays=arrays)
    if arrays is not None:
        _validate_term_count_metadata(metadata, arrays)

    blockers = sorted((unsupported | requested_terms) & FAIL_CLOSED_TERMS)
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        msg = f"prepared artifact declares unsupported force-field terms: {joined}"
        raise MLXCompatibilityError(msg)
    if blockers:
        joined = ", ".join(blockers)
        msg = f"prepared artifact requires unsupported production terms: {joined}"
        raise MLXCompatibilityError(msg)
    declared_blockers = sorted(str(item) for item in _jsonable_list(report.get("blockers")))
    if declared_blockers:
        joined = ", ".join(declared_blockers)
        msg = f"prepared artifact declares blockers: {joined}"
        raise MLXCompatibilityError(msg)

    unknown = sorted(term for term in requested_terms if term and term not in SUPPORTED_FORCE_TERMS)
    if unknown:
        joined = ", ".join(unknown)
        msg = f"prepared artifact requires terms not implemented in mlx_atomistic: {joined}"
        raise MLXCompatibilityError(msg)

    _metadata_replica_exchange_payload(metadata)

    electrostatics_mode = _metadata_electrostatics_mode(metadata)
    if electrostatics_mode == "pme" or "pme" in requested_terms:
        _pme_config_from_artifact(metadata, arrays, required=True)

    if require_production and not bool(report.get("production_force_field", False)):
        msg = "prepared artifact is not marked as a production force-field export"
        raise MLXCompatibilityError(msg)

    if (
        require_production
        or bool(report.get("production_force_field", False))
        or bool(report.get("physical_units", False))
    ):
        unit_system_cls = _md_unit_system_cls()
        return unit_system_cls.from_metadata(dict(metadata.get("units", {})))
    return None


def artifact_readiness_report(
    metadata: dict[str, Any],
    *,
    require_production: bool = False,
    arrays: dict[str, np.ndarray] | None = None,
) -> ReadinessReport:
    """Return the artifact compatibility gate as a shared readiness report."""

    metadata = _metadata_with_normalized_report(metadata, arrays)
    report = dict(metadata.get("compatibility_report", {}))
    readiness_metadata = {
        "require_production": bool(require_production),
        "production_force_field": bool(report.get("production_force_field", False)),
        "physical_units": bool(report.get("physical_units", False)),
        "electrostatics_model": _metadata_electrostatics_mode(metadata),
        "required_terms": sorted(_metadata_terms(metadata)),
    }
    try:
        validate_mlx_compatibility(
            metadata,
            require_production=require_production,
            arrays=arrays,
        )
    except MLXCompatibilityError as exc:
        return ReadinessReport(
            name="artifact",
            status="blocked",
            blockers=(f"artifact:{exc}",),
            metadata=readiness_metadata,
        )
    status = "ready" if readiness_metadata["production_force_field"] else "proof-level"
    if require_production:
        status = "ready"
    return ReadinessReport(
        name="artifact",
        status=status,
        metadata=readiness_metadata,
    )


def _validate_declared_term_arrays(
    metadata: dict[str, Any],
    *,
    requested_terms: set[str],
    arrays: dict[str, np.ndarray] | None,
) -> None:
    if arrays is None:
        return
    report = dict(metadata.get("compatibility_report", {}))
    declared_terms = set(requested_terms)
    declared_terms |= {
        _normalize_term(term) for term in _jsonable_list(report.get("unsupported_terms"))
    }
    declared_terms |= {
        _normalize_term(term) for term in _jsonable_list(report.get("rejected_terms"))
    }
    array_terms = {
        "charmm_cmap_terms": "charmm_cmap",
        "rb_dihedrals": "rb_dihedral",
        "rb_c0": "rb_dihedral",
        "rb_c1": "rb_dihedral",
        "rb_c2": "rb_dihedral",
        "rb_c3": "rb_dihedral",
        "rb_c4": "rb_dihedral",
        "rb_c5": "rb_dihedral",
        "urey_bradley_terms": "urey_bradley",
        "nbfix_pairs": "nbfix_pair_overrides",
        "nbfix_type_pairs": "nbfix_pair_overrides",
        "nbfix_type_sigma": "nbfix_pair_overrides",
        "nbfix_type_epsilon": "nbfix_pair_overrides",
        "custom_force_indices": "custom_force",
        "gbsa_radius": "gbsa",
        "gbsa_scale": "gbsa",
        "virtual_site_parent_atoms": "virtual_site",
        "virtual_site_weights": "virtual_site",
        "virtual_site_types": "virtual_site",
    }
    hidden = sorted(
        {
            term_name
            for array_name, term_name in array_terms.items()
            if _array_present(arrays, array_name) and term_name not in declared_terms
        }
    )
    if hidden:
        joined = ", ".join(hidden)
        msg = f"prepared artifact contains undeclared force-field arrays: {joined}"
        raise MLXCompatibilityError(msg)


def _optional_string_pairs(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    array = np.asarray(arrays.get(name, np.empty((0, 2), dtype=str)))
    if array.size == 0:
        return np.empty((0, 2), dtype=str)
    if array.ndim != 2 or array.shape[1] != 2:
        msg = f"{name} must have shape (n, 2)"
        raise ValueError(msg)
    return array.astype(str, copy=False)


def _required_string_pairs(
    arrays: dict[str, np.ndarray],
    name: str,
    term_name: str,
) -> np.ndarray:
    try:
        values = _optional_string_pairs(arrays, name)
    except ValueError as err:
        raise MLXCompatibilityError(str(err)) from err
    if values.shape[0] == 0:
        msg = f"{term_name} artifacts must include {name}"
        raise MLXCompatibilityError(msg)
    return values


def _load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_ARRAYS if name not in data]
        if missing:
            msg = f"prepared artifact is missing arrays: {', '.join(missing)}"
            raise ValueError(msg)
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return arrays


def _optional_indices(arrays: dict[str, np.ndarray], name: str, width: int) -> np.ndarray:
    array = np.asarray(arrays.get(name, np.empty((0, width), dtype=np.int32)), dtype=np.int32)
    if array.size == 0:
        return np.empty((0, width), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != width:
        msg = f"{name} must have shape (n, {width})"
        raise ValueError(msg)
    return array


def _optional_vector(arrays: dict[str, np.ndarray], name: str, count: int) -> np.ndarray:
    array = np.asarray(arrays.get(name, np.empty((0,), dtype=np.float32)), dtype=np.float32)
    if count == 0 and array.size == 0:
        return np.empty((0,), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must have shape ({count},)"
        raise ValueError(msg)
    return array


def load_prepared_mlx_artifact(
    path: str | Path,
    *,
    require_production: bool = False,
) -> PreparedMLXArtifact:
    """Load and validate a prepared MLX artifact directory or JSON path."""

    input_path = Path(path)
    base_dir = input_path.parent if input_path.is_file() else input_path
    json_path = input_path if input_path.is_file() else base_dir / PREPARED_JSON_NAME
    npz_path = base_dir / PREPARED_NPZ_NAME
    if not json_path.exists():
        msg = f"missing prepared artifact metadata: {json_path}"
        raise FileNotFoundError(msg)
    if not npz_path.exists():
        msg = f"missing prepared artifact arrays: {npz_path}"
        raise FileNotFoundError(msg)

    arrays = _load_arrays(npz_path)
    metadata = _metadata_with_normalized_report(json.loads(json_path.read_text()), arrays)
    unit_system = validate_mlx_compatibility(
        metadata,
        require_production=require_production,
        arrays=arrays,
    )
    _validate_core_shapes(arrays)
    _validate_electrostatics_arrays(metadata, arrays)
    _validate_requested_term_arrays(metadata, arrays)
    if require_production or bool(
        dict(metadata.get("compatibility_report", {})).get("production_force_field", False)
    ):
        _validate_production_arrays(metadata, arrays)
    return PreparedMLXArtifact(
        base_dir=base_dir,
        metadata=metadata,
        arrays=arrays,
        unit_system=unit_system,
    )


def _validate_core_shapes(arrays: dict[str, np.ndarray]) -> None:
    n_atoms = int(np.asarray(arrays["positions"]).shape[0])
    if n_atoms <= 0:
        msg = "prepared artifact must contain at least one atom"
        raise ValueError(msg)
    for name in ["symbols", "atom_names", "atom_types", "masses", "charges", "sigma", "epsilon"]:
        if np.asarray(arrays[name]).shape != (n_atoms,):
            msg = f"{name} must have shape ({n_atoms},)"
            raise ValueError(msg)
    for name in ["positions", "velocities"]:
        if np.asarray(arrays[name]).shape != (n_atoms, 3):
            msg = f"{name} must have shape ({n_atoms}, 3)"
            raise ValueError(msg)
    for name in [
        "ligand_mask",
        "receptor_mask",
        "water_mask",
        "ion_mask",
        "lipid_mask",
        "restraint_mask",
    ]:
        if (
            name in arrays
            and np.asarray(arrays[name]).size
            and np.asarray(arrays[name]).shape != (n_atoms,)
        ):
            msg = f"{name} must have shape ({n_atoms},)"
            raise ValueError(msg)


def _validate_production_arrays(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    report = dict(metadata.get("compatibility_report", {}))
    symbols = np.char.upper(np.asarray(arrays["symbols"], dtype=str))
    hydrogen_count = int(np.count_nonzero(symbols == "H"))
    if hydrogen_count <= 0:
        msg = "production artifact must contain explicit hydrogen atoms"
        raise MLXCompatibilityError(msg)
    if not bool(report.get("hydrogens_present", False)):
        msg = "production artifact metadata must declare hydrogens_present=true"
        raise MLXCompatibilityError(msg)
    metadata_hydrogen_count = int(report.get("hydrogen_count", hydrogen_count))
    if metadata_hydrogen_count != hydrogen_count:
        msg = (
            "production artifact hydrogen_count metadata does not match arrays: "
            f"metadata={metadata_hydrogen_count}, arrays={hydrogen_count}"
        )
        raise MLXCompatibilityError(msg)
    parameter_source = str(metadata.get("parameter_source", "")).lower()
    if not parameter_source or "generic" in parameter_source or "demo" in parameter_source:
        msg = "production artifact must declare a non-demo parameter_source"
        raise MLXCompatibilityError(msg)
    for name in ["positions", "velocities", "masses", "charges", "sigma", "epsilon"]:
        values = np.asarray(arrays[name])
        if not np.all(np.isfinite(values)):
            msg = f"production artifact array {name} contains non-finite values"
            raise MLXCompatibilityError(msg)
    masses = np.asarray(arrays["masses"], dtype=np.float32)
    if "virtual_site" in _metadata_terms(metadata):
        if np.any(masses < 0.0) or not np.any(masses > 0.0):
            msg = "production artifact masses must be non-negative with positive real atoms"
            raise MLXCompatibilityError(msg)
    elif np.any(masses <= 0.0):
        msg = "production artifact masses must be positive"
        raise MLXCompatibilityError(msg)
    _validate_hmr_and_virtual_site_policy(metadata, arrays, symbols)


def _validate_hmr_and_virtual_site_policy(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    symbols: np.ndarray,
) -> None:
    report = dict(metadata.get("compatibility_report", {}))
    water_model = str(
        metadata.get("water_model")
        or report.get("water_model")
        or report.get("solvent_model")
        or ""
    ).lower()
    if bool(report.get("virtual_sites_present", False)) and "virtual_site" not in _metadata_terms(
        metadata
    ):
        msg = "virtual_site artifacts are not supported by mlx_atomistic"
        raise MLXCompatibilityError(msg)
    if any(model in water_model for model in ("tip5p", "opc", "advanced_water")):
        msg = f"virtual_site water model is not supported: {water_model}"
        raise MLXCompatibilityError(msg)
    if "tip4p" in water_model and "virtual_site" not in _metadata_terms(metadata):
        msg = f"virtual_site water model requires virtual_site metadata: {water_model}"
        raise MLXCompatibilityError(msg)

    masses = np.asarray(arrays["masses"], dtype=np.float32)
    hydrogen_masses = masses[symbols == "H"]
    if hydrogen_masses.size == 0:
        return
    hmr_status = str(report.get("hydrogen_mass_repartitioning", "")).lower()
    hmr_represented = hmr_status in {
        "represented_by_masses",
        "static_masses",
        "present_represented_by_masses",
        "transformed_by_mlx_atomistic",
    }
    hidden_hmr = bool(np.any(hydrogen_masses > 1.25))
    if hidden_hmr and not hmr_represented:
        msg = (
            "hydrogen_mass_repartitioning detected from hydrogen masses but not "
            "declared as represented_by_masses"
        )
        raise MLXCompatibilityError(msg)
    if hmr_represented and "distance_constraint" not in _metadata_terms(metadata):
        msg = "hydrogen_mass_repartitioning production artifacts require distance_constraint"
        raise MLXCompatibilityError(msg)
    hmr_state = hmr_state_from_metadata(metadata)
    if bool(dict(hmr_state.get("policy", {})).get("virtual_sites_supported", False)):
        msg = "hydrogen_mass_repartitioning metadata must not claim virtual-site support"
        raise MLXCompatibilityError(msg)
    if bool(hmr_state.get("provenance_available", False)):
        _validate_hmr_provenance(metadata, arrays, symbols, hmr_state)


def _validate_hmr_provenance(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    symbols: np.ndarray,
    hmr_state: dict[str, Any],
) -> None:
    masses = np.asarray(arrays["masses"], dtype=np.float64)
    transformed = np.asarray(hmr_state.get("transformed_masses", []), dtype=np.float64)
    original = np.asarray(hmr_state.get("original_masses", []), dtype=np.float64)
    if original.shape != masses.shape or transformed.shape != masses.shape:
        msg = "hydrogen_mass_repartitioning provenance masses must match artifact masses"
        raise MLXCompatibilityError(msg)
    if not np.allclose(transformed, masses, rtol=1e-6, atol=1e-6):
        msg = "hydrogen_mass_repartitioning transformed masses do not match artifact masses"
        raise MLXCompatibilityError(msg)
    if not np.isclose(
        float(np.sum(original, dtype=np.float64)),
        float(np.sum(transformed, dtype=np.float64)),
        rtol=0.0,
        atol=1e-6,
    ):
        msg = "hydrogen_mass_repartitioning provenance does not preserve total mass"
        raise MLXCompatibilityError(msg)
    selected = hmr_state.get("selected_hydrogens", [])
    if not isinstance(selected, list) or not selected:
        msg = "hydrogen_mass_repartitioning provenance must list selected_hydrogens"
        raise MLXCompatibilityError(msg)
    for record in selected:
        if not isinstance(record, dict):
            msg = "hydrogen_mass_repartitioning selected_hydrogens entries must be objects"
            raise MLXCompatibilityError(msg)
        hydrogen_index = int(record.get("hydrogen_index", -1))
        if (
            hydrogen_index < 0
            or hydrogen_index >= masses.shape[0]
            or symbols[hydrogen_index] != "H"
        ):
            msg = "hydrogen_mass_repartitioning selected_hydrogens must reference hydrogens"
            raise MLXCompatibilityError(msg)


def _validate_electrostatics_arrays(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    mode = _metadata_electrostatics_mode(metadata)
    if mode not in {"ewald_reference", "pme"}:
        return
    cell = _cell_from_artifact_arrays(arrays, mode=mode)
    if not cell.is_orthorhombic:
        msg = (
            f"{mode} artifacts require an orthorhombic cell; "
            "triclinic cell_matrix is not supported by this electrostatics path"
        )
        raise MLXCompatibilityError(msg)
    charges = np.asarray(arrays["charges"], dtype=np.float32)
    if not np.all(np.isfinite(charges)):
        msg = f"{mode} artifacts must include finite charges"
        raise MLXCompatibilityError(msg)
    pme_config = None
    charge_tolerance = 1e-5
    if mode == "pme":
        pme_config = _pme_config_from_artifact(metadata, arrays, required=True)
        charge_tolerance = float(pme_config.charge_tolerance)
    net_charge = float(np.sum(charges, dtype=np.float64))
    if abs(net_charge) > charge_tolerance:
        msg = f"{mode} artifacts must be neutral; net_charge={net_charge:g}"
        raise MLXCompatibilityError(msg)


def _cell_from_artifact_arrays(arrays: dict[str, np.ndarray], *, mode: str) -> Cell:
    cell_matrix = np.asarray(arrays.get("cell_matrix", np.asarray([])), dtype=np.float32)
    if cell_matrix.size != 0:
        if cell_matrix.shape != (3, 3) or not np.all(np.isfinite(cell_matrix)):
            msg = f"{mode} artifacts must include finite cell_matrix with shape (3, 3)"
            raise MLXCompatibilityError(msg)
        try:
            return Cell.triclinic(cell_matrix.tolist())
        except ValueError as err:
            raise MLXCompatibilityError(str(err)) from err
    cell_lengths = np.asarray(arrays.get("cell_lengths", np.asarray([])), dtype=np.float32)
    if cell_lengths.shape != (3,) or not np.all(np.isfinite(cell_lengths)):
        msg = f"{mode} artifacts must include finite cell_lengths with shape (3,)"
        raise MLXCompatibilityError(msg)
    if np.any(cell_lengths <= 0.0):
        msg = f"{mode} artifacts must include positive cell_lengths"
        raise MLXCompatibilityError(msg)
    return Cell.orthorhombic(cell_lengths.astype(np.float32).tolist())


def _required_indices(
    arrays: dict[str, np.ndarray],
    name: str,
    width: int,
    term_name: str,
) -> np.ndarray:
    try:
        values = _optional_indices(arrays, name, width)
    except ValueError as err:
        raise MLXCompatibilityError(str(err)) from err
    if values.shape[0] == 0:
        msg = f"{term_name} artifacts must include {name}"
        raise MLXCompatibilityError(msg)
    return values


def _required_vector(
    arrays: dict[str, np.ndarray],
    name: str,
    count: int,
    term_name: str,
) -> np.ndarray:
    try:
        values = _optional_vector(arrays, name, count)
    except ValueError as err:
        raise MLXCompatibilityError(str(err)) from err
    if values.shape != (count,):
        msg = f"{term_name} artifacts must include {name} with shape ({count},)"
        raise MLXCompatibilityError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{term_name} artifacts must include finite {name}"
        raise MLXCompatibilityError(msg)
    return values


def _rb_arrays_present(arrays: dict[str, np.ndarray]) -> bool:
    return _array_present(arrays, "rb_dihedrals") or any(
        _array_present(arrays, name) for name in RB_COEFFICIENT_ARRAYS
    )


def _validate_rb_arrays(arrays: dict[str, np.ndarray], n_atoms: int) -> np.ndarray:
    rb_dihedrals = _required_indices(arrays, "rb_dihedrals", 4, "rb_dihedral")
    if np.any(rb_dihedrals < 0) or np.any(rb_dihedrals >= n_atoms):
        msg = "rb_dihedrals contain atom indices outside [0, atom_count)"
        raise MLXCompatibilityError(msg)
    for name in RB_COEFFICIENT_ARRAYS:
        _required_vector(arrays, name, rb_dihedrals.shape[0], "rb_dihedral")
    return rb_dihedrals


def _required_mask(arrays: dict[str, np.ndarray], name: str, n_atoms: int, term_name: str) -> None:
    values = np.asarray(arrays.get(name, np.asarray([])), dtype=bool)
    if values.shape != (n_atoms,):
        msg = f"{term_name} artifacts must include {name} with shape ({n_atoms},)"
        raise MLXCompatibilityError(msg)
    if not np.any(values):
        msg = f"{term_name} artifacts must include at least one true value in {name}"
        raise MLXCompatibilityError(msg)


def _validate_cmap_arrays(arrays: dict[str, np.ndarray]) -> None:
    terms = _required_indices(arrays, "charmm_cmap_terms", 8, "charmm_cmap")
    grids = np.asarray(arrays.get("charmm_cmap_grids", np.asarray([])), dtype=np.float32)
    if grids.ndim != 3 or grids.shape[1] != grids.shape[2]:
        msg = "charmm_cmap artifacts must include charmm_cmap_grids with shape (n_maps, grid, grid)"
        raise MLXCompatibilityError(msg)
    if grids.shape[1] < 4:
        msg = "charmm_cmap artifacts must use at least a 4x4 periodic grid"
        raise MLXCompatibilityError(msg)
    if not np.all(np.isfinite(grids)):
        msg = "charmm_cmap artifacts must include finite charmm_cmap_grids"
        raise MLXCompatibilityError(msg)
    indices = np.asarray(arrays.get("charmm_cmap_grid_indices", np.asarray([])), dtype=np.int32)
    if indices.shape != (terms.shape[0],):
        msg = f"charmm_cmap_grid_indices must have shape ({terms.shape[0]},)"
        raise MLXCompatibilityError(msg)
    if np.any(indices < 0) or np.any(indices >= grids.shape[0]):
        msg = "charmm_cmap_grid_indices contain indices outside charmm_cmap_grids"
        raise MLXCompatibilityError(msg)


def _validate_requested_term_arrays(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    requested_terms = _metadata_terms(metadata)
    n_atoms = int(np.asarray(arrays["positions"]).shape[0])
    if "rb_dihedral" in requested_terms:
        _validate_rb_arrays(arrays, n_atoms)
    if "charmm_cmap" in requested_terms:
        _validate_cmap_arrays(arrays)
    if "urey_bradley" in requested_terms:
        terms = _required_indices(arrays, "urey_bradley_terms", 3, "urey_bradley")
        _required_vector(arrays, "urey_bradley_k", terms.shape[0], "urey_bradley")
        _required_vector(arrays, "urey_bradley_distance", terms.shape[0], "urey_bradley")
    if "nbfix_pair_overrides" in requested_terms:
        has_legacy_pairs = _array_present(arrays, "nbfix_pairs")
        has_type_pairs = any(
            _array_present(arrays, name)
            for name in ("nbfix_type_pairs", "nbfix_type_sigma", "nbfix_type_epsilon")
        )
        if has_legacy_pairs:
            pairs = _required_indices(arrays, "nbfix_pairs", 2, "nbfix_pair_overrides")
            _required_vector(arrays, "nbfix_sigma", pairs.shape[0], "nbfix_pair_overrides")
            _required_vector(arrays, "nbfix_epsilon", pairs.shape[0], "nbfix_pair_overrides")
        elif has_type_pairs:
            pairs = _required_string_pairs(
                arrays,
                "nbfix_type_pairs",
                "nbfix_pair_overrides",
            )
            _required_vector(
                arrays,
                "nbfix_type_sigma",
                pairs.shape[0],
                "nbfix_pair_overrides",
            )
            _required_vector(
                arrays,
                "nbfix_type_epsilon",
                pairs.shape[0],
                "nbfix_pair_overrides",
            )
            atom_types = np.asarray(arrays.get("atom_types", np.asarray([])), dtype=str)
            if atom_types.shape != (n_atoms,):
                msg = "nbfix_pair_overrides artifacts require atom_types for every atom"
                raise MLXCompatibilityError(msg)
        else:
            msg = (
                "nbfix_pair_overrides artifacts must include "
                "nbfix_pairs or nbfix_type_pairs"
            )
            raise MLXCompatibilityError(msg)
    if "distance_constraint" in requested_terms:
        constraints = _required_indices(arrays, "constraints", 2, "distance_constraint")
        _required_vector(arrays, "constraint_distance", constraints.shape[0], "distance_constraint")
    if "virtual_site" in requested_terms:
        _validate_virtual_site_arrays(arrays)
    if "nonbonded_exception" in requested_terms:
        exceptions = _required_indices(
            arrays,
            "nonbonded_exception_pairs",
            2,
            "nonbonded_exception",
        )
        _required_vector(
            arrays,
            "nonbonded_exception_charge_product",
            exceptions.shape[0],
            "nonbonded_exception",
        )
        _required_vector(
            arrays,
            "nonbonded_exception_sigma",
            exceptions.shape[0],
            "nonbonded_exception",
        )
        _required_vector(
            arrays,
            "nonbonded_exception_epsilon",
            exceptions.shape[0],
            "nonbonded_exception",
        )
    if "custom_force" in requested_terms:
        _required_indices(arrays, "custom_force_indices", 2, "custom_force")
    if "gbsa" in requested_terms:
        _required_vector(arrays, "gbsa_radius", n_atoms, "gbsa")
        _required_vector(arrays, "gbsa_scale", n_atoms, "gbsa")
    for term_name, mask_name in [
        ("lipid", "lipid_mask"),
        ("water", "water_mask"),
        ("ion", "ion_mask"),
        ("receptor", "receptor_mask"),
        ("ligand", "ligand_mask"),
    ]:
        if term_name in requested_terms:
            _required_mask(arrays, mask_name, n_atoms, term_name)


def _validate_virtual_site_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = [
        name
        for name in ("virtual_site_parent_atoms", "virtual_site_weights", "virtual_site_types")
        if name not in arrays
    ]
    if missing:
        msg = f"virtual_site artifacts must include arrays: {', '.join(missing)}"
        raise MLXCompatibilityError(msg)
    parent_atoms = np.asarray(arrays["virtual_site_parent_atoms"], dtype=np.int32)
    weights = np.asarray(arrays["virtual_site_weights"], dtype=np.float32)
    types = np.asarray(arrays["virtual_site_types"], dtype=str)
    n_atoms = int(np.asarray(arrays["positions"]).shape[0])
    if parent_atoms.size == 0:
        msg = "virtual_site artifacts must include at least one virtual site"
        raise MLXCompatibilityError(msg)
    if parent_atoms.ndim != 2 or parent_atoms.shape[1] != 4:
        msg = "virtual_site_parent_atoms must have shape (n, 4)"
        raise MLXCompatibilityError(msg)
    if weights.shape != parent_atoms.shape:
        msg = f"virtual_site_weights must have shape {parent_atoms.shape}"
        raise MLXCompatibilityError(msg)
    if not np.all(np.isfinite(weights)):
        msg = "virtual_site_weights must be finite"
        raise MLXCompatibilityError(msg)
    if types.shape != (parent_atoms.shape[0],):
        msg = f"virtual_site_types must have shape ({parent_atoms.shape[0]},)"
        raise MLXCompatibilityError(msg)
    if np.any(parent_atoms < 0) or np.any(parent_atoms >= n_atoms):
        msg = "virtual_site_parent_atoms contain atom indices outside [0, atom_count)"
        raise MLXCompatibilityError(msg)
    supported = {"tip4p", "tip4p_ew", "tip4pew"}
    site_indices: list[int] = []
    for parents, site_type in zip(parent_atoms, types, strict=True):
        normalized_type = str(site_type).strip().lower().replace("-", "_")
        if normalized_type not in supported:
            msg = f"unsupported virtual_site type: {site_type}"
            raise MLXCompatibilityError(msg)
        if len(set(int(item) for item in parents.tolist())) != 4:
            msg = "tip4p virtual sites require three distinct parents and one distinct site atom"
            raise MLXCompatibilityError(msg)
        site_indices.append(int(parents[3]))
    if len(set(site_indices)) != len(site_indices):
        msg = "virtual_site atom indices must be unique"
        raise MLXCompatibilityError(msg)


def _validate_term_count_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    report = dict(metadata.get("compatibility_report", {}))
    declared = {
        str(key): int(value)
        for key, value in dict(report.get("term_counts_normalized", {})).items()
    }
    actual = {
        str(key): int(value)
        for key, value in dict(
            report.get(
                "array_term_counts",
                normalize_compatibility_report({}, arrays=arrays).get(
                    "array_term_counts",
                    {},
                ),
            )
        ).items()
    }
    mismatches = [
        f"{term}:metadata={declared[term]}:arrays={actual_count}"
        for term, actual_count in sorted(actual.items())
        if term in declared and declared[term] != actual_count
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        msg = f"prepared artifact term_counts metadata does not match arrays: {joined}"
        raise MLXCompatibilityError(msg)


def build_mlx_system_from_artifact(
    artifact: PreparedMLXArtifact,
    *,
    restraint_k: float = 0.0,
    constraint_max_iterations: int = 20,
    eager_nonbonded_pair_limit: int | None = DEFAULT_EAGER_NONBONDED_PAIR_LIMIT,
) -> tuple[MMSystem, list, DistanceConstraints | None]:
    """Convert a prepared artifact into `MMSystem`, force terms, and constraints."""

    arrays = dict(artifact.arrays)
    requested_terms = _metadata_terms(artifact.metadata)
    layout = _virtual_site_layout(arrays) if "virtual_site" in requested_terms else None
    if layout is not None:
        arrays = _reordered_virtual_site_arrays(arrays, layout)
    n_atoms = int(np.asarray(arrays["positions"]).shape[0])
    impropers = _optional_indices(arrays, "impropers", 4)
    _validate_declared_term_arrays(
        artifact.metadata,
        requested_terms=requested_terms,
        arrays=arrays,
    )
    rb_dihedrals = (
        _validate_rb_arrays(arrays, n_atoms)
        if "rb_dihedral" in requested_terms
        else np.empty((0, 4), dtype=np.int32)
    )
    exception_pairs = _optional_indices(arrays, "nonbonded_exception_pairs", 2)
    nonbonded_pairs = _optional_indices(arrays, "nonbonded_pairs", 2)
    constraints = _optional_indices(arrays, "constraints", 2)
    constraint_distances = _optional_vector(arrays, "constraint_distance", constraints.shape[0])
    charge_products = _optional_vector(
        arrays,
        "nonbonded_exception_charge_product",
        exception_pairs.shape[0],
    )
    exception_sigma = _optional_vector(
        arrays,
        "nonbonded_exception_sigma",
        exception_pairs.shape[0],
    )
    exception_epsilon = _optional_vector(
        arrays,
        "nonbonded_exception_epsilon",
        exception_pairs.shape[0],
    )
    protocol_metadata = dict(artifact.metadata.get("protocol_metadata", {}))
    nonbonded_metadata = dict(protocol_metadata.get("nonbonded", {}))
    nonbonded_cutoff = float(
        artifact.metadata.get("nonbonded_cutoff")
        or nonbonded_metadata.get("cutoff")
        or 10.0
    )

    virtual_sites, virtual_site_types = _virtual_sites_from_arrays(arrays)
    force_topology = Topology.from_sequences(
        n_atoms=n_atoms,
        bonds=np.asarray(arrays["bonds"], dtype=np.int32),
        angles=np.asarray(arrays["angles"], dtype=np.int32),
        dihedrals=np.concatenate(
            [np.asarray(arrays["dihedrals"], dtype=np.int32), rb_dihedrals],
            axis=0,
        ),
        impropers=impropers,
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=exception_pairs,
        exclude_bonds=True,
        nonbonded_cutoff=nonbonded_cutoff,
        eager_nonbonded_pair_limit=eager_nonbonded_pair_limit,
        virtual_sites=virtual_sites,
        virtual_site_types=virtual_site_types,
    )
    virtual_site_manager = (
        VirtualSiteManager(virtual_sites, n_real_atoms=n_atoms - len(virtual_sites))
        if virtual_sites
        else None
    )
    system_atom_count = (
        virtual_site_manager.n_real_atoms if virtual_site_manager is not None else n_atoms
    )
    system_topology = Topology.from_sequences(
        n_atoms=system_atom_count,
        bonds=_real_only_indices(np.asarray(arrays["bonds"], dtype=np.int32), system_atom_count),
        angles=_real_only_indices(np.asarray(arrays["angles"], dtype=np.int32), system_atom_count),
        dihedrals=_real_only_indices(
            np.concatenate([np.asarray(arrays["dihedrals"], dtype=np.int32), rb_dihedrals], axis=0),
            system_atom_count,
        ),
        impropers=_real_only_indices(impropers, system_atom_count),
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32)[:system_atom_count],
        nonbonded_exception_pairs=_real_only_indices(exception_pairs, system_atom_count),
        exclude_bonds=True,
        nonbonded_cutoff=nonbonded_cutoff,
        eager_nonbonded_pair_limit=eager_nonbonded_pair_limit,
        virtual_sites=virtual_sites,
        virtual_site_types=virtual_site_types,
    )
    system = MMSystem.from_sequences(
        symbols=tuple(str(item) for item in arrays["symbols"][:system_atom_count].tolist()),
        atom_names=tuple(str(item) for item in arrays["atom_names"][:system_atom_count].tolist()),
        atom_types=tuple(str(item) for item in arrays["atom_types"][:system_atom_count].tolist()),
        positions=np.asarray(arrays["positions"], dtype=np.float32)[:system_atom_count],
        velocities=np.asarray(arrays["velocities"], dtype=np.float32)[:system_atom_count],
        masses=np.asarray(arrays["masses"], dtype=np.float32)[:system_atom_count],
        charges=np.asarray(arrays["charges"], dtype=np.float32)[:system_atom_count],
        topology=system_topology,
        cell=artifact.cell,
        virtual_sites=virtual_site_manager,
    )

    terms = []
    if np.asarray(arrays["bonds"]).shape[0] > 0:
        terms.append(
            HarmonicBondPotential(
                arrays["bonds"],
                k=arrays["bond_k"],
                length=arrays["bond_length"],
            )
        )
    if np.asarray(arrays["angles"]).shape[0] > 0:
        terms.append(
            HarmonicAnglePotential(
                arrays["angles"],
                k=arrays["angle_k"],
                angle=arrays["angle_theta"],
            )
        )
    if np.asarray(arrays["dihedrals"]).shape[0] > 0:
        terms.append(
            PeriodicDihedralPotential(
                arrays["dihedrals"],
                k=arrays["dihedral_k"],
                periodicity=arrays["dihedral_periodicity"],
                phase=arrays["dihedral_phase"],
            )
        )
    if rb_dihedrals.shape[0] > 0:
        terms.append(
            RBDihedralPotential(
                rb_dihedrals,
                c0=_required_vector(arrays, "rb_c0", rb_dihedrals.shape[0], "rb_dihedral"),
                c1=_required_vector(arrays, "rb_c1", rb_dihedrals.shape[0], "rb_dihedral"),
                c2=_required_vector(arrays, "rb_c2", rb_dihedrals.shape[0], "rb_dihedral"),
                c3=_required_vector(arrays, "rb_c3", rb_dihedrals.shape[0], "rb_dihedral"),
                c4=_required_vector(arrays, "rb_c4", rb_dihedrals.shape[0], "rb_dihedral"),
                c5=_required_vector(arrays, "rb_c5", rb_dihedrals.shape[0], "rb_dihedral"),
            )
        )
    if impropers.shape[0] > 0:
        improper_potential = _improper_dihedral_potential_cls()
        terms.append(
            improper_potential(
                impropers,
                k=_optional_vector(arrays, "improper_k", impropers.shape[0]),
                periodicity=_optional_vector(arrays, "improper_periodicity", impropers.shape[0]),
                phase=_optional_vector(arrays, "improper_phase", impropers.shape[0]),
            )
        )
    if "urey_bradley" in requested_terms:
        urey_terms = _required_indices(arrays, "urey_bradley_terms", 3, "urey_bradley")
        terms.append(
            CHARMMUreyBradleyPotential(
                urey_terms,
                k=_required_vector(arrays, "urey_bradley_k", urey_terms.shape[0], "urey_bradley"),
                distance=_required_vector(
                    arrays,
                    "urey_bradley_distance",
                    urey_terms.shape[0],
                    "urey_bradley",
                ),
            )
        )
    if "charmm_cmap" in requested_terms:
        _validate_cmap_arrays(arrays)
        terms.append(
            CHARMMCMAPPotential(
                charmm_cmap_terms=np.asarray(arrays["charmm_cmap_terms"], dtype=np.int32),
                cmap_grids=np.asarray(arrays["charmm_cmap_grids"], dtype=np.float32),
                cmap_indices=np.asarray(arrays["charmm_cmap_grid_indices"], dtype=np.int32),
            )
        )
    if "custom_force" in requested_terms:
        custom_force_metadata = dict(
            artifact.metadata.get("custom_force", {})
            or artifact.metadata.get("protocol_metadata", {}).get("custom_force", {})
        )
        custom_indices = _required_indices(arrays, "custom_force_indices", 2, "custom_force")
        expression = custom_force_metadata.get("expression", "0.5 * k * (r - r0) ** 2")
        term_type = str(custom_force_metadata.get("term_type", "bond"))
        custom_params = {}
        for param_index in range(
            int(
                custom_force_metadata.get("parameter_count", 0)
                or custom_force_metadata.get("n_parameters", 0)
                or 0
            )
        ):
            param_name = str(
                custom_force_metadata.get(
                    f"parameter_{param_index}_name",
                    f"p{param_index}",
                )
            )
            param_key = f"custom_force_{param_name}"
            if param_key in arrays:
                custom_params[param_name] = _required_vector(
                    arrays, param_key, custom_indices.shape[0], "custom_force",
                )
        global_params = dict(custom_force_metadata.get("global_parameters", {}))
        terms.append(
            CustomForcePotential(
                indices=custom_indices,
                expression=expression,
                parameters=custom_params,
                global_parameters=global_params,
                term_type=term_type,
            )
        )

    units = artifact.unit_system
    coulomb_constant = (
        _coulomb_constant_kj_mol_angstrom()
        if units is None
        else units.coulomb_constant
    )
    electrostatics_mode = _metadata_electrostatics_mode(artifact.metadata)
    pme_config = (
        _pme_config_from_artifact(artifact.metadata, arrays, required=True)
        if electrostatics_mode == "pme"
        else None
    )
    if "gbsa" in requested_terms:
        gbsa_metadata = dict(
            artifact.metadata.get("gbsa", {})
            or artifact.metadata.get("protocol_metadata", {}).get("gbsa", {})
        )
        terms.append(
            GBSAForcePotential(
                charges=np.asarray(arrays["charges"], dtype=np.float32),
                radius=_required_vector(arrays, "gbsa_radius", n_atoms, "gbsa"),
                scale=_required_vector(arrays, "gbsa_scale", n_atoms, "gbsa"),
                solvent_dielectric=float(gbsa_metadata.get("solvent_dielectric", 78.5)),
                solute_dielectric=float(gbsa_metadata.get("solute_dielectric", 1.0)),
                surface_area_energy=float(
                    gbsa_metadata.get("surface_area_energy", 0.0225936)
                ),
                probe_radius=float(gbsa_metadata.get("probe_radius", 1.4)),
                radius_offset=float(gbsa_metadata.get("radius_offset", 0.09)),
                coulomb_constant=coulomb_constant,
            )
        )
    nbfix_pairs = np.empty((0, 2), dtype=np.int32)
    nbfix_sigma = np.asarray([], dtype=np.float32)
    nbfix_epsilon = np.asarray([], dtype=np.float32)
    nbfix_type_pairs = np.empty((0, 2), dtype=str)
    nbfix_type_sigma = np.asarray([], dtype=np.float32)
    nbfix_type_epsilon = np.asarray([], dtype=np.float32)
    if "nbfix_pair_overrides" in requested_terms:
        if _array_present(arrays, "nbfix_pairs"):
            nbfix_pairs = _required_indices(arrays, "nbfix_pairs", 2, "nbfix_pair_overrides")
            nbfix_sigma = _required_vector(
                arrays,
                "nbfix_sigma",
                nbfix_pairs.shape[0],
                "nbfix_pair_overrides",
            )
            nbfix_epsilon = _required_vector(
                arrays,
                "nbfix_epsilon",
                nbfix_pairs.shape[0],
                "nbfix_pair_overrides",
            )
        if _array_present(arrays, "nbfix_type_pairs"):
            nbfix_type_pairs = _required_string_pairs(
                arrays,
                "nbfix_type_pairs",
                "nbfix_pair_overrides",
            )
            nbfix_type_sigma = _required_vector(
                arrays,
                "nbfix_type_sigma",
                nbfix_type_pairs.shape[0],
                "nbfix_pair_overrides",
            )
            nbfix_type_epsilon = _required_vector(
                arrays,
                "nbfix_type_epsilon",
                nbfix_type_pairs.shape[0],
                "nbfix_pair_overrides",
            )
            atom_type_names = set(np.asarray(arrays["atom_types"], dtype=str).tolist())
            applicable_type_pairs = np.asarray(
                [
                    str(left) in atom_type_names and str(right) in atom_type_names
                    for left, right in nbfix_type_pairs.tolist()
                ],
                dtype=bool,
            )
            nbfix_type_pairs = nbfix_type_pairs[applicable_type_pairs]
            nbfix_type_sigma = nbfix_type_sigma[applicable_type_pairs]
            nbfix_type_epsilon = nbfix_type_epsilon[applicable_type_pairs]
    nonbonded_kwargs = {
        "sigma": np.asarray(arrays["sigma"], dtype=np.float32),
        "epsilon": np.asarray(arrays["epsilon"], dtype=np.float32),
        "charges": np.asarray(arrays["charges"], dtype=np.float32),
        "coulomb_constant": coulomb_constant,
        "cutoff": nonbonded_cutoff,
        "lj_shift": False,
        "electrostatics": electrostatics_mode,
        "switch_distance": artifact.metadata.get("switch_distance"),
        "topology": force_topology,
        "exception_pairs": exception_pairs,
        "exception_charge_products": charge_products,
        "exception_sigma": exception_sigma,
        "exception_epsilon": exception_epsilon,
        "pme_config": pme_config,
        "atom_types": np.asarray(arrays["atom_types"], dtype=str),
        "nbfix_pairs": nbfix_pairs,
        "nbfix_sigma": nbfix_sigma,
        "nbfix_epsilon": nbfix_epsilon,
        "nbfix_type_pairs": nbfix_type_pairs,
        "nbfix_type_sigma": nbfix_type_sigma,
        "nbfix_type_epsilon": nbfix_type_epsilon,
    }
    lambda_terms = requested_terms & {"soft_core_lj", "lambda_scaled_nonbonded"}
    if lambda_terms:
        lambda_term = (
            "soft_core_lj" if "soft_core_lj" in lambda_terms else "lambda_scaled_nonbonded"
        )
        nonbonded_kwargs["lambda_lj"] = _metadata_lambda_value(
            artifact.metadata,
            "lambda_lj",
            term=lambda_term,
        )
        nonbonded_kwargs["lambda_electrostatics"] = _metadata_lambda_value(
            artifact.metadata,
            "lambda_electrostatics",
            term=lambda_term,
        )
    nonbonded = NonbondedPotential(
        **nonbonded_kwargs,
    )
    if lambda_terms:
        nonbonded = SoftCoreNonbondedPotential(
            nonbonded,
            lambda_lj=nonbonded_kwargs["lambda_lj"],
            lambda_electrostatics=nonbonded_kwargs["lambda_electrostatics"],
        )
    if "charmm_force_switch_nonbonded" in requested_terms:
        if (
            electrostatics_mode != "cutoff"
            or artifact.metadata.get("switch_distance") is None
            or exception_pairs.shape[0] > 0
            or np.asarray(arrays["bonds"]).shape[0] > 0
            or np.asarray(arrays["angles"]).shape[0] > 0
            or np.asarray(arrays["dihedrals"]).shape[0] > 0
            or rb_dihedrals.shape[0] > 0
            or impropers.shape[0] > 0
        ):
            msg = (
                "charmm_force_switch_nonbonded artifacts cannot be faithfully "
                "combined with topology exclusions, exceptions, PME, or missing "
                "switch_distance yet"
            )
            raise MLXCompatibilityError(msg)
        terms.append(
            CHARMMForceSwitchNonbondedPotential(
                sigma=np.asarray(arrays["sigma"], dtype=np.float32),
                epsilon=np.asarray(arrays["epsilon"], dtype=np.float32),
                charges=np.asarray(arrays["charges"], dtype=np.float32),
                coulomb_constant=coulomb_constant,
                cutoff=nonbonded_cutoff,
                switch_distance=float(artifact.metadata["switch_distance"]),
            )
        )
    elif _metadata_terms(artifact.metadata) & {"pair_restricted_lj_coulomb"}:
        pair_restricted_nonbonded = _pair_restricted_nonbonded_potential_cls()
        terms.append(pair_restricted_nonbonded(nonbonded, nonbonded_pairs))
    else:
        terms.append(nonbonded)

    if restraint_k > 0.0 and "restraint_mask" in arrays and "reference_positions" in arrays:
        positional_restraint = _positional_restraint_potential_cls()
        terms.append(
            positional_restraint(
                reference_positions=np.asarray(arrays["reference_positions"], dtype=np.float32),
                mask=np.asarray(arrays["restraint_mask"], dtype=bool),
                k=restraint_k,
            )
        )
    distance_constraints = None
    if constraints.shape[0] > 0:
        distance_constraints = DistanceConstraints(
            constraints,
            distances=constraint_distances,
            max_iterations=constraint_max_iterations,
        )
    return system, terms, distance_constraints


def _virtual_sites_from_arrays(
    arrays: dict[str, np.ndarray],
) -> tuple[tuple[ThreeParticleAverage, ...], tuple[str, ...]]:
    parent_atoms = np.asarray(
        arrays.get("virtual_site_parent_atoms", np.empty((0, 4), dtype=np.int32)),
        dtype=np.int32,
    )
    if parent_atoms.size == 0:
        return (), ()
    weights = np.asarray(arrays.get("virtual_site_weights", np.empty((0, 4))), dtype=np.float32)
    types = np.asarray(arrays.get("virtual_site_types", np.asarray([], dtype=str)), dtype=str)
    if parent_atoms.ndim != 2 or parent_atoms.shape[1] != 4:
        msg = "virtual_site_parent_atoms must have shape (n, 4)"
        raise MLXCompatibilityError(msg)
    if weights.shape != parent_atoms.shape:
        msg = f"virtual_site_weights must have shape {parent_atoms.shape}"
        raise MLXCompatibilityError(msg)
    if types.shape != (parent_atoms.shape[0],):
        msg = f"virtual_site_types must have shape ({parent_atoms.shape[0]},)"
        raise MLXCompatibilityError(msg)

    virtual_sites: list[ThreeParticleAverage] = []
    virtual_site_types: list[str] = []
    for parents, row_weights, site_type in zip(parent_atoms, weights, types, strict=True):
        normalized_type = str(site_type).strip().lower().replace("-", "_")
        active = [int(parent) for parent in parents[:3].tolist()]
        if normalized_type in {"tip4p", "tip4p_ew", "tip4pew"}:
            if len(active) != 3:
                msg = "tip4p virtual sites require exactly three parent atoms"
                raise MLXCompatibilityError(msg)
            virtual_sites.append(tip4p_ew_virtual_site(active[0], active[1], active[2]))
            virtual_site_types.append("tip4p_ew")
            continue
        if normalized_type == "three_particle_average":
            if len(active) != 3:
                msg = "three_particle_average virtual sites require exactly three parent atoms"
                raise MLXCompatibilityError(msg)
            virtual_sites.append(
                ThreeParticleAverage(
                    particle1=active[0],
                    particle2=active[1],
                    particle3=active[2],
                    weight1=float(row_weights[0]),
                    weight2=float(row_weights[1]),
                    weight3=float(row_weights[2]),
                )
            )
            virtual_site_types.append(normalized_type)
            continue
        msg = f"unsupported virtual_site type: {site_type}"
        raise MLXCompatibilityError(msg)
    return tuple(virtual_sites), tuple(virtual_site_types)


def _real_only_indices(indices: np.ndarray, real_atom_count: int) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, indices.shape[1] if indices.ndim == 2 else 0), dtype=np.int32)
    mask = np.all(indices < real_atom_count, axis=1)
    return np.asarray(indices[mask], dtype=np.int32)


def _virtual_site_layout(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _validate_virtual_site_arrays(arrays)
    parent_atoms = np.asarray(arrays["virtual_site_parent_atoms"], dtype=np.int32)
    n_atoms = int(np.asarray(arrays["positions"]).shape[0])
    site_indices = parent_atoms[:, 3].astype(np.int32)
    real_indices = np.asarray(
        [index for index in range(n_atoms) if index not in set(site_indices.tolist())],
        dtype=np.int32,
    )
    permutation = np.concatenate([real_indices, site_indices])
    inverse = np.empty((n_atoms,), dtype=np.int32)
    inverse[permutation] = np.arange(n_atoms, dtype=np.int32)
    return {"permutation": permutation, "inverse": inverse}


def _reordered_virtual_site_arrays(
    arrays: dict[str, np.ndarray],
    layout: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    permutation = layout["permutation"]
    inverse = layout["inverse"]
    reordered = dict(arrays)
    for name in [
        "symbols",
        "atom_names",
        "atom_types",
        "residue_names",
        "residue_ids",
        "chain_ids",
        "positions",
        "velocities",
        "masses",
        "charges",
        "sigma",
        "epsilon",
        "ligand_mask",
        "receptor_mask",
        "restraint_mask",
        "reference_positions",
        "gbsa_radius",
        "gbsa_scale",
        "water_mask",
        "ion_mask",
        "lipid_mask",
    ]:
        if name in reordered and np.asarray(reordered[name]).shape[:1] == (len(permutation),):
            reordered[name] = np.asarray(reordered[name])[permutation]
    for name, width in [
        ("bonds", 2),
        ("angles", 3),
        ("dihedrals", 4),
        ("rb_dihedrals", 4),
        ("impropers", 4),
        ("nonbonded_pairs", 2),
        ("constraints", 2),
        ("nonbonded_exception_pairs", 2),
        ("urey_bradley_terms", 3),
        ("nbfix_pairs", 2),
    ]:
        if name in reordered:
            values = np.asarray(reordered[name], dtype=np.int32)
            if values.size == 0:
                reordered[name] = np.empty((0, width), dtype=np.int32)
            else:
                reordered[name] = inverse[values]
    parent_atoms = np.asarray(reordered["virtual_site_parent_atoms"], dtype=np.int32)
    reordered["virtual_site_parent_atoms"] = inverse[parent_atoms]
    return reordered


__all__ = [
    "MLXCompatibilityError",
    "PreparedMLXArtifact",
    "artifact_readiness_report",
    "build_mlx_system_from_artifact",
    "load_prepared_mlx_artifact",
    "validate_mlx_compatibility",
]
