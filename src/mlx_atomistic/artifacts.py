"""Strict loaders for prepared MLX molecular mechanics artifacts."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import (
    CHARMMCMAPPotential,
    CHARMMForceSwitchNonbondedPotential,
    CHARMMUreyBradleyPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    NonbondedPotential,
    PeriodicDihedralPotential,
    PMEConfig,
)
from mlx_atomistic.mm import MMSystem
from mlx_atomistic.nonbonded import normalize_nonbonded_electrostatics
from mlx_atomistic.topology import Topology

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
        "nonbonded_lj_coulomb",
        "pair_restricted_lj_coulomb",
        "nonbonded_exception",
        "ewald_reference_electrostatics",
        "distance_constraint",
        "positional_restraint",
        "pme",
        "charmm_cmap",
        "urey_bradley",
        "nbfix_pair_overrides",
        "charmm_force_switch_nonbonded",
        "lipid",
        "water",
        "ion",
        "receptor",
        "ligand",
    }
)
FAIL_CLOSED_TERMS = frozenset(
    {
        "ljpme",
        "barostat",
        "npt",
        "npt_barostat",
        "virtual_site",
        "virtual_sites",
        "hmr_or_virtual_site_policy_required",
        "hydrogen_mass_repartitioning",
        "virtual_sites_or_hydrogen_mass_repartitioning_not_checked",
        "drude",
        "polarizable",
        "gbsa",
        "reactive",
        "bond_breaking",
        "qm_mm",
    }
)
TERM_ALIASES = {
    "bonds": "harmonic_bond",
    "angles": "harmonic_angle",
    "dihedrals": "periodic_dihedral",
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
        cell_lengths = np.asarray(self.arrays.get("cell_lengths", np.asarray([])))
        if cell_lengths.size == 0:
            return None
        if cell_lengths.shape != (3,):
            msg = "cell_lengths must have shape (3,)"
            raise ValueError(msg)
        return Cell.orthorhombic(cell_lengths.astype(np.float32).tolist())


def _jsonable_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_term(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return TERM_ALIASES.get(normalized, normalized)


def _metadata_terms(metadata: dict[str, Any]) -> set[str]:
    report = dict(metadata.get("compatibility_report", {}))
    required = {_normalize_term(term) for term in _jsonable_list(report.get("required_terms"))}
    supported = {_normalize_term(term) for term in _jsonable_list(report.get("supported_terms"))}
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


def _array_present(arrays: dict[str, np.ndarray] | None, name: str) -> bool:
    return arrays is not None and name in arrays and np.asarray(arrays[name]).size != 0


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
    assignment_order: int,
    charge_tolerance: float,
) -> tuple[int, int, int]:
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
        msg = "pme artifacts require pme_assignment_order=2"
        raise MLXCompatibilityError(msg) from err
    if (
        not np.isfinite(assignment_order_value)
        or assignment_order_value != np.floor(assignment_order_value)
        or int(assignment_order_value) != 2
    ):
        msg = "pme artifacts require pme_assignment_order=2"
        raise MLXCompatibilityError(msg)
    if not np.isfinite(charge_tolerance) or charge_tolerance < 0.0:
        msg = "pme artifacts require finite non-negative pme_charge_tolerance"
        raise MLXCompatibilityError(msg)
    return tuple(int(item) for item in mesh_values.tolist())


def _pme_config_from_artifact(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
    *,
    required: bool = False,
) -> PMEConfig | None:
    has_array_config = _array_present(arrays, "pme_mesh_shape")
    if arrays is not None and has_array_config:
        mesh_shape = np.asarray(arrays["pme_mesh_shape"])
        if mesh_shape.shape != (3,):
            msg = "pme artifacts must include pme_mesh_shape with shape (3,)"
            raise MLXCompatibilityError(msg)
        try:
            alpha = float(_pme_scalar_from_arrays(arrays, "pme_alpha"))
            real_cutoff = float(_pme_scalar_from_arrays(arrays, "pme_real_cutoff"))
            assignment_order = float(_pme_scalar_from_arrays(arrays, "pme_assignment_order"))
            charge_tolerance = float(_pme_scalar_from_arrays(arrays, "pme_charge_tolerance"))
            mesh_shape_tuple = _validate_pme_config_values(
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
            mesh_shape_tuple = _validate_pme_config_values(
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

    report = dict(metadata.get("compatibility_report", {}))
    unsupported = {
        _normalize_term(term) for term in _jsonable_list(report.get("unsupported_terms"))
    }
    unsupported |= {
        _normalize_term(term) for term in _jsonable_list(report.get("rejected_terms"))
    }
    requested_terms = _metadata_terms(metadata)
    _validate_declared_term_arrays(metadata, requested_terms=requested_terms, arrays=arrays)

    blockers = sorted((unsupported | requested_terms) & FAIL_CLOSED_TERMS)
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        msg = f"prepared artifact declares unsupported force-field terms: {joined}"
        raise MLXCompatibilityError(msg)
    if blockers:
        joined = ", ".join(blockers)
        msg = f"prepared artifact requires unsupported production terms: {joined}"
        raise MLXCompatibilityError(msg)

    unknown = sorted(term for term in requested_terms if term and term not in SUPPORTED_FORCE_TERMS)
    if unknown:
        joined = ", ".join(unknown)
        msg = f"prepared artifact requires terms not implemented in mlx_atomistic: {joined}"
        raise MLXCompatibilityError(msg)

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
        "urey_bradley_terms": "urey_bradley",
        "nbfix_pairs": "nbfix_pair_overrides",
        "nbfix_type_pairs": "nbfix_pair_overrides",
        "nbfix_type_sigma": "nbfix_pair_overrides",
        "nbfix_type_epsilon": "nbfix_pair_overrides",
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

    metadata = json.loads(json_path.read_text())
    arrays = _load_arrays(npz_path)
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
    if np.any(np.asarray(arrays["masses"], dtype=np.float32) <= 0.0):
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
    if bool(report.get("virtual_sites_present", False)):
        msg = "virtual_site artifacts are not supported by mlx_atomistic"
        raise MLXCompatibilityError(msg)
    if any(model in water_model for model in ("tip4p", "tip5p", "opc")):
        msg = f"virtual_site water model is not supported: {water_model}"
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


def _validate_electrostatics_arrays(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    mode = _metadata_electrostatics_mode(metadata)
    if mode not in {"ewald_reference", "pme"}:
        return
    cell_lengths = np.asarray(arrays.get("cell_lengths", np.asarray([])), dtype=np.float32)
    if cell_lengths.shape != (3,) or not np.all(np.isfinite(cell_lengths)):
        msg = f"{mode} artifacts must include finite cell_lengths with shape (3,)"
        raise MLXCompatibilityError(msg)
    if np.any(cell_lengths <= 0.0):
        msg = f"{mode} artifacts must include positive cell_lengths"
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
    for term_name, mask_name in [
        ("lipid", "lipid_mask"),
        ("water", "water_mask"),
        ("ion", "ion_mask"),
        ("receptor", "receptor_mask"),
        ("ligand", "ligand_mask"),
    ]:
        if term_name in requested_terms:
            _required_mask(arrays, mask_name, n_atoms, term_name)


def build_mlx_system_from_artifact(
    artifact: PreparedMLXArtifact,
    *,
    restraint_k: float = 0.0,
    constraint_max_iterations: int = 20,
) -> tuple[MMSystem, list, DistanceConstraints | None]:
    """Convert a prepared artifact into `MMSystem`, force terms, and constraints."""

    arrays = artifact.arrays
    n_atoms = artifact.atom_count
    impropers = _optional_indices(arrays, "impropers", 4)
    exception_pairs = _optional_indices(arrays, "nonbonded_exception_pairs", 2)
    nonbonded_pairs = _optional_indices(arrays, "nonbonded_pairs", 2)
    constraints = _optional_indices(arrays, "constraints", 2)
    constraint_distances = _optional_vector(arrays, "constraint_distance", constraints.shape[0])
    requested_terms = _metadata_terms(artifact.metadata)
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
    nonbonded_cutoff = float(artifact.metadata.get("nonbonded_cutoff", 10.0))

    topology = Topology.from_sequences(
        n_atoms=n_atoms,
        bonds=np.asarray(arrays["bonds"], dtype=np.int32),
        angles=np.asarray(arrays["angles"], dtype=np.int32),
        dihedrals=np.asarray(arrays["dihedrals"], dtype=np.int32),
        impropers=impropers,
        partial_charges=np.asarray(arrays["charges"], dtype=np.float32),
        nonbonded_exception_pairs=exception_pairs,
        exclude_bonds=True,
        nonbonded_cutoff=nonbonded_cutoff,
    )
    system = MMSystem.from_sequences(
        symbols=tuple(str(item) for item in arrays["symbols"].tolist()),
        atom_names=tuple(str(item) for item in arrays["atom_names"].tolist()),
        atom_types=tuple(str(item) for item in arrays["atom_types"].tolist()),
        positions=np.asarray(arrays["positions"], dtype=np.float32),
        velocities=np.asarray(arrays["velocities"], dtype=np.float32),
        masses=np.asarray(arrays["masses"], dtype=np.float32),
        charges=np.asarray(arrays["charges"], dtype=np.float32),
        topology=topology,
        cell=artifact.cell,
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
        "topology": topology,
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
    nonbonded = NonbondedPotential(
        **nonbonded_kwargs,
    )
    if "charmm_force_switch_nonbonded" in requested_terms:
        if (
            electrostatics_mode != "cutoff"
            or artifact.metadata.get("switch_distance") is None
            or exception_pairs.shape[0] > 0
            or np.asarray(arrays["bonds"]).shape[0] > 0
            or np.asarray(arrays["angles"]).shape[0] > 0
            or np.asarray(arrays["dihedrals"]).shape[0] > 0
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


__all__ = [
    "MLXCompatibilityError",
    "PreparedMLXArtifact",
    "build_mlx_system_from_artifact",
    "load_prepared_mlx_artifact",
    "validate_mlx_compatibility",
]
