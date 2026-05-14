"""Artifact schema for MLX-compatible prepared atomistic systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ARTIFACT_VERSION = 2
SUPPORTED_ARTIFACT_VERSIONS = frozenset({1, ARTIFACT_VERSION})


@dataclass(frozen=True)
class PreparedSystemMetadata:
    """JSON metadata stored next to a prepared MLX system."""

    artifact_version: int
    source: dict[str, Any]
    selections: dict[str, Any]
    units: dict[str, str]
    parameter_source: str
    compatibility_report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    created_at: str | None = None
    pme_config: dict[str, Any] = field(default_factory=dict)
    protocol_metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "created_at": self.created_at,
            "source": self.source,
            "selections": self.selections,
            "units": self.units,
            "parameter_source": self.parameter_source,
            "compatibility_report": self.compatibility_report,
            "warnings": list(self.warnings),
            "pme_config": dict(self.pme_config),
            "protocol_metadata": dict(self.protocol_metadata),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> PreparedSystemMetadata:
        return cls(
            artifact_version=int(payload["artifact_version"]),
            created_at=payload.get("created_at"),
            source=dict(payload.get("source", {})),
            selections=dict(payload.get("selections", {})),
            units=dict(payload.get("units", {})),
            parameter_source=str(payload.get("parameter_source", "")),
            compatibility_report=dict(payload.get("compatibility_report", {})),
            warnings=[str(item) for item in payload.get("warnings", [])],
            pme_config=dict(payload.get("pme_config", {})),
            protocol_metadata=dict(payload.get("protocol_metadata", {})),
        )


@dataclass(frozen=True)
class PreparedSystem:
    """Numerical arrays needed to run a prepared system in `mlx_atomistic`."""

    metadata: PreparedSystemMetadata
    symbols: np.ndarray
    atom_names: np.ndarray
    atom_types: np.ndarray
    residue_names: np.ndarray
    residue_ids: np.ndarray
    chain_ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    bonds: np.ndarray
    bond_k: np.ndarray
    bond_length: np.ndarray
    angles: np.ndarray
    angle_k: np.ndarray
    angle_theta: np.ndarray
    dihedrals: np.ndarray
    dihedral_k: np.ndarray
    dihedral_periodicity: np.ndarray
    dihedral_phase: np.ndarray
    nonbonded_pairs: np.ndarray
    ligand_mask: np.ndarray
    receptor_mask: np.ndarray
    restraint_mask: np.ndarray
    reference_positions: np.ndarray
    cell_lengths: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    constraints: np.ndarray = field(default_factory=lambda: empty_indices(2))
    constraint_distance: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    impropers: np.ndarray = field(default_factory=lambda: empty_indices(4))
    improper_k: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    improper_periodicity: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    improper_phase: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    nonbonded_exception_pairs: np.ndarray = field(default_factory=lambda: empty_indices(2))
    nonbonded_exception_charge_product: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    nonbonded_exception_sigma: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    nonbonded_exception_epsilon: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    water_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    ion_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    lipid_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    pme_mesh_shape: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    pme_alpha: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    pme_real_cutoff: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    pme_assignment_order: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int32)
    )
    pme_charge_tolerance: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    pme_deconvolve_assignment: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=bool)
    )
    charmm_cmap_terms: np.ndarray = field(default_factory=lambda: empty_indices(8))
    charmm_cmap_grid_indices: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int32)
    )
    charmm_cmap_grids: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float32)
    )
    urey_bradley_terms: np.ndarray = field(default_factory=lambda: empty_indices(3))
    urey_bradley_k: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    urey_bradley_distance: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float32)
    )
    nbfix_pairs: np.ndarray = field(default_factory=lambda: empty_indices(2))
    nbfix_sigma: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    nbfix_epsilon: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    nbfix_type_pairs: np.ndarray = field(default_factory=lambda: empty_string_pairs())
    nbfix_type_sigma: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    nbfix_type_epsilon: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))

    @property
    def atom_count(self) -> int:
        return int(self.positions.shape[0])

    def validate(self) -> None:
        """Validate shape and index consistency."""

        n_atoms = self.atom_count
        if n_atoms <= 0:
            msg = "prepared system must contain at least one atom"
            raise ValueError(msg)
        for name in [
            "symbols",
            "atom_names",
            "atom_types",
            "residue_names",
            "chain_ids",
        ]:
            array = np.asarray(getattr(self, name))
            if array.shape != (n_atoms,):
                msg = f"{name} must have shape ({n_atoms},)"
                raise ValueError(msg)
        for name in [
            "residue_ids",
            "masses",
            "charges",
            "sigma",
            "epsilon",
            "ligand_mask",
            "receptor_mask",
            "restraint_mask",
        ]:
            array = np.asarray(getattr(self, name))
            if array.shape != (n_atoms,):
                msg = f"{name} must have shape ({n_atoms},)"
                raise ValueError(msg)
        for name in ["positions", "velocities", "reference_positions"]:
            array = np.asarray(getattr(self, name), dtype=np.float32)
            if array.shape != (n_atoms, 3):
                msg = f"{name} must have shape ({n_atoms}, 3)"
                raise ValueError(msg)
        for name in ["water_mask", "ion_mask", "lipid_mask"]:
            array = np.asarray(getattr(self, name), dtype=bool)
            if array.size != 0 and array.shape != (n_atoms,):
                msg = f"{name} must be empty or have shape ({n_atoms},)"
                raise ValueError(msg)
        self._validate_index_array("bonds", 2)
        self._validate_index_array("angles", 3)
        self._validate_index_array("dihedrals", 4)
        self._validate_index_array("impropers", 4)
        self._validate_index_array("nonbonded_pairs", 2)
        self._validate_index_array("constraints", 2)
        self._validate_index_array("nonbonded_exception_pairs", 2)
        self._validate_index_array("charmm_cmap_terms", 8)
        self._validate_index_array("urey_bradley_terms", 3)
        self._validate_index_array("nbfix_pairs", 2)
        self._validate_string_pair_array("nbfix_type_pairs")
        self._validate_parameter_length("bond_k", "bonds")
        self._validate_parameter_length("bond_length", "bonds")
        self._validate_parameter_length("angle_k", "angles")
        self._validate_parameter_length("angle_theta", "angles")
        self._validate_parameter_length("dihedral_k", "dihedrals")
        self._validate_parameter_length("dihedral_periodicity", "dihedrals")
        self._validate_parameter_length("dihedral_phase", "dihedrals")
        self._validate_parameter_length("improper_k", "impropers")
        self._validate_parameter_length("improper_periodicity", "impropers")
        self._validate_parameter_length("improper_phase", "impropers")
        self._validate_parameter_length("constraint_distance", "constraints")
        self._validate_parameter_length(
            "nonbonded_exception_charge_product",
            "nonbonded_exception_pairs",
        )
        self._validate_parameter_length("nonbonded_exception_sigma", "nonbonded_exception_pairs")
        self._validate_parameter_length("nonbonded_exception_epsilon", "nonbonded_exception_pairs")
        self._validate_parameter_length("charmm_cmap_grid_indices", "charmm_cmap_terms")
        self._validate_parameter_length("urey_bradley_k", "urey_bradley_terms")
        self._validate_parameter_length("urey_bradley_distance", "urey_bradley_terms")
        self._validate_parameter_length("nbfix_sigma", "nbfix_pairs")
        self._validate_parameter_length("nbfix_epsilon", "nbfix_pairs")
        self._validate_parameter_length("nbfix_type_sigma", "nbfix_type_pairs")
        self._validate_parameter_length("nbfix_type_epsilon", "nbfix_type_pairs")
        self._validate_nbfix_parameters()
        self._validate_pme_arrays()
        self._validate_charmm_cmap_grids()
        cell_lengths = np.asarray(self.cell_lengths, dtype=np.float32)
        if cell_lengths.size not in {0, 3}:
            msg = "cell_lengths must be empty or have shape (3,)"
            raise ValueError(msg)
        if cell_lengths.size == 3 and cell_lengths.shape != (3,):
            msg = "cell_lengths must be empty or have shape (3,)"
            raise ValueError(msg)
        if self.metadata.artifact_version not in SUPPORTED_ARTIFACT_VERSIONS:
            msg = (
                f"unsupported artifact version {self.metadata.artifact_version}; "
                f"expected one of {sorted(SUPPORTED_ARTIFACT_VERSIONS)}"
            )
            raise ValueError(msg)

    def _validate_index_array(self, name: str, width: int) -> None:
        array = np.asarray(getattr(self, name), dtype=np.int32)
        if array.ndim != 2 or array.shape[1] != width:
            msg = f"{name} must have shape (n, {width})"
            raise ValueError(msg)
        if array.size == 0:
            return
        if np.any(array < 0) or np.any(array >= self.atom_count):
            msg = f"{name} contains atom indices outside [0, atom_count)"
            raise ValueError(msg)

    def _validate_parameter_length(self, parameter_name: str, index_name: str) -> None:
        values = np.asarray(getattr(self, parameter_name))
        index_count = int(np.asarray(getattr(self, index_name)).shape[0])
        if values.shape != (index_count,):
            msg = f"{parameter_name} must have shape ({index_count},)"
            raise ValueError(msg)

    def _validate_string_pair_array(self, name: str) -> None:
        array = np.asarray(getattr(self, name), dtype=str)
        if array.ndim != 2 or array.shape[1] != 2:
            msg = f"{name} must have shape (n, 2)"
            raise ValueError(msg)
        if array.size and np.any(np.char.str_len(array) == 0):
            msg = f"{name} must contain non-empty atom type identifiers"
            raise ValueError(msg)

    def _validate_nbfix_parameters(self) -> None:
        for sigma_name, epsilon_name in [
            ("nbfix_sigma", "nbfix_epsilon"),
            ("nbfix_type_sigma", "nbfix_type_epsilon"),
        ]:
            sigma = np.asarray(getattr(self, sigma_name), dtype=np.float32)
            epsilon = np.asarray(getattr(self, epsilon_name), dtype=np.float32)
            if sigma.size and (not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0)):
                msg = f"{sigma_name} values must be finite and positive"
                raise ValueError(msg)
            if epsilon.size and (not np.all(np.isfinite(epsilon)) or np.any(epsilon < 0.0)):
                msg = f"{epsilon_name} values must be finite and non-negative"
                raise ValueError(msg)

    def _validate_pme_arrays(self) -> None:
        mesh_shape = np.asarray(self.pme_mesh_shape)
        scalar_names = [
            "pme_alpha",
            "pme_real_cutoff",
            "pme_assignment_order",
            "pme_charge_tolerance",
            "pme_deconvolve_assignment",
        ]
        has_any = mesh_shape.size != 0 or any(
            np.asarray(getattr(self, name)).size for name in scalar_names
        )
        if not has_any:
            return
        if mesh_shape.shape != (3,):
            msg = "pme_mesh_shape must be empty or have shape (3,)"
            raise ValueError(msg)
        try:
            mesh_values = np.asarray(mesh_shape, dtype=np.float64)
        except (TypeError, ValueError) as err:
            msg = "pme_mesh_shape dimensions must be finite integers >= 4"
            raise ValueError(msg) from err
        if (
            not np.all(np.isfinite(mesh_values))
            or not np.all(mesh_values == np.floor(mesh_values))
            or np.any(mesh_values < 4)
        ):
            msg = "pme_mesh_shape dimensions must be finite integers >= 4"
            raise ValueError(msg)
        for name in scalar_names:
            values = np.asarray(getattr(self, name))
            if values.shape != (1,):
                msg = f"{name} must be empty or have shape (1,)"
                raise ValueError(msg)
        alpha = float(np.asarray(self.pme_alpha, dtype=np.float32)[0])
        real_cutoff = float(np.asarray(self.pme_real_cutoff, dtype=np.float32)[0])
        assignment_order = float(np.asarray(self.pme_assignment_order, dtype=np.float32)[0])
        charge_tolerance = float(np.asarray(self.pme_charge_tolerance, dtype=np.float32)[0])
        if not np.isfinite(alpha) or alpha <= 0.0:
            msg = "pme_alpha must be finite and positive"
            raise ValueError(msg)
        if not np.isfinite(real_cutoff) or real_cutoff <= 0.0:
            msg = "pme_real_cutoff must be finite and positive"
            raise ValueError(msg)
        if (
            not np.isfinite(assignment_order)
            or assignment_order != np.floor(assignment_order)
            or int(assignment_order) != 2
        ):
            msg = "pme_assignment_order must be 2"
            raise ValueError(msg)
        if not np.isfinite(charge_tolerance) or charge_tolerance < 0.0:
            msg = "pme_charge_tolerance must be finite and non-negative"
            raise ValueError(msg)

    def _validate_charmm_cmap_grids(self) -> None:
        grids = np.asarray(self.charmm_cmap_grids, dtype=np.float32)
        terms = np.asarray(self.charmm_cmap_terms, dtype=np.int32)
        if terms.shape[0] == 0 and grids.size == 0:
            return
        if grids.ndim != 3 or grids.shape[1] != grids.shape[2]:
            msg = "charmm_cmap_grids must have shape (n_maps, grid, grid)"
            raise ValueError(msg)
        if grids.shape[1] < 4:
            msg = "charmm_cmap_grids must use at least a 4x4 periodic grid"
            raise ValueError(msg)
        if not np.all(np.isfinite(grids)):
            msg = "charmm_cmap_grids must be finite"
            raise ValueError(msg)
        indices = np.asarray(self.charmm_cmap_grid_indices, dtype=np.int32)
        if indices.size and (np.any(indices < 0) or np.any(indices >= grids.shape[0])):
            msg = "charmm_cmap_grid_indices contain indices outside charmm_cmap_grids"
            raise ValueError(msg)


def empty_indices(width: int) -> np.ndarray:
    """Return a typed empty index array with a fixed width."""

    return np.empty((0, width), dtype=np.int32)


def empty_string_pairs() -> np.ndarray:
    """Return a typed empty string pair array."""

    return np.empty((0, 2), dtype=str)
