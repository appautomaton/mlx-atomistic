"""Identity and calculation contracts for periodic SCF artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic._artifact_identity import (
    ArtifactIntegrityError,
    canonical_json_bytes,
    sha256_bytes,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDFTSystem,
    PeriodicSCFConfig,
    _eigensolve_provenance,
)
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.pseudopotentials import (
    PseudopotentialData,
    PseudopotentialFormat,
)
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

PERIODIC_SCF_CHECKPOINT_KIND = "periodic-scf-checkpoint"
PERIODIC_SCF_CHECKPOINT_SCHEMA = "mlx-atomistic.periodic-scf-checkpoint.v1"
PERIODIC_SCF_CHECKPOINT_PAYLOAD = "checkpoint.json"
PERIODIC_SCF_COMMAND_KIND = "periodic-scf"
_CALCULATION_CONTRACT_SCHEMA = "mlx-atomistic.periodic-scf-calculation.v2"
_LEGACY_CALCULATION_CONTRACT_SCHEMA = "mlx-atomistic.periodic-scf-calculation.v1"
_SHA256_FIELDS = (
    "workload_fingerprint",
    "protocol_fingerprint",
    "runtime_fingerprint",
    "execution_contract_fingerprint",
)

def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_complete_execution_contract(contract: Mapping[str, object]) -> None:
    mapping_fields = (
        "solver",
        "initialization",
        "settings_override",
        "lock",
        "environment",
        "host_protocol",
    )
    if contract.get("schema_version") != (
        "mlx-atomistic.dft-runtime-execution-contract.v1"
    ) or not isinstance(contract.get("command_kind"), str):
        msg = "periodic checkpoint execution contract schema or command is incomplete"
        raise ValueError(msg)
    if any(not isinstance(contract.get(field_name), Mapping) for field_name in mapping_fields):
        msg = "periodic checkpoint execution contract is missing a required object"
        raise ValueError(msg)
    if not isinstance(contract.get("synchronization"), str) or not contract.get("synchronization"):
        msg = "periodic checkpoint execution synchronization identity is missing"
        raise ValueError(msg)

    solver = contract["solver"]
    if not isinstance(solver.get("scf"), Mapping) or not isinstance(
        solver.get("davidson"),
        Mapping,
    ):
        msg = "periodic checkpoint execution solver identity is incomplete"
        raise ValueError(msg)
    lock = contract["lock"]
    if (
        lock.get("path") != "uv.lock"
        or type(lock.get("byte_size")) is not int
        or lock["byte_size"] < 0
        or not _is_sha256(lock.get("sha256"))
    ):
        msg = "periodic checkpoint lock identity is incomplete"
        raise ValueError(msg)
    environment = contract["environment"]
    required_environment = {
        "python_version",
        "python_implementation",
        "mlx_version",
        "default_device",
        "metal_available",
        "selected_device",
        "precision",
        "full_grid_precision",
        "projected_eigensolve_device",
        "projected_eigensolve_backend",
        "projected_eigensolve_precision",
        "projected_eigensolve_output_precision",
    }
    if not required_environment.issubset(environment):
        msg = "periodic checkpoint runtime environment identity is incomplete"
        raise ValueError(msg)
    host_protocol = contract["host_protocol"]
    required_host = {
        "model",
        "model_identifier",
        "chip",
        "machine",
        "macos",
        "power_source",
        "low_power_mode",
    }
    if not required_host.issubset(host_protocol) or not isinstance(
        host_protocol.get("macos"),
        Mapping,
    ):
        msg = "periodic checkpoint host identity is incomplete"
        raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicSCFExecutionIdentity:
    """Path-independent pre-execution identity required for checkpoint reuse.

    Args:
        workload_fingerprint: Canonical workload and GTH-resource fingerprint.
        protocol_fingerprint: Frozen measurement-protocol source fingerprint.
        runtime_fingerprint: Complete executing periodic-DFT source fingerprint.
        execution_contract_fingerprint: Hash of ``execution_contract``.
        execution_contract: Full canonical pre-run execution contract.
    """

    workload_fingerprint: str
    protocol_fingerprint: str
    runtime_fingerprint: str
    execution_contract_fingerprint: str
    execution_contract: Mapping[str, object]

    def __post_init__(self) -> None:
        values = {field_name: getattr(self, field_name) for field_name in _SHA256_FIELDS}
        if not all(_is_sha256(value) for value in values.values()):
            msg = "periodic checkpoint identity fields must be lowercase SHA-256 values"
            raise ValueError(msg)
        try:
            contract = json.loads(canonical_json_bytes(dict(self.execution_contract)))
        except (TypeError, ValueError) as error:
            msg = "periodic checkpoint execution contract must be finite canonical JSON"
            raise ValueError(msg) from error
        _validate_complete_execution_contract(contract)
        if (
            contract.get("workload_fingerprint") != self.workload_fingerprint
            or contract.get("protocol_fingerprint") != self.protocol_fingerprint
            or contract.get("runtime_fingerprint") != self.runtime_fingerprint
        ):
            msg = "periodic checkpoint execution contract decomposition is inconsistent"
            raise ValueError(msg)
        observed = sha256_bytes(canonical_json_bytes(contract))
        if observed != self.execution_contract_fingerprint:
            msg = "periodic checkpoint execution contract fingerprint is inconsistent"
            raise ValueError(msg)
        object.__setattr__(self, "execution_contract", contract)

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, object],
    ) -> PeriodicSCFExecutionIdentity:
        """Construct identity from the existing DFT execution-context mapping.

        Args:
            context: Mapping returned by the frozen ``build_execution_context``
                function, or an equivalent path-independent mapping.

        Returns:
            Validated checkpoint execution identity.
        """

        if not isinstance(context, Mapping):
            msg = "periodic checkpoint execution context must be an object"
            raise ValueError(msg)
        contract = context.get("execution_contract")
        if not isinstance(contract, Mapping):
            msg = "periodic checkpoint context is missing its execution contract"
            raise ValueError(msg)
        return cls(
            workload_fingerprint=str(contract.get("workload_fingerprint", "")),
            protocol_fingerprint=str(context.get("protocol_fingerprint", "")),
            runtime_fingerprint=str(context.get("runtime_fingerprint", "")),
            execution_contract_fingerprint=str(context.get("execution_contract_fingerprint", "")),
            execution_contract=dict(contract),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the four non-circular artifact-manifest identity fields."""

        return {field_name: getattr(self, field_name) for field_name in _SHA256_FIELDS}


def _coerce_identity(
    value: PeriodicSCFExecutionIdentity | Mapping[str, object],
) -> PeriodicSCFExecutionIdentity:
    if isinstance(value, PeriodicSCFExecutionIdentity):
        return PeriodicSCFExecutionIdentity(
            workload_fingerprint=value.workload_fingerprint,
            protocol_fingerprint=value.protocol_fingerprint,
            runtime_fingerprint=value.runtime_fingerprint,
            execution_contract_fingerprint=value.execution_contract_fingerprint,
            execution_contract=value.execution_contract,
        )
    if not isinstance(value, Mapping):
        msg = "periodic checkpoint execution context must be an object"
        raise ValueError(msg)
    return PeriodicSCFExecutionIdentity.from_context(value)


def _config_payload(config: PeriodicSCFConfig) -> dict[str, object]:
    return {
        "max_iterations": config.max_iterations,
        "density_tolerance": config.density_tolerance,
        "energy_tolerance_hartree": config.energy_tolerance,
        "orbital_tolerance": config.orbital_tolerance,
        "min_iterations": config.min_iterations,
        "mixing_beta": config.mixing_beta,
        "mixer": config.mixer,
        "adaptive_eigensolver_tolerance": config.adaptive_eigensolver_tolerance,
        "initial_eigensolver_tolerance": config.initial_eigensolver_tolerance,
        "eigensolver_tolerance_scale": config.eigensolver_tolerance_scale,
        "smearing": (
            None
            if config.smearing is None
            else {
                "method": "fermi-dirac",
                "width_hartree": config.smearing.width_hartree,
            }
        ),
        "spin": (
            None
            if config.spin is None
            else {
                "mode": config.spin.mode,
                "magnetization": config.spin.magnetization,
                "initial_magnetization": config.spin.initial_magnetization,
                "magnetization_mixing_beta": config.spin.magnetization_mixing_beta,
            }
        ),
        "davidson": {
            "max_iterations": config.davidson.max_iterations,
            "tolerance": config.davidson.tolerance,
            "max_subspace_size": config.davidson.max_subspace_size,
            "preconditioner_floor": config.davidson.preconditioner_floor,
        },
        "batch_policy": config.batch_policy(),
    }


def periodic_scf_execution_settings(
    config: PeriodicSCFConfig | None = None,
) -> dict[str, object]:
    """Return settings that bind a frozen execution context to periodic SCF.

    Args:
        config: Exact SCF controls. Defaults to ``PeriodicSCFConfig``.

    Returns:
        Mapping suitable for ``build_execution_context(settings_override=...)``.
    """

    scf_config = PeriodicSCFConfig() if config is None else config
    return {"periodic_scf": _config_payload(scf_config)}


def _solver_identity(config_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "davidson": dict(config_payload["davidson"]),
        "scf": {
            "density_tolerance": config_payload["density_tolerance"],
            "energy_tolerance_hartree": config_payload["energy_tolerance_hartree"],
            "max_iterations": config_payload["max_iterations"],
            "min_iterations": config_payload["min_iterations"],
            "mixer": config_payload["mixer"],
            "mixing_beta": config_payload["mixing_beta"],
            "orbital_tolerance": config_payload["orbital_tolerance"],
            "adaptive_eigensolver_tolerance": config_payload["adaptive_eigensolver_tolerance"],
            "initial_eigensolver_tolerance": config_payload["initial_eigensolver_tolerance"],
            "eigensolver_tolerance_scale": config_payload["eigensolver_tolerance_scale"],
            "smearing": config_payload["smearing"],
            "spin": config_payload["spin"],
        },
    }


def _array_input_identity(values: object) -> dict[str, object]:
    try:
        array = np.ascontiguousarray(np.asarray(values))
    except (TypeError, ValueError) as error:
        msg = "periodic checkpoint initial state must contain concrete arrays"
        raise ValueError(msg) from error
    if array.dtype.hasobject or array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        msg = "periodic checkpoint initial arrays must be finite numeric values"
        raise ValueError(msg)
    header = {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
    }
    digest = sha256_bytes(canonical_json_bytes(header) + b"\0" + array.tobytes())
    return {**header, "sha256": digest}


def periodic_scf_initialization_identity(
    *,
    initial_density: object | None = None,
    initial_coefficients: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build a path-independent identity for fresh periodic-SCF initialization.

    Args:
        initial_density: Optional caller-supplied density grid.
        initial_coefficients: Optional caller-supplied coefficient stack per
            explicit k-point.

    Returns:
        Canonical initialization identity for an execution contract.
    """

    density_identity: object = "uniform-electron-count-over-cell-volume"
    if initial_density is not None:
        density_identity = {
            "kind": "caller-supplied-density",
            **_array_input_identity(initial_density),
        }
    coefficient_identity: object = "lowest-kinetic-active-plane-waves"
    if initial_coefficients is not None:
        coefficient_identity = {
            "kind": "caller-supplied-explicit-kpoint-coefficients",
            "lanes": [_array_input_identity(values) for values in initial_coefficients],
        }
    return {
        "density": density_identity,
        "orbitals": coefficient_identity,
        "random_seed": None,
    }


def _validate_execution_calculation_binding(
    identity: PeriodicSCFExecutionIdentity,
    calculation: Mapping[str, object],
) -> None:
    contract = identity.execution_contract
    settings = contract.get("settings_override")
    calculation_config = calculation.get("config")
    environment = contract.get("environment")
    eigensolve = calculation.get("eigensolve")
    environment_matches = isinstance(environment, Mapping) and (
        environment.get("selected_device") == calculation.get("selected_device")
        and environment.get("precision") == calculation.get("precision")
        and isinstance(eigensolve, Mapping)
        and all(environment.get(key) == value for key, value in eigensolve.items())
    )
    if (
        contract.get("command_kind") != PERIODIC_SCF_COMMAND_KIND
        or not isinstance(settings, Mapping)
        or settings.get("periodic_scf") != calculation_config
        or not isinstance(calculation_config, Mapping)
        or contract.get("solver") != _solver_identity(calculation_config)
        or not environment_matches
    ):
        msg = (
            "periodic checkpoint execution contract is not bound to the exact "
            "SCF and batch settings"
        )
        raise ArtifactIntegrityError(msg)


def _validate_initialization_binding(
    identity: PeriodicSCFExecutionIdentity,
    *,
    initial_density: object | None,
    initial_coefficients: Sequence[object] | None,
) -> None:
    observed = periodic_scf_initialization_identity(
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
    )
    if identity.execution_contract.get("initialization") != observed:
        msg = "periodic checkpoint initialization does not match its execution contract"
        raise ArtifactIntegrityError(msg)


def _single_pseudopotential_payload(
    pseudo: PseudopotentialData,
) -> dict[str, object]:
    """Return one canonical periodic GTH pseudopotential payload."""

    if pseudo.format != PseudopotentialFormat.GTH:
        msg = "periodic SCF checkpoints require a GTH pseudopotential"
        raise ValueError(msg)
    return {
        "element": pseudo.element,
        "format": pseudo.format.value,
        "valence_charge": pseudo.valence_charge,
        "gth_rloc": pseudo.gth_rloc,
        "gth_coefficients": list(pseudo.gth_coefficients),
        "gth_channels": [
            {
                "angular_momentum": channel.angular_momentum,
                "radius": channel.radius,
                "coupling_matrix": [list(row) for row in channel.coupling_matrix],
            }
            for channel in pseudo.gth_channels
        ],
    }


def _pseudopotential_payload(system: PeriodicDFTSystem) -> dict[str, object]:
    """Return a deduplicated species table and ordered per-ion assignments."""

    species: list[dict[str, object]] = []
    species_by_payload: dict[bytes, int] = {}
    atom_species: list[int] = []
    for pseudo in system.pseudopotentials:
        payload = _single_pseudopotential_payload(pseudo)
        identity = canonical_json_bytes(payload)
        species_index = species_by_payload.get(identity)
        if species_index is None:
            species_index = len(species)
            species_by_payload[identity] = species_index
            species.append(payload)
        atom_species.append(species_index)
    return {
        "species": species,
        "atom_species": atom_species,
    }


def _upgrade_legacy_calculation_contract(
    calculation: Mapping[str, object],
) -> dict[str, object]:
    """Upgrade one homogeneous v1 calculation identity to the v2 species form."""

    upgraded = json.loads(canonical_json_bytes(dict(calculation)))
    if upgraded.get("schema_version") != _LEGACY_CALCULATION_CONTRACT_SCHEMA:
        return upgraded
    system = upgraded.get("system")
    if not isinstance(system, dict) or not isinstance(system.get("pseudopotential"), dict):
        msg = "legacy periodic SCF calculation contract is incomplete"
        raise ArtifactIntegrityError(msg)
    positions = system.get("positions_bohr")
    if not isinstance(positions, list) or not positions:
        msg = "legacy periodic SCF calculation positions are incomplete"
        raise ArtifactIntegrityError(msg)
    pseudo = system.pop("pseudopotential")
    system["pseudopotentials"] = {
        "species": [pseudo],
        "atom_species": [0] * len(positions),
    }
    upgraded["schema_version"] = _CALCULATION_CONTRACT_SCHEMA
    return upgraded


def periodic_scf_calculation_contract(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
) -> dict[str, object]:
    """Build the path-independent calculation contract used by checkpoints.

    Args:
        system: Periodic GTH system.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        n_bands: Computed band count. Fixed occupations default to half the
            electron count; smearing requires additional empty bands.
        config: SCF controls. Defaults to ``PeriodicSCFConfig``.
        xc_functional: Exchange-correlation functional. Only the deterministic
            production PBE path is checkpointable.

    Returns:
        Canonical JSON-compatible calculation settings and physics identity.

    Raises:
        ValueError: If a custom exchange-correlation implementation lacks the
            stable production-PBE identity.
    """

    from mlx_atomistic.dft.gga import ProductionPBEExchangeCorrelation

    scf_config = PeriodicSCFConfig() if config is None else config
    bands = int(round(system.electron_count / 2.0)) if n_bands is None else n_bands
    if xc_functional is not None and type(xc_functional) is not (ProductionPBEExchangeCorrelation):
        msg = "periodic checkpointing supports only the stable production PBE path"
        raise ValueError(msg)
    return {
        "schema_version": _CALCULATION_CONTRACT_SCHEMA,
        "system": {
            "cell_matrix_bohr": np.asarray(
                system.grid.cell.matrix,
                dtype=np.float64,
            ).tolist(),
            "grid_shape": list(system.grid.shape),
            "positions_bohr": np.asarray(system.positions, dtype=np.float64).tolist(),
            "electron_count": system.electron_count,
            "pseudopotentials": _pseudopotential_payload(system),
        },
        "cutoff_hartree": float(cutoff_hartree),
        "n_bands": int(bands),
        "kpoints": [
            {
                "reduced_kpoint": list(point.vector),
                "weight": point.weight,
                "coordinate_system": point.coordinate_system,
            }
            for point in kpoint_mesh.points
        ],
        "xc_functional": "production-pbe-v1",
        "config": _config_payload(scf_config),
        "precision": "complex64/float32",
        "selected_device": str(mx.default_device()),
        "eigensolve": _eigensolve_provenance(),
    }


def _calculation_fingerprint(contract: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(contract)))
