"""Construction and normalization for production nonbonded force terms."""

from __future__ import annotations

from dataclasses import dataclass, fields

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array
from mlx_atomistic.interaction_engine import _interaction32_topology_digest
from mlx_atomistic.nonbonded import (
    NonbondedBackend,
    NonbondedElectrostatics,
    NonbondedExecutionConfig,
)
from mlx_atomistic.pme import PMEConfig, PMEExecutionPlan
from mlx_atomistic.topology import Topology


@dataclass(frozen=True)
class _NonbondedConstructionSpec:
    """Raw inputs required to construct a production nonbonded force term."""

    sigma: object
    epsilon: object
    charges: object
    coulomb_constant: float
    cutoff: float | None
    switch_distance: float | None
    topology: Topology | None
    lj_one_four_scale: float
    coulomb_one_four_scale: float
    exception_pairs: object
    exception_charge_products: object | None
    exception_sigma: object | None
    exception_epsilon: object | None
    atom_types: object | None
    nbfix_pairs: object
    nbfix_sigma: object | None
    nbfix_epsilon: object | None
    nbfix_type_pairs: object
    nbfix_type_sigma: object | None
    nbfix_type_epsilon: object | None
    backend: NonbondedBackend
    electrostatics: NonbondedElectrostatics
    pme_config: PMEConfig | None
    pme_plan: PMEExecutionPlan | None
    tile_size: int
    memory_budget_bytes: int | None
    lambda_lj: float
    lambda_electrostatics: float
    use_dispersion_correction: bool


@dataclass(frozen=True)
class _ExceptionState:
    pairs_numpy: np.ndarray
    pair_set: set[tuple[int, int]]
    pairs: mx.array
    charge_products: mx.array
    sigma: mx.array
    epsilon: mx.array


@dataclass(frozen=True)
class _NBFixState:
    pairs_numpy: np.ndarray
    pairs: mx.array
    sigma: mx.array
    epsilon: mx.array


@dataclass(frozen=True)
class _TypeNBFixState:
    pairs: np.ndarray
    sigma: mx.array
    epsilon: mx.array
    atom_type_ids: mx.array
    pair_ids: mx.array
    type_count: int
    sigma_table: mx.array
    epsilon_table: mx.array


@dataclass(frozen=True)
class _TopologyCorrectionState:
    correction_pairs_numpy: np.ndarray
    correction_pairs: mx.array
    correction_charge_products: mx.array
    one_four_pairs: mx.array
    one_four_charge_products: mx.array
    lj_one_four_pairs_numpy: np.ndarray
    lj_one_four_pairs: mx.array
    sparse_pairs: mx.array
    sparse_charge_products: mx.array
    sparse_lj_sigma: mx.array
    sparse_lj_epsilon: mx.array
    lj_exclusion_offsets: mx.array
    lj_exclusion_right: mx.array
    lj_one_four_offsets: mx.array
    lj_one_four_right: mx.array
    exceptions_excluded_by_topology: bool


@dataclass(frozen=True)
class _PreparedNonbondedState:
    """Normalized state installed on ``NonbondedPotential`` exactly once."""

    sigma: mx.array
    epsilon: mx.array
    charges: mx.array
    exception_pairs: mx.array
    exception_charge_products: mx.array
    exception_sigma: mx.array
    exception_epsilon: mx.array
    nbfix_pairs: mx.array
    nbfix_sigma: mx.array
    nbfix_epsilon: mx.array
    nbfix_type_pairs: np.ndarray
    nbfix_type_sigma: mx.array
    nbfix_type_epsilon: mx.array
    _atom_type_ids: mx.array
    _nbfix_type_pair_ids: mx.array
    _nbfix_type_count: int
    _nbfix_type_sigma_table: mx.array
    _nbfix_type_epsilon_table: mx.array
    _exception_pair_set: frozenset[tuple[int, int]]
    _exception_pair_codes: np.ndarray
    _ewald_correction_pairs_cache: mx.array
    _ewald_correction_charge_products: mx.array
    _ewald_one_four_pairs_cache: mx.array
    _ewald_one_four_charge_products: mx.array
    _sparse_correction_pairs: mx.array
    _sparse_correction_charge_products: mx.array
    _sparse_correction_lj_sigma: mx.array
    _sparse_correction_lj_epsilon: mx.array
    _aligned_lj_exclusion_pairs: mx.array
    _aligned_lj_one_four_pairs: mx.array
    _interaction32_topology_digest: str
    _aligned_lj_exclusion_offsets: mx.array
    _aligned_lj_exclusion_right: mx.array
    _aligned_lj_one_four_offsets: mx.array
    _aligned_lj_one_four_right: mx.array
    _exceptions_excluded_by_topology: bool
    backend: NonbondedBackend
    electrostatics: NonbondedElectrostatics
    tile_size: int
    memory_budget_bytes: int | None
    lambda_lj: float
    lambda_electrostatics: float
    analytic_virial_supported: bool

    def install(self, target: object) -> None:
        """Install normalized fields on a frozen public facade."""

        for state_field in fields(self):
            object.__setattr__(target, state_field.name, getattr(self, state_field.name))


def _encoded_pairs(pairs: set[tuple[int, int]], n_atoms: int) -> np.ndarray:
    if not pairs:
        return np.empty((0,), dtype=np.int64)
    array = np.asarray(tuple(pairs), dtype=np.int64)
    left = np.minimum(array[:, 0], array[:, 1])
    right = np.maximum(array[:, 0], array[:, 1])
    return np.sort(left * np.int64(n_atoms) + right)


def _left_pair_csr(pairs: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return offsets and right atoms grouped by normalized left atom."""

    pair_array = np.asarray(pairs, dtype=np.int32)
    if pair_array.size == 0:
        return (
            np.zeros((n_atoms + 1,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    left = np.minimum(pair_array[:, 0], pair_array[:, 1])
    right = np.maximum(pair_array[:, 0], pair_array[:, 1])
    order = np.lexsort((right, left))
    left = left[order]
    right = right[order]
    counts = np.bincount(left, minlength=n_atoms)
    offsets = np.empty((n_atoms + 1,), dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    return offsets, right.astype(np.int32, copy=False)


def _normalize_base_parameters(
    spec: _NonbondedConstructionSpec,
) -> tuple[mx.array, mx.array, mx.array]:
    sigma = as_mx_array(spec.sigma)
    epsilon = as_mx_array(spec.epsilon)
    charges = as_mx_array(spec.charges)
    if sigma.ndim != 1 or epsilon.ndim != 1 or charges.ndim != 1:
        msg = "sigma, epsilon, and charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if sigma.shape != epsilon.shape or sigma.shape != charges.shape:
        msg = "sigma, epsilon, and charges must have matching shapes"
        raise ValueError(msg)
    if bool(np.any(np.asarray(sigma) <= 0.0)):
        msg = "sigma values must be positive"
        raise ValueError(msg)
    if bool(np.any(np.asarray(epsilon) < 0.0)):
        msg = "epsilon values must be non-negative"
        raise ValueError(msg)
    if not bool(np.all(np.isfinite(np.asarray(charges)))):
        msg = "charges must be finite"
        raise ValueError(msg)
    if not np.isfinite(float(spec.coulomb_constant)):
        msg = "coulomb_constant must be finite"
        raise ValueError(msg)
    if spec.topology is not None and spec.topology.n_atoms != sigma.shape[0]:
        msg = "topology.n_atoms must match nonbonded parameter length"
        raise ValueError(msg)
    if spec.cutoff is not None and spec.cutoff <= 0.0:
        msg = "cutoff must be positive"
        raise ValueError(msg)
    if spec.switch_distance is not None:
        if spec.cutoff is None:
            msg = "switch_distance requires a cutoff"
            raise ValueError(msg)
        if spec.switch_distance < 0.0 or spec.switch_distance >= spec.cutoff:
            msg = "switch_distance must be non-negative and smaller than cutoff"
            raise ValueError(msg)
    if spec.lj_one_four_scale < 0.0 or spec.coulomb_one_four_scale < 0.0:
        msg = "1-4 scaling factors must be non-negative"
        raise ValueError(msg)
    for name, value in (
        ("lambda_lj", spec.lambda_lj),
        ("lambda_electrostatics", spec.lambda_electrostatics),
    ):
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            msg = f"{name} must be finite and in [0, 1]"
            raise ValueError(msg)
    return sigma, epsilon, charges


def _normalize_exception_state(
    spec: _NonbondedConstructionSpec,
    atom_count: int,
) -> _ExceptionState:
    pairs = np.asarray(spec.exception_pairs, dtype=np.int32)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        msg = "exception_pairs must have shape (n, 2)"
        raise ValueError(msg)
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= atom_count)):
        msg = "exception_pairs contain atom indices outside [0, n_atoms)"
        raise ValueError(msg)
    count = pairs.shape[0]
    charge_products = (
        np.asarray([], dtype=np.float32)
        if spec.exception_charge_products is None
        else np.asarray(spec.exception_charge_products, dtype=np.float32)
    )
    sigma = (
        np.asarray([], dtype=np.float32)
        if spec.exception_sigma is None
        else np.asarray(spec.exception_sigma, dtype=np.float32)
    )
    epsilon = (
        np.asarray([], dtype=np.float32)
        if spec.exception_epsilon is None
        else np.asarray(spec.exception_epsilon, dtype=np.float32)
    )
    for name, values in (
        ("exception_charge_products", charge_products),
        ("exception_sigma", sigma),
        ("exception_epsilon", epsilon),
    ):
        if count == 0 and values.size == 0:
            values.resize((0,), refcheck=False)
        if values.shape != (count,):
            msg = f"{name} must have shape ({count},)"
            raise ValueError(msg)
    if np.any(sigma < 0.0) or np.any(epsilon < 0.0):
        msg = "exception sigma and epsilon values must be non-negative"
        raise ValueError(msg)
    pair_set = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs.tolist()}
    return _ExceptionState(
        pairs_numpy=pairs,
        pair_set=pair_set,
        pairs=mx.array(pairs, dtype=mx.int32),
        charge_products=as_mx_array(charge_products),
        sigma=as_mx_array(sigma),
        epsilon=as_mx_array(epsilon),
    )


def _normalize_nbfix_state(
    spec: _NonbondedConstructionSpec,
    atom_count: int,
) -> _NBFixState:
    pairs = np.asarray(spec.nbfix_pairs, dtype=np.int32)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        msg = "nbfix_pairs must have shape (n, 2)"
        raise ValueError(msg)
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= atom_count)):
        msg = "nbfix_pairs contain atom indices outside [0, n_atoms)"
        raise ValueError(msg)
    pair_set: set[tuple[int, int]] = set()
    for left, right in pairs.tolist():
        if left == right:
            msg = "nbfix_pairs must not contain self pairs"
            raise ValueError(msg)
        pair = (min(int(left), int(right)), max(int(left), int(right)))
        if pair in pair_set:
            msg = "nbfix_pairs must not contain duplicate pairs"
            raise ValueError(msg)
        pair_set.add(pair)
    count = pairs.shape[0]
    sigma = (
        np.asarray([], dtype=np.float32)
        if spec.nbfix_sigma is None
        else np.asarray(spec.nbfix_sigma, dtype=np.float32)
    )
    epsilon = (
        np.asarray([], dtype=np.float32)
        if spec.nbfix_epsilon is None
        else np.asarray(spec.nbfix_epsilon, dtype=np.float32)
    )
    for name, values in (("nbfix_sigma", sigma), ("nbfix_epsilon", epsilon)):
        if count == 0 and values.size == 0:
            values.resize((0,), refcheck=False)
        if values.shape != (count,):
            msg = f"{name} must have shape ({count},)"
            raise ValueError(msg)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        msg = "nbfix_sigma values must be finite and positive"
        raise ValueError(msg)
    if np.any(~np.isfinite(epsilon)) or np.any(epsilon < 0.0):
        msg = "nbfix_epsilon values must be finite and non-negative"
        raise ValueError(msg)
    return _NBFixState(
        pairs_numpy=pairs,
        pairs=mx.array(pairs, dtype=mx.int32),
        sigma=as_mx_array(sigma),
        epsilon=as_mx_array(epsilon),
    )


def _normalize_type_nbfix_state(
    spec: _NonbondedConstructionSpec,
    atom_count: int,
) -> _TypeNBFixState:
    pairs = np.asarray(spec.nbfix_type_pairs, dtype=str)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=str)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        msg = "nbfix_type_pairs must have shape (n, 2)"
        raise ValueError(msg)
    count = pairs.shape[0]
    sigma = (
        np.asarray([], dtype=np.float32)
        if spec.nbfix_type_sigma is None
        else np.asarray(spec.nbfix_type_sigma, dtype=np.float32)
    )
    epsilon = (
        np.asarray([], dtype=np.float32)
        if spec.nbfix_type_epsilon is None
        else np.asarray(spec.nbfix_type_epsilon, dtype=np.float32)
    )
    for name, values in (
        ("nbfix_type_sigma", sigma),
        ("nbfix_type_epsilon", epsilon),
    ):
        if count == 0 and values.size == 0:
            values.resize((0,), refcheck=False)
        if values.shape != (count,):
            msg = f"{name} must have shape ({count},)"
            raise ValueError(msg)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        msg = "nbfix_type_sigma values must be finite and positive"
        raise ValueError(msg)
    if np.any(~np.isfinite(epsilon)) or np.any(epsilon < 0.0):
        msg = "nbfix_type_epsilon values must be finite and non-negative"
        raise ValueError(msg)
    if count > 0 and np.any(np.char.str_len(pairs) == 0):
        msg = "nbfix_type_pairs must not contain empty type names"
        raise ValueError(msg)
    seen_pairs: set[tuple[str, str]] = set()
    for left, right in pairs.tolist():
        pair = tuple(sorted((str(left), str(right))))
        if pair in seen_pairs:
            msg = "nbfix_type_pairs must not contain duplicate type pairs"
            raise ValueError(msg)
        seen_pairs.add(pair)

    atom_type_ids = np.empty((0,), dtype=np.int32)
    pair_ids = np.empty((0, 2), dtype=np.int32)
    type_count = 0
    sigma_table = np.empty((0,), dtype=np.float32)
    epsilon_table = np.empty((0,), dtype=np.float32)
    if count > 0:
        if spec.atom_types is None:
            msg = "atom_types are required when nbfix_type_pairs are provided"
            raise ValueError(msg)
        atom_types = np.asarray(spec.atom_types, dtype=str)
        if atom_types.shape != (atom_count,):
            msg = "atom_types must have shape (n_atoms,)"
            raise ValueError(msg)
        type_to_id = {atom_type: index for index, atom_type in enumerate(sorted(set(atom_types)))}
        missing_type_names = sorted(
            {
                str(atom_type)
                for pair in pairs.tolist()
                for atom_type in pair
                if str(atom_type) not in type_to_id
            }
        )
        if missing_type_names:
            msg = "nbfix_type_pairs reference atom types absent from atom_types: " + ", ".join(
                missing_type_names
            )
            raise ValueError(msg)
        atom_type_ids = np.asarray(
            [type_to_id[atom_type] for atom_type in atom_types],
            dtype=np.int32,
        )
        pair_ids = np.asarray(
            [[type_to_id[str(left)], type_to_id[str(right)]] for left, right in pairs.tolist()],
            dtype=np.int32,
        )
        type_count = len(type_to_id)
        table_size = type_count * type_count
        sigma_table = np.zeros((table_size,), dtype=np.float32)
        epsilon_table = np.zeros((table_size,), dtype=np.float32)
        for type_pair_ids, sigma_value, epsilon_value in zip(
            pair_ids,
            sigma,
            epsilon,
            strict=True,
        ):
            left_id, right_id = (int(type_pair_ids[0]), int(type_pair_ids[1]))
            forward = left_id * type_count + right_id
            reverse = right_id * type_count + left_id
            sigma_table[[forward, reverse]] = sigma_value
            epsilon_table[[forward, reverse]] = epsilon_value
    return _TypeNBFixState(
        pairs=pairs,
        sigma=as_mx_array(sigma),
        epsilon=as_mx_array(epsilon),
        atom_type_ids=mx.array(atom_type_ids, dtype=mx.int32),
        pair_ids=mx.array(pair_ids, dtype=mx.int32),
        type_count=type_count,
        sigma_table=mx.array(sigma_table, dtype=mx.float32),
        epsilon_table=mx.array(epsilon_table, dtype=mx.float32),
    )


def _build_topology_correction_state(
    spec: _NonbondedConstructionSpec,
    charges: mx.array,
    exceptions: _ExceptionState,
) -> _TopologyCorrectionState:
    correction_pair_set = set(exceptions.pair_set)
    if spec.topology is not None:
        correction_pair_set.update(spec.topology.exclusion_set)
    correction_pairs_numpy = np.asarray(
        sorted(correction_pair_set),
        dtype=np.int32,
    ).reshape((-1, 2))
    excluded_one_four_pairs = set(correction_pair_set)
    one_four_pairs_numpy = np.asarray(
        (
            []
            if spec.topology is None or spec.coulomb_one_four_scale == 1.0
            else sorted(spec.topology.one_four_set - excluded_one_four_pairs)
        ),
        dtype=np.int32,
    ).reshape((-1, 2))
    lj_one_four_pairs_numpy = np.asarray(
        (
            []
            if spec.topology is None or spec.lj_one_four_scale == 1.0
            else sorted(spec.topology.one_four_set - excluded_one_four_pairs)
        ),
        dtype=np.int32,
    ).reshape((-1, 2))
    atom_count = int(charges.shape[0])
    exclusion_offsets, exclusion_right = _left_pair_csr(
        correction_pairs_numpy,
        atom_count,
    )
    one_four_offsets, one_four_right = _left_pair_csr(
        lj_one_four_pairs_numpy,
        atom_count,
    )
    correction_pairs = mx.array(correction_pairs_numpy, dtype=mx.int32)
    one_four_pairs = mx.array(one_four_pairs_numpy, dtype=mx.int32)
    lj_one_four_pairs = mx.array(lj_one_four_pairs_numpy, dtype=mx.int32)
    correction_charge_products = -(
        charges[correction_pairs[:, 0]] * charges[correction_pairs[:, 1]]
    )
    one_four_charge_products = (
        (spec.coulomb_one_four_scale - 1.0)
        * charges[one_four_pairs[:, 0]]
        * charges[one_four_pairs[:, 1]]
    )
    sparse_pairs = mx.concatenate(
        (correction_pairs, exceptions.pairs, one_four_pairs),
        axis=0,
    )
    sparse_charge_products = mx.concatenate(
        (
            correction_charge_products,
            exceptions.charge_products,
            one_four_charge_products,
        ),
        axis=0,
    )
    sparse_lj_sigma = mx.concatenate(
        (
            mx.zeros((correction_pairs_numpy.shape[0],), dtype=mx.float32),
            exceptions.sigma,
            mx.zeros((one_four_pairs_numpy.shape[0],), dtype=mx.float32),
        ),
        axis=0,
    )
    sparse_lj_epsilon = mx.concatenate(
        (
            mx.zeros((correction_pairs_numpy.shape[0],), dtype=mx.float32),
            exceptions.epsilon,
            mx.zeros((one_four_pairs_numpy.shape[0],), dtype=mx.float32),
        ),
        axis=0,
    )
    return _TopologyCorrectionState(
        correction_pairs_numpy=correction_pairs_numpy,
        correction_pairs=correction_pairs,
        correction_charge_products=correction_charge_products,
        one_four_pairs=one_four_pairs,
        one_four_charge_products=one_four_charge_products,
        lj_one_four_pairs_numpy=lj_one_four_pairs_numpy,
        lj_one_four_pairs=lj_one_four_pairs,
        sparse_pairs=sparse_pairs,
        sparse_charge_products=sparse_charge_products,
        sparse_lj_sigma=sparse_lj_sigma,
        sparse_lj_epsilon=sparse_lj_epsilon,
        lj_exclusion_offsets=mx.array(exclusion_offsets, dtype=mx.int32),
        lj_exclusion_right=mx.array(exclusion_right, dtype=mx.int32),
        lj_one_four_offsets=mx.array(one_four_offsets, dtype=mx.int32),
        lj_one_four_right=mx.array(one_four_right, dtype=mx.int32),
        exceptions_excluded_by_topology=(
            spec.topology is not None and exceptions.pair_set.issubset(spec.topology.exclusion_set)
        ),
    )


def _validated_execution_config(
    spec: _NonbondedConstructionSpec,
    charges: mx.array,
) -> NonbondedExecutionConfig:
    config = NonbondedExecutionConfig(
        backend=spec.backend,
        electrostatics=spec.electrostatics,
        tile_size=spec.tile_size,
        memory_budget_bytes=spec.memory_budget_bytes,
    )
    if (
        float(spec.lambda_lj) < 1.0 or float(spec.lambda_electrostatics) < 1.0
    ) and config.electrostatics != "cutoff":
        msg = "soft-core lambda scaling currently supports cutoff electrostatics only"
        raise ValueError(msg)
    if config.electrostatics == "pme":
        if spec.pme_config is None:
            msg = "PME electrostatics requires pme_config"
            raise ValueError(msg)
        if not np.isfinite(float(spec.pme_config.alpha)) or spec.pme_config.alpha <= 0.0:
            msg = "PME electrostatics requires finite positive pme_config.alpha"
            raise ValueError(msg)
        if spec.pme_config.real_cutoff is not None and (
            not np.isfinite(float(spec.pme_config.real_cutoff))
            or spec.pme_config.real_cutoff <= 0.0
        ):
            msg = "PME electrostatics requires finite positive pme_config.real_cutoff when provided"
            raise ValueError(msg)
        if (
            not np.isfinite(float(spec.pme_config.charge_tolerance))
            or spec.pme_config.charge_tolerance < 0.0
        ):
            msg = "PME electrostatics requires finite non-negative pme_config.charge_tolerance"
            raise ValueError(msg)
        net_charge = float(np.sum(np.asarray(charges, dtype=np.float64), dtype=np.float64))
        if (
            abs(net_charge) > spec.pme_config.charge_tolerance
            and spec.pme_config.background_policy != "uniform_neutralizing_plasma"
        ):
            msg = (
                "PME electrostatics requires a neutral system unless "
                "background_policy='uniform_neutralizing_plasma'; "
                f"net_charge={net_charge:g}"
            )
            raise ValueError(msg)
        if spec.pme_plan is not None:
            if not isinstance(spec.pme_plan, PMEExecutionPlan):
                msg = "pme_plan must be a PMEExecutionPlan instance"
                raise TypeError(msg)
            spec.pme_plan.validate(
                spec.pme_plan.cell,
                config=spec.pme_config,
                coulomb_constant=spec.coulomb_constant,
            )
    elif spec.pme_plan is not None:
        msg = "pme_plan requires electrostatics='pme'"
        raise ValueError(msg)
    if spec.use_dispersion_correction and config.electrostatics != "pme":
        msg = "LJ dispersion correction currently requires PME electrostatics"
        raise ValueError(msg)
    if spec.use_dispersion_correction and spec.switch_distance is not None:
        msg = "analytic LJ dispersion correction does not support switching"
        raise ValueError(msg)
    return config


def _prepare_nonbonded_state(
    spec: _NonbondedConstructionSpec,
) -> _PreparedNonbondedState:
    """Validate raw inputs and lower them into immutable runtime state."""

    sigma, epsilon, charges = _normalize_base_parameters(spec)
    atom_count = int(sigma.shape[0])
    exceptions = _normalize_exception_state(spec, atom_count)
    nbfix = _normalize_nbfix_state(spec, atom_count)
    type_nbfix = _normalize_type_nbfix_state(spec, atom_count)
    topology = _build_topology_correction_state(spec, charges, exceptions)
    config = _validated_execution_config(spec, charges)
    analytic_supported = (
        config.electrostatics in {"cutoff", "pme"}
        and float(spec.lambda_lj) == 1.0
        and float(spec.lambda_electrostatics) == 1.0
        and nbfix.pairs_numpy.shape[0] == 0
        and type_nbfix.pairs.shape[0] == 0
        and (
            config.electrostatics != "pme"
            or (spec.pme_config is not None and spec.pme_config.real_cutoff is not None)
        )
    )
    return _PreparedNonbondedState(
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        exception_pairs=exceptions.pairs,
        exception_charge_products=exceptions.charge_products,
        exception_sigma=exceptions.sigma,
        exception_epsilon=exceptions.epsilon,
        nbfix_pairs=nbfix.pairs,
        nbfix_sigma=nbfix.sigma,
        nbfix_epsilon=nbfix.epsilon,
        nbfix_type_pairs=type_nbfix.pairs,
        nbfix_type_sigma=type_nbfix.sigma,
        nbfix_type_epsilon=type_nbfix.epsilon,
        _atom_type_ids=type_nbfix.atom_type_ids,
        _nbfix_type_pair_ids=type_nbfix.pair_ids,
        _nbfix_type_count=type_nbfix.type_count,
        _nbfix_type_sigma_table=type_nbfix.sigma_table,
        _nbfix_type_epsilon_table=type_nbfix.epsilon_table,
        _exception_pair_set=frozenset(exceptions.pair_set),
        _exception_pair_codes=_encoded_pairs(exceptions.pair_set, atom_count),
        _ewald_correction_pairs_cache=topology.correction_pairs,
        _ewald_correction_charge_products=topology.correction_charge_products,
        _ewald_one_four_pairs_cache=topology.one_four_pairs,
        _ewald_one_four_charge_products=topology.one_four_charge_products,
        _sparse_correction_pairs=topology.sparse_pairs,
        _sparse_correction_charge_products=topology.sparse_charge_products,
        _sparse_correction_lj_sigma=topology.sparse_lj_sigma,
        _sparse_correction_lj_epsilon=topology.sparse_lj_epsilon,
        _aligned_lj_exclusion_pairs=topology.correction_pairs,
        _aligned_lj_one_four_pairs=topology.lj_one_four_pairs,
        _interaction32_topology_digest=_interaction32_topology_digest(
            atom_count,
            topology.correction_pairs_numpy,
            topology.lj_one_four_pairs_numpy,
        ),
        _aligned_lj_exclusion_offsets=topology.lj_exclusion_offsets,
        _aligned_lj_exclusion_right=topology.lj_exclusion_right,
        _aligned_lj_one_four_offsets=topology.lj_one_four_offsets,
        _aligned_lj_one_four_right=topology.lj_one_four_right,
        _exceptions_excluded_by_topology=topology.exceptions_excluded_by_topology,
        backend=config.backend,
        electrostatics=config.electrostatics,
        tile_size=config.tile_size,
        memory_budget_bytes=config.memory_budget_bytes,
        lambda_lj=float(spec.lambda_lj),
        lambda_electrostatics=float(spec.lambda_electrostatics),
        analytic_virial_supported=analytic_supported,
    )
