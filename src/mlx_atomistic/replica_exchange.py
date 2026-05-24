"""Replica exchange molecular dynamics driver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, isfinite, sqrt
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.forcefields import NonbondedPotential, SoftCoreNonbondedPotential
from mlx_atomistic.md import (
    DistanceConstraints,
    ForceTerm,
    LangevinThermostat,
    NeighborListManager,
    RuntimeReporter,
    SimulationConfig,
    SimulationState,
    _energy_forces_from_terms,
    simulate_nvt,
)


@dataclass(frozen=True)
class SwapAttempt:
    """One attempted exchange between adjacent thermodynamic states."""

    step: int
    left: int
    right: int
    probability: float
    accepted: bool
    log_acceptance: float
    random_value: float


@dataclass(frozen=True)
class ReplicaExchangeResult:
    """Final states and diagnostics from a replica exchange run."""

    replica_states: tuple[SimulationState, ...]
    temperatures: tuple[float, ...]
    lambdas: tuple[float | None, ...]
    swap_attempts: tuple[SwapAttempt, ...]
    accepted_swaps: int
    energy_history: mx.array
    state_index_history: mx.array


def metropolis_acceptance_probability(
    energy_left_at_left: float,
    energy_right_at_right: float,
    energy_left_at_right: float,
    energy_right_at_left: float,
    beta_left: float,
    beta_right: float,
) -> tuple[float, float]:
    """Return generalized replica-exchange Metropolis probability and log value."""

    log_acceptance = (
        beta_left * energy_left_at_left
        + beta_right * energy_right_at_right
        - beta_left * energy_left_at_right
        - beta_right * energy_right_at_left
    )
    if log_acceptance >= 0.0:
        return 1.0, log_acceptance
    return exp(log_acceptance), log_acceptance


def temperature_exchange_probability(
    energy_left: float,
    energy_right: float,
    beta_left: float,
    beta_right: float,
) -> tuple[float, float]:
    """Return the temperature-exchange Metropolis probability."""

    return metropolis_acceptance_probability(
        energy_left,
        energy_right,
        energy_right,
        energy_left,
        beta_left,
        beta_right,
    )


def simulate_replica_exchange(
    initial_states: list[SimulationState] | tuple[SimulationState, ...],
    force_terms: ForceTerm
    | list[ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]]
    | tuple[ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...], ...],
    *,
    temperatures: list[float] | tuple[float, ...],
    lambdas: list[float | None] | tuple[float | None, ...] | None = None,
    cell: Cell | None = None,
    config: SimulationConfig | None = None,
    neighbor_manager: NeighborListManager | None = None,
    constraints: DistanceConstraints | None = None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None = None,
    swap_interval: int = 10,
    thermostat_friction: float = 1.0,
    seed: int | None = None,
) -> ReplicaExchangeResult:
    """Run adjacent-pair temperature or Hamiltonian replica exchange.

    `force_terms` may be one shared force-term set or one explicit nested force-term
    set per replica, e.g. `[(term_a,), (term_b,)]`.
    When `lambdas` is provided with a shared lambda-scalable nonbonded term, per-replica
    soft-core nonbonded terms are created for Hamiltonian exchange.
    """

    if config is None:
        config = SimulationConfig()
    _validate_supported_runtime_inputs(
        neighbor_manager=neighbor_manager,
        constraints=constraints,
        reporters=reporters,
    )
    if swap_interval <= 0:
        msg = "swap_interval must be positive"
        raise ValueError(msg)
    states = tuple(initial_states)
    if len(states) < 2:
        msg = "replica exchange requires at least two replicas"
        raise ValueError(msg)
    temps = tuple(float(temperature) for temperature in temperatures)
    if len(temps) != len(states):
        msg = "temperatures must match the number of replicas"
        raise ValueError(msg)
    if any((not isfinite(temperature) or temperature <= 0.0) for temperature in temps):
        msg = "temperatures must be finite and positive"
        raise ValueError(msg)
    lambda_values = _normalize_lambdas(lambdas, len(states))
    term_sets = _replica_force_terms(force_terms, len(states), lambda_values)
    rng = np.random.default_rng(seed)

    energy_history = [_state_energies(states, term_sets, cell=cell)]
    state_indices = list(range(len(states)))
    state_index_history = [list(state_indices)]
    attempts: list[SwapAttempt] = []
    current_states = states

    completed = 0
    swap_round = 0
    while completed < config.steps:
        chunk_steps = min(swap_interval, config.steps - completed)
        current_states = tuple(
            _run_replica_chunk(
                state,
                term_sets[index],
                temperature=temps[index],
                cell=cell,
                config=config,
                steps=chunk_steps,
                seed=None if seed is None else seed + index + 1009 * completed,
                thermostat_friction=thermostat_friction,
            )
            for index, state in enumerate(current_states)
        )
        completed += chunk_steps
        energies = _state_energies(current_states, term_sets, cell=cell)
        current_states, state_indices, energies, chunk_attempts = _attempt_adjacent_swaps(
            current_states,
            state_indices,
            energies,
            term_sets,
            temps,
            config.boltzmann_constant,
            cell=cell,
            rng=rng,
            step=config.initial_step + completed,
            parity=swap_round % 2,
        )
        swap_round += 1
        attempts.extend(chunk_attempts)
        energy_history.append(energies)
        state_index_history.append(list(state_indices))

    return ReplicaExchangeResult(
        replica_states=current_states,
        temperatures=temps,
        lambdas=lambda_values,
        swap_attempts=tuple(attempts),
        accepted_swaps=sum(1 for attempt in attempts if attempt.accepted),
        energy_history=mx.array(np.asarray(energy_history, dtype=np.float32)),
        state_index_history=mx.array(np.asarray(state_index_history, dtype=np.int32)),
    )


def _run_replica_chunk(
    state: SimulationState,
    terms: tuple[ForceTerm, ...],
    *,
    temperature: float,
    cell: Cell | None,
    config: SimulationConfig,
    steps: int,
    seed: int | None,
    thermostat_friction: float,
) -> SimulationState:
    chunk_config = replace(
        config,
        steps=steps,
        initial_step=state.step,
        initial_time=state.time,
    )
    result = simulate_nvt(
        state.positions,
        state.velocities,
        masses=state.masses,
        cell=cell,
        force_terms=terms,
        config=chunk_config,
        thermostat=LangevinThermostat(
            temperature=temperature,
            friction=thermostat_friction,
            seed=seed,
        ),
    )
    return result.final_state


def _attempt_adjacent_swaps(
    states: tuple[SimulationState, ...],
    state_indices: list[int],
    energies: list[float],
    term_sets: tuple[tuple[ForceTerm, ...], ...],
    temperatures: tuple[float, ...],
    boltzmann_constant: float,
    *,
    cell: Cell | None,
    rng: np.random.Generator,
    step: int,
    parity: int,
) -> tuple[tuple[SimulationState, ...], list[int], list[float], list[SwapAttempt]]:
    mutable_states = list(states)
    mutable_indices = list(state_indices)
    mutable_energies = list(energies)
    attempts: list[SwapAttempt] = []
    for left in range(parity, len(states) - 1, 2):
        right = left + 1
        beta_left = 1.0 / (boltzmann_constant * temperatures[left])
        beta_right = 1.0 / (boltzmann_constant * temperatures[right])
        energy_left_at_right = _potential_energy(
            mutable_states[right].positions,
            term_sets[left],
            cell=cell,
        )
        energy_right_at_left = _potential_energy(
            mutable_states[left].positions,
            term_sets[right],
            cell=cell,
        )
        probability, log_acceptance = metropolis_acceptance_probability(
            mutable_energies[left],
            mutable_energies[right],
            energy_left_at_right,
            energy_right_at_left,
            beta_left,
            beta_right,
        )
        random_value = float(rng.random())
        accepted = random_value < probability
        if accepted:
            left_state = _rescale_state_temperature(
                mutable_states[right],
                old_temperature=temperatures[right],
                new_temperature=temperatures[left],
                terms=term_sets[left],
                cell=cell,
            )
            right_state = _rescale_state_temperature(
                mutable_states[left],
                old_temperature=temperatures[left],
                new_temperature=temperatures[right],
                terms=term_sets[right],
                cell=cell,
            )
            mutable_states[left], mutable_states[right] = left_state, right_state
            mutable_indices[left], mutable_indices[right] = (
                mutable_indices[right],
                mutable_indices[left],
            )
            mutable_energies[left], mutable_energies[right] = (
                energy_left_at_right,
                energy_right_at_left,
            )
        attempts.append(
            SwapAttempt(
                step=step,
                left=left,
                right=right,
                probability=probability,
                accepted=accepted,
                log_acceptance=log_acceptance,
                random_value=random_value,
            )
        )
    return tuple(mutable_states), mutable_indices, mutable_energies, attempts


def _rescale_state_temperature(
    state: SimulationState,
    *,
    old_temperature: float,
    new_temperature: float,
    terms: tuple[ForceTerm, ...],
    cell: Cell | None,
) -> SimulationState:
    scale = sqrt(new_temperature / old_temperature)
    velocities = state.velocities * scale
    _, forces = _energy_forces_from_terms(state.positions, terms, cell=cell, pairs=None)
    return replace(state, velocities=velocities, forces=forces)


def _state_energies(
    states: tuple[SimulationState, ...],
    term_sets: tuple[tuple[ForceTerm, ...], ...],
    *,
    cell: Cell | None,
) -> list[float]:
    return [
        _potential_energy(state.positions, terms, cell=cell)
        for state, terms in zip(states, term_sets, strict=True)
    ]


def _potential_energy(
    positions: mx.array,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
) -> float:
    energy, _ = _energy_forces_from_terms(positions, terms, cell=cell, pairs=None)
    return float(np.asarray(energy))


def _normalize_lambdas(
    lambdas: list[float | None] | tuple[float | None, ...] | None,
    count: int,
) -> tuple[float | None, ...]:
    if lambdas is None:
        return tuple(None for _ in range(count))
    values = tuple(None if value is None else float(value) for value in lambdas)
    if len(values) != count:
        msg = "lambdas must match the number of replicas"
        raise ValueError(msg)
    if any(
        value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0)
        for value in values
    ):
        msg = "lambdas must be finite values in [0, 1]"
        raise ValueError(msg)
    return values


def _replica_force_terms(
    force_terms: ForceTerm
    | list[ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]]
    | tuple[ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...], ...],
    count: int,
    lambdas: tuple[float | None, ...],
) -> tuple[tuple[ForceTerm, ...], ...]:
    if _looks_like_per_replica_terms(force_terms, count):
        term_sets = tuple(_as_term_tuple(term_set) for term_set in force_terms)  # type: ignore[arg-type]
    else:
        term_sets = tuple(_as_term_tuple(force_terms) for _ in range(count))  # type: ignore[arg-type]
    if any(value is not None for value in lambdas):
        return tuple(
            _lambda_scaled_terms(term_set, 1.0 if lambda_value is None else lambda_value)
            for term_set, lambda_value in zip(term_sets, lambdas, strict=True)
        )
    return term_sets


def _looks_like_per_replica_terms(force_terms: object, count: int) -> bool:
    if not isinstance(force_terms, (list, tuple)) or len(force_terms) != count:
        return False
    return any(isinstance(item, (list, tuple)) for item in force_terms)


def _as_term_tuple(
    force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...],
) -> tuple[ForceTerm, ...]:
    if isinstance(force_terms, tuple):
        return force_terms
    if isinstance(force_terms, list):
        return tuple(force_terms)
    return (force_terms,)


def _lambda_scaled_terms(
    force_terms: tuple[ForceTerm, ...],
    lambda_value: float,
) -> tuple[ForceTerm, ...]:
    scaled: list[ForceTerm] = []
    for term in force_terms:
        if isinstance(term, SoftCoreNonbondedPotential):
            scaled.append(
                replace(
                    term,
                    lambda_lj=lambda_value,
                    lambda_electrostatics=lambda_value,
                )
            )
        elif isinstance(term, NonbondedPotential):
            scaled.append(
                SoftCoreNonbondedPotential(
                    term,
                    lambda_lj=lambda_value,
                    lambda_electrostatics=lambda_value,
                )
            )
        else:
            scaled.append(term)
    return tuple(scaled)


def replica_exchange_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return normalized replica-exchange artifact metadata."""

    payload = metadata.get("replica_exchange", {})
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = "replica_exchange metadata must be an object"
        raise ValueError(msg)
    _validate_replica_exchange_metadata_payload(payload, error_cls=ValueError)
    return dict(payload)


def _validate_supported_runtime_inputs(
    *,
    neighbor_manager: NeighborListManager | None,
    constraints: DistanceConstraints | None,
    reporters: RuntimeReporter | list[RuntimeReporter] | tuple[RuntimeReporter, ...] | None,
) -> None:
    if neighbor_manager is not None:
        msg = "replica exchange does not yet support neighbor_manager runtime inputs"
        raise ValueError(msg)
    if constraints is not None:
        msg = "replica exchange does not yet support constrained runtime inputs"
        raise ValueError(msg)
    if reporters is not None:
        msg = "replica exchange does not yet support reporters"
        raise ValueError(msg)


def _validate_replica_exchange_metadata_payload(
    payload: dict[str, Any],
    *,
    error_cls: type[Exception],
) -> None:
    if "swap_interval" in payload:
        try:
            interval = int(payload["swap_interval"])
        except (TypeError, ValueError) as err:
            msg = "replica_exchange swap_interval must be a positive integer"
            raise error_cls(msg) from err
        if interval <= 0 or float(interval) != float(payload["swap_interval"]):
            msg = "replica_exchange swap_interval must be a positive integer"
            raise error_cls(msg)
    temperatures = _metadata_float_sequence(payload, "temperatures", error_cls=error_cls)
    lambdas = _metadata_float_sequence(payload, "lambdas", error_cls=error_cls)
    if temperatures is not None and (
        len(temperatures) < 2
        or any(not isfinite(value) or value <= 0.0 for value in temperatures)
    ):
        msg = "replica_exchange temperatures must contain finite positive values"
        raise error_cls(msg)
    if lambdas is not None and (
        len(lambdas) < 2
        or any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in lambdas)
    ):
        msg = "replica_exchange lambdas must contain finite values in [0, 1]"
        raise error_cls(msg)
    if temperatures is not None and lambdas is not None and len(temperatures) != len(lambdas):
        msg = "replica_exchange temperatures and lambdas must have matching lengths"
        raise error_cls(msg)


def _metadata_float_sequence(
    payload: dict[str, Any],
    name: str,
    *,
    error_cls: type[Exception],
) -> tuple[float, ...] | None:
    if name not in payload:
        return None
    raw_values = payload[name]
    if not isinstance(raw_values, (list, tuple)):
        msg = f"replica_exchange {name} must be a list"
        raise error_cls(msg)
    try:
        return tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as err:
        msg = f"replica_exchange {name} must contain numeric values"
        raise error_cls(msg) from err


def replica_exchange_initial_state(positions, velocities, masses) -> SimulationState:
    """Build an initial replica state with zero placeholder forces."""

    positions_array = as_mx_array(positions)
    velocities_array = as_mx_array(velocities)
    return SimulationState(
        positions=positions_array,
        velocities=velocities_array,
        masses=as_mx_array(masses),
        forces=mx.zeros_like(positions_array),
    )
