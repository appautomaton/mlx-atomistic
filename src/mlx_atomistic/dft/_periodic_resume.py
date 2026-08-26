"""Periodic SCF checkpoint capture and resume validation."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from mlx_atomistic.dft._compact import _CompactLaneState
from mlx_atomistic.dft._periodic_models import PeriodicKPointResult
from mlx_atomistic.dft._periodic_state import _PeriodicSCFContinuationState
from mlx_atomistic.dft.grids import RealSpaceGrid
from mlx_atomistic.dft.kpoints import TimeReversalOwnership, _independent_pair
from mlx_atomistic.dft.mixing import LinearMixer, PulayDIISMixer
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis


def _owned_device_copy(values: mx.array, *, dtype: mx.Dtype) -> mx.array:
    source = mx.array(values).astype(dtype)
    copied = (source + mx.zeros_like(source)).astype(dtype)
    mx.eval(copied)
    return copied


def _continuation_state_from_boundary(
    *,
    completed_iteration: int,
    density: mx.array,
    owned_results: Sequence[PeriodicKPointResult],
    previous_energy: float,
    energy_by_term: dict[str, float],
    history: Sequence[dict[str, float | int | str | None]],
    mixer: LinearMixer | PulayDIISMixer,
    ownership: TimeReversalOwnership,
    lineage: tuple[str, ...],
    spin_densities: tuple[mx.array, mx.array] | None = None,
    down_owned_results: Sequence[PeriodicKPointResult] = (),
    magnetization_mixer: LinearMixer | None = None,
) -> _PeriodicSCFContinuationState:
    owned_coefficients, owned_lanes = _capture_owned_results(owned_results)
    if spin_densities is None:
        down_owned_coefficients: tuple[tuple[int, mx.array], ...] = ()
        down_owned_lanes: tuple[dict[str, object], ...] = ()
        magnetization_mixer_state = None
    else:
        if magnetization_mixer is None:
            raise RuntimeError("spin checkpoint requires a magnetization mixer")
        down_owned_coefficients, down_owned_lanes = _capture_owned_results(
            down_owned_results
        )
        magnetization_mixer_state = magnetization_mixer._checkpoint_state()
    return _PeriodicSCFContinuationState(
        completed_iteration=completed_iteration,
        density=_owned_device_copy(density, dtype=mx.float32),
        owned_coefficients=owned_coefficients,
        owned_lanes=owned_lanes,
        previous_energy=float(previous_energy),
        energy_by_term=dict(energy_by_term),
        history=tuple(dict(row) for row in history),
        mixer_state=mixer._checkpoint_state(),
        ownership=ownership.to_dict(),
        lineage=lineage,
        spin_densities=(
            None
            if spin_densities is None
            else (
                _owned_device_copy(spin_densities[0], dtype=mx.float32),
                _owned_device_copy(spin_densities[1], dtype=mx.float32),
            )
        ),
        down_owned_coefficients=down_owned_coefficients,
        down_owned_lanes=down_owned_lanes,
        magnetization_mixer_state=magnetization_mixer_state,
    )


def _capture_owned_results(
    owned_results: Sequence[PeriodicKPointResult],
) -> tuple[tuple[tuple[int, mx.array], ...], tuple[dict[str, object], ...]]:
    owned_coefficients: list[tuple[int, mx.array]] = []
    owned_lanes: list[dict[str, object]] = []
    for result in owned_results:
        if result.explicit_index is None:
            msg = "checkpoint state requires explicit owner indices"
            raise RuntimeError(msg)
        compact = result.eigen._compact_coefficients
        if not isinstance(compact, _CompactLaneState):
            msg = "checkpoint state requires compact owner coefficients"
            raise RuntimeError(msg)
        owned_coefficients.append(
            (
                result.explicit_index,
                _owned_device_copy(compact.values, dtype=mx.complex64),
            )
        )
        owned_lanes.append(
            {
                "owner_index": result.explicit_index,
                "reduced_kpoint": list(result.reduced_kpoint),
                "active_count": result.basis.active_count,
                "basis_fingerprint": result.basis.basis_fingerprint,
                "basis_order_fingerprint": result.basis.order_fingerprint,
                "lane_id": result.basis.lane_id,
            }
        )
    return tuple(owned_coefficients), tuple(owned_lanes)


def _resume_ownership(
    rebuilt: TimeReversalOwnership,
    stored: dict[str, object],
) -> TimeReversalOwnership:
    if rebuilt.to_dict() == stored:
        return rebuilt
    stored_entries = stored.get("entries")
    if not isinstance(stored_entries, list) or len(stored_entries) != len(rebuilt.entries):
        msg = "periodic resume ownership payload is malformed"
        raise ValueError(msg)
    candidate = rebuilt
    for index, rebuilt_entry in enumerate(rebuilt.entries):
        stored_entry = stored_entries[index]
        if not isinstance(stored_entry, dict):
            msg = "periodic resume ownership entry is malformed"
            raise ValueError(msg)
        if (
            rebuilt_entry.role in {"owner", "partner"}
            and stored_entry.get("role") == "independent"
            and stored_entry.get("fallback_reason") == "initial_coefficients_time_reversal_mismatch"
        ):
            candidate = _independent_pair(
                candidate,
                index,
                "initial_coefficients_time_reversal_mismatch",
            )
    if candidate.to_dict() != stored:
        msg = "periodic resume ownership is not a valid stored fallback refinement"
        raise ValueError(msg)
    return candidate


def _validate_resume_metadata(
    state: _PeriodicSCFContinuationState,
    ownership: TimeReversalOwnership,
) -> None:
    if state.completed_iteration <= 0:
        msg = "periodic resume iteration must be positive"
        raise ValueError(msg)
    if state.ownership != ownership.to_dict():
        msg = "periodic resume ownership does not match the rebuilt topology"
        raise ValueError(msg)
    if len(state.history) != state.completed_iteration or any(
        row.get("iteration") != index for index, row in enumerate(state.history, start=1)
    ):
        msg = "periodic resume history does not match its iteration cursor"
        raise ValueError(msg)
    if not np.isfinite(state.previous_energy):
        msg = "periodic resume energy must be finite"
        raise ValueError(msg)
    if (
        not state.history
        or not np.isclose(
            float(state.history[-1]["total_energy_hartree"]),
            state.previous_energy,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(state.energy_by_term.get("total", float("nan"))),
            state.previous_energy,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        msg = "periodic resume energy state is internally inconsistent"
        raise ValueError(msg)


def _restore_density(
    state: _PeriodicSCFContinuationState,
    *,
    grid: RealSpaceGrid,
    electron_count: float,
) -> mx.array:
    density = mx.array(state.density)
    if density.shape != grid.shape or density.dtype != mx.float32:
        msg = "periodic resume density has incompatible shape or dtype"
        raise ValueError(msg)
    density_finite = mx.all(mx.isfinite(density))
    density_minimum = mx.min(density)
    density_count = mx.sum(density) * grid.dv
    mx.eval(density, density_finite, density_minimum, density_count)
    if (
        not bool(density_finite)
        or float(density_minimum) < -1e-7
        or abs(float(density_count) - electron_count) > 1e-4
    ):
        msg = "periodic resume density is non-finite, negative, or misnormalized"
        raise ValueError(msg)
    return _owned_device_copy(density, dtype=mx.float32)


def _validated_lane_map(
    state: _PeriodicSCFContinuationState,
    ownership: TimeReversalOwnership,
) -> dict[int, dict[str, object]]:
    coefficient_map = state.coefficient_map
    if len(coefficient_map) != len(state.owned_coefficients) or set(coefficient_map) != set(
        ownership.owned_indices
    ):
        msg = "periodic resume owner coefficient inventory is inconsistent"
        raise ValueError(msg)
    lane_map = {
        int(lane["owner_index"]): lane
        for lane in state.owned_lanes
        if isinstance(lane, dict) and "owner_index" in lane
    }
    if len(lane_map) != len(state.owned_lanes) or set(lane_map) != set(ownership.owned_indices):
        msg = "periodic resume owner lane inventory is inconsistent"
        raise ValueError(msg)
    return lane_map


def _expected_lane_identity(
    owner_index: int,
    basis: PlaneWaveBasis,
    ownership: TimeReversalOwnership,
) -> dict[str, object]:
    return {
        "owner_index": owner_index,
        "reduced_kpoint": list(ownership.entry_for(owner_index).reduced_kpoint),
        "active_count": basis.active_count,
        "basis_fingerprint": basis.basis_fingerprint,
        "basis_order_fingerprint": basis.order_fingerprint,
        "lane_id": basis.lane_id,
    }


def _restore_owner_states(
    state: _PeriodicSCFContinuationState,
    *,
    bases: Sequence[PlaneWaveBasis],
    ownership: TimeReversalOwnership,
    band_count: int,
) -> dict[int, _CompactLaneState]:
    coefficient_map = state.coefficient_map
    lane_map = _validated_lane_map(state, ownership)
    previous_states: dict[int, _CompactLaneState] = {}
    finite_checks: list[mx.array] = []
    for owner_index in ownership.owned_indices:
        basis = bases[owner_index]
        expected_lane = _expected_lane_identity(owner_index, basis, ownership)
        if lane_map[owner_index] != expected_lane:
            msg = "periodic resume owner lane identity does not match rebuilt bases"
            raise ValueError(msg)
        values = mx.array(coefficient_map[owner_index])
        if values.dtype != mx.complex64 or values.shape != (
            band_count,
            basis.active_count,
        ):
            msg = "periodic resume owner coefficients have incompatible shape or dtype"
            raise ValueError(msg)
        copied = _owned_device_copy(values, dtype=mx.complex64)
        finite_checks.append(mx.all(mx.isfinite(copied)))
        previous_states[owner_index] = basis._state_from_compact(copied)
    mx.eval(*finite_checks)
    if not all(bool(finite) for finite in finite_checks):
        msg = "periodic resume owner coefficients must be finite"
        raise ValueError(msg)
    return previous_states


def _restore_down_owner_states(
    state: _PeriodicSCFContinuationState,
    *,
    bases: Sequence[PlaneWaveBasis],
    ownership: TimeReversalOwnership,
    band_count: int,
) -> dict[int, _CompactLaneState]:
    coefficient_map = state.down_coefficient_map
    if len(coefficient_map) != len(state.down_owned_coefficients) or set(
        coefficient_map
    ) != set(ownership.owned_indices):
        raise ValueError("periodic spin resume down-channel inventory is inconsistent")
    lane_map = {
        int(lane["owner_index"]): lane
        for lane in state.down_owned_lanes
        if isinstance(lane, dict) and "owner_index" in lane
    }
    if len(lane_map) != len(state.down_owned_lanes) or set(lane_map) != set(
        ownership.owned_indices
    ):
        raise ValueError("periodic spin resume down-channel lanes are inconsistent")
    previous_states: dict[int, _CompactLaneState] = {}
    finite_checks: list[mx.array] = []
    for owner_index in ownership.owned_indices:
        basis = bases[owner_index]
        if lane_map[owner_index] != _expected_lane_identity(
            owner_index,
            basis,
            ownership,
        ):
            raise ValueError("periodic spin resume down-channel lane identity differs")
        values = mx.array(coefficient_map[owner_index])
        if values.dtype != mx.complex64 or values.shape != (
            band_count,
            basis.active_count,
        ):
            raise ValueError("periodic spin resume down coefficients are incompatible")
        copied = _owned_device_copy(values, dtype=mx.complex64)
        finite_checks.append(mx.all(mx.isfinite(copied)))
        previous_states[owner_index] = basis._state_from_compact(copied)
    mx.eval(*finite_checks)
    if not all(bool(finite) for finite in finite_checks):
        raise ValueError("periodic spin resume down coefficients must be finite")
    return previous_states


def _restore_continuation_state(
    state: _PeriodicSCFContinuationState,
    *,
    bases: Sequence[PlaneWaveBasis],
    ownership: TimeReversalOwnership,
    band_count: int,
    grid: RealSpaceGrid,
    electron_count: float,
    mixer: LinearMixer | PulayDIISMixer,
) -> tuple[
    mx.array,
    dict[int, _CompactLaneState],
    float,
    list[dict[str, float | int | str | None]],
    dict[str, float],
]:
    _validate_resume_metadata(state, ownership)
    density = _restore_density(state, grid=grid, electron_count=electron_count)
    previous_states = _restore_owner_states(
        state,
        bases=bases,
        ownership=ownership,
        band_count=band_count,
    )
    mixer._restore_checkpoint_state(state.mixer_state, expected_shape=grid.shape)
    return (
        density,
        previous_states,
        float(state.previous_energy),
        [dict(row) for row in state.history],
        dict(state.energy_by_term),
    )


def _restore_spin_continuation_state(
    state: _PeriodicSCFContinuationState,
    *,
    up_bases: Sequence[PlaneWaveBasis],
    down_bases: Sequence[PlaneWaveBasis],
    ownership: TimeReversalOwnership,
    band_count: int,
    grid: RealSpaceGrid,
    electron_count: float,
    expected_magnetization: float | None,
    mixer: LinearMixer | PulayDIISMixer,
    magnetization_mixer: LinearMixer,
) -> tuple[
    mx.array,
    tuple[dict[int, _CompactLaneState], dict[int, _CompactLaneState]],
    tuple[mx.array, mx.array],
    float,
    list[dict[str, float | int | str | None]],
    dict[str, float],
]:
    density, up_states, previous_energy, history, energy_terms = (
        _restore_continuation_state(
            state,
            bases=up_bases,
            ownership=ownership,
            band_count=band_count,
            grid=grid,
            electron_count=electron_count,
            mixer=mixer,
        )
    )
    if state.spin_densities is None or state.magnetization_mixer_state is None:
        raise ValueError("periodic spin resume payload is incomplete")
    spin_densities = tuple(
        _owned_device_copy(values, dtype=mx.float32) for values in state.spin_densities
    )
    if any(values.shape != grid.shape for values in spin_densities):
        raise ValueError("periodic spin resume densities have incompatible shapes")
    up, down = spin_densities
    finite = (
        mx.all(mx.isfinite(up))
        & mx.all(mx.isfinite(down))
        & (mx.min(up) >= 0.0)
        & (mx.min(down) >= 0.0)
    )
    total_error = mx.max(mx.abs((up + down) - density))
    up_count = mx.sum(up) * grid.dv
    down_count = mx.sum(down) * grid.dv
    mx.eval(finite, total_error, up_count, down_count)
    observed_total = float(up_count + down_count)
    observed_moment = float(up_count - down_count)
    if (
        not bool(finite)
        or float(total_error) > 2.0e-6
        or abs(observed_total - electron_count) > 1.0e-4
        or (
            expected_magnetization is not None
            and abs(observed_moment - expected_magnetization) > 1.0e-4
        )
    ):
        raise ValueError("periodic spin resume densities violate charge or moment")
    down_states = _restore_down_owner_states(
        state,
        bases=down_bases,
        ownership=ownership,
        band_count=band_count,
    )
    magnetization_mixer._restore_checkpoint_state(
        state.magnetization_mixer_state,
        expected_shape=grid.shape,
    )
    return (
        density,
        (up_states, down_states),
        (up, down),
        previous_energy,
        history,
        energy_terms,
    )
