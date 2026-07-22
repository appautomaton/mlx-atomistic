from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PeriodicSCFConfig,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    ReciprocalGrid,
    XCResult,
    run_periodic_scf,
)
from mlx_atomistic.dft._compact import _CompactBatch
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.mixing import PulayDIISMixer
from mlx_atomistic.dft.periodic_gth import _GTHProjectorCache
from mlx_atomistic.dft.periodic_scf import (
    PeriodicEigenResult,
    PeriodicKPointResult,
    _DavidsonEngine,
    _DavidsonLaneRequest,
    _DavidsonScheduler,
    _density_from_kpoints,
    _initial_coefficients,
    _next_scf_eigensolver_tolerance,
    _plan_compact_submissions,
    _scf_eigensolver_tolerance,
    _stable_compact_capacity_groups,
)
from mlx_atomistic.dft.runtime_state import serialize_periodic_scf_state


def _bases() -> tuple[PlaneWaveBasis, PlaneWaveBasis]:
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    reciprocal = ReciprocalGrid.from_real_space(grid)
    return (
        PlaneWaveBasis.from_reduced_kpoint(
            grid,
            3.0,
            (0.0, 0.0, 0.0),
            reciprocal_grid=reciprocal,
            lane_label="batch:gamma",
        ),
        PlaneWaveBasis.from_reduced_kpoint(
            grid,
            3.0,
            (0.25, 0.0, 0.0),
            reciprocal_grid=reciprocal,
            lane_label="batch:shifted",
        ),
    )


def _state(
    basis: PlaneWaveBasis,
    *,
    vectors: int = 2,
    seed: int,
):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(vectors, basis.active_count)) + 1j * rng.normal(
        size=(vectors, basis.active_count)
    )
    return basis._state_from_compact(mx.array(values.astype(np.complex64)))


def _kpoint_result(
    basis: PlaneWaveBasis,
    state,
    *,
    reduced_kpoint: tuple[float, float, float],
    weight: float,
) -> PeriodicKPointResult:
    eigen = PeriodicEigenResult._from_compact(
        eigenvalues=mx.zeros((state.vector_count,), dtype=mx.float32),
        compact_coefficients=state,
        basis=basis,
        residuals=mx.zeros((state.vector_count,), dtype=mx.float32),
        orthonormality_error=0.0,
        iterations=1,
        converged=True,
        subspace_size=state.vector_count,
        restart_count=0,
    )
    return PeriodicKPointResult(
        reduced_kpoint=reduced_kpoint,
        weight=weight,
        basis=basis,
        eigen=eigen,
    )


def _hydrogen_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )


def _paired_scf_problem(
    *,
    batch_size: int,
) -> tuple[PeriodicDFTSystem, KPointMesh, PeriodicSCFConfig]:
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        _hydrogen_gth(),
    )
    mesh = KPointMesh(
        [
            KPoint((-0.25, 0.0, 0.0), weight=0.25, coordinate_system="reduced"),
            KPoint((0.25, 0.0, 0.0), weight=0.25, coordinate_system="reduced"),
            KPoint((0.0, -0.25, 0.0), weight=0.25, coordinate_system="reduced"),
            KPoint((0.0, 0.25, 0.0), weight=0.25, coordinate_system="reduced"),
        ]
    )
    config = PeriodicSCFConfig(
        max_iterations=6,
        min_iterations=2,
        density_tolerance=0.2,
        energy_tolerance=0.1,
        orbital_tolerance=2e-3,
        mixing_beta=0.5,
        mixer="diis",
        davidson=PeriodicDavidsonConfig(
            max_iterations=20,
            tolerance=2e-3,
            max_subspace_size=12,
        ),
        kpoint_batch_size=batch_size,
        hpsi_shape_policy="stable",
    )
    return system, mesh, config


def test_default_batch_and_projector_cache_policy_stays_bounded():
    config = PeriodicSCFConfig()

    assert config.adaptive_eigensolver_tolerance is False
    assert config.batch_policy() == {
        "kpoint_batch_size": 8,
        "max_batch_padding_fraction": 0.25,
        "max_batch_transient_bytes": 512 * 1024 * 1024,
        "hpsi_shape_policy": "finite-buckets",
        "hpsi_lane_capacity_buckets": [1, 2, 4, 8],
        "hpsi_vector_capacity_buckets": [4, 8, 16],
    }
    assert _GTHProjectorCache.DEFAULT_BUDGET_BYTES == 256 * 1024 * 1024


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adaptive_eigensolver_tolerance", 1, "must be a bool"),
        ("initial_eigensolver_tolerance", 0.0, "finite and positive"),
        ("initial_eigensolver_tolerance", float("nan"), "finite and positive"),
        ("eigensolver_tolerance_scale", False, "finite and positive"),
        ("eigensolver_tolerance_scale", float("inf"), "finite and positive"),
        ("hpsi_shape_policy", "dynamic", "must be 'stable' or 'finite-buckets'"),
    ],
)
def test_adaptive_eigensolver_controls_reject_malformed_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        PeriodicSCFConfig(**{field: value})


def test_adaptive_eigensolver_tolerance_is_monotone_and_strictly_floored():
    config = PeriodicSCFConfig(
        adaptive_eigensolver_tolerance=True,
        initial_eigensolver_tolerance=1e-2,
        eigensolver_tolerance_scale=0.1,
        davidson=PeriodicDavidsonConfig(tolerance=1e-6),
    )
    history = [
        {"eigensolver_tolerance": 1e-2, "density_residual": 1.0},
        {"eigensolver_tolerance": 3.125e-3, "density_residual": 0.1},
        {"eigensolver_tolerance": 3.125e-4, "density_residual": 1e-8},
    ]

    assert _scf_eigensolver_tolerance(config, history, 32.0) == pytest.approx(1e-6)
    assert _next_scf_eigensolver_tolerance(config, 1e-4, 1.0, 32.0) == 1e-4
    assert _next_scf_eigensolver_tolerance(config, 1e-4, 0.0, 32.0) == 1e-6


def test_adaptive_eigensolver_resume_requires_an_exact_recorded_schedule():
    config = PeriodicSCFConfig(
        adaptive_eigensolver_tolerance=True,
        davidson=PeriodicDavidsonConfig(tolerance=1e-6),
    )

    with pytest.raises(ValueError, match="malformed tolerance schedule"):
        _scf_eigensolver_tolerance(config, [{"density_residual": 0.1}], 32.0)
    with pytest.raises(ValueError, match="inconsistent tolerance schedule"):
        _scf_eigensolver_tolerance(
            config,
            [{"eigensolver_tolerance": 1e-3, "density_residual": 0.1}],
            32.0,
        )


def test_adaptive_scf_records_the_tolerance_and_keeps_the_strict_final_gate():
    system, mesh, fixed = _paired_scf_problem(batch_size=2)
    config = replace(
        fixed,
        adaptive_eigensolver_tolerance=True,
        initial_eigensolver_tolerance=5e-2,
        eigensolver_tolerance_scale=0.1,
    )
    observer = RuntimeObserver(detail_events=False)

    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=config,
        observer=observer,
    )

    tolerances = [float(row["eigensolver_tolerance"]) for row in result.history]
    assert tolerances[0] == 5e-2
    assert tolerances == sorted(tolerances, reverse=True)
    assert all(value >= config.davidson.tolerance for value in tolerances)
    assert result.converged
    assert max(point.eigen.residuals.item() for point in result.kpoints) <= (
        config.orbital_tolerance
    )
    iteration_events = [
        event
        for event in observer.snapshot()["events"]
        if event["event"] == "scf_iteration"
    ]
    assert [event["eigensolver_tolerance"] for event in iteration_events[::2]] == tolerances
    assert [event["eigensolver_tolerance"] for event in iteration_events[1::2]] == tolerances


def test_compact_hpsi_batches_ragged_lanes_with_one_fft_pair(monkeypatch):
    first, second = _bases()
    states = (_state(first, seed=10), _state(second, seed=11))
    potential = mx.full(first.grid.shape, 0.35, dtype=mx.float32)
    mx.eval(potential)
    operators = tuple(
        PeriodicKohnShamOperator._from_shared_potential(basis, potential)
        for basis in (first, second)
    )
    expected = tuple(
        operator._apply_compact(state) for operator, state in zip(operators, states, strict=True)
    )
    observer = RuntimeObserver()
    observed_operators = tuple(
        PeriodicKohnShamOperator._from_shared_potential(
            basis,
            potential,
            observer=observer,
        )
        for basis in (first, second)
    )
    assert (
        observed_operators[0]._effective_local_potential
        is observed_operators[1]._effective_local_potential
    )
    calls = {"fftn": 0, "ifftn": 0}
    original_fftn = mx.fft.fftn
    original_ifftn = mx.fft.ifftn

    def counted_fftn(*args, **kwargs):
        calls["fftn"] += 1
        return original_fftn(*args, **kwargs)

    def counted_ifftn(*args, **kwargs):
        calls["ifftn"] += 1
        return original_ifftn(*args, **kwargs)

    monkeypatch.setattr(mx.fft, "fftn", counted_fftn)
    monkeypatch.setattr(mx.fft, "ifftn", counted_ifftn)
    outcome = PeriodicKohnShamOperator._apply_compact_batch(
        observed_operators,
        states,
        observer=observer,
    )

    assert not outcome.failures
    assert outcome.batch is not None
    assert calls == {"fftn": 1, "ifftn": 1}
    for index, reference in enumerate(expected):
        np.testing.assert_allclose(
            np.asarray(outcome.action_for(index).values),
            np.asarray(reference.values),
            atol=3e-6,
        )
    retained_action = np.asarray(outcome.action_for(0).values).copy()
    states[0].values[:] = mx.zeros_like(states[0].values)
    mx.eval(states[0].values)
    np.testing.assert_array_equal(
        np.asarray(outcome.action_for(0).values),
        retained_action,
    )
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == 1
    assert work["fft_submissions"] == 2
    assert work["fft_vector_equivalents"] == 8
    assert work["padding_elements"] == abs(first.active_count - second.active_count)


def test_compact_hpsi_stable_capacity_masks_lane_vector_and_active_padding():
    first, second = _bases()
    states = (
        _state(first, vectors=1, seed=112),
        _state(second, vectors=2, seed=113),
    )
    potential = mx.full(first.grid.shape, 0.35, dtype=mx.float32)
    mx.eval(potential)
    operators = tuple(
        PeriodicKohnShamOperator._from_shared_potential(basis, potential)
        for basis in (first, second)
    )
    expected = tuple(
        operator._apply_compact(state) for operator, state in zip(operators, states, strict=True)
    )
    active_capacity = max(first.active_count, second.active_count)
    prepared = _CompactBatch.from_states(
        states,
        lane_capacity=3,
        vector_capacity=2,
        active_capacity=active_capacity,
    )
    observer = RuntimeObserver()

    outcome = PeriodicKohnShamOperator._apply_compact_batch(
        tuple(
            PeriodicKohnShamOperator._from_shared_potential(
                basis,
                potential,
                observer=observer,
            )
            for basis in (first, second)
        ),
        states,
        observer=observer,
        prepared_batch=prepared,
    )

    assert not outcome.failures
    assert outcome.batch is prepared
    assert prepared.values.shape == (3, 2, active_capacity)
    assert prepared.lane_count == 2
    assert prepared.lane_capacity == 3
    assert prepared.vector_counts == (1, 2)
    assert prepared.lane_padding_elements == 2 * active_capacity
    assert prepared.vector_padding_elements == active_capacity
    np.testing.assert_array_equal(
        np.asarray(prepared.values[2]),
        np.zeros((2, active_capacity), dtype=np.complex64),
    )
    for index, reference in enumerate(expected):
        observed = outcome.action_for(index)
        assert observed.values.shape == reference.values.shape
        np.testing.assert_allclose(
            np.asarray(observed.values),
            np.asarray(reference.values),
            atol=3e-6,
        )
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == 1
    assert work["hpsi_vector_equivalents"] == 3
    assert work["hpsi_submitted_vector_equivalents"] == 6
    assert work["hpsi_lane_padding_vector_equivalents"] == 2
    assert work["hpsi_vector_padding_equivalents"] == 1
    assert work["hpsi_submitted_vector_equivalents"] == (
        work["hpsi_vector_equivalents"]
        + work["hpsi_lane_padding_vector_equivalents"]
        + work["hpsi_vector_padding_equivalents"]
    )
    assert work["fft_vector_equivalents"] == 6
    memory = observer.snapshot()["memory"]
    assert memory["hpsi_fft_workspace_bytes"] == memory["fft_workspace_bytes"]
    assert memory["hpsi_peak_temporary_bytes"] == memory["peak_temporary_bytes"]


def test_compact_submission_planner_respects_byte_and_padding_bounds():
    first, second = _bases()
    repeated = PlaneWaveBasis.from_reduced_kpoint(
        first.grid,
        first.cutoff_hartree,
        (0.0, 0.0, 0.0),
        reciprocal_grid=first.reciprocal_grid,
        lane_label="batch:gamma-repeat",
    )
    equal_states = (
        _state(first, vectors=1, seed=20),
        _state(repeated, vectors=1, seed=21),
        _state(first, vectors=1, seed=22),
    )
    two_lane_budget = _CompactBatch.from_states(equal_states[:2]).estimated_transient_bytes
    byte_bounded = _plan_compact_submissions(
        equal_states,
        batch_cap=3,
        max_padding_fraction=0.25,
        max_transient_bytes=two_lane_budget,
    )

    assert not byte_bounded.failures
    assert [submission.indices for submission in byte_bounded.submissions] == [
        (0, 1),
        (2,),
    ]

    narrow = PlaneWaveBasis.from_reduced_kpoint(
        first.grid,
        0.5,
        (0.0, 0.0, 0.0),
        reciprocal_grid=first.reciprocal_grid,
        lane_label="batch:narrow",
    )
    padding_bounded = _plan_compact_submissions(
        (_state(narrow, vectors=1, seed=23), _state(second, vectors=1, seed=24)),
        batch_cap=2,
        max_padding_fraction=0.25,
        max_transient_bytes=_CompactBatch._DEFAULT_MAX_TRANSIENT_BYTES,
    )

    assert not padding_bounded.failures
    assert [submission.indices for submission in padding_bounded.submissions] == [
        (0,),
        (1,),
    ]


def test_stable_capacity_groups_split_only_when_padding_bound_requires_it():
    first, second = _bases()
    nearby_states = (
        _state(first, vectors=1, seed=114),
        _state(second, vectors=1, seed=115),
    )
    nearby = _stable_compact_capacity_groups(
        nearby_states,
        range(2),
        lane_capacity=2,
        vector_capacity=1,
        max_padding_fraction=0.25,
    )

    assert len(nearby) == 1
    assert nearby[0][0] == tuple(
        sorted(range(2), key=lambda index: nearby_states[index].layout.active_count)
    )
    assert nearby[0][1].active == max(state.layout.active_count for state in nearby_states)

    narrow = PlaneWaveBasis.from_reduced_kpoint(
        first.grid,
        0.5,
        (0.0, 0.0, 0.0),
        reciprocal_grid=first.reciprocal_grid,
        lane_label="batch:stable-narrow",
    )
    wide_states = (
        _state(narrow, vectors=1, seed=116),
        _state(second, vectors=1, seed=117),
    )
    split = _stable_compact_capacity_groups(
        wide_states,
        range(2),
        lane_capacity=2,
        vector_capacity=1,
        max_padding_fraction=0.25,
    )

    assert len(split) == 2
    assert [indices for indices, _capacity in split] == [(0,), (1,)]
    assert [capacity.active for _indices, capacity in split] == [
        narrow.active_count,
        second.active_count,
    ]


def test_gth_projectors_use_one_k_batched_matrix_path_and_batch_cache_guard(
    monkeypatch,
):
    first, second = _bases()
    states = (_state(first, seed=25), _state(second, seed=26))
    pseudo = _hydrogen_gth()
    references = []
    reference_gth = []
    for basis, state in zip((first, second), states, strict=True):
        gth = PeriodicGTHNonlocalOperator(
            pseudo,
            basis,
            ((1.0, 2.0, 3.0),),
        )
        reference_gth.append(gth)
        references.append(
            PeriodicKohnShamOperator(
                basis,
                mx.full(basis.grid.shape, 0.2),
                gth,
            )._apply_compact(state)
        )

    cache = _GTHProjectorCache(byte_budget=max(first.active_count, second.active_count) * 8)
    observer = RuntimeObserver()
    potential = mx.full(first.grid.shape, 0.2)
    mx.eval(potential)
    gth_operators = tuple(
        PeriodicGTHNonlocalOperator(
            pseudo,
            basis,
            ((1.0, 2.0, 3.0),),
            cache=cache,
        )
        for basis in (first, second)
    )
    operators = tuple(
        PeriodicKohnShamOperator._from_shared_potential(
            basis,
            potential,
            gth,
            observer,
        )
        for basis, gth in zip((first, second), gth_operators, strict=True)
    )
    prepared = _CompactBatch.from_states(states)
    complete_bytes = PeriodicKohnShamOperator._estimated_batch_transient_bytes(
        operators,
        prepared,
    )
    singleton_bytes = [
        PeriodicKohnShamOperator._estimated_batch_transient_bytes(
            (operator,),
            _CompactBatch.from_states((state,)),
        )
        for operator, state in zip(operators, states, strict=True)
    ]
    bounded_bytes = max(prepared.estimated_transient_bytes, *singleton_bytes)
    assert bounded_bytes < complete_bytes
    bounded_plan = _plan_compact_submissions(
        states,
        batch_cap=2,
        max_padding_fraction=0.25,
        max_transient_bytes=bounded_bytes,
        batch_byte_estimator=lambda indices, batch: (
            PeriodicKohnShamOperator._estimated_batch_transient_bytes(
                [operators[index] for index in indices],
                batch,
            )
        ),
    )
    assert [submission.indices for submission in bounded_plan.submissions] == [
        (0,),
        (1,),
    ]
    rejected = PeriodicKohnShamOperator._apply_compact_batch(
        operators,
        states,
        observer=observer,
        max_transient_bytes=bounded_bytes,
        prepared_batch=prepared,
    )
    assert not rejected.actions
    assert set(rejected.failures) == {0, 1}
    assert rejected.batch is None
    assert observer.snapshot()["work_counters"]["hpsi_calls"] == 0

    matmul_calls = 0
    original_matmul = mx.matmul

    def counted_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(mx, "matmul", counted_matmul)
    outcome = PeriodicKohnShamOperator._apply_compact_batch(
        operators,
        states,
        observer=observer,
    )

    assert not outcome.failures
    assert matmul_calls == 3
    for index, reference in enumerate(references):
        np.testing.assert_allclose(
            np.asarray(outcome.action_for(index).values),
            np.asarray(reference.values),
            atol=3e-6,
        )
    assert cache.entry_count == 1
    assert cache.current_bytes <= cache.byte_budget
    work = observer.snapshot()["work_counters"]
    assert work["projector_cache_misses"] == 2
    assert work["projector_elements_generated"] == (first.active_count + second.active_count)
    assert (
        observer.snapshot()["memory"]["projector_payload_bytes"]
        == (first.active_count + second.active_count) * 8
    )

    cache.close()
    for gth in reference_gth:
        gth.close()


def test_batched_hpsi_keeps_nonlocal_failure_lane_local():
    first, second = _bases()
    states = (_state(first, seed=30), _state(second, seed=31))

    class FailingNonlocal:
        def _apply_compact(self, coefficients, *, evaluate=True):
            raise RuntimeError("injected nonlocal lane failure")

    observer = RuntimeObserver()
    operators = (
        PeriodicKohnShamOperator(
            first,
            mx.full(first.grid.shape, 0.2),
            observer=observer,
        ),
        PeriodicKohnShamOperator(
            second,
            mx.full(second.grid.shape, 0.2),
            FailingNonlocal(),
            observer,
        ),
    )
    outcome = PeriodicKohnShamOperator._apply_compact_batch(
        operators,
        states,
        observer=observer,
    )

    assert set(outcome.actions) == {0}
    assert set(outcome.failures) == {1}
    with pytest.raises(RuntimeError, match="injected nonlocal lane failure"):
        outcome.action_for(1)
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == 1
    assert work["fft_submissions"] == 2


def test_batched_davidson_keeps_divergent_lane_restart_and_convergence_local():
    first, second = _bases()
    observer = RuntimeObserver()
    coordinates = first.grid.coordinates()
    potential = 0.3 * mx.cos(2.0 * np.pi * coordinates[..., 0] / first.grid.lengths[0])
    mx.eval(potential)
    operators = tuple(
        PeriodicKohnShamOperator._from_shared_potential(
            basis,
            potential,
            observer=observer,
        )
        for basis in (first, second)
    )
    requests = (
        _DavidsonLaneRequest(
            lane_id="fast",
            operator=operators[0],
            n_bands=1,
            config=PeriodicDavidsonConfig(
                max_iterations=4,
                tolerance=1.0,
                max_subspace_size=3,
            ),
            trial=_initial_coefficients(first, 1),
            observer=observer,
        ),
        _DavidsonLaneRequest(
            lane_id="slow",
            operator=operators[1],
            n_bands=1,
            config=PeriodicDavidsonConfig(
                max_iterations=4,
                tolerance=1e-10,
                max_subspace_size=3,
            ),
            trial=_initial_coefficients(second, 1),
            observer=observer,
        ),
    )
    outcome = _DavidsonEngine(scheduler=_DavidsonScheduler(batch_cap=2)).solve(requests)

    assert not outcome.failures
    assert outcome.result_for("fast").converged
    assert outcome.result_for("fast").iterations == 1
    assert not outcome.result_for("slow").converged
    assert outcome.result_for("slow").iterations == 4
    assert outcome.result_for("slow").restart_count > 0
    assert outcome.ready_rounds[0] == ("fast", "slow")
    assert outcome.submission_groups[:2] == (
        ("fast", "slow"),
        ("fast", "slow"),
    )
    assert all(group == ("slow",) for group in outcome.submission_groups[2:])


def test_density_batch_one_and_many_have_the_same_ordered_sum():
    first, second = _bases()
    states = (_state(first, seed=40), _state(second, seed=41))
    results = (
        _kpoint_result(
            first,
            states[0],
            reduced_kpoint=(0.0, 0.0, 0.0),
            weight=0.4,
        ),
        _kpoint_result(
            second,
            states[1],
            reduced_kpoint=(0.25, 0.0, 0.0),
            weight=0.6,
        ),
    )
    singleton_observer = RuntimeObserver()
    batched_observer = RuntimeObserver()
    singleton = _density_from_kpoints(
        results,
        occupation=2.0,
        batch_cap=1,
        observer=singleton_observer,
    )
    batched = _density_from_kpoints(
        results,
        occupation=2.0,
        batch_cap=2,
        observer=batched_observer,
    )

    np.testing.assert_allclose(np.asarray(batched), np.asarray(singleton), atol=3e-6)
    singleton_work = singleton_observer.snapshot()["work_counters"]
    batched_work = batched_observer.snapshot()["work_counters"]
    assert singleton_work["fft_submissions"] == 2
    assert batched_work["fft_submissions"] == 1
    assert singleton_work["fft_vector_equivalents"] == 4
    assert batched_work["fft_vector_equivalents"] == 4


def test_pulay_diis_only_moves_the_small_gram_matrix_to_numpy(monkeypatch):
    current = mx.full((8, 8, 8), 0.2, dtype=mx.float32)
    first_target = current + 0.01
    second_target = current + 0.02
    mixer = PulayDIISMixer(beta=0.4)
    array_type = type(current)
    original_asarray = np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, array_type) and int(value.size) > mixer.history_size**2:
            raise AssertionError("DIIS copied a full density grid to NumPy")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", guarded_asarray)
    first = mixer.mix(current, first_target)
    second = mixer.mix(first, second_target)
    mx.eval(second)
    monkeypatch.undo()

    assert all(isinstance(values, array_type) for values in mixer._densities)
    assert all(isinstance(values, array_type) for values in mixer._residuals)
    assert np.isfinite(np.asarray(second)).all()


def test_pulay_diis_restarts_when_coefficients_exceed_float32(monkeypatch):
    mixer = PulayDIISMixer(beta=0.4)
    current = mx.full((4, 4, 4), 0.2, dtype=mx.float32)
    first_target = current + 0.01
    first = mixer.mix(current, first_target)
    second_target = first + 0.02
    expected = (0.6 * first + 0.4 * second_target).astype(mx.float32)

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.array([1e300, -1e300, 0.0]),
    )
    second = mixer.mix(first, second_target)

    np.testing.assert_allclose(np.asarray(second), np.asarray(expected), atol=1e-7)
    assert mixer.metadata()["stored"] == 1
    assert mixer.metadata()["last_coefficients"] == [1.0]


def test_pulay_diis_restarts_when_combination_has_no_positive_mass(monkeypatch):
    mixer = PulayDIISMixer(beta=0.4)
    high = mx.full((4, 4, 4), 10.0, dtype=mx.float32)
    low = mx.full((4, 4, 4), 1.0, dtype=mx.float32)
    mixer.mix(high, high)

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.array([-1.0, 2.0, 0.0]),
    )
    mixed = mixer.mix(low, low)

    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(low))
    assert mixer.metadata()["stored"] == 1
    assert mixer.metadata()["last_coefficients"] == [1.0]


def test_pulay_diis_restarts_when_combination_has_mixed_signs(monkeypatch):
    mixer = PulayDIISMixer(beta=0.4)
    first = mx.array([10.0, 1.0], dtype=mx.float32)
    second = mx.array([1.0, 10.0], dtype=mx.float32)
    mixer.mix(first, first)

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.array([2.0, -1.0, 0.0]),
    )
    mixed = mixer.mix(second, second)

    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(second))
    assert mixer.metadata()["stored"] == 1
    assert mixer.metadata()["last_coefficients"] == [1.0]


def test_pulay_diis_restarts_when_finite_coefficients_overflow_product(monkeypatch):
    mixer = PulayDIISMixer(beta=0.4)
    maximum = np.finfo(np.float32).max
    high = mx.array([float(maximum)], dtype=mx.float32)
    low = mx.array([1.0], dtype=mx.float32)
    mixer.mix(high, high)

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.array([2.0, -1.0, 0.0]),
    )
    mixed = mixer.mix(low, low)

    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(low))
    assert mixer.metadata()["stored"] == 1
    assert mixer.metadata()["last_coefficients"] == [1.0]


def test_periodic_scf_rejects_nonfinite_xc_before_hpsi():
    system, mesh, config = _paired_scf_problem(batch_size=2)
    observer = RuntimeObserver(synchronize=mx.synchronize)

    class NonFiniteXC:
        name = "non-finite-test-xc"

        def evaluate(self, density, grid=None, *, density_floor=1e-12):
            del density_floor
            assert grid is not None
            return XCResult(
                name=self.name,
                energy_density=mx.zeros_like(density),
                potential=mx.full(grid.shape, float("nan")),
                total_energy=mx.array(0.0),
            )

    with pytest.raises(
        ValueError,
        match="SCF exchange-correlation result is non-finite",
    ):
        run_periodic_scf(
            system,
            cutoff_hartree=2.5,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            xc_functional=NonFiniteXC(),
            observer=observer,
        )

    snapshot = observer.snapshot()
    assert snapshot["work_counters"]["hpsi_calls"] == 0
    assert not any(event["event"] == "kpoint_batch" for event in snapshot["events"])


def test_representative_scf_k_batch_one_and_many_match_trajectory_and_events():
    singleton_problem = _paired_scf_problem(batch_size=1)
    batched_problem = _paired_scf_problem(batch_size=2)
    singleton_observer = RuntimeObserver(synchronize=mx.synchronize)
    batched_observer = RuntimeObserver(synchronize=mx.synchronize)
    singleton = run_periodic_scf(
        singleton_problem[0],
        cutoff_hartree=2.5,
        kpoint_mesh=singleton_problem[1],
        n_bands=1,
        config=singleton_problem[2],
        observer=singleton_observer,
    )
    batched = run_periodic_scf(
        batched_problem[0],
        cutoff_hartree=2.5,
        kpoint_mesh=batched_problem[1],
        n_bands=1,
        config=batched_problem[2],
        observer=batched_observer,
    )

    assert len(singleton.owned_kpoints) == len(batched.owned_kpoints) == 2
    assert singleton.iterations == batched.iterations
    assert singleton.total_energy == pytest.approx(batched.total_energy, abs=5e-5)
    assert singleton.electron_count == pytest.approx(
        batched.electron_count,
        abs=5e-6,
    )
    np.testing.assert_allclose(
        np.asarray(singleton.density),
        np.asarray(batched.density),
        atol=5e-5,
    )
    for left, right in zip(singleton.history, batched.history, strict=True):
        assert left["total_energy_hartree"] == pytest.approx(
            right["total_energy_hartree"],
            abs=5e-5,
        )
        assert left["density_residual"] == pytest.approx(
            right["density_residual"],
            abs=5e-5,
        )
        assert left["electron_count"] == pytest.approx(
            right["electron_count"],
            abs=5e-6,
        )
    for singleton_kpoint, batched_kpoint in zip(
        singleton.owned_kpoints,
        batched.owned_kpoints,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(singleton_kpoint.eigen.eigenvalues),
            np.asarray(batched_kpoint.eigen.eigenvalues),
            atol=5e-5,
        )
    assert batched.batch_policy == {
        "kpoint_batch_size": 2,
        "max_batch_padding_fraction": 0.25,
        "max_batch_transient_bytes": 512 * 1024 * 1024,
        "hpsi_shape_policy": "stable",
        "hpsi_lane_capacity_buckets": [1, 2, 4, 8],
        "hpsi_vector_capacity_buckets": [4, 8, 16],
    }
    assert batched.to_dict()["batch_policy"] == batched.batch_policy
    metadata = json.loads(serialize_periodic_scf_state(batched)["metadata.json"])
    assert metadata["batch_policy"] == batched.batch_policy
    events = [
        event for event in batched_observer.snapshot()["events"] if event["event"] == "kpoint_batch"
    ]
    assert events
    assert [event["status"] for event in events].count("started") == [
        event["status"] for event in events
    ].count("completed")
    assert any(event["batch_size"] == 2 for event in events)
    assert all("padding_elements" in event for event in events)
    assert all(
        event["estimated_transient_bytes"] >= event["compact_batch_transient_bytes"]
        for event in events
    )
    assert all(
        event["estimated_transient_bytes"] <= event["batch_policy"]["max_batch_transient_bytes"]
        for event in events
    )
    singleton_work = singleton_observer.snapshot()["work_counters"]
    batched_work = batched_observer.snapshot()["work_counters"]
    assert batched_work["hpsi_calls"] < singleton_work["hpsi_calls"]
    assert batched_work["hpsi_vector_equivalents"] == singleton_work["hpsi_vector_equivalents"]
    assert batched_work["fft_submissions"] < singleton_work["fft_submissions"]


def test_finite_hpsi_buckets_preserve_small_scf_trajectory():
    stable_system, stable_mesh, stable_config = _paired_scf_problem(batch_size=2)
    finite_system, finite_mesh, finite_config = _paired_scf_problem(batch_size=2)
    stable_observer = RuntimeObserver(synchronize=mx.synchronize)
    finite_observer = RuntimeObserver(synchronize=mx.synchronize)

    stable = run_periodic_scf(
        stable_system,
        cutoff_hartree=2.5,
        kpoint_mesh=stable_mesh,
        n_bands=1,
        config=stable_config,
        observer=stable_observer,
    )
    finite = run_periodic_scf(
        finite_system,
        cutoff_hartree=2.5,
        kpoint_mesh=finite_mesh,
        n_bands=1,
        config=replace(finite_config, hpsi_shape_policy="finite-buckets"),
        observer=finite_observer,
    )

    assert stable.iterations == finite.iterations
    assert stable.total_energy == pytest.approx(finite.total_energy, abs=5e-5)
    assert stable.electron_count == pytest.approx(finite.electron_count, abs=5e-6)
    np.testing.assert_allclose(
        np.asarray(stable.density),
        np.asarray(finite.density),
        atol=5e-5,
    )
    for left, right in zip(stable.owned_kpoints, finite.owned_kpoints, strict=True):
        np.testing.assert_allclose(
            np.asarray(left.eigen.eigenvalues),
            np.asarray(right.eigen.eigenvalues),
            atol=5e-5,
        )
    assert finite.batch_policy["hpsi_shape_policy"] == "finite-buckets"
    assert finite_observer.snapshot()["hpsi_shapes"]


def test_representative_k_batch_failure_event_identifies_only_failed_lane(
    monkeypatch,
):
    system, mesh, config = _paired_scf_problem(batch_size=2)
    observer = RuntimeObserver(synchronize=mx.synchronize)
    original_batch_apply = PeriodicGTHNonlocalOperator._apply_compact_batch

    def injected_batch_apply(operators, coefficients, *, batch, evaluate=True):
        if len(operators) > 1:
            raise RuntimeError("injected collective GTH evaluation failure")
        if operators[0].basis.kpoint_cartesian[1] < -1e-8:
            raise RuntimeError("injected representative projector failure")
        return original_batch_apply(
            operators,
            coefficients,
            batch=batch,
            evaluate=evaluate,
        )

    monkeypatch.setattr(
        PeriodicGTHNonlocalOperator,
        "_apply_compact_batch",
        staticmethod(injected_batch_apply),
    )
    with pytest.raises(
        RuntimeError,
        match="injected representative projector failure",
    ):
        run_periodic_scf(
            system,
            cutoff_hartree=2.5,
            kpoint_mesh=mesh,
            n_bands=1,
            config=config,
            observer=observer,
        )

    events = observer.snapshot()["events"]
    completed = [
        event
        for event in events
        if event["event"] == "kpoint_batch"
        and event["status"] == "completed"
        and event.get("failed_explicit_indices")
    ]
    assert completed
    assert completed[0]["batch_size"] == 2
    assert completed[0]["failed_explicit_indices"] == [2]
    failure_events = [event for event in events if event["event"] == "failure"]
    assert len(failure_events) == 1
    assert failure_events[0]["failed_explicit_indices"] == [2]
