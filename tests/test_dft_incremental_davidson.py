from __future__ import annotations

import gc
import weakref

import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.dft.periodic_scf as periodic_scf
from mlx_atomistic.dft import (
    GTHProjectorChannel,
    PeriodicDavidsonConfig,
    PeriodicGTHNonlocalOperator,
    PeriodicKohnShamOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.periodic_scf import (
    _Complex64RankPolicy,
    _DavidsonApplicationTicket,
    _DavidsonEngine,
    _DavidsonLaneRequest,
    _DavidsonScheduler,
    _FixedHamiltonianToken,
    _PairedDavidsonState,
)


class _FailureFrameSentinel:
    pass


def _problem(
    *,
    lane_label: str = "incremental",
) -> tuple[PlaneWaveBasis, PeriodicKohnShamOperator]:
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        3.5,
        (0.25, 0.125, 0.0),
        lane_label=lane_label,
    )
    coordinates = grid.coordinates()
    potential = (
        0.31 * mx.cos(2.0 * np.pi * coordinates[..., 0] / 6.0)
        + 0.17 * mx.sin(4.0 * np.pi * coordinates[..., 1] / 6.0)
        + 0.07 * mx.cos(2.0 * np.pi * coordinates[..., 2] / 6.0)
    )
    return basis, PeriodicKohnShamOperator(basis, potential)


def test_fixed_hamiltonian_snapshots_input_and_returns_fresh_potential_copies():
    basis, template = _problem(lane_label="potential:snapshot")
    source = template.effective_local_potential
    operator = PeriodicKohnShamOperator(basis, source)
    seed = periodic_scf._initial_coefficients(basis, 2)
    expected = operator._apply_compact(seed)
    mx.eval(expected.values)

    first_public = operator.effective_local_potential
    second_public = operator.effective_local_potential
    assert first_public is not second_public
    source[:] = mx.full(source.shape, 19.0)
    first_public[:] = mx.full(first_public.shape, -23.0)

    observed = operator._apply_compact(seed)
    current_public = operator.effective_local_potential
    np.testing.assert_allclose(
        np.asarray(observed.values),
        np.asarray(expected.values),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(current_public),
        np.asarray(second_public),
        atol=0.0,
    )
    assert not np.array_equal(np.asarray(current_public), np.asarray(source))
    assert not np.array_equal(np.asarray(current_public), np.asarray(first_public))


def test_davidson_callback_mutation_cannot_mix_fixed_hamiltonian_potentials():
    basis, template = _problem(lane_label="potential:callback")
    source = template.effective_local_potential
    reference_potential = mx.array(source)
    mx.eval(reference_potential)
    holder: dict[str, PeriodicKohnShamOperator] = {}
    mutated_iterations: list[int] = []

    def mutate_after_ritz(event):
        if event["event"] != "davidson_iteration" or mutated_iterations:
            return
        mutated_iterations.append(int(event["iteration"]))
        source[:] = mx.full(source.shape, 31.0)
        public_copy = holder["operator"].effective_local_potential
        public_copy[:] = mx.full(public_copy.shape, -37.0)
        public_kinetic = basis.active_kinetic_energies
        public_kinetic[:] = mx.full(public_kinetic.shape, 41.0)

    observer = RuntimeObserver(callback=mutate_after_ritz)
    operator = PeriodicKohnShamOperator(basis, source, observer=observer)
    holder["operator"] = operator
    config = PeriodicDavidsonConfig(
        max_iterations=4,
        tolerance=1e-10,
        max_subspace_size=12,
    )

    mutated = solve_periodic_eigenproblem(
        operator,
        n_bands=2,
        config=config,
        observer=observer,
    )
    reference = solve_periodic_eigenproblem(
        PeriodicKohnShamOperator(basis, reference_potential),
        n_bands=2,
        config=config,
    )

    assert mutated_iterations == [1]
    assert observer.snapshot()["work_counters"]["hpsi_calls"] == 4
    np.testing.assert_allclose(
        np.asarray(mutated.eigenvalues),
        np.asarray(reference.eigenvalues),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(mutated.residuals),
        np.asarray(reference.residuals),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(operator.effective_local_potential),
        np.asarray(reference_potential),
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("max_iterations", 2.5),
        ("max_iterations", float("nan")),
        ("max_iterations", float("inf")),
        ("max_iterations", True),
        ("max_subspace_size", 4.5),
        ("max_subspace_size", float("nan")),
        ("max_subspace_size", float("inf")),
        ("max_subspace_size", False),
        ("tolerance", float("nan")),
        ("tolerance", float("inf")),
        ("tolerance", True),
        ("preconditioner_floor", float("nan")),
        ("preconditioner_floor", float("inf")),
        ("preconditioner_floor", False),
    ),
)
def test_davidson_config_rejects_noninteger_or_nonfinite_controls(
    field_name,
    value,
):
    controls = {
        "max_iterations": 4,
        "tolerance": 1e-5,
        "max_subspace_size": 8,
        "preconditioner_floor": 0.5,
    }
    controls[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        PeriodicDavidsonConfig(**controls)


@pytest.mark.parametrize("n_bands", (1.5, float("nan"), float("inf"), True))
def test_davidson_rejects_noninteger_or_nonfinite_band_counts(n_bands):
    _, operator = _problem(lane_label="invalid-band-count")

    with pytest.raises(ValueError, match="n_bands"):
        solve_periodic_eigenproblem(operator, n_bands=n_bands)


def test_nonconvergent_davidson_stops_at_integer_iteration_bound():
    _, operator = _problem(lane_label="bounded-nonconvergence")
    observer = RuntimeObserver()
    result = solve_periodic_eigenproblem(
        operator,
        n_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=3,
            tolerance=1e-12,
            max_subspace_size=8,
        ),
        observer=observer,
    )

    assert not result.converged
    assert result.iterations == 3
    assert [
        event["iteration"]
        for event in observer.snapshot()["events"]
        if event["event"] == "davidson_iteration"
    ] == [1, 2, 3]


def test_incremental_hv_and_projected_blocks_are_reused(monkeypatch):
    basis, operator = _problem()
    observer = RuntimeObserver()
    application_widths: list[int] = []
    projected_widths: list[int] = []
    original_apply = PeriodicKohnShamOperator._apply_compact
    original_project = periodic_scf._subspace_matrix

    def recording_apply(self, coefficients, *, observer=None):
        application_widths.append(coefficients.vector_count)
        return original_apply(self, coefficients, observer=observer)

    def recording_project(vectors, applied):
        projected_widths.append(int(vectors.shape[0]))
        return original_project(vectors, applied)

    monkeypatch.setattr(PeriodicKohnShamOperator, "_apply_compact", recording_apply)
    monkeypatch.setattr(periodic_scf, "_subspace_matrix", recording_project)

    result = solve_periodic_eigenproblem(
        operator,
        n_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=4,
            tolerance=1e-10,
            max_subspace_size=12,
        ),
        observer=observer,
    )

    work = observer.snapshot()["work_counters"]
    event_widths = [
        event["subspace_size"]
        for event in observer.snapshot()["events"]
        if event["event"] == "davidson_iteration"
    ]
    assert result.iterations == 4
    assert application_widths == [2, 2, 2, 2]
    assert projected_widths == application_widths
    assert event_widths == [2, 4, 6, 8]
    assert work["hpsi_calls"] == len(application_widths)
    assert work["davidson_hv_new_vectors"] == sum(application_widths)
    assert work["davidson_hv_reused_vectors"] == sum(event_widths[:-1])
    assert work["projected_old_old_rebuilds"] == 0


def test_restart_transforms_v_and_hv_without_reapplying_h(monkeypatch):
    _, operator = _problem(lane_label="restart")
    observer = RuntimeObserver()
    application_widths: list[int] = []
    paired_transforms: list[tuple[int, int, int]] = []
    original_apply = PeriodicKohnShamOperator._apply_compact
    original_transform = _PairedDavidsonState.transform

    def recording_apply(self, coefficients, *, observer=None):
        application_widths.append(coefficients.vector_count)
        return original_apply(self, coefficients, observer=observer)

    def recording_transform(self, transform, *, token):
        result = original_transform(self, transform, token=token)
        paired_transforms.append(
            (self.vectors.vector_count, self.applied.vector_count, result.vector_count)
        )
        return result

    monkeypatch.setattr(PeriodicKohnShamOperator, "_apply_compact", recording_apply)
    monkeypatch.setattr(_PairedDavidsonState, "transform", recording_transform)

    result = solve_periodic_eigenproblem(
        operator,
        n_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=5,
            tolerance=1e-10,
            max_subspace_size=3,
        ),
        observer=observer,
    )

    work = observer.snapshot()["work_counters"]
    assert result.restart_count >= 1
    assert application_widths[0] == 2
    assert application_widths[1:]
    assert set(application_widths[1:]) == {1}
    assert all(v_width == hv_width for v_width, hv_width, _ in paired_transforms)
    assert any(output_width == 2 for _, _, output_width in paired_transforms)
    assert work["davidson_hv_new_vectors"] == sum(application_widths)
    assert work["projected_old_old_rebuilds"] == 0

    independently_applied = original_apply(
        operator,
        result._compact_coefficients,
        observer=None,
    )
    independent_values = mx.real(
        mx.sum(
            mx.conjugate(result._compact_coefficients.values)
            * independently_applied.values,
            axis=1,
        )
    )
    independent_residuals = mx.sqrt(
        mx.sum(
            mx.abs(
                independently_applied.values
                - independent_values[:, None]
                * result._compact_coefficients.values
            )
            ** 2,
            axis=1,
        )
    )
    np.testing.assert_allclose(
        np.asarray(result.eigenvalues),
        np.asarray(independent_values),
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.residuals),
        np.asarray(independent_residuals),
        atol=3e-5,
    )


def test_fixed_hamiltonian_token_invalidates_hv_across_every_new_solve(monkeypatch):
    basis, operator = _problem(lane_label="token")
    config = PeriodicDavidsonConfig(max_iterations=3, max_subspace_size=6)
    rank_policy = _Complex64RankPolicy()
    vectors = periodic_scf._initial_coefficients(basis, 2)
    applied = operator._apply_compact(vectors)
    first = _FixedHamiltonianToken.create(operator, config, 2, rank_policy)
    paired = _PairedDavidsonState.initialize(vectors, applied, first)

    second = _FixedHamiltonianToken.create(operator, config, 2, rank_policy)
    with pytest.raises(ValueError, match="cannot cross a solve token"):
        paired.require_token(second)

    changed_operator = PeriodicKohnShamOperator(
        basis,
        operator.effective_local_potential + 0.01,
    )
    with pytest.raises(ValueError, match=r"token does not match"):
        first.validate(changed_operator, config, 2, rank_policy)
    with pytest.raises(ValueError, match=r"token does not match"):
        first.validate(
            operator,
            PeriodicDavidsonConfig(max_iterations=4, max_subspace_size=6),
            2,
            rank_policy,
        )
    with pytest.raises(ValueError, match=r"token does not match"):
        first.validate(operator, config, 1, rank_policy)
    with pytest.raises(ValueError, match=r"token does not match"):
        first.validate(
            operator,
            config,
            2,
            _Complex64RankPolicy(
                relative_tolerance=2.0 * rank_policy.relative_tolerance
            ),
        )
    with pytest.raises(ValueError, match="cached Hamiltonian action"):
        solve_periodic_eigenproblem(
            operator,
            n_bands=2,
            config=config,
            initial_coefficients=applied,
        )

    grid = basis.grid
    changed_kpoint_basis = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        basis.cutoff_hartree,
        (0.125, 0.125, 0.0),
        reciprocal_grid=basis.reciprocal_grid,
        lane_label="token:kpoint",
    )
    changed_basis_order = PlaneWaveBasis.from_reduced_kpoint(
        grid,
        basis.cutoff_hartree + 0.5,
        (0.25, 0.125, 0.0),
        reciprocal_grid=basis.reciprocal_grid,
        lane_label="token:basis",
    )
    changed_cell_grid = RealSpaceGrid((6, 6, 6), (7.0, 6.0, 6.0))
    changed_cell_basis = PlaneWaveBasis.from_reduced_kpoint(
        changed_cell_grid,
        basis.cutoff_hartree,
        (0.25, 0.125, 0.0),
        lane_label="token:cell",
    )
    context_changes = (
        PeriodicKohnShamOperator(
            changed_kpoint_basis,
            operator.effective_local_potential,
        ),
        PeriodicKohnShamOperator(
            changed_basis_order,
            operator.effective_local_potential,
        ),
        PeriodicKohnShamOperator(
            changed_cell_basis,
            mx.zeros(changed_cell_grid.shape),
        ),
        PeriodicKohnShamOperator(
            basis,
            operator.effective_local_potential.astype(mx.float16),
        ),
    )
    for changed_context in context_changes:
        with pytest.raises(ValueError, match=r"token does not match"):
            first.validate(changed_context, config, 2, rank_policy)

    original_device = str(mx.default_device())
    monkeypatch.setattr(
        periodic_scf.mx,
        "default_device",
        lambda: f"changed:{original_device}",
    )
    with pytest.raises(ValueError, match=r"token does not match"):
        first.validate(operator, config, 2, rank_policy)


def test_fixed_hamiltonian_token_invalidates_geometry_and_pseudopotential():
    basis, local_operator = _problem(lane_label="token:gth")
    config = PeriodicDavidsonConfig(max_iterations=2, max_subspace_size=4)
    rank_policy = _Complex64RankPolicy()
    pseudo = PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )
    changed_pseudo = PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.1,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )
    nonlocal_operators = (
        PeriodicGTHNonlocalOperator(pseudo, basis, ((1.0, 2.0, 3.0),)),
        PeriodicGTHNonlocalOperator(pseudo, basis, ((1.5, 2.0, 3.0),)),
        PeriodicGTHNonlocalOperator(
            changed_pseudo,
            basis,
            ((1.0, 2.0, 3.0),),
        ),
    )
    try:
        operator = PeriodicKohnShamOperator(
            basis,
            local_operator.effective_local_potential,
            nonlocal_operators[0],
        )
        token = _FixedHamiltonianToken.create(
            operator,
            config,
            1,
            rank_policy,
        )
        for changed_nonlocal in nonlocal_operators[1:]:
            changed_operator = PeriodicKohnShamOperator(
                basis,
                local_operator.effective_local_potential,
                changed_nonlocal,
            )
            with pytest.raises(ValueError, match=r"token does not match"):
                token.validate(changed_operator, config, 1, rank_policy)
    finally:
        for nonlocal_operator in nonlocal_operators:
            nonlocal_operator.close()


def test_complex64_rank_policy_deflates_or_fails_near_dependent_input():
    policy = _Complex64RankPolicy()
    first = np.zeros(8, dtype=np.complex64)
    first[0] = 1.0
    near_duplicate = first.copy()
    near_duplicate[1] = 1e-8
    independent = np.zeros(8, dtype=np.complex64)
    independent[2] = 1.0
    values = mx.array(np.stack([first, near_duplicate, independent]))

    result = policy.orthonormalize(values, required_count=2)
    assert result.deflated_count == 1
    assert result.values.shape == (2, 8)
    assert policy.overlap_error(result.values) <= policy.guard_tolerance(2)

    with pytest.raises(ValueError, match="1 vectors but 2 are required"):
        policy.orthonormalize(values[:2], required_count=2)

    basis, operator = _problem(lane_label="rank")
    seed = periodic_scf._initial_coefficients(basis, 2)
    nearly_rank_one = basis._state_from_compact(
        mx.stack(
            [
                seed.values[0],
                seed.values[0] + 1e-8 * seed.values[1],
            ]
        )
    )
    with pytest.raises(ValueError, match="1 vectors but 2 are required"):
        solve_periodic_eigenproblem(
            operator,
            n_bands=2,
            config=PeriodicDavidsonConfig(max_iterations=2, max_subspace_size=6),
            initial_coefficients=nearly_rank_one,
        )


def test_scheduler_groups_batch_one_and_many_but_failure_stays_lane_local():
    basis, operator = _problem(lane_label="scheduler")
    config = PeriodicDavidsonConfig(max_iterations=2, max_subspace_size=6)
    rank_policy = _Complex64RankPolicy()
    token = _FixedHamiltonianToken.create(operator, config, 2, rank_policy)
    one = periodic_scf._initial_coefficients(basis, 1)
    two = periodic_scf._initial_coefficients(basis, 2)
    nonfinite = basis._state_from_compact(
        mx.full((1, basis.active_count), complex(np.nan, 0.0))
    )
    observer = RuntimeObserver()
    scheduler = _DavidsonScheduler(batch_cap=1)

    result = scheduler.apply(
        [
            _DavidsonApplicationTicket(
                lane_id="good",
                operator=operator,
                config=config,
                n_bands=2,
                rank_policy=rank_policy,
                token=token,
                vectors=one,
                observer=observer,
            ),
            _DavidsonApplicationTicket(
                lane_id="peer",
                operator=operator,
                config=config,
                n_bands=2,
                rank_policy=rank_policy,
                token=token,
                vectors=one,
                observer=observer,
            ),
            _DavidsonApplicationTicket(
                lane_id="divergent",
                operator=operator,
                config=config,
                n_bands=2,
                rank_policy=rank_policy,
                token=token,
                vectors=nonfinite,
                observer=observer,
            ),
            _DavidsonApplicationTicket(
                lane_id="wide",
                operator=operator,
                config=config,
                n_bands=2,
                rank_policy=rank_policy,
                token=token,
                vectors=two,
                observer=observer,
            ),
        ]
    )

    assert result.compatibility_groups == (("good", "peer"), ("wide",))
    assert result.groups == (("good",), ("peer",), ("wide",))
    assert result.submission_count == 3
    assert set(result.actions) == {"good", "peer", "wide"}
    assert set(result.failures) == {"divergent"}
    assert result.action_for("good").vector_count == 1
    stored_failure = result.failures["divergent"]
    assert stored_failure.__traceback__ is None
    assert stored_failure.__context__ is None
    assert stored_failure.__cause__ is None
    with pytest.raises(ValueError, match="must be finite") as raised_failure:
        result.action_for("divergent")
    assert raised_failure.value is not stored_failure
    assert stored_failure.__traceback__ is None
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == result.submission_count
    assert work["fft_submissions"] == 2 * result.submission_count

    with pytest.raises(ValueError, match="batch_cap must be one"):
        _DavidsonScheduler(batch_cap=2)
    with pytest.raises(ValueError, match="batch_cap must be one"):
        _DavidsonScheduler(batch_cap=True)

    singleton = scheduler.apply(
        [
            _DavidsonApplicationTicket(
                lane_id="single",
                operator=operator,
                config=config,
                n_bands=2,
                rank_policy=rank_policy,
                token=token,
                vectors=one,
                observer=observer,
            )
        ]
    )
    assert singleton.groups == (("single",),)


def test_incremental_multilane_engine_progresses_ragged_lanes_and_submits_work():
    basis, unobserved = _problem(lane_label="engine")
    observer = RuntimeObserver()
    operator = PeriodicKohnShamOperator(
        basis,
        unobserved.effective_local_potential,
        observer=observer,
    )
    nonfinite = basis._state_from_compact(
        mx.full((1, basis.active_count), complex(np.nan, 0.0))
    )
    requests = (
        _DavidsonLaneRequest(
            lane_id="one-band",
            operator=operator,
            n_bands=1,
            config=PeriodicDavidsonConfig(
                max_iterations=5,
                tolerance=0.03,
                max_subspace_size=5,
            ),
            trial=periodic_scf._initial_coefficients(basis, 1),
            observer=observer,
        ),
        _DavidsonLaneRequest(
            lane_id="two-band",
            operator=operator,
            n_bands=2,
            config=PeriodicDavidsonConfig(
                max_iterations=5,
                tolerance=1e-10,
                max_subspace_size=4,
            ),
            trial=periodic_scf._initial_coefficients(basis, 2),
            observer=observer,
        ),
        _DavidsonLaneRequest(
            lane_id="nonfinite",
            operator=operator,
            n_bands=1,
            config=PeriodicDavidsonConfig(max_iterations=2, max_subspace_size=3),
            trial=nonfinite,
            observer=observer,
        ),
    )

    outcome = _DavidsonEngine(
        scheduler=_DavidsonScheduler(batch_cap=1)
    ).solve(requests)

    assert set(outcome.results) == {"one-band", "two-band"}
    assert set(outcome.failures) == {"nonfinite"}
    one_band = outcome.result_for("one-band")
    two_band = outcome.result_for("two-band")
    assert one_band.converged
    assert one_band.iterations == 3
    assert one_band.restart_count == 0
    assert not two_band.converged
    assert two_band.iterations == 5
    assert two_band.restart_count == 3
    assert outcome.ready_rounds == (
        ("one-band", "two-band"),
        ("one-band", "two-band"),
        ("one-band", "two-band"),
        ("two-band",),
        ("two-band",),
    )
    assert outcome.compatibility_groups[:2] == (("one-band",), ("two-band",))
    assert all(len(group) == 1 for group in outcome.submission_groups)
    assert len(outcome.submission_groups) == 8
    assert outcome.scheduler_calls == 5
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == len(outcome.submission_groups)
    assert work["fft_submissions"] == 2 * len(outcome.submission_groups)
    assert work["hpsi_vector_equivalents"] == 13
    assert work["davidson_hv_new_vectors"] == 13
    assert work["davidson_hv_reused_vectors"] == 11
    iteration_events = [
        (event["lane_id"], event["iteration"])
        for event in observer.snapshot()["events"]
        if event["event"] == "davidson_iteration"
    ]
    assert iteration_events == [
        ("one-band", 1),
        ("two-band", 1),
        ("one-band", 2),
        ("two-band", 2),
        ("one-band", 3),
        ("two-band", 3),
        ("two-band", 4),
        ("two-band", 5),
    ]


def test_incremental_multilane_engine_splits_ready_waves_at_real_batch_cap_one():
    basis, unobserved = _problem(lane_label="engine:compatible")
    observer = RuntimeObserver()
    operator = PeriodicKohnShamOperator(
        basis,
        unobserved.effective_local_potential,
        observer=observer,
    )
    config = PeriodicDavidsonConfig(
        max_iterations=2,
        tolerance=1e-10,
        max_subspace_size=3,
    )
    requests = tuple(
        _DavidsonLaneRequest(
            lane_id=lane_id,
            operator=operator,
            n_bands=1,
            config=config,
            trial=periodic_scf._initial_coefficients(basis, 1),
            observer=observer,
        )
        for lane_id in ("left", "right")
    )

    outcome = _DavidsonEngine(
        scheduler=_DavidsonScheduler(batch_cap=1)
    ).solve(requests)

    assert outcome.ready_rounds == (("left", "right"), ("left", "right"))
    assert outcome.compatibility_groups == (
        ("left", "right"),
        ("left", "right"),
    )
    assert outcome.submission_groups == (
        ("left",),
        ("right",),
        ("left",),
        ("right",),
    )
    assert outcome.scheduler_calls == 2
    assert not outcome.failures
    work = observer.snapshot()["work_counters"]
    assert work["hpsi_calls"] == 4
    assert work["fft_submissions"] == 8


def test_incremental_multilane_initial_projection_failure_is_lane_local(monkeypatch):
    basis, unobserved = _problem(lane_label="engine:projected-failure")
    observer = RuntimeObserver()
    operator = PeriodicKohnShamOperator(
        basis,
        unobserved.effective_local_potential,
        observer=observer,
    )
    original_project = periodic_scf._subspace_matrix

    def injected_projection(vectors, applied):
        if int(vectors.shape[0]) == 2:
            raise RuntimeError("injected projected-state failure")
        return original_project(vectors, applied)

    monkeypatch.setattr(periodic_scf, "_subspace_matrix", injected_projection)
    requests = (
        _DavidsonLaneRequest(
            "healthy",
            operator,
            1,
            PeriodicDavidsonConfig(max_iterations=1, max_subspace_size=3),
            periodic_scf._initial_coefficients(basis, 1),
            observer,
        ),
        _DavidsonLaneRequest(
            "failed",
            operator,
            2,
            PeriodicDavidsonConfig(max_iterations=1, max_subspace_size=4),
            periodic_scf._initial_coefficients(basis, 2),
            observer,
        ),
    )

    outcome = _DavidsonEngine().solve(requests)

    assert set(outcome.results) == {"healthy"}
    assert set(outcome.failures) == {"failed"}
    with pytest.raises(RuntimeError, match="injected projected-state failure"):
        outcome.result_for("failed")
    assert observer.snapshot()["work_counters"]["hpsi_calls"] == 2


def test_incremental_failure_records_release_frames_and_raise_fresh_errors(monkeypatch):
    basis, template = _problem(lane_label="engine:detached-failure")
    healthy_operator = PeriodicKohnShamOperator(
        basis,
        template.effective_local_potential,
    )
    failed_operator = PeriodicKohnShamOperator(
        basis,
        template.effective_local_potential,
    )
    sentinel_refs: list[weakref.ReferenceType[_FailureFrameSentinel]] = []
    original_apply = PeriodicKohnShamOperator._apply_compact

    def injected_apply(self, coefficients, *, observer=None):
        if self is failed_operator:
            sentinel = _FailureFrameSentinel()
            sentinel_refs.append(weakref.ref(sentinel))
            frame_scratch = mx.zeros(
                (coefficients.vector_count, *basis.grid.shape),
                dtype=mx.complex64,
            )
            if frame_scratch.shape[0] != coefficients.vector_count:
                raise AssertionError("unreachable failure-frame scratch check")
            raise RuntimeError("injected detached-frame failure")
        return original_apply(self, coefficients, observer=observer)

    monkeypatch.setattr(
        PeriodicKohnShamOperator,
        "_apply_compact",
        injected_apply,
    )
    config = PeriodicDavidsonConfig(max_iterations=1, max_subspace_size=3)
    requests = tuple(
        _DavidsonLaneRequest(
            lane_id,
            operator,
            1,
            config,
            periodic_scf._initial_coefficients(basis, 1),
            None,
        )
        for lane_id, operator in (
            ("healthy", healthy_operator),
            ("failed", failed_operator),
        )
    )

    outcome = _DavidsonEngine().solve(requests)
    stored = outcome.failures["failed"]
    gc.collect()

    assert set(outcome.results) == {"healthy"}
    assert outcome.result_for("healthy").iterations == 1
    assert stored.__traceback__ is None
    assert stored.__context__ is None
    assert stored.__cause__ is None
    assert len(sentinel_refs) == 1
    assert sentinel_refs[0]() is None
    with pytest.raises(RuntimeError, match="injected detached-frame failure") as first:
        outcome.result_for("failed")
    with pytest.raises(RuntimeError, match="injected detached-frame failure") as second:
        outcome.result_for("failed")
    assert first.value is not stored
    assert second.value is not stored
    assert second.value is not first.value
    assert stored.__traceback__ is None
    assert stored.__context__ is None
    assert stored.__cause__ is None


def test_nonconverged_incremental_result_keeps_public_compatibility(monkeypatch):
    basis, operator = _problem(lane_label="failure")
    application_widths: list[int] = []
    engine_lane_counts: list[int] = []
    original_apply = PeriodicKohnShamOperator._apply_compact
    original_engine_solve = _DavidsonEngine.solve

    def recording_apply(self, coefficients, *, observer=None):
        application_widths.append(coefficients.vector_count)
        return original_apply(self, coefficients, observer=observer)

    def recording_engine_solve(self, requests):
        engine_lane_counts.append(len(requests))
        return original_engine_solve(self, requests)

    monkeypatch.setattr(PeriodicKohnShamOperator, "_apply_compact", recording_apply)
    monkeypatch.setattr(_DavidsonEngine, "solve", recording_engine_solve)
    result = solve_periodic_eigenproblem(
        operator,
        n_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=1,
            tolerance=1e-12,
            max_subspace_size=6,
        ),
    )

    coefficients = np.asarray(result.coefficients)
    assert not result.converged
    assert engine_lane_counts == [1]
    assert application_widths == [2]
    assert result.eigenvalues.dtype == mx.float32
    assert result.residuals.dtype == mx.float32
    assert result._compact_coefficients.values.dtype == mx.complex64
    assert coefficients.shape == (2, *basis.grid.shape)
    assert np.count_nonzero(coefficients[:, ~np.asarray(basis.mask)]) == 0
    assert np.isfinite(np.asarray(result.eigenvalues)).all()
    assert np.isfinite(np.asarray(result.residuals)).all()
