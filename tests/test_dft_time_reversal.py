from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.benchmarks.dft_runtime_contract import _kpoint_manifest
from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
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
    admit_time_reversal_bases,
    build_time_reversal_ownership,
    run_periodic_scf,
    solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import _independent_pair
from mlx_atomistic.dft.periodic_scf import (
    PeriodicEigenResult,
    _admit_initial_time_reversal,
)
from mlx_atomistic.dft.runtime_state import serialize_periodic_scf_state


def _oracle_module():
    path = Path(__file__).parents[1] / "scripts/run_dft_runtime_oracle.py"
    spec = importlib.util.spec_from_file_location(
        "dft_time_reversal_oracle_test_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DFT runtime oracle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hydrogen_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )


def _small_system() -> PeriodicDFTSystem:
    return PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        _hydrogen_gth(),
    )


def _paired_mesh() -> KPointMesh:
    return KPointMesh(
        [
            KPoint((-0.25, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
            KPoint((0.25, 0.0, 0.0), weight=0.5, coordinate_system="reduced"),
        ]
    )


def _scf_config() -> PeriodicSCFConfig:
    return PeriodicSCFConfig(
        max_iterations=6,
        min_iterations=2,
        density_tolerance=0.2,
        energy_tolerance=0.1,
        orbital_tolerance=2e-3,
        mixing_beta=0.5,
        mixer="linear",
        davidson=PeriodicDavidsonConfig(
            max_iterations=20,
            tolerance=2e-3,
            max_subspace_size=12,
        ),
    )


def _bases(
    grid: RealSpaceGrid,
    mesh: KPointMesh,
    *,
    cutoff: float = 2.5,
) -> list[PlaneWaveBasis]:
    reciprocal = ReciprocalGrid.from_real_space(grid)
    return [
        PlaneWaveBasis.from_reduced_kpoint(
            grid,
            cutoff,
            point.vector,
            reciprocal_grid=reciprocal,
            lane_label=f"oracle:{index}",
        )
        for index, point in enumerate(mesh.points)
    ]


def _independent_permutation(
    source: PlaneWaveBasis,
    target: PlaneWaveBasis,
    source_kpoint: tuple[float, float, float],
    target_kpoint: tuple[float, float, float],
) -> np.ndarray:
    source_g = np.asarray(source.active_integer_g, dtype=np.int64)
    target_g = np.asarray(target.active_integer_g, dtype=np.int64)
    shift = np.rint(
        np.asarray(source_kpoint, dtype=np.float64)
        + np.asarray(target_kpoint, dtype=np.float64)
    ).astype(np.int64)
    target_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(target_g)
    }
    permutation = np.asarray(
        [
            target_lookup[tuple(int(value) for value in (-row - shift))]
            for row in source_g
        ],
        dtype=np.int32,
    )
    assert np.array_equal(np.sort(permutation), np.arange(permutation.size))
    return permutation


def _time_reverse_numpy(values: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    transformed = np.empty(
        (values.shape[0], permutation.size),
        dtype=np.complex64,
    )
    transformed[:, permutation] = np.conjugate(values)
    return transformed


def test_manifest_oracle_matches_canonical_ownership_and_independent_permutations():
    mesh = MonkhorstPackGrid((6, 6, 6))
    manifest = _kpoint_manifest(6)
    topology = build_time_reversal_ownership(mesh)

    assert len(topology.entries) == len(manifest) == 216
    assert topology.owned_indices == tuple(range(108))
    for entry, expected in zip(topology.entries, manifest, strict=True):
        assert entry.explicit_index == expected["index"]
        np.testing.assert_allclose(
            entry.reduced_kpoint,
            expected["reduced_coordinates"],
            atol=0.0,
        )
        assert entry.original_weight == pytest.approx(
            expected["weight"]["numerator"] / expected["weight"]["denominator"],
            abs=0.0,
        )
        assert entry.owner_index == expected["owner_index"]
        assert entry.partner_index == expected["partner_index"]
        assert entry.role == expected["role"]

    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    bases = _bases(grid, mesh, cutoff=1.0)
    admitted = admit_time_reversal_bases(topology, bases)
    assert admitted.owned_indices == tuple(range(108))
    for entry in admitted.entries:
        partner_index = entry.partner_index
        assert partner_index is not None
        expected = _independent_permutation(
            bases[entry.explicit_index],
            bases[partner_index],
            entry.reduced_kpoint,
            admitted.entry_for(partner_index).reduced_kpoint,
        )
        np.testing.assert_array_equal(entry.time_reversal_permutation, expected)
    original = admitted.entries[0].time_reversal_permutation
    leaked = admitted.entries[0].time_reversal_permutation
    leaked[:] = 0
    leaked.shape = (leaked.size, 1)
    np.testing.assert_array_equal(
        admitted.entries[0].time_reversal_permutation,
        original,
    )


@pytest.mark.parametrize(
    ("vector", "weight"),
    [
        ((np.nan, 0.0, 0.0), 1.0),
        ((np.inf, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 0.0), np.nan),
        ((0.0, 0.0, 0.0), np.inf),
    ],
)
def test_nonfinite_kpoint_coordinates_and_weights_fail_closed(vector, weight):
    with pytest.raises(ValueError, match="finite"):
        KPoint(vector, weight=weight, coordinate_system="reduced")


def test_duplicate_and_multiple_partner_maps_fail_closed():
    duplicate = KPointMesh(
        [
            KPoint((0.2, 0.0, 0.0), coordinate_system="reduced"),
            KPoint((1.2, 0.0, 0.0), coordinate_system="reduced"),
        ]
    )
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        build_time_reversal_ownership(duplicate)

    epsilon = 0.75e-10
    multiply_claimed = KPointMesh(
        [
            KPoint((0.2, 0.0, 0.0), coordinate_system="reduced"),
            KPoint((-0.2 + epsilon, 0.0, 0.0), coordinate_system="reduced"),
            KPoint((-0.2 - epsilon, 0.0, 0.0), coordinate_system="reduced"),
        ]
    )
    with pytest.raises(ValueError, match="multiple time-reversal partners"):
        build_time_reversal_ownership(multiply_claimed)


def test_missing_partner_and_unequal_weight_fall_back_only_affected_points():
    missing = build_time_reversal_ownership(
        KPointMesh([KPoint((0.2, 0.0, 0.0), coordinate_system="reduced")])
    )
    assert missing.owned_indices == (0,)
    assert missing.entries[0].role == "independent"
    assert missing.entries[0].fallback_reason == "missing_time_reversal_partner"

    unequal = build_time_reversal_ownership(
        KPointMesh(
            [
                KPoint((0.2, 0.0, 0.0), weight=0.4, coordinate_system="reduced"),
                KPoint((-0.2, 0.0, 0.0), weight=0.6, coordinate_system="reduced"),
                KPoint((0.0, 0.0, 0.0), weight=1.0, coordinate_system="reduced"),
            ]
        )
    )
    assert unequal.owned_indices == (0, 1, 2)
    assert unequal.entries[0].fallback_reason == "unequal_time_reversal_weight"
    assert unequal.entries[1].fallback_reason == "unequal_time_reversal_weight"
    assert unequal.entries[2].role == "owner"
    assert unequal.entries[2].fallback_reason is None


def test_self_inverse_boundary_and_reciprocal_shift_permutations_are_exact():
    grid = RealSpaceGrid((7, 7, 7), (7.0, 7.0, 7.0))
    self_inverse_mesh = KPointMesh(
        [
            KPoint((0.0, 0.0, 0.0), coordinate_system="reduced"),
            KPoint((0.5, 0.0, 0.0), coordinate_system="reduced"),
        ]
    )
    self_inverse = build_time_reversal_ownership(self_inverse_mesh)
    bases = _bases(grid, self_inverse_mesh, cutoff=1.5)
    admitted = admit_time_reversal_bases(self_inverse, bases)
    assert admitted.owned_indices == (0, 1)
    assert admitted.entries[0].partner_index == 0
    assert admitted.entries[1].partner_index == 1
    assert admitted.entries[1].reciprocal_shift == (1, 0, 0)
    for index in range(2):
        expected = _independent_permutation(
            bases[index],
            bases[index],
            admitted.entries[index].reduced_kpoint,
            admitted.entries[index].reduced_kpoint,
        )
        np.testing.assert_array_equal(
            admitted.entries[index].time_reversal_permutation,
            expected,
        )

    shifted_mesh = KPointMesh(
        [
            KPoint((0.75, 0.0, 0.0), coordinate_system="reduced"),
            KPoint((0.25, 0.0, 0.0), coordinate_system="reduced"),
        ]
    )
    shifted_bases = _bases(grid, shifted_mesh, cutoff=1.5)
    shifted = admit_time_reversal_bases(
        build_time_reversal_ownership(shifted_mesh),
        shifted_bases,
    )
    assert shifted.owned_indices == (0,)
    assert shifted.entries[0].reciprocal_shift == (1, 0, 0)
    np.testing.assert_array_equal(
        shifted.entries[0].time_reversal_permutation,
        _independent_permutation(
            shifted_bases[0],
            shifted_bases[1],
            shifted.entries[0].reduced_kpoint,
            shifted.entries[1].reduced_kpoint,
        ),
    )


def test_nonbijective_active_bases_fall_back_to_independent_lanes():
    grid = RealSpaceGrid((7, 7, 7), (7.0, 7.0, 7.0))
    mesh = _paired_mesh()
    reciprocal = ReciprocalGrid.from_real_space(grid)
    bases = [
        PlaneWaveBasis.from_reduced_kpoint(
            grid,
            cutoff,
            point.vector,
            reciprocal_grid=reciprocal,
            lane_label=f"mismatch:{index}",
        )
        for index, (point, cutoff) in enumerate(
            zip(mesh.points, (1.0, 2.0), strict=True)
        )
    ]
    admitted = admit_time_reversal_bases(
        build_time_reversal_ownership(mesh),
        bases,
    )
    assert admitted.owned_indices == (0, 1)
    assert admitted.fallback_reasons == {
        0: "active_basis_time_reversal_mismatch",
        1: "active_basis_time_reversal_mismatch",
    }


def test_active_basis_admission_keeps_large_mlx_axes_off_numpy(monkeypatch):
    import mlx_atomistic.dft.kpoints as kpoints_module

    grid = RealSpaceGrid((7, 7, 7), (7.0, 7.0, 7.0))
    mesh = _paired_mesh()
    bases = _bases(grid, mesh, cutoff=1.5)
    topology = build_time_reversal_ownership(mesh)

    def forbidden_public_adapter(_self):
        raise AssertionError("active-basis admission read a public array adapter")

    monkeypatch.setattr(
        PlaneWaveBasis,
        "active_integer_g",
        property(forbidden_public_adapter),
    )
    monkeypatch.setattr(
        PlaneWaveBasis,
        "active_shifted_vectors",
        property(forbidden_public_adapter),
    )
    original_asarray = kpoints_module.np.asarray

    def reject_mlx_to_numpy(values, *args, **kwargs):
        if isinstance(values, mx.array):
            raise AssertionError("active MLX axis transferred to NumPy")
        return original_asarray(values, *args, **kwargs)

    monkeypatch.setattr(kpoints_module.np, "asarray", reject_mlx_to_numpy)
    admitted = admit_time_reversal_bases(topology, bases)

    assert admitted.owned_indices == (0,)
    assert all(
        entry._time_reversal_permutation is not None
        for entry in admitted.entries
    )


def test_gauge_and_degenerate_initial_subspaces_reuse_but_mismatch_preserves_both():
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    mesh = _paired_mesh()
    bases = _bases(grid, mesh)
    topology = admit_time_reversal_bases(
        build_time_reversal_ownership(mesh),
        bases,
    )
    permutation = np.asarray(topology.entries[0].time_reversal_permutation)
    rng = np.random.default_rng(73)
    owner_values = rng.normal(size=(3, bases[0].active_count)) + 1j * rng.normal(
        size=(3, bases[0].active_count)
    )
    owner_values = np.asarray(
        bases[0]._orthonormalize_compact(
            mx.array(owner_values.astype(np.complex64))
        )
    )
    partner_values = _time_reverse_numpy(owner_values, permutation)
    unitary = np.asarray(
        [[1.0, 1.0], [-1.0, 1.0]],
        dtype=np.complex64,
    ) / np.sqrt(2.0)
    nonoccupied = rng.normal(size=(1, bases[1].active_count)) + 1j * rng.normal(
        size=(1, bases[1].active_count)
    )
    rotated = np.concatenate(
        [unitary @ partner_values[:2], nonoccupied.astype(np.complex64)],
        axis=0,
    )
    initial = [
        bases[0]._layout.unpack_fresh(mx.array(owner_values)),
        bases[1]._layout.unpack_fresh(mx.array(rotated)),
    ]

    gauge_admitted, gauge_states = _admit_initial_time_reversal(
        topology,
        bases,
        initial,
        n_bands=2,
    )
    assert gauge_admitted.owned_indices == (0,)
    assert tuple(gauge_states) == (0,)

    full_span_same_but_occupied_differs = partner_values[[0, 2, 1]]
    mismatched, mismatch_states = _admit_initial_time_reversal(
        topology,
        bases,
        [
            initial[0],
            bases[1]._layout.unpack_fresh(
                mx.array(full_span_same_but_occupied_differs)
            ),
        ],
        n_bands=2,
    )
    assert mismatched.owned_indices == (0, 1)
    assert tuple(mismatch_states) == (0, 1)
    assert mismatched.fallback_reasons == {
        0: "initial_coefficients_time_reversal_mismatch",
        1: "initial_coefficients_time_reversal_mismatch",
    }

    too_short, short_states = _admit_initial_time_reversal(
        topology,
        bases,
        [initial[0], bases[1]._layout.unpack_fresh(mx.array(partner_values[:1]))],
        n_bands=2,
    )
    assert too_short.owned_indices == (0, 1)
    assert tuple(short_states) == (0, 1)


def test_independent_k_and_minus_k_solutions_match_gauge_invariant_oracles():
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    mesh = _paired_mesh()
    bases = _bases(grid, mesh)
    coordinates = grid.coordinates()
    potential = 0.2 + 0.1 * mx.cos(2.0 * np.pi * coordinates[..., 0] / 6.0)
    results = []
    nonlocal_operators = []
    try:
        for basis in bases:
            nonlocal_operator = PeriodicGTHNonlocalOperator(
                _hydrogen_gth(),
                basis,
                ((1.5, 2.0, 3.0),),
            )
            nonlocal_operators.append(nonlocal_operator)
            results.append(
                solve_periodic_eigenproblem(
                    PeriodicKohnShamOperator(
                        basis,
                        potential,
                        nonlocal_operator,
                    ),
                    n_bands=2,
                    config=PeriodicDavidsonConfig(
                        max_iterations=30,
                        tolerance=2e-4,
                        max_subspace_size=16,
                    ),
                )
            )
    finally:
        for operator in nonlocal_operators:
            operator.close()

    np.testing.assert_allclose(
        np.asarray(results[0].eigenvalues),
        np.asarray(results[1].eigenvalues),
        atol=4e-4,
    )
    permutation = _independent_permutation(
        bases[0],
        bases[1],
        mesh.points[0].vector,
        mesh.points[1].vector,
    )
    owner_values = np.asarray(results[0]._compact_coefficients.values)
    expected_partner = _time_reverse_numpy(owner_values, permutation)
    actual_partner = np.asarray(results[1]._compact_coefficients.values)
    overlap = expected_partner @ np.conjugate(actual_partner.T)
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, atol=5e-3)
    expected_density = np.sum(
        np.abs(np.asarray(bases[1]._to_real_compact(
            bases[1]._state_from_compact(mx.array(expected_partner))
        ))) ** 2,
        axis=0,
    )
    actual_density = np.sum(
        np.abs(np.asarray(bases[1]._to_real_compact(
            results[1]._compact_coefficients
        ))) ** 2,
        axis=0,
    )
    np.testing.assert_allclose(actual_density, expected_density, atol=5e-4)
    assert 2.0 * float(mx.sum(results[0].eigenvalues)) == pytest.approx(
        2.0 * float(mx.sum(results[1].eigenvalues)),
        abs=8e-4,
    )


def test_owner_only_scf_lazy_public_views_and_persistence(monkeypatch, tmp_path):
    observer = RuntimeObserver(synchronize=mx.synchronize)
    result = run_periodic_scf(
        _small_system(),
        cutoff_hartree=2.5,
        kpoint_mesh=_paired_mesh(),
        n_bands=1,
        config=_scf_config(),
        observer=observer,
    )

    assert len(result.kpoints) == 2
    assert len(result.owned_kpoints) == 1
    assert result.time_reversal_ownership is not None
    assert result.time_reversal_ownership.owned_indices == (0,)
    owner, partner = result.kpoints
    assert owner.eigen._compact_coefficients is not None
    assert partner.eigen._compact_coefficients is None
    assert not owner.eigen.is_time_reversal_view
    assert partner.eigen.is_time_reversal_view
    assert [item.reduced_kpoint for item in result.kpoints] == [
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    ]
    assert [item.weight for item in result.kpoints] == [0.5, 0.5]
    np.testing.assert_array_equal(owner.eigen.eigenvalues, partner.eigen.eigenvalues)

    before_snapshot = observer.snapshot()
    before = before_snapshot["work_counters"]
    owner_bytes = int(np.prod(owner.eigen._compact_coefficients.values.shape)) * 8
    assert before_snapshot["memory"]["persistent_coefficient_bytes"] == owner_bytes
    assert before_snapshot["memory"]["coefficient_payload_bytes"] == owner_bytes
    first = partner.eigen.coefficients
    mx.eval(first)
    first[:] = mx.zeros_like(first)
    second = partner.eigen.coefficients
    mx.eval(second)
    after = observer.snapshot()["work_counters"]
    assert first is not second
    assert second.dtype == mx.complex64
    assert second.shape == (1, *partner.basis.grid.shape)
    assert np.count_nonzero(np.asarray(second)) > 0
    assert np.count_nonzero(
        np.asarray(second)[:, ~np.asarray(partner.basis.mask)]
    ) == 0
    assert before["kpoint_lane_solves"] == result.iterations
    assert after["kpoint_lane_solves"] == before["kpoint_lane_solves"]
    assert after["partner_reconstructions"] == before["partner_reconstructions"] + 2
    assert "coefficients" not in vars(partner.eigen)

    entry = result.time_reversal_ownership.entries[0]
    expected = _time_reverse_numpy(
        np.asarray(owner.eigen._compact_coefficients.values),
        np.asarray(entry.time_reversal_permutation),
    )
    observed, _ = partner.basis._state_from_full(second)
    np.testing.assert_allclose(np.asarray(observed.values), expected, atol=3e-6)

    def forbidden(_self):
        raise AssertionError("serializer read the public coefficient adapter")

    monkeypatch.setattr(PeriodicEigenResult, "coefficients", property(forbidden))
    payloads = serialize_periodic_scf_state(result)
    serialized_work = observer.snapshot()["work_counters"]
    assert serialized_work["partner_reconstructions"] == after["partner_reconstructions"]
    metadata = json.loads(payloads["metadata.json"])
    assert metadata["schema_version"] == "mlx-atomistic.periodic-scf-compact-state.v2"
    assert metadata["kpoint_count"] == 2
    assert metadata["owned_lane_count"] == 1
    lane = metadata["owned_lanes"][0]
    assert lane["owner_index"] == 0
    assert lane["explicit_indices"] == [0, 1]
    assert lane["aggregate_weight"] == 1.0
    assert lane["reduced_kpoint"] == [-0.25, 0.0, 0.0]
    assert lane["compact_coefficient_file"] in payloads
    assert lane["compact_index_file"] in payloads
    assert lane["eigenvalue_file"] in payloads
    assert not any("0001" in name for name in payloads)

    for relative, content in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    oracle = _oracle_module()
    loaded_metadata, loaded_lanes = oracle._state_lanes(tmp_path)
    assert loaded_metadata == metadata
    assert loaded_lanes == [lane]
    dense, eigenvalues = oracle._load_lane_state(
        tmp_path,
        loaded_lanes[0],
        owner.basis.grid.shape,
    )
    expected_dense = np.asarray(
        owner.eigen._compact_coefficients.layout.unpack_fresh(
            owner.eigen._compact_coefficients.values
        )
    )
    np.testing.assert_array_equal(dense, expected_dense)
    np.testing.assert_array_equal(eigenvalues, np.asarray(owner.eigen.eigenvalues))


def test_mismatched_initial_coefficients_preserve_and_solve_both_explicit_lanes():
    system = _small_system()
    mesh = _paired_mesh()
    bases = _bases(system.grid, mesh)
    topology = admit_time_reversal_bases(
        build_time_reversal_ownership(mesh),
        bases,
    )
    permutation = np.asarray(topology.entries[0].time_reversal_permutation)
    owner_values = np.zeros((1, bases[0].active_count), dtype=np.complex64)
    partner_values = np.zeros((1, bases[1].active_count), dtype=np.complex64)
    owner_values[0, 0] = 1.0
    partner_values[0, (int(permutation[0]) + 1) % bases[1].active_count] = 1.0
    initial = [
        bases[0]._layout.unpack_fresh(mx.array(owner_values)),
        bases[1]._layout.unpack_fresh(mx.array(partner_values)),
    ]
    observer = RuntimeObserver(synchronize=mx.synchronize)

    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=_scf_config(),
        initial_coefficients=initial,
        observer=observer,
    )

    assert result.time_reversal_ownership is not None
    assert result.time_reversal_ownership.owned_indices == (0, 1)
    assert result.time_reversal_ownership.fallback_reasons == {
        0: "initial_coefficients_time_reversal_mismatch",
        1: "initial_coefficients_time_reversal_mismatch",
    }
    assert len(result.owned_kpoints) == 2
    assert all(not item.eigen.is_time_reversal_view for item in result.kpoints)
    work = observer.snapshot()["work_counters"]
    assert work["kpoint_lane_solves"] == 2 * result.iterations
    assert work["representative_lane_solves"] == 0


def test_representative_and_independent_small_scf_trajectories_match(monkeypatch):
    import mlx_atomistic.dft.periodic_scf as periodic_scf

    representative_observer = RuntimeObserver(synchronize=mx.synchronize)
    representative = run_periodic_scf(
        _small_system(),
        cutoff_hartree=2.5,
        kpoint_mesh=_paired_mesh(),
        n_bands=1,
        config=_scf_config(),
        observer=representative_observer,
    )

    original_admit = periodic_scf.admit_time_reversal_bases

    def force_independent(topology, bases):
        admitted = original_admit(topology, bases)
        return _independent_pair(admitted, 0, "independent_oracle")

    monkeypatch.setattr(
        periodic_scf,
        "admit_time_reversal_bases",
        force_independent,
    )
    independent_observer = RuntimeObserver(synchronize=mx.synchronize)
    independent = run_periodic_scf(
        _small_system(),
        cutoff_hartree=2.5,
        kpoint_mesh=_paired_mesh(),
        n_bands=1,
        config=_scf_config(),
        observer=independent_observer,
    )

    assert len(representative.owned_kpoints) == 1
    assert len(independent.owned_kpoints) == 2
    np.testing.assert_allclose(
        np.asarray(representative.density),
        np.asarray(independent.density),
        atol=4e-5,
    )
    assert representative.total_energy == pytest.approx(
        independent.total_energy,
        abs=5e-5,
    )
    for term, value in representative.energy_by_term.items():
        assert value == pytest.approx(independent.energy_by_term[term], abs=5e-5)
    assert len(representative.history) == len(independent.history)
    for reused_row, independent_row in zip(
        representative.history,
        independent.history,
        strict=True,
    ):
        assert reused_row["total_energy_hartree"] == pytest.approx(
            independent_row["total_energy_hartree"],
            abs=5e-5,
        )
        assert reused_row["density_residual"] == pytest.approx(
            independent_row["density_residual"],
            abs=5e-5,
        )
    reused_work = representative_observer.snapshot()["work_counters"]
    independent_work = independent_observer.snapshot()["work_counters"]
    assert reused_work["kpoint_lane_solves"] == representative.iterations
    assert independent_work["kpoint_lane_solves"] == 2 * independent.iterations
    assert reused_work["representative_lane_solves"] == representative.iterations
    assert independent_work["representative_lane_solves"] == 0
