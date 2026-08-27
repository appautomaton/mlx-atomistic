from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import pi, sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_atomistic.dft.periodic_upf as periodic_upf
from mlx_atomistic.dft import (
    BandPath,
    KPoint,
    KPointMesh,
    NonlocalProjectorData,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicKohnShamOperator,
    PeriodicSCFConfig,
    PeriodicUPFNonlocalOperator,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RadialGrid,
    RealSpaceGrid,
    RuntimeObserver,
    gth_local_reciprocal_coefficients,
    periodic_analytic_stress,
    periodic_finite_difference_stress,
    periodic_gth_local_forces,
    periodic_scf_calculation_contract,
    periodic_scf_forces,
    periodic_upf_local_forces,
    read_upf,
    run_periodic_band_structure,
    run_periodic_scf,
    upf_local_potential_grid,
    upf_local_reciprocal_coefficients,
)
from mlx_atomistic.dft._compact import _CompactBatch


def _upf_text(*, dij: str = "2.0 0.4 0.4 4.0", weights: str = "0.01 0.09 0.9") -> str:
    return f"""<UPF version="2.0.1">
<PP_HEADER element="Mg" z_valence="2" pseudo_type="NC"
 relativistic="scalar" is_ultrasoft="F" is_paw="F" has_so="F"
 core_correction="F" functional="PBE"/>
<PP_MESH>
  <PP_R>0.01 0.1 1.0</PP_R>
  <PP_RAB>{weights}</PP_RAB>
</PP_MESH>
<PP_LOCAL>-2.0 -1.0 -0.2</PP_LOCAL>
<PP_NONLOCAL>
  <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius="0.8">
    0.0 0.2 0.0
  </PP_BETA.1>
  <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius="0.9">
    0.0 0.3 0.0
  </PP_BETA.2>
  <PP_DIJ>{dij}</PP_DIJ>
</PP_NONLOCAL>
</UPF>
"""


def test_upf_parser_preserves_radial_quadrature_and_full_dij(tmp_path):
    source = tmp_path / "Mg.nc.UPF"
    source.write_text(_upf_text())

    pseudo = read_upf(source)

    assert pseudo.format == PseudopotentialFormat.UPF
    assert pseudo.periodic_upf_compatible is True
    np.testing.assert_allclose(
        pseudo.local_grid.integration_weights,
        (0.01, 0.09, 0.9),
    )
    assert pseudo.nonlocal_coupling_matrix == ((1.0, 0.2), (0.2, 2.0))
    assert pseudo.nonlocal_projectors[0].coefficients == (1.0, 0.2)
    assert pseudo.nonlocal_projectors[1].coefficients == (0.2, 2.0)
    assert pseudo.nonlocal_projectors[0].coupling == pytest.approx(1.0)
    assert pseudo.nonlocal_projectors[1].coupling == pytest.approx(2.0)
    assert all(
        projector.metadata["radial_representation"] == "r_beta"
        for projector in pseudo.nonlocal_projectors
    )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ('pseudo_type="NC"', 'pseudo_type="US"'),
        ('relativistic="scalar"', 'relativistic="full"'),
        ('has_so="F"', 'has_so="T"'),
        ('core_correction="F"', 'core_correction="T"'),
    ),
)
def test_upf_periodic_boundary_rejects_unsupported_physics(tmp_path, old, new):
    source = tmp_path / "unsupported.UPF"
    source.write_text(_upf_text().replace(old, new))

    assert read_upf(source).periodic_upf_compatible is False


@pytest.mark.parametrize(
    ("dij", "message"),
    (
        ("2.0 0.4 4.0", "size must match"),
        ("2.0 0.3 0.4 4.0", "finite and symmetric"),
    ),
)
def test_upf_parser_rejects_invalid_dij(tmp_path, dij, message):
    source = tmp_path / "invalid-dij.UPF"
    source.write_text(_upf_text(dij=dij))

    with pytest.raises(ValueError, match=message):
        read_upf(source)


def test_upf_parser_rejects_mismatched_radial_quadrature(tmp_path):
    source = tmp_path / "invalid-grid.UPF"
    source.write_text(_upf_text(weights="0.01 0.99"))

    with pytest.raises(ValueError, match="sizes do not match"):
        read_upf(source)


def test_upf_parser_rejects_ambiguous_projector_order(tmp_path):
    source = tmp_path / "duplicate-projector.UPF"
    source.write_text(_upf_text().replace('PP_BETA.1 index="1"', 'PP_BETA.1 index="2"'))

    with pytest.raises(ValueError, match="unique and contiguous"):
        read_upf(source)


def _matched_gth_and_numerical_upf():
    gth = PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.GTH,
        valence_charge=4.0,
        gth_rloc=0.44,
        gth_coefficients=(-6.26928833,),
    )
    radii = np.linspace(0.0, 12.0, 6001, dtype=np.float64)
    spacing = float(radii[1] - radii[0])
    upf = PseudopotentialData(
        element="Si",
        format=PseudopotentialFormat.UPF,
        valence_charge=4.0,
        local_grid=RadialGrid(
            radii,
            gth.local_potential(radii),
            integration_weights=np.full(radii.shape, spacing),
        ),
    )
    return gth, upf


def _synthetic_periodic_upf() -> PseudopotentialData:
    _gth, local_upf = _matched_gth_and_numerical_upf()
    radii = local_upf.local_grid.radii

    def projector(l_value: int, alpha: float) -> NonlocalProjectorData:
        return NonlocalProjectorData(
            angular_momentum=l_value,
            values=tuple(radii ** (l_value + 1) * np.exp(-alpha * radii**2)),
            radial_grid=local_upf.local_grid,
            metadata={"radial_representation": "r_beta"},
        )

    return replace(
        local_upf,
        nonlocal_projectors=(
            projector(0, 0.65),
            projector(0, 1.10),
            projector(1, 0.80),
            projector(2, 0.95),
        ),
        nonlocal_coupling_matrix=(
            (1.20, -0.15, 0.0, 0.0),
            (-0.15, 0.85, 0.0, 0.0),
            (0.0, 0.0, 0.70, 0.0),
            (0.0, 0.0, 0.0, 0.45),
        ),
        metadata={
            "version": "synthetic-test-v1",
            "pseudo_type": "NC",
            "relativistic": "scalar",
            "functional": "PBE",
            "is_ultrasoft": False,
            "is_paw": False,
            "has_so": False,
            "core_correction": False,
        },
    )


def test_periodic_upf_local_transform_matches_qe_gth_oracle():
    gth, upf = _matched_gth_and_numerical_upf()
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    position = ((1.1, 2.2, 3.3),)

    expected = np.asarray(
        gth_local_reciprocal_coefficients(gth, basis, position)
    )
    observed = np.asarray(
        upf_local_reciprocal_coefficients(upf, basis, position)
    )
    potential = np.asarray(upf_local_potential_grid(upf, basis, position))

    np.testing.assert_allclose(observed, expected, rtol=2.0e-5, atol=2.0e-7)
    assert np.isfinite(potential).all()


def test_periodic_upf_local_force_matches_qe_gth_oracle(monkeypatch):
    gth, upf = _matched_gth_and_numerical_upf()
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 4.0)
    positions = ((1.1, 2.2, 3.3), (5.0, 4.0, 2.0))
    coordinates = np.asarray(grid.coordinates())
    density = (
        0.02
        + 0.004 * np.cos(2.0 * np.pi * coordinates[..., 0] / 8.0)
        + 0.003 * np.sin(2.0 * np.pi * coordinates[..., 1] / 8.0)
    ).astype(np.float32)
    transform = periodic_upf._upf_local_radial_transform
    transform_calls = 0

    def observed_transform(*args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return transform(*args, **kwargs)

    monkeypatch.setattr(
        periodic_upf,
        "_upf_local_radial_transform",
        observed_transform,
    )

    expected = np.asarray(
        periodic_gth_local_forces(density, gth, basis, positions)
    )
    observed = np.asarray(
        periodic_upf_local_forces(density, upf, basis, positions)
    )

    np.testing.assert_allclose(observed, expected, rtol=3.0e-5, atol=2.0e-7)
    assert transform_calls == 1


def test_periodic_upf_identity_is_content_bound_and_path_independent(tmp_path):
    first_path = tmp_path / "first.UPF"
    second_path = tmp_path / "nested" / "second.UPF"
    changed_path = tmp_path / "changed.UPF"
    second_path.parent.mkdir()
    first_path.write_text(_upf_text())
    second_path.write_text(_upf_text())
    changed_path.write_text(_upf_text(dij="2.0 0.6 0.6 4.0"))

    def system(path):
        return PeriodicDFTSystem(
            (8.0, 8.0, 8.0),
            (8, 8, 8),
            ((1.0, 2.0, 3.0),),
            read_upf(path),
        )

    assert system(first_path).fingerprint == system(second_path).fingerprint
    assert system(first_path).fingerprint != system(changed_path).fingerprint


@pytest.mark.parametrize("angular_momentum", (0, 1, 2))
def test_upf_nonlocal_radial_transform_matches_analytic_oracle(
    angular_momentum,
):
    alpha = 0.73
    radii = np.linspace(0.0, 12.0, 12001, dtype=np.float64)
    spacing = float(radii[1] - radii[0])
    stored_beta = radii ** (angular_momentum + 1) * np.exp(-alpha * radii**2)
    projector = NonlocalProjectorData(
        angular_momentum=angular_momentum,
        values=tuple(stored_beta),
        radial_grid=RadialGrid(
            radii,
            np.zeros_like(radii),
            integration_weights=np.full(radii.shape, spacing),
        ),
    )
    q = np.asarray((0.0, 0.37, 1.25), dtype=np.float64)
    volume = 91.0

    observed = periodic_upf._upf_projector_radial_transform(
        projector,
        q,
        volume=volume,
    )
    expected = (
        4.0
        * pi
        / sqrt(volume)
        * sqrt(pi)
        * q**angular_momentum
        / (2.0 ** (angular_momentum + 2) * alpha ** (angular_momentum + 1.5))
        * np.exp(-(q**2) / (4.0 * alpha))
    )

    np.testing.assert_allclose(observed, expected, rtol=2.0e-11, atol=2.0e-12)


def test_periodic_upf_batches_radial_setup_by_q_and_angular_channel(monkeypatch):
    grid = RealSpaceGrid((12, 12, 12), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis(grid, 8.0)
    pseudo = _synthetic_periodic_upf()
    calls = []
    original = periodic_upf.spherical_jn

    def observed_spherical_jn(angular_momentum, values):
        calls.append((angular_momentum, values.shape[0]))
        return original(angular_momentum, values)

    monkeypatch.setattr(periodic_upf, "spherical_jn", observed_spherical_jn)
    operator = PeriodicUPFNonlocalOperator(
        pseudo,
        basis,
        ((1.0, 2.0, 3.0),),
    )

    assert [angular_momentum for angular_momentum, _size in calls] == [0, 1, 2]
    assert max(size for _angular_momentum, size in calls) < basis.active_count
    operator.close()


def test_periodic_upf_nonlocal_is_hermitian_and_preserves_full_dij():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, 0.125, -0.25))
    pseudo = _synthetic_periodic_upf()
    operator = PeriodicUPFNonlocalOperator(pseudo, basis, ((1.0, 2.0, 3.0),))
    rng = np.random.default_rng(72)
    left = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(
                np.complex64
            )
        )
    )
    right = basis.normalize(
        mx.array(
            (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(
                np.complex64
            )
        )
    )

    left_right = mx.sum(mx.conjugate(left) * operator.apply(right))
    right_left = mx.sum(mx.conjugate(right) * operator.apply(left))
    expected_coupling = np.zeros((10, 10), dtype=np.float32)
    expected_coupling[:2, :2] = ((1.20, -0.15), (-0.15, 0.85))
    expected_coupling[2:5, 2:5] = np.eye(3, dtype=np.float32) * 0.70
    expected_coupling[5:, 5:] = np.eye(5, dtype=np.float32) * 0.45

    np.testing.assert_allclose(
        np.asarray(left_right),
        np.asarray(mx.conjugate(right_left)),
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(operator._flattened_coupling),
        expected_coupling,
        atol=1.0e-7,
    )
    assert operator.to_dict()["angular_projector_count_per_ion"] == 10


def test_periodic_upf_nonlocal_forces_match_fixed_orbital_derivative():
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    basis = PlaneWaveBasis.from_reduced_kpoint(grid, 4.0, (0.25, 0.125, -0.25))
    pseudo = _synthetic_periodic_upf()
    position = np.asarray(((1.0, 2.0, 3.0),), dtype=np.float64)
    rng = np.random.default_rng(81)
    trial = rng.normal(size=(2, *grid.shape)) + 1j * rng.normal(
        size=(2, *grid.shape)
    )
    orbitals = basis.orthonormalize(mx.array(trial.astype(np.complex64)))
    occupations = (2.0, 0.75)

    observed = np.asarray(
        PeriodicUPFNonlocalOperator(pseudo, basis, position).forces(
            orbitals,
            occupations=occupations,
        )
    )
    reference = np.zeros_like(position)
    displacement = 2.0e-3
    for axis in range(3):
        plus = position.copy()
        minus = position.copy()
        plus[0, axis] += displacement
        minus[0, axis] -= displacement
        energy_plus = float(
            PeriodicUPFNonlocalOperator(pseudo, basis, plus).energy(
                orbitals,
                occupations=occupations,
            )
        )
        energy_minus = float(
            PeriodicUPFNonlocalOperator(pseudo, basis, minus).energy(
                orbitals,
                occupations=occupations,
            )
        )
        reference[0, axis] = -(energy_plus - energy_minus) / (2.0 * displacement)

    np.testing.assert_allclose(observed, reference, atol=8.0e-5, rtol=5.0e-4)


def test_periodic_upf_nonlocal_uses_shared_kpoint_batch_path(monkeypatch):
    grid = RealSpaceGrid((8, 8, 8), (8.0, 8.0, 8.0))
    bases = (
        PlaneWaveBasis.from_reduced_kpoint(grid, 3.0, (0.0, 0.0, 0.0)),
        PlaneWaveBasis.from_reduced_kpoint(grid, 3.0, (0.25, 0.0, 0.0)),
    )
    pseudo = _synthetic_periodic_upf()
    rng = np.random.default_rng(91)
    states = tuple(
        basis._state_from_compact(
            mx.array(
                (
                    rng.normal(size=(2, basis.active_count))
                    + 1j * rng.normal(size=(2, basis.active_count))
                ).astype(np.complex64)
            )
        )
        for basis in bases
    )
    nonlocal_operators = tuple(
        PeriodicUPFNonlocalOperator(pseudo, basis, ((1.0, 2.0, 3.0),))
        for basis in bases
    )
    potential = mx.full(grid.shape, 0.2, dtype=mx.float32)
    operators = tuple(
        PeriodicKohnShamOperator(basis, potential, nonlocal_operator)
        for basis, nonlocal_operator in zip(bases, nonlocal_operators, strict=True)
    )
    expected = tuple(
        operator._apply_compact(state)
        for operator, state in zip(operators, states, strict=True)
    )
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
        prepared_batch=_CompactBatch.from_states(states),
    )

    assert not outcome.failures
    assert matmul_calls == 3
    for index, reference in enumerate(expected):
        np.testing.assert_allclose(
            np.asarray(outcome.action_for(index).values),
            np.asarray(reference.values),
            atol=3.0e-6,
        )


def test_periodic_upf_nonlocal_rejects_cross_angular_dij():
    pseudo = replace(
        _synthetic_periodic_upf(),
        nonlocal_coupling_matrix=(
            (1.20, -0.15, 0.10, 0.0),
            (-0.15, 0.85, 0.0, 0.0),
            (0.10, 0.0, 0.70, 0.0),
            (0.0, 0.0, 0.0, 0.45),
        ),
    )
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))

    with pytest.raises(ValueError, match="cannot couple different angular"):
        PeriodicUPFNonlocalOperator(
            pseudo,
            PlaneWaveBasis(grid, 2.0),
            ((1.0, 2.0, 3.0),),
        )


def test_periodic_upf_executes_through_scf_and_records_projector_work():
    pseudo = _synthetic_periodic_upf()
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        pseudo,
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    observer = RuntimeObserver(detail_events=False)

    result = run_periodic_scf(
        system,
        cutoff_hartree=2.0,
        kpoint_mesh=mesh,
        n_bands=2,
        config=PeriodicSCFConfig(
            max_iterations=2,
            min_iterations=2,
            density_tolerance=1.0e6,
            energy_tolerance=1.0e6,
            orbital_tolerance=1.0e6,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=12,
                tolerance=5.0e-3,
                max_subspace_size=12,
            ),
        ),
        observer=observer,
    )

    assert result.system_fingerprint == system.fingerprint
    assert result.converged
    assert np.isfinite(result.total_energy)
    work = observer.snapshot()["work_counters"]
    assert work["projector_elements_generated"] > 0
    assert work["projector_traffic_elements"] > 0
    forces = periodic_scf_forces(system, result)
    assert np.isfinite(np.asarray(forces.forces)).all()
    assert forces.provenance["pseudopotential_format"] == "upf"
    assert set(forces.to_dict()["force_by_term_hartree_per_bohr"]) == {
        "local_upf",
        "nonlocal_upf",
        "ion_ewald",
    }
    bands = run_periodic_band_structure(
        system,
        result,
        BandPath(
            [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
        ),
        n_bands=2,
        config=PeriodicDavidsonConfig(
            max_iterations=12,
            tolerance=5.0e-3,
            max_subspace_size=12,
        ),
    )
    assert np.isfinite(np.asarray(bands.eigenvalues)).all()


@pytest.mark.data
def test_qe_oncv_silicon_executes_periodic_upf_scf():
    source = Path(
        "vendors/quantum-espresso/QEHeat/examples/pseudo/Si_ONCV_PBE-1.1.upf"
    )
    assert sha256(source.read_bytes()).hexdigest() == (
        "d66fa1bb73367f9a2fc34286ecd0876fa1c8c79b25f3b12b30b5b8dc2164148c"
    )
    pseudo = read_upf(source)
    assert pseudo.periodic_upf_compatible
    system = PeriodicDFTSystem(
        (10.26, 10.26, 10.26),
        (16, 16, 16),
        ((0.0, 0.0, 0.0), (5.13, 5.13, 5.13)),
        pseudo,
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    observer = RuntimeObserver(detail_events=False)

    result = run_periodic_scf(
        system,
        cutoff_hartree=4.0,
        kpoint_mesh=mesh,
        n_bands=4,
        config=PeriodicSCFConfig(
            max_iterations=1,
            min_iterations=1,
            density_tolerance=1.0,
            energy_tolerance=1.0,
            orbital_tolerance=1.0,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=12,
                tolerance=5.0e-3,
                max_subspace_size=16,
            ),
        ),
        observer=observer,
    )

    assert result.system_fingerprint == system.fingerprint
    assert np.isfinite(result.total_energy)
    assert observer.snapshot()["work_counters"]["projector_elements_generated"] > 0


def test_periodic_scf_rejects_mixed_pseudopotential_formats_before_execution():
    upf = _synthetic_periodic_upf()
    gth, _local_upf = _matched_gth_and_numerical_upf()
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        pseudopotentials=(upf, gth),
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )

    with pytest.raises(ValueError, match="one pseudopotential format"):
        run_periodic_scf(
            system,
            cutoff_hartree=2.0,
            kpoint_mesh=mesh,
            n_bands=4,
            config=PeriodicSCFConfig(max_iterations=1),
        )


def test_periodic_upf_stress_fails_closed_before_electronic_work():
    pseudo = _synthetic_periodic_upf()
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        pseudo,
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )

    for evaluator in (periodic_analytic_stress, periodic_finite_difference_stress):
        with pytest.raises(ValueError, match="requires GTH input"):
            evaluator(
                system,
                cutoff_hartree=2.0,
                kpoint_mesh=mesh,
                n_bands=2,
            )


def test_periodic_upf_checkpoint_contract_is_content_bound():
    pseudo = _synthetic_periodic_upf()
    changed_coupling = list(list(row) for row in pseudo.nonlocal_coupling_matrix)
    changed_coupling[0][0] += 0.01
    systems = tuple(
        PeriodicDFTSystem(
            (6.0, 6.0, 6.0),
            (6, 6, 6),
            ((3.0, 3.0, 3.0),),
            candidate,
        )
        for candidate in (
            pseudo,
            replace(
                pseudo,
                nonlocal_coupling_matrix=tuple(
                    tuple(row) for row in changed_coupling
                ),
            ),
        )
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    contracts = tuple(
        periodic_scf_calculation_contract(
            system,
            cutoff_hartree=2.0,
            kpoint_mesh=mesh,
            n_bands=2,
        )
        for system in systems
    )
    species = contracts[0]["system"]["pseudopotentials"]["species"][0]

    assert species["format"] == "upf"
    assert len(species["content_sha256"]) == 64
    assert species["dij_dimension"] == 4
    assert contracts[0] != contracts[1]
