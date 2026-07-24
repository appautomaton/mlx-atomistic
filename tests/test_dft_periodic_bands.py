from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    BandPath,
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicFrozenDensity,
    PeriodicSCFConfig,
    PseudopotentialData,
    PseudopotentialFormat,
    fold_band_path_to_supercell,
    run_periodic_band_structure,
    run_periodic_scf,
    unfold_periodic_band_structure,
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


@pytest.fixture(scope="module")
def converged_gamma_scf():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        _hydrogen_gth(),
    )
    mesh = KPointMesh(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=1,
        config=PeriodicSCFConfig(
            max_iterations=12,
            min_iterations=2,
            density_tolerance=2e-3,
            energy_tolerance=2e-3,
            orbital_tolerance=5e-4,
            mixing_beta=0.5,
            mixer="linear",
            davidson=PeriodicDavidsonConfig(
                max_iterations=24,
                tolerance=5e-4,
                max_subspace_size=16,
            ),
        ),
    )
    assert result.converged
    return system, result


def _solver_config() -> PeriodicDavidsonConfig:
    return PeriodicDavidsonConfig(
        max_iterations=30,
        tolerance=8e-4,
        max_subspace_size=20,
    )


def test_periodic_gamma_band_matches_converged_scf_eigenvalue(converged_gamma_scf):
    system, scf_result = converged_gamma_scf
    path = BandPath(
        [KPoint((0.0, 0.0, 0.0), label="Γ", coordinate_system="reduced")]
    )

    bands = run_periodic_band_structure(
        system,
        scf_result,
        path,
        n_bands=2,
        config=_solver_config(),
    )

    assert bands.eigenvalues.shape == (1, 2)
    assert bands.residuals.shape == (1, 2)
    assert bands.occupied_band_count == 1
    assert bands.density_source == "scf_result"
    assert bands.points[0].eigen.converged
    assert float(bands.eigenvalues[0, 0]) == pytest.approx(
        float(scf_result.kpoints[0].eigen.eigenvalues[0]),
        abs=5e-3,
    )
    assert bands.to_dict()["self_consistency_iterations"] == 0


def test_periodic_short_path_matches_portable_frozen_density(converged_gamma_scf):
    system, scf_result = converged_gamma_scf
    path = BandPath.line(
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        count=3,
        start_label="Γ",
        end_label="Q",
        coordinate_system="reduced",
    )
    frozen = PeriodicFrozenDensity(
        density=scf_result.density,
        cutoff_hartree=2.5,
        electron_count=2.0,
    )

    from_result = run_periodic_band_structure(
        system,
        scf_result,
        path,
        n_bands=2,
        config=_solver_config(),
    )
    from_density = run_periodic_band_structure(
        system,
        frozen,
        path,
        n_bands=2,
        config=_solver_config(),
    )

    np.testing.assert_allclose(
        np.asarray(from_density.eigenvalues),
        np.asarray(from_result.eigenvalues),
        atol=2e-5,
    )
    assert from_density.density_source == "frozen_density"
    assert all(point.basis.active_count > 2 for point in from_density.points)
    assert all(point.eigen.converged for point in from_density.points)


def test_periodic_guard_bands_are_solved_but_not_published(converged_gamma_scf):
    system, scf_result = converged_gamma_scf
    path = BandPath(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )

    bands = run_periodic_band_structure(
        system,
        scf_result,
        path,
        n_bands=2,
        guard_bands=1,
        config=_solver_config(),
    )

    assert bands.eigenvalues.shape == (1, 2)
    assert bands.points[0].eigen.eigenvalues.shape == (2,)
    assert bands.guard_band_count == 1
    assert bands.to_dict()["guard_band_count"] == 1


def test_periodic_band_source_and_path_validation(converged_gamma_scf):
    system, scf_result = converged_gamma_scf
    reduced = BandPath(
        [KPoint((0.0, 0.0, 0.0), coordinate_system="reduced")]
    )
    cartesian = BandPath([KPoint.gamma()])

    with pytest.raises(ValueError, match="converged SCF"):
        run_periodic_band_structure(system, replace(scf_result, converged=False), reduced)
    with pytest.raises(ValueError, match="reduced-coordinate"):
        run_periodic_band_structure(system, scf_result, cartesian)
    with pytest.raises(ValueError, match="occupied band count"):
        run_periodic_band_structure(system, scf_result, reduced, n_bands=0)
    with pytest.raises(ValueError, match="guard_bands"):
        run_periodic_band_structure(system, scf_result, reduced, guard_bands=-1)
    mismatched_cell = PeriodicDFTSystem(
        (7.0, 7.0, 7.0),
        system.grid.shape,
        system.positions,
        system.pseudopotential,
    )
    with pytest.raises(ValueError, match="basis grid and cell"):
        run_periodic_band_structure(mismatched_cell, scf_result, reduced)
    with pytest.raises(ValueError, match="shape"):
        run_periodic_band_structure(
            system,
            PeriodicFrozenDensity(mx.ones((2, 2, 2)), 2.5, 2.0),
            reduced,
        )
    with pytest.raises(ValueError, match="electron count"):
        run_periodic_band_structure(
            system,
            PeriodicFrozenDensity(scf_result.density, 2.5, 4.0),
            reduced,
        )


def test_fcc_primitive_path_folds_into_conventional_silicon_cell():
    lattice = 10.0
    primitive = 0.5 * lattice * np.asarray(
        ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
    )
    conventional = lattice * np.eye(3)
    path = BandPath(
        [
            KPoint((0.0, 0.0, 0.0), label="Γ", coordinate_system="reduced"),
            KPoint((0.5, 0.0, 0.5), label="X", coordinate_system="reduced"),
            KPoint((0.5, 0.25, 0.75), label="W", coordinate_system="reduced"),
        ]
    )

    folded = fold_band_path_to_supercell(primitive, conventional, path)

    assert folded.volume_ratio == 4
    np.testing.assert_array_equal(
        folded.supercell_transform,
        ((-1, 1, 1), (1, -1, 1), (1, 1, -1)),
    )
    np.testing.assert_allclose(
        [point.vector for point in folded.supercell_path.points],
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        atol=1e-12,
    )
    assert np.all(np.diff(folded.path_distances) > 0.0)


def test_unfolding_identity_cell_has_unit_spectral_weight(converged_gamma_scf):
    system, scf_result = converged_gamma_scf
    path = BandPath.line(
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        count=2,
        coordinate_system="reduced",
    )
    cell = np.diag(np.asarray(system.grid.lengths, dtype=np.float64))
    folded = fold_band_path_to_supercell(cell, cell, path)
    bands = run_periodic_band_structure(
        system,
        scf_result,
        folded.supercell_path,
        n_bands=2,
        config=_solver_config(),
    )

    unfolded = unfold_periodic_band_structure(bands, folded)

    np.testing.assert_allclose(
        np.asarray(unfolded.spectral_weights),
        1.0,
        atol=2e-5,
    )
    assert unfolded.primitive_occupied_band_count == 1
