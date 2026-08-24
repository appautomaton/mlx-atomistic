from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    MonkhorstPackGrid,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicEigenResult,
    PeriodicFermiDiracSmearing,
    PeriodicKPointResult,
    PeriodicSCFConfig,
    PlaneWaveBasis,
    PseudopotentialData,
    PseudopotentialFormat,
    RealSpaceGrid,
    periodic_scf_forces,
    run_periodic_scf,
)
from mlx_atomistic.dft._compact import _CompactBatch
from mlx_atomistic.dft._periodic_davidson import _initial_trial
from mlx_atomistic.dft._periodic_density import _density_from_kpoints
from mlx_atomistic.dft._periodic_occupations import _resolve_periodic_occupations


def _hydrogen_gth() -> PseudopotentialData:
    return PseudopotentialData(
        element="H",
        format=PseudopotentialFormat.GTH,
        valence_charge=1.0,
        gth_rloc=0.25,
        gth_coefficients=(-1.0,),
        gth_channels=(GTHProjectorChannel(0, 0.3, ((0.5,),)),),
    )


def _aluminum_gth() -> PseudopotentialData:
    """Return the Al GTH-PBE-q3 entry from CP2K GTH_POTENTIALS."""

    return PseudopotentialData(
        element="Al",
        format=PseudopotentialFormat.GTH,
        valence_charge=3.0,
        gth_rloc=0.45,
        gth_coefficients=(-7.55476126,),
        gth_channels=(
            GTHProjectorChannel(
                0,
                0.48743529,
                ((6.95993832, -1.88883584), (-1.88883584, 2.43847659)),
            ),
            GTHProjectorChannel(1, 0.56218949, ((1.86529857,),)),
        ),
    )


def test_weighted_fermi_dirac_solver_conserves_electrons_and_entropy():
    width = 0.1
    result = _resolve_periodic_occupations(
        ((-0.2, 0.2), (-0.2, 0.2)),
        (0.25, 0.75),
        electron_count=2.0,
        smearing_width_hartree=width,
    )
    probability = 1.0 / (np.exp(-2.0) + 1.0)
    expected_entropy = -4.0 * (
        probability * np.log(probability)
        + (1.0 - probability) * np.log1p(-probability)
    )

    assert result.chemical_potential == pytest.approx(0.0, abs=1e-13)
    assert result.electron_count == pytest.approx(2.0, abs=1e-12)
    assert result.electronic_entropy == pytest.approx(expected_entropy, abs=1e-12)
    for occupations in result.occupations:
        assert sum(occupations) == pytest.approx(2.0, abs=1e-12)
        assert occupations[0] == pytest.approx(2.0 * probability, abs=1e-12)


def test_periodic_occupation_models_fail_closed_at_capacity_boundaries():
    with pytest.raises(ValueError, match="two electrons per computed band"):
        _resolve_periodic_occupations(
            ((-0.2, 0.2),),
            (1.0,),
            electron_count=3.0,
            smearing_width_hartree=None,
        )
    with pytest.raises(ValueError, match="partially empty band"):
        _resolve_periodic_occupations(
            ((-0.2, 0.2),),
            (1.0,),
            electron_count=4.0,
            smearing_width_hartree=0.1,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        PeriodicFermiDiracSmearing(0.0)


def test_band_resolved_density_uses_each_resolved_occupation():
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    basis = PlaneWaveBasis(grid, 3.0)
    state = _initial_trial(basis, 2, None)
    eigen = PeriodicEigenResult._from_compact(
        eigenvalues=mx.array((-0.2, 0.3), dtype=mx.float32),
        compact_coefficients=state,
        basis=basis,
        residuals=mx.zeros((2,), dtype=mx.float32),
        orthonormality_error=0.0,
        iterations=1,
        converged=True,
        subspace_size=2,
        restart_count=0,
    )
    point = PeriodicKPointResult(
        reduced_kpoint=(0.0, 0.0, 0.0),
        weight=1.0,
        basis=basis,
        eigen=eigen,
        occupations=(0.25, 0.75),
    )
    observed = _density_from_kpoints((point,))
    orbitals = _CompactBatch.from_states((state,)).to_real()[0, :2]
    expected = mx.sum(
        mx.array((0.25, 0.75), dtype=mx.float32)[:, None, None, None]
        * mx.abs(orbitals) ** 2,
        axis=0,
    )

    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), atol=2e-7)
    assert float(mx.sum(observed) * grid.dv) == pytest.approx(1.0, abs=2e-6)


def test_periodic_scf_smearing_supports_odd_electron_count_and_forces():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        _hydrogen_gth(),
    )
    mesh = KPointMesh(
        (KPoint((0.0, 0.0, 0.0), weight=1.0, coordinate_system="reduced"),)
    )
    smearing = PeriodicFermiDiracSmearing(width_hartree=0.2)
    config = PeriodicSCFConfig(
        max_iterations=8,
        min_iterations=2,
        density_tolerance=0.25,
        energy_tolerance=0.2,
        orbital_tolerance=3e-3,
        mixing_beta=0.5,
        mixer="linear",
        smearing=smearing,
        davidson=PeriodicDavidsonConfig(
            max_iterations=20,
            tolerance=3e-3,
            max_subspace_size=12,
        ),
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=mesh,
        n_bands=2,
        config=config,
    )

    assert result.converged
    assert result.electron_count == pytest.approx(1.0, abs=2e-5)
    assert result.smearing_width_hartree == 0.2
    assert result.chemical_potential is not None
    assert result.electronic_entropy > 0.0
    assert result.internal_energy is not None
    assert result.total_energy < result.internal_energy
    assert result.energy_by_term["entropy_correction"] < 0.0
    occupations = result.kpoints[0].occupations
    assert occupations is not None
    assert sum(occupations) == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < occupations[1] < occupations[0] < 2.0
    force = periodic_scf_forces(system, result)
    assert np.all(np.isfinite(np.asarray(force.forces)))


@pytest.mark.slow
def test_fcc_aluminum_smoke_exercises_weighted_metallic_occupations():
    lattice_bohr = 7.65
    fractional_positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ),
        dtype=np.float64,
    )
    system = PeriodicDFTSystem(
        (lattice_bohr,) * 3,
        (8, 8, 8),
        fractional_positions * lattice_bohr,
        _aluminum_gth(),
    )
    config = PeriodicSCFConfig(
        max_iterations=6,
        min_iterations=2,
        density_tolerance=0.5,
        energy_tolerance=0.5,
        orbital_tolerance=2e-2,
        mixing_beta=0.5,
        mixer="linear",
        smearing=PeriodicFermiDiracSmearing(width_hartree=0.05),
        davidson=PeriodicDavidsonConfig(
            max_iterations=20,
            tolerance=2e-2,
            max_subspace_size=24,
        ),
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=3.0,
        kpoint_mesh=MonkhorstPackGrid((2, 2, 2)),
        n_bands=8,
        config=config,
    )

    assert result.converged
    assert result.electron_count == pytest.approx(12.0, abs=2e-4)
    assert result.chemical_potential is not None
    assert result.electronic_entropy > 0.0
    assert any(
        1e-3 < occupation < 1.999
        for point in result.kpoints
        for occupation in point.occupations or ()
    )
    assert all(
        np.isfinite(value)
        for value in (
            result.total_energy,
            result.internal_energy,
            result.chemical_potential,
        )
    )


def test_fixed_periodic_scf_still_rejects_odd_electron_count():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        _hydrogen_gth(),
    )
    mesh = KPointMesh(
        (KPoint((0.0, 0.0, 0.0), weight=1.0, coordinate_system="reduced"),)
    )

    with pytest.raises(ValueError, match="two electrons per computed band"):
        run_periodic_scf(
            system,
            cutoff_hartree=2.5,
            kpoint_mesh=mesh,
            n_bands=1,
        )
