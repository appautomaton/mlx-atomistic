from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.dft import (
    GTHProjectorChannel,
    KPoint,
    KPointMesh,
    PeriodicCollinearSpinConfig,
    PeriodicDavidsonConfig,
    PeriodicDFTSystem,
    PeriodicFermiDiracSmearing,
    PeriodicSCFConfig,
    PseudopotentialData,
    PseudopotentialFormat,
    run_periodic_scf,
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


def _gamma_mesh() -> KPointMesh:
    return KPointMesh(
        (KPoint((0.0, 0.0, 0.0), weight=1.0, coordinate_system="reduced"),)
    )


def _bounded_config(
    *,
    spin: PeriodicCollinearSpinConfig | None = None,
    smearing: PeriodicFermiDiracSmearing | None = None,
) -> PeriodicSCFConfig:
    return PeriodicSCFConfig(
        max_iterations=8,
        min_iterations=2,
        density_tolerance=0.3,
        energy_tolerance=0.2,
        orbital_tolerance=5.0e-3,
        mixing_beta=0.5,
        mixer="linear",
        spin=spin,
        smearing=smearing,
        davidson=PeriodicDavidsonConfig(
            max_iterations=20,
            tolerance=5.0e-3,
            max_subspace_size=12,
        ),
    )


def test_periodic_spin_config_fails_closed_on_ambiguous_modes():
    with pytest.raises(ValueError, match="does not accept"):
        PeriodicCollinearSpinConfig(mode="unconstrained", magnetization=0.0)
    with pytest.raises(ValueError, match="requires Fermi-Dirac"):
        PeriodicSCFConfig(
            spin=PeriodicCollinearSpinConfig(
                mode="unconstrained",
                magnetization=None,
            )
        )


def test_zero_magnetization_spin_scf_reproduces_unpolarized_limit():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((2.0, 3.0, 3.0), (4.0, 3.0, 3.0)),
        _hydrogen_gth(),
    )
    unpolarized = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=_gamma_mesh(),
        n_bands=1,
        config=_bounded_config(),
    )
    polarized = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=_gamma_mesh(),
        n_bands=1,
        config=_bounded_config(spin=PeriodicCollinearSpinConfig()),
    )

    assert unpolarized.converged
    assert polarized.converged
    assert polarized.kpoints == ()
    assert len(polarized.spin_channels) == 2
    assert polarized.integrated_magnetization == pytest.approx(0.0, abs=2.0e-5)
    assert polarized.total_energy == pytest.approx(unpolarized.total_energy, abs=3.0e-4)
    np.testing.assert_allclose(
        np.asarray(polarized.density),
        np.asarray(unpolarized.density),
        atol=3.0e-4,
    )
    assert polarized.to_dict()["spin_channels"][0]["label"] == "up"


def test_fixed_spin_scf_preserves_charge_and_requested_magnetization():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        _hydrogen_gth(),
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=_gamma_mesh(),
        n_bands=1,
        config=_bounded_config(
            spin=PeriodicCollinearSpinConfig(
                mode="fixed_magnetization",
                magnetization=1.0,
            )
        ),
    )

    assert result.converged
    assert result.electron_count == pytest.approx(1.0, abs=2.0e-5)
    assert result.integrated_magnetization == pytest.approx(1.0, abs=2.0e-5)
    assert result.spin_channels[0].electron_count == pytest.approx(1.0, abs=2.0e-5)
    assert result.spin_channels[1].electron_count == pytest.approx(0.0, abs=2.0e-5)
    assert all(
        row["integrated_magnetization"] == pytest.approx(1.0, abs=1.0e-12)
        for row in result.history
    )


def test_unconstrained_spin_scf_uses_one_shared_fermi_level():
    system = PeriodicDFTSystem(
        (6.0, 6.0, 6.0),
        (6, 6, 6),
        ((3.0, 3.0, 3.0),),
        _hydrogen_gth(),
    )
    config = _bounded_config(
        spin=PeriodicCollinearSpinConfig(
            mode="unconstrained",
            magnetization=None,
            initial_magnetization=0.5,
        ),
        smearing=PeriodicFermiDiracSmearing(width_hartree=0.2),
    )
    result = run_periodic_scf(
        system,
        cutoff_hartree=2.5,
        kpoint_mesh=_gamma_mesh(),
        n_bands=2,
        config=config,
    )

    assert result.converged
    assert result.electron_count == pytest.approx(1.0, abs=2.0e-5)
    assert result.chemical_potential is not None
    assert result.spin_channels[0].chemical_potential == pytest.approx(
        result.chemical_potential,
        abs=1.0e-12,
    )
    assert result.spin_channels[1].chemical_potential == pytest.approx(
        result.chemical_potential,
        abs=1.0e-12,
    )
