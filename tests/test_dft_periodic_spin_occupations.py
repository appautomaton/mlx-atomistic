from __future__ import annotations

import numpy as np
import pytest

from mlx_atomistic.dft._periodic_spin_occupations import (
    _resolve_periodic_spin_occupations,
)


def _spectra():
    return (
        (
            np.asarray((-0.6, -0.2, 0.3, 0.8)),
            np.asarray((-0.5, -0.1, 0.4, 0.9)),
        ),
        (
            np.asarray((-0.4, 0.0, 0.5, 1.0)),
            np.asarray((-0.3, 0.1, 0.6, 1.1)),
        ),
    )


def test_fixed_spin_occupations_conserve_charge_and_magnetization():
    result = _resolve_periodic_spin_occupations(
        _spectra(),
        (0.5, 0.5),
        electron_count=6.0,
        smearing_width_hartree=None,
        magnetization=2.0,
    )

    assert result.electron_counts == (4.0, 2.0)
    assert result.occupations[0] == ((1.0, 1.0, 1.0, 1.0),) * 2
    assert result.occupations[1] == ((1.0, 1.0, 0.0, 0.0),) * 2
    assert result.electronic_entropy == 0.0


def test_unconstrained_spin_occupations_share_one_fermi_level():
    result = _resolve_periodic_spin_occupations(
        _spectra(),
        (0.5, 0.5),
        electron_count=4.0,
        smearing_width_hartree=0.02,
        magnetization=None,
    )

    assert sum(result.electron_counts) == pytest.approx(4.0, abs=1.0e-11)
    assert result.chemical_potentials[0] == result.chemical_potentials[1]
    assert result.shared_chemical_potential == result.chemical_potentials[0]
    assert result.electronic_entropy > 0.0


def test_smeared_fixed_magnetization_uses_two_channel_fermi_levels():
    result = _resolve_periodic_spin_occupations(
        _spectra(),
        (0.5, 0.5),
        electron_count=5.0,
        smearing_width_hartree=0.03,
        magnetization=1.0,
    )

    assert result.electron_counts[0] == pytest.approx(3.0, abs=1.0e-11)
    assert result.electron_counts[1] == pytest.approx(2.0, abs=1.0e-11)
    assert result.chemical_potentials[0] != result.chemical_potentials[1]
    assert result.shared_chemical_potential is None


def test_spin_occupation_modes_fail_closed():
    with pytest.raises(ValueError, match="require Fermi-Dirac"):
        _resolve_periodic_spin_occupations(
            _spectra(),
            (0.5, 0.5),
            electron_count=4.0,
            smearing_width_hartree=None,
            magnetization=None,
        )
    with pytest.raises(ValueError, match="integer channel"):
        _resolve_periodic_spin_occupations(
            _spectra(),
            (0.5, 0.5),
            electron_count=5.0,
            smearing_width_hartree=None,
            magnetization=0.5,
        )
