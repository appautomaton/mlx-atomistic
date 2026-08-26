from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_atomistic.dft import (
    PeriodicDOSResult,
    PeriodicEigenResult,
    PeriodicKPointResult,
    PeriodicSCFResult,
    PeriodicSpinChannelResult,
    PlaneWaveBasis,
    RealSpaceGrid,
    periodic_density_of_states,
)
from mlx_atomistic.dft._periodic_davidson import _initial_trial


def _point(
    energies: tuple[float, ...],
    occupations: tuple[float, ...],
    *,
    lane: str,
) -> PeriodicKPointResult:
    grid = RealSpaceGrid((6, 6, 6), (6.0, 6.0, 6.0))
    basis = PlaneWaveBasis(grid, 3.0, lane_label=lane)
    state = _initial_trial(basis, len(energies), None)
    eigen = PeriodicEigenResult._from_compact(
        eigenvalues=mx.array(energies, dtype=mx.float32),
        compact_coefficients=state,
        basis=basis,
        residuals=mx.zeros((len(energies),), dtype=mx.float32),
        orthonormality_error=0.0,
        iterations=1,
        converged=True,
        subspace_size=len(energies),
        restart_count=0,
    )
    return PeriodicKPointResult(
        reduced_kpoint=(0.0, 0.0, 0.0),
        weight=1.0,
        basis=basis,
        eigen=eigen,
        occupations=occupations,
    )


def _result(
    *,
    electron_count: float,
    kpoints: tuple[PeriodicKPointResult, ...] = (),
    spin_channels: tuple[PeriodicSpinChannelResult, ...] = (),
    chemical_potential: float | None = None,
    converged: bool = True,
) -> PeriodicSCFResult:
    density = mx.full((6, 6, 6), electron_count / 216.0)
    return PeriodicSCFResult(
        converged=converged,
        status="converged" if converged else "max_iterations",
        iterations=2,
        total_energy=-1.0,
        electron_count=electron_count,
        density_residual=1.0e-7,
        energy_delta=1.0e-8,
        density=density,
        kpoints=kpoints,
        energy_by_term={"total": -1.0},
        history=(),
        timings={},
        chemical_potential=chemical_potential,
        spin_channels=spin_channels,
    )


def test_fixed_periodic_dos_integrates_states_and_electrons():
    source = _result(
        electron_count=4.0,
        kpoints=(_point((-0.5, 0.2), (2.0, 2.0), lane="fixed"),),
    )
    result = periodic_density_of_states(
        source,
        broadening_hartree=0.02,
        energy_points=4001,
    )

    assert isinstance(result, PeriodicDOSResult)
    assert result.fermi_level_convention == "highest_occupied_computed_state"
    assert result.fermi_level_hartree == pytest.approx(0.2, abs=1.0e-7)
    assert result.expected_state_count == pytest.approx(4.0)
    assert result.integrated_state_count == pytest.approx(4.0, abs=2.0e-4)
    assert result.expected_electron_count == pytest.approx(4.0)
    assert result.integrated_electron_count == pytest.approx(4.0, abs=2.0e-4)
    assert result.to_dict()["energy_point_count"] == 4001


def test_spin_periodic_dos_has_shared_grid_and_channel_counts():
    up_point = _point((-0.4, 0.3), (1.0, 0.75), lane="spin-up")
    down_point = _point((-0.3, 0.4), (0.25, 0.0), lane="spin-down")
    up_density = mx.full((6, 6, 6), 1.75 / 216.0)
    down_density = mx.full((6, 6, 6), 0.25 / 216.0)
    source = _result(
        electron_count=2.0,
        chemical_potential=0.05,
        spin_channels=(
            PeriodicSpinChannelResult(
                "up",
                1.75,
                up_density,
                (up_point,),
                0.05,
            ),
            PeriodicSpinChannelResult(
                "down",
                0.25,
                down_density,
                (down_point,),
                0.05,
            ),
        ),
    )
    result = periodic_density_of_states(source, energy_points=3001)

    assert result.fermi_level_convention == "shared_fermi_dirac_chemical_potential"
    assert result.fermi_level_hartree == 0.05
    assert [channel.label for channel in result.channels] == ["up", "down"]
    assert result.expected_state_count == pytest.approx(4.0)
    assert result.integrated_state_count == pytest.approx(4.0, abs=2.0e-4)
    assert result.expected_electron_count == pytest.approx(2.0)
    assert result.integrated_electron_count == pytest.approx(2.0, abs=2.0e-4)
    assert result.channels[0].integrated_electron_count == pytest.approx(1.75, abs=2.0e-4)
    assert result.channels[1].integrated_electron_count == pytest.approx(0.25, abs=2.0e-4)


def test_periodic_dos_fails_closed_on_invalid_source_and_controls():
    point = _point((-0.5,), (2.0,), lane="invalid")
    unconverged = _result(electron_count=2.0, kpoints=(point,), converged=False)

    with pytest.raises(ValueError, match="converged"):
        periodic_density_of_states(unconverged)
    with pytest.raises(ValueError, match="positive"):
        periodic_density_of_states(
            _result(electron_count=2.0, kpoints=(point,)),
            broadening_hartree=0.0,
        )
    with pytest.raises(TypeError, match="PeriodicSCFResult"):
        periodic_density_of_states(object())
