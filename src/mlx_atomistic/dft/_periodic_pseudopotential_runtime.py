"""Format dispatch for production periodic pseudopotential operators."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_atomistic.dft.periodic_gth import (
    PeriodicGTHNonlocalOperator,
    _ProjectorCache,
    gth_local_potential_grid,
    periodic_gth_local_forces,
)
from mlx_atomistic.dft.periodic_upf import (
    PeriodicUPFNonlocalOperator,
    periodic_upf_local_forces,
    upf_local_potential_grid,
)
from mlx_atomistic.dft.plane_wave import PlaneWaveBasis
from mlx_atomistic.dft.pseudopotentials import (
    PseudopotentialData,
    PseudopotentialFormat,
)

_PeriodicNonlocalOperator = PeriodicGTHNonlocalOperator | PeriodicUPFNonlocalOperator
_PERIODIC_NONLOCAL_TYPES = (
    PeriodicGTHNonlocalOperator,
    PeriodicUPFNonlocalOperator,
)


def _periodic_pseudopotential_format(
    pseudopotentials: Sequence[PseudopotentialData],
) -> PseudopotentialFormat:
    """Return the single executable format assigned to a periodic system."""

    formats = {pseudopotential.format for pseudopotential in pseudopotentials}
    if len(formats) != 1:
        raise ValueError(
            "periodic execution requires one pseudopotential format per system"
        )
    format_value = next(iter(formats))
    if format_value == PseudopotentialFormat.UPF and any(
        not pseudopotential.periodic_upf_compatible
        for pseudopotential in pseudopotentials
    ):
        raise ValueError(
            "periodic UPF execution requires scalar norm-conserving input "
            "without augmentation, SOC, or nonlinear core correction"
        )
    return format_value


def _periodic_nonlocal_operator(
    pseudopotentials: Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
    *,
    cache: _ProjectorCache,
) -> _PeriodicNonlocalOperator:
    """Build the format-specific compact nonlocal operator."""

    format_value = _periodic_pseudopotential_format(pseudopotentials)
    operator_type = (
        PeriodicGTHNonlocalOperator
        if format_value == PseudopotentialFormat.GTH
        else PeriodicUPFNonlocalOperator
    )
    return operator_type(pseudopotentials, basis, positions, cache=cache)


def _periodic_local_potential_grid(
    pseudopotentials: Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Build the format-specific periodic local potential."""

    format_value = _periodic_pseudopotential_format(pseudopotentials)
    if format_value == PseudopotentialFormat.GTH:
        return gth_local_potential_grid(pseudopotentials, basis, positions)
    return upf_local_potential_grid(pseudopotentials, basis, positions)


def _periodic_local_forces(
    density: mx.array,
    pseudopotentials: Sequence[PseudopotentialData],
    basis: PlaneWaveBasis,
    positions: Sequence[Sequence[float]],
) -> mx.array:
    """Evaluate the format-specific analytic local ionic forces."""

    format_value = _periodic_pseudopotential_format(pseudopotentials)
    if format_value == PseudopotentialFormat.GTH:
        return periodic_gth_local_forces(
            density,
            pseudopotentials,
            basis,
            positions,
        )
    return periodic_upf_local_forces(
        density,
        pseudopotentials,
        basis,
        positions,
    )
