"""Thermodynamic observables shared by molecular-dynamics integrators."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_atomistic.constraints import DistanceConstraints
from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.nonbonded import normalize_molecule_ids


def kinetic_energy(
    velocities: mx.array,
    masses: mx.array,
    *,
    kinetic_energy_scale: float = 1.0,
) -> mx.array:
    """Return the total kinetic energy in the configured unit system.

    Args:
        velocities: Per-particle velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``.
        kinetic_energy_scale: Multiplicative factor converting the raw kinetic
            quantity into the configured energy unit. Defaults to ``1.0``.

    Returns:
        Scalar kinetic energy ``½ · scale · Σ_i m_i |v_i|²``.
    """

    velocities = as_mx_array(velocities)
    masses = as_mx_array(masses)
    return kinetic_energy_scale * 0.5 * mx.sum(masses[:, None] * velocities * velocities)


def _remove_center_of_mass_velocity(
    velocities: mx.array,
    masses: mx.array,
    *,
    total_mass: mx.array | None = None,
) -> mx.array:
    if total_mass is None:
        total_mass = mx.sum(masses)
    center_velocity = mx.sum(masses[:, None] * velocities, axis=0) / total_mass
    return velocities - center_velocity


def instantaneous_temperature(
    velocities: mx.array,
    masses: mx.array,
    *,
    dof: int | None = None,
    kinetic_energy_scale: float = 1.0,
    boltzmann_constant: float = 1.0,
) -> mx.array:
    """Return the instantaneous temperature from the kinetic energy.

    Args:
        velocities: Per-particle velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``.
        dof: Degrees of freedom in the equipartition denominator; ``None``
            uses ``velocities.size`` (no constraints removed). Defaults to ``None``.
        kinetic_energy_scale: Energy-unit factor forwarded to
            `kinetic_energy`. Defaults to ``1.0``.
        boltzmann_constant: Boltzmann constant in the configured units.
            Defaults to ``1.0`` (reduced units).

    Returns:
        Scalar temperature ``2·E_kin / (dof · k_B)``.
    """

    if dof is None:
        dof = velocities.size
    return (
        2.0
        * kinetic_energy(
            velocities,
            masses,
            kinetic_energy_scale=kinetic_energy_scale,
        )
        / (dof * boltzmann_constant)
    )


def virial_tensor(positions: mx.array, forces: mx.array) -> mx.array:
    """Return the non-periodic configurational virial tensor.

    Args:
        positions: Particle coordinates, shape ``(n_particles, 3)``.
        forces: Forces on each particle, shape ``(n_particles, 3)``.

    Returns:
        The ``(3, 3)`` virial tensor ``positionsᵀ · forces``.

    Raises:
        ValueError: If ``positions`` and ``forces`` do not both have
            shape ``(n_particles, 3)``.
    """

    positions = as_mx_array(positions)
    forces = as_mx_array(forces)
    if positions.shape != forces.shape or positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions and forces must both have shape (n_particles, 3)"
        raise ValueError(msg)
    return mx.transpose(positions) @ forces


def kinetic_pressure_tensor(
    velocities: mx.array,
    masses: mx.array,
    *,
    kinetic_energy_scale: float = 1.0,
    molecule_ids: object | None = None,
) -> mx.array:
    """Return the kinetic momentum-flux tensor in the configured units.

    Args:
        velocities: Per-particle velocities, shape ``(n_particles, 3)``.
        masses: Per-particle masses, shape ``(n_particles,)``.
        kinetic_energy_scale: Multiplicative factor converting the raw kinetic
            quantity into the configured energy unit. Defaults to ``1.0``.
        molecule_ids: Optional contiguous per-particle molecule identifiers.
            When present, the tensor uses molecular center-of-mass momenta.

    Returns:
        The ``(3, 3)`` tensor ``scale · Σ_i m_i v_i ⊗ v_i``.

    Raises:
        ValueError: If ``velocities`` is not ``(n_particles, 3)`` or
            ``masses`` is not ``(n_particles,)``.
    """

    velocities = as_mx_array(velocities)
    masses = as_mx_array(masses)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        msg = "velocities must have shape (n_particles, 3)"
        raise ValueError(msg)
    if masses.shape != (velocities.shape[0],):
        msg = "masses must have shape (n_particles,)"
        raise ValueError(msg)
    if molecule_ids is not None:
        ids = normalize_molecule_ids(
            molecule_ids,
            particle_count=velocities.shape[0],
        )
        molecule_count = int(np.max(ids)) + 1
        indices = mx.array(ids, dtype=mx.int32)
        molecule_masses = mx.zeros((molecule_count,), dtype=masses.dtype).at[indices].add(masses)
        momenta = masses[:, None] * velocities
        molecule_momenta = (
            mx.zeros((molecule_count, 3), dtype=velocities.dtype).at[indices].add(momenta)
        )
        molecule_velocities = molecule_momenta / molecule_masses[:, None]
        weighted_velocities = molecule_masses[:, None] * molecule_velocities
        return kinetic_energy_scale * mx.transpose(molecule_velocities) @ weighted_velocities
    weighted_velocities = masses[:, None] * velocities
    return kinetic_energy_scale * mx.transpose(velocities) @ weighted_velocities


def _pressure_diagnostics_from_virial(
    virial: mx.array | None,
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    *,
    cell: Cell | None,
    kinetic_energy_scale: float,
    enabled: bool,
    molecule_ids: object | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Assemble pressure from the virial owned by one diagnostic evaluation."""

    if not enabled:
        zeros = mx.zeros((3, 3), dtype=positions.dtype)
        return zeros, zeros, mx.sum(positions[:, 0] * 0.0)
    if virial is None:
        msg = "enabled pressure diagnostics require a configurational virial"
        raise RuntimeError(msg)
    if cell is None:
        zeros = mx.zeros((3, 3), dtype=virial.dtype)
        return virial, zeros, mx.sum(virial * 0.0)
    volume = cell.volume
    if float(np.asarray(volume)) <= 0.0:
        msg = "pressure diagnostics require positive cell volume"
        raise ValueError(msg)
    kinetic_tensor = kinetic_pressure_tensor(
        velocities,
        masses,
        kinetic_energy_scale=kinetic_energy_scale,
        molecule_ids=molecule_ids,
    )
    tensor = (kinetic_tensor + virial) / volume
    return virial, tensor, mx.trace(tensor) / 3.0


def _temperature_degrees_of_freedom(
    positions: mx.array,
    constraints: DistanceConstraints | None,
) -> int:
    dof = int(positions.size)
    if constraints is not None:
        dof -= int(constraints.pairs.shape[0])
    if positions.shape[0] > 1:
        dof -= 3
    return max(1, dof)
