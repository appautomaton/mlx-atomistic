"""Initial configurations and velocities for reduced-unit MD."""

from __future__ import annotations

from math import ceil

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.md import instantaneous_temperature


def simple_cubic_lattice(n_particles: int, *, density: float = 0.8) -> tuple[mx.array, Cell]:
    """Return positions on a simple cubic lattice and a cubic cell."""

    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)
    if density <= 0.0:
        msg = "density must be positive"
        raise ValueError(msg)

    cells_per_axis = ceil(n_particles ** (1.0 / 3.0))
    length = (n_particles / density) ** (1.0 / 3.0)
    spacing = length / cells_per_axis

    positions = []
    for ix in range(cells_per_axis):
        for iy in range(cells_per_axis):
            for iz in range(cells_per_axis):
                positions.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing])
                if len(positions) == n_particles:
                    return as_mx_array(positions), Cell.cubic(length)

    return as_mx_array(positions), Cell.cubic(length)


def fcc_lattice(n_particles: int, *, density: float = 0.8) -> tuple[mx.array, Cell]:
    """Return positions on an FCC lattice and a cubic cell."""

    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)
    if density <= 0.0:
        msg = "density must be positive"
        raise ValueError(msg)

    cells_per_axis = ceil((n_particles / 4.0) ** (1.0 / 3.0))
    length = (n_particles / density) ** (1.0 / 3.0)
    lattice_constant = length / cells_per_axis
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    positions = []
    for ix in range(cells_per_axis):
        for iy in range(cells_per_axis):
            for iz in range(cells_per_axis):
                origin = np.array([ix, iy, iz], dtype=np.float32)
                for offset in basis:
                    positions.append(((origin + offset) * lattice_constant).tolist())
                    if len(positions) == n_particles:
                        return as_mx_array(positions), Cell.cubic(length)

    return as_mx_array(positions), Cell.cubic(length)


def random_velocities(
    n_particles: int,
    *,
    temperature: float = 1.0,
    masses=None,
    seed: int | None = None,
) -> mx.array:
    """Return Maxwell-like random velocities in reduced units."""

    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)
    if temperature < 0.0:
        msg = "temperature must be non-negative"
        raise ValueError(msg)

    if masses is None:
        masses_np = np.ones((n_particles,), dtype=np.float32)
    else:
        masses_np = np.asarray(masses, dtype=np.float32)
        if masses_np.shape != (n_particles,):
            msg = "masses must have shape (n_particles,)"
            raise ValueError(msg)

    rng = np.random.default_rng(seed)
    std = np.sqrt(temperature / masses_np)[:, None]
    velocities = rng.normal(0.0, std, size=(n_particles, 3)).astype(np.float32)
    return as_mx_array(velocities)


def remove_center_of_mass_velocity(velocities, masses=None) -> mx.array:
    """Remove center-of-mass velocity."""

    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * velocities.shape[0]) if masses is None else as_mx_array(masses)

    total_mass = mx.sum(masses)
    center_velocity = mx.sum(masses[:, None] * velocities, axis=0) / total_mass
    return velocities - center_velocity


def rescale_temperature(velocities, masses=None, *, temperature: float = 1.0) -> mx.array:
    """Rescale velocities to a target reduced temperature."""

    velocities = as_mx_array(velocities)
    masses = as_mx_array([1.0] * velocities.shape[0]) if masses is None else as_mx_array(masses)

    current = float(np.array(instantaneous_temperature(velocities, masses)))
    if current <= 0.0:
        msg = "cannot rescale velocities with zero instantaneous temperature"
        raise ValueError(msg)
    return velocities * (temperature / current) ** 0.5


def thermal_velocities(
    n_particles: int,
    *,
    temperature: float = 1.0,
    masses=None,
    seed: int | None = None,
) -> mx.array:
    """Return random velocities with zero COM velocity and target temperature."""

    velocities = random_velocities(n_particles, temperature=temperature, masses=masses, seed=seed)
    velocities = remove_center_of_mass_velocity(velocities, masses)
    return rescale_temperature(velocities, masses, temperature=temperature)
