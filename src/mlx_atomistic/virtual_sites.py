"""Virtual-site geometry definitions for molecular mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import as_mx_array

TIP4P_EW_OH_DISTANCE_ANGSTROM = 0.9572
TIP4P_EW_HOH_ANGLE_DEGREES = 104.52
TIP4P_EW_OM_DISTANCE_ANGSTROM = 0.1250


def tip4p_ew_m_site_weights(
    *,
    oh_distance: float = TIP4P_EW_OH_DISTANCE_ANGSTROM,
    hoh_angle_degrees: float = TIP4P_EW_HOH_ANGLE_DEGREES,
    om_distance: float = TIP4P_EW_OM_DISTANCE_ANGSTROM,
) -> tuple[float, float, float]:
    """Return OpenMM-compatible O/H/H weights for the TIP4P-Ew M site."""

    half_angle = np.deg2rad(float(hoh_angle_degrees)) / 2.0
    hydrogen_weight = float(om_distance) / (2.0 * float(oh_distance) * np.cos(half_angle))
    oxygen_weight = 1.0 - 2.0 * hydrogen_weight
    return float(oxygen_weight), float(hydrogen_weight), float(hydrogen_weight)


def tip4p_ew_virtual_site(
    oxygen: int,
    hydrogen1: int,
    hydrogen2: int,
    *,
    oh_distance: float = TIP4P_EW_OH_DISTANCE_ANGSTROM,
    hoh_angle_degrees: float = TIP4P_EW_HOH_ANGLE_DEGREES,
    om_distance: float = TIP4P_EW_OM_DISTANCE_ANGSTROM,
) -> ThreeParticleAverage:
    """Create the TIP4P-Ew charge-site geometry for one water molecule."""

    w_oxygen, w_h1, w_h2 = tip4p_ew_m_site_weights(
        oh_distance=oh_distance,
        hoh_angle_degrees=hoh_angle_degrees,
        om_distance=om_distance,
    )
    return ThreeParticleAverage(
        particle1=int(oxygen),
        particle2=int(hydrogen1),
        particle3=int(hydrogen2),
        weight1=w_oxygen,
        weight2=w_h1,
        weight3=w_h2,
    )


def tip4p_ew_reference_positions() -> np.ndarray:
    """Return O, H1, H2, M coordinates for an ideal TIP4P-Ew water in angstrom."""

    half_angle = np.deg2rad(TIP4P_EW_HOH_ANGLE_DEGREES) / 2.0
    oxygen = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    h1 = np.asarray(
        [
            TIP4P_EW_OH_DISTANCE_ANGSTROM * np.cos(half_angle),
            TIP4P_EW_OH_DISTANCE_ANGSTROM * np.sin(half_angle),
            0.0,
        ],
        dtype=np.float32,
    )
    h2 = np.asarray([h1[0], -h1[1], 0.0], dtype=np.float32)
    m_site = tip4p_ew_virtual_site(0, 1, 2)
    m = np.asarray(m_site.compute_position(mx.array(np.stack([oxygen, h1, h2]))), dtype=np.float32)
    return np.stack([oxygen, h1, h2, m]).astype(np.float32)


def _cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class TwoParticleAverage:
    """Virtual site at a weighted average of two parent atom positions."""

    particle1: int
    particle2: int
    weight1: float
    weight2: float

    def __post_init__(self) -> None:
        if self.particle1 < 0 or self.particle2 < 0:
            msg = "particle indices must be non-negative"
            raise ValueError(msg)
        if self.particle1 == self.particle2:
            msg = "particle1 and particle2 must be distinct"
            raise ValueError(msg)

    @property
    def parent_atoms(self) -> tuple[int, int]:
        """Indices of the two parent atoms."""

        return (self.particle1, self.particle2)

    def compute_position(self, positions: mx.array) -> mx.array:
        """Return the virtual-site position from the parent-atom positions.

        Args:
            positions: Cartesian positions of all atoms, shape ``(n_atoms, 3)``.

        Returns:
            The virtual-site position, shape ``(3,)``.
        """

        positions = as_mx_array(positions)
        return self.weight1 * positions[self.particle1] + self.weight2 * positions[self.particle2]


@dataclass(frozen=True)
class ThreeParticleAverage:
    """Virtual site at a weighted average of three parent atom positions."""

    particle1: int
    particle2: int
    particle3: int
    weight1: float
    weight2: float
    weight3: float

    def __post_init__(self) -> None:
        if self.particle1 < 0 or self.particle2 < 0 or self.particle3 < 0:
            msg = "particle indices must be non-negative"
            raise ValueError(msg)
        if len({self.particle1, self.particle2, self.particle3}) != 3:
            msg = "particle1, particle2, and particle3 must be distinct"
            raise ValueError(msg)

    @property
    def parent_atoms(self) -> tuple[int, int, int]:
        """Indices of the three parent atoms."""

        return (self.particle1, self.particle2, self.particle3)

    def compute_position(self, positions: mx.array) -> mx.array:
        """Return the virtual-site position from the parent-atom positions.

        Args:
            positions: Cartesian positions of all atoms, shape ``(n_atoms, 3)``.

        Returns:
            The virtual-site position, shape ``(3,)``.
        """

        positions = as_mx_array(positions)
        return (
            self.weight1 * positions[self.particle1]
            + self.weight2 * positions[self.particle2]
            + self.weight3 * positions[self.particle3]
        )


@dataclass(frozen=True)
class OutOfPlane:
    """Virtual site computed from three parent atoms using cross-product offset.

    Following the OpenMM convention, the position is computed as:
        p = p1*w1 + p2*w2 + p3*w3 + d*(cross(p2-p1, p3-p1) / |cross(p2-p1, p3-p1)|)
    where d is the out-of-plane distance.
    """

    particle1: int
    particle2: int
    particle3: int
    weight1: float
    weight2: float
    weight3: float
    distance: float

    def __post_init__(self) -> None:
        if self.particle1 < 0 or self.particle2 < 0 or self.particle3 < 0:
            msg = "particle indices must be non-negative"
            raise ValueError(msg)
        if len({self.particle1, self.particle2, self.particle3}) != 3:
            msg = "particle1, particle2, and particle3 must be distinct"
            raise ValueError(msg)

    @property
    def parent_atoms(self) -> tuple[int, int, int]:
        """Indices of the three parent atoms."""

        return (self.particle1, self.particle2, self.particle3)

    def compute_position(self, positions: mx.array) -> mx.array:
        """Return the virtual-site position from the parent-atom positions.

        Args:
            positions: Cartesian positions of all atoms, shape ``(n_atoms, 3)``.

        Returns:
            The virtual-site position, shape ``(3,)``.
        """

        positions = as_mx_array(positions)
        p1 = positions[self.particle1]
        p2 = positions[self.particle2]
        p3 = positions[self.particle3]
        v12 = p2 - p1
        v13 = p3 - p1
        cross = _cross(v12, v13)
        cross_norm = mx.sqrt(mx.maximum(mx.sum(cross * cross), 1e-12))
        cross_unit = cross / cross_norm
        return (
            self.weight1 * p1
            + self.weight2 * p2
            + self.weight3 * p3
            + self.distance * cross_unit
        )


@dataclass(frozen=True)
class LocalCoordinates:
    """Virtual site defined by local coordinate offsets from three parent atoms.

    Three orthonormal basis vectors are constructed from the parent atoms:
        u1 = (p2 - p1) / |p2 - p1|
        u3 = cross(u1, p3 - p1) / |cross(u1, p3 - p1)|
        u2 = cross(u3, u1)
    The virtual site position is:
        p = p1*w1 + p2*w2 + p3*w3 + x*u1 + y*u2 + z*u3
    """

    particle1: int
    particle2: int
    particle3: int
    weight1: float
    weight2: float
    weight3: float
    local_x: float
    local_y: float
    local_z: float

    def __post_init__(self) -> None:
        if self.particle1 < 0 or self.particle2 < 0 or self.particle3 < 0:
            msg = "particle indices must be non-negative"
            raise ValueError(msg)
        if len({self.particle1, self.particle2, self.particle3}) != 3:
            msg = "particle1, particle2, and particle3 must be distinct"
            raise ValueError(msg)

    @property
    def parent_atoms(self) -> tuple[int, int, int]:
        """Indices of the three parent atoms."""

        return (self.particle1, self.particle2, self.particle3)

    def compute_position(self, positions: mx.array) -> mx.array:
        """Return the virtual-site position from the parent-atom positions.

        Args:
            positions: Cartesian positions of all atoms, shape ``(n_atoms, 3)``.

        Returns:
            The virtual-site position, shape ``(3,)``.
        """

        positions = as_mx_array(positions)
        p1 = positions[self.particle1]
        p2 = positions[self.particle2]
        p3 = positions[self.particle3]
        v12 = p2 - p1
        v13 = p3 - p1
        v12_norm = mx.sqrt(mx.maximum(mx.sum(v12 * v12), 1e-12))
        u1 = v12 / v12_norm
        cross = _cross(u1, v13)
        cross_norm = mx.sqrt(mx.maximum(mx.sum(cross * cross), 1e-12))
        u3 = cross / cross_norm
        u2 = _cross(u3, u1)
        return (
            self.weight1 * p1
            + self.weight2 * p2
            + self.weight3 * p3
            + self.local_x * u1
            + self.local_y * u2
            + self.local_z * u3
        )


def compute_virtual_site_positions(
    virtual_sites: tuple, positions: mx.array
) -> mx.array:
    """Compute positions for a sequence of virtual sites given parent atom positions."""

    if not virtual_sites:
        return mx.array(np.empty((0, 3), dtype=np.float32), dtype=mx.float32)
    site_positions = [site.compute_position(positions) for site in virtual_sites]
    return mx.stack(site_positions)


def _redistribute_vs_forces_linear(
    parents: tuple[int, ...], weights: tuple[float, ...], vs_force: mx.array
) -> dict[int, mx.array]:
    """Redistribute force for linear-combination virtual sites (constant weights)."""

    contributions = {}
    for idx, weight in zip(parents, weights, strict=True):
        contributions[idx] = weight * vs_force
    return contributions


def _redistribute_vs_forces_vjp(
    vs: OutOfPlane | LocalCoordinates,
    positions: mx.array,
    vs_force: mx.array,
) -> dict[int, mx.array]:
    """Redistribute force via VJP for position-dependent virtual-site geometries."""

    def scalar_product(pos: mx.array) -> mx.array:
        vs_pos = vs.compute_position(pos)
        return mx.sum(vs_force * vs_pos)

    grad = mx.grad(scalar_product)(positions)
    contributions = {}
    for parent_idx in vs.parent_atoms:
        contributions[parent_idx] = grad[parent_idx]
    return contributions


def redistribute_virtual_site_force(
    vs: TwoParticleAverage | ThreeParticleAverage | OutOfPlane | LocalCoordinates,
    positions: mx.array,
    vs_force: mx.array,
) -> dict[int, mx.array]:
    """Distribute the force on a single virtual site to its parent atoms.

    For linear-combination types (TwoParticleAverage, ThreeParticleAverage) the
    redistribution uses the constant weights.  For position-dependent types
    (OutOfPlane, LocalCoordinates) a vector-Jacobian product (VJP) is computed
    via ``mx.grad``.
    """

    if isinstance(vs, TwoParticleAverage):
        return _redistribute_vs_forces_linear(
            (vs.particle1, vs.particle2),
            (vs.weight1, vs.weight2),
            vs_force,
        )
    if isinstance(vs, ThreeParticleAverage):
        return _redistribute_vs_forces_linear(
            (vs.particle1, vs.particle2, vs.particle3),
            (vs.weight1, vs.weight2, vs.weight3),
            vs_force,
        )
    if isinstance(vs, (OutOfPlane, LocalCoordinates)):
        return _redistribute_vs_forces_vjp(vs, positions, vs_force)
    msg = f"Unsupported virtual site type: {type(vs).__name__}"
    raise TypeError(msg)


@dataclass(frozen=True)
class VirtualSiteManager:
    """Manages virtual site position reconstruction and force redistribution.

    Real atom indices occupy positions ``0 .. n_real_atoms - 1``.  Virtual
    site indices follow at ``n_real_atoms .. n_real_atoms + n_virtual_sites - 1``
    in the same order as ``virtual_sites``.
    """

    virtual_sites: tuple
    n_real_atoms: int

    @property
    def n_virtual_sites(self) -> int:
        """Number of managed virtual sites."""

        return len(self.virtual_sites)

    @property
    def n_total_atoms(self) -> int:
        """Total atom count: real atoms plus virtual sites."""

        return self.n_real_atoms + len(self.virtual_sites)

    def extend_positions(self, positions: mx.array) -> mx.array:
        """Append virtual site positions to real-atom positions.

        Returns an array of shape ``(n_total_atoms, 3)`` whose first
        ``n_real_atoms`` rows equal *positions* and whose remaining rows are
        reconstructed from the parent atoms.
        """

        positions = as_mx_array(positions)
        if not self.virtual_sites:
            return positions
        vs_positions = compute_virtual_site_positions(self.virtual_sites, positions)
        return mx.concatenate([positions, vs_positions], axis=0)

    def reconstruct_positions(self, positions: mx.array) -> mx.array:
        """Recompute virtual site rows from current parent atom positions.

        *positions* must have shape ``(n_total_atoms, 3)``.  The first
        ``n_real_atoms`` rows are taken as-is; the remaining rows are
        overwritten with freshly reconstructed virtual site positions.
        """

        positions = as_mx_array(positions)
        if not self.virtual_sites:
            return positions
        vs_positions = compute_virtual_site_positions(
            self.virtual_sites, positions[: self.n_real_atoms]
        )
        return positions.at[self.n_real_atoms :].set(vs_positions)

    def redistribute_forces(
        self, forces: mx.array, positions: mx.array
    ) -> mx.array:
        """Redistribute virtual site forces to parent atoms.

        *forces* has shape ``(n_total_atoms, 3)`` and *positions* has shape
        ``(n_total_atoms, 3)``.  Returns a ``(n_real_atoms, 3)`` array where
        each virtual site's force has been distributed to its parents.
        """

        if not self.virtual_sites:
            return forces[: self.n_real_atoms]
        real_forces = forces[: self.n_real_atoms]
        for i, vs in enumerate(self.virtual_sites):
            vs_idx = self.n_real_atoms + i
            vs_force = forces[vs_idx]
            contributions = redistribute_virtual_site_force(vs, positions, vs_force)
            for parent_idx, parent_force in contributions.items():
                real_forces = real_forces.at[parent_idx].add(parent_force)
        return real_forces


__all__ = [
    "LocalCoordinates",
    "OutOfPlane",
    "TIP4P_EW_HOH_ANGLE_DEGREES",
    "TIP4P_EW_OH_DISTANCE_ANGSTROM",
    "TIP4P_EW_OM_DISTANCE_ANGSTROM",
    "ThreeParticleAverage",
    "TwoParticleAverage",
    "VirtualSiteManager",
    "compute_virtual_site_positions",
    "redistribute_virtual_site_force",
    "tip4p_ew_m_site_weights",
    "tip4p_ew_reference_positions",
    "tip4p_ew_virtual_site",
]
