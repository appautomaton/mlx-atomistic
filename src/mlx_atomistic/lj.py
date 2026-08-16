"""Lennard-Jones force terms for reduced-unit simulations."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.metal_kernels import fused_lj_forces
from mlx_atomistic.neighbors import NeighborBlocks
from mlx_atomistic.nonbonded import (
    DEFAULT_DENSE_MEMORY_BUDGET_BYTES,
    NonbondedBackend,
    NonbondedExecutionConfig,
    choose_nonbonded_backend,
    dense_lj_energy_forces,
    estimate_dense_nonbonded_bytes,
)
from mlx_atomistic.topology import Topology, _isin_sorted_codes


@dataclass(frozen=True)
class LennardJonesPotential:
    """Naive all-pairs Lennard-Jones potential in reduced units."""

    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float | None = 2.5
    shift: bool = True
    topology: Topology | None = None
    one_four_scale: float = 1.0
    backend: NonbondedBackend = "auto"
    tile_size: int = 512
    memory_budget_bytes: int | None = DEFAULT_DENSE_MEMORY_BUDGET_BYTES
    name: str = "lj"
    supports_virial: bool = True
    analytic_virial_supported: bool = False
    use_fused_kernel: bool = False

    def __post_init__(self) -> None:
        if self.cutoff is not None and self.cutoff <= 0.0:
            msg = "cutoff must be positive"
            raise ValueError(msg)
        if self.one_four_scale < 0.0:
            msg = "one_four_scale must be non-negative"
            raise ValueError(msg)
        config = NonbondedExecutionConfig(
            backend=self.backend,
            tile_size=self.tile_size,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        object.__setattr__(self, "tile_size", config.tile_size)
        object.__setattr__(self, "memory_budget_bytes", config.memory_budget_bytes)
        object.__setattr__(self, "_pair_scale_cache", None)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: object | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return potential energy and forces for positions with shape ``(n_particles, 3)``.

        Args:
            positions: Particle coordinates, shape ``(n_particles, 3)``.
            cell: Optional periodic cell for minimum-image distances. Defaults to ``None``.
            pairs: Optional neighbor list, neighbor blocks, or dense pair array; the
                nonbonded backend is chosen automatically. Defaults to ``None``.

        Returns:
            An ``(energy, forces)`` tuple: scalar LJ energy and forces of shape
                ``(n_particles, 3)``.

        Raises:
            ValueError: If ``positions`` is not ``(n_particles, 3)`` or a lazy
                topology is used without a runtime pair provider.
        """

        positions = as_mx_array(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "positions must have shape (n_particles, 3)"
            raise ValueError(msg)
        if (
            self.topology is not None
            and pairs is None
            and self.topology.nonbonded_pair_policy == "lazy"
        ):
            msg = (
                "lazy topology requires a runtime nonbonded pair provider; "
                "full dense pair materialization was not requested"
            )
            raise ValueError(msg)

        estimated_bytes = estimate_dense_nonbonded_bytes(positions.shape[0], components="lj")
        concrete_backend = choose_nonbonded_backend(
            requested=self.backend,
            n_atoms=positions.shape[0],
            pairs_provided=pairs is not None,
            estimated_dense_bytes=estimated_bytes,
            memory_budget_bytes=self.memory_budget_bytes,
        )
        if concrete_backend in {"mlx_dense", "mlx_tiled"}:
            return dense_lj_energy_forces(
                positions,
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                shift=self.shift,
                cell=cell,
                topology=self.topology,
                one_four_scale=self.one_four_scale,
                tile_size=self.tile_size if concrete_backend == "mlx_tiled" else None,
            )

        if isinstance(pairs, NeighborBlocks):
            return self._block_energy_forces(positions, pairs, cell)
        if self.topology is not None:
            filtered_pairs, scales = self._topology_pairs_and_scales(pairs)
            return self._pair_energy_forces(positions, filtered_pairs, cell, scales=scales)
        if (
            self.use_fused_kernel
            and pairs is not None
            and isinstance(pairs, mx.array)
            and cell is not None
            and cell.is_orthorhombic
            and self.cutoff is not None
        ):
            return fused_lj_forces(
                positions,
                pairs,
                mx.diag(cell.matrix),
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
                shift=self.shift,
            )
        if pairs is not None:
            return self._pair_energy_forces(positions, pairs, cell)

        displacement = positions[:, None, :] - positions[None, :, :]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = r2 > 0.0
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)

        safe_r2 = mx.where(pair_mask, r2, 1.0)
        sigma2_over_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6

        pair_energy = 4.0 * self.epsilon * (inv_r12 - inv_r6)
        if self.shift and self.cutoff is not None:
            sigma2_over_rc2 = (self.sigma * self.sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * self.epsilon * (inv_rc12 - inv_rc6)
        pair_energy = mx.where(pair_mask, pair_energy, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar, 0.0)
        forces = mx.sum(scalar[:, :, None] * displacement, axis=1)

        energy = 0.5 * mx.sum(pair_energy)
        return energy, forces

    def _topology_pairs_and_scales(self, pairs) -> tuple[mx.array, mx.array]:
        topology = self.topology
        if topology is None:
            msg = "topology is required"
            raise ValueError(msg)
        if pairs is None and topology.nonbonded_pair_policy == "lazy":
            msg = (
                "lazy topology requires a runtime nonbonded pair provider; "
                "full dense pair materialization was not requested"
            )
            raise ValueError(msg)
        if pairs is not None:
            cache_key = (id(pairs), self.one_four_scale)
            cache = self._pair_scale_cache
            if cache is not None and cache[0] == cache_key:
                return cache[1]
        filtered_pairs = topology.nonbonded_pairs(pairs)
        if float(self.one_four_scale) == 1.0:
            scales = mx.array(1.0, dtype=mx.float32)
        else:
            scales = topology.pair_scales(
                filtered_pairs,
                one_four_scale=self.one_four_scale,
            )
        if pairs is not None:
            object.__setattr__(self, "_pair_scale_cache", (cache_key, (filtered_pairs, scales)))
        return filtered_pairs, scales

    def _pair_energy_forces(
        self,
        positions: mx.array,
        pairs: mx.array,
        cell: Cell | None,
        *,
        scales: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        pairs = as_mx_array(pairs, dtype=mx.int32)
        forces = mx.zeros_like(positions)
        if pairs.shape[0] == 0:
            return mx.sum(positions[:, 0] * 0.0), forces
        if scales is None:
            scales = mx.array(1.0, dtype=mx.float32)

        i = pairs[:, 0]
        j = pairs[:, 1]
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = r2 > 0.0
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)

        safe_r2 = mx.where(pair_mask, r2, 1.0)
        sigma2_over_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6

        pair_energy = 4.0 * self.epsilon * (inv_r12 - inv_r6)
        if self.shift and self.cutoff is not None:
            sigma2_over_rc2 = (self.sigma * self.sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * self.epsilon * (inv_rc12 - inv_rc6)
        pair_energy = mx.where(pair_mask, pair_energy * scales, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar * scales, 0.0)
        pair_forces = scalar[:, None] * displacement
        forces = forces.at[i].add(pair_forces).at[j].add(-pair_forces)

        return mx.sum(pair_energy), forces

    def _block_mask_and_scales(self, blocks: NeighborBlocks) -> tuple[mx.array, mx.array]:
        if self.topology is None:
            return blocks.valid_mask, mx.array(1.0, dtype=mx.float32)

        cache_key = ("blocks", id(blocks), self.one_four_scale)
        cache = self._pair_scale_cache
        if cache is not None and cache[0] == cache_key:
            return cache[1]

        left = np.asarray(blocks.left, dtype=np.int32).reshape(-1)
        right = np.asarray(blocks.right, dtype=np.int32).reshape(-1)
        valid = np.asarray(blocks.valid_mask, dtype=bool).reshape(-1)
        n_atoms = self.topology.n_atoms
        if np.any(left[valid] < 0) or np.any(right[valid] < 0):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)
        if np.any(left[valid] >= n_atoms) or np.any(right[valid] >= n_atoms):
            msg = "pairs contain atom indices outside [0, n_atoms)"
            raise ValueError(msg)

        normalized_left = np.minimum(left, right).astype(np.int64, copy=False)
        normalized_right = np.maximum(left, right).astype(np.int64, copy=False)
        codes = normalized_left * np.int64(n_atoms) + normalized_right
        keep = valid & ~_isin_sorted_codes(codes, self.topology._exclusion_codes)
        mask = mx.array(keep.reshape(blocks.left.shape))
        if float(self.one_four_scale) == 1.0 or self.topology._one_four_codes.size == 0:
            scales = mx.array(1.0, dtype=mx.float32)
        else:
            one_four = _isin_sorted_codes(codes, self.topology._one_four_codes)
            scales_np = np.where(one_four, float(self.one_four_scale), 1.0).astype(np.float32)
            scales = mx.array(scales_np.reshape(blocks.left.shape), dtype=mx.float32)
        object.__setattr__(self, "_pair_scale_cache", (cache_key, (mask, scales)))
        return mask, scales

    def _block_energy_forces(
        self,
        positions: mx.array,
        blocks: NeighborBlocks,
        cell: Cell | None,
    ) -> tuple[mx.array, mx.array]:
        forces = mx.zeros_like(positions)
        if blocks.candidate_count == 0:
            return mx.sum(positions[:, 0] * 0.0), forces

        i = blocks.left
        j = blocks.right
        displacement = positions[i] - positions[j]
        if cell is not None:
            displacement = cell.minimum_image(displacement)

        topology_mask, scales = self._block_mask_and_scales(blocks)
        r2 = mx.sum(displacement * displacement, axis=-1)
        pair_mask = topology_mask & (r2 > 0.0)
        if self.cutoff is not None:
            pair_mask = pair_mask & (r2 < self.cutoff * self.cutoff)

        safe_r2 = mx.where(pair_mask, r2, 1.0)
        sigma2_over_r2 = (self.sigma * self.sigma) / safe_r2
        inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
        inv_r12 = inv_r6 * inv_r6

        pair_energy = 4.0 * self.epsilon * (inv_r12 - inv_r6)
        if self.shift and self.cutoff is not None:
            sigma2_over_rc2 = (self.sigma * self.sigma) / (self.cutoff * self.cutoff)
            inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
            inv_rc12 = inv_rc6 * inv_rc6
            pair_energy = pair_energy - 4.0 * self.epsilon * (inv_rc12 - inv_rc6)
        pair_energy = mx.where(pair_mask, pair_energy * scales, 0.0)

        scalar = 24.0 * self.epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
        scalar = mx.where(pair_mask, scalar * scales, 0.0)
        pair_forces = scalar[..., None] * displacement
        flat_i = mx.reshape(i, (-1,))
        flat_j = mx.reshape(j, (-1,))
        flat_forces = mx.reshape(pair_forces, (-1, 3))
        forces = forces.at[flat_i].add(flat_forces).at[flat_j].add(-flat_forces)

        return mx.sum(pair_energy), forces
