"""Fused Metal kernels for recurring molecular force paths.

Collapses the per-step pairwise LJ force op-chain (gather -> minimum image -> r^2 ->
LJ scalar -> scatter-add) into a single ``mx.fast.metal_kernel`` dispatch. Diagnostic
kernels write per-pair energy without contention, while force-only kernels omit those
outputs and reductions entirely.

The simple kernel covers scalar reduced-unit LJ. The parameterized kernel covers
per-atom Lorentz-Berthelot parameters, topology scales, shifts, and smooth switching
for the production biomolecular path. A separate force-only kernel combines standard
bond, angle, periodic-torsion, improper, and CHARMM correction-map interactions into
one output. Unsupported cases fall back transparently.

Because ``tests/conftest.py`` forces ``MLX_ATOMISTIC_DEVICE=cpu``, the kernel is built
lazily on first use (not at import) so importing this module never triggers a Metal
device load.
"""

from __future__ import annotations

from math import isfinite, pi, sqrt
from typing import NamedTuple

import mlx.core as mx

from mlx_atomistic.core import as_mx_array

# Kernel body only; mx.fast.metal_kernel generates the signature from the input/output
# names. ``atomic_outputs=True`` makes every output a ``device atomic<float>*`` -- forces
# accumulate via atomic_fetch_add; pair_energy is written once per thread to its own slot.
_LJ_FORCE_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)npair[0]) {
        return;
    }
    int i = pairs_i[t];
    int j = pairs_j[t];

    float dx = positions[3 * i + 0] - positions[3 * j + 0];
    float dy = positions[3 * i + 1] - positions[3 * j + 1];
    float dz = positions[3 * i + 2] - positions[3 * j + 2];

    // orthorhombic minimum image: matches Cell.minimum_image (disp - L * round(disp / L)).
    // rint() is round-half-to-even, matching mx.round.
    float lx = box[0];
    float ly = box[1];
    float lz = box[2];
    dx -= lx * rint(dx / lx);
    dy -= ly * rint(dy / ly);
    dz -= lz * rint(dz / lz);

    float r2 = dx * dx + dy * dy + dz * dz;

    float eps = params[0];
    float sig2 = params[1];
    float cut2 = params[2];
    float eshift = params[3];

    float e = 0.0f;
    if (r2 > 0.0f && r2 < cut2) {
        float sig2_over_r2 = sig2 / r2;
        float inv_r6 = sig2_over_r2 * sig2_over_r2 * sig2_over_r2;
        float inv_r12 = inv_r6 * inv_r6;
        float scalar = 24.0f * eps * (2.0f * inv_r12 - inv_r6) / r2;
        float fx = scalar * dx;
        float fy = scalar * dy;
        float fz = scalar * dz;
        atomic_fetch_add_explicit(&forces[3 * i + 0], fx, memory_order_relaxed);
        atomic_fetch_add_explicit(&forces[3 * i + 1], fy, memory_order_relaxed);
        atomic_fetch_add_explicit(&forces[3 * i + 2], fz, memory_order_relaxed);
        atomic_fetch_add_explicit(&forces[3 * j + 0], -fx, memory_order_relaxed);
        atomic_fetch_add_explicit(&forces[3 * j + 1], -fy, memory_order_relaxed);
        atomic_fetch_add_explicit(&forces[3 * j + 2], -fz, memory_order_relaxed);
        e = 4.0f * eps * (inv_r12 - inv_r6) - eshift;
    }
    atomic_store_explicit(&pair_energy[t], e, memory_order_relaxed);
"""

_BONDED_FORCE_HEADER = r"""
struct MLXAtomisticDihedralGeometry {
    float phi;
    float3 delta_ab;
    float3 delta_bc;
    float3 delta_cd;
    float3 cross_ab_bc;
    float3 cross_bc_cd;
};

float3 mlx_atomistic_bonded_displacement(
    device const float* positions,
    int left,
    int right,
    constant const float* box
) {
    float3 displacement = float3(
        positions[3 * left + 0] - positions[3 * right + 0],
        positions[3 * left + 1] - positions[3 * right + 1],
        positions[3 * left + 2] - positions[3 * right + 2]
    );
    displacement.x -= box[0] * rint(displacement.x / box[0]);
    displacement.y -= box[1] * rint(displacement.y / box[1]);
    displacement.z -= box[2] * rint(displacement.z / box[2]);
    return displacement;
}

void mlx_atomistic_add_bonded_force(
    device atomic<float>* forces,
    int atom,
    float3 force
) {
    atomic_fetch_add_explicit(
        &forces[3 * atom + 0], force.x, memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        &forces[3 * atom + 1], force.y, memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        &forces[3 * atom + 2], force.z, memory_order_relaxed
    );
}

MLXAtomisticDihedralGeometry mlx_atomistic_dihedral_geometry(
    device const float* positions,
    int atom_i,
    int atom_j,
    int atom_k,
    int atom_m,
    constant const float* box
) {
    MLXAtomisticDihedralGeometry geometry;
    geometry.delta_ab = mlx_atomistic_bonded_displacement(
        positions, atom_j, atom_i, box
    );
    geometry.delta_bc = mlx_atomistic_bonded_displacement(
        positions, atom_j, atom_k, box
    );
    geometry.delta_cd = mlx_atomistic_bonded_displacement(
        positions, atom_m, atom_k, box
    );
    geometry.cross_ab_bc = cross(geometry.delta_ab, geometry.delta_bc);
    geometry.cross_bc_cd = cross(geometry.delta_bc, geometry.delta_cd);
    float norm_cross_1 = sqrt(max(
        dot(geometry.cross_ab_bc, geometry.cross_ab_bc), 1.0e-12f
    ));
    float norm_cross_2 = sqrt(max(
        dot(geometry.cross_bc_cd, geometry.cross_bc_cd), 1.0e-12f
    ));
    float cosine = clamp(
        dot(geometry.cross_ab_bc, geometry.cross_bc_cd)
            / (norm_cross_1 * norm_cross_2),
        -0.999999f,
        0.999999f
    );
    float angle = acos(cosine);
    float sign = dot(geometry.delta_ab, geometry.cross_bc_cd) < 0.0f
        ? -1.0f
        : 1.0f;
    geometry.phi = angle * sign;
    return geometry;
}

void mlx_atomistic_apply_dihedral_force(
    device atomic<float>* forces,
    int atom_i,
    int atom_j,
    int atom_k,
    int atom_m,
    MLXAtomisticDihedralGeometry geometry,
    float force_derivative
) {
    float norm_cross_1 = max(
        dot(geometry.cross_ab_bc, geometry.cross_ab_bc), 1.0e-12f
    );
    float norm_cross_2 = max(
        dot(geometry.cross_bc_cd, geometry.cross_bc_cd), 1.0e-12f
    );
    float norm_bc2 = max(dot(geometry.delta_bc, geometry.delta_bc), 1.0e-12f);
    float norm_bc = sqrt(norm_bc2);
    float3 force_i = (
        -force_derivative * norm_bc / norm_cross_1
    ) * geometry.cross_ab_bc;
    float3 force_m = (
        force_derivative * norm_bc / norm_cross_2
    ) * geometry.cross_bc_cd;
    float factor_j = dot(geometry.delta_ab, geometry.delta_bc) / norm_bc2;
    float factor_k = dot(geometry.delta_cd, geometry.delta_bc) / norm_bc2;
    float3 shared = factor_j * force_i - factor_k * force_m;
    float3 force_j = -(force_i - shared);
    float3 force_k = -(force_m + shared);
    mlx_atomistic_add_bonded_force(forces, atom_i, force_i);
    mlx_atomistic_add_bonded_force(forces, atom_j, force_j);
    mlx_atomistic_add_bonded_force(forces, atom_k, force_k);
    mlx_atomistic_add_bonded_force(forces, atom_m, force_m);
}
"""

_BONDED_FORCE_SOURCE = r"""
    uint task = thread_position_in_grid.x;
    uint bond_count = (uint)counts[0];
    uint angle_count = (uint)counts[1];
    uint dihedral_count = (uint)counts[2];
    uint improper_count = (uint)counts[3];
    uint cmap_count = (uint)counts[4];
    uint cmap_grid_size = (uint)counts[5];
    uint correction_count = (uint)counts[6];
    uint total_count = (
        bond_count + angle_count + dihedral_count + improper_count + cmap_count
            + correction_count
    );
    if (task >= total_count) {
        return;
    }

    if (task < bond_count) {
        int atom_i = bond_atoms[2 * task + 0];
        int atom_j = bond_atoms[2 * task + 1];
        float3 displacement = mlx_atomistic_bonded_displacement(
            positions, atom_i, atom_j, box
        );
        float distance = sqrt(max(dot(displacement, displacement), 1.0e-12f));
        float scalar = -bond_k[task] * (
            distance - bond_length[task]
        ) / distance;
        float3 force = scalar * displacement;
        mlx_atomistic_add_bonded_force(forces, atom_i, force);
        mlx_atomistic_add_bonded_force(forces, atom_j, -force);
        return;
    }
    task -= bond_count;

    if (task < angle_count) {
        int atom_i = angle_atoms[3 * task + 0];
        int atom_j = angle_atoms[3 * task + 1];
        int atom_k = angle_atoms[3 * task + 2];
        float3 left = mlx_atomistic_bonded_displacement(
            positions, atom_i, atom_j, box
        );
        float3 right = mlx_atomistic_bonded_displacement(
            positions, atom_k, atom_j, box
        );
        float left_norm2 = max(dot(left, left), 1.0e-12f);
        float right_norm2 = max(dot(right, right), 1.0e-12f);
        float left_norm = sqrt(left_norm2);
        float right_norm = sqrt(right_norm2);
        float cosine = clamp(
            dot(left, right) / (left_norm * right_norm),
            -0.999999f,
            0.999999f
        );
        float theta = acos(cosine);
        float sin_theta = sqrt(max(1.0f - cosine * cosine, 1.0e-12f));
        float prefactor = angle_k[task] * (
            theta - angle_target[task]
        ) / sin_theta;
        float3 left_force = prefactor * (
            right / (left_norm * right_norm) - cosine * left / left_norm2
        );
        float3 right_force = prefactor * (
            left / (left_norm * right_norm) - cosine * right / right_norm2
        );
        float3 center_force = -(left_force + right_force);
        mlx_atomistic_add_bonded_force(forces, atom_i, left_force);
        mlx_atomistic_add_bonded_force(forces, atom_j, center_force);
        mlx_atomistic_add_bonded_force(forces, atom_k, right_force);
        return;
    }
    task -= angle_count;

    if (task < dihedral_count) {
        int atom_i = dihedral_atoms[4 * task + 0];
        int atom_j = dihedral_atoms[4 * task + 1];
        int atom_k = dihedral_atoms[4 * task + 2];
        int atom_m = dihedral_atoms[4 * task + 3];
        MLXAtomisticDihedralGeometry geometry = mlx_atomistic_dihedral_geometry(
            positions, atom_i, atom_j, atom_k, atom_m, box
        );
        float periodic_angle = (
            dihedral_periodicity[task] * geometry.phi + dihedral_phase[task]
        );
        float force_derivative = (
            dihedral_k[task]
            * dihedral_periodicity[task]
            * sin(periodic_angle)
        );
        mlx_atomistic_apply_dihedral_force(
            forces,
            atom_i,
            atom_j,
            atom_k,
            atom_m,
            geometry,
            force_derivative
        );
        return;
    }
    task -= dihedral_count;

    if (task < improper_count) {
        int atom_i = improper_atoms[4 * task + 0];
        int atom_j = improper_atoms[4 * task + 1];
        int atom_k = improper_atoms[4 * task + 2];
        int atom_m = improper_atoms[4 * task + 3];
        MLXAtomisticDihedralGeometry geometry = mlx_atomistic_dihedral_geometry(
            positions, atom_i, atom_j, atom_k, atom_m, box
        );
        float periodicity = improper_periodicity[task];
        float shifted_angle = geometry.phi + improper_phase[task];
        float force_derivative;
        if (periodicity == 0.0f) {
            float harmonic_delta = atan2(sin(shifted_angle), cos(shifted_angle));
            force_derivative = -2.0f * improper_k[task] * harmonic_delta;
        } else {
            force_derivative = (
                improper_k[task]
                * periodicity
                * sin(periodicity * geometry.phi + improper_phase[task])
            );
        }
        mlx_atomistic_apply_dihedral_force(
            forces,
            atom_i,
            atom_j,
            atom_k,
            atom_m,
            geometry,
            force_derivative
        );
        return;
    }
    task -= improper_count;

    if (task < cmap_count) {
        int atom_i = cmap_atoms[8 * task + 0];
        int atom_j = cmap_atoms[8 * task + 1];
        int atom_k = cmap_atoms[8 * task + 2];
        int atom_m = cmap_atoms[8 * task + 3];
        int atom_n = cmap_atoms[8 * task + 4];
        int atom_o = cmap_atoms[8 * task + 5];
        int atom_p = cmap_atoms[8 * task + 6];
        int atom_q = cmap_atoms[8 * task + 7];
        MLXAtomisticDihedralGeometry phi_geometry = mlx_atomistic_dihedral_geometry(
            positions, atom_i, atom_j, atom_k, atom_m, box
        );
        MLXAtomisticDihedralGeometry psi_geometry = mlx_atomistic_dihedral_geometry(
            positions, atom_n, atom_o, atom_p, atom_q, box
        );

        float scale = (float)cmap_grid_size / (2.0f * M_PI_F);
        float phi_scaled = -phi_geometry.phi * scale;
        float psi_scaled = -psi_geometry.phi * scale;
        phi_scaled -= floor(phi_scaled / (float)cmap_grid_size) * cmap_grid_size;
        psi_scaled -= floor(psi_scaled / (float)cmap_grid_size) * cmap_grid_size;
        float phi_floor = floor(phi_scaled);
        float psi_floor = floor(psi_scaled);
        uint phi_index = (uint)phi_floor;
        uint psi_index = (uint)psi_floor;
        float phi_t = phi_scaled - phi_floor;
        float psi_t = psi_scaled - psi_floor;
        float phi_t2 = phi_t * phi_t;
        float psi_t2 = psi_t * psi_t;
        float phi_powers[4] = {1.0f, phi_t, phi_t2, phi_t2 * phi_t};
        float psi_powers[4] = {1.0f, psi_t, psi_t2, psi_t2 * psi_t};
        float phi_derivatives[4] = {0.0f, 1.0f, 2.0f * phi_t, 3.0f * phi_t2};
        float psi_derivatives[4] = {0.0f, 1.0f, 2.0f * psi_t, 3.0f * psi_t2};
        uint coefficient_base = 16u * (
            ((uint)cmap_indices[task] * cmap_grid_size + phi_index)
                * cmap_grid_size
            + psi_index
        );
        float derivative_phi_t = 0.0f;
        float derivative_psi_t = 0.0f;
        for (uint row = 0; row < 4; ++row) {
            for (uint column = 0; column < 4; ++column) {
                float coefficient = cmap_coefficients[
                    coefficient_base + 4u * row + column
                ];
                derivative_phi_t += (
                    coefficient * phi_derivatives[row] * psi_powers[column]
                );
                derivative_psi_t += (
                    coefficient * phi_powers[row] * psi_derivatives[column]
                );
            }
        }
        // This helper accepts -dE/d(angle).  The CMAP lookup coordinate is
        // -angle*scale, so the two signs cancel.
        mlx_atomistic_apply_dihedral_force(
            forces,
            atom_i,
            atom_j,
            atom_k,
            atom_m,
            phi_geometry,
            scale * derivative_phi_t
        );
        mlx_atomistic_apply_dihedral_force(
            forces,
            atom_n,
            atom_o,
            atom_p,
            atom_q,
            psi_geometry,
            scale * derivative_psi_t
        );
        return;
    }
    task -= cmap_count;

    if (task < correction_count) {
        int atom_i = correction_atoms[2 * task + 0];
        int atom_j = correction_atoms[2 * task + 1];
        float3 displacement = mlx_atomistic_bonded_displacement(
            positions, atom_i, atom_j, box
        );
        float r2 = dot(displacement, displacement);
        if (r2 <= 0.0f) {
            return;
        }
        float inv_distance = rsqrt(r2);
        float inv_r2 = inv_distance * inv_distance;
        float scalar = correction_coulomb[0] * correction_charge_products[task]
            * inv_r2 * inv_distance;
        float epsilon = correction_lj_epsilon[task];
        if (epsilon > 0.0f) {
            float sigma = correction_lj_sigma[task];
            float sigma2_over_r2 = sigma * sigma * inv_r2;
            float inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
            float inv_r12 = inv_r6 * inv_r6;
            scalar += 24.0f * epsilon * (2.0f * inv_r12 - inv_r6) * inv_r2;
        }
        float3 force = scalar * displacement;
        mlx_atomistic_add_bonded_force(forces, atom_i, force);
        mlx_atomistic_add_bonded_force(forces, atom_j, -force);
    }
"""

_kernel_singleton = None
_parameterized_kernel_singleton = None
_pme_direct_kernel_singleton = None
_pme_direct_virial_kernel_singleton = None
_pme_direct_force_only_kernel_singleton = None
_prepared_pme_direct_force_only_kernel_singleton = None
_tile_pme_direct_kernel_singleton = None
_tile_pme_direct_force_only_kernel_singleton = None
_tile_nbfix_pme_direct_kernel_singleton = None
_tile_nbfix_pme_direct_force_only_kernel_singleton = None
_interaction32_pack_kernel_singleton = None
_interaction32_force_kernel_singleton = None
_interaction32_canonical_force_kernel_singleton = None
_interaction32_fused_half_canonical_force_kernel_singleton = None
_interaction32_fused_half_nbfix_canonical_force_kernel_singleton = None
_interaction32_scatter_kernel_singleton = None
_interaction32_block_geometry_kernel_singleton = None
_interaction32_ordinary_count_kernel_singleton = None
_interaction32_ordinary_cached_count_kernel_singleton = None
_interaction32_ordinary_scatter_kernel_singleton = None
_interaction32_ordinary_cached_scatter_kernel_singleton = None
_interaction32_outer_inner_mode_count_kernel_singleton = None
_interaction32_outer_inner_mode_scatter_kernel_singleton = None
_interaction32_special_pair_words_kernel_singleton = None
_interaction32_special_block_scatter_kernel_singleton = None
_interaction32_special_work_kernel_singleton = None
_owner_compute32_force_kernel_singleton = None
_sparse_pme_correction_force_only_kernel_singleton = None
_pme_cutoff_correction_virial_kernel_singleton = None
_pme_order5_spread_kernel_singleton = None
_pme_order5_interpolate_kernel_singleton = None
_pme_order5_force_only_kernel_singleton = None
_pme_order5_normalized_real_grid_force_only_kernel_singleton = None
_pme_order5_complex_grid_force_only_kernel_singleton = None
_aligned_topology_lj_scales_kernel_singleton = None
_neighbor_cell_pair_candidates_kernel_singleton = None
_neighbor_pair_cutoff_mask_kernel_singleton = None
_neighbor_pair_ordered_scatter_kernel_singleton = None
_neighbor_cell_atom_blocks_kernel_singleton = None
_neighbor_cell_tile_candidates_kernel_singleton = None
_neighbor_tile_membership_kernel_singleton = None
_neighbor_tile_ordered_scatter_kernel_singleton = None
_neighbor_tile_left_counts_kernel_singleton = None
_neighbor_tile_left_scatter_kernel_singleton = None
_neighbor_tile_column_scatter_kernel_singleton = None
_neighbor_tile_force_groups_kernel_singleton = None
_neighbor_tile_member_counts_kernel_singleton = None
_neighbor_tile_pair_scatter_kernel_singleton = None
_tile_topology_lj_masks_kernel_singleton = None
_shake_cluster_position_kernel_singleton = None
_shake_cluster_velocity_kernel_singleton = None
_settle_water_position_kernel_singleton = None
_settle_water_velocity_kernel_singleton = None
_langevin_baoab_drift_kernel_singleton = None
_dense_constraint_apply_kernel_singleton = None
_small_constraint_cluster_position_kernel_singleton = None
_small_constraint_cluster_velocity_kernel_singleton = None
_fused_bonded_force_only_kernel_singleton = None

# Neighbor compaction preserves short runs of a common left atom, so one worker
# can sum those contributions locally before issuing global atomics.
_PREPARED_PME_PAIRS_PER_WORKER = 8

_TILE_PME_BLOCK_SIZE = 4
_TILE_PME_LANES_PER_TILE = _TILE_PME_BLOCK_SIZE * _TILE_PME_BLOCK_SIZE
_TILE_PME_MASK_WORD_COUNT = (_TILE_PME_LANES_PER_TILE + 31) // 32
_TILE_PME_COLUMN_DESCRIPTOR_MEMBER_SHIFT = 28
_TILE_PME_COLUMN_DESCRIPTOR_INDEX_MASK = (
    1 << _TILE_PME_COLUMN_DESCRIPTOR_MEMBER_SHIFT
) - 1
_TILE_PME_COLUMN_DESCRIPTOR_MEMBER_DIVISOR = (
    1 << _TILE_PME_COLUMN_DESCRIPTOR_MEMBER_SHIFT
)
_TILE_PME_COLUMN_DESCRIPTOR_TILE_LIMIT = (
    _TILE_PME_COLUMN_DESCRIPTOR_MEMBER_DIVISOR // _TILE_PME_BLOCK_SIZE
)
_TILE_BUILD_BLOCK_SIZE = 8
_TILE_BUILD_SUBTILES_PER_TILE = (_TILE_BUILD_BLOCK_SIZE // _TILE_PME_BLOCK_SIZE) ** 2
_TILE_PME_GROUPS_PER_THREADGROUP = 4
_TILE_PME_DISPATCH_WIDTH = 32 * _TILE_PME_GROUPS_PER_THREADGROUP
_TILE_MEMBERSHIP_THREADGROUP_SIZE = _TILE_PME_LANES_PER_TILE
_TILE_BUILD_MEMBERSHIP_THREADGROUP_SIZE = 32
_TILE_PME_THREADGROUP_TEMPORARY_BYTES = (
    _TILE_PME_BLOCK_SIZE * (1 + 3 + 1 + 1 + 1) * 4 * _TILE_PME_GROUPS_PER_THREADGROUP
)


def _tile_pme_threadgroup_count(force_group_count: int) -> int:
    """Return threadgroups needed to cover every spatial-tile force group."""

    packing = _TILE_PME_GROUPS_PER_THREADGROUP
    return (force_group_count + packing - 1) // packing


# The float32 erf/expm1 implementation below is adapted from MLX's Metal
# kernels:
# https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels
#
# MLX copyright and license:
# Copyright © 2023 Apple Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The expm1 portion carries this original license:
# Copyright (c) 2015-2023 Norbert Juffa
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Keeping MLX's approximation preserves the direct-space PME arithmetic used by
# ``mx.erf``.
_ERF_HEADER = r"""
float mlx_atomistic_expm1f_scaled(float a, float b) {
    float f, j, r, s, t, u, v, x, y;
    int i;
    j = metal::fma(1.442695f, a, 12582912.0f);
    j = j - 12582912.0f;
    i = (int)j;
    f = metal::fma(j, -6.93145752e-1f, a);
    s = f * f;
    if (a == 0.0f) {
        s = a;
    }
    r = 1.97350979e-4f;
    r = metal::fma(r, f, 1.39309070e-3f);
    r = metal::fma(r, f, 8.33343994e-3f);
    r = metal::fma(r, f, 4.16668020e-2f);
    r = metal::fma(r, f, 1.66666716e-1f);
    r = metal::fma(r, f, 4.99999970e-1f);
    u = (j == 1.0f) ? (f + 0.5f) : f;
    v = metal::fma(r, s, u);
    s = 0.5f * b;
    t = metal::ldexp(s, i);
    y = t - s;
    x = (t - y) - s;
    r = metal::fma(v, t, x) + y;
    r = r + r;
    if (j == 0.0f) {
        r = v;
    }
    if (j == 1.0f) {
        r = v + v;
    }
    return r;
}

float mlx_atomistic_expm1f(float a) {
    float r = mlx_atomistic_expm1f_scaled(a, 1.0f);
    if (metal::abs(a - 1.0f) > 88.0f) {
        r = metal::pow(2.0f, a);
        r = metal::fma(r, r, -1.0f);
    }
    return r;
}

float mlx_atomistic_erf(float a) {
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f) {
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f);
        r = metal::fma(r, t, -6.34846687e-1f);
        r = metal::fma(r, t, -1.28717512e-1f);
        r = metal::fma(r, t, -t);
        r = -mlx_atomistic_expm1f(r);
        r = metal::copysign(r, a);
    } else {
        r = -5.96761703e-4f;
        r = metal::fma(r, s, 4.99119423e-3f);
        r = metal::fma(r, s, -2.67681349e-2f);
        r = metal::fma(r, s, 1.12819925e-1f);
        r = metal::fma(r, s, -3.76125336e-1f);
        r = metal::fma(r, s, 1.28379166e-1f);
        r = metal::fma(r, a, a);
    }
    return r;
}

// Complementary error function for non-negative arguments only. On the branch
// above 0.927734375 the erf implementation forms 1 - exp(r) from the same
// polynomial, so erfc is exp(r) directly: one fewer transcendental round trip
// through expm1 and no cancelling subtraction. Screened Coulomb arguments are
// alpha times a positive distance, so the negative half is never reached.
float mlx_atomistic_erfc_nonnegative(float a) {
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f) {
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f);
        r = metal::fma(r, t, -6.34846687e-1f);
        r = metal::fma(r, t, -1.28717512e-1f);
        r = metal::fma(r, t, -t);
        return metal::exp(r);
    }
    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
    return 1.0f - r;
}

// Return erfc(a) and exp(-a*a) for a non-negative Ewald argument while
// evaluating the exponential once.  The erfc approximation is the
// Abramowitz-Stegun 7.1.26 form used by OpenMM's single-precision kernels.
float2 mlx_atomistic_ewald_erfc_exp(float a) {
    float exponential = metal::exp(-a * a);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float polynomial = metal::fma(1.061405429f, t, -1.453152027f);
    polynomial = metal::fma(polynomial, t, 1.421413741f);
    polynomial = metal::fma(polynomial, t, -0.284496736f);
    polynomial = metal::fma(polynomial, t, 0.254829592f);
    return float2(polynomial * t * exponential, exponential);
}

"""

_PARAMETERIZED_LJ_FORCE_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)counts[0]) {
        return;
    }
    int i = pairs_i[t];
    int j = pairs_j[t];

    float dx = positions[3 * i + 0] - positions[3 * j + 0];
    float dy = positions[3 * i + 1] - positions[3 * j + 1];
    float dz = positions[3 * i + 2] - positions[3 * j + 2];
    float lx = box[0];
    float ly = box[1];
    float lz = box[2];
    dx -= lx * rint(dx / lx);
    dy -= ly * rint(dy / ly);
    dz -= lz * rint(dz / lz);

    float r2 = dx * dx + dy * dy + dz * dz;
    float cut2 = params[0];
    float pair_energy_value = 0.0f;
    if (r2 > 0.0f && r2 < cut2) {
        float sigma_ij = 0.5f * (sigma[i] + sigma[j]);
        float epsilon_ij = sqrt(epsilon[i] * epsilon[j]);
        float sigma2_over_r2 = sigma_ij * sigma_ij / r2;
        float inv_r6 =
            sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
        float inv_r12 = inv_r6 * inv_r6;
        float unswitched_energy =
            4.0f * epsilon_ij * (inv_r12 - inv_r6);
        if (params[1] > 0.5f) {
            float sigma2_over_rc2 = sigma_ij * sigma_ij / cut2;
            float inv_rc6 =
                sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2;
            unswitched_energy -=
                4.0f * epsilon_ij * (inv_rc6 * inv_rc6 - inv_rc6);
        }

        float distance = sqrt(r2);
        float switch_value = 1.0f;
        float switch_derivative = 0.0f;
        if (params[2] > 0.5f) {
            float switch_distance = params[3];
            float width = params[4];
            float x = clamp(
                (distance - switch_distance) / width,
                0.0f,
                1.0f
            );
            float x2 = x * x;
            float x3 = x2 * x;
            float x4 = x3 * x;
            float x5 = x4 * x;
            switch_value = 1.0f - (
                10.0f * x3 - 15.0f * x4 + 6.0f * x5
            );
            if (distance > switch_distance && distance < params[5]) {
                switch_derivative = -(
                    30.0f * x2 - 60.0f * x3 + 30.0f * x4
                ) / width;
            }
        }

        float scale = scales[counts[1] == 1 ? 0 : t];
        pair_energy_value = unswitched_energy * switch_value * scale;
        float scalar = (
            24.0f
            * epsilon_ij
            * (2.0f * inv_r12 - inv_r6)
            / r2
            * switch_value
            - unswitched_energy * switch_derivative / distance
        ) * scale;
        float fx = scalar * dx;
        float fy = scalar * dy;
        float fz = scalar * dz;
        atomic_fetch_add_explicit(
            &forces[3 * i + 0], fx, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * i + 1], fy, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * i + 2], fz, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * j + 0], -fx, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * j + 1], -fy, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * j + 2], -fz, memory_order_relaxed
        );
    }
    atomic_store_explicit(
        &pair_energy[t], pair_energy_value, memory_order_relaxed
    );
"""

_PARAMETERIZED_PME_DIRECT_SOURCE = r"""
#ifdef MLX_ATOMISTIC_PAIRS_PER_WORKER
    // Compact neighbor pairs retain cell-task order. Processing a short run
    // per worker reduces dispatch width and lets repeated left-atom forces
    // accumulate in registers before reaching the global atomic buffer.
    uint worker = thread_position_in_grid.x;
    uint first_pair = worker * MLX_ATOMISTIC_PAIRS_PER_WORKER;
    if (first_pair >= (uint)npair[0]) {
        return;
    }
    int accumulated_i = -1;
    float accumulated_fx = 0.0f;
    float accumulated_fy = 0.0f;
    float accumulated_fz = 0.0f;
    for (
        uint local_pair = 0;
        local_pair < MLX_ATOMISTIC_PAIRS_PER_WORKER;
        local_pair++
    ) {
        uint t = first_pair + local_pair;
        if (t >= (uint)npair[0]) {
            break;
        }
#else
    uint t = thread_position_in_grid.x;
    if (t >= (uint)npair[0]) {
        return;
    }
#endif
    int i = pairs_i[t];
    int j = pairs_j[t];

    float dx = positions[3 * i + 0] - positions[3 * j + 0];
    float dy = positions[3 * i + 1] - positions[3 * j + 1];
    float dz = positions[3 * i + 2] - positions[3 * j + 2];
    float lx = box[0];
    float ly = box[1];
    float lz = box[2];
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
    dx -= lx * rint(dx * box[3]);
    dy -= ly * rint(dy * box[4]);
    dz -= lz * rint(dz * box[5]);
#else
    dx -= lx * rint(dx / lx);
    dy -= ly * rint(dy / ly);
    dz -= lz * rint(dz / lz);
#endif

    float r2 = dx * dx + dy * dy + dz * dz;
#ifndef MLX_ATOMISTIC_FORCE_ONLY
    float lj_energy_value = 0.0f;
    float coulomb_energy_value = 0.0f;
#endif
#ifdef MLX_ATOMISTIC_VIRIAL
    float virial_x = 0.0f;
    float virial_y = 0.0f;
    float virial_z = 0.0f;
#endif
    if (r2 > 0.0f && r2 < params[0]) {
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
        float inv_distance = rsqrt(r2);
        float inv_r2 = inv_distance * inv_distance;
        float distance = r2 * inv_distance;
#else
        float distance = sqrt(r2);
#endif
        float scalar = 0.0f;
        float lj_scale = lj_scales[t];
        if (lj_scale != 0.0f) {
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
            float sigma_ij = sigma[i] + sigma[j];
            float epsilon_ij = epsilon[i] * epsilon[j];
            float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
#else
            float sigma_ij = 0.5f * (sigma[i] + sigma[j]);
            float epsilon_ij = sqrt(epsilon[i] * epsilon[j]);
            float sigma2_over_r2 = sigma_ij * sigma_ij / r2;
#endif
            float inv_r6 =
                sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
            float inv_r12 = inv_r6 * inv_r6;
            float unswitched_energy =
                4.0f * epsilon_ij * (inv_r12 - inv_r6);
            if (params[1] > 0.5f) {
                float sigma2_over_rc2 =
                    sigma_ij * sigma_ij / params[0];
                float inv_rc6 =
                    sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2;
                unswitched_energy -=
                    4.0f
                    * epsilon_ij
                    * (inv_rc6 * inv_rc6 - inv_rc6);
            }

            float switch_value = 1.0f;
            float switch_derivative = 0.0f;
            if (params[2] > 0.5f) {
                float x = clamp(
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
                    (distance - params[3]) * params[9],
#else
                    (distance - params[3]) / params[4],
#endif
                    0.0f,
                    1.0f
                );
                float x2 = x * x;
                float x3 = x2 * x;
                float x4 = x3 * x;
                float x5 = x4 * x;
                switch_value = 1.0f - (
                    10.0f * x3 - 15.0f * x4 + 6.0f * x5
                );
                if (distance > params[3] && distance < params[5]) {
                    switch_derivative = -(
                        30.0f * x2 - 60.0f * x3 + 30.0f * x4
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
                    ) * params[9];
#else
                    ) / params[4];
#endif
                }
            }
#ifndef MLX_ATOMISTIC_FORCE_ONLY
            lj_energy_value =
                unswitched_energy * switch_value * lj_scale;
#endif
            scalar += (
                24.0f
                * epsilon_ij
                * (2.0f * inv_r12 - inv_r6)
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
                * inv_r2
#else
                / r2
#endif
                * switch_value
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
                - unswitched_energy * switch_derivative * inv_distance
#else
                - unswitched_energy * switch_derivative / distance
#endif
            ) * lj_scale;
        }

        float qij = charges[i] * charges[j];
        float erfc_term =
            1.0f - mlx_atomistic_erf(params[7] * distance);
#ifndef MLX_ATOMISTIC_FORCE_ONLY
        coulomb_energy_value =
            params[6] * qij * erfc_term / distance;
#endif
        scalar += params[6] * qij * (
#ifdef MLX_ATOMISTIC_PREPARED_INVARIANTS
            erfc_term * inv_r2 * inv_distance
            + params[8] * exp(-params[7] * params[7] * r2) * inv_r2
#else
            erfc_term / (r2 * distance)
            + params[8] * exp(-params[7] * params[7] * r2) / r2
#endif
        );

        float fx = scalar * dx;
        float fy = scalar * dy;
        float fz = scalar * dz;
#ifdef MLX_ATOMISTIC_VIRIAL
        virial_x = dx * fx;
        virial_y = dy * fy;
        virial_z = dz * fz;
#endif
#ifdef MLX_ATOMISTIC_PAIRS_PER_WORKER
        if (accumulated_i != i) {
            if (accumulated_i >= 0) {
                atomic_fetch_add_explicit(
                    &forces[3 * accumulated_i + 0],
                    accumulated_fx,
                    memory_order_relaxed
                );
                atomic_fetch_add_explicit(
                    &forces[3 * accumulated_i + 1],
                    accumulated_fy,
                    memory_order_relaxed
                );
                atomic_fetch_add_explicit(
                    &forces[3 * accumulated_i + 2],
                    accumulated_fz,
                    memory_order_relaxed
                );
            }
            accumulated_i = i;
            accumulated_fx = 0.0f;
            accumulated_fy = 0.0f;
            accumulated_fz = 0.0f;
        }
        accumulated_fx += fx;
        accumulated_fy += fy;
        accumulated_fz += fz;
#else
        atomic_fetch_add_explicit(
            &forces[3 * i + 0], fx, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * i + 1], fy, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * i + 2], fz, memory_order_relaxed
        );
#endif
        atomic_fetch_add_explicit(
            &forces[3 * j + 0], -fx, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * j + 1], -fy, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * j + 2], -fz, memory_order_relaxed
        );
    }
#ifndef MLX_ATOMISTIC_FORCE_ONLY
    atomic_store_explicit(
        &pair_lj_energy[t], lj_energy_value, memory_order_relaxed
    );
    atomic_store_explicit(
        &pair_coulomb_energy[t],
        coulomb_energy_value,
        memory_order_relaxed
    );
#endif
#ifdef MLX_ATOMISTIC_VIRIAL
    atomic_store_explicit(
        &pair_virial[3 * t + 0], virial_x, memory_order_relaxed
    );
    atomic_store_explicit(
        &pair_virial[3 * t + 1], virial_y, memory_order_relaxed
    );
    atomic_store_explicit(
        &pair_virial[3 * t + 2], virial_z, memory_order_relaxed
    );
#endif
#ifdef MLX_ATOMISTIC_PAIRS_PER_WORKER
    }
    if (accumulated_i >= 0) {
        atomic_fetch_add_explicit(
            &forces[3 * accumulated_i + 0],
            accumulated_fx,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * accumulated_i + 1],
            accumulated_fy,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * accumulated_i + 2],
            accumulated_fz,
            memory_order_relaxed
        );
    }
#endif
"""

_PARAMETERIZED_PME_DIRECT_VIRIAL_SOURCE = (
    "#define MLX_ATOMISTIC_VIRIAL 1\n" + _PARAMETERIZED_PME_DIRECT_SOURCE
)
_PARAMETERIZED_PME_DIRECT_FORCE_ONLY_SOURCE = (
    "#define MLX_ATOMISTIC_FORCE_ONLY 1\n" + _PARAMETERIZED_PME_DIRECT_SOURCE
)
_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE = (
    "#define MLX_ATOMISTIC_FORCE_ONLY 1\n"
    "#define MLX_ATOMISTIC_PREPARED_INVARIANTS 1\n"
    f"#define MLX_ATOMISTIC_PAIRS_PER_WORKER {_PREPARED_PME_PAIRS_PER_WORKER}u\n"
    + _PARAMETERIZED_PME_DIRECT_SOURCE
)

# One 32-lane SIMD group owns up to 32 non-empty right-atom columns for a shared
# four-atom left block. Each lane accumulates its right atom across all four
# left atoms, while SIMD reductions produce the four left forces without the
# repeated threadgroup barriers and pair-force scratch buffers used by the
# previous tile schedule. Compact column descriptors prevent lanes from being
# assigned to tile columns whose four membership bits are all zero. Each
# descriptor also carries those four membership bits, avoiding an indirect
# tile-mask read in the recurring force kernel.
#
# A threadgroup carries `_TILE_PME_GROUPS_PER_THREADGROUP` independent SIMD
# groups so a core has several force groups in flight against memory latency,
# and each lane reuses its three mask words across all four left slots. Both are
# exact: the same floats are read and the same arithmetic runs, only the
# dispatch shape and the load count change. Trailing SIMD groups beyond the
# last force group clamp their indices and are held inactive rather than
# returning early, because every thread must reach the barrier below.
_TILE_PREPARED_PME_DIRECT_SOURCE = r"""
    uint tg_thread = thread_position_in_threadgroup.x;
    uint lane = tg_thread & 31u;
    uint sub = tg_thread >> 5u;
    uint group_id = threadgroup_position_in_grid.x * GROUPS_PER_TGu + sub;
    int total_groups = ngroups[0];
    bool group_active = group_id < (uint)total_groups;
    uint group = group_active ? group_id : (uint)(total_groups - 1);
    int group_start = force_group_starts[group];
    int group_count = force_group_counts[group];
    uint first_column = (uint)force_columns[group_start];

    threadgroup int left_atoms[4 * GROUPS_PER_TG];
    threadgroup float left_positions[12 * GROUPS_PER_TG];
    threadgroup float left_half_sigma[4 * GROUPS_PER_TG];
    threadgroup float left_sqrt_epsilon[4 * GROUPS_PER_TG];
    threadgroup float left_charges[4 * GROUPS_PER_TG];
#ifdef MLX_ATOMISTIC_NBFIX
    threadgroup int left_type_ids[4 * GROUPS_PER_TG];
#endif
    uint lbase = sub * 4u;
    uint pbase = sub * 12u;

    int first_tile = (int)((first_column & 0x0fffffffu) >> 2);
    int left_block = tile_blocks[2 * first_tile + 0];
    if (lane < 4u) {
        int left_atom = atom_blocks[4 * left_block + lane];
        left_atoms[lbase + lane] = left_atom;
        int safe_left = max(left_atom, 0);
        left_positions[pbase + 3 * lane + 0] = positions[3 * safe_left + 0];
        left_positions[pbase + 3 * lane + 1] = positions[3 * safe_left + 1];
        left_positions[pbase + 3 * lane + 2] = positions[3 * safe_left + 2];
        left_half_sigma[lbase + lane] = half_sigma[safe_left];
        left_sqrt_epsilon[lbase + lane] = sqrt_epsilon[safe_left];
        left_charges[lbase + lane] = charges[safe_left];
#ifdef MLX_ATOMISTIC_NBFIX
        left_type_ids[lbase + lane] = atom_type_ids[safe_left];
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool tile_active = group_active && lane < (uint)group_count;
    int safe_local_column = min((int)lane, group_count - 1);
    uint encoded_column = (uint)force_columns[group_start + safe_local_column];
    int tile = (int)((encoded_column & 0x0fffffffu) >> 2);
    uint right_slot = encoded_column & 3u;
    uint column_member_mask = encoded_column >> 28;
    int right_block = tile_blocks[2 * tile + 1];
    int right_atom = atom_blocks[4 * right_block + right_slot];
    int safe_right = max(right_atom, 0);
    float right_x = positions[3 * safe_right + 0];
    float right_y = positions[3 * safe_right + 1];
    float right_z = positions[3 * safe_right + 2];
    float right_half_sigma_value = half_sigma[safe_right];
    float right_sqrt_epsilon_value = sqrt_epsilon[safe_right];
    float right_charge = charges[safe_right];
#ifdef MLX_ATOMISTIC_NBFIX
    int right_type_id = atom_type_ids[safe_right];
#endif
    float accumulated_right_fx = 0.0f;
    float accumulated_right_fy = 0.0f;
    float accumulated_right_fz = 0.0f;
#ifndef MLX_ATOMISTIC_FORCE_ONLY
    float accumulated_lj_energy = 0.0f;
    float accumulated_coulomb_energy = 0.0f;
#endif
    uint lj_word = lj_enabled_mask[tile];
    uint one_four_word = lj_one_four_mask[tile];
    float box_lx = box[0];
    float box_ly = box[1];
    float box_lz = box[2];
    float box_ix = box[3];
    float box_iy = box[4];
    float box_iz = box[5];
    float cutoff2 = params[0];
    float shift_flag = params[1];
    float switch_flag = params[2];
    float switch_start = params[3];
    float switch_end = params[5];
    float coulomb = params[6];
    float ewald_alpha = params[7];
    float ewald_self = params[8];
    float inv_switch_width = params[9];
    float one_four_scale = params[10];
#ifdef MLX_ATOMISTIC_NBFIX
    int nbfix_type_count = (int)params[11];
#endif

    for (uint left_slot = 0u; left_slot < 4u; left_slot++) {
        uint bit = 4u * left_slot + right_slot;
        bool member = tile_active
            && ((column_member_mask >> left_slot) & 1u) != 0u;
        int left_atom = left_atoms[lbase + left_slot];
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;
#ifndef MLX_ATOMISTIC_FORCE_ONLY
        float pair_lj_energy = 0.0f;
        float pair_coulomb_energy = 0.0f;
#endif
        if (member && left_atom >= 0 && right_atom >= 0) {
            float dx = left_positions[pbase + 3 * left_slot + 0] - right_x;
            float dy = left_positions[pbase + 3 * left_slot + 1] - right_y;
            float dz = left_positions[pbase + 3 * left_slot + 2] - right_z;
            dx -= box_lx * rint(dx * box_ix);
            dy -= box_ly * rint(dy * box_iy);
            dz -= box_lz * rint(dz * box_iz);
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 0.0f && r2 < cutoff2) {
                float inv_distance = rsqrt(r2);
                float inv_r2 = inv_distance * inv_distance;
                float distance = r2 * inv_distance;
                float scalar = 0.0f;
                bool lj_enabled = ((lj_word >> bit) & 1u) != 0u;
                if (lj_enabled) {
                    bool one_four = ((one_four_word >> bit) & 1u) != 0u;
                    float lj_scale = one_four ? one_four_scale : 1.0f;
                    float sigma_ij = left_half_sigma[lbase + left_slot]
                        + right_half_sigma_value;
                    float epsilon_ij = left_sqrt_epsilon[lbase + left_slot]
                        * right_sqrt_epsilon_value;
#ifdef MLX_ATOMISTIC_NBFIX
                    int nbfix_index =
                        left_type_ids[lbase + left_slot] * nbfix_type_count
                        + right_type_id;
                    float nbfix_sigma_value = nbfix_sigma[nbfix_index];
                    if (nbfix_sigma_value > 0.0f) {
                        sigma_ij = nbfix_sigma_value;
                        epsilon_ij = nbfix_epsilon[nbfix_index];
                    }
#endif
                    float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                    float inv_r6 =
                        sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
                    float inv_r12 = inv_r6 * inv_r6;
                    // The unswitched energy reaches the force only through a
                    // product with switch_derivative, which is identically zero
                    // when switching is disabled, so the whole chain including
                    // the shift correction is dead work in that case.
                    float unswitched_energy = 0.0f;
                    float switch_value = 1.0f;
                    float switch_derivative = 0.0f;
                    if (switch_flag > 0.5f) {
                        unswitched_energy =
                            4.0f * epsilon_ij * (inv_r12 - inv_r6);
                        if (shift_flag > 0.5f) {
                            float sigma2_over_rc2 =
                                sigma_ij * sigma_ij / cutoff2;
                            float inv_rc6 = sigma2_over_rc2
                                * sigma2_over_rc2 * sigma2_over_rc2;
                            unswitched_energy -= 4.0f * epsilon_ij
                                * (inv_rc6 * inv_rc6 - inv_rc6);
                        }
                        float x = clamp(
                            (distance - switch_start) * inv_switch_width,
                            0.0f,
                            1.0f
                        );
                        float x2 = x * x;
                        float x3 = x2 * x;
                        float x4 = x3 * x;
                        float x5 = x4 * x;
                        switch_value = 1.0f
                            - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                        if (distance > switch_start && distance < switch_end) {
                            switch_derivative = -(
                                30.0f * x2 - 60.0f * x3 + 30.0f * x4
                            ) * inv_switch_width;
                        }
                    }
#ifndef MLX_ATOMISTIC_FORCE_ONLY
                    if (switch_flag <= 0.5f) {
                        unswitched_energy =
                            4.0f * epsilon_ij * (inv_r12 - inv_r6);
                        if (shift_flag > 0.5f) {
                            float sigma2_over_rc2 =
                                sigma_ij * sigma_ij / cutoff2;
                            float inv_rc6 = sigma2_over_rc2
                                * sigma2_over_rc2 * sigma2_over_rc2;
                            unswitched_energy -= 4.0f * epsilon_ij
                                * (inv_rc6 * inv_rc6 - inv_rc6);
                        }
                    }
                    pair_lj_energy =
                        unswitched_energy * switch_value * lj_scale;
#endif
                    scalar += (
                        24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                        * inv_r2 * switch_value
                        - unswitched_energy * switch_derivative * inv_distance
                    ) * lj_scale;
                }

                float qij = left_charges[lbase + left_slot] * right_charge;
                float erfc_term =
                    mlx_atomistic_erfc_nonnegative(ewald_alpha * distance);
#ifndef MLX_ATOMISTIC_FORCE_ONLY
                pair_coulomb_energy =
                    coulomb * qij * erfc_term * inv_distance;
#endif
                scalar += coulomb * qij * (
                    erfc_term * inv_r2 * inv_distance
                    + ewald_self * exp(-ewald_alpha * ewald_alpha * r2) * inv_r2
                );
                fx = scalar * dx;
                fy = scalar * dy;
                fz = scalar * dz;
            }
        }

        accumulated_right_fx -= fx;
        accumulated_right_fy -= fy;
        accumulated_right_fz -= fz;
#ifndef MLX_ATOMISTIC_FORCE_ONLY
        accumulated_lj_energy += pair_lj_energy;
        accumulated_coulomb_energy += pair_coulomb_energy;
#endif
        float reduced_left_fx = simd_sum(fx);
        float reduced_left_fy = simd_sum(fy);
        float reduced_left_fz = simd_sum(fz);
        if (
            group_active
            && lane == left_slot
            && left_atom >= 0
            && (
                reduced_left_fx != 0.0f
                || reduced_left_fy != 0.0f
                || reduced_left_fz != 0.0f
            )
        ) {
            atomic_fetch_add_explicit(
                &forces[3 * left_atom + 0], reduced_left_fx, memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &forces[3 * left_atom + 1], reduced_left_fy, memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &forces[3 * left_atom + 2], reduced_left_fz, memory_order_relaxed
            );
        }
    }

    if (
        tile_active
        && right_atom >= 0
        && (
            accumulated_right_fx != 0.0f
            || accumulated_right_fy != 0.0f
            || accumulated_right_fz != 0.0f
        )
    ) {
        atomic_fetch_add_explicit(
            &forces[3 * right_atom + 0], accumulated_right_fx, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * right_atom + 1], accumulated_right_fy, memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &forces[3 * right_atom + 2], accumulated_right_fz, memory_order_relaxed
        );
    }
#ifndef MLX_ATOMISTIC_FORCE_ONLY
    float reduced_lj_energy = simd_sum(accumulated_lj_energy);
    float reduced_coulomb_energy = simd_sum(accumulated_coulomb_energy);
    if (group_active && lane == 0u) {
        atomic_store_explicit(
            &group_lj_energy[group], reduced_lj_energy, memory_order_relaxed
        );
        atomic_store_explicit(
            &group_coulomb_energy[group],
            reduced_coulomb_energy,
            memory_order_relaxed
        );
    }
#endif
""".replace("GROUPS_PER_TG", str(_TILE_PME_GROUPS_PER_THREADGROUP))

_TILE_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE = (
    "#define MLX_ATOMISTIC_FORCE_ONLY 1\n" + _TILE_PREPARED_PME_DIRECT_SOURCE
)

_TILE_NBFIX_PREPARED_PME_DIRECT_SOURCE = (
    "#define MLX_ATOMISTIC_NBFIX 1\n" + _TILE_PREPARED_PME_DIRECT_SOURCE
)

_TILE_NBFIX_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE = (
    "#define MLX_ATOMISTIC_FORCE_ONLY 1\n"
    "#define MLX_ATOMISTIC_NBFIX 1\n"
    + _TILE_PREPARED_PME_DIRECT_SOURCE
)

_INTERACTION32_GROUPS_PER_THREADGROUP = 4


class _Interaction32ForceStages(NamedTuple):
    """Lazy arrays for profiling the experimental interaction-force graph."""

    packed_posq: mx.array
    packed_lj: mx.array
    ordered_forces: mx.array
    forces: mx.array


_INTERACTION32_PACK_SOURCE = r"""
    uint ordered = thread_position_in_grid.x;
    uint padded_count = (uint)counts[0];
    if (ordered >= padded_count) {
        return;
    }
    int atom = atom_order[ordered];
    bool valid = atom >= 0 && atom < counts[1];
    int safe_atom = valid ? atom : 0;
    packed_posq[4 * ordered + 0] = valid ? positions[3 * safe_atom + 0] : 0.0f;
    packed_posq[4 * ordered + 1] = valid ? positions[3 * safe_atom + 1] : 0.0f;
    packed_posq[4 * ordered + 2] = valid ? positions[3 * safe_atom + 2] : 0.0f;
    packed_posq[4 * ordered + 3] = valid ? charges[safe_atom] : 0.0f;
    packed_lj[2 * ordered + 0] = valid ? half_sigma[safe_atom] : 0.0f;
    packed_lj[2 * ordered + 1] = valid ? sqrt_epsilon[safe_atom] : 0.0f;
"""

_INTERACTION32_BLOCK_GEOMETRY_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    uint atom_count = (uint)counts[1];
    if (block >= block_count) {
        return;
    }

    uint ordered = 32u * block + lane;
    int atom = atom_order[ordered];
    bool valid = atom >= 0 && atom < (int)atom_count;
    int safe_atom = valid ? atom : 0;
    int reference_atom = atom_order[32u * block];
    float3 reference = float3(
        positions[3 * reference_atom + 0],
        positions[3 * reference_atom + 1],
        positions[3 * reference_atom + 2]
    );
    float3 value = float3(
        positions[3 * safe_atom + 0],
        positions[3 * safe_atom + 1],
        positions[3 * safe_atom + 2]
    );
    float3 delta = value - reference;
    delta.x -= box[0] * rint(delta.x * box[3]);
    delta.y -= box[1] * rint(delta.y * box[4]);
    delta.z -= box[2] * rint(delta.z * box[5]);
    float3 unwrapped = reference + delta;
    float weight = valid ? 1.0f : 0.0f;
    float valid_count = simd_sum(weight);
    float3 center = float3(
        simd_sum(valid ? unwrapped.x : 0.0f),
        simd_sum(valid ? unwrapped.y : 0.0f),
        simd_sum(valid ? unwrapped.z : 0.0f)
    ) / valid_count;
    float3 centered = valid ? unwrapped - center : float3(0.0f);
    float3 extent = float3(
        simd_max(abs(centered.x)),
        simd_max(abs(centered.y)),
        simd_max(abs(centered.z))
    );
    float radius2 = simd_max(dot(centered, centered));
    if (lane == 0u) {
        center.x -= box[0] * floor(center.x * box[3]);
        center.y -= box[1] * floor(center.y * box[4]);
        center.z -= box[2] * floor(center.z * box[5]);
        center_radius[4 * block + 0] = center.x;
        center_radius[4 * block + 1] = center.y;
        center_radius[4 * block + 2] = center.z;
        center_radius[4 * block + 3] = sqrt(radius2);
        half_extent[3 * block + 0] = extent.x;
        half_extent[3 * block + 1] = extent.y;
        half_extent[3 * block + 2] = extent.z;
    }
"""

_INTERACTION32_ORDINARY_COMMON_SOURCE = r"""
inline bool mlx_atomistic_interaction32_blocks_may_interact(
    uint left_block,
    uint right_block,
    device const float* center_radius,
    device const float* half_extent,
    thread const float* box,
    float search_radius,
    float search_radius2
) {
    float3 delta = float3(
        center_radius[4 * right_block + 0]
            - center_radius[4 * left_block + 0],
        center_radius[4 * right_block + 1]
            - center_radius[4 * left_block + 1],
        center_radius[4 * right_block + 2]
            - center_radius[4 * left_block + 2]
    );
    delta.x -= box[0] * rint(delta.x * box[3]);
    delta.y -= box[1] * rint(delta.y * box[4]);
    delta.z -= box[2] * rint(delta.z * box[5]);
    float sphere_limit = search_radius
        + center_radius[4 * left_block + 3]
        + center_radius[4 * right_block + 3];
    if (dot(delta, delta) >= sphere_limit * sphere_limit) {
        return false;
    }
    float3 separated = max(
        abs(delta)
            - float3(
                half_extent[3 * left_block + 0]
                    + half_extent[3 * right_block + 0],
                half_extent[3 * left_block + 1]
                    + half_extent[3 * right_block + 1],
                half_extent[3 * left_block + 2]
                    + half_extent[3 * right_block + 2]
            ),
        float3(0.0f)
    );
    return dot(separated, separated) < search_radius2;
}

inline uint mlx_atomistic_interaction32_half_mode(
    int right_atom,
    device const float* positions,
    threadgroup const int* left_atoms,
    threadgroup const float* left_positions,
    thread const float* box,
    float search_radius2
) {
    if (right_atom < 0) {
        return 0u;
    }
    float3 right = float3(
        positions[3 * right_atom + 0],
        positions[3 * right_atom + 1],
        positions[3 * right_atom + 2]
    );
    bool first = false;
    bool second = false;
    for (uint left_slot = 0u; left_slot < 32u; left_slot++) {
        if (left_atoms[left_slot] < 0) {
            continue;
        }
        float3 delta = float3(
            left_positions[3 * left_slot + 0] - right.x,
            left_positions[3 * left_slot + 1] - right.y,
            left_positions[3 * left_slot + 2] - right.z
        );
        delta.x -= box[0] * rint(delta.x * box[3]);
        delta.y -= box[1] * rint(delta.y * box[4]);
        delta.z -= box[2] * rint(delta.z * box[5]);
        bool member = dot(delta, delta) < search_radius2;
        first = first || (member && left_slot < 16u);
        second = second || (member && left_slot >= 16u);
    }
    return (first ? 1u : 0u) | (second ? 2u : 0u);
}
"""

_INTERACTION32_ORDINARY_COUNT_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint traversal_index = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    if (traversal_index >= block_count) {
        return;
    }
    uint left_block = (uint)block_traversal[traversal_index];
    threadgroup int left_atoms[32];
    threadgroup float left_positions[96];
    int left_atom = atom_order[32u * left_block + lane];
    int safe_left = max(left_atom, 0);
    left_atoms[lane] = left_atom;
    left_positions[3u * lane + 0u] = positions[3 * safe_left + 0];
    left_positions[3u * lane + 1u] = positions[3 * safe_left + 1];
    left_positions[3u * lane + 2u] = positions[3 * safe_left + 2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_box[6];
    for (uint axis = 0u; axis < 6u; axis++) {
        local_box[axis] = box[axis];
    }
    float search_radius = params[0];
    float search_radius2 = params[1];
    uint mode1_count = 0u;
    uint mode2_count = 0u;
    uint mode3_count = 0u;
#ifdef MLX_ATOMISTIC_INTERACTION32_RETAIN_MODES
    uint pair_row_start = traversal_index
        * (2u * block_count - traversal_index - 1u) / 2u;
#endif
    for (
        uint right_index = traversal_index + 1u;
        right_index < block_count;
        right_index++
    ) {
#ifdef MLX_ATOMISTIC_INTERACTION32_RETAIN_MODES
        uint pair_index = pair_row_start + right_index - traversal_index - 1u;
#endif
        uint right_block = (uint)block_traversal[right_index];
        uint low_block = min(left_block, right_block);
        uint high_block = max(left_block, right_block);
        int block_code = (int)(low_block * block_count + high_block);
        bool include_lane = false;
        if (lane == 0u) {
            uint special_word = special_pair_words[(uint)block_code >> 5u];
            bool special = (
                (special_word >> ((uint)block_code & 31u)) & 1u
            ) != 0u;
            if (!special) {
                include_lane = mlx_atomistic_interaction32_blocks_may_interact(
                    left_block,
                    right_block,
                    center_radius,
                    half_extent,
                    local_box,
                    search_radius,
                    search_radius2
                );
            }
        }
        bool include = simd_broadcast(include_lane, 0u);
        if (!include) {
            continue;
        }
        int right_ordered = 32 * (int)right_block + (int)lane;
        int right_atom = atom_order[right_ordered];
        uint mode = mlx_atomistic_interaction32_half_mode(
            right_atom,
            positions,
            left_atoms,
            left_positions,
            local_box,
            search_radius2
        );
        // One 32-lane count fits in six bits, so one collective carries all modes.
        uint packed_count = simd_sum(
            mode == 1u ? 1u : (mode == 2u ? 64u : (mode == 3u ? 4096u : 0u))
        );
        mode1_count += packed_count & 63u;
        mode2_count += (packed_count >> 6u) & 63u;
        mode3_count += (packed_count >> 12u) & 63u;
#ifdef MLX_ATOMISTIC_INTERACTION32_RETAIN_MODES
        // Each lane owns a disjoint two-bit field, making these sums exact packs.
        uint mode_shift = 2u * (lane & 15u);
        uint packed_low = simd_sum(lane < 16u ? mode << mode_shift : 0u);
        uint packed_high = simd_sum(lane >= 16u ? mode << mode_shift : 0u);
        if (lane == 0u) {
            mode_words[2u * pair_index + 0u] = packed_low;
            mode_words[2u * pair_index + 1u] = packed_high;
        }
#endif
    }
    if (lane == 0u) {
        mode_counts[3u * traversal_index + 0u] = (int)mode1_count;
        mode_counts[3u * traversal_index + 1u] = (int)mode2_count;
        mode_counts[3u * traversal_index + 2u] = (int)mode3_count;
    }
"""

_INTERACTION32_ORDINARY_CACHED_SCATTER_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint traversal_index = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    if (traversal_index >= block_count) {
        return;
    }
    uint left_block = (uint)block_traversal[traversal_index];
    uint run_base = 3u * traversal_index;
    if (lane == 0u) {
        for (uint mode_slot = 0u; mode_slot < 3u; mode_slot++) {
            uint run = run_base + mode_slot;
            int tile_count = mode_tile_counts[run];
            int tile_start = mode_tile_prefix[run] - tile_count;
            for (int local_tile = 0; local_tile < tile_count; local_tile++) {
                int tile = tile_start + local_tile;
                ordinary_left_blocks[tile] = (int)left_block;
                ordinary_half_modes[tile] = (int)mode_slot + 1;
            }
        }
    }

    uint seen1 = 0u;
    uint seen2 = 0u;
    uint seen3 = 0u;
    uint pair_row_start = traversal_index
        * (2u * block_count - traversal_index - 1u) / 2u;
    for (
        uint right_index = traversal_index + 1u;
        right_index < block_count;
        right_index++
    ) {
        uint pair_index = pair_row_start + right_index - traversal_index - 1u;
        uint right_block = (uint)block_traversal[right_index];
        int right_ordered = 32 * (int)right_block + (int)lane;
        uint packed_modes = mode_words[2u * pair_index + (lane >> 4u)];
        uint mode = (packed_modes >> (2u * (lane & 15u))) & 3u;
        uint is1 = mode == 1u ? 1u : 0u;
        uint is2 = mode == 2u ? 1u : 0u;
        uint is3 = mode == 3u ? 1u : 0u;
        // Six-bit fields keep all three ranks and counts in one collective.
        uint packed_flags = is1 | (is2 << 6u) | (is3 << 12u);
        uint packed_rank = simd_prefix_exclusive_sum(packed_flags);
        uint packed_count = simd_sum(packed_flags);
        uint rank1 = packed_rank & 63u;
        uint rank2 = (packed_rank >> 6u) & 63u;
        uint rank3 = (packed_rank >> 12u) & 63u;
        uint count1 = packed_count & 63u;
        uint count2 = (packed_count >> 6u) & 63u;
        uint count3 = (packed_count >> 12u) & 63u;
        uint local_entry = mode == 1u
            ? seen1 + rank1
            : (mode == 2u ? seen2 + rank2 : seen3 + rank3);
        if (mode != 0u) {
            uint run = run_base + mode - 1u;
            int tile_count = mode_tile_counts[run];
            int tile_start = mode_tile_prefix[run] - tile_count;
            uint tile = (uint)tile_start + local_entry / 32u;
            uint slot = local_entry & 31u;
            ordinary_right_atoms[32u * tile + slot] = right_ordered;
        }
        seen1 += count1;
        seen2 += count2;
        seen3 += count3;
    }
"""

_INTERACTION32_ORDINARY_SCATTER_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint traversal_index = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    if (traversal_index >= block_count) {
        return;
    }
    uint left_block = (uint)block_traversal[traversal_index];
    threadgroup int left_atoms[32];
    threadgroup float left_positions[96];
    int left_atom = atom_order[32u * left_block + lane];
    int safe_left = max(left_atom, 0);
    left_atoms[lane] = left_atom;
    left_positions[3u * lane + 0u] = positions[3 * safe_left + 0];
    left_positions[3u * lane + 1u] = positions[3 * safe_left + 1];
    left_positions[3u * lane + 2u] = positions[3 * safe_left + 2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_box[6];
    for (uint axis = 0u; axis < 6u; axis++) {
        local_box[axis] = box[axis];
    }
    float search_radius = params[0];
    float search_radius2 = params[1];
    uint run_base = 3u * traversal_index;
    if (lane == 0u) {
        for (uint mode_slot = 0u; mode_slot < 3u; mode_slot++) {
            uint run = run_base + mode_slot;
            int tile_count = mode_tile_counts[run];
            int tile_start = mode_tile_prefix[run] - tile_count;
            for (int local_tile = 0; local_tile < tile_count; local_tile++) {
                int tile = tile_start + local_tile;
                ordinary_left_blocks[tile] = (int)left_block;
                ordinary_half_modes[tile] = (int)mode_slot + 1;
            }
        }
    }

    uint seen1 = 0u;
    uint seen2 = 0u;
    uint seen3 = 0u;
    for (
        uint right_index = traversal_index + 1u;
        right_index < block_count;
        right_index++
    ) {
        uint right_block = (uint)block_traversal[right_index];
        uint low_block = min(left_block, right_block);
        uint high_block = max(left_block, right_block);
        int block_code = (int)(low_block * block_count + high_block);
        bool include_lane = false;
        if (lane == 0u) {
            uint special_word = special_pair_words[(uint)block_code >> 5u];
            bool special = (
                (special_word >> ((uint)block_code & 31u)) & 1u
            ) != 0u;
            if (!special) {
                include_lane = mlx_atomistic_interaction32_blocks_may_interact(
                    left_block,
                    right_block,
                    center_radius,
                    half_extent,
                    local_box,
                    search_radius,
                    search_radius2
                );
            }
        }
        bool include = simd_broadcast(include_lane, 0u);
        if (!include) {
            continue;
        }
        int right_ordered = 32 * (int)right_block + (int)lane;
        int right_atom = atom_order[right_ordered];
        uint mode = mlx_atomistic_interaction32_half_mode(
            right_atom,
            positions,
            left_atoms,
            left_positions,
            local_box,
            search_radius2
        );
        uint is1 = mode == 1u ? 1u : 0u;
        uint is2 = mode == 2u ? 1u : 0u;
        uint is3 = mode == 3u ? 1u : 0u;
        // Six-bit fields keep all three ranks and counts in one collective.
        uint packed_flags = is1 | (is2 << 6u) | (is3 << 12u);
        uint packed_rank = simd_prefix_exclusive_sum(packed_flags);
        uint packed_count = simd_sum(packed_flags);
        uint rank1 = packed_rank & 63u;
        uint rank2 = (packed_rank >> 6u) & 63u;
        uint rank3 = (packed_rank >> 12u) & 63u;
        uint count1 = packed_count & 63u;
        uint count2 = (packed_count >> 6u) & 63u;
        uint count3 = (packed_count >> 12u) & 63u;
        uint local_entry = mode == 1u
            ? seen1 + rank1
            : (mode == 2u ? seen2 + rank2 : seen3 + rank3);
        if (mode != 0u) {
            uint run = run_base + mode - 1u;
            int tile_count = mode_tile_counts[run];
            int tile_start = mode_tile_prefix[run] - tile_count;
            uint tile = (uint)tile_start + local_entry / 32u;
            uint slot = local_entry & 31u;
            ordinary_right_atoms[32u * tile + slot] = right_ordered;
        }
        seen1 += count1;
        seen2 += count2;
        seen3 += count3;
    }
"""

_INTERACTION32_OUTER_INNER_MODE_COUNT_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint traversal_index = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    uint padded_count = (uint)counts[1];
    if (traversal_index >= block_count) {
        return;
    }
    uint left_block = (uint)block_traversal[traversal_index];
    threadgroup int left_atoms[32];
    threadgroup float left_positions[96];
    int left_atom = atom_order[32u * left_block + lane];
    int safe_left = max(left_atom, 0);
    left_atoms[lane] = left_atom;
    left_positions[3u * lane + 0u] = positions[3 * safe_left + 0];
    left_positions[3u * lane + 1u] = positions[3 * safe_left + 1];
    left_positions[3u * lane + 2u] = positions[3 * safe_left + 2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_box[6];
    for (uint axis = 0u; axis < 6u; axis++) {
        local_box[axis] = box[axis];
    }
    uint counts_by_mode[3] = {0u, 0u, 0u};
    uint run_base = 3u * traversal_index;
    for (uint outer_mode = 0u; outer_mode < 3u; outer_mode++) {
        uint run = run_base + outer_mode;
        int tile_count = outer_tile_counts[run];
        int tile_start = outer_tile_prefix[run] - tile_count;
        for (int local_tile = 0; local_tile < tile_count; local_tile++) {
            uint tile = (uint)(tile_start + local_tile);
            int right_ordered = outer_right_atoms[32u * tile + lane];
            int right_atom = right_ordered >= 0 && right_ordered < (int)padded_count
                ? atom_order[right_ordered]
                : -1;
            uint mode = mlx_atomistic_interaction32_half_mode(
                right_atom,
                positions,
                left_atoms,
                left_positions,
                local_box,
                params[0]
            );
            cached_modes[32u * tile + lane] = mode;
            // One 32-lane count fits in six bits, so one collective carries all modes.
            uint packed_count = simd_sum(
                mode == 1u
                    ? 1u
                    : (mode == 2u ? 64u : (mode == 3u ? 4096u : 0u))
            );
            counts_by_mode[0] += packed_count & 63u;
            counts_by_mode[1] += (packed_count >> 6u) & 63u;
            counts_by_mode[2] += (packed_count >> 12u) & 63u;
        }
    }
    if (lane == 0u) {
        mode_counts[run_base + 0u] = (int)counts_by_mode[0];
        mode_counts[run_base + 1u] = (int)counts_by_mode[1];
        mode_counts[run_base + 2u] = (int)counts_by_mode[2];
    }
"""

_INTERACTION32_OUTER_INNER_MODE_SCATTER_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint traversal_index = threadgroup_position_in_grid.x;
    uint block_count = (uint)counts[0];
    if (traversal_index >= block_count) {
        return;
    }
    uint left_block = (uint)block_traversal[traversal_index];
    uint run_base = 3u * traversal_index;
    if (lane == 0u) {
        for (uint mode_slot = 0u; mode_slot < 3u; mode_slot++) {
            uint run = run_base + mode_slot;
            int tile_count = inner_tile_counts[run];
            int tile_start = inner_tile_prefix[run] - tile_count;
            for (int local_tile = 0; local_tile < tile_count; local_tile++) {
                int tile = tile_start + local_tile;
                inner_left_blocks[tile] = (int)left_block;
                inner_half_modes[tile] = (int)mode_slot + 1;
            }
        }
    }

    uint seen1 = 0u;
    uint seen2 = 0u;
    uint seen3 = 0u;
    for (uint outer_mode = 0u; outer_mode < 3u; outer_mode++) {
        uint outer_run = run_base + outer_mode;
        int outer_tile_count = outer_tile_counts[outer_run];
        int outer_tile_start = outer_tile_prefix[outer_run] - outer_tile_count;
        for (int local_tile = 0; local_tile < outer_tile_count; local_tile++) {
            uint tile = (uint)(outer_tile_start + local_tile);
            int right_ordered = outer_right_atoms[32u * tile + lane];
            uint mode = cached_modes[32u * tile + lane];
            uint is1 = mode == 1u ? 1u : 0u;
            uint is2 = mode == 2u ? 1u : 0u;
            uint is3 = mode == 3u ? 1u : 0u;
            // Six-bit fields keep all three ranks and counts in one collective.
            uint packed_flags = is1 | (is2 << 6u) | (is3 << 12u);
            uint packed_rank = simd_prefix_exclusive_sum(packed_flags);
            uint packed_count = simd_sum(packed_flags);
            uint rank1 = packed_rank & 63u;
            uint rank2 = (packed_rank >> 6u) & 63u;
            uint rank3 = (packed_rank >> 12u) & 63u;
            uint count1 = packed_count & 63u;
            uint count2 = (packed_count >> 6u) & 63u;
            uint count3 = (packed_count >> 12u) & 63u;
            uint local_entry = mode == 1u
                ? seen1 + rank1
                : (mode == 2u ? seen2 + rank2 : seen3 + rank3);
            if (mode != 0u) {
                uint inner_run = run_base + mode - 1u;
                int inner_tile_count = inner_tile_counts[inner_run];
                int inner_tile_start = inner_tile_prefix[inner_run] - inner_tile_count;
                uint output_tile = (uint)inner_tile_start + local_entry / 32u;
                uint output_slot = local_entry & 31u;
                inner_right_atoms[32u * output_tile + output_slot] = right_ordered;
            }
            seen1 += count1;
            seen2 += count2;
            seen3 += count3;
        }
    }
"""

_INTERACTION32_SPECIAL_PAIR_WORDS_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= (uint)counts[0] || special_unique[index] == 0) {
        return;
    }
    uint code = (uint)special_codes[index];
    atomic_fetch_or_explicit(
        &special_pair_words[code >> 5u],
        1u << (code & 31u),
        memory_order_relaxed
    );
"""

_INTERACTION32_SPECIAL_BLOCK_SCATTER_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint raw_code_count = (uint)counts[0];
    uint block_count = (uint)counts[1];
    if (index >= raw_code_count || special_unique[index] == 0) {
        return;
    }
    uint code = (uint)special_codes[index];
    int output = special_prefix[index] - 1;
    special_blocks[2 * output + 0] = (int)(code / block_count);
    special_blocks[2 * output + 1] = (int)(code % block_count);
"""

_INTERACTION32_SPECIAL_WORK_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint tile_count = (uint)counts[0];
    uint padded_count = (uint)counts[1];
    if (tile >= tile_count) {
        return;
    }

    int left_block = special_blocks[2u * tile + 0u];
    int right_block = special_blocks[2u * tile + 1u];
    uint left_ordered = 32u * (uint)left_block + lane;
    uint right_ordered = 32u * (uint)right_block + lane;
    threadgroup int left_atoms[32];
    left_atoms[lane] = atom_order[left_ordered];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int right_atom = atom_order[right_ordered];
    uint enabled_word = 0u;
    uint one_four_word = 0u;
    if (right_atom >= 0) {
        for (uint left_slot = 0u; left_slot < 32u; left_slot++) {
            int left_atom = left_atoms[left_slot];
            if (left_atom < 0 || left_atom == right_atom) {
                continue;
            }
            int topology_class = -1;
            int start = topology_offsets[left_atom];
            int stop = topology_offsets[left_atom + 1];
            for (int index = start; index < stop; index++) {
                if (topology_neighbors[index] == right_atom) {
                    topology_class = topology_classes[index];
                    break;
                }
            }
            if (topology_class != 0) {
                enabled_word |= 1u << left_slot;
            }
            if (topology_class == 1) {
                one_four_word |= 1u << left_slot;
            }
        }
    }

    uint first_work = 2u * tile;
    uint second_work = first_work + 1u;
    int stored_right = right_atom >= 0 ? (int)right_ordered : (int)padded_count;
    special_right_atoms[32u * first_work + lane] = stored_right;
    special_right_atoms[32u * second_work + lane] = stored_right;
    special_lj_enabled[32u * first_work + lane] = enabled_word;
    special_lj_enabled[32u * second_work + lane] = enabled_word;
    special_lj_one_four[32u * first_work + lane] = one_four_word;
    special_lj_one_four[32u * second_work + lane] = one_four_word;
    if (lane == 0u) {
        int diagonal = left_block == right_block ? 1 : 0;
        special_left_blocks[first_work] = left_block;
        special_left_blocks[second_work] = left_block;
        special_left_slices[first_work] = 0;
        special_left_slices[second_work] = 1;
        special_diagonal[first_work] = diagonal;
        special_diagonal[second_work] = diagonal;
    }
"""

_INTERACTION32_FORCE_SOURCE = r"""
    uint tg_thread = thread_position_in_threadgroup.x;
    uint lane = tg_thread & 31u;
    uint sub = tg_thread >> 5u;
    uint dispatch_groups = (uint)counts[4];
    uint global_work = threadgroup_position_in_grid.x * dispatch_groups + sub;
    uint ordinary_work_count = (uint)counts[0];
    uint special_work_count = (uint)counts[1];
    uint total_work_count = ordinary_work_count + special_work_count;
    bool in_ordinary = global_work < ordinary_work_count;
    bool in_special = global_work >= ordinary_work_count
        && global_work < total_work_count;
    bool special = in_special || (ordinary_work_count == 0u && !in_ordinary);
    uint work = in_special ? global_work - ordinary_work_count : global_work;
    bool work_active = threads_per_simdgroup == 32u
        && global_work < total_work_count;
    uint safe_work = work_active ? work : 0u;
    uint group = safe_work;

    int ordinary_start = special ? 0 : ordinary_group_starts[group];
#ifdef MLX_ATOMISTIC_INTERACTION32_FUSED_HALF
    int ordinary_half_mode = special ? 0 : ordinary_half_modes[ordinary_start];
    uint left_slice_size = special
        ? (uint)counts[5]
        : (ordinary_half_mode == 3 ? 32u : 16u);
    uint left_slice = special
        ? (uint)special_left_slices[group]
        : (ordinary_half_mode == 2 ? 1u : 0u);
#else
    uint left_slice_size = (uint)counts[5];
    uint left_slice = special
        ? (uint)special_left_slices[group]
        : (uint)ordinary_left_slices[ordinary_start];
#endif
    uint interaction_count = special
        ? 1u
        : (uint)ordinary_group_counts[group];
    int left_block = special
        ? special_left_blocks[group]
        : ordinary_left_blocks[ordinary_start];
#ifdef MLX_ATOMISTIC_INTERACTION32_FUSED_HALF
    threadgroup int left_ordered_buffer[32 * GROUPS_PER_TG];
    threadgroup int left_atom_buffer[32 * GROUPS_PER_TG];
    threadgroup uint left_valid_buffer[32 * GROUPS_PER_TG];
    threadgroup float left_posq_buffer[128 * GROUPS_PER_TG];
    threadgroup float left_lj_buffer[64 * GROUPS_PER_TG];
    uint left_base = 32u * sub;
    uint posq_base = 128u * sub;
    uint lj_base = 64u * sub;
#else
    threadgroup int left_ordered_buffer[16 * GROUPS_PER_TG];
    threadgroup int left_atom_buffer[16 * GROUPS_PER_TG];
    threadgroup uint left_valid_buffer[16 * GROUPS_PER_TG];
    threadgroup float left_posq_buffer[64 * GROUPS_PER_TG];
    threadgroup float left_lj_buffer[32 * GROUPS_PER_TG];
    uint left_base = 16u * sub;
    uint posq_base = 64u * sub;
    uint lj_base = 32u * sub;
#endif
#ifdef MLX_ATOMISTIC_NBFIX
    threadgroup int left_type_buffer[32 * GROUPS_PER_TG];
#endif
#ifdef MLX_ATOMISTIC_INTERACTION32_ACTIVE_COMPACTION
    threadgroup int active_right_atom_buffer[32 * GROUPS_PER_TG];
    threadgroup float active_right_posq_buffer[128 * GROUPS_PER_TG];
    threadgroup float active_right_lj_buffer[64 * GROUPS_PER_TG];
#ifdef MLX_ATOMISTIC_NBFIX
    threadgroup int active_right_type_buffer[32 * GROUPS_PER_TG];
#endif
#endif
    if (lane < left_slice_size) {
        int left_ordered = 32 * left_block
            + (int)left_slice_size * (int)left_slice + (int)lane;
        bool left_valid = work_active
            && left_ordered < counts[2]
            && atom_order[left_ordered] >= 0;
        int left_atom = left_valid ? atom_order[left_ordered] : 0;
        left_ordered_buffer[left_base + lane] = left_ordered;
        left_atom_buffer[left_base + lane] = left_atom;
        left_valid_buffer[left_base + lane] = left_valid ? 1u : 0u;
#ifdef MLX_ATOMISTIC_INTERACTION32_CANONICAL
        left_posq_buffer[posq_base + 4u * lane + 0u] =
            positions[3 * left_atom + 0];
        left_posq_buffer[posq_base + 4u * lane + 1u] =
            positions[3 * left_atom + 1];
        left_posq_buffer[posq_base + 4u * lane + 2u] =
            positions[3 * left_atom + 2];
        left_posq_buffer[posq_base + 4u * lane + 3u] = charges[left_atom];
        left_lj_buffer[lj_base + 2u * lane + 0u] = half_sigma[left_atom];
        left_lj_buffer[lj_base + 2u * lane + 1u] = sqrt_epsilon[left_atom];
#ifdef MLX_ATOMISTIC_NBFIX
        left_type_buffer[left_base + lane] = atom_type_ids[left_atom];
#endif
#else
        left_posq_buffer[posq_base + 4u * lane + 0u] =
            packed_posq[4 * left_ordered + 0];
        left_posq_buffer[posq_base + 4u * lane + 1u] =
            packed_posq[4 * left_ordered + 1];
        left_posq_buffer[posq_base + 4u * lane + 2u] =
            packed_posq[4 * left_ordered + 2];
        left_posq_buffer[posq_base + 4u * lane + 3u] =
            packed_posq[4 * left_ordered + 3];
        left_lj_buffer[lj_base + 2u * lane + 0u] =
            packed_lj[2 * left_ordered + 0];
        left_lj_buffer[lj_base + 2u * lane + 1u] =
            packed_lj[2 * left_ordered + 1];
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float box_lx = box[0];
    float box_ly = box[1];
    float box_lz = box[2];
    float box_ix = box[3];
    float box_iy = box[4];
    float box_iz = box[5];
    float cutoff2 = params[0];
    float shift_flag = params[1];
    float switch_flag = params[2];
    float switch_start = params[3];
    float switch_end = params[5];
    float coulomb = params[6];
    float ewald_alpha = params[7];
    float ewald_self = params[8];
    float inv_switch_width = params[9];
    float one_four_scale = params[10];
#ifdef MLX_ATOMISTIC_NBFIX
    int nbfix_type_count = (int)params[11];
#endif
    float3 owned_left_force = float3(0.0f);
#ifdef MLX_ATOMISTIC_INTERACTION32_ACTIVE_COMPACTION
    if (!special) {
        // The Verlet schedule intentionally retains a wide shell.  Cull right
        // atoms against the current left-slice AABB, compact the survivors
        // inside this SIMD group, and transpose the pair loop so the number of
        // column reductions follows useful right work instead of padded width.
        uint safe_left_lane = min(lane, left_slice_size - 1u);
        bool owned_left_valid = work_active
            && lane < left_slice_size
            && left_valid_buffer[left_base + safe_left_lane] != 0u;
        int owned_left_atom = left_atom_buffer[left_base + safe_left_lane];
        float4 owned_left_posq = float4(
            left_posq_buffer[posq_base + 4u * safe_left_lane + 0u],
            left_posq_buffer[posq_base + 4u * safe_left_lane + 1u],
            left_posq_buffer[posq_base + 4u * safe_left_lane + 2u],
            left_posq_buffer[posq_base + 4u * safe_left_lane + 3u]
        );
        float2 owned_left_lj = float2(
            left_lj_buffer[lj_base + 2u * safe_left_lane + 0u],
            left_lj_buffer[lj_base + 2u * safe_left_lane + 1u]
        );
#ifdef MLX_ATOMISTIC_NBFIX
        int owned_left_type = left_type_buffer[left_base + safe_left_lane];
#endif
        float3 left_reference = float3(
            simd_broadcast(left_posq_buffer[posq_base + 0u], 0u),
            simd_broadcast(left_posq_buffer[posq_base + 1u], 0u),
            simd_broadcast(left_posq_buffer[posq_base + 2u], 0u)
        );
        float3 left_delta = owned_left_posq.xyz - left_reference;
        left_delta.x -= box_lx * rint(left_delta.x * box_ix);
        left_delta.y -= box_ly * rint(left_delta.y * box_iy);
        left_delta.z -= box_lz * rint(left_delta.z * box_iz);
        float3 left_unwrapped = left_reference + left_delta;
        float3 left_minimum = float3(
            simd_min(owned_left_valid ? left_unwrapped.x : 1.0e20f),
            simd_min(owned_left_valid ? left_unwrapped.y : 1.0e20f),
            simd_min(owned_left_valid ? left_unwrapped.z : 1.0e20f)
        );
        float3 left_maximum = float3(
            simd_max(owned_left_valid ? left_unwrapped.x : -1.0e20f),
            simd_max(owned_left_valid ? left_unwrapped.y : -1.0e20f),
            simd_max(owned_left_valid ? left_unwrapped.z : -1.0e20f)
        );

        for (uint interaction = 0u; interaction < interaction_count; interaction++) {
            uint tile = (uint)ordinary_start + interaction;
            int owned_right_ordered = ordinary_right_atoms[32 * tile + lane];
            bool owned_right_valid = work_active
                && owned_right_ordered >= 0
                && owned_right_ordered < counts[2]
                && atom_order[owned_right_ordered] >= 0;
            int owned_right_atom = owned_right_valid
                ? atom_order[owned_right_ordered]
                : 0;
            float4 owned_right_posq = float4(
                positions[3 * owned_right_atom + 0],
                positions[3 * owned_right_atom + 1],
                positions[3 * owned_right_atom + 2],
                charges[owned_right_atom]
            );
            float2 owned_right_lj = float2(
                half_sigma[owned_right_atom],
                sqrt_epsilon[owned_right_atom]
            );
#ifdef MLX_ATOMISTIC_NBFIX
            int owned_right_type = atom_type_ids[owned_right_atom];
#endif
            float3 right_delta = owned_right_posq.xyz - left_reference;
            right_delta.x -= box_lx * rint(right_delta.x * box_ix);
            right_delta.y -= box_ly * rint(right_delta.y * box_iy);
            right_delta.z -= box_lz * rint(right_delta.z * box_iz);
            float3 right_unwrapped = left_reference + right_delta;
            float3 aabb_gap = max(
                max(left_minimum - right_unwrapped, right_unwrapped - left_maximum),
                float3(0.0f)
            );
            uint right_active = (
                owned_right_valid && dot(aabb_gap, aabb_gap) < cutoff2
            ) ? 1u : 0u;
            uint active_rank = simd_prefix_exclusive_sum(right_active);
            uint active_count = simd_sum(right_active);
            if (right_active != 0u) {
                active_right_atom_buffer[left_base + active_rank] = owned_right_atom;
                active_right_posq_buffer[posq_base + 4u * active_rank + 0u] =
                    owned_right_posq.x;
                active_right_posq_buffer[posq_base + 4u * active_rank + 1u] =
                    owned_right_posq.y;
                active_right_posq_buffer[posq_base + 4u * active_rank + 2u] =
                    owned_right_posq.z;
                active_right_posq_buffer[posq_base + 4u * active_rank + 3u] =
                    owned_right_posq.w;
                active_right_lj_buffer[lj_base + 2u * active_rank + 0u] =
                    owned_right_lj.x;
                active_right_lj_buffer[lj_base + 2u * active_rank + 1u] =
                    owned_right_lj.y;
#ifdef MLX_ATOMISTIC_NBFIX
                active_right_type_buffer[left_base + active_rank] = owned_right_type;
#endif
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint right_slot = 0u; right_slot < active_count; right_slot++) {
                int right_atom = active_right_atom_buffer[left_base + right_slot];
                float4 right_posq = float4(
                    active_right_posq_buffer[posq_base + 4u * right_slot + 0u],
                    active_right_posq_buffer[posq_base + 4u * right_slot + 1u],
                    active_right_posq_buffer[posq_base + 4u * right_slot + 2u],
                    active_right_posq_buffer[posq_base + 4u * right_slot + 3u]
                );
                float2 right_lj = float2(
                    active_right_lj_buffer[lj_base + 2u * right_slot + 0u],
                    active_right_lj_buffer[lj_base + 2u * right_slot + 1u]
                );
#ifdef MLX_ATOMISTIC_NBFIX
                int right_type = active_right_type_buffer[left_base + right_slot];
#endif
                float3 pair_force = float3(0.0f);
                if (owned_left_valid) {
                    float dx = owned_left_posq.x - right_posq.x;
                    float dy = owned_left_posq.y - right_posq.y;
                    float dz = owned_left_posq.z - right_posq.z;
                    dx -= box_lx * rint(dx * box_ix);
                    dy -= box_ly * rint(dy * box_iy);
                    dz -= box_lz * rint(dz * box_iz);
                    float r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 > 0.0f && r2 < cutoff2) {
                        float inv_distance = rsqrt(r2);
                        float inv_r2 = inv_distance * inv_distance;
                        float distance = r2 * inv_distance;
                        float sigma_ij = owned_left_lj.x + right_lj.x;
                        float epsilon_ij = owned_left_lj.y * right_lj.y;
#ifdef MLX_ATOMISTIC_NBFIX
                        int nbfix_index = owned_left_type * nbfix_type_count + right_type;
                        float nbfix_sigma_value = nbfix_sigma[nbfix_index];
                        if (nbfix_sigma_value > 0.0f) {
                            sigma_ij = nbfix_sigma_value;
                            epsilon_ij = nbfix_epsilon[nbfix_index];
                        }
#endif
                        float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                        float inv_r6 = sigma2_over_r2
                            * sigma2_over_r2 * sigma2_over_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        float unswitched_energy = 0.0f;
                        float switch_value = 1.0f;
                        float switch_derivative = 0.0f;
                        if (switch_flag > 0.5f) {
                            unswitched_energy = 4.0f * epsilon_ij
                                * (inv_r12 - inv_r6);
                            if (shift_flag > 0.5f) {
                                float sigma2_over_rc2 = sigma_ij * sigma_ij / cutoff2;
                                float inv_rc6 = sigma2_over_rc2
                                    * sigma2_over_rc2 * sigma2_over_rc2;
                                unswitched_energy -= 4.0f * epsilon_ij
                                    * (inv_rc6 * inv_rc6 - inv_rc6);
                            }
                            float x = clamp(
                                (distance - switch_start) * inv_switch_width,
                                0.0f,
                                1.0f
                            );
                            float x2 = x * x;
                            float x3 = x2 * x;
                            float x4 = x3 * x;
                            float x5 = x4 * x;
                            switch_value = 1.0f
                                - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                            if (distance > switch_start && distance < switch_end) {
                                switch_derivative = -(
                                    30.0f * x2 - 60.0f * x3 + 30.0f * x4
                                ) * inv_switch_width;
                            }
                        }
                        float scalar =
                            24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                                * inv_r2 * switch_value
                            - unswitched_energy * switch_derivative * inv_distance;
                        float qij = owned_left_posq.w * right_posq.w;
                        float2 ewald_terms =
                            mlx_atomistic_ewald_erfc_exp(ewald_alpha * distance);
                        scalar += coulomb * qij * (
                            ewald_terms.x * inv_r2 * inv_distance
                            + ewald_self * ewald_terms.y * inv_r2
                        );
                        pair_force = float3(
                            scalar * dx,
                            scalar * dy,
                            scalar * dz
                        );
                    }
                }
                owned_left_force += pair_force;
                float3 right_force = -float3(
                    simd_sum(pair_force.x),
                    simd_sum(pair_force.y),
                    simd_sum(pair_force.z)
                );
                if (lane == 0u && any(right_force != float3(0.0f))) {
                    atomic_fetch_add_explicit(
                        &ordered_forces[3 * right_atom + 0],
                        right_force.x,
                        memory_order_relaxed
                    );
                    atomic_fetch_add_explicit(
                        &ordered_forces[3 * right_atom + 1],
                        right_force.y,
                        memory_order_relaxed
                    );
                    atomic_fetch_add_explicit(
                        &ordered_forces[3 * right_atom + 2],
                        right_force.z,
                        memory_order_relaxed
                    );
                }
            }
        }
        if (owned_left_valid && any(owned_left_force != float3(0.0f))) {
            atomic_fetch_add_explicit(
                &ordered_forces[3 * owned_left_atom + 0],
                owned_left_force.x,
                memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &ordered_forces[3 * owned_left_atom + 1],
                owned_left_force.y,
                memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &ordered_forces[3 * owned_left_atom + 2],
                owned_left_force.z,
                memory_order_relaxed
            );
        }
        return;
    }
#endif
    for (uint interaction = 0u; interaction < interaction_count; interaction++) {
        uint tile = special ? group : (uint)ordinary_start + interaction;
        int owned_right = special
            ? special_right_atoms[32 * tile + lane]
            : ordinary_right_atoms[32 * tile + lane];
        bool owned_right_valid = work_active
            && owned_right >= 0
            && owned_right < counts[2]
            && atom_order[owned_right] >= 0;
        int safe_right = owned_right_valid ? owned_right : 0;
#ifdef MLX_ATOMISTIC_INTERACTION32_CANONICAL
        int right_atom = owned_right_valid ? atom_order[owned_right] : 0;
        float4 right_posq = float4(
            positions[3 * right_atom + 0],
            positions[3 * right_atom + 1],
            positions[3 * right_atom + 2],
            charges[right_atom]
        );
        float2 right_lj = float2(
            half_sigma[right_atom],
            sqrt_epsilon[right_atom]
        );
#ifdef MLX_ATOMISTIC_NBFIX
        int right_type_id = atom_type_ids[right_atom];
#endif
#else
        int right_atom = safe_right;
        float4 right_posq = float4(
            packed_posq[4 * safe_right + 0],
            packed_posq[4 * safe_right + 1],
            packed_posq[4 * safe_right + 2],
            packed_posq[4 * safe_right + 3]
        );
        float2 right_lj = float2(
            packed_lj[2 * safe_right + 0],
            packed_lj[2 * safe_right + 1]
        );
#endif
        uint lj_word = special ? special_work_lj_enabled[32 * tile + lane] : 0u;
        uint one_four_word = special
            ? special_work_lj_one_four[32 * tile + lane]
            : 0u;
        bool diagonal = special && special_diagonal[tile] != 0;

        float3 right_force = float3(0.0f);
        for (uint left_slot = 0u; left_slot < left_slice_size; left_slot++) {
            int left_ordered = left_ordered_buffer[left_base + left_slot];
            bool left_valid = left_valid_buffer[left_base + left_slot] != 0u;
            float4 left_posq = float4(
                left_posq_buffer[posq_base + 4u * left_slot + 0u],
                left_posq_buffer[posq_base + 4u * left_slot + 1u],
                left_posq_buffer[posq_base + 4u * left_slot + 2u],
                left_posq_buffer[posq_base + 4u * left_slot + 3u]
            );
            float2 left_lj = float2(
                left_lj_buffer[lj_base + 2u * left_slot + 0u],
                left_lj_buffer[lj_base + 2u * left_slot + 1u]
            );
            uint absolute_left_slot = left_slice_size * left_slice + left_slot;
            bool member = left_valid && owned_right_valid
                && (!diagonal || left_ordered < owned_right);
            bool lj_enabled = !special
                || ((lj_word >> absolute_left_slot) & 1u) != 0u;
            bool one_four = special
                && ((one_four_word >> absolute_left_slot) & 1u) != 0u;
            float3 pair_force = float3(0.0f);
            if (member) {
                float dx = left_posq.x - right_posq.x;
                float dy = left_posq.y - right_posq.y;
                float dz = left_posq.z - right_posq.z;
                dx -= box_lx * rint(dx * box_ix);
                dy -= box_ly * rint(dy * box_iy);
                dz -= box_lz * rint(dz * box_iz);
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > 0.0f && r2 < cutoff2) {
                    float inv_distance = rsqrt(r2);
                    float inv_r2 = inv_distance * inv_distance;
                    float distance = r2 * inv_distance;
                    float scalar = 0.0f;
                    if (lj_enabled) {
                        float sigma_ij = left_lj.x + right_lj.x;
                        float epsilon_ij = left_lj.y * right_lj.y;
#ifdef MLX_ATOMISTIC_NBFIX
                        int nbfix_index =
                            left_type_buffer[left_base + left_slot]
                                * nbfix_type_count
                            + right_type_id;
                        float nbfix_sigma_value = nbfix_sigma[nbfix_index];
                        if (nbfix_sigma_value > 0.0f) {
                            sigma_ij = nbfix_sigma_value;
                            epsilon_ij = nbfix_epsilon[nbfix_index];
                        }
#endif
                        float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                        float inv_r6 = sigma2_over_r2
                            * sigma2_over_r2 * sigma2_over_r2;
                        float inv_r12 = inv_r6 * inv_r6;
                        float unswitched_energy = 0.0f;
                        float switch_value = 1.0f;
                        float switch_derivative = 0.0f;
                        if (switch_flag > 0.5f) {
                            unswitched_energy =
                                4.0f * epsilon_ij * (inv_r12 - inv_r6);
                            if (shift_flag > 0.5f) {
                                float sigma2_over_rc2 =
                                    sigma_ij * sigma_ij / cutoff2;
                                float inv_rc6 = sigma2_over_rc2
                                    * sigma2_over_rc2 * sigma2_over_rc2;
                                unswitched_energy -= 4.0f * epsilon_ij
                                    * (inv_rc6 * inv_rc6 - inv_rc6);
                            }
                            float x = clamp(
                                (distance - switch_start) * inv_switch_width,
                                0.0f,
                                1.0f
                            );
                            float x2 = x * x;
                            float x3 = x2 * x;
                            float x4 = x3 * x;
                            float x5 = x4 * x;
                            switch_value = 1.0f
                                - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                            if (distance > switch_start && distance < switch_end) {
                                switch_derivative = -(
                                    30.0f * x2 - 60.0f * x3 + 30.0f * x4
                                ) * inv_switch_width;
                            }
                        }
                        float lj_scale = one_four ? one_four_scale : 1.0f;
                        scalar += (
                            24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                                * inv_r2 * switch_value
                            - unswitched_energy * switch_derivative * inv_distance
                        ) * lj_scale;
                    }

                    float qij = left_posq.w * right_posq.w;
                    float alpha_distance = ewald_alpha * distance;
                    float erfc_term;
                    float gaussian_term;
#ifdef MLX_ATOMISTIC_INTERACTION32_SHARED_EWALD_EXP
                    float2 ewald_terms =
                        mlx_atomistic_ewald_erfc_exp(alpha_distance);
                    erfc_term = ewald_terms.x;
                    gaussian_term = ewald_terms.y;
#else
                    erfc_term =
                        mlx_atomistic_erfc_nonnegative(alpha_distance);
                    gaussian_term = exp(-ewald_alpha * ewald_alpha * r2);
#endif
                    scalar += coulomb * qij * (
                        erfc_term * inv_r2 * inv_distance
                        + ewald_self * gaussian_term * inv_r2
                    );
                    pair_force = float3(scalar * dx, scalar * dy, scalar * dz);
                }
            }
            right_force -= pair_force;
            float3 reduced_left = float3(
                simd_sum(pair_force.x),
                simd_sum(pair_force.y),
                simd_sum(pair_force.z)
            );
            if (lane == left_slot) {
                owned_left_force += reduced_left;
            }
        }
        if (owned_right_valid && any(right_force != float3(0.0f))) {
            atomic_fetch_add_explicit(
                &ordered_forces[3 * right_atom + 0],
                right_force.x,
                memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &ordered_forces[3 * right_atom + 1],
                right_force.y,
                memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &ordered_forces[3 * right_atom + 2],
                right_force.z,
                memory_order_relaxed
            );
        }
    }
    int owned_left = left_ordered_buffer[
        left_base + min(lane, left_slice_size - 1u)
    ];
#ifdef MLX_ATOMISTIC_INTERACTION32_CANONICAL
    int left_force_atom = left_atom_buffer[
        left_base + min(lane, left_slice_size - 1u)
    ];
#else
    int left_force_atom = owned_left;
#endif
    bool owned_left_valid = work_active
        && lane < left_slice_size
        && left_valid_buffer[left_base + lane] != 0u;
    if (owned_left_valid && any(owned_left_force != float3(0.0f))) {
        atomic_fetch_add_explicit(
            &ordered_forces[3 * left_force_atom + 0],
            owned_left_force.x,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &ordered_forces[3 * left_force_atom + 1],
            owned_left_force.y,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &ordered_forces[3 * left_force_atom + 2],
            owned_left_force.z,
            memory_order_relaxed
        );
    }
""".replace("GROUPS_PER_TG", str(_INTERACTION32_GROUPS_PER_THREADGROUP))

_INTERACTION32_SCATTER_SOURCE = r"""
    uint ordered = thread_position_in_grid.x;
    if (ordered >= (uint)counts[0]) {
        return;
    }
    int atom = atom_order[ordered];
    if (atom < 0 || atom >= counts[1]) {
        return;
    }
    forces[3 * atom + 0] = ordered_forces[3 * ordered + 0];
    forces[3 * atom + 1] = ordered_forces[3 * ordered + 1];
    forces[3 * atom + 2] = ordered_forces[3 * ordered + 2];
"""

_OWNER_COMPUTE32_FORCE_SOURCE = r"""
    uint tg_thread = thread_position_in_threadgroup.x;
    uint lane = tg_thread & 31u;
    uint sub = tg_thread >> 5u;
    uint dispatch_groups = (uint)counts[3];
    uint candidate_block =
        threadgroup_position_in_grid.x * dispatch_groups + sub;
    bool block_active = threads_per_simdgroup == 32u
        && candidate_block < (uint)counts[0];
    uint block = block_active ? candidate_block : 0u;
    int owner_ordered = 32 * (int)block + (int)lane;
    bool owner_valid = block_active
        && owner_ordered < counts[1]
        && atom_order[owner_ordered] >= 0;
    int owner_atom = owner_valid ? atom_order[owner_ordered] : 0;
    float4 owner_posq = float4(
        positions[3 * owner_atom + 0],
        positions[3 * owner_atom + 1],
        positions[3 * owner_atom + 2],
        charges[owner_atom]
    );
    float2 owner_lj = float2(
        half_sigma[owner_atom],
        sqrt_epsilon[owner_atom]
    );

    float box_lx = box[0];
    float box_ly = box[1];
    float box_lz = box[2];
    float box_ix = box[3];
    float box_iy = box[4];
    float box_iz = box[5];
    float cutoff2 = params[0];
    float shift_flag = params[1];
    float switch_flag = params[2];
    float switch_start = params[3];
    float switch_end = params[5];
    float coulomb = params[6];
    float ewald_alpha = params[7];
    float ewald_self = params[8];
    float inv_switch_width = params[9];
    float one_four_scale = params[10];
    float3 owner_force = float3(0.0f);
    float3 owner_force_compensation = float3(0.0f);

    int right_start = owner_offsets[block];
    int right_stop = owner_offsets[block + 1u];
    for (int tile_start = right_start; tile_start < right_stop; tile_start += 32) {
        int owned_right_ordered = right_atoms[tile_start + (int)lane];
        bool owned_right_valid = owned_right_ordered >= 0
            && owned_right_ordered < counts[1]
            && atom_order[owned_right_ordered] >= 0;
        int owned_right_atom = owned_right_valid
            ? atom_order[owned_right_ordered]
            : 0;
        float4 owned_right_posq = float4(
            positions[3 * owned_right_atom + 0],
            positions[3 * owned_right_atom + 1],
            positions[3 * owned_right_atom + 2],
            charges[owned_right_atom]
        );
        float2 owned_right_lj = float2(
            half_sigma[owned_right_atom],
            sqrt_epsilon[owned_right_atom]
        );

        for (uint rotation = 0u; rotation < 32u; rotation++) {
            ushort source_lane = (ushort)((lane + rotation) & 31u);
            bool right_valid = simd_shuffle(
                owned_right_valid ? 1u : 0u,
                source_lane
            ) != 0u;
            float4 right_posq = float4(
                simd_shuffle(owned_right_posq.x, source_lane),
                simd_shuffle(owned_right_posq.y, source_lane),
                simd_shuffle(owned_right_posq.z, source_lane),
                simd_shuffle(owned_right_posq.w, source_lane)
            );
            float2 right_lj = float2(
                simd_shuffle(owned_right_lj.x, source_lane),
                simd_shuffle(owned_right_lj.y, source_lane)
            );
            if (owner_valid && right_valid) {
                float dx = owner_posq.x - right_posq.x;
                float dy = owner_posq.y - right_posq.y;
                float dz = owner_posq.z - right_posq.z;
                dx -= box_lx * rint(dx * box_ix);
                dy -= box_ly * rint(dy * box_iy);
                dz -= box_lz * rint(dz * box_iz);
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > 0.0f && r2 < cutoff2) {
                    float inv_distance = rsqrt(r2);
                    float inv_r2 = inv_distance * inv_distance;
                    float distance = r2 * inv_distance;
                    float sigma_ij = owner_lj.x + right_lj.x;
                    float epsilon_ij = owner_lj.y * right_lj.y;
                    float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                    float inv_r6 = sigma2_over_r2
                        * sigma2_over_r2 * sigma2_over_r2;
                    float inv_r12 = inv_r6 * inv_r6;
                    float unswitched_energy = 0.0f;
                    float switch_value = 1.0f;
                    float switch_derivative = 0.0f;
                    if (switch_flag > 0.5f) {
                        unswitched_energy =
                            4.0f * epsilon_ij * (inv_r12 - inv_r6);
                        if (shift_flag > 0.5f) {
                            float sigma2_over_rc2 =
                                sigma_ij * sigma_ij / cutoff2;
                            float inv_rc6 = sigma2_over_rc2
                                * sigma2_over_rc2 * sigma2_over_rc2;
                            unswitched_energy -= 4.0f * epsilon_ij
                                * (inv_rc6 * inv_rc6 - inv_rc6);
                        }
                        float x = clamp(
                            (distance - switch_start) * inv_switch_width,
                            0.0f,
                            1.0f
                        );
                        float x2 = x * x;
                        float x3 = x2 * x;
                        float x4 = x3 * x;
                        float x5 = x4 * x;
                        switch_value = 1.0f
                            - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                        if (distance > switch_start && distance < switch_end) {
                            switch_derivative = -(
                                30.0f * x2 - 60.0f * x3 + 30.0f * x4
                            ) * inv_switch_width;
                        }
                    }
                    float scalar =
                        24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                            * inv_r2 * switch_value
                        - unswitched_energy * switch_derivative * inv_distance;
                    float qij = owner_posq.w * right_posq.w;
                    float erfc_term = mlx_atomistic_erfc_nonnegative(
                        ewald_alpha * distance
                    );
                    scalar += coulomb * qij * (
                        erfc_term * inv_r2 * inv_distance
                        + ewald_self * exp(-ewald_alpha * ewald_alpha * r2)
                            * inv_r2
                    );
                    float3 contribution = float3(
                        scalar * dx,
                        scalar * dy,
                        scalar * dz
                    );
                    float3 corrected = contribution - owner_force_compensation;
                    float3 next_force = owner_force + corrected;
                    owner_force_compensation =
                        (next_force - owner_force) - corrected;
                    owner_force = next_force;
                }
            }
        }
    }

    if (owner_valid) {
        int topology_start = topology_offsets[owner_atom];
        int topology_stop = topology_offsets[owner_atom + 1];
        for (int t = topology_start; t < topology_stop; t++) {
            int neighbor = topology_neighbors[t];
            float dx = owner_posq.x - positions[3 * neighbor + 0];
            float dy = owner_posq.y - positions[3 * neighbor + 1];
            float dz = owner_posq.z - positions[3 * neighbor + 2];
            dx -= box_lx * rint(dx * box_ix);
            dy -= box_ly * rint(dy * box_iy);
            dz -= box_lz * rint(dz * box_iz);
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 0.0f && r2 < cutoff2) {
                float inv_distance = rsqrt(r2);
                float inv_r2 = inv_distance * inv_distance;
                float distance = r2 * inv_distance;
                float sigma_ij = owner_lj.x + half_sigma[neighbor];
                float epsilon_ij = owner_lj.y * sqrt_epsilon[neighbor];
                float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                float inv_r6 = sigma2_over_r2
                    * sigma2_over_r2 * sigma2_over_r2;
                float inv_r12 = inv_r6 * inv_r6;
                float unswitched_energy = 0.0f;
                float switch_value = 1.0f;
                float switch_derivative = 0.0f;
                if (switch_flag > 0.5f) {
                    unswitched_energy =
                        4.0f * epsilon_ij * (inv_r12 - inv_r6);
                    if (shift_flag > 0.5f) {
                        float sigma2_over_rc2 = sigma_ij * sigma_ij / cutoff2;
                        float inv_rc6 = sigma2_over_rc2
                            * sigma2_over_rc2 * sigma2_over_rc2;
                        unswitched_energy -= 4.0f * epsilon_ij
                            * (inv_rc6 * inv_rc6 - inv_rc6);
                    }
                    float x = clamp(
                        (distance - switch_start) * inv_switch_width,
                        0.0f,
                        1.0f
                    );
                    float x2 = x * x;
                    float x3 = x2 * x;
                    float x4 = x3 * x;
                    float x5 = x4 * x;
                    switch_value = 1.0f
                        - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                    if (distance > switch_start && distance < switch_end) {
                        switch_derivative = -(
                            30.0f * x2 - 60.0f * x3 + 30.0f * x4
                        ) * inv_switch_width;
                    }
                }
                float correction_scale = topology_classes[t] == 0
                    ? -1.0f
                    : one_four_scale - 1.0f;
                float scalar = correction_scale * (
                    24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                        * inv_r2 * switch_value
                    - unswitched_energy * switch_derivative * inv_distance
                );
                float3 contribution = float3(
                    scalar * dx,
                    scalar * dy,
                    scalar * dz
                );
                float3 corrected = contribution - owner_force_compensation;
                float3 next_force = owner_force + corrected;
                owner_force_compensation =
                    (next_force - owner_force) - corrected;
                owner_force = next_force;
            }
        }
        forces[3 * owner_atom + 0] = owner_force.x;
        forces[3 * owner_atom + 1] = owner_force.y;
        forces[3 * owner_atom + 2] = owner_force.z;
    }
"""

_SPARSE_PME_CORRECTION_FORCE_ONLY_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)npair[0]) {
        return;
    }

    int atom_i = pairs_i[t];
    int atom_j = pairs_j[t];
    float dx = positions[3 * atom_i + 0] - positions[3 * atom_j + 0];
    float dy = positions[3 * atom_i + 1] - positions[3 * atom_j + 1];
    float dz = positions[3 * atom_i + 2] - positions[3 * atom_j + 2];
    dx -= box[0] * rint(dx * box[3]);
    dy -= box[1] * rint(dy * box[4]);
    dz -= box[2] * rint(dz * box[5]);
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 0.0f) {
        return;
    }

    float inv_distance = rsqrt(r2);
    float inv_r2 = inv_distance * inv_distance;
    float scalar = params[0] * charge_products[t] * inv_r2 * inv_distance;
    float epsilon = lj_epsilon[t];
    if (epsilon > 0.0f) {
        float sigma2_over_r2 = lj_sigma[t] * lj_sigma[t] * inv_r2;
        float inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
        float inv_r12 = inv_r6 * inv_r6;
        scalar += 24.0f * epsilon * (2.0f * inv_r12 - inv_r6) * inv_r2;
    }

    float fx = scalar * dx;
    float fy = scalar * dy;
    float fz = scalar * dz;
    atomic_fetch_add_explicit(&forces[3 * atom_i + 0], fx, memory_order_relaxed);
    atomic_fetch_add_explicit(&forces[3 * atom_i + 1], fy, memory_order_relaxed);
    atomic_fetch_add_explicit(&forces[3 * atom_i + 2], fz, memory_order_relaxed);
    atomic_fetch_add_explicit(&forces[3 * atom_j + 0], -fx, memory_order_relaxed);
    atomic_fetch_add_explicit(&forces[3 * atom_j + 1], -fy, memory_order_relaxed);
    atomic_fetch_add_explicit(&forces[3 * atom_j + 2], -fz, memory_order_relaxed);
"""

_PME_CUTOFF_CORRECTION_VIRIAL_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)npair[0]) {
        return;
    }
    int i = pairs_i[t];
    int j = pairs_j[t];

    float raw_dx = positions[3 * i + 0] - positions[3 * j + 0];
    float raw_dy = positions[3 * i + 1] - positions[3 * j + 1];
    float raw_dz = positions[3 * i + 2] - positions[3 * j + 2];
    float center_dx = centers[3 * i + 0] - centers[3 * j + 0];
    float center_dy = centers[3 * i + 1] - centers[3 * j + 1];
    float center_dz = centers[3 * i + 2] - centers[3 * j + 2];
    float lx = box[0];
    float ly = box[1];
    float lz = box[2];
    float dx = raw_dx - lx * rint(raw_dx / lx);
    float dy = raw_dy - ly * rint(raw_dy / ly);
    float dz = raw_dz - lz * rint(raw_dz / lz);
    float base_r2 = dx * dx + dy * dy + dz * dz;
    bool base_cutoff =
        base_r2 > 0.0f && base_r2 < params[0];
    float lj_scale = lj_scales[t];
    bool base_lj = base_cutoff && lj_scale != 0.0f;
    float base_safe_r2 = base_lj ? base_r2 : 1.0f;
    float base_distance = sqrt(base_safe_r2);
    float sigma_ij = 0.5f * (sigma[i] + sigma[j]);
    float epsilon_ij = sqrt(epsilon[i] * epsilon[j]);
    float base_sigma2_over_r2 =
        sigma_ij * sigma_ij / base_safe_r2;
    float base_inv_r6 =
        base_sigma2_over_r2
        * base_sigma2_over_r2
        * base_sigma2_over_r2;
    float base_d_energy_d_distance = (
        24.0f
        * epsilon_ij
        * (base_inv_r6 - 2.0f * base_inv_r6 * base_inv_r6)
        / base_distance
    );
    float qij = charges[i] * charges[j];
    float strain_epsilon = params[2];

    for (int axis = 0; axis < 3; axis++) {
        float3 strained_displacement[2];
        float lj_pair_energy[2] = {0.0f, 0.0f};
        float coulomb_boundary_energy[2] = {0.0f, 0.0f};
        for (int sign_index = 0; sign_index < 2; sign_index++) {
            float signed_epsilon = (
                sign_index == 0 ? strain_epsilon : -strain_epsilon
            );
            float slx = lx * (axis == 0 ? 1.0f + signed_epsilon : 1.0f);
            float sly = ly * (axis == 1 ? 1.0f + signed_epsilon : 1.0f);
            float slz = lz * (axis == 2 ? 1.0f + signed_epsilon : 1.0f);
            float sdx = raw_dx + (
                axis == 0 ? signed_epsilon * center_dx : 0.0f
            );
            float sdy = raw_dy + (
                axis == 1 ? signed_epsilon * center_dy : 0.0f
            );
            float sdz = raw_dz + (
                axis == 2 ? signed_epsilon * center_dz : 0.0f
            );
            sdx -= slx * rint(sdx / slx);
            sdy -= sly * rint(sdy / sly);
            sdz -= slz * rint(sdz / slz);
            strained_displacement[sign_index] = float3(sdx, sdy, sdz);
            float r2 = sdx * sdx + sdy * sdy + sdz * sdz;
            bool strained_cutoff = r2 > 0.0f && r2 < params[0];
            float safe_r2 = (
                strained_cutoff || base_cutoff ? r2 : 1.0f
            );
            if (params[5] > 0.5f && strained_cutoff && lj_scale != 0.0f) {
                float sigma2_over_r2 =
                    sigma_ij * sigma_ij / safe_r2;
                float inv_r6 =
                    sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
                lj_pair_energy[sign_index] = (
                    4.0f
                    * epsilon_ij
                    * (inv_r6 * inv_r6 - inv_r6)
                    * lj_scale
                );
            }
            if (params[6] > 0.5f) {
                float distance = sqrt(safe_r2);
                float mask_delta = (
                    (strained_cutoff ? 1.0f : 0.0f)
                    - (base_cutoff ? 1.0f : 0.0f)
                );
                coulomb_boundary_energy[sign_index] = (
                    mask_delta
                    * params[3]
                    * qij
                    * (1.0f - mlx_atomistic_erf(params[4] * distance))
                    / distance
                );
            }
        }
        float correction = 0.0f;
        if (params[5] > 0.5f) {
            float finite_strain_lj_virial = -(
                lj_pair_energy[0] - lj_pair_energy[1]
            ) / (2.0f * strain_epsilon);
            float3 displacement_derivative = (
                strained_displacement[0] - strained_displacement[1]
            ) / (2.0f * strain_epsilon);
            float distance_derivative = (
                dx * displacement_derivative.x
                + dy * displacement_derivative.y
                + dz * displacement_derivative.z
            ) / base_distance;
            float local_lj_virial = base_lj ? (
                -base_d_energy_d_distance
                * distance_derivative
                * lj_scale
            ) : 0.0f;
            correction += finite_strain_lj_virial - local_lj_virial;
        }
        if (params[6] > 0.5f) {
            correction -= (
                coulomb_boundary_energy[0]
                - coulomb_boundary_energy[1]
            ) / (2.0f * strain_epsilon);
        }
        atomic_store_explicit(
            &pair_correction[3 * t + axis],
            correction,
            memory_order_relaxed
        );
    }
"""

_PME_ORDER5_HEADER = r"""
inline void mlx_atomistic_pme_order5_weights(
    float3 fraction,
    thread float3* weights,
    thread float3* derivatives
) {
    weights[4] = float3(0.0f);
    weights[1] = fraction;
    weights[0] = float3(1.0f) - fraction;
    for (int j = 3; j < 5; j++) {
        float divisor = 1.0f / (float)(j - 1);
        weights[j - 1] = divisor * fraction * weights[j - 2];
        for (int k = 1; k < (j - 1); k++) {
            weights[j - k - 1] = divisor * (
                (fraction + float3((float)k)) * weights[j - k - 2]
                + (float3((float)(j - k)) - fraction)
                * weights[j - k - 1]
            );
        }
        weights[0] =
            divisor * (float3(1.0f) - fraction) * weights[0];
    }

    derivatives[0] = -weights[0];
    for (int j = 1; j < 5; j++) {
        derivatives[j] = weights[j - 1] - weights[j];
    }

    const float scale = 0.25f;
    weights[4] = scale * fraction * weights[3];
    for (int j = 1; j < 4; j++) {
        weights[4 - j] = scale * (
            (fraction + float3((float)j)) * weights[3 - j]
            + (float3((float)(5 - j)) - fraction) * weights[4 - j]
        );
    }
    weights[0] =
        scale * (float3(1.0f) - fraction) * weights[0];
}
"""

_PME_ORDER5_SPREAD_SOURCE = r"""
    uint lane = thread_position_in_grid.x;
    uint atom_count = (uint)counts[0];
    if (lane >= atom_count * 5u) {
        return;
    }
    uint atom = lane / 5u;
    int z_offset = (int)(lane - atom * 5u);

    float3 scaled;
    scaled.x = (
        positions[3 * atom + 0]
        - floor(positions[3 * atom + 0] / cell[0]) * cell[0]
    ) / cell[0] * (float)mesh[0];
    scaled.y = (
        positions[3 * atom + 1]
        - floor(positions[3 * atom + 1] / cell[1]) * cell[1]
    ) / cell[1] * (float)mesh[1];
    scaled.z = (
        positions[3 * atom + 2]
        - floor(positions[3 * atom + 2] / cell[2]) * cell[2]
    ) / cell[2] * (float)mesh[2];
    int3 anchor = int3(
        (int)floor(scaled.x) - 2,
        (int)floor(scaled.y) - 2,
        (int)floor(scaled.z) - 2
    );
    float3 fraction = scaled - floor(scaled);
    float3 weights[5];
    float3 derivatives[5];
    mlx_atomistic_pme_order5_weights(
        fraction,
        weights,
        derivatives
    );

    int nz = mesh[2];
    int ny = mesh[1];
    int z = (anchor.z + z_offset + nz) % nz;
    float qz = charges[atom] * weights[z_offset].z;
    for (int x_offset = 0; x_offset < 5; x_offset++) {
        int x = (anchor.x + x_offset + mesh[0]) % mesh[0];
        float qzx = qz * weights[x_offset].x;
        for (int y_offset = 0; y_offset < 5; y_offset++) {
            int y = (anchor.y + y_offset + ny) % ny;
            int index = (x * ny + y) * nz + z;
            float value = qzx * weights[y_offset].y;
            atomic_fetch_add_explicit(
                &charge_grid[index],
                value,
                memory_order_relaxed
            );
        }
    }
"""

_PME_ORDER5_INTERPOLATE_SOURCE = r"""
    uint atom = thread_position_in_grid.x;
    if (atom >= (uint)counts[0]) {
        return;
    }

    float3 scaled;
    scaled.x = (
        positions[3 * atom + 0]
        - floor(positions[3 * atom + 0] / cell[0]) * cell[0]
    ) / cell[0] * (float)mesh[0];
    scaled.y = (
        positions[3 * atom + 1]
        - floor(positions[3 * atom + 1] / cell[1]) * cell[1]
    ) / cell[1] * (float)mesh[1];
    scaled.z = (
        positions[3 * atom + 2]
        - floor(positions[3 * atom + 2] / cell[2]) * cell[2]
    ) / cell[2] * (float)mesh[2];
    int3 anchor = int3(
        (int)floor(scaled.x) - 2,
        (int)floor(scaled.y) - 2,
        (int)floor(scaled.z) - 2
    );
    float3 fraction = scaled - floor(scaled);
    float3 weights[5];
    float3 derivatives[5];
    mlx_atomistic_pme_order5_weights(
        fraction,
        weights,
        derivatives
    );

    int ny = mesh[1];
    int nz = mesh[2];
#ifdef MLX_ATOMISTIC_PME_WRITE_ENERGY
    float potential = 0.0f;
#endif
    float3 gradient = float3(0.0f);
    for (int x_offset = 0; x_offset < 5; x_offset++) {
        int x = (anchor.x + x_offset + mesh[0]) % mesh[0];
        for (int y_offset = 0; y_offset < 5; y_offset++) {
            int y = (anchor.y + y_offset + ny) % ny;
            for (int z_offset = 0; z_offset < 5; z_offset++) {
                int z = (anchor.z + z_offset + nz) % nz;
                int grid_index = (x * ny + y) * nz + z;
#ifdef MLX_ATOMISTIC_PME_COMPLEX_GRID
                float grid_value = potential_grid[grid_index].real;
#else
                float grid_value = potential_grid[grid_index];
#endif
                float wx = weights[x_offset].x;
                float wy = weights[y_offset].y;
                float wz = weights[z_offset].z;
#ifdef MLX_ATOMISTIC_PME_WRITE_ENERGY
                potential += wx * wy * wz * grid_value;
#endif
                gradient.x += (
                    derivatives[x_offset].x * wy * wz * grid_value
                );
                gradient.y += (
                    wx * derivatives[y_offset].y * wz * grid_value
                );
                gradient.z += (
                    wx * wy * derivatives[z_offset].z * grid_value
                );
            }
        }
    }

    float charge = charges[atom];
#if defined(MLX_ATOMISTIC_PME_COMPLEX_GRID) \
    || defined(MLX_ATOMISTIC_PME_NORMALIZED_REAL_GRID)
    float reciprocal_scale =
        (float)mesh[0] * (float)mesh[1] * (float)mesh[2];
#else
    float reciprocal_scale = 1.0f;
#endif
#ifdef MLX_ATOMISTIC_PME_WRITE_ENERGY
    atom_energy[atom] = 0.5f * charge * potential;
#endif
    forces[3 * atom + 0] =
        -charge * reciprocal_scale * gradient.x * (float)mesh[0] / cell[0];
    forces[3 * atom + 1] =
        -charge * reciprocal_scale * gradient.y * (float)mesh[1] / cell[1];
    forces[3 * atom + 2] =
        -charge * reciprocal_scale * gradient.z * (float)mesh[2] / cell[2];
"""

_ALIGNED_TOPOLOGY_LJ_SCALES_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)counts[0]) {
        return;
    }

    int left = pairs_i[t];
    int right = pairs_j[t];
    if (left > right) {
        int temporary = left;
        left = right;
        right = temporary;
    }

    bool excluded = false;
    int lower = 0;
    int upper = counts[1];
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        int known_left = excluded_i[middle];
        int known_right = excluded_j[middle];
        if (
            known_left < left
            || (known_left == left && known_right < right)
        ) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (
        lower < counts[1]
        && excluded_i[lower] == left
        && excluded_j[lower] == right
    ) {
        excluded = true;
    }

    float scale = excluded ? 0.0f : 1.0f;
    if (!excluded && params[0] != 1.0f && counts[2] > 0) {
        lower = 0;
        upper = counts[2];
        while (lower < upper) {
            int middle = lower + (upper - lower) / 2;
            int known_left = one_four_i[middle];
            int known_right = one_four_j[middle];
            if (
                known_left < left
                || (known_left == left && known_right < right)
            ) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        if (
            lower < counts[2]
            && one_four_i[lower] == left
            && one_four_j[lower] == right
        ) {
            scale = params[0];
        }
    }
    scales[t] = scale;
"""

_NEIGHBOR_PAIR_CUTOFF_MASK_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)counts[0]) {
        return;
    }

    int i = pairs_i[t];
    int j = pairs_j[t];
    float dx = positions[3 * i + 0] - positions[3 * j + 0];
    float dy = positions[3 * i + 1] - positions[3 * j + 1];
    float dz = positions[3 * i + 2] - positions[3 * j + 2];
    dx -= box[0] * rint(dx / box[0]);
    dy -= box[1] * rint(dy / box[1]);
    dz -= box[2] * rint(dz / box[2]);
    close[t] = dx * dx + dy * dy + dz * dz < params[0];
"""

_NEIGHBOR_CELL_PAIR_CANDIDATES_SOURCE = r"""
    uint task = thread_position_in_grid.x;
    if (task >= (uint)counts[0]) {
        return;
    }

    int left_cell = cell_pairs[2 * task + 0];
    int right_cell = cell_pairs[2 * task + 1];
    int left_start = cell_starts[left_cell];
    int right_start = cell_starts[right_cell];
    int left_count = cell_counts[left_cell];
    int right_count = cell_counts[right_cell];
    int output = task_offsets[task];

    if (left_cell == right_cell) {
        for (int left = 0; left < left_count; left++) {
            int atom_i = sorted_atoms[left_start + left];
            for (int right = left + 1; right < right_count; right++) {
                pairs_i[output] = atom_i;
                pairs_j[output] = sorted_atoms[right_start + right];
                output++;
            }
        }
        return;
    }

    for (int left = 0; left < left_count; left++) {
        int atom_i = sorted_atoms[left_start + left];
        for (int right = 0; right < right_count; right++) {
            pairs_i[output] = atom_i;
            pairs_j[output] = sorted_atoms[right_start + right];
            output++;
        }
    }
"""

_NEIGHBOR_PAIR_ORDERED_SCATTER_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    if (t >= (uint)counts[0] || !close[t]) {
        return;
    }
    uint out = (uint)(prefix[t] - 1);
    int i = pairs_i[t];
    int j = pairs_j[t];
    accepted_i[out] = min(i, j);
    accepted_j[out] = max(i, j);
"""

_NEIGHBOR_CELL_ATOM_BLOCKS_SOURCE = r"""
    uint cell = thread_position_in_grid.x;
    if (cell >= (uint)counts[0]) {
        return;
    }

    int atom_start = cell_starts[cell];
    int atom_count = cell_counts[cell];
    int block_start = cell_block_starts[cell];
    int block_count = cell_block_counts[cell];
    for (int local_block = 0; local_block < block_count; ++local_block) {
        int block = block_start + local_block;
        for (int slot = 0; slot < 8; ++slot) {
            int local_atom = 8 * local_block + slot;
            atom_blocks[8 * block + slot] = local_atom < atom_count
                ? sorted_atoms[atom_start + local_atom]
                : -1;
        }
    }
"""

_NEIGHBOR_CELL_TILE_CANDIDATES_SOURCE = r"""
    uint task = thread_position_in_grid.x;
    if (task >= (uint)counts[0]) {
        return;
    }

    int left_cell = cell_pairs[2 * task + 0];
    int right_cell = cell_pairs[2 * task + 1];
    int left_start = cell_block_starts[left_cell];
    int right_start = cell_block_starts[right_cell];
    int left_count = cell_block_counts[left_cell];
    int right_count = cell_block_counts[right_cell];
    int output = task_offsets[task];

    if (left_cell == right_cell) {
        for (int left = 0; left < left_count; left++) {
            for (int right = left; right < right_count; right++) {
                tile_left[output] = left_start + left;
                tile_right[output] = right_start + right;
                output++;
            }
        }
        return;
    }

    for (int left = 0; left < left_count; left++) {
        for (int right = 0; right < right_count; right++) {
            tile_left[output] = left_start + left;
            tile_right[output] = right_start + right;
            output++;
        }
    }
"""

_NEIGHBOR_TILE_MEMBERSHIP_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint top_left_slot = lane >> 3;
    uint bottom_left_slot = top_left_slot + 4u;
    uint right_slot = lane & 7u;

    int left_block = tile_blocks[2 * tile + 0];
    int right_block = tile_blocks[2 * tile + 1];

    // One SIMD group owns the full 8x8 coarse tile. Its first 16 lanes load
    // each atom exactly once, then SIMD shuffles broadcast those positions to
    // the 32 lanes. Each lane evaluates one top-half and one bottom-half pair.
    int owned_atom = -1;
    if (lane < 8u) {
        owned_atom = atom_blocks[8 * left_block + lane];
    } else if (lane < 16u) {
        owned_atom = atom_blocks[8 * right_block + lane - 8u];
    }
    int safe_owned_atom = max(owned_atom, 0);
    float owned_x = positions[3 * safe_owned_atom + 0];
    float owned_y = positions[3 * safe_owned_atom + 1];
    float owned_z = positions[3 * safe_owned_atom + 2];

    int atom_i_top = simd_shuffle(owned_atom, top_left_slot);
    int atom_i_bottom = simd_shuffle(owned_atom, bottom_left_slot);
    int atom_j = simd_shuffle(owned_atom, 8u + right_slot);
    float ix_top = simd_shuffle(owned_x, top_left_slot);
    float iy_top = simd_shuffle(owned_y, top_left_slot);
    float iz_top = simd_shuffle(owned_z, top_left_slot);
    float ix_bottom = simd_shuffle(owned_x, bottom_left_slot);
    float iy_bottom = simd_shuffle(owned_y, bottom_left_slot);
    float iz_bottom = simd_shuffle(owned_z, bottom_left_slot);
    float jx = simd_shuffle(owned_x, 8u + right_slot);
    float jy = simd_shuffle(owned_y, 8u + right_slot);
    float jz = simd_shuffle(owned_z, 8u + right_slot);

    bool top_valid = atom_i_top >= 0 && atom_j >= 0;
    bool bottom_valid = atom_i_bottom >= 0 && atom_j >= 0;
    if (left_block == right_block) {
        top_valid = top_valid && top_left_slot < right_slot;
        bottom_valid = bottom_valid && bottom_left_slot < right_slot;
    }

    bool top_close = false;
    if (top_valid) {
        float dx = ix_top - jx;
        float dy = iy_top - jy;
        float dz = iz_top - jz;
        dx -= box[0] * rint(dx / box[0]);
        dy -= box[1] * rint(dy / box[1]);
        dz -= box[2] * rint(dz / box[2]);
        top_close = dx * dx + dy * dy + dz * dz < params[0];
    }

    bool bottom_close = false;
    if (bottom_valid) {
        float dx = ix_bottom - jx;
        float dy = iy_bottom - jy;
        float dz = iz_bottom - jz;
        dx -= box[0] * rint(dx / box[0]);
        dy -= box[1] * rint(dy / box[1]);
        dz -= box[2] * rint(dz / box[2]);
        bottom_close = dx * dx + dy * dy + dz * dz < params[0];
    }

    uint bit = 1u << (4u * top_left_slot + (right_slot & 3u));
    uint word0 = simd_sum(
        top_close && right_slot < 4u ? bit : 0u
    );
    uint word1 = simd_sum(
        top_close && right_slot >= 4u ? bit : 0u
    );
    uint word2 = simd_sum(
        bottom_close && right_slot < 4u ? bit : 0u
    );
    uint word3 = simd_sum(
        bottom_close && right_slot >= 4u ? bit : 0u
    );
    uint exact_total = simd_sum(
        (top_close ? 1u : 0u) + (bottom_close ? 1u : 0u)
    );
    if (lane == 0u) {
        subtile_mask[4 * tile + 0] = word0;
        subtile_mask[4 * tile + 1] = word1;
        subtile_mask[4 * tile + 2] = word2;
        subtile_mask[4 * tile + 3] = word3;
        member_counts[tile] = (int)exact_total;
        subtile_counts[tile] =
            (word0 != 0u ? 1 : 0)
            + (word1 != 0u ? 1 : 0)
            + (word2 != 0u ? 1 : 0)
            + (word3 != 0u ? 1 : 0);
    }
"""

_NEIGHBOR_TILE_ORDERED_SCATTER_SOURCE = r"""
    uint tile = thread_position_in_grid.x;
    if (tile >= (uint)counts[0] || subtile_counts[tile] == 0) {
        return;
    }
    uint output = (uint)(prefix[tile] - subtile_counts[tile]);
    int coarse_left = tile_blocks[2 * tile + 0];
    int coarse_right = tile_blocks[2 * tile + 1];
    for (uint quadrant = 0u; quadrant < 4u; quadrant++) {
        uint word = subtile_mask[4 * tile + quadrant];
        if (word == 0u) {
            continue;
        }
        accepted_tile_blocks[2 * output + 0] =
            2 * coarse_left + (int)(quadrant >> 1);
        accepted_tile_blocks[2 * output + 1] =
            2 * coarse_right + (int)(quadrant & 1u);
        accepted_member_mask[output] = word;
        output++;
    }
"""

_NEIGHBOR_TILE_LEFT_COUNTS_SOURCE = r"""
    uint cell = thread_position_in_grid.x;
    if (cell >= (uint)counts[0]) {
        return;
    }
    int left_start = cell_block_starts[cell];
    int left_count = cell_block_counts[cell];
    int task_total = counts[1];
    int lower = 0;
    int upper = task_total;
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        if (cell_pairs[2 * middle + 0] < (int)cell) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    int task_start = lower;
    upper = task_total;
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        if (cell_pairs[2 * middle + 0] <= (int)cell) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    int task_count = lower - task_start;
    for (int local_left = 0; local_left < left_count; local_left++) {
        int top_count = 0;
        int bottom_count = 0;
        for (int local_task = 0; local_task < task_count; local_task++) {
            int task = task_start + local_task;
            if (task_tile_counts[task] == 0) {
                continue;
            }
            int right_cell = cell_pairs[2 * task + 1];
            int right_count = cell_block_counts[right_cell];
            int candidate = task_offsets[task];
            int right_begin = 0;
            if (right_cell == (int)cell) {
                candidate += local_left * left_count
                    - local_left * (local_left - 1) / 2;
                right_begin = local_left;
            } else {
                candidate += local_left * right_count;
            }
            for (int local_right = right_begin; local_right < right_count; local_right++) {
                top_count += subtile_mask[4 * candidate + 0] != 0u ? 1 : 0;
                top_count += subtile_mask[4 * candidate + 1] != 0u ? 1 : 0;
                bottom_count += subtile_mask[4 * candidate + 2] != 0u ? 1 : 0;
                bottom_count += subtile_mask[4 * candidate + 3] != 0u ? 1 : 0;
                candidate++;
            }
        }
        int coarse_left = left_start + local_left;
        fine_tile_counts[2 * coarse_left + 0] = top_count;
        fine_tile_counts[2 * coarse_left + 1] = bottom_count;
    }
"""

_NEIGHBOR_TILE_LEFT_SCATTER_SOURCE = r"""
    uint cell = thread_position_in_grid.x;
    if (cell >= (uint)counts[0]) {
        return;
    }
    int left_start = cell_block_starts[cell];
    int left_count = cell_block_counts[cell];
    int task_total = counts[1];
    int lower = 0;
    int upper = task_total;
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        if (cell_pairs[2 * middle + 0] < (int)cell) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    int task_start = lower;
    upper = task_total;
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        if (cell_pairs[2 * middle + 0] <= (int)cell) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    int task_count = lower - task_start;
    for (int local_left = 0; local_left < left_count; local_left++) {
        int coarse_left = left_start + local_left;
        int top_output = fine_tile_prefix[2 * coarse_left + 0]
            - fine_tile_counts[2 * coarse_left + 0];
        int bottom_output = fine_tile_prefix[2 * coarse_left + 1]
            - fine_tile_counts[2 * coarse_left + 1];
        for (int local_task = 0; local_task < task_count; local_task++) {
            int task = task_start + local_task;
            if (task_tile_counts[task] == 0) {
                continue;
            }
            int right_cell = cell_pairs[2 * task + 1];
            int right_start = cell_block_starts[right_cell];
            int right_count = cell_block_counts[right_cell];
            int candidate = task_offsets[task];
            int right_begin = 0;
            if (right_cell == (int)cell) {
                candidate += local_left * left_count
                    - local_left * (local_left - 1) / 2;
                right_begin = local_left;
            } else {
                candidate += local_left * right_count;
            }
            for (int local_right = right_begin; local_right < right_count; local_right++) {
                int coarse_right = right_start + local_right;
                for (int quadrant = 0; quadrant < 2; quadrant++) {
                    uint word = subtile_mask[4 * candidate + quadrant];
                    if (word == 0u) {
                        continue;
                    }
                    accepted_tile_blocks[2 * top_output + 0] = 2 * coarse_left;
                    accepted_tile_blocks[2 * top_output + 1] =
                        2 * coarse_right + quadrant;
                    accepted_member_mask[top_output] = word;
                    top_output++;
                }
                for (int quadrant = 2; quadrant < 4; quadrant++) {
                    uint word = subtile_mask[4 * candidate + quadrant];
                    if (word == 0u) {
                        continue;
                    }
                    accepted_tile_blocks[2 * bottom_output + 0] = 2 * coarse_left + 1;
                    accepted_tile_blocks[2 * bottom_output + 1] =
                        2 * coarse_right + (quadrant & 1);
                    accepted_member_mask[bottom_output] = word;
                    bottom_output++;
                }
                candidate++;
            }
        }
    }
"""

_NEIGHBOR_TILE_COLUMN_SCATTER_SOURCE = r"""
    uint tile = thread_position_in_grid.x;
    if (tile >= (uint)counts[0] || active_column_counts[tile] == 0) {
        return;
    }

    uint output = (uint)(prefix[tile] - active_column_counts[tile]);
    uint word = member_mask[tile];
    for (uint column = 0u; column < 4u; column++) {
        uint column_pattern = 0x1111u << column;
        if ((word & column_pattern) != 0u) {
            uint column_members = 0u;
            for (uint left = 0u; left < 4u; left++) {
                column_members |= (
                    (word >> (4u * left + column)) & 1u
                ) << left;
            }
            uint descriptor = (column_members << 28) | (4u * tile + column);
            force_columns[output] = (int)descriptor;
            output++;
        }
    }
"""

_NEIGHBOR_TILE_FORCE_GROUPS_SOURCE = r"""
    uint block = thread_position_in_grid.x;
    if (block >= (uint)counts[0] || work_counts[block] == 0) {
        return;
    }

    int work_count = work_counts[block];
    int work_start = work_prefix[block] - work_count;
    int group_count = group_counts[block];
    int group_start = group_prefix[block] - group_count;
    int items_per_group = counts[1];
    for (int local = 0; local < group_count; local++) {
        int consumed = local * items_per_group;
        int output = group_start + local;
        force_group_starts[output] = work_start + consumed;
        force_group_counts[output] = min(items_per_group, work_count - consumed);
    }
"""

_NEIGHBOR_TILE_MEMBER_COUNTS_SOURCE = r"""
    uint tile = thread_position_in_grid.x;
    if (tile >= (uint)counts[0]) {
        return;
    }
    uint total = 0u;
    uint word = member_mask[tile];
    for (uint bit = 0u; bit < 16u; bit++) {
        total += (word >> bit) & 1u;
    }
    member_counts[tile] = (int)total;
"""

_NEIGHBOR_TILE_PAIR_SCATTER_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint bit_index = lane;
    threadgroup uint scan[16];

    uint word = member_mask[tile];
    uint active = (word >> bit_index) & 1u;
    scan[lane] = active;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < 16u; offset <<= 1u) {
        uint addend = lane >= offset ? scan[lane - offset] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[lane] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active == 0u) {
        return;
    }

    uint base = (uint)(member_prefix[tile] - member_counts[tile]);
    uint output = base + scan[lane] - 1u;
    int left_block = tile_blocks[2 * tile + 0];
    int right_block = tile_blocks[2 * tile + 1];
    int atom_i = atom_blocks[4 * left_block + (lane >> 2)];
    int atom_j = atom_blocks[4 * right_block + (lane & 3u)];
    accepted_i[output] = min(atom_i, atom_j);
    accepted_j[output] = max(atom_i, atom_j);
"""

_TILE_TOPOLOGY_LJ_MASKS_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint bit_index = lane;
    threadgroup uint enabled[16];
    threadgroup uint one_four[16];

    uint member_word = member_mask[tile];
    bool member = ((member_word >> bit_index) & 1u) != 0u;
    int left_block = tile_blocks[2 * tile + 0];
    int right_block = tile_blocks[2 * tile + 1];
    int atom_i = atom_blocks[4 * left_block + (lane >> 2)];
    int atom_j = atom_blocks[4 * right_block + (lane & 3u)];
    int left = min(atom_i, atom_j);
    int right = max(atom_i, atom_j);

    bool excluded = false;
    if (member) {
        int start = excluded_offsets[left];
        int stop = excluded_offsets[left + 1];
        for (int index = start; index < stop; ++index) {
            if (excluded_right[index] == right) {
                excluded = true;
                break;
            }
        }
    }

    bool scaled = false;
    if (member && !excluded) {
        int start = one_four_offsets[left];
        int stop = one_four_offsets[left + 1];
        for (int index = start; index < stop; ++index) {
            if (one_four_right[index] == right) {
                scaled = true;
                break;
            }
        }
    }
    enabled[lane] = member && !excluded ? 1u : 0u;
    one_four[lane] = scaled ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint enabled_word = 0u;
        uint one_four_word = 0u;
        for (uint bit = 0u; bit < 16u; bit++) {
            enabled_word |= enabled[bit] << bit;
            one_four_word |= one_four[bit] << bit;
        }
        lj_enabled_mask[tile] = enabled_word;
        lj_one_four_mask[tile] = one_four_word;
    }
"""

_CONSTRAINT_HEADER = r"""
inline float3 constraint_load3(device const float* values, int atom) {
    return float3(
        values[3 * atom + 0],
        values[3 * atom + 1],
        values[3 * atom + 2]
    );
}

inline float3 constraint_minimum_image(
    float3 value,
    constant const float* box,
    int periodic
) {
    if (periodic != 0) {
        value.x -= box[0] * rint(value.x / box[0]);
        value.y -= box[1] * rint(value.y / box[1]);
        value.z -= box[2] * rint(value.z / box[2]);
    }
    return value;
}

inline float3 constraint_safe_normalize(float3 value) {
    return value * rsqrt(max(dot(value, value), 1.0e-20f));
}
"""

_DENSE_CONSTRAINT_APPLY_SOURCE = r"""
    uint atom = thread_position_in_grid.x;
    if (atom >= (uint)params[0]) {
        return;
    }

    float3 delta = float3(0.0f);
    if (atom < (uint)params[1]) {
        int family = owner_family[atom];
        int row = owner_rows[atom];
        int slot = owner_slots[atom];
        if (family == 1 && params[2] != 0) {
            delta = constraint_load3(settle_deltas, 3 * row + slot);
        }
        else if (family == 2 && params[3] != 0) {
            delta = constraint_load3(shake_deltas, 4 * row + slot);
        }
    }

    float3 value = constraint_load3(base_values, atom) + delta;
    constrained[3 * atom + 0] = value.x;
    constrained[3 * atom + 1] = value.y;
    constrained[3 * atom + 2] = value.z;
"""

_SMALL_CONSTRAINT_CLUSTER_POSITION_SOURCE = r"""
    uint cluster = thread_position_in_grid.x;
    if (cluster >= (uint)params[0]) {
        return;
    }
    int atom_count = atom_counts[cluster];
    int pair_count = pair_counts[cluster];
    int iterations = params[1];
    int periodic = params[2];
    float3 reference[4];
    float3 original[4];
    float3 current[4];
    float inverse_mass[4];
    float3 reference_delta[3];

    for (int slot = 0; slot < 4; slot++) {
        bool valid = slot < atom_count;
        int atom = valid ? cluster_atoms[4 * cluster + slot] : 0;
        reference[slot] = constraint_load3(reference_positions, atom);
        original[slot] = constraint_load3(predicted_positions, atom);
        current[slot] = original[slot];
        inverse_mass[slot] = valid ? 1.0f / masses[atom] : 0.0f;
    }
    for (int pair = 0; pair < 3; pair++) {
        int left_slot = pair_slots[6 * cluster + 2 * pair + 0];
        int right_slot = pair_slots[6 * cluster + 2 * pair + 1];
        reference_delta[pair] = pair < pair_count
            ? constraint_minimum_image(
                reference[left_slot] - reference[right_slot],
                box,
                periodic
            )
            : float3(0.0f);
    }

    for (int iteration = 0; iteration < iterations; iteration++) {
        float3 correction[4] = {
            float3(0.0f),
            float3(0.0f),
            float3(0.0f),
            float3(0.0f)
        };
        for (int pair = 0; pair < pair_count; pair++) {
            int left_slot = pair_slots[6 * cluster + 2 * pair + 0];
            int right_slot = pair_slots[6 * cluster + 2 * pair + 1];
            float3 displacement = constraint_minimum_image(
                current[left_slot] - current[right_slot],
                box,
                periodic
            );
            float target = target_distances[3 * cluster + pair];
            float error_squared = target * target - dot(displacement, displacement);
            float inverse_mass_sum =
                inverse_mass[left_slot] + inverse_mass[right_slot];
            float denominator = 2.0f * inverse_mass_sum
                * dot(displacement, reference_delta[pair]);
            float safe_denominator = abs(denominator) > 1.0e-20f
                ? denominator
                : (denominator < 0.0f ? -1.0e-20f : 1.0e-20f);
            float multiplier = error_squared / safe_denominator;
            float3 pair_correction = multiplier * reference_delta[pair];
            correction[left_slot] += inverse_mass[left_slot] * pair_correction;
            correction[right_slot] -= inverse_mass[right_slot] * pair_correction;
        }
        for (int slot = 0; slot < atom_count; slot++) {
            current[slot] += correction[slot];
        }
    }

    for (int slot = 0; slot < 4; slot++) {
        uint output = 12u * cluster + 3u * (uint)slot;
        float3 delta = slot < atom_count
            ? current[slot] - original[slot]
            : float3(0.0f);
        deltas[output + 0] = delta.x;
        deltas[output + 1] = delta.y;
        deltas[output + 2] = delta.z;
    }
"""

_SMALL_CONSTRAINT_CLUSTER_VELOCITY_SOURCE = r"""
    uint cluster = thread_position_in_grid.x;
    if (cluster >= (uint)params[0]) {
        return;
    }
    int atom_count = atom_counts[cluster];
    int pair_count = pair_counts[cluster];
    int iterations = params[1];
    int periodic = params[2];
    float3 original[4];
    float3 current[4];
    float inverse_mass[4];
    float3 unit[3];
    float weight_left[3];
    float weight_right[3];

    for (int slot = 0; slot < 4; slot++) {
        bool valid = slot < atom_count;
        int atom = valid ? cluster_atoms[4 * cluster + slot] : 0;
        original[slot] = constraint_load3(velocities, atom);
        current[slot] = original[slot];
        inverse_mass[slot] = valid ? 1.0f / masses[atom] : 0.0f;
    }
    for (int pair = 0; pair < 3; pair++) {
        int left_slot = pair_slots[6 * cluster + 2 * pair + 0];
        int right_slot = pair_slots[6 * cluster + 2 * pair + 1];
        if (pair < pair_count) {
            int left_atom = cluster_atoms[4 * cluster + left_slot];
            int right_atom = cluster_atoms[4 * cluster + right_slot];
            float3 displacement = constraint_minimum_image(
                constraint_load3(positions, left_atom)
                    - constraint_load3(positions, right_atom),
                box,
                periodic
            );
            unit[pair] = constraint_safe_normalize(displacement);
            float inverse_mass_sum =
                inverse_mass[left_slot] + inverse_mass[right_slot];
            weight_left[pair] = inverse_mass[left_slot] / inverse_mass_sum;
            weight_right[pair] = inverse_mass[right_slot] / inverse_mass_sum;
        }
        else {
            unit[pair] = float3(0.0f);
            weight_left[pair] = 0.0f;
            weight_right[pair] = 0.0f;
        }
    }

    for (int iteration = 0; iteration < iterations; iteration++) {
        float3 correction[4] = {
            float3(0.0f),
            float3(0.0f),
            float3(0.0f),
            float3(0.0f)
        };
        for (int pair = 0; pair < pair_count; pair++) {
            int left_slot = pair_slots[6 * cluster + 2 * pair + 0];
            int right_slot = pair_slots[6 * cluster + 2 * pair + 1];
            float relative = dot(
                current[left_slot] - current[right_slot],
                unit[pair]
            );
            float3 pair_correction = relative * unit[pair];
            correction[left_slot] -= weight_left[pair] * pair_correction;
            correction[right_slot] += weight_right[pair] * pair_correction;
        }
        for (int slot = 0; slot < atom_count; slot++) {
            current[slot] += correction[slot];
        }
    }

    for (int slot = 0; slot < 4; slot++) {
        uint output = 12u * cluster + 3u * (uint)slot;
        float3 delta = slot < atom_count
            ? current[slot] - original[slot]
            : float3(0.0f);
        deltas[output + 0] = delta.x;
        deltas[output + 1] = delta.y;
        deltas[output + 2] = delta.z;
    }
"""

_SETTLE_WATER_POSITION_SOURCE = r"""
    uint water = thread_position_in_grid.x;
    if (water >= (uint)params[0]) {
        return;
    }
    int oxygen = waters[3 * water + 0];
    int hydrogen_a = waters[3 * water + 1];
    int hydrogen_b = waters[3 * water + 2];
    int periodic = params[1];

    float3 reference_o = constraint_load3(reference_positions, oxygen);
    float3 reference_a = constraint_load3(reference_positions, hydrogen_a);
    float3 reference_b = constraint_load3(reference_positions, hydrogen_b);
    float3 predicted_o = constraint_load3(predicted_positions, oxygen);
    float3 predicted_a = constraint_load3(predicted_positions, hydrogen_a);
    float3 predicted_b = constraint_load3(predicted_positions, hydrogen_b);
    float3 old_b = constraint_minimum_image(reference_a - reference_o, box, periodic);
    float3 old_c = constraint_minimum_image(reference_b - reference_o, box, periodic);
    float3 step_o = constraint_minimum_image(predicted_o - reference_o, box, periodic);
    float3 step_a = constraint_minimum_image(predicted_a - reference_a, box, periodic);
    float3 step_b = constraint_minimum_image(predicted_b - reference_b, box, periodic);

    float mass_o = masses[oxygen];
    float mass_a = masses[hydrogen_a];
    float mass_b = masses[hydrogen_b];
    float total_mass = mass_o + mass_a + mass_b;
    float3 center = (
        mass_o * step_o
        + mass_a * (old_b + step_a)
        + mass_b * (old_c + step_b)
    ) / total_mass;
    float3 centered_o = step_o - center;
    float3 centered_a = old_b + step_a - center;
    float3 centered_b = old_c + step_b - center;

    float3 axis_z = constraint_safe_normalize(cross(old_b, old_c));
    float3 axis_x = constraint_safe_normalize(cross(centered_o, axis_z));
    float3 axis_y = constraint_safe_normalize(cross(axis_z, axis_x));
    float old_b_x = dot(old_b, axis_x);
    float old_b_y = dot(old_b, axis_y);
    float old_c_x = dot(old_c, axis_x);
    float old_c_y = dot(old_c, axis_y);
    float centered_o_z = dot(centered_o, axis_z);
    float centered_a_x = dot(centered_a, axis_x);
    float centered_a_y = dot(centered_a, axis_y);
    float centered_a_z = dot(centered_a, axis_z);
    float centered_b_x = dot(centered_b, axis_x);
    float centered_b_y = dot(centered_b, axis_y);
    float centered_b_z = dot(centered_b, axis_z);

    float half_hh = 0.5f * geometry[1];
    float oxygen_to_h_axis = sqrt(max(geometry[0] * geometry[0] - half_hh * half_hh, 0.0f));
    float oxygen_radius = oxygen_to_h_axis * (mass_a + mass_b) / total_mass;
    float hydrogen_radius = oxygen_to_h_axis - oxygen_radius;
    float sin_phi = clamp(centered_o_z / oxygen_radius, -1.0f, 1.0f);
    float cos_phi = sqrt(max(1.0f - sin_phi * sin_phi, 0.0f));
    float sin_psi = clamp(
        (centered_a_z - centered_b_z) / max(2.0f * half_hh * cos_phi, 1.0e-20f),
        -1.0f,
        1.0f
    );
    float cos_psi = sqrt(max(1.0f - sin_psi * sin_psi, 0.0f));

    float oxygen_y = oxygen_radius * cos_phi;
    float hydrogen_x = -half_hh * cos_psi;
    float hydrogen_a_y = -hydrogen_radius * cos_phi - half_hh * sin_psi * sin_phi;
    float hydrogen_b_y = -hydrogen_radius * cos_phi + half_hh * sin_psi * sin_phi;
    float hydrogen_x_squared = hydrogen_x * hydrogen_x;
    float current_hh_squared = 4.0f * hydrogen_x_squared
        + (hydrogen_a_y - hydrogen_b_y) * (hydrogen_a_y - hydrogen_b_y)
        + (centered_a_z - centered_b_z) * (centered_a_z - centered_b_z);
    float delta_x = 2.0f * hydrogen_x + sqrt(max(
        4.0f * hydrogen_x_squared - current_hh_squared + geometry[1] * geometry[1],
        0.0f
    ));
    hydrogen_x -= 0.5f * delta_x;

    float alpha = hydrogen_x * (old_b_x - old_c_x)
        + old_b_y * hydrogen_a_y + old_c_y * hydrogen_b_y;
    float beta = hydrogen_x * (old_c_y - old_b_y)
        + old_b_x * hydrogen_a_y + old_c_x * hydrogen_b_y;
    float gamma = old_b_x * centered_a_y - centered_a_x * old_b_y
        + old_c_x * centered_b_y - centered_b_x * old_c_y;
    float alpha_beta_squared = alpha * alpha + beta * beta;
    float sin_theta = (
        alpha * gamma - beta * sqrt(max(alpha_beta_squared - gamma * gamma, 0.0f))
    ) / max(alpha_beta_squared, 1.0e-20f);
    sin_theta = clamp(sin_theta, -1.0f, 1.0f);
    float cos_theta = sqrt(max(1.0f - sin_theta * sin_theta, 0.0f));

    float oxygen_x = -oxygen_y * sin_theta;
    oxygen_y *= cos_theta;
    float hydrogen_a_x = hydrogen_x * cos_theta - hydrogen_a_y * sin_theta;
    hydrogen_a_y = hydrogen_x * sin_theta + hydrogen_a_y * cos_theta;
    float hydrogen_b_x = -hydrogen_x * cos_theta - hydrogen_b_y * sin_theta;
    hydrogen_b_y = -hydrogen_x * sin_theta + hydrogen_b_y * cos_theta;

    float3 projected_o = reference_o + center
        + oxygen_x * axis_x + oxygen_y * axis_y + centered_o_z * axis_z;
    float3 projected_a = reference_o + center
        + hydrogen_a_x * axis_x + hydrogen_a_y * axis_y + centered_a_z * axis_z;
    float3 projected_b = reference_o + center
        + hydrogen_b_x * axis_x + hydrogen_b_y * axis_y + centered_b_z * axis_z;
    float3 delta_o = projected_o - predicted_o;
    float3 delta_a = projected_a - predicted_a;
    float3 delta_b = projected_b - predicted_b;
    uint output = 9 * water;
    deltas[output + 0] = delta_o.x;
    deltas[output + 1] = delta_o.y;
    deltas[output + 2] = delta_o.z;
    deltas[output + 3] = delta_a.x;
    deltas[output + 4] = delta_a.y;
    deltas[output + 5] = delta_a.z;
    deltas[output + 6] = delta_b.x;
    deltas[output + 7] = delta_b.y;
    deltas[output + 8] = delta_b.z;
"""

_SETTLE_WATER_VELOCITY_SOURCE = r"""
    uint water = thread_position_in_grid.x;
    if (water >= (uint)params[0]) {
        return;
    }
    int oxygen = waters[3 * water + 0];
    int hydrogen_a = waters[3 * water + 1];
    int hydrogen_b = waters[3 * water + 2];
    int periodic = params[1];

    float3 position_o = constraint_load3(positions, oxygen);
    float3 position_a = constraint_load3(positions, hydrogen_a);
    float3 position_b = constraint_load3(positions, hydrogen_b);
    float3 velocity_o = constraint_load3(velocities, oxygen);
    float3 velocity_a = constraint_load3(velocities, hydrogen_a);
    float3 velocity_b = constraint_load3(velocities, hydrogen_b);
    float3 q_oh_a = constraint_minimum_image(position_o - position_a, box, periodic);
    float3 q_oh_b = constraint_minimum_image(position_o - position_b, box, periodic);
    float3 q_hh = constraint_minimum_image(position_a - position_b, box, periodic);

    float inverse_o = 1.0f / masses[oxygen];
    float inverse_a = 1.0f / masses[hydrogen_a];
    float inverse_b = 1.0f / masses[hydrogen_b];
    float dot_oh = dot(q_oh_a, q_oh_b);
    float dot_a_hh = dot(q_oh_a, q_hh);
    float dot_b_hh = dot(q_oh_b, q_hh);
    float3 row_0 = float3(
        (inverse_o + inverse_a) * dot(q_oh_a, q_oh_a),
        inverse_o * dot_oh,
        -inverse_a * dot_a_hh
    );
    float3 row_1 = float3(
        inverse_o * dot_oh,
        (inverse_o + inverse_b) * dot(q_oh_b, q_oh_b),
        inverse_b * dot_b_hh
    );
    float3 row_2 = float3(
        -inverse_a * dot_a_hh,
        inverse_b * dot_b_hh,
        (inverse_a + inverse_b) * dot(q_hh, q_hh)
    );
    float3 rhs = -float3(
        dot(q_oh_a, velocity_o - velocity_a),
        dot(q_oh_b, velocity_o - velocity_b),
        dot(q_hh, velocity_a - velocity_b)
    );
    float3 cross_12 = cross(row_1, row_2);
    float determinant = dot(row_0, cross_12);
    float safe_determinant = fabs(determinant) > 1.0e-20f ? determinant : 1.0f;
    float3 multipliers = (
        rhs.x * cross_12
        + rhs.y * cross(row_2, row_0)
        + rhs.z * cross(row_0, row_1)
    ) / safe_determinant;
    float3 correction_o = inverse_o * (
        multipliers.x * q_oh_a + multipliers.y * q_oh_b
    );
    float3 correction_a = inverse_a * (
        -multipliers.x * q_oh_a + multipliers.z * q_hh
    );
    float3 correction_b = inverse_b * (
        -multipliers.y * q_oh_b - multipliers.z * q_hh
    );
    uint output = 9 * water;
    deltas[output + 0] = correction_o.x;
    deltas[output + 1] = correction_o.y;
    deltas[output + 2] = correction_o.z;
    deltas[output + 3] = correction_a.x;
    deltas[output + 4] = correction_a.y;
    deltas[output + 5] = correction_a.z;
    deltas[output + 6] = correction_b.x;
    deltas[output + 7] = correction_b.y;
    deltas[output + 8] = correction_b.z;
"""

_SHAKE_CLUSTER_POSITION_SOURCE = r"""
    uint cluster = thread_position_in_grid.x;
    if (cluster >= (uint)params[0]) {
        return;
    }

    int peripheral_count = peripheral_counts[cluster];
    int atom[4];
    float x[4];
    float y[4];
    float z[4];
    float base_x[4];
    float base_y[4];
    float base_z[4];
    float inverse_mass[4];
    for (int slot = 0; slot < 4; slot++) {
        atom[slot] = cluster_atoms[4 * cluster + slot];
        if (slot <= peripheral_count) {
            int index = atom[slot];
            x[slot] = predicted_positions[3 * index + 0];
            y[slot] = predicted_positions[3 * index + 1];
            z[slot] = predicted_positions[3 * index + 2];
            base_x[slot] = x[slot];
            base_y[slot] = y[slot];
            base_z[slot] = z[slot];
            inverse_mass[slot] = 1.0f / masses[index];
        } else {
            x[slot] = 0.0f;
            y[slot] = 0.0f;
            z[slot] = 0.0f;
            base_x[slot] = 0.0f;
            base_y[slot] = 0.0f;
            base_z[slot] = 0.0f;
            inverse_mass[slot] = 0.0f;
        }
    }

    float reference_x[3];
    float reference_y[3];
    float reference_z[3];
    for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
        int slot = peripheral + 1;
        int center_atom = atom[0];
        int outer_atom = atom[slot];
        float dx = reference_positions[3 * center_atom + 0]
            - reference_positions[3 * outer_atom + 0];
        float dy = reference_positions[3 * center_atom + 1]
            - reference_positions[3 * outer_atom + 1];
        float dz = reference_positions[3 * center_atom + 2]
            - reference_positions[3 * outer_atom + 2];
        if (params[2] != 0) {
            dx -= box[0] * rint(dx / box[0]);
            dy -= box[1] * rint(dy / box[1]);
            dz -= box[2] * rint(dz / box[2]);
        }
        reference_x[peripheral] = dx;
        reference_y[peripheral] = dy;
        reference_z[peripheral] = dz;
    }

    float target_squared = distances[cluster] * distances[cluster];
    for (int iteration = 0; iteration < params[1]; iteration++) {
        float center_delta_x = 0.0f;
        float center_delta_y = 0.0f;
        float center_delta_z = 0.0f;
        for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
            int slot = peripheral + 1;
            float dx = x[0] - x[slot];
            float dy = y[0] - y[slot];
            float dz = z[0] - z[slot];
            if (params[2] != 0) {
                dx -= box[0] * rint(dx / box[0]);
                dy -= box[1] * rint(dy / box[1]);
                dz -= box[2] * rint(dz / box[2]);
            }
            float reference_dot = dx * reference_x[peripheral]
                + dy * reference_y[peripheral]
                + dz * reference_z[peripheral];
            float denominator = 2.0f
                * (inverse_mass[0] + inverse_mass[slot])
                * reference_dot;
            if (fabs(denominator) <= 1.0e-20f) {
                denominator = denominator < 0.0f ? -1.0e-20f : 1.0e-20f;
            }
            float error_squared = target_squared - (dx * dx + dy * dy + dz * dz);
            float multiplier = error_squared / denominator;
            float correction_x = multiplier * reference_x[peripheral];
            float correction_y = multiplier * reference_y[peripheral];
            float correction_z = multiplier * reference_z[peripheral];
            center_delta_x += inverse_mass[0] * correction_x;
            center_delta_y += inverse_mass[0] * correction_y;
            center_delta_z += inverse_mass[0] * correction_z;
            x[slot] -= inverse_mass[slot] * correction_x;
            y[slot] -= inverse_mass[slot] * correction_y;
            z[slot] -= inverse_mass[slot] * correction_z;
        }
        x[0] += center_delta_x;
        y[0] += center_delta_y;
        z[0] += center_delta_z;
    }

    for (int slot = 0; slot < 4; slot++) {
        uint output = 12 * cluster + 3 * slot;
        if (slot <= peripheral_count) {
            deltas[output + 0] = x[slot] - base_x[slot];
            deltas[output + 1] = y[slot] - base_y[slot];
            deltas[output + 2] = z[slot] - base_z[slot];
        } else {
            deltas[output + 0] = 0.0f;
            deltas[output + 1] = 0.0f;
            deltas[output + 2] = 0.0f;
        }
    }
"""

_SHAKE_CLUSTER_VELOCITY_SOURCE = r"""
    uint cluster = thread_position_in_grid.x;
    if (cluster >= (uint)params[0]) {
        return;
    }

    int peripheral_count = peripheral_counts[cluster];
    int center_atom = cluster_atoms[4 * cluster];
    float3 center_velocity = constraint_load3(velocities, center_atom);
    float inverse_center = 1.0f / masses[center_atom];
    float3 unit[3] = {
        float3(0.0f),
        float3(0.0f),
        float3(0.0f)
    };
    float3 outer_velocity[3] = {
        float3(0.0f),
        float3(0.0f),
        float3(0.0f)
    };
    float inverse_outer[3] = {0.0f, 0.0f, 0.0f};
    float relative[3] = {0.0f, 0.0f, 0.0f};
    for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
        int outer_atom = cluster_atoms[4 * cluster + peripheral + 1];
        float3 displacement = constraint_minimum_image(
            constraint_load3(positions, center_atom)
                - constraint_load3(positions, outer_atom),
            box,
            params[2]
        );
        unit[peripheral] = constraint_safe_normalize(displacement);
        outer_velocity[peripheral] = constraint_load3(velocities, outer_atom);
        inverse_outer[peripheral] = 1.0f / masses[outer_atom];
        relative[peripheral] = dot(
            center_velocity - outer_velocity[peripheral],
            unit[peripheral]
        );
    }

    float coupling_01 = inverse_center * dot(unit[0], unit[1]);
    float coupling_02 = inverse_center * dot(unit[0], unit[2]);
    float coupling_12 = inverse_center * dot(unit[1], unit[2]);
    float3 row_0 = float3(
        peripheral_count > 0 ? inverse_center + inverse_outer[0] : 1.0f,
        peripheral_count > 1 ? coupling_01 : 0.0f,
        peripheral_count > 2 ? coupling_02 : 0.0f
    );
    float3 row_1 = float3(
        peripheral_count > 1 ? coupling_01 : 0.0f,
        peripheral_count > 1 ? inverse_center + inverse_outer[1] : 1.0f,
        peripheral_count > 2 ? coupling_12 : 0.0f
    );
    float3 row_2 = float3(
        peripheral_count > 2 ? coupling_02 : 0.0f,
        peripheral_count > 2 ? coupling_12 : 0.0f,
        peripheral_count > 2 ? inverse_center + inverse_outer[2] : 1.0f
    );
    float3 rhs = -float3(relative[0], relative[1], relative[2]);
    float3 cross_12 = cross(row_1, row_2);
    float determinant = dot(row_0, cross_12);
    float safe_determinant = fabs(determinant) > 1.0e-20f ? determinant : 1.0f;
    float3 multipliers = (
        rhs.x * cross_12
        + rhs.y * cross(row_2, row_0)
        + rhs.z * cross(row_0, row_1)
    ) / safe_determinant;

    float3 center_delta = float3(0.0f);
    for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
        center_delta += inverse_center * multipliers[peripheral] * unit[peripheral];
    }
    uint center_output = 12 * cluster;
    deltas[center_output + 0] = center_delta.x;
    deltas[center_output + 1] = center_delta.y;
    deltas[center_output + 2] = center_delta.z;
    for (int peripheral = 0; peripheral < 3; peripheral++) {
        uint output = center_output + 3 * (peripheral + 1);
        float3 outer_delta = peripheral < peripheral_count
            ? -inverse_outer[peripheral] * multipliers[peripheral] * unit[peripheral]
            : float3(0.0f);
        deltas[output + 0] = outer_delta.x;
        deltas[output + 1] = outer_delta.y;
        deltas[output + 2] = outer_delta.z;
    }
"""

_LANGEVIN_BAOAB_DRIFT_SOURCE = r"""
    uint atom = thread_position_in_grid.x;
    if (atom >= (uint)counts[0]) {
        return;
    }

    uint offset = 3u * atom;
    float half_dt = params[0];
    float velocity_decay = params[1];
    bool wrap = params[2] != 0.0f;
    float acceleration_scale = force_scale_over_mass[atom];
    float thermal = thermal_scale[atom];

    for (uint axis = 0u; axis < 3u; axis++) {
        float velocity_half = velocities[offset + axis]
            + half_dt * acceleration_scale * forces[offset + axis];
        float position = positions[offset + axis] + half_dt * velocity_half;
        if (wrap) {
            float length = box[axis];
            position -= length * floor(position / length);
        }
        float middle_velocity = velocity_decay * velocity_half
            + thermal * noise[offset + axis];
        position += half_dt * middle_velocity;
        if (wrap) {
            float length = box[axis];
            position -= length * floor(position / length);
        }
        next_positions[offset + axis] = position;
        middle_velocities[offset + axis] = middle_velocity;
    }
"""


def _lj_force_kernel():
    """Return the cached fused-LJ Metal kernel, building it on first call."""

    global _kernel_singleton
    if _kernel_singleton is None:
        _kernel_singleton = mx.fast.metal_kernel(
            name="fused_lj_force",
            input_names=["positions", "pairs_i", "pairs_j", "box", "params", "npair"],
            output_names=["forces", "pair_energy"],
            source=_LJ_FORCE_SOURCE,
            atomic_outputs=True,
        )
    return _kernel_singleton


def _parameterized_lj_force_kernel():
    """Return the cached parameterized-LJ Metal kernel."""

    global _parameterized_kernel_singleton
    if _parameterized_kernel_singleton is None:
        _parameterized_kernel_singleton = mx.fast.metal_kernel(
            name="fused_parameterized_lj_force",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "scales",
                "params",
                "counts",
            ],
            output_names=["forces", "pair_energy"],
            source=_PARAMETERIZED_LJ_FORCE_SOURCE,
            atomic_outputs=True,
        )
    return _parameterized_kernel_singleton


def _parameterized_pme_direct_kernel():
    """Return the cached fused LJ/PME-direct Metal kernel."""

    global _pme_direct_kernel_singleton
    if _pme_direct_kernel_singleton is None:
        _pme_direct_kernel_singleton = mx.fast.metal_kernel(
            name="fused_parameterized_pme_direct",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "charges",
                "lj_scales",
                "params",
                "npair",
            ],
            output_names=[
                "forces",
                "pair_lj_energy",
                "pair_coulomb_energy",
            ],
            source=_PARAMETERIZED_PME_DIRECT_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _pme_direct_kernel_singleton


def _parameterized_pme_direct_force_only_kernel():
    """Return the cached force-only fused LJ/PME-direct Metal kernel."""

    global _pme_direct_force_only_kernel_singleton
    if _pme_direct_force_only_kernel_singleton is None:
        _pme_direct_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="fused_parameterized_pme_direct_force_only",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "charges",
                "lj_scales",
                "params",
                "npair",
            ],
            output_names=["forces"],
            source=_PARAMETERIZED_PME_DIRECT_FORCE_ONLY_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _pme_direct_force_only_kernel_singleton


def _prepared_pme_direct_force_only_kernel():
    """Return the cached force-only kernel for setup-precomputed invariants."""

    global _prepared_pme_direct_force_only_kernel_singleton
    if _prepared_pme_direct_force_only_kernel_singleton is None:
        _prepared_pme_direct_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="prepared_parameterized_pme_direct_force_only",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "charges",
                "lj_scales",
                "params",
                "npair",
            ],
            output_names=["forces"],
            source=_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _prepared_pme_direct_force_only_kernel_singleton


def _tile_pme_direct_force_only_kernel():
    """Return the cached spatial 4x4 tile direct-force kernel."""

    global _tile_pme_direct_force_only_kernel_singleton
    if _tile_pme_direct_force_only_kernel_singleton is None:
        _tile_pme_direct_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="spatial_tile_prepared_parameterized_pme_direct_force_only",
            input_names=[
                "positions",
                "atom_blocks",
                "tile_blocks",
                "lj_enabled_mask",
                "lj_one_four_mask",
                "force_columns",
                "force_group_starts",
                "force_group_counts",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "params",
                "ngroups",
            ],
            output_names=["forces"],
            source=_TILE_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _tile_pme_direct_force_only_kernel_singleton


def _tile_nbfix_pme_direct_force_only_kernel():
    """Return the cached NBFIX-aware spatial 4x4 direct-force kernel."""

    global _tile_nbfix_pme_direct_force_only_kernel_singleton
    if _tile_nbfix_pme_direct_force_only_kernel_singleton is None:
        _tile_nbfix_pme_direct_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="spatial_tile_nbfix_parameterized_pme_direct_force_only",
            input_names=[
                "positions",
                "atom_blocks",
                "tile_blocks",
                "lj_enabled_mask",
                "lj_one_four_mask",
                "force_columns",
                "force_group_starts",
                "force_group_counts",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "atom_type_ids",
                "nbfix_sigma",
                "nbfix_epsilon",
                "params",
                "ngroups",
            ],
            output_names=["forces"],
            source=_TILE_NBFIX_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _tile_nbfix_pme_direct_force_only_kernel_singleton


def _tile_pme_direct_kernel():
    """Return the cached spatial 4x4 tile direct energy/force kernel."""

    global _tile_pme_direct_kernel_singleton
    if _tile_pme_direct_kernel_singleton is None:
        _tile_pme_direct_kernel_singleton = mx.fast.metal_kernel(
            name="spatial_tile_prepared_parameterized_pme_direct",
            input_names=[
                "positions",
                "atom_blocks",
                "tile_blocks",
                "lj_enabled_mask",
                "lj_one_four_mask",
                "force_columns",
                "force_group_starts",
                "force_group_counts",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "params",
                "ngroups",
            ],
            output_names=["forces", "group_lj_energy", "group_coulomb_energy"],
            source=_TILE_PREPARED_PME_DIRECT_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _tile_pme_direct_kernel_singleton


def _tile_nbfix_pme_direct_kernel():
    """Return the cached NBFIX-aware spatial 4x4 energy/force kernel."""

    global _tile_nbfix_pme_direct_kernel_singleton
    if _tile_nbfix_pme_direct_kernel_singleton is None:
        _tile_nbfix_pme_direct_kernel_singleton = mx.fast.metal_kernel(
            name="spatial_tile_nbfix_parameterized_pme_direct",
            input_names=[
                "positions",
                "atom_blocks",
                "tile_blocks",
                "lj_enabled_mask",
                "lj_one_four_mask",
                "force_columns",
                "force_group_starts",
                "force_group_counts",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "atom_type_ids",
                "nbfix_sigma",
                "nbfix_epsilon",
                "params",
                "ngroups",
            ],
            output_names=["forces", "group_lj_energy", "group_coulomb_energy"],
            source=_TILE_NBFIX_PREPARED_PME_DIRECT_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _tile_nbfix_pme_direct_kernel_singleton


def _interaction32_pack_kernel():
    """Return the cached 32-atom record-packing Metal kernel."""

    global _interaction32_pack_kernel_singleton
    if _interaction32_pack_kernel_singleton is None:
        _interaction32_pack_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_pack_atom_records",
            input_names=[
                "positions",
                "atom_order",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "counts",
            ],
            output_names=["packed_posq", "packed_lj"],
            source=_INTERACTION32_PACK_SOURCE,
        )
    return _interaction32_pack_kernel_singleton


def _interaction32_block_geometry_kernel():
    """Return the packed 32-atom block-bounds Metal kernel."""

    global _interaction32_block_geometry_kernel_singleton
    if _interaction32_block_geometry_kernel_singleton is None:
        _interaction32_block_geometry_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_block_geometry",
            input_names=["positions", "atom_order", "box", "counts"],
            output_names=["center_radius", "half_extent"],
            source=_INTERACTION32_BLOCK_GEOMETRY_SOURCE,
        )
    return _interaction32_block_geometry_kernel_singleton


def _interaction32_ordinary_count_kernel():
    """Return the packed 32-atom ordinary-membership count kernel."""

    global _interaction32_ordinary_count_kernel_singleton
    if _interaction32_ordinary_count_kernel_singleton is None:
        _interaction32_ordinary_count_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_ordinary_mode_counts",
            input_names=[
                "positions",
                "atom_order",
                "center_radius",
                "half_extent",
                "block_traversal",
                "special_pair_words",
                "box",
                "params",
                "counts",
            ],
            output_names=["mode_counts"],
            source=_INTERACTION32_ORDINARY_COUNT_SOURCE,
            header=_INTERACTION32_ORDINARY_COMMON_SOURCE,
        )
    return _interaction32_ordinary_count_kernel_singleton


def _interaction32_ordinary_cached_count_kernel():
    """Return the count kernel that retains packed two-bit membership modes."""

    global _interaction32_ordinary_cached_count_kernel_singleton
    if _interaction32_ordinary_cached_count_kernel_singleton is None:
        _interaction32_ordinary_cached_count_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_ordinary_cached_mode_counts",
            input_names=[
                "positions",
                "atom_order",
                "center_radius",
                "half_extent",
                "block_traversal",
                "special_pair_words",
                "box",
                "params",
                "counts",
            ],
            output_names=["mode_counts", "mode_words"],
            source=(
                "#define MLX_ATOMISTIC_INTERACTION32_RETAIN_MODES 1\n"
                + _INTERACTION32_ORDINARY_COUNT_SOURCE
            ),
            header=_INTERACTION32_ORDINARY_COMMON_SOURCE,
        )
    return _interaction32_ordinary_cached_count_kernel_singleton


def _interaction32_ordinary_scatter_kernel():
    """Return the compact packed 32-atom ordinary-membership scatter kernel."""

    global _interaction32_ordinary_scatter_kernel_singleton
    if _interaction32_ordinary_scatter_kernel_singleton is None:
        _interaction32_ordinary_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_ordinary_mode_scatter",
            input_names=[
                "positions",
                "atom_order",
                "center_radius",
                "half_extent",
                "block_traversal",
                "special_pair_words",
                "mode_tile_counts",
                "mode_tile_prefix",
                "box",
                "params",
                "counts",
            ],
            output_names=[
                "ordinary_left_blocks",
                "ordinary_right_atoms",
                "ordinary_half_modes",
            ],
            source=_INTERACTION32_ORDINARY_SCATTER_SOURCE,
            header=_INTERACTION32_ORDINARY_COMMON_SOURCE,
        )
    return _interaction32_ordinary_scatter_kernel_singleton


def _interaction32_ordinary_cached_scatter_kernel():
    """Return the scatter kernel that decodes retained membership modes."""

    global _interaction32_ordinary_cached_scatter_kernel_singleton
    if _interaction32_ordinary_cached_scatter_kernel_singleton is None:
        _interaction32_ordinary_cached_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_ordinary_cached_mode_scatter",
            input_names=[
                "block_traversal",
                "mode_words",
                "mode_tile_counts",
                "mode_tile_prefix",
                "counts",
            ],
            output_names=[
                "ordinary_left_blocks",
                "ordinary_right_atoms",
                "ordinary_half_modes",
            ],
            source=_INTERACTION32_ORDINARY_CACHED_SCATTER_SOURCE,
            header=_INTERACTION32_ORDINARY_COMMON_SOURCE,
        )
    return _interaction32_ordinary_cached_scatter_kernel_singleton


def _interaction32_outer_inner_mode_count_kernel():
    """Return the outer-schedule inner-membership count kernel."""

    global _interaction32_outer_inner_mode_count_kernel_singleton
    if _interaction32_outer_inner_mode_count_kernel_singleton is None:
        _interaction32_outer_inner_mode_count_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_outer_inner_mode_count",
            input_names=[
                "positions",
                "atom_order",
                "block_traversal",
                "outer_right_atoms",
                "outer_tile_counts",
                "outer_tile_prefix",
                "box",
                "params",
                "counts",
            ],
            output_names=["mode_counts", "cached_modes"],
            source=_INTERACTION32_OUTER_INNER_MODE_COUNT_SOURCE,
            header=_INTERACTION32_ORDINARY_COMMON_SOURCE,
        )
    return _interaction32_outer_inner_mode_count_kernel_singleton


def _interaction32_outer_inner_mode_scatter_kernel():
    """Return the cached outer-to-inner mode scatter kernel."""

    global _interaction32_outer_inner_mode_scatter_kernel_singleton
    if _interaction32_outer_inner_mode_scatter_kernel_singleton is None:
        _interaction32_outer_inner_mode_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_outer_inner_mode_scatter",
            input_names=[
                "block_traversal",
                "outer_right_atoms",
                "cached_modes",
                "outer_tile_counts",
                "outer_tile_prefix",
                "inner_tile_counts",
                "inner_tile_prefix",
                "counts",
            ],
            output_names=[
                "inner_left_blocks",
                "inner_right_atoms",
                "inner_half_modes",
            ],
            source=_INTERACTION32_OUTER_INNER_MODE_SCATTER_SOURCE,
        )
    return _interaction32_outer_inner_mode_scatter_kernel_singleton


def _interaction32_special_pair_words_kernel():
    """Return the atomic special-block bitset Metal kernel."""

    global _interaction32_special_pair_words_kernel_singleton
    if _interaction32_special_pair_words_kernel_singleton is None:
        _interaction32_special_pair_words_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_special_pair_words",
            input_names=["special_codes", "special_unique", "counts"],
            output_names=["special_pair_words"],
            source=_INTERACTION32_SPECIAL_PAIR_WORDS_SOURCE,
            atomic_outputs=True,
        )
    return _interaction32_special_pair_words_kernel_singleton


def _interaction32_special_block_scatter_kernel():
    """Return the compact special-block scatter Metal kernel."""

    global _interaction32_special_block_scatter_kernel_singleton
    if _interaction32_special_block_scatter_kernel_singleton is None:
        _interaction32_special_block_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_special_block_scatter",
            input_names=[
                "special_codes",
                "special_unique",
                "special_prefix",
                "counts",
            ],
            output_names=["special_blocks"],
            source=_INTERACTION32_SPECIAL_BLOCK_SCATTER_SOURCE,
        )
    return _interaction32_special_block_scatter_kernel_singleton


def _interaction32_special_work_kernel():
    """Return the two-half special-topology work Metal kernel."""

    global _interaction32_special_work_kernel_singleton
    if _interaction32_special_work_kernel_singleton is None:
        _interaction32_special_work_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_special_work",
            input_names=[
                "atom_order",
                "special_blocks",
                "topology_offsets",
                "topology_neighbors",
                "topology_classes",
                "counts",
            ],
            output_names=[
                "special_left_blocks",
                "special_left_slices",
                "special_right_atoms",
                "special_lj_enabled",
                "special_lj_one_four",
                "special_diagonal",
            ],
            source=_INTERACTION32_SPECIAL_WORK_SOURCE,
        )
    return _interaction32_special_work_kernel_singleton


def _interaction32_force_kernel():
    """Return the cached SIMD-native 32-atom direct-force Metal kernel."""

    global _interaction32_force_kernel_singleton
    if _interaction32_force_kernel_singleton is None:
        _interaction32_force_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_pme_direct_force",
            input_names=[
                "packed_posq",
                "packed_lj",
                "atom_order",
                "ordinary_left_blocks",
                "ordinary_left_slices",
                "ordinary_right_atoms",
                "ordinary_group_starts",
                "ordinary_group_counts",
                "special_left_blocks",
                "special_left_slices",
                "special_right_atoms",
                "special_work_lj_enabled",
                "special_work_lj_one_four",
                "special_diagonal",
                "box",
                "params",
                "counts",
            ],
            output_names=["ordered_forces"],
            source=_INTERACTION32_FORCE_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _interaction32_force_kernel_singleton


def _interaction32_canonical_force_kernel():
    """Return the cached direct-canonical 32-atom force Metal kernel."""

    global _interaction32_canonical_force_kernel_singleton
    if _interaction32_canonical_force_kernel_singleton is None:
        _interaction32_canonical_force_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_pme_direct_canonical_force",
            input_names=[
                "positions",
                "atom_order",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "ordinary_left_blocks",
                "ordinary_left_slices",
                "ordinary_right_atoms",
                "ordinary_group_starts",
                "ordinary_group_counts",
                "special_left_blocks",
                "special_left_slices",
                "special_right_atoms",
                "special_work_lj_enabled",
                "special_work_lj_one_four",
                "special_diagonal",
                "box",
                "params",
                "counts",
            ],
            output_names=["ordered_forces"],
            source=(
                "#define MLX_ATOMISTIC_INTERACTION32_CANONICAL 1\n" + _INTERACTION32_FORCE_SOURCE
            ),
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _interaction32_canonical_force_kernel_singleton


def _interaction32_fused_half_canonical_force_kernel():
    """Return the cached half-membership-aware canonical force Metal kernel."""

    global _interaction32_fused_half_canonical_force_kernel_singleton
    if _interaction32_fused_half_canonical_force_kernel_singleton is None:
        _interaction32_fused_half_canonical_force_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_fused_half_pme_direct_canonical_force",
            input_names=[
                "positions",
                "atom_order",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "ordinary_left_blocks",
                "ordinary_right_atoms",
                "ordinary_half_modes",
                "ordinary_group_starts",
                "ordinary_group_counts",
                "special_left_blocks",
                "special_left_slices",
                "special_right_atoms",
                "special_work_lj_enabled",
                "special_work_lj_one_four",
                "special_diagonal",
                "box",
                "params",
                "counts",
            ],
            output_names=["ordered_forces"],
            source=(
                "#define MLX_ATOMISTIC_INTERACTION32_CANONICAL 1\n"
                "#define MLX_ATOMISTIC_INTERACTION32_FUSED_HALF 1\n"
                "#define MLX_ATOMISTIC_INTERACTION32_SHARED_EWALD_EXP 1\n"
                "#define MLX_ATOMISTIC_INTERACTION32_ACTIVE_COMPACTION 1\n"
                + _INTERACTION32_FORCE_SOURCE
            ),
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _interaction32_fused_half_canonical_force_kernel_singleton


def _interaction32_fused_half_nbfix_canonical_force_kernel():
    """Return the NBFIX-aware fused-half canonical force Metal kernel."""

    global _interaction32_fused_half_nbfix_canonical_force_kernel_singleton
    if _interaction32_fused_half_nbfix_canonical_force_kernel_singleton is None:
        _interaction32_fused_half_nbfix_canonical_force_kernel_singleton = (
            mx.fast.metal_kernel(
                name="interaction32_fused_half_nbfix_pme_direct_canonical_force",
                input_names=[
                    "positions",
                    "atom_order",
                    "half_sigma",
                    "sqrt_epsilon",
                    "charges",
                    "atom_type_ids",
                    "nbfix_sigma",
                    "nbfix_epsilon",
                    "ordinary_left_blocks",
                    "ordinary_right_atoms",
                    "ordinary_half_modes",
                    "ordinary_group_starts",
                    "ordinary_group_counts",
                    "special_left_blocks",
                    "special_left_slices",
                    "special_right_atoms",
                    "special_work_lj_enabled",
                    "special_work_lj_one_four",
                    "special_diagonal",
                    "box",
                    "params",
                    "counts",
                ],
                output_names=["ordered_forces"],
                source=(
                    "#define MLX_ATOMISTIC_INTERACTION32_CANONICAL 1\n"
                    "#define MLX_ATOMISTIC_INTERACTION32_FUSED_HALF 1\n"
                    "#define MLX_ATOMISTIC_INTERACTION32_SHARED_EWALD_EXP 1\n"
                    "#define MLX_ATOMISTIC_INTERACTION32_ACTIVE_COMPACTION 1\n"
                    "#define MLX_ATOMISTIC_NBFIX 1\n"
                    + _INTERACTION32_FORCE_SOURCE
                ),
                header=_ERF_HEADER,
                atomic_outputs=True,
            )
        )
    return _interaction32_fused_half_nbfix_canonical_force_kernel_singleton


def _interaction32_scatter_kernel():
    """Return the cached unique-writer ordered-force scatter kernel."""

    global _interaction32_scatter_kernel_singleton
    if _interaction32_scatter_kernel_singleton is None:
        _interaction32_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="interaction32_scatter_ordered_force",
            input_names=["ordered_forces", "atom_order", "counts"],
            output_names=["forces"],
            source=_INTERACTION32_SCATTER_SOURCE,
        )
    return _interaction32_scatter_kernel_singleton


def _owner_compute32_force_kernel():
    """Return the cached no-atomic owner-computes direct-force Metal kernel."""

    global _owner_compute32_force_kernel_singleton
    if _owner_compute32_force_kernel_singleton is None:
        _owner_compute32_force_kernel_singleton = mx.fast.metal_kernel(
            name="owner_compute32_pme_direct_force",
            input_names=[
                "positions",
                "atom_order",
                "owner_offsets",
                "right_atoms",
                "topology_offsets",
                "topology_neighbors",
                "topology_classes",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "params",
                "counts",
            ],
            output_names=["forces"],
            source=_OWNER_COMPUTE32_FORCE_SOURCE,
            header=_ERF_HEADER,
        )
    return _owner_compute32_force_kernel_singleton


def _sparse_pme_correction_force_only_kernel():
    """Return the cached fused sparse PME-correction force kernel."""

    global _sparse_pme_correction_force_only_kernel_singleton
    if _sparse_pme_correction_force_only_kernel_singleton is None:
        _sparse_pme_correction_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="sparse_pme_correction_force_only",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "charge_products",
                "lj_sigma",
                "lj_epsilon",
                "params",
                "npair",
            ],
            output_names=["forces"],
            source=_SPARSE_PME_CORRECTION_FORCE_ONLY_SOURCE,
            atomic_outputs=True,
        )
    return _sparse_pme_correction_force_only_kernel_singleton


def _parameterized_pme_direct_virial_kernel():
    """Return the cached diagnostic fused LJ/PME-direct virial kernel."""

    global _pme_direct_virial_kernel_singleton
    if _pme_direct_virial_kernel_singleton is None:
        _pme_direct_virial_kernel_singleton = mx.fast.metal_kernel(
            name="fused_parameterized_pme_direct_virial",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "charges",
                "lj_scales",
                "params",
                "npair",
            ],
            output_names=[
                "forces",
                "pair_lj_energy",
                "pair_coulomb_energy",
                "pair_virial",
            ],
            source=_PARAMETERIZED_PME_DIRECT_VIRIAL_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _pme_direct_virial_kernel_singleton


def _pme_cutoff_correction_virial_kernel():
    """Return the cached fused PME cutoff-correction virial kernel."""

    global _pme_cutoff_correction_virial_kernel_singleton
    if _pme_cutoff_correction_virial_kernel_singleton is None:
        _pme_cutoff_correction_virial_kernel_singleton = mx.fast.metal_kernel(
            name="fused_pme_cutoff_correction_virial",
            input_names=[
                "positions",
                "centers",
                "pairs_i",
                "pairs_j",
                "box",
                "sigma",
                "epsilon",
                "charges",
                "lj_scales",
                "params",
                "npair",
            ],
            output_names=["pair_correction"],
            source=_PME_CUTOFF_CORRECTION_VIRIAL_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _pme_cutoff_correction_virial_kernel_singleton


def _pme_order5_spread_kernel():
    """Return the cached order-five PME charge-spread Metal kernel."""

    global _pme_order5_spread_kernel_singleton
    if _pme_order5_spread_kernel_singleton is None:
        _pme_order5_spread_kernel_singleton = mx.fast.metal_kernel(
            name="pme_order5_spread",
            input_names=["positions", "charges", "cell", "mesh", "counts"],
            output_names=["charge_grid"],
            source=_PME_ORDER5_SPREAD_SOURCE,
            header=_PME_ORDER5_HEADER,
            atomic_outputs=True,
        )
    return _pme_order5_spread_kernel_singleton


def _pme_order5_interpolate_kernel():
    """Return the cached order-five PME interpolation Metal kernel."""

    global _pme_order5_interpolate_kernel_singleton
    if _pme_order5_interpolate_kernel_singleton is None:
        _pme_order5_interpolate_kernel_singleton = mx.fast.metal_kernel(
            name="pme_order5_interpolate",
            input_names=[
                "positions",
                "charges",
                "potential_grid",
                "cell",
                "mesh",
                "counts",
            ],
            output_names=["atom_energy", "forces"],
            source=_PME_ORDER5_INTERPOLATE_SOURCE,
            header=(_PME_ORDER5_HEADER + "\n#define MLX_ATOMISTIC_PME_WRITE_ENERGY 1\n"),
        )
    return _pme_order5_interpolate_kernel_singleton


def _pme_order5_force_only_kernel():
    """Return the cached order-five PME force-only interpolation kernel."""

    global _pme_order5_force_only_kernel_singleton
    if _pme_order5_force_only_kernel_singleton is None:
        _pme_order5_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="pme_order5_force_only",
            input_names=[
                "positions",
                "charges",
                "potential_grid",
                "cell",
                "mesh",
                "counts",
            ],
            output_names=["forces"],
            source=_PME_ORDER5_INTERPOLATE_SOURCE,
            header=_PME_ORDER5_HEADER,
        )
    return _pme_order5_force_only_kernel_singleton


def _pme_order5_normalized_real_grid_force_only_kernel():
    """Return the cached force kernel consuming a normalized real FFT grid."""

    global _pme_order5_normalized_real_grid_force_only_kernel_singleton
    if _pme_order5_normalized_real_grid_force_only_kernel_singleton is None:
        _pme_order5_normalized_real_grid_force_only_kernel_singleton = (
            mx.fast.metal_kernel(
                name="pme_order5_normalized_real_grid_force_only",
                input_names=[
                    "positions",
                    "charges",
                    "potential_grid",
                    "cell",
                    "mesh",
                    "counts",
                ],
                output_names=["forces"],
                source=_PME_ORDER5_INTERPOLATE_SOURCE,
                header=(
                    _PME_ORDER5_HEADER
                    + "\n#define MLX_ATOMISTIC_PME_NORMALIZED_REAL_GRID 1\n"
                ),
            )
        )
    return _pme_order5_normalized_real_grid_force_only_kernel_singleton


def _pme_order5_complex_grid_force_only_kernel():
    """Return the cached force kernel consuming an unscaled complex FFT grid."""

    global _pme_order5_complex_grid_force_only_kernel_singleton
    if _pme_order5_complex_grid_force_only_kernel_singleton is None:
        _pme_order5_complex_grid_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="pme_order5_complex_grid_force_only",
            input_names=[
                "positions",
                "charges",
                "potential_grid",
                "cell",
                "mesh",
                "counts",
            ],
            output_names=["forces"],
            source=_PME_ORDER5_INTERPOLATE_SOURCE,
            header=(_PME_ORDER5_HEADER + "\n#define MLX_ATOMISTIC_PME_COMPLEX_GRID 1\n"),
        )
    return _pme_order5_complex_grid_force_only_kernel_singleton


def _aligned_topology_lj_scales_kernel():
    """Return the cached topology-scale alignment Metal kernel."""

    global _aligned_topology_lj_scales_kernel_singleton
    if _aligned_topology_lj_scales_kernel_singleton is None:
        _aligned_topology_lj_scales_kernel_singleton = mx.fast.metal_kernel(
            name="aligned_topology_lj_scales",
            input_names=[
                "pairs_i",
                "pairs_j",
                "excluded_i",
                "excluded_j",
                "one_four_i",
                "one_four_j",
                "params",
                "counts",
            ],
            output_names=["scales"],
            source=_ALIGNED_TOPOLOGY_LJ_SCALES_SOURCE,
        )
    return _aligned_topology_lj_scales_kernel_singleton


def _neighbor_pair_cutoff_mask_kernel():
    """Return the cached neighbor cutoff-mask Metal kernel."""

    global _neighbor_pair_cutoff_mask_kernel_singleton
    if _neighbor_pair_cutoff_mask_kernel_singleton is None:
        _neighbor_pair_cutoff_mask_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_pair_cutoff_mask",
            input_names=[
                "positions",
                "pairs_i",
                "pairs_j",
                "box",
                "params",
                "counts",
            ],
            output_names=["close"],
            source=_NEIGHBOR_PAIR_CUTOFF_MASK_SOURCE,
        )
    return _neighbor_pair_cutoff_mask_kernel_singleton


def _neighbor_cell_pair_candidates_kernel():
    """Return the cached spatial neighbor-candidate Metal kernel."""

    global _neighbor_cell_pair_candidates_kernel_singleton
    if _neighbor_cell_pair_candidates_kernel_singleton is None:
        _neighbor_cell_pair_candidates_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_cell_pair_candidates",
            input_names=[
                "sorted_atoms",
                "cell_starts",
                "cell_counts",
                "cell_pairs",
                "task_offsets",
                "counts",
            ],
            output_names=["pairs_i", "pairs_j"],
            source=_NEIGHBOR_CELL_PAIR_CANDIDATES_SOURCE,
        )
    return _neighbor_cell_pair_candidates_kernel_singleton


def _neighbor_pair_ordered_scatter_kernel():
    """Return the cached deterministic neighbor-compaction Metal kernel."""

    global _neighbor_pair_ordered_scatter_kernel_singleton
    if _neighbor_pair_ordered_scatter_kernel_singleton is None:
        _neighbor_pair_ordered_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_pair_ordered_scatter",
            input_names=[
                "pairs_i",
                "pairs_j",
                "close",
                "prefix",
                "counts",
            ],
            output_names=["accepted_i", "accepted_j"],
            source=_NEIGHBOR_PAIR_ORDERED_SCATTER_SOURCE,
        )
    return _neighbor_pair_ordered_scatter_kernel_singleton


def _neighbor_cell_tile_candidates_kernel():
    """Return the cached spatial cell-block tile-emission kernel."""

    global _neighbor_cell_tile_candidates_kernel_singleton
    if _neighbor_cell_tile_candidates_kernel_singleton is None:
        _neighbor_cell_tile_candidates_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_cell_tile_candidates",
            input_names=[
                "cell_block_starts",
                "cell_block_counts",
                "cell_pairs",
                "task_offsets",
                "counts",
            ],
            output_names=["tile_left", "tile_right"],
            source=_NEIGHBOR_CELL_TILE_CANDIDATES_SOURCE,
        )
    return _neighbor_cell_tile_candidates_kernel_singleton


def _neighbor_cell_atom_blocks_kernel():
    """Return the cached device-resident cell-to-atom-block kernel."""

    global _neighbor_cell_atom_blocks_kernel_singleton
    if _neighbor_cell_atom_blocks_kernel_singleton is None:
        _neighbor_cell_atom_blocks_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_cell_atom_blocks",
            input_names=[
                "sorted_atoms",
                "cell_starts",
                "cell_counts",
                "cell_block_starts",
                "cell_block_counts",
                "counts",
            ],
            output_names=["atom_blocks"],
            source=_NEIGHBOR_CELL_ATOM_BLOCKS_SOURCE,
        )
    return _neighbor_cell_atom_blocks_kernel_singleton


def _neighbor_tile_membership_kernel():
    """Return the cached coarse-tile membership and subdivision kernel."""

    global _neighbor_tile_membership_kernel_singleton
    if _neighbor_tile_membership_kernel_singleton is None:
        _neighbor_tile_membership_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_membership",
            input_names=["positions", "atom_blocks", "tile_blocks", "box", "params"],
            output_names=["subtile_mask", "member_counts", "subtile_counts"],
            source=_NEIGHBOR_TILE_MEMBERSHIP_SOURCE,
        )
    return _neighbor_tile_membership_kernel_singleton


def _neighbor_tile_ordered_scatter_kernel():
    """Return the cached non-empty subtile compaction kernel."""

    global _neighbor_tile_ordered_scatter_kernel_singleton
    if _neighbor_tile_ordered_scatter_kernel_singleton is None:
        _neighbor_tile_ordered_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_ordered_scatter",
            input_names=[
                "tile_blocks",
                "subtile_mask",
                "subtile_counts",
                "prefix",
                "counts",
            ],
            output_names=["accepted_tile_blocks", "accepted_member_mask"],
            source=_NEIGHBOR_TILE_ORDERED_SCATTER_SOURCE,
        )
    return _neighbor_tile_ordered_scatter_kernel_singleton


def _neighbor_tile_left_counts_kernel():
    """Return the cached exact-tile counts-by-left-block kernel."""

    global _neighbor_tile_left_counts_kernel_singleton
    if _neighbor_tile_left_counts_kernel_singleton is None:
        _neighbor_tile_left_counts_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_left_counts",
            input_names=[
                "subtile_mask",
                "cell_block_starts",
                "cell_block_counts",
                "cell_pairs",
                "task_offsets",
                "task_tile_counts",
                "counts",
            ],
            output_names=["fine_tile_counts"],
            source=_NEIGHBOR_TILE_LEFT_COUNTS_SOURCE,
        )
    return _neighbor_tile_left_counts_kernel_singleton


def _neighbor_tile_left_scatter_kernel():
    """Return the cached exact-tile left-grouped scatter kernel."""

    global _neighbor_tile_left_scatter_kernel_singleton
    if _neighbor_tile_left_scatter_kernel_singleton is None:
        _neighbor_tile_left_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_left_scatter",
            input_names=[
                "subtile_mask",
                "cell_block_starts",
                "cell_block_counts",
                "cell_pairs",
                "task_offsets",
                "task_tile_counts",
                "fine_tile_counts",
                "fine_tile_prefix",
                "counts",
            ],
            output_names=["accepted_tile_blocks", "accepted_member_mask"],
            source=_NEIGHBOR_TILE_LEFT_SCATTER_SOURCE,
        )
    return _neighbor_tile_left_scatter_kernel_singleton


def _neighbor_tile_column_scatter_kernel():
    """Return the cached non-empty tile-column compaction kernel."""

    global _neighbor_tile_column_scatter_kernel_singleton
    if _neighbor_tile_column_scatter_kernel_singleton is None:
        _neighbor_tile_column_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_column_scatter",
            input_names=["member_mask", "active_column_counts", "prefix", "counts"],
            output_names=["force_columns"],
            source=_NEIGHBOR_TILE_COLUMN_SCATTER_SOURCE,
        )
    return _neighbor_tile_column_scatter_kernel_singleton


def _neighbor_tile_force_groups_kernel():
    """Return the cached spatial-tile force-group schedule kernel."""

    global _neighbor_tile_force_groups_kernel_singleton
    if _neighbor_tile_force_groups_kernel_singleton is None:
        _neighbor_tile_force_groups_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_force_groups",
            input_names=[
                "work_counts",
                "work_prefix",
                "group_counts",
                "group_prefix",
                "counts",
            ],
            output_names=["force_group_starts", "force_group_counts"],
            source=_NEIGHBOR_TILE_FORCE_GROUPS_SOURCE,
        )
    return _neighbor_tile_force_groups_kernel_singleton


def _neighbor_tile_member_counts_kernel():
    """Return the cached compact-tile membership counter."""

    global _neighbor_tile_member_counts_kernel_singleton
    if _neighbor_tile_member_counts_kernel_singleton is None:
        _neighbor_tile_member_counts_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_member_counts",
            input_names=["member_mask", "counts"],
            output_names=["member_counts"],
            source=_NEIGHBOR_TILE_MEMBER_COUNTS_SOURCE,
        )
    return _neighbor_tile_member_counts_kernel_singleton


def _neighbor_tile_pair_scatter_kernel():
    """Return the cached tile-membership diagnostic-pair decoder."""

    global _neighbor_tile_pair_scatter_kernel_singleton
    if _neighbor_tile_pair_scatter_kernel_singleton is None:
        _neighbor_tile_pair_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_pair_scatter",
            input_names=[
                "atom_blocks",
                "tile_blocks",
                "member_mask",
                "member_counts",
                "member_prefix",
            ],
            output_names=["accepted_i", "accepted_j"],
            source=_NEIGHBOR_TILE_PAIR_SCATTER_SOURCE,
        )
    return _neighbor_tile_pair_scatter_kernel_singleton


def _tile_topology_lj_masks_kernel():
    """Return the cached tile-aligned topology-mask kernel."""

    global _tile_topology_lj_masks_kernel_singleton
    if _tile_topology_lj_masks_kernel_singleton is None:
        _tile_topology_lj_masks_kernel_singleton = mx.fast.metal_kernel(
            name="tile_topology_lj_masks",
            input_names=[
                "atom_blocks",
                "tile_blocks",
                "member_mask",
                "excluded_offsets",
                "excluded_right",
                "one_four_offsets",
                "one_four_right",
            ],
            output_names=["lj_enabled_mask", "lj_one_four_mask"],
            source=_TILE_TOPOLOGY_LJ_MASKS_SOURCE,
        )
    return _tile_topology_lj_masks_kernel_singleton


def _shake_cluster_position_kernel():
    """Return the cached disjoint SHAKE-cluster position kernel."""

    global _shake_cluster_position_kernel_singleton
    if _shake_cluster_position_kernel_singleton is None:
        _shake_cluster_position_kernel_singleton = mx.fast.metal_kernel(
            name="shake_cluster_positions",
            input_names=[
                "reference_positions",
                "predicted_positions",
                "masses",
                "cluster_atoms",
                "peripheral_counts",
                "distances",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SHAKE_CLUSTER_POSITION_SOURCE,
        )
    return _shake_cluster_position_kernel_singleton


def _shake_cluster_velocity_kernel():
    """Return the cached analytical SHAKE-cluster velocity kernel."""

    global _shake_cluster_velocity_kernel_singleton
    if _shake_cluster_velocity_kernel_singleton is None:
        _shake_cluster_velocity_kernel_singleton = mx.fast.metal_kernel(
            name="shake_cluster_velocities",
            input_names=[
                "positions",
                "velocities",
                "masses",
                "cluster_atoms",
                "peripheral_counts",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SHAKE_CLUSTER_VELOCITY_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _shake_cluster_velocity_kernel_singleton


def _small_constraint_cluster_position_kernel():
    """Return the cached small-component position-constraint kernel."""

    global _small_constraint_cluster_position_kernel_singleton
    if _small_constraint_cluster_position_kernel_singleton is None:
        _small_constraint_cluster_position_kernel_singleton = mx.fast.metal_kernel(
            name="small_constraint_cluster_position",
            input_names=[
                "reference_positions",
                "predicted_positions",
                "masses",
                "cluster_atoms",
                "atom_counts",
                "pair_slots",
                "pair_counts",
                "target_distances",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SMALL_CONSTRAINT_CLUSTER_POSITION_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _small_constraint_cluster_position_kernel_singleton


def _small_constraint_cluster_velocity_kernel():
    """Return the cached small-component velocity-constraint kernel."""

    global _small_constraint_cluster_velocity_kernel_singleton
    if _small_constraint_cluster_velocity_kernel_singleton is None:
        _small_constraint_cluster_velocity_kernel_singleton = mx.fast.metal_kernel(
            name="small_constraint_cluster_velocity",
            input_names=[
                "positions",
                "velocities",
                "masses",
                "cluster_atoms",
                "atom_counts",
                "pair_slots",
                "pair_counts",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SMALL_CONSTRAINT_CLUSTER_VELOCITY_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _small_constraint_cluster_velocity_kernel_singleton


def _settle_water_position_kernel():
    """Return the cached analytical SETTLE position kernel."""

    global _settle_water_position_kernel_singleton
    if _settle_water_position_kernel_singleton is None:
        _settle_water_position_kernel_singleton = mx.fast.metal_kernel(
            name="settle_water_positions",
            input_names=[
                "reference_positions",
                "predicted_positions",
                "masses",
                "waters",
                "geometry",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SETTLE_WATER_POSITION_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _settle_water_position_kernel_singleton


def _settle_water_velocity_kernel():
    """Return the cached analytical SETTLE velocity kernel."""

    global _settle_water_velocity_kernel_singleton
    if _settle_water_velocity_kernel_singleton is None:
        _settle_water_velocity_kernel_singleton = mx.fast.metal_kernel(
            name="settle_water_velocities",
            input_names=[
                "positions",
                "velocities",
                "masses",
                "waters",
                "box",
                "params",
            ],
            output_names=["deltas"],
            source=_SETTLE_WATER_VELOCITY_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _settle_water_velocity_kernel_singleton


def _langevin_baoab_drift_kernel():
    """Return the cached fused Langevin BAOAB kick-drift-thermostat kernel."""

    global _langevin_baoab_drift_kernel_singleton
    if _langevin_baoab_drift_kernel_singleton is None:
        _langevin_baoab_drift_kernel_singleton = mx.fast.metal_kernel(
            name="langevin_baoab_drift",
            input_names=[
                "positions",
                "velocities",
                "forces",
                "force_scale_over_mass",
                "thermal_scale",
                "noise",
                "box",
                "params",
                "counts",
            ],
            output_names=["next_positions", "middle_velocities"],
            source=_LANGEVIN_BAOAB_DRIFT_SOURCE,
        )
    return _langevin_baoab_drift_kernel_singleton


def _dense_constraint_apply_kernel():
    """Return the cached dense disjoint-constraint application kernel."""

    global _dense_constraint_apply_kernel_singleton
    if _dense_constraint_apply_kernel_singleton is None:
        _dense_constraint_apply_kernel_singleton = mx.fast.metal_kernel(
            name="dense_constraint_apply",
            input_names=[
                "base_values",
                "owner_family",
                "owner_rows",
                "owner_slots",
                "settle_deltas",
                "shake_deltas",
                "params",
            ],
            output_names=["constrained"],
            source=_DENSE_CONSTRAINT_APPLY_SOURCE,
            header=_CONSTRAINT_HEADER,
        )
    return _dense_constraint_apply_kernel_singleton


def _fused_bonded_force_only_kernel():
    """Return the cached force-only bonded-interaction Metal kernel."""

    global _fused_bonded_force_only_kernel_singleton
    if _fused_bonded_force_only_kernel_singleton is None:
        _fused_bonded_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="fused_bonded_force_only",
            input_names=[
                "positions",
                "box",
                "bond_atoms",
                "bond_k",
                "bond_length",
                "angle_atoms",
                "angle_k",
                "angle_target",
                "dihedral_atoms",
                "dihedral_k",
                "dihedral_periodicity",
                "dihedral_phase",
                "improper_atoms",
                "improper_k",
                "improper_periodicity",
                "improper_phase",
                "cmap_atoms",
                "cmap_indices",
                "cmap_coefficients",
                "correction_atoms",
                "correction_charge_products",
                "correction_lj_sigma",
                "correction_lj_epsilon",
                "correction_coulomb",
                "counts",
            ],
            output_names=["forces"],
            source=_BONDED_FORCE_SOURCE,
            header=_BONDED_FORCE_HEADER,
            atomic_outputs=True,
        )
    return _fused_bonded_force_only_kernel_singleton


def neighbor_pair_cutoff_mask(
    positions: mx.array,
    pairs_i: mx.array,
    pairs_j: mx.array,
    box_lengths: mx.array,
    *,
    search_radius: float,
) -> mx.array:
    """Return the periodic cutoff mask for aligned candidate pairs on Metal.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs_i: Left atom indices with shape ``(n_pairs,)``.
        pairs_j: Right atom indices with shape ``(n_pairs,)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        search_radius: Positive neighbor search radius.

    Returns:
        Boolean mask with one entry per candidate pair.

    Raises:
        ValueError: If an input shape or the search radius is invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs_i = as_mx_array(pairs_i, dtype=mx.int32)
    pairs_j = as_mx_array(pairs_j, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if pairs_i.ndim != 1 or pairs_j.shape != pairs_i.shape:
        msg = "pairs_i and pairs_j must have matching one-dimensional shapes"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if not isfinite(float(search_radius)) or float(search_radius) <= 0.0:
        msg = "search_radius must be finite and positive"
        raise ValueError(msg)

    pair_count = int(pairs_i.shape[0])
    if pair_count == 0:
        return mx.zeros((0,), dtype=mx.bool_)
    params = mx.array([float(search_radius) ** 2], dtype=mx.float32)
    counts = mx.array([pair_count], dtype=mx.int32)
    threads = min(256, pair_count)
    (close,) = _neighbor_pair_cutoff_mask_kernel()(
        inputs=[
            positions,
            pairs_i,
            pairs_j,
            box_lengths,
            params,
            counts,
        ],
        output_shapes=[(pair_count,)],
        output_dtypes=[mx.bool_],
        grid=(pair_count, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return close


def _neighbor_cell_pair_candidates(
    sorted_atoms: mx.array,
    cell_starts: mx.array,
    cell_counts: mx.array,
    cell_pairs: mx.array,
    task_offsets: mx.array,
    *,
    candidate_count: int,
) -> tuple[mx.array, mx.array]:
    """Emit atom-pair candidates for occupied spatial-cell tasks on Metal.

    Args:
        sorted_atoms: Atom indices grouped by cell with shape ``(n_atoms,)``.
        cell_starts: Starting index of each cell in ``sorted_atoms``.
        cell_counts: Number of atoms in each spatial cell.
        cell_pairs: Canonical cell-pair tasks with shape ``(n_tasks, 2)``.
        task_offsets: Exclusive output offset for each task.
        candidate_count: Total output candidates across all tasks.

    Returns:
        Aligned left and right atom-index arrays. Same-cell tasks emit each
        unique pair once; cross-cell tasks emit their Cartesian product.

    Raises:
        ValueError: If shapes or counts are inconsistent.
    """

    sorted_atoms = as_mx_array(sorted_atoms, dtype=mx.int32)
    cell_starts = as_mx_array(cell_starts, dtype=mx.int32)
    cell_counts = as_mx_array(cell_counts, dtype=mx.int32)
    cell_pairs = as_mx_array(cell_pairs, dtype=mx.int32)
    task_offsets = as_mx_array(task_offsets, dtype=mx.int32)
    if sorted_atoms.ndim != 1:
        msg = "sorted_atoms must be one-dimensional"
        raise ValueError(msg)
    if cell_starts.ndim != 1 or cell_counts.shape != cell_starts.shape:
        msg = "cell_starts and cell_counts must have matching one-dimensional shapes"
        raise ValueError(msg)
    if cell_pairs.ndim != 2 or cell_pairs.shape[1] != 2:
        msg = "cell_pairs must have shape (n_tasks, 2)"
        raise ValueError(msg)
    if task_offsets.shape != (cell_pairs.shape[0],):
        msg = "task_offsets must contain one output offset per cell-pair task"
        raise ValueError(msg)
    if candidate_count < 0:
        msg = "candidate_count must be non-negative"
        raise ValueError(msg)

    task_count = int(cell_pairs.shape[0])
    if task_count == 0 or candidate_count == 0:
        empty = mx.zeros((0,), dtype=mx.int32)
        return empty, empty
    threads = min(256, task_count)
    pairs_i, pairs_j = _neighbor_cell_pair_candidates_kernel()(
        inputs=[
            sorted_atoms,
            cell_starts,
            cell_counts,
            cell_pairs,
            task_offsets,
            mx.array([task_count], dtype=mx.int32),
        ],
        output_shapes=[(candidate_count,), (candidate_count,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(task_count, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return pairs_i, pairs_j


def _neighbor_cell_atom_blocks_sized(
    sorted_atoms: mx.array,
    cell_starts: mx.array,
    cell_counts: mx.array,
    cell_block_starts: mx.array,
    cell_block_counts: mx.array,
    *,
    block_capacity: int,
) -> mx.array:
    """Scatter cell-sorted atoms into padded eight-atom blocks on Metal."""

    sorted_atoms = as_mx_array(sorted_atoms, dtype=mx.int32)
    cell_starts = as_mx_array(cell_starts, dtype=mx.int32)
    cell_counts = as_mx_array(cell_counts, dtype=mx.int32)
    cell_block_starts = as_mx_array(cell_block_starts, dtype=mx.int32)
    cell_block_counts = as_mx_array(cell_block_counts, dtype=mx.int32)
    cell_count = int(cell_counts.shape[0])
    for name, values in (
        ("cell_starts", cell_starts),
        ("cell_block_starts", cell_block_starts),
        ("cell_block_counts", cell_block_counts),
    ):
        if values.shape != (cell_count,):
            msg = f"{name} must contain one value per cell"
            raise ValueError(msg)
    if sorted_atoms.ndim != 1:
        msg = "sorted_atoms must be one-dimensional"
        raise ValueError(msg)
    if block_capacity < 0:
        msg = "block_capacity must be non-negative"
        raise ValueError(msg)
    if cell_count == 0 or block_capacity == 0:
        return mx.zeros((0, _TILE_BUILD_BLOCK_SIZE), dtype=mx.int32)
    threads = min(256, cell_count)
    (atom_blocks,) = _neighbor_cell_atom_blocks_kernel()(
        inputs=[
            sorted_atoms,
            cell_starts,
            cell_counts,
            cell_block_starts,
            cell_block_counts,
            mx.array([cell_count], dtype=mx.int32),
        ],
        output_shapes=[(block_capacity, _TILE_BUILD_BLOCK_SIZE)],
        output_dtypes=[mx.int32],
        grid=(cell_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=-1,
    )
    return atom_blocks


def _neighbor_cell_tile_candidates(
    cell_block_starts: mx.array,
    cell_block_counts: mx.array,
    cell_pairs: mx.array,
    task_offsets: mx.array,
    *,
    candidate_count: int,
) -> mx.array:
    """Emit spatial block-pair tile candidates on Metal."""

    cell_block_starts = as_mx_array(cell_block_starts, dtype=mx.int32)
    cell_block_counts = as_mx_array(cell_block_counts, dtype=mx.int32)
    cell_pairs = as_mx_array(cell_pairs, dtype=mx.int32)
    task_offsets = as_mx_array(task_offsets, dtype=mx.int32)
    if cell_block_starts.ndim != 1 or cell_block_counts.shape != cell_block_starts.shape:
        msg = "cell block starts and counts must have matching one-dimensional shapes"
        raise ValueError(msg)
    if cell_pairs.ndim != 2 or cell_pairs.shape[1] != 2:
        msg = "cell_pairs must have shape (n_tasks, 2)"
        raise ValueError(msg)
    if task_offsets.shape != (cell_pairs.shape[0],):
        msg = "task_offsets must contain one output offset per cell-pair task"
        raise ValueError(msg)
    if candidate_count < 0:
        msg = "candidate_count must be non-negative"
        raise ValueError(msg)
    task_count = int(cell_pairs.shape[0])
    if task_count == 0 or candidate_count == 0:
        return mx.zeros((0, 2), dtype=mx.int32)
    threads = min(256, task_count)
    tile_left, tile_right = _neighbor_cell_tile_candidates_kernel()(
        inputs=[
            cell_block_starts,
            cell_block_counts,
            cell_pairs,
            task_offsets,
            mx.array([task_count], dtype=mx.int32),
        ],
        output_shapes=[(candidate_count,), (candidate_count,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(task_count, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return mx.stack((tile_left, tile_right), axis=1)


def _neighbor_tile_membership(
    positions: mx.array,
    atom_blocks: mx.array,
    tile_blocks: mx.array,
    box_lengths: mx.array,
    *,
    search_radius: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Encode 8x8 membership as four packed 4x4 subtile masks."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_BUILD_BLOCK_SIZE:
        msg = f"atom_blocks must have shape (n_blocks, {_TILE_BUILD_BLOCK_SIZE})"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if not isfinite(float(search_radius)) or float(search_radius) <= 0.0:
        msg = "search_radius must be finite and positive"
        raise ValueError(msg)
    tile_count = int(tile_blocks.shape[0])
    if tile_count == 0:
        return (
            mx.zeros((0, _TILE_BUILD_SUBTILES_PER_TILE), dtype=mx.uint32),
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
        )
    subtile_mask, member_counts, subtile_counts = _neighbor_tile_membership_kernel()(
        inputs=[
            positions,
            atom_blocks,
            tile_blocks,
            box_lengths,
            mx.array([float(search_radius) ** 2], dtype=mx.float32),
        ],
        output_shapes=[
            (tile_count, _TILE_BUILD_SUBTILES_PER_TILE),
            (tile_count,),
            (tile_count,),
        ],
        output_dtypes=[mx.uint32, mx.int32, mx.int32],
        grid=(tile_count * _TILE_BUILD_MEMBERSHIP_THREADGROUP_SIZE, 1, 1),
        threadgroup=(_TILE_BUILD_MEMBERSHIP_THREADGROUP_SIZE, 1, 1),
        init_value=0,
    )
    return subtile_mask, member_counts, subtile_counts


def _neighbor_tile_ordered_scatter_sized(
    tile_blocks: mx.array,
    subtile_mask: mx.array,
    subtile_counts: mx.array,
    prefix: mx.array,
    *,
    accepted_count: int,
) -> tuple[mx.array, mx.array]:
    """Compact non-empty 4x4 subtiles from coarse 8x8 candidates."""

    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    subtile_mask = as_mx_array(subtile_mask, dtype=mx.uint32)
    subtile_counts = as_mx_array(subtile_counts, dtype=mx.int32)
    prefix = as_mx_array(prefix, dtype=mx.int32)
    tile_count = int(tile_blocks.shape[0])
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    mask_shape = (tile_count, _TILE_BUILD_SUBTILES_PER_TILE)
    if subtile_mask.shape != mask_shape:
        msg = f"subtile_mask must have shape (n_tiles, {_TILE_BUILD_SUBTILES_PER_TILE})"
        raise ValueError(msg)
    if subtile_counts.shape != (tile_count,) or prefix.shape != (tile_count,):
        msg = "subtile_counts and prefix must contain one value per coarse tile"
        raise ValueError(msg)
    if accepted_count < 0 or accepted_count > tile_count * _TILE_BUILD_SUBTILES_PER_TILE:
        msg = "accepted_count must fit within the subdivided candidate tile count"
        raise ValueError(msg)
    if accepted_count == 0:
        return (
            mx.zeros((0, 2), dtype=mx.int32),
            mx.zeros((0, _TILE_PME_MASK_WORD_COUNT), dtype=mx.uint32),
        )
    accepted_tiles, accepted_mask = _neighbor_tile_ordered_scatter_kernel()(
        inputs=[
            tile_blocks,
            subtile_mask,
            subtile_counts,
            prefix,
            mx.array([tile_count], dtype=mx.int32),
        ],
        output_shapes=[
            (accepted_count, 2),
            (accepted_count, _TILE_PME_MASK_WORD_COUNT),
        ],
        output_dtypes=[mx.int32, mx.uint32],
        grid=(tile_count, 1, 1),
        threadgroup=(min(256, tile_count), 1, 1),
        init_value=0,
    )
    return accepted_tiles, accepted_mask


def _neighbor_tile_left_counts(
    subtile_mask: mx.array,
    cell_block_starts: mx.array,
    cell_block_counts: mx.array,
    cell_pairs: mx.array,
    task_offsets: mx.array,
    task_tile_counts: mx.array,
    *,
    block_capacity: int,
) -> mx.array:
    """Count exact 4x4 subtiles directly by their final left block."""

    subtile_mask = as_mx_array(subtile_mask, dtype=mx.uint32)
    cell_block_starts = as_mx_array(cell_block_starts, dtype=mx.int32)
    cell_block_counts = as_mx_array(cell_block_counts, dtype=mx.int32)
    cell_pairs = as_mx_array(cell_pairs, dtype=mx.int32)
    task_offsets = as_mx_array(task_offsets, dtype=mx.int32)
    task_tile_counts = as_mx_array(task_tile_counts, dtype=mx.int32)
    cell_count = int(cell_block_starts.shape[0])
    task_count = int(cell_pairs.shape[0])
    if cell_block_counts.shape != (cell_count,):
        msg = "cell block starts and counts must have matching vector shapes"
        raise ValueError(msg)
    if cell_pairs.ndim != 2 or cell_pairs.shape[1] != 2:
        msg = "cell_pairs must have shape (n_tasks, 2)"
        raise ValueError(msg)
    if task_offsets.shape != (task_count,) or task_tile_counts.shape != (task_count,):
        msg = "task offsets and tile counts must contain one value per task"
        raise ValueError(msg)
    if subtile_mask.ndim != 2 or subtile_mask.shape[1] != _TILE_BUILD_SUBTILES_PER_TILE:
        msg = f"subtile_mask must have {_TILE_BUILD_SUBTILES_PER_TILE} columns"
        raise ValueError(msg)
    if block_capacity < 0:
        msg = "block_capacity must be non-negative"
        raise ValueError(msg)
    if cell_count == 0 or block_capacity == 0:
        return mx.zeros((2 * block_capacity,), dtype=mx.int32)
    threads = min(256, cell_count)
    (fine_tile_counts,) = _neighbor_tile_left_counts_kernel()(
        inputs=[
            subtile_mask,
            cell_block_starts,
            cell_block_counts,
            cell_pairs,
            task_offsets,
            task_tile_counts,
            mx.array([cell_count, task_count], dtype=mx.int32),
        ],
        output_shapes=[(2 * block_capacity,)],
        output_dtypes=[mx.int32],
        grid=(cell_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0,
    )
    return fine_tile_counts


def _neighbor_tile_left_scatter_sized(
    subtile_mask: mx.array,
    cell_block_starts: mx.array,
    cell_block_counts: mx.array,
    cell_pairs: mx.array,
    task_offsets: mx.array,
    task_tile_counts: mx.array,
    fine_tile_counts: mx.array,
    fine_tile_prefix: mx.array,
    *,
    accepted_count: int,
) -> tuple[mx.array, mx.array]:
    """Scatter exact subtiles in deterministic final-left-block order."""

    subtile_mask = as_mx_array(subtile_mask, dtype=mx.uint32)
    cell_block_starts = as_mx_array(cell_block_starts, dtype=mx.int32)
    cell_block_counts = as_mx_array(cell_block_counts, dtype=mx.int32)
    cell_pairs = as_mx_array(cell_pairs, dtype=mx.int32)
    task_offsets = as_mx_array(task_offsets, dtype=mx.int32)
    task_tile_counts = as_mx_array(task_tile_counts, dtype=mx.int32)
    fine_tile_counts = as_mx_array(fine_tile_counts, dtype=mx.int32)
    fine_tile_prefix = as_mx_array(fine_tile_prefix, dtype=mx.int32)
    cell_count = int(cell_block_starts.shape[0])
    task_count = int(cell_pairs.shape[0])
    if cell_block_counts.shape != (cell_count,):
        msg = "cell block starts and counts must have matching vector shapes"
        raise ValueError(msg)
    if cell_pairs.ndim != 2 or cell_pairs.shape[1] != 2:
        msg = "cell_pairs must have shape (n_tasks, 2)"
        raise ValueError(msg)
    if task_offsets.shape != (task_count,) or task_tile_counts.shape != (task_count,):
        msg = "task offsets and tile counts must contain one value per task"
        raise ValueError(msg)
    if subtile_mask.ndim != 2 or subtile_mask.shape[1] != _TILE_BUILD_SUBTILES_PER_TILE:
        msg = f"subtile_mask must have {_TILE_BUILD_SUBTILES_PER_TILE} columns"
        raise ValueError(msg)
    if fine_tile_prefix.shape != fine_tile_counts.shape or fine_tile_counts.ndim != 1:
        msg = "fine tile counts and prefix must have matching vector shapes"
        raise ValueError(msg)
    if accepted_count < 0:
        msg = "accepted_count must be non-negative"
        raise ValueError(msg)
    if accepted_count == 0:
        return (
            mx.zeros((0, 2), dtype=mx.int32),
            mx.zeros((0, _TILE_PME_MASK_WORD_COUNT), dtype=mx.uint32),
        )
    threads = min(256, cell_count)
    accepted_tiles, accepted_mask = _neighbor_tile_left_scatter_kernel()(
        inputs=[
            subtile_mask,
            cell_block_starts,
            cell_block_counts,
            cell_pairs,
            task_offsets,
            task_tile_counts,
            fine_tile_counts,
            fine_tile_prefix,
            mx.array([cell_count, task_count], dtype=mx.int32),
        ],
        output_shapes=[
            (accepted_count, 2),
            (accepted_count, _TILE_PME_MASK_WORD_COUNT),
        ],
        output_dtypes=[mx.int32, mx.uint32],
        grid=(cell_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0,
    )
    return accepted_tiles, accepted_mask


def _neighbor_tile_column_scatter_sized(
    member_mask: mx.array,
    active_column_counts: mx.array,
    prefix: mx.array,
    *,
    accepted_count: int,
) -> mx.array:
    """Compact non-empty right columns and pack their four membership bits."""

    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    active_column_counts = as_mx_array(active_column_counts, dtype=mx.int32)
    prefix = as_mx_array(prefix, dtype=mx.int32)
    tile_count = int(member_mask.shape[0])
    if tile_count > _TILE_PME_COLUMN_DESCRIPTOR_TILE_LIMIT:
        msg = (
            "tile count exceeds the packed force-column descriptor capacity "
            f"({_TILE_PME_COLUMN_DESCRIPTOR_TILE_LIMIT})"
        )
        raise ValueError(msg)
    if member_mask.shape != (tile_count, _TILE_PME_MASK_WORD_COUNT):
        msg = f"member_mask must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    if active_column_counts.shape != (tile_count,) or prefix.shape != (tile_count,):
        msg = "active_column_counts and prefix must contain one value per tile"
        raise ValueError(msg)
    if accepted_count < 0 or accepted_count > tile_count * _TILE_PME_BLOCK_SIZE:
        msg = "accepted_count must fit within the tile-column inventory"
        raise ValueError(msg)
    if accepted_count == 0:
        return mx.zeros((0,), dtype=mx.int32)
    (force_columns,) = _neighbor_tile_column_scatter_kernel()(
        inputs=[
            member_mask,
            active_column_counts,
            prefix,
            mx.array([tile_count], dtype=mx.int32),
        ],
        output_shapes=[(accepted_count,)],
        output_dtypes=[mx.int32],
        grid=(tile_count, 1, 1),
        threadgroup=(min(256, tile_count), 1, 1),
        init_value=0,
    )
    return force_columns


def _neighbor_tile_force_groups_sized(
    work_counts: mx.array,
    work_prefix: mx.array,
    group_counts: mx.array,
    group_prefix: mx.array,
    *,
    accepted_count: int,
    items_per_group: int,
) -> tuple[mx.array, mx.array]:
    """Build contiguous same-left-block force groups from work-item counts."""

    work_counts = as_mx_array(work_counts, dtype=mx.int32)
    work_prefix = as_mx_array(work_prefix, dtype=mx.int32)
    group_counts = as_mx_array(group_counts, dtype=mx.int32)
    group_prefix = as_mx_array(group_prefix, dtype=mx.int32)
    block_count = int(work_counts.shape[0])
    if any(values.shape != (block_count,) for values in (work_prefix, group_counts, group_prefix)):
        msg = "work and group counts/prefixes must have matching one-dimensional shapes"
        raise ValueError(msg)
    if items_per_group <= 0:
        msg = "items_per_group must be positive"
        raise ValueError(msg)
    if accepted_count < 0:
        msg = "accepted_count must be non-negative"
        raise ValueError(msg)
    if accepted_count == 0:
        empty = mx.zeros((0,), dtype=mx.int32)
        return empty, empty
    force_group_starts, force_group_sizes = _neighbor_tile_force_groups_kernel()(
        inputs=[
            work_counts,
            work_prefix,
            group_counts,
            group_prefix,
            mx.array([block_count, items_per_group], dtype=mx.int32),
        ],
        output_shapes=[(accepted_count,), (accepted_count,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(block_count, 1, 1),
        threadgroup=(min(256, block_count), 1, 1),
        init_value=0,
    )
    return force_group_starts, force_group_sizes


def _neighbor_tile_member_pairs_sized(
    atom_blocks: mx.array,
    tile_blocks: mx.array,
    member_mask: mx.array,
    member_counts: mx.array,
    member_prefix: mx.array,
    *,
    accepted_count: int,
) -> mx.array:
    """Decode diagnostic pairs from already-computed tile membership."""

    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    member_counts = as_mx_array(member_counts, dtype=mx.int32)
    member_prefix = as_mx_array(member_prefix, dtype=mx.int32)
    tile_count = int(tile_blocks.shape[0])
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = f"atom_blocks must have shape (n_blocks, {_TILE_PME_BLOCK_SIZE})"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if member_mask.shape != (tile_count, _TILE_PME_MASK_WORD_COUNT):
        msg = f"member_mask must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    if member_counts.shape != (tile_count,) or member_prefix.shape != (tile_count,):
        msg = "member counts and prefix must contain one value per tile"
        raise ValueError(msg)
    if accepted_count < 0 or (tile_count == 0 and accepted_count != 0):
        msg = "accepted_count is incompatible with tile membership"
        raise ValueError(msg)
    if accepted_count == 0:
        return mx.zeros((0, 2), dtype=mx.int32)
    accepted_i, accepted_j = _neighbor_tile_pair_scatter_kernel()(
        inputs=[
            atom_blocks,
            tile_blocks,
            member_mask,
            member_counts,
            member_prefix,
        ],
        output_shapes=[(accepted_count,), (accepted_count,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(tile_count * _TILE_PME_LANES_PER_TILE, 1, 1),
        threadgroup=(_TILE_MEMBERSHIP_THREADGROUP_SIZE, 1, 1),
        init_value=0,
    )
    return mx.stack((accepted_i, accepted_j), axis=1)


def _neighbor_tile_member_counts(member_mask: mx.array) -> mx.array:
    """Count active lanes in already-compacted spatial-tile masks."""

    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    if member_mask.ndim != 2 or member_mask.shape[1] != _TILE_PME_MASK_WORD_COUNT:
        msg = f"member_mask must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    tile_count = int(member_mask.shape[0])
    if tile_count == 0:
        return mx.zeros((0,), dtype=mx.int32)
    return _neighbor_tile_member_counts_kernel()(
        inputs=[
            member_mask,
            mx.array([tile_count], dtype=mx.int32),
        ],
        output_shapes=[(tile_count,)],
        output_dtypes=[mx.int32],
        grid=(tile_count, 1, 1),
        threadgroup=(min(256, tile_count), 1, 1),
        init_value=0,
    )[0]


def tile_topology_lj_masks(
    atom_blocks: mx.array,
    tile_blocks: mx.array,
    member_mask: mx.array,
    excluded_offsets: mx.array,
    excluded_right: mx.array,
    one_four_offsets: mx.array,
    one_four_right: mx.array,
) -> tuple[mx.array, mx.array]:
    """Build tile-aligned LJ masks from atom-local sparse topology rows."""

    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    excluded_offsets = as_mx_array(excluded_offsets, dtype=mx.int32)
    excluded_right = as_mx_array(excluded_right, dtype=mx.int32)
    one_four_offsets = as_mx_array(one_four_offsets, dtype=mx.int32)
    one_four_right = as_mx_array(one_four_right, dtype=mx.int32)
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = f"atom_blocks must have shape (n_blocks, {_TILE_PME_BLOCK_SIZE})"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    tile_count = int(tile_blocks.shape[0])
    if member_mask.shape != (tile_count, _TILE_PME_MASK_WORD_COUNT):
        msg = f"member_mask must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    for name, offsets, right in (
        ("excluded", excluded_offsets, excluded_right),
        ("one_four", one_four_offsets, one_four_right),
    ):
        if offsets.ndim != 1 or offsets.shape[0] < 1:
            msg = f"{name}_offsets must be a non-empty one-dimensional array"
            raise ValueError(msg)
        if right.ndim != 1:
            msg = f"{name}_right must be one-dimensional"
            raise ValueError(msg)
    if tile_count == 0:
        empty = mx.zeros((0, _TILE_PME_MASK_WORD_COUNT), dtype=mx.uint32)
        return empty, empty
    enabled_mask, one_four_mask = _tile_topology_lj_masks_kernel()(
        inputs=[
            atom_blocks,
            tile_blocks,
            member_mask,
            excluded_offsets,
            excluded_right,
            one_four_offsets,
            one_four_right,
        ],
        output_shapes=[
            (tile_count, _TILE_PME_MASK_WORD_COUNT),
            (tile_count, _TILE_PME_MASK_WORD_COUNT),
        ],
        output_dtypes=[mx.uint32, mx.uint32],
        grid=(tile_count * _TILE_PME_LANES_PER_TILE, 1, 1),
        threadgroup=(_TILE_MEMBERSHIP_THREADGROUP_SIZE, 1, 1),
        init_value=0,
    )
    return enabled_mask, one_four_mask


def neighbor_pair_ordered_scatter(
    pairs_i: mx.array,
    pairs_j: mx.array,
    close: mx.array,
    prefix: mx.array,
) -> tuple[mx.array, mx.array]:
    """Scatter accepted neighbor candidates by their stable prefix positions.

    Args:
        pairs_i: Left atom indices with shape ``(n_pairs,)``.
        pairs_j: Right atom indices with shape ``(n_pairs,)``.
        close: Boolean cutoff mask with shape ``(n_pairs,)``.
        prefix: Inclusive integer prefix sum of ``close``.

    Returns:
        Candidate-sized left and right output buffers whose leading accepted
        entries preserve the input candidate order.

    Raises:
        ValueError: If the inputs are not matching one-dimensional arrays.
    """

    pairs_i = as_mx_array(pairs_i, dtype=mx.int32)
    return _neighbor_pair_ordered_scatter_sized(
        pairs_i,
        pairs_j,
        close,
        prefix,
        accepted_count=int(pairs_i.shape[0]),
    )


def _neighbor_pair_ordered_scatter_sized(
    pairs_i: mx.array,
    pairs_j: mx.array,
    close: mx.array,
    prefix: mx.array,
    *,
    accepted_count: int,
) -> tuple[mx.array, mx.array]:
    """Scatter accepted candidates into an explicitly sized output."""

    pairs_i = as_mx_array(pairs_i, dtype=mx.int32)
    pairs_j = as_mx_array(pairs_j, dtype=mx.int32)
    close = as_mx_array(close, dtype=mx.bool_)
    prefix = as_mx_array(prefix, dtype=mx.int32)
    if pairs_i.ndim != 1:
        msg = "pairs_i must be one-dimensional"
        raise ValueError(msg)
    if pairs_j.shape != pairs_i.shape or close.shape != pairs_i.shape:
        msg = "pairs_i, pairs_j, and close must have matching shapes"
        raise ValueError(msg)
    if prefix.shape != pairs_i.shape:
        msg = "prefix must match the candidate-pair shape"
        raise ValueError(msg)

    pair_count = int(pairs_i.shape[0])
    if pair_count == 0:
        empty = mx.zeros((0,), dtype=mx.int32)
        return empty, empty
    output_count = int(accepted_count)
    if output_count < 0 or output_count > pair_count:
        msg = "accepted_count must fit within the candidate-pair count"
        raise ValueError(msg)
    if output_count == 0:
        empty = mx.zeros((0,), dtype=mx.int32)
        return empty, empty
    threads = min(256, pair_count)
    accepted_i, accepted_j = _neighbor_pair_ordered_scatter_kernel()(
        inputs=[
            pairs_i,
            pairs_j,
            close,
            prefix,
            mx.array([pair_count], dtype=mx.int32),
        ],
        output_shapes=[(output_count,), (output_count,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(pair_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0,
    )
    return accepted_i, accepted_j


def _small_constraint_cluster_position_deltas(
    reference_positions: mx.array,
    predicted_positions: mx.array,
    masses: mx.array,
    cluster_atoms: mx.array,
    atom_counts: mx.array,
    pair_slots: mx.array,
    pair_counts: mx.array,
    target_distances: mx.array,
    box_lengths: mx.array,
    *,
    max_iterations: int,
    periodic: bool,
) -> mx.array:
    """Return position deltas for disjoint constraint components of up to four atoms."""

    reference_positions = as_mx_array(reference_positions, dtype=mx.float32)
    predicted_positions = as_mx_array(predicted_positions, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    cluster_atoms = as_mx_array(cluster_atoms, dtype=mx.int32)
    atom_counts = as_mx_array(atom_counts, dtype=mx.int32)
    pair_slots = as_mx_array(pair_slots, dtype=mx.int32)
    pair_counts = as_mx_array(pair_counts, dtype=mx.int32)
    target_distances = as_mx_array(target_distances, dtype=mx.float32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    if reference_positions.shape != predicted_positions.shape:
        raise ValueError("reference and predicted positions must have matching shapes")
    if predicted_positions.ndim != 2 or predicted_positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if masses.shape != (predicted_positions.shape[0],):
        raise ValueError("masses must have shape (n_atoms,)")
    if cluster_atoms.ndim != 2 or cluster_atoms.shape[1] != 4:
        raise ValueError("cluster_atoms must have shape (n_clusters, 4)")
    cluster_count = int(cluster_atoms.shape[0])
    if atom_counts.shape != (cluster_count,):
        raise ValueError("atom_counts must have shape (n_clusters,)")
    if pair_slots.shape != (cluster_count, 3, 2):
        raise ValueError("pair_slots must have shape (n_clusters, 3, 2)")
    if pair_counts.shape != (cluster_count,):
        raise ValueError("pair_counts must have shape (n_clusters,)")
    if target_distances.shape != (cluster_count, 3):
        raise ValueError("target_distances must have shape (n_clusters, 3)")
    if box_lengths.shape != (3,):
        raise ValueError("box_lengths must have shape (3,)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if cluster_count == 0:
        return mx.zeros((0, 4, 3), dtype=mx.float32)
    (deltas,) = _small_constraint_cluster_position_kernel()(
        inputs=[
            reference_positions,
            predicted_positions,
            masses,
            cluster_atoms,
            atom_counts,
            pair_slots,
            pair_counts,
            target_distances,
            box_lengths,
            mx.array(
                [cluster_count, max_iterations, int(periodic)],
                dtype=mx.int32,
            ),
        ],
        output_shapes=[(cluster_count, 4, 3)],
        output_dtypes=[mx.float32],
        grid=(cluster_count, 1, 1),
        threadgroup=(min(256, cluster_count), 1, 1),
        init_value=0.0,
    )
    return deltas


def _small_constraint_cluster_velocity_deltas(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    cluster_atoms: mx.array,
    atom_counts: mx.array,
    pair_slots: mx.array,
    pair_counts: mx.array,
    box_lengths: mx.array,
    *,
    max_iterations: int,
    periodic: bool,
) -> mx.array:
    """Return velocity deltas for disjoint components of up to four atoms."""

    positions = as_mx_array(positions, dtype=mx.float32)
    velocities = as_mx_array(velocities, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    cluster_atoms = as_mx_array(cluster_atoms, dtype=mx.int32)
    atom_counts = as_mx_array(atom_counts, dtype=mx.int32)
    pair_slots = as_mx_array(pair_slots, dtype=mx.int32)
    pair_counts = as_mx_array(pair_counts, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (n_atoms,)")
    if cluster_atoms.ndim != 2 or cluster_atoms.shape[1] != 4:
        raise ValueError("cluster_atoms must have shape (n_clusters, 4)")
    cluster_count = int(cluster_atoms.shape[0])
    if atom_counts.shape != (cluster_count,):
        raise ValueError("atom_counts must have shape (n_clusters,)")
    if pair_slots.shape != (cluster_count, 3, 2):
        raise ValueError("pair_slots must have shape (n_clusters, 3, 2)")
    if pair_counts.shape != (cluster_count,):
        raise ValueError("pair_counts must have shape (n_clusters,)")
    if box_lengths.shape != (3,):
        raise ValueError("box_lengths must have shape (3,)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if cluster_count == 0:
        return mx.zeros((0, 4, 3), dtype=mx.float32)
    (deltas,) = _small_constraint_cluster_velocity_kernel()(
        inputs=[
            positions,
            velocities,
            masses,
            cluster_atoms,
            atom_counts,
            pair_slots,
            pair_counts,
            box_lengths,
            mx.array(
                [cluster_count, max_iterations, int(periodic)],
                dtype=mx.int32,
            ),
        ],
        output_shapes=[(cluster_count, 4, 3)],
        output_dtypes=[mx.float32],
        grid=(cluster_count, 1, 1),
        threadgroup=(min(256, cluster_count), 1, 1),
        init_value=0.0,
    )
    return deltas


def _dense_constraint_apply(
    base_values: mx.array,
    owner_family: mx.array,
    owner_rows: mx.array,
    owner_slots: mx.array,
    settle_deltas: mx.array,
    shake_deltas: mx.array,
    params: mx.array,
) -> mx.array:
    """Apply disjoint SETTLE/SHAKE deltas through one dense Metal write."""

    base_values = as_mx_array(base_values, dtype=mx.float32)
    owner_family = as_mx_array(owner_family, dtype=mx.int32)
    owner_rows = as_mx_array(owner_rows, dtype=mx.int32)
    owner_slots = as_mx_array(owner_slots, dtype=mx.int32)
    settle_deltas = as_mx_array(settle_deltas, dtype=mx.float32)
    shake_deltas = as_mx_array(shake_deltas, dtype=mx.float32)
    params = as_mx_array(params, dtype=mx.int32)
    if base_values.ndim != 2 or base_values.shape[1] != 3:
        msg = "base_values must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if not (
        owner_family.ndim == 1
        and owner_rows.shape == owner_family.shape
        and owner_slots.shape == owner_family.shape
    ):
        msg = "constraint owner maps must be matching one-dimensional arrays"
        raise ValueError(msg)
    if settle_deltas.ndim != 3 or settle_deltas.shape[1:] != (3, 3):
        msg = "settle_deltas must have shape (n_waters, 3, 3)"
        raise ValueError(msg)
    if shake_deltas.ndim != 3 or shake_deltas.shape[1:] != (4, 3):
        msg = "shake_deltas must have shape (n_clusters, 4, 3)"
        raise ValueError(msg)
    if params.shape != (4,):
        msg = "constraint apply params must have shape (4,)"
        raise ValueError(msg)
    atom_count = int(base_values.shape[0])
    if atom_count == 0:
        return base_values
    (constrained,) = _dense_constraint_apply_kernel()(
        inputs=[
            base_values,
            owner_family,
            owner_rows,
            owner_slots,
            settle_deltas,
            shake_deltas,
            params,
        ],
        output_shapes=[base_values.shape],
        output_dtypes=[mx.float32],
        grid=(atom_count, 1, 1),
        threadgroup=(min(256, atom_count), 1, 1),
        init_value=0.0,
    )
    return constrained


def _settle_water_position_deltas(
    reference_positions: mx.array,
    predicted_positions: mx.array,
    masses: mx.array,
    waters: mx.array,
    box_lengths: mx.array,
    *,
    oh_distance: float,
    hh_distance: float,
    periodic: bool,
) -> mx.array:
    """Return analytical SETTLE position deltas from one Metal dispatch."""

    reference_positions = as_mx_array(reference_positions, dtype=mx.float32)
    predicted_positions = as_mx_array(predicted_positions, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    waters = as_mx_array(waters, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    water_count = int(waters.shape[0])
    if reference_positions.shape != predicted_positions.shape:
        msg = "reference and predicted positions must have matching shapes"
        raise ValueError(msg)
    if predicted_positions.ndim != 2 or predicted_positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if masses.shape != (predicted_positions.shape[0],):
        msg = "masses must have shape (n_atoms,)"
        raise ValueError(msg)
    if waters.ndim != 2 or waters.shape[1] != 3:
        msg = "waters must have shape (n_waters, 3)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if water_count == 0:
        return mx.zeros((0, 3, 3), dtype=mx.float32)
    threads = min(256, water_count)
    (deltas,) = _settle_water_position_kernel()(
        inputs=[
            reference_positions,
            predicted_positions,
            masses,
            waters,
            mx.array([oh_distance, hh_distance], dtype=mx.float32),
            box_lengths,
            mx.array([water_count, int(periodic)], dtype=mx.int32),
        ],
        output_shapes=[(water_count, 3, 3)],
        output_dtypes=[mx.float32],
        grid=(water_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return deltas


def _settle_water_velocity_deltas(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    waters: mx.array,
    box_lengths: mx.array,
    *,
    periodic: bool,
) -> mx.array:
    """Return analytical SETTLE velocity deltas from one Metal dispatch."""

    positions = as_mx_array(positions, dtype=mx.float32)
    velocities = as_mx_array(velocities, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    waters = as_mx_array(waters, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    water_count = int(waters.shape[0])
    if positions.shape != velocities.shape:
        msg = "positions and velocities must have matching shapes"
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if masses.shape != (positions.shape[0],):
        msg = "masses must have shape (n_atoms,)"
        raise ValueError(msg)
    if waters.ndim != 2 or waters.shape[1] != 3:
        msg = "waters must have shape (n_waters, 3)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if water_count == 0:
        return mx.zeros((0, 3, 3), dtype=mx.float32)
    threads = min(256, water_count)
    (deltas,) = _settle_water_velocity_kernel()(
        inputs=[
            positions,
            velocities,
            masses,
            waters,
            box_lengths,
            mx.array([water_count, int(periodic)], dtype=mx.int32),
        ],
        output_shapes=[(water_count, 3, 3)],
        output_dtypes=[mx.float32],
        grid=(water_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return deltas


def _shake_cluster_position_deltas(
    reference_positions: mx.array,
    predicted_positions: mx.array,
    masses: mx.array,
    cluster_atoms: mx.array,
    peripheral_counts: mx.array,
    distances: mx.array,
    box_lengths: mx.array,
    *,
    max_iterations: int,
    periodic: bool,
) -> mx.array:
    """Return per-cluster SHAKE position deltas from one Metal dispatch."""

    reference_positions = as_mx_array(reference_positions, dtype=mx.float32)
    predicted_positions = as_mx_array(predicted_positions, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    cluster_atoms = as_mx_array(cluster_atoms, dtype=mx.int32)
    peripheral_counts = as_mx_array(peripheral_counts, dtype=mx.int32)
    distances = as_mx_array(distances, dtype=mx.float32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    cluster_count = int(cluster_atoms.shape[0])
    if reference_positions.shape != predicted_positions.shape:
        msg = "reference and predicted positions must have matching shapes"
        raise ValueError(msg)
    if predicted_positions.ndim != 2 or predicted_positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if masses.shape != (predicted_positions.shape[0],):
        msg = "masses must have shape (n_atoms,)"
        raise ValueError(msg)
    if cluster_atoms.ndim != 2 or cluster_atoms.shape[1] != 4:
        msg = "cluster_atoms must have shape (n_clusters, 4)"
        raise ValueError(msg)
    if peripheral_counts.shape != (cluster_count,):
        msg = "peripheral_counts must have shape (n_clusters,)"
        raise ValueError(msg)
    if distances.shape != (cluster_count,):
        msg = "distances must have shape (n_clusters,)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if max_iterations <= 0:
        msg = "max_iterations must be positive"
        raise ValueError(msg)
    if cluster_count == 0:
        return mx.zeros((0, 4, 3), dtype=mx.float32)
    threads = min(256, cluster_count)
    (deltas,) = _shake_cluster_position_kernel()(
        inputs=[
            reference_positions,
            predicted_positions,
            masses,
            cluster_atoms,
            peripheral_counts,
            distances,
            box_lengths,
            mx.array(
                [cluster_count, max_iterations, int(periodic)],
                dtype=mx.int32,
            ),
        ],
        output_shapes=[(cluster_count, 4, 3)],
        output_dtypes=[mx.float32],
        grid=(cluster_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return deltas


def _shake_cluster_velocity_deltas(
    positions: mx.array,
    velocities: mx.array,
    masses: mx.array,
    cluster_atoms: mx.array,
    peripheral_counts: mx.array,
    box_lengths: mx.array,
    *,
    periodic: bool,
) -> mx.array:
    """Return exact per-cluster RATTLE velocity deltas from one Metal dispatch."""

    positions = as_mx_array(positions, dtype=mx.float32)
    velocities = as_mx_array(velocities, dtype=mx.float32)
    masses = as_mx_array(masses, dtype=mx.float32)
    cluster_atoms = as_mx_array(cluster_atoms, dtype=mx.int32)
    peripheral_counts = as_mx_array(peripheral_counts, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    cluster_count = int(cluster_atoms.shape[0])
    if positions.shape != velocities.shape:
        msg = "positions and velocities must have matching shapes"
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if masses.shape != (positions.shape[0],):
        msg = "masses must have shape (n_atoms,)"
        raise ValueError(msg)
    if cluster_atoms.ndim != 2 or cluster_atoms.shape[1] != 4:
        msg = "cluster_atoms must have shape (n_clusters, 4)"
        raise ValueError(msg)
    if peripheral_counts.shape != (cluster_count,):
        msg = "peripheral_counts must have shape (n_clusters,)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    if cluster_count == 0:
        return mx.zeros((0, 4, 3), dtype=mx.float32)
    threads = min(256, cluster_count)
    (deltas,) = _shake_cluster_velocity_kernel()(
        inputs=[
            positions,
            velocities,
            masses,
            cluster_atoms,
            peripheral_counts,
            box_lengths,
            mx.array(
                [cluster_count, 0, int(periodic)],
                dtype=mx.int32,
            ),
        ],
        output_shapes=[(cluster_count, 4, 3)],
        output_dtypes=[mx.float32],
        grid=(cluster_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return deltas


def _fused_langevin_baoab_drift(
    positions: mx.array,
    velocities: mx.array,
    forces: mx.array,
    force_scale_over_mass: mx.array,
    thermal_scale: mx.array,
    noise: mx.array,
    box: mx.array,
    parameters: mx.array,
    counts: mx.array,
) -> tuple[mx.array, mx.array]:
    """Run the first BAOAB kick, both drifts, and Langevin thermostat on Metal."""

    atom_count = int(positions.shape[0])
    if atom_count == 0:
        return positions, velocities
    threads = min(256, atom_count)
    next_positions, middle_velocities = _langevin_baoab_drift_kernel()(
        inputs=[
            positions,
            velocities,
            forces,
            force_scale_over_mass,
            thermal_scale,
            noise,
            box,
            parameters,
            counts,
        ],
        output_shapes=[positions.shape, velocities.shape],
        output_dtypes=[positions.dtype, velocities.dtype],
        grid=(atom_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return next_positions, middle_velocities


def aligned_topology_lj_scales(
    pairs: mx.array,
    excluded_pairs: mx.array,
    one_four_pairs: mx.array,
    *,
    one_four_scale: float,
) -> mx.array:
    """Build pair-aligned topology LJ scales on Metal.

    Args:
        pairs: Candidate atom pairs with shape ``(n_pairs, 2)``.
        excluded_pairs: Sorted excluded pairs with shape ``(n_excluded, 2)``.
        one_four_pairs: Sorted non-excluded 1-4 pairs with shape ``(n_one_four, 2)``.
        one_four_scale: LJ scale assigned to 1-4 pairs.

    Returns:
        Float32 scales with shape ``(n_pairs,)``. Excluded pairs are zero,
        ordinary pairs are one, and 1-4 pairs use ``one_four_scale``.

    Raises:
        ValueError: If any pair array has the wrong shape or the scale is invalid.
    """

    pairs = as_mx_array(pairs, dtype=mx.int32)
    excluded_pairs = as_mx_array(excluded_pairs, dtype=mx.int32)
    one_four_pairs = as_mx_array(one_four_pairs, dtype=mx.int32)
    for name, values in (
        ("pairs", pairs),
        ("excluded_pairs", excluded_pairs),
        ("one_four_pairs", one_four_pairs),
    ):
        if values.ndim != 2 or values.shape[1] != 2:
            msg = f"{name} must have shape (n, 2)"
            raise ValueError(msg)
    if not isfinite(float(one_four_scale)) or float(one_four_scale) < 0.0:
        msg = "one_four_scale must be finite and non-negative"
        raise ValueError(msg)

    pair_count = int(pairs.shape[0])
    if pair_count == 0:
        return mx.zeros((0,), dtype=mx.float32)
    if excluded_pairs.shape[0] == 0 and (
        one_four_pairs.shape[0] == 0 or float(one_four_scale) == 1.0
    ):
        return mx.ones((pair_count,), dtype=mx.float32)

    counts = mx.array(
        [
            pair_count,
            int(excluded_pairs.shape[0]),
            int(one_four_pairs.shape[0]),
        ],
        dtype=mx.int32,
    )
    params = mx.array([float(one_four_scale)], dtype=mx.float32)
    threads = min(256, pair_count)
    (scales,) = _aligned_topology_lj_scales_kernel()(
        inputs=[
            pairs[:, 0],
            pairs[:, 1],
            excluded_pairs[:, 0],
            excluded_pairs[:, 1],
            one_four_pairs[:, 0],
            one_four_pairs[:, 1],
            params,
            counts,
        ],
        output_shapes=[(pair_count,)],
        output_dtypes=[mx.float32],
        grid=(pair_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return scales


def pme_order5_charge_grid(
    positions: mx.array,
    charges: mx.array,
    cell_lengths: mx.array,
    mesh_shape: tuple[int, int, int],
) -> mx.array:
    """Spread particle charges onto a PME mesh with one Metal dispatch.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        charges: Per-atom charges with shape ``(n_atoms,)``.
        cell_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        mesh_shape: Three positive PME mesh dimensions.

    Returns:
        The float32 order-five B-spline charge grid.

    Raises:
        ValueError: If input shapes or mesh dimensions are invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    cell_lengths = as_mx_array(cell_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    atom_count = int(positions.shape[0])
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if cell_lengths.shape != (3,):
        msg = "cell_lengths must have shape (3,)"
        raise ValueError(msg)
    if len(mesh_shape) != 3 or any(int(size) <= 0 for size in mesh_shape):
        msg = "mesh_shape must contain three positive dimensions"
        raise ValueError(msg)
    normalized_mesh = tuple(int(size) for size in mesh_shape)
    grid_size = normalized_mesh[0] * normalized_mesh[1] * normalized_mesh[2]
    mesh = mx.array(normalized_mesh, dtype=mx.int32)
    counts = mx.array([atom_count], dtype=mx.int32)
    work_items = atom_count * 5
    if work_items == 0:
        return mx.zeros(normalized_mesh, dtype=mx.float32)
    threads = min(256, work_items)
    (charge_grid,) = _pme_order5_spread_kernel()(
        inputs=[positions, charges, cell_lengths, mesh, counts],
        output_shapes=[(grid_size,)],
        output_dtypes=[mx.float32],
        grid=(work_items, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return mx.reshape(charge_grid, normalized_mesh)


def pme_order5_energy_forces(
    positions: mx.array,
    charges: mx.array,
    potential_grid: mx.array,
    cell_lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    """Interpolate order-five PME energy and forces with one Metal dispatch.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        charges: Per-atom charges with shape ``(n_atoms,)``.
        potential_grid: Scalar three-dimensional reciprocal potential mesh.
        cell_lengths: Orthorhombic cell lengths with shape ``(3,)``.

    Returns:
        Scalar reciprocal energy and ``(n_atoms, 3)`` forces.

    Raises:
        ValueError: If input shapes are invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    potential_grid = as_mx_array(potential_grid, dtype=mx.float32)
    cell_lengths = as_mx_array(cell_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    atom_count = int(positions.shape[0])
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if potential_grid.ndim != 3:
        msg = "potential_grid must be three-dimensional"
        raise ValueError(msg)
    if cell_lengths.shape != (3,):
        msg = "cell_lengths must have shape (3,)"
        raise ValueError(msg)
    if atom_count == 0:
        return mx.array(0.0, dtype=mx.float32), mx.zeros_like(positions)
    mesh = mx.array(potential_grid.shape, dtype=mx.int32)
    counts = mx.array([atom_count], dtype=mx.int32)
    threads = min(256, atom_count)
    atom_energy, forces = _pme_order5_interpolate_kernel()(
        inputs=[
            positions,
            charges,
            potential_grid,
            cell_lengths,
            mesh,
            counts,
        ],
        output_shapes=[(atom_count,), (atom_count, 3)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(atom_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return mx.sum(atom_energy), forces


def _pme_order5_forces(
    positions: mx.array,
    charges: mx.array,
    potential_grid: mx.array,
    cell_lengths: mx.array,
) -> mx.array:
    """Interpolate order-five PME forces without producing particle energies."""

    positions = as_mx_array(positions, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    potential_grid = as_mx_array(potential_grid, dtype=mx.float32)
    cell_lengths = as_mx_array(cell_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    atom_count = int(positions.shape[0])
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if potential_grid.ndim != 3:
        msg = "potential_grid must be three-dimensional"
        raise ValueError(msg)
    if cell_lengths.shape != (3,):
        msg = "cell_lengths must have shape (3,)"
        raise ValueError(msg)
    if atom_count == 0:
        return mx.zeros_like(positions)
    mesh = mx.array(potential_grid.shape, dtype=mx.int32)
    counts = mx.array([atom_count], dtype=mx.int32)
    threads = min(256, atom_count)
    (forces,) = _pme_order5_force_only_kernel()(
        inputs=[
            positions,
            charges,
            potential_grid,
            cell_lengths,
            mesh,
            counts,
        ],
        output_shapes=[(atom_count, 3)],
        output_dtypes=[mx.float32],
        grid=(atom_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def _pme_order5_forces_from_normalized_real_grid(
    positions: mx.array,
    charges: mx.array,
    potential_grid: mx.array,
    cell_lengths: mx.array,
) -> mx.array:
    """Interpolate forces directly from a normalized real inverse FFT grid."""

    positions = as_mx_array(positions, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    potential_grid = as_mx_array(potential_grid, dtype=mx.float32)
    cell_lengths = as_mx_array(cell_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    atom_count = int(positions.shape[0])
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if potential_grid.ndim != 3:
        msg = "potential_grid must be three-dimensional"
        raise ValueError(msg)
    if cell_lengths.shape != (3,):
        msg = "cell_lengths must have shape (3,)"
        raise ValueError(msg)
    if atom_count == 0:
        return mx.zeros_like(positions)
    mesh = mx.array(potential_grid.shape, dtype=mx.int32)
    counts = mx.array([atom_count], dtype=mx.int32)
    threads = min(256, atom_count)
    (forces,) = _pme_order5_normalized_real_grid_force_only_kernel()(
        inputs=[
            positions,
            charges,
            potential_grid,
            cell_lengths,
            mesh,
            counts,
        ],
        output_shapes=[(atom_count, 3)],
        output_dtypes=[mx.float32],
        grid=(atom_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def _pme_order5_forces_from_complex_grid(
    positions: mx.array,
    charges: mx.array,
    potential_grid: mx.array,
    cell_lengths: mx.array,
) -> mx.array:
    """Interpolate forces directly from an unscaled complex inverse FFT grid."""

    positions = as_mx_array(positions, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    potential_grid = as_mx_array(potential_grid, dtype=mx.complex64)
    cell_lengths = as_mx_array(cell_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    atom_count = int(positions.shape[0])
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if potential_grid.ndim != 3:
        msg = "potential_grid must be three-dimensional"
        raise ValueError(msg)
    if cell_lengths.shape != (3,):
        msg = "cell_lengths must have shape (3,)"
        raise ValueError(msg)
    if atom_count == 0:
        return mx.zeros_like(positions)
    mesh = mx.array(potential_grid.shape, dtype=mx.int32)
    counts = mx.array([atom_count], dtype=mx.int32)
    threads = min(256, atom_count)
    (forces,) = _pme_order5_complex_grid_force_only_kernel()(
        inputs=[
            positions,
            charges,
            potential_grid,
            cell_lengths,
            mesh,
            counts,
        ],
        output_shapes=[(atom_count, 3)],
        output_dtypes=[mx.float32],
        grid=(atom_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def _fused_bonded_force_only(
    positions: mx.array,
    box_lengths: mx.array,
    bond_atoms: mx.array,
    bond_k: mx.array,
    bond_length: mx.array,
    angle_atoms: mx.array,
    angle_k: mx.array,
    angle_target: mx.array,
    dihedral_atoms: mx.array,
    dihedral_k: mx.array,
    dihedral_periodicity: mx.array,
    dihedral_phase: mx.array,
    improper_atoms: mx.array,
    improper_k: mx.array,
    improper_periodicity: mx.array,
    improper_phase: mx.array,
    cmap_atoms: mx.array,
    cmap_indices: mx.array,
    cmap_coefficients: mx.array,
    correction_atoms: mx.array | None = None,
    correction_charge_products: mx.array | None = None,
    correction_lj_sigma: mx.array | None = None,
    correction_lj_epsilon: mx.array | None = None,
    correction_coulomb_constant: float = 0.0,
) -> mx.array:
    """Evaluate bonded families and optional sparse PME corrections in one dispatch."""

    positions = as_mx_array(positions, dtype=mx.float32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    bond_atoms = as_mx_array(bond_atoms, dtype=mx.int32)
    bond_k = as_mx_array(bond_k, dtype=mx.float32)
    bond_length = as_mx_array(bond_length, dtype=mx.float32)
    angle_atoms = as_mx_array(angle_atoms, dtype=mx.int32)
    angle_k = as_mx_array(angle_k, dtype=mx.float32)
    angle_target = as_mx_array(angle_target, dtype=mx.float32)
    dihedral_atoms = as_mx_array(dihedral_atoms, dtype=mx.int32)
    dihedral_k = as_mx_array(dihedral_k, dtype=mx.float32)
    dihedral_periodicity = as_mx_array(dihedral_periodicity, dtype=mx.float32)
    dihedral_phase = as_mx_array(dihedral_phase, dtype=mx.float32)
    improper_atoms = as_mx_array(improper_atoms, dtype=mx.int32)
    improper_k = as_mx_array(improper_k, dtype=mx.float32)
    improper_periodicity = as_mx_array(improper_periodicity, dtype=mx.float32)
    improper_phase = as_mx_array(improper_phase, dtype=mx.float32)
    cmap_atoms = as_mx_array(cmap_atoms, dtype=mx.int32)
    cmap_indices = as_mx_array(cmap_indices, dtype=mx.int32)
    cmap_coefficients = as_mx_array(cmap_coefficients, dtype=mx.float32)
    correction_inputs = (
        correction_atoms,
        correction_charge_products,
        correction_lj_sigma,
        correction_lj_epsilon,
    )
    has_corrections = any(value is not None for value in correction_inputs)
    if has_corrections and not all(value is not None for value in correction_inputs):
        msg = "sparse PME correction inputs must be provided together"
        raise ValueError(msg)
    if has_corrections:
        correction_atoms = as_mx_array(correction_atoms, dtype=mx.int32)
        correction_charge_products = as_mx_array(
            correction_charge_products,
            dtype=mx.float32,
        )
        correction_lj_sigma = as_mx_array(correction_lj_sigma, dtype=mx.float32)
        correction_lj_epsilon = as_mx_array(correction_lj_epsilon, dtype=mx.float32)
    else:
        correction_atoms = mx.zeros((0, 2), dtype=mx.int32)
        correction_charge_products = mx.zeros((0,), dtype=mx.float32)
        correction_lj_sigma = mx.zeros((0,), dtype=mx.float32)
        correction_lj_epsilon = mx.zeros((0,), dtype=mx.float32)

    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if box_lengths.shape != (3,):
        msg = "box_lengths must have shape (3,)"
        raise ValueError(msg)
    counts = (
        bond_atoms.shape[0],
        angle_atoms.shape[0],
        dihedral_atoms.shape[0],
        improper_atoms.shape[0],
        cmap_atoms.shape[0],
        correction_atoms.shape[0],
    )
    arrays = (
        (bond_atoms, bond_k, bond_length, 2, "bond"),
        (angle_atoms, angle_k, angle_target, 3, "angle"),
        (
            dihedral_atoms,
            dihedral_k,
            dihedral_periodicity,
            4,
            "dihedral",
        ),
        (
            improper_atoms,
            improper_k,
            improper_periodicity,
            4,
            "improper",
        ),
    )
    for atoms, first_parameter, second_parameter, width, family in arrays:
        count = atoms.shape[0]
        if atoms.ndim != 2 or atoms.shape[1] != width:
            msg = f"{family}_atoms must have shape (n, {width})"
            raise ValueError(msg)
        if first_parameter.shape != (count,) or second_parameter.shape != (count,):
            msg = f"{family} parameters must have shape ({count},)"
            raise ValueError(msg)
    if dihedral_phase.shape != (counts[2],):
        msg = f"dihedral_phase must have shape ({counts[2]},)"
        raise ValueError(msg)
    if improper_phase.shape != (counts[3],):
        msg = f"improper_phase must have shape ({counts[3]},)"
        raise ValueError(msg)
    if cmap_atoms.ndim != 2 or cmap_atoms.shape[1] != 8:
        msg = "cmap_atoms must have shape (n, 8)"
        raise ValueError(msg)
    if cmap_indices.shape != (counts[4],):
        msg = f"cmap_indices must have shape ({counts[4]},)"
        raise ValueError(msg)
    if cmap_coefficients.ndim != 5 or cmap_coefficients.shape[-2:] != (4, 4):
        msg = "cmap_coefficients must have shape (n_maps, grid, grid, 4, 4)"
        raise ValueError(msg)
    if cmap_coefficients.shape[1] != cmap_coefficients.shape[2]:
        msg = "cmap coefficient grids must be square"
        raise ValueError(msg)
    if correction_atoms.ndim != 2 or correction_atoms.shape[1] != 2:
        msg = "correction_atoms must have shape (n, 2)"
        raise ValueError(msg)
    for name, values in (
        ("correction_charge_products", correction_charge_products),
        ("correction_lj_sigma", correction_lj_sigma),
        ("correction_lj_epsilon", correction_lj_epsilon),
    ):
        if values.shape != (counts[5],):
            msg = f"{name} must have shape ({counts[5]},)"
            raise ValueError(msg)
    if not isfinite(float(correction_coulomb_constant)):
        msg = "correction_coulomb_constant must be finite"
        raise ValueError(msg)

    total_count = sum(counts)
    if total_count == 0:
        return mx.zeros_like(positions)
    count_array = mx.array(
        (*counts[:5], cmap_coefficients.shape[1], counts[5]),
        dtype=mx.int32,
    )
    threads = min(256, total_count)
    (forces,) = _fused_bonded_force_only_kernel()(
        inputs=[
            positions,
            box_lengths,
            mx.reshape(bond_atoms, (-1,)),
            bond_k,
            bond_length,
            mx.reshape(angle_atoms, (-1,)),
            angle_k,
            angle_target,
            mx.reshape(dihedral_atoms, (-1,)),
            dihedral_k,
            dihedral_periodicity,
            dihedral_phase,
            mx.reshape(improper_atoms, (-1,)),
            improper_k,
            improper_periodicity,
            improper_phase,
            mx.reshape(cmap_atoms, (-1,)),
            cmap_indices,
            mx.reshape(cmap_coefficients, (-1,)),
            mx.reshape(correction_atoms, (-1,)),
            correction_charge_products,
            correction_lj_sigma,
            correction_lj_epsilon,
            mx.array([float(correction_coulomb_constant)], dtype=mx.float32),
            count_array,
        ],
        output_shapes=[positions.shape],
        output_dtypes=[mx.float32],
        grid=(total_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def fused_lj_forces(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float,
    shift: bool,
) -> tuple[mx.array, mx.array]:
    """Fused LJ energy + forces via a single Metal kernel (orthorhombic, scalar LJ).

    Mirrors ``LennardJonesPotential._pair_energy_forces`` semantics: a half neighbor
    list ``pairs`` of shape ``(M, 2)``, an ``r^2`` cutoff mask, and an optional energy
    shift at the cutoff. ``box_lengths`` are the orthorhombic edge lengths
    (``mx.diag(cell.matrix)``). Returns ``(energy_scalar, forces)`` with forces
    shape ``(N, 3)``.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)
    if cutoff is None:
        msg = "fused_lj_forces requires a finite cutoff"
        raise ValueError(msg)

    pairs_i = pairs[:, 0]
    pairs_j = pairs[:, 1]

    sigma2 = float(sigma) * float(sigma)
    cut2 = float(cutoff) * float(cutoff)
    if shift:
        sig2_over_rc2 = sigma2 / cut2
        inv_rc6 = sig2_over_rc2 * sig2_over_rc2 * sig2_over_rc2
        e_shift = 4.0 * float(epsilon) * (inv_rc6 * inv_rc6 - inv_rc6)
    else:
        e_shift = 0.0
    params = mx.array([float(epsilon), sigma2, cut2, e_shift], dtype=mx.float32)
    box = as_mx_array(box_lengths, dtype=mx.float32)
    npair = mx.array([n_pairs], dtype=mx.int32)

    threads = 256 if n_pairs >= 256 else n_pairs
    forces, pair_energy = _lj_force_kernel()(
        inputs=[positions, pairs_i, pairs_j, box, params, npair],
        output_shapes=[(n_atoms, 3), (n_pairs,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_pairs, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return mx.sum(pair_energy), forces


def fused_parameterized_lj_forces(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
) -> tuple[mx.array, mx.array]:
    """Evaluate parameterized LJ energy and forces with one Metal dispatch.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs: Half-neighbor pairs with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        scales: Either one shared scale or one scale per pair.
        cutoff: Finite LJ cutoff.
        shift: Whether to subtract each pair's cutoff energy.
        switch_distance: Optional start of the smooth potential switch.

    Returns:
        Scalar LJ energy and an ``(n_atoms, 3)`` force array.

    Raises:
        ValueError: If the cutoff or scale count is invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    sigma = as_mx_array(sigma, dtype=mx.float32)
    epsilon = as_mx_array(epsilon, dtype=mx.float32)
    scales = as_mx_array(scales, dtype=mx.float32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        return mx.sum(positions[:, 0] * 0.0), mx.zeros_like(positions)
    if cutoff is None or cutoff <= 0.0:
        msg = "fused_parameterized_lj_forces requires a positive cutoff"
        raise ValueError(msg)
    scale_count = int(scales.size)
    if scale_count not in {1, n_pairs}:
        msg = "scales must contain one value or one value per pair"
        raise ValueError(msg)
    scales = mx.reshape(scales, (scale_count,))

    pairs_i = pairs[:, 0]
    pairs_j = pairs[:, 1]
    cutoff_value = float(cutoff)
    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - float(switch_distance)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(has_switch),
            switch_value,
            switch_width,
            cutoff_value,
        ],
        dtype=mx.float32,
    )
    counts = mx.array([n_pairs, scale_count], dtype=mx.int32)
    box = as_mx_array(box_lengths, dtype=mx.float32)
    threads = 256 if n_pairs >= 256 else n_pairs
    forces, pair_energy = _parameterized_lj_force_kernel()(
        inputs=[
            positions,
            pairs_i,
            pairs_j,
            box,
            sigma,
            epsilon,
            scales,
            params,
            counts,
        ],
        output_shapes=[(n_atoms, 3), (n_pairs,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_pairs, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return mx.sum(pair_energy), forces


def fused_parameterized_pme_direct_components(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Evaluate separated LJ/PME direct energies and combined forces in one dispatch.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs: Shared half-neighbor candidates with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        charges: Per-atom partial charges.
        lj_scales: One aligned LJ scale per candidate; zero excludes LJ only.
        cutoff: Shared finite LJ and PME real-space cutoff.
        shift: Whether to subtract each LJ pair's cutoff energy.
        switch_distance: Optional start of the smooth LJ potential switch.
        coulomb_constant: Coulomb prefactor in the configured units.
        alpha: Ewald splitting parameter.

    Returns:
        Combined energy, ``(n_atoms, 3)`` forces, LJ energy, and direct-space
        Coulomb energy.

    Raises:
        ValueError: If the cutoff, alpha, or aligned scale count is invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    sigma = as_mx_array(sigma, dtype=mx.float32)
    epsilon = as_mx_array(epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    lj_scales = as_mx_array(lj_scales, dtype=mx.float32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        zero = mx.sum(positions[:, 0] * 0.0)
        return zero, mx.zeros_like(positions), zero, zero
    if cutoff is None or cutoff <= 0.0:
        msg = "fused_parameterized_pme_direct_forces requires a positive cutoff"
        raise ValueError(msg)
    if alpha <= 0.0:
        msg = "fused_parameterized_pme_direct_forces requires positive alpha"
        raise ValueError(msg)
    if int(lj_scales.size) != n_pairs:
        msg = "lj_scales must contain one aligned value per pair"
        raise ValueError(msg)
    lj_scales = mx.reshape(lj_scales, (n_pairs,))

    cutoff_value = float(cutoff)
    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - float(switch_distance)
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(has_switch),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
        ],
        dtype=mx.float32,
    )
    npair = mx.array([n_pairs], dtype=mx.int32)
    box = as_mx_array(box_lengths, dtype=mx.float32)
    threads = 256 if n_pairs >= 256 else n_pairs
    forces, pair_lj_energy, pair_coulomb_energy = _parameterized_pme_direct_kernel()(
        inputs=[
            positions,
            pairs[:, 0],
            pairs[:, 1],
            box,
            sigma,
            epsilon,
            charges,
            lj_scales,
            params,
            npair,
        ],
        output_shapes=[
            (n_atoms, 3),
            (n_pairs,),
            (n_pairs,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_pairs, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    lj_energy = mx.sum(pair_lj_energy)
    coulomb_energy = mx.sum(pair_coulomb_energy)
    return lj_energy + coulomb_energy, forces, lj_energy, coulomb_energy


def fused_parameterized_pme_direct_components_virial(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Evaluate direct components, forces, and atomic diagonal virial.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs: Shared half-neighbor candidates with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        charges: Per-atom partial charges.
        lj_scales: One aligned LJ scale per candidate; zero excludes LJ only.
        cutoff: Shared finite LJ and PME real-space cutoff.
        shift: Whether to subtract each LJ pair's cutoff energy.
        switch_distance: Optional start of the smooth LJ potential switch.
        coulomb_constant: Coulomb prefactor in the configured units.
        alpha: Ewald splitting parameter.

    Returns:
        Combined energy, forces, LJ energy, direct Coulomb energy, and the
        three diagonal atomic-virial components.

    Raises:
        ValueError: If the cutoff, alpha, or aligned scale count is invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    sigma = as_mx_array(sigma, dtype=mx.float32)
    epsilon = as_mx_array(epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    lj_scales = as_mx_array(lj_scales, dtype=mx.float32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        zero = mx.sum(positions[:, 0] * 0.0)
        return (
            zero,
            mx.zeros_like(positions),
            zero,
            zero,
            mx.zeros((3,), dtype=mx.float32),
        )
    if cutoff is None or cutoff <= 0.0:
        msg = "fused_parameterized_pme_direct_components_virial requires a positive cutoff"
        raise ValueError(msg)
    if alpha <= 0.0:
        msg = "fused_parameterized_pme_direct_components_virial requires positive alpha"
        raise ValueError(msg)
    if int(lj_scales.size) != n_pairs:
        msg = "lj_scales must contain one aligned value per pair"
        raise ValueError(msg)
    lj_scales = mx.reshape(lj_scales, (n_pairs,))

    cutoff_value = float(cutoff)
    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - float(switch_distance)
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(has_switch),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
        ],
        dtype=mx.float32,
    )
    npair = mx.array([n_pairs], dtype=mx.int32)
    box = as_mx_array(box_lengths, dtype=mx.float32)
    threads = 256 if n_pairs >= 256 else n_pairs
    forces, pair_lj_energy, pair_coulomb_energy, pair_virial = (
        _parameterized_pme_direct_virial_kernel()(
            inputs=[
                positions,
                pairs[:, 0],
                pairs[:, 1],
                box,
                sigma,
                epsilon,
                charges,
                lj_scales,
                params,
                npair,
            ],
            output_shapes=[
                (n_atoms, 3),
                (n_pairs,),
                (n_pairs,),
                (n_pairs, 3),
            ],
            output_dtypes=[
                mx.float32,
                mx.float32,
                mx.float32,
                mx.float32,
            ],
            grid=(n_pairs, 1, 1),
            threadgroup=(threads, 1, 1),
            init_value=0.0,
        )
    )
    lj_energy = mx.sum(pair_lj_energy)
    coulomb_energy = mx.sum(pair_coulomb_energy)
    return (
        lj_energy + coulomb_energy,
        forces,
        lj_energy,
        coulomb_energy,
        mx.sum(pair_virial, axis=0),
    )


def fused_pme_cutoff_correction_virial(
    positions: mx.array,
    molecule_centers: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    strain_epsilon: float,
    coulomb_constant: float,
    alpha: float,
    include_lj: bool,
    include_coulomb: bool,
) -> mx.array:
    """Evaluate the finite-strain cutoff correction in one Metal dispatch.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        molecule_centers: Per-atom geometric molecule centers, shape
            ``(n_atoms, 3)``.
        pairs: Compact cutoff-shell pairs with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        charges: Per-atom partial charges.
        lj_scales: One aligned LJ scale per candidate; zero excludes LJ.
        cutoff: Shared finite LJ and PME real-space cutoff.
        strain_epsilon: Positive central finite-strain displacement.
        coulomb_constant: Coulomb prefactor in the configured units.
        alpha: Ewald splitting parameter.
        include_lj: Whether to include the unswitched LJ correction.
        include_coulomb: Whether to include the PME real-space boundary term.

    Returns:
        A diagonal ``(3, 3)`` cutoff-correction virial tensor.

    Raises:
        ValueError: If shapes or positive scalar parameters are invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    molecule_centers = as_mx_array(molecule_centers, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    sigma = as_mx_array(sigma, dtype=mx.float32)
    epsilon = as_mx_array(epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    lj_scales = as_mx_array(lj_scales, dtype=mx.float32)
    atom_count = int(positions.shape[0])
    pair_count = int(pairs.shape[0])
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if molecule_centers.shape != positions.shape:
        msg = "molecule_centers must match positions"
        raise ValueError(msg)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        msg = "pairs must have shape (n_pairs, 2)"
        raise ValueError(msg)
    if sigma.shape != (atom_count,) or epsilon.shape != (atom_count,):
        msg = "sigma and epsilon must have shape (n_atoms,)"
        raise ValueError(msg)
    if charges.shape != (atom_count,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if int(lj_scales.size) != pair_count:
        msg = "lj_scales must contain one aligned value per pair"
        raise ValueError(msg)
    if cutoff <= 0.0 or strain_epsilon <= 0.0 or alpha <= 0.0:
        msg = "cutoff, strain_epsilon, and alpha must be positive"
        raise ValueError(msg)
    if pair_count == 0 or not (include_lj or include_coulomb):
        return mx.zeros((3, 3), dtype=mx.float32)

    params = mx.array(
        [
            float(cutoff) * float(cutoff),
            float(cutoff),
            float(strain_epsilon),
            float(coulomb_constant),
            float(alpha),
            float(include_lj),
            float(include_coulomb),
        ],
        dtype=mx.float32,
    )
    npair = mx.array([pair_count], dtype=mx.int32)
    threads = min(256, pair_count)
    (pair_correction,) = _pme_cutoff_correction_virial_kernel()(
        inputs=[
            positions,
            molecule_centers,
            pairs[:, 0],
            pairs[:, 1],
            as_mx_array(box_lengths, dtype=mx.float32),
            sigma,
            epsilon,
            charges,
            mx.reshape(lj_scales, (pair_count,)),
            params,
            npair,
        ],
        output_shapes=[(pair_count, 3)],
        output_dtypes=[mx.float32],
        grid=(pair_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return mx.diag(mx.sum(pair_correction, axis=0))


def fused_parameterized_pme_direct_forces(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> tuple[mx.array, mx.array]:
    """Evaluate combined LJ plus PME direct-space energy and forces.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs: Shared half-neighbor candidates with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        charges: Per-atom partial charges.
        lj_scales: One aligned LJ scale per candidate; zero excludes LJ only.
        cutoff: Shared finite LJ and PME real-space cutoff.
        shift: Whether to subtract each LJ pair's cutoff energy.
        switch_distance: Optional start of the smooth LJ potential switch.
        coulomb_constant: Coulomb prefactor in the configured units.
        alpha: Ewald splitting parameter.

    Returns:
        Combined scalar direct-space energy and ``(n_atoms, 3)`` forces.
    """

    energy, forces, _, _ = fused_parameterized_pme_direct_components(
        positions,
        pairs,
        box_lengths,
        sigma,
        epsilon,
        charges,
        lj_scales,
        cutoff=cutoff,
        shift=shift,
        switch_distance=switch_distance,
        coulomb_constant=coulomb_constant,
        alpha=alpha,
    )
    return energy, forces


def fused_parameterized_pme_direct_force_only(
    positions: mx.array,
    pairs: mx.array,
    box_lengths: mx.array,
    sigma: mx.array,
    epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> mx.array:
    """Evaluate combined LJ plus PME direct-space forces without energy outputs.

    Args:
        positions: Atomic coordinates with shape ``(n_atoms, 3)``.
        pairs: Shared half-neighbor candidates with shape ``(n_pairs, 2)``.
        box_lengths: Orthorhombic cell lengths with shape ``(3,)``.
        sigma: Per-atom LJ sigma values.
        epsilon: Per-atom LJ epsilon values.
        charges: Per-atom partial charges.
        lj_scales: One aligned LJ scale per candidate; zero excludes LJ only.
        cutoff: Shared finite LJ and PME real-space cutoff.
        shift: Whether to subtract each LJ pair's cutoff energy.
        switch_distance: Optional start of the smooth LJ potential switch.
        coulomb_constant: Coulomb prefactor in the configured units.
        alpha: Ewald splitting parameter.

    Returns:
        An ``(n_atoms, 3)`` force array. The kernel allocates no per-pair
        energy outputs.

    Raises:
        ValueError: If the cutoff, alpha, or aligned scale count is invalid.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    sigma = as_mx_array(sigma, dtype=mx.float32)
    epsilon = as_mx_array(epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    lj_scales = as_mx_array(lj_scales, dtype=mx.float32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        return mx.zeros_like(positions)
    if cutoff is None or cutoff <= 0.0:
        msg = "fused_parameterized_pme_direct_force_only requires a positive cutoff"
        raise ValueError(msg)
    if alpha <= 0.0:
        msg = "fused_parameterized_pme_direct_force_only requires positive alpha"
        raise ValueError(msg)
    if int(lj_scales.size) != n_pairs:
        msg = "lj_scales must contain one aligned value per pair"
        raise ValueError(msg)
    lj_scales = mx.reshape(lj_scales, (n_pairs,))

    cutoff_value = float(cutoff)
    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - float(switch_distance)
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(has_switch),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
        ],
        dtype=mx.float32,
    )
    npair = mx.array([n_pairs], dtype=mx.int32)
    box = as_mx_array(box_lengths, dtype=mx.float32)
    threads = 256 if n_pairs >= 256 else n_pairs
    (forces,) = _parameterized_pme_direct_force_only_kernel()(
        inputs=[
            positions,
            pairs[:, 0],
            pairs[:, 1],
            box,
            sigma,
            epsilon,
            charges,
            lj_scales,
            params,
            npair,
        ],
        output_shapes=[(n_atoms, 3)],
        output_dtypes=[mx.float32],
        grid=(n_pairs, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def _prepared_parameterized_pme_direct_force_only(
    positions: mx.array,
    pairs: mx.array,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    lj_scales: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    coulomb_constant: float,
    alpha: float,
) -> mx.array:
    """Evaluate prepared LJ plus PME direct forces with cached invariants."""

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    lj_scales = as_mx_array(lj_scales, dtype=mx.float32)
    n_atoms = positions.shape[0]
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        return mx.zeros_like(positions)
    if cutoff is None or cutoff <= 0.0:
        msg = "prepared PME direct forces require a positive cutoff"
        raise ValueError(msg)
    if alpha <= 0.0:
        msg = "prepared PME direct forces require positive alpha"
        raise ValueError(msg)
    if box.shape != (6,):
        msg = "box_lengths_and_inverses must have shape (6,)"
        raise ValueError(msg)
    if half_sigma.shape != (n_atoms,) or sqrt_epsilon.shape != (n_atoms,):
        msg = "prepared LJ parameters must have shape (n_atoms,)"
        raise ValueError(msg)
    if charges.shape != (n_atoms,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if int(lj_scales.size) != n_pairs:
        msg = "lj_scales must contain one aligned value per pair"
        raise ValueError(msg)
    lj_scales = mx.reshape(lj_scales, (n_pairs,))

    cutoff_value = float(cutoff)
    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - switch_value
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(has_switch),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
            1.0 / switch_width,
        ],
        dtype=mx.float32,
    )
    npair = mx.array([n_pairs], dtype=mx.int32)
    worker_count = (n_pairs + _PREPARED_PME_PAIRS_PER_WORKER - 1) // _PREPARED_PME_PAIRS_PER_WORKER
    threads = min(64, worker_count)
    (forces,) = _prepared_pme_direct_force_only_kernel()(
        inputs=[
            positions,
            pairs[:, 0],
            pairs[:, 1],
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            lj_scales,
            params,
            npair,
        ],
        output_shapes=[(n_atoms, 3)],
        output_dtypes=[mx.float32],
        grid=(worker_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0.0,
    )
    return forces


def _tile_parameterized_pme_direct_force_only(
    positions: mx.array,
    atom_blocks: mx.array,
    tile_blocks: mx.array,
    member_mask: mx.array,
    lj_enabled_mask: mx.array,
    lj_one_four_mask: mx.array,
    force_columns: mx.array,
    force_group_starts: mx.array,
    force_group_counts: mx.array,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    atom_type_ids: mx.array | None = None,
    nbfix_type_sigma: mx.array | None = None,
    nbfix_type_epsilon: mx.array | None = None,
    nbfix_type_count: int = 0,
    _return_energy: bool = False,
) -> mx.array | tuple[mx.array, mx.array, mx.array]:
    """Evaluate prepared direct forces from exact spatial 4x4 tiles.

    Type-pair NBFIX parameters select a dedicated kernel specialization. The
    ordinary tile kernel and its input contract remain unchanged when the
    optional lookup table is absent.
    """

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    lj_enabled_mask = as_mx_array(lj_enabled_mask, dtype=mx.uint32)
    lj_one_four_mask = as_mx_array(lj_one_four_mask, dtype=mx.uint32)
    force_columns = as_mx_array(force_columns, dtype=mx.int32)
    force_group_starts = as_mx_array(force_group_starts, dtype=mx.int32)
    force_group_counts = as_mx_array(force_group_counts, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    nbfix_inputs = (atom_type_ids, nbfix_type_sigma, nbfix_type_epsilon)
    has_nbfix = any(value is not None for value in nbfix_inputs)
    if has_nbfix and not all(value is not None for value in nbfix_inputs):
        msg = "NBFIX tile inputs must be provided together"
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    n_atoms = int(positions.shape[0])
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = f"atom_blocks must have shape (n_blocks, {_TILE_PME_BLOCK_SIZE})"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    tile_count = int(tile_blocks.shape[0])
    mask_shape = (tile_count, _TILE_PME_MASK_WORD_COUNT)
    if member_mask.shape != mask_shape:
        msg = f"member_mask must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    if lj_enabled_mask.shape != mask_shape or lj_one_four_mask.shape != mask_shape:
        msg = f"tile LJ masks must have shape (n_tiles, {_TILE_PME_MASK_WORD_COUNT})"
        raise ValueError(msg)
    if force_columns.ndim != 1:
        msg = "force_columns must be a vector"
        raise ValueError(msg)
    column_count = int(force_columns.shape[0])
    group_count = int(force_group_starts.shape[0])
    if force_group_starts.ndim != 1 or force_group_counts.shape != (group_count,):
        msg = "tile force group starts and counts must have matching vector shapes"
        raise ValueError(msg)
    if box.shape != (6,):
        msg = "box_lengths_and_inverses must have shape (6,)"
        raise ValueError(msg)
    if half_sigma.shape != (n_atoms,) or sqrt_epsilon.shape != (n_atoms,):
        msg = "prepared LJ parameters must have shape (n_atoms,)"
        raise ValueError(msg)
    if charges.shape != (n_atoms,):
        msg = "charges must have shape (n_atoms,)"
        raise ValueError(msg)
    if has_nbfix:
        atom_type_ids = as_mx_array(atom_type_ids, dtype=mx.int32)
        nbfix_type_sigma = as_mx_array(nbfix_type_sigma, dtype=mx.float32)
        nbfix_type_epsilon = as_mx_array(nbfix_type_epsilon, dtype=mx.float32)
        if atom_type_ids.shape != (n_atoms,):
            msg = "atom_type_ids must have shape (n_atoms,)"
            raise ValueError(msg)
        if nbfix_type_count <= 0:
            msg = "nbfix_type_count must be positive when NBFIX inputs are provided"
            raise ValueError(msg)
        table_shape = (nbfix_type_count * nbfix_type_count,)
        if (
            nbfix_type_sigma.shape != table_shape
            or nbfix_type_epsilon.shape != table_shape
        ):
            msg = "NBFIX parameter tables must have shape (nbfix_type_count ** 2,)"
            raise ValueError(msg)
    elif nbfix_type_count != 0:
        msg = "nbfix_type_count requires NBFIX tile inputs"
        raise ValueError(msg)
    if cutoff is None or not isfinite(float(cutoff)) or cutoff <= 0.0:
        msg = "tile PME direct forces require a finite positive cutoff"
        raise ValueError(msg)
    if not isfinite(float(alpha)) or alpha <= 0.0:
        msg = "tile PME direct forces require finite positive alpha"
        raise ValueError(msg)
    if not isfinite(float(coulomb_constant)):
        msg = "coulomb_constant must be finite"
        raise ValueError(msg)
    if not isfinite(float(one_four_scale)) or one_four_scale < 0.0:
        msg = "one_four_scale must be finite and non-negative"
        raise ValueError(msg)
    cutoff_value = float(cutoff)
    if switch_distance is not None and (
        not isfinite(float(switch_distance))
        or float(switch_distance) < 0.0
        or float(switch_distance) >= cutoff_value
    ):
        msg = "switch_distance must be finite, non-negative, and below cutoff"
        raise ValueError(msg)
    if tile_count == 0:
        forces = mx.zeros_like(positions)
        if _return_energy:
            zero = mx.array(0.0, dtype=mx.float32)
            return zero, zero, forces
        return forces
    if column_count == 0 or group_count == 0:
        msg = "non-empty tile geometry requires force columns and groups"
        raise ValueError(msg)

    has_switch = switch_distance is not None
    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - switch_value
    alpha_value = float(alpha)
    parameter_values = [
        cutoff_value * cutoff_value,
        float(bool(shift)),
        float(has_switch),
        switch_value,
        switch_width,
        cutoff_value,
        float(coulomb_constant),
        alpha_value,
        2.0 * alpha_value / sqrt(pi),
        1.0 / switch_width,
        float(one_four_scale),
    ]
    if has_nbfix:
        parameter_values.append(float(nbfix_type_count))
    params = mx.array(parameter_values, dtype=mx.float32)
    inputs = [
        positions,
        atom_blocks,
        tile_blocks,
        lj_enabled_mask,
        lj_one_four_mask,
        force_columns,
        force_group_starts,
        force_group_counts,
        box,
        half_sigma,
        sqrt_epsilon,
        charges,
    ]
    if has_nbfix:
        inputs.extend(
            (atom_type_ids, nbfix_type_sigma, nbfix_type_epsilon),
        )
    inputs.extend((params, mx.array([group_count], dtype=mx.int32)))
    dispatch = {
        "grid": (
            _tile_pme_threadgroup_count(group_count) * _TILE_PME_DISPATCH_WIDTH,
            1,
            1,
        ),
        "threadgroup": (_TILE_PME_DISPATCH_WIDTH, 1, 1),
        "init_value": 0.0,
    }
    if _return_energy:
        kernel = (
            _tile_nbfix_pme_direct_kernel()
            if has_nbfix
            else _tile_pme_direct_kernel()
        )
        forces, group_lj_energy, group_coulomb_energy = kernel(
            inputs=inputs,
            output_shapes=[(n_atoms, 3), (group_count,), (group_count,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            **dispatch,
        )
        return mx.sum(group_lj_energy), mx.sum(group_coulomb_energy), forces
    kernel = (
        _tile_nbfix_pme_direct_force_only_kernel()
        if has_nbfix
        else _tile_pme_direct_force_only_kernel()
    )
    (forces,) = kernel(
        inputs=inputs,
        output_shapes=[(n_atoms, 3)],
        output_dtypes=[mx.float32],
        **dispatch,
    )
    return forces


def _interaction32_block_geometry(
    positions: mx.array,
    atom_order: mx.array,
    box_lengths_and_inverses: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compute periodic centers, radii, and extents for packed atom blocks."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions.shape[0])
    if atom_count == 0:
        raise ValueError("block geometry requires at least one atom")
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    if int(atom_order.shape[0]) < atom_count:
        raise ValueError("atom_order cannot be shorter than the atom count")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    block_count = int(atom_order.shape[0]) // 32
    return _interaction32_block_geometry_kernel()(
        inputs=[
            positions,
            atom_order,
            box,
            mx.array([block_count, atom_count], dtype=mx.int32),
        ],
        output_shapes=[(block_count, 4), (block_count, 3)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(block_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=0.0,
    )


def _interaction32_ordinary_mode_counts(
    positions: mx.array,
    atom_order: mx.array,
    center_radius: mx.array,
    half_extent: mx.array,
    block_traversal: mx.array,
    special_pair_words: mx.array,
    box_lengths_and_inverses: mx.array,
    *,
    search_radius: float,
    retain_modes: bool = True,
) -> tuple[mx.array, mx.array | None]:
    """Count ordinary modes and retain a packed two-bit membership cache."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    center_radius = as_mx_array(center_radius, dtype=mx.float32)
    half_extent = as_mx_array(half_extent, dtype=mx.float32)
    block_traversal = as_mx_array(block_traversal, dtype=mx.int32)
    special_pair_words = as_mx_array(special_pair_words, dtype=mx.uint32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    block_count = int(atom_order.shape[0]) // 32
    if center_radius.shape != (block_count, 4):
        raise ValueError("center_radius must have shape (n_blocks, 4)")
    if half_extent.shape != (block_count, 3):
        raise ValueError("half_extent must have shape (n_blocks, 3)")
    if block_traversal.shape != (block_count,):
        raise ValueError("block_traversal must have shape (n_blocks,)")
    special_word_shape = ((block_count * block_count + 31) // 32,)
    if special_pair_words.shape != special_word_shape:
        raise ValueError("special_pair_words must cover the dense block-code space")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    radius = float(search_radius)
    block_pair_count = block_count * (block_count - 1) // 2
    inputs = [
        positions,
        atom_order,
        center_radius,
        half_extent,
        block_traversal,
        special_pair_words,
        box,
        mx.array([radius, radius * radius], dtype=mx.float32),
        mx.array([block_count], dtype=mx.int32),
    ]
    if retain_modes:
        mode_counts, mode_words = _interaction32_ordinary_cached_count_kernel()(
            inputs=inputs,
            output_shapes=[(block_count, 3), (2 * block_pair_count,)],
            output_dtypes=[mx.int32, mx.uint32],
            grid=(block_count * 32, 1, 1),
            threadgroup=(32, 1, 1),
            init_value=0,
        )
        return mode_counts, mode_words
    (mode_counts,) = _interaction32_ordinary_count_kernel()(
        inputs=inputs,
        output_shapes=[(block_count, 3)],
        output_dtypes=[mx.int32],
        grid=(block_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=0,
    )
    return mode_counts, None


def _interaction32_ordinary_scatter_sized(
    positions: mx.array,
    atom_order: mx.array,
    center_radius: mx.array,
    half_extent: mx.array,
    block_traversal: mx.array,
    special_pair_words: mx.array,
    mode_words: mx.array | None,
    mode_tile_counts: mx.array,
    mode_tile_prefix: mx.array,
    box_lengths_and_inverses: mx.array,
    *,
    search_radius: float,
    accepted_tile_count: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Scatter compact ordinary rows from packed two-bit membership modes."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    center_radius = as_mx_array(center_radius, dtype=mx.float32)
    half_extent = as_mx_array(half_extent, dtype=mx.float32)
    block_traversal = as_mx_array(block_traversal, dtype=mx.int32)
    special_pair_words = as_mx_array(special_pair_words, dtype=mx.uint32)
    if mode_words is not None:
        mode_words = as_mx_array(mode_words, dtype=mx.uint32)
    mode_tile_counts = as_mx_array(mode_tile_counts, dtype=mx.int32)
    mode_tile_prefix = as_mx_array(mode_tile_prefix, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    block_count = int(atom_order.shape[0]) // 32
    block_pair_count = block_count * (block_count - 1) // 2
    run_shape = (block_count * 3,)
    if mode_tile_counts.shape != run_shape or mode_tile_prefix.shape != run_shape:
        raise ValueError("mode tile counts and prefixes must have shape (3*n_blocks,)")
    if center_radius.shape != (block_count, 4):
        raise ValueError("center_radius must have shape (n_blocks, 4)")
    if half_extent.shape != (block_count, 3):
        raise ValueError("half_extent must have shape (n_blocks, 3)")
    if block_traversal.shape != (block_count,):
        raise ValueError("block_traversal must have shape (n_blocks,)")
    special_word_shape = ((block_count * block_count + 31) // 32,)
    if special_pair_words.shape != special_word_shape:
        raise ValueError("special_pair_words must cover the dense block-code space")
    if mode_words is not None and mode_words.shape != (2 * block_pair_count,):
        raise ValueError("mode_words must store two packed words per block pair")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    if accepted_tile_count < 0:
        raise ValueError("accepted_tile_count must be non-negative")
    if accepted_tile_count == 0:
        return (
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0, 32), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
        )
    if mode_words is not None:
        return _interaction32_ordinary_cached_scatter_kernel()(
            inputs=[
                block_traversal,
                mode_words,
                mode_tile_counts,
                mode_tile_prefix,
                mx.array([block_count], dtype=mx.int32),
            ],
            output_shapes=[
                (accepted_tile_count,),
                (accepted_tile_count, 32),
                (accepted_tile_count,),
            ],
            output_dtypes=[mx.int32, mx.int32, mx.int32],
            grid=(block_count * 32, 1, 1),
            threadgroup=(32, 1, 1),
            init_value=int(atom_order.shape[0]),
        )
    radius = float(search_radius)
    return _interaction32_ordinary_scatter_kernel()(
        inputs=[
            positions,
            atom_order,
            center_radius,
            half_extent,
            block_traversal,
            special_pair_words,
            mode_tile_counts,
            mode_tile_prefix,
            box,
            mx.array([radius, radius * radius], dtype=mx.float32),
            mx.array([block_count], dtype=mx.int32),
        ],
        output_shapes=[
            (accepted_tile_count,),
            (accepted_tile_count, 32),
            (accepted_tile_count,),
        ],
        output_dtypes=[mx.int32, mx.int32, mx.int32],
        grid=(block_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=int(atom_order.shape[0]),
    )


def _interaction32_outer_inner_mode_counts(
    positions: mx.array,
    atom_order: mx.array,
    block_traversal: mx.array,
    outer_right_atoms: mx.array,
    outer_tile_counts: mx.array,
    outer_tile_prefix: mx.array,
    box_lengths_and_inverses: mx.array,
    *,
    search_radius: float,
) -> tuple[mx.array, mx.array]:
    """Classify outer-schedule right entries at a smaller search radius."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    block_traversal = as_mx_array(block_traversal, dtype=mx.int32)
    outer_right_atoms = as_mx_array(outer_right_atoms, dtype=mx.int32)
    outer_tile_counts = as_mx_array(outer_tile_counts, dtype=mx.int32)
    outer_tile_prefix = as_mx_array(outer_tile_prefix, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    block_count = int(atom_order.shape[0]) // 32
    if block_traversal.shape != (block_count,):
        raise ValueError("block_traversal must have shape (n_blocks,)")
    if outer_right_atoms.ndim != 2 or outer_right_atoms.shape[1] != 32:
        raise ValueError("outer_right_atoms must have shape (n_tiles, 32)")
    run_shape = (3 * block_count,)
    if outer_tile_counts.shape != run_shape or outer_tile_prefix.shape != run_shape:
        raise ValueError("outer tile counts and prefixes must have shape (3*n_blocks,)")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    if not isfinite(float(search_radius)) or search_radius <= 0.0:
        raise ValueError("search_radius must be finite and positive")
    radius = float(search_radius)
    return _interaction32_outer_inner_mode_count_kernel()(
        inputs=[
            positions,
            atom_order,
            block_traversal,
            outer_right_atoms,
            outer_tile_counts,
            outer_tile_prefix,
            box,
            mx.array([radius * radius], dtype=mx.float32),
            mx.array([block_count, int(atom_order.shape[0])], dtype=mx.int32),
        ],
        output_shapes=[run_shape, outer_right_atoms.shape],
        output_dtypes=[mx.int32, mx.uint32],
        grid=(block_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=0,
    )


def _interaction32_outer_inner_mode_scatter_sized(
    atom_order: mx.array,
    block_traversal: mx.array,
    outer_right_atoms: mx.array,
    cached_modes: mx.array,
    outer_tile_counts: mx.array,
    outer_tile_prefix: mx.array,
    inner_tile_counts: mx.array,
    inner_tile_prefix: mx.array,
    *,
    accepted_tile_count: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Scatter cached inner modes into one retained outer-sized capacity."""

    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    block_traversal = as_mx_array(block_traversal, dtype=mx.int32)
    outer_right_atoms = as_mx_array(outer_right_atoms, dtype=mx.int32)
    cached_modes = as_mx_array(cached_modes, dtype=mx.uint32)
    outer_tile_counts = as_mx_array(outer_tile_counts, dtype=mx.int32)
    outer_tile_prefix = as_mx_array(outer_tile_prefix, dtype=mx.int32)
    inner_tile_counts = as_mx_array(inner_tile_counts, dtype=mx.int32)
    inner_tile_prefix = as_mx_array(inner_tile_prefix, dtype=mx.int32)
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    block_count = int(atom_order.shape[0]) // 32
    if block_traversal.shape != (block_count,):
        raise ValueError("block_traversal must have shape (n_blocks,)")
    if outer_right_atoms.ndim != 2 or outer_right_atoms.shape[1] != 32:
        raise ValueError("outer_right_atoms must have shape (n_tiles, 32)")
    if cached_modes.shape != outer_right_atoms.shape:
        raise ValueError("cached_modes must match outer_right_atoms")
    run_shape = (3 * block_count,)
    run_arrays = (
        outer_tile_counts,
        outer_tile_prefix,
        inner_tile_counts,
        inner_tile_prefix,
    )
    if any(array.shape != run_shape for array in run_arrays):
        raise ValueError("outer and inner tile metadata must have shape (3*n_blocks,)")
    if accepted_tile_count < 0:
        raise ValueError("accepted_tile_count must be non-negative")
    if accepted_tile_count == 0:
        return (
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0, 32), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
        )
    return _interaction32_outer_inner_mode_scatter_kernel()(
        inputs=[
            block_traversal,
            outer_right_atoms,
            cached_modes,
            outer_tile_counts,
            outer_tile_prefix,
            inner_tile_counts,
            inner_tile_prefix,
            mx.array([block_count], dtype=mx.int32),
        ],
        output_shapes=[
            (accepted_tile_count,),
            (accepted_tile_count, 32),
            (accepted_tile_count,),
        ],
        output_dtypes=[mx.int32, mx.int32, mx.int32],
        grid=(block_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=int(atom_order.shape[0]),
    )


def _interaction32_special_pair_words(
    special_codes: mx.array,
    special_unique: mx.array,
    *,
    block_count: int,
) -> mx.array:
    """Pack special block codes into a constant-time membership bitset."""

    special_codes = as_mx_array(special_codes, dtype=mx.int32)
    special_unique = as_mx_array(special_unique, dtype=mx.int32)
    if special_codes.ndim != 1 or special_unique.shape != special_codes.shape:
        raise ValueError("special codes and unique flags must be matching vectors")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    code_count = int(special_codes.shape[0])
    word_count = (block_count * block_count + 31) // 32
    if code_count == 0:
        return mx.zeros((word_count,), dtype=mx.uint32)
    (words,) = _interaction32_special_pair_words_kernel()(
        inputs=[
            special_codes,
            special_unique,
            mx.array([code_count], dtype=mx.int32),
        ],
        output_shapes=[(word_count,)],
        output_dtypes=[mx.uint32],
        grid=(code_count, 1, 1),
        threadgroup=(min(256, code_count), 1, 1),
        init_value=0,
    )
    return words


def _interaction32_special_blocks_sized(
    special_codes: mx.array,
    special_unique: mx.array,
    special_prefix: mx.array,
    *,
    block_count: int,
    special_count: int,
    block_capacity: int | None = None,
) -> mx.array:
    """Compact flagged 32-atom block pairs into a sized device array."""

    special_codes = as_mx_array(special_codes, dtype=mx.int32)
    special_unique = as_mx_array(special_unique, dtype=mx.int32)
    special_prefix = as_mx_array(special_prefix, dtype=mx.int32)
    raw_code_count = int(special_codes.shape[0])
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if special_codes.ndim != 1 or raw_code_count < block_count:
        raise ValueError("special_codes must contain sorted diagonal block codes")
    if special_unique.shape != (raw_code_count,):
        raise ValueError("special_unique must contain one marker per raw block code")
    if special_prefix.shape != (raw_code_count,):
        raise ValueError("special_prefix must contain one value per raw block code")
    if special_count < 0 or special_count > raw_code_count:
        raise ValueError("special_count is incompatible with the block inventory")
    if block_capacity is None:
        block_capacity = special_count
    if block_capacity < special_count:
        raise ValueError("special block capacity is below the logical inventory")
    if block_capacity == 0:
        return mx.zeros((0, 2), dtype=mx.int32)
    threads = min(256, raw_code_count)
    (special_blocks,) = _interaction32_special_block_scatter_kernel()(
        inputs=[
            special_codes,
            special_unique,
            special_prefix,
            mx.array([raw_code_count, block_count], dtype=mx.int32),
        ],
        output_shapes=[(block_capacity, 2)],
        output_dtypes=[mx.int32],
        grid=(raw_code_count, 1, 1),
        threadgroup=(threads, 1, 1),
        init_value=0,
    )
    return special_blocks


def _interaction32_special_work_two_halves(
    atom_order: mx.array,
    special_blocks: mx.array,
    topology_offsets: mx.array,
    topology_neighbors: mx.array,
    topology_classes: mx.array,
    *,
    work_capacity: int | None = None,
) -> tuple[mx.array, ...]:
    """Build conservative two-half work and topology masks for special blocks."""

    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    special_blocks = as_mx_array(special_blocks, dtype=mx.int32)
    topology_offsets = as_mx_array(topology_offsets, dtype=mx.int32)
    topology_neighbors = as_mx_array(topology_neighbors, dtype=mx.int32)
    topology_classes = as_mx_array(topology_classes, dtype=mx.int32)
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    if special_blocks.ndim != 2 or special_blocks.shape[1] != 2:
        raise ValueError("special_blocks must have shape (n_special, 2)")
    atom_count = int(topology_offsets.shape[0]) - 1
    if atom_count < 1:
        raise ValueError("topology_offsets must contain one boundary per atom")
    topology_count = int(topology_neighbors.shape[0])
    if topology_classes.shape != (topology_count,):
        raise ValueError("topology neighbors and classes must have matching shapes")
    special_count = int(special_blocks.shape[0])
    work_count = 2 * special_count
    if work_capacity is None:
        work_capacity = work_count
    if work_capacity < work_count:
        raise ValueError("special work capacity is below the logical inventory")
    if work_capacity == 0:
        return (
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((0, 32), dtype=mx.int32),
            mx.zeros((0, 32), dtype=mx.uint32),
            mx.zeros((0, 32), dtype=mx.uint32),
            mx.zeros((0,), dtype=mx.int32),
        )
    return _interaction32_special_work_kernel()(
        inputs=[
            atom_order,
            special_blocks,
            topology_offsets,
            topology_neighbors,
            topology_classes,
            mx.array([special_count, int(atom_order.shape[0])], dtype=mx.int32),
        ],
        output_shapes=[
            (work_capacity,),
            (work_capacity,),
            (work_capacity, 32),
            (work_capacity, 32),
            (work_capacity, 32),
            (work_capacity,),
        ],
        output_dtypes=[
            mx.int32,
            mx.int32,
            mx.int32,
            mx.uint32,
            mx.uint32,
            mx.int32,
        ],
        grid=(special_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=0,
    )


def _interaction32_pme_direct_force_only(
    positions: mx.array,
    atom_order: mx.array,
    ordinary_left_blocks: mx.array,
    ordinary_left_slices: mx.array,
    ordinary_right_atoms: mx.array,
    ordinary_group_starts: mx.array,
    ordinary_group_counts: mx.array,
    special_left_blocks: mx.array,
    special_left_slices: mx.array,
    special_right_atoms: mx.array,
    special_work_lj_enabled: mx.array,
    special_work_lj_one_four: mx.array,
    special_diagonal: mx.array,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    _return_stages: bool = False,
    _canonical_records: bool = True,
    _simdgroups_per_threadgroup: int = _INTERACTION32_GROUPS_PER_THREADGROUP,
    _left_slice_size: int = 16,
) -> mx.array | _Interaction32ForceStages:
    """Evaluate the experimental SIMD-native 32-atom direct-force schedule."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    ordinary_left_blocks = as_mx_array(ordinary_left_blocks, dtype=mx.int32)
    ordinary_left_slices = as_mx_array(ordinary_left_slices, dtype=mx.int32)
    ordinary_right_atoms = as_mx_array(ordinary_right_atoms, dtype=mx.int32)
    ordinary_group_starts = as_mx_array(ordinary_group_starts, dtype=mx.int32)
    ordinary_group_counts = as_mx_array(ordinary_group_counts, dtype=mx.int32)
    special_left_blocks = as_mx_array(special_left_blocks, dtype=mx.int32)
    special_left_slices = as_mx_array(special_left_slices, dtype=mx.int32)
    special_right_atoms = as_mx_array(special_right_atoms, dtype=mx.int32)
    special_work_lj_enabled = as_mx_array(special_work_lj_enabled, dtype=mx.uint32)
    special_work_lj_one_four = as_mx_array(special_work_lj_one_four, dtype=mx.uint32)
    special_diagonal = as_mx_array(special_diagonal, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions.shape[0])
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    padded_count = int(atom_order.shape[0])
    ordinary_count = int(ordinary_left_blocks.shape[0])
    if ordinary_left_blocks.ndim != 1 or ordinary_right_atoms.shape != (
        ordinary_count,
        32,
    ):
        raise ValueError("ordinary interaction arrays must have shapes (n,) and (n, 32)")
    if ordinary_left_slices.shape != (ordinary_count,):
        raise ValueError("ordinary_left_slices must have one entry per ordinary tile")
    ordinary_group_count = int(ordinary_group_starts.shape[0])
    if ordinary_group_starts.ndim != 1 or ordinary_group_counts.shape != (ordinary_group_count,):
        raise ValueError("ordinary interaction groups must have matching vector shapes")
    special_count = int(special_left_blocks.shape[0])
    if special_left_slices.shape != (special_count,) or special_diagonal.shape != (special_count,):
        raise ValueError("special work metadata must have matching vector shapes")
    special_shape = (special_count, 32)
    if (
        special_right_atoms.shape != special_shape
        or special_work_lj_enabled.shape != special_shape
        or special_work_lj_one_four.shape != special_shape
    ):
        raise ValueError("special work arrays must have shape (n_special_work, 32)")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    parameter_shape = (atom_count,)
    if (
        half_sigma.shape != parameter_shape
        or sqrt_epsilon.shape != parameter_shape
        or charges.shape != parameter_shape
    ):
        raise ValueError("prepared nonbonded parameters must match the atom count")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if not isfinite(float(alpha)) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not isfinite(float(coulomb_constant)):
        raise ValueError("coulomb_constant must be finite")
    if not isfinite(float(one_four_scale)) or one_four_scale < 0.0:
        raise ValueError("one_four_scale must be finite and non-negative")
    if not 1 <= _simdgroups_per_threadgroup <= _INTERACTION32_GROUPS_PER_THREADGROUP:
        raise ValueError("SIMD groups per threadgroup must be between one and four")
    if _left_slice_size not in (4, 8, 16):
        raise ValueError("left slice size must be 4, 8, or 16")
    cutoff_value = float(cutoff)
    if switch_distance is not None and (
        not isfinite(float(switch_distance))
        or float(switch_distance) < 0.0
        or float(switch_distance) >= cutoff_value
    ):
        raise ValueError("switch_distance must be finite, non-negative, and below cutoff")

    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - switch_value
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(switch_distance is not None),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
            1.0 / switch_width,
            float(one_four_scale),
        ],
        dtype=mx.float32,
    )
    pack_counts = mx.array([padded_count, atom_count], dtype=mx.int32)
    counts = mx.array(
        [
            ordinary_group_count,
            special_count,
            padded_count,
            atom_count,
            _simdgroups_per_threadgroup,
            _left_slice_size,
        ],
        dtype=mx.int32,
    )
    schedule_inputs = [
        ordinary_left_blocks,
        ordinary_left_slices,
        ordinary_right_atoms,
        ordinary_group_starts,
        ordinary_group_counts,
        special_left_blocks,
        special_left_slices,
        special_right_atoms,
        special_work_lj_enabled,
        special_work_lj_one_four,
        special_diagonal,
        box,
        params,
        counts,
    ]
    if _canonical_records:
        packed_posq = positions
        packed_lj = half_sigma
        force_inputs = [
            positions,
            atom_order,
            half_sigma,
            sqrt_epsilon,
            charges,
            *schedule_inputs,
        ]
        force_kernel = _interaction32_canonical_force_kernel()
        force_shape = positions.shape
    else:
        packed_posq, packed_lj = _interaction32_pack_kernel()(
            inputs=[
                positions,
                atom_order,
                half_sigma,
                sqrt_epsilon,
                charges,
                pack_counts,
            ],
            output_shapes=[(padded_count, 4), (padded_count, 2)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(padded_count, 1, 1),
            threadgroup=(min(256, padded_count), 1, 1),
        )
        force_inputs = [packed_posq, packed_lj, atom_order, *schedule_inputs]
        force_kernel = _interaction32_force_kernel()
        force_shape = (padded_count, 3)

    def dispatch() -> mx.array:
        work_count = ordinary_group_count + special_count
        if work_count == 0:
            return mx.zeros(force_shape, dtype=mx.float32)
        threadgroups = (work_count + _simdgroups_per_threadgroup - 1) // _simdgroups_per_threadgroup
        dispatch_width = 32 * _simdgroups_per_threadgroup
        (ordered_forces,) = force_kernel(
            inputs=force_inputs,
            output_shapes=[force_shape],
            output_dtypes=[mx.float32],
            grid=(threadgroups * dispatch_width, 1, 1),
            threadgroup=(dispatch_width, 1, 1),
            init_value=0.0,
        )
        return ordered_forces

    ordered_forces = dispatch()
    if _canonical_records:
        forces = ordered_forces
    else:
        (forces,) = _interaction32_scatter_kernel()(
            inputs=[ordered_forces, atom_order, pack_counts],
            output_shapes=[positions.shape],
            output_dtypes=[mx.float32],
            grid=(padded_count, 1, 1),
            threadgroup=(min(256, padded_count), 1, 1),
        )
    if _return_stages:
        return _Interaction32ForceStages(
            packed_posq=packed_posq,
            packed_lj=packed_lj,
            ordered_forces=ordered_forces,
            forces=forces,
        )
    return forces


def _interaction32_fused_half_pme_direct_force_only(
    positions: mx.array,
    atom_order: mx.array,
    ordinary_left_blocks: mx.array,
    ordinary_right_atoms: mx.array,
    ordinary_half_modes: mx.array,
    ordinary_group_starts: mx.array,
    ordinary_group_counts: mx.array,
    special_left_blocks: mx.array,
    special_left_slices: mx.array,
    special_right_atoms: mx.array,
    special_work_lj_enabled: mx.array,
    special_work_lj_one_four: mx.array,
    special_diagonal: mx.array,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    atom_type_ids: mx.array | None = None,
    nbfix_type_sigma: mx.array | None = None,
    nbfix_type_epsilon: mx.array | None = None,
    nbfix_type_count: int = 0,
    _simdgroups_per_threadgroup: int = _INTERACTION32_GROUPS_PER_THREADGROUP,
) -> mx.array:
    """Evaluate fused 16-atom memberships with one right-force write."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    ordinary_left_blocks = as_mx_array(ordinary_left_blocks, dtype=mx.int32)
    ordinary_right_atoms = as_mx_array(ordinary_right_atoms, dtype=mx.int32)
    ordinary_half_modes = as_mx_array(ordinary_half_modes, dtype=mx.int32)
    ordinary_group_starts = as_mx_array(ordinary_group_starts, dtype=mx.int32)
    ordinary_group_counts = as_mx_array(ordinary_group_counts, dtype=mx.int32)
    special_left_blocks = as_mx_array(special_left_blocks, dtype=mx.int32)
    special_left_slices = as_mx_array(special_left_slices, dtype=mx.int32)
    special_right_atoms = as_mx_array(special_right_atoms, dtype=mx.int32)
    special_work_lj_enabled = as_mx_array(special_work_lj_enabled, dtype=mx.uint32)
    special_work_lj_one_four = as_mx_array(special_work_lj_one_four, dtype=mx.uint32)
    special_diagonal = as_mx_array(special_diagonal, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    nbfix_inputs = (atom_type_ids, nbfix_type_sigma, nbfix_type_epsilon)
    has_nbfix = any(value is not None for value in nbfix_inputs)
    if has_nbfix and not all(value is not None for value in nbfix_inputs):
        raise ValueError("NBFIX interaction inputs must be provided together")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions.shape[0])
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    padded_count = int(atom_order.shape[0])
    ordinary_count = int(ordinary_left_blocks.shape[0])
    ordinary_shape = (ordinary_count, 32)
    if (
        ordinary_left_blocks.ndim != 1
        or ordinary_right_atoms.shape != ordinary_shape
        or ordinary_half_modes.shape != (ordinary_count,)
    ):
        raise ValueError("fused ordinary arrays must have shapes (n,) and (n, 32)")
    ordinary_group_count = int(ordinary_group_starts.shape[0])
    if ordinary_group_starts.ndim != 1 or ordinary_group_counts.shape != (
        ordinary_group_count,
    ):
        raise ValueError("ordinary interaction groups must have matching vector shapes")
    special_count = int(special_left_blocks.shape[0])
    if special_left_slices.shape != (special_count,) or special_diagonal.shape != (
        special_count,
    ):
        raise ValueError("special work metadata must have matching vector shapes")
    special_shape = (special_count, 32)
    if (
        special_right_atoms.shape != special_shape
        or special_work_lj_enabled.shape != special_shape
        or special_work_lj_one_four.shape != special_shape
    ):
        raise ValueError("special work arrays must have shape (n_special_work, 32)")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    parameter_shape = (atom_count,)
    if (
        half_sigma.shape != parameter_shape
        or sqrt_epsilon.shape != parameter_shape
        or charges.shape != parameter_shape
    ):
        raise ValueError("prepared nonbonded parameters must match the atom count")
    if has_nbfix:
        atom_type_ids = as_mx_array(atom_type_ids, dtype=mx.int32)
        nbfix_type_sigma = as_mx_array(nbfix_type_sigma, dtype=mx.float32)
        nbfix_type_epsilon = as_mx_array(nbfix_type_epsilon, dtype=mx.float32)
        if atom_type_ids.shape != parameter_shape:
            raise ValueError("atom_type_ids must match the atom count")
        if nbfix_type_count <= 0:
            raise ValueError("nbfix_type_count must be positive with NBFIX inputs")
        table_shape = (nbfix_type_count * nbfix_type_count,)
        if (
            nbfix_type_sigma.shape != table_shape
            or nbfix_type_epsilon.shape != table_shape
        ):
            raise ValueError("NBFIX tables must have shape (nbfix_type_count ** 2,)")
    elif nbfix_type_count != 0:
        raise ValueError("nbfix_type_count requires NBFIX interaction inputs")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if not isfinite(float(alpha)) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not isfinite(float(coulomb_constant)):
        raise ValueError("coulomb_constant must be finite")
    if not isfinite(float(one_four_scale)) or one_four_scale < 0.0:
        raise ValueError("one_four_scale must be finite and non-negative")
    if not 1 <= _simdgroups_per_threadgroup <= _INTERACTION32_GROUPS_PER_THREADGROUP:
        raise ValueError("SIMD groups per threadgroup must be between one and four")
    cutoff_value = float(cutoff)
    if switch_distance is not None and (
        not isfinite(float(switch_distance))
        or float(switch_distance) < 0.0
        or float(switch_distance) >= cutoff_value
    ):
        raise ValueError("switch_distance must be finite, non-negative, and below cutoff")

    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - switch_value
    alpha_value = float(alpha)
    parameter_values = [
        cutoff_value * cutoff_value,
        float(bool(shift)),
        float(switch_distance is not None),
        switch_value,
        switch_width,
        cutoff_value,
        float(coulomb_constant),
        alpha_value,
        2.0 * alpha_value / sqrt(pi),
        1.0 / switch_width,
        float(one_four_scale),
    ]
    if has_nbfix:
        parameter_values.append(float(nbfix_type_count))
    params = mx.array(parameter_values, dtype=mx.float32)
    counts = mx.array(
        [
            ordinary_group_count,
            special_count,
            padded_count,
            atom_count,
            _simdgroups_per_threadgroup,
            16,
        ],
        dtype=mx.int32,
    )
    work_count = ordinary_group_count + special_count
    if work_count == 0:
        return mx.zeros(positions.shape, dtype=mx.float32)
    threadgroups = (
        work_count + _simdgroups_per_threadgroup - 1
    ) // _simdgroups_per_threadgroup
    dispatch_width = 32 * _simdgroups_per_threadgroup
    inputs = [
        positions,
        atom_order,
        half_sigma,
        sqrt_epsilon,
        charges,
    ]
    if has_nbfix:
        inputs.extend((atom_type_ids, nbfix_type_sigma, nbfix_type_epsilon))
    inputs.extend(
        (
            ordinary_left_blocks,
            ordinary_right_atoms,
            ordinary_half_modes,
            ordinary_group_starts,
            ordinary_group_counts,
            special_left_blocks,
            special_left_slices,
            special_right_atoms,
            special_work_lj_enabled,
            special_work_lj_one_four,
            special_diagonal,
            box,
            params,
            counts,
        )
    )
    kernel = (
        _interaction32_fused_half_nbfix_canonical_force_kernel()
        if has_nbfix
        else _interaction32_fused_half_canonical_force_kernel()
    )
    (forces,) = kernel(
        inputs=inputs,
        output_shapes=[positions.shape],
        output_dtypes=[mx.float32],
        grid=(threadgroups * dispatch_width, 1, 1),
        threadgroup=(dispatch_width, 1, 1),
        init_value=0.0,
    )
    return forces


def _owner_compute32_pme_direct_force_only(
    positions: mx.array,
    atom_order: mx.array,
    owner_offsets: mx.array,
    right_atoms: mx.array,
    topology_offsets: mx.array,
    topology_neighbors: mx.array,
    topology_classes: mx.array,
    box_lengths_and_inverses: mx.array,
    half_sigma: mx.array,
    sqrt_epsilon: mx.array,
    charges: mx.array,
    *,
    cutoff: float,
    shift: bool,
    switch_distance: float | None,
    one_four_scale: float,
    coulomb_constant: float,
    alpha: float,
    _simdgroups_per_threadgroup: int = 4,
) -> mx.array:
    """Evaluate the no-atomic owner-computes direct-force schedule."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_order = as_mx_array(atom_order, dtype=mx.int32)
    owner_offsets = as_mx_array(owner_offsets, dtype=mx.int32)
    right_atoms = as_mx_array(right_atoms, dtype=mx.int32)
    topology_offsets = as_mx_array(topology_offsets, dtype=mx.int32)
    topology_neighbors = as_mx_array(topology_neighbors, dtype=mx.int32)
    topology_classes = as_mx_array(topology_classes, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    atom_count = int(positions.shape[0])
    if atom_order.ndim != 1 or atom_order.shape[0] % 32 != 0:
        raise ValueError("atom_order must be a padded vector divisible by 32")
    padded_count = int(atom_order.shape[0])
    block_count = padded_count // 32
    if owner_offsets.shape != (block_count + 1,):
        raise ValueError("owner_offsets must have one boundary per owner block")
    if right_atoms.ndim != 1 or right_atoms.shape[0] % 32 != 0:
        raise ValueError("right_atoms must be a padded vector divisible by 32")
    if topology_offsets.shape != (atom_count + 1,):
        raise ValueError("topology_offsets must have one boundary per atom")
    topology_count = int(topology_neighbors.shape[0])
    if topology_neighbors.ndim != 1 or topology_classes.shape != (topology_count,):
        raise ValueError("topology neighbors and classes must have matching vector shapes")
    if box.shape != (6,):
        raise ValueError("box_lengths_and_inverses must have shape (6,)")
    parameter_shape = (atom_count,)
    if (
        half_sigma.shape != parameter_shape
        or sqrt_epsilon.shape != parameter_shape
        or charges.shape != parameter_shape
    ):
        raise ValueError("prepared nonbonded parameters must match the atom count")
    if not isfinite(float(cutoff)) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if not isfinite(float(alpha)) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not isfinite(float(coulomb_constant)):
        raise ValueError("coulomb_constant must be finite")
    if not isfinite(float(one_four_scale)) or one_four_scale < 0.0:
        raise ValueError("one_four_scale must be finite and non-negative")
    if not 1 <= _simdgroups_per_threadgroup <= 4:
        raise ValueError("SIMD groups per threadgroup must be between one and four")
    cutoff_value = float(cutoff)
    if switch_distance is not None and (
        not isfinite(float(switch_distance))
        or float(switch_distance) < 0.0
        or float(switch_distance) >= cutoff_value
    ):
        raise ValueError("switch_distance must be finite, non-negative, and below cutoff")

    switch_value = 0.0 if switch_distance is None else float(switch_distance)
    switch_width = 1.0 if switch_distance is None else cutoff_value - switch_value
    alpha_value = float(alpha)
    params = mx.array(
        [
            cutoff_value * cutoff_value,
            float(bool(shift)),
            float(switch_distance is not None),
            switch_value,
            switch_width,
            cutoff_value,
            float(coulomb_constant),
            alpha_value,
            2.0 * alpha_value / sqrt(pi),
            1.0 / switch_width,
            float(one_four_scale),
        ],
        dtype=mx.float32,
    )
    counts = mx.array(
        [block_count, padded_count, atom_count, _simdgroups_per_threadgroup],
        dtype=mx.int32,
    )
    threadgroups = (
        block_count + _simdgroups_per_threadgroup - 1
    ) // _simdgroups_per_threadgroup
    dispatch_width = 32 * _simdgroups_per_threadgroup
    (forces,) = _owner_compute32_force_kernel()(
        inputs=[
            positions,
            atom_order,
            owner_offsets,
            right_atoms,
            topology_offsets,
            topology_neighbors,
            topology_classes,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            params,
            counts,
        ],
        output_shapes=[positions.shape],
        output_dtypes=[mx.float32],
        grid=(threadgroups * dispatch_width, 1, 1),
        threadgroup=(dispatch_width, 1, 1),
    )
    return forces


def fused_sparse_pme_correction_forces(
    positions: mx.array,
    pairs: mx.array,
    box_lengths_and_inverses: mx.array,
    charge_products: mx.array,
    lj_sigma: mx.array,
    lj_epsilon: mx.array,
    *,
    coulomb_constant: float,
) -> mx.array:
    """Evaluate sparse PME exclusions, exceptions, and 1-4 force corrections."""

    positions = as_mx_array(positions, dtype=mx.float32)
    pairs = as_mx_array(pairs, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    charge_products = as_mx_array(charge_products, dtype=mx.float32)
    lj_sigma = as_mx_array(lj_sigma, dtype=mx.float32)
    lj_epsilon = as_mx_array(lj_epsilon, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        msg = "pairs must have shape (n_pairs, 2)"
        raise ValueError(msg)
    pair_count = int(pairs.shape[0])
    for name, values in (
        ("charge_products", charge_products),
        ("lj_sigma", lj_sigma),
        ("lj_epsilon", lj_epsilon),
    ):
        if values.shape != (pair_count,):
            msg = f"{name} must have shape (n_pairs,)"
            raise ValueError(msg)
    if box.shape != (6,):
        msg = "box_lengths_and_inverses must have shape (6,)"
        raise ValueError(msg)
    if not isfinite(float(coulomb_constant)):
        msg = "coulomb_constant must be finite"
        raise ValueError(msg)
    if pair_count == 0:
        return mx.zeros_like(positions)
    (forces,) = _sparse_pme_correction_force_only_kernel()(
        inputs=[
            positions,
            pairs[:, 0],
            pairs[:, 1],
            box,
            charge_products,
            lj_sigma,
            lj_epsilon,
            mx.array([float(coulomb_constant)], dtype=mx.float32),
            mx.array([pair_count], dtype=mx.int32),
        ],
        output_shapes=[positions.shape],
        output_dtypes=[mx.float32],
        grid=(pair_count, 1, 1),
        threadgroup=(min(256, pair_count), 1, 1),
        init_value=0.0,
    )
    return forces
