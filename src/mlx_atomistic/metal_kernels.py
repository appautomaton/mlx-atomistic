"""Fused Metal kernels for recurring molecular force paths.

Collapses the per-step pairwise LJ force op-chain (gather -> minimum image -> r^2 ->
LJ scalar -> scatter-add) into a single ``mx.fast.metal_kernel`` dispatch. Diagnostic
kernels write per-pair energy without contention, while force-only kernels omit those
outputs and reductions entirely.

The simple kernel covers scalar reduced-unit LJ. The parameterized kernel covers
per-atom Lorentz-Berthelot parameters, topology scales, shifts, and smooth switching
for the production biomolecular path. A separate force-only kernel combines standard
bond, angle, periodic-torsion, and improper interactions into one output. Unsupported
cases fall back transparently.

Because ``tests/conftest.py`` forces ``MLX_ATOMISTIC_DEVICE=cpu``, the kernel is built
lazily on first use (not at import) so importing this module never triggers a Metal
device load.
"""

from __future__ import annotations

from math import isfinite, pi, sqrt

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
    uint total_count = bond_count + angle_count + dihedral_count + improper_count;
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
    }
"""

_kernel_singleton = None
_parameterized_kernel_singleton = None
_pme_direct_kernel_singleton = None
_pme_direct_virial_kernel_singleton = None
_pme_direct_force_only_kernel_singleton = None
_prepared_pme_direct_force_only_kernel_singleton = None
_tile_pme_direct_force_only_kernel_singleton = None
_sparse_pme_correction_force_only_kernel_singleton = None
_pme_cutoff_correction_virial_kernel_singleton = None
_pme_order5_spread_kernel_singleton = None
_pme_order5_interpolate_kernel_singleton = None
_pme_order5_force_only_kernel_singleton = None
_aligned_topology_lj_scales_kernel_singleton = None
_neighbor_cell_pair_candidates_kernel_singleton = None
_neighbor_pair_cutoff_mask_kernel_singleton = None
_neighbor_pair_ordered_scatter_kernel_singleton = None
_neighbor_cell_tile_candidates_kernel_singleton = None
_neighbor_tile_membership_kernel_singleton = None
_neighbor_tile_ordered_scatter_kernel_singleton = None
_neighbor_tile_force_groups_kernel_singleton = None
_neighbor_tile_pair_scatter_kernel_singleton = None
_tile_topology_lj_masks_kernel_singleton = None
_shake_cluster_position_kernel_singleton = None
_shake_cluster_velocity_kernel_singleton = None
_settle_water_position_kernel_singleton = None
_settle_water_velocity_kernel_singleton = None
_fused_bonded_force_only_kernel_singleton = None

# Neighbor compaction preserves short runs of a common left atom, so one worker
# can sum those contributions locally before issuing global atomics.
_PREPARED_PME_PAIRS_PER_WORKER = 8

_TILE_PME_BLOCK_SIZE = 8
_TILE_PME_LANES_PER_TILE = _TILE_PME_BLOCK_SIZE * _TILE_PME_BLOCK_SIZE
_TILE_PME_THREADGROUP_SIZE = _TILE_PME_LANES_PER_TILE
_TILE_MEMBERSHIP_THREADGROUP_SIZE = _TILE_PME_LANES_PER_TILE
_TILE_PME_THREADGROUP_TEMPORARY_BYTES = (
    2 * _TILE_PME_BLOCK_SIZE * 4
    + 2 * _TILE_PME_BLOCK_SIZE * (3 + 1 + 1 + 1) * 4
    + 3 * _TILE_PME_LANES_PER_TILE * 4
)

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

# One 64-thread group walks a short contiguous run of tiles with a common left
# block. It loads that block once and accumulates its force across the run, while
# each right block is still reduced locally. This preserves exact 8x8 geometry
# but removes repeated global writes and parameter gathers for the shared side.
_TILE_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    int group_start = force_group_starts[group];
    int group_count = force_group_counts[group];

    threadgroup int left_atoms[8];
    threadgroup int right_atoms[8];
    threadgroup float left_positions[24];
    threadgroup float right_positions[24];
    threadgroup float left_half_sigma[8];
    threadgroup float right_half_sigma[8];
    threadgroup float left_sqrt_epsilon[8];
    threadgroup float right_sqrt_epsilon[8];
    threadgroup float left_charges[8];
    threadgroup float right_charges[8];
    threadgroup float pair_fx[64];
    threadgroup float pair_fy[64];
    threadgroup float pair_fz[64];

    int left_block = tile_blocks[2 * group_start + 0];
    if (lane < 8u) {
        int left_atom = atom_blocks[8 * left_block + lane];
        left_atoms[lane] = left_atom;
        int safe_left = max(left_atom, 0);
        left_positions[3 * lane + 0] = positions[3 * safe_left + 0];
        left_positions[3 * lane + 1] = positions[3 * safe_left + 1];
        left_positions[3 * lane + 2] = positions[3 * safe_left + 2];
        left_half_sigma[lane] = half_sigma[safe_left];
        left_sqrt_epsilon[lane] = sqrt_epsilon[safe_left];
        left_charges[lane] = charges[safe_left];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulated_left_fx = 0.0f;
    float accumulated_left_fy = 0.0f;
    float accumulated_left_fz = 0.0f;
    uint left_slot = lane >> 3;
    uint right_slot = lane & 7u;
    uint word = lane >> 5;
    uint bit = lane & 31u;

    for (int local_tile = 0; local_tile < group_count; local_tile++) {
        int tile = group_start + local_tile;
        int tile_left_block = tile_blocks[2 * tile + 0];
        int right_block = tile_blocks[2 * tile + 1];
        bool same_block = tile_left_block == right_block;
        if (lane < 8u) {
            int right_atom = atom_blocks[8 * right_block + lane];
            right_atoms[lane] = right_atom;
            int safe_right = max(right_atom, 0);
            right_positions[3 * lane + 0] = positions[3 * safe_right + 0];
            right_positions[3 * lane + 1] = positions[3 * safe_right + 1];
            right_positions[3 * lane + 2] = positions[3 * safe_right + 2];
            right_half_sigma[lane] = half_sigma[safe_right];
            right_sqrt_epsilon[lane] = sqrt_epsilon[safe_right];
            right_charges[lane] = charges[safe_right];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool member = ((member_mask[2 * tile + word] >> bit) & 1u) != 0u;
        int i = left_atoms[left_slot];
        int j = right_atoms[right_slot];
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;
        if (member && i >= 0 && j >= 0) {
            float dx = left_positions[3 * left_slot + 0]
                - right_positions[3 * right_slot + 0];
            float dy = left_positions[3 * left_slot + 1]
                - right_positions[3 * right_slot + 1];
            float dz = left_positions[3 * left_slot + 2]
                - right_positions[3 * right_slot + 2];
            float lx = box[0];
            float ly = box[1];
            float lz = box[2];
            dx -= lx * rint(dx * box[3]);
            dy -= ly * rint(dy * box[4]);
            dz -= lz * rint(dz * box[5]);
            float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 0.0f && r2 < params[0]) {
                float inv_distance = rsqrt(r2);
                float inv_r2 = inv_distance * inv_distance;
                float distance = r2 * inv_distance;
                float scalar = 0.0f;
                bool lj_enabled =
                    ((lj_enabled_mask[2 * tile + word] >> bit) & 1u) != 0u;
                if (lj_enabled) {
                    bool one_four =
                        ((lj_one_four_mask[2 * tile + word] >> bit) & 1u) != 0u;
                    float lj_scale = one_four ? params[10] : 1.0f;
                    float sigma_ij = left_half_sigma[left_slot]
                        + right_half_sigma[right_slot];
                    float epsilon_ij = left_sqrt_epsilon[left_slot]
                        * right_sqrt_epsilon[right_slot];
                    float sigma2_over_r2 = sigma_ij * sigma_ij * inv_r2;
                    float inv_r6 =
                        sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
                    float inv_r12 = inv_r6 * inv_r6;
                    float unswitched_energy =
                        4.0f * epsilon_ij * (inv_r12 - inv_r6);
                    if (params[1] > 0.5f) {
                        float sigma2_over_rc2 = sigma_ij * sigma_ij / params[0];
                        float inv_rc6 = sigma2_over_rc2
                            * sigma2_over_rc2 * sigma2_over_rc2;
                        unswitched_energy -= 4.0f * epsilon_ij
                            * (inv_rc6 * inv_rc6 - inv_rc6);
                    }

                    float switch_value = 1.0f;
                    float switch_derivative = 0.0f;
                    if (params[2] > 0.5f) {
                        float x = clamp(
                            (distance - params[3]) * params[9],
                            0.0f,
                            1.0f
                        );
                        float x2 = x * x;
                        float x3 = x2 * x;
                        float x4 = x3 * x;
                        float x5 = x4 * x;
                        switch_value = 1.0f
                            - (10.0f * x3 - 15.0f * x4 + 6.0f * x5);
                        if (distance > params[3] && distance < params[5]) {
                            switch_derivative = -(
                                30.0f * x2 - 60.0f * x3 + 30.0f * x4
                            ) * params[9];
                        }
                    }
                    scalar += (
                        24.0f * epsilon_ij * (2.0f * inv_r12 - inv_r6)
                        * inv_r2 * switch_value
                        - unswitched_energy * switch_derivative * inv_distance
                    ) * lj_scale;
                }

                float qij = left_charges[left_slot] * right_charges[right_slot];
                float erfc_term =
                    1.0f - mlx_atomistic_erf(params[7] * distance);
                scalar += params[6] * qij * (
                    erfc_term * inv_r2 * inv_distance
                    + params[8] * exp(-params[7] * params[7] * r2) * inv_r2
                );
                fx = scalar * dx;
                fy = scalar * dy;
                fz = scalar * dz;
            }
        }
        pair_fx[lane] = fx;
        pair_fy[lane] = fy;
        pair_fz[lane] = fz;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8u) {
            float reduced_fx = 0.0f;
            float reduced_fy = 0.0f;
            float reduced_fz = 0.0f;
            for (uint other = 0u; other < 8u; other++) {
                uint pair_lane = 8u * lane + other;
                reduced_fx += pair_fx[pair_lane];
                reduced_fy += pair_fy[pair_lane];
                reduced_fz += pair_fz[pair_lane];
            }
            if (same_block) {
                for (uint other = 0u; other < 8u; other++) {
                    uint pair_lane = 8u * other + lane;
                    reduced_fx -= pair_fx[pair_lane];
                    reduced_fy -= pair_fy[pair_lane];
                    reduced_fz -= pair_fz[pair_lane];
                }
            }
            accumulated_left_fx += reduced_fx;
            accumulated_left_fy += reduced_fy;
            accumulated_left_fz += reduced_fz;
        } else if (lane < 16u && !same_block) {
            uint slot = lane - 8u;
            float reduced_fx = 0.0f;
            float reduced_fy = 0.0f;
            float reduced_fz = 0.0f;
            for (uint other = 0u; other < 8u; other++) {
                uint pair_lane = 8u * other + slot;
                reduced_fx -= pair_fx[pair_lane];
                reduced_fy -= pair_fy[pair_lane];
                reduced_fz -= pair_fz[pair_lane];
            }
            int atom = right_atoms[slot];
            if (
                atom >= 0
                && (reduced_fx != 0.0f || reduced_fy != 0.0f || reduced_fz != 0.0f)
            ) {
                atomic_fetch_add_explicit(
                    &forces[3 * atom + 0], reduced_fx, memory_order_relaxed
                );
                atomic_fetch_add_explicit(
                    &forces[3 * atom + 1], reduced_fy, memory_order_relaxed
                );
                atomic_fetch_add_explicit(
                    &forces[3 * atom + 2], reduced_fz, memory_order_relaxed
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < 8u) {
        int atom = left_atoms[lane];
        if (
            atom >= 0
            && (
                accumulated_left_fx != 0.0f
                || accumulated_left_fy != 0.0f
                || accumulated_left_fz != 0.0f
            )
        ) {
            atomic_fetch_add_explicit(
                &forces[3 * atom + 0], accumulated_left_fx, memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &forces[3 * atom + 1], accumulated_left_fy, memory_order_relaxed
            );
            atomic_fetch_add_explicit(
                &forces[3 * atom + 2], accumulated_left_fz, memory_order_relaxed
            );
        }
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
                float grid_value = potential_grid[(x * ny + y) * nz + z];
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
#ifdef MLX_ATOMISTIC_PME_WRITE_ENERGY
    atom_energy[atom] = 0.5f * charge * potential;
#endif
    forces[3 * atom + 0] =
        -charge * gradient.x * (float)mesh[0] / cell[0];
    forces[3 * atom + 1] =
        -charge * gradient.y * (float)mesh[1] / cell[1];
    forces[3 * atom + 2] =
        -charge * gradient.z * (float)mesh[2] / cell[2];
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
    uint left_slot = lane >> 3;
    uint right_slot = lane & 7u;
    threadgroup uint active[64];

    int left_block = tile_blocks[2 * tile + 0];
    int right_block = tile_blocks[2 * tile + 1];
    int atom_i = atom_blocks[8 * left_block + left_slot];
    int atom_j = atom_blocks[8 * right_block + right_slot];
    bool valid = atom_i >= 0 && atom_j >= 0;
    if (left_block == right_block) {
        valid = valid && left_slot < right_slot;
    }
    bool close = false;
    if (valid) {
        float dx = positions[3 * atom_i + 0] - positions[3 * atom_j + 0];
        float dy = positions[3 * atom_i + 1] - positions[3 * atom_j + 1];
        float dz = positions[3 * atom_i + 2] - positions[3 * atom_j + 2];
        dx -= box[0] * rint(dx / box[0]);
        dy -= box[1] * rint(dy / box[1]);
        dz -= box[2] * rint(dz / box[2]);
        close = dx * dx + dy * dy + dz * dz < params[0];
    }
    active[lane] = close ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < 2u) {
        uint word = 0u;
        uint base = 32u * lane;
        for (uint bit = 0u; bit < 32u; bit++) {
            word |= active[base + bit] << bit;
        }
        member_mask[2 * tile + lane] = word;
    }
    if (lane == 0u) {
        int total = 0;
        for (uint index = 0u; index < 64u; index++) {
            total += (int)active[index];
        }
        member_counts[tile] = total;
    }
"""

_NEIGHBOR_TILE_ORDERED_SCATTER_SOURCE = r"""
    uint tile = thread_position_in_grid.x;
    if (tile >= (uint)counts[0] || member_counts[tile] == 0) {
        return;
    }
    uint output = (uint)(prefix[tile] - 1);
    accepted_tile_blocks[2 * output + 0] = tile_blocks[2 * tile + 0];
    accepted_tile_blocks[2 * output + 1] = tile_blocks[2 * tile + 1];
    accepted_member_mask[2 * output + 0] = member_mask[2 * tile + 0];
    accepted_member_mask[2 * output + 1] = member_mask[2 * tile + 1];
"""

_NEIGHBOR_TILE_FORCE_GROUPS_SOURCE = r"""
    uint block = thread_position_in_grid.x;
    if (block >= (uint)counts[0] || tile_counts[block] == 0) {
        return;
    }

    int tile_count = tile_counts[block];
    int tile_start = tile_prefix[block] - tile_count;
    int group_count = group_counts[block];
    int group_start = group_prefix[block] - group_count;
    int tiles_per_group = counts[1];
    for (int local = 0; local < group_count; local++) {
        int consumed = local * tiles_per_group;
        int output = group_start + local;
        force_group_starts[output] = tile_start + consumed;
        force_group_counts[output] = min(tiles_per_group, tile_count - consumed);
    }
"""

_NEIGHBOR_TILE_PAIR_SCATTER_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint word_index = lane >> 5;
    uint bit_index = lane & 31u;
    threadgroup uint scan[64];

    uint word = member_mask[2 * tile + word_index];
    uint active = (word >> bit_index) & 1u;
    scan[lane] = active;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < 64u; offset <<= 1u) {
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
    int atom_i = atom_blocks[8 * left_block + (lane >> 3)];
    int atom_j = atom_blocks[8 * right_block + (lane & 7u)];
    accepted_i[output] = min(atom_i, atom_j);
    accepted_j[output] = max(atom_i, atom_j);
"""

_TILE_TOPOLOGY_LJ_MASKS_SOURCE = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint word_index = lane >> 5;
    uint bit_index = lane & 31u;
    threadgroup uint enabled[64];
    threadgroup uint one_four[64];

    uint member_word = member_mask[2 * tile + word_index];
    bool member = ((member_word >> bit_index) & 1u) != 0u;
    int left_block = tile_blocks[2 * tile + 0];
    int right_block = tile_blocks[2 * tile + 1];
    int atom_i = atom_blocks[8 * left_block + (lane >> 3)];
    int atom_j = atom_blocks[8 * right_block + (lane & 7u)];
    int left = min(atom_i, atom_j);
    int right = max(atom_i, atom_j);

    bool excluded = false;
    if (member) {
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
        excluded = lower < counts[1]
            && excluded_i[lower] == left
            && excluded_j[lower] == right;
    }

    bool scaled = false;
    if (member && !excluded && counts[2] > 0) {
        int lower = 0;
        int upper = counts[2];
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
        scaled = lower < counts[2]
            && one_four_i[lower] == left
            && one_four_j[lower] == right;
    }
    enabled[lane] = member && !excluded ? 1u : 0u;
    one_four[lane] = scaled ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < 2u) {
        uint enabled_word = 0u;
        uint one_four_word = 0u;
        uint base = 32u * lane;
        for (uint bit = 0u; bit < 32u; bit++) {
            enabled_word |= enabled[base + bit] << bit;
            one_four_word |= one_four[base + bit] << bit;
        }
        lj_enabled_mask[2 * tile + lane] = enabled_word;
        lj_one_four_mask[2 * tile + lane] = one_four_word;
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
    int atom[4];
    float vx[4];
    float vy[4];
    float vz[4];
    float base_vx[4];
    float base_vy[4];
    float base_vz[4];
    float inverse_mass[4];
    float unit_x[3];
    float unit_y[3];
    float unit_z[3];
    for (int slot = 0; slot < 4; slot++) {
        atom[slot] = cluster_atoms[4 * cluster + slot];
        if (slot <= peripheral_count) {
            int index = atom[slot];
            vx[slot] = velocities[3 * index + 0];
            vy[slot] = velocities[3 * index + 1];
            vz[slot] = velocities[3 * index + 2];
            base_vx[slot] = vx[slot];
            base_vy[slot] = vy[slot];
            base_vz[slot] = vz[slot];
            inverse_mass[slot] = 1.0f / masses[index];
        } else {
            vx[slot] = 0.0f;
            vy[slot] = 0.0f;
            vz[slot] = 0.0f;
            base_vx[slot] = 0.0f;
            base_vy[slot] = 0.0f;
            base_vz[slot] = 0.0f;
            inverse_mass[slot] = 0.0f;
        }
    }

    for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
        int slot = peripheral + 1;
        int center_atom = atom[0];
        int outer_atom = atom[slot];
        float dx = positions[3 * center_atom + 0] - positions[3 * outer_atom + 0];
        float dy = positions[3 * center_atom + 1] - positions[3 * outer_atom + 1];
        float dz = positions[3 * center_atom + 2] - positions[3 * outer_atom + 2];
        if (params[2] != 0) {
            dx -= box[0] * rint(dx / box[0]);
            dy -= box[1] * rint(dy / box[1]);
            dz -= box[2] * rint(dz / box[2]);
        }
        float inverse_length = rsqrt(max(dx * dx + dy * dy + dz * dz, 1.0e-20f));
        unit_x[peripheral] = dx * inverse_length;
        unit_y[peripheral] = dy * inverse_length;
        unit_z[peripheral] = dz * inverse_length;
    }

    for (int iteration = 0; iteration < params[1]; iteration++) {
        float center_delta_x = 0.0f;
        float center_delta_y = 0.0f;
        float center_delta_z = 0.0f;
        for (int peripheral = 0; peripheral < peripheral_count; peripheral++) {
            int slot = peripheral + 1;
            float relative = (vx[0] - vx[slot]) * unit_x[peripheral]
                + (vy[0] - vy[slot]) * unit_y[peripheral]
                + (vz[0] - vz[slot]) * unit_z[peripheral];
            float weight_center = inverse_mass[0]
                / (inverse_mass[0] + inverse_mass[slot]);
            float weight_outer = inverse_mass[slot]
                / (inverse_mass[0] + inverse_mass[slot]);
            float correction_x = relative * unit_x[peripheral];
            float correction_y = relative * unit_y[peripheral];
            float correction_z = relative * unit_z[peripheral];
            center_delta_x -= weight_center * correction_x;
            center_delta_y -= weight_center * correction_y;
            center_delta_z -= weight_center * correction_z;
            vx[slot] += weight_outer * correction_x;
            vy[slot] += weight_outer * correction_y;
            vz[slot] += weight_outer * correction_z;
        }
        vx[0] += center_delta_x;
        vy[0] += center_delta_y;
        vz[0] += center_delta_z;
    }

    for (int slot = 0; slot < 4; slot++) {
        uint output = 12 * cluster + 3 * slot;
        if (slot <= peripheral_count) {
            deltas[output + 0] = vx[slot] - base_vx[slot];
            deltas[output + 1] = vy[slot] - base_vy[slot];
            deltas[output + 2] = vz[slot] - base_vz[slot];
        } else {
            deltas[output + 0] = 0.0f;
            deltas[output + 1] = 0.0f;
            deltas[output + 2] = 0.0f;
        }
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
    """Return the cached spatial 8x8 tile direct-force kernel."""

    global _tile_pme_direct_force_only_kernel_singleton
    if _tile_pme_direct_force_only_kernel_singleton is None:
        _tile_pme_direct_force_only_kernel_singleton = mx.fast.metal_kernel(
            name="spatial_tile_prepared_parameterized_pme_direct_force_only",
            input_names=[
                "positions",
                "atom_blocks",
                "tile_blocks",
                "member_mask",
                "lj_enabled_mask",
                "lj_one_four_mask",
                "force_group_starts",
                "force_group_counts",
                "box",
                "half_sigma",
                "sqrt_epsilon",
                "charges",
                "params",
            ],
            output_names=["forces"],
            source=_TILE_PREPARED_PME_DIRECT_FORCE_ONLY_SOURCE,
            header=_ERF_HEADER,
            atomic_outputs=True,
        )
    return _tile_pme_direct_force_only_kernel_singleton


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
            header=(
                _PME_ORDER5_HEADER
                + "\n#define MLX_ATOMISTIC_PME_WRITE_ENERGY 1\n"
            ),
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


def _neighbor_tile_membership_kernel():
    """Return the cached exact spatial-tile membership kernel."""

    global _neighbor_tile_membership_kernel_singleton
    if _neighbor_tile_membership_kernel_singleton is None:
        _neighbor_tile_membership_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_membership",
            input_names=["positions", "atom_blocks", "tile_blocks", "box", "params"],
            output_names=["member_mask", "member_counts"],
            source=_NEIGHBOR_TILE_MEMBERSHIP_SOURCE,
        )
    return _neighbor_tile_membership_kernel_singleton


def _neighbor_tile_ordered_scatter_kernel():
    """Return the cached non-empty spatial-tile compaction kernel."""

    global _neighbor_tile_ordered_scatter_kernel_singleton
    if _neighbor_tile_ordered_scatter_kernel_singleton is None:
        _neighbor_tile_ordered_scatter_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_ordered_scatter",
            input_names=[
                "tile_blocks",
                "member_mask",
                "member_counts",
                "prefix",
                "counts",
            ],
            output_names=["accepted_tile_blocks", "accepted_member_mask"],
            source=_NEIGHBOR_TILE_ORDERED_SCATTER_SOURCE,
        )
    return _neighbor_tile_ordered_scatter_kernel_singleton


def _neighbor_tile_force_groups_kernel():
    """Return the cached spatial-tile force-group schedule kernel."""

    global _neighbor_tile_force_groups_kernel_singleton
    if _neighbor_tile_force_groups_kernel_singleton is None:
        _neighbor_tile_force_groups_kernel_singleton = mx.fast.metal_kernel(
            name="neighbor_tile_force_groups",
            input_names=[
                "tile_counts",
                "tile_prefix",
                "group_counts",
                "group_prefix",
                "counts",
            ],
            output_names=["force_group_starts", "force_group_counts"],
            source=_NEIGHBOR_TILE_FORCE_GROUPS_SOURCE,
        )
    return _neighbor_tile_force_groups_kernel_singleton


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
                "excluded_i",
                "excluded_j",
                "one_four_i",
                "one_four_j",
                "counts",
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
    """Return the cached disjoint SHAKE-cluster velocity kernel."""

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
        )
    return _shake_cluster_velocity_kernel_singleton


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
    if (
        cell_block_starts.ndim != 1
        or cell_block_counts.shape != cell_block_starts.shape
    ):
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
) -> tuple[mx.array, mx.array]:
    """Encode exact cutoff-plus-skin membership for spatial 8x8 tiles."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    box_lengths = as_mx_array(box_lengths, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = "atom_blocks must have shape (n_blocks, 8)"
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
            mx.zeros((0, 2), dtype=mx.uint32),
            mx.zeros((0,), dtype=mx.int32),
        )
    member_mask, member_counts = _neighbor_tile_membership_kernel()(
        inputs=[
            positions,
            atom_blocks,
            tile_blocks,
            box_lengths,
            mx.array([float(search_radius) ** 2], dtype=mx.float32),
        ],
        output_shapes=[(tile_count, 2), (tile_count,)],
        output_dtypes=[mx.uint32, mx.int32],
        grid=(tile_count * _TILE_PME_LANES_PER_TILE, 1, 1),
        threadgroup=(_TILE_MEMBERSHIP_THREADGROUP_SIZE, 1, 1),
        init_value=0,
    )
    return member_mask, member_counts


def _neighbor_tile_ordered_scatter_sized(
    tile_blocks: mx.array,
    member_mask: mx.array,
    member_counts: mx.array,
    prefix: mx.array,
    *,
    accepted_count: int,
) -> tuple[mx.array, mx.array]:
    """Compact non-empty spatial tiles into explicitly sized outputs."""

    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    member_counts = as_mx_array(member_counts, dtype=mx.int32)
    prefix = as_mx_array(prefix, dtype=mx.int32)
    tile_count = int(tile_blocks.shape[0])
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if member_mask.shape != (tile_count, 2):
        msg = "member_mask must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if member_counts.shape != (tile_count,) or prefix.shape != (tile_count,):
        msg = "member_counts and prefix must contain one value per tile"
        raise ValueError(msg)
    if accepted_count < 0 or accepted_count > tile_count:
        msg = "accepted_count must fit within the candidate tile count"
        raise ValueError(msg)
    if accepted_count == 0:
        return (
            mx.zeros((0, 2), dtype=mx.int32),
            mx.zeros((0, 2), dtype=mx.uint32),
        )
    accepted_tiles, accepted_mask = _neighbor_tile_ordered_scatter_kernel()(
        inputs=[
            tile_blocks,
            member_mask,
            member_counts,
            prefix,
            mx.array([tile_count], dtype=mx.int32),
        ],
        output_shapes=[(accepted_count, 2), (accepted_count, 2)],
        output_dtypes=[mx.int32, mx.uint32],
        grid=(tile_count, 1, 1),
        threadgroup=(min(256, tile_count), 1, 1),
        init_value=0,
    )
    return accepted_tiles, accepted_mask


def _neighbor_tile_force_groups_sized(
    tile_counts: mx.array,
    tile_prefix: mx.array,
    group_counts: mx.array,
    group_prefix: mx.array,
    *,
    accepted_count: int,
    tiles_per_group: int,
) -> tuple[mx.array, mx.array]:
    """Build contiguous same-left-block force groups from tile row counts."""

    tile_counts = as_mx_array(tile_counts, dtype=mx.int32)
    tile_prefix = as_mx_array(tile_prefix, dtype=mx.int32)
    group_counts = as_mx_array(group_counts, dtype=mx.int32)
    group_prefix = as_mx_array(group_prefix, dtype=mx.int32)
    block_count = int(tile_counts.shape[0])
    if any(
        values.shape != (block_count,)
        for values in (tile_prefix, group_counts, group_prefix)
    ):
        msg = "tile and group counts/prefixes must have matching one-dimensional shapes"
        raise ValueError(msg)
    if tiles_per_group <= 0:
        msg = "tiles_per_group must be positive"
        raise ValueError(msg)
    if accepted_count < 0:
        msg = "accepted_count must be non-negative"
        raise ValueError(msg)
    if accepted_count == 0:
        empty = mx.zeros((0,), dtype=mx.int32)
        return empty, empty
    force_group_starts, force_group_sizes = _neighbor_tile_force_groups_kernel()(
        inputs=[
            tile_counts,
            tile_prefix,
            group_counts,
            group_prefix,
            mx.array([block_count, tiles_per_group], dtype=mx.int32),
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
        msg = "atom_blocks must have shape (n_blocks, 8)"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if member_mask.shape != (tile_count, 2):
        msg = "member_mask must have shape (n_tiles, 2)"
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


def tile_topology_lj_masks(
    atom_blocks: mx.array,
    tile_blocks: mx.array,
    member_mask: mx.array,
    excluded_pairs: mx.array,
    one_four_pairs: mx.array,
) -> tuple[mx.array, mx.array]:
    """Build tile-aligned LJ eligibility and 1-4 masks on Metal."""

    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    excluded_pairs = as_mx_array(excluded_pairs, dtype=mx.int32)
    one_four_pairs = as_mx_array(one_four_pairs, dtype=mx.int32)
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = "atom_blocks must have shape (n_blocks, 8)"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    tile_count = int(tile_blocks.shape[0])
    if member_mask.shape != (tile_count, 2):
        msg = "member_mask must have shape (n_tiles, 2)"
        raise ValueError(msg)
    for name, values in (
        ("excluded_pairs", excluded_pairs),
        ("one_four_pairs", one_four_pairs),
    ):
        if values.ndim != 2 or values.shape[1] != 2:
            msg = f"{name} must have shape (n, 2)"
            raise ValueError(msg)
    if tile_count == 0:
        empty = mx.zeros((0, 2), dtype=mx.uint32)
        return empty, empty
    enabled_mask, one_four_mask = _tile_topology_lj_masks_kernel()(
        inputs=[
            atom_blocks,
            tile_blocks,
            member_mask,
            excluded_pairs[:, 0],
            excluded_pairs[:, 1],
            one_four_pairs[:, 0],
            one_four_pairs[:, 1],
            mx.array(
                [tile_count, int(excluded_pairs.shape[0]), int(one_four_pairs.shape[0])],
                dtype=mx.int32,
            ),
        ],
        output_shapes=[(tile_count, 2), (tile_count, 2)],
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
    max_iterations: int,
    periodic: bool,
) -> mx.array:
    """Return per-cluster RATTLE velocity deltas from one Metal dispatch."""

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
    if max_iterations <= 0:
        msg = "max_iterations must be positive"
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
) -> mx.array:
    """Evaluate four standard bonded families in one force-only dispatch."""

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

    total_count = sum(counts)
    if total_count == 0:
        return mx.zeros_like(positions)
    count_array = mx.array(counts, dtype=mx.int32)
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
    worker_count = (
        n_pairs + _PREPARED_PME_PAIRS_PER_WORKER - 1
    ) // _PREPARED_PME_PAIRS_PER_WORKER
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
) -> mx.array:
    """Evaluate prepared direct forces from exact spatial 8x8 tiles."""

    positions = as_mx_array(positions, dtype=mx.float32)
    atom_blocks = as_mx_array(atom_blocks, dtype=mx.int32)
    tile_blocks = as_mx_array(tile_blocks, dtype=mx.int32)
    member_mask = as_mx_array(member_mask, dtype=mx.uint32)
    lj_enabled_mask = as_mx_array(lj_enabled_mask, dtype=mx.uint32)
    lj_one_four_mask = as_mx_array(lj_one_four_mask, dtype=mx.uint32)
    force_group_starts = as_mx_array(force_group_starts, dtype=mx.int32)
    force_group_counts = as_mx_array(force_group_counts, dtype=mx.int32)
    box = as_mx_array(box_lengths_and_inverses, dtype=mx.float32)
    half_sigma = as_mx_array(half_sigma, dtype=mx.float32)
    sqrt_epsilon = as_mx_array(sqrt_epsilon, dtype=mx.float32)
    charges = as_mx_array(charges, dtype=mx.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = "positions must have shape (n_atoms, 3)"
        raise ValueError(msg)
    n_atoms = int(positions.shape[0])
    if atom_blocks.ndim != 2 or atom_blocks.shape[1] != _TILE_PME_BLOCK_SIZE:
        msg = "atom_blocks must have shape (n_blocks, 8)"
        raise ValueError(msg)
    if tile_blocks.ndim != 2 or tile_blocks.shape[1] != 2:
        msg = "tile_blocks must have shape (n_tiles, 2)"
        raise ValueError(msg)
    tile_count = int(tile_blocks.shape[0])
    mask_shape = (tile_count, 2)
    if member_mask.shape != mask_shape:
        msg = "member_mask must have shape (n_tiles, 2)"
        raise ValueError(msg)
    if lj_enabled_mask.shape != mask_shape or lj_one_four_mask.shape != mask_shape:
        msg = "tile LJ masks must have shape (n_tiles, 2)"
        raise ValueError(msg)
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
        return mx.zeros_like(positions)
    if group_count == 0:
        msg = "non-empty tile geometry requires at least one force group"
        raise ValueError(msg)

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
            float(one_four_scale),
        ],
        dtype=mx.float32,
    )
    (forces,) = _tile_pme_direct_force_only_kernel()(
        inputs=[
            positions,
            atom_blocks,
            tile_blocks,
            member_mask,
            lj_enabled_mask,
            lj_one_four_mask,
            force_group_starts,
            force_group_counts,
            box,
            half_sigma,
            sqrt_epsilon,
            charges,
            params,
        ],
        output_shapes=[(n_atoms, 3)],
        output_dtypes=[mx.float32],
        grid=(group_count * _TILE_PME_THREADGROUP_SIZE, 1, 1),
        threadgroup=(_TILE_PME_THREADGROUP_SIZE, 1, 1),
        init_value=0.0,
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
