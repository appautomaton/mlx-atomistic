"""Fused Metal kernels for recurring Lennard-Jones force paths.

Collapses the per-step pairwise LJ force op-chain (gather -> minimum image -> r^2 ->
LJ scalar -> scatter-add) into a single ``mx.fast.metal_kernel`` dispatch. Diagnostic
kernels write per-pair energy without contention, while force-only kernels omit those
outputs and reductions entirely.

The simple kernel covers scalar reduced-unit LJ. The parameterized kernel covers
per-atom Lorentz-Berthelot parameters, topology scales, shifts, and smooth switching
for the production biomolecular path. Unsupported cases fall back transparently.

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

_kernel_singleton = None
_parameterized_kernel_singleton = None
_pme_direct_kernel_singleton = None
_pme_direct_virial_kernel_singleton = None
_pme_direct_force_only_kernel_singleton = None
_prepared_pme_direct_force_only_kernel_singleton = None
_pme_cutoff_correction_virial_kernel_singleton = None
_pme_order5_spread_kernel_singleton = None
_pme_order5_interpolate_kernel_singleton = None
_aligned_topology_lj_scales_kernel_singleton = None
_neighbor_cell_pair_candidates_kernel_singleton = None
_neighbor_pair_cutoff_mask_kernel_singleton = None
_neighbor_pair_ordered_scatter_kernel_singleton = None
_shake_cluster_position_kernel_singleton = None
_shake_cluster_velocity_kernel_singleton = None
_settle_water_position_kernel_singleton = None
_settle_water_velocity_kernel_singleton = None

# Neighbor compaction preserves short runs of a common left atom, so one worker
# can sum those contributions locally before issuing global atomics.
_PREPARED_PME_PAIRS_PER_WORKER = 8

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
    dx -= lx * floor(dx * box[3] + 0.5f);
    dy -= ly * floor(dy * box[4] + 0.5f);
    dz -= lz * floor(dz * box[5] + 0.5f);
#else
    dx -= lx * floor(dx / lx + 0.5f);
    dy -= ly * floor(dy / ly + 0.5f);
    dz -= lz * floor(dz / lz + 0.5f);
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
    float potential = 0.0f;
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
                potential += wx * wy * wz * grid_value;
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
    atom_energy[atom] = 0.5f * charge * potential;
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
            header=_PME_ORDER5_HEADER,
        )
    return _pme_order5_interpolate_kernel_singleton


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
