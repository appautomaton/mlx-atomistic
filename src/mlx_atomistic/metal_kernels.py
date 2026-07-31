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
_pme_cutoff_correction_virial_kernel_singleton = None
_pme_order5_spread_kernel_singleton = None
_pme_order5_interpolate_kernel_singleton = None
_aligned_topology_lj_scales_kernel_singleton = None
_neighbor_pair_cutoff_mask_kernel_singleton = None

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
    uint t = thread_position_in_grid.x;
    if (t >= (uint)npair[0]) {
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
    dx -= lx * floor(dx / lx + 0.5f);
    dy -= ly * floor(dy / ly + 0.5f);
    dz -= lz * floor(dz / lz + 0.5f);

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
        float distance = sqrt(r2);
        float scalar = 0.0f;
        float lj_scale = lj_scales[t];
        if (lj_scale != 0.0f) {
            float sigma_ij = 0.5f * (sigma[i] + sigma[j]);
            float epsilon_ij = sqrt(epsilon[i] * epsilon[j]);
            float sigma2_over_r2 = sigma_ij * sigma_ij / r2;
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
                    (distance - params[3]) / params[4],
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
                    ) / params[4];
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
                / r2
                * switch_value
                - unswitched_energy * switch_derivative / distance
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
            erfc_term / (r2 * distance)
            + params[8] * exp(-params[7] * params[7] * r2) / r2
        );

        float fx = scalar * dx;
        float fy = scalar * dy;
        float fz = scalar * dz;
#ifdef MLX_ATOMISTIC_VIRIAL
        virial_x = dx * fx;
        virial_y = dy * fy;
        virial_z = dz * fz;
#endif
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
"""

_PARAMETERIZED_PME_DIRECT_VIRIAL_SOURCE = (
    "#define MLX_ATOMISTIC_VIRIAL 1\n" + _PARAMETERIZED_PME_DIRECT_SOURCE
)
_PARAMETERIZED_PME_DIRECT_FORCE_ONLY_SOURCE = (
    "#define MLX_ATOMISTIC_FORCE_ONLY 1\n" + _PARAMETERIZED_PME_DIRECT_SOURCE
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
