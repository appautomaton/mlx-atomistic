"""Fused Metal kernels for recurring Lennard-Jones force paths.

Collapses the per-step pairwise LJ force op-chain (gather -> minimum image -> r^2 ->
LJ scalar -> scatter-add) into a single ``mx.fast.metal_kernel`` dispatch. Forces use an
atomic scatter into a half neighbor list; per-pair energy is written to its own slot
(no contention) and summed by the caller -- this keeps the energy accurate (needed by the
periodic-virial finite-difference path) without a single-cell energy-atomic hot spot.

The simple kernel covers scalar reduced-unit LJ. The parameterized kernel covers
per-atom Lorentz-Berthelot parameters, topology scales, shifts, and smooth switching
for the production biomolecular path. Unsupported cases fall back transparently.

Because ``tests/conftest.py`` forces ``MLX_ATOMISTIC_DEVICE=cpu``, the kernel is built
lazily on first use (not at import) so importing this module never triggers a Metal
device load.
"""

from __future__ import annotations

from math import pi, sqrt

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
    float lj_energy_value = 0.0f;
    float coulomb_energy_value = 0.0f;
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
            lj_energy_value =
                unswitched_energy * switch_value * lj_scale;
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
        coulomb_energy_value =
            params[6] * qij * erfc_term / distance;
        scalar += params[6] * qij * (
            erfc_term / (r2 * distance)
            + params[8] * exp(-params[7] * params[7] * r2) / r2
        );

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
        &pair_lj_energy[t], lj_energy_value, memory_order_relaxed
    );
    atomic_store_explicit(
        &pair_coulomb_energy[t],
        coulomb_energy_value,
        memory_order_relaxed
    );
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
