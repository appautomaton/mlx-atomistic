# MD Reciprocal PME Spread-Launch Verdict on M5 Max

Date: 2026-08-13

## Decision

Retain the existing order-five PME charge-spread Metal kernel. Three launch
geometry experiments preserved charge-grid and reciprocal-force parity, but
none passed the predeclared performance gate on both 5DFR and JAC. No candidate
entered the production path, and the 75-step complete-wall gate was not run.

The result closes charge-spread thread-layout tuning as the next reciprocal PME
optimization. A fresh production force-only profile shows that charge spread,
forward FFT, inverse FFT plus influence, and interpolation are now balanced.
The largest isolated row is the inverse FFT path, which remains an MLX runtime
operation rather than a project-owned Metal kernel.

## Candidates

The retained kernel launches five work items per atom. Each work item owns one
z offset and performs the 25 x-y atomic additions for that slice.

Three alternatives were tested in sequence:

1. `simd32_xy_slices` assigned 25 active lanes per atom and five z additions
   per active lane. It repeated coordinate and B-spline weight work in all 25
   lanes.
2. `simd32_shared_xy` kept the 25-lane write layout but calculated the anchor,
   charge, and 15 weights once in lane zero and distributed them with
   `simd_shuffle`.
3. `simd8_shared_z` preserved the retained five z workers and 25 additions per
   active worker. Four padded eight-lane atom groups shared each SIMD group,
   and the first lane in each atom group calculated and distributed the common
   values.

The first two candidates tested whether more parallel atomic writes could hide
the serial 25-add loop. The third isolated weight recomputation while preserving
the retained atomic-write pattern.

## Correctness

All candidates passed same-input charge-grid and complete reciprocal-force
comparisons. The maximum candidate-minus-control differences were:

| Candidate | Workload | Charge grid | Reciprocal force, kJ/mol/A |
| --- | --- | ---: | ---: |
| `simd32_xy_slices` | 5DFR | `5.22e-8` | `2.57e-5` |
| `simd32_xy_slices` | JAC | `4.47e-8` | `3.67e-5` |
| `simd32_shared_xy` | 5DFR | `4.47e-8` | `2.62e-5` |
| `simd32_shared_xy` | JAC | `4.47e-8` | `3.77e-5` |
| `simd8_shared_z` | 5DFR | `3.73e-8` | `1.91e-5` |
| `simd8_shared_z` | JAC | `2.24e-8` | `2.15e-5` |

The admission limits were `2e-6` for the charge grid and `2e-4 kJ/mol/A` for
reciprocal force. Numerical correctness therefore did not decide the verdict.

## Performance

The benchmark alternated control-first and candidate-first timing blocks. Each
sample contained ten individually synchronized evaluations, preserving the
per-call launch and synchronization shape while reducing frequency-transition
noise. The gate required at least 5% charge-spread speedup, at least 1% complete
reciprocal-force-graph speedup, passing parity, and positive results in both
call-order partitions.

| Candidate | Workload | Charge spread | Reciprocal graph | Stable directions |
| --- | --- | ---: | ---: | --- |
| `simd32_xy_slices` | 5DFR | -31.5% | -30.2% | no |
| `simd32_xy_slices` | JAC | -63.5% | -28.1% | no |
| `simd32_shared_xy` | 5DFR | -34.2% | -23.4% | no |
| `simd32_shared_xy` | JAC | -96.0% | -31.0% | no |
| `simd8_shared_z` | 5DFR | +1.08% | +0.43% | no |
| `simd8_shared_z` | JAC | +1.11% | +0.72% | yes |

The 25-lane variants show that exposing more atomic writes is more expensive
than the original serial loop, even when the common arithmetic is shared. The
eight-lane variant confirms that weight sharing is directionally plausible on
JAC, but its sub-percent graph effect is below the gate and is not stable on
5DFR. The experimental kernels were removed after the verdict.

Raw local reports are under `results/reciprocal-pme-launch/`:

- `simd32-xy-{5dfr,jac}.json`
- `simd32-shared-xy-{5dfr,jac}.json`
- `simd8-shared-z-{5dfr,jac}.json`

## Fresh production-stage profile

The existing PME profiler contained a stale admission check: it built compact
`mlx_cell_pairs`, then required the former `block_candidate`/`blocks` policy.
Its final checks already required the current `compact_pair`/`pairs` contract.
The admission condition now matches the production contract.

The profiler also used generic MLX assignment and interpolation rows in its
stage summary even on the order-five Metal route. It now emits and prefers the
exact recurring force-only stages while retaining the generic rows as reference
diagnostics. A seven-sample JAC run measured:

| Production reciprocal stage | Median |
| --- | ---: |
| Order-five Metal charge spread | 0.394 ms |
| Forward FFT | 0.441 ms |
| Influence multiplication and inverse FFT | 0.554 ms |
| Complex-grid Metal force interpolation | 0.384 ms |
| Complete compiled reciprocal force graph | 1.237 ms |

Each isolated row includes its own synchronization and must not be summed to
predict the compiled graph. The profile output is
`results/reciprocal-pme-launch/current-jac-production-stages/pme-profile.json`.

## Boundary and next action

Do not add another charge-spread launch geometry without a materially different
algorithm, such as reducing the number of global atomic additions rather than
redistributing the same 125 additions. Do not replace or rewrite MLX FFT based
on this profile.

Reciprocal PME is no longer the primary end-to-end MD target. Future kernel work
must return to a fresh complete MD route profile and select the largest retained
project-owned stage. A reciprocal interpolation prototype is justified only if
that profile shows reciprocal PME has again become material to complete wall.
