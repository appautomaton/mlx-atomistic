# Retained-stack Phase 5 confirmation on Apple M5 Max

Date: 2026-08-11

This rollup closes the cross-regime confirmation after commit
`20199e46d16d57e79559897d61109e846bca1629`. It covers the retained MLX/Metal
product runtime only. OpenMM is executed separately as a manifest-bound
reference and never enters the MLX runtime path.

## Decision summary

- The deferred exact-pair molecular-dynamics route is retained after
  synchronized profiling, two 750-step 5DFR runs, two 750-step JAC runs, and a
  1,500-step JAC memory-settling run.
- A cross-SIMD-group atomic-reduction Metal kernel is `no-go`: its measured
  schedule admits too little complete-step headroom to clear the timing noise.
- The two Phase 2 density-functional theory (DFT) Metal boundary candidates
  were removed, so there is no retained candidate kernel requiring a new
  Carbon or MgO transfer claim. The accepted Si, C, and MgO scientific ledger
  remains the product boundary.
- A fresh, contemporaneous, manifest-matched JAC comparison reduces the
  MLX/OpenMM latency ratio from the historical `9.7586x` to `5.2701x` by the
  ratio of the two-run medians.

## Post-change 5DFR profile

The formal `charged_pme profile` matrix used 10 warmups, 75 measured steps,
0.004 ps, 300 K, seed 17, a 9 A cutoff, 5.5 A skin, and one-step neighbor
checks. It ran pair controls before and after two clean tile samples, one
synchronized instrumented tile sample, and the tile-versus-pair force oracle.
Every science, route, inventory, memory, and force-parity check passed.

| Complete-wall sample | Seconds |
| --- | ---: |
| Pair control before | 0.201014 |
| Tile clean | 0.105260 |
| Tile clean repeat | 0.104415 |
| Pair control after | 0.190088 |

The pair median was 0.195551 seconds and the tile median was 0.104837 seconds,
a 46.39% reduction for this pair-versus-tile profile matrix. This comparison is
separate from the baseline-commit comparison in the Phase 4 verdict.

The synchronized sample intentionally inserts completion barriers. Its route
shares diagnose ordering but are not uninstrumented complete-wall shares.

| Route or route family | Wall | Instrumented share |
| --- | ---: | ---: |
| SETTLE and SHAKE constraints combined | 89.487 ms | 27.44% |
| `direct_spatial_tiles` | 51.854 ms | 15.90% |
| Integration and thermostat | 50.839 ms | 15.59% |
| Reciprocal Particle Mesh Ewald (PME) | 29.053 ms | 8.91% |
| Neighbor update and rebuild | 28.868 ms | 8.85% |
| Force aggregation | 26.951 ms | 8.26% |

Neighbor work is no longer the dominant route. Direct tiles are the largest
single producer, but they are effectively tied with integration and are below
the combined constraint work.

## Sustained confirmation

| Workload | Measured steps | Wall samples | Rebuilds | Metal peak | Result |
| --- | ---: | --- | --- | ---: | --- |
| 5DFR, 23,558 atoms | 750 | 1.626131 s; 1.456710 s | 22; 23 | 116.1; 116.2 MB | Both passed, both physical-memory plateaus passed |
| JAC, 94,232 atoms | 750 | 6.761443 s; 6.735158 s | 22; 22 | 494.1; 493.6 MB | Both passed; one short-run plateau passed |
| JAC, 94,232 atoms | 1,500 | 13.342558 s | 45 | 494.1 MB | Passed; physical memory settled near 4.68 GB |

Every run kept `diagnostic_pairs_materialized = false` and reported zero
compact-pair bytes. The first 750-step JAC trace ended while the MLX allocator
cache was still growing; the second passed, and the 1,500-step trace showed the
process flattening at about 4.68 GB. Final active MLX arrays were about 158 MB,
the allocator cache was about 4.03 GB, and there was no rebuild-proportional
unbounded growth.

## Atomic-kernel decision

The current direct kernel already reduces every force group locally and packs
four independent SIMD groups into one Metal threadgroup. A possible successor
would merge repeated endpoint blocks across those four groups using bounded
threadgroup memory.

The exact schedules reject that implementation before code is written:

| Schedule bound | 5DFR | JAC |
| --- | ---: | ---: |
| Left endpoint block writes removable within one packed threadgroup | 73.30% | 73.38% |
| Right endpoint block writes removable | 0.0102% | 0.0022% |
| All endpoint block writes removable | 15.19% | 15.19% |

Historical ablation assigns only about 15% of direct-kernel work to all atomic
writes. Removing at most 15.19% of those writes therefore bounds the ideal
kernel gain near 2.3% of the direct kernel before adding barriers, comparisons,
and threadgroup scratch. Diluted through the complete step, that is below the
measured timing noise. The earlier global two-pass reducer remains closed
because its 264 MB temporary made complete wall slower. No new MD kernel is
retained from Phase 5.

## Refreshed OpenMM context

Two fresh MLX runs and two immediately following OpenMM/OpenCL reference runs
used the exact 94,232-atom modern JAC artifact, 10 warmups, 75 measured steps,
single precision, fixed-cell Langevin-middle dynamics, and the same PME,
constraint, timing-boundary, and final-device-completion contract. The machine
reported AC Power and Low Power Mode for the refresh.

| Engine | Sample 1 | Sample 2 | Median |
| --- | ---: | ---: | ---: |
| MLX/Metal | 0.533019 s | 0.524504 s | 0.528761 s |
| OpenMM/OpenCL reference | 0.097707 s | 0.102956 s | 0.100332 s |

Both generated runtime comparisons passed every manifest and timing check.
The individual ratios were `5.4553x` and `5.0944x`; the ratio of medians is
`5.2701x`. The former admitted MLX row was 1.004613 seconds, so the current MLX
median is 47.37% lower while the reference median remains about 0.100 seconds.
This ratio applies only to this protocol, host, precision, and power state.

## DFT boundary and next work

The DFT Phase 2 candidates were correctly removed after their complete-wall
gates failed. Existing committed evidence already validates the accepted
diamond-Si, diamond-C, and rock-salt-MgO workloads, including the declared MgO
pseudopotential and force-precision limits. Phase 5 does not turn those narrow
rows into a broader chemistry claim.

The retained Hpsi profiler says the next DFT optimization is the amount of work
submitted to the existing MLX fast Fourier transform (FFT): padded vector
capacity, shape scheduling, and avoidable FFT-vector equivalents. It is not
another narrow scatter/gather Metal kernel. For MD, another kernel should wait
for a new profile that exposes a larger attributable boundary or a design that
removes more than the 2.3% direct-kernel bound without proportional storage.

Raw generated evidence is gitignored under
`results/md-post-pair-elision/`. Scientific DFT boundaries are recorded in
[`dft-material-validation.md`](./dft-material-validation.md), the rejected Hpsi
experiments in
[`dft-hpsi-metal-boundary-m5max.md`](./dft-hpsi-metal-boundary-m5max.md), and
the retained Phase 4 MD change in
[`md-neighbor-roundtrip-verdict-m5max.md`](./md-neighbor-roundtrip-verdict-m5max.md).
