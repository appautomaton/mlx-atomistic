# Fused Direct Space and sparse-correction verdict on Apple M5 Max

Date: 2026-08-13

## Decision

Reject the single-output Direct Space and sparse Particle Mesh Ewald (PME)
correction prototype. It reduced the synchronized nonbonded route and produced
a small sustained JAC improvement, but the result did not transfer to 5DFR.
The 5DFR production trajectory rebuilt its neighbor list more often and became
slower. The candidate Metal source, runtime wiring, and candidate-only tests
were removed.

Retain only the profiler attribution improvement developed before the
prototype. The former `force_aggregation` bucket mixed three different
operations. Synchronized profiles now report:

- `pme_force_aggregation` for Direct Space, reciprocal PME, and correction
  force-array aggregation;
- `force_term_aggregation` for bonded and nonbonded force-array aggregation;
- `virtual_site_force_redistribution` only when virtual sites are present.

The last change also stops timing a no-op redistribution when a workload has no
virtual sites. None of these profiler-only changes modifies production runtime
behavior.

## Measured opportunity

The post-packed-descriptor JAC profile separated the relevant force-call costs:

| Route | Time per force call |
| --- | ---: |
| Direct Space spatial tiles | 1.742 ms |
| Sparse PME corrections | 0.269 ms |
| PME force aggregation | 0.198 ms |
| Force-term aggregation | 0.201 ms |

The candidate appended sparse-correction SIMD groups to the existing Direct
Space Metal dispatch. Both kinds of work atomically accumulated into one force
output. Reciprocal PME then required a two-array aggregation instead of a
three-array aggregation. The formulas, minimum-image convention, and sparse
Lennard-Jones exception parameters were unchanged.

A synchronized JAC profile measured the combined Direct Space and correction
route at 1.744 ms per force call. Compared with 2.011 ms for the two separate
routes, this was a 13.3% reduction. The complete synchronized PME force path was
about 9% shorter. Metal half-box and tile-parity tests passed.

## Isolated evidence

The benchmark profiler interleaved six separate-output and six single-output
calls in each process after two warmups. A 5DFR process measured a 6.05%
candidate reduction. Three JAC processes measured `-0.87%`, `+1.43%`, and
`+2.11%`; the median direction was a 1.43% improvement. The slow-state JAC
sample demonstrated that combining dispatches did not win in every hardware
performance state.

Immediate combined-force parity remained within the existing float32 atomic
tolerances. The observed root-mean-square candidate-minus-control force delta
was about `2.8e-5 kJ mol^-1 A^-1`, and the maximum was at most
`3.74e-4 kJ mol^-1 A^-1`.

## Complete-wall evidence

All independent processes used 10 warmups, a 0.004 ps timestep, 300 K, seed 17,
a 5.5 Angstrom neighbor skin, one-step neighbor checks, and the
`mlx_cell_tiles` backend. Control and candidate processes were position
balanced. Every runtime check passed.

Two 75-step samples per arm did not support retention:

| Workload | Control median | Candidate median | Candidate result | Rebuilds |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 98.308 ms | 102.384 ms | 4.15% slower | 2 / 2 |
| JAC | 373.900 ms | 382.485 ms | 2.30% slower | 2 / 2 |

Longer JAC runs were encouraging but small. Two 750-step samples per arm all
rebuilt 22 times. The control median was 6.116 s and the candidate median was
6.058 s, a 0.95% candidate improvement.

The same 750-step gate failed to transfer to 5DFR:

| Arm | Samples | Rebuild counts |
| --- | --- | --- |
| Control | 1.468 s, 1.488 s | 22, 21 |
| Candidate | 1.576 s, 1.502 s | 23, 23 |

The candidate median was 4.14% slower. The rebuild counts differ, so this row
is not a clean kernel speed ratio. It is nevertheless the relevant production
result: changing atomic accumulation order perturbed the trajectory enough to
move neighbor rebuild points, and the complete runtime became slower. A design
that wins only when those consequences are excluded is not a transferable MD
optimization.

Raw generated reports remain gitignored under
`results/fused-direct-correction/`. The detached control was commit `ff94d19`.

## Boundary

Do not revive this mixed-dispatch design without a force-accumulation strategy
that preserves or controls trajectory-level rebuild behavior. The refined
profile shows that aggregation remains measurable, but eliminating an array is
not sufficient by itself. A future candidate must demonstrate equal-rebuild
complete-wall improvement on both 5DFR and JAC before retention.
