# Neighbor wait attribution on M5 Max

Status: attribution retained; scheduling and parameter-branch candidates
rejected on 2026-08-12.

## Question

The clean constrained NVT runtime submits direct, reciprocal, correction, and
bonded forces asynchronously. The next neighbor displacement admission calls
`mx.eval`, so `NeighborListManager.update_wall_seconds` includes both actual
neighbor work and completion of upstream Metal work. This experiment separated
those costs and tested whether changing the asynchronous submission boundary
could reduce complete wall time.

## Boundary attribution

A read-only benchmark evaluated incoming positions before entering the existing
MLX scalar displacement check. The first interval therefore measured upstream
completion, while the second measured the finite check, minimum-image
displacement reduction, scalar materialization, and rebuild decision with the
input already complete.

The 10-warmup plus 75-measured-step runs made 87 manager calls:

| Workload | Upstream completion | Pure displacement check | Upstream fraction |
| --- | ---: | ---: | ---: |
| 5DFR | 71.53 ms | 27.16 ms | 72.5% |
| JAC | 197.93 ms | 25.76 ms | 88.5% |

The apparent JAC neighbor bottleneck is predominantly an accounting boundary
for earlier GPU work. A new displacement kernel cannot remove that wait.

A fresh synchronized 20-step JAC route profile reconciled 99% of its
instrumented wall time. The leading force routes were:

| Route | Wall time | Calls |
| --- | ---: | ---: |
| `direct_spatial_tiles` | 30.71 ms | 19 |
| `reciprocal_pme` | 16.38 ms | 19 |
| `force_aggregation` | 6.71 ms | 57 |
| `pme_exceptions_corrections` | 5.69 ms | 19 |
| `bonded_fused` | 4.34 ms | 19 |

Constraint and integration routes remain visible in a synchronized profile,
but the preceding final-kick and combined-constraint experiments already
showed that their isolated wins do not transfer to sustained JAC wall time.

## Async submission experiments

The control submits `next_forces` immediately after force graph construction.
Two alternatives were measured:

1. Keep the force submission and additionally submit the final velocity graph.
2. Remove the early force submission and submit only the final velocity graph.

The first alternative created two dependent asynchronous command-buffer
boundaries. Some 75-step JAC samples grew from approximately 0.4 seconds to
more than 4.5 seconds. This was queue backpressure, not a physics failure; all
runs remained finite and passed with two rebuilds.

The second alternative was stable but delayed useful force execution. After a
system-stability probe, three 750-step JAC pairs produced:

| Route | Median wall time | Individual times | Rebuilds |
| --- | ---: | --- | --- |
| Existing early force submission | 6.357827 s | 5.9852 / 6.3578 / 6.5228 s | 22 / 22 / 22 |
| Late velocity-only submission | 6.624113 s | 6.5553 / 6.6241 / 8.2453 s | 22 / 22 / 22 |

The late submission was 4.19% slower at equal rebuild count. Both scheduling
candidates were removed.

## Direct-kernel parameter sparsity experiment

Both prepared workloads have zero Lennard-Jones epsilon on 59.70% of atoms.
A small force-only kernel candidate skipped sigma, inverse-power, and switching
arithmetic unless both pair endpoints had positive square-root epsilon.
Coulomb arithmetic remained unchanged, and Metal parity tests passed.

Two independent JAC tile-inventory timings gave a combined control median of
1.8022 ms and candidate median of 1.8240 ms. The candidate was 1.21% slower.
The added SIMD lane divergence cost more than the zero-parameter arithmetic it
removed, so this candidate was also removed.

## Decision

Keep the existing early force-only asynchronous submission, MLX scalar neighbor
admission, and branch-free Lennard-Jones arithmetic. Do not optimize the
neighbor displacement check based on its inclusive wall counter, and do not
move the force submission later in the step.

The next direct-force experiment must reduce work that every charged pair
performs. The screened-Coulomb `erfc` and Gaussian terms are the remaining
per-pair arithmetic target. Any approximation must first pass fixed-input
force RMS and maximum-error gates, followed by equal-rebuild sustained JAC
wall time. The reciprocal PME route is the fallback target if no accurate
direct-Coulomb reduction clears those gates.

## Raw evidence

Local, gitignored reports are under:

- `results/md-neighbor-attribution/`
- `results/md-velocity-submit/`
- `results/md-velocity-submit-late/`

Control runs used detached commit `5b7bc13`. The admitted production protocol
used the 5.5 Angstrom skin, one-step neighbor checks, seed 17, ten warmups,
final-only sampling and diagnostics, and the spatial tile backend.
