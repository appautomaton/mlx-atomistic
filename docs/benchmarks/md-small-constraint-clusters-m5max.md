# Small Constraint-Cluster Metal Kernel on M5 Max

## Verdict

Retain the small-component constraint route. On the 94,232-atom JAC workload,
the production-length two-run median fell from 11.7073 seconds to 5.5895
seconds for 750 measured steps, a 52.3% complete-wall reduction. The candidate
kept all 89,160 distance constraints on-device and passed the runtime admission
checks.

The route applies only when every connected constraint component contains at
most four atoms and three constrained pairs. Larger or unsupported graphs keep
the existing MLX SHAKE/RATTLE fallback. The 5DFR workload already uses its
specialized SETTLE and central-atom SHAKE kernels, so this change does not alter
that force/integration path.

## Design

The previous generic route issued 20 full-array Jacobi iterations for each
position projection and each velocity projection. Molecular dynamics invokes
one position and two velocity projections per step, so JAC queued roughly 60
large gather/scatter layers every step.

The retained Metal kernels assign one thread to each independent component.
That thread loads up to four atoms and three pairs into fixed local arrays,
runs the same iteration equations locally, and emits one delta per owned atom.
A schedule built once at `DistanceConstraints` construction maps those deltas
back to canonical atom indices. No new package or runtime dependency is
required.

The JAC constraint graph contains 31,252 connected components:

| Component size | Count |
| --- | ---: |
| 2 atoms | 1,832 |
| 3 atoms | 29,024 |
| 4 atoms | 396 |

All components contain at most three edges, so the complete graph is eligible.

## Isolated Kernel Result

The isolated interleaved benchmark compares the retained kernels with the
unchanged generic MLX equations on the same prepared JAC state.

| Operation | Generic median | Metal cluster median | Reduction |
| --- | ---: | ---: | ---: |
| Position projection | 2.4747 ms | 0.3473 ms | 86.0% |
| Velocity projection | 1.6219 ms | 0.2954 ms | 81.8% |

Position output differed by `3.54e-6 A` root-mean-square and `4.58e-5 A`
maximum. Velocity output differed by `4.94e-8 A/ps` root-mean-square and
`8.34e-7 A/ps` maximum. The candidate's maximum position-constraint residual
was `3.29e-5 A`, below the generic result on that predicted state.

Raw result:
`results/interaction-engine-v2/small-constraints-jac.json`.

## Complete Molecular-Dynamics Result

Each row is a two-run median from a position-balanced control/candidate/
candidate/control sequence. The control source is commit `4da45d5`; both
variants use the same prepared input, seed, 4 fs step, 9 A cutoff, 5.5 A skin,
order-five PME, and fixed orthorhombic cell.

| Workload | Steps | Control | Candidate | Measured change |
| --- | ---: | ---: | ---: | ---: |
| JAC, 94,232 atoms | 75 | 0.9786 s | 0.3513 s | 64.1% lower |
| JAC, 94,232 atoms | 750 | 11.7073 s | 5.5895 s | 52.3% lower |
| 5DFR, 23,558 atoms | 75 | 0.1348 s | 0.0968 s | non-regression only |
| 5DFR, 23,558 atoms | 750 | 1.5170 s | 1.3786 s | non-regression only |

The 5DFR measurements must not be credited as a kernel gain because that
workload remains on the unchanged specialized constraint route. They establish
that the new schedule and route inventory do not regress that production path.

For the JAC 750-step runs, every candidate completed 22 neighbor rebuilds,
remained finite, reused one PME plan, avoided neighbor fallback, and passed the
new constraint-route admission check. Candidate maximum constraint residuals
were `4.60e-5` to `4.75e-5 A`; control residuals were about `5.00e-5 A`.
Candidate peak MLX memory was 1,143.6--1,144.6 MiB versus 1,140.5 MiB for the
control, an increase below 0.4%.

Raw results live under:

- `results/small-constraint-clusters/jac-75/`
- `results/small-constraint-clusters/jac-750/`
- `results/small-constraint-clusters/5dfr-75/`
- `results/small-constraint-clusters/5dfr-750/`

## Synchronized Profile

A 20-step synchronized JAC profile fell from 249.75 ms to 122.31 ms, a 51.0%
wall reduction. Constraint route totals changed as follows:

| Route | Generic | Cluster kernel |
| --- | ---: | ---: |
| Position | 67.30 ms | 9.67 ms |
| Pre-force velocity | 35.69 ms | 6.81 ms |
| Final velocity | 37.83 ms | 6.30 ms |

Raw profiles:
`results/interaction-engine-v2/profile-jac-20.json` and
`results/interaction-engine-v2/profile-jac-small-clusters-20.json`.

## Reproducer

Run Metal measurements outside a restricted sandbox:

```bash
uv run python scripts/benchmark_small_constraint_clusters.py \
  results/scalable-charged-pme-runtime/jac-2x2x1/prepared \
  --warmups 3 --samples 10 --block-count 4 \
  --out results/interaction-engine-v2/small-constraints-jac.json
```

The complete-step runs use the existing charged-PME runtime command with the
same prepared directory, `--warmups 10`, and either `--steps 75` or
`--steps 750`.

## Provenance

- Date: 2026-08-14
- Engine: `mlx_atomistic`
- Runtime: MLX 0.31.2, Python 3.13.12
- Platform: Metal on Apple M5 Max, macOS 26.5.2
- Control commit: `4da45d5`
- Raw outputs: gitignored `results/`

Two investigated alternatives were not retained. A neighbor-list-free fixed
cell-pair direct kernel was 43--69% slower and exceeded the force-parity
envelope. Spatially reordering PME atom spread/interpolation was flat to slower
on JAC. Their production code was removed rather than carried as dormant
complexity.
