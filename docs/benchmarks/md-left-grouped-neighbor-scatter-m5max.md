# Adaptive Left-Grouped Neighbor Scatter on Apple M5 Max

Date: 2026-08-14

## Decision

Retain a hybrid exact-tile scatter for the Metal spatial Neighbor builder.
Small candidate inventories keep the existing compact-then-sort route. Large,
normally occupied inventories use two new Metal kernels that count and scatter
exact 4-by-4 tiles directly in final left-block order. The decision depends on
runtime inventory, not a workload name or atom count.

The retained boundary is:

- at least 3,000,000 coarse candidate tiles; and
- no more than eight coarse 8-atom blocks in any spatial cell.

The occupancy guard matters because one thread owns one cell in the new
kernels. It keeps pathologically concentrated cells on the more parallel
compact-and-sort route. Its maximum is included in the builder's existing
inventory reduction, so the guard adds no new host synchronization.

This work improves the device-resident schedule producer for the production
4-by-4 interaction tiles and 32-column force groups. It does not route the
runtime through the experimental 32-atom interaction engine.

## Root cause and implementation

The previous large-system builder first compacted every non-empty 4-by-4
subtile from the coarse 8-by-8 candidates, then applied a global
`mx.argsort(tile_blocks[:, 0])`. The sort was required because each recurring
Direct Space force group must own one left block, but it operated over more
than seven million exact tiles on the 92k-94k atom workloads.

The spatial cell-pair template is already ordered by left cell. The retained
large-inventory route reuses that structure entirely on device:

1. One Metal thread derives its left cell's task range with two binary searches
   over the existing cell-pair template.
2. The first kernel traverses the task's coarse candidates and counts non-empty
   top and bottom subtiles directly by final 4-atom left block.
3. MLX prefix-scans only that left-block count vector.
4. The second kernel repeats the same deterministic traversal and scatters
   exact tile blocks and membership masks directly into final left-block order.

The recurring force schedule and Direct Space consumer are unchanged. No
per-tile metadata is added, and candidate masks, task offsets, occupancy, and
final tile inventories remain device resident until the pre-existing sized
allocation boundaries.

## Boundary experiments

An unconditional left-grouped route was not general. On 5DFR, with about 1.07
million coarse candidate tiles, it changed complete step wall from 1.1287 ms to
1.1677 ms, a 3.46% regression. At the 47,116-atom JAC 2-by-1-by-1 midpoint,
about 2.29 million candidates, complete rebuild wall was effectively tied:
13.70 ms for control and 13.78 ms for the candidate. At the 94,232-atom JAC
2-by-2-by-1 point, about 4.72 million candidates, the new route became clearly
faster.

An earlier implementation materialized per-cell task ranges on the host. It
reduced exact scatter work, but raised the JAC host-task stage from 1.49 ms to
4.84 ms and made the complete rebuild slower. The retained kernels instead
derive those ranges by binary search on device.

The 3 million candidate crossover leaves margin above the measured 2.29
million tie. A separate concentrated-cell Metal test forces the candidate
threshold to zero and verifies that nine coarse blocks in one cell still select
compact-and-sort.

## Structural profiles

The opt-in synchronized profiler attributes the exact scatter and complete
rebuild. It is not used as the clean throughput result.

| Workload | Candidate tiles | Route | Exact scatter/order | Rebuild wall | Reduction |
| --- | ---: | --- | ---: | ---: | ---: |
| 5DFR control | 1.07M | compact + `argsort` | 1.072 ms | 7.351 ms | control |
| 5DFR hybrid | 1.07M | compact + `argsort` | 1.164 ms | 7.515 ms | boundary guard retained old route |
| JAC control | 4.71M | compact + `argsort` | 5.953 ms | 27.457 ms | control |
| JAC hybrid | 4.72M | left-grouped Metal | 2.971 ms | 23.541 ms | 50.10% stage, 14.26% rebuild |
| GPCRmd control | 4.53M | compact + `argsort` | 6.605 ms | 29.136 ms | control |
| GPCRmd hybrid | 4.53M | left-grouped Metal | 3.117 ms | 24.519 ms | 52.81% stage, 15.85% rebuild |

The final hybrid profiles report a maximum of three coarse blocks per cell for
all three workloads. They report `left_grouped_scatter=0` for 5DFR and `1` for
JAC and GPCRmd, directly recording the selected route.

Raw structural profiles are under
`results/md-suite/left-grouped-scatter/*-profile.json`.

## Interleaved complete-trajectory A/B

Each arm ran 10 warmup and 750 measured fixed-cell NVT steps with seed 17, a
4 fs timestep, 9 Angstrom cutoff, 5.5 Angstrom neighbor skin, Metal spatial
tiles, and boundary-only sampling and diagnostics. The order for every workload
was control 1, candidate 1, candidate 2, control 2, control 3, candidate 3.

| Workload | Atoms | Control median | Candidate median | Step-wall reduction | Throughput increase | Metal peak median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5DFR | 23,558 | 1.0881 ms | 1.0910 ms | -0.26% | -0.26% | 301.1 -> 301.3 MB |
| JAC 4-cell | 94,232 | 3.3983 ms | 3.2597 ms | 4.08% | 4.25% | 1265.8 -> 1073.4 MB |
| GPCRmd 729 | 92,001 | 3.9123 ms | 3.8200 ms | 2.36% | 2.42% | 1275.4 -> 1099.1 MB |

Every candidate sample was faster than every control sample on JAC and
GPCRmd. The 0.26% 5DFR change is treated as no material regression and remains
well inside the 3% cross-system gate. All 18 runs passed finite-state,
constraint, memory, fixed-cell, lazy-topology, Neighbor representation, and PME
plan-reuse checks. GPCRmd also exercises CHARMM NBFIX and CMAP force terms.

Raw clean outputs are under
`results/md-suite/left-grouped-scatter/final-ab/`.

## Correctness

The Neighbor CPU suite and the related force-runtime, forcefield, MD, and
nonbonded suites pass 133 tests. The complete fused Metal kernel suite passes
31 tests. Metal coverage explicitly forces the left-grouped path for periodic
tile membership, Direct Space force parity, force-column scheduling, and exact
pair materialization. A second test certifies the concentrated-cell fallback.

`ruff check src tests` passes, and the static API documentation generator emits
all 57 pages without a signature/docstring mismatch.

## Reproducer

One clean candidate arm was run with:

```bash
uv run python -m mlx_atomistic.benchmarks.charged_pme runtime \
  --prepared results/larger-system-scaling/jac-2x2x1-modern/prepared \
  --warmups 10 \
  --steps 750 \
  --dt-ps 0.004 \
  --temperature-k 300 \
  --seed 17 \
  --neighbor-skin 5.5 \
  --neighbor-check-interval 1 \
  --sample-interval 750 \
  --diagnostic-interval 750 \
  --neighbor-backend mlx_cell_tiles \
  --out results/md-suite/left-grouped-scatter/final-ab/jac-candidate-01.json
```

Add `--neighbor-rebuild-profile` to reproduce the structural attribution. The
control worktree was fixed at `3f281e0`; candidate runs used the same
environment and prepared artifacts with only this adaptive scatter change.
