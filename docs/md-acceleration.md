# MLX-First MD Acceleration

This milestone keeps Python as the user-facing API but moves the MD nonbonded
hot path into MLX arrays and focused custom Metal kernels. The current priority
is recurring fixed-cell PME throughput on Apple Silicon while preserving the
existing force, constraint, and restart validation gates.

## Backends

- `mlx_dense` evaluates all pair interactions with dense MLX arrays. This is a
  serious baseline on M-series machines with large unified memory.
- `mlx_tiled` evaluates row blocks against all particles, reducing peak memory
  while keeping the pair math in MLX.
- `mlx_pairs` evaluates an explicit pair list with MLX gather/scatter. This is
  the path used with prebuilt neighbor lists.
- `mlx_dense_pairs` is a small-system neighbor-list backend selected by the
  neighbor manager's `auto` policy. It evaluates the dense periodic distance
  mask in MLX, then records explicit CPU `argwhere` compaction metadata because
  this MLX runtime does not expose dynamic `argwhere`/`nonzero` pair emission.
- `mlx_cell_pairs` is the large-system neighbor-list backend selected by the
  neighbor manager's `auto` policy above the dense-pair atom limit. On Metal it
  bins and stably orders atoms in MLX, prunes a fine cell grid by bounding-box
  distance, emits bounded candidate batches with a custom Metal kernel, and
  compacts exact pairs by prefix scan. Only small cell-occupancy and task
  metadata reaches the CPU; coordinates and compact pairs remain in MLX. The
  CPU backend retains the established NumPy cell-list implementation.
- `mlx_cell_blocks` keeps the periodic cell/bin candidate search in a
  fixed-shape block representation. It remains available for fixed-shape
  consumers and historical benchmark reproduction.
- `python_neighbor` means the Python/NumPy cell-list builder is included in the
  benchmark before MLX pair evaluation.
- `auto` uses dense MLX when no pair list is supplied and the dense memory
  estimate fits the configured budget; otherwise it falls back to tiled MLX.
  For neighbor-list managers, `auto` selects `mlx_dense_pairs` for supported
  small systems and `mlx_cell_pairs` above the small-system limit. The
  production PME runner explicitly selects `mlx_cell_pairs`.

## Current Hot-Path Recommendation

Measure with `python -m mlx_atomistic.benchmarks.md_acceleration --json` before
changing a force path. The large-system neighbor emitter and recurring direct
force path are now Metal-native. Production remains on compact explicit pairs.
The first exact 8x8 atom-tile implementation was parity-correct but slower end
to end and has been removed; it is evidence for a different future layout, not
an alternate production backend.

The remaining candidates, in priority order, are:

- a spatially coherent direct-force layout that avoids both the global compact-
  pair scan and the rejected 8x8 tile route's padded-lane expansion;
- fused rigid-water position and velocity projection, after an independent
  parity gate for the full SETTLE update;
- further neighbor rebuild work only when a matched trajectory shows it beats
  the retained fixed-cell topology cache.

For GPCRmd-scale periodic systems, dense all-pairs is not viable. The current
large-system route uses `mlx_cell_pairs`. The historical implementation used
CPU-side periodic bins and candidate arrays; the current Metal route uses a
fine device-resident grid and an AABB-pruned canonical half-stencil, then emits
and compacts candidates on Metal for the existing MLX pair force path. A
92,001-atom GPCRmd 729 short-range proof run under
`/tmp/mlx-atomistic-gpcrmd-729-mlx-cell-pairs` selected
`nonbonded_runtime.backend=mlx_cell_pairs` with no fallback. It tested
`candidate_count=390934237` local candidates, emitted `pair_count=48933140`,
and recorded `elapsed_wall_seconds=23.05953904206399`,
`neighbor_rebuild_wall_seconds=1.3890800829976797`,
`force_evaluation_wall_seconds=11.277963876025751`, `skin=2.5`, and one
rebuild. Those figures describe the historical hybrid implementation. The
under-5-second GPCRmd stretch target remains blocked by force evaluation,
per-step synchronization, and the cost of rebuilding and consuming a global
explicit pair array. Dynamic compaction is no longer a CPU bottleneck on Metal.

### Spatial Metal neighbor result

The retained 2026-07-31 JAC fixed-cell PME ladder used the same 9 A cutoff,
5.5 A skin, ten warmup steps, and 750 measured steps as its immediate control.
These are one-time engineering measurements on an M5 Max, not a CI gate.

| Atoms | Prior MLX | Spatial MLX | Time reduction | OpenMM artifact | Peak process memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23,558 | 12.6688 s | 8.7643 s | 30.8% | 0.6810 s | 1.76 GB |
| 47,116 | 20.5108 s | 13.9166 s | 32.1% | 1.0945 s | 2.76 GB |
| 94,232 | 40.4986 s | 24.3424 s | 39.9% | 1.6861 s | 4.80 GB |

At 94,232 atoms the current 25x25x12 grid emits 184.1 million candidates,
down 60.7% from 468.3 million, and retains about 60.5 million exact pairs. All
three runs were finite, passed the existing constraint behavior, stayed below
40 GB, and had passing late-memory plateaus. The result is a substantial
retained gain but does not meet the one-order-of-magnitude OpenMM stretch
target. The next large-system lever is spatial tiles or cell-pair tasks that
feed the direct-space force kernel without materializing and repeatedly
scanning the full explicit pair array.

The OpenMM values in this historical ladder are provisional context. The local
artifact does not bind integration, timestep, thermostat, constraints,
completion, timing boundary, and workload identity strongly enough for a
publishable cross-engine ratio. Its 1.6861-second value therefore defines only
the provisional 16.8615-second ten-times stretch target.

### Prepared recurring-force result

A later fixed-cell pass retained four additional changes without changing the
cutoff, PME mesh, timestep, thermostat, or constraint tolerances:

1. Fixed cell-pair topology is cached while dynamic occupancies and exact pair
   membership are recomputed at every rebuild.
2. Box reciprocals and Lorentz-Berthelot particle invariants are prepared once
   for the recurring direct-space kernel.
3. The first of two mathematically redundant SETTLE velocity projections is
   skipped only for disjoint rigid-water groups; mixed or overlapping
   constraints retain the full projection.
4. Each prepared Metal worker processes eight adjacent compact pairs and
   accumulates repeated left-atom forces in registers before global atomics.

The fixed-topology cache first reduced the complete 94,232-atom, 750-step run
from 24.3424 to 19.1477 seconds. The retained prepared-force and constraint
changes then produced a complete 18.0356-second run: 25.9% below the original
spatial result, with a 3.43 GB peak. This is the latest stable complete number
for the retained stack before the eight-pair worker change. No formal
MLX/OpenMM ratio is reported because the OpenMM artifact did not pass the
current comparability manifest gate.

Because later complete samples were thermally throttled, the eight-pair worker
was admitted with position-balanced A/B evidence instead of an unmatched full
run. On the 60.5-million-pair 94,232-atom list it reduced the isolated direct
kernel by about 25%. Alternating 75-step trajectories reduced median measured
time from 5.1243 to 4.8785 seconds (4.8%) while keeping constraint residuals
near 1.5e-5 A. A bounded 92,001-atom GPCRmd 729 run also passed all finite-state,
constraint, HMR, PME-plan reuse, trajectory, and checkpoint checks with a
5.60 GB process peak. The provisional one-order-of-magnitude target requires
a stable complete time at or below 16.8615 seconds and has not yet been
demonstrated.

### Rejected atom-tile result

The 2026-07-31 atom-tile experiment tested the full route instead of assuming
that a faster kernel would make the trajectory faster. Its isolated 8x8 Metal
kernel reduced median direct-force latency from 0.005548 to 0.004257 seconds,
a 30.33% win, and matched the explicit-pair force within the established
float32 tolerance. Initial integration lost badly because tile construction
and CPU topology binding raised the 94,232-atom, 75-step wall time from the
1.850210-second explicit-pair control to 34.429230 seconds.

One bounded lifecycle correction then derived tiles from the exact compact-
pair oracle and packed topology masks on Metal. It reduced the atom-tile wall
time from 34.821962 to 2.963867 seconds, geometry construction from 7.933098 to
0.280091 seconds, and topology binding from 24.648279 to 0.012191 seconds. All
finite-state, constraint, neighbor, PME-reuse, route, and no-fallback checks
passed. The corrected route nevertheless remained 60.19% slower than explicit
pairs and increased the short-run process-tree peak from 4.17 to 8.06 GB. Its
9,664,362 tiles represented 60,504,316 exact pairs through 618,519,168 padded
lanes, so the construction fix did not make the complete representation
competitive.

The conditional repeat and 750-step proof were not run because the first exact
comparison was decisive and projected about 29.6 seconds for 750 steps, above
both the retained complete result and the 16.8615-second stretch target. The
tile kernel, production routing, experimental selector, and candidate-only
metadata were removed. Exact compact-pair ownership remains as the verified
production foundation.

A fresh synthetic orthorhombic parity ladder now validates this route at
1k/4k/16k/50k/92,001 atoms against the tiled all-pairs MLX oracle. At 92,001
atoms, the compact build took 0.545 s, the explicitly synchronized pair-force
evaluation took 0.068 s, and the tiled oracle took 112.1 s; relative energy
delta was `4.56e-7` and maximum absolute force delta was `8.49e-7`. This
2026-07-13 result remains diagnostic because the local real-fixture cache was
unavailable for that measurement. A later source-backed GPCRmd 729 run now
passes a separate bounded fixed-cell parity/NVT/restart gate using
`mlx_cell_blocks`/`NeighborBlocks`; it does not change the classification of the
synthetic neighbor row. The production runner has since moved to compact
`mlx_cell_pairs`. Retained NPT diagnostic reuse, reciprocal-PME graph
compilation, and fused parameterized LJ/direct-PME Metal kernels then reduced a
matched 75-step DHFR NPT prefix from 142.87 to a repeated median of 13.77
seconds. Process-tree peak memory fell from 27.33 GB to 5.18--6.11 GB across
the retained samples, and the same numerical gates passed. Order-five
reciprocal PME now also uses dedicated Metal charge-spread and
potential-derivative interpolation kernels, reducing the 2,269-atom alanine
50-step fixed-cell median from 0.853 to 0.537 seconds without pressure
diagnostics and from 1.313 to 0.987 seconds with analytic pressure diagnostics.
The artifact loader now also routes complete three-site water constraint
triangles to a batched MLX rigid-water projector while preserving a fail-closed
generic path for incomplete or mixed geometries. That reduced the same
fixed-cell median from 0.537 to 0.419 seconds and the analytic-pressure median
from 0.987 to 0.863 seconds. The complete 100-step NVT plus 1,000-step NPT
alanine check then passed all 16 unchanged science gates in 15.899 seconds,
down from 23.110 seconds, with a `3.34e-6` A maximum constraint error and a
0.94 GB process-tree peak. Fixed-coordinate OpenMM parity remained within the
accepted energy and force gates. These measurements were made on an M5 Max in
low-power mode. This is bounded optimization and one-picosecond stability
evidence, not a production-length validation. Charged fixed-cell PME also has
a separate 94,232-atom JAC validation. See
[`docs/benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md`](./benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md)
and
[`docs/benchmarks/scalable-charged-pme-runtime-m5max.md`](./benchmarks/scalable-charged-pme-runtime-m5max.md),
and
[`docs/benchmarks/gpcrmd-729-pme-runtime-m5max.md`](./benchmarks/gpcrmd-729-pme-runtime-m5max.md).

For the active solvated ligand-receptor notebook system, the near-term GPU
occupancy lever is independent replica batching. The system is only a few
hundred atoms, so one trajectory does not expose much work to the GPU. Use the
prep APIs to advance multiple physically independent velocity seeds in one MLX
loop:

```python
from mlx_atomistic.prep.replicas import run_ligand_receptor_replicas

run_ligand_receptor_replicas(
    "notebooks/ligand-receptor-motion/data/mlx-real-md/example-200ps-r4",
    replicas=4,
    selected_replica=0,
    steps=200000,
    dt=0.001,
    sample_interval=100,
)
```

For repeatable profiling across durations and replica counts:

```python
from mlx_atomistic.prep.replicas import profile_ligand_receptor_performance

profile_ligand_receptor_performance(
    "notebooks/ligand-receptor-motion/data/perf/replica-profile",
    durations_ps=[5, 50, 200],
    replica_counts=[1, 4, 8, 16],
    dt=0.001,
    sample_interval=100,
)
```

The profile reports wall time, per-replica and aggregate steps/s, aggregate
ps/s, GPU-visible atoms, dense pair slots, force-evaluation cost, constraint
projection cost, diagnostic cost, max constraint error, and artifact size.

## Interpreting The Benchmark

The benchmark reports `ms_per_eval`, `neighbor_rebuild_ms_per_eval`,
`force_eval_ms_per_eval`, an estimated dense memory footprint,
`ns_per_day_at_dt_0_002`, and force/energy deltas relative to dense MLX. The
`ns_per_day` number is only a throughput-style indicator because the MD engine
uses reduced internal units. Use the separated rebuild and force timings when
deciding whether the current bottleneck is pair construction or pair evaluation.
