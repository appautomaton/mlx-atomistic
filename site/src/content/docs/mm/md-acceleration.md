---
title: "MLX-First MD Acceleration"
---


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
- `mlx_cell_tiles` spatially sorts atoms into eight-atom blocks, retains exact
  cutoff-plus-skin membership masks over non-empty 8x8 tiles, and groups tiles
  sharing a left block for the prepared orthorhombic PME force route. It keeps
  compact pairs as a diagnostic oracle during this development stage.
- `python_neighbor` means the Python/NumPy cell-list builder is included in the
  benchmark before MLX pair evaluation.
- `auto` uses dense MLX when no pair list is supplied and the dense memory
  estimate fits the configured budget; otherwise it falls back to tiled MLX.
  For neighbor-list managers, `auto` selects `mlx_dense_pairs` for supported
  small systems and `mlx_cell_pairs` above the small-system limit. The
  charged-PME performance runner explicitly selects `mlx_cell_tiles`.
  Production fixed-cell PME also selects tiles only inside the measured Metal
  envelope: 90,000--100,000 atoms, order-5 assignment, 9 A cutoff, 5.5 A skin,
  orthorhombic cell, and no NBFIX. Every other PME case keeps compact pairs.

## Current Hot-Path Recommendation

Measure with `python -m mlx_atomistic.benchmarks.md_acceleration --json` before
changing a force path. The large-system neighbor emitter and recurring direct
force path are now Metal-native. The first canonical-ID 8x8 atom-tile
implementation was parity-correct but slower end to end and was removed. The
later retained route instead creates blocks from spatial order and groups
same-left tiles, while preserving canonical atom indices at force scatter.

The current production path also uses specialized rigid-water/constraint
kernels and one fused standard-bonded force dispatch. The charged-PME
development runner now selects spatial tiles inside its narrow supported
envelope and fails its profile gate on a compact-pair fallback. The production
runner uses the same conservative envelope and records the selected backend in
checkpoints; resume pins that backend rather than silently changing the force
representation. General
large-system execution remains pair-oriented until the tile envelope is
broadened deliberately. The next performance work should reduce remaining
direct atomics and host-controlled rebuild work rather than add another wrapper
around the same 8x8 schedule.

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
5.60 GB process peak. At that point, the provisional one-order-of-magnitude
target still required a stable complete time at or below 16.8615 seconds and
had not yet been demonstrated.

### Complete-step pipeline result

The 2026-08-01 pipeline pass used the same 94,232-atom JAC 2x2x1 system,
9 A cutoff, 128x128x64 PME mesh, 4 fs timestep, 300 K target, 1/ps friction,
ten warmup steps, and 75 measured steps throughout. It retained two changes:

1. The 89,160 constraints are partitioned once into 28,092 exact three-site
   SETTLE waters and 3,160 disjoint SHAKE clusters. The recurring position and
   velocity projections use their specialized batched Metal kernels; no JAC
   constraint remains on the generic path.
2. Ordinary force-only steps combine the standard bond, angle, periodic
   torsion, and improper families into one atomic Metal output. This removes
   four dense 94,232x3 force outputs, their four outer additions, and the
   equivalent of 13 separate scatter stages. Energy, virial, and component
   diagnostics still use the original independently tested force terms.

| Retained state | 75-step wall time | Change from corrected control |
| --- | ---: | ---: |
| Corrected pipeline control | 1.588653 s | baseline |
| Specialized constraints | 1.142599--1.190142 s | 25.1--28.1% lower |
| Constraints plus fused bonded forces, two-run median | 1.003149 s | 36.9% lower |
| Matched OpenMM/OpenCL, single precision | 0.102947 s | reference |

The final MLX/OpenMM ratio is `9.7586x`, so this named workload now meets the
one-order-of-magnitude stretch target, narrowly. The manifest-bound comparison
matches atom and force-term inventory, cell, PME method/cutoff/alpha/mesh,
constraints, fixed-cell Langevin-middle protocol, timestep, warmups, measured
steps, and timing boundary. Both engines include explicit final-device
completion in the timer and exclude initialization and I/O. Their random-number
implementations differ, so this is matched protocol throughput rather than
trajectory identity.

The two fused MLX samples were 1.001684 and 1.004613 seconds. The latter had a
`3.21e-5 A` maximum constraint residual and a 3.86 GB process-tree peak. A
fixed-coordinate production-force comparison gave `1.90e-4 kJ/mol/A` RMS and
`0.01692 kJ/mol/A` maximum absolute force delta against a
`437.69 kJ/mol/A` maximum reference force, consistent with the existing
float32 atomic-scatter envelope. The OpenMM process tree peaked at 0.401 GB.

Two attractive-looking experiments were removed after complete-step tests:

- The atom-tile direct kernel improved isolated force time from 0.005970 to
  0.004718 seconds, but its padded representation increased the full 75-step
  run from 1.190142 to 1.506723 seconds and raised peak memory from about 3.90
  to 6.06 GB.
- A constraint-aware compiled trajectory block reduced neighbor rebuilding,
  but repeated a much larger lazy graph. Even after rebuilds fell from nine to
  two, it took 1.87596 seconds versus the 1.190142-second control.

A synchronized post-constraint profile assigned 35.91% of wall time to direct
LJ plus screened Coulomb, 14.48% to the then-unfused bonded terms, 12.12% to
neighbor update/rebuild, 8.53% to constraints, 7.36% to force aggregation,
6.16% to diagnostics, 6.06% to reciprocal PME, and 4.34% to PME corrections.
The fused bonded route addressed the second item. The next pass therefore
changed the direct-space work layout instead of adding another Python-loop
cleanup.

### Retained spatial-direct increment

The retained 2026-08-01 development candidate is structurally different from
the rejected atom-tile experiment below. It builds spatially sorted eight-atom
blocks, evaluates exact cutoff membership for each 8x8 block pair on Metal,
and compacts only accepted tiles. Accepted tiles are sorted by their left block
and scheduled in groups of up to four, allowing one threadgroup to reuse the
left atom coordinates and parameters and to reduce its atomic force writes.
Small cell-count and task metadata still cross the host boundary during a
rebuild. The existing exact compact-pair inventory remains the correctness
oracle.

The same pass combines sparse PME exclusions, Coulomb exceptions, 1-4 terms,
and LJ exceptions into one Metal force buffer. This is a force-only production
optimization; diagnostic energy and component paths remain independently
observable.

| Bounded development check | Before | Retained candidate | Change |
| --- | ---: | ---: | ---: |
| Isolated direct-space force | 5.52227 ms | 3.79931 ms | 31.20% lower |
| Sparse PME correction force | 0.568062 ms | 0.262083 ms | 53.86% lower |
| Complete 75-step trajectory, median | 0.933094 s | 0.601954 s | 35.49% lower |

The spatial inventory contains 60,502,167 exact pairs in 2,751,445 tiles, or
176,092,480 padded lanes with 34.36% occupancy. Direct-force agreement with the
compact-pair route was `6.32e-5 kJ/mol/A` RMS and `6.71e-4 kJ/mol/A` maximum;
the fused correction comparison was `4.68e-5 kJ/mol/A` RMS and
`3.05e-4 kJ/mol/A` maximum. The complete trajectory remained finite, reused
its PME plan, ended below a `3.51e-5 A` maximum constraint residual, and peaked
at 5.69 GB across the process tree with a passing late-memory plateau check.
The current development representation intentionally retains the 484 MB exact
pair array as a diagnostic oracle; tile geometry, topology masks, and that
oracle total about 578 MB of persistent indexed state. Per-pair LJ scales are
no longer constructed for an admitted tile route.

The complete gate used the position-balanced order pairs, tiles, tiles, pairs.
The pair samples were 1.045140 and 0.821048 seconds; the tile samples were
0.577860 and 0.626048 seconds. This clears the 0.85-second development target,
but the candidate does not replace the existing manifest-bound 1.003149-second
MLX result or establish a new OpenMM ratio. The main remaining costs are
direct-space work, neighbor update/rebuild, constraints, reciprocal PME, force
aggregation, and Python-side launch orchestration.

### Production routing follow-up

A fresh raw 75-step bundle on 2026-08-01 reproduced the retained route at a
0.558220-second tile median versus a 0.859394-second compact-pair control
median, 35.04% lower. All force, inventory, constraint, finite-state, PME-plan,
and 40 GB memory gates passed. The recurring order-5 reciprocal path now omits
particle-energy output and reduction during force-only steps. Its synchronized
route time fell from 0.081776 to 0.067008 seconds across 74 calls, 18.06%, while
complete clean wall remained statistically flat at 0.560536 seconds.

Two bounded direct-force experiments were rejected. Raising the same-left tile
group width from four to eight improved the isolated tile kernel by 5.53% but
changed complete wall from 0.558220 to 0.559019 seconds. A two-pass non-atomic
right-block reducer produced a 264 MB temporary tile-force buffer, reduced the
direct advantage over pairs to 8.52%, and raised complete wall to 0.734520
seconds. Both experiments were removed. The retained tile route is now selected
by production only inside the measured envelope above; compact pairs remain the
fallback, and checkpoint resume preserves the originally recorded backend.

### Rejected atom-tile result

The 2026-07-31 canonical-ID atom-tile experiment tested the full route instead
of assuming that a faster kernel would make the trajectory faster. Its tiles
were derived from the compact-pair list, unlike the retained spatially built
route above. Its isolated 8x8 Metal
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
candidate tile kernel, production routing, experimental selector, and metadata
were removed. Exact compact-pair ownership remains the verified correctness
foundation for the later spatial route.

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
[`docs/benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md`](../benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md)
and
[`docs/benchmarks/scalable-charged-pme-runtime-m5max.md`](../benchmarks/scalable-charged-pme-runtime-m5max.md),
and
[`docs/benchmarks/gpcrmd-729-pme-runtime-m5max.md`](../benchmarks/gpcrmd-729-pme-runtime-m5max.md).

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
