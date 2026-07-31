---
title: "MLX-First MD Acceleration"
---


This milestone keeps Python as the user-facing API but moves the MD nonbonded
hot path toward MLX array execution. The current priority is to measure how far
dense and tiled MLX pair evaluation can go on Apple Silicon before introducing
custom Metal kernels.

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
changing a force path. The large-system neighbor emitter is now Metal-native,
so the next scaling decision is whether to replace the explicit global pair
array with a spatial tile representation consumed directly by the force
kernel.

The likely candidates are:

- dense/tiled pair evaluation, if MLX force accumulation dominates runtime;
- neighbor-list construction, if `python_neighbor` is much slower than
  `mlx_pairs`;
- DFT projector or solver work, if MD dense/tiled already scales well enough.

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

| Atoms | Prior MLX | Spatial MLX | Time reduction | OpenMM | OpenMM / MLX | Peak process memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23,558 | 12.6688 s | 8.7643 s | 30.8% | 0.6810 s | 12.9x | 1.76 GB |
| 47,116 | 20.5108 s | 13.9166 s | 32.1% | 1.0945 s | 12.7x | 2.76 GB |
| 94,232 | 40.4986 s | 24.3424 s | 39.9% | 1.6861 s | 14.4x | 4.80 GB |

At 94,232 atoms the current 25x25x12 grid emits 184.1 million candidates,
down 60.7% from 468.3 million, and retains about 60.5 million exact pairs. All
three runs were finite, passed the existing constraint behavior, stayed below
40 GB, and had passing late-memory plateaus. The result is a substantial
retained gain but does not meet the one-order-of-magnitude OpenMM stretch
target. The next large-system lever is spatial tiles or cell-pair tasks that
feed the direct-space force kernel without materializing and repeatedly
scanning the full explicit pair array.

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
