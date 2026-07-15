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
  neighbor manager's `auto` policy above the dense-pair atom limit. It keeps the
  periodic cell/bin search space bounded, batches neighboring-cell candidates
  into MLX distance-filter chunks, and emits compact pairs through CPU
  compaction.
- `mlx_cell_blocks` keeps the periodic cell/bin candidate search in a
  fixed-shape block representation. Production PME selects this backend so LJ
  and direct-space Coulomb share `NeighborBlocks` without materializing a dense
  topology pair cache.
- `python_neighbor` means the Python/NumPy cell-list builder is included in the
  benchmark before MLX pair evaluation.
- `auto` uses dense MLX when no pair list is supplied and the dense memory
  estimate fits the configured budget; otherwise it falls back to tiled MLX.
  For neighbor-list managers, `auto` selects `mlx_dense_pairs` for supported
  small systems and `mlx_cell_pairs` above the small-system limit. The
  production PME runner explicitly selects `mlx_cell_blocks`.

## Current Hot-Path Recommendation

Do not write a custom Metal kernel first. The first decision point should come
from `python -m mlx_atomistic.benchmarks.md_acceleration --json`.

The likely candidates are:

- dense/tiled pair evaluation, if MLX force accumulation dominates runtime;
- neighbor-list construction, if `python_neighbor` is much slower than
  `mlx_pairs`;
- DFT projector or solver work, if MD dense/tiled already scales well enough.

For the current code shape, the most suspicious path is still neighbor-list
construction: it uses Python dictionaries, Python sets, and NumPy loops. Dense
MLX all-pairs should be measured first because the target machine has enough
memory for thousands of particles.

For GPCRmd-scale periodic systems, dense all-pairs is not viable. The current
large-system route uses `mlx_cell_pairs`: CPU-side periodic binning bounds the
candidate space, MLX evaluates batched neighboring-cell distance filters, and
the backend emits compact pairs for the existing MLX pair force path. A
92,001-atom GPCRmd 729 short-range proof run under
`/tmp/mlx-atomistic-gpcrmd-729-mlx-cell-pairs` selected
`nonbonded_runtime.backend=mlx_cell_pairs` with no fallback. It tested
`candidate_count=390934237` local candidates, emitted `pair_count=48933140`,
and recorded `elapsed_wall_seconds=23.05953904206399`,
`neighbor_rebuild_wall_seconds=1.3890800829976797`,
`force_evaluation_wall_seconds=11.277963876025751`, `skin=2.5`, and one
rebuild. Direct rebuild microbenchmarks on 512-8192 particle FCC systems showed
the batched `mlx_cell_pairs` builder matching `periodic_cell_list` pairs and
running about 2.5-3x faster than the CPU builder. The under-5-second GPCRmd
stretch target remains blocked by force-evaluation cost, import/orchestration
overhead, and remaining CPU dynamic compaction.

A native pair emitter would benefit more than the GPCRmd notebook. It would
accelerate any cutoff-based periodic run that currently pays CPU-side cost for
compact neighbor-pair construction, including solvated proteins,
protein-ligand systems, water boxes, membranes, coarse-grained systems, and
benchmarks that use explicit pair lists. The difficult part is dynamic pair-list
emission: MLX can evaluate dense distance masks and pair math well, but the
runtime does not currently expose a NumPy-style one-argument `where(mask)` or
`nonzero(mask)` compaction API for emitting variable-length `(i, j)` pairs.
`mlx_cell_pairs` is the current pragmatic hybrid: bounded cell candidates and
MLX distance filtering, with dynamic compaction still CPU-side.

A fresh synthetic orthorhombic parity ladder now validates this route at
1k/4k/16k/50k/92,001 atoms against the tiled all-pairs MLX oracle. At 92,001
atoms, the compact build took 0.545 s, the explicitly synchronized pair-force
evaluation took 0.068 s, and the tiled oracle took 112.1 s; relative energy
delta was `4.56e-7` and maximum absolute force delta was `8.49e-7`. The result
is diagnostic rather than a GPCRmd production claim because the local
real-fixture cache was unavailable. Charged fixed-cell PME now has a separate
94,232-atom JAC validation using `mlx_cell_blocks`; that result does not convert
the synthetic neighbor row into a GPCRmd run. See
[`docs/benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md`](../benchmarks/scalable-neighbor-nonbonded-runtime-m5max.md)
and
[`docs/benchmarks/scalable-charged-pme-runtime-m5max.md`](../benchmarks/scalable-charged-pme-runtime-m5max.md).

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
