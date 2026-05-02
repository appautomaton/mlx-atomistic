# MLX-First MD Acceleration

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
- `python_neighbor` means the Python/NumPy cell-list builder is included in the
  benchmark before MLX pair evaluation.
- `auto` uses dense MLX when no pair list is supplied and the dense memory
  estimate fits the configured budget; otherwise it falls back to tiled MLX.

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
large-system route uses compact periodic cell-list pairs built on the CPU and
then evaluates pair interactions through MLX. The latest 92,001-atom GPCRmd 729
proof artifact uses `periodic_cell_list`, `skin=2.5`, one neighbor rebuild over
10 integration steps, scalar unit pair scales when no 1-4 scaling is active, and
writes 11 sampled frames in about 13.9 seconds.

A native or hybrid MLX pair emitter would benefit more than the GPCRmd notebook.
It would accelerate any cutoff-based periodic run that currently pays CPU-side
cost for compact neighbor-pair construction, including solvated proteins,
protein-ligand systems, water boxes, membranes, coarse-grained systems, and
benchmarks that use explicit pair lists. The difficult part is dynamic pair-list
emission: MLX can evaluate dense distance masks and pair math well, but the
runtime does not currently expose a NumPy-style one-argument `where(mask)` or
`nonzero(mask)` compaction API for emitting variable-length `(i, j)` pairs.
Until that exists, a pragmatic target is a hybrid path that does more candidate
distance testing in MLX while keeping bounded compaction and topology filtering
CPU-side.

For the active solvated ligand-receptor notebook system, the near-term GPU
occupancy lever is independent replica batching. The system is only a few
hundred atoms, so one trajectory does not expose much work to the GPU. Use the
prep CLI to advance multiple physically independent velocity seeds in one MLX
loop:

```bash
uv run atomistic-prep run-ligand-receptor-replicas \
  --out notebooks/ligand-receptor-motion/data/mlx-real-md/example-200ps-r4 \
  --replicas 4 \
  --selected-replica 0 \
  --steps 200000 \
  --dt 0.001 \
  --sample-interval 100
```

For repeatable profiling across durations and replica counts:

```bash
uv run atomistic-prep profile-ligand-receptor-performance \
  --out notebooks/ligand-receptor-motion/data/perf/replica-profile \
  --durations-ps 5 50 200 \
  --replicas 1 4 8 16 \
  --dt 0.001 \
  --sample-interval 100
```

The profile reports wall time, per-replica and aggregate steps/s, aggregate
ps/s, GPU-visible atoms, dense pair slots, force-evaluation cost, constraint
projection cost, diagnostic cost, max constraint error, and artifact size.

## Interpreting The Benchmark

The benchmark reports `ms_per_eval`, an estimated dense memory footprint,
`ns_per_day_at_dt_0_002`, and force/energy deltas relative to dense MLX. The
`ns_per_day` number is only a throughput-style indicator because the MD engine
uses reduced internal units.
