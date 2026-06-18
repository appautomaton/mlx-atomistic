# Same-Workload LJ Scaling: MLX vs OpenMM vs LAMMPS (M5 Max)

Date: 2026-06-18 (MLX re-measured after the `mlx_cell_pairs` neighbor-default
switch; OpenMM and LAMMPS reference rows are carried over unchanged from the
2026-06-17 run on the same machine — only the MLX runtime changed.)

This is the first **production-scale** same-workload throughput comparison of the
`mlx_atomistic` runtime against the OpenMM and LAMMPS reference engines on a
single Apple M5 Max. It exists because every prior committed MLX number was
smoke-sized (4–32 atoms, 1 step), where fixed overhead dominates and the real
O(N) force/neighbor costs are invisible. The goal here is an honest baseline, not
a favorable one.

## What is measured

A pure Lennard-Jones fluid (ε = σ = 1, cutoff 2.5σ) integrated under NVE/Langevin
at reduced density 0.8, swept across a particle-count ladder and run on each
engine's GPU path on the same machine. The comparison metric is **`steps_per_s`**
(integration steps per wall-second). Reduced-unit `ns/day` is intentionally not
compared across engines because the reduced timestep has no common physical
meaning.

## Method

| Engine | Path | Command surface | Geometry / units |
| --- | --- | --- | --- |
| `mlx_atomistic` | MLX/Metal GPU | `mlx_atomistic.benchmarks.md_performance` | `fcc_lattice`, reduced units, auto backend |
| OpenMM | OpenCL | `scripts/benchmark_openmm_opencl.py` | synthetic LJ, physical (nm/ps) units |
| LAMMPS | GPU/OpenCL (`lj/cut/gpu`) | `scripts/benchmark_lammps_opencl.py` | same `fcc_lattice`, reduced (`units lj`) |

- **Identical workload per size.** All engines run the same `(particles, steps)`
  at each ladder point; the aggregator (`same_workload_compare.build_scaling_summary`)
  only emits a ratio when particle and step counts match and both engines ran `ok`.
- **MLX and LAMMPS share geometry.** The LAMMPS script reuses the product's own
  `fcc_lattice` (same N, box, initial positions) at reduced density 0.8, so the
  MLX↔LAMMPS comparison is genuinely the same physics (same potential, cutoff,
  units). OpenMM runs the same particle/step counts in physical units and is a
  throughput reference, not a bit-identical workload.
- **MLX runs its own auto-selected backend per size**: dense all-pairs below the
  neighbor-list threshold (1536 atoms), and the `mlx_cell_pairs` neighbor backend
  (host-compacted real pairs) above it. This compares MLX at its best per size,
  not a single fixed path.
- **Per-size step counts.** Cheap small systems run many steps; expensive large
  systems run fewer, so each point reaches steady state without an unbounded run.
  Steps are matched across engines within each size.

### Two measurement corrections made for this comparison

1. **OpenMM async timing.** OpenCL enqueues integration kernels asynchronously, so
   `integrator.step(n)` can return before the GPU has done the work. The timed
   region now forces the queue to drain (`context.getState(getEnergy=True)`) so
   `wall_s` reflects real compute, not just kernel enqueue. Without this, short
   OpenMM runs reported absurd rates (e.g. ~280k steps/s at 60 steps).
2. **LAMMPS GPU verification.** The LAMMPS log is parsed to confirm the
   `lj/cut/gpu` pair build is actually active; if it cannot be confirmed the row
   is reported as a CPU-only `diagnostic` rather than claimed as OpenCL.

## Honest caveats

- This is a **throughput** comparison (steps/s), not a validation of identical
  trajectories. Neighbor strategy, thermostat, and integrator details differ
  across engines.
- On the Apple GPU, **both LAMMPS and MLX build neighbor lists on the host (CPU)**
  and run the pair force on the GPU (LAMMPS logs "GPU does not support neighbor
  lists on device, switching to host"). This is the documented neighbor-build
  cost, not a defect.
- MLX uses reduced-unit LJ; OpenMM uses physical units. Only `steps_per_s` is
  compared, and only where N and step counts match.

## Results

Measured on an Apple M5 Max (128 GB), reduced density 0.8, `dt = 0.002`, per-size
step counts (3000 / 2000 / 800 / 300). MLX and LAMMPS share the same `fcc_lattice`
geometry and reduced LJ units; OpenMM runs the same particle/step counts in
physical units. All four sizes are `comparable` and all MLX runs conserved energy
(see note below). The MLX rows are the current best per size: above the dense
threshold (1536 atoms) the neighbor backend is `mlx_cell_pairs` (compacted real
pairs) and the integrator runs the **compiled batched-block fast path**
(`block_size=16`, `neighbor_skin=1.2`) — `block_size` Langevin substeps per
compiled block with one host sync per block instead of per step.

| atoms | MLX steps/s | MLX backend | OpenMM steps/s | OpenMM/MLX | LAMMPS steps/s | LAMMPS/MLX |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000  | 1966.4 | `mlx_dense` | 24024.7 | 12.2× | 4358.1 | 2.2× |
| 4000  | 1354.4 | `mlx_cell_pairs` + batched | 4819.9 | 3.6× | 2165.1 | 1.6× |
| 16000 | 345.6  | `mlx_cell_pairs` + batched | 4171.0 | 12.1× | 664.6 | 1.9× |
| 50000 | 101.9  | `mlx_cell_pairs` + batched | 3193.6 | 31.3× | 234.0 | 2.3× |

`ratio = reference_steps_per_s / mlx_steps_per_s` (> 1 means the reference engine
runs more steps/s — the MLX gap).

Two compounding MLX changes produced these numbers, both at identical physics:

1. **Neighbor backend `mlx_cell_blocks` → `mlx_cell_pairs`** (compact real pairs
   instead of ~11× padded candidates): 99.5 / 31.8 / 8.5 → 649 / 180 / 57 steps/s
   at 4k / 16k / 50k (**~6×**). Locked by
   `tests/test_neighbors.py::test_default_backend_switch_preserves_lj_physics`.
2. **Compiled batched-block integrator** (sync once per block, not per step):
   649 / 180 / 57 → 1354 / 346 / 102 steps/s (**2.2× / 1.9× / 1.8×**). The batched
   path uses the same seeded threaded PRNG and the same per-step arithmetic, so it
   reproduces the per-step trajectory to floating-point precision (≈1e-4 energy
   over the run; differences are summation-order ULPs from a larger skin, the same
   class of difference as changing the rebuild interval) — locked by
   `tests/test_nvt.py::test_batched_langevin_matches_per_step`.

Cumulatively MLX is **13.6× / 10.9× / 12.0×** faster at 4k / 16k / 50k than the
original `mlx_cell_blocks` baseline.

### Interpretation

- **MLX now tracks LAMMPS within ~1.6–2.3× across the ladder** — same reduced-unit
  LJ physics, both building neighbor lists on the host with the pair force on the
  GPU — down from ~20–27× at the start. The two levers were structural, not
  micro-optimization: stop computing masked padded candidates, and stop paying a
  host round-trip every step.
- **The batched fast path removes the per-step host sync.** With a managed
  neighbor list the integrator could not be compiled and synced (`mx.eval`) every
  step. Running `block_size` Langevin substeps as one compiled block — with the
  neighbor displacement check at block boundaries and a larger skin so rebuilds
  stay rare — cuts the host round-trips by `block_size`× and lets MLX fuse the
  step. A fixed-list NVE micro-loop showed the ceiling: per-step sync caps 4k at
  ~3.2k steps/s, while syncing once per ~50 steps reaches ~9k.
- **The residual gap to OpenMM is structural.** OpenMM throughput is nearly flat
  (24k → 3.2k steps/s, ~7.5× for a 50× size increase); MLX goes 1966 → 102 (~19×).
  OpenMM keeps the *entire* step on-device (fused tile kernels, neighbor list on
  device, no host round-trip). MLX still rebuilds the neighbor list on the host
  (`argwhere` compaction), which is ~60–75% of the remaining wall time at scale and
  forces a host sync per block.
- The next closure lever is therefore an **on-device neighbor rebuild** — a Metal
  pair-emitter kernel (`mx.fast.metal_kernel` exists) to eliminate the host
  `argwhere` compaction the `mlx_cell_pairs` path still uses — plus reducing the
  per-step Langevin RNG cost inside the block.

### Energy-conservation note (why these numbers are trustworthy)

This ladder was first run before a correctness fix and exposed a critical bug:
MLX MD did **not conserve energy** on a standard LJ fluid — total energy climbed
every step and diverged to `NaN` within a few thousand steps, which also corrupted
the throughput numbers (a diverging system changes the per-step work and crashed
the neighbor backend at 16k/50k). Root cause: `Cell.wrap` round-tripped through
fractional coordinates, so in float32 it nudged boundary atoms by ~1e-2 each step
(not an exact lattice translation), doing spurious work. The fix makes `wrap` a
direct `x - L·floor(x/L)` translation (matching `minimum_image`); NVE now conserves
energy over long runs and a regression test guards it
(`tests/test_nve.py::test_simulate_nve_conserves_energy_over_long_run`,
`tests/test_core.py::test_wrap_is_exact_lattice_translation`). The numbers above are
post-fix: every MLX run completed with finite, bounded energy drift.

## Reproduce

```bash
uv run python scripts/run_same_workload_lj_scaling.py \
  --sizes 1000,4000,16000,50000 --steps 3000,2000,800,300 \
  --block-size 16 --neighbor-skin 1.2
```

(`--block-size 1` reproduces the per-step baseline; `--block-size 16
--neighbor-skin 1.2` is the batched fast path reported above. Both are the same
physics — the batched path matches the per-step trajectory to floating-point
precision.)

Raw per-engine JSON and the aggregated `summary.json` are written under the
gitignored `results/same-workload-lj-scaling/`. The MLX command stays in the
`mlx_atomistic.benchmarks` module; OpenMM and LAMMPS commands stay under
`scripts/`, per the reference-engine boundary.
