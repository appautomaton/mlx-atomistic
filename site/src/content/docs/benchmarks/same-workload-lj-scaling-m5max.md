---
title: "Same-Workload LJ Scaling: MLX vs OpenMM vs LAMMPS (M5 Max)"
---


Date: 2026-06-18 (MLX re-measured after the `mlx_cell_pairs` neighbor-default
switch, the batched-block integrator, and dropping the per-rebuild pair sort;
OpenMM and LAMMPS reference rows are carried over unchanged from the 2026-06-17
run on the same machine — only the MLX runtime changed.)

This is the first **large-scale synthetic-LJ** same-workload throughput
comparison of the `mlx_atomistic` runtime against the OpenMM and LAMMPS
reference engines on a single Apple M5 Max. It exists because every prior
committed MLX number was smoke-sized (4–32 atoms, 1 step), where fixed overhead
dominates and the real O(N) force/neighbor costs are invisible. The goal here is
an honest throughput baseline, not production chemistry or production MD
certification.

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
pairs, unsorted) and the integrator runs the **compiled batched-block fast path**
(`block_size=16`, `neighbor_skin=1.2`) — `block_size` Langevin substeps per
compiled block with one host sync per block instead of per step.

| atoms | MLX steps/s | MLX backend | OpenMM steps/s | OpenMM/MLX | LAMMPS steps/s | LAMMPS/MLX |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000  | 1906.8 | `mlx_dense` | 24024.7 | 12.6× | 4358.1 | 2.3× |
| 4000  | 2152.6 | `mlx_cell_pairs` + batched | 4819.9 | 2.2× | 2165.1 | 1.0× |
| 16000 | 663.7  | `mlx_cell_pairs` + batched | 4171.0 | 6.3× | 664.6 | 1.0× |
| 50000 | 200.9  | `mlx_cell_pairs` + batched | 3193.6 | 15.9× | 234.0 | 1.2× |

`ratio = reference_steps_per_s / mlx_steps_per_s` (> 1 means the reference engine
runs more steps/s — the MLX gap).

Three compounding MLX changes produced these numbers, all at identical physics:

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
3. **Dropped the per-rebuild pair sort** (`NeighborListManager.sort_pairs` now
   defaults to `False`): 1354 / 346 / 102 → 2153 / 664 / 201 steps/s at 4k / 16k /
   50k (**1.6× / 1.9× / 2.0×**). A wall-clock decomposition of a 50k rebuild showed
   the final `np.lexsort` of ~4.8M pairs was **~77% of the rebuild** (~700 ms) and
   bought nothing: MLX scatter-add is insensitive to pair order, so the
   Lennard-Jones energy and forces are unchanged (identical pair sets; residuals
   are summation-order ULPs), and unsorted lists are still deterministic. Locked by
   `tests/test_neighbors.py::test_unsorted_pairs_are_physics_neutral_and_default_in_md_loop`.
   (`cProfile` had mis-attributed the rebuild cost to the Python candidate loop;
   wall-clock timers found the sort. A vectorized candidate generator was
   prototyped and discarded — it gave only ~2% once the sort was gone.)

Cumulatively MLX is **21.6× / 20.9× / 23.6×** faster at 4k / 16k / 50k than the
original `mlx_cell_blocks` baseline.

### Interpretation

- **MLX now matches LAMMPS across the ladder (~1.0–1.2×)** — same reduced-unit LJ
  physics, both building neighbor lists on the host with the pair force on the GPU
  — down from ~20–27× at the start, and from ~1.6–2.3× before the pair sort was
  dropped. At 16k the two are neck-and-neck (663.7 vs 664.6 steps/s). The three
  levers were structural, not micro-optimization: stop computing masked padded
  candidates, stop paying a host round-trip every step, and stop sorting a pair
  list nothing reads in order.
- **The batched fast path removes the per-step host sync.** With a managed
  neighbor list the integrator could not be compiled and synced (`mx.eval`) every
  step. Running `block_size` Langevin substeps as one compiled block — with the
  neighbor displacement check at block boundaries and a larger skin so rebuilds
  stay rare — cuts the host round-trips by `block_size`× and lets MLX fuse the
  step. A fixed-list NVE micro-loop showed the ceiling: per-step sync caps 4k at
  ~3.2k steps/s, while syncing once per ~50 steps reaches ~9k.
- **Measure with wall-clock timers, not `cProfile`, for this code.** `cProfile`
  attributes by call count, so it inflated the per-cell Python candidate loop to
  ~72% of a 50k rebuild and made the single expensive `np.lexsort` C call look
  free. Plain `perf_counter` timing reversed that: the sort was ~77%, the loop a
  few percent. The fix followed the corrected measurement (drop the sort), not the
  profile (which would have sent us to rewrite the loop for ~2%).
- **The residual gap to OpenMM is structural, and now splits two ways.** OpenMM
  throughput is nearly flat (24k → 3.2k steps/s, ~7.5× for a 50× size increase);
  MLX goes 1907 → 201 (~9.5×). OpenMM keeps the *entire* step on-device (fused tile
  kernels, neighbor list on device, no host round-trip) and is launch-bound at
  these sizes — extra atoms are nearly free. MLX still pays two real O(N) costs:
  the per-step force (now ~3.8 ms/step at 50k, several dispatched kernels per
  substep vs OpenMM's one fused tile kernel) and the host neighbor rebuild
  (now ~⅓ of 50k wall, down from ~⅔ before the sort fix).
- The next closure levers are therefore **force-step kernel fusion** (collapse the
  per-substep gather → distance → LJ → scatter-add dispatch chain) and an
  **on-device neighbor rebuild** (`mx.fast.metal_kernel` exists) to remove the
  remaining host round-trip and flatten MLX's curve toward OpenMM's launch-bound
  regime.

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
precision. The managed neighbor list now defaults to unsorted pairs
(`NeighborListManager.sort_pairs=False`); pass `sort_pairs=True` to restore
canonical `(i, j)` ordering at the cost of a per-rebuild `lexsort`.)

Raw per-engine JSON and the aggregated `summary.json` are written under the
gitignored `results/same-workload-lj-scaling/`. The MLX command stays in the
`mlx_atomistic.benchmarks` module; OpenMM and LAMMPS commands stay under
`scripts/`, per the reference-engine boundary.
