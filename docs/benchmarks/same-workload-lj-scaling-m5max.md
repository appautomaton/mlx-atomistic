# Same-Workload LJ Scaling: MLX vs OpenMM vs LAMMPS (M5 Max)

Date: 2026-06-17

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
  neighbor-list threshold (1536 atoms), `mlx_cell`/neighbor backends above it.
  This compares MLX at its best per size, not a single fixed path.
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
(see note below).

| atoms | MLX steps/s | MLX backend | OpenMM steps/s | OpenMM/MLX | LAMMPS steps/s | LAMMPS/MLX |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000  | 1874.6 | `mlx_dense` | 24024.7 | 12.8× | 4358.1 | 2.3× |
| 4000  | 99.5   | `mlx_cell_blocks` | 4819.9 | 48.4× | 2165.1 | 21.8× |
| 16000 | 31.8   | `mlx_cell_blocks` | 4171.0 | 131.2× | 664.6 | 20.9× |
| 50000 | 8.5    | `mlx_cell_blocks` | 3193.6 | 375.7× | 234.0 | 27.5× |

`ratio = reference_steps_per_s / mlx_steps_per_s` (> 1 means the reference engine
runs more steps/s — the MLX gap).

### Interpretation

- **The gap widens sharply with system size**: OpenMM is ~13× faster at 1k atoms
  but ~376× faster at 50k. OpenMM throughput is nearly flat across the ladder
  (24k → 3.2k steps/s, ~7.5× for a 50× size increase); MLX throughput collapses
  (1875 → 8.5 steps/s, ~220×). MLX scales poorly, not just slowly.
- **There is a cliff at the dense → neighbor-list backend transition.** MLX's
  `auto` policy uses dense all-pairs below 1536 atoms and the
  `mlx_cell_blocks` neighbor backend above it. Crossing that boundary (1k → 4k)
  drops MLX from 1875 to 99 steps/s — a ~19× slowdown for 4× the atoms. The
  neighbor path, not raw pair math, is the dominant production-scale cost. This
  matches the prior audit: neighbor build and CPU-side pair compaction are the
  bottleneck (MLX has no on-device variable-length `where`/`nonzero` emitter).
- **LAMMPS isolates the units question.** LAMMPS uses the identical reduced-unit
  LJ physics and also builds neighbor lists on the host (GPU pair force only), yet
  is ~20–27× faster than MLX at scale. So the gap is real MLX runtime overhead,
  not an artifact of comparing reduced units to OpenMM's physical units.
- The single fastest closure lever is therefore the neighbor-list/compaction path
  (a padded on-device neighbor list or a Metal pair-emitter kernel), consistent
  with the optimization backlog in `performance-audit-baseline.md`.

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
  --sizes 1000,4000,16000,50000 --steps 3000,2000,800,300
```

Raw per-engine JSON and the aggregated `summary.json` are written under the
gitignored `results/same-workload-lj-scaling/`. The MLX command stays in the
`mlx_atomistic.benchmarks` module; OpenMM and LAMMPS commands stay under
`scripts/`, per the reference-engine boundary.
