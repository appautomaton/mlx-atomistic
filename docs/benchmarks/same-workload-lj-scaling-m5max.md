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
(see note below). The MLX rows below are the post-switch numbers: above the dense
threshold the default neighbor backend is now `mlx_cell_pairs` (compacted real
pairs) rather than the former `mlx_cell_blocks` (padded candidate blocks).

| atoms | MLX steps/s | MLX backend | OpenMM steps/s | OpenMM/MLX | LAMMPS steps/s | LAMMPS/MLX |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1000  | 1974.3 | `mlx_dense` | 24024.7 | 12.2× | 4358.1 | 2.2× |
| 4000  | 649.2  | `mlx_cell_pairs` | 4819.9 | 7.4× | 2165.1 | 3.3× |
| 16000 | 180.2  | `mlx_cell_pairs` | 4171.0 | 23.1× | 664.6 | 3.7× |
| 50000 | 57.0   | `mlx_cell_pairs` | 3193.6 | 56.0× | 234.0 | 4.1× |

`ratio = reference_steps_per_s / mlx_steps_per_s` (> 1 means the reference engine
runs more steps/s — the MLX gap).

For reference, the prior `mlx_cell_blocks` default measured 99.5 / 31.8 / 8.5
steps/s at 4k / 16k / 50k. Switching to `mlx_cell_pairs` is a **6.5× / 5.7× / 6.7×**
MLX speedup at identical physics (energy and forces agree to float precision; a
regression test locks this in
`tests/test_neighbors.py::test_default_backend_switch_preserves_lj_physics`).

### Interpretation

- **The dense → neighbor-list cliff is now mild.** MLX's `auto` policy uses dense
  all-pairs below 1536 atoms and a neighbor list above it. Crossing that boundary
  (1k → 4k) drops MLX from 1974 to 649 steps/s — a ~3× slowdown for 4× the atoms,
  versus the ~19× cliff the old `mlx_cell_blocks` default produced. The fix was to
  stop evaluating padded candidate blocks: the former default generated ~11× more
  candidate pair-math than real pairs (256-wide padded blocks over a 27-cell
  stencil, ~91% masked) and rebuilt them on the host every few steps. Compacting
  to real pairs (`mlx_cell_pairs`) removes that waste at identical physics.
- **The residual gap is real per-step overhead, not the neighbor representation.**
  LAMMPS uses the identical reduced-unit LJ physics and also builds neighbor lists
  on the host (GPU pair force only). After the switch it is ~3.3–4.1× faster than
  MLX at 4k–50k, down from ~20–27× with the old default. That remaining ~3–4× is
  genuine MLX runtime overhead — per-step host sync (`mx.eval` each step),
  Langevin RNG, diagnostics, and per-term energy bookkeeping in the production
  loop — not the cost of compaction.
- **MLX still scales worse than OpenMM, but far less so.** OpenMM throughput is
  nearly flat across the ladder (24k → 3.2k steps/s, ~7.5× for a 50× size
  increase); MLX now goes 1974 → 57 steps/s (~35×), versus ~220× before the
  switch. The OpenMM/MLX ratio grows from ~12× at 1k to ~56× at 50k because OpenMM
  keeps the entire step on-device (fused tile kernels, no per-step host
  round-trip), which is the structural advantage MLX has yet to match.
- The next closure levers are therefore (1) the per-step loop overhead measured
  against the clean NVE micro-loop, and (2) keeping the step on-device to cut the
  per-step host sync — an on-device neighbor/pair-emitter kernel
  (`mx.fast.metal_kernel` exists) rather than the host compaction the
  `mlx_cell_pairs` path still uses.

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
