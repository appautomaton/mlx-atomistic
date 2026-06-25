---
title: "Same-Workload Comparison Matrix"
---


Date: 2026-05-23

Scope: `.agent/work/2026-05-23-same-workload-openmm-benchmark-comparison`.
This matrix defines the first MLX/OpenMM comparison pairs. It is a routing
artifact for benchmark implementation, not a performance report.

## Rules

- Compare `mlx_atomistic` to OpenMM only when workload, physics, hardware, and
  metric family match.
- Keep MLX commands under `src/mlx_atomistic/benchmarks/` module entry points.
- Keep OpenMM commands under `scripts/`; OpenMM remains a reference/dev
  surface, not a product runtime dependency.
- Write raw JSON/CSV under gitignored `results/same-workload-openmm-comparison/`.
- Mark rows as `comparable`, `diagnostic`, or `blocked`; do not compute a ratio
  for diagnostic or blocked rows.
- For the synthetic-LJ scaling ladder, LAMMPS is a first-class third engine
  (reduced-unit `lj/cut/gpu`, same `fcc_lattice` geometry as MLX). For the
  semantic smoke pairs below (GBSA/TIP4P/DHFR), LAMMPS remains deferred.

## Pair Matrix

| Pair id | Workload | MLX command | OpenMM command | Metric family | Comparable status | Output paths | Caveat or blocker policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lj-synthetic-loop` | synthetic LJ full-loop / nonbonded smoke | `uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 32 --steps 1 --sample-interval 1 --diagnostic-interval 1 --evaluation-interval 1 --json` | `uv run python scripts/benchmark_openmm_opencl.py --platform OpenCL --particles 32 --steps 1 --warmup-steps 0 --spacing-nm 1.0 --json` | `steps/s`; optionally `ns/day` when timestep and step semantics are aligned | `comparable` when both sides are `ok` and use the same particle/step count; otherwise `blocked` | `results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json`; `results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json` | Use only as a tiny controlled smoke row. Do not extrapolate to DHFR, ApoA1, PME, or production throughput. |
| `gbsa-obc-small` | small GBSA/OBC energy and force evaluation | `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json` | `uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json` | `ms/eval` or per-step timing | `comparable` only if OpenMM side evaluates the same OBC-style implicit-solvent physics; `blocked` if the reference script does not yet expose this case | `results/same-workload-openmm-comparison/mlx-gbsa-obc-small.json`; `results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json` | OpenMM reference must report the exact force setup or return `blocked` with the unsupported feature reason. |
| `tip4p-ew-water` | TIP4P-Ew virtual-site / water row | `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json` | `uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json` | `ms/eval` or per-step timing | `comparable` only if both sides report the same TIP4P-Ew virtual-site or water-workload operation; `diagnostic` if the MLX side measures reconstruction and OpenMM measures full water force evaluation | `results/same-workload-openmm-comparison/mlx-tip4p-ew-water.json`; `results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json` | Do not compare virtual-site reconstruction alone against full OpenMM water dynamics. |
| `dhfr-implicit` | DHFR implicit GBSA/OBC real-system stretch | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json` | `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json` | `ns/day` | `comparable` for the one-step MLX/OpenMM Reference smoke row when both sides are `ok` and use `0.004 ps` | `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json`; `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json`; `results/same-workload-openmm-comparison/summary.json` | Use as a narrow runtime/artifact smoke comparison. OpenMM OpenCL DHFR implicit remains context only. |
| `dhfr-explicit-pme` | DHFR explicit PME real-system stretch | `uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json` | `uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json` | `ns/day` | `blocked` until MLX has a scientifically valid neutral PME artifact or charged-PME policy | `results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json`; `results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json` | Current local Amber20/JAC PME artifact has `net_charge=-11`; OpenMM OpenCL DHFR PME remains context only. |

## Production Scaling Ladder

The `lj-synthetic-loop` row above is a tiny controlled smoke. The production-scale
throughput comparison is the synthetic-LJ size ladder (1k/4k/16k/50k) across MLX,
OpenMM, and LAMMPS, driven by `scripts/run_same_workload_lj_scaling.py` and
aggregated by `same_workload_compare.build_scaling_summary` (matched per size by
`(atom_count, step_count)`). Results, method, and caveats:
`docs/benchmarks/same-workload-lj-scaling-m5max.md`. Raw JSON under gitignored
`results/same-workload-lj-scaling/`.

```bash
uv run python scripts/run_same_workload_lj_scaling.py \
  --sizes 1000,4000,16000,50000 --steps 3000,2000,800,300
```

## Required Report Behavior

The final comparison report should answer five questions for each row:

1. Did both engines run?
2. Are the workload and metric actually comparable?
3. What raw output files support the row?
4. If comparable, what is the measured ratio?
5. If not comparable, what exact blocker or caveat prevents comparison?

OpenMM DHFR, ApoA1, Cellulose, and STMV reports may be cited as reference
context, but they are not direct comparisons until a matching `mlx_atomistic`
row exists.
