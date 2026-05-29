---
active_change: 2026-05-23-dhfr-runnable-benchmarks
stage: verify
---

# Status

## Current Change

- active change: `2026-05-23-dhfr-runnable-benchmarks`
- current stage: `verify`

## What Is True Now

- The active frame is `.agent/work/2026-05-23-dhfr-benchmark-artifact-runtime/SPEC.md`.
- The active design is `.agent/work/2026-05-23-dhfr-benchmark-artifact-runtime/DESIGN.md`.
- The active plan is `.agent/work/2026-05-23-dhfr-benchmark-artifact-runtime/PLAN.md`.
- The user selected both implicit DHFR and explicit PME DHFR as in-scope.
- The goal is to make DHFR a runnable `mlx_atomistic` benchmark path, then refresh MLX/OpenMM comparison reporting with comparable, diagnostic, or blocked DHFR rows.
- The previous benchmark-ladder work remains the reference boundary: ratios only when MLX/OpenMM semantics and metrics match.
- Local DHFR-related inputs exist: OpenMM stock DHFR PDBs under `vendors/openmm/examples/benchmarks/`, Amber20/JAC inputs under `results/inputs/Amber20_Benchmark_Suite/PME/`, and existing OpenMM DHFR raw output under `results/openmm-opencl-dhfr-m5max.json`.
- The plan has eight slices: readiness, OpenMM reference, artifact import, implicit MLX runtime, explicit PME MLX runtime/readiness, comparison classification, docs refresh, and regression gate.

## Next Step

Run `auto-eng-review` before execution because the plan crosses benchmark schema, OpenMM reference behavior, AMBER artifact import, PME readiness, runtime boundaries, and benchmark reporting.

## Open Risks

- Explicit PME DHFR may expose missing PME, topology, constraint, long-range correction, or artifact-readiness gaps.
- Exact DHFR input files may not be available locally; the implementation must fail closed with a concrete acquisition/preparation blocker instead of uncontrolled downloads.
- Full MLX/OpenMM benchmark execution may require Metal/OpenCL access outside restrictive sandboxes.
