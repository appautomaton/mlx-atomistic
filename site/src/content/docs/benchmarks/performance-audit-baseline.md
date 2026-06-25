---
title: "Performance Audit Baseline"
---


Date: 2026-05-22

This report is the committed Slice 6 audit summary for
`2026-05-22-performance-audit-harness-hardening`. Raw JSON is gitignored under
`results/performance-audit-harness-hardening/`.

## Baseline Runs

| Benchmark | Engine | Command | Raw output | Status |
| --- | --- | --- | --- | --- |
| force-term microbenchmarks | `mlx_atomistic` | `uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 1 --particles 16 --json` | `results/performance-audit-harness-hardening/mm-force-terms-fast.json` | ok |
| nonbonded acceleration split | `mlx_atomistic` | `uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 16 --evaluations 1 --json` | `results/performance-audit-harness-hardening/md-acceleration-fast.json` | ok |
| full MD smoke | `mlx_atomistic` | `uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 32 --steps 1 --sample-interval 1 --diagnostic-interval 1 --evaluation-interval 1 --json` | `results/performance-audit-harness-hardening/md-performance-fast.json` | ok |
| Phase 3 physics smoke | `mlx_atomistic` | `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json` | `results/performance-audit-harness-hardening/phase3-physics-fast.json` | ok |
| PME missing-fixture blocked smoke | `mlx_atomistic` | `uv run python -m mlx_atomistic.benchmarks.pme_performance --fixture-dir results/missing-pme-fixture --iterations 1 --warmups 0 --json` | `results/performance-audit-harness-hardening/pme-blocked-fast.json` | blocked |
| OpenMM unavailable-platform smoke | `openmm-reference` | `uv run python scripts/benchmark_openmm_opencl.py --platform DefinitelyMissing --particles 16 --steps 1 --json` | `results/performance-audit-harness-hardening/openmm-blocked-fast.json` | blocked |
| LAMMPS OpenCL smoke | `lammps-reference` | `uv run python scripts/benchmark_lammps_opencl.py --particles 16 --steps 1 --json` | `results/performance-audit-harness-hardening/lammps-fast.json` | ok |

## Measured Rows

| Row | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| full MD synthetic LJ, dense backend | `steps_per_s` | 53.073 | `md-performance-fast.json` |
| full MD force evaluation | `force_eval_ms_per_step` | 0.221 | `md-performance-fast.json` |
| nonbonded `mlx_tiled` | `ms_per_eval` | 0.521 | `md-acceleration-fast.json` |
| nonbonded `mlx_dense` | `ms_per_eval` | 0.855 | `md-acceleration-fast.json` |
| nonbonded `mlx_pairs` force eval | `ms_per_eval` | 0.864 | `md-acceleration-fast.json` |
| `mlx_pairs` neighbor build | `neighbor_build_ms_per_eval` | 2.423 | `md-acceleration-fast.json` |
| `python_neighbor` total eval | `ms_per_eval` | 2.376 | `md-acceleration-fast.json` |
| Phase 3 replica exchange | `ms_per_eval` | 9.552 | `phase3-physics-fast.json` |
| GBSA/OBC energy and forces | `ms_per_eval` | 7.791 | `phase3-physics-fast.json` |
| TIP4P-Ew M-site reconstruction | `ms_per_eval` | 5.526 | `phase3-physics-fast.json` |
| soft-core lambda grid | `ms_per_eval` | 3.389 | `phase3-physics-fast.json` |
| virtual-site force redistribution | `ms_per_eval` | 1.125 | `phase3-physics-fast.json` |
| LAMMPS synthetic OpenCL smoke | `steps_per_s` | 700.076 | `lammps-fast.json` |

The reference-engine rows are context only. The OpenMM row intentionally uses a
missing platform and proves fail-soft behavior; the LAMMPS row is a tiny
synthetic smoke run and is not an apples-to-apples production target.

## Ranked Optimization Backlog

1. **Replica-exchange runtime materialization and serial replica execution.**
   Evidence: `phase3-physics-fast.json` reports
   `two_replica_temperature_exchange` at `9.552 ms/eval`, the slowest fast row,
   with `history_materialization_count: 8`.
   Reproducer: `uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json`.

2. **GBSA/OBC force evaluation.**
   Evidence: `phase3-physics-fast.json` reports
   `gbsa_obc_energy_forces` at `7.791 ms/eval` for only four atoms, while the
   surface-area term is `0.903 ms/eval`.
   Reproducer: same Phase 3 command above.

3. **TIP4P-Ew virtual-site reconstruction and advanced-water overhead.**
   Evidence: `phase3-physics-fast.json` reports TIP4P-Ew reconstruction at
   `5.526 ms/eval`; `mm-force-terms-fast.json` reports synchronized
   `tip4p-ew-reconstruct` at `1.158 ms/eval` and force redistribution at
   `0.866 ms/eval`.
   Reproducer: `uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 1 --particles 16 --json`.

4. **Neighbor-list build and pair-compaction overhead.**
   Evidence: `md-acceleration-fast.json` reports `mlx_pairs`
   `neighbor_build_ms_per_eval: 2.423`, larger than its force eval row
   (`0.864 ms/eval`) and the dense/tiled diagnostic rows at this small size.
   Reproducer: `uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 16 --evaluations 1 --json`.

5. **Full-loop MD synchronization/cadence path.**
   Evidence: `md-performance-fast.json` reports `53.073 steps/s` for a one-step
   smoke with `force_eval_ms_per_step: 0.221`; this row should be rerun at
   opt-in sizes before any custom-kernel work.
   Reproducer: `uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 32 --steps 1 --sample-interval 1 --diagnostic-interval 1 --evaluation-interval 1 --json`.

## Follow-On Spec Recommendation

The next optimization spec should target **replica-exchange and GBSA/OBC
Phase 3 overhead first**, then neighbor-list build/compaction. Custom Metal
kernel work remains deferred until opt-in larger-system runs reproduce these
rankings beyond fast synthetic probes.
