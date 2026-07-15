---
title: "Same-Workload DHFR Stretch Status"
---


Date: 2026-07-15

Engine pair: `mlx_atomistic` vs `openmm-reference`.
Pair ids: `dhfr-implicit`, `dhfr-explicit-pme`.

These are the real-system DHFR stretch rows for the same-workload comparison
plan. `dhfr-implicit` runs as a one-step MLX/OpenMM Reference smoke comparison.
`dhfr-explicit-pme` now runs under the explicit
`uniform_neutralizing_plasma` convention and passes fixed-coordinate OpenMM
energy/complete-force parity, but it still has no throughput ratio because the
available runtime rows do not share a matching manifest.

## Row Status

| Pair id | Intended system | Physics target | Metric | Status | Raw output |
| --- | --- | --- | --- | --- | --- |
| `dhfr-implicit` | DHFR implicit solvent, ~2.5k atoms | implicit GBSA/OBC target | `ns/day` | `comparable` one-step smoke row; ratio in summary | MLX: `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json`; OpenMM Reference: `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json`; summary: `results/same-workload-openmm-comparison/summary.json` |
| `dhfr-explicit-pme` | Charged AMBER20 JAC explicit PME, 23,558 atoms | explicit solvent PME with uniform neutralizing plasma | energy/complete-force parity; one-step MLX `ns/day` diagnostic | parity passed; MLX runtime passed; no throughput ratio | `results/scalable-charged-pme-runtime/jac-1x/charged_pme_parity_report.json`; `results/scalable-charged-pme-runtime/jac-1x/runtime-smoke.json` |

## OpenMM Reference

The reference side is already measured in
[`openmm-opencl-dhfr.md`](./openmm-opencl-dhfr.md):

| OpenMM system | Size | Result |
| --- | ---: | ---: |
| DHFR Implicit GBSA | ~2.5k atoms | 1762.1 ns/day |
| DHFR Explicit PME | 23.6k atoms | 752.5 ns/day |

The committed OpenMM OpenCL numbers are reference context, not the same thing
as the one-step OpenMM Reference shape-check rows used by the same-workload
comparison helper.

## MLX Status

`mlx_atomistic` can resolve the local DHFR inputs. The two rows now differ:

- `dhfr-implicit`: runnable. The prep path derives a DHFR GBSA/OBC artifact
  from OpenMM `amber99sb.xml` and `amber99_obc.xml`, including `gbsa_radius`,
  `gbsa_scale`, bonded terms, torsions, constraints, and nonbonded exceptions.
  The MLX runtime currently reports a one-step `0.004 ps` smoke row.
- `dhfr-explicit-pme`: runnable with the explicit
  `uniform_neutralizing_plasma` policy. The 23,558-atom fixed-coordinate report
  passed with `0.00013610 kJ/mol/atom` total-energy error,
  `1.05591e-5` relative energy error, `0.07247 kJ/mol/nm` force RMS, and
  `1.18571 kJ/mol/nm` maximum force error. The one-step MLX NVT smoke reported
  `0.047433 ns/day`, one PME plan build, lazy topology,
  `mlx_cell_blocks`, and no materialized dense pair cache. These values come
  from the two JAC 1x raw files listed in the row table.

The comparison helper computes a ratio only when operation, atom count, step
count, timing metric, and PME configuration match. The charged explicit-PME
energy/force comparison meets that contract and passes. The one-step MLX NVT
timing does not have a matching OpenMM runtime manifest, so the performance
ratio remains suppressed.

## Reproducer Boundary

MLX row reproducers:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --amber-topology results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop --amber-coordinates results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd --json > results/scalable-charged-pme-runtime/jac-1x/runtime-smoke.json
```

OpenMM same-workload shape-check and charged parity reproducers:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --with openmm python scripts/run_charged_pme_parity.py --mlx-prepared results/dhfr-artifacts/dhfr-explicit-pme --amber-prmtop results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop --amber-coordinates results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd --replicas 1,1,1 --platform OpenCL --out results/scalable-charged-pme-runtime/jac-1x
```

OpenMM OpenCL context is preserved as internal reference evidence; it is not a
package workflow or a required PyPI validation step.

The OpenMM Reference commands above are one-step same-workload shape checks.
They are not the OpenCL performance numbers in the context table.
