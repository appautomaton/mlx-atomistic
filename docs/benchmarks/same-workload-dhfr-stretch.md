# Same-Workload DHFR Stretch Status

Date: 2026-05-23

Engine pair: `mlx_atomistic` vs `openmm-reference`.
Pair ids: `dhfr-implicit`, `dhfr-explicit-pme`.

These are the real-system DHFR stretch rows for the same-workload comparison
plan. `dhfr-implicit` now runs as a one-step MLX/OpenMM Reference smoke
comparison. `dhfr-explicit-pme` remains blocked on the MLX side, so it still
has no ratio.

## Row Status

| Pair id | Intended system | Physics target | Metric | Status | Raw output |
| --- | --- | --- | --- | --- | --- |
| `dhfr-implicit` | DHFR implicit solvent, ~2.5k atoms | implicit GBSA/OBC target | `ns/day` | `comparable` one-step smoke row; ratio in summary | MLX: `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json`; OpenMM Reference: `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json`; summary: `results/same-workload-openmm-comparison/summary.json` |
| `dhfr-explicit-pme` | DHFR explicit PME, 23.6k atoms | explicit solvent PME target | `ns/day` | `blocked`; no ratio | MLX: `results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json`; OpenMM Reference: `results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json` |

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
- `dhfr-explicit-pme`: blocked before PME runtime because the local Amber20/JAC
  artifact has `net_charge=-11` and the current MLX PME artifact policy
  requires neutral systems.

The comparison helper computes a ratio only for `dhfr-implicit`. It keeps
`dhfr-explicit-pme` blocked and does not compute a ratio for that row.

## Reproducer Boundary

MLX row reproducers:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json
```

OpenMM same-workload shape-check reproducers:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json
```

OpenMM OpenCL context reproducer:

```bash
cd vendors/openmm/examples/benchmarks
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --project ../../../.. \
  python benchmark.py \
    --platform OpenCL \
    --test pme \
    --seconds 30 \
    --precision single \
    --outfile ../../../../results/openmm-opencl-dhfr-m5max.json
```

The OpenMM Reference commands above are one-step same-workload shape checks.
They are not the OpenCL performance numbers in the context table.
