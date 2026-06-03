# M5 Max Reference Engines

This page is the current single-machine reference benchmark map for OpenMM and
LAMMPS on the Apple M5 Max host. These are reference-engine results, not
`mlx_atomistic` runtime performance.

## Run Summary

| Field | Value |
| --- | --- |
| Raw manifest | `results/m5max-reference/manifest.json` |
| Created | 2026-06-01T01:40:45Z |
| Host | `AppCubics-MacBook-Pro.local` |
| Hardware | Apple M5 Max, Darwin arm64 |
| OpenMM | 8.5.1.dev-f7fa0c2, OpenCL platform available |
| LAMMPS | 20250722, `PKG_GPU=ON`, `GPU_API=opencl`, `GPU_PREC=single` |
| Harness | `scripts/benchmark_m5max_reference.py` |
| Manifest validation | `ok` |
| Suite status | `blocked`, because LAMMPS `rhodo` cannot run with this single-precision OpenCL GPU build |

The LAMMPS package metadata and executable path come from the active
`main/.venv` environment. The stale `main/.venv/bin/lmp` console script is not
the reproducer surface; the harness records and invokes the packaged executable
under `main/.venv/lib/python3.13/site-packages/lammps/lmp`.

## OpenMM Reference Results

The harness reran the upstream OpenMM benchmark script with OpenCL, single
precision, and 30-second timing targets. Raw wrappers and upstream JSON live
under `results/m5max-reference/openmm/`.

| Case | Upstream tests | Status | Worst-row ns/day | Raw wrapper |
| --- | --- | ---: | ---: | --- |
| DHFR | `gbsa`, `rf`, `pme` | `ok` | 751.19 | `results/m5max-reference/openmm/dhfr.json` |
| ApoA1 | `apoa1rf`, `apoa1pme`, `apoa1ljpme` | `ok` | 171.31 | `results/m5max-reference/openmm/apoa1.json` |
| Amber20 | `amber20-cellulose`, `amber20-stmv` | `ok` | 18.01 | `results/m5max-reference/openmm/amber20.json` |

Detailed committed OpenMM reports remain linked here and were not overwritten by
this rerun:

- [OpenMM OpenCL DHFR](./openmm-opencl-dhfr.md)
- [OpenMM OpenCL ApoA1](./openmm-opencl-apoa1.md)
- [OpenMM OpenCL Amber20 Cellulose and STMV](./openmm-opencl-amber20.md)

The fresh rerun agrees with the existing story: OpenMM's OpenCL backend runs the
selected named systems on the M5 Max, including the large Amber20 Cellulose and
STMV systems, but these numbers are reference ceilings for this repo.

## LAMMPS Reference Results

LAMMPS coverage uses the five official top-level inputs from
`vendors/lammps/bench/`. The harness copies each input into
`results/m5max-reference/lammps/<case>/work/` before running, so `vendors/`
stays read-only.

| Case | Status | Acceleration class | Loop time, s | Raw wrapper |
| --- | ---: | --- | ---: | --- |
| `lj` | `ok` | `full_gpu_opencl` | 0.298473 | `results/m5max-reference/lammps/lj.json` |
| `eam` | `ok` | `full_gpu_opencl` | 0.725426 | `results/m5max-reference/lammps/eam.json` |
| `chain` | `diagnostic` | `partial_gpu_opencl` | 0.544019 | `results/m5max-reference/lammps/chain.json` |
| `chute` | `diagnostic` | `cpu_only_diagnostic` | 0.108473 | `results/m5max-reference/lammps/chute.json` |
| `rhodo` | `blocked` | `partial_gpu_opencl` | N/A | `results/m5max-reference/lammps/rhodo.json` |

`chain` and `chute` are not failures of the harness. They are diagnostic because
the official inputs do not map fully to `/gpu` styles in the local
LAMMPS OpenCL build. `rhodo` is a real blocker:

```text
ERROR: PPPM was compiled for double precision floating point but GPU device supports single precision only.
```

See [LAMMPS OpenCL on M5 Max](./lammps-opencl-m5max.md) for the per-case style
mapping.

## MLX Boundary

The MLX benchmarks in this repo are still development and smoke coverage for
the product runtime. They are useful for catching regressions and proving
specific code paths, but they are not yet same-workload production MD results
against OpenMM or LAMMPS. Keep OpenMM/LAMMPS reference-engine numbers in this
page and MLX runtime claims in the MLX benchmark docs.

## Reproducer

Run final GPU/OpenCL measurements from a host terminal or an approved
outside-sandbox command. Sandboxed sessions can hide GPU devices or trigger
Metal/OpenCL cleanup errors.

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py environment --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py openmm --dry-run --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py lammps --classify-only --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py run --seconds 30 --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py validate --manifest results/m5max-reference/manifest.json --json
```

Amber20 inputs are external and gitignored. For this run, the official
`Amber20_Benchmark_Suite.tar.gz` was downloaded and extracted under
`results/inputs/Amber20_Benchmark_Suite/`.
