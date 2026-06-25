---
title: "LAMMPS OpenCL on Apple M5 Max"
---


Engine: `lammps-reference`. This report covers the five official top-level
LAMMPS benchmark inputs from `vendors/lammps/bench/` on the single Apple M5 Max
host. It does not describe `mlx_atomistic` runtime performance.

## Result

| Case | Description | Status | Acceleration class | Loop time, s | Blocker or caveat |
| --- | --- | ---: | --- | ---: | --- |
| `lj` | Atomic Lennard-Jones fluid | `ok` | `full_gpu_opencl` | 0.298473 | none |
| `eam` | Bulk Cu EAM solid | `ok` | `full_gpu_opencl` | 0.725426 | none |
| `chain` | FENE bead-spring polymer melt | `diagnostic` | `partial_gpu_opencl` | 0.544019 | bond and Langevin styles are not mapped to GPU styles |
| `chute` | Granular chute flow | `diagnostic` | `cpu_only_diagnostic` | 0.108473 | no relevant mapped GPU/OpenCL styles in this build |
| `rhodo` | Rhodopsin in solvated lipid bilayer | `blocked` | `partial_gpu_opencl` | N/A | PPPM double-precision build conflicts with the single-precision Apple GPU path |

The raw records are under `results/m5max-reference/lammps/`. Each run copies the
official input files into a case-local work directory before invoking LAMMPS, so
the shared vendor tree stays unchanged.

## Environment

| Field | Value |
| --- | --- |
| LAMMPS version | 20250722 |
| Packaged executable | `.venv/lib/python3.13/site-packages/lammps/lmp` |
| GPU package | `PKG_GPU=ON` |
| GPU API | `opencl` |
| GPU precision | `single` |
| Device | Apple M5 Max |
| Run command root | `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py` |

Do not rely on `.venv/bin/lmp` as the final reproducer. The harness bypasses the
console script and records the active packaged executable.

## Acceleration Mapping

| Case | Relevant styles | GPU/OpenCL mapping |
| --- | --- | --- |
| `lj` | `pair lj/cut`, `fix nve` | both map to `lj/cut/gpu` and `nve/gpu` |
| `eam` | `pair eam`, `fix nve` | both map to `eam/gpu` and `nve/gpu` |
| `chain` | `bond fene`, `pair lj/cut`, `fix nve`, `fix langevin` | pair and NVE map to GPU; bond and Langevin do not |
| `chute` | `pair gran/hooke/history`, `fix gravity`, `fix freeze`, `fix nve/sphere` | no mapped GPU styles in this local build |
| `rhodo` | harmonic bonds, CHARMM angles/dihedrals, `lj/charmm/coul/long`, `pppm`, `shake`, `npt` | pair, PPPM, and NPT have GPU mappings; bonded terms and SHAKE do not |

The classification is about this local LAMMPS OpenCL build, not about every
possible LAMMPS build. A different build or precision setting can change the
mapping and the `rhodo` outcome.

## Blocked Case

`rhodo` reaches PPPM initialization and then fails:

```text
ERROR: PPPM was compiled for double precision floating point but GPU device supports single precision only.
```

That is why the suite manifest status is `blocked` even though the manifest
validates structurally. The failure is preserved in
`results/m5max-reference/lammps/rhodo.json`, with the LAMMPS log under
`results/m5max-reference/lammps/rhodo/work/log.lammps`.

## Reproducer

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py lammps --classify-only --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py lammps --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_m5max_reference.py validate --manifest results/m5max-reference/manifest.json --json
```

Use a host terminal or approved outside-sandbox command for final GPU/OpenCL
measurements. The sandbox can hide device access and should only be used for
dry classification or schema tests.
