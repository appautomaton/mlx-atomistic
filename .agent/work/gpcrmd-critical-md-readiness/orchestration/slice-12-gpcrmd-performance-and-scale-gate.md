# Slice 12: GPCRmd Performance And Scale Gate

## Result

Completed. The selected GPCRmd MLX run path now has a repeatable short benchmark
surface that writes JSON/CSV rows for runnable systems and explicit blocker rows
for blocked systems.

## Scope

- Added `mlx_atomistic.prep.gpcrmd_benchmark.benchmark_gpcrmd_mlx(...)`.
- Added API command:

  ```bash
  uv run mlx_atomistic.prep Python API benchmark-gpcrmd-mlx \
    --target <id> \
    --cache <gpcrmd-cache-or-manifest> \
    --out <benchmark-dir> \
    --durations-ps 0.01 \
    --json
  ```

- Added shared benchmark helpers under `mlx_atomistic.benchmarks.gpcrmd_runtime`.
- Benchmark rows report:
  - import/run/total wall time;
  - integration steps/s and ps/s;
  - atom, water, ion, lipid counts;
  - PME mesh shape/size when present;
  - final pair count and rebuild count;
  - memory and artifact size;
  - finite trajectory diagnostics;
  - blocker strings for non-runnable cases.
- Electrostatics comparison requests are explicit and honest: the runnable row
  uses the electrostatics mode encoded in the prepared artifact; requested
  cutoff/Ewald/PME variants block unless they match that prepared artifact.

## Files Changed

- `src/mlx_atomistic/prep/gpcrmd_benchmark.py`
- `src/mlx_atomistic/prep/`
- `src/mlx_atomistic/benchmarks/gpcrmd_runtime.py`
- `tests/test_gpcrmd_registry.py`

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py -k "benchmark"`
  - `3 passed, 25 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "benchmark or performance or gpcrmd"`
  - `60 passed, 248 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py`
  - `28 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gpcrmd_benchmark.py src/mlx_atomistic/benchmarks/gpcrmd_runtime.py src/mlx_atomistic/prep/ tests/test_gpcrmd_registry.py tests/test_benchmarks.py`
  - `All checks passed!`
- Blocked API smoke emitted blocker JSON and exited nonzero as expected for an
  unknown target.
- External-engine scan over touched runtime benchmark files found no OpenMM,
  GROMACS, LAMMPS, subprocess, or shell simulation calls.

## Remaining Risks

- This is a runtime/performance gate, not a biological sampling claim.
- Cutoff/Ewald/PME performance comparisons require distinct prepared artifacts
  whose electrostatics settings are physically valid; this slice does not mutate
  artifacts to create fake comparison modes.
