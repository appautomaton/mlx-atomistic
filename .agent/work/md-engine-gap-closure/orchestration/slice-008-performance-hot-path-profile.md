# Slice 8: Performance Hot-Path Profile

## Status

Complete at decision checkpoint.

## What ran

- `uv run python -m mlx_atomistic.benchmarks.md_performance --json`
- `uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 2000 --steps 1000 --csv results/md-engine-gap-closure/performance/mlx-synthetic-2000-1000.csv --json`
- `uv run python scripts/benchmark_openmm_opencl.py --particles 2000 --steps 10000 --warmup-steps 100 --json --csv results/md-engine-gap-closure/performance/openmm-opencl-2000-10000.csv`
- `uv run python -m cProfile -o results/md-engine-gap-closure/performance/mlx-md-performance-2000.prof -m mlx_atomistic.benchmarks.md_performance --sizes 2000 --steps 100 --json`
- `uv run python scripts/benchmark_openmm_opencl.py --help`

## Evidence

- Default MLX profile completed on `mlx==0.31.2`, default device `Device(gpu, 0)`,
  Metal available.
- MLX synthetic LJ, 2000 atoms, 1000 steps: `729.5651978507578` steps/s.
- OpenMM OpenCL synthetic LJ, 2000 atoms, 10000 steps: `19088.50935994495`
  steps/s on `Apple M5 Max`, OpenCL single precision.
- cProfile for MLX 2000 atoms / 100 steps ranked `simulate_nvt` and
  `_eval_runtime_state` as the visible Python-side hot path; dense nonbonded
  pair evaluation is the algorithmic bottleneck (`1999000` dense pairs,
  `224000000` estimated dense bytes).
- Local profile summary written to
  `results/md-engine-gap-closure/performance/SUMMARY.md`.

## Decision

No optimization patch was started. The measured next optimization target should
be a focused follow-up spec around nonbonded pair/neighbor behavior and runtime
diagnostic synchronization. PME FFT needs its own PME-specific profile.
