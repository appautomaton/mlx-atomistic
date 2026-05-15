# Validation And Performance

This milestone turns the MD/MM code from "it runs" into "we can measure whether
it is correct, stable, and worth optimizing." The key idea is simple: before
writing custom Metal kernels, we need repeatable force checks, stability runs,
and benchmark rows that identify the hot path.

## Force Validation

`mlx_atomistic.validation` compares each force term against central
finite-difference forces:

```python
from mlx_atomistic.validation import run_force_validation_suite

results = run_force_validation_suite(seed=7, cases_per_term=1)
```

Each `ForceValidationResult` reports:

- maximum absolute force error
- RMS force error
- atom and coordinate of the worst error
- seed, tolerance, and pass/fail status

The command-line gauntlet emits JSON or CSV:

```bash
uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json
uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --csv validation.csv
```

The default suite is intentionally small for development. Increase
`--cases-per-term` before trusting a larger code change.

## Stability Diagnostics

The stability benchmark runs:

- bonded-chain NVE at multiple `dt` values
- LJ-liquid NVE
- LJ-liquid Langevin NVT

It records energy drift, relative drift, mean/final temperature, pair counts,
neighbor-list rebuilds, and nonfinite diagnostics:

```bash
uv run python -m mlx_atomistic.benchmarks.stability --json
uv run python -m mlx_atomistic.benchmarks.stability --sizes 128,512,2048 --csv stability.csv
```

`8192` particles is supported as an opt-in size, but it should not be part of
routine development checks.

## Performance Harnesses

The LJ MD benchmark now supports CSV:

```bash
uv run python -m mlx_atomistic.benchmarks.lj_md --sizes 128,512,2048 --steps 20 --json
uv run python -m mlx_atomistic.benchmarks.lj_md --sizes 128,512,2048 --steps 20 --csv lj.csv
```

The MM force-term benchmark separates the current hot-path candidates:

- bonded autodiff terms
- neighbor-list construction
- LJ pair-list evaluation
- direct cutoff Coulomb evaluation
- combined mixed LJ+Coulomb nonbonded evaluation
- distance-constraint projection

```bash
uv run python -m mlx_atomistic.benchmarks.mm_force_terms --particles 128 --evaluations 20 --json
```

Use these rows to decide where a custom Metal kernel belongs. At this stage the
right answer should come from timing data, not intuition.

## Development Gate

For normal development:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.stability --sizes 16 --steps 2 --bonded-steps 2 --dt-values 0.001 --json
```

For serious local performance work on Apple Silicon, run the larger benchmark
matrix outside the fast test loop and keep the JSON/CSV artifacts for comparison.

## OpenMM/OpenCL Reference

OpenMM is not a product runtime dependency, but it is useful as a reference
ceiling for local GPU/OpenCL throughput. The standalone showcase script keeps
that comparison under `scripts/`:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --particles 4096 --steps 1000 --platform OpenCL --csv openmm-opencl.csv
```

The script emits OpenMM version, available platforms, selected platform
properties, steps/s, ns/day, and final energy/finite-state diagnostics.

## Benchmark Reports

Per-run benchmark write-ups live under [`docs/benchmarks/`](./benchmarks/),
indexed by `docs/benchmarks/README.md`. Each report records hardware,
engine version, config, the reproducer command, and external reference
numbers (e.g. `openmm.org/benchmarks`, HECBioSim) so a future result can be
compared without re-deriving context. Raw JSON output is written to the
gitignored `results/` directory.
