# Testing

The suite is tiered so the everyday lane is fast and deterministic, while heavier
or dependency-bound tests run on demand.

## Tiers (pytest markers)

- *(unmarked)* — fast unit tests: pure functions and tiny synthesized systems
  (2–10 atoms, 1–5 steps). The default lane; runs in seconds.
- `slow` — long-running physics / SCF / benchmark tests (>~1s).
- `integration` — multi-component end-to-end flows. A label, not a speed gate:
  fast integration tests still run in the fast lane; slow ones also carry `slow`.
- `reference` — require an external reference engine (OpenMM / LAMMPS). Skipped
  unless the run includes `--run-reference`.
- `data` — require a heavy, gitignored dataset (e.g. notebook OpenMM run reports).
  Skipped unless the run includes `--run-data`.
- `gpu` — require a visible Metal GPU. Skipped unless the run includes
  `--run-gpu`.
- `perf` — exercise DFT performance and formal-evidence governance. Skipped
  unless the run includes `--run-perf`; performance seals do not gate
  pre-v0.1 correctness.

Markers are registered with `--strict-markers`, so a typo'd marker fails fast.

## Commands

Local equivalent of the required pull-request lane:

```bash
uv run --locked --no-default-groups --extra prep --group test python -m pytest -m "not slow"
```

GitHub Actions runs that command on `ubuntu-22.04` with the `mlx-cpu` extra.
The opt-in `reference`, `data`, `gpu`, and `perf` tests skip unless their
matching flag is passed.

The scheduled/manual CPU lane includes `slow` tests and coverage:

```bash
uv run --locked --no-default-groups --extra prep --group test python -m pytest --cov=mlx_atomistic --cov-report=term-missing
```

Run explicit reference or data tiers only after provisioning those local
surfaces:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-reference -m reference
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-data -m data
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test python -m pytest --run-gpu -m gpu
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --extra prep --group test python -m pytest --run-perf -m perf
```

The Linux CPU result is the routine correctness reference. Metal is a local
development and optimization instrument: use it for real workloads, Metal-path
changes, and occasional CPU-versus-GPU parity, not as a routine CI tier.

## Dependency groups

- `test` — pytest, pytest-cov, pytest-xdist, ruff. Light and selected as the
  default `uv` group; no reference engines, so the fast CI lane never has to
  build LAMMPS.
- `reference` — OpenMM (PyPI wheel) and LAMMPS (built from source with
  GPU/OpenCL).
- `dev` — `test` + `reference` for opt-in local validation.

## Conventions

- Prove physics on tiny synthesized systems; never mock the numerics.
- Mock or guard only external boundaries: reference engines, file I/O, downloads.
- Keep heavy/gitignored datasets out of the fast lane; tag such tests `data`.
- Write outputs under `tmp_path`, never a fixed shared path.
