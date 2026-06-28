---
title: "Testing"
---


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

Markers are registered with `--strict-markers`, so a typo'd marker fails fast.

## Commands

Local fast lane — no reference engines, so it never builds LAMMPS:

```bash
uv run --locked --no-default-groups --group test python -m pytest -m "not slow and not integration and not reference and not data and not gpu"
```

Hosted CI/package boundary lane — the deterministic subset GitHub Actions runs
before packaging:

```bash
uv run --locked --no-default-groups --group test python -m pytest tests/test_runtime_boundaries.py
```

Package suite + coverage — local Apple Silicon release gate. Reference-engine
and vendor-data tests remain separate opt-in lanes:

```bash
uv run --locked --no-default-groups --group test python -m pytest -m "not reference and not data and not gpu" --cov=mlx_atomistic --cov-report=term-missing --cov-fail-under=80
```

Run explicit reference or data tiers only after provisioning those local
surfaces:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-reference -m reference
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group dev python -m pytest --run-data -m data
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run --locked --no-default-groups --group test python -m pytest --run-gpu -m gpu
```

MLX runtime tests require a local Apple Silicon host with stable MLX execution.
Headless, virtualized, or hosted macOS sessions can collect tests and run static
package-boundary checks, but they are not the runtime validation environment for
`0.0.1`.

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
