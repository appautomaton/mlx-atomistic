# Testing

The suite is tiered so the everyday lane is fast and deterministic, while heavier
or dependency-bound tests run on demand.

## Tiers (pytest markers)

- *(unmarked)* — fast unit tests: pure functions and tiny synthesized systems
  (2–10 atoms, 1–5 steps). The default lane; runs in seconds.
- `slow` — long-running physics / SCF / benchmark tests (>~1s).
- `integration` — multi-component end-to-end flows. A label, not a speed gate:
  fast integration tests still run in the fast lane; slow ones also carry `slow`.
- `reference` — require an external reference engine (OpenMM / LAMMPS). Guarded
  with `pytest.importorskip`, so they skip cleanly when the engine is absent.
- `data` — require a heavy, gitignored dataset (e.g. notebook OpenMM run reports).
  Skip-guarded when the data is absent.

Markers are registered with `--strict-markers`, so a typo'd marker fails fast.

## Commands

Fast lane — parallel, what CI runs on every push/PR (no reference engines, so it
never builds LAMMPS):

```bash
uv run --group test python -m pytest -n auto -m "not slow and not reference and not data"
```

Full suite + coverage — what CI runs nightly / on demand (builds LAMMPS, runs
every tier, gates coverage):

```bash
uv run --group dev python -m pytest --cov=mlx_atomistic --cov-report=term-missing --cov-fail-under=80
```

Run a single tier, e.g. just the slow tests:

```bash
uv run --group dev python -m pytest -m slow
```

## Dependency groups

- `test` — pytest, pytest-cov, pytest-xdist, ruff. Light; no reference engines,
  so the fast CI lane never has to build LAMMPS.
- `reference` — OpenMM (PyPI wheel) and LAMMPS (built from source with
  GPU/OpenCL).
- `dev` — `test` + `reference` (the full local/CI environment).

## Conventions

- Prove physics on tiny synthesized systems; never mock the numerics.
- Mock or guard only external boundaries: reference engines, file I/O, downloads.
- Keep heavy/gitignored datasets out of the fast lane; tag such tests `data`.
- Write outputs under `tmp_path`, never a fixed shared path — this keeps tests
  parallel-safe under `-n auto`.
