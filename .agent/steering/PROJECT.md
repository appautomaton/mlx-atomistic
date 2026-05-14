# Project

## One-Liner

`mlx-atomistic` is a Python 3.13 `uv` package for Apple Silicon-native atomistic simulation experiments built on MLX and Metal (`README.md`, `pyproject.toml`).

## Why This Repo Exists

- The repository exists to explore lightweight, validated MD and DFT building blocks on local Apple Silicon rather than to ship a heavyweight production DFT engine (`README.md`).
- It prioritizes small examples, validation notebooks, visualization utilities, and benchmarkable kernels before broader chemistry coverage (`README.md`, `notebooks/README.md`).

## Current Users or Operators

- Primary operators are developers and notebook users working from the repo-local `uv` environment (`README.md`, `notebooks/README.md`, `AGENTS.md`).
- Secondary operators are Python API users preparing ligand-receptor artifacts or running benchmark surfaces through installed console scripts (`pyproject.toml`, `src/mlx_atomistic/prep/`).

## Current System Model

- request or event flow: users run `uv` commands, import `mlx_atomistic`, execute notebooks, call `mlx_atomistic.prep` APIs, or run benchmark modules/scripts (`README.md`, `pyproject.toml`, `src/mlx_atomistic/prep/`).
- primary surfaces: package API, preparation APIs, benchmark API/modules, notebooks, tests, and lint (`src/mlx_atomistic/__init__.py`, `pyproject.toml`, `notebooks/README.md`).
- critical dependencies: Python 3.13, MLX, NumPy, SciPy, optional notebook/prep/viz extras, Pytest, Ruff, and Hatchling (`.python-version`, `pyproject.toml`).

## Major Surfaces

| Surface | Path | Responsibility |
|---------|------|----------------|
| Core package | `src/mlx_atomistic/` | MD, DFT, force fields, validation, topology, runtime, trajectory, visualization, and benchmark APIs. |
| Preparation package | `src/mlx_atomistic/prep/` | Prepared-artifact import/build/run workflows exposed through Python APIs. |
| Notebooks | `notebooks/` | Jupyter-first workflows, active ligand-receptor visualization, and archived provenance. |
| Tests | `tests/` | Regression coverage for MD, DFT, topology, validation, prep, runtime, visualization, and artifacts. |
| Vendor references | `vendors/` | Reference source trees only, not dependencies or package inputs. |

## Stack Summary

- Python 3.13 is pinned by `.python-version` and `pyproject.toml` (`.python-version`, `pyproject.toml`).
- `uv` is the project execution and environment manager (`README.md`, `AGENTS.md`).
- MLX is the local GPU execution dependency; NumPy and SciPy are the core scientific dependencies (`pyproject.toml`).
- Hatchling builds the package, Pytest runs tests, and Ruff covers style/linting (`pyproject.toml`).

## Commands

- install: `uv sync --extra notebook --extra prep --extra viz --group dev` (`README.md`, `notebooks/README.md`).
- notebook dev: `uv run jupyter lab` (`README.md`, `notebooks/README.md`).
- targeted regression: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py tests/test_ligand_receptor_motion.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py` was verified and passed with `154 passed`.
- source lint: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` was verified and passed.
- full lint: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check .` was verified and currently fails on notebook Ruff findings.

## Decision Principles Already Visible In The Repo

- Prefer repo-local `uv` execution over global Python tools (`AGENTS.md`, `README.md`).
- Keep source under `src/mlx_atomistic/` and notebooks under `notebooks/` (`AGENTS.md`, `README.md`).
- Treat `vendors/` as reference material unless a task explicitly changes that boundary (`AGENTS.md`, `README.md`).
- Avoid heavyweight chemistry or ML helper packages without concrete need (`AGENTS.md`, `pyproject.toml`).
- Keep active notebooks focused and archive old milestone/provenance notebooks separately (`notebooks/README.md`).

## Evidence Anchors

- `README.md`
- `AGENTS.md`
- `.python-version`
- `pyproject.toml`
- `src/mlx_atomistic/__init__.py`
- `src/mlx_atomistic/prep/`
- `notebooks/README.md`
