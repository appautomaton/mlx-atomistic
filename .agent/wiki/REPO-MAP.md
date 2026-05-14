# Repo Map

## One-Sentence Model

- `mlx-atomistic` is a Python 3.13 `uv` package for Apple Silicon-native atomistic simulation experiments built on MLX and Metal, with MD, DFT, notebook, benchmark, and preparation surfaces (`README.md`, `pyproject.toml`).

## What This Repository Owns

- Package code for the core `mlx_atomistic` API under `src/mlx_atomistic/`, including molecular mechanics, DFT building blocks, trajectory adapters, validation, benchmarks, and visualization exports (`src/mlx_atomistic/__init__.py`).
- The `mlx_atomistic.prep` subpackage for prepared-artifact and ligand-receptor workflow APIs (`pyproject.toml`, `src/mlx_atomistic/prep/`).
- Jupyter-first workflow notebooks under `notebooks/workflows/` plus a current ligand-receptor visualization workflow and archived provenance notebooks (`notebooks/README.md`).
- Reference vendor source trees under `vendors/`; they are explicitly layout-only/reference material, not package inputs (`README.md`, `AGENTS.md`).

## Runtime Surfaces

| Surface | Path | Role | Entry Points | Notes |
|---------|------|------|--------------|-------|
| Python package API | `src/mlx_atomistic/` | MLX atomistic simulation, DFT, MD, validation, visualization | imports from `mlx_atomistic` | Public API is exported through `src/mlx_atomistic/__init__.py`. |
| Preparation API | `src/mlx_atomistic/prep/` | Build/import prepared systems and run short MLX workflows | Python imports from `mlx_atomistic.prep` | No prep console command is declared. |
| Benchmark API | `src/mlx_atomistic/benchmarks/` | MD and DFT benchmark/validation entry points | `uv run mlx-atomistic-benchmark ...`; module benchmarks | Declared as `mlx-atomistic-benchmark = mlx_atomistic.benchmarks.md_performance:main` in `pyproject.toml`; examples are listed in `README.md`. |
| Notebook workflows | `notebooks/workflows/` and `notebooks/ligand-receptor-motion/` | Narrative, plotted, executable validation and visualization workflows | `uv run jupyter lab` | Active notebooks are described in `notebooks/README.md`. |
| Tests and lint | `tests/`, `pyproject.toml` | Regression and style surfaces | `uv run pytest`; `uv run ruff check ...` | `pytest` is configured in `pyproject.toml`; Ruff excludes `vendors` and targets Python 3.13. |

## Stack and Infrastructure

- Python is pinned to 3.13 through `.python-version` and `requires-python = ">=3.13,<3.14"` (`.python-version`, `pyproject.toml`).
- Package management and execution use `uv`; project setup and notebook setup commands are documented in `README.md` and `notebooks/README.md`.
- Runtime dependencies are small and MLX-first: `mlx`, `numpy`, and `scipy`; notebook/prep/viz dependencies live behind optional extras (`pyproject.toml`).
- Build uses Hatchling; tests use Pytest; lint uses Ruff with `vendors` and `.venv` excluded (`pyproject.toml`).

## Commands That Work Today

- install: `uv sync --extra notebook --extra viz --group dev` is the documented full notebook/viz setup (`README.md`, `notebooks/README.md`).
- dev: `uv run jupyter lab` is the documented notebook runtime (`README.md`, `notebooks/README.md`).
- test: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` was verified and passed with `162 passed`.
- source lint: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts` was verified and passed.
- full lint: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check .` was verified and currently fails on notebook Ruff issues, including archived notebooks and a few active workflow notebooks.

## Apps, Packages, and Boundaries

- Wheel packages include only `src/mlx_atomistic`, whose `prep/` subpackage is canonical (`pyproject.toml`).
- Source code should stay under `src/mlx_atomistic/`, including the canonical `prep/` subpackage (`AGENTS.md`, `pyproject.toml`).
- Notebooks belong under `notebooks/` and should import from the `uv` environment (`AGENTS.md`, `notebooks/README.md`).
- `vendors/` is a reference boundary only and should not be treated as dependency or build input without an explicit boundary change (`AGENTS.md`, `README.md`).

## External Systems and Integrations

- Apple Silicon GPU execution is through MLX/Metal (`README.md`, `pyproject.toml`).
- Optional preparation and visualization extras integrate with scientific Python packages such as Gemmi, ParmEd, RDKit, ASE, MDAnalysis, MDTraj, NGLView, py3Dmol, ProLIF, pandas, matplotlib, and Plotly (`pyproject.toml`).
- The active ligand-receptor notebook workflow builds and analyzes MLX-generated trajectories rather than relying on public trajectory results (`notebooks/README.md`).

## Existing Conventions

### Observed

- Use `uv run ...` and a writable `UV_CACHE_DIR` in sandboxed runs when needed (`README.md`, verified commands).
- Keep notebooks compact and narrative-first, with equations, plots, and diagnostics where useful (`notebooks/README.md`).
- Keep archived notebooks as provenance instead of active workflow truth (`notebooks/README.md`).

### Inferred

- Current near-term work should prefer strengthening source, tests, and active notebooks before broadening dependency or production-chemistry scope (`README.md`, `pyproject.toml`, `notebooks/README.md`).

### Needs Confirmation

- Whether full-repo Ruff should cover archived notebooks or whether archived notebooks should be excluded/normalized (`pyproject.toml`, `notebooks/README.md`, verified Ruff failure).
- Whether a future prep API should be reintroduced or the Python API should remain the only preparation surface (`pyproject.toml`, `src/mlx_atomistic/prep/`).

## Verification and Release Surfaces

- Pytest is the main regression gate and is currently green (`pyproject.toml`, verified `uv run pytest`).
- Ruff is the style gate for source/tests/scripts, but full-repo Ruff is not green because notebooks are linted too (`pyproject.toml`, verified Ruff commands).
- Console scripts declared in `pyproject.toml` expose only the benchmark API; prep workflows use Python APIs.

## Likely Hotspots for the First Changes

- Notebook lint policy and active-vs-archive notebook hygiene (`notebooks/README.md`, verified full Ruff failure).
- Preparation API production boundaries and fail-closed behavior around prepared artifacts (`src/mlx_atomistic/prep/`).
- Benchmark and validation consistency across DFT/MD surfaces (`README.md`, `src/mlx_atomistic/__init__.py`).

## Sources Read

- `.agent/.automaton/state/current.json` - active `bootstrap` / `frame` state.
- `.agent/steering/STATUS.md` - scaffold-level steering status before refresh.
- `README.md` - purpose, setup, initial scope, benchmark examples, and layout.
- `pyproject.toml` - package metadata, dependencies, extras, scripts, tests, and lint config.
- `.python-version` - Python version pin.
- `src/mlx_atomistic/__init__.py` - public API and owned simulation surfaces.
- `src/mlx_atomistic/prep/` - preparation APIs and workflow surface.
- `notebooks/README.md` - active notebook workflows, archive boundary, and regeneration command.

## Open Questions

- Should archived notebooks be excluded from full-repo Ruff, or should they be auto-normalized so `uv run ruff check .` is green?
- What is the first user-facing slice after bootstrap: notebook hygiene, preparation workflow hardening, benchmark reporting, or DFT/MD capability expansion?

## Import Verdict

- steering confidence: high for repo shape, stack, and current verification state; medium for roadmap priorities.
- recommended next skill: `auto-frame` for a specific first change, or `auto-plan` if the next slice is already accepted.
