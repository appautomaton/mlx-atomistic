# Requirements

## Product Commitments

### Observed

- The project targets Apple Silicon-native atomistic simulation experiments built on MLX and Metal (`README.md`, `pyproject.toml`).
- Initial scope is lightweight DFT building blocks, molecular mechanics, validation notebooks, and visualization utilities rather than a heavyweight production DFT engine (`README.md`).
- Notebook workflows are first-class and should run from the project `uv` environment (`README.md`, `notebooks/README.md`, `AGENTS.md`).
- Active ligand-receptor visualization should use MLX-generated trajectory outputs, not stale public trajectory analysis as active results (`notebooks/README.md`).

### Inferred

- User-facing work should preserve a scientific, validation-oriented posture and avoid implying production chemistry coverage beyond the implemented surfaces (`README.md`, `docs/` references in `README.md`).
- Roadmap items should improve correctness, visibility, and workflow reliability before widening package dependencies (`README.md`, `pyproject.toml`, `AGENTS.md`).

### Needs Confirmation

- The next primary user-facing slice after bootstrap has not been selected (`.agent/.automaton/state/current.json`, `.agent/wiki/REPO-MAP.md`).
- Full-repo Ruff policy for archived notebooks is unresolved because `uv run ruff check .` currently fails on notebook findings (`pyproject.toml`, `notebooks/README.md`).

## Technical Constraints and Invariants

### Observed

- Use Python 3.13 through `.python-version` and `requires-python = ">=3.13,<3.14"` (`.python-version`, `pyproject.toml`).
- Use `uv run ...` and `uv sync ...` for Python environment and command execution (`AGENTS.md`, `README.md`).
- Keep package source under `src/mlx_atomistic/` unless the task explicitly targets the peer `atomistic_prep` package (`AGENTS.md`, `pyproject.toml`).
- Keep notebooks under `notebooks/` and import package code through the `uv` environment (`AGENTS.md`, `notebooks/README.md`).
- Treat `vendors/` as reference material only (`AGENTS.md`, `README.md`).
- Do not add heavyweight chemistry or ML helper packages without concrete need (`AGENTS.md`, `pyproject.toml`).

### Inferred

- Optional extras should remain the boundary for notebook, visualization, and preparation dependencies (`pyproject.toml`).
- Console scripts should remain stable because they are declared package entry points (`pyproject.toml`, `src/atomistic_prep/cli.py`).

### Needs Confirmation

- Whether `atomistic_prep` remains a separate installable package long-term or becomes an internal workflow layer (`pyproject.toml`, `src/atomistic_prep/cli.py`).

## Quality and Operational Expectations

- testing bar: `uv run pytest` is the main regression gate and currently passes with `162 passed` (`pyproject.toml`, verified command).
- source lint bar: `uv run ruff check src tests scripts` currently passes (`pyproject.toml`, verified command).
- full lint state: `uv run ruff check .` currently fails on notebook lint findings and should not be claimed green until policy or notebooks change (`pyproject.toml`, verified command).
- sandbox reliability: use `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache` when the default home `uv` cache is not writable (`README.md`, verified commands).

## Integration Boundaries

- upstream or downstream systems: MLX/Metal for Apple Silicon execution, optional scientific Python packages for notebooks/prep/viz, and local Jupyter for notebook workflows (`README.md`, `pyproject.toml`, `notebooks/README.md`).
- external contracts that cannot break: package imports from `mlx_atomistic`, console scripts `atomistic-prep` and `mlx-atomistic-benchmark`, notebook workflow locations, and vendor reference boundary (`src/mlx_atomistic/__init__.py`, `pyproject.toml`, `notebooks/README.md`, `AGENTS.md`).

## Non-Goals

- Do not turn `vendors/` into package dependencies or build inputs without an explicit task changing that boundary (`AGENTS.md`, `README.md`).
- Do not add broad/heavy chemistry or ML helper packages without a concrete implementation need (`AGENTS.md`, `pyproject.toml`).
- Do not treat archived notebooks as active workflow truth (`notebooks/README.md`).
- Do not present this as a broad production DFT engine beyond the lightweight validated scope in the README (`README.md`).

## Open Risks and Unknowns

- Notebook lint policy is inconsistent with a green full-repo Ruff command today (`pyproject.toml`, `notebooks/README.md`, verified command).
- The first post-bootstrap feature/change is not yet framed (`.agent/.automaton/state/current.json`, `.agent/wiki/REPO-MAP.md`).
- Optional preparation and visualization extras are intentionally broad; dependency expansion should stay justified by specific workflow needs (`pyproject.toml`, `AGENTS.md`).

## Evidence Anchors

- `README.md`
- `AGENTS.md`
- `.python-version`
- `pyproject.toml`
- `src/mlx_atomistic/__init__.py`
- `src/atomistic_prep/cli.py`
- `notebooks/README.md`
- `.agent/.automaton/state/current.json`
