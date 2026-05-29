# Repo Map

## One-Sentence Model

Apple Silicon-native atomistic simulation library: molecular mechanics, MD integrators, and plane-wave DFT on MLX/Metal with validation against OpenMM and LAMMPS references.

## What This Repository Owns

- The `mlx_atomistic` Python package: MM force fields, MD integrators (NVE, NVT Langevin, NPT), DFT/SCF, geometry optimization, trajectory I/O, visualization
- Preparation tooling for importing real molecular systems (GPCRMD, PDB, CHARMM)
- Benchmark and validation infrastructure with OpenMM/LAMMPS parity checks
- Jupyter notebooks for exploratory validation

## Runtime Surfaces

| Surface | Path | Role | Entry Points | Notes |
|---------|------|------|--------------|-------|
| Core library | `src/mlx_atomistic/` | Primary product | `import mlx_atomistic` | ~30 modules + `dft/` and `prep/` subpackages |
| DFT subpackage | `src/mlx_atomistic/dft/` | Plane-wave DFT/SCF engine | `from mlx_atomistic.dft import run_scf, optimize_geometry` | 21 modules: SCF, XC, pseudopotentials, k-points, stress |
| Prep subpackage | `src/mlx_atomistic/prep/` | Import and system preparation | `from mlx_atomistic.prep import ...` | GPCRMD, PDB, topology import, artifact prep |
| Benchmarks | `src/mlx_atomistic/benchmarks/` | Performance and validation CLI | `python -m mlx_atomistic.benchmarks.lj_md` | 18 modules: LJ MD, MM forces, stability, DFT SCF, Ewald |
| CLI benchmark entry | `pyproject.toml:22` | Script entry point | `mlx-atomistic-benchmark` | Maps to `md_performance:main` |
| Notebooks | `notebooks/` | Exploratory validation and visualization | Jupyter | `workflows/`, `archive/`, `ligand-receptor-motion/` |
| Test suite | `tests/` | Unit and integration tests | `pytest` (testpaths: `tests/`) | 47 test files |
| Scripts | `scripts/` | Reference parity and production MD runs | `uv run python scripts/...` | OpenMM comparison, GPCRMD CHARMM runs |
| Vendor references | `vendors/` | Reference source trees (not imported) | N/A | CP2K, GROMACS, LAMMPS, OpenMM, Quantum ESPRESSO |

## Stack and Infrastructure

- Language: Python 3.13 (pinned `>=3.13,<3.14`, `.python-version`)
- GPU runtime: MLX (`mlx>=0.31.1`) for Apple Metal array operations
- Core deps: `numpy>=2.0`, `scipy>=1.14`
- Build: hatchling via `uv` (`tool.hatch.build.targets.wheel.packages = ["src/mlx_atomistic"]`)
- Test: pytest (`-q`, testpaths `tests/`)
- Lint: ruff (E, F, I, UP, B, SIM; line-length 100; target py313; excludes `vendors/`, `.venv/`)
- Dev deps (group): OpenMM, LAMMPS (+ MPICH), pytest, ruff
- Optional extras: `notebook` (Jupyter), `prep` (gemmi, parmed, rdkit), `viz` (ase, matplotlib, plotly, MDAnalysis, MDTraj, nglview, py3dmol, PROLIF)
- No CI config found (inferred: not yet configured)

## Commands That Work Today

- install: `uv sync --extra notebook --extra prep --extra viz --group dev`
- test: `uv run pytest`
- lint: `uv run ruff check src/mlx_atomistic/ tests/`
- bench (LJ MD): `uv run python -m mlx_atomistic.benchmarks.lj_md --particles 256 --steps 20`
- bench (MM forces): `uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 20 --json`
- bench (validation): `uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json`
- bench (stability): `uv run python -m mlx_atomistic.benchmarks.stability --json`
- bench (DFT SCF): `uv run python -m mlx_atomistic.benchmarks.dft_scf --sizes 8,16,24,32 --iterations 5 --mixer both --json`

## Apps, Packages, and Boundaries

- Single package: `mlx_atomistic` (src layout under `src/mlx_atomistic/`)
- `dft/` subpackage: self-contained DFT engine (SCF, XC, pseudopotentials, geometry optimization, band structure, stress)
- `prep/` subpackage: system preparation pipeline (imports, GPCRMD benchmarks, replica setup)
- `benchmarks/` subpackage: runnable validation and performance scripts
- `vendors/`: reference source trees only; explicitly excluded from imports (`CLAUDE.md:9`, `.gitignore`)
- No multi-app or monorepo structure; single installable package

## External Systems and Integrations

- MLX: primary compute backend (Apple Silicon GPU arrays, Metal kernels)
- OpenMM (dev dep): reference validation target for MD parity
- LAMMPS (dev dep, built from source with GPU/OpenCL): reference for MD benchmarks
- ASE, MDAnalysis, MDTraj (viz extra): trajectory format interop
- gemmi, parmed, rdkit (prep extra): molecular structure import
- GPCRMD: benchmark dataset for protein-ligand systems

## Existing Conventions

### Observed

- Source under `src/mlx_atomistic/` (hatchling src layout) — `pyproject.toml:68`
- `uv` for all Python execution — `AGENTS.md`, `CLAUDE.md`
- `vendors/` is reference-only, not imported — `CLAUDE.md:9`, `vendors/README.md`
- LJ reduced units for MD path — `README.md:42`, `docs/units.md`
- Sparse trajectory frames separate from dense per-step diagnostics — `README.md:42`
- Public API re-exported through `__init__.py` (332 lines of `__all__`) — `src/mlx_atomistic/__init__.py`
- Ruff excludes `vendors/` and `.venv/` — `pyproject.toml:75`

### Inferred

- No CI pipeline configured yet (no `.github/workflows/`, no `.gitlab-ci.yml`)
- Jupyter-first visualization philosophy (notebooks are first-class, not afterthought)
- Validation against established engines (OpenMM, LAMMPS, CP2K, QE) is a core quality gate

### Needs Confirmation

- Whether typecheck is enforced as a hard requirement or local convention (no `mypy` or `pyright` config found)
- Whether `mlx-atomistic-benchmark` CLI entry point is intended as the primary user-facing CLI or just an internal convenience

## Verification and Release Surfaces

- Test: `uv run pytest` (47 test files)
- Lint: `uv run ruff check src/mlx_atomistic/ tests/`
- Benchmarks: `python -m mlx_atomistic.benchmarks.*` with `--json` flag for structured output
- Validation suite: `mlx_atomistic.benchmarks.validation_gauntlet` and `stability`
- No release automation, CI, or formal versioning beyond `0.1.0` in `pyproject.toml`

## Likely Hotspots for the First Changes

- DFT subpackage is actively expanding (geometry optimization, band structure, stress, nonlocal pseudopotentials all recently added)
- Production MD validation against OpenMM GPCRMD reference is in active development (many scripts and production_md_ test fixtures)
- Prep pipeline for importing real molecular systems is under active development

## Sources Read

- `README.md` — project scope, setup, runtime boundary, benchmarks, layout
- `pyproject.toml` — dependencies, build config, test/lint config, entry points
- `src/mlx_atomistic/__init__.py` — public API surface (332 lines of exports)
- `src/mlx_atomistic/` directory listing — package structure (30 entries)
- `src/mlx_atomistic/dft/` directory listing — DFT subpackage (21 modules)
- `src/mlx_atomistic/prep/` directory listing — prep subpackage (14 modules)
- `src/mlx_atomistic/benchmarks/` directory listing — benchmark suite (18 modules)
- `tests/` directory listing — 47 test files
- `scripts/` directory listing — 13 reference/validation scripts
- `docs/` directory listing — 12+ documentation files
- `vendors/` directory listing — 5 reference source trees + lock file
- `.gitignore` — excluded paths, vendor policy, data artifacts
- `CLAUDE.md` / `AGENTS.md` — project guidance (same content)

## Open Questions

- Is `mlx-atomistic-benchmark` intended as a user-facing CLI or internal convenience?
- Is there a typecheck requirement beyond ruff?
- Is there a preferred CI provider for future automation?

## Import Verdict

- steering confidence: high
- recommended next skill: `auto-frame` (active change is `bootstrap` at stage `frame`)