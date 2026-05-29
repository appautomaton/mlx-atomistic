# Requirements

## Product Commitments

### Observed

- MLX/Metal is the sole GPU compute backend; CUDA is not a target — `README.md:3`, `pyproject.toml:16` (`mlx>=0.31.1`)
- MD uses Lennard-Jones reduced units by default; unit policy documented in `docs/units.md` — `README.md:42`
- Sparse trajectory frames are stored separately from dense per-step diagnostics — `README.md:42`
- Jupyter-first visualization for structures, densities, orbitals, and SCF convergence — `README.md:36`
- Public API is re-exported through `__init__.py` with an explicit `__all__` — `src/mlx_atomistic/__init__.py`
- DFT path provides spin-unpolarized Γ-point plane-wave SCF with geometry optimization and band structure — `README.md:52-53`

### Inferred

- The package aims to serve both interactive notebook users (experiment-driven) and programmatic API users (protocol-driven)
- Production MD against real molecular systems (GPCRMD) is a quality gate, not just unit tests

### Needs Confirmation

- Whether the CLI entry point (`mlx-atomistic-benchmark`) is intended to grow into a full user-facing CLI or remains internal

## Technical Constraints and Invariants

### Observed

- Python 3.13 only (pinned `>=3.13,<3.14`) — `pyproject.toml:10`
- `uv` is the only supported environment and package manager — `AGENTS.md`, `CLAUDE.md`
- Source code lives under `src/mlx_atomistic/` (hatchling src layout) — `pyproject.toml:68`
- `vendors/` is reference material only, not imported as project code — `CLAUDE.md:9`
- No heavyweight chemistry or ML helper packages without concrete need — `CLAUDE.md:10`
- Ruff is the linter with rules E, F, I, UP, B, SIM; line-length 100, target py313 — `pyproject.toml:74-80`
- LAMMPS is built from source with GPU/OpenCL enabled via `uv` config-settings — `pyproject.toml:63-65`

### Inferred

- MLX array operations must stay on-device to preserve GPU performance; unnecessary host transfers are a correctness concern
- Test coverage of DFT is advancing rapidly (test_dft.py, test_dft_optimization.py, test_dft_production_core.py, etc.)

### Needs Confirmation

- Type checking enforcement level (no mypy/pyright config found)

## Quality and Operational Expectations

- testing bar: `uv run pytest` across 47 test files; validation gauntlet (`benchmarks.validation_gauntlet`) and stability checks (`benchmarks.stability`) for numerical correctness
- release or deployment constraint: no CI, no release automation observed; currently `0.1.0` pre-release
- security or reliability expectation: no secrets in repo; `.gitignore` excludes `.env`, `.env.*`; data artifacts (`*.npy`, `*.npz`, `*.h5`, etc.) are gitignored

## Integration Boundaries

- upstream systems: MLX (Apple), NumPy, SciPy
- downstream consumers: Jupyter notebooks, Python scripts
- external contracts that cannot break: public API exports in `__init__.py` `__all__`; `TrajectoryRecord` I/O format; force validation parity with OpenMM and LAMMPS
- reference engines (dev-only): OpenMM ≥8.5, LAMMPS (GPU/OpenCL), CP2K, GROMACS, Quantum ESPRESSO — `vendors/vendor-lock.json`, `vendors/README.md`

## Non-Goals

- CUDA or non-Apple-Silicon GPU support (inferred from MLX dependency and README framing)
- Heavyweight chemistry/ML packages as core dependencies (explicit: `CLAUDE.md:10`)
- Importing or building against vendor source trees (explicit: `CLAUDE.md:9`)
- CI/CD or automated release pipeline (no config found, not currently in scope)

## Open Risks and Unknowns

- DFT subpackage is expanding fast; API surface may still be stabilizing
- Production MD benchmarking against GPCRMD reference is actively being developed (many new test fixtures)
- No CI pipeline; correctness relies on local test runs
- Typecheck is not enforced in the toolchain

## Evidence Anchors

- `pyproject.toml` — dependency pins, build config, test/lint config
- `src/mlx_atomistic/__init__.py` — public API contract
- `tests/` — 47 test files across MM, MD, DFT, validation, I/O, protocols
- `CLAUDE.md` / `AGENTS.md` — project guidance rules
- `.gitignore` — excluded artifacts and vendor policy