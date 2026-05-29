# Project

## One-Liner

Apple Silicon-native atomistic simulation library: MLX/Metal-accelerated molecular mechanics, MD integrators, and plane-wave DFT with validation against established engines.

## Why This Repo Exists

- Provide a GPU-native simulation library that runs entirely on Apple Silicon via MLX, without depending on CUDA or x86 HPC clusters — `README.md:3-5`
- Validate simulation correctness against established engines (OpenMM, LAMMPS, CP2K, Quantum ESPRESSO) rather than reimplementing from scratch — `README.md:10-14`, `tests/test_openmm_mlx_parity.py`, `scripts/openmm_mlx_parity.py`
- Offer an experiment-first, Jupyter-friendly surface for atomistic simulations with incremental milestones — `README.md:36-37`

## Current Users or Operators

- Researchers running MD and DFT experiments on Apple Silicon machines
- Developers benchmarking MLX MD implementations against OpenMM and LAMMPS references
- Notebook users exploring structures, densities, orbitals, and SCF convergence

## Current System Model

- request or event flow: Python API call → MLX array construction → Metal kernel dispatch → result arrays back to Python
- primary surfaces: `mlx_atomistic` package (MM forces, MD integrators, DFT/SCF, protocols, I/O, validation)
- critical dependencies: MLX (GPU compute), NumPy/SciPy (fallback和支持), OpenMM + LAMMPS (dev-only reference validation)

## Major Surfaces

| Surface | Path | Responsibility |
|---------|------|----------------|
| Core (Atoms, Cell, units) | `src/mlx_atomistic/core.py`, `units.py` | Primitive types and unit system |
| MM force fields | `src/mlx_atomistic/mm.py`, `forcefields.py`, `charmm_terms.py` | Force field definitions, bonded/nonbonded terms |
| Topology | `src/mlx_atomistic/topology.py` | Programmatic topology with exclusions, 1-4 scaling, partial charges |
| Nonbonded + PME | `src/mlx_atomistic/nonbonded.py`, `pme.py`, `neighbors.py`, `cell_list.py` | Neighbor lists, Ewald/PME electrostatics |
| MD integrators | `src/mlx_atomistic/md.py`, `minimize.py`, `steering.py` | NVE, Langevin NVT, energy minimization, steered MD |
| DFT engine | `src/mlx_atomistic/dft/` (21 modules) | Plane-wave DFT: SCF, XC, pseudopotentials, geometry optimization, band structure, stress |
| Protocols | `src/mlx_atomistic/protocols.py`, `initialize.py` | Higher-level MD workflows (MinimizeThenNVT) |
| I/O + trajectories | `src/mlx_atomistic/io.py`, `trajectory_adapters.py`, `artifacts.py` | Trajectory I/O, MDTraj/MDAnalysis adapters, artifact prep |
| Validation | `src/mlx_atomistic/validation.py`, `diagnostics.py` | Force validation, platform validation, energy drift checks |
| Visualization | `src/mlx_atomistic/visualization.py`, `examples.py` | Jupyter-first structure/density/orbital rendering |
| Prep pipeline | `src/mlx_atomistic/prep/` (14 modules) | GPCRMD import, PDB, topology import, replica setup |
| Benchmarks | `src/mlx_atomistic/benchmarks/` (18 modules) | Performance and validation CLI scripts |

## Stack Summary

- Python 3.13, MLX (Metal GPU), NumPy, SciPy
- Build: hatchling, managed by `uv`
- Test: pytest, Lint: ruff
- Dev reference engines: OpenMM 8.5+, LAMMPS (built from source with GPU/OpenCL)

## Commands

- install: `uv sync --extra notebook --extra prep --extra viz --group dev`
- test: `uv run pytest`
- lint: `uv run ruff check src/mlx_atomistic/ tests/`

## Decision Principles Already Visible In The Repo

- Apple Silicon first: MLX/Metal is the compute path; no CUDA dependency — `README.md:3`
- Reduced units for MD: LJ reduced units as default unit system — `README.md:42`, `docs/units.md`
- Validation against established engines: every force term and integrator has an OpenMM or LAMMPS parity check — `tests/test_openmm_mlx_parity.py`, `tests/test_force_scopes.py`
- Incremental milestones: small validated examples before broader coverage — `README.md:37`
- Vendor reference only: `vendors/` trees are comparison targets, not import dependencies — `CLAUDE.md:9`
- No heavyweight chemistry packages without concrete need — `CLAUDE.md:10`

## Evidence Anchors

- `src/mlx_atomistic/__init__.py` — 332-line public API surface defining the contract
- `pyproject.toml` — single installable package, pinned Python, MLX dependency
- `tests/` — 47 test files covering MM, MD, DFT, validation, I/O, protocols
- `README.md` — scope, runtime boundary, benchmarks, layout