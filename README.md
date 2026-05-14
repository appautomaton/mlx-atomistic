# mlx-atomistic

Apple Silicon-native atomistic simulation experiments built on MLX and Metal.

This project targets Python 3.13 through `uv` and uses MLX for local GPU execution on Apple Silicon. The first milestones are lightweight DFT building blocks, validation notebooks, and visualization utilities rather than a heavy production DFT engine.

## Runtime Boundary

`mlx_atomistic` is the primary trajectory generator and product runtime in this
repo. OpenMM, LAMMPS, and the source trees under `vendors/` are reference or
validation surfaces; they do not replace the MLX runtime path.

See `docs/runtime-boundaries.md` for the dependency roles and current
OpenMM/LAMMPS provenance.

## Setup

```bash
uv venv --python 3.13
uv sync --extra notebook --extra prep --extra viz --group dev
uv run python -m ipykernel install --user --name mlx-atomistic --display-name "mlx-atomistic"
uv run jupyter lab
```

If `uv` cannot use the home cache in a sandboxed run, use a writable cache:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv sync --extra notebook --extra prep --extra viz --group dev
```

## Initial Scope

- MLX-backed arrays and kernels for atomistic/DFT experiments.
- Molecular mechanics in reduced units: LJ, Coulomb, harmonic bonds/angles, and periodic torsions.
- Programmatic topology with exclusions, 1-4 scaling, and partial charges.
- Jupyter-first visualization of structures, densities, orbitals, and SCF convergence.
- Small, validated examples before broader chemistry coverage.

See `docs/units.md` for the internal unit policy.

The MD path currently uses Lennard-Jones reduced units and keeps sparse trajectory
frames separately from dense per-step diagnostics. NVE is available for energy
drift checks, and Langevin NVT is available for seeded temperature-controlled
experiments.

See `docs/molecular-mechanics.md` for the topology and force-field surface.
See `docs/validation-and-performance.md` for the force-validation, stability,
and benchmark workflow.
See `docs/real-mm-core.md` for typed systems, force-field assignment,
constraints, trajectory I/O, and the combined nonbonded path.
See `docs/dft-foundations.md` for the spin-unpolarized Γ-point plane-wave DFT
prototype.
See `docs/dft-scf-core.md` for the exchange-correlation, SCF mixing, force, and
DFT benchmark surface.

## Benchmarks

```bash
uv run python -m mlx_atomistic.benchmarks.lj_md --particles 256 --steps 20
uv run python -m mlx_atomistic.benchmarks.lj_md --sizes 128,512,2048 --steps 20 --json
uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 20 --json
uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json
uv run python -m mlx_atomistic.benchmarks.stability --json
uv run python -m mlx_atomistic.benchmarks.dft_scf --sizes 8,16,24,32 --iterations 5 --mixer both --json
```

## Layout

- `src/mlx_atomistic/`: core MLX/Metal simulation package code.
- `src/mlx_atomistic/prep/`: preparation/import tooling for MLX-ready artifacts.
- `notebooks/`: exploratory notebooks and visual validation.
- `tests/`: focused unit tests.
- `vendors/`: reference source trees; not imported as project code.
