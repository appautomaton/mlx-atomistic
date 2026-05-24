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
- Molecular mechanics in reduced and physical units: LJ, Coulomb, harmonic
  bonds/angles, periodic torsions, Ryckaert-Bellemans torsions, and bounded PME.
- Programmatic topology with exclusions, 1-4 scaling, and partial charges.
- Native prepared-system imports for accepted AMBER `prmtop`/`inpcrd`, CHARMM
  PSF/parameter, and GROMACS `.top`/`.gro` subsets.
- Jupyter-first visualization of structures, densities, orbitals, and SCF convergence.
- Small, validated examples before broader chemistry coverage.

See `docs/units.md` for the internal unit policy.

Low-level MD kernels still accept Lennard-Jones reduced-unit inputs unless a
caller converts values at the API boundary. Prepared-system artifacts use
explicit physical-unit metadata for imported force-field systems. Both paths keep
sparse trajectory frames separately from dense per-step diagnostics. NVE is
available for energy drift checks, and Langevin NVT is available for seeded
temperature-controlled experiments.

## Documentation

- `docs/molecular-mechanics.md`: topology, force-field terms, virtual sites,
  custom forces, GBSA/OBC, soft-core lambda scaling, and replica exchange.
- `docs/production-md.md`: prepared-system artifacts, accepted parser paths,
  TIP4P-Ew, PME, runtime readiness, and production MD gates.
- `docs/real-mm-core.md`: typed systems, force-field assignment, constraints,
  trajectory I/O, and the combined nonbonded path.
- `docs/validation-and-performance.md`: force validation, stability checks, and
  benchmark workflow.
- `docs/runtime-boundaries.md`: MLX runtime boundary and reference-engine policy.
- `docs/units.md`: internal unit policy.
- `docs/dft-foundations.md`: spin-unpolarized Γ-point plane-wave DFT prototype.
- `docs/dft-scf-core.md`: exchange-correlation, SCF mixing, force, and DFT
  benchmark surface.

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
