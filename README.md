# mlx-atomistic

Apple Silicon-native atomistic simulation experiments built on MLX and Metal.

This project targets Python 3.13 through `uv` and uses MLX for local GPU execution on Apple Silicon. The first milestones are lightweight DFT building blocks, validation notebooks, and visualization utilities rather than a heavy production DFT engine.

## Setup

```bash
uv venv --python 3.13
uv sync --extra notebook --extra viz --group dev
uv run python -m ipykernel install --user --name mlx-atomistic --display-name "mlx-atomistic"
uv run jupyter lab
```

If `uv` cannot use the home cache in a sandboxed run, use a writable cache:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv sync --extra notebook --extra viz --group dev
```

## Initial Scope

- MLX-backed arrays and kernels for atomistic/DFT experiments.
- Lennard-Jones molecular dynamics in reduced units as the first engine slice.
- Jupyter-first visualization of structures, densities, orbitals, and SCF convergence.
- Small, validated examples before broader chemistry coverage.

See `docs/units.md` for the internal unit policy.

The MD path currently uses Lennard-Jones reduced units and keeps sparse trajectory
frames separately from dense per-step diagnostics, so energy drift can be checked
without storing every position frame.

## Benchmarks

```bash
uv run python -m mlx_atomistic.benchmarks.lj_md --particles 256 --steps 20
```

## Layout

- `src/mlx_atomistic/`: package code.
- `notebooks/`: exploratory notebooks and visual validation.
- `tests/`: focused unit tests.
- `vendors/`: reference source trees; not imported as project code.
