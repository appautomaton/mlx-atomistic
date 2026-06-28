---
title: Overview
description: Apple Silicon-native atomistic simulation with MLX and Metal.
---

mlx-atomistic is an experimental Apple Silicon-native atomistic simulation
runtime built on MLX and Metal. It targets Python 3.13 through `uv` and uses
MLX for local GPU execution on Apple Silicon.

## What it is

`mlx_atomistic` is the primary trajectory generator and product runtime in this
repo. OpenMM, LAMMPS, and the source trees under `vendors/` are reference or
validation surfaces; they do not replace the MLX runtime path.

## Two scales, one runtime

- **Density Functional Theory** — spin-unpolarized Γ-point plane-wave Kohn-Sham
  SCF, LDA plus public-alpha PBE-PZ81 GGA diagnostics, non-SCF k-point/band
  diagnostics, pseudopotentials (GTH / UPF), forces, stress, and geometry
  optimization prototypes.
- **Molecular Mechanics** — Lennard-Jones, Coulomb, harmonic bonds/angles,
  periodic + Ryckaert–Bellemans torsions, bounded PME, NVE and Langevin NVT.

## First milestones

Lightweight DFT building blocks, validation notebooks, and visualization
utilities rather than a heavy production DFT engine. Small, validated examples
before broader chemistry coverage.

## Try the PyPI Alpha

```bash
uv run --no-project --python 3.13 --with mlx-atomistic \
  python -c "import mlx_atomistic as ma; print(ma.__version__)"
```

Extras are opt-in for checkout workflows: `prep` for topology/prep imports,
`viz` for visualization, and `notebook` for Jupyter.

## Checkout Setup

```bash
uv venv --python 3.13
uv sync --extra notebook --extra prep --extra viz
uv run python -m ipykernel install --user --name mlx-atomistic --display-name "mlx-atomistic"
uv run jupyter lab
```

If `uv` cannot use the home cache in a sandboxed run, use a writable cache:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv sync --extra notebook --extra prep --extra viz
```

## Where to go next

- [Foundations](/mlx-atomistic/foundations/) — units, runtime boundaries, testing
- [DFT](/mlx-atomistic/dft/) — SCF core, pseudopotentials, numerics
- [Molecular Mechanics](/mlx-atomistic/mm/) — topology, force fields, production MD
- [Benchmarks](/mlx-atomistic/benchmarks/) — validation and performance results
