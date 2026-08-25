# Overview

`mlx_atomistic` is an experimental Apple Silicon-native atomistic simulation
runtime built on MLX and Metal. It targets Python 3.13 through `uv` and uses MLX
for local GPU execution on Apple Silicon.

## Runtime Boundary

`mlx_atomistic` is the primary trajectory and electronic-structure runtime in
this repository. OpenMM, LAMMPS, and source trees under `vendors/` are reference
or validation surfaces. They never replace the MLX runtime path.

## Two Scales, One Runtime

- **Density Functional Theory** provides a legacy Gamma-point teaching and
  reference path plus a separate periodic PBE-GTH implementation with
  Monkhorst-Pack integration, block-Davidson solves, frozen-density bands, and
  periodic forces. Verified claims remain limited to published silicon,
  carbon, and magnesium-oxide workloads.
- **Molecular Mechanics** provides Lennard-Jones and Coulomb nonbonded work,
  bonded force fields, bounded Particle Mesh Ewald, constrained fixed-cell
  constant-particle-number, volume, and energy dynamics, and Langevin
  constant-particle-number, volume, and temperature dynamics.

## Install the Alpha

```bash
uv run --no-project --python 3.13 --with mlx-atomistic \
  python -c "import mlx_atomistic as ma; print(ma.__version__)"
```

Checkout extras remain optional: `prep` for topology import, `viz` for
visualization, and `notebook` for Jupyter.

```bash
uv venv --python 3.13
uv sync --extra notebook --extra prep --extra viz
uv run python -m ipykernel install --user \
  --name mlx-atomistic --display-name "mlx-atomistic"
uv run jupyter lab
```

If a sandbox cannot use the home cache, select a writable cache:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache \
  uv sync --extra notebook --extra prep --extra viz
```

## Documentation Map

- [Runtime boundaries](./runtime-boundaries.md)
- [Testing](./testing.md)
- [Molecular Mechanics](./molecular-mechanics.md)
- [Production MD](./production-md.md)
- [MD acceleration](./md-acceleration.md)
- [Experimental Metal interaction engine](./metal-interaction-engine.md)
- [DFT foundations](./dft-foundations.md)
- [DFT production core](./dft-production-core.md)
- [DFT roadmap](./dft-roadmap.md)
- [Validation and performance](./validation-and-performance.md)
- [Benchmarks](./benchmarks/README.md)
- [Release checklist](./release.md)
