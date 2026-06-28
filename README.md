<p align="center">
  <a href="https://appautomaton.github.io/mlx-atomistic/">
    <img src="https://appautomaton.github.io/mlx-atomistic/og.png" alt="mlx-atomistic — Apple Silicon-native molecular dynamics and DFT runtime on MLX and Metal" width="760">
  </a>
</p>

<h1 align="center">mlx-atomistic</h1>

<p align="center">
  Apple&nbsp;Silicon-native alpha <b>molecular dynamics</b> and <b>density-functional-theory</b> runtime,
  built directly on <a href="https://github.com/ml-explore/mlx">MLX</a> and Metal —<br>
  it runs the GPU on your Mac, with no CUDA, server, or cloud.
</p>

<p align="center">
  <a href="https://appautomaton.github.io/mlx-atomistic/"><img alt="Documentation" src="https://img.shields.io/badge/docs-appautomaton.github.io-6c5ce7?style=flat-square&logo=readthedocs&logoColor=white"></a>
  <a href="https://github.com/appautomaton/mlx-atomistic/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/appautomaton/mlx-atomistic/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.13" src="https://img.shields.io/badge/python-3.13-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1d1d1f?style=flat-square">
  <a href="https://appautomaton.github.io"><img alt="Part of App Automaton" src="https://img.shields.io/badge/part%20of-App%20Automaton-6c5ce7?style=flat-square"></a>
</p>

<p align="center">
  <b><a href="https://appautomaton.github.io/mlx-atomistic/">Documentation</a></b> ·
  <a href="https://appautomaton.github.io/mlx-atomistic/overview/">Overview</a> ·
  <a href="https://appautomaton.github.io/mlx-atomistic/mm/molecular-mechanics/">Molecular mechanics</a> ·
  <a href="https://appautomaton.github.io/mlx-atomistic/dft/dft-scf-core/">DFT</a> ·
  <a href="https://appautomaton.github.io/mlx-atomistic/api/">API reference</a> ·
  <a href="https://appautomaton.github.io/mlx-atomistic/benchmarks/">Benchmarks</a>
</p>

---

## What is mlx-atomistic?

**mlx-atomistic is an experimental Apple Silicon-native runtime for molecular dynamics (MD)
and density functional theory (DFT)**, built directly on Apple's [MLX](https://github.com/ml-explore/mlx)
array framework and the Metal GPU backend. It runs the simulation kernels on the
GPU in your Mac — no CUDA, no remote cluster, no cloud. The `0.0.1` package is
a strict alpha preview: early plane-wave DFT building blocks, molecular-mechanics
force terms, prepared-system imports, and Jupyter-first visualization.

`mlx_atomistic` is the primary trajectory generator and product runtime in this
repo. OpenMM, LAMMPS, and the source trees under `vendors/` are reference and
validation surfaces only — they never replace the MLX runtime path.

## Features

- **Apple Silicon native** — MLX arrays execute through the Metal backend on
  your machine. No CUDA, no server, no cloud.
- **Plane-wave DFT building blocks** — Γ-point Kohn–Sham SCF, LDA plus
  public-alpha PBE-PZ81 GGA diagnostics, non-SCF k-point/band diagnostics,
  Davidson/preconditioned-residual eigensolver diagnostics, and GTH/UPF
  pseudopotentials with proof-level local + nonlocal projector diagnostics.
- **Molecular-mechanics building blocks** — Lennard-Jones, Coulomb, harmonic
  bonds and angles, periodic + Ryckaert-Bellemans torsions, NVE/Langevin NVT,
  bounded PME proof surfaces, and proof-level barostat/NPT diagnostics.
- **Prepared-system imports** — AMBER `prmtop`/`inpcrd`, CHARMM PSF/parameter, and
  GROMACS `.top`/`.gro` subsets, with explicit physical-unit metadata.
- **Reference-validation ready** — OpenMM and LAMMPS surfaces are opt-in local
  validation lanes, not package/runtime dependencies.
- **Self-documenting** — Google-style docstrings generate the [API reference](https://appautomaton.github.io/mlx-atomistic/api/)
  and an [`llms.txt`](https://appautomaton.github.io/mlx-atomistic/llms.txt) for agentic tools.

## Quick start

Install the alpha package from PyPI into a Python 3.13 environment:

```bash
uv run --no-project --python 3.13 --with mlx-atomistic \
  python -c "import mlx_atomistic as ma; print(ma.__version__)"
```

For notebook or development work from a checkout:

```bash
uv venv --python 3.13
uv sync --extra notebook --extra prep --extra viz
uv run python -m ipykernel install --user --name mlx-atomistic --display-name "mlx-atomistic"
uv run jupyter lab
```

Plain `uv sync` uses the light test group by default. OpenMM and LAMMPS are
installed only when you explicitly request the `reference` or `dev` group.

If `uv` cannot use the home cache in a sandboxed run, point it at a writable cache:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv sync --extra notebook --extra prep --extra viz
```

## Benchmarks

```bash
uv run python -m mlx_atomistic.benchmarks.lj_md --particles 256 --steps 20
uv run python -m mlx_atomistic.benchmarks.lj_md --sizes 128,512,2048 --steps 20 --json
uv run python -m mlx_atomistic.benchmarks.mm_force_terms --evaluations 20 --json
uv run python -m mlx_atomistic.benchmarks.validation_gauntlet --json
uv run python -m mlx_atomistic.benchmarks.stability --json
uv run python -m mlx_atomistic.benchmarks.dft_scf --sizes 8,16,24,32 --iterations 5 --mixer both --json
```

## Documentation

The full documentation — narrative guides plus an auto-generated API reference —
lives at **[appautomaton.github.io/mlx-atomistic](https://appautomaton.github.io/mlx-atomistic/)**:

- [Overview](https://appautomaton.github.io/mlx-atomistic/overview/) — what the runtime is and how the pieces fit.
- [Molecular mechanics](https://appautomaton.github.io/mlx-atomistic/mm/molecular-mechanics/) — topology, force-field terms, virtual sites, GBSA/OBC, soft-core λ, replica exchange.
- [Density functional theory](https://appautomaton.github.io/mlx-atomistic/dft/dft-scf-core/) — plane-wave SCF, exchange-correlation, mixing, forces.
- [API reference](https://appautomaton.github.io/mlx-atomistic/api/) — generated directly from the package docstrings.
- [Benchmarks](https://appautomaton.github.io/mlx-atomistic/benchmarks/) — validation gauntlet, stability, and performance methodology.

The narrative source lives in [`docs/`](docs/) and the site itself in [`site/`](site/);
the API pages are regenerated from the package on every deploy, so they never drift
from the code.

## Runtime boundary

`mlx_atomistic` is the product runtime. Low-level MD kernels accept Lennard-Jones
reduced-unit inputs unless a caller converts at the API boundary; prepared-system
artifacts carry explicit physical-unit metadata. Sparse trajectory frames are kept
separately from dense per-step diagnostics. NVE is available for energy-drift
checks and Langevin NVT for seeded temperature-controlled runs. See
[`docs/runtime-boundaries.md`](docs/runtime-boundaries.md) for the dependency roles
and reference-engine provenance, and [`docs/units.md`](docs/units.md) for the unit policy.

## Part of App Automaton

mlx-atomistic is part of **[App Automaton](https://appautomaton.github.io)** — open
skills, harnesses, and on-device tools for engineering with AI coding agents, with a
focus on local-first, Apple-Silicon-native execution.

## License

[MIT](LICENSE).
