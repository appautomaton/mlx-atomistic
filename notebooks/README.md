# Notebooks

Use the `mlx-atomistic` Jupyter kernel from the project `uv` environment.

Planned notebooks:

- `00-runtime-and-visualization.ipynb`: verify MLX/Metal, render atoms, and plot simple grids.
- `01-lj-md.ipynb`: run and plot a small Lennard-Jones MD trajectory.
- `02-lj-md-performance.ipynb`: compare all-pairs and neighbor-list LJ force paths.
- `03-molecular-mechanics.ipynb`: inspect bonded terms, charged systems, and energy decomposition.
- `04-validation-performance.ipynb`: run force validation, stability diagnostics, and benchmark summaries.
- `05-real-mm-core.ipynb`: build typed MM systems, apply constraints, save/load trajectories, and compare force paths.
- `06-density-grid.ipynb`: inspect toy charge-density arrays and slices.
- `07-scf-convergence.ipynb`: plot initial SCF iteration traces.
- `08-xc-and-mixing.ipynb`: compare exchange-only and LDA XC with linear and DIIS mixing.
- `09-dft-forces.ipynb`: compare local Gaussian pseudopotential forces with finite differences.
- `10-dft-performance.ipynb`: inspect DFT timing breakdowns and small grid scaling.
- `11-ks-operator.ipynb`: inspect Kohn-Sham operator components on a compact grid.
- `12-dense-vs-operator.ipynb`: compare dense reference matrix-vector products with operator application.
- `13-scf-force-consistency.ipynb`: compare SCF total-energy finite-difference forces with reported forces.
