# Notebooks

Use the project `uv` environment:

```bash
uv sync --extra notebook --extra viz --group dev
uv run jupyter lab
```

The active workflow notebooks live under `notebooks/workflows/`. They are meant
to be compact lab workbooks: narrative first, equations where useful, then small
executable cells with plots and diagnostics.

`notebooks/ligand-receptor-motion/` is the active macromolecular visualization
workflow. Its primary path imports or loads a strict GPCRmd prepared artifact,
runs the short NVT proof with `mlx_atomistic`, then visualizes and analyzes only
that `mlx_atomistic` trajectory. OpenMM preview notebooks in that directory are
`openmm-reference` workflows for comparison and visualization smoke checks; they
are not production runtime output. GPCRmd reference trajectories are comparison
context only.

## Workflow Notebooks

- `workflows/01-md-molecular-mechanics.ipynb`  
  Typed topology, bonded/nonbonded force composition, constraints, NVE energy
  drift, energy decomposition, and a water-like trajectory visualization.

- `workflows/02-md-validation-performance.ipynb`  
  Finite-difference force validation, LJ liquid NVE diagnostics, neighbor-list
  pair counts, rebuild counts, temperature, and drift.

- `workflows/03-dft-density-scf.ipynb`  
  Kohn-Sham density on a real-space grid, effective-potential slices, SCF
  energy/residual traces, orbital residuals, and energy decomposition.

- `workflows/04-dft-pseudopotentials-nonlocal.ipynb`  
  UPF/GTH parsing, radial local potentials, nonlocal projector inspection, and
  local-only versus local+nonlocal SCF diagnostics.

- `workflows/05-dft-solvers-spin-kpoints.ipynb`  
  Dense reference versus Davidson-style solving, fractional occupations,
  collinear spin-density diagnostics, k-point meshes, and non-SCF band plots.

- `workflows/06-dft-relaxation-reference.ipynb`  
  Geometry relaxation history, finite-difference stress, dense SCF restarts, and
  static reference comparison plumbing.

## Archive

`archive/atp-pocket-mlx-demo/` contains the old ATP/P2X4 pocket notebook as
historical reference. It is no longer the active macromolecule visualization
workflow.

`archive/milestone-trace/` contains the old numbered notebooks from earlier
milestone development. They are retained as provenance, not as the recommended
learning path.

## Regenerating

The curated workflow notebooks are generated from:

```bash
uv run python scripts/notebooks/rebuild_workflow_notebooks.py
```
