# Ligand-Receptor Motion Notebook

This notebook uses the GPCRmd MLX runtime as the active path:

```text
atomistic_prep imports or loads a strict GPCRmd prepared artifact
mlx_atomistic has already written the short-range prototype proof trajectory
the notebook loads, visualizes, and analyzes only the saved MLX trajectory diagnostics
```

No OpenMM, LAMMPS, GROMACS, or other external MD engine is run. GPCRmd
reference trajectories are comparison context only; they are not notebook
results.

## Active Example

The notebook loads `trajectory.npz` only when `gpcrmd_mlx_run_report.json`
records a completed `run_gpcrmd_mlx` result and `trajectory.npz` metadata has
`engine="mlx_atomistic"` and `workflow="run_gpcrmd_mlx"`.
When inputs are missing or the GPCRmd artifact is blocked, the notebook displays
the blocker JSON and stops before trajectory visualization.

The default artifact is `data/gpcrmd-mlx/729-proof/`, a 1-step, 2-frame
short-range prototype proof. It is not a production PME trajectory and should
not be interpreted as binding/unbinding or long-timescale GPCR dynamics.

The default target is the selected GPCRmd beta1 adrenergic receptor system
tracked by `atomistic-prep run-gpcrmd-mlx`. Local cache paths can point at a
GPCRmd manifest, package directory, or a directory that already contains
`prepared_system.json` and `prepared_system.npz`.

## Run

```bash
uv sync --extra viz --group dev
uv run jupyter lab notebooks/ligand-receptor-motion
```

The notebook does not run MD. Regenerate the proof artifact outside the notebook
when needed:

```bash
uv run atomistic-prep run-gpcrmd-mlx \
  --target gpcrmd-729-beta1-5f8u-cyanopindolol \
  --cache /path/to/gpcrmd-cache-or-manifest \
  --out notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof \
  --steps 1 \
  --dt 0.0005 \
  --sample-interval 1
```

Viewer defaults show the ligand, receptor pocket, nearby waters/ions, and a
bounded membrane context instead of rendering all solvent/lipids at once.

Generated outputs under `data/gpcrmd-mlx/`, `data/cache/`, `data/prepared/`,
and `data/processed/` are ignored by git.
