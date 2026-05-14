# Ligand-Receptor Motion Notebook

This notebook uses the GPCRmd MLX runtime as the active path:

```text
mlx_atomistic.prep imports or loads a strict GPCRmd prepared artifact
mlx_atomistic has already written the short-range prototype proof trajectory
the notebook loads, visualizes, and analyzes only the saved mlx_atomistic diagnostics
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

The default artifact is `data/gpcrmd-mlx/729-50steps-sample50/`, a 50-step
short-range prototype proof saved with `sample_interval=50`. It contains two
saved frames, the initial frame and the final step-50 frame. It is not a
production PME trajectory and should not be interpreted as binding/unbinding or
long-timescale GPCR dynamics.

The default target is the selected GPCRmd beta1 adrenergic receptor system.
Local cache paths can point at a GPCRmd manifest, package directory, or a directory that already contains
`prepared_system.json` and `prepared_system.npz`.

## Run

```bash
uv sync --extra viz --group dev
uv run jupyter lab notebooks/ligand-receptor-motion
```

The MLX notebook does not run MD. Regenerate the proof artifact outside the
notebook when needed:

```python
from mlx_atomistic.prep.runner import run_gpcrmd_mlx

run_gpcrmd_mlx(
    target_id="gpcrmd-729-beta1-5f8u-cyanopindolol",
    cache="/path/to/gpcrmd-cache-or-manifest",
    out="notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-50steps-sample50",
    steps=50,
    dt=0.0005,
    sample_interval=50,
    diagnostic_interval=50,
)
```

Viewer defaults show the ligand, receptor pocket, nearby waters/ions, and a
bounded membrane context instead of rendering all solvent/lipids at once.
Regenerate with `--sample-interval 1` and a different output dataset name if
you want the notebook animation to include every integration step.

## OpenMM Preview

`02-openmm-2000-step-preview.ipynb` loads the local `openmm-reference` preview artifact at
`data/openmm-preview/729-2000-opencl-restrained-preview/`. It is a 2000-step
OpenMM `OpenCL` restrained visualization smoke run sampled every 20 steps, with
the ligand restraint target translated 8 A along x so ligand motion is visible.
It is a reference preview, not production runtime output, force-field parity, or
a production GPCRmd protocol.

Regenerate it with:

```bash
uv run python scripts/run_openmm_gpcrmd_preview.py \
  --out notebooks/ligand-receptor-motion/data/openmm-preview/729-2000-opencl-restrained-preview \
  --steps 2000 \
  --sample-interval 20 \
  --platform OpenCL \
  --dt-ps 0.0005 \
  --temperature 300 \
  --friction 1 \
  --restraint-k 1000 \
  --ligand-restraint-k 50000 \
  --ligand-translation-A 8 0 0 \
  --nonbonded-mode none
```

## OpenMM CHARMM/PME MD Preview

`03-openmm-charmm-pme-md-preview.ipynb` loads a dense short full-MD `openmm-reference`
artifact at `data/openmm-md/729-200-opencl-charmm-pme-dense-preview/`. It uses
the GPCRmd PSF/PDB/PRM inputs through OpenMM's CHARMM loaders with PME
electrostatics, HBond constraints, rigid water, and Langevin dynamics. The
default trial is 200 steps sampled every 2 steps, giving 101 saved frames for a
lightweight production-physics reference preview. It is not production runtime
output from `mlx_atomistic`.

Regenerate it with:

```bash
uv run python scripts/run_openmm_gpcrmd_charmm_md.py \
  --out notebooks/ligand-receptor-motion/data/openmm-md/729-200-opencl-charmm-pme-dense-preview \
  --steps 200 \
  --sample-interval 2 \
  --platform OpenCL \
  --dt-ps 0.00025 \
  --temperature 310 \
  --friction 0.1 \
  --minimize-steps 500 \
  --positions-source prepared
```

`04-openmm-50k-sparse-preview.ipynb` loads the longer sparse `openmm-reference` artifact at
`data/openmm-md/729-50000-opencl-charmm-pme-sample11/`. It keeps the 50,000-step
full CHARMM/PME run but samples only every 5000 steps, giving 11 saved frames.
This is the lighter notebook to open when you want the 5 ps run without loading
the denser 101-frame visualization artifact.

Regenerate it with:

```bash
uv run python scripts/run_openmm_gpcrmd_charmm_md.py \
  --out notebooks/ligand-receptor-motion/data/openmm-md/729-50000-opencl-charmm-pme-sample11 \
  --steps 50000 \
  --sample-interval 5000 \
  --platform OpenCL \
  --dt-ps 0.0001 \
  --temperature 310 \
  --friction 0.1 \
  --minimize-steps 2000 \
  --positions-source prepared
```

Generated outputs under `data/gpcrmd-mlx/`, `data/openmm-preview/`,
`data/openmm-md/`, `data/cache/`, `data/prepared/`, and `data/processed/` are
ignored by git.
