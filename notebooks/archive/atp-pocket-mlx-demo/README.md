# Archived ATP-Pocket MLX Demo

Historical notebook-first example for the ATP/P2X4 pocket MLX demo. The active
macromolecular visualization workflow now lives in
`notebooks/ligand-receptor-motion/`.

Use the optional notebook and visualization extras:

```bash
uv sync --extra notebook --extra viz
uv run jupyter lab notebooks/archive/atp-pocket-mlx-demo
```

## Contents

- `01-jupyter-macromolecule-visualization.ipynb`  
  py3Dmol for static structures, Plotly for preloaded trajectory animation, MDAnalysis for
  ATP-receptor contact traces and residue tables. For the bundled 4DW1 example,
  the notebook builds the internal production MLX artifact if it is missing or
  stale, runs MLX when `data/prepared/4dw1-atp/trajectory.npz` is missing or
  stale, then animates and analyzes that saved trajectory. The trajectory view
  uses one preloaded Plotly player for the trajectory with visible controls,
  keeps a translucent cyan ATP copy at frame 0 as a static reference pose, and
  plots ATP center-of-mass motion relative to the receptor pocket.
- `helpers/`  
  Local notebook helper modules for configuration, static py3Dmol views,
  production artifact/run workflow, Plotly trajectory preview, and analysis.
  These are kept beside the notebook so the notebook remains readable while
  still running from either repo-root or notebook-local working directories.

The checked-in example data includes:

- `data/4dw1_atp_bound_p2x4.pdb`: ATP-bound P2X4 receptor coordinates from RCSB.
- `data/atp_pubchem_5957_3d.sdf`: real 3D ATP ligand from PubChem CID 5957.

The same path can be reproduced from the CLI:

```bash
uv run atomistic-prep prepare-p2x4-atp --backend production_mlx \
  --pdb notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb \
  --out notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp --force
uv run atomistic-prep validate --prepared notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp --require-production
uv run atomistic-prep run-mlx --prepared notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp \
  --require-production --steps 10000 --sample-interval 25 --dt 0.002 \
  --temperature 300 --friction 10 --restraint-k 5 \
  --minimize-steps 50 --equilibration-steps 100 \
  --constraint-max-iterations 4 --diagnostic-interval 25 --force
uv run atomistic-prep benchmark-p2x4-atp \
  --prepared notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp \
  --durations-ps 1 10 20
```

Raw PDB coordinates are not general production MD input. The bundled 4DW1
example works because `atomistic_prep` has versioned internal templates for that
specific ATP pocket: explicit hydrogens, atom types, charges, bonded terms,
constraints, and nonbonded exceptions. Other systems should come from AMBER or
CHARMM topology/parameter import.
