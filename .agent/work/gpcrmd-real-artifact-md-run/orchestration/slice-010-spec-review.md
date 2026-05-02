# Slice 010 Spec Review

- Slice: Notebook GPCRmd Artifact Consumer
- Status: approved

## Findings

- The notebook shows blocker JSON and stops before visualization when a usable MLX GPCRmd trajectory is unavailable.
- Trajectory loading is restricted to `mlx_atomistic` GPCRmd workflow metadata.
- Viewer defaults avoid rendering all solvent/lipids and instead select receptor pocket, nearby waters/ions, and bounded membrane context.
- Analysis uses minimum-image/PBC correction and avoids production or long-timescale claims.

## Evidence

- `notebooks/ligand-receptor-motion/01-ligand-receptor-translational-motion.ipynb`: loads through `load_gpcrmd_mlx_artifact`, displays `bundle.blocker_json()`, and stops before visualization when no processed trajectory exists.
- `notebooks/ligand-receptor-motion/helpers/mlx_real_md.py`: blocks trajectories unless metadata has `engine="mlx_atomistic"`, `kind="gpcrmd_mlx_nvt"`, and `workflow="run_gpcrmd_mlx"`.
- `notebooks/ligand-receptor-motion/helpers/visualization.py`: selects receptor pocket, nearby waters/ions, and bounded lipid context.
- `notebooks/ligand-receptor-motion/helpers/motion_analysis.py`: applies minimum-image/PBC correction for contact, water, ion, and residue distance analysis.
- `notebooks/ligand-receptor-motion/README.md`: avoids production, binding/unbinding, and long-timescale GPCR dynamics claims.
- Required command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_ligand_receptor_motion.py tests/test_trajectory_adapters.py -k "gpcrmd or notebook or pbc"`: 12 passed, 9 deselected.
