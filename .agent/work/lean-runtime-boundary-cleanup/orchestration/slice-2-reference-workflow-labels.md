# Slice 2: Reference Workflow Labels And Artifact Hygiene

## Status

complete

## Route

direct

## Files Changed

- `notebooks/README.md`: labels the active trajectory as `mlx_atomistic` and OpenMM notebooks as `openmm-reference`.
- `notebooks/ligand-receptor-motion/README.md`: labels OpenMM artifacts as `openmm-reference` and clarifies that they are not production runtime output.
- `scripts/run_openmm_gpcrmd_preview.py`: adds `artifact_label="openmm-reference"` to generated metadata and reports.
- `scripts/run_openmm_gpcrmd_charmm_md.py`: adds `artifact_label="openmm-reference"` to generated metadata and reports.
- `.gitignore`: clarifies that generated MLX and OpenMM reference outputs are ignored.

## Verification

- `rg -n "mlx_atomistic|openmm-reference|reference preview|generated.*ignored|not production" notebooks/README.md notebooks/ligand-receptor-motion/README.md scripts/run_openmm_gpcrmd_preview.py scripts/run_openmm_gpcrmd_charmm_md.py .gitignore` passed.
- `git check-ignore notebooks/ligand-receptor-motion/data/openmm-md/example/trajectory.npz notebooks/ligand-receptor-motion/data/openmm-preview/example/trajectory.npz notebooks/ligand-receptor-motion/data/gpcrmd-mlx/example/trajectory.npz` passed.

## Notes

- Existing `engine="openmm"` metadata remains intact for compatibility; `artifact_label="openmm-reference"` carries the boundary label.
