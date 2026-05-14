# SPEC: GPCRmd Real Artifact MLX MD Run

## Bounded Goal

Use the downloaded GPCRmd 729 artifacts to produce a strict MLX-ready prepared artifact, run a short real `mlx_atomistic` NVT trajectory, and make `notebooks/ligand-receptor-motion/` visualize/analyze that MLX-generated trajectory.

## Selected Lenses

- product
- engineering
- runtime
- design

## Constraints

- No OpenMM, LAMMPS, GROMACS, ACEMD, NAMD, or other external MD engine may run; `mlx_atomistic` generates the trajectory.
- `vendors/` may be read only as reference material for semantics and algorithms; it must not become a dependency, runtime import, or build input.
- The active result must use the real downloaded GPCRmd 729 PSF/PDB/PRM/protocol artifacts, not public reference trajectory playback, toy fixtures, or forced-motion demos.
- The workflow must fail closed when a required CHARMM/GPCRmd term, box/protocol field, topology feature, or scalable nonbonded path is unsupported.
- The notebook must show blocker JSON when blocked and show trajectory visualization/analysis only when `trajectory.npz` is created by `mlx_atomistic`.

## Blocking Questions Or Assumptions

- Assumption: the first successful run target is a short NVT proof trajectory, not long production GPCRmd sampling.
- Assumption: the downloaded GPCRmd files under `notebooks/ligand-receptor-motion/data/gpcrmd-cache/729/` are available locally but ignored by git.
- Assumption: exact CHARMM fidelity is required for import validation, but the first MLX run may use conservative runtime settings such as shorter `dt`, sparse diagnostics, and a short step count.
- Assumption: if the full 92k-atom system is still too slow or memory-heavy after scalable nonbonded routing, the accepted outcome is an exact blocker report plus the next runtime slice, not a fabricated trajectory.
- Blocking question: none for framing.

## Required Outcome

- `mlx_atomistic.prep Python API run-gpcrmd-mlx --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 ...` either writes a real MLX `trajectory.npz` or returns precise blockers.
- On success, `prepared_system.json`, `prepared_system.npz`, `view.pdb`, `gpcrmd_import_report.json`, `gpcrmd_mlx_run_report.json`, and `trajectory.npz` describe the same GPCRmd target.
- The notebook loads that trajectory, labels it as `mlx_atomistic`, displays frame/time metadata, and runs PBC-aware ligand/receptor analysis.

## Known Implementation Risks

- GPCRmd CHARMM import currently needs PSF-derived `MASS` handling before ParmEd can load atom type `CT3`.
- CHARMM Urey-Bradley, CMAP, and possible NBFIX data must be extracted into the artifact arrays instead of being dropped.
- GPCRmd protocol tarballs must be parsed for `input.xsc` box vectors and constraint/HMR/virtual-site policy.
- The 92k-atom system cannot rely on dense all-pairs topology or nonbonded execution.
- Notebook execution may need degraded visualization defaults for a large membrane/solvent system.

## Anti-Goals

- Do not claim natural ligand binding, unbinding, or long-timescale GPCR dynamics from the short NVT proof.
- Do not use GPCRmd reference trajectories as the active project result.
- Do not implement NPT or a membrane barostat in this change unless the short NVT proof cannot be honestly gated without it.
- Do not broaden into all OpenMM/LAMMPS feature parity.
- Do not commit downloaded GPCRmd data or generated trajectory artifacts.

## Recommended Next Skill

- `auto-plan`
