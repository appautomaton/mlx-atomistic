# SPEC: GPCRmd-Backed MLX Real MD Target

## Bounded Goal

Make GPCRmd the first real-system target for `notebooks/ligand-receptor-motion/` by defining a workflow that uses GPCRmd data as reference input/validation context while requiring any newly generated trajectory shown as a project result to come from `mlx_atomistic`.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- `mlx_atomistic` is the only runtime trajectory generator; OpenMM, LAMMPS, GROMACS, and other MD engines may not run simulations for this workflow.
- `vendors/` remains reference-only and must not become a package input, runtime dependency, or build source.
- GPCRmd trajectories may be used as reference data for system selection, import checks, observables, and comparison, but not as the main claimed MLX result.
- `atomistic_prep` owns complete-system import/build responsibilities; `mlx_atomistic` owns simulation capabilities and must fail closed on unsupported physics or missing terms.
- Performance probing must use short, repeatable runs before any long notebook run; the workflow should expose wall time, steps/s, constraint error, and artifact size.

## Blocking Questions Or Assumptions

- Assumption: the first target should be a ligand-bound GPCRmd system with downloadable coordinates, topology/parameters, box, water, ions, and reference trajectory/protocol metadata.
- Assumption: the first deliverable can stop at an explicit compatibility report if current MLX physics cannot yet run the chosen full GPCR system.
- Assumption: short solvated NVT is acceptable as the first MLX-generated run; natural binding, unbinding, membrane equilibration, and ns-us sampling are later milestones.
- Assumption: missing core capabilities likely include PME/Ewald electrostatics, lipid/membrane scale handling, robust periodic neighbor lists, and possibly NPT/barostat support.

## Anti-Goals

- Do not make a downloaded public trajectory the main result of the active notebook.
- Do not revive the benzene pull or any steered/toy motion as the main workflow.
- Do not claim ligand binding, unbinding, egress, or production-quality GPCR dynamics from short MLX NVT.
- Do not add OpenMM, LAMMPS, GROMACS, or Quantum ESPRESSO as runtime simulation dependencies.
- Do not broaden into full force-field parity or PyPI release work in this change.

## Acceptance Shape

- The active workflow has one named GPCRmd target candidate or a short list with a clear selection gate.
- The repo can inspect/cache the target metadata and report whether the system is MLX-runnable today.
- If runnable, the notebook uses an MLX-generated trajectory and labels GPCRmd as reference context.
- If not runnable, the notebook and CLI state the exact missing capabilities instead of falling back to fake or downloaded motion.
