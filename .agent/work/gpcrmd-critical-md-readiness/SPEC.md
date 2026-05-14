# SPEC: GPCRmd-Critical MLX MD Readiness

## Bounded Goal

Implement the GPCRmd-critical missing engine and import capabilities so a complete GPCRmd membrane/solvent system can be imported, validated, simulated by `mlx_atomistic`, and visualized from an MLX-generated trajectory.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- `mlx_atomistic` remains the only trajectory generator; OpenMM, LAMMPS, GROMACS, Quantum ESPRESSO, and other engines may be used only as reference material.
- Scope is limited to GPCRmd-critical readiness: PME mesh electrostatics, CHARMM/GPCR force-field terms, lipid/membrane support, scalable periodic nonbonded execution, constraints, virial/pressure, NPT or membrane barostat if required, and GPCRmd import/validation.
- Implementations must be complete enough for the selected GPCRmd target path, not placeholder APIs or silent approximations.
- `mlx_atomistic.prep` owns topology/parameter/protocol import and complete-system validation; `mlx_atomistic` owns simulation, force terms, integration, diagnostics, and fail-closed unsupported-term reporting.
- Each capability must be gated by focused correctness/performance tests before being used in a full GPCRmd run.

## Blocking Questions Or Assumptions

- Assumption: the first target remains GPCRmd target 729 / PDB 5F8U unless planning identifies a better complete GPCRmd target with fewer non-electrostatic blockers.
- Assumption: the first full readiness proof can be a short MLX run on the imported GPCRmd system; ns-us biological sampling is out of scope for this change.
- Assumption: PME mesh must be validated against the existing Ewald reference on small neutral periodic systems before GPCRmd scale.
- Assumption: if the selected GPCRmd files require unsupported CHARMM terms such as CMAP, NBFIX, force-switching, or membrane/lipid parameters, those terms are in scope for this readiness effort.
- Assumption: if the selected protocol requires NPT or membrane pressure coupling, virial/pressure and the needed barostat path are in scope; otherwise short NVT can be the first runtime proof.

## Anti-Goals

- Do not add broad OpenMM/LAMMPS feature parity that does not unblock GPCRmd simulation.
- Do not add FEP/TI/metadynamics, antibody developability, polymer tooling, reactive chemistry, GBSA, or transport-analysis modules in this change.
- Do not make downloaded GPCRmd trajectories the project result; they are reference/validation context only.
- Do not claim full GPCRmd readiness from a reduced toy system, stripped subset, or biased steering demo.
- Do not hide unsupported physics behind permissive fallbacks; imports and runs must fail closed with exact blockers.

## Acceptance Shape

- A selected complete GPCRmd target can be imported into strict MLX artifacts with topology, parameters, water/ions/lipids, box, masks, constraints, exceptions, and protocol metadata.
- Required GPCRmd-critical force terms and electrostatics execute in `mlx_atomistic` with finite energies/forces and diagnostics.
- The imported system can run a short MLX protocol without NaNs, with finite temperature/energy/constraint/pressure diagnostics and saved trajectory artifacts.
- The active ligand-receptor notebook loads only the MLX-generated GPCRmd trajectory for the main result and labels any GPCRmd reference trajectory as comparison context.
- Remaining unsupported GPCRmd requirements, if any, are named exactly and block execution rather than producing an approximate run.
