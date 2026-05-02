# SPEC: PME/Ewald Periodic Electrostatics

## Bounded Goal

Add a validated periodic electrostatics path to `mlx_atomistic`, starting with an Ewald reference implementation and then a PME-compatible interface, so solvated periodic prepared systems can fail closed on precise remaining blockers instead of always rejecting `pme_ewald_periodic_electrostatics`.

## Selected Lenses

- engineering
- runtime

## Constraints

- `mlx_atomistic` remains the only simulation engine; external MD engines may be read as references but not called at runtime.
- Scope is electrostatics only: do not implement lipids, CHARMM CMAP, NPT/barostat, virtual sites, or full GPCRmd runtime in this change.
- Start with orthorhombic periodic cells, real units, neutral systems, exclusions, and 1-4 exceptions; fail closed for unsupported cells or non-neutral charged systems.
- Keep APIs compatible with existing `NonbondedPotential`, prepared artifacts, diagnostics, and benchmark surfaces.
- Use short correctness and performance fixtures before attempting any large GPCRmd-derived system.

## Blocking Questions Or Assumptions

- Assumption: the first production-quality milestone should be `ewald_reference`, not mesh PME, because it gives a debuggable force/energy oracle inside our own code.
- Assumption: PME mesh acceleration is the next slice after Ewald reference parity is verified on small neutral periodic systems.
- Assumption: GPCRmd dynamics 729 remains blocked after this slice until lipid/CHARMM CMAP/scale support is addressed.

## Anti-Goals

- Do not claim full GPCRmd/GPCR membrane MD support from PME/Ewald alone.
- Do not introduce OpenMM, LAMMPS, GROMACS, or other MD engines as runtime dependencies.
- Do not hide direct-cutoff electrostatics behind a `pme` label.
- Do not optimize for 92k-atom GPCRmd scale before the periodic electrostatics math is validated on small systems.

## Acceptance Shape

- `mlx_atomistic` exposes an explicit electrostatics mode for direct cutoff, Ewald reference, and future PME.
- Ewald energy and forces pass finite-difference, invariance, and convergence tests on small neutral periodic systems.
- Prepared-artifact compatibility reports can distinguish `pme_ewald_periodic_electrostatics` from later blockers such as lipids, CMAP, virtual sites, and scale.
- Benchmarks report the cost of direct cutoff vs Ewald reference on small and medium periodic fixtures.
