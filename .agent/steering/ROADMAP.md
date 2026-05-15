# Roadmap

## Direction

Make `mlx_atomistic` credible for production-oriented biomolecular MD on Apple
Silicon while keeping the repo lean. MLX is the runtime; OpenMM and LAMMPS are
reference and validation surfaces, not runtime dependencies.

## Current Evidence

- `run_mlx` can produce finite short NVT trajectories for local proof fixtures:
  113-atom T4L-benzene and 259-atom solvated ligand-receptor.
- Those proof fixtures are intentionally not production systems:
  `production_force_field=false`, and `require_production=True` rejects them.
- The solvated proof fixture uses `short_range_electrostatics_prototype`, not
  production PME.
- No committed production prepared artifact is currently available for a repeatable
  roadmap baseline.

## Capability Ladder

1. **Evidence Baseline**
   Keep local `results/gap-discovery/` runs reproducible and add one
   production-marked fixed-topology artifact when available.

2. **PME Production Readiness**
   Turn periodic long-range electrostatics from gated/prototype behavior into a
   validated production path with OpenMM force and energy parity.

3. **OpenMM Parity Harness**
   Compare MLX and OpenMM on the same fixed-topology system using matched timestep,
   temperature, friction, cutoff, constraints, and force-field inputs. Add only the
   minimal reporter hooks needed to capture parity diagnostics.

4. **Real-System Coverage**
   Validate AMBER/CHARMM artifact imports, close box-shape limits such as triclinic
   support where needed, and define the TIP4P/virtual-site/HMR policy.

5. **NPT / Barostat**
   Add pressure coupling after PME and virial diagnostics are credible.

6. **Runner Usability**
   Add runner-level checkpoint/restart and DCD/XTC-facing output once the physics
   path is worth preserving.

7. **Performance**
   Profile against OpenMM OpenCL on Apple Silicon and optimize the confirmed hot
   paths only after correctness targets are fixed.

## Deferred

- DFT needs its own capability audit before being mixed into this MD roadmap.
- AMOEBA/Drude, FEP/TI/BAR, REMD, GBSA, arbitrary custom forces, and broad raw-PDB
  workflow ergonomics are demand-driven follow-ons.
- Vendor checkouts remain reference-only unless a future spec explicitly changes
  that boundary.
