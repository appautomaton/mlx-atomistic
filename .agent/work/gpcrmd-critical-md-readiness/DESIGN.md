# DESIGN: GPCRmd-Critical MLX MD Readiness

## Boundary

This change turns GPCRmd from a compatibility target into an MLX runtime target. External engines remain reference material only. `mlx_atomistic.prep` imports and validates complete GPCRmd systems; `mlx_atomistic` runs the simulation and refuses unsupported physics.

## Data Flow

```text
GPCRmd cache / metadata
  -> term and protocol inventory
  -> topology/parameter import
  -> strict prepared_system artifact
  -> MLX compatibility validation
  -> MLX minimize/equilibrate/short-run protocol
  -> trajectory.npz + diagnostics
  -> notebook visualization / comparison
```

## Capability Gates

The full GPCRmd run is allowed only after these gates pass:

1. Selected target inventory names every required force term, parameter source, water/lipid model, protocol requirement, and box/constraint feature.
2. PME mesh and its `NonbondedPotential` integration match the existing Ewald reference on small neutral periodic fixtures.
3. CHARMM/GPCR terms used by the selected target have finite energy/force tests.
4. Periodic nonbonded execution scales beyond dense toy systems and reports memory/runtime estimates.
5. Import writes strict artifacts without silently dropping terms.
6. MLX protocol emits finite energy, temperature, pressure/virial, constraint, and per-term diagnostics.

## Engine Surface

`mlx_atomistic` must add only terms needed by the selected GPCRmd path:

- PME mesh electrostatics;
- CHARMM-relevant terms such as CMAP, Urey-Bradley, NBFIX/pair overrides, force-switching, and lipid nonbonded/bonded parameters when present;
- scalable periodic neighbor/cell lists;
- virial and pressure diagnostics;
- NPT or membrane barostat only if required by the chosen protocol;
- artifact-loader support for those exact terms.

Broader OpenMM/LAMMPS features stay out of scope unless the GPCRmd target requires them.

## Prep Surface

`mlx_atomistic.prep` owns GPCRmd import:

- inspect the actual cached topology/parameter/protocol files;
- parse atom names/types, residues, water/ions/lipids, box vectors, masks, constraints, exclusions, exceptions, and CHARMM-specific parameters;
- export strict artifacts or block with exact missing terms.

## Parallelization Model

After the target inventory gate, independent workers can proceed on disjoint write sets:

- PME standalone math in `src/mlx_atomistic/pme.py`;
- CHARMM term primitives in `src/mlx_atomistic/charmm_terms.py`;
- scalable periodic pair construction in `src/mlx_atomistic/cell_list.py` / `src/mlx_atomistic/neighbors.py`.

Integration slices are serial because they touch shared artifact schemas, `NonbondedPotential`, protocol runners, and notebook behavior.

## Readiness Claim

The project can claim GPCRmd readiness only when a selected complete GPCRmd system imports into strict MLX artifacts and a short MLX-generated trajectory runs without NaNs. A downloaded GPCRmd trajectory may be used only as comparison context.
