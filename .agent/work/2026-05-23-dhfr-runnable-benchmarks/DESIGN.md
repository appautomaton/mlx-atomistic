# DESIGN: DHFR Runnable Benchmarks

## Architecture Summary

Enable DHFR benchmarks by producing real MLX prepared artifacts first, then reusing existing artifact/runtime primitives to run short bounded MD rows. Keep OpenMM outside product runtime: OpenMM may appear in scripts/tests/prep helpers used to extract reference parameters, but `mlx_atomistic` benchmark execution must consume saved artifacts and run through MLX code.

## Main Decisions

### Implicit DHFR

The implicit row cannot be repaired by inventing only `gbsa_radius` and `gbsa_scale`. It needs a complete prepared artifact sourced from the same OpenMM force-field setup used by the reference script:

```python
ForceField("amber99sb.xml", "amber99_obc.xml")
```

The prep path should extract enough OpenMM `System` data to populate `PreparedSystem`:

- particles, masses, positions, velocities
- atom/residue names and inferred symbols
- `NonbondedForce` charges, sigma, epsilon, exclusions/exceptions
- `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`
- `GBSAOBCForce` charge/radius/scale and metadata
- HBond constraints
- provenance and compatibility metadata

GBSA radii should be stored in Angstrom because existing MLX GBSA tests and artifacts use Angstrom.

### Explicit PME DHFR

The local Amber20/JAC `NONBONDED_PARM_INDEX` has two negative entries, both OW/HW water O-H pairs. The corresponding HW LJ self epsilon is zero, and OpenMM builds the system. The AMBER importer should treat those pair entries as zero-LJ only when all safety conditions hold:

- no nonzero `HBOND_ACOEF` or `HBOND_BCOEF`
- negative pair maps only to water O/H or H/O atom types
- standard mixed LJ epsilon is zero
- metadata records the policy and affected type pairs

All other negative pair-index cases remain blocked.

### Runtime

Once artifacts are saved and pass `load_prepared_mlx_artifact(require_production=True)`, use `build_mlx_system_from_artifact()` and `run_nvt()`/existing MD primitives for short bounded `--steps` runs. The benchmark row should report `ns_per_day` from `steps * dt_ps / 1000 / wall_s * 86400`, with finite energy/temperature evidence when available.

### Comparison

The same-workload helper already has DHFR semantic gates. It should compute DHFR ratios only when both MLX and OpenMM rows are `ok` and have matching:

- atom count
- `ns_per_day`
- step count
- `dt_ps`
- solvent model
- electrostatics model

Blocked or diagnostic rows remain valid outputs but do not satisfy the final runnable benchmark target unless a new unavoidable blocker is proven.

## File Boundaries

- `scripts/`: OpenMM reference/prep extraction surfaces.
- `src/mlx_atomistic/prep/`: import/extraction code that creates `PreparedSystem`.
- `src/mlx_atomistic/benchmarks/dhfr.py`: DHFR CLI orchestration, artifact prep/readiness/runtime payloads.
- `src/mlx_atomistic/benchmarks/same_workload_compare.py`: ratio gating only.
- `docs/benchmarks/`: human-readable benchmark results and reproduction commands.

## Verification Strategy

Use small direct commands first:

- importer unit tests for AMBER negative pair policy
- implicit artifact prep tests for GBSA arrays and metadata
- `dhfr --prepare --json` for both cases
- `dhfr --steps 1 --json` for both cases
- same-workload comparison summary with both DHFR raw rows
- focused lint and pytest in Metal-visible environment

