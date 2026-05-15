# SPEC: MD Engine Gap Closure

## Bounded Goal

Close the important `mlx_atomistic` production-MD engine gaps through an ordered
phase sequence that starts with an OpenMM parity fixture and then implements PME,
runtime observability/restart, HMR or virtual-site policy, NPT, trajectory output,
and measured performance work.

## Broader Intent

Make `mlx_atomistic` a credible MLX-native MD runtime for real biomolecular
simulation workflows on Apple Silicon while keeping OpenMM, GROMACS, LAMMPS, and
`vendors/` as reference or validation surfaces, not product dependencies.

This spec preserves the verified gap audit rather than replacing it. The source
inputs are:

- `.agent/work/md-engine-capability-gap-matrix/EVIDENCE-INDEX.md`
- `.agent/work/md-engine-capability-gap-matrix/GAP-MATRIX.md`
- `.agent/work/md-engine-capability-gap-matrix/BACKLOG.md`
- `.agent/work/md-engine-gap-closure/spec/mature-framework-gap-comparison.md`

## Work Scale And Shape

- scale: multi-phase capability objective
- shape: parity + feature closure + runtime production-readiness

This is one coherent objective because every phase depends on the same product
outcome: MLX-generated trajectories should be physically credible, observable,
restartable, and comparable to OpenMM on selected production artifacts.

## Selected Lenses

- product
- engineering
- runtime

## Constraints And Risks

- Use `uv run ...` for Python execution.
- Keep source under `src/mlx_atomistic/`.
- Keep generated trajectories, parity reports, and benchmark outputs under ignored
  `results/` or temporary paths unless a later plan explicitly commits a small
  fixture.
- Treat OpenMM as the primary executable parity reference. Current local evidence
  shows OpenMM exposes `Reference`, `CPU`, and `OpenCL` platforms through `uv`.
- Treat GROMACS and LAMMPS as algorithm/source references unless a later plan adds
  a bounded runnable probe.
- Do not build GROMACS as part of this work.
- Do not skip the parity fixture and jump directly into PME/NPT implementation.
  PME, NPT, checkpointing, and performance need a shared test system and tolerance
  harness.
- Keep the repo lean: no heavyweight chemistry or ML helper packages without a
  concrete need tied to one phase.
- Preserve fail-closed behavior for unsupported artifacts. A partial feature must
  not silently accept production artifacts it cannot represent.

## Required Outcome

The implementation roadmap for this spec must close the important gaps in this
order unless fresh evidence changes the dependency graph:

1. **Production artifact parity fixture:** select a small real AMBER or CHARMM
   artifact, build MLX terms and an OpenMM reference from the same source, compare
   total/component energies and forces with explicit tolerances, and record
   unsupported terms exactly.
2. **PME production readiness:** make PME production-executable for the selected
   fixture, update `pme_readiness_report` semantics, and verify OpenMM PME
   force/energy parity.
3. **Runtime observability:** add a minimal reporter/callback surface for frames,
   scalar state data, diagnostics, and parity traces without changing physics.
4. **Checkpoint/restart boundary:** serialize and restore the state needed for
   long runs, including positions, velocities, step/time, thermostat/RNG state,
   neighbor-list policy, force-term metadata, and diagnostic cursor.
5. **Constraints, HMR, and virtual-site policy:** either implement the support
   needed for common 2-4 fs workflows or reject those artifacts with exact
   metadata-driven blockers.
6. **NPT/barostat:** add pressure coupling after PME and virial diagnostics are
   credible, starting with a Monte Carlo barostat unless planning evidence selects
   another implementation.
7. **Production trajectory output:** expose DCD/XTC output through the runtime
   reporting/output surface while keeping native NPZ useful for diagnostics.
8. **Performance pass:** profile the parity fixture after correctness gates pass,
   then optimize neighbor, pair, PME FFT, or Metal/MLX hot paths based on measured
   evidence.

## Acceptance Criteria

- A plan derived from this spec maps every phase to the verified gap row IDs from
  `GAP-MATRIX.md`.
- Phase 1 produces a runnable OpenMM-vs-MLX force/energy parity harness before
  PME/NPT implementation begins.
- PME is not marked production-ready until `pme_readiness_report` is green for a
  selected fixture and OpenMM energy/force parity passes explicit tolerances.
- NPT is not implemented before PME force/energy parity and virial diagnostics are
  validated on the same fixture family.
- Reporter and checkpoint work are verified with restart/resume behavior and do
  not replace physics validation.
- DCD/XTC output is treated as a product runtime surface, not just an analysis
  adapter, once reporter infrastructure exists.
- Performance work is driven by profiling after correctness gates; no custom
  Metal or MLX-kernel work is accepted without measured hot-path evidence.
- Deferred capabilities remain named and out of scope: raw PDB plus force-field
  one-shot API, thermostat variety beyond Langevin, triclinic cells after the
  first orthorhombic PME target, LJPME/dispersion correction, polarizable force
  fields, free energy, REMD, metadynamics, and GBSA.
- No DFT scope is added.
- No GROMACS build or GROMACS runtime dependency is added.

## Linked Detail Files

- `spec/mature-framework-gap-comparison.md`: normative build/validate/defer
  decisions comparing `mlx_atomistic` to OpenMM, GROMACS, and LAMMPS for this
  gap-closure objective.

## Blocking Questions Or Assumptions

- Assumption: "important gaps" means production-blocking MD engine gaps from the
  verified audit, not long-tail scientific coverage.
- Assumption: the first parity fixture should be small enough for fast local tests
  but realistic enough to include production force-field metadata and nonbonded
  exceptions.
- Assumption: OpenMM remains the reference for executable physics parity; GROMACS
  and LAMMPS remain design/performance references unless a later plan says
  otherwise.
- Assumption: Apple Silicon/MLX remains the target runtime path.

## Anti-Goals

- Do not implement code in this framing change.
- Do not replace `mlx_atomistic` with OpenMM, GROMACS, or LAMMPS.
- Do not broaden into DFT capability work.
- Do not add a raw PDB-to-production-system API before the prepared-artifact
  parity path is credible.
- Do not claim production readiness from toy fixtures alone.
- Do not commit large generated trajectories or benchmark outputs.
