# SPEC: MD Integrator and Constraint Parity

## Bounded Goal

Add Phase 1 MD runtime parity capabilities for biomolecular production setup: stronger minimization, water/constraint handling, triclinic periodic cells, Nose-Hoover NVT, and anisotropic or membrane-aware MC NPT, each validated against OpenMM on bounded fixtures.

## Broader Intent

This preserves the roadmap-scale goal of making `mlx_atomistic` a credible Apple Silicon MD engine, while decomposing the full 20-gap feature-parity intake into the first executable phase recorded in `.agent/steering/ROADMAP.md`.

## Work Scale And Shape

- Scale: large roadmap phase.
- Shape: parity implementation with runtime and validation gates.

## Selected Lenses

- product
- engineering
- runtime

## Target User Or Stakeholder

Researchers and engineer-users who want to set up, equilibrate, and run real biomolecular MD workflows on Apple Silicon without hitting missing minimizer, constraint, cell, thermostat, or barostat surfaces before force-field and parser phases begin.

## Constraints And Risks That Change Implementation

- Use Python 3.13 and project execution through `uv`; keep source changes under `src/mlx_atomistic/`.
- Keep `vendors/` reference-only. OpenMM is a dev/reference validation surface, not runtime product code.
- Do not add heavyweight chemistry or ML packages unless a specific Phase 1 validation fixture cannot be represented without one.
- Preserve existing orthorhombic `Cell`, Langevin NVT, isotropic MC NPT, checkpoint, reporter, neighbor-list, PME, and artifact-loading behavior unless a compatibility change is explicitly required and tested.
- Triclinic support is a cross-cutting risk because `Cell`, wrapping, minimum-image math, neighbor lists, nonbonded terms, PME, virial diagnostics, pressure, barostat proposals, and artifact cell metadata may all assume three independent lengths.
- SETTLE and HMR are coupled to masses, constraint degrees of freedom, velocity projection, timestep stability, and water topology recognition; they must fail closed when required water geometry or masses are unavailable.
- Anisotropic and membrane barostats depend on pressure/virial diagnostics being meaningful for the supported force terms. Unsupported force terms must produce clear blockers rather than silent NPT claims.
- OpenMM parity should use small bounded fixtures for development proof. This phase does not certify the 92k-atom GPCRmd fixture; prior evidence says that larger fixture is still blocked by lazy-topology runtime nonbonded pair provisioning.

## Required Outcome

The Phase 1 runtime surface must expose and validate these parity targets:

- P1.1 Minimization: selectable steepest-descent compatibility plus L-BFGS and conjugate-gradient minimization paths with convergence diagnostics and neighbor-list-aware force evaluation.
- P1.2 Constraints: analytical SETTLE for supported water molecules, existing generic distance constraints retained, and hydrogen mass repartitioning that preserves total mass and records provenance.
- P1.3 Triclinic PBC: a cell representation and periodic operations that support full 3x3 triclinic boxes while preserving current cubic and orthorhombic APIs.
- P1.4 Thermostat: Nose-Hoover NVT support distinct from the existing Langevin BAOAB path, with deterministic state and reporter/checkpoint visibility.
- P1.5 Barostat: anisotropic and membrane/semi-isotropic MC barostat modes that update cell shape, rescale coordinates consistently, and report attempts/acceptance.
- P1.6 Validation: OpenMM-backed parity checks for each new capability plus one bounded end-to-end protocol that runs minimize -> NVT -> NPT on a small explicit-solvent biomolecular or water-rich fixture.

## Acceptance Criteria

- AC1: Existing behavior remains compatible: cubic/orthorhombic cell tests, current NVE/NVT/NPT tests, checkpoint/restart tests, runtime reporter tests, artifact-loading tests, PME tests, and production-probe blocker tests still pass.
- AC2: Minimization tests show L-BFGS and conjugate-gradient reduce potential energy and force norms on deterministic fixtures, stop on tolerance or iteration bounds, and produce OpenMM-comparable minimized energies for at least one small molecular mechanics fixture.
- AC3: SETTLE tests keep water O-H and H-H distances within declared tolerance, remove constrained relative velocity components, interoperate with existing generic constraints, and fail clearly for non-water or malformed water topology.
- AC4: HMR tests preserve total system mass, increase selected hydrogen masses according to configuration, reduce bonded heavy-atom masses consistently, and serialize enough metadata for artifacts/checkpoints to explain the repartitioning.
- AC5: Triclinic tests cover construction from a full cell matrix, wrapping, fractional conversion, minimum-image displacement, neighbor-list candidate generation, nonbonded energy/force evaluation, and compatibility with existing `Cell.cubic()` and `Cell.orthorhombic()` callers.
- AC6: Nose-Hoover NVT tests produce finite trajectories, bounded temperature behavior on a small fixture, deterministic continuation from state/checkpoint metadata, and separate metadata from Langevin thermostat reports.
- AC7: Anisotropic and membrane barostat tests verify allowed cell-axis updates, coordinate scaling, pressure metadata, attempt/acceptance accounting, constraint compatibility, and fail-closed behavior when required virial support is absent.
- AC8: OpenMM parity fixtures are stored or generated reproducibly, run through `uv run ...`, and compare agreed observables with explicit tolerances: minimized energy, constrained water geometry, triclinic periodic distances, thermostat temperature statistics, and barostat cell/volume trends.
- AC9: A bounded end-to-end proof command or test runs minimize -> Nose-Hoover NVT -> anisotropic or membrane MC NPT using the new surfaces and records finite energies, finite coordinates, finite cell state, and no unsupported-term blockers.
- AC10: Final verification passes `uv run ruff check src tests scripts` and `uv run pytest`.

## Scope Coverage Decisions

- Included from the intake: Phase 1 minimizer, constraint, HMR, triclinic PBC, Nose-Hoover thermostat, anisotropic/membrane barostat, and OpenMM parity validation surfaces.
- Deferred to later roadmap phases: force-field file parsers, Ryckaert-Bellemans dihedrals, higher-order PME interpolation, virtual sites, GBSA/OBC, custom force expressions, replica exchange, alchemical free energy, CI/release automation, and broad documentation restructuring.
- Deferred intake decisions: GBSA/OBC phase priority and custom force expression API shape do not change Phase 1 implementation or verification, so they remain roadmap decisions rather than blockers for this spec.

## Anti-Goals

- Do not claim broad production-MD readiness or GPCRmd 729 runtime success from this phase.
- Do not implement CUDA, ROCm, x86 HPC support, reactive force fields, machine-learned potentials, or materials-science potential families.
- Do not import from `vendors/` or make OpenMM/LAMMPS product runtime dependencies.
- Do not implement force-field parser breadth, virtual sites, implicit solvent, alchemical methods, replica exchange, or custom algebraic force expressions in this phase.
- Do not make API churn in unrelated DFT, visualization, notebook, or benchmark surfaces beyond what Phase 1 runtime compatibility requires.
