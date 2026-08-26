# DFT Roadmap

This roadmap defines the bounded path from the current specialized periodic
runtime to a general core for common solid-state DFT. It does not target full
feature parity with Quantum ESPRESSO or CP2K. MLX remains the product runtime;
reference engines remain validation surfaces.

## Current Baseline

The retained periodic path provides PBE-PW92, reciprocal-space GTH operators,
weighted k-points, block-Davidson eigensolves, fixed and Fermi-Dirac
occupations, frozen-density bands, analytic fixed-cell forces, and atomic
SCF checkpoint/resume. A fail-closed fixed-cell periodic ionic optimizer adds
accepted-state electronic continuation and a separate atomic outer checkpoint.
Periodic analytic stress and variable-cell controllers now add fixed integer-G
trajectories, differentiable frozen-energy derivatives, cutoff-convergence
admission, and accepted-cell checkpoints. Their bounded 2H-Silicon material
gate is closed.
Periodic collinear spin now has a shared two-channel SCF controller, spin-PBE,
fixed and unconstrained occupations, symmetry-breaking initial seeds, and
spin-aware checkpoint/resume. Its deterministic numerical gates and the
source-bound bcc Iron PBE/GTH-q16 material golden pass. Fe q8 retains a declared
magnetic transferability failure.
Scalar and spin-resolved total density of states, portable charge and
magnetization volumes, and reusable cutoff/k-point/smearing comparisons now
share source-bound result contracts. A bounded finite-displacement Gamma-point
phonon core adds explicit displacement symmetry, raw reciprocity and ASR
diagnostics, no-pickle partial-sample restart, and a verified two-atom Silicon
material gate.
Real and reciprocal grids, k-points, Ewald terms, GTH operators, forces, and
state identity share one full-rank periodic cell matrix contract.
Material-level verification remains narrow: Silicon, Carbon, and simple-metal
Aluminum have accepted equation-of-state workloads, Silicon has one accepted
orthorhombic relaxation and one accepted hexagonal relaxation, and MgO retains
a declared force and transferability boundary.

## Program Boundary

The target is a general solid-state core that supports common insulating,
metallic, and collinear-magnetic crystals across ordinary Bravais cells. It
must provide trustworthy energy, bands, forces, stress, ionic relaxation, cell
relaxation, and restart behavior within an explicit pseudopotential and
functional envelope.

## Capability Coverage

The general-core claim is bounded by the following matrix. A row is complete
only when its implementation and material-level evidence both pass.

| Capability | General-core commitment | Current boundary | Delivery |
| --- | --- | --- | --- |
| Exchange-correlation | PBE-PW92 production envelope | Implemented; verified material set is narrow | Every material phase |
| Pseudopotentials | Broad, fingerprinted GTH transferability | s/p/d and chemistry coverage is complete; strict MgO and historical identity gates remain blocked | Phase 7 |
| Crystal geometry | Ordinary full-rank periodic cells | Implemented; one source-bound hexagonal Silicon case and bounded low-symmetry oracles are verified | Phase 3 |
| Electronic states | Insulators, simple metals, and collinear magnets | One simple metal and one collinear bcc Iron q16 workload are verified | Phases 1 and 5 |
| Electronic observables | Energy, density, occupations, bands, total DOS, and Fermi level | Implemented; scalar/spin DOS, portable density volumes, and reusable convergence reports pass deterministic gates | Phase 6 |
| Mechanical observables | Analytic forces and validated stress | Forces retain an MgO boundary; analytic periodic stress is verified for the bounded 2H-Si workflow | Phase 4 |
| Structural workflows | Fixed-cell ionic and variable-cell relaxation | Fixed-cell and one 2H-Si variable-cell trajectory are verified; broad transferability remains open | Phases 2 and 4 |
| Lattice dynamics | Bounded finite-displacement phonons | Gamma-point core and source-bound Silicon gate are verified; dispersion and non-analytic response are deferred | Phase 8 |
| Reproducibility | Source-bound inputs, convergence studies, restart, and explicit evidence labels | SCF, ionic, cell, spin, density-volume, and phonon-sample artifacts have fail-closed identities | Every phase |
| Runtime quality | Measured complete wall and peak memory on representative Apple Silicon workloads | Existing DFT controls cover a narrow workload set | Every phase |

## Architecture Rules

The following rules apply across all phases and prevent feature-specific
subsystems from fragmenting the runtime:

- Reuse the project-level `Cell` as the sole cell geometry type. General-cell
  work extends DFT grids, reciprocal transforms, Ewald terms, and public system
  construction; it does not introduce a second DFT cell abstraction.
- Keep immutable calculation identity separate from mutable solver and workflow
  state. Positions, cell, pseudopotentials, k-points, occupations, and
  functional settings remain fingerprinted inputs.
- Separate optimization mathematics from the electronic evaluator. L-BFGS,
  step clipping, and line-search policy may be shared, while legacy and
  periodic SCF adapters retain their own scientific state contracts.
- Treat inner SCF continuation and outer workflows as different artifact
  layers. Geometry, cell, and phonon workflows checkpoint accepted outer states
  without weakening the existing SCF checkpoint identity.
- Add periodic spin through an explicit channel dimension and shared
  Hamiltonian machinery, not by copying the complete unpolarized controller.
- Preserve the accepted fixed-occupation and orthorhombic paths as compatibility
  oracles while new dimensions are introduced.

The dependency order is:

```text
metallic protocol lock -> metallic golden validation

analytic forces -> fixed-cell ionic relaxation -> general cell geometry
                                                   |-> stress -> variable-cell relaxation
                                                   |-> periodic collinear spin
                                                   |      `-> DOS and convergence workflows
                                                   `-> phonons

pseudopotential transferability expands across every phase
```

## Phase 1: Close Metallic Scientific Validation

Status: complete. The source-bound Aluminum workload passes every locked gate.

- Lock one Aluminum reference protocol with the same structure, functional,
  pseudopotential identity, Fermi-Dirac width, and free-energy definition used
  locally. Converge the representation-specific basis and k-point integration
  rather than equating unlike CP2K Gaussian-grid and local plane-wave cutoffs.
- Retain source fingerprints and parsed inputs so the comparison is
  reproducible without placing a reference engine on the MLX runtime path.
- Validate electron count, chemical potential, occupations, free energy, and
  at least one material observable such as an equation-of-state curve.

The bounded scientific contract and delivery plan are recorded in
[DFT Metallic Validation](./dft-metallic-validation.md).

Exit gate: closed. The accepted profile and current-verified measurements are
recorded in the linked evidence summary.

## Phase 2: Add Periodic Fixed-Cell Ionic Relaxation

Status: complete. The scientific contract and current-verified Silicon result
are recorded in
[DFT Periodic Fixed-Cell Relaxation](./dft-periodic-relaxation.md).

- Add a periodic optimizer that consumes converged analytic forces.
- Reuse SCF density and compact eigenspaces between accepted ionic steps.
- Preserve fixed cell, k-point, pseudopotential, checkpoint, and provenance
  identity throughout the trajectory.
- Fail closed on unconverged SCF states, non-finite forces, and unsupported cell
  modes.

Exit gate: closed. The source-bound Silicon relaxation passes the locked force,
displacement, energy, and reference-geometry gates. Deterministic CPU coverage
locks restart equivalence, and the Metal workload verifies atomic outer
checkpoint publication.

## Phase 3: Generalize Periodic Cell Geometry

Status: complete. The matrix convention, compatibility boundary, implementation
and current-verified evidence are recorded in
[DFT General Periodic Cell Geometry](./dft-general-cell-geometry.md).

- Replace orthorhombic-only assumptions with one full-rank `3 x 3` cell matrix
  contract across real and reciprocal grids, plane-wave bases, k-points,
  Ewald terms, local and nonlocal GTH operators, forces, and fingerprints.
- Preserve the existing orthorhombic numerical trajectory as a strict
  compatibility case.
- Add representative cubic, hexagonal, and low-symmetry cells.

Exit gate: closed. Cell-coordinate transforms, reciprocal identities, Ewald and
GTH invariances, analytic force derivatives, SCF state identity, and the
source-bound 2H-Silicon relaxation pass their locked gates. Existing
orthorhombic compatibility tests remain unchanged.

## Phase 4: Add Stress And Variable-Cell Relaxation

Status: complete, exit gate closed. The sign, free-energy, frozen-variational,
optimization, restart, and evidence contracts are locked in
[DFT Stress And Variable-Cell Relaxation](./dft-stress-variable-cell.md).

- Establish a reliable periodic stress tensor with an explicit sign, unit, and
  free-energy convention.
- Validate stress against controlled cell finite differences before using it
  in optimization.
- Add bounded cell-only and coupled ion/cell relaxation with restartable state.

Exit gate: closed. Electronic stress terms pass isotropic and shear numerical
derivatives, the float64 analytic Ewald tensor passes a full-rank cell
derivative, and deterministic cell-only, coupled, and checkpoint workflows
pass. The source-bound 2H-Silicon trajectory converges from `0.995` to
`0.9981142` of the accepted scale with `0.1886%` lattice error. Its 25-to-35 Ha
pressure drift is `2.17019e-7 Ha/bohr^3`, below the locked
`5e-6 Ha/bohr^3` Pulay gate.

## Phase 5: Add Periodic Collinear Spin

Status: complete. The scientific and architecture contract is recorded in
[DFT Periodic Collinear Spin](./dft-periodic-spin.md).

- Carry separate spin-up and spin-down densities, occupations, potentials, and
  convergence diagnostics through periodic SCF.
- Support fixed magnetization and unconstrained collinear modes with explicit
  electron-count contracts.
- Add non-magnetic equivalence and magnetic material golden cases.

Exit gate: closed. The unpolarized limit, charge and fixed-moment conservation,
shared-Fermi-level occupations, checkpoint equivalence, and Quantum ESPRESSO
PW92 spin oracle pass. The bcc Iron PBE/GTH-q16 workload passes energy ordering,
published moment, cutoff, k-point, complete-wall, and logical-memory gates. The
q8 variant remains a documented transferability failure.

## Phase 6: Add Core Observables And Convergence Workflows

Status: closed. The shared scientific and artifact contract is recorded in
[DFT Core Observables And Convergence](./dft-core-observables.md).

- Add total density of states for fixed and smeared calculations, including
  spin-resolved output when spin is active.
- Add a portable volumetric density export for charge and magnetization fields.
- Generalize the existing material-specific cutoff and k-point checks into
  reusable convergence workflows, including a smearing-width study for metals.
- Keep projected density of states out of this phase until nonlocal projector
  conventions have the required fidelity.

Exit gate: integrated density of states reproduces the declared electron count,
the insulating and metallic Fermi-level conventions are explicit, volumetric
exports round-trip their cell and density normalization, and convergence
reports retain exact source and runtime identities.

Exit evidence: deterministic scalar and spin-resolved DOS integration,
full-rank charge and magnetization volume round-trips, strict malformed-artifact
rejection, and shared cutoff, k-point, and smearing comparison tests pass. No
material workload was rerun merely to exercise these analysis contracts.

## Phase 7: Expand Pseudopotential Transferability

Status: implementation complete; scientific exit blocked. The fail-closed
matrix and efficient material decision policy are defined in
[Periodic GTH Transferability](./dft-gth-transferability.md).

- Strengthen periodic GTH convention fidelity instead of treating parser
  success as scientific validation. Existing UPF support retains its separate
  proof-level boundary.
- Cover representative `s`, `p`, and `d`-block elements, ionic compounds,
  multiple oxidation environments, and both local and nonlocal force terms.
- Bind every claim to an exact resource fingerprint and matching reference
  protocol.

Exit gate: a multi-material matrix passes locked energy, structural, and force
thresholds without element-specific runtime branches.

Current exit result: coverage is complete, but the gate remains blocked by the
locked MgO q2 bulk-derivative and total-force residuals and incomplete exact
calculation/runtime identities for older project-derived summaries. The current
Fe q16 full-versus-reduced SCF oracle passes after rotated-density
reconstruction, but its broader legacy evidence still lacks refreshed v2
identities. The primitive Mg q10 screen failed its orbital-residual gate and
was produced before SCF density reconstruction, so no full q10 EOS was run.

## Phase 8: Add Finite-Displacement Phonons

Status: closed. The bounded Gamma-point numerical, symmetry, and restart
contract is defined in
[Finite-Displacement Phonons](./dft-finite-displacement-phonons.md).

- Build force-constant matrices from symmetry-independent finite
  displacements only after force, relaxation, cell, and restart contracts are
  stable.
- Enforce translational sum rules as diagnostics rather than silently repairing
  invalid force data.
- Validate frequencies and eigenvectors for bounded crystals.

Exit gate: displacement convergence, acoustic modes, restart equivalence, and
reference frequencies pass declared thresholds.

Exit evidence: the source-bound two-atom diamond-Silicon PBE/GTH-q4 workload
passes raw reciprocity and ASR diagnostics, three acoustic translation modes,
the optical-triplet reference boundary, `0.02 -> 0.01 bohr` displacement
convergence, exact partial-sample restart, complete-wall, and logical-memory
gates. Eight SCFs were required; no ASR correction was imposed.

## Cross-Cutting Gates

Every phase must satisfy the same delivery gates; a feature is not retained
only because its public API exists:

- Numerical: conservation laws, finite values, derivatives, restart
  equivalence, and compatibility oracles pass locked tolerances.
- Scientific: at least one source-bound material case passes without changing
  thresholds after observing the result.
- Identity: source inputs, pseudopotentials, protocols, runtime source, and
  accepted workflow state have deterministic fingerprints.
- Performance: complete-wall time and peak memory are measured on a
  representative Apple Silicon workload; performance regressions are either
  removed or explicitly accepted as a capability cost.
- Portability: routine correctness remains covered by CPU continuous
  integration, while Metal remains an optimization and parity instrument.
- Product boundary: MLX remains the execution path; reference engines and
  heavyweight chemistry packages do not become runtime dependencies.
- Documentation: capability, evidence, and known boundaries are updated in the
  canonical docs in the same change.

## General-Core Exit Criteria

The project may describe the periodic runtime as a general core for common
solid-state DFT only when all of the following hold:

- ordinary full-rank crystal cells are supported;
- insulating, simple metallic, and collinear-magnetic paths have material
  goldens;
- energy, bands, forces, stress, fixed-cell relaxation, and variable-cell
  relaxation form consistent workflows;
- the GTH production envelope is backed by a multi-material transferability
  matrix;
- total density of states and reusable cutoff, k-point, and smearing
  convergence reports are available;
- bounded finite-displacement phonons pass displacement and reference
  frequency gates;
- checkpoint/restart preserves every supported scientific state;
- complete-wall performance and peak memory remain measured on representative
  Apple Silicon workloads;
- unsupported physics fails closed and remains documented.

Current exit audit: every criterion above is implemented and has bounded
evidence except the broad GTH production envelope. Phase 7 has complete
coverage, but its strict matrix remains failed by the locked MgO q2
bulk-derivative and total-force residuals, the rejected q10 candidate, and
missing exact calculation/runtime identities in older summaries. The Fe q16
point-group method oracle now passes, but that does not erase the remaining
identity and MgO blockers. The project therefore must not yet claim the general
core defined here. No Phase 6 or Phase 8 implementation gap remains hidden
behind that blocker.

## Deferred Beyond The General Core

The following are named post-core programs, not hidden work inside the phases
above:

- functional breadth: DFT+U, nonlocal dispersion, meta-GGA, and hybrid
  functionals;
- relativistic and magnetic breadth: non-collinear spin and spin-orbit
  coupling;
- difficult electrostatics: charged-defect corrections, slab dipole
  corrections, and isolated-boundary molecular electrostatics;
- projected analysis: projected density of states, population analysis, and
  projector-dependent bonding descriptors;
- response physics: density-functional perturbation theory, dielectric and
  optical response, and electron-phonon workflows;
- reaction and excited-state workflows: nudged elastic band and time-dependent
  DFT;
- platform scale: automatic crystallographic symmetry discovery, distributed
  execution, multi-device scheduling, and broad molecular DFT;
- alternate periodic pseudopotential envelopes: production UPF support beyond
  the required GTH general-core envelope.

Each item requires its own roadmap or bounded extension after the general-core
exit audit. None is an implied blocker for the claim defined here.

## Delivery Rule

Each phase begins with source and protocol research. Implementation starts only
after inputs, numerical semantics, acceptance thresholds, and evidence labels
are locked. A task tracker may record bounded implementation progress; it does
not own program-level scientific decisions or weaken exit gates.
