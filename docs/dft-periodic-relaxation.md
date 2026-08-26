# DFT Periodic Fixed-Cell Relaxation

This document records the completed scientific and engineering contract for
Phase 2 of the [DFT roadmap](./dft-roadmap.md). It promotes ionic relaxation
from the legacy teaching surface to the periodic PBE/GTH runtime without
adding variable-cell physics or a reference-engine dependency.

## Objective

Add a restartable fixed-cell optimizer for `PeriodicDFTSystem`. Every energy
and force evaluation must use the existing weighted-k-point periodic SCF and
analytic periodic-force paths. The workflow must reuse accepted electronic
state in memory, fail closed on unconverged SCF, and retain immutable
calculation identity across an outer restart.

The implementation may share pure optimization mathematics with the legacy
workflow. It must not reuse the legacy controller or its permissive SCF status
policy.

## Public Contract

The periodic surface exposes an explicit configuration, result, and entry
point rather than overloading `optimize_geometry`:

```python
result = optimize_periodic_geometry(
    system,
    cutoff_hartree=25.0,
    kpoint_mesh=mesh,
    config=PeriodicGeometryOptimizationConfig(),
    scf_config=scf_config,
)
```

The cell, FFT shape, ordered pseudopotentials, electron count, k-point mesh,
band count, exchange-correlation functional, and SCF controls are immutable for
the complete trajectory. Only Cartesian ionic positions change. Unsupported
cell or coupled ion/cell modes fail during configuration validation.

## Optimization Semantics

The default optimizer is limited-memory BFGS with a history of five accepted
curvature pairs. A steepest-descent fallback is used when the history is empty,
non-finite, has invalid curvature, or does not produce a descent direction.
Each proposed displacement is clipped by maximum per-ion norm.

Backtracking line search starts from the configured inverse-Hessian scale,
shrinks by `0.5`, and accepts only a converged, finite SCF state satisfying an
Armijo energy-decrease condition with coefficient `1e-4`. Rejected trials do
not update optimizer history or continuation state. The workflow never runs a
duplicate SCF merely to refresh an already accepted electronic state.

Convergence requires both maximum force norm and RMS Cartesian force below
their thresholds. After the initial geometry, maximum ionic displacement must
also be below its threshold. Energy change remains a diagnostic and cannot by
itself declare convergence.

These choices follow the established fixed-cell separation in
[Quantum ESPRESSO](https://www.quantum-espresso.org/Doc/user_guide_PDF/pw_user_guide.pdf),
the force and displacement criteria documented by
[CP2K](https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/GEO_OPT.html), and the
bounded maximum-step and restart model used by
[ASE](https://docs.ase-lib.org/_modules/ase/optimize/bfgs.html). Those projects
are design and validation references only.

## Electronic Continuation And Efficiency

Within one uninterrupted workflow, each line-search trial starts from the
density and compact owner-aware eigenspaces of the last accepted periodic SCF.
This preserves time-reversal ownership and avoids full-grid coefficient
materialization. A rejected trial cannot become the seed for a later trial.

The result records complete wall time, SCF and force timings, SCF iterations,
line-search evaluations, accepted steps, and whether density and coefficient
continuation were used. Deterministic CPU tests verify that continuation comes
only from the last accepted state and that rejected trials cannot seed later
work. Work counts remain the primary efficiency evidence; wall time is
device- and power-state-specific.

## Outer Checkpoint Contract

An opt-in checkpoint is published only after an accepted outer step. It stores
the accepted positions, density seed, force state, energy, optimizer history,
accepted-step history, and next-step cursor in an atomic, hash-inventoried
generation. Compact eigenspaces are reused in memory but are not required in
the portable outer artifact; resumed SCF may reconstruct them from the stored
density.

Resume requires the caller to provide the original system and exact immutable
calculation controls. A mismatch in source system, cell, grid, ordered
pseudopotentials, electron count, k-points, bands, functional, SCF controls, or
optimizer controls fails closed. Inner SCF checkpoints and outer relaxation
checkpoints remain separate artifact types.

## Scientific Protocol

The source geometry is `Si-Diamond.xsf` from Materials Cloud record
`yf0rj-w3r97`, DOI `10.24435/materialscloud:s4-3h`. The source file has SHA-256
`96c53bf0a1caa8a5afc99baeabb19f727483644152a5ca9e0e68efea3d3c972e` and
defines the ideal two-sublattice diamond internal coordinates.

The local validation uses the already accepted conventional eight-atom Silicon
workload:

- PBE-PW92 and `Si GTH-PBE-q4`;
- fixed conventional cubic lattice `5.460859 A` from the verified local EOS;
- `25 Ha` cutoff, `56 x 56 x 56` FFT grid, and `6 x 6 x 6`
  Monkhorst-Pack mesh;
- fixed occupations with 16 doubly occupied bands;
- one atom on the `(0.25, 0.25, 0.25)` sublattice displaced by
  `(+0.04, -0.03, +0.02) A` from the ideal source-defined position.

The fixed local lattice is intentional: this phase validates internal ionic
relaxation, not cell relaxation or exact reproduction of the source EOS
protocol. Geometry comparison removes one uniform periodic translation before
measuring atom-wise minimum-image error.

## Acceptance Criteria

Every retained electronic state must be converged and finite, preserve the
electron count within `1e-4` per cell, and produce a finite analytic periodic
force decomposition.

The production relaxation must satisfy thresholds locked before execution:

- maximum per-ion force norm no greater than `5e-4 Ha/bohr`;
- RMS Cartesian force no greater than `3e-4 Ha/bohr`;
- final maximum ionic step no greater than `3e-3 bohr`;
- maximum translation-aligned error from ideal diamond positions no greater
  than `0.01 A`;
- every accepted energy satisfies the Armijo contract;
- checkpoint/resume reproduces status and accepted-step count, with final
  energy within `5e-6 Ha` and positions within `5e-4 bohr` of an uninterrupted
  run.

The report must record complete wall time, peak physical memory, electronic
work counters, continuation use, source and runtime fingerprints, and the
power state. Thresholds are not relaxed after observing the result.

## Accepted Result

The current-verified Metal run used the locked eight-atom Silicon protocol on
AC power with low-power mode disabled. It converged in seven accepted ionic
steps, nine SCF evaluations, and eight line-search evaluations. One trial was
rejected without contaminating subsequent continuation state. Every accepted
energy satisfied the Armijo contract.

The final state passed every locked material gate:

- energy: `-31.509274637639624 Ha`;
- maximum force norm: `8.832269562418987e-5 Ha/bohr`;
- RMS Cartesian force: `2.4370971021545013e-5 Ha/bohr`;
- maximum final ionic step: `0.0014801492501048006 bohr`;
- maximum translation-aligned geometry error: `0.0003113522679746047 A`.

Complete wall time was `296.674 s`, and peak physical memory was `7.980 GB`.
Hamiltonian application consumed `166.807 s` and orthogonalization consumed
`48.402 s`; these measurements identify performance work but do not change the
scientific gate. The accepted local report is retained under
`results/dft-silicon-relaxation/uninterrupted-v2/` with workload fingerprint
`d4cbb4abe682895be010362078629354201570377d0ae3c3194ced39d7ad4426` and
runtime fingerprint
`176c2a3fa6ac019135bd8cc8a23f4109681f80f2879ad3a6558b78fccf3a2ba7`.

The Metal workflow also published an atomic checkpoint after its second
accepted step. Exact split-run versus uninterrupted restart equivalence is
covered by the deterministic CPU oracle. A second minutes-long Silicon resume
run was intentionally not used as duplicate evidence.

## Delivery

Delivered in this phase:

1. Shared pure optimization mathematics without changing the legacy numerical
   trajectory.
2. A fail-closed periodic optimizer and immutable `with_positions` system
   operation.
3. Atomic accepted-step checkpoints with strict resume identity.
4. CPU correctness, continuation, failure, and restart tests.
5. A source-bound Silicon validation on Metal; raw calculations remain under
   gitignored `results/`.
6. Updated capability and evidence boundaries. Repository CI remains the merge
   gate.

## Out Of Scope

This phase does not add general cells, stress, variable-cell relaxation,
constraints, symmetry enforcement, molecular DFT, spin polarization,
transition-state search, or a general optimizer plugin system. These remain
separate roadmap work.
