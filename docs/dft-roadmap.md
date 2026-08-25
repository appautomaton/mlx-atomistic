# DFT Roadmap

This roadmap defines the bounded path from the current specialized periodic
runtime to a general core for common solid-state DFT. It does not target full
feature parity with Quantum ESPRESSO or CP2K. MLX remains the product runtime;
reference engines remain validation surfaces.

## Current Baseline

The retained periodic path provides PBE-PW92, reciprocal-space GTH operators,
weighted k-points, block-Davidson eigensolves, fixed and Fermi-Dirac
occupations, frozen-density bands, analytic fixed-cell forces, and atomic
checkpoint/resume. Material-level verification remains narrow: Silicon and
Carbon have accepted equation-of-state workloads, while MgO retains a declared
force and transferability boundary. Metallic execution is covered, but a
matched metallic accuracy reference is not yet closed.

## Program Boundary

The target is a general solid-state core that supports common insulating,
metallic, and collinear-magnetic crystals across ordinary Bravais cells. It
must provide trustworthy energy, bands, forces, stress, ionic relaxation, cell
relaxation, and restart behavior within an explicit pseudopotential and
functional envelope.

The dependency order is:

```text
metallic protocol lock -> metallic golden validation

analytic forces -> fixed-cell ionic relaxation
                         |
                         v
general cell geometry -> stress -> variable-cell relaxation -> phonons
          |
          v
periodic collinear spin

pseudopotential transferability expands across every phase
```

## Phase 1: Close Metallic Scientific Validation

Status: active research; no product-code change starts before protocol lock.

- Lock one Aluminum reference protocol with the same structure,
  pseudopotential identity, Fermi-Dirac width, k-point mesh, cutoff, and
  free-energy definition used locally.
- Retain source fingerprints and parsed inputs so the comparison is
  reproducible without placing a reference engine on the MLX runtime path.
- Validate electron count, chemical potential, occupations, free energy, and
  at least one material observable such as an equation-of-state curve.

Exit gate: a source-bound metallic report passes declared numerical and
scientific thresholds. The current Aluminum smoke remains execution evidence
until this gate closes.

## Phase 2: Add Periodic Fixed-Cell Ionic Relaxation

Status: planned after Phase 1 scientific closure.

- Add a periodic optimizer that consumes converged analytic forces.
- Reuse SCF density and compact eigenspaces between accepted ionic steps.
- Preserve fixed cell, k-point, pseudopotential, checkpoint, and provenance
  identity throughout the trajectory.
- Fail closed on unconverged SCF states, non-finite forces, and unsupported cell
  modes.

Exit gate: force and displacement convergence are reproducible across restart,
energy decreases according to the optimizer contract, and at least one
source-bound crystal relaxation agrees with a reference geometry.

## Phase 3: Generalize Periodic Cell Geometry

Status: planned after the fixed-cell relaxation contract is stable.

- Replace orthorhombic-only assumptions with one full-rank `3 x 3` cell matrix
  contract across real and reciprocal grids, plane-wave bases, k-points,
  Ewald terms, local and nonlocal GTH operators, forces, and fingerprints.
- Preserve the existing orthorhombic numerical trajectory as a strict
  compatibility case.
- Add representative cubic, hexagonal, and low-symmetry cells.

Exit gate: cell-coordinate transforms, reciprocal identities, energies, and
forces pass analytic or finite-difference checks, and accepted orthorhombic
goldens do not regress.

## Phase 4: Add Stress And Variable-Cell Relaxation

Status: blocked on Phase 3 cell geometry.

- Establish a reliable periodic stress tensor with an explicit sign, unit, and
  free-energy convention.
- Validate stress against controlled cell finite differences before using it
  in optimization.
- Add bounded cell-only and coupled ion/cell relaxation with restartable state.

Exit gate: stress passes numerical derivatives for insulating and smeared
systems, and relaxed cells reproduce declared reference lattice observables.

## Phase 5: Add Periodic Collinear Spin

Status: planned after the general cell contract to avoid duplicating a global
geometry refactor across spin channels.

- Carry separate spin-up and spin-down densities, occupations, potentials, and
  convergence diagnostics through periodic SCF.
- Support fixed magnetization and unconstrained collinear modes with explicit
  electron-count contracts.
- Add non-magnetic equivalence and magnetic material golden cases.

Exit gate: the unpolarized limit reproduces the existing path, total charge and
magnetization are conserved, and at least one magnetic crystal passes a
source-bound energy and moment comparison.

## Phase 6: Expand Pseudopotential Transferability

Status: evidence expands incrementally with every earlier phase; the final
matrix closes after spin support.

- Strengthen GTH and UPF convention fidelity instead of treating parser success
  as scientific validation.
- Cover representative `s`, `p`, and `d`-block elements, ionic compounds,
  multiple oxidation environments, and both local and nonlocal force terms.
- Bind every claim to an exact resource fingerprint and matching reference
  protocol.

Exit gate: a multi-material matrix passes locked energy, structural, and force
thresholds without element-specific runtime branches.

## Phase 7: Add Finite-Displacement Phonons

Status: blocked on stable forces, ionic relaxation, cell geometry, and restart.

- Build force-constant matrices from symmetry-independent finite
  displacements only after force, relaxation, cell, and restart contracts are
  stable.
- Enforce translational sum rules as diagnostics rather than silently repairing
  invalid force data.
- Validate frequencies and eigenvectors for bounded crystals.

Exit gate: displacement convergence, acoustic modes, restart equivalence, and
reference frequencies pass declared thresholds.

## General-Core Exit Criteria

The project may describe the periodic runtime as a general core for common
solid-state DFT only when all of the following hold:

- ordinary full-rank crystal cells are supported;
- insulating, simple metallic, and collinear-magnetic paths have material
  goldens;
- energy, bands, forces, stress, fixed-cell relaxation, and variable-cell
  relaxation form consistent workflows;
- GTH or UPF claims are backed by a multi-material transferability matrix;
- checkpoint/restart preserves every supported scientific state;
- complete-wall performance and peak memory remain measured on representative
  Apple Silicon workloads;
- unsupported physics fails closed and remains documented.

## Deferred Beyond The General Core

Non-collinear spin, spin-orbit coupling, hybrid functionals, time-dependent
DFT, density-functional perturbation theory, electron-phonon workflows,
reaction-path methods, distributed execution, and broad molecular DFT remain
outside this roadmap. They require separate scientific programs rather than
being appended to an active core feature.

## Delivery Rule

Each phase begins with source and protocol research. Implementation starts only
after inputs, numerical semantics, acceptance thresholds, and evidence labels
are locked. Automaton may track one bounded implementation phase after that
lock; it does not own this program-level scientific roadmap.
