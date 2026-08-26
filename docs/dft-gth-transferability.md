# Periodic GTH Transferability

This document is the scientific and engineering contract for Phase 7 of the
[DFT roadmap](./dft-roadmap.md). It defines the production boundary of the
periodic Goedecker-Teter-Hutter (GTH) path. Universal pseudopotential accuracy
is not inferred from parser support, projector identities, or one material.

## Runtime Boundary

The product runtime remains element-agnostic. `PeriodicDFTSystem` supplies one
`PseudopotentialData` value per ion, and the local potential, nonlocal
projectors, forces, stress, and self-consistent field controller dispatch from
that data. Phase 7 must not add element names, material names, or fitted
corrections to the runtime path.

The admitted GTH envelope supports real nonlocal channels through angular
momentum `l = 2`. Parsed UPF input and GTH channels above this boundary remain
proof-level and are not promoted by this phase.

## Transferability Matrix

The matrix is computed from locked observables, thresholds, and evidence
identities. It must cover all of the following without selecting only passing
metrics from a source report:

- representative s-, p-, and d-block elements;
- covalent, metallic, ionic, and magnetic solid environments;
- homogeneous and mixed-species cells;
- local and nonlocal energy and force execution;
- insulating fixed occupations, metallic smearing, and collinear spin.

Every material record binds the exact pseudopotential resource, reference
bundle or protocol, calculation contract, and runtime identity used by its
evidence. A missing or malformed SHA-256 identity fails closed. A
historical-frozen or project-derived result stays labeled as such and cannot be
reported as current verification.

## Scientific Gates

Equation-of-state cases retain the shared locked thresholds in `dft_eos.py`.
Magnetic Iron retains its published-moment, magnetic-energy-ordering, cutoff,
and k-point gates. Force coverage requires both:

- deterministic local and nonlocal component derivatives across the admitted
  `l = 0`, `l = 1`, and `l = 2` projector envelope;
- at least one mixed-species self-consistent total-force comparison.

The existing Mg-q2/O-q6 MgO case passes its locked lattice, bulk-modulus, and
equation-of-state shape gates, but its bulk derivative remains outside the
strict 15 percent gate and three finite-difference force components remain
outside `1e-4 Ha/bohr`. Those values remain failures. Phase 7 may evaluate the
already prepared Mg-q10 alternative, but only after one bounded feasibility
point passes numerical, memory, and complete-wall controls. Thresholds are not
changed after observing a result.

## Report Semantics

The matrix has three independent outcomes:

- `coverage_complete`: every declared block, environment, occupation mode,
  species mode, and force component has evidence;
- `strict_science_passed`: every required locked scientific metric passes;
- `production_envelope_verified`: coverage and strict science both pass with
  source-bound current or accepted project-derived evidence.

Known residuals are emitted as explicit blockers. Coverage never converts a
scientific failure into a pass, and a single successful material never
generalizes the entire GTH family.

## Efficient Execution Policy

Deterministic parser, identity, projector, and matrix tests run before any
material calculation. Material work follows a fail-early ladder: one central
feasibility point, then only the adjacent cutoff or k-point samples required to
make a decision, and finally a full curve only for an admitted candidate.
Existing source-bound evidence is reused when its fingerprints still match.
No material run is repeated merely to reformat a report.

## Current Matrix Result

Run the deterministic report with:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_gth_transferability --json
```

The committed contract currently reports:

- `coverage_complete = true` across the declared block, environment, species,
  occupation, and force axes;
- `strict_science_passed = false` because the locked Mg-q2/O-q6 bulk-derivative
  and total-force metrics remain outside their thresholds;
- `identity_complete = false` because several older project-derived EOS
  summaries do not retain exact calculation and runtime fingerprints;
- `production_envelope_verified = false`.

The one permitted Mg-q10/O-q6 feasibility point used the two-atom primitive
cell, a 36-cubed FFT grid, eight occupied bands, and cubic k-point symmetry. It
reduced complete wall time from an interrupted conventional-cell run exceeding
five minutes to `40.989 s`, with `57,201,084` peak temporary bytes. It was not
admitted: after 80 SCF iterations its maximum orbital residual was
`4.329e-6`, above the locked `2e-6` limit. No seven-point q10 EOS was run.

The primitive representation and reusable reciprocal-symmetry basis transform
are retained because their deterministic geometry and quadrature gates pass.
They improve future validation efficiency without adding a material branch to
the product runtime. The failed q10 candidate remains in the matrix rather than
being deleted or reclassified.
