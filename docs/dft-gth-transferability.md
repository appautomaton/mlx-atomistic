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

Point-group k-point reduction has a separate method gate. Aggregating orbit
weights is sufficient for invariant quadrature, but a self-consistent field
calculation must also reconstruct every rotated orbital density. Until that
runtime path exists, a reduced SCF is admitted only by a matched full-versus-
reduced material oracle. Time-reversal ownership remains valid because paired
states have the same real-space density.

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
  and total-force metrics remain outside their thresholds and the Fe q16
  full-versus-reduced SCF oracle fails;
- `identity_complete = false` because several older project-derived EOS
  summaries do not retain exact calculation and runtime fingerprints;
- `production_envelope_verified = false`.

The historical Mg-q10/O-q6 feasibility point used the two-atom primitive cell,
a 36-cubed FFT grid, eight occupied bands, and point-group-reduced k-points. It
took `40.989 s`, used `57,201,084` peak temporary bytes, and stopped after 80
SCF iterations with a `4.329e-6` orbital residual above the locked `2e-6`
limit. The point is also method-invalid for admission because it aggregated
orbit weights without reconstructing rotated SCF densities. It remains in the
matrix as rejected historical evidence; no seven-point q10 EOS was run.

The primitive geometry and reciprocal-basis transform remain valid. Future
MgO candidates use a full mesh with time reversal only unless a matched
full-versus-reduced SCF oracle passes. The same rule blocks the existing Fe q16
transferability row: its current matched oracle found a `0.005622 Ha/atom`
free-energy difference and a `0.07977` Bohr-magneton-per-atom moment difference,
above the locked `5e-5 Ha/atom` and `0.02` gates. The full-mesh moment is
`2.33818`, close to the published `2.33` context, but full-mesh cutoff,
k-point, and magnetic-ordering checks have not been completed.
