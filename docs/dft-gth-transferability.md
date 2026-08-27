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
calculation must also reconstruct every rotated orbital density. The runtime
now retains exact orbit operations and applies device-resident FFT-grid
permutations to scalar and collinear-spin densities. A matched full-versus-
reduced material oracle remains required for scientific admission. Time-
reversal ownership composes with the same plan because paired states have the
same real-space density.

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

The current Mg-q10/O-q6 feasibility point used the two-atom primitive cell, a
36-cubed FFT grid, eight occupied bands, and point-group density reconstruction.
It took `39.892 s`, used `63,770,520` peak temporary bytes, and stopped after 80
SCF iterations with a `3.5695e-6` direct orbital residual above the locked
`2e-6` limit. A continuation seeded from that final density and using direct
Rayleigh-quotient refinement improved the residual to `3.0126e-6` without an
extra Hamiltonian application, but it also reached 80 iterations and remained
outside the gate. The remaining blocker is orthogonal subspace convergence,
not stale projected eigenvalues. The fail-early policy therefore skipped the
matched full-mesh oracle and seven-point q10 EOS.

The primitive geometry and reciprocal-basis transform remain valid. A future
MgO candidate may use a v2 point-group mesh only after a matched full-versus-
reduced SCF oracle passes. The current Fe q16 oracle now passes: the reduced and
full free energies differ by `4.0505e-6 Ha/atom`, and their moments differ by
`0.002563` Bohr magnetons per atom, below the locked `5e-5 Ha/atom` and `0.02`
gates. The full-mesh moment is `2.33818`, close to the published `2.33` context.
The row remains project-derived because its older cutoff, k-point, and magnetic-
ordering evidence has not yet been regenerated with complete v2 calculation
and runtime identities.

A separate historical-frozen screen evaluated the CP2K 2026.1 UZH PBE
Mg-q2/O-q6 resources, which CP2K recommends as a matched UZH protocol for new
GPW inputs. The source-bound local workload fingerprint was
`7990b1e9302d23dfab55059dac0242f80418c31bae406bb4f7627cbc9aba492b`.
At the established 70 Ha and 6-by-6-by-6 conventional-cell representation,
point-group reconstruction reproduced the matched full central energy within
`6.90e-10 Ha/cell` and reduced wall from `105.081` to `8.840 s`. The seven-point
curve gave `a₀ = 4.252650 Å`, `B₀ = 146.708 GPa`, `B₀′ = 3.35102`, and Δ
`0.461 meV/atom`. Its bulk-derivative relative error was `18.09%`, above the
unchanged `15%` gate and worse than the standard q2 result. The UZH candidate
is therefore rejected, and its force ladder was not run.

The matching UZH Mg-q10/O-q6 primitive central feasibility point was also
rejected before a full-mesh oracle. It reached 80 SCF iterations in `39.286 s`
with a `5.4187e-6` direct orbital residual, above the unchanged `2e-6` gate.
Its density and energy criteria passed, so neither a full mesh nor an EOS curve
was run. This closes the prepared UZH q10 candidate without weakening the
eigensolver requirement.
