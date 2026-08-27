# DFT Pseudopotentials

The DFT layer includes an ion-model surface while keeping the engine small and
inspectable. The legacy teaching runtime and the periodic plane-wave runtime
share parsed UPF and GTH data but retain separate execution boundaries.

## What Is Implemented

- `PseudopotentialData` stores parsed local potential data, valence charge, and
  nonlocal metadata.
- `Ion` and `IonCollection` place parsed pseudopotentials at periodic ion
  centers.
- `LocalPseudopotentialField` builds `V_local(r)` on a real-space DFT grid.
- `NonlocalPseudopotentialOperator` applies ion-aware separable projectors when
  parsed projector metadata is available.
- `DFTSystem` accepts `IonCollection` and defaults the electron count to the
  sum of valence charges for neutral systems.
- `run_scf(...)` records pseudopotential diagnostics:
  `pseudopotential_format`, `ion_count`, `valence_electron_count`,
  `nonlocal_available`, and `nonlocal_applied`.

## UPF

`read_upf(path)` reads UPF v2-style XML files from Quantum ESPRESSO-style
sources. The current path uses:

- `PP_HEADER` for element and valence charge.
- `PP_MESH/PP_R` and `PP_RAB` for radial samples and quadrature weights.
- `PP_LOCAL` for the local potential.
- `PP_BETA.*` tags for nonlocal projector metadata.
- The complete symmetric `PP_DIJ` matrix, converted from Rydberg to Hartree.

The legacy teaching path interpolates the local potential onto its real-space
grid. Its proof-level nonlocal operator still consumes only the diagonal
couplings and is not a production UPF implementation.

The periodic layer implements a scalar norm-conserving UPF path. It follows
Quantum ESPRESSO's compensated radial transform: subtract `erf(r) / r` before
quadrature, restore the analytic reciprocal-space Coulomb tail, and use the
finite `G=0` alpha term. Literal source oracles and a numerical GTH-equivalence
test cover the transform and fixed-cell local force. The compact nonlocal
operator preserves the complete symmetric `PP_DIJ`, evaluates radial
projectors with normalized real harmonics through `l=2`, and reuses the same
bounded k-point batch backend as periodic GTH. Periodic SCF, frozen-density
bands, analytic fixed-cell forces, and checkpoint identity dispatch through
that format-neutral runtime boundary.

The periodic boundary is fail-closed. Only scalar norm-conserving input
without ultrasoft augmentation, PAW, spin-orbit terms, or nonlinear core
correction is currently eligible. Parsing an unsupported file preserves its
identity; it does not make that physics executable. Analytic stress and
variable-cell workflows also remain GTH-only.

UPF setup deduplicates identical `|G+k|` magnitudes and batches radial
transforms by angular channel and radial grid. This removes repeated spherical
Bessel evaluation without moving any SCF hot-path work off the device. A
source-bound Quantum ESPRESSO Si ONCV input passes the periodic SCF smoke gate;
that execution check is not material-accuracy certification.

## GTH

`read_gth(path, element=..., name=...)` reads both single GTH files and CP2K
database entries. The local GTH potential is evaluated analytically:

```text
V_local(r) = -Z_ion erf(r / √2 r_loc) / r
             + exp[-0.5(r/r_loc)²] Σᵢ cᵢ(r/r_loc)²ⁱ
```

The derivative of this local form is used for fixed-density ion-force checks.
GTH nonlocal channel metadata is parsed and applied by the same separable
operator path when projector metadata is present.

The periodic plane-wave GTH path implements normalized real spherical
harmonics through `l=2`, including the five d-channel projectors in the
Quantum ESPRESSO ordering. Analytic harmonic identities, Hermiticity, and cell
translation invariance are deterministic gates. This implementation support
does not by itself certify d-block material transferability. The bcc Iron
PBE/GTH-q16 cutoff and k-point study now passes the Phase 5 material gate,
while the matching q8 study retains a failed magnetic-moment gate. Phase 7
now computes a multi-material coverage and science matrix rather than
generalizing from q16. The current Fe q16 full-versus-reduced SCF oracle passes
after exact rotated-density reconstruction. Coverage is complete, but the
production GTH envelope remains unverified because locked MgO residuals and
older evidence-identity blockers are retained.

## Forces

For ion-backed systems, reported forces include:

```text
F_total = F_local electron-ion + F_center-center + F_nonlocal correction
```

The legacy nonlocal term is a fixed-orbital finite-difference correction. The
periodic GTH and scalar norm-conserving UPF paths instead use analytic
projector-phase derivatives at a converged fixed-cell SCF state. These
consistency gates do not by themselves establish broad material accuracy.

## Current Limits

- The legacy UPF nonlocal operator remains proof-level and diagonal-only. The
  separate periodic implementation consumes full `PP_DIJ` but is scientifically
  verified only at source-oracle and execution-smoke level.
- Ultrasoft, PAW, spin-orbit, nonlinear-core-correction, and UPF analytic-stress
  terms are not implemented.
- This pseudopotential milestone does not certify geometry or stress. The
  current periodic runtime separately provides verified fixed-cell Silicon
  relaxation and one bounded analytic-stress/variable-cell 2H-Silicon path;
  broader pseudopotential transferability remains open.
- Vendor checkouts remain reference material only; the package does not import
  Quantum ESPRESSO or CP2K code.

## Benchmark

Run:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --json
```

The benchmark compares compact Gaussian, UPF-local, and GTH-local SCF cases and
reports timing plus pseudopotential diagnostics when explicit pseudopotential
files are supplied. Without extra inputs, the installed package runs only the
self-contained Gaussian case:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --json
uv run python -m mlx_atomistic.benchmarks.dft_pseudopotential --upf path/to/pseudo.upf --gth path/to/pseudo.gth --gth-element H --json
```
