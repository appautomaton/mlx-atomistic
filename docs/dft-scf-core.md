# DFT SCF Core

The DFT layer now has a more explicit SCF core while staying intentionally
small: spin-unpolarized Γ-point SCF, orthorhombic cells, local Gaussian toy
pseudopotentials, and ion-backed GTH/UPF pseudopotential prototypes.

## Exchange-Correlation

DFT does not solve directly for the full many-electron wavefunction. In the
Kohn-Sham construction, auxiliary orbitals generate the density `ρ(r)`, and the
unknown many-electron physics is approximated by an exchange-correlation
functional.

This milestone exposes that as a small API:

- `DiracExchange`
- `LDACorrelationPZ81`
- `LDAExchangeCorrelation`
- `PBEExchangeCorrelation`

Each functional returns an energy density, total energy, and potential `v_xc`.
The default SCF path uses LDA exchange plus PZ81 correlation.
`PBEExchangeCorrelation` is public alpha API: it uses the PBE GGA gradient form
with the package's PZ81 uniform correlation baseline, so it reports itself as a
PBE-PZ81 alpha diagnostic rather than full PW92-backed production PBE.

## SCF Mixing

SCF iteration repeatedly maps an input density to a new output density. Directly
replacing the old density is often unstable, so the code now has two mixers:

- `LinearMixer`: conservative and predictable.
- `PulayDIISMixer`: uses residual history to accelerate convergence.

`SCFResult` records density residual, potential residual, energy delta,
convergence status, failure reason, timing totals, and mixer metadata.

## Restartable State

`run_scf(...)` accepts `initial_density` and `initial_orbitals`. This is enough
for continuation workflows and future geometry steps. A restart density is
renormalized to the requested electron count so continuation does not silently
change the number of electrons.

## Forces

`local_pseudopotential_forces(...)` computes Hellmann-Feynman-style forces on
Gaussian centers from a converged or fixed density. Ion-backed systems add
center-center forces, and systems that apply nonlocal projectors report a
fixed-orbital finite-difference nonlocal correction. These are diagnostic alpha
forces, not production DFT force claims.

## Performance Evidence

The DFT benchmark now reports per-case timings for Hartree solves, XC
evaluation, kinetic application, mixer cost, force evaluation, total SCF time,
and a compact FFT probe. This makes the next optimization decision evidence
based instead of speculative.
