# DFT SCF Core

The DFT layer now has a more explicit SCF core while staying intentionally
small: spin-unpolarized, Γ-point, orthorhombic cells, and local Gaussian
pseudopotentials only.

## Exchange-Correlation

DFT does not solve directly for the full many-electron wavefunction. In the
Kohn-Sham construction, auxiliary orbitals generate the density `ρ(r)`, and the
unknown many-electron physics is approximated by an exchange-correlation
functional.

This milestone exposes that as a small API:

- `DiracExchange`
- `LDACorrelationPZ81`
- `LDAExchangeCorrelation`

Each functional returns an energy density, total energy, and potential `v_xc`.
The default SCF path now uses LDA exchange plus PZ81 correlation.

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
Gaussian centers from a converged or fixed density. These are diagnostic forces
for the toy local pseudopotential model, not production DFT forces and not yet a
geometry optimizer.

## Performance Evidence

The DFT benchmark now reports per-case timings for Hartree solves, XC
evaluation, kinetic application, mixer cost, force evaluation, total SCF time,
and a compact FFT probe. This makes the next optimization decision evidence
based instead of speculative.
