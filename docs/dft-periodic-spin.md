# DFT Periodic Collinear Spin

This document is the scientific and engineering contract for Phase 5 of the
[DFT roadmap](./dft-roadmap.md). It extends the existing periodic PBE/GTH
runtime with collinear spin without creating a second SCF controller.

## Objective

Support common collinear-magnetic solids with explicit spin-up and spin-down
densities, potentials, occupations, and diagnostics. Preserve the current
unpolarized trajectory byte-for-byte at the public contract boundary wherever
floating execution order is unchanged.

Non-collinear magnetism, spin-orbit coupling, constrained local moments,
DFT+U, and magnetic forces or stress are separate phases.

## Public Contract

`PeriodicSCFConfig` retains `unpolarized` as its default. Collinear runs select
one of two electron-allocation modes:

- `fixed_magnetization`: the caller supplies total magnetization
  `M = N_up - N_down`; `N_up = (N + M) / 2` and
  `N_down = (N - M) / 2` remain fixed throughout SCF.
- `unconstrained`: both channels share one Fermi level and electrons may move
  between them. This mode requires finite-temperature occupations so level
  crossings remain well defined.

Unconstrained runs may supply an `initial_magnetization` seed. The seed only
splits the initial channel densities; it does not constrain the converged
moment. Fixed-magnetization runs reject a separate seed because their channel
electron counts already define the initial and accepted moment.

The result reports total density, the two spin densities, magnetization
density, integrated magnetization, per-channel k-point states and electron
counts, and either one shared or two fixed-channel chemical potentials. Total
charge and the requested fixed magnetization are hard convergence gates.

## Exchange-Correlation Functional

The production functional is spin-PBE with the Perdew-Wang 1992 uniform-gas
baseline. Exchange uses exact spin scaling:

```text
E_x[n_up, n_down] = 1/2 E_x^PBE[2 n_up] + 1/2 E_x^PBE[2 n_down]
```

Correlation uses PW92 spin interpolation for the local baseline and the PBE
spin-scaling factor

```text
zeta = (n_up - n_down) / (n_up + n_down)
phi = ((1 + zeta)^(2/3) + (1 - zeta)^(2/3)) / 2
```

in the PBE gradient correction. The two local potentials are obtained from one
MLX automatic-differentiation graph. At `n_up = n_down`, total energy, total
density, and both channel potentials must reproduce the existing unpolarized
PBE path within locked float32 tolerances.

## Runtime Architecture

The physical channel dimension belongs above the existing k-point scheduler:

```text
shared cell / GTH / Hartree(total density)
                    |
          spin-PBE potentials [up, down]
                    |
       existing k-point + Davidson machinery
             /                     \
       up channel               down channel
             \                     /
         occupation and density reduction
```

Cell geometry, plane-wave bases, local GTH, nonlocal GTH, Ewald energy,
k-point symmetry, compact batching, and Davidson code remain shared. Each spin
channel receives `V_local + V_H[n_up+n_down] + V_xc,spin`; no spin-specific
copy of kinetic or pseudopotential code is allowed.

The mixer acts on charge and magnetization channels, not on two unrelated
densities. This makes the unpolarized subspace explicit and permits separate
bounded damping of magnetic oscillations.

The float32 spin-PBE graph applies a `1e-7 bohr^-3` per-channel numerical floor
and a bounded polarization edge. These controls keep the functional derivative
finite for a completely empty minority channel. Channel density construction,
normalization, and reported electron counts still preserve an exactly empty
channel when its occupation target is zero.

## Implementation Status

The shared controller, fixed and unconstrained occupation modes, charge and
magnetization mixing, explicit channel results, non-magnetic equivalence, and
atomic checkpoint/resume are implemented. Checkpoints persist total, spin-up,
spin-down, compact orbital, charge-mixer, and magnetization-mixer state under
the existing validated artifact envelope.

The Phase 5 material exit gate is closed by the fingerprinted bcc Iron
PBE/GTH-q16 workload. Its one-atom primitive cell uses exact index-2 unfolding
of conventional-cell k-points, Fermi-Dirac width `0.01 Ha`, a 150 Ha selected
cutoff, and an unconstrained initial moment of `2.2` electrons.

The selected 4x4x4 calculation converged to `2.41795` Bohr magnetons per atom;
the 6x6x6 check gave `2.33319`, against the locked published PBE context value
of `2.33`. The 200 Ha cutoff check changed the moment by `0.00035` and free
energy by `0.000136 Ha/atom`; the k-point check changed them by `0.08476` and
`0.000812 Ha/atom`. The magnetic state is `0.023828 Ha/atom` below the matched
unpolarized state.

The selected point took `10.727 s` on an Apple M5 Max in battery low-power
mode. Runtime accounting reported `110.32 MB` peak temporary storage,
`15.66 MB` persistent compact coefficients, and `10.18 MB` persistent
projectors. These are logical runtime counters, not process resident memory.

Historical-frozen Fe GTH-PBE-q8 evidence remains an explicit transferability
failure: after cutoff and k-point convergence it retained about `2.98` Bohr
magnetons per atom, outside the locked `2.33 +/- 0.40` material gate. The
threshold was not weakened; the q8 evidence predates a telemetry-only runtime
fingerprint change and is not labeled current-verified.

## Acceptance Criteria

- spin-PBE energy and both potentials pass finite-difference derivatives;
- equal spin densities reproduce unpolarized PBE energy and potential;
- channel exchange is invariant under swapping up and down;
- fixed and unconstrained occupations conserve total electron count;
- fixed magnetization conserves `N_up - N_down` at every accepted iteration;
- zero-magnetization periodic SCF reproduces the existing unpolarized result;
- checkpoint/resume reproduces both spin densities and channel occupations;
- one source-bound magnetic crystal passes energy, moment, and numerical
  convergence gates without running a reference engine on the MLX path;
- complete wall time and peak memory are measured once after the protocol is
  locked.

## Delivery Order

1. Implement spin-PBE and spin-resolved occupation oracles.
2. Add immutable spin configuration and result contracts.
3. Lift periodic effective-potential, eigensolve, density, and mixing stages to
   a shared two-channel controller.
4. Add checkpoint identity and exact resume.
5. Close non-magnetic equivalence and bounded magnetic numerical gates.
6. Lock and run one source-bound magnetic material validation.
7. Synchronize capability documentation and merge only after CI passes.

## Evidence Boundary

Deterministic functional and engine tests establish numerical semantics, not
material accuracy. The material gate uses a fingerprinted GTH-PBE source,
cell, k-point mesh, cutoff, smearing width, and magnetic reference. The verified
numerical claim is limited to the Fe q16 workload. The current point-group
density-reconstruction oracle gives reduced and full free energies of
`-123.62366184` and `-123.62366589 Ha/atom`, a `4.0505e-6 Ha/atom` difference
below the locked `5e-5` gate. Their moments are `2.34075` and `2.33818` Bohr
magnetons per atom, a `0.002563` difference below the `0.02` gate. This closes
the point-group SCF method blocker. The older cutoff, k-point, and ordering
artifacts predate the v2 orbit contract, so Phase 7 still requires refreshed
exact identities before treating the complete q16 row as current
transferability evidence. The q8 failure remains explicit.
