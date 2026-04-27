# Units and MD Diagnostics

`mlx-atomistic` uses explicit internal unit systems instead of SI in the numerical kernels.

## Molecular Dynamics

The v1 MD engine uses Lennard-Jones reduced units. Inputs passed to MD kernels are
dimensionless unless a caller explicitly converts them before calling the API:

```text
σ = 1
ε = 1
m = 1
k_B = 1
```

Derived units:

```text
length      = σ
energy      = ε
mass        = m
time        = τ = σ sqrt(m / ε)
force       = ε / σ
temperature = ε / k_B
velocity    = σ / τ
```

Typical example values such as `dt = 0.005`, `temperature = 1.0`, `density = 0.8`,
and `cutoff = 2.5` are reduced-unit values.

The code exposes this convention as `mlx_atomistic.units.LJ_REDUCED_UNITS`.
For material-specific reduced-unit experiments, create a `LennardJonesReducedUnits`
instance with explicit `sigma`, `epsilon`, `mass`, and `boltzmann` values, then
convert external values at the API boundary.

## NVE Output Contract

`simulate_nve()` separates trajectory storage from diagnostics:

- `sample_interval` controls sparse trajectory frames only.
- `sampled_positions`, `sampled_velocities`, `sampled_steps`, and `sampled_time`
  are stored at step `0`, every sampled step, and the final step.
- `potential_energy`, `kinetic_energy`, `total_energy`, `temperature`,
  `pair_count`, and `rebuild_count` are dense per-step diagnostics with length
  `steps + 1`.

This lets long runs keep a small trajectory while still retaining enough scalar
diagnostics to judge integrator health.

For NVE, the first correctness check is total-energy drift:

```text
E(t) = U(t) + K(t)
ΔE(t) = E(t) - E(0)
```

`NVEResult.energy_drift` and `NVEResult.max_energy_drift` expose those values.
Small nonzero drift is expected from finite `dt` and floating-point arithmetic.
For a stable NVE smoke test, `max(|ΔE|)` should remain small over short runs and
should improve when `dt` is reduced.

## DFT

The planned DFT prototype will use atomic units internally:

```text
ℏ = 1
m_e = 1
e = 1
4πε₀ = 1
```

DFT user-facing conversion helpers can be added once the first Γ-point plane-wave pseudopotential prototype exists. Until then, examples should state their unit convention explicitly.
