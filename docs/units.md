# Units and MD Diagnostics

`mlx-atomistic` uses explicit internal unit systems instead of SI in the numerical kernels.

## Molecular Dynamics

Low-level MD kernels use Lennard-Jones reduced units. Inputs passed directly to
those kernels are dimensionless unless a caller explicitly converts them before
calling the API:

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

Prepared-system artifacts are the physical-unit path. Their metadata records
coordinate, mass, charge, energy, time, and temperature units, and the artifact
gate converts accepted AMBER, CHARMM, and GROMACS subsets into runtime force
terms that preserve that declared unit contract.

## NVE Output Contract

`simulate_nve()` separates trajectory storage from diagnostics:

- `sample_interval` controls sparse trajectory frames only.
- `sampled_positions`, `sampled_velocities`, `sampled_steps`, and `sampled_time`
  are stored at step `0`, every sampled step, and the final step.
- `potential_energy`, `kinetic_energy`, `total_energy`, `temperature`,
  `pair_count`, and `rebuild_count` are dense per-step diagnostics with length
  `steps + 1`.
- `potential_energy_by_term` stores dense per-term potential-energy series when
  multiple force terms are composed.

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

## NVT and Langevin Dynamics

`simulate_nvt()` uses a Langevin thermostat with BAOAB integration. It keeps the
same sparse trajectory and dense diagnostics contract as `simulate_nve()`, but
adds `target_temperature` and `temperature_error`.

The v1 thermostat parameters are:

```text
T = target reduced temperature
γ = friction in τ⁻¹
```

`γ = 0` disables stochastic thermostatting, so the NVT integrator reduces to the
same force/position/velocity updates as NVE. With `γ > 0`, random kicks exchange
energy with an implicit heat bath. That means total energy is not conserved in
NVT; use temperature statistics rather than `ΔE(t)` as the primary health check.

Seeded NVT runs use local MLX PRNG keys, so the same seed should reproduce the
same trajectory without changing global random state.

## Diagnostic Summaries

`mlx_atomistic.diagnostics.summarize_md_result()` converts an MD result into a
small dictionary of Python scalars for notebooks and CLI output. It reports
initial/final/mean temperature, total-energy drift, and neighbor-list pair/rebuild
counts when those fields are present. It also reports per-term potential-energy
summaries when energy decomposition is available. For NVT results it reports
target temperature and final/mean temperature error.

When distance constraints are active, NVE/NVT results also include dense
`constraint_max_error` diagnostics in the same reduced length unit as positions.

## DFT

The DFT prototype uses atomic units internally:

```text
ℏ = 1
m_e = 1
e = 1
4πε₀ = 1
```

The current Γ-point plane-wave prototype keeps examples explicit about units.
User-facing conversion helpers can be added once the DFT API grows beyond toy
systems.
