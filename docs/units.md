# Units

`mlx-atomistic` uses explicit internal unit systems instead of SI in the numerical kernels.

## Molecular Dynamics

The v1 MD engine uses Lennard-Jones reduced units:

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

Typical example values such as `dt = 0.005`, `temperature = 1.0`, `density = 0.8`, and `cutoff = 2.5` are reduced-unit values.

## DFT

The planned DFT prototype will use atomic units internally:

```text
ℏ = 1
m_e = 1
e = 1
4πε₀ = 1
```

DFT user-facing conversion helpers can be added once the first Γ-point plane-wave pseudopotential prototype exists. Until then, examples should state their unit convention explicitly.
