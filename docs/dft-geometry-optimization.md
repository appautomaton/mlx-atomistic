# DFT Geometry Optimization

Milestone 5 adds a fixed-cell relaxation workflow for the current DFT stack. It optimizes only ion-center positions and keeps the orthorhombic cell fixed. Forces come from `run_scf(...).forces`, so the workflow is only as physical as the local pseudopotential force model underneath it.

This is a workflow and consistency milestone. It is not a claim that the current DFT layer is chemically validated production DFT.

## Workflow

The public entry point is:

```python
from mlx_atomistic.dft import GeometryOptimizationConfig, optimize_geometry

result = optimize_geometry(system, config=GeometryOptimizationConfig(max_steps=5))
```

Each geometry step does:

1. Run SCF at the current ion positions.
2. Use the current force array `F = -∂E/∂R`.
3. Propose an ion-position step with an L-BFGS-style inverse-Hessian update.
4. Fall back to steepest descent if the L-BFGS direction is invalid.
5. Backtrack the step size until the SCF energy is finite and not higher than the previous accepted geometry.
6. Wrap accepted ion positions back into the periodic cell.
7. Reuse the previous SCF density and orbitals as continuation inputs for the next line-search trial.

The optimizer accepts SCF results with status `converged` or `max_iterations`. It rejects SCF failures, nonfinite energies, and exhausted line searches.

## Results

`GeometryOptimizationResult.to_dict()` is JSON-safe and includes:

- final status, final energy, final positions, and final maximum force;
- per-step energy, energy delta, maximum force, RMS force, force norm, step norm, and accepted step size;
- per-step SCF status, SCF iteration count, residual, electron count, and timing summary;
- final SCF summary without dense array payloads.

Valid statuses are:

- `converged`
- `max_steps`
- `line_search_failed`
- `scf_failed`
- `nonfinite`

## Restart-Like Continuation

Within one `optimize_geometry(...)` run, SCF is continued by passing the previous step's density and orbitals into `run_scf(...)`. This is a practical acceleration and stability feature for local relaxation workflows because adjacent geometries usually have similar densities.

The compressed NPZ helpers save and load the accepted geometry history:

```python
from mlx_atomistic.dft import save_geometry_optimization, load_geometry_optimization

save_geometry_optimization("relaxation.npz", result, metadata={"system": "gaussian-dimer"})
record = load_geometry_optimization("relaxation.npz")
```

The NPZ record stores positions, forces, energies, maximum forces, statuses, JSON metadata, and JSON step history. It does not store enough dense SCF state to resume a calculation by itself yet.

## CLI

Compact built-in demos are available through:

```bash
uv run python -m mlx_atomistic.dft.optimize --system gaussian-dimer --steps 5 --json
```

Supported built-in demo systems:

- `gaussian-dimer`: toy two-center Gaussian local potential.

The benchmark entry point runs the same self-contained demo by default:

```bash
uv run python -m mlx_atomistic.benchmarks.dft_geometry --json
```

## Current Scientific Boundary

This milestone is still a proof-level fixed-cell relaxation workflow, not a
chemically certified production materials optimizer. Spin/k-point diagnostics,
nonlocal projectors, finite-difference stress, and geometry optimization exist as
prototype surfaces; production validation and cell relaxation remain out of scope.
