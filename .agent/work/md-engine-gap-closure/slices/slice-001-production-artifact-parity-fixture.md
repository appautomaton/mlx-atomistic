# Slice 001: Production Artifact Parity Fixture Detail

## Purpose

Create the shared evidence fixture that every later MD engine gap uses. This is
the first implementation slice because PME, NPT, checkpointing, DCD/XTC output,
and performance all need a common system where MLX and OpenMM can be compared at
the same coordinates.

## Execution Contract

1. Inventory candidate prepared artifacts and fixture factories already present
   in the repo.
2. Prefer the smallest artifact that is realistic, production-gated, and can be
   represented by both MLX and OpenMM.
3. Prefer AMBER first if a small artifact is available because it exercises the
   most common fixed-charge path with fewer CHARMM-specific terms.
4. Use CHARMM only if the available AMBER path is not realistic or cannot build
   an OpenMM reference.
5. Do not use toy Lennard-Jones-only systems as production evidence.
6. Keep large generated artifacts and parity output under
   `results/md-engine-gap-closure/parity-fixture/`.

## Required Harness Behavior

- Load the selected artifact through the production artifact path.
- Build MLX runtime system and force terms from the artifact.
- Build an OpenMM reference from the same topology/parameter source.
- Evaluate at one fixed coordinate frame first.
- Compare:
  - total potential energy
  - component energies where names can be mapped honestly
  - force array shape, finiteness, and tolerance statistics
  - unsupported or unmapped force terms
- Write a small machine-readable parity report under `results/`.
- Keep tests small enough for routine local execution.

## Stop Conditions

Stop and record the blocker instead of forcing the harness if:

- no candidate artifact loads with `require_production=True`;
- the selected artifact needs unsupported terms that cannot be failed closed;
- OpenMM cannot build the same reference system from available source files;
- energy units, coordinate units, or component mapping cannot be made explicit.

## Expected Verification

```sh
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_openmm_mlx_parity.py
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/run_openmm_mlx_parity.py --fixture <selected-fixture> --out results/md-engine-gap-closure/parity-fixture
```

## Handoff Decision

After this slice:

- If force/energy parity passes for the fixed-coordinate fixture, proceed to PME
  production readiness.
- If parity fails in bonded, exception, CMAP, NBFIX, or unit conversion paths,
  fix artifact conversion/term mapping before PME.
- If only PME is missing, keep the fixture and start Slice 2.
