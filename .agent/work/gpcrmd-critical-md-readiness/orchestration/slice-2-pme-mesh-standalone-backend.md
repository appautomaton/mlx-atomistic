# Slice 2 Orchestration: PME Mesh Standalone Backend

## Route

- Route used: subagent implementation with coordinator verification.
- Implementer: `019de541-71e5-74d3-9fad-8c8183475e40` (`DONE`).
- Spec reviewer: `019de548-074e-7c50-93bd-fbd6dc2227f3` (`APPROVED`).
- Quality reviewer first pass: `019de549-6320-7992-bd9b-4c24c2d810a8` (`CHANGES_REQUESTED`).
- Fix implementer: `019de54b-cb25-7e00-beb8-e32992b15db6` (`DONE`).
- Quality reviewer final pass: `019de54d-c728-7b31-8adb-6f0435f15dc2` (`APPROVED`).

## Scope

- Executed only Slice 2.
- Added standalone PME mesh electrostatics.
- Did not integrate PME into `NonbondedPotential`, artifacts, runtime commands, notebooks, CHARMM terms, or neighbor-list execution.
- No external MD engines were used.

## Evidence

- Added `src/mlx_atomistic/pme.py` with:
  - `PMEConfig`;
  - CIC charge assignment;
  - FFT reciprocal mesh solve;
  - influence function;
  - CIC interpolation;
  - PME energy/forces;
  - diagnostics.
- Added `tests/test_pme.py` with:
  - charge-assignment conservation;
  - Ewald-reference comparison;
  - wrapping invariance;
  - non-neutral refusal;
  - invalid mesh/cell refusal;
  - non-finite input refusal;
  - benchmark comparison assertions.
- Extended `src/mlx_atomistic/benchmarks/ewald_reference.py` with PME comparison fields and `--pme-mesh`.
- Quality review found and the fix addressed two fail-closed validation gaps:
  - non-finite positions/charges now raise;
  - fractional mesh dimensions now raise instead of truncating.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or ewald"`: `27 passed, 181 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/pme.py src/mlx_atomistic/benchmarks/ewald_reference.py tests/test_pme.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.ewald_reference --atoms 4 --evaluations 1 --json`: passed.
  - `pme_finite`: `true`
  - `pme_energy_abs_error`: `1.6093254089355469e-06`
  - `pme_force_max_abs_error`: `1.0779127478599548e-05`

## Stop Or Continue

- Slice 2 has `Auto-continue: yes`.
- Next slice is Slice 3, with Slice 4 still parallel-safe if the next execution window chooses to dispatch it separately.
