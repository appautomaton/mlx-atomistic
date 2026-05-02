# Slice 5 Orchestration: PME NonbondedPotential Integration

## Scope

- Change: `gpcrmd-critical-md-readiness`
- Slice: `slice_5_pme_nonbondedpotential_integration`
- Route: subagent implementation with spec and quality review
- Files in scope:
  - `src/mlx_atomistic/nonbonded.py`
  - `src/mlx_atomistic/forcefields.py`
  - `tests/test_forcefields.py`
  - `tests/test_pme.py`
- Narrow test update:
  - `tests/test_nonbonded_acceleration.py` was updated because the required selector still asserted PME validation must raise `NotImplementedError`.

## Implementation

- Implementer `019de579-1d68-7290-8be5-d881408e83c1` made `electrostatics="pme"` executable in `NonbondedPotential`.
- `NonbondedPotential` now accepts explicit `pme_config`.
- PME mode evaluates full-system PME Coulomb, direct-space LJ, explicit exception LJ/Coulomb, and exclusion/1-4 Coulomb corrections.
- PME mode rejects missing `pme_config`, missing cell, restricted-pair evaluation, invalid cells, and non-neutral systems.
- PME component reporting includes real, reciprocal, self, exclusion correction, exception, 1-4 correction, and diagnostics.

## Reviews

- Spec review `019de57d-1768-7913-a6dc-f219d1272781`: `APPROVED`.
- First quality review `019de57f-0739-70d3-971c-57122fb94af5`: `CHANGES_REQUESTED`.
  - Required fail-closed validation for non-finite Coulomb constant and invalid PME config fields.
  - Required PME exception-correction coverage with a nonzero exception charge-product override.
- Fix implementer `019de579-1d68-7290-8be5-d881408e83c1`: `DONE`.
  - Added non-finite/invalid validation before PME evaluation.
  - Added nonzero exception override energy/component/force tests.
- Quality re-review `019de583-665e-7ee1-953e-c97a02540017`: `APPROVED`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "pme or ewald" tests/test_pme.py`
  - Result: `20 passed, 12 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "pme or nonbonded or forcefields"`
  - Result: `54 passed, 196 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/nonbonded.py src/mlx_atomistic/forcefields.py tests/test_forcefields.py tests/test_pme.py tests/test_nonbonded_acceleration.py`
  - Result: `All checks passed`

## Notes

- PME artifact schema and importer wiring remain later slices.
- This slice does not claim GPCRmd runtime readiness by itself.
