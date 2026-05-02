# Slice 3 Orchestration: CHARMM/GPCR Force-Term Primitives

## Scope

- Change: `gpcrmd-critical-md-readiness`
- Slice: `slice_3_charmm_gpcr_force_term_primitives`
- Route: subagent implementation with spec and quality review
- Files in scope:
  - `src/mlx_atomistic/charmm_terms.py`
  - `src/mlx_atomistic/forcefields.py`
  - `tests/test_charmm_terms.py`

## Implementation

- Implementer `019de551-6ec5-76d1-aef1-5a2350f4edad` added CHARMM/GPCR force-term primitives:
  - `CHARMMCMAPPotential`
  - `CHARMMUreyBradleyPotential`
  - `CHARMMForceSwitchNonbondedPotential`
  - `CHARMMNBFIXPairOverridePotential`
- Public CHARMM names are re-exported from `mlx_atomistic.forcefields`.
- Tests cover finite energy/force behavior and finite-difference checks where practical.

## Reviews

- Spec review `019de559-350e-7c11-85bc-e19b988c6870`: `APPROVED`.
- First quality review `019de55a-9fa2-7471-b109-9e368fe17786`: `CHANGES_REQUESTED`.
  - Required a real CHARMM LJ force-switch primitive instead of generic smooth energy switching.
  - Required NBFIX restricted-pair handling to fail closed or filter consistently.
  - Required CMAP reference tests for grid nodes, periodic seam, and multiple maps.
  - Required stricter validation for invalid Urey-Bradley and NBFIX inputs.
- Fix implementer `019de55d-3217-7193-8bec-49f4895c8eb8`: `DONE`.
  - Implemented CHARMM LJ force-switch behavior with reference-value tests.
  - Made NBFIX restricted `pairs=` evaluation fail closed consistently.
  - Added CMAP node, seam, and multi-map tests.
  - Added invalid Urey-Bradley and NBFIX validation tests.
- Second quality review `019de566-08d2-7c32-9670-c7d8df2dc205`: `CHANGES_REQUESTED`.
  - Required fail-closed validation for non-finite NBFIX base nonbonded arrays and cutoff/switch settings.
  - Required fail-closed validation for non-finite force-switch Coulomb constant.
- Final fix implementer `019de55d-3217-7193-8bec-49f4895c8eb8`: `DONE`.
  - Added non-finite/invalid validation for NBFIX base parameters.
  - Added non-finite Coulomb constant validation for the force-switch primitive.
- Final quality re-review `019de569-f500-7fd3-b00c-b41ffd9c2b99`: `APPROVED`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_charmm_terms.py`
  - Result: `31 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "charmm or cmap or nbfix"`
  - Result: `31 passed, 208 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/charmm_terms.py src/mlx_atomistic/forcefields.py tests/test_charmm_terms.py`
  - Result: `All checks passed`

## Notes

- Artifact importer wiring remains a later slice.
- PME wiring remains a later slice.
- Pre-existing dirty changes in `src/mlx_atomistic/forcefields.py` were not reverted.
