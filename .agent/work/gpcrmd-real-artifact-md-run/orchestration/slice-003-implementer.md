# Slice 003 Implementer

- Slice: NBFIX Export Schema
- Status: completed
- Route: subagent implementation with one narrow follow-up fix

## Implementation Outcome

- Added compact NBFIX type-pair arrays: `nbfix_type_pairs`, `nbfix_type_sigma`, and `nbfix_type_epsilon`.
- Preserved legacy explicit atom-pair NBFIX arrays: `nbfix_pairs`, `nbfix_sigma`, and `nbfix_epsilon`.
- Exported ParmEd CHARMM atom-type NBFIX overrides with type identifiers, sigma in angstrom, and epsilon in kJ/mol.
- Added compatibility `term_details.nbfix_pair_overrides` with concrete converted values and source values.
- Added fail-closed import blockers for malformed, missing, conflicting, nonfinite, nonpositive, and distinct 1-4 NBFIX values.
- Added loader metadata-hiding validation for compact NBFIX arrays.

## Files Changed

- `src/atomistic_prep/schema.py`: compact NBFIX schema fields and validation.
- `src/atomistic_prep/io.py`: JSON/NPZ save-load defaults for compact NBFIX arrays.
- `src/atomistic_prep/topology_import.py`: ParmEd NBFIX type-pair extraction and reporting.
- `src/atomistic_prep/gpcrmd.py`: GPCRmd report propagation for concrete NBFIX values.
- `src/mlx_atomistic/artifacts.py`: undeclared-array guard for compact NBFIX arrays.
- `tests/test_atomistic_prep.py`: NBFIX export, round-trip, and fail-closed cases.
- `tests/test_production_artifacts.py`: compact NBFIX metadata-hiding regression coverage.

## Implementer Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_atomistic_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py -k "nbfix or charmm or gpcrmd"`: 41 passed, 53 deselected before follow-up; 43 passed, 53 deselected after follow-up.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py -k "nbfix or undeclared"`: 3 passed, 35 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/atomistic_prep/schema.py src/atomistic_prep/io.py src/atomistic_prep/topology_import.py src/atomistic_prep/gpcrmd.py tests/test_atomistic_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/artifacts.py tests/test_production_artifacts.py`: passed.
- Real GPCRmd 729 import probe exported `/tmp/mlx-atomistic-gpcrmd-729-slice3` with 92,001 atoms, 37 compact NBFIX type pairs, and 0 legacy explicit NBFIX pairs.

## Residual Concern

Runtime NBFIX semantics are still Slice 4. The Slice 3 artifact is strict and loadable, but `mlx_atomistic` still needs to apply compact type-pair overrides during nonbonded evaluation before the real GPCRmd system can be stepped.
