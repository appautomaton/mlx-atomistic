# Slice 7 Code Quality Review: Native GROMACS Top/Gro Import

## Status

APPROVED

## Summary

- Native GROMACS importer and routing changes are scoped to Slice 7 surfaces.
- The prior `gen-pairs = no` gap is corrected: `[ pairs ]` records now fail closed unless generated pair parameters are enabled.
- No maintainability or regression blockers remain in the reviewed GROMACS importer/routing surface.

## Issues

- none

## Evidence

- `src/mlx_atomistic/prep/gromacs.py` validates `gen-pairs = no` with `[ pairs ]` as `unsupported_terms:gromacs_pairs_without_generated_parameters`.
- `src/mlx_atomistic/prep/gromacs.py` continues to reject explicit pair parameter rows as `unsupported_terms:gromacs_explicit_pair_parameters`.
- `tests/test_gromacs_import.py` covers the `gen-pairs = no` mutation while retaining `[ pairs ]`.
- Required Slice 7 pytest passed: `12 passed, 123 deselected`.
- Targeted Ruff passed for the touched Slice 7 files.
- `git diff --check` passed for the touched Slice 7 files.

## Residual Risk

- Broader production `.top` coverage remains intentionally narrow because preprocessor features, `[pairtypes]`, explicit pair parameters, and generated pairs without records fail closed.
