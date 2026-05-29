# Slice 5: GBSA/OBC Implicit Solvent

## Status: DONE

## Summary

Implemented `GBSAForcePotential` with GB-OBC Born radii, electrostatic solvation energy, ACE surface-area energy, and MLX-gradient forces. Added GBSA artifact support and persistence through `PreparedSystem` save/load. Added OpenMM GB-OBC parity against a tracked protein fixture and regressions for full-pair NoCutoff GBSA behavior under MD runtime pair passing.

## Files Changed

- `src/mlx_atomistic/gbsa.py`: New GBSA/OBC + ACE force term, full-pair NoCutoff semantics.
- `src/mlx_atomistic/forcefields.py`: Re-exported `GBSAForcePotential`.
- `src/mlx_atomistic/artifacts.py`: Added GBSA supported-term handling, validation, and runtime term construction.
- `src/mlx_atomistic/__init__.py`: Package export for `GBSAForcePotential`.
- `src/mlx_atomistic/prep/schema.py`: GBSA prepared-system fields and validation.
- `src/mlx_atomistic/prep/io.py`: GBSA arrays included in optional NPZ save/load path.
- `tests/test_gbsa.py`: GBSA correctness, artifact, persistence, OpenMM parity, and runtime-pair regression tests.

## Verification

- `uv run pytest tests/test_gbsa.py tests/test_production_artifacts.py -k "gbsa or implicit or obc" && uv run ruff check src/mlx_atomistic/gbsa.py src/mlx_atomistic/forcefields.py`: `8 passed, 73 deselected`; ruff passed.
- `uv run pytest`: `712 passed`.
- `git diff --check`: passed.

## Reviewer Verdicts

- Spec review: CHANGES_REQUESTED for GBSA persistence and protein fixture parity, then APPROVED after fixes.
- Quality review: CHANGES_REQUESTED for runtime neighbor-pair truncation, then APPROVED after GBSA switched to full-pair NoCutoff semantics.

## Unresolved Risks

- None for Slice 5.
