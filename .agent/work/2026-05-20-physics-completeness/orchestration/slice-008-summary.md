# Slice 8: Phase 3 Regression and OpenMM Parity Gate

## Status: DONE

## Summary

Ran the Phase 3 aggregate regression gate after virtual sites, TIP4P-Ew, custom forces, GBSA, soft-core/lambda, and replica exchange landed. Fixed two Ruff formatting issues in `tests/test_custom_force.py` from the earlier custom-force slice.

## Files Changed

- `tests/test_custom_force.py`: Import formatting and line-wrap cleanup required by full ruff gate.

## Verification

- `uv run ruff check src tests scripts && uv run pytest`: ruff passed, `736 passed`.
- `git diff --check`: passed.

## Unresolved Risks

- None for Slice 8.
