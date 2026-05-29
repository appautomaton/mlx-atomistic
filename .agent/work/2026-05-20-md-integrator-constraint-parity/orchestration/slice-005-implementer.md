# Slice 5 Implementer

Status: DONE

## Summary

- Implemented deterministic hydrogen mass repartitioning as a prep-level mass transform.
- HMR increases selected hydrogen masses, subtracts the matching delta from bonded heavy atoms, and preserves total mass.
- HMR provenance records policy, original masses, transformed masses, selected hydrogens, heavy atoms, and total masses.
- Artifact/checkpoint metadata reports HMR state without treating HMR as a force term or claiming virtual-site support.

## Quality Fix

After quality review, the implementer:

- rejected non-floating prepared mass arrays and validated mass preservation after final dtype conversion;
- labeled explicit hydrogen-subset HMR as `explicit_hydrogen_indices` and persisted selected indices in policy metadata.

## Verification

- `uv run pytest tests/test_hmr.py tests/test_production_artifacts.py tests/test_checkpoint_restart.py` -> `60 passed`.
- Targeted Ruff checks for HMR/artifact/checkpoint tests passed.
