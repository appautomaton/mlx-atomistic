# Slice 6: Soft-Core Potentials and Lambda Scaling

## Status: DONE

## Summary

Implemented cutoff-path soft-core LJ/Coulomb lambda scaling, explicit `energy_forces_dlambda()` derivatives, `SoftCoreNonbondedPotential`, and artifact support for `soft_core_lj` / `lambda_scaled_nonbonded`. Non-cutoff PME/Ewald alchemical derivatives fail closed with explicit `ValueError`.

## Files Changed

- `src/mlx_atomistic/forcefields.py`: Lambda fields, soft-core pair math, opt-in derivative surface, wrapper.
- `src/mlx_atomistic/artifacts.py`: Soft-core/lambda supported terms, metadata parsing, artifact construction.
- `tests/test_soft_core.py`: Finite-overlap, endpoint, derivative, wrapper, artifact, metadata, fail-closed, and default diagnostics regression tests.

## Verification

- `uv run pytest tests/test_soft_core.py tests/test_forcefields.py tests/test_production_artifacts.py -k "soft_core or lambda or alchemical" && uv run ruff check src/mlx_atomistic/nonbonded.py src/mlx_atomistic/forcefields.py`: `8 passed, 108 deselected`; ruff passed.
- `uv run pytest`: `720 passed`.
- `git diff --check`: passed.

## Reviewer Verdicts

- Spec review: APPROVED; cutoff-only/fail-closed PME/Ewald is compatible with Slice 6 scope.
- Quality review: CHANGES_REQUESTED for term-scoped metadata and non-cutoff derivative `KeyError`, then APPROVED after fixes.

## Regression Fix

Full regression initially failed because default nonbonded component diagnostics included `dU_dlambda_*` keys. The implementation now keeps derivative keys off default diagnostics and exposes derivatives through `energy_forces_dlambda()`.

## Unresolved Risks

- PME/Ewald alchemical soft-core remains unsupported by design and fails closed.
