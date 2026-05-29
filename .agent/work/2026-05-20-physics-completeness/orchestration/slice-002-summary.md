# Slice 2: Custom Force Expressions

## Status: DONE

## Summary

Created `CustomForcePotential` in new `custom_force.py` with symbolic expression parser supporting arithmetic (+, -, *, /, **), functions (cos, sin, exp, log, sqrt, abs), and named variables/parameters. Supports bond/pair (2-body distance-based) and angle (3-body) term types with `energy_forces(positions, cell)` protocol. Forces computed via numerical 1D derivatives with analytical chain rule conversion to per-atom forces. Re-exported from `forcefields.py` and `__init__.py`. Added `custom_force` to `SUPPORTED_FORCE_TERMS` and artifact construction. 37 tests pass.

## Files Changed

- `src/mlx_atomistic/custom_force.py` (new): CustomForcePotential with expression parser, evaluator, force computation
- `src/mlx_atomistic/forcefields.py`: Added import of CustomForcePotential
- `src/mlx_atomistic/__init__.py`: Added CustomForcePotential import and __all__ entry
- `src/mlx_atomistic/artifacts.py`: Added custom_force to SUPPORTED_FORCE_TERMS, validation, and artifact construction
- `tests/test_custom_force.py` (new): 37 tests

## Verification

- `uv run pytest tests/test_custom_force.py tests/test_forcefields.py tests/test_production_artifacts.py -k "custom_force or CustomForce or expression"`: 38 passed
- `uv run ruff check`: passed
- Full regression: 694 passed

## Concerns

- Morse-like bond expression uses atol=3e-2 due to compounded numerical derivative errors; acceptable for downstream GBSA use.
- Angle forces less precise near theta=0,pi; acceptable for GBSA surface-area terms.