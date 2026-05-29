# Slice 7 Spec Review

Status: APPROVED

## Summary

- Slice 7 meets the accepted barostat contract after the constraint-compatibility test fix.

## Issues

- Initial review requested a fix because the named NPT constraint-compatibility test did not pass `constraints=` into `simulate_npt`.
- The fix added a real `DistanceConstraints` object and final geometry/error assertions.

## Evidence

- `tests/test_npt.py` exercises isotropic, anisotropic, membrane/semi-isotropic, constraint-compatible, and fail-closed NPT paths.
- `src/mlx_atomistic/md.py` exposes MC barostat mode policy, attempts, acceptances, final cell, volume history, and metadata.
- `src/mlx_atomistic/protocols.py` accepts bounded MC NPT and membrane MC NPT proof-mode requests.
- Verification observed: `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"` -> `19 passed, 5 deselected`.
- Targeted Ruff checks passed.

## Residual Risk

- MC pressure behavior remains bounded-fixture/proof-level pending Slice 8 parity coverage.
