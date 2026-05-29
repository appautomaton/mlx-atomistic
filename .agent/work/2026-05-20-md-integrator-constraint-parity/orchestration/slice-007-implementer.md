# Slice 7 Implementer

Status: DONE

## Summary

- Added `MonteCarloBarostat` mode selection for isotropic, anisotropic, and membrane/semi-isotropic MC NPT proposals.
- Added matrix-cell coordinate rescaling for accepted barostat proposals and explicit mode metadata.
- Extended NPT reporter and protocol metadata to expose barostat mode, pressure intent, attempts, acceptances, and final cell/volume.
- Preserved fail-closed pressure-coupled behavior for unsupported virial terms.

## Review Fixes

After spec and quality review, the implementer/coordinator:

- changed the constraint-compatibility NPT test to pass a real `DistanceConstraints` object and assert final constrained geometry;
- updated NPT result assembly so accepted barostat moves replace the final public sampled frame, velocities, diagnostics, energies, and saved trajectory state with the post-barostat final state;
- threaded neighbor-managed pairs through old/proposed/final NPT barostat energy and diagnostic paths, including lazy-topology coverage.

## Verification

- `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"` -> `19 passed, 5 deselected in 0.94s`.
- `uv run ruff check src/mlx_atomistic/md.py src/mlx_atomistic/io.py src/mlx_atomistic/protocols.py tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py` -> passed.
