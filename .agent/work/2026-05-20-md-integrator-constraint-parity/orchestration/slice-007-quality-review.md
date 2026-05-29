# Slice 7 Code Quality Review

Status: APPROVED

## Summary

- Code quality approved after two focused fixes to the public NPT output state and neighbor-managed barostat paths.

## Issues

- Accepted barostat moves initially updated `final_state`/`final_cell` while delegated public sampled trajectory arrays still reflected the pre-barostat NVT result.
- Post-barostat energy and diagnostic rebuilds initially ignored `neighbor_manager`, which broke lazy-topology paths requiring runtime nonbonded pairs.

## Fix Evidence

- `src/mlx_atomistic/md.py` now rebuilds the delegated NVT result final sampled frame, velocity frame, diagnostics, energies, pressure values, constraint error, and nonbonded report from the post-barostat final state and cell.
- `src/mlx_atomistic/md.py` now uses manager-backed pairs for old/proposed/final NPT barostat energy and diagnostic paths and updates the manager to the accepted final cell.
- `tests/test_npt.py` asserts accepted NPT final sampled positions/velocities match `final_state`, saved trajectory output matches the final cell/state, and lazy-topology NPT uses neighbor-manager pair/rebuild counts.
- `uv run pytest tests/test_npt.py tests/test_virial_pressure.py tests/test_runtime_reporters.py -k "barostat or npt or pressure"` -> `19 passed, 5 deselected in 0.94s`.
- Targeted Ruff checks passed.

## Residual Risk

- Remaining risk is limited to proof-level MC NPT physics validation; broad OpenMM parity is intentionally deferred to Slice 8.
