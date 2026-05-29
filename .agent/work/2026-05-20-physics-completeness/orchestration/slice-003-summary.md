# Slice 3: Virtual-Site Force Redistribution and Constraint Integration

## Status: DONE

## Summary

Implemented virtual-site MD integration through `SimulationConfig.virtual_sites`. Force evaluation reconstructs virtual-site positions, evaluates on real plus virtual atoms, then redistributes virtual-site forces back to real atoms. `simulate_nvt` and `simulate_npt` public function signatures remain unchanged. Constraints, kinetics, and stored trajectory state remain real-atom only.

## Files Changed

- `src/mlx_atomistic/virtual_sites.py`: Added lint-clean strict zip in force redistribution helper.
- `src/mlx_atomistic/md.py`: Added virtual-site-aware evaluator wrappers, reconstruction before force evaluation, redistributed real-atom forces, and virtual-site-aware dense nonbonded reports.
- `tests/test_virtual_sites.py`: Added redistribution, timestep reconstruction, diagnostic consistency, and signature-preservation coverage.

## Verification

- `uv run pytest tests/test_virtual_sites.py tests/test_md.py tests/test_constraints.py tests/test_nvt.py -k "virtual_site or redistribute or vsite" && uv run ruff check src/mlx_atomistic/virtual_sites.py src/mlx_atomistic/md.py`: `25 passed, 18 deselected`; ruff passed.
- `uv run pytest`: `698 passed`.
- `git diff --check`: passed.

## Reviewer Verdicts

- Spec review: APPROVED.
- Quality review: CHANGES_REQUESTED, then APPROVED after fixes.

## Resolved Issues

- Restored `simulate_nvt` thermostat annotation to `LangevinThermostat | None`.
- Fixed dense `nonbonded_report["pair_count"]` to use virtual-site evaluation positions consistently.

## Unresolved Risks

- None for Slice 3.
