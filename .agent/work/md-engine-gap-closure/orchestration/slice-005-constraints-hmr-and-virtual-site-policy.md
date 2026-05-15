# Slice 5: Constraints, HMR, And Virtual-Site Policy

## Status

Complete at decision checkpoint.

## What Changed

- Added a focused constraint stability test for a constrained NVE run.
- Added fail-closed virtual-site policy for explicit virtual-site metadata and
  TIP4P/TIP5P/OPC-style water model metadata.
- Added HMR policy:
  - hidden heavy hydrogen masses fail closed;
  - declared `hydrogen_mass_repartitioning=represented_by_masses` is accepted;
  - HMR requested as a runtime force term fails closed.
- Updated GPCRmd policy blocker naming to
  `hmr_or_virtual_site_policy_required`.

## Evidence

Verification:

```sh
uv run pytest tests/test_protocols.py tests/test_production_artifacts.py tests/test_constraints.py tests/test_gpcrmd_registry.py -q
uv run ruff check src/mlx_atomistic/artifacts.py src/mlx_atomistic/prep/gpcrmd.py tests/test_production_artifacts.py tests/test_constraints.py tests/test_gpcrmd_registry.py
```

All verification commands passed.

## Decision

Virtual-site implementation is not required for this wave. The accepted policy
is exact rejection unless a future slice explicitly implements virtual sites.
HMR is represented only as static artifact masses with explicit metadata.
