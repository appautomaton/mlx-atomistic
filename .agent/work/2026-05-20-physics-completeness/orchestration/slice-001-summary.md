# Slice 1: Virtual-Site Geometry and Topology Integration

## Status: DONE

## Summary

Implemented VirtualSite base classes (TwoParticleAverage, ThreeParticleAverage, OutOfPlane, LocalCoordinates) in new `virtual_sites.py`. Added `virtual_sites` and `virtual_site_types` fields to `Topology` with backward-compatible defaults. Moved `virtual_site` from `FAIL_CLOSED_TERMS` to `SUPPORTED_FORCE_TERMS` in `artifacts.py`; kept `tip4p`/`opc`/`advanced_water` blocked. Added virtual-site arrays to `PreparedSystem` validation. 21 tests pass covering geometry, topology integration, and edge cases.

## Files Changed

- `src/mlx_atomistic/virtual_sites.py` (new): VirtualSite base classes with position computation
- `src/mlx_atomistic/topology.py`: Added virtual_sites/virtual_site_types fields
- `src/mlx_atomistic/artifacts.py`: Moved virtual_site to SUPPORTED_FORCE_TERMS, updated HMR validation
- `src/mlx_atomistic/prep/schema.py`: Added virtual-site array fields and validation
- `src/mlx_atomistic/__init__.py`: Added VirtualSite exports
- `tests/test_virtual_sites.py` (new): 21 tests

## Verification

- `uv run pytest tests/test_virtual_sites.py tests/test_topology.py tests/test_production_artifacts.py -k "virtual_site or topology"`: 28 passed
- `uv run ruff check`: All checks passed
- Full regression: 694 passed

## Concerns

- None material. Pre-existing circular import note from Slice 2 does not affect test execution.