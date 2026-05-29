# Slice 6 Implementer: Native CHARMM PSF/Parameter Import

## Status

DONE

## Files Changed

- `src/mlx_atomistic/prep/topology_import.py`
- `src/mlx_atomistic/prep/__init__.py`
- `tests/test_mlx_prep.py`
- `tests/fixtures/charmm/native-mini.psf`
- `tests/fixtures/charmm/native-mini.prm`
- `tests/fixtures/charmm/native-mini.pdb`

## Implementation Summary

- Added native `import_charmm_psf` exported from `mlx_atomistic.prep`.
- Parsed supported PSF atoms, charges, masses, bonds, angles, dihedrals, CMAP records, coordinates, and box metadata without ParmEd.
- Parsed CHARMM parameter BONDS, ANGLES with Urey-Bradley, DIHEDRALS, CMAP, NONBONDED, and NBFIX records into existing `PreparedSystem` arrays.
- Preserved the existing `import_charmm_with_parmed` compatibility path and added matching fail-closed overflow validation to its CMAP/NBFIX helpers.
- Added explicit blockers for unsupported PSF records, virtual-site/lone-pair records, unsupported water models, malformed or unsupported parameter records, malformed NBFIX, non-empty HBOND records, malformed numeric values, float32-overflowing values, missing supported-subset parameters, unsupported harmonic impropers, and distinct NBFIX 1-4 values.
- Added a minimal native CHARMM fixture and focused tests covering supported mappings, artifact round trip, blockers, and ParmEd compatibility helpers.

## Verification

- Focused native CHARMM parser tests passed in sandbox: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py -k "charmm_psf"` -> `24 passed, 80 deselected` with the known MLX Metal atexit warning.
- Focused ParmEd compatibility tests passed in sandbox: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py -k "charmm_parmed"` -> `4 passed, 102 deselected` with the known MLX Metal atexit warning.
- Required Slice 6 gate passed outside the sandbox: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py -k "charmm or cmap or urey or nbfix"` -> `69 passed, 152 deselected`.
- Targeted Ruff passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/__init__.py tests/test_mlx_prep.py tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py`.

## Concerns

- Native CHARMM support is intentionally bounded to the accepted subset and explicit blockers; broad CHARMM grammar coverage remains out of scope.
