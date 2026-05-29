# Slice 007 Implementer: Native GROMACS Top/Gro Import

## Status

DONE

## Files Changed

- `src/mlx_atomistic/prep/gromacs.py`
- `src/mlx_atomistic/prep/topology_import.py`
- `src/mlx_atomistic/prep/__init__.py`
- `src/mlx_atomistic/prep/gpcrmd.py`
- `tests/fixtures/gromacs/native-mini.top`
- `tests/fixtures/gromacs/native-mini.gro`
- `tests/test_gromacs_import.py`
- `tests/test_gpcrmd_registry.py`

## Implementation Summary

- Added a native supported-subset GROMACS `.top`/`.gro` parser with explicit section parsing.
- Covered `[defaults]`, `[atomtypes]`, `[moleculetype]`, `[atoms]`, `[bonds]`, `[angles]`, `[dihedrals]`, `[pairs]`, `[exclusions]`, `[system]`, `[molecules]`, coordinate import, molecule expansion, and periodic box import for the fixture.
- Mapped function-type-3 GROMACS RB records directly into `PreparedSystem` RB arrays using the existing `cos(phi - pi)` RB convention.
- Converted GROMACS nm coordinate, sigma, bond-length, and box units to angstrom, and converted harmonic bond force constants from kJ/mol/nm^2 to kJ/mol/angstrom^2.
- Added fail-closed blockers for preprocessing directives, unsupported sections, unsupported nonbonded combination rules, generated-pair cases outside the declared subset, virtual sites, B-state atom records, unsupported function types, malformed records, and invalid boxes.
- Exported `import_gromacs_top_gro` from `mlx_atomistic.prep`.
- Updated GPCRmd `.top` routing to inspect ambiguous `.top` content and distinguish GROMACS topology files from AMBER `%FLAG` topology files.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gromacs_import.py -q` -> passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gromacs.py src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/__init__.py src/mlx_atomistic/prep/gpcrmd.py tests/test_gromacs_import.py tests/test_gpcrmd_registry.py` -> passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gromacs_import.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py -k "gromacs or rb"` -> `12 passed, 123 deselected` after the `gen-pairs = no` correction.

## Concerns

- The parser intentionally rejects GROMACS preprocessing, includes, generated 1-4 pair inference without explicit `[ pairs ]`, `[ pairs ]` when `gen-pairs = no`, virtual sites, constraints/SETTLE, unsupported function types, explicit pair parameter rows, `[pairtypes]`, and combination rules other than 2. Those remain explicit blockers instead of silent partial imports.
- Pytest emits the existing MLX Metal atexit warning in this sandbox after artifact-related imports, but the tests pass.
