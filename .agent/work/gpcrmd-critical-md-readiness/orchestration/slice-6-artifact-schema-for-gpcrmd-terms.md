# Slice 6 Orchestration: Artifact Schema For GPCRmd Terms

## Scope

- Change: `gpcrmd-critical-md-readiness`
- Slice: `slice_6_artifact_schema_for_gpcrmd_terms`
- Route: subagent implementation with spec review and focused re-review
- Files in scope:
  - `src/mlx_atomistic/artifacts.py`
  - `src/atomistic_prep/schema.py`
  - `src/atomistic_prep/io.py`
  - `tests/test_production_artifacts.py`

## Implementation

- Implementer `019de585-912c-7c82-be5e-9cdbc887a0e4` added v2 prepared artifact fields while accepting v1 artifacts.
- New optional artifact fields cover PME config arrays, CHARMM CMAP, Urey-Bradley, NBFIX, lipid masks, and protocol metadata.
- `load_prepared_mlx_artifact` validates requested PME, CHARMM, lipid, water, ion, receptor, ligand, exception, and constraint arrays before build.
- `build_mlx_system_from_artifact` now wires PME `NonbondedPotential`, CMAP, Urey-Bradley, and NBFIX where it can do so without silently dropping required behavior.
- NBFIX/force-switch combinations that cannot yet be represented faithfully with topology exclusions, exceptions, or PME remain fail-closed.

## Reviews

- First spec review `019de58f-2133-7403-9084-f3b4eea4afdb`: `CHANGES_REQUESTED`.
  - Invalid PME scalar settings could load and only fail later during build.
- Fix implementer `019de585-912c-7c82-be5e-9cdbc887a0e4`: `DONE`.
  - Added load-time and save-time PME scalar validation.
- First re-review `019de594-3ce2-7853-9b84-d4632683904f`: `CHANGES_REQUESTED`.
  - Fractional PME mesh dimensions were still truncated by integer casts.
- Final fix implementer `019de585-912c-7c82-be5e-9cdbc887a0e4`: `DONE`.
  - Validates PME mesh dimensions before lossy casts.
- Final re-review `019de598-7f36-7a51-9f1f-1c112bec2040`: `APPROVED`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_atomistic_prep.py`
  - Result: `55 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "artifacts or schema or pme or charmm"`
  - Result: `83 passed, 192 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/artifacts.py src/atomistic_prep/schema.py src/atomistic_prep/io.py tests/test_production_artifacts.py`
  - Result: `All checks passed`

## Notes

- GPCRmd importer implementation remains Slice 7.
- Runtime protocol and notebook consumption remain later slices.
