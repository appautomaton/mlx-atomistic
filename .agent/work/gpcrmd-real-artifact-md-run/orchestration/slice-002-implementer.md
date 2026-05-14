# Slice 002 Implementer

- Slice: Strict CHARMM Term Export
- Status: completed with concern resolved
- Route: subagent implementation

## Implementation Outcome

- Exported CHARMM Urey-Bradley arrays from ParmEd structures into prepared-system artifacts.
- Exported CHARMM CMAP term indices, grid indices, and grids into prepared-system artifacts.
- Detected NBFIX atom-type pair overrides and reported them as a fail-closed required/rejected term because the current runtime path cannot combine NBFIX faithfully with topology exclusions, exceptions, or PME.
- Added metadata and array validation so CHARMM arrays cannot be hidden by editing compatibility metadata only.
- Updated GPCRmd blocked import reports to include prepared force-term fields when parsing succeeds but fail-closed blockers remain.

## Files Changed

- `src/mlx_atomistic/prep/topology_import.py`: CHARMM Urey-Bradley and CMAP export; NBFIX detection and rejection metadata; GPCRmd ligand residue `P32` masking.
- `src/mlx_atomistic/prep/gpcrmd.py`: prepared force-term report propagation into blocked GPCRmd import reports.
- `src/mlx_atomistic/prep/runner.py`: in-memory prepared-system compatibility validation now passes arrays.
- `src/mlx_atomistic/artifacts.py`: array-level metadata-hiding validation for CHARMM terms.
- `tests/test_mlx_prep.py`: Urey-Bradley, CMAP, NBFIX rejection, and `P32` ligand masking coverage.
- `tests/test_production_artifacts.py`: hidden CHARMM array rejection for disk and in-memory paths.
- `tests/test_gpcrmd_registry.py`: blocked prepared GPCRmd import report coverage.

## Implementer Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py::test_charmm_arrays_cannot_be_hidden_by_metadata_only tests/test_production_artifacts.py::test_in_memory_runner_rejects_hidden_charmm_arrays`: 2 passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/runner.py tests/test_production_artifacts.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py::test_gpcrmd_blocked_import_surfaces_prepared_force_term_report tests/test_gpcrmd_registry.py::test_gpcrmd_charmm_import_receives_psf_mass_prelude_and_protocol_box tests/test_gpcrmd_registry.py::test_gpcrmd_import_reports_unsupported_charmm_terms_without_export`: 3 passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gpcrmd.py tests/test_gpcrmd_registry.py`: passed.
- Real GPCRmd 729 import probe: blocker remains `rejected_terms:nbfix_pair_overrides`; report includes `charmm_cmap_terms=317`, `urey_bradley_terms=49223`, and `nbfix_pair_overrides=37`.

## Residual Concern

NBFIX remains a fail-closed blocker for the real GPCRmd 729 artifact. This is allowed by Slice 2 acceptance, but it prevents a real GPCRmd MLX run until the plan adds NBFIX-compatible runtime/export support or explicitly chooses a blocker-only path.
