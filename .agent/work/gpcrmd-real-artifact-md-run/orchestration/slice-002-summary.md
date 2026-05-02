# Slice 002 Summary

- Slice: Strict CHARMM Term Export
- Status: completed
- Execution route: subagent implementation with spec and quality review
- Stop reason: Slice 2 has `Auto-continue: no`; execution stops at the CHARMM-term checkpoint.

## What Changed

- CHARMM Urey-Bradley terms are exported into `urey_bradley_terms`, `urey_bradley_k`, and `urey_bradley_distance`.
- CHARMM CMAP terms are exported into `charmm_cmap_terms`, `charmm_cmap_grid_indices`, and `charmm_cmap_grids`.
- NBFIX pair overrides are detected and reported as `rejected_terms:nbfix_pair_overrides` when the current MLX runtime cannot represent them faithfully.
- Compatibility reports now include supported, required, rejected, and counted CHARMM terms on the blocked GPCRmd path.
- Prepared artifacts and in-memory prepared systems reject nonempty CHARMM arrays hidden by metadata-only edits.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_charmm_terms.py tests/test_production_artifacts.py tests/test_atomistic_prep.py tests/test_gpcrmd_registry.py -k "charmm or cmap or urey or nbfix or gpcrmd"`: 71 passed, 53 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/atomistic_prep/topology_import.py src/atomistic_prep/gpcrmd.py src/atomistic_prep/runner.py src/mlx_atomistic/artifacts.py tests/test_atomistic_prep.py tests/test_production_artifacts.py tests/test_gpcrmd_registry.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run atomistic-prep gpcrmd-import --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out /tmp/mlx-atomistic-gpcrmd-729-slice2 --json`: command completed, exported no artifact, and reported blocker `rejected_terms:nbfix_pair_overrides`.
- Real GPCRmd 729 report includes prepared force-term counts: `charmm_cmap_terms=317`, `urey_bradley_terms=49223`, and `nbfix_pair_overrides=37`.

## Reviews

- Implementer: completed with concern resolved.
- Spec review: approved after one correction.
- Quality review: approved.

## Next

The real GPCRmd 729 artifact still cannot export/run because NBFIX remains a fail-closed blocker. The next plan step should be corrected before continuing: add NBFIX-compatible runtime/export support, or explicitly choose a blocker-only large-system runtime gate.
