# Slice 003 Summary

- Slice: NBFIX Export Schema
- Status: completed
- Execution route: subagent implementation with spec and quality review
- Stop reason: Slice 3 has `Auto-continue: no`; execution stops at the NBFIX export checkpoint.

## What Changed

- Prepared artifacts now carry compact NBFIX type-pair override arrays instead of metadata-only counts.
- Each NBFIX override records atom-type identifiers plus converted sigma and epsilon values.
- GPCRmd 729 import now reports 37 concrete NBFIX overrides in `term_details.nbfix_pair_overrides`.
- Legacy explicit atom-pair NBFIX arrays remain valid and backward-compatible.
- Compact NBFIX arrays cannot be hidden by deleting metadata declarations.
- Malformed, missing, nonfinite, nonpositive, conflicting, or distinct 1-4 NBFIX entries fail closed during import.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py -k "nbfix or undeclared"`: 3 passed, 35 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py -k "nbfix or charmm or gpcrmd"`: 43 passed, 53 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/schema.py src/mlx_atomistic/prep/io.py src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/gpcrmd.py tests/test_mlx_prep.py tests/test_gpcrmd_registry.py tests/test_production_artifacts.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/artifacts.py tests/test_production_artifacts.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run mlx_atomistic.prep Python API gpcrmd-import --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out /tmp/mlx-atomistic-gpcrmd-729-slice3 --json`: exported a real prepared artifact with 92,001 atoms and 37 compact NBFIX type pairs.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python - <<'PY' ... load_prepared_mlx_artifact('/tmp/mlx-atomistic-gpcrmd-729-slice3', require_production=True) ... PY`: loaded the artifact and reported `artifact_loaded 92001 (37, 2)`.

## Reviews

- Implementer: completed.
- Spec review: approved.
- Quality review: approved after one correction.

## Next

Slice 4 must implement runtime NBFIX semantics so compact type-pair LJ overrides are applied without double-counting base nonbonded terms and without affecting exclusions or explicit exceptions.
