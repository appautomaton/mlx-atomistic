# Slice 7: GPCRmd Topology/Parameter Import

## Result

- Status: completed
- Route: subagent implementer with spec and quality review
- Auto-continue: no
- Stop reason: Slice 7 is the planned checkpoint before virial/pressure diagnostics.

## Scope

Implemented the `gpcrmd-import` path that either writes strict MLX prepared artifacts or returns exact blockers. The slice stayed inside the GPCRmd prep/import surface and tests.

## Files Changed

- `src/mlx_atomistic/prep/gpcrmd.py`: added prepared-artifact import attempt flow, AMBER/CHARMM import routing, GPCRmd metadata enrichment, unsupported-term blocker mapping, and stale prepared-artifact cleanup on blocked imports.
- `src/mlx_atomistic/prep/topology_import.py`: extended topology imports with water/ion/lipid/receptor/ligand masks, inferred X-H constraints, term-count metadata, and fail-closed CHARMM unsupported-term detection.
- `tests/test_gpcrmd_registry.py`: added tiny GPCRmd AMBER export coverage, unsupported CHARMM blocker coverage, API JSON coverage, and stale-artifact cleanup regression coverage.
- `tests/test_mlx_prep.py`: added CHARMM/ParmEd-shaped unsupported-term rejection coverage.

## Review Loop

- Implementer: `DONE`
- Spec review 1: `CHANGES_REQUESTED`
  - Issue: CHARMM import could silently drop CMAP, Urey-Bradley, NBFIX/pair override, or related CHARMM terms while claiming compatibility.
  - Fix: CHARMM/ParmEd import now fails closed with `unsupported_terms:<term>` blockers before exporting when those containers are present.
- Spec review 2: `APPROVED`
- Quality review 1: `CHANGES_REQUESTED`
  - Issue: failed imports could leave stale `prepared_system.json`, `prepared_system.npz`, or `view.pdb` in a reused output directory.
  - Fix: blocked import attempts now remove only those generated prepared-artifact files and preserve unrelated files.
- Quality review 2: `APPROVED`

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py tests/test_mlx_prep.py`
  - Result: `44 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd or topology_import or mlx_atomistic.prep"`
  - Result: `44 passed, 236 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gpcrmd.py src/mlx_atomistic/prep/topology_import.py src/mlx_atomistic/prep/ tests/test_gpcrmd_registry.py tests/test_mlx_prep.py`
  - Result: `All checks passed!`
- Fixture API smoke with generated tiny GPCRmd AMBER cache:
  - `uv run mlx_atomistic.prep Python API gpcrmd-import ... --json`
  - Result: `"exported": true`, empty blockers, and `prepared_system.json`, `prepared_system.npz`, `view.pdb` were written.
- Selected target missing-cache API smoke:
  - `uv run mlx_atomistic.prep Python API gpcrmd-import --target gpcrmd-729-beta1-5f8u-cyanopindolol --cache <missing> --out <tmp> --json`
  - Result: `"exported": false` with exact missing cache/topology/model/parameters/protocol/box-vector blockers.

## Remaining Risks

- Full CHARMM export for selected-target CMAP, Urey-Bradley, NBFIX/pair override, and related terms is not completed in this slice. The importer now blocks exactly instead of dropping those terms.
- The selected GPCRmd target still needs a mounted/downloaded complete cache before real import can proceed.
