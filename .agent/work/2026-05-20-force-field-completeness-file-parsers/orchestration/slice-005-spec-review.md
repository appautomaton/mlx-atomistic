# Slice 5 Spec Review

## Status

APPROVED

## Summary

- Slice 5 matches the requested AMBER import completion and Phase 2 metadata alignment.
- The implementation preserves the requested AMBER topology and parameter surfaces, derives topology-specific 1-4 scaling when present, fails closed with explicit `unsupported_terms:*` blockers, and keeps the AMBER parity artifact path including PME order metadata.

## Issues

- none

## Evidence

- `.agent/work/2026-05-20-force-field-completeness-file-parsers/PLAN.md:105` defines the Slice 5 objective, criteria, verification command, and produced IDs.
- `src/mlx_atomistic/prep/topology_import.py` implements native AMBER import, unsupported-record checks, topology/restart validation, preserved arrays, Phase 2 metadata, fail-closed blockers, topology-derived 1-4 scaling, exclusion parsing, and exception construction.
- `scripts/openmm_mlx_parity.py` catches AMBER `TopologyImportError` into blocked parity reports, emits Phase 2 evidence IDs, and applies PME artifact metadata while clearing stale `cell_matrix`.
- `tests/test_mlx_prep.py` covers accepted fixture counts, topology-derived 1-4 scaling, fixed-width numeric records, residue validation, periodic/restart handling, unsupported records, malformed records, LJ/exclusion edge cases, and explicit empty exclusions.
- `tests/test_openmm_mlx_parity.py` covers AMBER production artifact parity, PME assignment-order metadata, stale matrix clearing, and blocked unsupported/malformed reports.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "amber"` passed outside the sandbox: `54 passed, 46 deselected`.
- Targeted Ruff passed for the touched Slice 5 files.
