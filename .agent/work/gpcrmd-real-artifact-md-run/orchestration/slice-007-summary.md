# Slice 007 Summary

- Slice: Scalable Periodic Nonbonded Gate
- Status: completed
- Execution route: subagent route with one quality-review follow-up fix
- Stop reason: Slice 7 has `Auto-continue: no`; execution stops at the scalable-nonbonded checkpoint.

## What Changed

- Large lazy-topology nonbonded paths now require compact runtime neighbor pairs and refuse dense/tiled all-pairs fallback.
- Compact neighbor pairs are filtered through topology exclusions and explicit nonbonded exceptions without dense matrices.
- NBFIX type-pair LJ substitution works on neighbor-list pairs.
- Nonbonded runtime results report backend, pair count, cutoff, skin, rebuild count, and estimated pair memory.
- The production runner and protocol path pass a neighbor manager through minimization, equilibration, and production NVT.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_forcefields.py -k "neighbor or periodic or large or nbfix or exception"`: 24 passed, 27 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/nonbonded.py src/mlx_atomistic/neighbors.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/md.py src/mlx_atomistic/minimize.py src/mlx_atomistic/protocols.py src/atomistic_prep/runner.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py`: passed.
- Follow-up regression verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py -q`: 17 passed.
- Protocol verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_protocols.py -q`: 6 passed.

## Reviews

- Implementer: completed with concern about extra protocol/minimization files.
- Spec review: approved; extra protocol/minimization files are in scope for the production path.
- Quality review: approved after moving lazy-topology guards before dense/tiled backend selection.

## Next

Slice 8 should encode PME readiness for GPCRmd. The large-system real-space nonbonded route now has compact neighbor pairs and fail-closed dense fallback behavior; electrostatics remains the next gate.
