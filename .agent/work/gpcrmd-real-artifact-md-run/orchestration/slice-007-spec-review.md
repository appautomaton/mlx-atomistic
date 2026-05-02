# Slice 007 Spec Review

- Slice: Scalable Periodic Nonbonded Gate
- Status: approved

## Findings

- Slice 7 acceptance criteria are met.
- `minimize.py` and `protocols.py` changes are in scope because the production runner can perform minimization and equilibration before production integration, and those stages require the same compact pair route.
- The follow-up guard movement is in scope because direct lazy-topology calls must refuse dense/tiled fallback before backend selection.

## Evidence

- `src/mlx_atomistic/topology.py`: runtime pair arrays are filtered against topology exclusions without materializing full dense pairs when pairs are provided.
- `src/mlx_atomistic/forcefields.py`: explicit exception pairs are removed from runtime neighbor pairs, NBFIX mixed LJ parameters are applied to provided neighbor-list pairs, and lazy topology with no runtime pairs raises before backend selection.
- `src/mlx_atomistic/md.py`: lazy large topology refuses dense/tiled fallback when no `NeighborListManager` or runtime pairs are available, and reports backend, pair count, cutoff, skin, rebuild count, and estimated pair memory.
- `src/atomistic_prep/runner.py`: production runner creates a compact neighbor manager for lazy periodic cutoff nonbonded systems.
- `src/mlx_atomistic/minimize.py` and `src/mlx_atomistic/protocols.py`: minimization, equilibration, and production receive the neighbor manager.
- `tests/test_nonbonded_acceleration.py`: covers lazy topology filtering, exception removal, NBFIX on neighbor pairs, dense fallback refusal, and runtime reporting.
- Required command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_forcefields.py -k "neighbor or periodic or large or nbfix or exception"`: 24 passed, 27 deselected.
