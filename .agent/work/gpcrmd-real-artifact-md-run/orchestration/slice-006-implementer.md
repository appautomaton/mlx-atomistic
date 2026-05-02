# Slice 006 Implementer

- Slice: Lazy Large-Topology Contract
- Status: completed
- Route: subagent implementation with one quality-review follow-up fix

## Implementation Outcome

- Added lazy/eager topology pair policy so large systems defer `_nonbonded_pairs` until explicitly requested.
- Preserved compact exclusions, explicit nonbonded exception pairs, and 1-4 pairs as arrays and sets.
- Added `nonbonded_build_report` with pair policy, atom count, cutoff, exclusions, exceptions, 1-4 count, and nonbonded pair count.
- Passed artifact nonbonded cutoff into topology report metadata.
- Added guards so artifact nonbonded and `LennardJonesPotential` pair-list paths do not implicitly materialize dense pairs for lazy topologies.

## Files Changed

- `src/mlx_atomistic/topology.py`: lazy topology pair policy, compact metadata, and explicit materialization path.
- `src/mlx_atomistic/artifacts.py`: artifact cutoff propagation into topology metadata.
- `src/mlx_atomistic/forcefields.py`: lazy topology runtime guard for artifact nonbonded paths.
- `src/mlx_atomistic/md.py`: lazy topology runtime guard for `LennardJonesPotential` pair-list paths.
- `tests/test_topology.py`: lazy topology contract coverage.
- `tests/test_production_artifacts.py`: large artifact build coverage without dense pair materialization.
- `tests/test_nonbonded_acceleration.py`: regression coverage for `LennardJonesPotential` with lazy topology and no pair provider.

## Implementer Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_topology.py tests/test_production_artifacts.py -k "large or topology or nonbonded_pairs or exception or nbfix"`: 9 passed, 35 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "large or topology or nonbonded_pairs or exception or nbfix"`: 12 passed, 15 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_forcefields.py -k "large or topology or nonbonded_pairs or exception or nbfix"`: 16 passed, 50 deselected.
- Follow-up fix verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py tests/test_topology.py tests/test_production_artifacts.py::test_large_artifact_build_defers_dense_topology_pairs`: 18 passed.
- Coordinator style verification after formatting fixes: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/topology.py src/mlx_atomistic/artifacts.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/md.py tests/test_topology.py tests/test_production_artifacts.py tests/test_nonbonded_acceleration.py`: passed.

## Residual Concern

Slice 7 still needs the compact periodic neighbor-pair provider. Lazy topology now blocks before dense runtime fallback instead of providing scalable nonbonded execution.
