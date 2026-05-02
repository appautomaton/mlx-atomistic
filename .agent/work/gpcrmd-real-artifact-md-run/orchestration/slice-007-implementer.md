# Slice 007 Implementer

- Slice: Scalable Periodic Nonbonded Gate
- Status: completed with one quality-review follow-up fix
- Route: subagent implementation

## Implementation Outcome

- Added fail-closed guards so lazy large-topology nonbonded paths cannot use dense or tiled all-pairs fallback without runtime pairs.
- Wired compact periodic neighbor pairs through the production runner and NVT protocol path, including minimization and equilibration stages used before production integration.
- Added nonbonded runtime reporting with backend, neighbor pair count, cutoff, skin, rebuild count, and estimated pair memory.
- Added regression tests for topology filtering, explicit exception removal, NBFIX substitution on neighbor-list pairs, compact backend reporting, and direct dense-fallback refusal.

## Files Changed

- `src/mlx_atomistic/md.py`: compact-backend guard, runtime nonbonded report, and lazy-topology guard before LJ backend selection.
- `src/mlx_atomistic/forcefields.py`: lazy-topology guard before `NonbondedPotential` backend selection and existing runtime-pair filtering/NBFIX path coverage.
- `src/mlx_atomistic/minimize.py`: minimization force evaluations can use neighbor-manager pairs.
- `src/mlx_atomistic/protocols.py`: minimization, equilibration, and production NVT pass the neighbor manager.
- `src/atomistic_prep/runner.py`: creates compact neighbor manager for lazy periodic cutoff nonbonded systems and records runtime metadata.
- `tests/test_nonbonded_acceleration.py`: focused Slice 7 regression coverage.

## Implementer Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_forcefields.py -k "neighbor or periodic or large or nbfix or exception"`: 24 passed, 25 deselected before the follow-up fix; 24 passed, 27 deselected after coordinator cleanup.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import atomistic_prep.runner; import mlx_atomistic.protocols; import mlx_atomistic.minimize'`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_protocols.py -q`: 6 passed.
- Follow-up fix verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py -q`: 17 passed.
- Coordinator style verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/nonbonded.py src/mlx_atomistic/neighbors.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/md.py src/mlx_atomistic/minimize.py src/mlx_atomistic/protocols.py src/atomistic_prep/runner.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py`: passed.

## Residual Concern

Slice 8 still needs the electrostatics readiness decision. Slice 7 only provides the compact real-space nonbonded route and fail-closed dense fallback behavior.
