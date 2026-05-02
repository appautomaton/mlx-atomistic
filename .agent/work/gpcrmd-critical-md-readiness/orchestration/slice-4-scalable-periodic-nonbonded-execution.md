# Slice 4 Orchestration: Scalable Periodic Nonbonded Execution

## Scope

- Change: `gpcrmd-critical-md-readiness`
- Slice: `slice_4_scalable_periodic_nonbonded_execution`
- Route: subagent implementation with spec and quality review
- Files in scope:
  - `src/mlx_atomistic/cell_list.py`
  - `src/mlx_atomistic/neighbors.py`
  - `src/mlx_atomistic/benchmarks/md_acceleration.py`
  - `tests/test_neighbors.py`
  - `tests/test_nonbonded_acceleration.py`

## Implementation

- Implementer `019de56c-cdb7-7822-b2f4-b386aa7f2c82` added deterministic periodic cell-list and pair-list construction.
- `build_neighbor_list` now uses the cell-list helper and exposes backend and memory estimates through `NeighborList`.
- `md_acceleration` benchmark rows now report pair count, rebuild count, pair-memory estimate, dense-memory estimate, and backend.
- Tests cover deterministic periodic pair counts, compact storage on a larger periodic fixture, rebuild policy, and small-fixture dense-vs-pair nonbonded agreement.

## Reviews

- Spec review `019de572-0c36-78f0-a9b9-9e3c137a1783`: `APPROVED`.
- First quality review `019de573-8968-75f3-8fed-1e29038cbe49`: `CHANGES_REQUESTED`.
  - Required fail-closed validation in `NeighborListManager.needs_rebuild()` so non-finite positions cannot reuse a stale neighbor list.
- Fix implementer `019de56c-cdb7-7822-b2f4-b386aa7f2c82`: `DONE`.
  - Added shape and finite-position validation before interval skipping or displacement math.
  - Added a regression test for `manager.update()` after an initial valid build.
- Quality re-review `019de576-b369-7171-af2d-16ef207dc43b`: `APPROVED`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_neighbors.py tests/test_nonbonded_acceleration.py`
  - Result: `18 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "neighbor or nonbonded_acceleration"`
  - Result: `21 passed, 223 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/cell_list.py src/mlx_atomistic/neighbors.py src/mlx_atomistic/benchmarks/md_acceleration.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py`
  - Result: `All checks passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.md_acceleration --sizes 16 --backends python_neighbor,mlx_pairs --evaluations 1 --json`
  - Result: emitted rows containing `backend`, `pairs`, `rebuild_count`, `estimated_pair_bytes`, and `estimated_dense_bytes`.

## Concerns

- The pair builder is still Python/NumPy-side construction; it removes dense all-pairs memory but is not a Metal pair-list builder.
- PME production wiring remains Slice 5.
