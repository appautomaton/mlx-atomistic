# Slice 006 Spec Review

- Slice: Lazy Large-Topology Contract
- Status: approved

## Findings

- Lazy topology behavior is implemented for synthetic large systems and the real GPCRmd 729 artifact.
- Compact exclusions, exception pairs, and 1-4 pairs remain available without dense `_nonbonded_pairs`.
- Small-system `nonbonded_pairs()` behavior is preserved.
- The `md.py` guard and nonbonded acceleration regression test are within Slice 6 scope because they prevent another topology consumer from materializing dense lazy pairs.

## Evidence

- `src/mlx_atomistic/topology.py`: computes pair count, chooses eager vs lazy policy, and leaves `_nonbonded_pairs` as `None` when above the eager limit.
- `src/mlx_atomistic/topology.py`: exposes `nonbonded_build_report` with pair policy, atom count, cutoff, exclusions, exceptions, 1-4 count, and nonbonded pair count.
- `src/mlx_atomistic/topology.py`: materializes dense pairs only when `nonbonded_pairs()` is explicitly called.
- `src/mlx_atomistic/md.py`: raises before lazy topology can materialize full dense pairs without explicit runtime pairs.
- `tests/test_topology.py`: verifies lazy metadata, compact arrays, and explicit materialization.
- `tests/test_production_artifacts.py`: verifies artifact build defers dense topology pairs and runtime evaluation reaches the lazy nonbonded gate.
- `tests/test_nonbonded_acceleration.py`: verifies `backend="mlx_pairs"` requires an explicit pair provider for lazy topology and keeps `_nonbonded_pairs` unset.
- Required command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_forcefields.py -k "large or topology or nonbonded_pairs or exception or nbfix"`: 16 passed, 50 deselected.
