# Slice 006 Summary

- Slice: Lazy Large-Topology Contract
- Status: completed
- Execution route: subagent route with one quality-review follow-up fix
- Stop reason: Slice 6 has `Auto-continue: no`; execution stops at the lazy-topology checkpoint.

## What Changed

- Large `Topology` instances now use a lazy nonbonded pair policy when dense all-pair materialization would exceed the eager limit.
- Compact exclusions, explicit nonbonded exception pairs, and 1-4 pairs remain available without dense pair arrays.
- `Topology.nonbonded_build_report` reports pair policy, atom count, cutoff, exclusions, exceptions, 1-4 count, and nonbonded pair count.
- Artifact construction passes nonbonded cutoff metadata into topology construction.
- Artifact nonbonded paths and `LennardJonesPotential` now fail closed for lazy topologies without explicit runtime pairs instead of materializing dense pairs implicitly.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_production_artifacts.py tests/test_forcefields.py -k "large or topology or nonbonded_pairs or exception or nbfix"`: 16 passed, 50 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/topology.py src/mlx_atomistic/artifacts.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/md.py tests/test_topology.py tests/test_production_artifacts.py tests/test_nonbonded_acceleration.py`: passed.
- Subagent follow-up verification: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py tests/test_topology.py tests/test_production_artifacts.py::test_large_artifact_build_defers_dense_topology_pairs`: 18 passed.
- Spec reviewer observed real GPCRmd 729 build with `atom_count=92001`, `pair_policy=lazy`, and `pairs_cached=False`, followed by the lazy runtime nonbonded pair-provider gate.

## Reviews

- Implementer: completed.
- Spec review: approved, including the targeted `md.py` guard after follow-up.
- Quality review: approved after one requested fix for `LennardJonesPotential` implicit materialization.

## Next

Slice 7 should add the scalable periodic nonbonded gate and compact runtime pair-provider route. The current blocker is no longer topology construction; it is the missing compact runtime nonbonded pair provider for the 92k-atom artifact.
