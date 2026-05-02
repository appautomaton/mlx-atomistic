# Slice 006 Quality Review

- Slice: Lazy Large-Topology Contract
- Status: approved after one requested fix

## Initial Finding

- important: `LennardJonesPotential` could still call `topology.nonbonded_pairs(None)` and materialize dense pairs for lazy large topologies outside the artifact `NonbondedPotential` path.

## Resolution

- `src/mlx_atomistic/md.py` now rejects lazy topologies without an explicit runtime pair provider before calling `topology.nonbonded_pairs(None)`.
- `tests/test_nonbonded_acceleration.py` covers the `backend="mlx_pairs"` lazy-topology regression and asserts `_nonbonded_pairs` remains `None`.

## Final Review

- Status: approved.
- Issues: none.
- Residual risk: Slice 7 must still provide scalable runtime neighbor pairs; Slice 6 intentionally only blocks dense fallback.

## Evidence

- `src/mlx_atomistic/md.py`: lazy/no-provider guard runs before `topology.nonbonded_pairs(pairs)`.
- `src/mlx_atomistic/topology.py`: defers `_nonbonded_pairs` when the nonbonded pair count exceeds the eager limit.
- `tests/test_nonbonded_acceleration.py`: covers the LJ lazy-topology regression.
- `tests/test_production_artifacts.py`: covers large artifact build deferring dense topology pairs through the runtime gate.
- Focused re-review command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py::test_lj_pairs_backend_requires_pair_provider_for_lazy_topology tests/test_production_artifacts.py::test_large_artifact_build_defers_dense_topology_pairs tests/test_topology.py::test_large_topology_defers_dense_nonbonded_pairs_until_requested`: 3 passed.
