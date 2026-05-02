# Slice 004 Summary

- Slice: NBFIX Runtime Semantics
- Status: completed
- Execution route: subagent route with coordinator repair after implementer timeouts
- Stop reason: Slice 4 has `Auto-continue: no`; execution stops at the runtime-semantics checkpoint.

## What Changed

- `NonbondedPotential` now accepts compact atom-type NBFIX and legacy explicit-pair NBFIX inputs.
- LJ pair parameters use NBFIX sigma/epsilon for matching pairs.
- Coulomb remains unchanged by NBFIX.
- Topology exclusions and explicit nonbonded exceptions remain authoritative.
- Artifact construction passes NBFIX arrays into the normal `nonbonded` term and no longer appends a separate `nbfix_pair_overrides` term.
- PME/Ewald Coulomb paths can coexist with NBFIX because LJ and Coulomb are evaluated separately.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "nbfix"`: 6 passed, 21 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_charmm_terms.py tests/test_forcefields.py tests/test_production_artifacts.py -k "nbfix or nonbonded or pme or exception"`: 64 passed, 32 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/forcefields.py src/mlx_atomistic/nonbonded.py src/mlx_atomistic/charmm_terms.py src/mlx_atomistic/artifacts.py tests/test_charmm_terms.py tests/test_forcefields.py tests/test_production_artifacts.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python - <<'PY' ... load_prepared_mlx_artifact('/tmp/mlx-atomistic-gpcrmd-729-slice3', require_production=True) ... PY`: loaded 92,001 atoms and 37 NBFIX type pairs.

## Reviews

- Implementer: subagent implementation attempted twice, but both workers timed out.
- Spec review: approved.
- Quality review: approved.

## Next

Slice 5 should run the real GPCRmd prepared import/export check. The next known runtime blocker is dense nonbonded topology materialization for the 92k-atom system, which belongs to Slice 6.
