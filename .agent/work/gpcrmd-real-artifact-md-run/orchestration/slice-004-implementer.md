# Slice 004 Implementer

- Slice: NBFIX Runtime Semantics
- Status: completed with coordinator repair
- Route: subagent implementation attempted; coordinator completed one narrow fix after subagent timeout

## Implementation Outcome

- Added NBFIX runtime inputs to `NonbondedPotential`: legacy explicit atom pairs and compact atom-type pairs.
- Substituted NBFIX sigma/epsilon in LJ pair parameter selection, keeping Coulomb unchanged.
- Kept explicit nonbonded exceptions authoritative by removing exception pairs before regular LJ/Coulomb evaluation and adding exception terms separately.
- Routed artifacts with NBFIX into the normal `nonbonded` force term instead of adding a second NBFIX force term.
- Allowed NBFIX with PME/Ewald Coulomb paths because the override affects LJ parameters only.
- Added focused tests for NBFIX type pairs, explicit pairs, exclusions, exceptions, unknown type rejection, artifact construction, and PME with NBFIX.

## Files Changed

- `src/mlx_atomistic/forcefields.py`: NBFIX parameters and LJ substitution in `NonbondedPotential`.
- `src/mlx_atomistic/artifacts.py`: artifact-to-runtime wiring for compact and legacy NBFIX arrays.
- `tests/test_forcefields.py`: runtime NBFIX behavior coverage.
- `tests/test_production_artifacts.py`: artifact construction and exception-precedence coverage.

## Subagent Notes

- First implementer subagent timed out after partial edits and was shut down.
- Second implementer subagent also timed out after partial follow-up and was shut down.
- Coordinator inspected the partial patch, ran focused tests, and fixed one artifact bug: `nbfix_type_epsilon` was loaded after type-pair filtering instead of before it.

## Verification

- Sandbox test run failed with `No Metal device available`; verification was rerun outside the sandbox with approved `uv run pytest`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_forcefields.py -k "nbfix"`: 6 passed, 21 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_charmm_terms.py tests/test_forcefields.py tests/test_production_artifacts.py -k "nbfix or nonbonded or pme or exception"`: 64 passed, 32 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/forcefields.py src/mlx_atomistic/nonbonded.py src/mlx_atomistic/charmm_terms.py src/mlx_atomistic/artifacts.py tests/test_charmm_terms.py tests/test_forcefields.py tests/test_production_artifacts.py`: passed.
- GPCRmd load-only check loaded 92,001 atoms and 37 compact NBFIX type pairs from `/tmp/mlx-atomistic-gpcrmd-729-slice3`.

## Residual Concern

Building the full GPCRmd runtime system no longer rejects NBFIX, but it fails when `Topology` tries to materialize dense nonbonded pairs for 92,001 atoms: `ValueError: Shape dimension falls outside supported int range.` This is the planned Slice 6 large-system runtime gate.
