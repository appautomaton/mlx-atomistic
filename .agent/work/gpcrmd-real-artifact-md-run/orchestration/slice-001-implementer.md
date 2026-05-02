# Slice 001 Implementer

- Slice: GPCRmd Cache And Protocol Normalization
- Status: DONE
- Route: worker subagent

## Summary

- Added deterministic GPCRmd role-path reporting.
- Added PSF-derived CHARMM `MASS` prelude generation and GPCRmd import wiring so ParmEd receives missing atom types such as `CT3`.
- Parsed GPCRmd protocol `input.xsc` files and propagated box vectors/lengths into import details, prepared metadata, and `cell_lengths`.
- Confirmed real GPCRmd 729 import now reaches strict CHARMM blockers: `charmm_cmap_terms`, `nbfix_pair_overrides`, and `urey_bradley_terms`.

## Files Changed

- `src/atomistic_prep/gpcrmd.py`: deterministic role paths, import details, MASS prelude wiring, protocol XSC box parsing.
- `src/atomistic_prep/topology_import.py`: PSF-derived CHARMM MASS prelude helper.
- `tests/test_gpcrmd_registry.py`: cache role resolution, MASS prelude wiring, protocol box propagation tests.
- `tests/test_atomistic_prep.py`: direct PSF MASS prelude unit test.

## Verification Reported

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py tests/test_atomistic_prep.py -k "gpcrmd or charmm"`: passed, 33 passed / 22 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/atomistic_prep/gpcrmd.py src/atomistic_prep/topology_import.py tests/test_gpcrmd_registry.py tests/test_atomistic_prep.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run atomistic-prep gpcrmd-import --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out /tmp/mlx-atomistic-gpcrmd-slice1-probe --json`: command completed; exported false with expected fail-closed CHARMM term blockers, not CT3.

## Concerns

- None blocking.
