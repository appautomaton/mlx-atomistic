# Slice 001 Summary

- Slice: GPCRmd Cache And Protocol Normalization
- Status: completed
- Execution route: subagent implementation with spec and quality review
- Stop reason: execution window intentionally stops before Slice 2, the strict CHARMM term export checkpoint.

## What Changed

- GPCRmd cache inspection now reports deterministic `resolved_role_paths`.
- GPCRmd CHARMM import now builds a PSF-derived MASS prelude when the PSF contains atom types absent from the parameter file.
- GPCRmd import details now include source role paths, derived MASS metadata, and protocol box metadata.
- Protocol `input.xsc` files near extracted GPCRmd protocol archives are parsed into box vectors and cell lengths.
- Prepared GPCRmd artifacts receive `cell_lengths` from protocol metadata before later validation.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py tests/test_mlx_prep.py -k "gpcrmd or charmm"`: 33 passed, 22 deselected.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gpcrmd.py src/mlx_atomistic/prep/topology_import.py tests/test_gpcrmd_registry.py tests/test_mlx_prep.py`: passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run mlx_atomistic.prep Python API gpcrmd-import --cache notebooks/ligand-receptor-motion/data/gpcrmd-cache/729 --out /tmp/mlx-atomistic-gpcrmd-slice1-probe --json`: command completed and returned expected blockers `unsupported_terms:charmm_cmap_terms`, `unsupported_terms:nbfix_pair_overrides`, and `unsupported_terms:urey_bradley_terms`.
- Temporary probe output under `/tmp/mlx-atomistic-gpcrmd-slice1-probe` was removed.
- `git ls-files notebooks/ligand-receptor-motion/data`: no tracked files.
- `git status --ignored --short notebooks/ligand-receptor-motion/data`: data directory is ignored.

## Next

Slice 2 should implement strict CHARMM term export for Urey-Bradley, CMAP, and NBFIX handling or stop with a precise unsupported-term blocker. Do not proceed by dropping any of those terms.
