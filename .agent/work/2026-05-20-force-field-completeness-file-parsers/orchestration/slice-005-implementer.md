# Slice 5 Implementer: AMBER Import Completion And Phase 2 Metadata Alignment

## Status

DONE

## Files Changed

- `src/mlx_atomistic/prep/topology_import.py`
- `scripts/openmm_mlx_parity.py`
- `tests/test_mlx_prep.py`
- `tests/test_openmm_mlx_parity.py`
- `tests/fixtures/amber/alanine-dipeptide-implicit.prmtop`
- `tests/fixtures/amber/alanine-dipeptide-implicit.inpcrd`

## Implementation Summary

- Completed native AMBER `prmtop`/`inpcrd` import for the accepted ff14SB-style fixture.
- Preserved AMBER atoms, residues, charges, masses, LJ parameters, bonds, angles, periodic dihedrals, impropers, constraints, nonbonded exceptions, periodic box metadata, and Phase 2 compatibility metadata.
- Derived AMBER 1-4 electrostatic/LJ scaling from `SCEE_SCALE_FACTOR` and `SCNB_SCALE_FACTOR` when present, with standard AMBER fallback only when absent.
- Parsed AMBER explicit exclusion records while preserving the distinction between absent tables and explicit empty/zero-sentinel tables.
- Added broad fail-closed validation for unsupported or malformed AMBER records, including malformed atom arrays, residue pointers, fixed-width numeric records, restart values, bond/angle/dihedral parameters, LJ tables, atom types, exclusions, 1-4 scaling, periodic box metadata, and schema validation failures.
- Updated AMBER OpenMM/MLX parity handling so `TopologyImportError` returns blocked reports with parsed `unsupported_terms`.
- Added PME assignment-order opt-in for AMBER parity artifacts and cleared stale `cell_matrix` metadata when PME cell lengths override the imported box.
- Updated AMBER parity evidence IDs to Phase 2 traceability: `AC-03`, `AC-07`, `P2-PARSE-01`, and `P2-PARITY-01`.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py -k "amber"` passed outside the sandbox: `54 passed, 46 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/topology_import.py scripts/openmm_mlx_parity.py tests/test_mlx_prep.py tests/test_openmm_mlx_parity.py` passed.

## Concerns

- None for Slice 5 completion. Residual risk remains that broader AMBER variants beyond the accepted subset rely on fail-closed blockers rather than broad real-fixture coverage.
