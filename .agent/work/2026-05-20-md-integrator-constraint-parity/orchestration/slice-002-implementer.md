# Slice 2 Implementer

Status: DONE

## Summary

- Implemented triclinic runtime propagation through matrix-capable `Cell`.
- Dense neighbor and nonbonded paths use `Cell.minimum_image()`.
- Compact neighbor, PME, and isotropic NPT paths fail closed for unsupported triclinic cases.
- Pressure and virial diagnostics use matrix cell volume and fractional strain.
- Trajectory/checkpoint/artifact payloads preserve triclinic matrices while keeping orthorhombic length-vector compatibility.

## Quality Fix

After quality review, the implementer added `cell_matrix` through the prepared-system schema/save/load path and preserved GPCRmd full box vectors.

## Verification

- `uv run pytest tests/test_triclinic_cell.py tests/test_neighbors.py tests/test_nonbonded_acceleration.py tests/test_pme.py tests/test_virial_pressure.py tests/test_checkpoint_restart.py` -> `94 passed`.
- `uv run pytest tests/test_mlx_prep.py` -> `31 passed`.
- Focused prep matrix tests passed.
- Focused Ruff check passed for prep matrix-persistence files.
