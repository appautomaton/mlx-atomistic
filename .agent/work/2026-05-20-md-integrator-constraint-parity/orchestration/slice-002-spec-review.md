# Slice 2 Spec Review

Status: APPROVED

## Summary

- Slice 2 matches the requested triclinic runtime propagation scope.
- Matrix cells are preserved in prep/artifact/checkpoint/trajectory paths.
- Unsupported compact-neighbor and PME triclinic paths fail closed.

## Issues

- none

## Evidence

- `Cell.volume` uses the matrix determinant and `Cell.minimum_image()` uses fractional coordinates.
- Dense neighbor pairs use `Cell.minimum_image()`; compact neighbor backends reject triclinic cells.
- PME rejects triclinic cells and pressure diagnostics use `cell.volume`.
- `PreparedSystem.cell_matrix` validates, persists, loads, and GPCRmd box application preserves matrix vectors.
- Verification observed: targeted Slice 2 pytest set passed and `tests/test_mlx_prep.py` passed.
