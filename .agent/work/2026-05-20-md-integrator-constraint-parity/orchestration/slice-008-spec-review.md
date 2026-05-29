# Slice 8 Spec Review

Status: APPROVED

## Summary

- Slice 8 meets the Phase 1 parity and bounded end-to-end proof contract.

## Issues

- none

## Evidence

- `tests/test_openmm_mlx_parity.py` covers OpenMM-backed minimized energy, constrained water geometry, triclinic periodic distance, Nose-Hoover temperature statistics, and anisotropic barostat cell/volume trends with explicit tolerances.
- `tests/test_md_phase1_end_to_end.py` runs the bounded `minimize -> Nose-Hoover NVT -> anisotropic MC NPT` proof and checks finite outputs and protocol metadata.
- `scripts/openmm_mlx_parity.py` records OpenMM as reference-only evidence, not a product runtime dependency, and includes `AC8` in platform evidence.
- Final verification observed: targeted parity/e2e tests passed, global Ruff passed, and full pytest passed with `486 passed`.

## Residual Risk

- This remains bounded Phase 1 proof, not broad GPCRmd production-MD certification. Larger fixture evidence preserves that blocker boundary.
