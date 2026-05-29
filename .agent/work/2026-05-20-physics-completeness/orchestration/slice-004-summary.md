# Slice 4: TIP4P-Ew Water Model and Parity

## Status: DONE

## Summary

Implemented narrow TIP4P-Ew geometry, parser/artifact metadata, virtual-site persistence, strict virtual-site artifact validation, real-atom `MMSystem` construction with `MMSystem.virtual_sites`, and persisted periodic TIP4P-Ew OpenMM PME parity. Spec review approved after corrections.

Quality re-review found a remaining runtime-integration gap: artifact-built TIP4P systems carry `system.virtual_sites`, but standard artifact runner paths created `SimulationConfig` without `virtual_sites=system.virtual_sites`. The plan was corrected and runner propagation was implemented.

## Files Changed By Implementation

- `src/mlx_atomistic/virtual_sites.py`: TIP4P-Ew constants/factories/geometry helpers.
- `src/mlx_atomistic/prep/topology_import.py`: Narrow AMBER TIP4P-Ew virtual-site detection/population.
- `src/mlx_atomistic/prep/schema.py`: Virtual-site parent padding support.
- `src/mlx_atomistic/prep/io.py`: Virtual-site array persistence.
- `src/mlx_atomistic/artifacts.py`: TIP4P policy, strict virtual-site validation, real/virtual split, runtime virtual-site manager attachment.
- `src/mlx_atomistic/mm.py`: Optional `virtual_sites` runtime boundary on `MMSystem`.
- `scripts/openmm_mlx_parity.py`: Persisted periodic TIP4P-Ew artifact parity helper.
- `tests/test_virtual_sites.py`: Geometry, persistence, validation, and runtime split coverage.
- `tests/test_openmm_mlx_parity.py`: TIP4P-Ew artifact/PME parity assertions.
- `src/mlx_atomistic/prep/runner.py`: Propagates `system.virtual_sites` into runner-created `SimulationConfig` paths.

## Verification Observed

- Targeted TIP4P/virtual-site tests passed after final correction: `31 passed, 26 deselected`.
- Targeted ruff reported passing.
- Full regression passed after final correction: `704 passed`.
- `git diff --check`: passed.

## Review Verdicts

- Spec review: CHANGES_REQUESTED for persistence, then APPROVED after `prep/io.py` correction.
- Quality review: CHANGES_REQUESTED for validation/runtime/parity gaps, CHANGES_REQUESTED for unresolved runner propagation, then APPROVED after runner propagation.

## Stop Reason Resolved

Reviewer requested changes twice for the same runtime-integration issue. Plan correction was applied, then the implementation was completed and re-reviewed.

## Plan Correction Applied

Added `src/mlx_atomistic/prep/runner.py` to Slice 4 touch scope and require runner-created `SimulationConfig` objects to propagate `system.virtual_sites` for NVT, NPT, minimize/equilibration, and steered paths as applicable.

## Final Evidence

- Runner-created configs propagate `system.virtual_sites` on direct NVT, NPT, virtual-site minimize/equilibration, and steered NVT paths.
- Persisted TIP4P-Ew artifact through `run_mlx(..., steps=0, minimize_steps=0, equilibration_steps=0, require_production=True)` returns real-atom runtime positions without manual wiring.
