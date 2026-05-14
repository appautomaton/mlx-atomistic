# Slice 008 Quality Review

- Slice: PME Readiness Gate For GPCRmd
- Status: approved after requested fixes

## Initial Findings

- important: legacy tiny GPCRmd tests and prototype callers relied on default electrostatics while default PME now blocks.
- important: `short-range-prototype` initially changed only metadata and could still execute artifact-declared PME.
- minor: benchmark rows could label a completed prototype run as requested PME.

## Resolution

- Tiny trajectory-producing tests and benchmark callers now pass `electrostatics="short-range-prototype"` explicitly.
- Prototype mode now forces runtime cutoff electrostatics before force-term construction.
- Prototype trajectory metadata remains non-production and cannot be confused with production PME.
- Benchmark completed rows label the actual runnable request as `short-range-prototype`.

## Final Review

- Status: approved.
- Issues: none.
- Residual risk: production PME remains unavailable until a non-NumPy production backend is implemented and validated.

## Evidence

- `src/mlx_atomistic/prep/runner.py`: blocks GPCRmd runs when electrostatics readiness has blockers.
- `src/mlx_atomistic/prep/runner.py`: forces runtime cutoff only for `short-range-prototype` and records non-production metadata.
- `src/mlx_atomistic/pme.py`: reports `pme_backend_not_production_executable:current_backend=numpy_reference`.
- `tests/test_production_artifacts.py`: verifies a PME artifact prototype run emits cutoff terms, not PME terms.
- `tests/test_gpcrmd_registry.py`: verifies benchmark rows label actual prototype execution.
- Focused regression commands and Ruff checks passed.
