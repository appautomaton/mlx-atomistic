# Slice 8: Virial And Pressure Diagnostics

## Result

- Status: completed
- Route: subagent implementer with spec and quality review
- Auto-continue: yes

## Scope

Added diagnostic-only virial and pressure reporting needed before any later pressure-coupled runtime gate. No NPT or barostat behavior was added.

## Files Changed

- `src/mlx_atomistic/md.py`: added virial/pressure diagnostic arrays to NVE/NVT results, explicit virial support validation, pressure diagnostics, and periodic orthorhombic cell-strain virial diagnostics.
- `src/mlx_atomistic/io.py`: saved and loaded `virial_tensor`, `pressure_tensor`, and `pressure` in native trajectory artifacts, with zero defaults for older artifacts.
- `src/mlx_atomistic/forcefields.py`: added explicit `supports_virial=True` declarations to supported built-in force terms.
- `src/mlx_atomistic/charmm_terms.py`: added explicit `supports_virial=True` declarations to supported CHARMM primitive terms.
- `tests/test_virial_pressure.py`: added finite periodic diagnostics, sparse diagnostic-axis, trajectory round-trip, older-artifact default, explicit support-gate, and periodic wrap-invariance coverage.

## Review Loop

- Implementer: `DONE`
- Spec review 1: `CHANGES_REQUESTED`
  - Issue: virial support was inferred from module membership, so internal terms could be accepted without explicit support.
  - Fix: `_term_supports_virial()` now accepts only explicit `supports_virial` or dedicated virial diagnostic methods.
- Spec review 2: `APPROVED`
- Quality review 1: `CHANGES_REQUESTED`
  - Issue: periodic virial used wrapped absolute coordinates and was origin/wrap dependent.
  - Fix: periodic virial now uses explicit-support-gated orthorhombic cell-strain finite differences with fractional coordinates held fixed.
- Spec review 3: `APPROVED`
- Quality review 2: `APPROVED`

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "virial or pressure or trajectory"`
  - Result: `16 passed, 272 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nve.py tests/test_nvt.py tests/test_energy_decomposition.py tests/test_real_mm_core.py tests/test_virial_pressure.py`
  - Result: `29 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/md.py src/mlx_atomistic/forcefields.py src/mlx_atomistic/charmm_terms.py src/mlx_atomistic/io.py tests/test_virial_pressure.py`
  - Result: `All checks passed!`

## Remaining Risks

- Periodic virial diagnostics are diagonal-only for orthorhombic cell strain in this slice. Off-diagonal virials and barostat moves remain out of scope for Slice 8.
