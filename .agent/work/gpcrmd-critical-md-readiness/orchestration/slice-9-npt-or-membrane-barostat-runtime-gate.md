# Slice 9: NPT Or Membrane-Barostat Runtime Gate

## Result

- Status: completed
- Route: subagent implementer with spec and quality review
- Auto-continue: no

## Scope

The selected GPCRmd target is NVT, so this slice implemented the deferred path: a strict short-NVT proof gate. No NPT, membrane barostat, or volume-move mechanics were added.

## Files Changed

- `src/mlx_atomistic/protocols.py`: added the NVT-only protocol compatibility gate, exact protocol blocker reporting, pre-run validation in `run_minimize_then_nvt`, and protocol metadata on `ProtocolResult`.
- `src/atomistic_prep/runner.py`: validates prepared artifact protocol metadata before MLX run paths and saves normalized protocol metadata into trajectories.
- `tests/test_protocols.py`: covers accepted short-NVT metadata, rejected NPT/barostat/membrane-barostat requests, pre-run failure, and result metadata.
- `tests/test_atomistic_prep.py`: covers runner-level protocol rejection before system build and normalized NVT trajectory metadata persistence.

## Review Loop

- Implementer: `DONE`
- Spec review 1: `APPROVED`
- Quality review 1: `CHANGES_REQUESTED`
  - Issue: `atomistic_prep.runner.run_mlx` could bypass the protocol gate by ignoring artifact `protocol_metadata`, including its direct `simulate_nvt` branch.
  - Fix: runner paths now validate artifact protocol metadata before system construction and save normalized protocol metadata after caller overrides.
- Spec review 2: `APPROVED`
- Quality review 2: `APPROVED`

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "npt or barostat or protocol"`
  - Result: `10 passed, 286 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_protocols.py tests/test_atomistic_prep.py -k "protocol or npt or barostat"`
  - Result: `8 passed, 22 deselected`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/protocols.py src/atomistic_prep/runner.py tests/test_protocols.py tests/test_atomistic_prep.py`
  - Result: `All checks passed!`

## Remaining Risks

- Barostat support remains intentionally unimplemented. GPCRmd runtime proof mode is short NVT only; NPT and membrane-barostat requests fail closed with exact blockers.
