# Slice 7: Replica Exchange

## Status: DONE

## Summary

Implemented `simulate_replica_exchange` with adjacent odd/even swaps, temperature exchange, lambda-scaled Hamiltonian exchange, deterministic seeding, swap diagnostics, and energy/state history. Added artifact metadata validation for `replica_exchange` and package exports.

## Files Changed

- `src/mlx_atomistic/replica_exchange.py`: New replica exchange driver, result/attempt dataclasses, Metropolis helpers, validation.
- `src/mlx_atomistic/artifacts.py`: Replica exchange metadata acceptance and validation.
- `src/mlx_atomistic/__init__.py`: Public exports for replica exchange APIs.
- `tests/test_replica_exchange.py`: Probability, deterministic swaps, explicit Hamiltonian exchange, metadata validation, unsupported runtime input, and odd/even scheduling tests.

## Verification

- `uv run pytest tests/test_replica_exchange.py -k "replica or exchange or histogram" && uv run ruff check src/mlx_atomistic/replica_exchange.py src/mlx_atomistic/md.py`: `16 passed`; ruff passed.
- `uv run pytest`: `736 passed`.
- `git diff --check`: passed.

## Reviewer Verdicts

- Spec review: CHANGES_REQUESTED for reversed Metropolis sign, then APPROVED after fix.
- Quality review: CHANGES_REQUESTED for metadata validation, Hamiltonian ambiguity, unsupported runtime inputs, and swap scheduling; CHANGES_REQUESTED again for shared multi-term Hamiltonian regression; APPROVED after final fix.

## Unresolved Risks

- Replica exchange fails closed for unsupported constrained/lazy-neighbor/reporter production inputs rather than silently dropping them.
