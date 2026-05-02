# Slice 007 Quality Review

- Slice: Scalable Periodic Nonbonded Gate
- Status: approved after one requested fix

## Initial Finding

- important: direct `NonbondedPotential` and `LennardJonesPotential` calls with lazy topology and no runtime pairs could select dense/tiled backend before the lazy-topology guard.

## Resolution

- `src/mlx_atomistic/forcefields.py` now checks lazy topology with no runtime pairs before `choose_nonbonded_backend` in `NonbondedPotential._pair_components`.
- `src/mlx_atomistic/md.py` now checks lazy topology with no runtime pairs before `choose_nonbonded_backend` in `LennardJonesPotential.energy_forces`.
- `tests/test_nonbonded_acceleration.py` covers both direct `backend="auto"` low-memory fallback guards.

## Final Review

- Status: approved.
- Issues: none.
- Residual risk: Slice 8 must still prevent PME/electrostatics from being treated as production-ready without validation.

## Evidence

- `src/mlx_atomistic/forcefields.py`: `NonbondedPotential._pair_components` rejects lazy topology with no pairs before backend selection.
- `src/mlx_atomistic/md.py`: `LennardJonesPotential.energy_forces` rejects lazy topology with no pairs before backend selection.
- `tests/test_nonbonded_acceleration.py`: regression tests cover both direct auto-backend lazy-topology guards.
- Focused follow-up command passed: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_nonbonded_acceleration.py -q`: 17 passed.
