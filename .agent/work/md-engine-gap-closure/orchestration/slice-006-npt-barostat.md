# Slice 6: NPT / Barostat

## Status

Complete at decision checkpoint.

## What changed

- Added a minimal orthorhombic `MonteCarloBarostat`, `NPTResult`, and
  `simulate_npt` path in `src/mlx_atomistic/md.py`.
- Updated protocol gating so `ensemble=NPT` with `barostat=monte_carlo` is
  accepted, while missing/unsupported barostats and membrane barostats still
  fail closed.
- Routed `prep.run_mlx` NPT requests through `simulate_npt`, persist final NPT
  cell metadata, and checkpoint the final NPT cell.
- Added a physical pressure conversion constant for Angstrom/kJ/mol artifacts:
  `ATM_TO_KJ_PER_MOL_ANGSTROM3`.
- Added `scripts/run_openmm_mlx_npt_parity.py` for a short OpenMM-vs-MLX PME
  NPT volume comparison on the shared AMBER fixture.
- Updated tests that encoded the old "all NPT is blocked" policy.

## Evidence

- `uv run pytest tests/test_npt.py tests/test_protocols.py tests/test_mlx_prep.py -q`
  passed: `36 passed`.
- `uv run pytest tests/test_npt.py tests/test_protocols.py tests/test_mlx_prep.py tests/test_runtime_boundaries.py -q`
  passed: `44 passed`.
- `uv run pytest tests/test_virial_pressure.py tests/test_protocols.py tests/test_npt.py -q`
  passed: `16 passed`.
- `uv run ruff check src/mlx_atomistic/md.py src/mlx_atomistic/protocols.py src/mlx_atomistic/prep/runner.py src/mlx_atomistic/units.py scripts/run_openmm_mlx_npt_parity.py tests/test_npt.py tests/test_protocols.py tests/test_mlx_prep.py tests/test_runtime_boundaries.py`
  passed.
- `uv run python scripts/run_openmm_mlx_npt_parity.py --fixture amber-alanine-dipeptide-implicit --out results/md-engine-gap-closure/npt-parity`
  passed.

## NPT parity result

- Fixture: `amber-alanine-dipeptide-implicit`
- Atom count: `22`
- OpenMM platform: `Reference`
- PME readiness: `ready`
- MLX volume ratio: `1.005016357421875`
- OpenMM volume ratio: `0.9974641762766988`
- Absolute ratio delta: `0.0075521811451760845`
- Acceptance bound: `0.25`
- MLX barostat attempts: `1`
- MLX accepted attempts: `1`

## Notes

The plan's direct `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ...` form
ran inside the restricted sandbox path in this session and failed with
`No Metal device available`. The same tests and parity command passed through
the project-standard approved `uv run ...` path with device access.

## Decision

The first supported orthorhombic Monte Carlo NPT path is credible enough for
the next output-polish slice. Future work should still improve physical depth:
barostat attempts currently run as the first supported short-NPT proof path,
not as a mature long-production NPT engine with analytic PME virial.
