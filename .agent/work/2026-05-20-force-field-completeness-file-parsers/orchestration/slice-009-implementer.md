# Slice 9 Implementer: OpenMM Parity Harness For Accepted Imports

Status: `DONE`

Route: subagent implementer with coordinator corrections and independent re-review.

Implemented:
- Generalized `scripts/openmm_mlx_parity.py` from AMBER-only parity to AMBER, CHARMM, and GROMACS accepted-import parity.
- Added report fields for reference engine role, artifact/readiness state, source paths, platform evidence, unsupported terms, component errors, force max/RMS errors, and pass/block status.
- Added tracked parity fixtures for AMBER, CHARMM, and GROMACS instead of relying on vendor paths.
- Added CHARMM OpenMM loading through PSF/PDB plus RTF/PRM, including nonzero charge, LJ, Urey-Bradley, CMAP, and nonbonded component coverage.
- Added GROMACS OpenMM loading through `.top`/`.gro` and RB component reporting.
- Added `--source-kind` CLI routing for AMBER, CHARMM, and GROMACS parity runs.

Review corrections:
- Blocked reports now serialize `openmm_mlx_parity_report.json` on blocked paths.
- CHARMM parity no longer relies on a zeroed component fixture for acceptance; tests assert nonzero OpenMM magnitudes for claimed CHARMM components.
- HarmonicBondForce-to-Urey-Bradley mapping is source/count-aware, and ambiguous extra harmonic-bond forces are blocked.
- CLI dispatch is covered by routing regression tests.
- The compatibility normalizer was moved to `src/mlx_atomistic/compatibility.py` so artifact validation no longer imports the prep layer.

Changed paths:
- `scripts/openmm_mlx_parity.py`
- `scripts/run_openmm_mlx_parity.py`
- `src/mlx_atomistic/compatibility.py`
- `src/mlx_atomistic/artifacts.py`
- `src/mlx_atomistic/prep/schema.py`
- `tests/test_openmm_mlx_parity.py`
- `tests/fixtures/amber/`
- `tests/fixtures/charmm/`
- `tests/fixtures/gromacs/`

Verification:
- `uv run pytest tests/test_openmm_mlx_parity.py -k "amber or charmm or gromacs or pme"` passed outside the sandbox: `20 passed, 6 deselected`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_runtime_boundaries.py` passed: `8 passed` with the known MLX Metal atexit warning.
- `uv run pytest tests/test_mlx_prep.py tests/test_production_artifacts.py -k "charmm or cmap or urey or nbfix"` passed outside the sandbox: `39 passed, 140 deselected`.
- `uv run pytest tests/test_production_artifacts.py tests/test_mlx_prep.py -k "artifact or compatibility or unsupported or rb or pme or charmm or gromacs"` passed outside the sandbox: `117 passed, 62 deselected`.
- Targeted Ruff passed for Slice 9 and compatibility-boundary files.
- `git diff --check` passed for Slice 9 and compatibility-boundary files.

Concerns:
- MLX runtime parity gates require Metal access; sandboxed runs fail on device loading, while the same gates pass outside the sandbox.
