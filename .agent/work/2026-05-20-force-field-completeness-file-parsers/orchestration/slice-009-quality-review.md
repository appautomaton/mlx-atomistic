# Slice 9 Code Quality Review: OpenMM Parity Harness For Accepted Imports

Status: `APPROVED`

Summary:
- Initial review requested changes for missing blocked-report serialization, zeroed CHARMM component parity, brittle Urey-Bradley mapping, and missing CLI routing.
- Re-review found no remaining findings.

Issues fixed:
- `important`: `_blocked_report` now writes `openmm_mlx_parity_report.json`, so blocked runs preserve the report contract.
- `important`: CHARMM parity asserts nonzero OpenMM component magnitudes for each claimed CHARMM component.
- `important`: `HarmonicBondForce` mapping is source/count-aware and treats ambiguous extra harmonic-bond force groups as unsupported.
- `minor`: `scripts/run_openmm_mlx_parity.py` now routes AMBER, CHARMM, and GROMACS through format-specific fixtures and arguments.
- `minor`: Shared compatibility normalization moved out of `prep.schema` into `mlx_atomistic.compatibility`, restoring the runtime/prep boundary.

Evidence:
- `scripts/openmm_mlx_parity.py` writes blocked reports and blocks unsupported OpenMM force classes.
- `tests/test_openmm_mlx_parity.py` covers missing fixture JSON, OpenMM reference failure JSON, nonzero CHARMM components, ambiguous harmonic-bond blocking, and CLI dispatch.
- `tests/test_runtime_boundaries.py` passes after the compatibility helper move.

Verification:
- Slice 9 parity gate passed outside the sandbox: `20 passed, 6 deselected`.
- Runtime boundary gate passed: `8 passed`.
- CHARMM artifact/prep regression passed outside the sandbox: `39 passed, 140 deselected`.
- Cross-format compatibility gate passed outside the sandbox: `117 passed, 62 deselected`.
- Targeted Ruff and `git diff --check` passed.

Residual risk:
- None beyond the existing Metal-device requirement for MLX runtime tests.
