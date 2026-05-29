# Slice 9 Spec Review: OpenMM Parity Harness For Accepted Imports

Status: `APPROVED`

Summary:
- Initial review requested changes because the CHARMM acceptance fixture used zeroed charge, LJ, and CMAP terms, proving wiring but not meaningful CHARMM component parity.
- Re-review approved after nonzero CHARMM component parity, blocked-report serialization, ambiguity blocking, and CLI routing landed.

Issues fixed:
- `important`: CHARMM parity now checks nonzero OpenMM component magnitudes for `bond`, `angle`, `torsion`, `urey_bradley`, `charmm_cmap`, and `nonbonded`.
- `important`: Ambiguous extra harmonic-bond OpenMM forces are blocked instead of silently mapped to Urey-Bradley.
- `important`: Blocked report paths now still produce the machine-readable parity report.
- `minor`: The CLI can route all accepted source kinds through `--source-kind`.

Evidence:
- `tests/test_openmm_mlx_parity.py` covers AMBER, CHARMM, GROMACS, blocked-report JSON, component mapping, and CLI routing.
- `scripts/openmm_mlx_parity.py` carries the report fields required by Slice 9 and maps AMBER/CHARMM/GROMACS OpenMM force classes into parity components.
- `scripts/run_openmm_mlx_parity.py` routes `amber`, `charmm`, and `gromacs`.
- `tests/test_runtime_boundaries.py` confirms OpenMM remains outside product runtime imports.

Verification:
- Slice 9 parity gate passed outside the sandbox: `20 passed, 6 deselected`.
- Runtime boundary gate passed: `8 passed`.

Residual risk:
- None beyond the Metal-capable environment required for MLX/OpenMM parity execution.
