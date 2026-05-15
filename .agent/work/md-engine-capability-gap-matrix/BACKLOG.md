# BACKLOG: MD Engine Capability Gap Closure

## First Wave

These are ordered by production impact and dependency order.

| Order | Candidate SPEC | Matrix rows | Outcome |
| --- | --- | --- | --- |
| 1 | `production-artifact-openmm-parity-fixture` | `T2-02`, `T2-03`, `T2-06`, `T4-04`, `T6-02`, `T6-07` | Pick one small real AMBER or CHARMM artifact, build MLX terms from the same artifact, and compare component energy/forces against OpenMM at a fixed coordinate frame. |
| 2 | `pme-production-readiness` | `T1-05`, `T1-06`, `T1-08`, `T4-04`, `T5-03` | Replace or promote the PME backend only when force/energy parity and readiness checks pass. Make `pme_readiness_report` green for the selected fixture. |
| 3 | `runtime-reporters-and-diagnostics` | `T3-02`, `T3-05`, `T4-05`, `T5-04` | Add a minimal reporter/callback surface for sampled frames, state data, diagnostics, and parity traces without changing physics. |
| 4 | `checkpoint-restart-boundary` | `T3-03`, `T3-07`, `T4-05` | Serialize positions, velocities, step/time, thermostat/RNG state, neighbor-list policy, force-term metadata, and diagnostic cursor so long runs can resume cleanly. |
| 5 | `constraints-hmr-virtual-sites-policy` | `T1-07`, `T1-11`, `T2-07`, `T6-05` | Decide which 2-4 fs production workflows are supported now. Implement HMR/virtual sites or make artifact rejection exact and user-facing. |
| 6 | `npt-barostat` | `T1-04`, `T1-08`, `T2-01`, `T3-02` | Add NPT after PME and virial diagnostics are credible. Start with a Monte Carlo barostat and validate density/volume behavior against OpenMM. |
| 7 | `native-dcd-xtc-output` | `T3-04`, `T3-02`, `T6-06` | Expose DCD/XTC output through the reporter/output API, likely routing through MDTraj initially while keeping the product surface first-class. |
| 8 | `performance-hotpath-profile` | `T5-01`, `T5-02`, `T5-04`, `T5-05`, `T5-06`, `T5-07` | Profile the parity fixture after correctness gates. Only then choose MLX graph changes or custom Metal kernels for neighbor/pair/PME hotspots. |

## Deferred

These should not block the first production-ready MD wave.

| Item | Matrix rows | Reason |
| --- | --- | --- |
| Raw PDB plus force-field one-shot API | `T6-03` | Useful ergonomics, but the artifact path is already the product boundary. |
| Thermostat variety beyond Langevin | `T1-03` | Coverage expansion after NVT/NPT parity is trustworthy. |
| Triclinic cells | `T1-10` | Important for broad workflows, but first PME target can stay orthorhombic. |
| LJPME/dispersion correction | `T1-13` | Defer until Coulomb PME is production-ready. |
| Drude/AMOEBA/polarizable models | `T2-09` | Long-tail force-field coverage. |
| GROMACS/LAMMPS executable parity | `T4-06` | Source reference is enough until a specific validation need appears. |
| Mixed precision policy | `T5-08` | Needs measured correctness/performance data after PME/NPT. |
| FEP/TI/BAR, REMD, metadynamics, GBSA | not first-wave rows | Scientific coverage expansion, not required for fixed-charge production MD parity. |

## Next SPEC

Recommended next SPEC: `production-artifact-openmm-parity-fixture`.

Reason: PME, NPT, reporter, checkpoint, and performance work all need a stable
fixture and tolerance harness. Without a shared OpenMM/MLX artifact, PME changes
will be validated against local tests only, and performance numbers will not say
whether the trajectory is physically credible.

Suggested acceptance criteria for that SPEC:

- Select one small real AMBER or CHARMM prepared artifact that loads through
  `load_prepared_mlx_artifact(..., require_production=True)`.
- Build an OpenMM system from the same topology/parameter source.
- Compare total and component energies at one coordinate frame.
- Compare MLX forces against OpenMM forces with explicit tolerances.
- Record unsupported terms exactly instead of silently dropping them.
- Produce local results under ignored `results/` and commit only the small harness,
  tests, and planning evidence if implementation follows.

## Decision Checkpoint

Stop here before implementation. The next change should be framed as its own SPEC
using this backlog, with `production-artifact-openmm-parity-fixture` as the
default unless the user chooses to jump directly to PME.
