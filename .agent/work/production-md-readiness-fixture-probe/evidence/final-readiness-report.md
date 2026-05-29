# Production MD Readiness Fixture Probe

- fixture: `gpcrmd-729-beta1-5f8u-cyanopindolol`
- status: `blocked`
- bounded pass: `false`

## Blocking Categories

- `topology_terms`: lazy topology requires a runtime nonbonded pair provider; full dense pair materialization was not requested
  - command: `run_minimize_then_nvt bounded production probe`
  - next: fix MLX runtime blocker before bounded fixture execution
- `performance_runtime`: bounded run attempted but did not complete
  - command: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/run_mlx_production_md_probe.py --candidate .agent/work/production-md-readiness-fixture-probe/evidence/candidate-fixture.json --out .agent/work/production-md-readiness-fixture-probe/evidence/mlx-probe.json`
  - next: fix runtime blocker before timing claims
- `output_restart`: no trajectory, checkpoint, or restart output because bounded MLX run blocked
  - command: `MLX probe output check`
  - next: record output/restart behavior after runtime blocker is fixed

## Category Matrix

| Category | Status | Prevents Pass | Observed Result |
| --- | --- | --- | --- |
| `artifact_source` | `passed` | `false` | selected local GPCRmd cache fixture with 92001 atoms |
| `preparation` | `passed` | `false` | prepared artifact exported for selected fixture |
| `topology_terms` | `blocked` | `true` | lazy topology requires a runtime nonbonded pair provider; full dense pair materialization was not requested |
| `forcefield_terms` | `passed` | `false` | prepared terms represented: 16 required term families |
| `constraints_hmr_virtual_sites` | `partial` | `false` | hmr_or_virtual_site_policy_required |
| `electrostatics_pme` | `partial` | `false` | periodic explicit membrane system requires PME-scale electrostatics |
| `npt_barostat` | `passed` | `false` | selected fixture protocol is NVT; no barostat required |
| `integrator_protocol` | `partial` | `false` | NVT short proof protocol is accepted; NPT is not required |
| `stability_finiteness` | `partial` | `false` | MLX prep produced finite positions and velocities; energies are unavailable because bounded run blocked |
| `parity_tolerance` | `partial` | `false` | OpenMM reference ran with finite outputs; comparison is bounded by documented protocol divergences |
| `performance_runtime` | `blocked` | `true` | bounded run attempted but did not complete |
| `output_restart` | `blocked` | `true` | no trajectory, checkpoint, or restart output because bounded MLX run blocked |
| `dependency_boundary` | `passed` | `false` | MLX probe imported no reference engines or vendors |

## Production Claim Boundary

This report is one bounded fixture probe. It is not broad production MD certification.
