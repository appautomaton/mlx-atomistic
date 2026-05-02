# VERIFY: GPCRmd Real Artifact MLX MD Run

## Verification: Slice 11 Full Verification And Handoff

- Criterion: Focused GPCRmd tests pass.
  - Result: PASS
  - Evidence: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py tests/test_ligand_receptor_motion.py tests/test_production_artifacts.py -k "gpcrmd or electrostatics or notebook or pbc"` passed with `47 passed, 44 deselected`.
  - Gap: none.

- Criterion: Full test suite status is known.
  - Result: PASS
  - Evidence: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests notebooks/ligand-receptor-motion` passed with `All checks passed`. `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest` passed with `342 passed in 25.60s` after the neighbor-list tuning pass.
  - Gap: none.

- Criterion: Run report or blocker report path is documented.
  - Result: PASS
  - Evidence: Run report exists at `notebooks/ligand-receptor-motion/data/gpcrmd-mlx/729-proof/gpcrmd_mlx_run_report.json`; it reports `status=ran`, `trajectory_written=true`, `engine=mlx_atomistic`, `workflow=run_gpcrmd_mlx`, `electrostatics_route=short-range-prototype`, `steps=10`, `sampled_frame_count=11`, `simulated_time_ps=0.005`, `elapsed_wall_seconds=13.944710792042315`, `integration_steps_per_second=0.7171177767061757`, `nonbonded_runtime.backend=periodic_cell_list`, `nonbonded_runtime.skin=2.5`, `nonbonded_runtime.rebuild_count=1`, `total_energy_finite=true`, and `temperature_finite=true`.
  - Gap: none.

- Criterion: No generated GPCRmd data or trajectory artifact is staged for commit.
  - Result: PASS
  - Evidence: `git status --short` showed no staged entries. The generated trajectory/report files are under ignored notebook data paths and do not appear as staged changes.
  - Gap: none.

## Content Checks

- Audience: PASS. The handoff artifacts address future maintainers and notebook users by naming exact report/trajectory paths, proof limitations, and next actions.
- Thesis: PASS. The core claim is that execution produced a real MLX GPCRmd short-range prototype proof and full verification passed; the verification evidence supports that claim.
- Content anti-goals: PASS. The notes do not claim production PME, binding/unbinding, or long-timescale GPCR dynamics.
- Channel and format: PASS. This is a repository verification artifact under `.agent/work/gpcrmd-real-artifact-md-run/VERIFY.md`.
- Source policy and factual risk: PASS. Claims are tied to local command output and run-report observations.
- Anti-slop scan: PASS. No promotional claims or unsupported significance inflation found.

## Overall

- Overall: PASS
- Remaining gaps: Production PME remains intentionally unavailable; the generated trajectory is a 10-step short-range prototype proof, not a production PME trajectory or long-timescale GPCR dynamics.
- Recommended next skill: none; verification stage is complete.
