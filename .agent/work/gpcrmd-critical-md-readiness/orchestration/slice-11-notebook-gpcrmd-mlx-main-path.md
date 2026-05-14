# Slice 11: Notebook GPCRmd MLX Main Path

## Result

Completed. `notebooks/ligand-receptor-motion/` now treats the MLX-generated
GPCRmd trajectory as the main result path.

## Scope

- Added a GPCRmd-specific notebook helper that runs or loads
  `mlx_atomistic.prep.runner.run_gpcrmd_mlx`.
- Kept `ensure_mlx_ligand_receptor_bundle(...)` as a compatibility alias while
  routing it to the GPCRmd MLX workflow.
- Updated the ligand-receptor notebook and README copy so GPCRmd reference
  trajectories are comparison context only.
- Added stale-artifact guards before visualization:
  - blocked runtime reports remain blocked;
  - requested target mismatch blocks cached output reuse;
  - corrupt `trajectory.npz` blocks instead of raising;
  - missing/corrupt prepared artifacts block instead of raising;
  - target/dynamics, atom-count, and mask-length mismatches block.

## Files Changed

- `notebooks/ligand-receptor-motion/helpers/mlx_real_md.py`
- `notebooks/ligand-receptor-motion/01-ligand-receptor-translational-motion.ipynb`
- `notebooks/ligand-receptor-motion/README.md`
- `notebooks/README.md`
- `tests/test_ligand_receptor_motion.py`
- `tests/test_mlx_prep.py`
- `.gitignore`

## Review

- Spec review: approved after adding the corrupt trajectory blocker path.
- Quality review: approved after adding the requested target mismatch guard for
  cached bundles.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_ligand_receptor_motion.py`
  - `15 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "ligand_receptor_motion or notebook or gpcrmd"`
  - `47 passed, 258 deselected`
- `rg -n "downloaded.*main|fake|benzene|steered" notebooks/ligand-receptor-motion notebooks/README.md`
  - no matches
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check notebooks/ligand-receptor-motion/helpers/mlx_real_md.py tests/test_ligand_receptor_motion.py tests/test_mlx_prep.py`
  - `All checks passed!`
- Notebook code cells parse with `ast.parse`.

## Remaining Risks

- The notebook shows the short NVT GPCRmd proof trajectory only when the selected
  target can import and run. Blocked targets intentionally stop before MD
  visualization and show blocker JSON.
- Full biological sampling and NPT/membrane barostat behavior remain out of
  scope for this slice.
