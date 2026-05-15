---
active_change: docs-hygiene-follow-up
stage: complete
---

# Status

## Current Change

- active change: `lean-runtime-boundary-cleanup`
- current stage: `verify`

## What Is True Now

- Office-hours direction is approved: keep the repo lean by making `mlx_atomistic` the product runtime and treating OpenMM, LAMMPS, and `vendors/` as reference or validation surfaces only.
- OpenMM in the project `.venv` was removed and reinstalled through `uv`; it resolves from the PyPI `openmm==8.5.1` macOS arm64 wheel and exposes `Reference`, `CPU`, and `OpenCL` platforms.
- LAMMPS is configured as a `uv` local build from the upstream PyPI source package with `PKG_GPU=ON`, `GPU_API=opencl`, and `GPU_PREC=single`; the installed runtime reports `has_GPU=True`.
- OpenMM provenance was rechecked as `uv`/PyPI `openmm==8.5.1` with platforms `Reference`, `CPU`, and `OpenCL`.
- LAMMPS provenance was rechecked as a `uv` local build with runtime version `20250722` and GPU package support `True`.
- `mlx_atomistic.prep` is now the only preparation/import package surface; the legacy package shim and prep console command have been removed.
- Current setup docs use the full notebook/prep/viz environment: `uv sync --extra notebook --extra prep --extra viz --group dev`.
- Local `main` tracks `origin/main` at `https://github.com/appautomaton/mlx-atomistic.git`.
- Historical `.agent/work/**` records are retained as provenance and are not expected to be rewritten for every package/command rename.
- Docs-hygiene verification passed: old prep CLI wording is absent from active docs, `OpenMM.ipynb` has no code-cell outputs, source/test/script Ruff passed, `uv lock --check` passed, and the targeted MLX regression suite reported `154 passed in 6.75s` on an approved unsandboxed run.

## Next Step

No execution gaps remain for `docs-hygiene-follow-up`.

## Open Risks

- The active OpenMM exploratory notebook must stay output-free or be archived later; the current cleanup clears outputs in place.
- Dependency metadata must stay lean without deleting useful reference-engine workflows that are still valuable for validation.
- LAMMPS runtime checks require unsandboxed execution on this machine because MPI initialization is blocked by sandbox network-interface policy.
