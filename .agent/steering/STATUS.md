---
active_change: lean-runtime-boundary-cleanup
stage: verify
---

# Status

## Current Change

- active change: `lean-runtime-boundary-cleanup`
- current stage: `verify`

## What Is True Now

- Office-hours direction is approved: keep the repo lean by making `mlx_atomistic` the product runtime and treating OpenMM, LAMMPS, and `vendors/` as reference or validation surfaces only.
- OpenMM in the project `.venv` was removed and reinstalled through `uv`; it resolves from the PyPI `openmm==8.5.1` macOS arm64 wheel and exposes `Reference`, `CPU`, and `OpenCL` platforms.
- LAMMPS is configured as a `uv` local build from the upstream PyPI source package with `PKG_GPU=ON`, `GPU_API=opencl`, and `GPU_PREC=single`; the installed runtime reports `has_GPU=True`.
- `SPEC.md` exists at `.agent/work/lean-runtime-boundary-cleanup/SPEC.md`.
- `PLAN.md` exists at `.agent/work/lean-runtime-boundary-cleanup/PLAN.md`.
- Execution is complete through Slice 3.
- `VERIFY.md` exists at `.agent/work/lean-runtime-boundary-cleanup/VERIFY.md`.
- Targeted verification passed: boundary/import tests plus `tests/test_mlx_prep.py` reported `32 passed in 6.31s` on an approved unsandboxed run, and `ruff check src tests scripts` reported `All checks passed!`.
- Final auto-verify passed: fresh targeted pytest reported `32 passed in 4.39s`, source/test/script Ruff passed, OpenMM/LAMMPS provenance checks passed, and all 7 acceptance criteria passed.
- OpenMM provenance was rechecked as `uv`/PyPI `openmm==8.5.1` with platforms `Reference`, `CPU`, and `OpenCL`.
- LAMMPS provenance was rechecked as a `uv` local build with runtime version `20250722` and GPU package support `True`.
- `mlx_atomistic.prep` is now the only preparation/import package surface; the legacy package shim and prep console command have been removed.
- Follow-on prep migration verification passed before removal: targeted prep/boundary pytest reported `129 passed in 6.80s` on an approved unsandboxed run, and source/test/script Ruff passed.

## Next Step

No execution gaps remain for `lean-runtime-boundary-cleanup`.

## Open Risks

- Existing notebooks and generated artifacts can blur the line between `mlx_atomistic` output and external-engine reference output unless labels and docs are made explicit.
- Dependency metadata must stay lean without deleting useful reference-engine workflows that are still valuable for validation.
- LAMMPS runtime checks require unsandboxed execution on this machine because MPI initialization is blocked by sandbox network-interface policy.
