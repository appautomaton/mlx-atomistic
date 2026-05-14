# Slice 1: Central Runtime Boundary

## Status

complete

## Route

direct

## Files Changed

- `README.md`: added the repo-level runtime-boundary pointer and clarified package layout.
- `docs/runtime-boundaries.md`: added the authoritative boundary note for `mlx_atomistic`, OpenMM, LAMMPS, and `vendors/`.
- `pyproject.toml`: added dependency-role comments while keeping OpenMM and LAMMPS outside `project.dependencies`.

## Verification

- `rg -n "primary trajectory generator|reference|OpenMM|LAMMPS|vendors" README.md docs/runtime-boundaries.md pyproject.toml` passed.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -c 'import tomllib; ...'` passed after an approved unsandboxed rerun. The first sandboxed run failed while fetching `hatchling` for the local editable package rebuild.

## Notes

- No core dependency relocation was needed; `project.dependencies` already stayed lean.
