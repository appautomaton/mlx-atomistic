# SPEC: Lean Runtime Boundary Cleanup

## Bounded Goal

Make the repository structure and project metadata clearly communicate that `mlx_atomistic` is the product runtime, while OpenMM, LAMMPS, and vendor checkouts are reference or validation surfaces only.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Keep `src/mlx_atomistic/` as the core MLX/Metal atomistic simulation library and avoid expanding it into a general external-engine wrapper.
- Keep `src/mlx_atomistic/prep/` as import/preparation tooling that may support MLX-ready artifact creation but does not make OpenMM or LAMMPS the main trajectory path.
- Treat OpenMM as a PyPI-installed reference and preview engine; do not add local OpenMM build work.
- Treat LAMMPS as a locally built upstream reference engine for GPU/OpenCL semantics and neighbor-list behavior; do not make it part of the primary runtime.
- Keep `vendors/` read-only reference material; do not import, build against, or copy vendor source into package code.
- Preserve existing useful reference notebooks and generated evidence, but label them so they cannot be mistaken for `mlx_atomistic` product output.
- Use `uv run ...` and the project environment for validation.

## Required Behavior

- Project docs state the runtime boundary in one place: `mlx_atomistic` generates project-owned trajectories; OpenMM and LAMMPS are reference, preview, or validation tools.
- Dependency metadata separates core package dependencies from dev/reference-engine dependencies and documents why OpenMM and LAMMPS are present.
- Notebook and artifact documentation consistently labels outputs as `mlx_atomistic`, `openmm-reference`, or `lammps-reference` where applicable.
- Any scripts that run OpenMM or LAMMPS are named, documented, or located so they read as reference/preview workflows rather than production runtime entrypoints.
- Generated or heavyweight artifacts remain out of source control, or are explicitly documented as local ignored outputs when they must exist for notebooks.
- Existing active runtime specs continue to treat external engines as non-runtime references unless a future spec explicitly changes that boundary.

## Acceptance Criteria

- A new contributor can answer from repo docs and metadata that the primary trajectory generator is `mlx_atomistic`, not OpenMM or LAMMPS.
- `pyproject.toml` keeps core runtime dependencies lean and keeps OpenMM/LAMMPS in non-core dependency surfaces.
- A repository scan of source and scripts finds no unintended OpenMM/LAMMPS runtime imports in `src/mlx_atomistic/`; any external-engine usage outside `vendors/` is limited to prep, scripts, notebooks, tests, or documented reference workflows.
- Notebook docs under `notebooks/ligand-receptor-motion/` label OpenMM outputs as reference or preview artifacts and MLX outputs as `mlx_atomistic` outputs.
- `.gitignore` or local artifact documentation covers generated OpenMM/LAMMPS/MLX trajectory outputs that should not be committed.
- Validation records the current OpenMM and LAMMPS provenance clearly: OpenMM is the `uv`/PyPI wheel with OpenCL available; LAMMPS is a `uv` local build from upstream PyPI source with GPU/OpenCL enabled.
- Source/test/script validation status is known after the cleanup.

## Blocking Questions Or Assumptions

- Assumption: the repo should stay MLX-first and lean, not become an engine orchestration project.
- Assumption: OpenMM and LAMMPS remain valuable enough to keep as dev/reference tools, but not as core dependencies.
- Assumption: existing generated notebook artifacts may remain locally available, but the cleanup should avoid committing new heavyweight data.
- Blocking question: none.

## Anti-Goals

- Do not build OpenMM locally.
- Do not remove OpenMM or LAMMPS entirely if doing so would lose useful reference and validation workflows.
- Do not implement new MD physics, PME, neighbor-list algorithms, or performance optimizations in this change.
- Do not reorganize the whole repository tree unless required to make the runtime boundary unambiguous.
- Do not rewrite notebooks for visual design or scientific content beyond boundary labels and artifact clarity.
- Do not change `vendors/` from reference-only material.
