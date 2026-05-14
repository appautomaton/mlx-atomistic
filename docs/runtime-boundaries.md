# Runtime Boundaries

`mlx_atomistic` is the primary trajectory generator and product runtime for this
repo. The project can use external engines for reference runs and validation,
but project-owned trajectories should come from the MLX implementation unless a
future spec explicitly changes that boundary.

## Core Runtime

- `src/mlx_atomistic/` contains the MLX/Metal simulation library: force terms,
  neighbor-list/runtime paths, integrators, reports, and benchmarks.
- Core package dependencies stay lean: MLX, NumPy, and SciPy are the runtime
  surface in `project.dependencies`.
- `src/mlx_atomistic/prep/` prepares and validates inputs for MLX-ready artifacts.
  It may use chemistry tooling, but it should not turn OpenMM or LAMMPS into the
  main runtime path.

## Reference Engines

- OpenMM is a reference and preview engine. In the current `uv` environment it
  resolves from the PyPI `openmm==8.5.1` macOS arm64 wheel and exposes
  `Reference`, `CPU`, and `OpenCL` platforms. We do not build OpenMM locally for
  this project.
- LAMMPS is a reference engine for GPU/OpenCL semantics and neighbor-list
  behavior. It is configured as a `uv` local build from the upstream PyPI source
  package with `PKG_GPU=ON`, `GPU_API=opencl`, and `GPU_PREC=single`.
- OpenMM and LAMMPS remain outside `project.dependencies`; they belong to
  dev/reference workflows.

## Vendor Checkouts

`vendors/` contains local reference source trees only. These trees are not
Python package inputs, are not imported by `mlx_atomistic`, and are not built by
`uv sync`. Use them for architecture study, algorithm references, and validation
planning.

## Notebook And Artifact Labels

Notebook data and generated reports should make the engine explicit:

- `mlx_atomistic`: product runtime output.
- `openmm-reference`: OpenMM preview or validation output.
- `lammps-reference`: LAMMPS reference or validation output.

Generated trajectories and heavyweight science artifacts should stay local and
ignored unless a later spec explicitly approves committing a small fixture.
