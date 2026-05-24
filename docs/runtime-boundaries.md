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

## Platform Boundary

`mlx_atomistic.runtime.get_platform_boundary_report()` describes the local
mini-platform boundary without importing reference engines. The report names the
product runtime, active MLX runtime information, reference-engine policy, and
local concept groups for runtime/backend, system/artifact, protocol, readiness,
validation, and DFT/QM scope.

## Reference Engines

- OpenMM is a reference and preview engine. In the current `uv` environment it
  resolves from the PyPI `openmm==8.5.1` macOS arm64 wheel and exposes
  `Reference`, `CPU`, and `OpenCL` platforms. We do not build OpenMM locally for
  this project.
- LAMMPS is a reference engine for GPU/OpenCL semantics and neighbor-list
  behavior. It is configured as a `uv` local build from the upstream PyPI source
  package with `PKG_GPU=ON`, `GPU_API=opencl`, and `GPU_PREC=single`.
- GROMACS is a reference for biomolecular MD staging, PME/nonbonded performance
  shape, preprocessing boundaries, and trajectory-analysis conventions.
- CP2K and Quantum ESPRESSO are references for DFT/QM suite boundaries and force
  environment discipline, not product runtime engines for this package.
- OpenMM and LAMMPS remain outside `project.dependencies`; reference engines
  belong to dev/reference workflows unless a future spec explicitly changes
  that boundary.

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
- `gromacs-reference`: GROMACS reference or validation output.
- `cp2k-reference` / `qe-reference`: electronic-structure reference outputs.

Generated trajectories and heavyweight science artifacts should stay local and
ignored unless a later spec explicitly approves committing a small fixture.

## Platform Evidence

Runtime proof paths now carry compact platform metadata:

- prepared MLX trajectories and checkpoints record `platform_boundary` and
  `platform_readiness`;
- OpenMM parity reports record `platform_evidence` and label OpenMM as
  reference-only validation;
- the Phase 3 GPCRmd 729 fixture probe records OpenMM reference evidence,
  MLX prep/load/readiness evidence, and a blocker matrix without turning OpenMM
  or `vendors/` into runtime dependencies;
- MD performance payloads record `platform_evidence` for finite-output proof
  cases;
- DFT/QM scope is reported by `get_dft_qm_scope_report()` and
  `dft_qm_scope_readiness_report()`.

The current large-fixture production-readiness probe is blocked at MLX runtime
topology/nonbonded provisioning, not at fixture selection or strict artifact
loading. It should be read as a bounded blocker report, not as a production MD
certification.
