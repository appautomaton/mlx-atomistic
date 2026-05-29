# DESIGN: DHFR Benchmark Artifact And Runtime Path

## Architecture Approach

Add DHFR as a named benchmark capability, not as a special case hidden inside the existing LJ/OpenMM script.

Primary surfaces:

- `src/mlx_atomistic/benchmarks/dhfr.py`: MLX-facing DHFR readiness and benchmark CLI.
- `scripts/benchmark_openmm_dhfr.py`: OpenMM reference CLI for DHFR rows.
- `src/mlx_atomistic/benchmarks/same_workload_compare.py`: comparison-pair registry and strict DHFR classification.
- `docs/benchmarks/`: human-readable ladder/report updates.
- `results/same-workload-openmm-comparison/`: gitignored raw MLX/OpenMM DHFR JSON outputs.

Use existing prep/runtime machinery first:

- AMBER import: `mlx_atomistic.prep.topology_import.import_amber_prmtop`.
- Artifact validation: `load_prepared_mlx_artifact`, `artifact_readiness_report`, and `build_mlx_system_from_artifact`.
- PME readiness: existing PME config/readiness helpers before any explicit PME runtime claim.
- Runtime execution: existing MLX NVT/NPT runner paths where feasible; otherwise emit normalized blockers.

## Input Strategy

Local DHFR-related assets already exist:

- OpenMM stock DHFR PDBs:
  - `vendors/openmm/examples/benchmarks/5dfr_minimized.pdb`
  - `vendors/openmm/examples/benchmarks/5dfr_solv-cube_equil.pdb`
- Amber20/JAC explicit PME inputs:
  - `results/inputs/Amber20_Benchmark_Suite/PME/Topologies/JAC.prmtop`
  - `results/inputs/Amber20_Benchmark_Suite/PME/Coordinates/JAC.inpcrd`
- Existing OpenMM raw context:
  - `results/openmm-opencl-dhfr-m5max.json`

Because `results/inputs/` is gitignored, every command must fail closed with a concrete input blocker when these local inputs are absent. No silent download is allowed.

## Pair Model

Use two DHFR pair IDs:

- `dhfr-implicit`: implicit-solvent DHFR, expected metric `ns_per_day` or a normalized equivalent derived from step timing.
- `dhfr-explicit-pme`: explicit PME DHFR/JAC, expected metric `ns_per_day` or a normalized equivalent derived from step timing.

The comparison layer may compute a ratio only when:

- both sides are `ok`,
- pair ID matches,
- atom counts match,
- solvent model matches,
- PME/implicit semantics match,
- timing metric matches,
- timestep and step-count semantics are compatible.

Otherwise the row remains `diagnostic` or `blocked`.

## Main Risks

- Implicit DHFR probably needs AMBER GBSA/OBC radii and scale extraction that may not yet be represented by `import_amber_prmtop`.
- DHFR-size implicit runtime may exceed eager all-pairs behavior and require lazy topology plus neighbor-provider support.
- Explicit PME DHFR/JAC is likely above the current PME production atom-count envelope and may initially produce a concrete readiness blocker.
- OpenMM stock PDB benchmark semantics and Amber20/JAC semantics are not automatically equivalent; the report must not merge them without evidence.
