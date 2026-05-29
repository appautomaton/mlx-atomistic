# DESIGN: Same-Workload Benchmark Comparison

## Architecture Approach

The comparison layer keeps three boundaries separate:

1. MLX product benchmark execution remains in `src/mlx_atomistic/benchmarks/`.
2. OpenMM reference execution remains in `scripts/` and may fail soft with normalized `blocked` payloads.
3. Committed interpretation remains in `docs/benchmarks/`, with raw JSON/CSV under gitignored `results/`.

This preserves the existing runtime boundary: `mlx_atomistic` does not import OpenMM or LAMMPS.

## Comparison Shape

Each comparison pair should have a small metadata record:

- pair id
- workload name
- MLX command and raw output path
- OpenMM command and raw output path
- physics assumptions
- metric family
- comparable status: `comparable`, `diagnostic`, or `blocked`
- blocker or caveat when not comparable

At least one controlled pair must produce a measured ratio. Other pairs may be `blocked` or `diagnostic` only when the reason is concrete.

## Must-Ship Pairs

Controlled pairs are the must-ship path:

| Pair | Required state |
| --- | --- |
| Synthetic LJ full-loop/nonbonded | measured or blocked with command evidence |
| GBSA/OBC small system | measured or blocked with command evidence |
| TIP4P-Ew water/virtual-site | measured or blocked with command evidence |

The real-system DHFR-style row is a stretch target. It may ship as `blocked` if the MLX side cannot yet load or run matching physics.

## Reporting Rules

The report must not rank MLX against OpenMM unless a pair uses the same workload and metric family. OpenMM DHFR/ApoA1/STMV reference numbers may appear only as context unless a matching MLX row exists.

## Verification Boundary

MLX and OpenMM GPU execution may need unsandboxed Metal/OpenCL access. Sandbox-specific failures must be recorded as environment blockers, not silently converted into performance claims.
