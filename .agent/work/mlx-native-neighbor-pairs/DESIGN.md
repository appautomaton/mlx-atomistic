# DESIGN: MLX Native Neighbor Pairs

## Design Intent

Move the cutoff neighbor rebuild path from CPU-heavy NumPy cell-list construction toward an MLX/device-heavy backend while keeping the current `periodic_cell_list` implementation as the correctness oracle and explicit fallback.

The design must not revive dense all-pairs materialization for GPCRmd-scale lazy topologies. If MLX cannot compact a dynamic mask into a giant flat `(i, j)` array efficiently, the implementation should prefer a bounded tile/block representation and adapt force evaluation to that representation instead of forcing the old shape.

## Current Runtime Shape

- `src/mlx_atomistic/neighbors.py` owns `NeighborList` and `NeighborListManager`.
- `build_neighbor_list(...)` currently calls the CPU-side periodic cell-list builder and returns compact MLX `int32` pairs.
- `src/mlx_atomistic/forcefields.py` accepts explicit pairs for lazy topologies and refuses dense fallback when a runtime pair provider is required.
- `src/mlx_atomistic/md.py` reports the neighbor backend through `result.nonbonded_report`.
- `src/mlx_atomistic/prep/runner.py` creates the GPCRmd production neighbor manager with `periodic_cell_list`, `skin=2.5`, and `sort_pairs=False`.

## Smallest Correct Architecture

1. Keep `periodic_cell_list` as the oracle backend.
2. Add a backend contract around neighbor rebuilds before adding MLX-specific logic.
3. Implement an MLX-dominant candidate/tile builder that can return either:
   - compact unique pairs when MLX compaction is viable, or
   - bounded interaction tiles with candidate masks/counts when compact pair emission is not viable.
4. Route unsupported systems to an explicit fallback or blocker with metadata, not silent dense all-pairs evaluation.
5. Extend force evaluation only as much as needed to consume the chosen representation while preserving topology filtering and explicit pair overrides.
6. Record timing and count metadata that separates rebuild work from force evaluation.

## Backend Contract

The neighbor path should expose enough metadata for execution and reports:

- backend name, for example `periodic_cell_list` and the chosen MLX backend name;
- representation kind, for example `pairs` or `tiles`;
- pair or candidate tile count;
- cutoff and skin;
- rebuild count;
- rebuild timing;
- fallback reason or unsupported-backend blocker when applicable;
- estimated pair/tile memory.

The exact Python type can be chosen during execution, but it should keep the existing compact `NeighborList.pairs` path stable for current callers.

## Correctness Model

The CPU `periodic_cell_list` pair set remains the oracle for small periodic systems. Pair order is not part of the public contract.

Regression coverage must preserve:

- exclusions;
- explicit nonbonded exceptions;
- NBFIX pair and type-pair overrides;
- 1-4 LJ and Coulomb scaling;
- cutoff and skin behavior;
- periodic minimum image behavior;
- fail-closed behavior for large lazy topologies.

## Performance Model

The first performance bar is not absolute speed. It is proving that rebuild CPU work is reduced and separately measurable.

The GPCRmd 729 proof target is a 10-step short-range prototype runtime under 5 seconds on the current machine. If the target is missed, the verification artifact must identify the bottleneck after MLX neighbor work is enabled.

## Out Of Scope

- Production PME.
- Custom Metal kernels.
- Removing `periodic_cell_list`.
- Optimizing unrelated bonded, CMAP, constraints, notebook rendering, or trajectory streaming paths.
