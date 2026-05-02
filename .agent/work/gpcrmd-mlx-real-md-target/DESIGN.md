# DESIGN: GPCRmd-Backed MLX Real MD Target

## System Boundary

The workflow has three layers:

1. `atomistic_prep` handles GPCRmd dataset metadata, cache inspection, topology/parameter import attempts, and prepared-artifact export.
2. `mlx_atomistic` handles compatibility validation and simulation only.
3. `notebooks/ligand-receptor-motion/` visualizes and analyzes MLX-generated trajectories, or displays the exact capability blockers.

No external MD engine runs. GPCRmd data is reference input and validation context only.

## Data Flow

```text
GPCRmd target metadata
  -> local cache / manifest
  -> package inspection
  -> topology/parameter import attempt
  -> MLX compatibility report
  -> prepared_system.* if compatible
  -> MLX short NVT trajectory if runnable
  -> notebook visualization / analysis
```

## Target Selection Gate

A target candidate is acceptable only if it has:

- ligand-bound GPCR system metadata;
- downloadable coordinates and topology/parameter files;
- explicit water, ions, and box vectors;
- protocol/reference trajectory metadata;
- manageable size for initial local inspection.

If multiple targets qualify, prefer the smallest ligand-bound all-atom membrane system with the most complete downloadable topology package.

## Compatibility Report Contract

The report must distinguish:

- `supported_now`: terms and metadata MLX can consume;
- `missing_input`: absent files or missing topology/parameter fields;
- `unsupported_physics`: PME/Ewald, virtual sites, CMAP, NPT/barostat, unsupported water/lipid terms, or other required features;
- `runtime_risk`: atom count, pair count, expected memory, and short-run performance estimate;
- `next_engine_slice`: the smallest engine capability that would unblock the target.

Failing compatibility is a valid outcome if the blockers are exact and reproducible.

## Notebook Contract

The active notebook must never substitute GPCRmd reference frames for an MLX result.

- If `trajectory.npz` is MLX-generated and compatible, visualize and analyze it.
- If no MLX trajectory exists but the target is compatible, run a short MLX probe.
- If the target is not compatible, show the compatibility report and stop before MD visualization.

## Testing Shape

Network-heavy and large-file paths stay outside default tests. Default tests use tiny manifests, small cached fixtures, and mocked package indexes. Real GPCRmd download/inspection is an explicit CLI or skipped integration path.
