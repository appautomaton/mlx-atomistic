# Dual-runtime baselines on Apple M5 Max

This page is the durable control record for maintainers optimizing the existing MLX/Metal DFT and molecular-dynamics runtimes. It reports only evidence generated from the current repository and the local artifacts named below. No timing result is admitted until Slice 2 validates the complete suite.

## Control contract

- One clean Git commit must identify every accepted timing artifact.
- Metal runs execute serially in separate bounded processes.
- Clean product wall time stays separate from synchronized profiling.
- A blocked provenance, numerical, route, power, timeout, memory, or integrity gate remains blocked.

The exact control commit and runtime fingerprints are pending until Slice 2.

## Host and fixture admission

The required host is Apple M5 Max with Low Power Mode active. Each timing artifact must record macOS, MLX, power source, thermal pressure when available, and process-tree peak memory. The molecular-dynamics fixture is the prepared 23,558-atom 5DFR system under `results/dhfr-npt-closure/prepared`.

Admission details are pending until Slice 2.

## Raw artifact namespace

Generated evidence lives under `results/dual-runtime-baselines/2026-08-10/`. The directory is gitignored and is not a package input or runtime dependency.

## DFT fixed-density

Pending until Slice 2: one warmup and five fresh measured samples with numerical and provenance admission.

## DFT full SCF

Pending until Slice 2: one process-cold, non-resumed calculation with a 300-second bound.

## MD clean 5DFR

Pending until Slice 2: two separate-process controls with 10 warmups and 75 measured steps.

## MD structural profile

Pending until Slice 2: one synchronized route profile and one separate graph capture. These runs will not be reported as clean product wall time.

## Bottleneck decision

Pending until Slice 2. The decision will report whether DFT `Hpsi` and MD neighbor rebuild or synchronization remain the first measured targets.

<!-- dual-runtime-summary:start -->
```json
{
  "schema_version": "mlx_atomistic.dual_runtime_summary.v1",
  "status": "pending until Slice 2"
}
```
<!-- dual-runtime-summary:end -->
