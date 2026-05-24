# Same-Workload OpenMM Comparison

Date: 2026-05-23

Scope: refreshed controlled `mlx_atomistic` vs `openmm-reference` comparison.
This report only compares rows where workload and metric match. Rows marked
`blocked`, `diagnostic`, or `deferred` keep their raw evidence but do not get
performance ratios.

## Summary

| Pair id | Status | MLX result | OpenMM result | Ratio | Raw output |
| --- | --- | ---: | ---: | ---: | --- |
| `lj-synthetic-loop` | `comparable` | 37.8469 steps/s | 727.9128 steps/s | 19.2331 OpenMM/MLX throughput | `results/same-workload-openmm-comparison/summary.json`; `results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json`; `results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json` |
| `gbsa-obc-small` | `comparable` | 3.3352 ms/eval | 0.001458 ms/eval | 0.0004372 OpenMM/MLX latency | `results/same-workload-openmm-comparison/summary.json`; `results/same-workload-openmm-comparison/mlx-phase3-controlled.json`; `results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json` |
| `tip4p-ew-water` | `comparable` | 6.0068 ms/eval | 0.0003339 ms/eval | 0.00005558 OpenMM/MLX latency | `results/same-workload-openmm-comparison/summary.json`; `results/same-workload-openmm-comparison/mlx-phase3-controlled.json`; `results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json` |
| `dhfr-implicit` | `comparable` | 0.3095 ns/day | 1.3136 ns/day | 4.2447 OpenMM/MLX throughput | `results/same-workload-openmm-comparison/summary.json`; `results/same-workload-openmm-comparison/mlx-dhfr-implicit.json`; `results/same-workload-openmm-comparison/openmm-dhfr-implicit.json`; [`same-workload-dhfr-stretch.md`](./same-workload-dhfr-stretch.md) |
| `dhfr-explicit-pme` | `blocked` | blocked: PME artifact must be neutral (`net_charge=-11`) | OpenMM Reference one-step row runs; OpenMM OpenCL context 752.5 ns/day | none | `results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json`; `results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json`; [`same-workload-dhfr-stretch.md`](./same-workload-dhfr-stretch.md) |

## Interpretation

`lj-synthetic-loop` is a tiny controlled MD row: one synthetic LJ step, 32
atoms, matching atom count, matching step count, and `steps/s` on both sides.
In that narrow row, the OpenMM OpenCL throughput is 19.2331 times the MLX
throughput.

That is not a production-MD conclusion. It says the current tiny MLX full-loop
path has lower throughput than OpenMM on this controlled smoke case. It does
not say how MLX compares on DHFR, ApoA1, PME, or larger systems.

`gbsa-obc-small` is now a controlled latency row. The MLX row is the
`obc_pair_accumulation_and_force` case from
`results/same-workload-openmm-comparison/mlx-phase3-controlled.json`. The
summary keeps the logical pair id `gbsa-obc-small`; the per-pair
`results/same-workload-openmm-comparison/mlx-gbsa-obc-small.json` file is a row
extract that points back to the combined phase3 source. The OpenMM Reference row
reports `status: "ok"`,
`fixture: "gbsa_obc_small"`, `ms_per_eval`, and an `obc_force_setup` payload in
`results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json`. Because
this is a latency metric, the lower OpenMM/MLX ratio means the OpenMM Reference
latency is lower for this controlled OBC operation.

`tip4p-ew-water` is also a controlled latency row. The MLX row is the
`m_site_reconstruction` case from
`results/same-workload-openmm-comparison/mlx-phase3-controlled.json`. The
summary keeps the logical pair id `tip4p-ew-water`; the per-pair
`results/same-workload-openmm-comparison/mlx-tip4p-ew-water.json` file is a row
extract that points back to the combined phase3 source. The OpenMM Reference row
reports `status: "ok"`,
`fixture: "tip4p_ew_water"`, `operation_semantics:
"virtual_site_reconstruction"`, and `openmm_operation:
"Context.computeVirtualSites"` in
`results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json`. Because
this is a latency metric, the lower OpenMM/MLX ratio means the OpenMM Reference
latency is lower for this controlled virtual-site reconstruction operation.

`dhfr-implicit` is now a real one-step same-workload smoke row. The MLX side
loads an OpenMM-derived DHFR GBSA/OBC artifact, builds MLX force terms, and runs
one bounded NVT step at `0.004 ps`. The OpenMM Reference side runs the matching
one-step implicit GBSA/OBC row. Because both rows are `ok`, use the ratio as a
narrow reference comparison for this smoke workload only.

`dhfr-explicit-pme` remains blocked on the MLX side for a concrete artifact
reason: the Amber20/JAC PME artifact is charged (`net_charge=-11`) and the
current PME artifact policy requires neutral systems. The old AMBER 10-12
blocker is no longer the active blocker for this local input. The comparison
helper correctly suppresses the explicit PME ratio while keeping the OpenMM
Reference raw row.

No row in this report is currently labeled `diagnostic` or `deferred`. The
benchmark ladder still uses those labels for rows whose operation semantics,
metric family, or reference mapping are not yet controlled enough for a ratio.

## Reproducer

Raw controlled outputs:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.md_performance --sizes 32 --steps 1 --sample-interval 1 --diagnostic-interval 1 --evaluation-interval 1 --json > results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --platform OpenCL --particles 32 --steps 1 --warmup-steps 0 --spacing-nm 1.0 --json > results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.phase3_physics --evaluations 1 --waters 1 --atoms 4 --replica-steps 1 --json > results/same-workload-openmm-comparison/mlx-phase3-controlled.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --case gbsa-obc-small --platform Reference --particles 4 --steps 1 --json > results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_opencl.py --case tip4p-ew-water --platform Reference --particles 4 --steps 1 --json > results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-implicit --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.dhfr --case dhfr-explicit-pme --steps 1 --json > results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-implicit --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-implicit.json
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python scripts/benchmark_openmm_dhfr.py --case dhfr-explicit-pme --platform Reference --steps 1 --json > results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json
```

Comparison summary:

```bash
UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run python -m mlx_atomistic.benchmarks.same_workload_compare --mlx-json results/same-workload-openmm-comparison/mlx-lj-synthetic-loop.json --mlx-json results/same-workload-openmm-comparison/mlx-phase3-controlled.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-implicit.json --mlx-json results/same-workload-openmm-comparison/mlx-dhfr-explicit-pme.json --openmm-json results/same-workload-openmm-comparison/openmm-lj-synthetic-loop.json --openmm-json results/same-workload-openmm-comparison/openmm-gbsa-obc-small.json --openmm-json results/same-workload-openmm-comparison/openmm-tip4p-ew-water.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-implicit.json --openmm-json results/same-workload-openmm-comparison/openmm-dhfr-explicit-pme.json --out results/same-workload-openmm-comparison/summary.json
```

These commands require local MLX/Metal access for MLX rows and OpenMM/OpenCL or
OpenMM Reference availability for OpenMM rows. A sandbox atexit warning from
MLX/Metal cleanup after the summary command is not a benchmark failure when the
summary JSON is written with `status: "ok"`.

## Next Optimization Target

The next optimization target should be MLX TIP4P-Ew virtual-site
reconstruction, followed by GBSA/OBC force evaluation if the TIP4P path does
not explain the shared latency overhead. This is based on measured comparable
rows only: `tip4p-ew-water` has the largest MLX latency in the refreshed
controlled summary, and both `tip4p-ew-water` and `gbsa-obc-small` use
controlled OpenMM Reference latency rows.

The `lj-synthetic-loop` result also shows a measured tiny full-loop throughput
gap, so force evaluation, reporting cadence, and synchronization remain valid
diagnostic follow-ups for that row. `dhfr-implicit` is now runnable, but its
one-step row should guide runtime-path hardening before it guides broad
performance optimization claims. `dhfr-explicit-pme` still needs a neutral PME
artifact policy before it can guide optimization.
