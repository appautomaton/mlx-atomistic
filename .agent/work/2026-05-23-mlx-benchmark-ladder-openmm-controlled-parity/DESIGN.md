# DESIGN: MLX Benchmark Ladder With OpenMM Controlled Parity

## Boundary

- MLX benchmark product code stays under `src/mlx_atomistic/benchmarks/`.
- OpenMM reference code stays under `scripts/`.
- Tests may import or execute OpenMM reference scripts, but product runtime code must not import OpenMM.
- Raw measured JSON/CSV stays under gitignored `results/`.
- Committed interpretation and benchmark ladder docs stay under `docs/benchmarks/`.

## Benchmark Ladder Model

The ladder is a documentation and metadata layer over existing benchmark code.
It should classify rows by why they exist:

| Layer | Purpose | Example rows |
| --- | --- | --- |
| micro/kernel | isolate local costs | force terms, neighbor build, virtual-site reconstruction |
| controlled MD | runnable same-workload smoke | synthetic LJ full loop |
| feature physics | MLX feature timing and correctness-adjacent costs | GBSA/OBC, TIP4P-Ew, soft-core/lambda, replica exchange |
| scaling | opt-in size sensitivity | size sweeps, neighbor policy, cadence |
| reference parity | OpenMM/LAMMPS mappings where semantics line up | LJ, GBSA/OBC, TIP4P-Ew |
| stretch | real systems not yet product-ready | DHFR-style rows |

Every row should have:

- pair id or ladder id
- layer
- MLX command
- reference command or deferred reference
- metric family
- raw output path
- comparison status: `comparable`, `diagnostic`, `blocked`, or `deferred`
- decision value: what optimization choice the row can support
- caveat/blocker when not comparable

## OpenMM Controlled Cases

`scripts/benchmark_openmm_opencl.py` already has case routing for:

- `synthetic-lj-periodic`
- `gbsa-obc-small`
- `tip4p-ew-water`

The current GBSA/OBC and TIP4P-Ew cases are placeholder-blocked. This change
should replace placeholders where safe:

- GBSA/OBC: use a small no-cutoff OBC-style force evaluation compatible with
  the MLX synthetic GBSA/OBC row. If exact setup cannot be represented, return
  `diagnostic` or `blocked` with the reason.
- TIP4P-Ew: prefer a virtual-site reconstruction operation if it can match the
  MLX operation. If OpenMM naturally exposes only full water-system evaluation,
  label the row `diagnostic` and suppress ratios.

Invalid numeric inputs remain validation errors before OpenMM loading or
case-specific blocked handling.

## Comparison Semantics

`same_workload_compare.py` remains the comparison gate. It should continue to
compute ratios only when:

- MLX status is `ok`
- OpenMM status is `ok`
- timing metrics match
- atom counts match
- step counts match for throughput metrics
- operation semantics are not marked diagnostic

Blocked or diagnostic rows must keep reasons and produce no ratio.

## Content Rules

Channel: benchmark docs.

Source policy: repo-generated JSON, existing repo benchmark docs, and the
reference URLs captured in SPEC. No unsupported external performance claims.

Factual risk: high for performance claims. Every numeric claim in the refreshed
report must trace to a raw output path or an existing committed benchmark doc.

Format: concise Markdown with tables, commands, raw paths, row status, and
next-optimization guidance.
