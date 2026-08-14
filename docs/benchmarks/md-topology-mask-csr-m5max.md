# Atom-Local Topology Mask Binding on Apple M5 Max

Date: 2026-08-14

## Decision

Retain atom-local compressed sparse row (CSR) topology lookup for Metal spatial
tiles. The change raises median complete-trajectory throughput by 7.96% on the
94,232-atom JAC workload, 7.41% on the 92,001-atom GPCRmd workload, and 6.01%
on the 23,558-atom 5DFR workload. It does not change neighbor membership, the
direct-force kernel, topology semantics, or the force-array layout.

## Root cause

Every spatial neighbor rebuild produces a new tile generation. The nonbonded
force binding then creates two tile-aligned bit masks:

- whether each active pair is eligible for Lennard-Jones work;
- whether an eligible pair uses the configured 1-4 Lennard-Jones scale.

The previous Metal kernel launched one lane for every position in every 4-by-4
tile. For each active pair it performed a binary search over the complete
sorted exclusion array and, when present, the complete 1-4 array. That meant
roughly 60 million active pair lookups per large-system rebuild against 138,836
JAC or 237,483 GPCRmd exclusion records.

Those records are globally large but atom-local sparse. When normalized by the
lower atom index, active left atoms have the following row degrees:

| Workload | Mean non-empty row | 90th percentile | Maximum |
| --- | ---: | ---: | ---: |
| JAC 4-cell | 2.12 | 3 | 16 |
| GPCRmd 729 | 3.34 | 9 | 19 |

The prepared nonbonded state now stores CSR offsets and right-atom arrays for
the same normalized exclusions and 1-4 pairs. A tile lane scans only
`offsets[left]:offsets[left+1]`. The output remains the same two packed masks,
so the recurring direct-force consumer and all correction terms are unchanged.

## Binding microbenchmark

Each row below built the same production spatial tiles once, then synchronized
five mask builds after one warmup. Times are medians.

| Workload | Active pairs | Tiles | Global search | Atom-local CSR | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| JAC 4-cell | 60,502,167 | 7,182,592 | 15.978 ms | 5.388 ms | 66.28% |
| GPCRmd 729 | 60,004,004 | 7,067,646 | 16.121 ms | 5.335 ms | 66.90% |

Raw samples are under `results/md-suite/tile-mask-{baseline,csr}-*.json`.

The synchronized whole-step profiler independently shows the expected route
movement:

| Workload | Previous binding wall | CSR binding wall | Bindings |
| --- | ---: | ---: | ---: |
| JAC 4-cell | 0.3624 s | 0.1311 s | 22 |
| GPCRmd 729 | 0.4473 s | 0.1610 s | 27 |

The counts are the initial generation plus measured rebuilds. The candidate
profile is `results/md-suite/csr-mask-profile.json`.

## Interleaved complete-trajectory A/B

Each process ran 10 warmup and 750 measured fixed-cell NVT steps with seed 17,
4 fs timestep, a 9 A cutoff, 5.5 A skin, Metal spatial tiles, and boundary-only
sampling and diagnostics. AC Low Power Mode was disabled. The order for each
workload was control 1, candidate 1, candidate 2, control 2, control 3,
candidate 3.

| Workload | Control median | CSR median | Step-wall reduction | Throughput increase |
| --- | ---: | ---: | ---: | ---: |
| 5DFR | 1.3112 ms | 1.2369 ms | 5.67% | 6.01% |
| JAC 4-cell | 4.0719 ms | 3.7718 ms | 7.37% | 7.96% |
| GPCRmd 729 | 4.8128 ms | 4.4809 ms | 6.89% | 7.41% |

All three paired directions improved on both large workloads. JAC candidate sample 1
performed one more measured rebuild than its control, and GPCRmd control sample
3 performed one more than its candidate. The median result is therefore not a
rebuild-count artifact. The smaller 5DFR validation had identical rebuild
counts in every run; its first pair moved by -0.98%, while the next two pairs
and the three-sample median improved. Every raw run passed finite-state,
constraint, memory, fixed-cell, lazy-topology, neighbor-representation, and
Particle Mesh Ewald (PME) plan-reuse checks. Raw outputs are under
`results/md-suite/csr-ab/`.

## Correctness and scope

The existing spatial-tile parity case covers exclusions, explicit exceptions,
non-unit 1-4 Lennard-Jones and Coulomb scales, periodic geometry, prepared
force-only execution, and compact-pair diagnostic parity. The full 30-case
Metal neighbor/nonbonded suite passes. CPU forcefield/runtime/benchmark tests
and production artifact/GPCRmd registry tests also pass.

CSR arrays are derived once from the same validated normalized pair sets used
by the prior kernel. CPU tile fallback, compact-pair execution, sparse PME
corrections, and NBFIX parameter lookup are unchanged. The retained performance
claim applies to generation binding for Metal spatial tiles; it does not claim
that neighbor construction or recurring Direct Space has been eliminated as a
bottleneck.
