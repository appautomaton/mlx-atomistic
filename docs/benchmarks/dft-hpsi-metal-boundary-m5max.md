# DFT Hpsi Metal boundary verdict on Apple M5 Max

Date: 2026-08-10

This record closes the private compact-Hamiltonian boundary experiment from
`2026-08-10-fuse-dft-hamiltonian-boundary`. The candidate fused compact scatter
and post-FFT gather/combine in two private Metal kernels while leaving the MLX
FFT and matrix multiplication paths intact. It was measured and then removed;
the product and test surfaces in final source commit
`6cf8d618b68b66f411e857f28318b4eb7ead4fe8` exactly match the recorded
pre-candidate inventory.

## Control and host

The forced MLX and forced Metal runs used candidate commit
`6d6af77174ce1a8ea775ea65c7f10dc3a87dc27e`, workload fingerprint
`d4cbb4abe682895be010362078629354201570377d0ae3c3194ced39d7ad4426`, and
runtime fingerprint
`8cc06604b28a0b66fe91273c6b4e1ad03eb4ec47a28ce976b59fbab88af6aa42`.
Both were fresh, formally admitted, and numerically valid. The host was an
Apple M5 Max MacBook Pro (`Mac17,7`) with 128 GB unified memory, macOS 26.5.2
build 25F84, MLX 0.31.2, Battery Power, and Low Power Mode active.

## Fixed-density result

After one warmup, each route produced five fresh samples. The MLX median was
0.2700237921 seconds and the Metal median was 0.2309044579 seconds, a 14.4874%
complete-wall improvement. Retention required an improvement strictly greater
than the 14.7222% control dispersion, so the timing gate failed by about 0.235
percentage points.

The candidate did pass its other fixed-density gates:

- Both private kernels executed 216 times. The targeted boundary used zero
  counted intermediate arrays versus 648 on the MLX route.
- Hpsi peak temporary memory was unchanged at 56,061,289 bytes.
- Process high water was 213,680,128 bytes for Metal versus 216,170,496 bytes
  for MLX.
- Bounded process-tree peak was 1,018,430,448 bytes for Metal versus
  1,052,279,816 bytes for MLX.
- Unified-memory high water was 353,346,655 bytes for Metal versus 402,547,791
  bytes for MLX.

Because the first mandatory performance gate failed, the protocol did not run
full self-consistent field (SCF), Carbon, or MgO comparisons. Their statuses are
`not-run-after-prior-failure`; this is a declared short-circuit, not missing
positive evidence.

## Source decision and next target

The result is `removed`. No candidate kernel, route selector, observer counter,
benchmark flag, or candidate test remains in the product tree. Hpsi remains the
largest measured density-functional theory (DFT) phase in the current control,
but this narrow scatter/gather fusion did not clear complete-wall noise. The
next DFT optimization should start with a fresh profile and target a larger
attributable part of Hpsi, such as its FFT boundary, before considering the
deferred orthogonalization and projected-eigensolve work.

Generated evidence is local and gitignored under
`results/dft-hpsi-metal-boundary/2026-08-10/`:

- MLX and Metal reports: `fixed-mlx/` and `fixed-metal/`
- Bounded memory traces: `fixed-mlx-memory.json` and `fixed-metal-memory.json`
- Final helper-bound comparison: `fixed-comparison-v2.json`
- Candidate decision: `candidate-decision-v2.json`
- Exact-removal inventory: `final-removal-inventory.json`
- Final decision: `decision.json`

The exact machine-readable decision summary is:

```json
{"candidate_commit":"6d6af77174ce1a8ea775ea65c7f10dc3a87dc27e","carbon_gate":"not-run-after-prior-failure","decision":"removed","final_commit":"6cf8d618b68b66f411e857f28318b4eb7ead4fe8","fixed_gate":"failed","full_gate":"not-run-after-prior-failure","mgo_gate":"not-run-after-prior-failure"}
```

This report characterizes only the stated repository commit, Silicon workload,
host, and power state. It makes no cross-engine or hardware-general performance
claim.

## Follow-up three-dimensional boundary experiment

Date: 2026-08-11

A benchmark-only stage profiler was added after the first candidate was
removed. On the same Apple M5 Max with 128 GB unified memory, macOS 26.5.2,
MLX 0.31.2, AC Power, and Normal Power Mode, the 64-vector control attributed
72.60% of independently synchronized Hpsi time to the local FFT path, 23.30%
to compact scatter, and 5.60% to GTH. These independently measured medians are
diagnostic and are not additive. In particular, the earlier 529 GB GTH value
was logical algorithmic traffic rather than observed device traffic and did
not identify the measured bottleneck.

A second private candidate then replaced the one-dimensional boundary dispatch
with a three-dimensional `(slot, vector, lane)` Metal grid and added an
unmasked fast path for the zero-padding fixed-density case. It fused compact
scatter and post-FFT gather, kinetic, and nonlocal combination while continuing
to use the MLX FFT. No C++ extension or additional package was introduced.

The synchronized 64-vector A-B-B-A microbenchmark averaged 7.842229 ms for the
MLX Hpsi boundary and 6.482854 ms for the forced Metal boundary, a 17.33%
elapsed-time reduction. Scatter alone averaged 1.856208 ms versus 0.375188 ms.
The complete fixed-density A-B-B-A gate was smaller: the two MLX medians
averaged 0.164337833 seconds and the two Metal medians averaged 0.153821958
seconds, a 6.3989% complete-wall reduction. All samples converged in 34
iterations with five restarts, the representative eigenvalues were identical,
and every forced-route Hpsi call was accounted for.

The frozen retention gate still required a complete-wall reduction strictly
greater than 14.7222%. The second candidate therefore also has decision
`removed`: no Metal kernel, selector, runtime counter, or forced route remains
in the product path. The stage profiler and its atomic capture support remain
as diagnostic infrastructure. The result says that custom Metal can accelerate
this boundary, but this boundary alone is not large enough to justify a default
runtime fork.

The control profiler and candidate evidence are local and gitignored under:

- `results/dft-hpsi-stage-profile/2026-08-11/control/`
- `results/dft-hpsi-boundary-abba/2026-08-11/`
- `results/dft-hpsi-fixed-abba/2026-08-11/`

The retained control profiler can be reproduced with:

```console
uv run python -m mlx_atomistic.benchmarks.dft_hpsi_profile \
  --manifest results/dft-hpsi-metal-boundary/2026-08-10/silicon-workload/manifest.json \
  --gth-source results/dft-hpsi-metal-boundary/2026-08-10/silicon-workload/resources/Si-GTH-PBE-q4.gth \
  --out results/dft-hpsi-stage-profile/control \
  --warmups 3 --samples 7 --json
```

### Local FFT decomposition

The retained profiler was then extended to materialize stable inputs and time
the inverse FFT, potential multiply, forward FFT, and compact gather
independently. Its artifact schema is
`mlx-atomistic.dft-hpsi-stage-profile.v3`. A stable 64-vector diagnostic control
reported 7.926791 ms for Hpsi and 5.750958 ms for the complete local FFT path.
The substage medians were 2.739459 ms for the inverse FFT, 0.538875 ms for the
potential multiply, 2.733708 ms for the forward FFT, and 0.605333 ms for
gather. The two FFT medians sum to 95.17% of the complete local FFT median.

The multiply and gather together account for only 14.43% of Hpsi before any
complete-wall dilution. Even an impossible zero-cost replacement would
therefore provide too little headroom for another narrow custom Metal kernel to
clear the complete fixed-density retention gate. The next optimization target
is not C++ or another boundary kernel. It is the amount of work submitted to
the existing MLX FFT: padded vector capacity, shape scheduling, and avoidable
FFT vector equivalents.

The stable decomposition artifact is local and gitignored at
`results/dft-hpsi-local-fft-profile/2026-08-11/control-v3-ac2/`. An earlier
control in the same directory was power-state unstable and is not used for the
decision.
