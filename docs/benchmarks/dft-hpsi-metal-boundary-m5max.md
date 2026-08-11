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
