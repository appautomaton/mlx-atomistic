# Packed spatial-column descriptor verdict on Apple M5 Max

Date: 2026-08-13

## Decision

Retain the packed spatial-column descriptor. The existing `int32` force-column
array now carries the four membership bits for its right-atom column in the
high nibble. The recurring Direct Space Metal kernel reads those bits from the
already-sequential descriptor instead of indirectly loading the tile-wide
membership word.

The change adds no array, dispatch, package, or runtime dependency. It preserves
the retained 4-by-4 execution tiles and 32-lane Single Instruction Multiple
Data (SIMD) force groups. OpenMM and LAMMPS remain reference surfaces and do not
enter the MLX runtime path.

## Why this candidate was selected

Fresh production inventories measured the member count of every right-atom
column:

| Workload | Active columns | 1-2 member columns | Pairs in 1-2 member columns | Full 4-member columns | Pairs in full columns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | 5,068,119 | 36.99% | 18.25% | 48.09% | 66.32% |
| JAC, 94,232 atoms | 20,932,623 | 37.32% | 18.49% | 47.53% | 65.78% |

This rejected a separate sparse-pair kernel before implementation. Sparse
columns contain too little of the expensive pair arithmetic to justify another
dispatch and a second force-accumulation policy. Splitting full and partial
columns would also add another schedule and dispatch to remove only a cheap
membership branch.

Packing membership into the descriptor instead covers every active column. It
removes one indirect membership load per column and simplifies the four inner
membership tests while leaving arithmetic and force ownership unchanged.

## Descriptor contract

The low 28 bits retain the original compact index and the high four bits hold
the column-local membership mask:

```text
bits 31..28  four left-slot membership bits
bits 27..2   tile index
bits  1..0   right-column index
```

The descriptor therefore supports up to 67,108,864 retained 4-by-4 tiles. The
builder raises an explicit error before packing a larger inventory rather than
silently overlapping membership and index bits. The 5DFR and JAC inventories
use 1,731,081 and 7,182,592 tiles, respectively.

## Correctness and isolated kernel evidence

The Metal parity test reconstructs every packed membership nibble from the
tile membership word and compares the tile force with the compact-pair route.
The production profiles also passed exact-pair inventory, force, correction,
memory, and backend gates.

Two independent profile processes per arm produced these medians. Each process
internally interleaved six compact-pair and six tile-force samples after two
warmups.

| Workload | Control Direct Space | Packed descriptor | Result |
| --- | ---: | ---: | ---: |
| 5DFR | 0.557854 ms | 0.546625 ms | 2.01% faster |
| JAC | 1.663344 ms | 1.639885 ms | 1.41% faster |

The JAC tile rebuild median moved from 60.156 ms to 61.641 ms, a 2.47% increase
from packing four bits per active column. Rebuilds are amortized across the
Verlet-list lifetime; complete production wall time is the retention gate.

## Complete-wall evidence

All complete runs used independent processes, a 0.004 ps timestep, 300 K,
seed 17, a 5.5 Angstrom neighbor skin, one-step neighbor checks, and the
`mlx_cell_tiles` backend. Samples were position-balanced between the control
at commit `f0f818c` and the candidate working tree.

Two 75-step 5DFR samples per arm had two rebuilds each. The control median was
100.397 ms and the candidate median was 98.844 ms, a 1.55% reduction.

The short JAC samples entered distinct host performance states, so their 1.88%
candidate median reduction was treated only as supporting evidence. Four
750-step samples per arm provided the decision surface:

| Workload | Control median | Packed descriptor median | Result |
| --- | ---: | ---: | ---: |
| JAC, 750 steps | 6.048536 s | 5.972529 s | 1.26% faster |

Both arms had the same rebuild distribution: three samples rebuilt 22 times
and one rebuilt 21 times. Restricting the comparison to the three 22-rebuild
samples per arm still favored the candidate by 1.15% (`6.057204 s` versus
`5.987457 s`). Every sample passed the runtime checks. Peak Metal memory stayed
near 1.141 GB in both arms, as expected from reusing the existing descriptor
array.

Generated reports remain gitignored under `results/packed-column-ab/`.

## Boundary

This result does not reopen the rejected 32-atom interaction engine or justify
a C++ MLX extension. It is a measured improvement to the retained project-owned
Metal route. Future Direct Space work should begin from a fresh synchronized
profile and must not infer that the remaining inactive tile lanes are equally
expensive; inactive membership branches do not execute the screened-Coulomb
arithmetic.
