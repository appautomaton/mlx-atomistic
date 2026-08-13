# Device-Resident 32-Atom Metal Interaction Engine

Status: research complete; bounded prototype approved; production integration
not yet approved.

## Decision

Build the engine in two gates.

1. Implement a fixed-coordinate, fixed-cell `32 x 32` force prototype with
   `mx.fast.metal_kernel`.  Include atom packing and force scattering in every
   performance result.  This gate requires no new package and answers the most
   important question first: does the interaction algorithm beat the retained
   `4 x 4` production route on Apple Silicon?
2. Only after that gate passes, implement a small MLX C++ `Primitive` that owns
   the persistent interaction state and encodes the rebuild and force kernels
   on the MLX Metal stream.  The native extension is not justified before the
   force algorithm has measured headroom.

This is not a port of OpenMM.  OpenMM supplies validated algorithmic ideas.  The
product remains the MLX/Metal runtime, with canonical MLX positions and forces
at the public boundary.

## Why this is a structural experiment

The retained direct kernel is already handwritten Metal.  It assigns one
32-lane Single Instruction Multiple Data (SIMD) group to as many as 32 active
right-atom columns that share a four-atom left block.  Another local rewrite of
that kernel cannot remove the current neighbor representation, its rebuild
transactions, or its small execution blocks.

The proposed engine changes all three together:

- atoms are packed into spatially coherent 32-atom blocks;
- the neighbor list stores a left block plus 32 packed right atoms;
- one SIMD group evaluates the resulting `32 x 32` interaction block;
- topology-bearing block pairs use a separate, small special-tile route;
- rebuild decisions and fixed-capacity state remain on the device.

That scope satisfies the reopening condition recorded in `docs/md-acceleration.md`:
it is spatially native, occupancy-bounded, and quantifies scheduled lanes before
implementation.

## Evidence from the current runtime

The current production path has these boundaries:

- `NeighborTiles` in `src/mlx_atomistic/neighbors.py` stores exact Verlet
  membership in four-atom execution blocks.
- `_build_mlx_spatial_cell_tile_list()` uses an eight-atom search block and
  emits four-atom execution tiles.  Several prefix-tail evaluations size later
  allocations on the host.
- `_tile_parameterized_pme_direct_force_only()` in
  `src/mlx_atomistic/metal_kernels.py` evaluates the direct Lennard-Jones and
  screened-Coulomb terms in one custom Metal kernel.
- `NeighborListManager._needs_rebuild_mlx_scalar()` materializes a device
  displacement result on the host every step.  The inclusive timer is mostly
  upstream GPU completion on JAC, so a smaller displacement kernel alone is
  not the target.

The retained profiles place direct force and amortized rebuild at approximately
28% and 24% of a 5DFR step, respectively.  Those numbers are large enough to
justify a design that changes both boundaries, but not a native-extension
rewrite whose force algorithm has not first won a microbenchmark.

## What OpenMM actually does

The relevant OpenMM OpenCL sources are local references under `vendors/openmm`:

- `platforms/opencl/src/OpenCLContext.cpp` fixes the OpenCL tile size at 32.
- `platforms/opencl/src/kernels/findInteractingBlocks.cl` lets one warp own a
  left block, rejects right blocks with sphere and axis-aligned bounding-box
  tests, keeps right atoms that are close to any left atom, and packs those atom
  identifiers in groups of 32.
- `platforms/opencl/src/kernels/nonbonded.cl` rotates 32 right atoms across 32
  left lanes and accumulates one left and one right force per lane.
- `platforms/common/src/ComputeContext.cpp` periodically reorders the full
  engine state.  That reordering is an important source of spatial locality.
- block pairs containing exclusions are removed from the ordinary neighbor
  list and evaluated by a separate exclusion-tile path.

Therefore the useful reference is not merely a `32 x 32` loop.  Its required
invariants are spatially coherent blocks, a separate topology route, persistent
capacity buffers, and a device rebuild flag.

OpenMM's exact implementation is not copied for three reasons:

1. Reordering the complete MLX state would disturb bonded, reciprocal Particle
   Mesh Ewald (PME), constraint, and integration graphs.  The first engine will
   instead pack a private direct-force view and scatter its result back to
   canonical atom order.
2. The MLX runtime already represents Coulomb exclusions and exceptions as a
   sparse correction after the ordinary direct term.  The new ordinary and
   special kernels preserve that convention.  Only Lennard-Jones enable and
   1-4 masks belong in special interaction tiles.
3. OpenMM uses fixed-point force buffers.  The retained MLX kernel already has
   validated float32 atomics on the target hardware, so fixed-point conversion
   is not part of the first experiment.

## Measured 32-atom inventory

`scripts/analyze_interaction_blocks.py` inventories the proposed schedule on
the real prepared coordinates.  It counts geometric pairs with a periodic
`cKDTree`, forms 32-atom blocks, maps topology exceptions to special block
pairs, applies block bounds, and packs ordinary right atoms exactly as the
proposed builder would.

Reproduce the measurements with:

```bash
uv run python scripts/analyze_interaction_blocks.py \
  results/larger-system-scaling/modern-jac-base/dhfr-explicit-pme/prepared_system.npz

uv run python scripts/analyze_interaction_blocks.py \
  results/scalable-charged-pme-runtime/jac-2x2x1/prepared/prepared_system.npz
```

The 9 Angstrom cutoff and 5.5 Angstrom skin produce the following result.  State
sizes include a 25% ordinary-tile capacity reserve and exclude transient packed
positions and force outputs.

| Workload | Ordering | Total 32-atom tiles | Scheduled pair lanes | Search-pair occupancy | Estimated persistent state |
| --- | --- | ---: | ---: | ---: | ---: |
| 5DFR, 23,558 atoms | canonical | 116,025 | 118.81 M | 12.73% | 18.94 MiB |
| 5DFR, 23,558 atoms | cell | 32,898 | 33.69 M | 44.90% | 6.08 MiB |
| 5DFR, 23,558 atoms | Morton | 33,204 | 34.00 M | 44.49% | 6.15 MiB |
| JAC, 94,232 atoms | canonical | 568,243 | 581.88 M | 10.40% | 92.14 MiB |
| JAC, 94,232 atoms | cell | 130,865 | 134.01 M | 45.15% | 24.23 MiB |
| JAC, 94,232 atoms | Morton | 132,802 | 135.99 M | 44.49% | 24.63 MiB |

The decision is unambiguous:

- canonical order is invalid for this design;
- the existing cell ordering is slightly better than Morton ordering and is
  much simpler to build;
- the maximum ordinary inventory is 80 tiles per left block on JAC and 78 on
  5DFR for the measured coordinates;
- JAC's 130,865 total tiles contain 14,714 special topology tiles;
- the estimated JAC schedule state is about 24 MiB, compared with about
  233 MiB of tile geometry plus topology masks in the retained JAC profiler;
- the new schedule executes about 16.6% more cheap pair lanes than the retained
  JAC active-column inventory (`134.01 M` versus `114.92 M`).

The last point is the central risk.  A 32-atom engine cannot win by reducing the
number of distance checks.  It must win by reusing packed atom data, replacing
millions of small tile descriptors, reducing global loads and atomics per
geometric pair, and removing rebuild allocation/readback transactions.

## Data model

The engine state is a functional token of MLX arrays.  It is consumed by one
step and replaced by the next state, which preserves MLX graph semantics while
allowing a C++ primitive to donate a uniquely owned backing buffer.

```text
InteractionState32
  atom_order                 int32 [padded_atoms]
  inverse_atom_order         int32 [atoms]
  block_center_radius        float32 [blocks, 4]
  block_half_extent          float32 [blocks, 4]
  ordinary_left_block        int32 [ordinary_capacity]
  ordinary_right_atoms       int32 [ordinary_capacity, 32]
  ordinary_count             uint32 [1]
  special_block_pair         int32 [special_count, 2]
  special_lj_enabled         uint32 [special_count, 32]
  special_lj_one_four        uint32 [special_count, 32]
  old_positions              float32 [atoms, 4]
  old_box                     float32 [6]
  rebuild_flag               uint32 [1]
  overflow_flag              uint32 [1]
  generation                 uint32 [1]
```

`ordinary_right_atoms[tile, lane]` is either a canonical atom identifier or a
sentinel equal to `atom_count`.  The left block identifies 32 atoms through
`atom_order`.  Ordinary tiles contain no topology masks.

Every diagonal block and every block pair containing at least one excluded,
exception, or 1-4 Lennard-Jones pair is a special tile.  Its two 32-word masks
describe the 32 left interactions for each right atom.  Coulomb remains enabled
for all valid geometric pairs because the existing sparse PME correction route
removes exclusions and installs exception values afterward.

## Atom ordering and packed views

The first implementation keeps canonical runtime state.  It does not reorder
the `System`, integrator, PME grid, constraints, or bonded parameter arrays.

At initialization or an explicit order refresh:

1. wrap positions into the orthorhombic cell;
2. calculate the same fine cell identifiers used by the retained spatial
   builder;
3. stable-sort canonical atom identifiers by cell identifier;
4. group consecutive identifiers into 32-atom blocks;
5. build the inverse order and topology-bearing special tiles.

At every force evaluation, a linear kernel packs canonical positions and
static charge/Lennard-Jones parameters into aligned block-order records.  A
linear unique-writer kernel scatters the accumulated direct force back to
canonical order.  Both costs must be included in every force benchmark.

The ordering is not refreshed at every neighbor rebuild.  The bounded prototype
keeps it fixed for a trajectory sample and records occupancy drift.  A retained
engine may refresh it on a long interval, initially 250 steps to match the
amortization scale used by OpenMM, but only if a measured quality metric shows
that the refresh pays for itself.

## Kernel family

### 1. `pack_atom_records_32`

Gather canonical `positions`, `charges`, `half_sigma`, and `sqrt_epsilon` into
block-order float records.  Static parameters are repacked only when atom order
changes; positions are packed each force evaluation.

### 2. `check_rebuild_and_bounds_32`

One thread per atom calculates minimum-image displacement from `old_positions`.
SIMD and threadgroup reductions set `rebuild_flag` when the maximum squared
displacement exceeds one quarter of the squared skin.  The same dispatch writes
32-atom block centers, radii, and half extents only when rebuilding.

The prototype is fixed-cell orthorhombic NVT.  A later NPT version must also
rebuild when the cell changes and must validate the displacement certificate in
fractional coordinates.

### 3. `find_interaction_blocks_32`

One SIMD group owns one left block.  Thirty-two lanes compare 32 right-block
bounds in parallel.  For an admitted ordinary block, every lane loads one right
atom and checks it against all 32 left atoms.  Right atoms within the search
radius of any left atom are compacted into a small threadgroup buffer and
flushed in groups of 32 through one global atomic capacity reservation.

The kernel returns immediately when `rebuild_flag` is zero.  It never sizes a
later allocation.  `ordinary_capacity` is fixed for the state generation.

### 4. `compute_ordinary_interactions_32`

One SIMD group owns one ordinary tile.  Lane `l` owns left atom `l` and the
right atom stored in slot `l`.  During rotation `k`, it evaluates left atom `l`
against right slot `(l+k) mod 32`.

The force on the left atom remains in a register.  The contribution for a
lane's original right atom is returned with a SIMD shuffle from source lane
`(l-k) mod 32`.  After 32 rotations each lane performs one three-component
atomic addition for its left atom and one for its original right atom.

Conceptually:

```text
for k in 0..31:
    right = simd_shuffle(owned_right, (lane + k) & 31)
    pair_force = direct_pair(left, right)
    left_force += pair_force
    right_force -= simd_shuffle(pair_force, (lane - k) & 31)
```

The actual Metal source must guard invalid atoms, zero distance, the force
cutoff, and a runtime `threads_per_simdgroup == 32` certificate.

### 5. `compute_special_interactions_32`

This uses the same rotation and accumulation pattern but reads one
Lennard-Jones enable word and one 1-4 word per right lane.  Diagonal tiles also
mask self pairs and one triangle.  Keeping this route separate prevents mask
bandwidth and branches from entering ordinary tiles.

### 6. `scatter_ordered_force_32`

One thread per valid ordered atom writes its direct force to the unique
canonical atom slot.  This scatter needs no atomic operation.

## Device residence and the MLX boundary

MLX custom Metal kernels are sufficient for the first force gate.  Their Python
API allocates outputs from host-provided shapes, so it cannot by itself preserve
a fixed-capacity output when a device rebuild branch returns early.

MLX's supported extension API provides the required second boundary:

- a custom `Primitive` has an `eval_gpu()` implementation;
- `mlx::core::metal::get_command_encoder()` encodes kernels on the active MLX
  stream;
- an output can share or donate a uniquely owned input buffer after an explicit
  `is_donatable()` check;
- several Metal dispatches and barriers can be encoded inside one primitive.

The primitive must remain functional from MLX's perspective.  There is no
Python-global mutable GPU buffer and no independent Metal command queue.  When
state is not donatable, the implementation allocates and copies before any
conditional write.  Buffer donation and no-rebuild aliasing require a dedicated
test before production integration.

The current project is a pure Python Hatchling build.  A native gate would add
build-time CMake and nanobind integration following MLX's extension mechanism.
Those are build dependencies, not molecular-runtime dependencies.  The build
change is deliberately deferred until the Python Metal force gate passes.

## Capacity and overflow correctness

Initialization may synchronously build once, read `ordinary_count`, and reserve
at least 125% of the observed tiles.  Later rebuilds use that fixed capacity and
set `overflow_flag` rather than writing out of bounds.

Overflow must never silently omit forces.  The retained native design needs a
small fixed-grid fallback dispatch.  When `overflow_flag` is zero its SIMD
groups return immediately.  When it is one, those groups stride across all
block pairs, apply the special masks where necessary, and compute a correct but
slow cutoff force for that step.  A later safe host boundary can grow capacity.

This rare fallback is preferable to a per-step scalar read.  If a bounded
prototype cannot demonstrate overflow correctness, it cannot be called
device-resident and cannot replace the production route.

## Prototype gates

### Gate A: inventory

Status: passed.

- Spatial ordering must reach at least 40% search-pair occupancy on both 5DFR
  and JAC.
- Estimated JAC persistent state must remain below 64 MiB.
- The ordering must be selected by evidence rather than copied from another
  engine.

Cell ordering reaches 44.90% on 5DFR and 45.15% on JAC with approximately
6.08 MiB and 24.23 MiB of state.

### Gate B: force algorithm

Build only the pack, ordinary force, special force, and scatter kernels.  The
schedule may be built by the research harness for this gate.

Correctness against the retained direct route on both prepared systems:

- force root-mean-square delta no greater than `1.0e-4 kJ/mol/A`;
- force maximum delta no greater than `1.0e-3 kJ/mol/A`;
- every value finite;
- exact same set of geometric and topology interactions;
- a non-32 SIMD width produces an explicit unsupported result and takes the
  retained route.

Performance, including pack and scatter:

- at least 15% lower synchronized force-only median on JAC;
- no regression larger than 2% on 5DFR;
- two interleaved directions agree;
- no result is accepted while Low Power Mode or power state differs between
  arms.

Failure closes the 32-atom force design.  Do not build the native primitive.

### Gate C: device builder

- randomized periodic small systems exactly match a brute-force search list;
- 5DFR and JAC exactly match the retained Verlet membership, allowing only a
  documented strict-boundary tie;
- no duplicate unordered atom pair is emitted;
- topology block pairs never enter the ordinary list;
- no prefix-tail allocation readback occurs;
- rebuild wall is at least 30% lower than the retained builder on both systems;
- the no-rebuild path does not rewrite or copy the capacity arrays.

### Gate D: native state and full trajectory

- buffer donation and non-donatable copy behavior pass targeted unit tests;
- overflow executes the correct fallback and is reported;
- 75-step interleaved complete-wall medians improve by at least 5% on 5DFR and
  8% on JAC;
- two 750-step JAC runs pass finite, constraint, route, memory-plateau, and
  equal-work gates;
- fixed-coordinate total/component energy and complete-force parity remain
  inside the existing charged-PME acceptance envelope;
- a fresh manifest-bound OpenMM comparison is required before publishing a new
  MLX/OpenMM ratio.

Only Gate D authorizes replacing `mlx_cell_tiles` as the default production
backend.  Until then, the new engine is opt-in and the current route is the
fallback.

## Explicit non-goals for the first prototype

- no full runtime atom reorder;
- no triclinic or changing-cell NPT support;
- no replacement of MLX FFT-based reciprocal PME;
- no fixed-point force buffer;
- no new chemistry or machine-learning dependency;
- no removal of the current production neighbor backend;
- no claim that a force-only microbenchmark is an end-to-end speedup.

## Primary API references

- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [Apple: porting Metal code to Apple Silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
- [Apple Metal feature-set tables](https://developer.apple.com/metal/capabilities/)
- [Apple indirect compute dispatch](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups%28indirectbuffer%3Aindirectbufferoffset%3Athreadsperthreadgroup%3A%29)

Apple explicitly advises querying the SIMD width rather than assuming it.  The
engine is optimized for the 32-lane Apple Silicon target, but must certify that
width and fall back safely on any different pipeline.
