# Production MLX MD Boundary

`mlx_atomistic` keeps the simulation engine lightweight. Core installs only MLX,
NumPy, and SciPy. Optional extras may parse chemistry/topology files or visualize
results, but `mlx_atomistic` owns trajectory generation.

## Dependency Extras

- `mlx-atomistic`: core MLX engine and reduced/physical-unit kernels.
- `mlx-atomistic[prep]`: topology/parameter import plus ligand chemistry/file
  parsing helpers. These tools do not run MD.
- `mlx-atomistic[viz]`: notebook visualization and trajectory analysis.

Raw PDB/mmCIF coordinates are accepted for visualization and selection, not as
general production MD input. Bundled examples are explicit exceptions:
`mlx_atomistic.prep` includes versioned internal templates for specific systems so
notebooks can build runnable MLX artifacts without an external simulator or
user-supplied topology files.

## Production Artifact Contract

A production artifact must include:

- explicit physical units for coordinates, mass, charge, energy, time, and
  temperature;
- topology arrays for bonds, angles, dihedrals, optional impropers, constraints,
  and nonbonded exceptions;
- per-atom LJ parameters and charges;
- force-field provenance and a compatibility report;
- no unsupported required terms.

`mlx_atomistic.artifacts.load_prepared_mlx_artifact(..., require_production=True)`
fails closed for reduced-unit demo artifacts, unsupported terms, missing arrays,
unsupported or incomplete PME/barostat requests, virtual sites,
Drude/polarizable terms, and other terms the MLX engine cannot yet represent
faithfully. Fixed-cell orthorhombic PME is a bounded production surface:
accepted artifacts must provide complete configuration/readiness metadata and
must fit the measured atom/mesh/cutoff/cell envelope. First-path NPT remains a
proof surface, and unsupported production cases remain blockers.

## Bounded GPCRmd 729 Fixed-Cell Result

The fresh production-readiness row uses
`gpcrmd-729-beta1-5f8u-cyanopindolol`: 92,001 atoms with receptor, ligand, POPC
membrane, TIP3P water, and sodium/chloride ions. Official source files were
reacquired with hashes, then parsed through the native CHARMM preparation path.

The fixed-cell orthorhombic NVT boundary now passes end to end:

- the strict prepared artifact and source workload manifest agree;
- independent OpenMM/OpenCL and MLX/Metal manifests match before numerical
  comparison;
- total/component energy and complete-force bounds pass;
- one warmup plus two measured 4 fs steps retain finite state with lazy
  `mlx_cell_blocks`/`NeighborBlocks`, shared LJ/direct-PME neighbors, one
  reusable PME plan, and no dense/tiled fallback;
- trajectory and checkpoint artifacts reload, and restart advances step/time
  from 3/0.012 ps to 4/0.016 ps without minimization or equilibration.

The quantitative record is in
[`gpcrmd-729-pme-runtime-m5max.md`](./benchmarks/gpcrmd-729-pme-runtime-m5max.md).
The regenerated blocker matrix marks the bounded fixture passed; the stale
`topology_terms` and `electrostatics_pme` blockers are no longer current.

This remains one bounded four-step fixed-cell result, not broad production MD
certification.

## Validated Charged Fixed-Cell PME Envelope

The product runtime now has a measured charged-PME validation workload:

- deterministic AMBER20 JAC 2x2x1 replication with 94,232 atoms;
- fixed orthorhombic cell, 128x128x64 mesh, order-5 assignment, and 9 A cutoff;
- explicit OpenMM-compatible `uniform_neutralizing_plasma` policy;
- independent OpenMM manifest match plus passing total/component energy and
  complete-force bounds;
- one warmup plus two measured finite NVT steps using one reusable PME plan,
  lazy topology, shared `NeighborBlocks`, and no fallback.

The quantitative record and the three gitignored raw JSON paths are in
[`scalable-charged-pme-runtime-m5max.md`](./benchmarks/scalable-charged-pme-runtime-m5max.md).
The measured readiness checks admit at most 100,000 atoms and 1,048,576 mesh
points for supported orthorhombic fixed-cell configurations; that admission
limit is not a claim that every chemistry or configuration inside the rectangle
has been certified.

Non-neutral artifacts still fail closed unless they explicitly select the
supported background policy. Existing artifacts without the new field retain
`reject_non_neutral`; unknown policies and metadata/array disagreement are
errors. A bound execution plan is reused only while cell, mesh, alpha, cutoff,
assignment order, deconvolution, Coulomb constant, dtype/backend/device, and
background policy remain compatible.

## Archived ATP-Receptor Workflow

The old ATP/P2X4 notebook has moved to
`notebooks/archive/atp-pocket-mlx-demo/`. It remains useful as historical
reference for the internal 4DW1 pocket artifact, but it is no longer the active
macromolecule visualization workflow. For that archived example:

1. build the prepared artifact with `prepare_p2x4_atp(..., backend="production_mlx")`
   and `save_prepared_system(...)` if the artifact is missing or stale;
2. validate the generated artifact with `require_production=True`;
3. run MLX minimization, restrained NVT warmup, and production NVT if
   `trajectory.npz` is missing or stale;
4. animate and analyze only the saved MLX coordinates with one preloaded Plotly
   trajectory player, visible controls, a translucent frame-0 ATP overlay, and
   ATP center-of-mass motion relative to the receptor pocket.

Expected Python API flow:

```python
from pathlib import Path

from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.prepare import prepare_p2x4_atp
from mlx_atomistic.prep.runner import run_mlx

prepared_dir = Path("notebooks/archive/atp-pocket-mlx-demo/data/prepared/4dw1-atp")
prepared = prepare_p2x4_atp(
    pdb_path=Path("notebooks/archive/atp-pocket-mlx-demo/data/4dw1_atp_bound_p2x4.pdb"),
    backend="production_mlx",
)
save_prepared_system(prepared, prepared_dir)
run_mlx(
    prepared_dir,
    require_production=True,
    steps=5000,
    sample_interval=25,
    dt=0.002,
    temperature=300,
    friction=10,
    restraint_k=5,
    minimize_steps=50,
    equilibration_steps=100,
)
```

General user systems still need real topology/parameter import first:

```python
from mlx_atomistic.prep import (
    import_amber_prmtop,
    import_charmm_psf,
    import_gromacs_top_gro,
)
```

Accepted imports can carry RB torsions and PME assignment-order metadata into
the strict artifact gate. PME assignment orders `2`, `4`, and `5` are accepted
when the artifact includes complete PME configuration arrays; unsupported
force-field terms still produce blockers rather than partial production runs.

The internal 4DW1 force field is fixed-topology classical MD: no ATP hydrolysis,
bond breaking, ligand docking/search, membrane, solvent, PME, or NPT.

## T4L / Benzene Forced-SMD Method Demo

The active macromolecular notebook is now
`notebooks/ligand-receptor-motion/01-ligand-receptor-translational-motion.ipynb`.
Its primary realistic path uses a public GLP-1R / Exendin-4 trajectory. The MLX
section builds a small soluble T4 lysozyme L99A / benzene artifact from PDB
`4W52` and runs forced steered MD:

```python
from pathlib import Path

from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.runner import run_steered_mlx
from mlx_atomistic.prep.t4l_benzene import prepare_t4l_benzene

prepared_dir = Path("notebooks/ligand-receptor-motion/data/prepared/t4l-benzene-smd")
save_prepared_system(prepare_t4l_benzene(), prepared_dir)
run_steered_mlx(prepared_dir, steps=25000, dt=0.001, sample_interval=50)
```

The T4L artifact is labeled `mlx_internal_t4l_benzene_forced_smd_demo_v2`. It
includes explicit hydrogens, topology arrays, simple internal parameters,
constraints, nonbonded exceptions, receptor/ligand masks, and steering
provenance. It is appropriate for demonstrating MLX-generated ligand translation
under a moving COM restraint. It is not a validated CHARMM/AMBER production force
field, does not represent natural diffusion, and does not infer a real benzene
egress route. The steering direction is a documented heuristic radial vector
from pocket-center to ligand-center.

The same notebook keeps the public GLP-1R / Exendin-4 trajectory as a labeled
`public_md` comparison. That comparison is analysis input only.

## Remaining Production Gaps

A GLP-1R / Exendin-4 production simulation generated by `mlx_atomistic` still
requires full membrane/solvent/ion setup, workload-specific PME/NPT validation,
validated CHARMM/AMBER force-field parity, and enhanced sampling beyond simple
SMD. The charged JAC PME result is a bounded fixed-cell validation surface, not
evidence for a complete membrane-production workflow.

For GPCRmd 729, the selected fixed-cell NVT fixture now has source-backed
preparation, independent parity, bounded finite execution, saved output, and
checkpoint continuation. The next gaps are larger than this closure:
production NPT and cell changes, analytic PME virial, triclinic PME,
production-length stability, broader GPCRmd coverage, and a general
membrane-production readiness claim. No OpenMM/MLX throughput ratio is valid
until both engines run a matching runtime manifest.
