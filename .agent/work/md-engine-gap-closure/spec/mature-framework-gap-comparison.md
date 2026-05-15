# Mature Framework Gap Comparison

## Purpose

This file turns the verified gap matrix into build decisions for
`mlx_atomistic`. The goal is not to copy OpenMM, GROMACS, or LAMMPS feature for
feature. The goal is to decide which mature-framework capabilities matter for a
credible MLX-native biomolecular MD runtime in this repo.

## Reference Baselines

| Framework | What it is useful for here | What not to copy directly |
| --- | --- | --- |
| OpenMM | Primary executable physics reference for biomolecular systems: force fields, PME, constraints, virtual sites, barostats, reporters, checkpoints, and Python API ergonomics. | Full platform/plugin architecture. MLX already gives a Metal-centered runtime path. |
| GROMACS | Throughput and algorithm reference: NBNXM pair kernels, PME/FFT split, GPU backend choices, GPU-resident execution, checkpoint/restart, trajectory output. | Full CLI/application stack and broad HPC tuning surface. |
| LAMMPS | Modularity reference: `fix` style integrators/barostats, PPPM/kspace, GPU/OpenCL package, dump formats, materials breadth. | Broad materials/reactive/polarizable ecosystem outside first biomolecular MD target. |

## Decision Rule

Build now when the capability is required for common fixed-charge biomolecular
production MD and cannot be delegated to a reference engine without losing
`mlx_atomistic` as the trajectory generator.

Validate now when source shows a partial implementation but no OpenMM parity or
real-artifact proof.

Defer when the capability is long-tail coverage, a mature-framework convenience
that does not block the prepared-artifact path, or a performance optimization
that should wait for correctness evidence.

## Build / Validate / Defer

| Area | Mature framework baseline | `mlx_atomistic` state | Decision for this spec |
| --- | --- | --- | --- |
| Production artifact parity | OpenMM can build a reference `System` from standard topology/force-field inputs. | AMBER/CHARMM artifact import exists, but there is no general MLX-vs-OpenMM force/energy parity harness. | **Build first.** This is the fixture all later work needs. |
| PME / long-range electrostatics | OpenMM/GROMACS/LAMMPS have production PME/PPPM paths. | PME exists but is explicitly gated by `PME_PRODUCTION_EXECUTABLE = False` and `pme_readiness_report`. | **Build after parity fixture.** Make PME production-ready only against explicit OpenMM tolerances. |
| Virial / pressure | Mature engines use virial accounting for pressure diagnostics and barostats. | Pressure/virial diagnostics exist, but do not drive volume changes. | **Validate with PME, then build NPT.** |
| NPT / barostat | Mature engines have pressure coupling, including MC or Nose-Hoover family barostats. | Protocol gate rejects NPT/barostat requests. | **Build after PME and virial validation.** Start with MC barostat unless planning evidence changes this. |
| Constraints | Mature engines apply constraints inside integration and support water/H-bond workflows. | Pair-distance constraints exist and are applied in NVE/NVT loops. | **Validate.** Long-run stability and production-artifact parity are needed before treating this as complete. |
| HMR / virtual sites / 4 fs workflows | Mature biomolecular setups support HMR and/or virtual-site water models. | GPCRmd path flags HMR/virtual-site policy as unchecked; virtual sites fail closed. | **Build policy and support or exact rejection.** This blocks common modern workflows. |
| AMBER force-field path | OpenMM and AMBER-family workflows are standard production paths. | AMBER import exists and exports production metadata, constraints, masks, and exceptions. | **Validate first.** Use it as the first parity fixture if a small enough artifact is available. |
| CHARMM force-field path | OpenMM/GROMACS support CHARMM, CMAP, NBFIX, membranes. | CHARMM/ParmEd import, CMAP, and NBFIX surfaces exist, but broad parity is unverified. | **Validate after or alongside AMBER.** Do not label as missing. |
| Reporters / callbacks | OpenMM reporters and LAMMPS dumps allow step-level observation and output. | `run_mlx` writes native NPZ after the run; no general step-level reporter surface. | **Build.** Needed for parity traces and long runs. |
| Checkpoint / restart | Mature engines can restart long production runs. | `restart_state_from_trajectory` exists, but no runner-level checkpoint with RNG/thermostat/neighbor metadata. | **Build after reporters.** |
| DCD/XTC output | Mature engines write analysis-ready trajectories directly. | NPZ is first-class; MDAnalysis/MDTraj adapters exist, but no runner output surface. | **Build after reporters.** Route through MDTraj if useful, but expose as product API. |
| Raw PDB plus force-field one-shot | OpenMM app layer handles PDB/Modeller/ForceField flows. | Project boundary is `PreparedSystem`; raw PDB is not general production input. | **Defer.** Keep artifact path first. |
| Triclinic cells | Mature engines support general periodic boxes. | Cell and PME code are orthorhombic. | **Defer until orthorhombic PME parity passes.** Important, but not first blocker. |
| Thermostat variety | Mature engines expose several thermostats/integrators. | Langevin NVT exists. | **Defer.** Coverage expansion after NPT path works. |
| LJPME / dispersion correction | Mature biomolecular engines support long-range LJ variants/corrections. | No first-class production LJPME/dispersion correction found. | **Defer until Coulomb PME is production-ready.** |
| Free energy / REMD / metadynamics | Mature ecosystems support these through engines or plugins. | No first-wave surface. | **Defer.** Not required for fixed-charge production MD credibility. |
| Polarizable force fields | OpenMM/LAMMPS have AMOEBA/Drude-like surfaces. | Fail-closed or absent in MLX path. | **Defer.** Long-tail coverage. |
| Pair-kernel / neighbor performance | GROMACS/OpenMM/LAMMPS have specialized GPU kernels and list structures. | MLX paths and cell-list/neighbor managers exist; custom Metal kernel work is not proven necessary yet. | **Profile later.** Optimize only after correctness gates identify the hot path. |
| PME FFT performance | Mature engines have optimized FFT/GPU FFT paths. | PME uses reference-style implementation today. | **Build as part of PME readiness if correctness passes and profiling shows it blocks throughput.** |
| Backend abstraction | OpenMM/GROMACS carry CUDA/OpenCL/SYCL/HIP/platform abstractions. | `mlx_atomistic` is MLX/Metal-first. | **Do not copy.** Treat MLX/Metal as the product backend and use other engines for references. |

## What This Means For The Spec

The first thing to make is not a faster kernel or a barostat. It is a shared
parity fixture:

1. Select a small real production artifact.
2. Build the MLX runtime system from it.
3. Build an OpenMM reference system from the same source.
4. Compare total energy, component energies, and forces at the same coordinates.
5. Record exact unsupported terms.

After that, the next build decisions are evidence-driven:

- If parity fails before PME, fix force-field/artifact conversion first.
- If parity passes but PME is blocked, work on PME readiness.
- If PME passes but trajectories cannot be observed or resumed, work on reporters
  and checkpointing before long runs.
- If PME and virial diagnostics pass, implement NPT.
- If performance is the blocker after correctness, profile and optimize the
  measured hot path.

## Not For This Spec

- Replacing OpenMM/GROMACS/LAMMPS as general-purpose frameworks.
- Building a GROMACS runtime dependency.
- Copying OpenMM's full platform plugin architecture.
- Adding raw PDB-to-production-system ergonomics before artifact parity.
- Adding DFT, free energy, REMD, metadynamics, or polarizable force fields.
