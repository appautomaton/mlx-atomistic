# GAP MATRIX: MD Engine Capability

## Reading Rules

- Status values: `implemented`, `partial`, `missing`, `unverified`.
- Production impact: `blocker`, `important`, or `deferred`.
- Recommended order names the roadmap dependency, not implementation detail.
- Evidence is path-level and intentionally compact; see `EVIDENCE-INDEX.md` for
  the inspected surface list.

## T1: Core MD Physics

Reference anchor: OpenMM exposes `System`/`Force`/`Context` plus PME,
integrators, constraints, virtual sites, and barostats in `vendors/openmm`.
GROMACS and LAMMPS source show mature PME/PPPM, NPT, GPU pair kernels, and
long-run execution surfaces.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T1-01 | NVE integration | implemented | `md.py:simulate`, `md.py:simulate_nve`; `tests/test_nve.py`, `benchmarks/stability.py` | important | validate longer energy drift on production artifacts | after PME/parity fixture |
| T1-02 | NVT Langevin/BAOAB | implemented | `md.py:LangevinThermostat`, `md.py:simulate_nvt`; `tests/test_nvt.py`, `protocols.py:run_minimize_then_nvt` | important | compare temperature/energy distributions with OpenMM | parity wave |
| T1-03 | Thermostat diversity | partial | Langevin exists; no Nose-Hoover/Andersen/v-rescale surface found; OpenMM has several integrator/thermostat classes | deferred | defer until common NVT/NPT path is credible | deferred |
| T1-04 | NPT/barostat | missing | `protocols.py:validate_gpcrmd_protocol_request` records `npt_barostat`, `barostat`, `membrane_barostat` blockers; `tests/test_protocols.py` asserts rejection | blocker | implement MC barostat after PME virial is validated | after PME |
| T1-05 | PME long-range electrostatics | partial | `pme.py` implements standalone PME, but `PME_PRODUCTION_EXECUTABLE = False` and `pme_readiness_report` blocks `numpy_reference`; `tests/test_pme.py` validates the gate | blocker | make PME production executable and parity-checked vs OpenMM PME | first wave |
| T1-06 | Ewald reference electrostatics | implemented | `nonbonded.py:ewald_reference_coulomb_energy_forces`; `benchmarks/ewald_reference.py`; `tests/test_ewald_reference.py` | important | keep as correctness oracle for small systems | supports PME |
| T1-07 | Constraints | partial | `constraints.py:DistanceConstraints`; `md.py` applies position/velocity projection; tests cover finite constraint diagnostics | blocker | long-run stability/HMR/virtual-site policy validation | parity wave |
| T1-08 | Pressure/virial diagnostics | partial | `md.py:virial_tensor`, `pressure_tensor`, `_pressure_diagnostics`; `tests/test_virial_pressure.py`; no coupling back to a barostat | blocker | validate virial with PME, then drive NPT acceptance | before NPT |
| T1-09 | Orthorhombic periodic cells | implemented | `core.py:Cell` is orthorhombic; `cell_list.py` builds orthorhombic periodic pairs; `tests/test_core.py`, `tests/test_neighbors.py` | important | keep for first PME/NPT target | PME baseline |
| T1-10 | Triclinic/non-orthorhombic cells | missing | no triclinic cell object found; PME and cell-list validation require orthorhombic lengths | important | add triclinic cell/minimum-image/neighbors after first PME parity | after first PME |
| T1-11 | Virtual sites/TIP4P-style water | missing | `artifacts.py` fail-closed terms include `virtual_site`/`virtual_sites`; `gpcrmd.py` blocker `virtual_sites_or_hydrogen_mass_repartitioning_not_checked` | blocker for 4 fs/TIP4P workflows | implement or explicitly gate water model policy | after PME, before broad artifacts |
| T1-12 | Water/ion masks and simple explicit water | partial | `PreparedSystem` includes `water_mask`/`ion_mask`; `prep/solvated_example.py` creates explicit waters but marks `production_force_field=False` | important | validate real TIP3P/CHARMM/AMBER water artifact parity | parity wave |
| T1-13 | Long-range dispersion/LJPME | missing | no production LJPME/dispersion-correction runner surface found; OpenMM/GROMACS benchmark docs include LJPME/PME variants | important | defer until Coulomb PME is production-ready | after PME |

## T2: Force-Field And Artifact Coverage

Reference anchor: OpenMM `app.ForceField` builds systems from topology at run
time; this repo uses `PreparedSystem` artifacts. GROMACS/LAMMPS are useful
references for file-format coverage and long-range method semantics, not runtime
dependencies.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T2-01 | Production artifact gate | implemented | `artifacts.py:validate_mlx_compatibility`; `load_prepared_mlx_artifact(..., require_production=True)`; `tests/test_production_artifacts.py` | blocker | keep fail-closed; update docs when source support changes | always-on |
| T2-02 | AMBER topology import | partial | `prep/topology_import.py:import_amber_prmtop`; exports masses, charges, bonds, angles, dihedrals, exceptions, masks, constraints; tests cover production fixture paths | blocker | run OpenMM parity fixture from same AMBER artifact | first wave |
| T2-03 | CHARMM/ParmEd import | partial | `prep/topology_import.py:import_charmm_with_parmed`; exports CMAP, NBFIX type overrides, masks, constraints; `tests/test_mlx_prep.py` covers import details | blocker | validate on a real CHARMM artifact against OpenMM | after AMBER fixture or parallel |
| T2-04 | CMAP runtime force term | partial | `charmm_terms.py:CHARMMCMAPPotential`; `artifacts.py` builds CMAP terms; `tests/test_charmm_terms.py` finite-difference checks | important | parity-check energy/forces against OpenMM CHARMM CMAP | parity wave |
| T2-05 | NBFIX pair overrides | partial | `charmm_terms.py:CHARMMNBFIXPairOverridePotential`; `topology_import.py` exports type-pair overrides; distinct 1-4 NBFIX values fail closed | important | validate real CHARMM NBFIX cases and neighbor-list behavior | parity wave |
| T2-06 | 1-4 exceptions/exclusions | implemented | `nonbonded_exception_*` arrays in `PreparedSystem`; `artifacts.py` builds nonbonded exceptions; tests cover presence and force evaluation | blocker | include in OpenMM parity target | first wave |
| T2-07 | Hydrogen mass repartitioning policy | unverified | GPCRmd gate names HMR/virtual-site policy as unchecked; no explicit HMR runtime policy found | blocker for 4 fs workflows | parse/record HMR or reject with artifact-level reason | after PME parity fixture |
| T2-08 | Ligand/small-molecule parameters | partial | internal 4DW1/T4L/solvated examples exist; `production_pocket.py` is fixed-template; general GAFF/OpenFF path not found | important | choose one production ligand artifact and parity-check terms | after core parity |
| T2-09 | Drude/AMOEBA/polarizable | missing | fail-closed terms include Drude/polarizable; no AMOEBA runtime surface found; OpenMM/LAMMPS have reference surfaces | deferred | defer until fixed-charge MD is production credible | deferred |

## T3: Runtime Production Usability

Reference anchor: OpenMM app has `DCDReporter`, `StateDataReporter`, and
`CheckpointReporter`. LAMMPS has dump styles including DCD/XTC in vendor source.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T3-01 | Native NPZ trajectory output | implemented | `io.py:save_npz_trajectory`, `load_npz_trajectory`; `prep/runner.py:run_mlx` writes `trajectory.npz`; tests cover metadata and finite arrays | important | keep as native debug/checkpoint-adjacent format | current |
| T3-02 | Step-level reporters/callbacks | missing | `run_mlx` saves after a run; no OpenMM-style reporter append/callback surface found | important | add minimal reporter callback before long parity runs | before long runs |
| T3-03 | Runner-level checkpoint/restart | partial | `io.py:restart_state_from_trajectory` recomputes forces from a trajectory frame; no runner checkpoint with RNG, neighbor, thermostat, metadata state found | blocker for multi-day runs | design checkpoint schema after reporter hook | usability wave |
| T3-04 | DCD/XTC first-class runner output | partial | `trajectory_adapters.py` can convert to MDAnalysis/MDTraj; no direct `run_mlx(..., output='dcd/xtc')` surface found; LAMMPS has DCD/XTC dump files | important | expose DCD/XTC writer through reporter/output API | after reporters |
| T3-05 | Runtime diagnostics and metadata | implemented | `NVE/NVTResult` stores energy, temperature, pressure, pair/rebuild counts, constraint error; `save_npz_trajectory` persists diagnostics and metadata | important | standardize diagnostic summaries for parity reports | first wave |
| T3-06 | Fail-closed error behavior | implemented | artifact/protocol gates reject unsupported production terms, NPT/barostat, invalid PME config, output overwrite cases | important | keep errors explicit as new features land | always-on |
| T3-07 | Reproducibility metadata/RNG state | partial | thermostat seed is recorded/configured; no serialized RNG state for mid-run resume found | blocker for exact restart | include RNG state in checkpoint design | usability wave |

## T4: Validation And Parity

Reference anchor: OpenMM is executable locally and should be the primary parity
engine. GROMACS/LAMMPS source inform what mature engines validate, but they are
not required to run for the first parity pass.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T4-01 | Finite-difference force validation | implemented | `validation.py:run_force_validation_suite`; `benchmarks/validation_gauntlet.py`; `tests/test_validation.py`, `tests/test_forcefields.py`, `tests/test_charmm_terms.py` | important | include PME/CMAP/NBFIX cases in default suite as they mature | current |
| T4-02 | Stability benchmark suite | implemented | `benchmarks/stability.py`, `benchmarks/md_performance.py`; tests cover finite results and JSON/CSV smoke | important | run on production artifact after PME gate | after PME |
| T4-03 | OpenMM performance baseline | implemented | `scripts/benchmark_openmm_opencl.py`; `docs/benchmarks/openmm-opencl-*.md`; live `uv` probe confirms OpenCL platform | important | keep as performance ceiling, not product path | current |
| T4-04 | OpenMM force/energy parity harness | partial | OpenMM preview/reference scripts exist, but no general parity harness mapping one artifact to component-wise MLX/OpenMM force/energy tolerances was found | blocker | build parity harness around selected AMBER/CHARMM artifact | first wave |
| T4-05 | Trajectory distribution parity | missing | no systematic temperature/PE/RMSD/RDF parity report for same system found | important | add after force/energy parity and reporter/diagnostic normalization | after T4-04 |
| T4-06 | GROMACS/LAMMPS parity runs | unverified | vendor source exists; no local `gmx`/`lmp` CLI path found in current shell; LAMMPS Python module is importable | deferred | use source as reference until a specific parity need appears | deferred |
| T4-07 | GPCRmd runtime benchmark surface | partial | `prep/gpcrmd_benchmark.py`, `tests/test_gpcrmd_registry.py` finite tiny-fixture benchmark; production-scale blockers still route to PME/NPT/HMR | important | rerun after PME production gate | after PME |

## T5: Performance And Backend

Reference anchor: GROMACS has NBNXM CUDA/HIP/OpenCL/SYCL kernels and GPU FFT
selection; OpenMM has OpenCL and common platform kernels; LAMMPS has GPU package
pair styles and PPPM code.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T5-01 | Neighbor-list scaling | partial | `cell_list.py`, `NeighborListManager`, `mlx_cell_pairs` tests and acceleration benchmarks; large/real artifact throughput still unverified | important | profile selected parity artifact before kernel work | after PME correctness |
| T5-02 | Dense/pair nonbonded kernels | partial | MLX pair paths and backend policy exist; no custom Metal fused pair kernel found | important | benchmark first; only implement custom kernels for measured hotspots | after parity |
| T5-03 | PME FFT/backend performance | partial | PME uses `numpy_reference`; GROMACS CMake shows GPU FFT library choices; OpenMM docs show PME OpenCL ceiling | blocker | replace/prove PME backend after correctness target is fixed | first wave with PME |
| T5-04 | OpenMM OpenCL throughput target | implemented as reference | docs report DHFR/ApoA1/Amber20 OpenMM OpenCL baselines on Apple Silicon; OpenMM platform probe passes | important | define target ratio after MLX can run same PME artifact | after PME |
| T5-05 | GROMACS GPU backend patterns | source-only | `vendors/gromacs/CMakeLists.txt` supports CUDA/OpenCL/SYCL/HIP; `src/gromacs/nbnxm/{cuda,hip,opencl,sycl}` present | important reference | study when choosing MLX/Metal kernel layout | performance wave |
| T5-06 | LAMMPS GPU/PPPM patterns | source-only | `vendors/lammps/src/GPU`, `lib/gpu/lal_pppm.*`, `src/EXTRA-DUMP/dump_{dcd,xtc}` present | important reference | study for modular NPT/dump/kspace patterns | performance/usability wave |
| T5-07 | Custom Metal/MLX fused kernels | missing | no product custom Metal op surface found in `src/mlx_atomistic`; current path relies on MLX ops and NumPy PME reference | important | defer until profiling isolates a hotspot | after correctness |
| T5-08 | Mixed precision policy | unverified | no explicit MD-wide mixed precision policy found beyond dtype choices in code/tests | deferred | document/benchmark after PME/NPT parity | deferred |

## T6: Prep And Workflow

Reference anchor: OpenMM `app.PDBFile`/`Modeller`/`ForceField.createSystem` is
the mature one-shot raw-input path. This repo deliberately routes real MD through
MLX-compatible prepared artifacts.

| ID | Capability | Status | Evidence | Impact | Next action | Order |
| --- | --- | --- | --- | --- | --- | --- |
| T6-01 | PreparedSystem schema | implemented | `prep/schema.py:PreparedSystem` contains positions, velocities, masses, cell lengths, constraints, masks, exceptions, PME arrays, CMAP/NBFIX arrays | blocker | keep as product boundary | current |
| T6-02 | Build MLX runtime from artifact | implemented | `artifacts.py:build_mlx_system_from_artifact`; `prep/runner.py:build_mlx_system`; tests build systems/terms/constraints | blocker | use as parity harness input | first wave |
| T6-03 | Raw PDB plus force field one-shot | missing | docs state raw PDB/mmCIF is accepted for visualization/selection, not general production MD input; no `ForceField.createSystem` equivalent found | deferred | defer unless user ergonomics become blocker | deferred |
| T6-04 | Specific demo artifacts | implemented | `prep/t4l_benzene.py`, `prep/solvated_example.py`, `prep/production_pocket.py`; tests run short MLX trajectories | important | keep as smoke fixtures, not production evidence | current |
| T6-05 | GPCRmd artifact conversion | partial | `prep/gpcrmd.py` converts/validates and annotates blockers; tests cover registry/runtime tiny paths; blockers remain PME/NPT/HMR/scale | important | rerun after PME and checkpoint/reporting work | after PME |
| T6-06 | Notebook/script entrypoints | partial | notebooks and helper scripts exist; docs warn which trajectories are product vs reference | important | keep output-free and tied to prepared artifacts | ongoing |
| T6-07 | Evidence baseline run | unverified | plan keeps a `prep.run_mlx` baseline as possible evidence slice; not run in this audit | important | run after choosing next SPEC fixture | first implementation spec |

## Matrix Conclusion

The production-blocking MD gaps are concentrated in:

1. PME production backend and OpenMM component parity (`T1-05`, `T4-04`,
   `T5-03`).
2. NPT/barostat after PME virial is trustworthy (`T1-04`, `T1-08`).
3. Artifact parity for AMBER/CHARMM plus constraints/HMR/virtual-site policy
   (`T2-02`, `T2-03`, `T2-07`, `T1-11`).
4. Long-run usability once trajectories are worth running longer
   (`T3-02`, `T3-03`, `T3-04`).
5. Performance only after correctness/parity exposes the real hot path
   (`T5-01` through `T5-07`).

The repo is not at zero. It already has NVE/NVT, constraints, force terms,
artifact gates, diagnostics, native NPZ trajectories, finite-difference
validation, benchmark scaffolding, and partial AMBER/CHARMM/PME surfaces. The
next work should close the production blockers rather than rewrite the engine
skeleton.
