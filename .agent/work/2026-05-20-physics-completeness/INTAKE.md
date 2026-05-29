# INTAKE: Physics Completeness and Production Hardening

## Work Scale

roadmap

## Work Shape

mixed (parity, feature, coverage, infrastructure)

## Objective Statement

Complete the remaining physics capabilities — virtual sites with force redistribution, custom force expressions, GBSA/OBC implicit solvent, soft-core potentials, lambda scaling, and replica exchange — then harden the project with CI, a minor version bump, and documentation restructuring.

## Broader Intent

This makes mlx-atomistic a complete biomolecular MD engine on Apple Silicon: any standard force field, water model, or free-energy method can run with validated OpenMM parity, and the project ships reliably.

## Target User or Stakeholder

Researchers and engineer-users who need production biomolecular MD — explicit TIP4P solvent, implicit GB solvent, or alchemical free energy — on Apple Silicon without depending on external engines at runtime.

## Desired Outcome

- Any standard biomolecular system (CHARMM36, AMBER ff14SB, GROMACS) with TIP4P water or GB implicit solvent runs with validated OpenMM parity.
- Alchemical free-energy calculations produce ΔG values within error bars of OpenMM reference.
- CI runs on every PR; docs are current; version is bumped to the next 0.x minor.

## Scope Boundary and Anti-Goals

### Included

- Virtual site framework: base classes (TwoParticleAverage, ThreeParticleAverage, OutOfPlane, LocalCoordinates), force redistribution from virtual sites to parent atoms, virtual-site constraint integration.
- TIP4P-Ew water model with validated energetics.
- Custom force expression framework (CustomForcePotential).
- GBSA/OBC implicit solvent (GB-OBC model with ACE surface-area calculation on MLX).
- Soft-core LJ and Coulomb potentials for alchemical transformations.
- Lambda-dependent NonbondedPotential scaling.
- Replica exchange driver running temperature and Hamiltonian exchange.
- GitHub Actions CI running pytest + ruff on PRs.
- Minor version bump (0.x → 0.x+1).
- Docs restructured and linked from README.

### Anti-Goals

- Do not broaden into production deployment, serving infrastructure, or distribution packaging.
- Do not add polarizable force fields (Drude, AMOEBA), machine-learned potentials, reactive force fields, or materials potentials.
- Do not add CUDA/ROCm GPU support.
- Do not target 1.0; this is a minor version bump only.
- Do not silently ignore unsupported force-field terms or virtual-site types — continue fail-closed.
- Do not introduce runtime OpenMM, GROMACS, or ParmEd dependencies for accepted execution paths.

### Deferred

- DFT or QM/MM integration improvements beyond current scope.
- Multi-GPU or multi-node parallelization.
- Advanced barostat modes beyond current MC isotropic/anisotropic/membrane.
- Further GROMACS preprocessing or CHARMM extension coverage beyond Phase 2's declared supported subset.

## Rejected Framings

- **Single "do everything" phase**: Too large for one spec; the roadmap needs decomposition into testable sub-specs per phase.
- **Virtual sites as Phase 3 only, everything else Phase 4**: Under-estimates the shared infrastructure (custom forces, multi-copy drivers) that connects GBSA, replica exchange, and alchemical work.
- **CI/docs as Phase 3 before physics**: Delays the highest-value capability work; CI is low-risk and better positioned after the codebase stabilizes from Phase 3 changes.

## Scope Preservation

This preserves the user's full stated intent: all remaining physics capabilities, followed by production hardening. The consolidation from 4 phases to 2 reduces coordination overhead without dropping coverage.

## Scope Coverage

### Included

- Virtual site framework, force redistribution, TIP4P-Ew
- Custom force expressions
- GBSA/OBC implicit solvent
- Soft-core potentials, lambda scaling
- Replica exchange
- CI, minor version bump, docs

### Deferred

- DFT/QM-MM improvements
- Multi-GPU parallelization
- Advanced barostat modes
- Further parser coverage beyond Phase 2
- Polarizable, ML, reactive, and materials force fields
- Production deployment/serving

### Anti-Goals

- 1.0 version target
- Runtime OpenMM/GROMACS/ParmEd dependencies for accepted paths
- Silent dropping of unsupported features
- Deployment infrastructure

### Needs Decision

None — user confirmed Option A with minor version bump.

## Selected Approach

**Option A: Physics Completeness then Production Hardening.** Virtual sites first (deepest nonbonded surgery, prerequisite for TIP4P and force redistribution), then custom forces → GBSA → soft-core/lambda → replica exchange, then CI/version/docs. This opens `nonbonded.py` once for the foundational change and builds every subsequent feature on the `ForceTerm` protocol and artifact pipeline that Phases 1+2 established.

## Key Assumptions and Risks

- Virtual sites require force redistribution architecture in `nonbonded.py` — highest implementation risk, touches the same module as soft-core potentials.
- GBSA/OBC requires accurate surface-area calculation as an MLX GPU kernel — nontrivial custom kernel work.
- Custom force expressions (needed by both GBSA and soft-core) are a shared prerequisite that must land before either downstream feature.
- Multi-copy simulation infrastructure (needed by both replica exchange and lambda windows) is a shared prerequisite for Phase 3's后半段.
- Phase 3 is large and must decompose into sub-specs: likely (a) virtual sites + TIP4P, (b) custom forces + GBSA, (c) soft-core + lambda + replica exchange.
- Phase 4 is low-risk and can be executed quickly once Phase 3 stabilizes.

## Recommended Next Skill

`auto-frame` — to bound Phase 3 into a SPEC.md with concrete acceptance criteria and decomposition guidance.