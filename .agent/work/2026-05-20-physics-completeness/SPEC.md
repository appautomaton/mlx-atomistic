# SPEC: Physics Completeness and Production Hardening

## Bounded Goal

Implement virtual sites with force redistribution, custom force expressions, GBSA/OBC implicit solvent, soft-core potentials with lambda scaling, and replica exchange so that any standard biomolecular system can run with validated OpenMM parity, then harden the project with CI, a minor version bump, and documentation restructuring.

## Broader Intent

This makes mlx-atomistic a complete biomolecular MD engine on Apple Silicon: any standard force field, water model, or free-energy method can run without external engines at runtime, and the project ships reliably.

## Work Scale And Shape

- Scale: roadmap-sized (two phases, multiple independent capability areas).
- Shape: mixed (parity, feature, coverage, infrastructure).
- Rationale: the work spans six capability areas (virtual sites, custom forces, GBSA, soft-core/lambda, replica exchange, CI/docs) across two phases. Each area has distinct acceptance criteria but shares the `ForceTerm` protocol, `NonbondedPotential`, artifact pipeline, and `md.py` simulation driver. The decomposition must preserve shared surfaces while enabling agentic parallelization.

## Selected Lenses

- product
- engineering
- runtime

## Target User Or Stakeholder

Researchers and engineer-users who need production biomolecular MD — explicit TIP4P solvent, implicit GB solvent, or alchemical free energy — on Apple Silicon without depending on external engines at runtime.

## Linked Detail Files

- `spec/gap-matrix.md`: normative Phase 3 and Phase 4 gap IDs, evidence, and acceptance links.

## Constraints And Risks

- Use `uv` for Python commands; keep source under `src/mlx_atomistic/`.
- Treat `vendors/` as reference-only; do not import from vendor checkouts.
- OpenMM is a test/reference oracle only; MLX execution must not require OpenMM at runtime.
- `nonbonded.py` (1,092 lines) is the highest-risk file: virtual-site force redistribution and soft-core potentials both modify it. These must be sequenced or isolated so that a subagent working on one does not conflict with a subagent working on the other.
- `ForceTerm` protocol (`md.py` line 42) is the integration spine: every new force term (GBSA, CustomForce, soft-core) implements `energy_forces(positions, cell, pairs)`. Subagents can add force terms in parallel if they add to `forcefields.py` without modifying existing term internals.
- `md.py` (2,798 lines) is the single-trajectory simulation driver. Multi-copy replica exchange needs a new driver surface, not modification of the existing `simulate_nvt`/`simulate_npt` signatures. The existing functions must remain unchanged.
- Custom force expressions are a shared prerequisite for GBSA (which uses them for GB energy) and potentially for soft-core (which could be expressed as a custom nonbonded term). Custom forces must land before GBSA.
- GBSA/OBC requires an MLX-efficient surface-area computation. This is the most uncertain GPU-kernel work; the spec allows a Python fallback initially with the acceptance criterion requiring numerical accuracy, not maximum throughput.
- Phase 3 is large enough to need sub-spec decomposition by the planner. The SPEC preserves the full scope; PLAN.md orders the slices and identifies parallel-safe groups.
- Phase 4 (CI/docs) is intentionally separate. It has zero merge-conflict risk with Phase 3 physics work and should execute after Phase 3 stabilizes.

## Agentic Parallelization Guidance

The following module boundaries enable subagents to work concurrently without merge conflicts:

**Parallel-safe groups (can run simultaneously):**

| Group | Modules Touched | Constraint |
|-------|----------------|------------|
| Virtual-site geometry | New `virtual_sites.py`, `topology.py` (add fields) | Add new fields to `Topology.__init__` default; do not modify existing field logic |
| Custom force expressions | New `custom_force.py`, `forcefields.py` (add class) | Add `CustomForcePotential` after existing terms; do not modify existing force-term classes |
| GBSA surface-area kernel | New `gbsa.py` | Depends on `CustomForcePotential` from custom-force group; must sequence after that group |
| CI/docs | `.github/`, `docs/`, `README.md`, `pyproject.toml` version | Zero overlap with physics modules |

**Sequentially required (cannot parallelize):**

| Dependency | Reason |
|------------|--------|
| Virtual-site geometry → Virtual-site force redistribution | Redistribution reads virtual-site geometry to compute parent-atom force weights |
| Virtual-site redistribution → TIP4P-Ew energy parity | TIP4P test uses redistributed forces |
| Custom forces → GBSA | GBSA uses the `CustomForcePotential` expression framework |
| Soft-core/lambda → Replica exchange (Hamiltonian) | Hamiltonian replica exchange needs lambda-scaled `NonbondedPotential`; temperature-only exchange does not |

**Merge-conflict hot zones (sequential access only):**

- `nonbonded.py`: virtual-site redistribution and soft-core both modify `NonbondedPotential`. These must be in separate slices.
- `artifacts.py`: each new term type (virtual_site, custom_force, gbsa, soft_core_lj) adds to `SUPPORTED_FORCE_TERMS` and `FAIL_CLOSED_TERMS`. Coordinate by adding terms in their own slice.
- `md.py`: replica exchange adds a new top-level function; does not modify existing simulation functions.

## Required Outcome

### Phase 3: Physics Completeness

1. **Virtual sites with force redistribution:** `VirtualSite` base classes compute positions from parent atoms; forces redistribute to parent atoms; `Topology` carries virtual-site definitions; parser entry points populate virtual-site topology for supported water models; artifact pipeline accepts and validates virtual-site terms.

2. **TIP4P-Ew water model:** TIP4P-Ew uses virtual-site geometry with validated OpenMM energy parity for a TIP4P water box.

3. **Custom force expressions:** `CustomForcePotential` accepts symbolic expressions and per-particle or per-pair parameters; energy and forces match finite-difference reference; artifact pipeline accepts `custom_force` terms.

4. **GBSA/OBC implicit solvent:** GB-OBC model with ACE surface-area approximation computes implicit solvent energy and forces; artifact pipeline accepts `gbsa` terms; validated against analytical reference and OpenMM GB-OBC energetics.

5. **Soft-core potentials and lambda scaling:** Soft-core LJ and Coulomb produce finite energies at `lambda < 1` and match hard-core potentials at `lambda = 1`; `NonbondedPotential` accepts lambda parameters; `dU/dlambda` is available; artifact pipeline accepts soft-core and lambda parameters.

6. **Replica exchange:** Multi-copy driver manages N replicas with temperature or Hamiltonian exchange; swap acceptance matches Metropolis criterion; energy histogram overlap confirms correct Boltzmann sampling.

7. **Existing Phase 1+2 tests pass after Phase 3 changes.**

### Phase 4: Production Hardening

8. **CI pipeline:** GitHub Actions runs `pytest` and `ruff check` on every PR.

9. **Minor version bump:** `pyproject.toml` version moves to next 0.x minor.

10. **Docs restructured and linked from README.**

## Acceptance Criteria

| ID | Check |
|----|-------|
| AC-01 | Virtual-site base classes (TwoParticleAverage, ThreeParticleAverage, OutOfPlane, LocalCoordinates) compute positions from parent atoms; forces redistribute to parent atoms; virtual-site positions are reconstructed each timestep; `Topology` carries virtual-site definitions; artifact pipeline accepts virtual-site terms and rejects undeclared types. |
| AC-02 | TIP4P-Ew water model produces correct geometry and matches OpenMM reference energetics within stated tolerance for a periodic water box. |
| AC-03 | `CustomForcePotential` evaluates energy and forces matching finite-difference reference for bond-like, angle-like, and nonbonded-like expressions; artifact pipeline accepts `custom_force` terms. |
| AC-04 | GB-OBC implicit solvent computes energy and forces matching OpenMM GB-OBC reference within stated tolerance for neutral and charged periodic fixtures；ACE surface-area term validated against analytical reference. |
| AC-05 | Soft-core LJ and Coulomb potentials produce finite energies at `lambda < 1`, match hard-core potentials at `lambda = 1`, and provide smooth `dU/dlambda` for thermodynamic integration; artifact pipeline accepts `soft_core_lj` and `lambda_scaled_nonbonded` parameters. |
| AC-06 | Replica exchange driver manages N replicas, performs temperature or Hamiltonian swaps at specified intervals, and produces Boltzmann-weighted energy histograms consistent with the expected temperature distribution. |
| AC-07 | GitHub Actions CI runs `pytest` and `ruff check` on PRs; version bumped to next 0.x minor; docs restructured and linked from README. |

## Scope Coverage Decisions

Included:
- Virtual site base classes, force redistribution, constraint integration, TIP4P-Ew.
- Custom force expression framework.
- GBSA/OBC implicit solvent with ACE surface area.
- Soft-core LJ and Coulomb, lambda-dependent nonbonded scaling.
- Replica exchange (temperature and Hamiltonian variants).
- GitHub Actions CI, minor version bump, docs restructuring.

Deferred to later phases or roadmap:
- DFT/QM-MM improvements.
- Multi-GPU or multi-node parallelization.
- Advanced barostat modes beyond current MC isotropic/anisotropic/membrane.
- Further parser coverage beyond Phase 2's declared supported subset.
- Polarizable force fields (Drude, AMOEBA).
- Machine-learned interatomic potentials.
- Reactive or materials force fields.
- CUDA/ROCm GPU support.
- Production deployment or serving infrastructure.

Anti-goals:
- Do not target version 1.0; this is a minor version bump only.
- Do not silently drop unsupported force-field terms or virtual-site types; continue fail-closed.
- Do not introduce runtime OpenMM, GROMACS, or ParmEd dependencies for accepted execution paths.
- Do not broaden into production deployment, serving, or distribution packaging.

## Review: Engineering

(To be filled by auto-eng-review after auto-ceo-review.)