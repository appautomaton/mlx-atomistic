# GPCRmd-Critical MLX MD Readiness Handoff

## Verdict

The GPCRmd-critical MLX runtime path is implemented and verified on fixture
artifacts, but the selected real GPCRmd target has not run because the required
GPCRmd package files are not present in the workspace.

The framework now fails closed with exact blockers instead of showing reference
or fake motion as an MLX result.

## Implemented

- PME mesh electrostatics and `NonbondedPotential(electrostatics="pme")`.
- CHARMM/GPCR primitives needed by the readiness plan: CMAP, Urey-Bradley,
  CHARMM LJ force-switch, and NBFIX pair overrides.
- Periodic scalable pair construction for real-space nonbonded work.
- Strict prepared-artifact schema for PME, CHARMM terms, lipid masks,
  constraints, exceptions, protocol metadata, and virial/pressure diagnostics.
- GPCRmd import path that writes MLX prepared artifacts or exact blocker reports.
- GPCRmd MLX runtime command:

  ```bash
  uv run atomistic-prep run-gpcrmd-mlx --target <id> --cache <gpcrmd-cache> --out <out> --json
  ```

- Notebook main path that consumes only the MLX-generated GPCRmd trajectory.
- GPCRmd benchmark command:

  ```bash
  uv run atomistic-prep benchmark-gpcrmd-mlx --target <id> --cache <gpcrmd-cache> --out <bench> --json
  ```

## Verification

- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest`
  - `308 passed`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src tests scripts`
  - `All checks passed!`
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run atomistic-prep gpcrmd-inspect --target gpcrmd-729-beta1-5f8u-cyanopindolol --cache /tmp/mlx-gpcrmd-readiness-empty-cache --compatibility --json`
  - completed and reported exact missing inputs.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run atomistic-prep run-gpcrmd-mlx --target gpcrmd-729-beta1-5f8u-cyanopindolol --cache /tmp/mlx-gpcrmd-readiness-empty-cache --out /tmp/mlx-gpcrmd-readiness-run --steps 2 --sample-interval 1 --dt 0.0005 --json`
  - exited blocked with exact blocker JSON and no trajectory.

## Selected Target Status

Selected target:

- `gpcrmd-729-beta1-5f8u-cyanopindolol`
- dynamics id: `729`
- PDB: `5F8U`
- receptor: Beta-1 adrenergic receptor
- ligand-bound membrane/solvent system, `92001` atoms
- POPC membrane, TIP3P water, sodium/chloride ions
- protocol metadata: NVT, 4 fs timestep, 3 reference replicates

Current blockers with an empty cache:

- `missing_input:file:topology:15286`
- `missing_input:file:model:17686`
- `missing_input:file:parameters:15290`
- `missing_input:file:protocol:17687`
- `missing_input:box_vectors:requires_model_or_coordinate_file`

Metadata-level remaining risk after real files are mounted:

- `virtual_sites_or_hydrogen_mass_repartitioning_not_checked`

This is not a runtime dependency on OpenMM, LAMMPS, GROMACS, or GPCRmd engines.
The blockers mean the local GPCRmd topology/parameter/model/protocol package has
not been provided for import and MLX simulation.

## Exact Next Command

Once the GPCRmd package files are downloaded or mounted locally:

```bash
uv run atomistic-prep run-gpcrmd-mlx \
  --target gpcrmd-729-beta1-5f8u-cyanopindolol \
  --cache <path-to-complete-gpcrmd-package-or-manifest> \
  --out notebooks/ligand-receptor-motion/data/gpcrmd-mlx/5f8u-cyanopindolol \
  --steps 2000 \
  --sample-interval 10 \
  --dt 0.001 \
  --force \
  --json
```

If that command still blocks, the blocker JSON is the next implementation list.
The most likely next blocker is parsing HMR/virtual-site policy from the real
topology/protocol files.
