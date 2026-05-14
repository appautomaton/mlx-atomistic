# Slice 1 Evidence: GPCRmd Target Inventory Gate

## Target Decision

- Status: fixed selected target.
- Target: `gpcrmd-729-beta1-5f8u-cyanopindolol`.
- Dynamics ID: `729`.
- PDB ID: `5F8U`.
- Receptor: Beta-1 adrenergic receptor.
- Ligand: `4-{[(2s)-3-(tert-butylamino)-2-hydroxypropyl]oxy}-3h-indole-2-carbonitrile`.
- Reason: this is the GPCRmd-critical explicit-solvent membrane target already selected by the registry. It exposes the engine capabilities the readiness plan must gate: PME, CHARMM36, POPC membrane/lipids, constraints/HMR or virtual-site policy, exceptions, and GPCRmd-scale nonbonded execution.
- Replacement target: none. A lower-blocker soluble target would not exercise the membrane/PME/lipid readiness path this change is scoped to prove.

## Required GPCRmd Package Files For MLX Import

- Topology file: GPCRmd file ID `15286`, role `topology`.
- Model/coordinates file: GPCRmd file ID `17686`, role `model`.
- Parameters file: GPCRmd file ID `15290`, role `parameters`.
- Protocol/starting files: GPCRmd file ID `17687`, role `protocol`.

Reference trajectories are optional analysis/comparison features, not MLX runtime inputs:

- Trajectory replica 1: GPCRmd file ID `15287`.
- Trajectory replica 2: GPCRmd file ID `15288`.
- Trajectory replica 3: GPCRmd file ID `15289`.

## System Inventory

- Total atoms: `92001`.
- Water model: `TIP3P`, water count `19944`.
- Lipid/membrane model: homogeneous `POPC`, lipid count `200`.
- Ions: sodium `57`, chloride `74`.
- Force field: `CHARMM c36 Jul 2020`.
- Software/protocol source: `ACEMD3, GPUGRID`.
- Ensemble: `NVT`.
- Timestep: `4.0 fs`.
- Replicates: `3`.
- Reference frame stride: `0.2 ns`.
- Accumulated reference time: `1.5 us`.
- Periodic box: required; vectors must come from model/protocol package files.
- Constraints: required; topology/protocol must define constrained bonds and any HMR or virtual-site policy implied by the 4 fs timestep.
- Exceptions: required; topology/parameters must define exclusions, 1-4 exceptions, and CHARMM pair overrides or NBFIX entries if present.

## Required Terms

- `pme_mesh_periodic_electrostatics`
- `charmm36_bonded_and_nonbonded_parameters`
- `charmm_cmap_terms`
- `membrane_lipid_force_field_terms`
- `popc_lipid_topology_and_parameters`
- `tip3p_water_model`
- `nonbonded_exclusions_and_1_4_exceptions`

## Optional Analysis Features

- GPCRmd reference trajectory comparison.
- Replica-level reference statistics.
- Frame-stride/reference-time analysis.

These are not accepted as substitutes for an MLX-generated trajectory.

## First Engine Blockers

- `pme_mesh_periodic_electrostatics`: first owned by Slice 2 and Slice 5.
- `membrane_lipid_force_field_terms`: first owned by Slice 3.
- `popc_lipid_topology_and_parameters`: first owned by Slice 3 and Slice 7.
- `charmm_cmap_terms`: first owned by Slice 3.
- `large_periodic_system_neighbor_list_scaling`: first owned by Slice 4.
- `virtual_sites_or_hydrogen_mass_repartitioning_not_checked`: first owned by Slice 6 and Slice 7.

No generic `production validation` blocker is used for this gate.

## Verification

- Manifest entries now require an existing path to count as present. Pathless or nonexistent required inputs remain visible in inspection JSON but appear as `present: false` and contribute `mlx_compatibility.missing_input`; optional reference trajectories remain optional.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests/test_gpcrmd_registry.py`: `18 passed`.
- `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run pytest tests -k "gpcrmd and inventory"`: `3 passed, 194 deselected`.
- Fixture API smoke: `uv run mlx_atomistic.prep Python API gpcrmd-inspect --target gpcrmd-729-beta1-5f8u-cyanopindolol --cache <fixture-cache> --compatibility --json` emitted `complete: true`, `mlx_readiness_inventory`, `pme_mesh_periodic_electrostatics`, and `reference_trajectory_comparison`.
- Targeted Ruff: `UV_CACHE_DIR=/tmp/mlx-atomistic-uv-cache uv run ruff check src/mlx_atomistic/prep/gpcrmd.py tests/test_gpcrmd_registry.py`: passed.
