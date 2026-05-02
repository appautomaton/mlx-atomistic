# Slice 1 Evidence: GPCRmd Target Registry And Selection Gate

## Source Anchors

- GPCRmd data-download docs state that simulation report pages expose outputs, trajectories, topologies, coordinates, protocols, and starting files: https://gpcrmd-docs.readthedocs.io/en/latest/data-download.html
- Primary candidate source page: https://www.gpcrmd.org/dynadb/dynamics/id/729/
- Candidate source fields used: dynamics ID 729, PDB 5F8U, beta-1 adrenergic receptor, orthosteric ligand, TIP3P water, POPC membrane, sodium/chloride ions, 92001 atoms, NVT, ACEMD3/GPUGRID, CHARMM c36 Jul 2020, model/topology/trajectory/parameters/starting-file IDs.

## Slice Output

- Added `atomistic_prep.gpcrmd` for offline registry loading, writing, selection reports, and fail-closed target selection.
- Added tests for the built-in GPCRmd target, offline fixture round-trip, exact missing-requirement errors, and unknown-target errors.

## Scope Boundary

- No GPCRmd downloads.
- No topology import.
- No external MD engine calls.
- No MLX simulation.
- No notebook changes.
