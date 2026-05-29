# DESIGN: MD Integrator and Constraint Parity

## Architecture Approach

Phase 1 should extend existing runtime contracts instead of replacing them. Current public entry points such as `Cell.cubic()`, `Cell.orthorhombic()`, `DistanceConstraints`, `simulate_nvt()`, `simulate_npt()`, checkpoint helpers, reporters, and OpenMM parity scripts remain compatibility anchors.

## Cell And Periodic Geometry

- Keep `Cell` as the single periodic-cell abstraction.
- Add full 3x3 matrix support while preserving the existing `lengths` property for cubic and orthorhombic callers.
- Implement wrapping and minimum-image through fractional coordinates:
  - Cartesian row vectors convert to fractional coordinates with the inverse cell matrix.
  - Wrapped fractional coordinates use `frac - floor(frac)`.
  - Minimum image uses `frac - round(frac)`.
  - Cartesian coordinates convert back through the cell matrix.
- Volume must come from the matrix determinant. Orthorhombic volume remains the product of lengths.
- Artifact, checkpoint, trajectory, and reporter metadata should persist enough cell shape information to distinguish orthorhombic and triclinic boxes.

## Force, Neighbor, PME, And Pressure Contracts

- Force terms should continue to receive `cell: Cell | None` and use `cell.minimum_image(...)`; that is the compatibility boundary for triclinic support.
- Neighbor-list implementations must either support triclinic cells directly or fail closed with a clear blocker for unsupported compact paths.
- PME and virial diagnostics are allowed to support a narrower triclinic subset initially only if unsupported configurations fail before claims are made.
- Pressure and barostat code must compute volume through the `Cell` API, not by assuming `cell.lengths` fully describes the box.

## Constraint And HMR Contracts

- Preserve `DistanceConstraints` as the generic pair-constraint path.
- Add an analytical SETTLE path for recognized water triplets. It should expose the same operational shape as `DistanceConstraints`: position projection, velocity projection, and max-error reporting.
- If multiple constraint implementations are active, expose a small combiner rather than forcing every caller to know about every constraint type.
- HMR should be a deterministic mass transformation with provenance, not a hidden side effect in the integrator. It must preserve total mass and record the original and transformed masses where artifacts or checkpoints need explanation.

## Thermostat And Barostat State

- Keep Langevin BAOAB unchanged.
- Add Nose-Hoover as a separate thermostat object with explicit deterministic state. Reporter and checkpoint payloads must identify the thermostat family and continuation cursor/state.
- Extend MC barostat behavior through explicit modes rather than implicit flags:
  - isotropic existing behavior remains compatible;
  - anisotropic proposes axis-specific scaling;
  - membrane or semi-isotropic keeps membrane-plane and normal-axis policy explicit.
- Barostat proposals must rescale coordinates consistently with the proposed cell matrix and must validate virial support before pressure-coupled claims.

## Validation Strategy

- Use OpenMM as a dev/reference engine only.
- Keep validation fixtures bounded and deterministic.
- Prefer targeted parity tests for each capability, then one small end-to-end proof that exercises minimize -> Nose-Hoover NVT -> anisotropic or membrane MC NPT.
- Do not use this phase to certify the large GPCRmd 729 fixture. Existing evidence says that fixture is blocked by lazy-topology runtime nonbonded pair provisioning.
