# Slice 2: PME Production Readiness

## Status

Complete at decision checkpoint.

## What Changed

- Replaced the PME production gate from `numpy_reference` / blocked to the
  executable `mlx_fft_cic` backend.
- Moved PME energy/force evaluation onto MLX operations for the real-space path,
  CIC charge assignment, mesh FFT reciprocal solve, and atom interpolation.
- Kept readiness fail-closed for missing config, invalid box/cutoff, non-neutral
  systems, invalid exception counts, and systems outside the current runtime
  envelope.
- Added the current runtime envelope to `pme_readiness_report`:
  orthorhombic cells, CIC assignment, and `max_atoms=4096`.
- Exposed the virial status as `finite_difference_cell_strain` with analytic
  PME virial marked unsupported.
- Extended the OpenMM-vs-MLX parity harness with `--pme`, explicit PME config,
  periodic box injection, OpenMM PME parameter matching, and PME readiness in
  the report.

## Fixture

- Fixture: `amber-alanine-dipeptide-implicit`
- AMBER source:
  - `vendors/openmm/wrappers/python/tests/systems/alanine-dipeptide-implicit.prmtop`
  - `vendors/openmm/wrappers/python/tests/systems/alanine-dipeptide-implicit.inpcrd`
- PME box: `40,40,40` Angstrom orthorhombic
- PME mesh: `48,48,48`
- PME alpha: `0.35` Angstrom^-1
- PME real cutoff: `10.0` Angstrom

## Evidence

Command:

```sh
uv run python scripts/run_openmm_mlx_parity.py --fixture amber-alanine-dipeptide-implicit --pme --out results/md-engine-gap-closure/pme-parity
```

Result:

- status: `passed`
- OpenMM method: `PME`
- OpenMM total energy: `-88.11968020163312` kJ/mol
- MLX total energy: `-88.14239501953125` kJ/mol
- total energy abs error: `0.022714817898133788` kJ/mol
- nonbonded component abs error: `0.023510570876212` kJ/mol
- force max abs error: `8.634396488319567` kJ/mol/nm
- force RMS abs error: `2.4919619801391475` kJ/mol/nm
- tolerances: total/component `0.05` kJ/mol, force max `12.0`
  kJ/mol/nm, force RMS `3.0` kJ/mol/nm
- PME readiness: `ready`
- PME backend: `mlx_fft_cic`
- PME blockers: `[]`

Report path:

- `results/md-engine-gap-closure/pme-parity/openmm_mlx_parity_report.json`

Verification:

```sh
uv run pytest tests/test_pme.py tests/test_openmm_mlx_parity.py -q
uv run pytest tests/test_pme.py tests/test_openmm_mlx_parity.py tests/test_production_artifacts.py -q
uv run ruff check src/mlx_atomistic/pme.py scripts/openmm_mlx_parity.py scripts/run_openmm_mlx_parity.py tests/test_pme.py tests/test_openmm_mlx_parity.py tests/test_production_artifacts.py
uv run pytest -q
```

All verification commands passed.

## Decision Checkpoint

Slice 2 is complete for the selected fixture family. PME is now executable and
OpenMM-parity checked for a small orthorhombic AMBER fixture. Larger systems,
triclinic boxes, and analytic PME virial are still outside this slice and should
stay explicit future scope.

Next planned slice: Slice 3, runtime reporters and diagnostics.
