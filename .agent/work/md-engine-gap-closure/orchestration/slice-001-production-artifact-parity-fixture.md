# Slice 001 Execution Evidence

## Route

Direct execution.

## Fixture

- Selected fixture: `amber-alanine-dipeptide-implicit`
- Source files:
  - `vendors/openmm/wrappers/python/tests/systems/alanine-dipeptide-implicit.prmtop`
  - `vendors/openmm/wrappers/python/tests/systems/alanine-dipeptide-implicit.inpcrd`
- Source kind: AMBER
- Atom count: 22

## Result

The fixed-coordinate OpenMM-vs-MLX parity check passed.

- OpenMM total energy: `-88.08858858077855` kJ/mol
- MLX total energy: `-88.08782958984375` kJ/mol
- Total energy absolute error: `0.0007589909347984758` kJ/mol
- Force max absolute error: `9.047958696432033` kJ/mol/nm
- Force RMS absolute error: `1.757840114951591` kJ/mol/nm
- Unsupported terms: none
- Local report: `results/md-engine-gap-closure/parity-fixture/openmm_mlx_parity_report.json`

## Implementation Notes

- Added a reference-script parity helper under `scripts/openmm_mlx_parity.py`.
- Added `scripts/run_openmm_mlx_parity.py` as the repeatable CLI entrypoint.
- Kept OpenMM imports out of `src/mlx_atomistic` to preserve the runtime boundary.
- Fixed the Angstrom-space Coulomb constant from `COULOMB_CONSTANT_KJ_MOL_NM / 10`
  to `COULOMB_CONSTANT_KJ_MOL_NM * 10`; the previous value caused pre-PME
  AMBER nonbonded parity to fail by 100x in electrostatics.

## Verification

```sh
uv run pytest tests/test_openmm_mlx_parity.py
```

Result: `3 passed in 0.20s`

```sh
uv run python scripts/run_openmm_mlx_parity.py --fixture amber-alanine-dipeptide-implicit --out results/md-engine-gap-closure/parity-fixture
```

Result: exit code `0`, report status `passed`.

```sh
uv run ruff check scripts/openmm_mlx_parity.py scripts/run_openmm_mlx_parity.py tests/test_openmm_mlx_parity.py tests/test_runtime_boundaries.py tests/test_units.py src/mlx_atomistic/units.py src/mlx_atomistic/artifacts.py
```

Result: `All checks passed!`

```sh
uv run pytest
```

Result: `354 passed in 26.36s`

## Checkpoint

Slice 1 reached its planned decision checkpoint. Since parity passes for the
small AMBER fixture and no unsupported terms were reported, Slice 2 can start:
PME production readiness.
