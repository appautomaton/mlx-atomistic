"""Symmetry-reduced finite-displacement phonons for periodic DFT systems."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes
from mlx_atomistic.dft._periodic_artifact_contracts import (
    periodic_scf_calculation_contract,
)
from mlx_atomistic.dft._periodic_models import PeriodicDFTSystem, PeriodicSCFConfig
from mlx_atomistic.dft._periodic_phonon_models import (
    PeriodicDisplacementMember,
    PeriodicDisplacementOrbit,
    PeriodicDisplacementPlan,
    PeriodicPhononConfig,
    PeriodicPhononConvergenceResult,
    PeriodicPhononResult,
    PeriodicPhononSample,
    PeriodicPhononSampleSet,
    PeriodicPhononSymmetry,
)
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import KPointMesh
from mlx_atomistic.dft.periodic_forces import periodic_scf_forces
from mlx_atomistic.dft.periodic_scf import run_periodic_scf
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional

ATOMIC_MASS_UNIT_TO_ELECTRON_MASS = 1822.888486209
HARTREE_TO_WAVENUMBER_CM1 = 219474.6313632


def _identity_symmetry() -> PeriodicPhononSymmetry:
    return PeriodicPhononSymmetry(np.eye(3), label="identity")


def _operation_key(operation: PeriodicPhononSymmetry) -> tuple[int, ...]:
    rotation = np.rint(operation.rotation_cartesian * 1.0e12).astype(np.int64)
    translation = np.remainder(operation.translation_fractional, 1.0)
    translation[np.isclose(translation, 1.0, atol=1.0e-12, rtol=0.0)] = 0.0
    translated = np.rint(translation * 1.0e12).astype(np.int64)
    return tuple(int(value) for value in (*rotation.flat, *translated.flat))


def _symmetry_operations(
    operations: Sequence[PeriodicPhononSymmetry],
) -> tuple[PeriodicPhononSymmetry, ...]:
    values = [_identity_symmetry()]
    seen = {_operation_key(values[0])}
    for operation in operations:
        if not isinstance(operation, PeriodicPhononSymmetry):
            raise TypeError("phonon symmetries must be PeriodicPhononSymmetry values")
        key = _operation_key(operation)
        if key not in seen:
            seen.add(key)
            values.append(operation)
    return tuple(values)


def _axis_mapping(rotation: np.ndarray) -> tuple[tuple[int, int], ...]:
    mapping = []
    for source_axis in range(3):
        direction = np.eye(3)[source_axis] @ rotation.T
        target_axis = int(np.argmax(np.abs(direction)))
        sign = 1 if direction[target_axis] > 0.0 else -1
        expected = np.zeros(3)
        expected[target_axis] = sign
        if not np.allclose(direction, expected, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "phonon symmetry rotations must map Cartesian axes by signed permutation"
            )
        mapping.append((target_axis, sign))
    return tuple(mapping)


def _atom_mapping(
    system: PeriodicDFTSystem,
    operation: PeriodicPhononSymmetry,
    *,
    position_tolerance_bohr: float,
) -> tuple[int, ...]:
    cell = np.asarray(system.grid.cell.matrix, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    fractional_rotation = cell @ operation.rotation_cartesian.T @ inverse
    rounded = np.rint(fractional_rotation)
    if not np.allclose(
        fractional_rotation,
        rounded,
        atol=1.0e-12,
        rtol=0.0,
    ) or abs(round(float(np.linalg.det(rounded)))) != 1:
        raise ValueError("phonon symmetry does not preserve the direct lattice")
    fractional = np.asarray(system.positions, dtype=np.float64) @ inverse
    transformed = (
        np.asarray(system.positions, dtype=np.float64)
        @ operation.rotation_cartesian.T
        @ inverse
        + operation.translation_fractional
    )
    mapping = []
    available = set(range(system.ion_count))
    for source_index, transformed_position in enumerate(transformed):
        matches = []
        for target_index in available:
            if system.symbols[target_index] != system.symbols[source_index]:
                continue
            difference = transformed_position - fractional[target_index]
            difference -= np.rint(difference)
            distance = float(np.linalg.norm(difference @ cell))
            if distance <= position_tolerance_bohr:
                matches.append(target_index)
        if len(matches) != 1:
            raise ValueError(
                "phonon symmetry must map every atom uniquely within its element"
            )
        target = matches[0]
        mapping.append(target)
        available.remove(target)
    return tuple(mapping)


def _dof_action(
    atom_mapping: Sequence[int],
    axis_mapping: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (3 * atom_mapping[atom] + axis_mapping[axis][0], axis_mapping[axis][1])
        for atom in range(len(atom_mapping))
        for axis in range(3)
    )


def _compose_actions(
    first: Sequence[tuple[int, int]],
    second: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            first[second_target][0],
            second_sign * first[second_target][1],
        )
        for second_target, second_sign in second
    )


def plan_periodic_phonon_displacements(
    system: PeriodicDFTSystem,
    *,
    config: PeriodicPhononConfig | None = None,
    symmetry_operations: Sequence[PeriodicPhononSymmetry] = (),
    position_tolerance_bohr: float = 1.0e-6,
) -> PeriodicDisplacementPlan:
    """Build a validated symmetry-independent Cartesian displacement plan.

    Args:
        system: Periodic DFT system at its equilibrium geometry.
        config: Phonon numerical controls. Defaults to `PeriodicPhononConfig`.
        symmetry_operations: Explicit affine crystal symmetries. Identity is
            always added.
        position_tolerance_bohr: Atomic symmetry matching tolerance in bohr.

    Returns:
        Fingerprint-bound displacement orbits covering all Cartesian degrees
        of freedom.

    Raises:
        TypeError: If inputs use unsupported types.
        ValueError: If the system, tolerance, symmetries, or group action fail.
    """

    if not isinstance(system, PeriodicDFTSystem):
        raise TypeError("system must be PeriodicDFTSystem")
    controls = PeriodicPhononConfig() if config is None else config
    if not isinstance(controls, PeriodicPhononConfig):
        raise TypeError("config must be PeriodicPhononConfig")
    if (
        isinstance(position_tolerance_bohr, (bool, np.bool_))
        or not np.isfinite(position_tolerance_bohr)
        or position_tolerance_bohr <= 0.0
    ):
        raise ValueError("position_tolerance_bohr must be finite and positive")
    operations = _symmetry_operations(symmetry_operations)
    atom_mappings = tuple(
        _atom_mapping(
            system,
            operation,
            position_tolerance_bohr=float(position_tolerance_bohr),
        )
        for operation in operations
    )
    axis_mappings = tuple(
        _axis_mapping(operation.rotation_cartesian) for operation in operations
    )
    actions = tuple(
        _dof_action(atom_mapping, axis_mapping)
        for atom_mapping, axis_mapping in zip(
            atom_mappings,
            axis_mappings,
            strict=True,
        )
    )
    action_keys = {tuple(value for pair in action for value in pair) for action in actions}
    if len(action_keys) != len(actions):
        raise ValueError("phonon symmetries must have unique degree-of-freedom actions")
    for first in actions:
        for second in actions:
            composed = _compose_actions(first, second)
            key = tuple(value for pair in composed for value in pair)
            if key not in action_keys:
                raise ValueError("phonon symmetry operations must form a closed group")
    remaining = set(range(3 * system.ion_count))
    orbits = []
    while remaining:
        representative = min(remaining)
        by_target: dict[int, PeriodicDisplacementMember] = {}
        for operation_index, action in enumerate(actions):
            target, sign = action[representative]
            by_target.setdefault(
                target,
                PeriodicDisplacementMember(target, operation_index, sign),
            )
        members = tuple(by_target[target] for target in sorted(by_target))
        orbit_targets = {member.dof_index for member in members}
        if not orbit_targets.issubset(remaining):
            raise ValueError("phonon symmetry orbits overlap inconsistently")
        remaining -= orbit_targets
        orbits.append(PeriodicDisplacementOrbit(representative, members))
    return PeriodicDisplacementPlan(
        system_fingerprint=system.fingerprint,
        symbols=system.symbols,
        displacement_bohr=controls.displacement_bohr,
        symmetry_operations=operations,
        atom_mappings=atom_mappings,
        axis_mappings=axis_mappings,
        orbits=tuple(orbits),
    )


def periodic_phonon_displaced_system(
    system: PeriodicDFTSystem,
    plan: PeriodicDisplacementPlan,
    representative_dof: int,
    direction_sign: int,
) -> PeriodicDFTSystem:
    """Return one plus or minus representative displaced periodic system.

    Args:
        system: Exact equilibrium system used to build ``plan``.
        plan: Matching displacement plan.
        representative_dof: Planned independent Cartesian degree of freedom.
        direction_sign: Minus one or plus one.

    Returns:
        New fixed-cell system with one Cartesian coordinate displaced.
    """

    if not isinstance(system, PeriodicDFTSystem):
        raise TypeError("system must be PeriodicDFTSystem")
    if not isinstance(plan, PeriodicDisplacementPlan):
        raise TypeError("plan must be PeriodicDisplacementPlan")
    if system.fingerprint != plan.system_fingerprint:
        raise ValueError("phonon displacement plan does not match the periodic system")
    if representative_dof not in plan.representative_dofs:
        raise ValueError("representative_dof is not planned for explicit evaluation")
    if direction_sign not in {-1, 1}:
        raise ValueError("direction_sign must be -1 or 1")
    positions = np.array(system.positions, dtype=np.float64, copy=True)
    atom, axis = divmod(representative_dof, 3)
    positions[atom, axis] += direction_sign * plan.displacement_bohr
    return system.with_positions(positions)


def evaluate_periodic_phonon_sample(
    system: PeriodicDFTSystem,
    plan: PeriodicDisplacementPlan,
    representative_dof: int,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    scf_config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicPhononSample:
    """Evaluate both central-force signs for one planned displacement.

    Args:
        system: Equilibrium periodic system matching ``plan``.
        plan: Symmetry-independent displacement plan.
        representative_dof: Planned degree of freedom to evaluate.
        cutoff_hartree: Plane-wave kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate electronic k-point mesh.
        n_bands: Optional computed band count.
        scf_config: Optional periodic SCF controls.
        xc_functional: Optional exchange-correlation functional.
        observer: Optional shared runtime observer for both signs.

    Returns:
        Fingerprinted minus/plus analytic-force sample.

    Raises:
        ValueError: If either displaced SCF does not converge.
    """

    forces: dict[int, np.ndarray] = {}
    fingerprints: dict[int, str] = {}
    for direction_sign in (-1, 1):
        displaced = periodic_phonon_displaced_system(
            system,
            plan,
            representative_dof,
            direction_sign,
        )
        calculation = periodic_scf_calculation_contract(
            displaced,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=scf_config,
            xc_functional=xc_functional,
        )
        result = run_periodic_scf(
            displaced,
            cutoff_hartree=cutoff_hartree,
            kpoint_mesh=kpoint_mesh,
            n_bands=n_bands,
            config=scf_config,
            xc_functional=xc_functional,
            observer=observer,
        )
        if not result.converged:
            raise ValueError(
                "periodic phonon displacement SCF did not converge: "
                f"dof={representative_dof} sign={direction_sign}"
            )
        force = periodic_scf_forces(displaced, result)
        forces[direction_sign] = np.asarray(force.forces, dtype=np.float64)
        fingerprints[direction_sign] = sha256_bytes(
            canonical_json_bytes(calculation)
        )
    return PeriodicPhononSample(
        representative_dof=representative_dof,
        minus_forces_hartree_per_bohr=forces[-1],
        plus_forces_hartree_per_bohr=forces[1],
        minus_calculation_fingerprint=fingerprints[-1],
        plus_calculation_fingerprint=fingerprints[1],
    )


def _force_transform(
    plan: PeriodicDisplacementPlan,
    operation_index: int,
) -> np.ndarray:
    transform = np.zeros((plan.dof_count, plan.dof_count), dtype=np.float64)
    atom_mapping = plan.atom_mappings[operation_index]
    axis_mapping = plan.axis_mappings[operation_index]
    for atom in range(plan.atom_count):
        for axis in range(3):
            target_axis, sign = axis_mapping[axis]
            source = 3 * atom + axis
            target = 3 * atom_mapping[atom] + target_axis
            transform[target, source] = sign
    return transform


def _translation_basis(atom_count: int) -> np.ndarray:
    translations = np.zeros((3 * atom_count, 3), dtype=np.float64)
    for atom in range(atom_count):
        translations[3 * atom : 3 * atom + 3] = np.eye(3)
    return translations


def _signed_frequencies(eigenvalues: np.ndarray) -> np.ndarray:
    return (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * HARTREE_TO_WAVENUMBER_CM1
    )


def assemble_periodic_phonons(
    plan: PeriodicDisplacementPlan,
    samples: PeriodicPhononSampleSet,
    masses_amu: Sequence[float],
    *,
    config: PeriodicPhononConfig | None = None,
) -> PeriodicPhononResult:
    """Assemble raw force constants and admitted Gamma-point phonon modes.

    Args:
        plan: Symmetry-independent displacement plan.
        samples: Complete central-force samples bound to ``plan``.
        masses_amu: Positive per-atom masses in atomic mass units.
        config: Numerical diagnostics. Its displacement must match ``plan``.

    Returns:
        Raw force constants, diagnostics, and modes only when raw force-
        constant gates pass.

    Raises:
        TypeError: If inputs use unsupported types.
        ValueError: If samples, masses, displacement, or reconstruction differ.
    """

    if not isinstance(plan, PeriodicDisplacementPlan):
        raise TypeError("plan must be PeriodicDisplacementPlan")
    if not isinstance(samples, PeriodicPhononSampleSet):
        raise TypeError("samples must be PeriodicPhononSampleSet")
    controls = (
        PeriodicPhononConfig(displacement_bohr=plan.displacement_bohr)
        if config is None
        else config
    )
    if not isinstance(controls, PeriodicPhononConfig):
        raise TypeError("config must be PeriodicPhononConfig")
    if not np.isclose(
        controls.displacement_bohr,
        plan.displacement_bohr,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("phonon config displacement differs from the plan")
    missing = samples.missing_representatives(plan)
    if missing:
        raise ValueError(f"phonon samples are incomplete: {list(missing)}")
    masses = np.asarray(masses_amu, dtype=np.float64)
    if (
        masses.shape != (plan.atom_count,)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0.0)
    ):
        raise ValueError("masses_amu must be finite, positive, and match the plan")
    by_dof = {sample.representative_dof: sample for sample in samples.samples}
    force_constants = np.empty((plan.dof_count, plan.dof_count), dtype=np.float64)
    assigned: set[int] = set()
    for orbit in plan.orbits:
        sample = by_dof[orbit.representative_dof]
        representative_column = -(
            sample.plus_forces_hartree_per_bohr
            - sample.minus_forces_hartree_per_bohr
        ).reshape(-1) / (2.0 * plan.displacement_bohr)
        for member in orbit.members:
            transform = _force_transform(plan, member.operation_index)
            force_constants[:, member.dof_index] = (
                member.direction_sign * transform @ representative_column
            )
            assigned.add(member.dof_index)
    if assigned != set(range(plan.dof_count)) or not np.isfinite(force_constants).all():
        raise ValueError("phonon force-constant reconstruction is incomplete")
    reciprocity = float(np.max(np.abs(force_constants - force_constants.T)))
    translations = _translation_basis(plan.atom_count)
    right_sum_rule = float(np.max(np.abs(force_constants @ translations)))
    left_sum_rule = float(np.max(np.abs(translations.T @ force_constants)))
    force_constants_passed = bool(
        reciprocity <= controls.reciprocity_tolerance_hartree_per_bohr2
        and right_sum_rule <= controls.sum_rule_tolerance_hartree_per_bohr2
        and left_sum_rule <= controls.sum_rule_tolerance_hartree_per_bohr2
    )
    common = {
        "plan_fingerprint": plan.fingerprint,
        "system_fingerprint": plan.system_fingerprint,
        "displacement_bohr": plan.displacement_bohr,
        "masses_amu": masses,
        "force_constants_hartree_per_bohr2": force_constants,
        "reciprocity_residual_hartree_per_bohr2": reciprocity,
        "right_sum_rule_residual_hartree_per_bohr2": right_sum_rule,
        "left_sum_rule_residual_hartree_per_bohr2": left_sum_rule,
        "force_constants_passed": force_constants_passed,
    }
    if not force_constants_passed:
        return PeriodicPhononResult(**common)
    symmetric = 0.5 * (force_constants + force_constants.T)
    mass_dof = np.repeat(masses * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS, 3)
    dynamical = symmetric / np.sqrt(np.outer(mass_dof, mass_dof))
    eigenvalues, mass_weighted = np.linalg.eigh(dynamical)
    frequencies = _signed_frequencies(eigenvalues)
    cartesian = mass_weighted / np.sqrt(mass_dof)[:, None]
    norms = np.linalg.norm(cartesian, axis=0)
    cartesian /= norms[None, :]
    acoustic_indices = tuple(
        int(value) for value in np.sort(np.argsort(np.abs(eigenvalues))[:3])
    )
    acoustic_max = float(np.max(np.abs(frequencies[list(acoustic_indices)])))
    mass_translations = translations * np.sqrt(mass_dof)[:, None]
    mass_translations, _ = np.linalg.qr(mass_translations)
    acoustic_vectors = mass_weighted[:, list(acoustic_indices)]
    overlaps = np.linalg.svd(
        mass_translations.T @ acoustic_vectors,
        compute_uv=False,
    )
    minimum_overlap = float(np.min(overlaps))
    acoustic_passed = bool(
        acoustic_max <= controls.acoustic_frequency_tolerance_cm1
        and minimum_overlap >= controls.acoustic_translation_overlap_minimum
    )
    optical_indices = sorted(set(range(plan.dof_count)) - set(acoustic_indices))
    stable = bool(
        not optical_indices
        or np.min(frequencies[optical_indices])
        >= -controls.imaginary_frequency_tolerance_cm1
    )
    return PeriodicPhononResult(
        **common,
        dynamical_eigenvalues_au=eigenvalues,
        frequencies_cm1=frequencies,
        mass_weighted_eigenvectors=mass_weighted,
        cartesian_eigenvectors=cartesian,
        acoustic_mode_indices=acoustic_indices,
        acoustic_max_abs_frequency_cm1=acoustic_max,
        acoustic_translation_overlap_minimum=minimum_overlap,
        acoustic_passed=acoustic_passed,
        stable=stable,
        valid=acoustic_passed and stable,
    )


def compare_periodic_phonon_displacements(
    coarse: PeriodicPhononResult,
    fine: PeriodicPhononResult,
    *,
    config: PeriodicPhononConfig | None = None,
) -> PeriodicPhononConvergenceResult:
    """Compare complete phonon modes at two displacement magnitudes.

    Args:
        coarse: Result at the larger displacement.
        fine: Result at the smaller displacement.
        config: Convergence tolerances. Defaults to `PeriodicPhononConfig`.

    Returns:
        Maximum frequency and eigenvalue drifts with aggregate status.
    """

    if not isinstance(coarse, PeriodicPhononResult) or not isinstance(
        fine,
        PeriodicPhononResult,
    ):
        raise TypeError("coarse and fine must be PeriodicPhononResult values")
    controls = PeriodicPhononConfig() if config is None else config
    if not isinstance(controls, PeriodicPhononConfig):
        raise TypeError("config must be PeriodicPhononConfig")
    if coarse.system_fingerprint != fine.system_fingerprint:
        raise ValueError("phonon convergence results must use the same system")
    if not coarse.valid or not fine.valid:
        raise ValueError("phonon convergence requires two valid mode results")
    if coarse.displacement_bohr <= fine.displacement_bohr:
        raise ValueError("coarse displacement must exceed fine displacement")
    if (
        coarse.frequencies_cm1 is None
        or fine.frequencies_cm1 is None
        or coarse.dynamical_eigenvalues_au is None
        or fine.dynamical_eigenvalues_au is None
        or coarse.frequencies_cm1.shape != fine.frequencies_cm1.shape
    ):
        raise ValueError("phonon convergence mode arrays are inconsistent")
    frequency_drift = float(
        np.max(np.abs(coarse.frequencies_cm1 - fine.frequencies_cm1))
    )
    eigenvalue_drift = float(
        np.max(
            np.abs(
                coarse.dynamical_eigenvalues_au
                - fine.dynamical_eigenvalues_au
            )
        )
    )
    passed = bool(
        frequency_drift <= controls.frequency_convergence_tolerance_cm1
        and eigenvalue_drift <= controls.eigenvalue_convergence_tolerance_au
    )
    return PeriodicPhononConvergenceResult(
        coarse_displacement_bohr=coarse.displacement_bohr,
        fine_displacement_bohr=fine.displacement_bohr,
        maximum_frequency_drift_cm1=frequency_drift,
        maximum_eigenvalue_drift_au=eigenvalue_drift,
        frequency_tolerance_cm1=controls.frequency_convergence_tolerance_cm1,
        eigenvalue_tolerance_au=controls.eigenvalue_convergence_tolerance_au,
        passed=passed,
    )


__all__ = [
    "ATOMIC_MASS_UNIT_TO_ELECTRON_MASS",
    "HARTREE_TO_WAVENUMBER_CM1",
    "PeriodicDisplacementMember",
    "PeriodicDisplacementOrbit",
    "PeriodicDisplacementPlan",
    "PeriodicPhononConfig",
    "PeriodicPhononConvergenceResult",
    "PeriodicPhononResult",
    "PeriodicPhononSample",
    "PeriodicPhononSampleSet",
    "PeriodicPhononSymmetry",
    "assemble_periodic_phonons",
    "compare_periodic_phonon_displacements",
    "evaluate_periodic_phonon_sample",
    "periodic_phonon_displaced_system",
    "plan_periodic_phonon_displacements",
]
