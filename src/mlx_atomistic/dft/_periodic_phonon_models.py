"""Data contracts for periodic finite-displacement phonons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlx_atomistic._artifact_identity import canonical_json_bytes, sha256_bytes


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _positive_finite(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class PeriodicPhononConfig:
    """Numerical gates for Gamma-point finite-displacement phonons.

    Args:
        displacement_bohr: Positive central-displacement magnitude in bohr.
        reciprocity_tolerance_hartree_per_bohr2: Maximum raw Hessian
            antisymmetry.
        sum_rule_tolerance_hartree_per_bohr2: Maximum left or right
            translational residual.
        acoustic_frequency_tolerance_cm1: Maximum absolute acoustic frequency.
        acoustic_translation_overlap_minimum: Minimum singular overlap with the
            exact mass-weighted translation subspace.
        imaginary_frequency_tolerance_cm1: Admitted non-acoustic numerical
            imaginary-frequency magnitude.
        frequency_convergence_tolerance_cm1: Maximum mode-frequency drift
            between displacement magnitudes.
        eigenvalue_convergence_tolerance_au: Maximum dynamical eigenvalue drift
            in atomic units.
    """

    displacement_bohr: float = 0.01
    reciprocity_tolerance_hartree_per_bohr2: float = 5.0e-4
    sum_rule_tolerance_hartree_per_bohr2: float = 5.0e-4
    acoustic_frequency_tolerance_cm1: float = 10.0
    acoustic_translation_overlap_minimum: float = 0.9
    imaginary_frequency_tolerance_cm1: float = 5.0
    frequency_convergence_tolerance_cm1: float = 5.0
    eigenvalue_convergence_tolerance_au: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "displacement_bohr",
            "reciprocity_tolerance_hartree_per_bohr2",
            "sum_rule_tolerance_hartree_per_bohr2",
            "acoustic_frequency_tolerance_cm1",
            "imaginary_frequency_tolerance_cm1",
            "frequency_convergence_tolerance_cm1",
            "eigenvalue_convergence_tolerance_au",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        overlap = self.acoustic_translation_overlap_minimum
        if (
            isinstance(overlap, (bool, np.bool_))
            or not np.isfinite(overlap)
            or not 0.0 < overlap <= 1.0
        ):
            raise ValueError(
                "acoustic_translation_overlap_minimum must lie in (0, 1]"
            )
        object.__setattr__(
            self,
            "acoustic_translation_overlap_minimum",
            float(overlap),
        )

    def to_dict(self) -> dict[str, float]:
        """Return the canonical JSON-safe numerical controls."""

        return {
            "displacement_bohr": self.displacement_bohr,
            "reciprocity_tolerance_hartree_per_bohr2": (
                self.reciprocity_tolerance_hartree_per_bohr2
            ),
            "sum_rule_tolerance_hartree_per_bohr2": (
                self.sum_rule_tolerance_hartree_per_bohr2
            ),
            "acoustic_frequency_tolerance_cm1": (
                self.acoustic_frequency_tolerance_cm1
            ),
            "acoustic_translation_overlap_minimum": (
                self.acoustic_translation_overlap_minimum
            ),
            "imaginary_frequency_tolerance_cm1": (
                self.imaginary_frequency_tolerance_cm1
            ),
            "frequency_convergence_tolerance_cm1": (
                self.frequency_convergence_tolerance_cm1
            ),
            "eigenvalue_convergence_tolerance_au": (
                self.eigenvalue_convergence_tolerance_au
            ),
        }


@dataclass(frozen=True)
class PeriodicPhononSymmetry:
    """Explicit affine crystal symmetry used for displacement reduction.

    Args:
        rotation_cartesian: Orthogonal Cartesian column-vector operation.
        translation_fractional: Fractional translation after rotation.
        label: Optional stable human-readable operation label.
    """

    rotation_cartesian: np.ndarray = field(repr=False, compare=False)
    translation_fractional: np.ndarray = field(repr=False, compare=False)
    label: str = ""

    def __init__(
        self,
        rotation_cartesian: object,
        translation_fractional: object = (0.0, 0.0, 0.0),
        *,
        label: str = "",
    ):
        rotation = np.array(rotation_cartesian, dtype=np.float64, copy=True)
        translation = np.array(
            translation_fractional,
            dtype=np.float64,
            copy=True,
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("phonon symmetry rotation must be a finite 3 x 3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12, rtol=0.0):
            raise ValueError("phonon symmetry rotation must be orthogonal")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("phonon symmetry translation must be a finite vector")
        if not isinstance(label, str):
            raise TypeError("phonon symmetry label must be a string")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation_cartesian", rotation)
        object.__setattr__(self, "translation_fractional", translation)
        object.__setattr__(self, "label", label)

    def to_dict(self) -> dict[str, object]:
        """Return the affine operation as JSON-safe metadata."""

        return {
            "rotation_cartesian": self.rotation_cartesian.tolist(),
            "translation_fractional": self.translation_fractional.tolist(),
            "label": self.label,
        }


@dataclass(frozen=True)
class PeriodicDisplacementMember:
    """One degree of freedom reconstructed from an orbit representative."""

    dof_index: int
    operation_index: int
    direction_sign: int

    def __post_init__(self) -> None:
        if type(self.dof_index) is not int or self.dof_index < 0:
            raise ValueError("displacement member dof_index must be non-negative")
        if type(self.operation_index) is not int or self.operation_index < 0:
            raise ValueError("displacement operation_index must be non-negative")
        if self.direction_sign not in {-1, 1}:
            raise ValueError("displacement direction_sign must be -1 or 1")

    def to_dict(self) -> dict[str, int]:
        """Return the reconstruction member as JSON-safe metadata."""

        return {
            "dof_index": self.dof_index,
            "operation_index": self.operation_index,
            "direction_sign": self.direction_sign,
        }


@dataclass(frozen=True)
class PeriodicDisplacementOrbit:
    """One symmetry-independent displacement and its reconstructed members."""

    representative_dof: int
    members: tuple[PeriodicDisplacementMember, ...]

    def __post_init__(self) -> None:
        if type(self.representative_dof) is not int or self.representative_dof < 0:
            raise ValueError("representative_dof must be non-negative")
        if not self.members:
            raise ValueError("displacement orbit must contain at least one member")
        indices = tuple(member.dof_index for member in self.members)
        if len(set(indices)) != len(indices) or self.representative_dof not in indices:
            raise ValueError(
                "displacement orbit members must be unique and include its representative"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the displacement orbit as JSON-safe metadata."""

        return {
            "representative_dof": self.representative_dof,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class PeriodicDisplacementPlan:
    """Fingerprint-bound symmetry-independent periodic displacement plan."""

    system_fingerprint: str
    symbols: tuple[str, ...]
    displacement_bohr: float
    symmetry_operations: tuple[PeriodicPhononSymmetry, ...]
    atom_mappings: tuple[tuple[int, ...], ...]
    axis_mappings: tuple[tuple[tuple[int, int], ...], ...]
    orbits: tuple[PeriodicDisplacementOrbit, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.system_fingerprint):
            raise ValueError("phonon plan system_fingerprint must be lowercase SHA-256")
        symbols = tuple(str(symbol) for symbol in self.symbols)
        if not symbols or any(not symbol for symbol in symbols):
            raise ValueError("phonon plan symbols must be non-empty")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "displacement_bohr",
            _positive_finite(self.displacement_bohr, "displacement_bohr"),
        )
        operation_count = len(self.symmetry_operations)
        if operation_count == 0:
            raise ValueError("phonon plan requires at least the identity symmetry")
        if len(self.atom_mappings) != operation_count or len(self.axis_mappings) != operation_count:
            raise ValueError("phonon plan symmetry mappings are incomplete")
        atom_count = len(symbols)
        for mapping in self.atom_mappings:
            if tuple(sorted(mapping)) != tuple(range(atom_count)):
                raise ValueError("phonon plan atom mappings must be bijections")
        for mapping in self.axis_mappings:
            if len(mapping) != 3 or tuple(sorted(axis for axis, _sign in mapping)) != (0, 1, 2):
                raise ValueError("phonon plan axis mappings must be signed permutations")
            if any(sign not in {-1, 1} for _axis, sign in mapping):
                raise ValueError("phonon plan axis mapping signs must be -1 or 1")
        member_dofs = [member.dof_index for orbit in self.orbits for member in orbit.members]
        if sorted(member_dofs) != list(range(3 * atom_count)):
            raise ValueError("phonon displacement orbits must partition all degrees of freedom")
        if tuple(sorted(self.representative_dofs)) != self.representative_dofs:
            raise ValueError("phonon displacement representatives must be sorted")

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the planned periodic cell."""

        return len(self.symbols)

    @property
    def dof_count(self) -> int:
        """Return the number of Cartesian degrees of freedom."""

        return 3 * self.atom_count

    @property
    def representative_dofs(self) -> tuple[int, ...]:
        """Return sorted symmetry-independent displaced degrees of freedom."""

        return tuple(orbit.representative_dof for orbit in self.orbits)

    def identity_dict(self) -> dict[str, object]:
        """Return the canonical plan identity without its derived fingerprint."""

        return {
            "schema_version": "mlx-atomistic.periodic-phonon-plan.v1",
            "system_fingerprint": self.system_fingerprint,
            "symbols": list(self.symbols),
            "displacement_bohr": self.displacement_bohr,
            "symmetry_operations": [
                operation.to_dict() for operation in self.symmetry_operations
            ],
            "atom_mappings": [list(mapping) for mapping in self.atom_mappings],
            "axis_mappings": [
                [[axis, sign] for axis, sign in mapping]
                for mapping in self.axis_mappings
            ],
            "orbits": [orbit.to_dict() for orbit in self.orbits],
        }

    @property
    def fingerprint(self) -> str:
        """Return the deterministic SHA-256 displacement-plan identity."""

        return sha256_bytes(canonical_json_bytes(self.identity_dict()))

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe plan."""

        return {**self.identity_dict(), "plan_fingerprint": self.fingerprint}


@dataclass(frozen=True)
class PeriodicPhononSample:
    """Central plus/minus force sample for one representative displacement."""

    representative_dof: int
    minus_forces_hartree_per_bohr: np.ndarray = field(repr=False, compare=False)
    plus_forces_hartree_per_bohr: np.ndarray = field(repr=False, compare=False)
    minus_calculation_fingerprint: str
    plus_calculation_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.representative_dof) is not int or self.representative_dof < 0:
            raise ValueError("phonon sample representative_dof must be non-negative")
        minus = np.array(self.minus_forces_hartree_per_bohr, dtype=np.float64, copy=True)
        plus = np.array(self.plus_forces_hartree_per_bohr, dtype=np.float64, copy=True)
        if (
            minus.ndim != 2
            or minus.shape[1:] != (3,)
            or plus.shape != minus.shape
            or not np.isfinite(minus).all()
            or not np.isfinite(plus).all()
        ):
            raise ValueError("phonon sample forces must be matching finite (N, 3) arrays")
        if not _is_sha256(self.minus_calculation_fingerprint) or not _is_sha256(
            self.plus_calculation_fingerprint
        ):
            raise ValueError("phonon sample calculations require SHA-256 fingerprints")
        minus.setflags(write=False)
        plus.setflags(write=False)
        object.__setattr__(self, "minus_forces_hartree_per_bohr", minus)
        object.__setattr__(self, "plus_forces_hartree_per_bohr", plus)

    @property
    def atom_count(self) -> int:
        """Return the force-sample atom count."""

        return int(self.minus_forces_hartree_per_bohr.shape[0])


@dataclass(frozen=True)
class PeriodicPhononSampleSet:
    """Immutable partial or complete central-force sample collection."""

    plan_fingerprint: str
    atom_count: int
    samples: tuple[PeriodicPhononSample, ...] = ()

    def __post_init__(self) -> None:
        if not _is_sha256(self.plan_fingerprint):
            raise ValueError("phonon sample set plan_fingerprint must be SHA-256")
        if type(self.atom_count) is not int or self.atom_count <= 0:
            raise ValueError("phonon sample set atom_count must be positive")
        indices = tuple(sample.representative_dof for sample in self.samples)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("phonon samples must be sorted and unique")
        if any(sample.atom_count != self.atom_count for sample in self.samples):
            raise ValueError("phonon sample atom counts are inconsistent")

    @classmethod
    def empty(cls, plan: PeriodicDisplacementPlan) -> PeriodicPhononSampleSet:
        """Return an empty sample collection bound to ``plan``."""

        return cls(plan.fingerprint, plan.atom_count)

    def with_sample(self, sample: PeriodicPhononSample) -> PeriodicPhononSampleSet:
        """Return a new collection containing one previously absent sample."""

        if sample.atom_count != self.atom_count:
            raise ValueError("phonon sample atom count differs from collection")
        if any(item.representative_dof == sample.representative_dof for item in self.samples):
            raise ValueError("phonon sample representative is already present")
        return PeriodicPhononSampleSet(
            self.plan_fingerprint,
            self.atom_count,
            tuple(sorted((*self.samples, sample), key=lambda item: item.representative_dof)),
        )

    def missing_representatives(
        self,
        plan: PeriodicDisplacementPlan,
    ) -> tuple[int, ...]:
        """Return planned representatives not present in this collection."""

        if plan.fingerprint != self.plan_fingerprint or plan.atom_count != self.atom_count:
            raise ValueError("phonon sample set does not match the displacement plan")
        present = {sample.representative_dof for sample in self.samples}
        if not present.issubset(plan.representative_dofs):
            raise ValueError("phonon sample set contains an unplanned representative")
        return tuple(dof for dof in plan.representative_dofs if dof not in present)


@dataclass(frozen=True)
class PeriodicPhononResult:
    """Raw force constants, diagnostics, and admitted Gamma-point modes."""

    plan_fingerprint: str
    system_fingerprint: str
    displacement_bohr: float
    masses_amu: np.ndarray = field(repr=False, compare=False)
    force_constants_hartree_per_bohr2: np.ndarray = field(repr=False, compare=False)
    reciprocity_residual_hartree_per_bohr2: float
    right_sum_rule_residual_hartree_per_bohr2: float
    left_sum_rule_residual_hartree_per_bohr2: float
    force_constants_passed: bool
    dynamical_eigenvalues_au: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    frequencies_cm1: np.ndarray | None = field(default=None, repr=False, compare=False)
    mass_weighted_eigenvectors: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    cartesian_eigenvectors: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    acoustic_mode_indices: tuple[int, int, int] | None = None
    acoustic_max_abs_frequency_cm1: float | None = None
    acoustic_translation_overlap_minimum: float | None = None
    acoustic_passed: bool = False
    stable: bool = False
    valid: bool = False

    def __post_init__(self) -> None:
        for name in ("plan_fingerprint", "system_fingerprint"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"phonon result {name} must be SHA-256")
        masses = np.array(self.masses_amu, dtype=np.float64, copy=True)
        constants = np.array(
            self.force_constants_hartree_per_bohr2,
            dtype=np.float64,
            copy=True,
        )
        if (
            masses.ndim != 1
            or masses.size == 0
            or not np.isfinite(masses).all()
            or np.any(masses <= 0.0)
            or constants.shape != (3 * masses.size, 3 * masses.size)
            or not np.isfinite(constants).all()
        ):
            raise ValueError("phonon result masses or force constants are invalid")
        arrays = {
            "masses_amu": masses,
            "force_constants_hartree_per_bohr2": constants,
        }
        mode_count = 3 * masses.size
        for name in (
            "dynamical_eigenvalues_au",
            "frequencies_cm1",
            "mass_weighted_eigenvectors",
            "cartesian_eigenvectors",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=np.float64, copy=True)
                if not np.isfinite(array).all():
                    raise ValueError(f"phonon result {name} must be finite")
                expected_shape = (
                    (mode_count, mode_count)
                    if name.endswith("eigenvectors")
                    else (mode_count,)
                )
                if array.shape != expected_shape:
                    raise ValueError(f"phonon result {name} has the wrong shape")
                arrays[name] = array
        mode_payloads = (
            self.dynamical_eigenvalues_au,
            self.frequencies_cm1,
            self.mass_weighted_eigenvectors,
            self.cartesian_eigenvectors,
        )
        if self.force_constants_passed != all(value is not None for value in mode_payloads):
            raise ValueError("phonon modes must exist exactly when force constants pass")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def mode_count(self) -> int:
        """Return the number of Gamma-point modes."""

        return 3 * int(self.masses_amu.size)

    def to_dict(self) -> dict[str, object]:
        """Return diagnostics and frequencies without dense matrices."""

        return {
            "plan_fingerprint": self.plan_fingerprint,
            "system_fingerprint": self.system_fingerprint,
            "displacement_bohr": self.displacement_bohr,
            "masses_amu": self.masses_amu.tolist(),
            "mode_count": self.mode_count,
            "reciprocity_residual_hartree_per_bohr2": (
                self.reciprocity_residual_hartree_per_bohr2
            ),
            "right_sum_rule_residual_hartree_per_bohr2": (
                self.right_sum_rule_residual_hartree_per_bohr2
            ),
            "left_sum_rule_residual_hartree_per_bohr2": (
                self.left_sum_rule_residual_hartree_per_bohr2
            ),
            "force_constants_passed": self.force_constants_passed,
            "frequencies_cm1": (
                None if self.frequencies_cm1 is None else self.frequencies_cm1.tolist()
            ),
            "acoustic_mode_indices": (
                None
                if self.acoustic_mode_indices is None
                else list(self.acoustic_mode_indices)
            ),
            "acoustic_max_abs_frequency_cm1": self.acoustic_max_abs_frequency_cm1,
            "acoustic_translation_overlap_minimum": (
                self.acoustic_translation_overlap_minimum
            ),
            "acoustic_passed": self.acoustic_passed,
            "stable": self.stable,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class PeriodicPhononConvergenceResult:
    """Displacement-convergence comparison for two complete phonon results."""

    coarse_displacement_bohr: float
    fine_displacement_bohr: float
    maximum_frequency_drift_cm1: float
    maximum_eigenvalue_drift_au: float
    frequency_tolerance_cm1: float
    eigenvalue_tolerance_au: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        """Return the displacement-convergence comparison."""

        return {
            "coarse_displacement_bohr": self.coarse_displacement_bohr,
            "fine_displacement_bohr": self.fine_displacement_bohr,
            "maximum_frequency_drift_cm1": self.maximum_frequency_drift_cm1,
            "maximum_eigenvalue_drift_au": self.maximum_eigenvalue_drift_au,
            "frequency_tolerance_cm1": self.frequency_tolerance_cm1,
            "eigenvalue_tolerance_au": self.eigenvalue_tolerance_au,
            "passed": self.passed,
        }
