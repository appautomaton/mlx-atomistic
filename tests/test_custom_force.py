"""Tests for CustomForcePotential expression evaluation and force correctness."""

from math import pi

import numpy as np
import pytest

from mlx_atomistic.custom_force import (
    CustomForcePotential,
    evaluate_expression,
    parse_expression,
)
from mlx_atomistic.forcefields import (
    CustomForcePotential as ExportedCustomForcePotential,
)
from mlx_atomistic.forcefields import (
    HarmonicAnglePotential,
    HarmonicBondPotential,
)


def _finite_difference_forces(term, positions, *, epsilon=1e-4):
    positions = np.array(positions, dtype=np.float32)
    forces = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += epsilon
            minus[atom, axis] -= epsilon
            e_plus = float(np.array(term.energy_forces(plus)[0]))
            e_minus = float(np.array(term.energy_forces(minus)[0]))
            forces[atom, axis] = -(e_plus - e_minus) / (2.0 * epsilon)
    return forces


def _assert_forces_match_fd(term, positions, *, atol=2e-3):
    _, forces = term.energy_forces(positions)
    fd_forces = _finite_difference_forces(term, positions)
    np.testing.assert_allclose(np.array(forces), fd_forces, atol=atol)


class TestExpressionParser:
    def test_simple_arithmetic(self):
        node = parse_expression("1.0 + 2.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(3.0, abs=1e-5)

    def test_multiply(self):
        node = parse_expression("3.0 * 2.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(6.0, abs=1e-5)

    def test_division(self):
        node = parse_expression("6.0 / 2.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(3.0, abs=1e-5)

    def test_power(self):
        node = parse_expression("2.0 ** 3.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(8.0, abs=1e-5)

    def test_variable(self):
        node = parse_expression("r")
        import mlx.core as mx

        result = evaluate_expression(node, {"r": mx.array(2.5, dtype=mx.float32)})
        assert float(result) == pytest.approx(2.5, abs=1e-5)

    def test_function_sin(self):
        node = parse_expression("sin(0.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_function_cos(self):
        node = parse_expression("cos(0.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(1.0, abs=1e-5)

    def test_function_exp(self):
        node = parse_expression("exp(0.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(1.0, abs=1e-5)

    def test_function_log(self):
        node = parse_expression("log(1.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_function_sqrt(self):
        node = parse_expression("sqrt(4.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(2.0, abs=1e-5)

    def test_function_abs(self):
        node = parse_expression("abs(-3.0)")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(3.0, abs=1e-5)

    def test_parentheses(self):
        node = parse_expression("(2.0 + 3.0) * 4.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(20.0, abs=1e-5)

    def test_negation(self):
        node = parse_expression("-1.0 + 3.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(2.0, abs=1e-5)

    def test_complex_expression(self):
        node = parse_expression("0.5 * k * (r - r0) ** 2")
        import mlx.core as mx

        env = {
            "k": mx.array([2.0, 3.0], dtype=mx.float32),
            "r": mx.array([1.5, 2.0], dtype=mx.float32),
            "r0": mx.array([1.0, 1.0], dtype=mx.float32),
        }
        result = evaluate_expression(node, env)
        expected = np.array(
            [0.5 * 2.0 * (1.5 - 1.0) ** 2, 0.5 * 3.0 * (2.0 - 1.0) ** 2],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_pi_constant(self):
        node = parse_expression("pi")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(pi, abs=1e-5)

    def test_operator_precedence(self):
        node = parse_expression("2.0 + 3.0 * 4.0")
        result = evaluate_expression(node, {})
        assert float(result) == pytest.approx(14.0, abs=1e-5)

    def test_undefined_variable_raises(self):
        node = parse_expression("undefined_var")
        with pytest.raises(ValueError, match="undefined variable"):
            evaluate_expression(node, {})

    def test_unsupported_token_raises(self):
        with pytest.raises(ValueError):
            parse_expression("@")


class TestCustomForceBondLike:
    def test_bond_like_expression_matches_harmonic_bond_energy(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="0.5 * k * (r - r0) ** 2",
            parameters={"k": [5.0], "r0": [1.0]},
            term_type="bond",
        )
        reference = HarmonicBondPotential([(0, 1)], k=5.0, length=1.0)
        e_custom, _ = custom.energy_forces(positions)
        e_ref, _ = reference.energy_forces(positions)
        assert float(np.array(e_custom)) == pytest.approx(
            float(np.array(e_ref)), abs=1e-4
        )

    def test_bond_like_forces_match_finite_difference(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="0.5 * k * (r - r0) ** 2",
            parameters={"k": [5.0], "r0": [1.0]},
            term_type="bond",
        )
        _assert_forces_match_fd(custom, positions)

    def test_bond_like_with_sin_expression_forces(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.3, 0.1]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="k * (1.0 - cos(r - r0))",
            parameters={"k": [3.0], "r0": [1.0]},
            term_type="bond",
        )
        _assert_forces_match_fd(custom, positions, atol=5e-3)

    def test_multiple_bonds(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.1, 0.0]], dtype=np.float32
        )
        custom = CustomForcePotential(
            indices=[(0, 1), (1, 2)],
            expression="0.5 * k * (r - r0) ** 2",
            parameters={"k": [4.0, 6.0], "r0": [1.0, 1.2]},
            term_type="bond",
        )
        _assert_forces_match_fd(custom, positions)

    def test_empty_indices_returns_zero(self):
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=np.empty((0, 2), dtype=np.int32),
            expression="0.5 * k * (r - r0) ** 2",
            parameters={},
            term_type="bond",
        )
        energy, forces = custom.energy_forces(positions)
        assert float(np.array(energy)) == pytest.approx(0.0, abs=1e-6)

    def test_morse_like_bond_expression(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.2, 0.1]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="D * (1.0 - exp(-a * (r - r0))) ** 2",
            parameters={"D": [10.0], "a": [1.5], "r0": [1.0]},
            global_parameters={},
            term_type="bond",
        )
        _assert_forces_match_fd(custom, positions, atol=3e-2)

    def test_global_parameters_in_expression(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="0.5 * k * (r - r0) ** 2",
            parameters={"r0": [1.0]},
            global_parameters={"k": 5.0},
            term_type="bond",
        )
        _assert_forces_match_fd(custom, positions)

    def test_lj_like_pair_expression(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)",
            parameters={"epsilon": [0.1], "sigma": [1.0]},
            term_type="pair",
        )
        _assert_forces_match_fd(custom, positions, atol=5e-3)


class TestCustomForceAngleLike:
    def test_angle_like_expression_matches_harmonic_angle_energy(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.9, 0.0]], dtype=np.float32
        )
        custom = CustomForcePotential(
            indices=[(0, 1, 2)],
            expression="0.5 * k * (theta - theta0) ** 2",
            parameters={"k": [2.0], "theta0": [pi / 2.0]},
            term_type="angle",
        )
        reference = HarmonicAnglePotential([(0, 1, 2)], k=2.0, angle=pi / 2.0)
        e_custom, _ = custom.energy_forces(positions)
        e_ref, _ = reference.energy_forces(positions)
        assert float(np.array(e_custom)) == pytest.approx(
            float(np.array(e_ref)), abs=1e-3
        )

    def test_angle_like_forces_match_finite_difference(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.9, 0.0]], dtype=np.float32
        )
        custom = CustomForcePotential(
            indices=[(0, 1, 2)],
            expression="0.5 * k * (theta - theta0) ** 2",
            parameters={"k": [2.0], "theta0": [pi / 2.0]},
            term_type="angle",
        )
        _assert_forces_match_fd(custom, positions, atol=5e-3)

    def test_angle_with_cos_expression(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.8, 0.1]], dtype=np.float32
        )
        custom = CustomForcePotential(
            indices=[(0, 1, 2)],
            expression="k * (1.0 + cos(theta - theta0))",
            parameters={"k": [1.5], "theta0": [pi]},
            term_type="angle",
        )
        _assert_forces_match_fd(custom, positions, atol=5e-3)


class TestCustomForceExport:
    def test_exported_from_package(self):
        assert ExportedCustomForcePotential is CustomForcePotential

    def test_exported_from_init(self):
        from mlx_atomistic import CustomForcePotential as InitCustom

        assert InitCustom is CustomForcePotential


class TestCustomForceValidation:
    def test_invalid_term_type_raises(self):
        with pytest.raises(ValueError, match="term_type"):
            CustomForcePotential(
                indices=[(0, 1)],
                expression="0.5 * k * (r - r0) ** 2",
                parameters={"k": [1.0], "r0": [1.0]},
                term_type="invalid",
            )

    def test_wrong_width_indices_raises(self):
        with pytest.raises(ValueError):
            CustomForcePotential(
                indices=[(0, 1, 2)],
                expression="0.5 * k * (r - r0) ** 2",
                parameters={"k": [1.0], "r0": [1.0]},
                term_type="bond",
            )

    def test_parameter_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CustomForcePotential(
                indices=[(0, 1)],
                expression="0.5 * k * (r - r0) ** 2",
                parameters={"k": [1.0, 2.0], "r0": [1.0]},
                term_type="bond",
            )

    def test_forces_with_cell_minimum_image(self):
        from mlx_atomistic.core import Cell

        cell = Cell.orthorhombic([6.0, 6.0, 6.0])
        positions = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="0.5 * k * (r - r0) ** 2",
            parameters={"k": [4.0], "r0": [1.0]},
            term_type="bond",
        )
        energy, forces = custom.energy_forces(positions, cell=cell)
        r_min_image = 0.5
        expected_energy = 0.5 * 4.0 * (r_min_image - 1.0) ** 2
        assert float(np.array(energy)) == pytest.approx(expected_energy, abs=1e-3)


class TestCustomForceNonbondedLike:
    def test_coulomb_like_pair(self):
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="c * q1 * q2 / r",
            parameters={"c": [1389.35], "q1": [1.0], "q2": [-1.0]},
            term_type="pair",
        )
        _assert_forces_match_fd(custom, positions, atol=0.1)

    def test_soft_core_like_expression(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.1]], dtype=np.float32)
        custom = CustomForcePotential(
            indices=[(0, 1)],
            expression="epsilon * (sigma ** 6) / (r ** 6 + alpha * sigma ** 6)",
            parameters={"epsilon": [0.5], "sigma": [1.0], "alpha": [0.5]},
            term_type="pair",
        )
        _assert_forces_match_fd(custom, positions, atol=5e-3)
