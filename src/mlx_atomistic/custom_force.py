"""Custom force terms with symbolic expression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_atomistic.core import Cell, as_mx_array


def _cf_norm(vector: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(mx.sum(vector * vector, axis=-1), 1e-12))


def _cf_cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _cf_parameter_array(value, *, count: int, name: str) -> mx.array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((count,), float(array), dtype=np.float32)
    if array.shape != (count,):
        msg = f"{name} must be scalar or have shape ({count},)"
        raise ValueError(msg)
    return as_mx_array(array)


def _cf_zero_energy(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in " \t\n\r":
            i += 1
            continue
        if ch == "*" and i + 1 < len(expr) and expr[i + 1] == "*":
            tokens.append("**")
            i += 2
            continue
        if ch in "+-*/(),":
            tokens.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            tokens.append(expr[start:i])
            continue
        if ch.isdigit() or ch == ".":
            start = i
            while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                i += 1
            tokens.append(expr[start:i])
            continue
        msg = f"unexpected character {ch!r} in expression"
        raise ValueError(msg)
    return tokens


_FUNCTION_NAMES = frozenset({"cos", "sin", "exp", "log", "sqrt", "abs"})


class _ExprNode:
    pass


@dataclass(frozen=True)
class _BinaryOp(_ExprNode):
    op: str
    left: _ExprNode
    right: _ExprNode


@dataclass(frozen=True)
class _UnaryOp(_ExprNode):
    op: str
    operand: _ExprNode


@dataclass(frozen=True)
class _FunctionCall(_ExprNode):
    name: str
    arg: _ExprNode


@dataclass(frozen=True)
class _Variable(_ExprNode):
    name: str


@dataclass(frozen=True)
class _Constant(_ExprNode):
    value: float


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected: str | None = None) -> str:
        tok = self._peek()
        if tok is None:
            msg = f"unexpected end of expression, expected {expected!r}"
            raise ValueError(msg)
        if expected is not None and tok != expected:
            msg = f"expected {expected!r}, got {tok!r}"
            raise ValueError(msg)
        self.pos += 1
        return tok

    def parse(self) -> _ExprNode:
        node = self._parse_additive()
        if self.pos < len(self.tokens):
            msg = f"unexpected token {self.tokens[self.pos]!r}"
            raise ValueError(msg)
        return node

    def _parse_additive(self) -> _ExprNode:
        left = self._parse_multiplicative()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_multiplicative()
            left = _BinaryOp(op, left, right)
        return left

    def _parse_multiplicative(self) -> _ExprNode:
        left = self._parse_power()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_power()
            left = _BinaryOp(op, left, right)
        return left

    def _parse_power(self) -> _ExprNode:
        base = self._parse_unary()
        if self._peek() == "**":
            self._consume()
            exp = self._parse_power()
            return _BinaryOp("**", base, exp)
        return base

    def _parse_unary(self) -> _ExprNode:
        if self._peek() == "-":
            self._consume()
            operand = self._parse_unary()
            return _UnaryOp("-", operand)
        if self._peek() == "+":
            self._consume()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> _ExprNode:
        tok = self._peek()
        if tok is None:
            msg = "unexpected end of expression"
            raise ValueError(msg)
        if tok == "(":
            self._consume("(")
            node = self._parse_additive()
            self._consume(")")
            return node
        if tok in _FUNCTION_NAMES:
            name = self._consume()
            self._consume("(")
            arg = self._parse_additive()
            self._consume(")")
            return _FunctionCall(name, arg)
        if tok.replace(".", "", 1).isdigit():
            text = self._consume()
            return _Constant(float(text))
        if tok[0].isalpha() or tok[0] == "_":
            text = self._consume()
            if text == "pi":
                return _Constant(float(np.pi))
            return _Variable(text)
        msg = f"unexpected token {tok!r}"
        raise ValueError(msg)


_MLX_OPS: dict[str, Any] = {
    "+": mx.add,
    "-": mx.subtract,
    "*": mx.multiply,
    "/": mx.divide,
    "**": mx.power,
}

_MLX_FUNCS: dict[str, Any] = {
    "cos": mx.cos,
    "sin": mx.sin,
    "exp": mx.exp,
    "log": mx.log,
    "sqrt": mx.sqrt,
    "abs": mx.abs,
}


def _evaluate(node: _ExprNode, env: dict[str, mx.array]) -> mx.array:
    if isinstance(node, _Constant):
        return mx.array(node.value, dtype=mx.float32)
    if isinstance(node, _Variable):
        if node.name not in env:
            msg = f"undefined variable {node.name!r} in expression"
            raise ValueError(msg)
        return env[node.name]
    if isinstance(node, _UnaryOp):
        operand = _evaluate(node.operand, env)
        if node.op == "-":
            return mx.negative(operand)
        msg = f"unsupported unary operator {node.op!r}"
        raise ValueError(msg)
    if isinstance(node, _BinaryOp):
        left = _evaluate(node.left, env)
        right = _evaluate(node.right, env)
        op_fn = _MLX_OPS.get(node.op)
        if op_fn is None:
            msg = f"unsupported binary operator {node.op!r}"
            raise ValueError(msg)
        return op_fn(left, right)
    if isinstance(node, _FunctionCall):
        arg = _evaluate(node.arg, env)
        func = _MLX_FUNCS.get(node.name)
        if func is None:
            msg = f"unsupported function {node.name!r}"
            raise ValueError(msg)
        return func(arg)
    msg = f"unsupported expression node {type(node).__name__}"
    raise ValueError(msg)


def parse_expression(expr: str) -> _ExprNode:
    """Parse a force-expression string into an evaluable expression tree.

    Args:
        expr: The expression source string.

    Returns:
        The parsed expression AST node.
    """

    tokens = _tokenize(expr)
    parser = _Parser(tokens)
    return parser.parse()


def evaluate_expression(node: _ExprNode, env: dict[str, mx.array]) -> mx.array:
    """Evaluate a parsed expression tree against a variable environment.

    Args:
        node: A parsed expression AST node.
        env: Mapping from variable name to its ``mx.array`` value.

    Returns:
        The evaluated result as an ``mx.array``.
    """

    return _evaluate(node, env)


@dataclass(frozen=True)
class CustomForcePotential:
    """Custom force term defined by a symbolic expression.

    The expression string is parsed at construction time into an AST of
    ``mlx.core`` operations.  Supported syntax:

    * arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
    * functions: ``cos``, ``sin``, ``exp``, ``log``, ``sqrt``, ``abs``
    * variables: geometric quantities (``r``, ``theta``) and named
      per-term parameters supplied via *parameters* or *global_parameters*.

    Parameters
    ----------
    indices : array-like
        ``(n_terms, W)`` index array where *W* is the term width
        (2 for bond/pair, 3 for angle).  Each row lists the atom
        indices participating in that term.
    expression : str
        Symbolic energy-per-term expression.
    parameters : dict[str, array-like], optional
        Named per-term parameter arrays, each with shape ``(n_terms,)``.
    global_parameters : dict[str, float], optional
        Named scalar parameters available in the expression.
    name : str
        Descriptive name for diagnostics.
    term_type : str
        One of ``"bond"`` (2-body distance), ``"angle"`` (3-body
        angle), or ``"pair"`` (nonbonded-like 2-body).
    supports_virial : bool
        Whether the term advertises virial support.
    """

    indices: object
    expression: str
    parameters: object = ()
    global_parameters: object = ()
    name: str = "custom_force"
    term_type: str = "bond"
    supports_virial: bool = True

    def __post_init__(self) -> None:
        valid_types = ("bond", "angle", "pair")
        if self.term_type not in valid_types:
            msg = f"term_type must be one of {valid_types}, got {self.term_type!r}"
            raise ValueError(msg)
        indices = np.asarray(self.indices, dtype=np.int32)
        if indices.size == 0:
            width = 2 if self.term_type in ("bond", "pair") else 3
            indices = np.empty((0, width), dtype=np.int32)
        if indices.ndim != 2:
            msg = "indices must have shape (n_terms, W)"
            raise ValueError(msg)
        expected_width = 2 if self.term_type in ("bond", "pair") else 3
        if indices.shape[1] != expected_width:
            msg = (
                f"indices must have shape (n_terms, {expected_width}) "
                f"for term_type={self.term_type!r}"
            )
            raise ValueError(msg)
        count = indices.shape[0]
        object.__setattr__(self, "indices", mx.array(indices, dtype=mx.int32))
        params: dict[str, mx.array] = {}
        raw_params = (
            dict(self.parameters) if not isinstance(self.parameters, (tuple, list)) else {}
        )
        for p_name, p_value in raw_params.items():
            params[p_name] = _cf_parameter_array(p_value, count=count, name=p_name)
        object.__setattr__(self, "parameters", params)
        global_params: dict[str, float] = {}
        raw_globals = (
            dict(self.global_parameters)
            if not isinstance(self.global_parameters, (tuple, list))
            else {}
        )
        for g_name, g_value in raw_globals.items():
            global_params[g_name] = float(g_value)
        object.__setattr__(self, "global_parameters", global_params)
        object.__setattr__(self, "_expr_ast", parse_expression(self.expression))

    def _build_env(
        self,
        positions: mx.array,
        cell: Cell | None,
    ) -> tuple[list[dict[str, mx.array]], dict[str, Any]]:
        indices_np = np.asarray(self.indices, dtype=np.int32)
        n_terms = indices_np.shape[0]
        if n_terms == 0:
            return [], {}
        geom: dict[str, Any] = {}
        env: dict[str, mx.array] = {}
        if self.term_type in ("bond", "pair"):
            i_idxs = self.indices[:, 0]
            j_idxs = self.indices[:, 1]
            displacement = positions[i_idxs] - positions[j_idxs]
            if cell is not None:
                displacement = cell.minimum_image(displacement)
            r = _cf_norm(displacement)
            geom["displacement"] = displacement
            geom["i_idxs"] = i_idxs
            geom["j_idxs"] = j_idxs
            geom["r"] = r
            env["r"] = r
            env["rx"] = displacement[:, 0]
            env["ry"] = displacement[:, 1]
            env["rz"] = displacement[:, 2]
        elif self.term_type == "angle":
            i_idxs = self.indices[:, 0]
            j_idxs = self.indices[:, 1]
            k_idxs = self.indices[:, 2]
            left = positions[i_idxs] - positions[j_idxs]
            right = positions[k_idxs] - positions[j_idxs]
            if cell is not None:
                left = cell.minimum_image(left)
                right = cell.minimum_image(right)
            left_norm = _cf_norm(left)
            right_norm = _cf_norm(right)
            cosine = mx.sum(left * right, axis=-1) / (left_norm * right_norm)
            cosine = mx.clip(cosine, -0.999999, 0.999999)
            theta = mx.arccos(cosine)
            sin_theta = mx.sqrt(mx.maximum(1.0 - cosine * cosine, 1e-12))
            geom["left"] = left
            geom["right"] = right
            geom["left_norm"] = left_norm
            geom["right_norm"] = right_norm
            geom["cosine"] = cosine
            geom["sin_theta"] = sin_theta
            geom["i_idxs"] = i_idxs
            geom["j_idxs"] = j_idxs
            geom["k_idxs"] = k_idxs
            env["theta"] = theta
            env["cos_theta"] = cosine
            env["sin_theta"] = sin_theta
        env["n_terms"] = n_terms
        for p_name, p_array in self.parameters.items():
            env[p_name] = p_array
        for g_name, g_value in self.global_parameters.items():
            env[g_name] = mx.array(g_value, dtype=mx.float32)
        return [env], geom

    def potential_energy(self, positions: mx.array, cell: Cell | None = None) -> mx.array:
        """Return the custom force-term energy from the symbolic expression.

        Args:
            positions: Atomic coordinates, shape ``(n_atoms, 3)``.
            cell: Optional periodic cell; when given, distances use the minimum-image
                convention. Defaults to ``None``.

        Returns:
            Total custom force-term energy as a scalar array.
        """

        positions = as_mx_array(positions)
        if self.indices.shape[0] == 0:
            return _cf_zero_energy(positions)
        envs, _ = self._build_env(positions, cell)
        per_term_energy = evaluate_expression(self._expr_ast, envs[0])
        return mx.sum(per_term_energy)

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return the custom force-term energy and per-atom forces.

        Args:
            positions: Atomic coordinates, shape ``(n_atoms, 3)``.
            cell: Optional periodic cell; when given, distances use the minimum-image
                convention. Defaults to ``None``.
            pairs: Accepted for interface uniformity and ignored; the term uses its
                stored index list. Defaults to ``None``.

        Returns:
            An ``(energy, forces)`` tuple: scalar energy and per-atom forces of shape
                ``(n_atoms, 3)``.
        """

        del pairs
        positions = as_mx_array(positions)
        if self.indices.shape[0] == 0:
            return _cf_zero_energy(positions), mx.zeros_like(positions)
        envs, geom = self._build_env(positions, cell)
        env = envs[0]
        per_term_energy = evaluate_expression(self._expr_ast, env)
        total_energy = mx.sum(per_term_energy)
        forces = mx.zeros_like(positions)
        if self.term_type in ("bond", "pair"):
            forces = self._bond_pair_forces(positions, cell, env, geom)
        elif self.term_type == "angle":
            forces = self._angle_forces(positions, cell, env, geom)
        return total_energy, forces

    def _bond_pair_forces(
        self,
        positions: mx.array,
        cell: Cell | None,
        env: dict[str, mx.array],
        geom: dict[str, Any],
    ) -> mx.array:
        displacement = geom["displacement"]
        i_idxs = geom["i_idxs"]
        j_idxs = geom["j_idxs"]
        r = geom["r"]
        eps = mx.array(1e-4, dtype=mx.float32)
        env_plus = dict(env)
        env_plus["r"] = r + eps
        env_minus = dict(env)
        env_minus["r"] = r - eps
        e_plus = evaluate_expression(self._expr_ast, env_plus)
        e_minus = evaluate_expression(self._expr_ast, env_minus)
        dedr = (e_plus - e_minus) / (2.0 * eps)
        safe_r = mx.maximum(r, mx.array(1e-12, dtype=mx.float32))
        pair_forces = -(dedr / safe_r)[:, None] * displacement
        forces = mx.zeros_like(positions).at[i_idxs].add(pair_forces).at[j_idxs].add(
            -pair_forces
        )
        return forces

    def _angle_forces(
        self,
        positions: mx.array,
        cell: Cell | None,
        env: dict[str, mx.array],
        geom: dict[str, Any],
    ) -> mx.array:
        left = geom["left"]
        right = geom["right"]
        left_norm = geom["left_norm"]
        right_norm = geom["right_norm"]
        cosine = geom["cosine"]
        sin_theta = geom["sin_theta"]
        i_idxs = geom["i_idxs"]
        j_idxs = geom["j_idxs"]
        k_idxs = geom["k_idxs"]
        eps = mx.array(1e-4, dtype=mx.float32)
        env_plus = dict(env)
        env_plus["theta"] = env["theta"] + eps
        env_plus["cos_theta"] = mx.cos(env_plus["theta"])
        env_plus["sin_theta"] = mx.sin(env_plus["theta"])
        env_minus = dict(env)
        env_minus["theta"] = env["theta"] - eps
        env_minus["cos_theta"] = mx.cos(env_minus["theta"])
        env_minus["sin_theta"] = mx.sin(env_minus["theta"])
        e_plus = evaluate_expression(self._expr_ast, env_plus)
        e_minus = evaluate_expression(self._expr_ast, env_minus)
        dedtheta = (e_plus - e_minus) / (2.0 * eps)
        prefactor = dedtheta / sin_theta
        left_force = prefactor[:, None] * (
            right / (left_norm * right_norm)[:, None]
            - cosine[:, None] * left / (left_norm * left_norm)[:, None]
        )
        right_force = prefactor[:, None] * (
            left / (left_norm * right_norm)[:, None]
            - cosine[:, None] * right / (right_norm * right_norm)[:, None]
        )
        center_force = -(left_force + right_force)
        forces = (
            mx.zeros_like(positions)
            .at[i_idxs]
            .add(left_force)
            .at[j_idxs]
            .add(center_force)
            .at[k_idxs]
            .add(right_force)
        )
        return forces