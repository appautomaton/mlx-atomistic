"""Fixed-Hamiltonian identity for one periodic Davidson solve."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from mlx_atomistic.dft._periodic_hamiltonian import PeriodicKohnShamOperator
from mlx_atomistic.dft._periodic_models import PeriodicDavidsonConfig
from mlx_atomistic.dft._periodic_orthonormalization import (
    _DAVIDSON_RANK_POLICY,
    _Complex64RankPolicy,
)


def _hamiltonian_context(
    operator: PeriodicKohnShamOperator,
    config: PeriodicDavidsonConfig,
    n_bands: int,
    rank_policy: _Complex64RankPolicy,
) -> tuple[object, ...]:
    nonlocal_context = (
        None
        if operator.nonlocal_operator is None
        else (
            id(operator.nonlocal_operator),
            operator.nonlocal_operator._context_identity,
        )
    )
    potential = operator._effective_local_potential
    return (
        id(operator),
        id(potential),
        tuple(int(value) for value in potential.shape),
        str(potential.dtype),
        operator.basis.basis_fingerprint,
        operator.basis.order_fingerprint,
        operator.basis._layout.lane_id,
        operator.basis.reciprocal_grid.fingerprint,
        tuple(float(value) for value in operator.basis.kpoint_cartesian),
        nonlocal_context,
        "complex64-float32",
        str(mx.default_device()),
        config.max_iterations,
        config.tolerance,
        config.max_subspace_size,
        config.preconditioner_floor,
        n_bands,
        "complex64-adaptive-choleskyqr-cgs2-mgs2-rank-v6",
        rank_policy.relative_tolerance,
    )


@dataclass(frozen=True, eq=False)
class _FixedHamiltonianToken:
    """Solve-local identity that prevents paired H(V) from crossing contexts."""

    context: tuple[object, ...]
    nonce: object = field(default_factory=object, repr=False)

    @classmethod
    def create(
        cls,
        operator: PeriodicKohnShamOperator,
        config: PeriodicDavidsonConfig,
        n_bands: int,
        rank_policy: _Complex64RankPolicy = _DAVIDSON_RANK_POLICY,
    ) -> _FixedHamiltonianToken:
        return cls(_hamiltonian_context(operator, config, n_bands, rank_policy))

    def validate(
        self,
        operator: PeriodicKohnShamOperator,
        config: PeriodicDavidsonConfig,
        n_bands: int,
        rank_policy: _Complex64RankPolicy = _DAVIDSON_RANK_POLICY,
    ) -> None:
        if self.context != _hamiltonian_context(
            operator,
            config,
            n_bands,
            rank_policy,
        ):
            msg = "Davidson H(V) token does not match the fixed Hamiltonian"
            raise ValueError(msg)
