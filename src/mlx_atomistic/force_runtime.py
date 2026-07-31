"""Prepared force-evaluation bindings for the molecular-dynamics hot loop."""

from __future__ import annotations

from dataclasses import dataclass
from types import NotImplementedType

import mlx.core as mx

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import NeighborList
from mlx_atomistic.virtual_sites import VirtualSiteManager


@dataclass(frozen=True)
class _BoundForcePipeline:
    """Force terms bound to one exact neighbor-list generation."""

    force_terms: tuple[object, ...]
    term_bindings: tuple[object | NotImplementedType, ...]
    interactions: object | None
    cell: Cell | None
    virtual_sites: VirtualSiteManager | None

    def forces(
        self,
        positions: mx.array,
        *,
        evaluation_positions: mx.array | None = None,
    ) -> mx.array:
        """Evaluate forces without repeating setup or neighbor admission.

        Args:
            positions: Real-atom coordinates.
            evaluation_positions: Already validated coordinates including any
                virtual sites. ``None`` derives them from ``positions``.

        Returns:
            Forces on the real atoms.
        """

        real_positions = as_mx_array(positions)
        eval_positions = (
            _evaluation_positions(real_positions, self.virtual_sites)
            if evaluation_positions is None
            else as_mx_array(evaluation_positions)
        )
        total_forces = mx.zeros_like(eval_positions)
        for term, binding in zip(
            self.force_terms,
            self.term_bindings,
            strict=True,
        ):
            bound_method = getattr(term, "_forces_from_binding", None)
            forces = (
                bound_method(eval_positions, binding)
                if binding is not NotImplemented and callable(bound_method)
                else NotImplemented
            )
            if forces is NotImplemented:
                runtime_method = getattr(term, "_runtime_forces", None)
                forces = (
                    runtime_method(
                        eval_positions,
                        cell=self.cell,
                        pairs=self.interactions,
                    )
                    if callable(runtime_method)
                    else NotImplemented
                )
            if forces is NotImplemented:
                _, forces = term.energy_forces(
                    eval_positions,
                    self.cell,
                    pairs=self.interactions,
                )
            total_forces = total_forces + as_mx_array(forces)
        if not self.force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        return _redistribute_forces(
            total_forces,
            eval_positions,
            self.virtual_sites,
        )


class _PreparedForcePipeline:
    """Own setup-validated force terms and cache their neighbor binding."""

    def __init__(
        self,
        force_terms: tuple[object, ...],
        *,
        cell: Cell | None,
        virtual_sites: VirtualSiteManager | None,
    ) -> None:
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        self.force_terms = force_terms
        self.cell = cell
        self.virtual_sites = virtual_sites
        self._cached_neighbor_list: NeighborList | None = None
        self._cached_binding: _BoundForcePipeline | None = None

    @classmethod
    def prepare(
        cls,
        force_terms: tuple[object, ...],
        *,
        cell: Cell | None,
        virtual_sites: VirtualSiteManager | None = None,
    ) -> _PreparedForcePipeline:
        """Create a fixed-cell force pipeline.

        Args:
            force_terms: Immutable force terms for this run segment.
            cell: Current periodic cell, or ``None`` for nonperiodic work.
            virtual_sites: Optional virtual-site mapping.

        Returns:
            A pipeline ready to bind a neighbor-list generation.
        """

        return cls(
            tuple(force_terms),
            cell=cell,
            virtual_sites=virtual_sites,
        )

    def bind(self, neighbor_list: NeighborList | None) -> _BoundForcePipeline:
        """Bind the current neighbor representation once.

        Args:
            neighbor_list: Current neighbor list, or ``None`` for dense work.

        Returns:
            A force evaluator bound to the exact list object.
        """

        if (
            neighbor_list is self._cached_neighbor_list
            and self._cached_binding is not None
        ):
            return self._cached_binding
        interactions = (
            None if neighbor_list is None else neighbor_list.interactions
        )
        term_bindings: list[object | NotImplementedType] = []
        for term in self.force_terms:
            prepare = getattr(term, "_prepare_force_binding", None)
            binding = (
                prepare(self.cell, interactions)
                if callable(prepare)
                else NotImplemented
            )
            term_bindings.append(binding)
        bound = _BoundForcePipeline(
            force_terms=self.force_terms,
            term_bindings=tuple(term_bindings),
            interactions=interactions,
            cell=self.cell,
            virtual_sites=self.virtual_sites,
        )
        self._cached_neighbor_list = neighbor_list
        self._cached_binding = bound
        return bound


def _evaluation_positions(
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return positions
    if positions.shape[0] != virtual_sites.n_real_atoms:
        msg = "positions must contain real atoms only when virtual sites are configured"
        raise ValueError(msg)
    return virtual_sites.extend_positions(positions)


def _redistribute_forces(
    forces: mx.array,
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return forces
    return virtual_sites.redistribute_forces(forces, positions)
