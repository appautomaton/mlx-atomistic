"""Prepared force-evaluation bindings for the molecular-dynamics hot loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import NotImplementedType
from typing import Any

import mlx.core as mx

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.neighbors import NeighborList
from mlx_atomistic.virtual_sites import VirtualSiteManager


class _ExclusiveRouteProfiler:
    """Collect opt-in, synchronized, non-overlapping runtime route timings."""

    def __init__(self) -> None:
        self._wall_started = perf_counter()
        self._routes: dict[str, dict[str, int | float]] = {}

    @staticmethod
    def start() -> float:
        """Return a route start timestamp."""

        return perf_counter()

    def finish(self, name: str, started: float, *values: Any) -> None:
        """Complete queued route work and record exclusive wall ownership."""

        arrays = tuple(_profile_arrays(values))
        completion_started = perf_counter()
        if arrays:
            mx.eval(*arrays)
        completion_seconds = perf_counter() - completion_started
        wall_seconds = perf_counter() - started
        route = self._routes.setdefault(
            name,
            {
                "count": 0,
                "wall_seconds": 0.0,
                "completion_seconds": 0.0,
            },
        )
        route["count"] = int(route["count"]) + 1
        route["wall_seconds"] = float(route["wall_seconds"]) + wall_seconds
        route["completion_seconds"] = (
            float(route["completion_seconds"]) + completion_seconds
        )

    def report(self) -> dict[str, Any]:
        """Return reconciled route totals for the instrumented execution."""

        instrumented_wall_seconds = perf_counter() - self._wall_started
        routes = {
            name: {
                **values,
                "graph_and_host_seconds": max(
                    0.0,
                    float(values["wall_seconds"])
                    - float(values["completion_seconds"]),
                ),
            }
            for name, values in sorted(self._routes.items())
        }
        accounted_seconds = sum(
            float(values["wall_seconds"]) for values in routes.values()
        )
        residual_seconds = instrumented_wall_seconds - accounted_seconds
        tolerance_seconds = max(1.0e-9, instrumented_wall_seconds * 1.0e-9)
        reconciled = residual_seconds >= -tolerance_seconds
        if residual_seconds < 0.0 and reconciled:
            residual_seconds = 0.0
        return {
            "mode": "synchronized_exclusive_routes",
            "instrumented_wall_seconds": instrumented_wall_seconds,
            "accounted_route_seconds": accounted_seconds,
            "residual_seconds": residual_seconds,
            "reconciliation_error_seconds": (
                instrumented_wall_seconds
                - accounted_seconds
                - residual_seconds
            ),
            "reconciled": reconciled,
            "routes": routes,
        }


def _profile_arrays(values: tuple[Any, ...]):
    for value in values:
        if isinstance(value, mx.array):
            yield value
        elif isinstance(value, dict):
            yield from _profile_arrays(tuple(value.values()))
        elif isinstance(value, tuple | list):
            yield from _profile_arrays(tuple(value))


@dataclass(frozen=True)
class _BoundForcePipeline:
    """Force terms bound to one exact neighbor-list generation."""

    force_terms: tuple[object, ...]
    term_bindings: tuple[object | NotImplementedType, ...]
    interactions: object | None
    cell: Cell | None
    virtual_sites: VirtualSiteManager | None
    route_profiler: _ExclusiveRouteProfiler | None = None

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
            profiled_bound_method = getattr(
                term,
                "_profile_forces_from_binding",
                None,
            )
            route_started = None
            if (
                self.route_profiler is not None
                and binding is not NotImplemented
                and callable(profiled_bound_method)
            ):
                forces = profiled_bound_method(
                    eval_positions,
                    binding,
                    self.route_profiler,
                )
            else:
                if self.route_profiler is not None:
                    route_started = self.route_profiler.start()
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
            if route_started is not None:
                self.route_profiler.finish(
                    "other_force_terms",
                    route_started,
                    forces,
                )
            aggregation_started = (
                None
                if self.route_profiler is None
                else self.route_profiler.start()
            )
            total_forces = total_forces + as_mx_array(forces)
            if aggregation_started is not None:
                self.route_profiler.finish(
                    "force_aggregation",
                    aggregation_started,
                    total_forces,
                )
        if not self.force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        redistribution_started = (
            None
            if self.route_profiler is None
            else self.route_profiler.start()
        )
        redistributed = _redistribute_forces(
            total_forces,
            eval_positions,
            self.virtual_sites,
        )
        if redistribution_started is not None:
            self.route_profiler.finish(
                "force_aggregation",
                redistribution_started,
                redistributed,
            )
        return redistributed


class _PreparedForcePipeline:
    """Own setup-validated force terms and cache their neighbor binding."""

    def __init__(
        self,
        force_terms: tuple[object, ...],
        *,
        cell: Cell | None,
        virtual_sites: VirtualSiteManager | None,
        route_profiler: _ExclusiveRouteProfiler | None,
    ) -> None:
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        self.force_terms = force_terms
        self.cell = cell
        self.virtual_sites = virtual_sites
        self.route_profiler = route_profiler
        self._cached_neighbor_list: NeighborList | None = None
        self._cached_binding: _BoundForcePipeline | None = None

    @classmethod
    def prepare(
        cls,
        force_terms: tuple[object, ...],
        *,
        cell: Cell | None,
        virtual_sites: VirtualSiteManager | None = None,
        route_profiler: _ExclusiveRouteProfiler | None = None,
    ) -> _PreparedForcePipeline:
        """Create a fixed-cell force pipeline.

        Args:
            force_terms: Immutable force terms for this run segment.
            cell: Current periodic cell, or ``None`` for nonperiodic work.
            virtual_sites: Optional virtual-site mapping.
            route_profiler: Optional synchronized benchmark-only route profiler.

        Returns:
            A pipeline ready to bind a neighbor-list generation.
        """

        return cls(
            tuple(force_terms),
            cell=cell,
            virtual_sites=virtual_sites,
            route_profiler=route_profiler,
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
            route_profiler=self.route_profiler,
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
