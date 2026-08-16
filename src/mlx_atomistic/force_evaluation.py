"""Composable force evaluation independent of integration policy."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx

from mlx_atomistic.core import Cell, as_mx_array
from mlx_atomistic.virtual_sites import VirtualSiteManager


class ForceTerm(Protocol):
    """Protocol for composable force terms."""

    def energy_forces(
        self,
        positions: mx.array,
        cell: Cell | None = None,
        pairs: object | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return potential energy and forces.

        Args:
            positions: Particle coordinates, shape ``(n_particles, 3)``.
            cell: Optional periodic cell for minimum-image distances. Defaults to ``None``.
            pairs: Optional precomputed neighbor/pair structure. Defaults to ``None``.

        Returns:
            An ``(energy, forces)`` tuple: scalar potential energy and forces of
                shape ``(n_particles, 3)``.
        """


def _as_force_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    if isinstance(force_terms, (list, tuple)):
        if not force_terms:
            msg = "force_terms must not be empty"
            raise ValueError(msg)
        return tuple(force_terms)
    return (force_terms,)


def _named_force_terms(force_terms: ForceTerm | list[ForceTerm] | tuple[ForceTerm, ...]):
    terms = _as_force_terms(force_terms)
    seen: dict[str, int] = {}
    named_terms = []
    for term in terms:
        base_name = str(getattr(term, "name", type(term).__name__))
        seen[base_name] = seen.get(base_name, 0) + 1
        name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}"
        named_terms.append((name, term))
    return tuple(named_terms)


def _virtual_site_evaluation_positions(
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return positions
    if positions.shape[0] != virtual_sites.n_real_atoms:
        msg = "positions must contain real atoms only when virtual sites are configured"
        raise ValueError(msg)
    return virtual_sites.extend_positions(positions)


def _redistribute_virtual_site_forces(
    forces: mx.array,
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    if virtual_sites is None or virtual_sites.n_virtual_sites == 0:
        return forces
    return virtual_sites.redistribute_forces(forces, positions)


def _neighbor_evaluation_positions(
    positions: mx.array,
    virtual_sites: VirtualSiteManager | None,
) -> mx.array:
    return _virtual_site_evaluation_positions(positions, virtual_sites)


def _groupable_potential_terms(
    force_terms: tuple[ForceTerm, ...],
    pairs: object | None,
) -> tuple[ForceTerm, ...]:
    if pairs is not None:
        return ()
    # Production force terms provide analytical `energy_forces`; using those is
    # faster than differentiating summed potential energies each MD step.
    # Potential-only custom terms may opt into autograd grouping explicitly.
    return tuple(
        term
        for term in force_terms
        if bool(getattr(term, "use_autograd_forces", False))
        and callable(getattr(term, "potential_energy", None))
        and not callable(getattr(term, "energy_forces_with_components", None))
    )


def _grouped_potential_energy_forces(
    positions: mx.array,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
) -> tuple[mx.array, mx.array]:
    def total_potential_energy(pos: mx.array) -> mx.array:
        total = None
        for term in terms:
            energy = term.potential_energy(pos, cell)
            total = energy if total is None else total + energy
        if total is None:
            return _zero_energy(pos)
        return total

    energy, gradient = mx.value_and_grad(total_potential_energy)(positions)
    return energy, -gradient


def _grouped_potential_forces(
    positions: mx.array,
    terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
) -> mx.array:
    def total_potential_energy(pos: mx.array) -> mx.array:
        total = None
        for term in terms:
            energy = term.potential_energy(pos, cell)
            total = energy if total is None else total + energy
        if total is None:
            return _zero_energy(pos)
        return total

    return -mx.grad(total_potential_energy)(positions)


def _forces_from_term_demands(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None,
) -> tuple[mx.array, int, int]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    grouped_terms = _groupable_potential_terms(force_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_forces = mx.zeros_like(eval_positions)
    optimized_terms = len(grouped_terms)
    fallback_terms = 0
    if grouped_terms:
        total_forces = total_forces + _grouped_potential_forces(
            eval_positions,
            grouped_terms,
            cell=cell,
        )
    for term in force_terms:
        if id(term) in grouped_ids:
            continue
        force_method = getattr(term, "_runtime_forces", None)
        forces = (
            force_method(eval_positions, cell=cell, pairs=pairs)
            if callable(force_method)
            else NotImplemented
        )
        if forces is NotImplemented:
            _, forces = term.energy_forces(eval_positions, cell, pairs=pairs)
            fallback_terms += 1
        else:
            optimized_terms += 1
        total_forces = total_forces + as_mx_array(forces)
    if optimized_terms + fallback_terms == 0:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return (
        _redistribute_virtual_site_forces(
            total_forces,
            eval_positions,
            virtual_sites,
        ),
        optimized_terms,
        fallback_terms,
    )


def _energy_from_term_demands(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None,
) -> tuple[mx.array, int, int]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    grouped_terms = _groupable_potential_terms(force_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    optimized_terms = 0
    fallback_terms = 0
    for term in force_terms:
        if id(term) in grouped_ids:
            energy = term.potential_energy(eval_positions, cell)
            optimized_terms += 1
        else:
            energy_method = getattr(term, "_runtime_energy", None)
            energy = (
                energy_method(eval_positions, cell=cell, pairs=pairs)
                if callable(energy_method)
                else NotImplemented
            )
            if energy is NotImplemented:
                potential_energy = getattr(term, "potential_energy", None)
                if pairs is None and callable(potential_energy):
                    energy = potential_energy(eval_positions, cell)
                    optimized_terms += 1
                else:
                    energy, _ = term.energy_forces(eval_positions, cell, pairs=pairs)
                    fallback_terms += 1
            else:
                optimized_terms += 1
        total_energy = energy if total_energy is None else total_energy + energy
    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, optimized_terms, fallback_terms


def _energy_forces_from_terms(
    positions: mx.array,
    force_terms: tuple[ForceTerm, ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> tuple[mx.array, mx.array]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    grouped_terms = _groupable_potential_terms(force_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(eval_positions)
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            eval_positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for term in force_terms:
        if id(term) in grouped_ids:
            continue
        energy, forces = term.energy_forces(eval_positions, cell, pairs=pairs)
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return total_energy, _redistribute_virtual_site_forces(
        total_forces,
        eval_positions,
        virtual_sites,
    )


def _energy_forces_by_term(
    positions: mx.array,
    force_terms: tuple[tuple[str, ForceTerm], ...],
    *,
    cell: Cell | None,
    pairs: object | None,
    virtual_sites: VirtualSiteManager | None = None,
) -> tuple[mx.array, mx.array, dict[str, mx.array]]:
    real_positions = as_mx_array(positions)
    eval_positions = _virtual_site_evaluation_positions(real_positions, virtual_sites)
    unnamed_terms = tuple(term for _, term in force_terms)
    grouped_terms = _groupable_potential_terms(unnamed_terms, pairs)
    grouped_ids = {id(term) for term in grouped_terms}
    total_energy = None
    total_forces = mx.zeros_like(eval_positions)
    energy_by_term = {}
    if grouped_terms:
        energy, forces = _grouped_potential_energy_forces(
            eval_positions,
            grouped_terms,
            cell=cell,
        )
        total_energy = energy
        total_forces = total_forces + forces
    for name, term in force_terms:
        if id(term) in grouped_ids:
            energy_by_term[name] = term.potential_energy(eval_positions, cell)
            continue

        combined_components = getattr(
            term,
            "_runtime_energy_forces_with_components",
            None,
        )
        if not callable(combined_components):
            combined_components = getattr(
                term,
                "energy_forces_with_components",
                None,
            )
        if callable(combined_components):
            energy, forces, components = combined_components(eval_positions, cell, pairs=pairs)
            for component_name, component_energy in components.items():
                if _is_energy_component(component_energy):
                    energy_by_term[f"{name}.{component_name}"] = component_energy
        else:
            energy, forces = term.energy_forces(eval_positions, cell, pairs=pairs)
            component_energies = getattr(term, "component_energies", None)
            if callable(component_energies):
                for component_name, component_energy in component_energies(
                    eval_positions,
                    cell=cell,
                    pairs=pairs,
                ).items():
                    if _is_energy_component(component_energy):
                        energy_by_term[f"{name}.{component_name}"] = component_energy
            else:
                energy_by_term[name] = energy
        total_energy = energy if total_energy is None else total_energy + energy
        total_forces = total_forces + forces

    if total_energy is None:
        msg = "force_terms must not be empty"
        raise ValueError(msg)
    return (
        total_energy,
        _redistribute_virtual_site_forces(total_forces, eval_positions, virtual_sites),
        energy_by_term,
    )


def _is_energy_component(value: object) -> bool:
    return isinstance(value, (mx.array, int, float))


def _zero_energy(positions: mx.array) -> mx.array:
    return mx.sum(positions[:, 0] * 0.0)


def _dense_pair_count(positions: mx.array) -> int:
    n_particles = positions.shape[0]
    return n_particles * (n_particles - 1) // 2
