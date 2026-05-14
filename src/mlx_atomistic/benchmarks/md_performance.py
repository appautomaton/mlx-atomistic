"""Run repeatable full-MD performance benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_atomistic.initialize import fcc_lattice, thermal_velocities
from mlx_atomistic.io import load_npz_trajectory
from mlx_atomistic.md import (
    LangevinThermostat,
    LennardJonesPotential,
    SimulationConfig,
    simulate_nvt,
)
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.nonbonded import (
    NonbondedBackend,
    choose_nonbonded_backend,
    estimate_dense_nonbonded_bytes,
)
from mlx_atomistic.runtime import get_runtime_info

BenchmarkMode = Literal["auto", "dense", "dynamic-neighbor"]


@dataclass(frozen=True)
class MDPerformanceResult:
    """One full-MD benchmark row."""

    case: str
    mode: str
    particles: int
    replicas: int
    steps: int
    dt: float
    simulated_ps: float
    wall_s: float
    steps_per_s: float
    ps_per_s: float
    frames: int
    diagnostic_points: int
    backend: str
    estimated_dense_bytes: int
    final_pair_count: int
    rebuild_count: int
    max_constraint_error: float
    energy_drift: float
    relative_energy_drift: float
    mean_temperature: float
    final_temperature: float
    max_rss_mb: float
    finite: bool

    def to_dict(self) -> dict:
        """Return a JSON- and CSV-safe row."""

        return asdict(self)


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        msg = "value must contain positive integers"
        raise ValueError(msg)
    return values


def _max_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB. Keep this diagnostic approximate.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _finite_result(result) -> bool:
    arrays = [
        np.asarray(result.sampled_positions),
        np.asarray(result.potential_energy),
        np.asarray(result.kinetic_energy),
        np.asarray(result.total_energy),
        np.asarray(result.temperature),
        np.asarray(result.constraint_max_error),
    ]
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _batched_lj_energy_forces(
    positions: mx.array,
    cell,
    *,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float | None = 2.5,
    shift: bool = True,
) -> tuple[mx.array, mx.array]:
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    if cell is not None:
        displacement = cell.minimum_image(displacement)
    r2 = mx.sum(displacement * displacement, axis=-1)
    pair_mask = r2 > 0.0
    if cutoff is not None:
        pair_mask = pair_mask & (r2 < cutoff * cutoff)

    safe_r2 = mx.where(pair_mask, r2, 1.0)
    sigma2_over_r2 = (sigma * sigma) / safe_r2
    inv_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2
    inv_r12 = inv_r6 * inv_r6
    pair_energy = 4.0 * epsilon * (inv_r12 - inv_r6)
    if shift and cutoff is not None:
        sigma2_over_rc2 = (sigma * sigma) / (cutoff * cutoff)
        inv_rc6 = sigma2_over_rc2 * sigma2_over_rc2 * sigma2_over_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        pair_energy = pair_energy - 4.0 * epsilon * (inv_rc12 - inv_rc6)
    pair_energy = mx.where(pair_mask, pair_energy, 0.0)
    scalar = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / safe_r2
    scalar = mx.where(pair_mask, scalar, 0.0)
    forces = mx.sum(scalar[:, :, :, None] * displacement, axis=2)
    return 0.5 * mx.sum(pair_energy, axis=(1, 2)), forces


def _batched_kinetic_energy(velocities: mx.array) -> mx.array:
    return 0.5 * mx.sum(velocities * velocities, axis=(1, 2))


def _batched_temperature(velocities: mx.array) -> mx.array:
    dof = max(1, int(velocities.shape[1]) * 3 - 3)
    return 2.0 * _batched_kinetic_energy(velocities) / dof


def _backend_for_case(
    *,
    requested: NonbondedBackend,
    particles: int,
    pairs_provided: bool,
) -> str:
    estimated = estimate_dense_nonbonded_bytes(particles, components="lj")
    if pairs_provided:
        return "dynamic-neighbor+mlx_pairs"
    return choose_nonbonded_backend(
        requested=requested,
        n_atoms=particles,
        pairs_provided=False,
        estimated_dense_bytes=estimated,
        memory_budget_bytes=None,
    )


def run_synthetic_case(
    *,
    particles: int,
    steps: int,
    dt: float,
    mode: BenchmarkMode = "auto",
    dense_threshold: int = 2048,
    sample_interval: int | None = None,
    diagnostic_interval: int | None = None,
    neighbor_check_interval: int = 1,
    temperature: float = 1.0,
    friction: float = 0.5,
    density: float = 0.8,
    seed: int = 11,
) -> MDPerformanceResult:
    """Run one synthetic LJ NVT benchmark case."""

    if sample_interval is None:
        sample_interval = max(1, steps)
    if diagnostic_interval is None:
        diagnostic_interval = max(1, sample_interval)
    positions, cell = fcc_lattice(particles, density=density)
    velocities = thermal_velocities(particles, temperature=temperature, seed=seed)

    use_neighbor = mode == "dynamic-neighbor" or (
        mode == "auto" and particles > dense_threshold
    )
    if mode == "dense":
        backend: NonbondedBackend = "mlx_dense"
    elif use_neighbor:
        backend = "mlx_pairs"
    else:
        backend = "auto"
    potential = LennardJonesPotential(cutoff=2.5, backend=backend)
    neighbor_manager = (
        NeighborListManager(
            cell,
            cutoff=potential.cutoff or 2.5,
            skin=0.4,
            check_interval=neighbor_check_interval,
        )
        if use_neighbor
        else None
    )

    config = SimulationConfig(
        dt=dt,
        steps=steps,
        sample_interval=sample_interval,
        diagnostic_interval=diagnostic_interval,
        compile_force_evaluator=neighbor_manager is None,
    )
    start = perf_counter()
    result = simulate_nvt(
        positions,
        velocities,
        cell=cell,
        force_terms=potential,
        neighbor_manager=neighbor_manager,
        config=config,
        thermostat=LangevinThermostat(temperature=temperature, friction=friction, seed=seed),
    )
    mx.eval(
        result.sampled_positions,
        result.sampled_velocities,
        result.total_energy,
        result.temperature,
        result.constraint_max_error,
        result.pair_count,
        result.rebuild_count,
    )
    wall_s = perf_counter() - start

    total_energy = np.asarray(result.total_energy, dtype=np.float64)
    initial_energy = float(total_energy[0])
    energy_drift = float(total_energy[-1] - initial_energy)
    relative_energy_drift = energy_drift / max(abs(initial_energy), 1e-12)
    simulated_ps = steps * dt
    return MDPerformanceResult(
        case="synthetic_lj",
        mode=mode,
        particles=particles,
        replicas=1,
        steps=steps,
        dt=dt,
        simulated_ps=simulated_ps,
        wall_s=wall_s,
        steps_per_s=steps / wall_s if wall_s > 0.0 else 0.0,
        ps_per_s=simulated_ps / wall_s if wall_s > 0.0 else 0.0,
        frames=int(np.asarray(result.sampled_positions).shape[0]),
        diagnostic_points=int(np.asarray(result.diagnostic_steps).shape[0]),
        backend=_backend_for_case(
            requested=backend,
            particles=particles,
            pairs_provided=use_neighbor,
        ),
        estimated_dense_bytes=estimate_dense_nonbonded_bytes(particles, components="lj"),
        final_pair_count=int(np.asarray(result.pair_count)[-1]),
        rebuild_count=int(np.asarray(result.rebuild_count)[-1]),
        max_constraint_error=float(np.max(np.asarray(result.constraint_max_error))),
        energy_drift=energy_drift,
        relative_energy_drift=relative_energy_drift,
        mean_temperature=float(np.mean(np.asarray(result.temperature))),
        final_temperature=float(np.asarray(result.temperature)[-1]),
        max_rss_mb=_max_rss_mb(),
        finite=_finite_result(result),
    )


def run_batched_synthetic_case(
    *,
    particles: int,
    replicas: int,
    steps: int,
    dt: float,
    sample_interval: int | None = None,
    diagnostic_interval: int | None = None,
    temperature: float = 1.0,
    friction: float = 0.5,
    density: float = 0.8,
    seed: int = 11,
) -> MDPerformanceResult:
    """Run a dense LJ NVT benchmark with independent replicas in one MLX graph."""

    if replicas <= 0:
        msg = "replicas must be positive"
        raise ValueError(msg)
    if sample_interval is None:
        sample_interval = max(1, steps)
    if diagnostic_interval is None:
        diagnostic_interval = max(1, sample_interval)

    base_positions, cell = fcc_lattice(particles, density=density)
    positions = mx.stack([base_positions for _ in range(replicas)])
    velocities = mx.stack(
        [
            thermal_velocities(particles, temperature=temperature, seed=seed + replica)
            for replica in range(replicas)
        ]
    )
    potential_energy, forces = _batched_lj_energy_forces(positions, cell)
    sampled_positions = [positions]
    diagnostic_steps = [0]
    total_energies = [potential_energy + _batched_kinetic_energy(velocities)]
    temperatures = [_batched_temperature(velocities)]

    key = mx.random.key(seed)
    velocity_decay = float(np.exp(-friction * dt))
    noise_scale = float(np.sqrt((1.0 - velocity_decay * velocity_decay) * temperature))
    start = perf_counter()
    for step in range(1, steps + 1):
        velocities_half = velocities + 0.5 * dt * forces
        positions = positions + 0.5 * dt * velocities_half
        positions = cell.wrap(positions)
        keys = mx.random.split(key, 2)
        key = keys[0]
        noise = mx.random.normal(velocities.shape, key=keys[1])
        velocities = velocity_decay * velocities_half + noise_scale * noise
        positions = positions + 0.5 * dt * velocities
        positions = cell.wrap(positions)
        potential_energy, forces = _batched_lj_energy_forces(positions, cell)
        velocities = velocities + 0.5 * dt * forces

        if step % sample_interval == 0 or step == steps:
            sampled_positions.append(positions)
        if step % diagnostic_interval == 0 or step == steps:
            diagnostic_steps.append(step)
            total_energies.append(potential_energy + _batched_kinetic_energy(velocities))
            temperatures.append(_batched_temperature(velocities))
        if step % 25 == 0 or step == steps:
            mx.eval(positions, velocities, forces, potential_energy)

    sampled_positions_array = mx.stack(sampled_positions)
    total_energy_array = mx.stack(total_energies)
    temperature_array = mx.stack(temperatures)
    mx.eval(sampled_positions_array, total_energy_array, temperature_array)
    wall_s = perf_counter() - start

    total_energy = np.asarray(total_energy_array, dtype=np.float64)
    per_replica_drift = total_energy[-1] - total_energy[0]
    mean_initial_energy = float(np.mean(total_energy[0]))
    energy_drift = float(np.mean(per_replica_drift))
    relative_energy_drift = energy_drift / max(abs(mean_initial_energy), 1e-12)
    simulated_ps = steps * dt
    return MDPerformanceResult(
        case="synthetic_lj_replicas",
        mode="batched-dense",
        particles=particles,
        replicas=replicas,
        steps=steps,
        dt=dt,
        simulated_ps=simulated_ps,
        wall_s=wall_s,
        steps_per_s=(steps * replicas) / wall_s if wall_s > 0.0 else 0.0,
        ps_per_s=(simulated_ps * replicas) / wall_s if wall_s > 0.0 else 0.0,
        frames=int(sampled_positions_array.shape[0]),
        diagnostic_points=len(diagnostic_steps),
        backend="batched_mlx_dense",
        estimated_dense_bytes=replicas * estimate_dense_nonbonded_bytes(
            particles,
            components="lj",
        ),
        final_pair_count=particles * (particles - 1) // 2,
        rebuild_count=0,
        max_constraint_error=0.0,
        energy_drift=energy_drift,
        relative_energy_drift=relative_energy_drift,
        mean_temperature=float(np.mean(np.asarray(temperature_array))),
        final_temperature=float(np.mean(np.asarray(temperature_array[-1]))),
        max_rss_mb=_max_rss_mb(),
        finite=bool(
            np.all(np.isfinite(np.asarray(sampled_positions_array)))
            and np.all(np.isfinite(total_energy))
            and np.all(np.isfinite(np.asarray(temperature_array)))
        ),
    )


def run_atp_case(
    *,
    prepared: Path,
    steps: int,
    dt: float,
    sample_interval: int,
    diagnostic_interval: int,
    constraint_max_iterations: int,
) -> MDPerformanceResult:
    """Run the prepared ATP-pocket benchmark through mlx_atomistic.prep."""

    from mlx_atomistic.prep.runner import run_mlx

    with tempfile.TemporaryDirectory(prefix="mlx-atp-perf-") as temp_dir:
        out = Path(temp_dir) / "trajectory.npz"
        start = perf_counter()
        run_mlx(
            prepared,
            out=out,
            steps=steps,
            sample_interval=sample_interval,
            dt=dt,
            require_production=True,
            minimize_steps=0,
            equilibration_steps=0,
            constraint_max_iterations=constraint_max_iterations,
            diagnostic_interval=diagnostic_interval,
        )
        wall_s = perf_counter() - start
        record = load_npz_trajectory(out)

    total_energy = np.asarray(record.total_energy, dtype=np.float64)
    initial_energy = float(total_energy[0])
    energy_drift = float(total_energy[-1] - initial_energy)
    relative_energy_drift = energy_drift / max(abs(initial_energy), 1e-12)
    particles = int(record.sampled_positions.shape[1])
    simulated_ps = steps * dt
    return MDPerformanceResult(
        case="atp_pocket",
        mode="production_mlx",
        particles=particles,
        replicas=1,
        steps=steps,
        dt=dt,
        simulated_ps=simulated_ps,
        wall_s=wall_s,
        steps_per_s=steps / wall_s if wall_s > 0.0 else 0.0,
        ps_per_s=simulated_ps / wall_s if wall_s > 0.0 else 0.0,
        frames=int(record.sampled_positions.shape[0]),
        diagnostic_points=int(record.diagnostic_steps.shape[0]),
        backend="artifact_force_terms",
        estimated_dense_bytes=estimate_dense_nonbonded_bytes(particles, components="combined"),
        final_pair_count=int(record.pair_count[-1]),
        rebuild_count=int(record.rebuild_count[-1]),
        max_constraint_error=float(np.max(record.constraint_max_error)),
        energy_drift=energy_drift,
        relative_energy_drift=relative_energy_drift,
        mean_temperature=float(np.mean(record.temperature)),
        final_temperature=float(record.temperature[-1]),
        max_rss_mb=_max_rss_mb(),
        finite=bool(
            np.all(np.isfinite(record.sampled_positions))
            and np.all(np.isfinite(record.total_energy))
            and np.all(np.isfinite(record.temperature))
        ),
    )


def build_payload(
    *,
    sizes: tuple[int, ...],
    steps: int,
    dt: float,
    mode: BenchmarkMode,
    dense_threshold: int,
    sample_interval: int,
    diagnostic_interval: int,
    neighbor_check_interval: int,
    replicas: int = 1,
    include_atp: bool = False,
    prepared: Path | None = None,
    constraint_max_iterations: int = 4,
) -> dict:
    """Run the selected benchmark cases and return a payload."""

    if replicas == 1:
        cases = [
            run_synthetic_case(
                particles=size,
                steps=steps,
                dt=dt,
                mode=mode,
                dense_threshold=dense_threshold,
                sample_interval=sample_interval,
                diagnostic_interval=diagnostic_interval,
                neighbor_check_interval=neighbor_check_interval,
            )
            for size in sizes
        ]
    else:
        cases = [
            run_batched_synthetic_case(
                particles=size,
                replicas=replicas,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                diagnostic_interval=diagnostic_interval,
            )
            for size in sizes
        ]
    if include_atp:
        if prepared is None:
            msg = "--include-atp requires --prepared"
            raise ValueError(msg)
        cases.append(
            run_atp_case(
                prepared=prepared,
                steps=steps,
                dt=dt,
                sample_interval=sample_interval,
                diagnostic_interval=diagnostic_interval,
                constraint_max_iterations=constraint_max_iterations,
            )
        )

    rows = [case.to_dict() for case in cases]
    return {
        "runtime": asdict(get_runtime_info()),
        "config": {
            "sizes": list(sizes),
            "steps": steps,
            "dt": dt,
            "mode": mode,
            "dense_threshold": dense_threshold,
            "sample_interval": sample_interval,
            "diagnostic_interval": diagnostic_interval,
            "neighbor_check_interval": neighbor_check_interval,
            "replicas": replicas,
            "include_atp": include_atp,
            "prepared": str(prepared) if prepared is not None else None,
        },
        "case_count": len(rows),
        "cases": rows,
    }


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="386,2000")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--mode", choices=["auto", "dense", "dynamic-neighbor"], default="auto")
    parser.add_argument("--dense-threshold", type=int, default=2048)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--neighbor-check-interval", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--include-atp", action="store_true")
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--constraint-max-iterations", type=int, default=4)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sizes = _parse_ints(args.sizes)
    if args.include_large:
        sizes = tuple(dict.fromkeys((*sizes, 10_000, 50_000)))
    payload = build_payload(
        sizes=sizes,
        steps=args.steps,
        dt=args.dt,
        mode=args.mode,
        dense_threshold=args.dense_threshold,
        sample_interval=args.sample_interval,
        diagnostic_interval=args.diagnostic_interval,
        neighbor_check_interval=args.neighbor_check_interval,
        replicas=args.replicas,
        include_atp=args.include_atp,
        prepared=args.prepared,
        constraint_max_iterations=args.constraint_max_iterations,
    )
    if args.csv is not None:
        _write_csv(args.csv, payload["cases"])

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = payload["runtime"]
    print(
        f"runtime mlx={runtime['mlx_version']} device={runtime['default_device']} "
        f"metal={runtime['metal_available']}"
    )
    print(
        "case,mode,particles,replicas,steps,wall_s,steps_per_s,ps_per_s,"
        "backend,pairs,rebuilds,max_constraint_error,finite"
    )
    for row in payload["cases"]:
        print(
            f"{row['case']},{row['mode']},{row['particles']},{row['replicas']},{row['steps']},"
            f"{row['wall_s']:.3f},{row['steps_per_s']:.3f},{row['ps_per_s']:.3f},"
            f"{row['backend']},{row['final_pair_count']},{row['rebuild_count']},"
            f"{row['max_constraint_error']:.6g},{row['finite']}"
        )


if __name__ == "__main__":
    main()
