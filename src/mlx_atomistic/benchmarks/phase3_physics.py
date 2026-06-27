"""Fast benchmark coverage for Phase 3 molecular-physics features."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from mlx_atomistic.benchmarks import (
    default_benchmark_command,
    get_hardware_info,
    normalize_benchmark_payload,
    normalize_benchmark_row,
)
from mlx_atomistic.forcefields import NonbondedPotential
from mlx_atomistic.gbsa import GBSAForcePotential
from mlx_atomistic.md import SimulationConfig, SimulationState
from mlx_atomistic.replica_exchange import simulate_replica_exchange
from mlx_atomistic.runtime import get_runtime_info
from mlx_atomistic.virtual_sites import VirtualSiteManager, tip4p_ew_virtual_site

COMMAND = default_benchmark_command("phase3_physics")
COMPARISON_OUTPUT_ROOT = "outputs/benchmarks/same-workload-openmm-comparison"


COMPARISON_PAIR_METADATA = {
    "gbsa_obc": {
        "comparison_pair_id": "gbsa-obc-small",
        "comparison_metric_family": "ms/eval",
        "comparison_raw_output_path": f"{COMPARISON_OUTPUT_ROOT}/mlx-gbsa-obc-small.json",
    },
    "tip4p_ew": {
        "comparison_pair_id": "tip4p-ew-water",
        "comparison_metric_family": "ms/eval",
        "comparison_raw_output_path": f"{COMPARISON_OUTPUT_ROOT}/mlx-tip4p-ew-water.json",
    },
    "virtual_sites": {
        "comparison_pair_id": "tip4p-ew-water",
        "comparison_metric_family": "ms/eval",
        "comparison_raw_output_path": f"{COMPARISON_OUTPUT_ROOT}/mlx-tip4p-ew-water.json",
    },
}


@dataclass(frozen=True)
class _HarmonicWell:
    k: float = 1.0
    name: str = "harmonic"
    supports_virial: bool = True

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        positions = mx.array(positions)
        return 0.5 * self.k * mx.sum(positions * positions), -self.k * positions


def _time_repeated(
    fn: Callable[[], object],
    *,
    evaluations: int,
    eval_outputs: Callable[[object], None],
) -> tuple[float, object]:
    value = None
    start = perf_counter()
    for _ in range(evaluations):
        value = fn()
        eval_outputs(value)
    elapsed = perf_counter() - start
    return elapsed * 1000.0 / evaluations, value


def _tip4p_real_positions(waters: int) -> mx.array:
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.7569503, 0.5858823, 0.0],
            [0.7569503, -0.5858823, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.asarray([[3.0 * idx, 0.0, 0.0] for idx in range(waters)], dtype=np.float32)
    return mx.array((base[None, :, :] + offsets[:, None, :]).reshape(waters * 3, 3))


def _tip4p_manager(waters: int) -> tuple[VirtualSiteManager, mx.array]:
    positions = _tip4p_real_positions(waters)
    sites = tuple(
        tip4p_ew_virtual_site(3 * idx, 3 * idx + 1, 3 * idx + 2) for idx in range(waters)
    )
    return VirtualSiteManager(sites, n_real_atoms=int(positions.shape[0])), positions


def _phase3_row(row: dict, *, fixture: str, evaluations: int) -> dict:
    normalized = normalize_benchmark_row(
        row,
        benchmark_name="phase3_physics",
        fixture=fixture,
        timing_metric="ms_per_eval",
        evaluation_count=evaluations,
    )
    comparison = COMPARISON_PAIR_METADATA.get(str(normalized.get("feature")))
    if comparison is not None:
        normalized.update(
            {
                **comparison,
                "comparison_role": "mlx",
                "comparison_command": COMMAND,
            }
        )
    return normalized


def _virtual_site_rows(*, evaluations: int, waters: int) -> list[dict]:
    manager, positions = _tip4p_manager(waters)

    reconstruct_ms, extended = _time_repeated(
        lambda: manager.extend_positions(positions),
        evaluations=evaluations,
        eval_outputs=lambda value: mx.eval(value),
    )
    full_positions = manager.extend_positions(positions)
    forces = mx.concatenate(
        [
            mx.zeros_like(full_positions[: manager.n_real_atoms]),
            mx.ones_like(full_positions[manager.n_real_atoms :]),
        ],
        axis=0,
    )
    redistribute_ms, redistributed = _time_repeated(
        lambda: manager.redistribute_forces(forces, full_positions),
        evaluations=evaluations,
        eval_outputs=lambda value: mx.eval(value),
    )

    return [
        _phase3_row(
            {
                "feature": "virtual_sites",
                "case": "tip4p_ew_reconstruct",
                "operation": "reconstruct_positions",
                "waters": waters,
                "atoms": manager.n_total_atoms,
                "virtual_site_count": manager.n_virtual_sites,
                "ms_per_eval": reconstruct_ms,
                "finite": bool(np.isfinite(np.asarray(extended)).all()),
            },
            fixture="tip4p_ew_virtual_sites",
            evaluations=evaluations,
        ),
        _phase3_row(
            {
                "feature": "virtual_sites",
                "case": "tip4p_ew_force_redistribute",
                "operation": "redistribute_forces",
                "waters": waters,
                "atoms": manager.n_total_atoms,
                "virtual_site_count": manager.n_virtual_sites,
                "ms_per_eval": redistribute_ms,
                "force_norm": float(mx.sqrt(mx.sum(redistributed * redistributed))),
            },
            fixture="tip4p_ew_virtual_sites",
            evaluations=evaluations,
        ),
        _phase3_row(
            {
                "feature": "tip4p_ew",
                "case": "tip4p_ew_m_site",
                "operation": "m_site_reconstruction",
                "waters": waters,
                "atoms": manager.n_total_atoms,
                "virtual_site_count": manager.n_virtual_sites,
                "ms_per_eval": reconstruct_ms,
                "tip4p_helper": "tip4p_ew_virtual_site",
                "json_csv_status": "available",
            },
            fixture="tip4p_ew",
            evaluations=evaluations,
        ),
    ]


def _gbsa_rows(*, evaluations: int, atoms: int) -> list[dict]:
    positions = mx.array(
        np.stack(
            [
                np.linspace(0.0, 1.5 * (atoms - 1), atoms),
                np.sin(np.arange(atoms, dtype=np.float32)) * 0.2,
                np.cos(np.arange(atoms, dtype=np.float32)) * 0.2,
            ],
            axis=1,
        ).astype(np.float32)
    )
    charges = np.where(np.arange(atoms) % 2 == 0, 0.4, -0.35).astype(np.float32)
    radii = np.linspace(1.45, 1.75, atoms, dtype=np.float32)
    scales = np.linspace(0.72, 0.85, atoms, dtype=np.float32)
    term = GBSAForcePotential(charges=charges, radius=radii, scale=scales)

    energy_forces_ms, energy_forces = _time_repeated(
        lambda: term.energy_forces(positions),
        evaluations=evaluations,
        eval_outputs=lambda value: mx.eval(value[0], value[1]),
    )
    surface_ms, surface_energy = _time_repeated(
        lambda: term.ace_surface_area_energy(positions),
        evaluations=evaluations,
        eval_outputs=lambda value: mx.eval(value),
    )
    energy, forces = energy_forces

    return [
        _phase3_row(
            {
                "feature": "gbsa_obc",
                "case": "gbsa_obc_energy_forces",
                "operation": "obc_pair_accumulation_and_force",
                "atoms": atoms,
                "obc_pair_count": atoms * (atoms - 1) // 2,
                "ms_per_eval": energy_forces_ms,
                "energy": float(energy),
                "force_norm": float(mx.sqrt(mx.sum(forces * forces))),
            },
            fixture="synthetic_gbsa_obc",
            evaluations=evaluations,
        ),
        _phase3_row(
            {
                "feature": "gbsa_obc",
                "case": "gbsa_ace_surface_area",
                "operation": "surface_area_term",
                "atoms": atoms,
                "obc_pair_count": atoms * (atoms - 1) // 2,
                "ms_per_eval": surface_ms,
                "surface_area_energy": float(surface_energy),
            },
            fixture="synthetic_gbsa_obc",
            evaluations=evaluations,
        ),
    ]


def _soft_core_row(*, evaluations: int) -> dict:
    positions = mx.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=mx.float32)
    lambdas = (0.25, 0.5, 0.75)
    base_kwargs = {
        "sigma": [1.0, 1.1],
        "epsilon": [0.25, 0.35],
        "charges": [0.5, -0.4],
        "cutoff": None,
        "lj_shift": False,
    }

    def evaluate_grid():
        outputs = []
        for lambda_value in lambdas:
            term = NonbondedPotential(
                **base_kwargs,
                lambda_lj=lambda_value,
                lambda_electrostatics=lambda_value,
            )
            outputs.append(term.energy_forces_dlambda(positions))
        return outputs

    def eval_grid(outputs):
        for energy, forces, derivatives in outputs:
            mx.eval(energy, forces, derivatives["lambda_lj"], derivatives["lambda_electrostatics"])

    lambda_grid_ms, outputs = _time_repeated(
        evaluate_grid,
        evaluations=evaluations,
        eval_outputs=eval_grid,
    )
    derivative_sum = sum(float(item[2]["lambda_lj"]) for item in outputs)

    return _phase3_row(
        {
            "feature": "soft_core_lambda",
            "case": "soft_core_energy_forces_dlambda_grid",
            "operation": "energy_forces_dlambda",
            "atoms": 2,
            "pair_count": 1,
            "lambda_grid": ";".join(str(value) for value in lambdas),
            "lambda_count": len(lambdas),
            "lambda_evaluation_count": evaluations * len(lambdas),
            "ms_per_eval": lambda_grid_ms,
            "derivative_sum": derivative_sum,
        },
        fixture="synthetic_soft_core_pair",
        evaluations=evaluations,
    )


def _replica_state(x: float) -> SimulationState:
    positions = mx.array([[x, 0.0, 0.0]], dtype=mx.float32)
    return SimulationState(
        positions=positions,
        velocities=mx.zeros_like(positions),
        masses=mx.array([1.0], dtype=mx.float32),
        forces=mx.zeros_like(positions),
    )


def _replica_exchange_row(*, evaluations: int, steps: int) -> dict:
    replicas = 2

    def run_exchange():
        return simulate_replica_exchange(
            [_replica_state(2.0), _replica_state(0.0)],
            _HarmonicWell(),
            temperatures=[1.0, 2.0],
            config=SimulationConfig(
                dt=0.001,
                steps=steps,
                sample_interval=1,
                diagnostic_interval=1,
            ),
            swap_interval=1,
            thermostat_friction=0.0,
            seed=7,
        )

    def eval_exchange(result):
        mx.eval(result.energy_history, result.state_index_history)
        for state in result.replica_states:
            mx.eval(state.positions, state.velocities, state.forces)

    ms_per_eval, result = _time_repeated(
        run_exchange,
        evaluations=evaluations,
        eval_outputs=eval_exchange,
    )
    wall_s = ms_per_eval / 1000.0
    attempts = len(result.swap_attempts)
    acceptance_rate = result.accepted_swaps / attempts if attempts else 0.0

    return _phase3_row(
        {
            "feature": "replica_exchange",
            "case": "two_replica_temperature_exchange",
            "operation": "simulate_replica_exchange",
            "atoms": replicas,
            "replicas": replicas,
            "steps": steps,
            "swap_attempts": attempts,
            "accepted_swaps": result.accepted_swaps,
            "acceptance_rate": acceptance_rate,
            "history_materialization_count": int(np.asarray(result.energy_history).size)
            + int(np.asarray(result.state_index_history).size),
            "per_replica_steps_per_s": steps / wall_s if wall_s > 0.0 else None,
            "aggregate_replica_steps_per_s": (steps * replicas) / wall_s
            if wall_s > 0.0
            else None,
            "ms_per_eval": ms_per_eval,
        },
        fixture="synthetic_replica_exchange",
        evaluations=evaluations,
    )


def build_payload(
    *,
    evaluations: int = 1,
    waters: int = 2,
    atoms: int = 4,
    replica_steps: int = 1,
) -> dict:
    """Run fast Phase 3 feature benchmark probes and return normalized rows."""

    if evaluations <= 0:
        msg = "evaluations must be positive"
        raise ValueError(msg)
    if waters <= 0:
        msg = "waters must be positive"
        raise ValueError(msg)
    if atoms < 2:
        msg = "atoms must be at least 2"
        raise ValueError(msg)
    if replica_steps <= 0:
        msg = "replica_steps must be positive"
        raise ValueError(msg)

    rows = [
        *_virtual_site_rows(evaluations=evaluations, waters=waters),
        *_gbsa_rows(evaluations=evaluations, atoms=atoms),
        _soft_core_row(evaluations=evaluations),
        _replica_exchange_row(evaluations=evaluations, steps=replica_steps),
    ]
    hardware = get_hardware_info()
    runtime = asdict(get_runtime_info())
    payload = {
        "benchmark_name": "phase3_physics",
        "fixture": "phase3_fast_synthetic",
        "hardware": hardware,
        "runtime": runtime,
        "config": {
            "evaluations": evaluations,
            "waters": waters,
            "atoms": atoms,
            "replica_steps": replica_steps,
        },
        "required_features": [
            "virtual_sites",
            "tip4p_ew",
            "gbsa_obc",
            "soft_core_lambda",
            "replica_exchange",
        ],
        "case_count": len(rows),
        "cases": rows,
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name="phase3_physics",
        fixture="phase3_fast_synthetic",
        timing_metric="ms_per_eval",
        hardware=hardware,
        runtime=runtime,
        evaluation_count=evaluations,
        finite=all(bool(row["finite"]) for row in rows),
        command=COMMAND,
    )


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=1)
    parser.add_argument("--waters", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--replica-steps", type=int, default=1)
    parser.add_argument("--csv", default=None, help="Optional path for per-case CSV output.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        evaluations=args.evaluations,
        waters=args.waters,
        atoms=args.atoms,
        replica_steps=args.replica_steps,
    )
    rows = payload["cases"]
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    runtime = get_runtime_info()
    print(
        f"runtime mlx={runtime.mlx_version} device={runtime.default_device} "
        f"metal={runtime.metal_available}"
    )
    for row in rows:
        print(
            f"{row['feature']:20s} {row['case']:36s} "
            f"evals={row['evaluation_count']} ms/eval={row['ms_per_eval']:.3f} "
            f"status={row['status']}"
        )


if __name__ == "__main__":
    main()
