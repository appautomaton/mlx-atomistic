"""Benchmark molecular mechanics force-term evaluation costs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from time import perf_counter

import mlx.core as mx

from mlx_atomistic.examples import bonded_chain_example, charged_dimer_example
from mlx_atomistic.runtime import get_runtime_info


@dataclass(frozen=True)
class ForceTermBenchmarkResult:
    term: str
    evaluations: int
    ms_per_eval: float
    energy: float


def run_term(term, positions, *, evaluations: int) -> ForceTermBenchmarkResult:
    """Measure repeated energy/force evaluations for one term."""

    energy = None
    forces = None
    start = perf_counter()
    for _ in range(evaluations):
        energy, forces = term.energy_forces(positions)
    if energy is not None and forces is not None:
        mx.eval(energy, forces)
    elapsed = perf_counter() - start
    return ForceTermBenchmarkResult(
        term=str(getattr(term, "name", type(term).__name__)),
        evaluations=evaluations,
        ms_per_eval=elapsed * 1000.0 / evaluations,
        energy=float(energy),
    )


def run_benchmark(*, evaluations: int) -> list[ForceTermBenchmarkResult]:
    """Run the default molecular mechanics force-term benchmark."""

    positions, _, _, bonded_terms = bonded_chain_example()
    charged_positions, _, _, charged_terms = charged_dimer_example()
    results = [run_term(term, positions, evaluations=evaluations) for term in bonded_terms]
    results.extend(
        run_term(term, charged_positions, evaluations=evaluations) for term in charged_terms
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluations <= 0:
        msg = "--evaluations must be positive"
        raise ValueError(msg)

    results = run_benchmark(evaluations=args.evaluations)
    if args.json:
        print(
            json.dumps(
                {
                    "runtime": asdict(get_runtime_info()),
                    "cases": [asdict(result) for result in results],
                },
                indent=2,
            )
        )
        return

    runtime = get_runtime_info()
    print(
        f"runtime mlx={runtime.mlx_version} device={runtime.default_device} "
        f"metal={runtime.metal_available}"
    )
    for result in results:
        print(
            f"{result.term:10s} evals={result.evaluations} "
            f"ms/eval={result.ms_per_eval:.3f} energy={result.energy:.6g}"
        )


if __name__ == "__main__":
    main()
