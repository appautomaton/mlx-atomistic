"""Thin read-only CLI adapter for periodic DFT checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mlx_atomistic._artifact_identity import canonical_json_bytes
from mlx_atomistic.dft.artifacts import inspect_periodic_scf_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlx_atomistic.benchmarks.dft_artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser(
        "inspect",
        help="validate one completed periodic-SCF checkpoint without modifying it",
    )
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--expected-execution-context")
    inspect.add_argument("--json", action="store_true")
    return parser


def _print(payload: object, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")
    else:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only checkpoint inspection command."""

    args = _parser().parse_args(argv)
    try:
        expected = None
        if args.expected_execution_context is not None:
            expected = json.loads(
                Path(args.expected_execution_context).expanduser().read_bytes()
            )
        payload = inspect_periodic_scf_checkpoint(
            args.artifact,
            expected_execution_context=expected,
        )
    except (OSError, TypeError, ValueError) as error:
        _print(
            {
                "status": "blocked",
                "error": str(error),
            },
            as_json=args.json,
        )
        return 2
    _print(payload, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
