"""Tests for the bounded periodic SCF development-gate command."""

from __future__ import annotations

import argparse

import pytest

from mlx_atomistic.benchmarks.dft_scf_smell import (
    _owner_points,
    _parser,
    _positive_integer,
)


def test_smell_parser_requires_explicit_science_inputs() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--mode", "adaptive"])


def test_positive_integer_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_integer("0")


def test_owner_points_selects_only_requested_representatives() -> None:
    workload = {
        "physics": {
            "kpoints": [
                {"index": 0, "role": "owner"},
                {"index": 1, "role": "partner"},
                {"index": 2, "role": "owner"},
            ]
        }
    }

    assert [point["index"] for point in _owner_points(workload, 2)] == [0, 2]


def test_owner_points_fails_closed_when_request_exceeds_manifest() -> None:
    workload = {"physics": {"kpoints": [{"index": 0, "role": "owner"}]}}

    with pytest.raises(ValueError, match="contains only 1 owners"):
        _owner_points(workload, 2)
