"""Tests for the bounded periodic SCF development-gate command."""

from __future__ import annotations

import argparse

import pytest

from mlx_atomistic.benchmarks.dft_scf_smell import (
    _hpsi_shape_profile,
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


def test_hpsi_shape_profile_ignores_started_events_and_selects_one_tail() -> None:
    events = [
        {
            "event": "kpoint_batch",
            "status": "started",
            "logical_vector_counts": [2],
            "lane_capacity": 8,
            "vector_count": 16,
        },
        *[
            {
                "event": "kpoint_batch",
                "status": "completed",
                "logical_vector_counts": [2],
                "lane_capacity": 8,
                "vector_count": 16,
            }
            for _ in range(4)
        ],
        {
            "event": "completion",
            "status": "converged",
        },
    ]

    profile = _hpsi_shape_profile(events)

    assert profile["baseline_calls"] == 4
    assert profile["baseline_logical_vector_equivalents"] == 8
    assert profile["baseline_submitted_vector_equivalents"] == 512
    assert profile["selected_tail_capacity"] == {"lanes": 1, "vectors": 4}


def test_hpsi_shape_profile_stops_when_no_tail_meets_reduction_gate() -> None:
    profile = _hpsi_shape_profile(
        [
            {
                "event": "kpoint_batch",
                "status": "completed",
                "logical_vector_counts": [16] * 8,
                "lane_capacity": 8,
                "vector_count": 16,
            }
        ]
    )

    assert profile["selected_tail_capacity"] is None
