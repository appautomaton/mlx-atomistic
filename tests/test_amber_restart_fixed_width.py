import pytest

from mlx_atomistic.prep.topology_import import _split_amber_restart_line

# AMBER restart/inpcrd coordinate lines are fixed-width 6F12.7. A value that
# needs all twelve columns (any coordinate <= -100.0000000, or >= 1000.0000000)
# leaves no space before the next field, so whitespace splitting merges the two
# into one unparseable token and the import fails with
# "unsupported_terms:amber_malformed_topology".


def test_split_recovers_fields_that_run_together():
    # Two adjacent fields with no separating space: -9.3590097 then -102.7707192.
    line = "  -9.3590097-102.7707192  19.9855003   1.2345678   2.3456789   3.4567890"
    assert line.split()[0] == "-9.3590097-102.7707192"  # the shape that used to break
    assert _split_amber_restart_line(line) == [
        "-9.3590097",
        "-102.7707192",
        "19.9855003",
        "1.2345678",
        "2.3456789",
        "3.4567890",
    ]


def test_split_handles_every_field_full():
    assert _split_amber_restart_line("-123.4567890-234.5678901") == [
        "-123.4567890",
        "-234.5678901",
    ]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("  1.0 2.0 3.0", ["1.0", "2.0", "3.0"]),  # free format, unchanged
        ("   1.2345678   2.3456789", ["1.2345678", "2.3456789"]),  # short final line
        ("  1.0D-01   2.0D+00", ["1.0D-01", "2.0D+00"]),  # Fortran D exponents
        ("           ", []),  # blank
        ("", []),
    ],
)
def test_split_preserves_existing_behaviour(line, expected):
    assert _split_amber_restart_line(line) == expected


def test_split_leaves_genuinely_corrupt_lines_to_the_caller():
    # Returning the whitespace tokens keeps the caller's float() ValueError path,
    # so a malformed file is still reported as malformed rather than silently read.
    assert _split_amber_restart_line("  1.0  abc") == ["1.0", "abc"]