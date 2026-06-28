import os

os.environ["MLX_ATOMISTIC_DEVICE"] = "cpu"

import mlx.core as mx
import pytest

_CPU_DEVICE = mx.Device(mx.cpu, 0)
mx.set_default_device(_CPU_DEVICE)
mx.set_default_stream(mx.new_stream(_CPU_DEVICE))


def pytest_configure():
    mx.set_default_device(_CPU_DEVICE)
    mx.set_default_stream(mx.new_stream(_CPU_DEVICE))


def pytest_addoption(parser):
    parser.addoption(
        "--run-data",
        action="store_true",
        default=False,
        help="run tests that require gitignored or externally mounted data",
    )
    parser.addoption(
        "--run-reference",
        action="store_true",
        default=False,
        help="run tests that require optional reference engines",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that require a Metal GPU",
    )


def pytest_collection_modifyitems(config, items):
    skip_data = pytest.mark.skip(reason="requires --run-data")
    skip_reference = pytest.mark.skip(reason="requires --run-reference")
    skip_gpu = pytest.mark.skip(reason="requires --run-gpu")
    run_data = config.getoption("--run-data")
    run_reference = config.getoption("--run-reference")
    run_gpu = config.getoption("--run-gpu")
    for item in items:
        if not run_data and item.get_closest_marker("data"):
            item.add_marker(skip_data)
        if not run_reference and item.get_closest_marker("reference"):
            item.add_marker(skip_reference)
        if not run_gpu and item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
