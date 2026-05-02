import os

os.environ["MLX_ATOMISTIC_DEVICE"] = "cpu"

import mlx.core as mx

_CPU_DEVICE = mx.Device(mx.cpu, 0)
mx.set_default_device(_CPU_DEVICE)
mx.set_default_stream(mx.new_stream(_CPU_DEVICE))


def pytest_configure():
    mx.set_default_device(_CPU_DEVICE)
    mx.set_default_stream(mx.new_stream(_CPU_DEVICE))
