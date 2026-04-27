import os

import mlx.core as mx

os.environ["MLX_ATOMISTIC_DEVICE"] = "cpu"
mx.set_default_device(mx.cpu)
mx.set_default_stream(mx.new_stream(mx.cpu))


def pytest_configure():
    mx.set_default_device(mx.cpu)
    mx.set_default_stream(mx.new_stream(mx.cpu))
