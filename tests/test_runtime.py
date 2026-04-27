from mlx_atomistic.runtime import RuntimeInfo


def test_runtime_info_shape():
    info = RuntimeInfo(mlx_version="0.0", default_device="Device(gpu, 0)", metal_available=True)

    assert info.mlx_version == "0.0"
    assert info.metal_available is True
