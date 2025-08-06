import pytest
import torch


@pytest.fixture
def tensor_shapes():
    return [(3,), (2, 3), (2, 3, 4)]


@pytest.fixture
def devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Fixture that provides device parametrization with automatic CUDA skip."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device_name
