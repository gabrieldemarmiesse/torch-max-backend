import pytest
import torch
from torch_max_backend import get_accelerators

import os

os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"


@pytest.fixture(params=["cpu", "cuda"])
def device(request, gpu_available: bool):
    device_name = request.param
    if not gpu_available and device_name == "cuda":
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture
def gpu_available() -> bool:
    return len(list(get_accelerators())) > 1


@pytest.fixture(params=[(3,), (2, 3)])
def tensor_shapes(request):
    return request.param


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
