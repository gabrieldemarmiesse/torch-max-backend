import pytest
import torch

from torch_max_backend.profiler import profile
from torch_max_backend import get_accelerators

import os


# Register your helper module for assertion rewriting
pytest.register_assert_rewrite("torch_max_backend.testing")


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


@pytest.fixture
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def max_gpu_available() -> bool:
    return len(list(get_accelerators())) > 1


@pytest.fixture(params=[(3,), (2, 3)])
def tensor_shapes(request):
    return request.param


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield


@pytest.fixture
def cuda_device(gpu_available: bool):
    if not gpu_available:
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture(params=["cpu", "gpu"])
def max_device(request, max_gpu_available: bool):
    if request.param == "cpu":
        yield (f"max_device:{len(get_accelerators()) - 1}")
    else:
        if not max_gpu_available:
            pytest.skip("You do not have a GPU supported by Max")
        yield ("max_device:0")


def pytest_sessionfinish(session, exitstatus):
    profile.print_stats()


def pytest_make_parametrize_id(config, val, argname):
    """Custom ID generation for parametrized tests"""
    if isinstance(val, torch.dtype):
        return str(val).split(".")[-1]
    # Return None to fall back to default behavior for other types
    return None
