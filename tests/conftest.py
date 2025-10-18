import os

import pytest
import torch

from torch_max_backend import get_accelerators, register_max_devices
from torch_max_backend.profiler import profile
from torch_max_backend.testing import Conf

# Register your helper module for assertion rewriting
pytest.register_assert_rewrite("torch_max_backend.testing")


os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"
accelerators = list(get_accelerators())


@pytest.fixture(params=["cpu", "cuda"])
def device(request, gpu_available: bool):
    device_name = request.param
    if not gpu_available and device_name == "cuda":
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture(
    params=[
        ("max_device:cpu", True),
        ("max_device:cpu", False),
        ("max_device:gpu", True),
        ("max_device:gpu", False),
        ("cpu", True),
        ("cuda", True),
    ]
)
def conf(request, max_gpu_available: bool, cuda_available: bool):
    device_name, compile = request.param
    # to use max_device:gpu, we need to have a max supported gpu
    if device_name == "max_device:gpu" and not max_gpu_available:
        pytest.skip("You do not have a GPU supported by Max")
    if device_name == "cuda" and not cuda_available:
        pytest.skip("Pytorch CUDA not available")

    # known issues:
    if device_name.startswith("max_device") and compile:
        pytest.xfail("Known issue: max_device with compilation is not supported yet")

    if device_name.startswith("max_device"):
        device_name = device_name.replace("gpu", "0")
        device_name = device_name.replace("cpu", str(len(accelerators) - 1))
        # Make sure the device is initialized
        register_max_devices()

    if device_name in ("cuda", "cpu"):
        device_name += ":0"

    return Conf(device=device_name, compile=compile)


@pytest.fixture
def gpu_available() -> bool:
    return len(accelerators) > 1


@pytest.fixture
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def max_gpu_available() -> bool:
    return len(accelerators) > 1


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
    if isinstance(val, Conf):
        return str(val)
    # Return None to fall back to default behavior for other types
    return None
