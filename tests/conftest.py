import os

os.environ["MODULAR_TELEMETRY_ENABLED"] = "0"

import pytest

os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"

# must be called before importing torch_max_backend
pytest.register_assert_rewrite("torch_max_backend.testing")


import torch
from mojo.paths import _build_mojo_source_package

from torch_max_backend import get_accelerators, register_max_devices
from torch_max_backend.profiler import profile
from torch_max_backend.testing import Conf
from torch_max_backend.torch_compile_backend import compiler

# from torch_max_backend.max_device.log_aten_calls import log_aten_calls

# log_aten_calls()


# TODO: remove this when
# https://github.com/modular/modular/issues/5495 is fixed
compiler.paths_to_mojo_kernels[0] = _build_mojo_source_package(
    compiler.paths_to_mojo_kernels[0]
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request, gpu_available: bool):
    device_name = request.param
    if not gpu_available and device_name == "cuda":
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture(
    params=[
        Conf("max_device:cpu", True),
        Conf("max_device:cpu", False),
        Conf("max_device:gpu", True),
        Conf("max_device:gpu", False),
        Conf("cpu", True),
        Conf("cuda", True),
    ]
)
def conf(request, max_gpu_available: bool, cuda_available: bool):
    conf = request.param
    # to use max_device:gpu, we need to have a max supported gpu
    if conf.device == "max_device:gpu" and not max_gpu_available:
        pytest.skip("You do not have a GPU supported by Max")
    if conf.device == "cuda" and not cuda_available:
        pytest.skip("Pytorch CUDA not available")

    # known issues:
    if conf.device.startswith("max_device") and conf.compile:
        pytest.xfail("Known issue: max_device with compilation is not supported yet")

    if conf.device.startswith("max_device"):
        conf.device = conf.device.replace("gpu", "0")
        conf.device = conf.device.replace("cpu", str(len(list(get_accelerators())) - 1))
        # Make sure the device is initialized
        register_max_devices()

    if conf.device == "cuda":
        conf.device += ":0"

    return conf


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
    if isinstance(val, Conf):
        return str(val)
    # Return None to fall back to default behavior for other types
    return None
