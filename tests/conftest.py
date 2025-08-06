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
