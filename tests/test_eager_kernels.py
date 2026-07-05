"""Tests for the Mojo-extension fast path used by max_device eager mode."""

import pytest
import torch

from torch_max_backend import register_max_devices
from torch_max_backend.flags import fast_eager_enabled

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_max_devices()


BINARY_OPS = [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum]
UNARY_OPS = [torch.relu, torch.exp]


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_binary_ops_match_cpu(max_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    y = torch.randn(33, 65).to(dtype) + 1.5  # avoid div-by-~0
    result = op(x.to(max_device), y.to(max_device))
    torch.testing.assert_close(result.cpu(), op(x, y))


@pytest.mark.parametrize("op", UNARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_unary_ops_match_cpu(max_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    result = op(x.to(max_device))
    torch.testing.assert_close(result.cpu(), op(x))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fast_binary_int_dtypes(max_device, dtype):
    x = torch.arange(100, dtype=dtype)
    y = torch.arange(100, dtype=dtype) * 3
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fast_path_is_used(max_device):
    """The eligible case must go through the Mojo kernel, not the fallback."""
    if not fast_eager_enabled():
        pytest.skip("fast eager path disabled")
    from torch_max_backend import eager_kernels

    calls = []
    original = eager_kernels.binary_op

    def spy(mojo_fn, lhs, rhs):
        calls.append(mojo_fn)
        return original(mojo_fn, lhs, rhs)

    eager_kernels.binary_op = spy
    try:
        x = torch.randn(8, 8).to(max_device)
        y = torch.randn(8, 8).to(max_device)
        _ = x + y
    finally:
        eager_kernels.binary_op = original
    assert len(calls) == 1


def test_fallback_broadcast(max_device):
    x = torch.randn(16, 16)
    y = torch.randn(16)
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fallback_scalar_other(max_device):
    x = torch.randn(16, 16)
    result = (x.to(max_device) + 2.5).cpu()
    torch.testing.assert_close(result, x + 2.5)


def test_fallback_alpha(max_device):
    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    result = torch.add(x.to(max_device), y.to(max_device), alpha=2.0).cpu()
    torch.testing.assert_close(result, torch.add(x, y, alpha=2.0))


def test_fallback_int_div(max_device):
    x = torch.arange(1, 65, dtype=torch.int32)
    y = torch.full((64,), 4, dtype=torch.int32)
    result = (x.to(max_device) / y.to(max_device)).cpu()
    # check_dtype=False: the graph-based fallback path promotes int div to
    # float64 where torch gives float32 — a pre-existing deviation
    # (reproduces with TORCH_MAX_BACKEND_FAST_EAGER=0).
    torch.testing.assert_close(result, x / y, check_dtype=False)


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (0, 5)])
def test_edge_case_shapes(max_device, shape):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    result = (x.to(max_device) + y.to(max_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_chained_fast_ops(max_device):
    """Outputs of fast ops must be valid inputs to further fast ops."""
    x = torch.randn(32, 32)
    y = torch.randn(32, 32)
    device_result = x.to(max_device)
    for _ in range(5):
        device_result = torch.relu(device_result * y.to(max_device) + y.to(max_device))
    expected = x
    for _ in range(5):
        expected = torch.relu(expected * y + y)
    torch.testing.assert_close(device_result.cpu(), expected)
