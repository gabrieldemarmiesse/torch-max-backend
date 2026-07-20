"""Direct eager ATen contracts used by AdamW and gradient clipping."""

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool) -> str:
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    register_mojo_devices()
    return "mojo:0"


def _watch_eager_op(call_checker: CallChecker, op_name: str) -> None:
    """Require the exact PrivateUse1 registration, not a decomposition twin."""
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS[op_name])


def test_lerp_scalar_broadcast_uses_narrowed_fp32_branch(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar")

    # This Python double rounds to exactly 0.5f. ATen narrows the scalar before
    # selecting its stable formula, so this also exercises the >= 0.5 branch.
    weight = 0.5 - 2**-30
    start = torch.tensor(
        [[[-1.0687099695205688, -2.0, 3.0]], [[4.0, -5.0, 6.0]]], dtype=torch.float32
    )
    end = torch.tensor(
        [[[2.028475284576416, 8.0, -3.0]] for _ in range(4)], dtype=torch.float32
    ).transpose(0, 1)
    assert start.shape == (2, 1, 3)
    assert end.shape == (1, 4, 3)

    expected = torch.ops.aten.lerp.Scalar(start, end, weight)
    actual = torch.ops.aten.lerp.Scalar(start.to(mojo_gpu), end.to(mojo_gpu), weight)

    assert actual.shape == expected.shape == (2, 4, 3)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_lerp_scalar_out_supports_strided_self_alias(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar_out")

    weight = 0.5 - 2**-30
    base = torch.arange(24, dtype=torch.float32).reshape(3, 8) - 7.0
    end = torch.tensor([[2.0, -3.0, 5.0, -7.0]], dtype=torch.float32)
    expected_base = base.clone()
    expected_base[:, 1::2] = torch.lerp(base[:, 1::2], end, weight)

    device_base = base.to(mojo_gpu)
    device_self = device_base[:, 1::2]
    device_end = end.to(mojo_gpu)
    assert not device_self._is_contiguous
    assert not device_self.is_contiguous()
    holder, ptr = device_self._holder, device_self._ptr

    returned = torch.ops.aten.lerp.Scalar_out(
        device_self, device_end, weight, out=device_self
    )

    assert returned is device_self
    assert device_self._holder is holder
    assert device_self._ptr == ptr
    torch.testing.assert_close(device_base.cpu(), expected_base, rtol=0, atol=0)
    torch.testing.assert_close(device_end.cpu(), end, rtol=0, atol=0)


def test_lerp_scalar_rejects_fp32_weight_overflow(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar")

    start = torch.tensor([1.0, -2.0], dtype=torch.float32)
    end = torch.tensor([3.0, 4.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cannot be converted.*float.*overflow"):
        torch.ops.aten.lerp.Scalar(start, end, 1e40)
    with pytest.raises(RuntimeError, match="cannot be converted.*float.*overflow"):
        torch.ops.aten.lerp.Scalar(start.to(mojo_gpu), end.to(mojo_gpu), 1e40)


@pytest.mark.parametrize("operation", ["lerp", "vector_norm"])
def test_optimizer_out_ops_reject_invalid_integer_output(
    mojo_gpu: str, call_checker: CallChecker, operation: str
):
    if operation == "lerp":
        op_name = "aten::lerp.Scalar_out"
        lhs = torch.tensor([1.0, -2.0], device=mojo_gpu)
        rhs = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.lerp.Scalar_out(lhs, rhs, 0.25, out=out)

    else:
        op_name = "aten::linalg_vector_norm.out"
        input = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                input, 2, None, False, dtype=None, out=out
            )

    _watch_eager_op(call_checker, op_name)
    out = torch.empty((), dtype=torch.int64, device=mojo_gpu)
    if operation == "lerp":
        out = torch.empty(2, dtype=torch.int64, device=mojo_gpu)

    with pytest.raises(RuntimeError):
        invoke(out)


@pytest.mark.parametrize("operation", ["lerp", "vector_norm"])
def test_optimizer_out_ops_resize_public_shape_metadata(
    mojo_gpu: str, call_checker: CallChecker, operation: str
):
    if operation == "lerp":
        op_name = "aten::lerp.Scalar_out"
        lhs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        rhs = lhs + 6.0
        expected = torch.lerp(lhs, rhs, 0.25)
        device_lhs = lhs.to(mojo_gpu)
        device_rhs = rhs.to(mojo_gpu)

        def invoke(out):
            return torch.ops.aten.lerp.Scalar_out(device_lhs, device_rhs, 0.25, out=out)

    else:
        op_name = "aten::linalg_vector_norm.out"
        input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expected = torch.linalg.vector_norm(input, dim=1)
        device_input = input.to(mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                device_input, 2, [1], False, dtype=None, out=out
            )

    _watch_eager_op(call_checker, op_name)
    out = torch.empty(0, dtype=torch.float32, device=mojo_gpu)
    marker = object()
    out.user_marker = marker
    version = out._version
    returned = invoke(out)

    assert returned is out
    assert out.user_marker is marker
    assert out._version == version + 1
    assert tuple(out._shape) == tuple(expected.shape)
    torch.testing.assert_close(out.cpu(), expected)
    # Python methods and operators that read TensorImpl directly must agree
    # with the eager payload after the identity-preserving resize.
    assert tuple(out.shape) == tuple(expected.shape)
    assert out.numel() == expected.numel()
    assert torch.numel(out) == expected.numel()
    assert out.ndimension() == expected.ndimension()
    assert out.flatten().shape == expected.flatten().shape


@pytest.mark.parametrize("case", ["strided", "empty"])
def test_linalg_vector_norm_out_strided_and_empty(
    mojo_gpu: str, call_checker: CallChecker, case: str
):
    _watch_eager_op(call_checker, "aten::linalg_vector_norm.out")

    if case == "strided":
        base = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
        input = base.t()
        device_input = base.to(mojo_gpu).t()
        dim = [1]
        keepdim = True
        assert not device_input._is_contiguous
    else:
        input = torch.empty((2, 0), dtype=torch.float32)
        device_input = input.to(mojo_gpu)
        dim = [1]
        keepdim = False

    expected = torch.linalg.vector_norm(input, 2, dim, keepdim)
    out = torch.empty_like(expected, device=mojo_gpu)
    holder, ptr = out._holder, out._ptr
    returned = torch.ops.aten.linalg_vector_norm.out(
        device_input, 2, dim, keepdim, dtype=None, out=out
    )

    assert returned is out
    assert out._holder is holder
    assert out._ptr == ptr
    torch.testing.assert_close(out.cpu(), expected)


def test_mul_tensor_inplace_preserves_strided_alias(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::mul_.Tensor")

    base = torch.arange(12, dtype=torch.float32).to(mojo_gpu)
    view = base[::2]
    observer = base.view(3, 4)
    coefficient = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)
    expected = torch.arange(12, dtype=torch.float32)
    expected[::2] *= 0.25
    holder, ptr = view._holder, view._ptr
    version = view._version

    returned = torch.ops.aten.mul_.Tensor(view, coefficient)

    assert returned is view
    assert view._holder is holder
    assert view._ptr == ptr
    assert view._version == version + 1
    torch.testing.assert_close(observer.cpu().reshape(-1), expected)


@pytest.mark.parametrize("storage_offset", [0, 2])
def test_mul_out_resize_preserves_existing_storage_alias(
    mojo_gpu: str, call_checker: CallChecker, storage_offset: int
):
    _watch_eager_op(call_checker, "aten::mul.out")

    base = torch.arange(8, dtype=torch.float32).to(mojo_gpu)
    out = base[storage_offset:storage_offset]
    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    holder = base._holder

    returned = torch.ops.aten.mul.out(lhs, rhs, out=out)

    assert returned is out
    assert out._holder is holder is base._holder
    assert out.shape == (2,)
    assert torch.numel(out) == 2
    torch.testing.assert_close(out.cpu(), torch.tensor([10.0, 21.0]))
    expected_base = torch.arange(8, dtype=torch.float32)
    expected_base[storage_offset : storage_offset + 2] = torch.tensor([10.0, 21.0])
    torch.testing.assert_close(base.cpu(), expected_base)


@pytest.mark.parametrize(
    ("op_name", "valid_dtype", "invalid_dtype"),
    [
        ("aten::linalg_vector_norm.out", torch.float32, torch.float16),
        ("aten::any.out", torch.uint8, torch.int64),
        ("aten::isin.Tensor_Tensor_out", torch.bool, torch.int64),
    ],
)
def test_out_variants_enforce_operation_specific_dtype_contracts(
    mojo_gpu: str,
    call_checker: CallChecker,
    op_name: str,
    valid_dtype: torch.dtype,
    invalid_dtype: torch.dtype,
):
    _watch_eager_op(call_checker, op_name)

    if op_name == "aten::linalg_vector_norm.out":
        input = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                input, 2, None, False, dtype=None, out=out
            )

        expected = torch.tensor(5.0, dtype=valid_dtype)
    elif op_name == "aten::any.out":
        input = torch.tensor([[0, 2, 0], [0, 0, 0]], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.any.out(input, 1, False, out=out)

        expected = torch.tensor([1, 0], dtype=valid_dtype)
    else:
        input = torch.tensor([1, 2, 3], device=mojo_gpu)
        test_elements = torch.tensor([2, 4], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.isin.Tensor_Tensor_out(
                input, test_elements, assume_unique=False, invert=False, out=out
            )

        expected = torch.tensor([False, True, False], dtype=valid_dtype)

    valid_out = torch.empty(0, dtype=valid_dtype, device=mojo_gpu)
    assert invoke(valid_out) is valid_out
    assert valid_out.dtype == valid_dtype
    torch.testing.assert_close(valid_out.cpu(), expected)

    invalid_out = torch.empty(0, dtype=invalid_dtype, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        invoke(invalid_out)
