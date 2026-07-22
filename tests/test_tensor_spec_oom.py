"""Host-only regression tests for TensorSpec allocator-error propagation."""

from types import SimpleNamespace

import pytest
import torch

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.eager_kernels import aten_fast

_CUDA_OOM = "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)"


def _tensor(device, *, dtype=aten_fast.DType.float32, shape=(2, 3), strides=(3, 1)):
    return SimpleNamespace(
        _device=device,
        _dtype=dtype,
        _shape=shape,
        _strides=strides,
        _is_contiguous=True,
    )


@pytest.mark.parametrize(
    "path",
    [
        "binary",
        "binary_cast",
        "binary_fill_rhs",
        "binary_fill_lhs",
        "unary",
        "reduce",
        "matmul",
        "scalar",
        "int_scalar",
        "logical",
        "batch_norm",
        "min_dim",
        "attention_decode",
    ],
)
def test_tensor_spec_fallbacks_propagate_device_oom(monkeypatch, path):
    """Fallback-only errors stay recoverable, but allocator OOM never does."""
    device = SimpleNamespace(label="gpu")
    lhs = _tensor(device)
    rhs = _tensor(device)
    bool_lhs = _tensor(device, dtype=aten_fast.DType.bool)
    query = _tensor(device, shape=(1, 1, 1, 4), strides=(4, 4, 4, 1))
    key = _tensor(device, shape=(1, 1, 2, 4), strides=(8, 8, 4, 1))
    value = _tensor(device, shape=(1, 1, 2, 4), strides=(8, 8, 4, 1))
    tensors = (lhs, rhs, bool_lhs, query, key, value)

    def as_tensor(candidate):
        return candidate if any(candidate is tensor for tensor in tensors) else None

    def raise_allocator_oom(*_args):
        raise NotImplementedError(_CUDA_OOM)

    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1)
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "logic_ops",
        SimpleNamespace(
            SubSpec=raise_allocator_oom, LogicalAndSpec=raise_allocator_oom
        ),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "data_movement_ops",
        SimpleNamespace(CastSpec=raise_allocator_oom),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "elementwise_ops",
        SimpleNamespace(
            FillSpec=raise_allocator_oom,
            NegSpec=raise_allocator_oom,
            AddScalarSpec=raise_allocator_oom,
            AddScalarIntSpec=raise_allocator_oom,
        ),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "reduction_ops",
        SimpleNamespace(SumSpec=raise_allocator_oom, MinDimSpec=raise_allocator_oom),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "matmul_ops",
        SimpleNamespace(MatmulSpec=raise_allocator_oom),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "nn_ops",
        SimpleNamespace(
            BatchNormSpec=raise_allocator_oom, AttnDecodeSpec=raise_allocator_oom
        ),
    )

    calls = {
        "binary": lambda: aten_fast._try_spec_binary("SubSpec", lhs, rhs),
        "binary_cast": lambda: aten_fast._try_spec_binary("SubSpec", bool_lhs, rhs),
        "binary_fill_rhs": lambda: aten_fast._try_spec_binary("SubSpec", lhs, 1.0),
        "binary_fill_lhs": lambda: aten_fast._try_spec_binary("SubSpec", 1.0, rhs),
        "unary": lambda: aten_fast._try_spec_unary("NegSpec", lhs),
        "reduce": lambda: aten_fast._try_spec_reduce("SumSpec", lhs, (1,), False),
        "matmul": lambda: aten_fast._try_spec_matmul("MatmulSpec", (lhs, rhs), 0),
        "scalar": lambda: aten_fast._try_spec_scalar("AddScalarSpec", lhs, 1.0),
        "int_scalar": lambda: aten_fast._try_spec_int_scalar(
            "AddScalarIntSpec", lhs, 1
        ),
        "logical": lambda: aten_fast._try_logical("LogicalAndSpec", bool_lhs, bool_lhs),
        "batch_norm": lambda: aten_fast._fast_batch_norm_inference(
            lhs, lhs, lhs, lhs, lhs, 1e-5
        ),
        "min_dim": lambda: aten_fast.fast_aten_min_dim(lhs, 1),
        "attention_decode": lambda: aten_fast.fast_aten_scaled_dot_product_attention(
            query, key, value
        ),
    }

    with pytest.raises(torch.OutOfMemoryError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        calls[path]()


def test_tensor_spec_unsupported_metadata_still_uses_fallback(monkeypatch):
    device = object()
    tensor = _tensor(device)

    def raise_unsupported(*_args):
        raise NotImplementedError("mojo spec neg: strided input is unsupported")

    monkeypatch.setattr(aten_fast, "_t", lambda _candidate: tensor)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda _tensor: object())
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "elementwise_ops",
        SimpleNamespace(NegSpec=raise_unsupported),
    )

    assert aten_fast._try_spec_unary("NegSpec", tensor) is None
