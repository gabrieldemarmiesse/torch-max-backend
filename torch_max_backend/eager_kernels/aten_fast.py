"""ATen-signature-compatible fast implementations for max_device eager mode.

Each function here is registered in `max_device_aten_ops.py` (through
`_eager_impl`) *instead of* its counterpart in `aten_functions.py` when the
fast eager path is enabled. When the inputs qualify (realized, contiguous,
supported dtype/layout), the op runs as one or a few Mojo kernel calls — no
graph building, no MLIR passes, no interpreter.

Functions receive `TorchMaxTensor` arguments directly (no generic
argument-conversion walk — that costs several microseconds per op) and
return `TorchMaxTensor` results. When the inputs don't qualify they return
the `NOT_HANDLED` sentinel and the registration falls back to the regular
`wrap_for_max_device(aten_functions...)` path, so behavior is unchanged for
everything the fast path doesn't cover.

Only the eager (max_device) path uses this module; the torch.compile
backend keeps using `aten_functions` directly.
"""

import math
from typing import no_type_check

from max import driver
from max.dtype import DType
from max.experimental.tensor import Tensor as MaxEagerTensor

from torch_max_backend import eager_kernels, is_running_tests
from torch_max_backend.eager_kernels import _ctx_ptr
from torch_max_backend.max_device.torch_max_tensor import TorchMaxTensor

# Returned when the inputs don't qualify; the registration then falls back
# to the graph-based implementation with the original arguments.
NOT_HANDLED = object()

# The Mojo kernels raise (instead of falling back) on dtypes they don't
# support; gate float-only ops here so those inputs take the regular path.
_FLOAT_DTYPES = (DType.float16, DType.bfloat16, DType.float32)

_INT_SCALAR_DTYPES = (DType.int32, DType.int64)

# Dtypes the data-movement (pure copy) kernels support.
_COPYABLE_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.bool,
)

_CAST_DTYPES = (
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.uint8,
    DType.bool,
)


@no_type_check
def _buffer_or_none(x) -> driver.Buffer | None:
    """The realized contiguous driver buffer of x, or None.

    Accepts both TorchMaxTensor (what registrations receive) and
    MaxEagerTensor (internal callers).
    """
    if isinstance(x, TorchMaxTensor):
        buffer = x._buffer
        if buffer is not None:
            # Fast-path outputs store the realized contiguous buffer
            # directly; no MaxEagerTensor is ever constructed for them.
            return buffer
        max_data = x._max_data_
        if max_data is None:
            return None
    elif isinstance(x, MaxEagerTensor):
        max_data = x
    else:
        return None
    if not max_data.real:
        return None
    buffer = max_data.driver_tensor
    if not buffer.is_contiguous:
        return None
    return buffer


@no_type_check
def _on_gpu(buffer: driver.Buffer) -> bool:
    return buffer.device.label == "gpu"


@no_type_check
def _wrap(buffer: driver.Buffer) -> TorchMaxTensor:
    return TorchMaxTensor._from_buffer(buffer)


@no_type_check
def _new_buffer(dtype: DType, shape, device: driver.Device) -> driver.Buffer:
    return driver.Buffer(dtype, tuple(shape), device)


@no_type_check
def _row_major_strides(shape) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


@no_type_check
def _permute_buffer(buffer: driver.Buffer, perm: list[int]) -> driver.Buffer:
    """Materialize a permutation of a contiguous buffer (rank <= 4)."""
    in_shape = tuple(buffer.shape)
    in_strides = _row_major_strides(in_shape)
    out_shape = [in_shape[p] for p in perm]
    strides = [in_strides[p] for p in perm]
    pad = 4 - len(out_shape)
    out = _new_buffer(buffer.dtype, out_shape, buffer.device)
    if buffer.num_elements > 0:
        eager_kernels.data_movement_ops.PermuteCopy(
            out,
            buffer,
            tuple([1] * pad + out_shape),
            tuple([0] * pad + strides),
            _ctx_ptr(buffer.device),
        )
    return out


@no_type_check
def _try_binary(mojo_fn, lhs, rhs):
    """Elementwise binary kernel on two same-shape tensors, or None."""
    lhs_buffer = _buffer_or_none(lhs)
    rhs_buffer = _buffer_or_none(rhs)
    if (
        lhs_buffer is None
        or rhs_buffer is None
        or lhs_buffer.dtype != rhs_buffer.dtype
        or lhs_buffer.shape != rhs_buffer.shape
        or lhs_buffer.device != rhs_buffer.device
    ):
        return None
    out = _new_buffer(lhs_buffer.dtype, lhs_buffer.shape, lhs_buffer.device)
    if lhs_buffer.num_elements > 0:
        mojo_fn(out, lhs_buffer, rhs_buffer, _ctx_ptr(lhs_buffer.device))
    return _wrap(out)


@no_type_check
def _try_unary(mojo_fn, x, dtypes=_COPYABLE_DTYPES):
    x_buffer = _buffer_or_none(x)
    if x_buffer is None or x_buffer.dtype not in dtypes:
        return None
    out = _new_buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, _ctx_ptr(x_buffer.device))
    return _wrap(out)


@no_type_check
def _try_scalar(mojo_fn, x, scalar):
    if not isinstance(scalar, int | float) or isinstance(scalar, bool):
        return None
    x_buffer = _buffer_or_none(x)
    if x_buffer is None or x_buffer.dtype not in _FLOAT_DTYPES:
        return None
    out = _new_buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, float(scalar), _ctx_ptr(x_buffer.device))
    return _wrap(out)


@no_type_check
def _try_add_scalar_int(x, scalar):
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        return None
    x_buffer = _buffer_or_none(x)
    if x_buffer is None or x_buffer.dtype not in _INT_SCALAR_DTYPES:
        return None
    out = _new_buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        eager_kernels.elementwise_ops.AddScalarInt(
            out, x_buffer, scalar, _ctx_ptr(x_buffer.device)
        )
    return _wrap(out)


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_add(input, other, alpha=1):
    if alpha == 1:
        result = _try_binary(eager_kernels.elementwise_ops.Add, input, other)
        if result is None:
            result = _try_scalar(eager_kernels.elementwise_ops.AddScalar, input, other)
        if result is None:
            result = _try_add_scalar_int(input, other)
        if result is not None:
            return result
    return NOT_HANDLED


@no_type_check
def fast_aten_add_(input, other, alpha=1):
    """In-place add into input's buffer. Returns None when unavailable.

    Called with MaxEagerTensor arguments from the aten::add_ registration.
    """
    if alpha != 1:
        return None
    out = _buffer_or_none(input)
    other_buffer = _buffer_or_none(other)
    if (
        out is None
        or other_buffer is None
        or out.dtype != other_buffer.dtype
        or out.shape != other_buffer.shape
        or out.device != other_buffer.device
    ):
        return None
    if out.num_elements > 0:
        eager_kernels.elementwise_ops.Add(out, out, other_buffer, _ctx_ptr(out.device))
    return input


@no_type_check
def fast_aten_sub(input, other, alpha=1):
    if alpha == 1:
        result = _try_binary(eager_kernels.elementwise_ops.Sub, input, other)
        if result is not None:
            return result
    return NOT_HANDLED


@no_type_check
def fast_aten_mul(input, other):
    result = _try_binary(eager_kernels.elementwise_ops.Mul, input, other)
    if result is None:
        result = _try_scalar(eager_kernels.elementwise_ops.MulScalar, input, other)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_div(input, other, *, rounding_mode=None):
    if rounding_mode is None:
        input_buffer = _buffer_or_none(input)
        if input_buffer is not None and input_buffer.dtype in _FLOAT_DTYPES:
            result = _try_binary(eager_kernels.elementwise_ops.Div, input, other)
            if result is not None:
                return result
    return NOT_HANDLED


@no_type_check
def fast_aten_maximum(x, y):
    result = _try_binary(eager_kernels.elementwise_ops.Max, x, y)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_minimum(x, y):
    result = _try_binary(eager_kernels.elementwise_ops.Min, x, y)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_relu(tensor):
    result = _try_unary(eager_kernels.elementwise_ops.Relu, tensor)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_exp(input):
    result = _try_unary(eager_kernels.elementwise_ops.Exp, input, _FLOAT_DTYPES)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_tanh(x):
    result = _try_unary(eager_kernels.elementwise_ops.Tanh, x, _FLOAT_DTYPES)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_pow(x, y):
    result = _try_scalar(eager_kernels.elementwise_ops.PowScalar, x, y)
    if result is not None:
        return result
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# View / shape metadata ops (zero copy: alias the same driver buffer)
# ---------------------------------------------------------------------------


@no_type_check
def _resolve_sizes(shape, numel: int) -> list[int] | None:
    sizes = list(shape)
    negative = [i for i, s in enumerate(sizes) if s == -1]
    if len(negative) > 1 or any(s < -1 for s in sizes):
        return None
    if negative:
        rest = math.prod(s for s in sizes if s != -1)
        if rest == 0 or numel % rest != 0:
            return None
        sizes[negative[0]] = numel // rest
    if math.prod(sizes) != numel:
        return None
    return sizes


@no_type_check
def _fast_view(tensor, shape):
    if len(shape) == 1 and isinstance(shape[0], list | tuple):
        shape = shape[0]
    if not all(isinstance(s, int) for s in shape):
        return NOT_HANDLED
    buffer = _buffer_or_none(tensor)
    if buffer is None or buffer.num_elements == 0:
        return NOT_HANDLED
    sizes = _resolve_sizes(shape, buffer.num_elements)
    if sizes is None:
        return NOT_HANDLED
    return _wrap(buffer.view(buffer.dtype, sizes))


@no_type_check
def fast_aten_view(tensor, *shape):
    return _fast_view(tensor, shape)


@no_type_check
def fast_aten__unsafe_view(tensor, *shape):
    return _fast_view(tensor, shape)


@no_type_check
def fast_aten_unsqueeze(tensor, dim):
    buffer = _buffer_or_none(tensor)
    if buffer is not None and buffer.num_elements > 0:
        sizes = list(buffer.shape)
        if -len(sizes) - 1 <= dim <= len(sizes):
            if dim < 0:
                dim += len(sizes) + 1
            sizes.insert(dim, 1)
            return _wrap(buffer.view(buffer.dtype, sizes))
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Transpose / permute (materializing copies)
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_t(input):
    buffer = _buffer_or_none(input)
    if (
        buffer is not None
        and len(buffer.shape) == 2
        and buffer.dtype in _COPYABLE_DTYPES
    ):
        return _wrap(_permute_buffer(buffer, [1, 0]))
    return NOT_HANDLED


@no_type_check
def fast_aten_transpose(input, dim0, dim1):
    buffer = _buffer_or_none(input)
    if (
        buffer is not None
        and isinstance(dim0, int)
        and isinstance(dim1, int)
        and len(buffer.shape) <= 4
        and buffer.dtype in _COPYABLE_DTYPES
    ):
        rank = len(buffer.shape)
        if -rank <= dim0 < rank and -rank <= dim1 < rank:
            dim0 %= rank
            dim1 %= rank
            perm = list(range(rank))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            return _wrap(_permute_buffer(buffer, perm))
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Split (narrow copies along one dim)
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_split(input, split_size, dim=0):
    buffer = _buffer_or_none(input)
    if (
        buffer is not None
        and buffer.num_elements > 0
        and buffer.dtype in _COPYABLE_DTYPES
        and isinstance(dim, int)
    ):
        shape = list(buffer.shape)
        rank = len(shape)
        if -rank <= dim < rank:
            dim %= rank
            if isinstance(split_size, int):
                lengths = [
                    min(split_size, shape[dim] - start)
                    for start in range(0, shape[dim], split_size)
                ]
            else:
                lengths = list(split_size)
            if (
                all(isinstance(x, int) and x > 0 for x in lengths)
                and sum(lengths) == shape[dim]
            ):
                inner = math.prod(shape[dim + 1 :])
                outer = math.prod(shape[:dim])
                src_stride = shape[dim] * inner
                ctx = _ctx_ptr(buffer.device)
                results = []
                offset = 0
                for length in lengths:
                    out_shape = shape.copy()
                    out_shape[dim] = length
                    out = _new_buffer(buffer.dtype, out_shape, buffer.device)
                    eager_kernels.data_movement_ops.NarrowCopy(
                        out,
                        buffer,
                        outer,
                        src_stride,
                        length * inner,
                        offset * inner,
                        ctx,
                    )
                    results.append(_wrap(out))
                    offset += length
                return results
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Concatenation (only the trivial single-non-empty case is fast)
# ---------------------------------------------------------------------------


@no_type_check
def _is_legacy_empty(t) -> bool:
    buffer = _buffer_or_none(t)
    return buffer is not None and len(buffer.shape) == 1 and buffer.num_elements == 0


@no_type_check
def fast_aten_cat(tensors, dim=0):
    # PyTorch's cat skips legacy "empty" (1-D, size-0) tensors, e.g.
    # uninitialized KV-caches. When only one real tensor remains, cat is
    # just a copy of it.
    real = [t for t in tensors if not _is_legacy_empty(t)]
    if len(real) == 1:
        buffer = _buffer_or_none(real[0])
        if (
            buffer is not None
            and buffer.num_elements > 0
            and buffer.dtype in _COPYABLE_DTYPES
        ):
            out = _new_buffer(buffer.dtype, buffer.shape, buffer.device)
            eager_kernels.data_movement_ops.NarrowCopy(
                out,
                buffer,
                1,
                buffer.num_elements,
                buffer.num_elements,
                0,
                _ctx_ptr(buffer.device),
            )
            return _wrap(out)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@no_type_check
def _fast_batch_norm_inference(input, weight, bias, running_mean, running_var, eps):
    in_buf = _buffer_or_none(input)
    if in_buf is None or in_buf.dtype not in _FLOAT_DTYPES or len(in_buf.shape) < 2:
        return NOT_HANDLED
    if in_buf.num_elements == 0:
        return NOT_HANDLED
    params = [_buffer_or_none(t) for t in (running_mean, running_var, weight, bias)]
    if any(p is None or p.dtype != in_buf.dtype for p in params):
        return NOT_HANDLED
    mean_buf, var_buf, gamma_buf, beta_buf = params
    channels = in_buf.shape[1]
    inner = math.prod(in_buf.shape[2:])
    out = _new_buffer(in_buf.dtype, in_buf.shape, in_buf.device)
    eager_kernels.nn_ops.BatchNormInference(
        out,
        in_buf,
        mean_buf,
        var_buf,
        gamma_buf,
        beta_buf,
        (float(eps), channels, inner),
        _ctx_ptr(in_buf.device),
    )
    # Inference mode returns empty (0,) tensors for the saved stats.
    return (
        _wrap(out),
        _wrap(_new_buffer(in_buf.dtype, (0,), in_buf.device)),
        _wrap(_new_buffer(in_buf.dtype, (0,), in_buf.device)),
    )


@no_type_check
def fast_aten_native_batch_norm(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
    if not training:
        return _fast_batch_norm_inference(
            input, weight, bias, running_mean, running_var, eps
        )
    return NOT_HANDLED


@no_type_check
def fast_aten__native_batch_norm_legit_no_training(
    input, weight, bias, running_mean, running_var, momentum, eps
):
    return _fast_batch_norm_inference(
        input, weight, bias, running_mean, running_var, eps
    )


@no_type_check
def fast_aten_native_layer_norm(input, normalized_shape, weight, bias, eps):
    in_buf = _buffer_or_none(input)
    if (
        in_buf is not None
        and in_buf.num_elements > 0
        and in_buf.dtype in _FLOAT_DTYPES
        and len(normalized_shape) == 1
        and len(in_buf.shape) >= 1
        and in_buf.shape[-1] == normalized_shape[0]
    ):
        gamma_buf = _buffer_or_none(weight)
        beta_buf = _buffer_or_none(bias)
        if (
            gamma_buf is not None
            and beta_buf is not None
            and gamma_buf.dtype == in_buf.dtype
            and beta_buf.dtype == in_buf.dtype
        ):
            cols = in_buf.shape[-1]
            rows = in_buf.num_elements // cols
            out = _new_buffer(in_buf.dtype, in_buf.shape, in_buf.device)
            stat_shape = tuple(in_buf.shape[:-1]) + (1,)
            mean = _new_buffer(DType.float32, stat_shape, in_buf.device)
            rstd = _new_buffer(DType.float32, stat_shape, in_buf.device)
            eager_kernels.nn_ops.LayerNorm(
                out,
                mean,
                rstd,
                in_buf,
                gamma_buf,
                beta_buf,
                (float(eps), rows, cols),
                _ctx_ptr(in_buf.device),
            )
            return _wrap(out), _wrap(mean), _wrap(rstd)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_mean(input, dim=None, keepdim=False, *, dtype=None):
    in_buf = _buffer_or_none(input) if dtype is None else None
    if (
        in_buf is not None
        and in_buf.num_elements > 0
        and in_buf.dtype in _FLOAT_DTYPES
        and isinstance(dim, list | tuple)
        and len(dim) > 0
        and all(isinstance(d, int) for d in dim)
    ):
        rank = len(in_buf.shape)
        dims = sorted(d % rank for d in dim)
        # Fast path: reducing exactly the trailing dims (contiguous rows).
        if dims == list(range(rank - len(dims), rank)):
            cols = math.prod(in_buf.shape[d] for d in dims)
            rows = in_buf.num_elements // cols
            lead_shape = list(in_buf.shape[: rank - len(dims)])
            out_shape = lead_shape + [1] * len(dims) if keepdim else lead_shape
            out = _new_buffer(in_buf.dtype, out_shape, in_buf.device)
            eager_kernels.nn_ops.MeanRows(
                out, in_buf, rows, cols, _ctx_ptr(in_buf.device)
            )
            return _wrap(out)
    return NOT_HANDLED


@no_type_check
def fast_aten_all(input, dim=None, keepdim=False):
    in_buf = _buffer_or_none(input)
    if (
        in_buf is not None
        and dim is None
        and not keepdim
        and in_buf.dtype == DType.bool
        and 0 < in_buf.num_elements < 65536
    ):
        out = _new_buffer(DType.bool, (), in_buf.device)
        eager_kernels.nn_ops.AllBool(
            out, in_buf, in_buf.num_elements, _ctx_ptr(in_buf.device)
        )
        return _wrap(out)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


@no_type_check
def _pair(x) -> tuple[int, int] | None:
    if isinstance(x, int):
        return (x, x)
    if (
        isinstance(x, list | tuple)
        and len(x) == 2
        and all(isinstance(i, int) for i in x)
    ):
        return (x[0], x[1])
    if isinstance(x, list | tuple) and len(x) == 1 and isinstance(x[0], int):
        return (x[0], x[0])
    return None


@no_type_check
def fast_aten_max_pool2d_with_indices(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
    in_buf = _buffer_or_none(input)
    kernel = _pair(kernel_size)
    strides = _pair(stride) if stride not in (None, []) else kernel
    pads = _pair(padding)
    dils = _pair(dilation)
    if (
        in_buf is not None
        and in_buf.num_elements > 0
        and in_buf.dtype in _FLOAT_DTYPES
        and len(in_buf.shape) == 4
        and not ceil_mode
        and None not in (kernel, strides, pads, dils)
    ):
        n, c, in_h, in_w = in_buf.shape
        kh, kw = kernel
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if out_h > 0 and out_w > 0:
            out = _new_buffer(in_buf.dtype, (n, c, out_h, out_w), in_buf.device)
            indices = _new_buffer(DType.int64, (n, c, out_h, out_w), in_buf.device)
            eager_kernels.nn_ops.MaxPool2dWithIndices(
                out,
                indices,
                in_buf,
                (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, n * c),
                _ctx_ptr(in_buf.device),
            )
            return _wrap(out), _wrap(indices)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Matmul family (GPU only: cuBLAS via the MAX linalg vendor bindings)
# ---------------------------------------------------------------------------


@no_type_check
def _fast_matmul(a_buf, b_buf) -> driver.Buffer | None:
    """C = A @ B for 2D contiguous same-dtype float buffers on GPU."""
    if a_buf is None or b_buf is None:
        return None
    if not (_on_gpu(a_buf) and a_buf.device == b_buf.device):
        return None
    if a_buf.dtype != b_buf.dtype or a_buf.dtype not in _FLOAT_DTYPES:
        return None
    if len(a_buf.shape) != 2 or len(b_buf.shape) != 2:
        return None
    m, k = a_buf.shape
    k2, n = b_buf.shape
    if k != k2 or 0 in (m, n, k):
        return None
    out = _new_buffer(a_buf.dtype, (m, n), a_buf.device)
    eager_kernels.matmul_ops.Matmul(
        out, a_buf, b_buf, (m, n, k, 0), _ctx_ptr(a_buf.device)
    )
    return out


@no_type_check
def fast_aten_mm(x, y):
    out = _fast_matmul(_buffer_or_none(x), _buffer_or_none(y))
    if out is not None:
        return _wrap(out)
    return NOT_HANDLED


@no_type_check
def fast_aten_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    if beta == 1 and alpha == 1:
        bias_buf = _buffer_or_none(input)
        a_buf = _buffer_or_none(mat1)
        b_buf = _buffer_or_none(mat2)
        if (
            bias_buf is not None
            and a_buf is not None
            and b_buf is not None
            and len(bias_buf.shape) == 1
            and len(b_buf.shape) == 2
            and bias_buf.shape[0] == b_buf.shape[1]
            and bias_buf.dtype == a_buf.dtype
        ):
            out = _fast_matmul(a_buf, b_buf)
            if out is not None:
                eager_kernels.matmul_ops.BiasAddRow(
                    out, bias_buf, out.shape[1], _ctx_ptr(out.device)
                )
                return _wrap(out)
    return NOT_HANDLED


@no_type_check
def fast_aten_bmm(input, mat2):
    a_buf = _buffer_or_none(input)
    b_buf = _buffer_or_none(mat2)
    if (
        a_buf is not None
        and b_buf is not None
        and _on_gpu(a_buf)
        and a_buf.device == b_buf.device
        and a_buf.dtype == b_buf.dtype
        and a_buf.dtype in _FLOAT_DTYPES
        and len(a_buf.shape) == 3
        and len(b_buf.shape) == 3
        and a_buf.shape[0] == b_buf.shape[0]
        and a_buf.shape[2] == b_buf.shape[1]
        and 0 not in a_buf.shape
        and 0 not in b_buf.shape
    ):
        batch, m, k = a_buf.shape
        n = b_buf.shape[2]
        out = _new_buffer(a_buf.dtype, (batch, m, n), a_buf.device)
        eager_kernels.matmul_ops.Bmm(
            out, a_buf, b_buf, (batch, m, n, k, 0), _ctx_ptr(a_buf.device)
        )
        return _wrap(out)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Convolution (GPU only). Batch-1/groups-1 runs as im2col + cuBLAS matmul
# with the torch (K,C,R,S) weight used as-is and NCHW output — no layout
# permutes; everything else calls the MAX conv kernel (cuDNN path for
# torch's filter layout) with NHWC permutes around it.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    in_buf = _buffer_or_none(input)
    w_buf = _buffer_or_none(weight)
    bias_buf = _buffer_or_none(bias) if bias is not None else None
    strides = _pair(list(stride))
    pads = _pair(list(padding))
    dils = _pair(list(dilation))
    if (
        in_buf is not None
        and w_buf is not None
        and not transposed
        and (bias is None or bias_buf is not None)
        and _on_gpu(in_buf)
        and in_buf.dtype == w_buf.dtype
        and in_buf.dtype in _FLOAT_DTYPES
        and len(in_buf.shape) == 4
        and len(w_buf.shape) == 4
        and isinstance(groups, int)
        and groups >= 1
        and None not in (strides, pads, dils)
        and 0 not in in_buf.shape
    ):
        n, c, in_h, in_w = in_buf.shape
        out_c, c_per_group, kh, kw = w_buf.shape
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if c_per_group * groups == c and out_h > 0 and out_w > 0:
            if bias_buf is not None and (
                bias_buf.dtype != in_buf.dtype or tuple(bias_buf.shape) != (out_c,)
            ):
                return NOT_HANDLED
            ctx = _ctx_ptr(in_buf.device)
            if n == 1 and groups == 1:
                cols = out_h * out_w
                crs = c * kh * kw
                if (kh, kw, sh, sw, ph, pw, dh, dw) == (1, 1, 1, 1, 0, 0, 1, 1):
                    col = in_buf.view(in_buf.dtype, (crs, cols))
                else:
                    col = _new_buffer(in_buf.dtype, (crs, cols), in_buf.device)
                    eager_kernels.conv_ops.Im2col(
                        col,
                        in_buf,
                        (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, c),
                        ctx,
                    )
                out = _new_buffer(in_buf.dtype, (out_c, cols), in_buf.device)
                eager_kernels.matmul_ops.Matmul(
                    out,
                    w_buf.view(w_buf.dtype, (out_c, crs)),
                    col,
                    (out_c, cols, crs, 0),
                    ctx,
                )
                if bias_buf is not None:
                    eager_kernels.conv_ops.BiasAddChan(out, bias_buf, cols, ctx)
                return _wrap(out.view(out.dtype, (1, out_c, out_h, out_w)))
            in_nhwc = _permute_buffer(in_buf, [0, 2, 3, 1])
            out_nhwc = _new_buffer(
                in_buf.dtype, (n, out_h, out_w, out_c), in_buf.device
            )
            eager_kernels.conv_ops.Conv2d(
                out_nhwc,
                in_nhwc,
                w_buf,
                (
                    n,
                    in_h,
                    in_w,
                    c,
                    out_c,
                    kh,
                    kw,
                    out_h,
                    out_w,
                    sh,
                    sw,
                    dh,
                    dw,
                    ph,
                    pw,
                    groups,
                ),
                ctx,
            )
            if bias_buf is not None:
                eager_kernels.matmul_ops.BiasAddRow(out_nhwc, bias_buf, out_c, ctx)
            return _wrap(_permute_buffer(out_nhwc, [0, 3, 1, 2]))
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Attention: decomposed scaled dot product attention (bmm + fused
# scale/causal softmax + bmm).
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    q_buf = _buffer_or_none(query)
    k_buf = _buffer_or_none(key)
    v_buf = _buffer_or_none(value)
    if (
        q_buf is not None
        and k_buf is not None
        and v_buf is not None
        and attn_mask is None
        and dropout_p == 0.0
        and _on_gpu(q_buf)
        and q_buf.dtype == k_buf.dtype == v_buf.dtype
        and q_buf.dtype in _FLOAT_DTYPES
        and len(q_buf.shape) == 4
        and tuple(k_buf.shape) == tuple(v_buf.shape)
        and tuple(q_buf.shape[:2]) == tuple(k_buf.shape[:2])
        and q_buf.shape[3] == k_buf.shape[3]
        and 0 not in q_buf.shape
        and 0 not in k_buf.shape
    ):
        b, h, q_len, head_dim = q_buf.shape
        kv_len = k_buf.shape[2]
        scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        ctx = _ctx_ptr(q_buf.device)
        q3 = q_buf.view(q_buf.dtype, (b * h, q_len, head_dim))
        k3 = k_buf.view(k_buf.dtype, (b * h, kv_len, head_dim))
        v3 = v_buf.view(v_buf.dtype, (b * h, kv_len, head_dim))
        scores = _new_buffer(q_buf.dtype, (b * h, q_len, kv_len), q_buf.device)
        # scores = q @ k^T (transpose_b=1)
        eager_kernels.matmul_ops.Bmm(
            scores, q3, k3, (b * h, q_len, kv_len, head_dim, 1), ctx
        )
        probs = _new_buffer(q_buf.dtype, (b * h, q_len, kv_len), q_buf.device)
        eager_kernels.nn_ops.SoftmaxRows(
            probs,
            scores,
            b * h * q_len,
            kv_len,
            float(scale_val),
            1 if is_causal else 0,
            q_len,
            ctx,
        )
        out = _new_buffer(q_buf.dtype, (b * h, q_len, head_dim), q_buf.device)
        eager_kernels.matmul_ops.Bmm(
            out, probs, v3, (b * h, q_len, head_dim, kv_len, 0), ctx
        )
        return _wrap(out.view(out.dtype, (b, h, q_len, head_dim)))
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_embedding(
    input, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    # `input` is the weight table, `weight` the indices (aten naming).
    table_buf = _buffer_or_none(input)
    idx_buf = _buffer_or_none(weight)
    if (
        table_buf is not None
        and idx_buf is not None
        and table_buf.dtype in _FLOAT_DTYPES
        and idx_buf.dtype in (DType.int32, DType.int64)
        and len(table_buf.shape) == 2
        and idx_buf.num_elements > 0
    ):
        row_len = table_buf.shape[1]
        out_shape = tuple(idx_buf.shape) + (row_len,)
        out = _new_buffer(table_buf.dtype, out_shape, table_buf.device)
        eager_kernels.nn_ops.Gather0(
            out,
            table_buf,
            idx_buf,
            idx_buf.num_elements,
            row_len,
            _ctx_ptr(table_buf.device),
        )
        return _wrap(out)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Scalar readback
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten__local_scalar_dense(tensor):
    buffer = _buffer_or_none(tensor)
    # bfloat16 is not representable in numpy; let it fall back.
    if (
        buffer is not None
        and buffer.num_elements == 1
        and buffer.dtype != DType.bfloat16
    ):
        return buffer.to_numpy().reshape(()).item()
    return NOT_HANDLED


def _instrument_call_counts():
    """Give every fast op a test-only call counter, mirroring what
    `aten_functions.map_to` does, so `CallChecker` can assert that an op
    was handled by either implementation."""
    import functools

    for name, func in list(globals().items()):
        if not name.startswith("fast_aten"):
            continue

        def make_wrapper(wrapped):
            @functools.wraps(wrapped)
            @no_type_check
            def wrapper(*args, **kwargs):
                wrapper.call_count += 1
                return wrapped(*args, **kwargs)

            wrapper.call_count = 0
            return wrapper

        globals()[name] = make_wrapper(func)


if is_running_tests.IS_RUNNING_TESTS:
    _instrument_call_counts()
