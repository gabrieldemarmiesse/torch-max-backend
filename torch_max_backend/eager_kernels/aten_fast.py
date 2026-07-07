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

# Dtypes the binary elementwise kernels support (no bool: the arithmetic
# kernels don't lower for it, so bool tensors take the regular path).
_BINARY_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)

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
        or lhs_buffer.dtype not in _BINARY_DTYPES
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
def _try_bool_and(lhs, rhs):
    """bool * bool (= logical AND) via the uint8 Mul kernel on byte views."""
    lhs_buffer = _buffer_or_none(lhs)
    rhs_buffer = _buffer_or_none(rhs)
    if (
        lhs_buffer is None
        or rhs_buffer is None
        or lhs_buffer.dtype != DType.bool
        or rhs_buffer.dtype != DType.bool
        or lhs_buffer.shape != rhs_buffer.shape
        or lhs_buffer.device != rhs_buffer.device
    ):
        return None
    out = _new_buffer(DType.bool, lhs_buffer.shape, lhs_buffer.device)
    if lhs_buffer.num_elements > 0:
        eager_kernels.elementwise_ops.Mul(
            out.view(DType.uint8, out.shape),
            lhs_buffer.view(DType.uint8, lhs_buffer.shape),
            rhs_buffer.view(DType.uint8, rhs_buffer.shape),
            _ctx_ptr(lhs_buffer.device),
        )
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
def _try_int_scalar(mojo_fn, x, scalar):
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        return None
    x_buffer = _buffer_or_none(x)
    if x_buffer is None or x_buffer.dtype not in _INT_SCALAR_DTYPES:
        return None
    out = _new_buffer(x_buffer.dtype, x_buffer.shape, x_buffer.device)
    if x_buffer.num_elements > 0:
        mojo_fn(out, x_buffer, scalar, _ctx_ptr(x_buffer.device))
    return _wrap(out)


# ---------------------------------------------------------------------------
# Broadcast helpers for the strided kernels (logic_ops / WhereSelect).
# Operands are described by the contiguous output's dims padded to rank 4
# plus per-operand element strides, 0 on broadcast dims.
# ---------------------------------------------------------------------------

# int64 scalars round-trip through the Fill kernel's Float64 argument.
_MAX_EXACT_INT = 2**53

_BITWISE_DTYPES = (
    DType.bool,
    DType.uint8,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
)

# Dtypes the broadcast-strided logic_ops kernels dispatch on (no float64,
# no bool: bool goes through uint8 views).
_BCAST_DTYPES = _FLOAT_DTYPES + (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)


@no_type_check
def _bcast_meta(*shapes):
    """Broadcast shapes (rank <= 4) for the strided kernels.

    Returns (out_shape, dims4, [strides4 per input]) or None when the
    shapes don't broadcast or exceed rank 4.
    """
    rank = max((len(s) for s in shapes), default=0)
    if rank > 4:
        return None
    padded = [[1] * (rank - len(s)) + list(s) for s in shapes]
    out = []
    for i in range(rank):
        size = 1
        for p in padded:
            if p[i] != 1:
                if size != 1 and p[i] != size:
                    return None
                size = p[i]
        out.append(size)
    dims = [1] * (4 - rank) + out
    all_strides = []
    for p in padded:
        st = _row_major_strides(p)
        st = [0 if p[i] == 1 else st[i] for i in range(rank)]
        all_strides.append([0] * (4 - rank) + st)
    return out, dims, all_strides


@no_type_check
def _scalar_buffer(value, dtype, device) -> driver.Buffer:
    """A 0-d buffer holding `value`, for stride-0 broadcast operands."""
    out = _new_buffer(dtype, (), device)
    eager_kernels.elementwise_ops.Fill(out, float(value), _ctx_ptr(device))
    return out


@no_type_check
def _cast_buffer(buf: driver.Buffer, dtype: DType) -> driver.Buffer:
    out = _new_buffer(dtype, buf.shape, buf.device)
    if buf.num_elements > 0:
        eager_kernels.data_movement_ops.Cast(out, buf, _ctx_ptr(buf.device))
    return out


@no_type_check
def _promoted_pair(a: driver.Buffer, b: driver.Buffer):
    """Same-dtype buffer pair following torch's promotion, or None.

    Only the promotions the generation loops hit: bool combined with any
    castable dtype, and int32 with int64.
    """
    if a.dtype == b.dtype:
        return a, b
    if a.dtype == DType.bool and b.dtype in _CAST_DTYPES:
        return _cast_buffer(a, b.dtype), b
    if b.dtype == DType.bool and a.dtype in _CAST_DTYPES:
        return a, _cast_buffer(b, a.dtype)
    if a.dtype == DType.int32 and b.dtype == DType.int64:
        return _cast_buffer(a, DType.int64), b
    if a.dtype == DType.int64 and b.dtype == DType.int32:
        return a, _cast_buffer(b, DType.int64)
    return None


@no_type_check
def _resolve_scalar(value, dtype: DType, device) -> driver.Buffer | None:
    """A 0-d stride-0 buffer holding a Python scalar in `dtype`, or None
    when the value doesn't embed losslessly."""
    if not isinstance(value, int | float):
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_INT:
            return None
        if dtype == DType.bool and value not in (0, 1):
            return None
    elif dtype not in _FLOAT_DTYPES + (DType.float64,):
        # float scalar against an int tensor promotes in torch; keep it
        # on the regular path.
        return None
    return _scalar_buffer(value, dtype, device)


@no_type_check
def _binary_operands(input, other):
    """Resolve (lhs_buffer, rhs_buffer) with equal dtypes for a broadcast
    binary/comparison kernel. Either operand may be a Python scalar (which
    becomes a 0-d stride-0 buffer of the tensor operand's dtype), or None
    if unresolvable.
    """
    in_buf = _buffer_or_none(input)
    other_buf = _buffer_or_none(other)
    if in_buf is not None and other_buf is not None:
        if other_buf.device != in_buf.device:
            return None
        return _promoted_pair(in_buf, other_buf)
    if in_buf is not None:
        scalar_buf = _resolve_scalar(other, in_buf.dtype, in_buf.device)
        return None if scalar_buf is None else (in_buf, scalar_buf)
    if other_buf is not None:
        # Scalar-first calls, e.g. rsub-style `1 - tensor`.
        scalar_buf = _resolve_scalar(input, other_buf.dtype, other_buf.device)
        return None if scalar_buf is None else (scalar_buf, other_buf)
    return None


@no_type_check
def _launch_bcast(kernel, out, operands, strides_dims):
    out_shape, dims, strides = strides_dims
    params = tuple(dims) + tuple(s for st in strides for s in st)
    kernel(out, *operands, params, _ctx_ptr(out.device))


@no_type_check
def _try_binary_bcast(kernel_name, lhs, rhs):
    """Broadcast-strided arithmetic on two operands (tensor or scalar)."""
    pair = _binary_operands(lhs, rhs)
    if pair is None:
        return None
    lhs_buf, rhs_buf = pair
    if lhs_buf.dtype not in _BCAST_DTYPES:
        return None
    if kernel_name == "DivBcast" and lhs_buf.dtype not in _FLOAT_DTYPES:
        return None
    meta = _bcast_meta(lhs_buf.shape, rhs_buf.shape)
    if meta is None:
        return None
    out = _new_buffer(lhs_buf.dtype, meta[0], lhs_buf.device)
    if out.num_elements > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (lhs_buf, rhs_buf), meta
        )
    return _wrap(out)


@no_type_check
def _try_cmp(kernel_name, input, other):
    """Broadcast-strided comparison -> bool tensor, or None."""
    pair = _binary_operands(input, other)
    if pair is None:
        return None
    lhs_buf, rhs_buf = pair
    if lhs_buf.dtype == DType.bool:
        lhs_buf = lhs_buf.view(DType.uint8, lhs_buf.shape)
        rhs_buf = rhs_buf.view(DType.uint8, rhs_buf.shape)
    if lhs_buf.dtype not in _BCAST_DTYPES:
        return None
    meta = _bcast_meta(lhs_buf.shape, rhs_buf.shape)
    if meta is None:
        return None
    out = _new_buffer(DType.bool, meta[0], lhs_buf.device)
    if out.num_elements > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (lhs_buf, rhs_buf), meta
        )
    return _wrap(out)


@no_type_check
def _try_bitwise(kernel_name, input, other):
    """Broadcast-strided bitwise op (bool via uint8 views), or None."""
    pair = _binary_operands(input, other)
    if pair is None:
        return None
    lhs_buf, rhs_buf = pair
    if lhs_buf.dtype not in _BITWISE_DTYPES:
        return None
    meta = _bcast_meta(lhs_buf.shape, rhs_buf.shape)
    if meta is None:
        return None
    out = _new_buffer(lhs_buf.dtype, meta[0], lhs_buf.device)
    kernel_out = out
    if lhs_buf.dtype == DType.bool:
        lhs_buf = lhs_buf.view(DType.uint8, lhs_buf.shape)
        rhs_buf = rhs_buf.view(DType.uint8, rhs_buf.shape)
        kernel_out = out.view(DType.uint8, out.shape)
    if out.num_elements > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name),
            kernel_out,
            (lhs_buf, rhs_buf),
            meta,
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
            result = _try_int_scalar(
                eager_kernels.elementwise_ops.AddScalarInt, input, other
            )
        if result is None:
            result = _try_binary_bcast("AddBcast", input, other)
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
        or out.dtype not in _BINARY_DTYPES
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
        if result is None and isinstance(other, int | float):
            result = _try_scalar(eager_kernels.elementwise_ops.AddScalar, input, -other)
            if result is None and isinstance(other, int):
                result = _try_int_scalar(
                    eager_kernels.elementwise_ops.AddScalarInt, input, -other
                )
        if result is None:
            result = _try_binary_bcast("SubBcast", input, other)
        if result is not None:
            return result
    return NOT_HANDLED


@no_type_check
def fast_aten_mul(input, other):
    result = _try_binary(eager_kernels.elementwise_ops.Mul, input, other)
    if result is None:
        result = _try_bool_and(input, other)
    if result is None:
        result = _try_scalar(eager_kernels.elementwise_ops.MulScalar, input, other)
    if result is None:
        result = _try_int_scalar(
            eager_kernels.elementwise_ops.MulScalarInt, input, other
        )
    if result is None:
        result = _try_binary_bcast("MulBcast", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_div(input, other, *, rounding_mode=None):
    if rounding_mode is None:
        input_buffer = _buffer_or_none(input)
        if input_buffer is not None and input_buffer.dtype in _FLOAT_DTYPES:
            result = _try_binary(eager_kernels.elementwise_ops.Div, input, other)
            if result is None:
                result = _try_binary_bcast("DivBcast", input, other)
            if result is not None:
                return result
    return NOT_HANDLED


@no_type_check
def fast_aten_fill_scalar(input, value):
    """Functional fill: new buffer, same shape/dtype, all elements = value."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_HANDLED
    input_buffer = _buffer_or_none(input)
    if input_buffer is None:
        return NOT_HANDLED
    out = _new_buffer(input_buffer.dtype, input_buffer.shape, input_buffer.device)
    if out.num_elements > 0:
        eager_kernels.elementwise_ops.Fill(out, float(value), _ctx_ptr(out.device))
    return _wrap(out)


@no_type_check
def fast_aten_fill__scalar(input, value):
    """In-place fill of input's buffer. Returns None when unavailable."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    out = _buffer_or_none(input)
    if out is None:
        return None
    if out.num_elements > 0:
        eager_kernels.elementwise_ops.Fill(out, float(value), _ctx_ptr(out.device))
    return input


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
# Comparisons and bitwise/logic ops (broadcast-strided kernels). These are
# the generation-loop bookkeeping ops: stopping criteria, attention-mask
# prep, position ids.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_eq(input, other):
    result = _try_cmp("EqBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_ne(input, other):
    result = _try_cmp("NeBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_lt(input, other):
    result = _try_cmp("LtBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_le(input, other):
    result = _try_cmp("LeBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_gt(input, other):
    result = _try_cmp("GtBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_ge(input, other):
    result = _try_cmp("GeBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_and(input, other):
    result = _try_bitwise("AndBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_or(input, other):
    result = _try_bitwise("OrBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_xor(input, other):
    result = _try_bitwise("XorBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_not(input):
    in_buf = _buffer_or_none(input)
    if in_buf is None or in_buf.dtype not in _BITWISE_DTYPES:
        return NOT_HANDLED
    out = _new_buffer(in_buf.dtype, in_buf.shape, in_buf.device)
    if in_buf.num_elements > 0:
        eager_kernels.logic_ops.BitwiseNot(out, in_buf, _ctx_ptr(in_buf.device))
    return _wrap(out)


@no_type_check
def fast_aten_isin(elements, test_elements, *, assume_unique=False, invert=False):
    el_buf = _buffer_or_none(elements)
    te_buf = _buffer_or_none(test_elements)
    if (
        el_buf is None
        or te_buf is None
        or el_buf.device != te_buf.device
        or el_buf.dtype != te_buf.dtype
        or el_buf.dtype not in (DType.int64, DType.int32)
    ):
        return NOT_HANDLED
    out = _new_buffer(DType.bool, el_buf.shape, el_buf.device)
    if el_buf.num_elements > 0:
        if te_buf.num_elements == 0:
            eager_kernels.elementwise_ops.Fill(
                out, 1.0 if invert else 0.0, _ctx_ptr(el_buf.device)
            )
        else:
            eager_kernels.logic_ops.IsIn(
                out,
                el_buf,
                te_buf,
                te_buf.num_elements,
                1 if invert else 0,
                _ctx_ptr(el_buf.device),
            )
    return _wrap(out)


@no_type_check
def fast_aten_where(condition, input, other):
    cond_buf = _buffer_or_none(condition)
    if cond_buf is None or cond_buf.dtype != DType.bool:
        return NOT_HANDLED
    pair = _binary_operands(input, other)
    if pair is None:
        return NOT_HANDLED
    a_buf, b_buf = pair
    if a_buf.device != cond_buf.device:
        return NOT_HANDLED
    meta = _bcast_meta(cond_buf.shape, a_buf.shape, b_buf.shape)
    if meta is None:
        return NOT_HANDLED
    out = _new_buffer(a_buf.dtype, meta[0], a_buf.device)
    if out.num_elements > 0:
        _launch_bcast(
            eager_kernels.data_movement_ops.WhereSelect,
            out,
            (cond_buf, a_buf, b_buf),
            meta,
        )
    return _wrap(out)


@no_type_check
def _masked_fill_buffer(input, mask, value) -> tuple | None:
    """Resolve (in_buf, mask_buf, value_buf, meta) for masked_fill, or None.

    `value` may be a Python scalar or a 0-d tensor of input's dtype. The
    output shape must equal input's shape (mask broadcasts up to input).
    """
    in_buf = _buffer_or_none(input)
    mask_buf = _buffer_or_none(mask)
    if (
        in_buf is None
        or mask_buf is None
        or mask_buf.dtype != DType.bool
        or mask_buf.device != in_buf.device
    ):
        return None
    val_buf = _buffer_or_none(value)
    if val_buf is not None:
        if (
            val_buf.dtype != in_buf.dtype
            or val_buf.device != in_buf.device
            or val_buf.num_elements != 1
        ):
            return None
        val_buf = val_buf.view(val_buf.dtype, ())
    elif isinstance(value, int | float):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
            return None
        if (
            isinstance(value, float)
            and in_buf.dtype not in _FLOAT_DTYPES
            and not value.is_integer()
        ):
            return None
        val_buf = _scalar_buffer(value, in_buf.dtype, in_buf.device)
    else:
        return None
    meta = _bcast_meta(mask_buf.shape, val_buf.shape, in_buf.shape)
    if meta is None or tuple(meta[0]) != tuple(in_buf.shape):
        return None
    return in_buf, mask_buf, val_buf, meta


@no_type_check
def fast_aten_masked_fill(input, mask, value):
    resolved = _masked_fill_buffer(input, mask, value)
    if resolved is None:
        return NOT_HANDLED
    in_buf, mask_buf, val_buf, meta = resolved
    out = _new_buffer(in_buf.dtype, in_buf.shape, in_buf.device)
    if out.num_elements > 0:
        _launch_bcast(
            eager_kernels.data_movement_ops.WhereSelect,
            out,
            (mask_buf, val_buf, in_buf),
            meta,
        )
    return _wrap(out)


@no_type_check
def fast_aten_masked_fill_(input, mask, value):
    """In-place masked fill into input's buffer. Returns None when
    unavailable (the registration then falls back to the graph path)."""
    resolved = _masked_fill_buffer(input, mask, value)
    if resolved is None:
        return None
    in_buf, mask_buf, val_buf, meta = resolved
    if in_buf.num_elements > 0:
        # Writing out == b is safe: each element reads and writes the
        # same index (input strides are contiguous in the output layout).
        _launch_bcast(
            eager_kernels.data_movement_ops.WhereSelect,
            in_buf,
            (mask_buf, val_buf, in_buf),
            meta,
        )
    return input


# ---------------------------------------------------------------------------
# View / shape metadata ops (zero copy: alias the same driver buffer)
# ---------------------------------------------------------------------------


@no_type_check
def _resolve_sizes(shape, numel: int) -> list[int] | None:
    # Single pass; view is the hottest op in transformer forwards.
    prod = 1
    neg = -1
    for i, s in enumerate(shape):
        if s >= 0:
            prod *= s
        elif s == -1:
            if neg >= 0:
                return None
            neg = i
        else:
            return None
    if neg < 0:
        return list(shape) if prod == numel else None
    if prod == 0 or numel % prod != 0:
        return None
    sizes = list(shape)
    sizes[neg] = numel // prod
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
            shape = list(buffer.shape)
            if dim0 == dim1 or shape[dim0] == 1 or shape[dim1] == 1:
                # Swapping a size-1 dim never reorders memory, so the
                # result is still contiguous: alias instead of copying.
                # (Decode-time transformers transpose (1, 1, h, d) q/k/v
                # every layer; this makes all of those zero-copy.)
                shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
                return _wrap(buffer.view(buffer.dtype, shape))
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
# Slice / select (narrow copies along one dim)
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_slice(input, dim=0, start=None, end=None, step=1):
    buf = _buffer_or_none(input)
    if (
        buf is None
        or buf.dtype not in _COPYABLE_DTYPES
        or step != 1
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    shape = list(buf.shape)
    rank = len(shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = shape[dim]
    start = 0 if start is None else start
    end = size if end is None else end
    if not isinstance(start, int) or not isinstance(end, int):
        return NOT_HANDLED
    if start < 0:
        start += size
    if end < 0:
        end += size
    start = min(max(start, 0), size)
    end = min(max(end, 0), size)
    length = max(end - start, 0)
    out_shape = shape.copy()
    out_shape[dim] = length
    out = _new_buffer(buf.dtype, out_shape, buf.device)
    if out.num_elements > 0:
        inner = math.prod(shape[dim + 1 :])
        eager_kernels.data_movement_ops.NarrowCopy(
            out,
            buf,
            math.prod(shape[:dim]),
            size * inner,
            length * inner,
            start * inner,
            _ctx_ptr(buf.device),
        )
    return _wrap(out)


@no_type_check
def fast_aten_select(input, dim, index):
    buf = _buffer_or_none(input)
    if (
        buf is None
        or buf.dtype not in _COPYABLE_DTYPES
        or not isinstance(dim, int)
        or not isinstance(index, int)
    ):
        return NOT_HANDLED
    shape = list(buf.shape)
    rank = len(shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = shape[dim]
    if index < 0:
        index += size
    if not 0 <= index < size:
        return NOT_HANDLED
    out_shape = shape[:dim] + shape[dim + 1 :]
    out = _new_buffer(buf.dtype, out_shape, buf.device)
    if out.num_elements > 0:
        inner = math.prod(shape[dim + 1 :])
        eager_kernels.data_movement_ops.NarrowCopy(
            out,
            buf,
            math.prod(shape[:dim]),
            size * inner,
            inner,
            index * inner,
            _ctx_ptr(buf.device),
        )
    return _wrap(out)


# ---------------------------------------------------------------------------
# Concatenation along any dim: one destination-strided narrow copy per
# input into the output's slot for that input.
# ---------------------------------------------------------------------------


@no_type_check
def _is_legacy_empty(t) -> bool:
    buffer = _buffer_or_none(t)
    return buffer is not None and len(buffer.shape) == 1 and buffer.num_elements == 0


@no_type_check
def fast_aten_cat(tensors, dim=0):
    # PyTorch's cat skips legacy "empty" (1-D, size-0) tensors, e.g.
    # uninitialized KV-caches.
    real = [t for t in tensors if not _is_legacy_empty(t)]
    if not real or not isinstance(dim, int):
        return NOT_HANDLED
    bufs = [_buffer_or_none(t) for t in real]
    first = bufs[0]
    if first is None or first.dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    rank = len(first.shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    for b in bufs:
        if (
            b is None
            or b.dtype != first.dtype
            or b.device != first.device
            or len(b.shape) != rank
            or any(i != dim and b.shape[i] != first.shape[i] for i in range(rank))
        ):
            return NOT_HANDLED
    out_shape = list(first.shape)
    out_shape[dim] = sum(b.shape[dim] for b in bufs)
    inner = math.prod(out_shape[dim + 1 :])
    outer = math.prod(out_shape[:dim])
    out = _new_buffer(first.dtype, out_shape, first.device)
    dst_stride = out_shape[dim] * inner
    ctx = _ctx_ptr(first.device)
    offset = 0
    for b in bufs:
        copy_len = b.shape[dim] * inner
        if copy_len > 0 and outer > 0:
            eager_kernels.data_movement_ops.NarrowCopyDst(
                out, b, outer, dst_stride, copy_len, offset, ctx
            )
        offset += copy_len
    return _wrap(out)


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
        and 0 < in_buf.num_elements < (1 << 22)
    ):
        out = _new_buffer(DType.bool, (), in_buf.device)
        eager_kernels.nn_ops.AllBool(
            out, in_buf, in_buf.num_elements, _ctx_ptr(in_buf.device)
        )
        return _wrap(out)
    return NOT_HANDLED


@no_type_check
def fast_aten_any(input, dim=None, keepdim=False):
    in_buf = _buffer_or_none(input)
    if (
        in_buf is not None
        and dim is None
        and not keepdim
        and in_buf.dtype == DType.bool
        and 0 < in_buf.num_elements < (1 << 22)
    ):
        out = _new_buffer(DType.bool, (), in_buf.device)
        eager_kernels.nn_ops.AnyBool(
            out, in_buf, in_buf.num_elements, _ctx_ptr(in_buf.device)
        )
        return _wrap(out)
    return NOT_HANDLED


_ROW_REDUCE_DTYPES = _FLOAT_DTYPES + (DType.int64, DType.int32)


@no_type_check
def fast_aten_argmax(input, dim=None, keepdim=False):
    in_buf = _buffer_or_none(input)
    if (
        in_buf is None
        or in_buf.num_elements == 0
        or in_buf.dtype not in _ROW_REDUCE_DTYPES
    ):
        return NOT_HANDLED
    rank = len(in_buf.shape)
    if dim is None:
        rows, cols = 1, in_buf.num_elements
        out_shape = [1] * rank if keepdim else []
    else:
        if not isinstance(dim, int) or rank == 0 or dim % rank != rank - 1:
            return NOT_HANDLED
        cols = in_buf.shape[-1]
        rows = in_buf.num_elements // cols
        out_shape = list(in_buf.shape[:-1]) + ([1] if keepdim else [])
    out = _new_buffer(DType.int64, out_shape, in_buf.device)
    eager_kernels.nn_ops.ArgmaxRows(out, in_buf, rows, cols, _ctx_ptr(in_buf.device))
    return _wrap(out)


@no_type_check
def fast_aten_max(input, *args, **kwargs):
    # Only the values-only overload max(Tensor) -> Tensor.
    if args or kwargs:
        return NOT_HANDLED
    in_buf = _buffer_or_none(input)
    if (
        in_buf is None
        or in_buf.num_elements == 0
        or in_buf.dtype not in _ROW_REDUCE_DTYPES
    ):
        return NOT_HANDLED
    out = _new_buffer(in_buf.dtype, (), in_buf.device)
    eager_kernels.nn_ops.MaxRows(
        out, in_buf, 1, in_buf.num_elements, _ctx_ptr(in_buf.device)
    )
    return _wrap(out)


@no_type_check
def fast_aten_cumsum(input, dim, *, dtype=None):
    in_buf = _buffer_or_none(input) if dtype is None else None
    if (
        in_buf is None
        or in_buf.num_elements == 0
        or in_buf.dtype not in (DType.int64, DType.int32, DType.float32)
    ):
        return NOT_HANDLED
    rank = len(in_buf.shape)
    if not isinstance(dim, int) or rank == 0 or dim % rank != rank - 1:
        return NOT_HANDLED
    cols = in_buf.shape[-1]
    rows = in_buf.num_elements // cols
    out = _new_buffer(in_buf.dtype, in_buf.shape, in_buf.device)
    eager_kernels.nn_ops.CumsumRows(out, in_buf, rows, cols, _ctx_ptr(in_buf.device))
    return _wrap(out)


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
            and _on_gpu(a_buf)
            and a_buf.device == b_buf.device
            and a_buf.dtype == b_buf.dtype
            and a_buf.dtype in _FLOAT_DTYPES
            and len(a_buf.shape) == 2
            and a_buf.shape[1] == b_buf.shape[0]
            and 0 not in a_buf.shape
            and 0 not in b_buf.shape
        ):
            m, k = a_buf.shape
            n = b_buf.shape[1]
            out = _new_buffer(a_buf.dtype, (m, n), a_buf.device)
            eager_kernels.matmul_ops.MatmulBias(
                out, a_buf, b_buf, bias_buf, (m, n, k, 0), _ctx_ptr(a_buf.device)
            )
            return _wrap(out)
    return NOT_HANDLED


@no_type_check
def fast_aten_linear(input, weight, bias=None):
    # linear(input, weight, bias) = input @ weight^T + bias. Registering the
    # composite op (instead of letting torch decompose it into t() + mm)
    # matters because our buffers are always dense: the t() would have to
    # materialize a transposed copy of the weight — ~154 MB per decode step
    # for a GPT-2 lm_head — while the GEMM kernel reads B transposed for free.
    in_buf = _buffer_or_none(input)
    w_buf = _buffer_or_none(weight)
    if (
        in_buf is not None
        and w_buf is not None
        and _on_gpu(in_buf)
        and in_buf.device == w_buf.device
        and in_buf.dtype == w_buf.dtype
        and in_buf.dtype in _FLOAT_DTYPES
        and len(in_buf.shape) >= 2
        and len(w_buf.shape) == 2
        and in_buf.shape[-1] == w_buf.shape[1]
        and 0 not in in_buf.shape
        and 0 not in w_buf.shape
    ):
        n, k = w_buf.shape
        bias_buf = None
        if bias is not None:
            bias_buf = _buffer_or_none(bias)
            if (
                bias_buf is None
                or bias_buf.dtype != in_buf.dtype
                or tuple(bias_buf.shape) != (n,)
            ):
                return NOT_HANDLED
        m = in_buf.num_elements // k
        out_shape = tuple(in_buf.shape[:-1]) + (n,)
        out = _new_buffer(in_buf.dtype, out_shape, in_buf.device)
        ctx = _ctx_ptr(in_buf.device)
        if bias_buf is not None:
            eager_kernels.matmul_ops.MatmulBias(
                out, in_buf, w_buf, bias_buf, (m, n, k, 1), ctx
            )
        else:
            eager_kernels.matmul_ops.Matmul(out, in_buf, w_buf, (m, n, k, 1), ctx)
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
# Convolution (GPU only), pure Mojo: batched im2col + the pure GEMM with
# the torch (K,C,R,S) weight used as-is and NCHW output — no layout
# permutes and no cuDNN. Grouped convolutions slice the channel-major
# im2col rows and weights per group with element offsets.
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
            cols = out_h * out_w
            ckk = c * kh * kw
            if (kh, kw, sh, sw, ph, pw, dh, dw) == (1, 1, 1, 1, 0, 0, 1, 1):
                # 1x1 stride-1 conv: NCHW input already is the col matrix.
                col = in_buf.view(in_buf.dtype, (n, ckk, cols))
            else:
                col = _new_buffer(in_buf.dtype, (n, ckk, cols), in_buf.device)
                eager_kernels.conv_ops.Im2col(
                    col,
                    in_buf,
                    (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, c, n),
                    ctx,
                )
            out = _new_buffer(in_buf.dtype, (n, out_c, cols), in_buf.device)
            if groups == 1:
                eager_kernels.matmul_ops.Bmm(
                    out,
                    w_buf.view(w_buf.dtype, (out_c, ckk)),
                    col,
                    (n, out_c, cols, ckk, 0, 1),  # a_shared=1: broadcast weights
                    ctx,
                )
            else:
                # Channel-major im2col rows make each group a contiguous
                # (crs_g, cols) slice; run one offset GEMM per (sample, group).
                crs_g = c_per_group * kh * kw
                oc_g = out_c // groups
                w_view = w_buf.view(w_buf.dtype, (out_c, crs_g))
                for s in range(n):
                    for g in range(groups):
                        eager_kernels.matmul_ops.Matmul(
                            out,
                            w_view,
                            col,
                            (
                                oc_g,
                                cols,
                                crs_g,
                                0,
                                (s * out_c + g * oc_g) * cols,
                                g * oc_g * crs_g,
                                (s * c + g * c_per_group) * kh * kw * cols,
                            ),
                            ctx,
                        )
            if bias_buf is not None:
                eager_kernels.conv_ops.BiasAddChan(out, bias_buf, (cols, out_c), ctx)
            return _wrap(out.view(out.dtype, (n, out_c, out_h, out_w)))
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
        if (
            q_len == 1
            and not is_causal
            and head_dim % 4 == 0
            and head_dim <= 256
            and kv_len <= 4096
        ):
            # Decode step: one fused kernel instead of bmm+softmax+bmm
            # (single launch, no scratch buffers, coalesced K/V reads).
            out = _new_buffer(q_buf.dtype, (b, h, 1, head_dim), q_buf.device)
            eager_kernels.nn_ops.AttnDecode(
                out,
                q_buf,
                k_buf,
                v_buf,
                (b * h, kv_len, head_dim, float(scale_val)),
                ctx,
            )
            return _wrap(out)
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
# Filled factories (full / ones / zeros / scalar_tensor): one allocation
# plus one Fill kernel. The registrations resolve torch dtype/device.
# ---------------------------------------------------------------------------

# What the Fill kernel dispatches on (no uint16/32/64).
_FILL_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
)


@no_type_check
def fast_filled(shape, value, dtype: DType, device: driver.Device):
    """A TorchMaxTensor of `shape` filled with `value`, or None."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int | float):
        return None
    if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
        return None
    if dtype not in _FILL_DTYPES:
        return None
    out = _new_buffer(dtype, tuple(shape), device)
    if out.num_elements > 0:
        eager_kernels.elementwise_ops.Fill(out, float(value), _ctx_ptr(device))
    return _wrap(out)


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
