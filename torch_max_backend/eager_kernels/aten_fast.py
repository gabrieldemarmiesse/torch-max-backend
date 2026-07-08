"""ATen-signature-compatible fast implementations for max_device eager mode.

Each function here is registered in `max_device_aten_ops.py` (through
`_eager_impl`). Tensors are `TorchMaxTensor`s: a Mojo `TensorHolder`
ownership token plus Python-side layout metadata (`_ptr`, `_shape`,
`_strides` in elements, `_offset`, `_dtype`, `_device`, `_is_contiguous`).
An op runs as one or a few Mojo kernel calls — no graph building, no MLIR
passes, no interpreter, and *no graph fallback*: when a function returns
the `NOT_HANDLED` sentinel, the registration raises `NotImplementedError`
naming the op (see docs/strided_owning_tensors_design.md).

View ops (view / permute / transpose / expand / slice / select / split /
squeeze / unsqueeze / alias) are ZERO-COPY: they return new wrappers
sharing the base tensor's holder with adjusted shape/strides/offset.
Compute kernels take contiguous operands; strided inputs either feed the
broadcast-strided kernels directly (rank <= 4, real strides) or are
materialized first via the `CopyStrided` primitive.

Only the eager (max_device) path uses this module; the torch.compile
backend keeps using `aten_functions` directly.
"""

import math
from typing import no_type_check

from max.dtype import DType

from torch_max_backend import eager_kernels, is_running_tests
from torch_max_backend.eager_kernels import _ctx_ptr
from torch_max_backend.max_device.torch_max_tensor import (
    TorchMaxTensor,
    _copy_strided_into,
    _pad8,
    _row_major_strides,
)

# Returned when the inputs don't qualify; the registration then raises
# NotImplementedError naming the op (there is no fallback anymore).
NOT_HANDLED = object()

# The Mojo kernels raise (instead of falling back) on dtypes they don't
# support; gate float-only ops here.
_FLOAT_DTYPES = (DType.float16, DType.bfloat16, DType.float32)

_INT_SCALAR_DTYPES = (DType.int32, DType.int64)

# Dtypes the binary elementwise kernels support (no bool: the arithmetic
# kernels don't lower for it, so bool tensors go through uint8 views or
# are rejected).
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
def _t(x) -> TorchMaxTensor | None:
    """x as a TorchMaxTensor (any layout), or None."""
    return x if isinstance(x, TorchMaxTensor) else None


@no_type_check
def _tc(x) -> TorchMaxTensor | None:
    """x as a *contiguous* TorchMaxTensor (materializing views), or None."""
    if isinstance(x, TorchMaxTensor):
        return x if x._is_contiguous else x._materialize_contiguous()
    return None


_alloc = TorchMaxTensor._alloc
_view_of = TorchMaxTensor._view_of


@no_type_check
def _on_gpu(t: TorchMaxTensor) -> bool:
    return t._device.label == "gpu"


@no_type_check
def _copy_into(dst: TorchMaxTensor, src: TorchMaxTensor) -> None:
    """dst[...] = src[...] for equal shapes/dtypes, any strides on both."""
    if dst._numel == 0:
        return
    if dst._is_contiguous and src._is_contiguous:
        eager_kernels.tensor_holder.copy_d2d(
            _ctx_ptr(dst._device), dst._ptr, src._ptr, dst._numel * dst._itemsize
        )
    else:
        _copy_strided_into(dst, src)


# ---------------------------------------------------------------------------
# Elementwise helper layer
# ---------------------------------------------------------------------------


@no_type_check
def _try_binary(mojo_fn, lhs, rhs):
    """Elementwise binary kernel on two contiguous same-shape tensors."""
    a = _t(lhs)
    b = _t(rhs)
    if (
        a is None
        or b is None
        or not a._is_contiguous
        or not b._is_contiguous
        or a._dtype not in _BINARY_DTYPES
        or a._dtype != b._dtype
        or a._shape != b._shape
        or a._device != b._device
    ):
        return None
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        mojo_fn(
            out._ptr, a._ptr, b._ptr, out._numel, a._dtype.value, _ctx_ptr(a._device)
        )
    return out


@no_type_check
def _try_unary(mojo_fn, x, dtypes=_COPYABLE_DTYPES):
    a = _tc(x)
    if a is None or a._dtype not in dtypes:
        return None
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        mojo_fn(out._ptr, a._ptr, out._numel, a._dtype.value, _ctx_ptr(a._device))
    return out


@no_type_check
def _try_bool_and(lhs, rhs):
    """bool * bool (= logical AND) via the uint8 Mul kernel."""
    a = _tc(lhs)
    b = _tc(rhs)
    if (
        a is None
        or b is None
        or a._dtype != DType.bool
        or b._dtype != DType.bool
        or a._shape != b._shape
        or a._device != b._device
    ):
        return None
    out = _alloc(a._shape, DType.bool, a._device)
    if out._numel > 0:
        eager_kernels.elementwise_ops.Mul(
            out._ptr, a._ptr, b._ptr, out._numel, DType.uint8.value, _ctx_ptr(a._device)
        )
    return out


@no_type_check
def _try_scalar(mojo_fn, x, scalar):
    if not isinstance(scalar, int | float) or isinstance(scalar, bool):
        return None
    a = _tc(x)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return None
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        mojo_fn(
            out._ptr,
            a._ptr,
            float(scalar),
            out._numel,
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def _try_int_scalar(mojo_fn, x, scalar):
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        return None
    a = _tc(x)
    if a is None or a._dtype not in _INT_SCALAR_DTYPES:
        return None
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        mojo_fn(
            out._ptr, a._ptr, scalar, out._numel, a._dtype.value, _ctx_ptr(a._device)
        )
    return out


# ---------------------------------------------------------------------------
# Broadcast helpers for the strided kernels (logic_ops / WhereSelect).
# Operands are described by the output's dims padded to rank 4 plus
# per-operand REAL element strides (0 on broadcast dims), so strided and
# expanded views feed these kernels with no materialization.
# ---------------------------------------------------------------------------


@no_type_check
def _bcast_meta(*tensors):
    """Broadcast metadata (rank <= 4) from the tensors' real strides.

    Returns (out_shape, dims4, [strides4 per tensor]) or None when the
    shapes don't broadcast or exceed rank 4.
    """
    rank = max((len(t._shape) for t in tensors), default=0)
    if rank > 4:
        return None
    out = []
    for i in range(rank):
        size = 1
        for t in tensors:
            j = i - (rank - len(t._shape))
            if j >= 0 and t._shape[j] != 1:
                if size != 1 and t._shape[j] != size:
                    return None
                size = t._shape[j]
        out.append(size)
    dims = [1] * (4 - rank) + out
    all_strides = []
    for t in tensors:
        pad = rank - len(t._shape)
        st = []
        for i in range(rank):
            j = i - pad
            if j < 0 or t._shape[j] == 1:
                st.append(0)
            else:
                st.append(t._strides[j])
        all_strides.append([0] * (4 - rank) + st)
    return out, dims, all_strides


@no_type_check
def _scalar_tensor_0d(value, dtype, device) -> TorchMaxTensor:
    """A 0-d tensor holding `value`, for stride-0 broadcast operands."""
    out = _alloc((), dtype, device)
    eager_kernels.elementwise_ops.Fill(
        out._ptr, float(value), 1, dtype.value, _ctx_ptr(device)
    )
    return out


@no_type_check
def _cast_tensor(x: TorchMaxTensor, dtype: DType) -> TorchMaxTensor:
    """Contiguous dtype cast through the Cast kernel."""
    a = _tc(x)
    out = _alloc(a._shape, dtype, a._device)
    if out._numel > 0:
        eager_kernels.data_movement_ops.Cast(
            out._ptr,
            dtype.value,
            a._ptr,
            a._dtype.value,
            out._numel,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def _promoted_pair(a: TorchMaxTensor, b: TorchMaxTensor):
    """Same-dtype tensor pair following torch's promotion, or None.

    Only the promotions the generation loops hit: bool combined with any
    castable dtype, and int32 with int64.
    """
    if a._dtype == b._dtype:
        return a, b
    if a._dtype == DType.bool and b._dtype in _CAST_DTYPES:
        return _cast_tensor(a, b._dtype), b
    if b._dtype == DType.bool and a._dtype in _CAST_DTYPES:
        return a, _cast_tensor(b, a._dtype)
    if a._dtype == DType.int32 and b._dtype == DType.int64:
        return _cast_tensor(a, DType.int64), b
    if a._dtype == DType.int64 and b._dtype == DType.int32:
        return a, _cast_tensor(b, DType.int64)
    return None


@no_type_check
def _resolve_scalar(value, dtype: DType, device) -> TorchMaxTensor | None:
    """A 0-d stride-0 tensor holding a Python scalar in `dtype`, or None
    when the value doesn't embed losslessly."""
    if not isinstance(value, int | float):
        return None
    if dtype not in _FILL_DTYPES:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_INT:
            return None
        if dtype == DType.bool and value not in (0, 1):
            return None
    elif dtype not in _FLOAT_DTYPES + (DType.float64,):
        # float scalar against an int tensor promotes in torch; not handled.
        return None
    return _scalar_tensor_0d(value, dtype, device)


@no_type_check
def _binary_operands(input, other):
    """Resolve (lhs, rhs) TorchMaxTensors with equal dtypes for a broadcast
    binary/comparison kernel. Either operand may be a Python scalar (which
    becomes a 0-d stride-0 tensor of the tensor operand's dtype), or None
    if unresolvable.
    """
    a = _t(input)
    b = _t(other)
    if a is not None and b is not None:
        if a._device != b._device:
            return None
        return _promoted_pair(a, b)
    if a is not None:
        scalar = _resolve_scalar(other, a._dtype, a._device)
        return None if scalar is None else (a, scalar)
    if b is not None:
        # Scalar-first calls, e.g. rsub-style `1 - tensor`.
        scalar = _resolve_scalar(input, b._dtype, b._device)
        return None if scalar is None else (scalar, b)
    return None


@no_type_check
def _launch_bcast(kernel, out, operands, meta, dtype):
    out_shape, dims, strides = meta
    params = tuple(dims) + tuple(s for st in strides for s in st)
    kernel(
        out._ptr,
        *[t._ptr for t in operands],
        params,
        dtype.value,
        _ctx_ptr(out._device),
    )


@no_type_check
def _try_binary_bcast(kernel_name, lhs, rhs):
    """Broadcast-strided arithmetic on two operands (tensor or scalar)."""
    pair = _binary_operands(lhs, rhs)
    if pair is None:
        return None
    a, b = pair
    if a._dtype not in _BCAST_DTYPES:
        return None
    if kernel_name == "DivBcast" and a._dtype not in _FLOAT_DTYPES:
        return None
    meta = _bcast_meta(a, b)
    if meta is None:
        return None
    out = _alloc(meta[0], a._dtype, a._device)
    if out._numel > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (a, b), meta, a._dtype
        )
    return out


@no_type_check
def _try_cmp(kernel_name, input, other):
    """Broadcast-strided comparison -> bool tensor, or None."""
    pair = _binary_operands(input, other)
    if pair is None:
        return None
    a, b = pair
    dtype = DType.uint8 if a._dtype == DType.bool else a._dtype
    if dtype not in _BCAST_DTYPES:
        return None
    meta = _bcast_meta(a, b)
    if meta is None:
        return None
    out = _alloc(meta[0], DType.bool, a._device)
    if out._numel > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (a, b), meta, dtype
        )
    return out


@no_type_check
def _try_bitwise(kernel_name, input, other):
    """Broadcast-strided bitwise op (bool via the uint8 dispatch), or None."""
    pair = _binary_operands(input, other)
    if pair is None:
        return None
    a, b = pair
    if a._dtype not in _BITWISE_DTYPES:
        return None
    meta = _bcast_meta(a, b)
    if meta is None:
        return None
    out = _alloc(meta[0], a._dtype, a._device)
    dtype = DType.uint8 if a._dtype == DType.bool else a._dtype
    if out._numel > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (a, b), meta, dtype
        )
    return out


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------


@no_type_check
def _scaled_operand(other, alpha):
    """other * alpha as a tensor, for add/sub with alpha != 1, or None."""
    b = _t(other)
    if b is None:
        if isinstance(other, int | float) and not isinstance(other, bool):
            return other * alpha
        return None
    scaled = _try_scalar(eager_kernels.elementwise_ops.MulScalar, b, alpha)
    if scaled is None:
        scaled = _try_int_scalar(eager_kernels.elementwise_ops.MulScalarInt, b, alpha)
    return scaled


@no_type_check
def fast_aten_add(input, other, alpha=1):
    if alpha != 1:
        other = _scaled_operand(other, alpha)
        if other is None:
            return NOT_HANDLED
    result = _try_binary(eager_kernels.elementwise_ops.Add, input, other)
    if result is None:
        result = _try_scalar(eager_kernels.elementwise_ops.AddScalar, input, other)
    if result is None:
        result = _try_int_scalar(
            eager_kernels.elementwise_ops.AddScalarInt, input, other
        )
    if result is None:
        result = _try_binary_bcast("AddBcast", input, other)
    if result is None:
        result = _try_binary(eager_kernels.elementwise_ops.Add, _tc(input), _tc(other))
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_add_(input, other, alpha=1):
    """In-place add into input. Returns None when unavailable."""
    dst = _t(input)
    if dst is None:
        return None
    # Direct in-place kernel when everything lines up.
    b = _t(other)
    if (
        alpha == 1
        and b is not None
        and dst._is_contiguous
        and b._is_contiguous
        and dst._dtype in _BINARY_DTYPES
        and dst._dtype == b._dtype
        and dst._shape == b._shape
        and dst._device == b._device
    ):
        if dst._numel > 0:
            eager_kernels.elementwise_ops.Add(
                dst._ptr,
                dst._ptr,
                b._ptr,
                dst._numel,
                dst._dtype.value,
                _ctx_ptr(dst._device),
            )
        return input
    # General path: functional result, then a (strided-safe) copy back.
    result = fast_aten_add(input, other, alpha)
    if (
        result is NOT_HANDLED
        or result._shape != dst._shape
        or result._dtype != dst._dtype
    ):
        return None
    _copy_into(dst, result)
    return input


@no_type_check
def fast_aten_sub(input, other, alpha=1):
    if alpha != 1:
        other = _scaled_operand(other, alpha)
        if other is None:
            return NOT_HANDLED
    result = _try_binary(eager_kernels.elementwise_ops.Sub, input, other)
    if (
        result is None
        and isinstance(other, int | float)
        and not isinstance(other, bool)
    ):
        result = _try_scalar(eager_kernels.elementwise_ops.AddScalar, input, -other)
        if result is None and isinstance(other, int):
            result = _try_int_scalar(
                eager_kernels.elementwise_ops.AddScalarInt, input, -other
            )
    if result is None:
        result = _try_binary_bcast("SubBcast", input, other)
    if result is None:
        result = _try_binary(eager_kernels.elementwise_ops.Sub, _tc(input), _tc(other))
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
    if result is None:
        result = _try_binary(eager_kernels.elementwise_ops.Mul, _tc(input), _tc(other))
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_div(input, other, *, rounding_mode=None):
    if rounding_mode is not None:
        return NOT_HANDLED
    a = _t(input)
    if a is not None and a._dtype in _FLOAT_DTYPES:
        result = _try_binary(eager_kernels.elementwise_ops.Div, input, other)
        if result is None:
            result = _try_binary_bcast("DivBcast", input, other)
        if result is None:
            result = _try_binary(
                eager_kernels.elementwise_ops.Div, _tc(input), _tc(other)
            )
        if result is not None:
            return result
        return NOT_HANDLED
    # Integer (and int-scalar) division promotes to float32 in torch.
    if a is not None and a._dtype in _INT_SCALAR_DTYPES + (
        DType.int8,
        DType.int16,
        DType.uint8,
    ):
        lhs = _cast_tensor(a, DType.float32)
        b = _t(other)
        if b is not None:
            if b._device != a._device:
                return NOT_HANDLED
            rhs = _cast_tensor(b, DType.float32) if b._dtype != DType.float32 else b
        elif isinstance(other, int | float) and not isinstance(other, bool):
            rhs = other
        else:
            return NOT_HANDLED
        return fast_aten_div(lhs, rhs)
    return NOT_HANDLED


@no_type_check
def fast_aten_fill_scalar(input, value):
    """Functional fill: new tensor, same shape/dtype, all elements = value."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_HANDLED
    a = _t(input)
    if a is None or a._dtype not in _FILL_DTYPES:
        return NOT_HANDLED
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        eager_kernels.elementwise_ops.Fill(
            out._ptr, float(value), out._numel, out._dtype.value, _ctx_ptr(out._device)
        )
    return out


@no_type_check
def fast_aten_fill__scalar(input, value):
    """In-place fill of input (any strides). Returns None when unavailable."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    a = _t(input)
    if a is None:
        return None
    if a._numel > 0:
        eager_kernels.tensor_holder.StridedFill(
            a._ptr,
            float(value),
            _pad8(a._shape, 1),
            _pad8(a._strides, 0),
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return input


@no_type_check
def fast_aten_maximum(x, y):
    result = _try_binary(eager_kernels.elementwise_ops.Max, x, y)
    if result is None:
        result = _try_binary(eager_kernels.elementwise_ops.Max, _tc(x), _tc(y))
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_minimum(x, y):
    result = _try_binary(eager_kernels.elementwise_ops.Min, x, y)
    if result is None:
        result = _try_binary(eager_kernels.elementwise_ops.Min, _tc(x), _tc(y))
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
# Unary elementwise suite
#
# One generic `_try_unary` call per op. The float-only transcendentals (and
# ceil/floor/gelu/sigmoid/silu) gate on `_FLOAT_DTYPES`; abs/neg/sign also
# dispatch on the integer dtypes the kernel handles. isnan / logical_not go
# through the unary-to-bool kernels and always produce a bool tensor.
# ---------------------------------------------------------------------------

# abs/neg/sign accept the signed ints + uint8 the Neg/Abs/Sign kernels
# dispatch on, plus the float dtypes (float64 works on the CPU device).
_SIGNED_UNARY_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)

# Dtypes the unary-to-bool kernels (IsNan / LogicalNot) dispatch on. bool
# tensors are routed through their uint8 storage below.
_BOOL_UNARY_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
)


@no_type_check
def _try_unary_bool(mojo_fn, x):
    """Unary op producing a bool tensor (isnan / logical_not)."""
    a = _tc(x)
    if a is None:
        return None
    kernel_dtype = DType.uint8 if a._dtype == DType.bool else a._dtype
    if kernel_dtype not in _BOOL_UNARY_DTYPES:
        return None
    out = _alloc(a._shape, DType.bool, a._device)
    if out._numel > 0:
        mojo_fn(out._ptr, a._ptr, out._numel, kernel_dtype.value, _ctx_ptr(a._device))
    return out


@no_type_check
def _unary_op(mojo_fn, x, dtypes=_FLOAT_DTYPES):
    result = _try_unary(mojo_fn, x, dtypes)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_abs(x):
    return _unary_op(eager_kernels.elementwise_ops.Abs, x, _SIGNED_UNARY_DTYPES)


@no_type_check
def fast_aten_neg(x):
    return _unary_op(eager_kernels.elementwise_ops.Neg, x, _SIGNED_UNARY_DTYPES)


@no_type_check
def fast_aten_sign(x):
    return _unary_op(eager_kernels.elementwise_ops.Sign, x, _SIGNED_UNARY_DTYPES)


@no_type_check
def fast_aten_ceil(x):
    return _unary_op(eager_kernels.elementwise_ops.Ceil, x)


@no_type_check
def fast_aten_floor(x):
    return _unary_op(eager_kernels.elementwise_ops.Floor, x)


@no_type_check
def fast_aten_acos(x):
    return _unary_op(eager_kernels.elementwise_ops.Acos, x)


@no_type_check
def fast_aten_asinh(x):
    return _unary_op(eager_kernels.elementwise_ops.Asinh, x)


@no_type_check
def fast_aten_atanh(x):
    return _unary_op(eager_kernels.elementwise_ops.Atanh, x)


@no_type_check
def fast_aten_cos(x):
    return _unary_op(eager_kernels.elementwise_ops.Cos, x)


@no_type_check
def fast_aten_cosh(x):
    return _unary_op(eager_kernels.elementwise_ops.Cosh, x)


@no_type_check
def fast_aten_erf(x):
    return _unary_op(eager_kernels.elementwise_ops.Erf, x)


@no_type_check
def fast_aten_log(x):
    return _unary_op(eager_kernels.elementwise_ops.Log, x)


@no_type_check
def fast_aten_log1p(x):
    return _unary_op(eager_kernels.elementwise_ops.Log1p, x)


@no_type_check
def fast_aten_reciprocal(x):
    return _unary_op(eager_kernels.elementwise_ops.Reciprocal, x)


@no_type_check
def fast_aten_rsqrt(x):
    return _unary_op(eager_kernels.elementwise_ops.Rsqrt, x)


@no_type_check
def fast_aten_sigmoid(x):
    return _unary_op(eager_kernels.elementwise_ops.Sigmoid, x)


@no_type_check
def fast_aten_silu(x):
    return _unary_op(eager_kernels.elementwise_ops.Silu, x)


@no_type_check
def fast_aten_sin(x):
    return _unary_op(eager_kernels.elementwise_ops.Sin, x)


@no_type_check
def fast_aten_sinh(x):
    return _unary_op(eager_kernels.elementwise_ops.Sinh, x)


@no_type_check
def fast_aten_sqrt(x):
    return _unary_op(eager_kernels.elementwise_ops.Sqrt, x)


@no_type_check
def fast_aten_tan(x):
    return _unary_op(eager_kernels.elementwise_ops.Tan, x)


@no_type_check
def fast_aten_gelu(input, approximate="none"):
    if approximate == "none":
        fn = eager_kernels.elementwise_ops.GeluNone
    elif approximate == "tanh":
        fn = eager_kernels.elementwise_ops.GeluTanh
    else:
        return NOT_HANDLED
    return _unary_op(fn, input)


@no_type_check
def fast_aten_isnan(x):
    result = _try_unary_bool(eager_kernels.elementwise_ops.IsNan, x)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_logical_not(x):
    result = _try_unary_bool(eager_kernels.elementwise_ops.LogicalNot, x)
    return result if result is not None else NOT_HANDLED


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
    a = _tc(input)
    if a is None or a._dtype not in _BITWISE_DTYPES:
        return NOT_HANDLED
    out = _alloc(a._shape, a._dtype, a._device)
    dtype = DType.uint8 if a._dtype == DType.bool else a._dtype
    if out._numel > 0:
        eager_kernels.logic_ops.BitwiseNot(
            out._ptr, a._ptr, out._numel, dtype.value, _ctx_ptr(a._device)
        )
    return out


@no_type_check
def fast_aten_isin(elements, test_elements, *, assume_unique=False, invert=False):
    el = _tc(elements)
    te = _tc(test_elements)
    if (
        el is None
        or te is None
        or el._device != te._device
        or el._dtype != te._dtype
        or el._dtype not in (DType.int64, DType.int32)
    ):
        return NOT_HANDLED
    out = _alloc(el._shape, DType.bool, el._device)
    if el._numel > 0:
        if te._numel == 0:
            eager_kernels.elementwise_ops.Fill(
                out._ptr,
                1.0 if invert else 0.0,
                out._numel,
                DType.bool.value,
                _ctx_ptr(el._device),
            )
        else:
            eager_kernels.logic_ops.IsIn(
                out._ptr,
                el._ptr,
                te._ptr,
                el._numel,
                te._numel,
                1 if invert else 0,
                el._dtype.value,
                _ctx_ptr(el._device),
            )
    return out


# ---------------------------------------------------------------------------
# Binary/ternary extras: remainder, floor_divide, pow(Tensor,Tensor),
# logical_and/xor, clamp, addcmul/addcdiv. All ride the broadcast-strided
# kernels (real strides, no materialization) except clamp, which is a
# contiguous unary.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_remainder(input, other):
    # Divisor-signed remainder (Python/torch `%`), float and int dtypes.
    result = _try_binary_bcast("RemainderBcast", input, other)
    if result is None:
        result = _try_binary_bcast("RemainderBcast", _tc(input), _tc(other))
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_floor_divide(input, other):
    # floor(input / other), float and int dtypes.
    result = _try_binary_bcast("FloorDivBcast", input, other)
    if result is None:
        result = _try_binary_bcast("FloorDivBcast", _tc(input), _tc(other))
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_pow_tensor_tensor(input, exponent):
    # Float-only (the kernel raises on ints, which would leave the output
    # unwritten); gate here so unsupported dtypes fall through cleanly.
    a = _t(input)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    result = _try_binary_bcast("PowBcast", input, exponent)
    if result is None:
        result = _try_binary_bcast("PowBcast", _tc(input), _tc(exponent))
    return result if result is not None else NOT_HANDLED


@no_type_check
def _try_logical(kernel_name, input, other):
    """logical_and / logical_xor: bool output from any input dtype pair.

    Same-dtype operands (in the bcast set) test nonzero-ness inline with no
    materialization; mixed dtypes are cast to bool first (which also does the
    nonzero test) and combined through the uint8 dispatch.
    """
    a = _t(input)
    b = _t(other)
    if a is None or b is None or a._device != b._device:
        return None
    if a._dtype == b._dtype and a._dtype in _BCAST_DTYPES:
        da, db, dtype = a, b, a._dtype
    elif a._dtype == DType.bool and b._dtype == DType.bool:
        da, db, dtype = a, b, DType.uint8
    else:
        # Mixed dtypes: reduce each to bool (nonzero test) via Cast.
        if a._dtype not in _CAST_DTYPES or b._dtype not in _CAST_DTYPES:
            return None
        da = a if a._dtype == DType.bool else _cast_tensor(a, DType.bool)
        db = b if b._dtype == DType.bool else _cast_tensor(b, DType.bool)
        dtype = DType.uint8
    meta = _bcast_meta(da, db)
    if meta is None:
        return None
    out = _alloc(meta[0], DType.bool, a._device)
    if out._numel > 0:
        _launch_bcast(
            getattr(eager_kernels.logic_ops, kernel_name), out, (da, db), meta, dtype
        )
    return out


@no_type_check
def fast_aten_logical_and(input, other):
    result = _try_logical("LogicalAndBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_logical_xor(input, other):
    result = _try_logical("LogicalXorBcast", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_clamp(input, min=None, max=None):
    a = _tc(input)
    if a is None or a._dtype not in _BCAST_DTYPES:
        return NOT_HANDLED
    if min is None and max is None:
        return NOT_HANDLED
    for bound in (min, max):
        if bound is not None and not isinstance(bound, int | float):
            return NOT_HANDLED
    has_min = min is not None
    has_max = max is not None
    lo = float(min) if has_min else 0.0
    hi = float(max) if has_max else 0.0
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        eager_kernels.logic_ops.ClampScalar(
            out._ptr,
            a._ptr,
            lo,
            hi,
            1 if has_min else 0,
            1 if has_max else 0,
            out._numel,
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def _try_addc(kernel_name, self, tensor1, tensor2, value, allow_int):
    a = _t(self)
    b = _t(tensor1)
    c = _t(tensor2)
    if a is None or b is None or c is None:
        return NOT_HANDLED
    if a._device != b._device or a._device != c._device:
        return NOT_HANDLED
    dtype = a._dtype
    if b._dtype != dtype or c._dtype != dtype:
        return NOT_HANDLED
    if dtype in _FLOAT_DTYPES:
        pass
    elif allow_int and dtype in (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ):
        pass
    else:
        return NOT_HANDLED
    if not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_HANDLED
    meta = _bcast_meta(a, b, c)
    if meta is None:
        return NOT_HANDLED
    out_shape, dims, strides = meta
    out = _alloc(out_shape, dtype, a._device)
    if out._numel > 0:
        params = tuple(dims) + tuple(s for st in strides for s in st)
        getattr(eager_kernels.logic_ops, kernel_name)(
            out._ptr,
            a._ptr,
            b._ptr,
            c._ptr,
            params,
            float(value),
            dtype.value,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def fast_aten_addcmul(self, tensor1, tensor2, value=1):
    return _try_addc("AddcmulBcast", self, tensor1, tensor2, value, True)


@no_type_check
def fast_aten_addcdiv(self, tensor1, tensor2, value=1):
    return _try_addc("AddcdivBcast", self, tensor1, tensor2, value, False)


@no_type_check
def fast_aten_where(condition, input, other):
    cond = _t(condition)
    if cond is None or cond._dtype != DType.bool:
        return NOT_HANDLED
    pair = _binary_operands(input, other)
    if pair is None:
        return NOT_HANDLED
    a, b = pair
    if a._device != cond._device:
        return NOT_HANDLED
    meta = _bcast_meta(cond, a, b)
    if meta is None:
        return NOT_HANDLED
    out = _alloc(meta[0], a._dtype, a._device)
    if out._numel > 0:
        _launch_bcast(
            eager_kernels.data_movement_ops.WhereSelect,
            out,
            (cond, a, b),
            meta,
            a._dtype,
        )
    return out


@no_type_check
def _masked_fill_operands(input, mask, value):
    """Resolve (in_t, mask_t, value_t, meta) for masked_fill, or None.

    `value` may be a Python scalar or a 1-element tensor of input's dtype.
    The output shape must equal input's shape (mask broadcasts up to input).
    """
    a = _t(input)
    m = _t(mask)
    if a is None or m is None or m._dtype != DType.bool or m._device != a._device:
        return None
    val = _t(value)
    if val is not None:
        if val._dtype != a._dtype or val._device != a._device or val._numel != 1:
            return None
        if len(val._shape) > 0:
            val = _view_of(val, (), (), val._offset)
    elif isinstance(value, int | float):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
            return None
        if (
            isinstance(value, float)
            and a._dtype not in _FLOAT_DTYPES
            and not value.is_integer()
        ):
            return None
        if a._dtype not in _FILL_DTYPES:
            return None
        val = _scalar_tensor_0d(value, a._dtype, a._device)
    else:
        return None
    meta = _bcast_meta(m, val, a)
    if meta is None or tuple(meta[0]) != tuple(a._shape):
        return None
    return a, m, val, meta


@no_type_check
def fast_aten_masked_fill(input, mask, value):
    resolved = _masked_fill_operands(input, mask, value)
    if resolved is None:
        return NOT_HANDLED
    a, m, val, meta = resolved
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        _launch_bcast(
            eager_kernels.data_movement_ops.WhereSelect,
            out,
            (m, val, a),
            meta,
            a._dtype,
        )
    return out


@no_type_check
def fast_aten_masked_fill_(input, mask, value):
    """In-place masked fill into input. Returns None when unavailable."""
    resolved = _masked_fill_operands(input, mask, value)
    if resolved is None:
        return None
    a, m, val, meta = resolved
    if a._numel > 0:
        if a._is_contiguous:
            # Writing out == a is safe: each element reads and writes the
            # same index (a's strides are the output layout).
            _launch_bcast(
                eager_kernels.data_movement_ops.WhereSelect,
                a,
                (m, val, a),
                meta,
                a._dtype,
            )
        else:
            result = fast_aten_masked_fill(input, mask, value)
            if result is NOT_HANDLED:
                return None
            _copy_into(a, result)
    return input


# ---------------------------------------------------------------------------
# View / shape-metadata ops. ALL ZERO-COPY: new wrappers over the same
# holder with adjusted shape/strides/offset.
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
def _compute_view_strides(old_shape, old_strides, new_shape):
    """Port of ATen's computeStride: strides for viewing `old` as
    `new_shape` without a copy, or None when impossible."""
    if len(old_shape) == 0:
        return (1,) * len(new_shape)
    numel = 1
    for s in old_shape:
        numel *= s
    if numel == 0:
        if tuple(old_shape) == tuple(new_shape):
            return tuple(old_strides)
        return _row_major_strides(new_shape)

    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base_stride = old_strides[-1]
    tensor_numel = 1
    view_numel = 1
    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]
        if tensor_d == 0 or (
            old_shape[tensor_d - 1] != 1
            and old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride
        ):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = old_strides[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return tuple(new_strides)


@no_type_check
def _fast_view(tensor, shape):
    if len(shape) == 1 and isinstance(shape[0], list | tuple):
        shape = shape[0]
    if not all(isinstance(s, int) for s in shape):
        return NOT_HANDLED
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    sizes = _resolve_sizes(shape, t._numel)
    if sizes is None:
        return NOT_HANDLED
    if t._is_contiguous:
        return _view_of(t, sizes, _row_major_strides(sizes), t._offset)
    new_strides = _compute_view_strides(t._shape, t._strides, sizes)
    if new_strides is None:
        # PyTorch's reshape reads the TensorImpl's (fake-contiguous) strides
        # and routes copy-requiring reshapes here too — materialize.
        c = t._materialize_contiguous()
        return _view_of(c, sizes, _row_major_strides(sizes), 0)
    return _view_of(t, sizes, new_strides, t._offset)


@no_type_check
def fast_aten_view(tensor, *shape):
    return _fast_view(tensor, shape)


@no_type_check
def fast_aten__unsafe_view(tensor, *shape):
    return _fast_view(tensor, shape)


@no_type_check
def fast_aten_unsqueeze(tensor, dim):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    rank = len(t._shape)
    if not -rank - 1 <= dim <= rank:
        return NOT_HANDLED
    if dim < 0:
        dim += rank + 1
    new_shape = t._shape[:dim] + (1,) + t._shape[dim:]
    inserted = t._shape[dim] * t._strides[dim] if dim < rank else 1
    new_strides = t._strides[:dim] + (inserted,) + t._strides[dim:]
    return _view_of(t, new_shape, new_strides, t._offset)


@no_type_check
def fast_aten_squeeze_dim(tensor, dim):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    if t._shape[dim] != 1:
        return _view_of(t, t._shape, t._strides, t._offset)
    new_shape = t._shape[:dim] + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + t._strides[dim + 1 :]
    return _view_of(t, new_shape, new_strides, t._offset)


@no_type_check
def fast_aten_alias(tensor):
    t = _t(tensor)
    if t is None:
        return NOT_HANDLED
    return _view_of(t, t._shape, t._strides, t._offset)


fast_aten_detach = fast_aten_alias


@no_type_check
def fast_aten_permute(input, dims):
    t = _t(input)
    if t is None or not isinstance(dims, list | tuple):
        return NOT_HANDLED
    rank = len(t._shape)
    if len(dims) != rank:
        return NOT_HANDLED
    perm = [d % rank if -rank <= d < rank else None for d in dims]
    if None in perm or len(set(perm)) != rank:
        return NOT_HANDLED
    new_shape = tuple(t._shape[p] for p in perm)
    new_strides = tuple(t._strides[p] for p in perm)
    return _view_of(t, new_shape, new_strides, t._offset)


@no_type_check
def fast_aten_t(input):
    t = _t(input)
    if t is None or len(t._shape) > 2:
        return NOT_HANDLED
    if len(t._shape) < 2:
        return _view_of(t, t._shape, t._strides, t._offset)
    return fast_aten_transpose(input, 0, 1)


@no_type_check
def fast_aten_transpose(input, dim0, dim1):
    t = _t(input)
    if t is None or not isinstance(dim0, int) or not isinstance(dim1, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0:
        return _view_of(t, t._shape, t._strides, t._offset)
    if not (-rank <= dim0 < rank and -rank <= dim1 < rank):
        return NOT_HANDLED
    dim0 %= rank
    dim1 %= rank
    shape = list(t._shape)
    strides = list(t._strides)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    return _view_of(t, shape, strides, t._offset)


@no_type_check
def fast_aten_expand(tensor, sizes, *, implicit=False):
    t = _t(tensor)
    if t is None or not isinstance(sizes, list | tuple):
        return NOT_HANDLED
    rank = len(t._shape)
    new_rank = len(sizes)
    if new_rank < rank:
        return NOT_HANDLED
    pad = new_rank - rank
    new_shape = []
    new_strides = []
    for i, s in enumerate(sizes):
        j = i - pad
        if j < 0:
            if s == -1:
                return NOT_HANDLED
            new_shape.append(s)
            new_strides.append(0)
        else:
            old_size = t._shape[j]
            if s == -1 or s == old_size:
                new_shape.append(old_size)
                new_strides.append(t._strides[j])
            elif old_size == 1:
                new_shape.append(s)
                new_strides.append(0)
            else:
                return NOT_HANDLED
    return _view_of(t, new_shape, new_strides, t._offset)


@no_type_check
def fast_aten_slice(input, dim=0, start=None, end=None, step=1):
    t = _t(input)
    if t is None or not isinstance(dim, int) or not isinstance(step, int) or step < 1:
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
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
    length = -(-length // step)  # ceil div for step > 1
    new_shape = t._shape[:dim] + (length,) + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + (t._strides[dim] * step,) + t._strides[dim + 1 :]
    new_offset = t._offset + start * t._strides[dim]
    return _view_of(t, new_shape, new_strides, new_offset)


@no_type_check
def fast_aten_select(input, dim, index):
    t = _t(input)
    if t is None or not isinstance(dim, int) or not isinstance(index, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
    if index < 0:
        index += size
    if not 0 <= index < size:
        return NOT_HANDLED
    new_shape = t._shape[:dim] + t._shape[dim + 1 :]
    new_strides = t._strides[:dim] + t._strides[dim + 1 :]
    new_offset = t._offset + index * t._strides[dim]
    return _view_of(t, new_shape, new_strides, new_offset)


@no_type_check
def fast_aten_split(input, split_size, dim=0):
    t = _t(input)
    if t is None or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    size = t._shape[dim]
    if isinstance(split_size, int):
        if split_size <= 0:
            return NOT_HANDLED
        lengths = [
            min(split_size, size - start) for start in range(0, size, split_size)
        ]
        if not lengths:
            lengths = [0]
    else:
        lengths = list(split_size)
        if (
            not all(isinstance(x, int) and x >= 0 for x in lengths)
            or sum(lengths) != size
        ):
            return NOT_HANDLED
    results = []
    offset = 0
    for length in lengths:
        results.append(fast_aten_slice(t, dim, offset, offset + length))
        offset += length
    if any(r is NOT_HANDLED for r in results):
        return NOT_HANDLED
    return results


@no_type_check
def fast_aten_split_with_sizes(input, split_sizes, dim=0):
    return fast_aten_split(input, list(split_sizes), dim)


@no_type_check
def fast_aten_unbind(input, dim=0):
    t = _t(input)
    if t is None or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    results = [fast_aten_select(t, dim, i) for i in range(t._shape[dim])]
    if any(r is NOT_HANDLED for r in results):
        return NOT_HANDLED
    return results


@no_type_check
def fast_aten_clone(input, *, memory_format=None):
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    return t._materialize_contiguous()


# ---------------------------------------------------------------------------
# Concatenation along any dim: one destination-strided narrow copy per
# input into the output's slot for that input.
# ---------------------------------------------------------------------------


@no_type_check
def _is_legacy_empty(t) -> bool:
    x = _t(t)
    return x is not None and len(x._shape) == 1 and x._numel == 0


@no_type_check
def fast_aten_cat(tensors, dim=0):
    # PyTorch's cat skips legacy "empty" (1-D, size-0) tensors, e.g.
    # uninitialized KV-caches.
    real = [x for x in tensors if not _is_legacy_empty(x)]
    if not real or not isinstance(dim, int):
        return NOT_HANDLED
    ins = [_tc(x) for x in real]
    first = ins[0]
    if first is None or first._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    rank = len(first._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    for b in ins:
        if (
            b is None
            or b._dtype != first._dtype
            or b._device != first._device
            or len(b._shape) != rank
            or any(i != dim and b._shape[i] != first._shape[i] for i in range(rank))
        ):
            return NOT_HANDLED
    out_shape = list(first._shape)
    out_shape[dim] = sum(b._shape[dim] for b in ins)
    inner = math.prod(out_shape[dim + 1 :])
    outer = math.prod(out_shape[:dim])
    out = _alloc(out_shape, first._dtype, first._device)
    dst_stride = out_shape[dim] * inner
    ctx = _ctx_ptr(first._device)
    offset = 0
    for b in ins:
        copy_len = b._shape[dim] * inner
        if copy_len > 0 and outer > 0:
            eager_kernels.data_movement_ops.NarrowCopyDst(
                out._ptr,
                b._ptr,
                outer,
                dst_stride,
                copy_len,
                offset,
                out._itemsize,
                ctx,
            )
        offset += copy_len
    return out


# ---------------------------------------------------------------------------
# Movement tail: stack / repeat / tril / triu / index / scatter /
# select_scatter / nonzero.
# ---------------------------------------------------------------------------

# Max tensor rank the strided (rank-8-padded) kernels accept.
_MAX_RANK = 8

# Dtypes ScatterDim dispatches on (mirrors SCATTER_DTYPES in
# data_movement_ops.mojo): the scalar value overload casts through Float64.
_SCATTER_DTYPES = _FLOAT_DTYPES + (
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
)


@no_type_check
def fast_aten_stack(tensors, dim=0):
    # stack = unsqueeze each input at `dim`, then concatenate along `dim`.
    # Both helpers normalize `dim` against the SAME (rank + 1) base, so a raw
    # (possibly negative) `dim` stays consistent between them.
    if not isinstance(tensors, list | tuple) or len(tensors) == 0:
        return NOT_HANDLED
    if not isinstance(dim, int):
        return NOT_HANDLED
    unsqueezed = []
    for x in tensors:
        u = fast_aten_unsqueeze(x, dim)
        if u is NOT_HANDLED:
            return NOT_HANDLED
        unsqueezed.append(u)
    return fast_aten_cat(unsqueezed, dim)


@no_type_check
def fast_aten_repeat(input, repeats):
    t = _tc(input)
    if t is None or t._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    if not isinstance(repeats, list | tuple):
        return NOT_HANDLED
    repeats = list(repeats)
    if not all(isinstance(r, int) and r >= 0 for r in repeats):
        return NOT_HANDLED
    rank = len(t._shape)
    # torch left-pads the input shape with 1s when len(repeats) > rank; fewer
    # repeats than dims is invalid.
    if len(repeats) < rank or len(repeats) > _MAX_RANK:
        return NOT_HANDLED
    n_out = len(repeats)
    padded_shape = (1,) * (n_out - rank) + tuple(t._shape)
    out_shape = tuple(padded_shape[i] * repeats[i] for i in range(n_out))
    padded_strides = _row_major_strides(padded_shape)
    out = _alloc(out_shape, t._dtype, t._device)
    if out._numel > 0:
        eager_kernels.data_movement_ops.TileCopy(
            out._ptr,
            t._ptr,
            _pad8(out_shape, 1),
            _pad8(padded_shape, 1),
            _pad8(padded_strides, 0),
            out._itemsize,
            _ctx_ptr(t._device),
        )
    return out


@no_type_check
def _fast_triangular(input, diagonal, upper):
    if not isinstance(diagonal, int):
        return NOT_HANDLED
    t = _tc(input)
    if t is None or t._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    if len(t._shape) < 2:
        return NOT_HANDLED
    out = _alloc(t._shape, t._dtype, t._device)
    if out._numel > 0:
        rows = t._shape[-2]
        cols = t._shape[-1]
        batch = t._numel // (rows * cols)
        eager_kernels.data_movement_ops.TriangularCopy(
            out._ptr,
            t._ptr,
            batch,
            rows,
            cols,
            diagonal,
            upper,
            out._itemsize,
            _ctx_ptr(t._device),
        )
    return out


@no_type_check
def fast_aten_tril(input, diagonal=0):
    return _fast_triangular(input, diagonal, 0)


@no_type_check
def fast_aten_triu(input, diagonal=0):
    return _fast_triangular(input, diagonal, 1)


@no_type_check
def fast_aten_index(input, indices):
    t = _t(input)
    if t is None or not isinstance(indices, list | tuple):
        return NOT_HANDLED
    non_none = [(i, x) for i, x in enumerate(indices) if x is not None]
    # Only the single-index-on-dim-0 cases are handled fast.
    if len(non_none) != 1 or non_none[0][0] != 0:
        return NOT_HANDLED
    idx = _t(non_none[0][1])
    if idx is None or idx._device != t._device:
        return NOT_HANDLED

    if idx._dtype in (DType.int32, DType.int64):
        # Gather whole rows along dim 0.
        src = _tc(t)
        if src is None or src._dtype not in _COPYABLE_DTYPES or len(src._shape) < 1:
            return NOT_HANDLED
        idx_c = _tc(idx)
        row_len = 1
        for s in src._shape[1:]:
            row_len *= s
        out_shape = tuple(idx_c._shape) + tuple(src._shape[1:])
        out = _alloc(out_shape, src._dtype, src._device)
        if out._numel > 0:
            eager_kernels.data_movement_ops.GatherRows(
                out._ptr,
                src._ptr,
                idx_c._ptr,
                idx_c._dtype.value,
                idx_c._numel,
                row_len,
                src._shape[0],
                out._itemsize,
                _ctx_ptr(src._device),
            )
        return out

    if idx._dtype == DType.bool:
        # Boolean mask: data-dependent output shape -> host bounce (syncs).
        cpu_self = t._to_cpu_tensor()
        cpu_indices = tuple(
            slice(None) if x is None else _t(x)._to_cpu_tensor() for x in indices
        )
        result = cpu_self[cpu_indices]
        return TorchMaxTensor._from_cpu(result, t._device)

    return NOT_HANDLED


@no_type_check
def _fast_scatter(input, dim, index, src, value):
    a = _t(input)
    idx = _t(index)
    if a is None or idx is None or not isinstance(dim, int):
        return NOT_HANDLED
    if idx._device != a._device or idx._dtype != DType.int64:
        return NOT_HANDLED
    if a._dtype not in _SCATTER_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    if rank == 0 or rank > 4 or not -rank <= dim < rank:
        return NOT_HANDLED
    dim %= rank
    if len(idx._shape) != rank:
        return NOT_HANDLED

    out = a._materialize_contiguous()  # clone(self)
    idx_c = _tc(idx)

    is_value = 0
    value_f = 0.0
    src_ptr = out._ptr  # unused in value mode; a valid pointer for the kernel
    src_strides4 = [0, 0, 0, 0]
    if src is not None:
        s = _tc(src)
        if (
            s is None
            or s._dtype != a._dtype
            or s._device != a._device
            or len(s._shape) != rank
        ):
            return NOT_HANDLED
        src_ptr = s._ptr
        src_strides4 = [0] * (4 - rank) + list(_row_major_strides(s._shape))
    else:
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int | float):
            return NOT_HANDLED
        is_value = 1
        if a._dtype == DType.bool:
            value_f = 1.0 if value != 0 else 0.0
        else:
            value_f = float(value)

    dims4 = [1] * (4 - rank) + list(idx_c._shape)
    out_strides4 = [0] * (4 - rank) + list(out._strides)
    idx_strides4 = [0] * (4 - rank) + list(idx_c._strides)
    dim_padded = dim + (4 - rank)
    params = (
        tuple(dims4)
        + tuple(out_strides4)
        + tuple(src_strides4)
        + tuple(idx_strides4)
        + (dim_padded,)
    )
    if idx_c._numel > 0:
        eager_kernels.data_movement_ops.ScatterDim(
            out._ptr,
            idx_c._ptr,
            src_ptr,
            params,
            is_value,
            value_f,
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def fast_aten_scatter_src(input, dim, index, src):
    return _fast_scatter(input, dim, index, src, None)


@no_type_check
def fast_aten_scatter_value(input, dim, index, value):
    return _fast_scatter(input, dim, index, None, value)


@no_type_check
def fast_aten_select_scatter(input, src, dim, index):
    a = _t(input)
    s = _t(src)
    if a is None or s is None or not isinstance(dim, int) or not isinstance(index, int):
        return NOT_HANDLED
    if s._device != a._device:
        return NOT_HANDLED
    out = fast_aten_clone(a)
    if out is NOT_HANDLED:
        return NOT_HANDLED
    view = fast_aten_select(out, dim, index)
    if view is NOT_HANDLED:
        return NOT_HANDLED
    if s._dtype != out._dtype:
        s = _cast_tensor(s, out._dtype)
    if tuple(s._shape) != tuple(view._shape):
        s = fast_aten_expand(s, view._shape)
        if s is NOT_HANDLED:
            return NOT_HANDLED
    _copy_into(view, s)
    return out


@no_type_check
def fast_aten_nonzero(input):
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    # Data-dependent output shape -> host bounce (syncs). `.nonzero()` returns
    # an int64 (N, ndim) tensor in C-order over the input's coordinates.
    result = t._to_cpu_tensor().nonzero()
    return TorchMaxTensor._from_cpu(result, t._device)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@no_type_check
def _fast_batch_norm_inference(input, weight, bias, running_mean, running_var, eps):
    a = _tc(input)
    if a is None or a._dtype not in _FLOAT_DTYPES or len(a._shape) < 2:
        return NOT_HANDLED
    if a._numel == 0:
        return NOT_HANDLED
    params = [_tc(x) for x in (running_mean, running_var, weight, bias)]
    if any(p is None or p._dtype != a._dtype for p in params):
        return NOT_HANDLED
    mean_t, var_t, gamma_t, beta_t = params
    channels = a._shape[1]
    inner = math.prod(a._shape[2:])
    out = _alloc(a._shape, a._dtype, a._device)
    eager_kernels.nn_ops.BatchNormInference(
        out._ptr,
        a._ptr,
        mean_t._ptr,
        var_t._ptr,
        gamma_t._ptr,
        beta_t._ptr,
        (float(eps), channels, inner, a._numel),
        a._dtype.value,
        _ctx_ptr(a._device),
    )
    # Inference mode returns empty (0,) tensors for the saved stats.
    return (out, _alloc((0,), a._dtype, a._device), _alloc((0,), a._dtype, a._device))


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
    a = _tc(input)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(normalized_shape) == 1
        and len(a._shape) >= 1
        and a._shape[-1] == normalized_shape[0]
    ):
        gamma = _tc(weight)
        beta = _tc(bias)
        if (
            gamma is not None
            and beta is not None
            and gamma._dtype == a._dtype
            and beta._dtype == a._dtype
        ):
            cols = a._shape[-1]
            rows = a._numel // cols
            out = _alloc(a._shape, a._dtype, a._device)
            stat_shape = tuple(a._shape[:-1]) + (1,)
            mean = _alloc(stat_shape, DType.float32, a._device)
            rstd = _alloc(stat_shape, DType.float32, a._device)
            eager_kernels.nn_ops.LayerNorm(
                out._ptr,
                mean._ptr,
                rstd._ptr,
                a._ptr,
                gamma._ptr,
                beta._ptr,
                (float(eps), rows, cols),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out, mean, rstd
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


_ROW_REDUCE_DTYPES = _FLOAT_DTYPES + (DType.int64, DType.int32)

# Dtypes any()/all() accept as input (the nonzero test works for all of them;
# the output is always bool). Matches reduction_ops' AnyRows/AllRows dispatch.
_ANYALL_DTYPES = _FLOAT_DTYPES + (
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
)


@no_type_check
def _torch_dtype_to_max(dtype):
    from max.experimental.torch.torch import torch_dtype_to_max

    try:
        return torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        return None


@no_type_check
def _norm_reduce_dims(dim, rank, empty_is_all):
    """Sorted unique normalized reduce dims, or None if the spec is invalid.

    `dim=None` always reduces every dim. An empty dim list reduces every dim
    when `empty_is_all` (sum/amax/amin/mean/var semantics) and nothing when
    not (any.dims/all.dims semantics). Duplicate or out-of-range dims -> None.
    """
    if dim is None:
        return list(range(rank))
    if isinstance(dim, int) and not isinstance(dim, bool):
        dims = [dim]
    elif isinstance(dim, list | tuple):
        dims = list(dim)
    else:
        return None
    if len(dims) == 0:
        return list(range(rank)) if empty_is_all else []
    seen = set()
    out = []
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool):
            return None
        if not -rank <= d < rank:
            return None
        d %= rank
        if d in seen:
            return None
        seen.add(d)
        out.append(d)
    return sorted(out)


@no_type_check
def _reduce_to_rows(t, reduce_dims, keepdim):
    """Materialize `t` with the reduce dims moved to the trailing positions.

    Returns (contiguous row-major tensor, rows, cols, out_shape) where the
    kernel reduces each of the `rows` contiguous rows of `cols` elements to
    one output element, and `out_shape` already respects `keepdim`. The
    kept dims stay in ascending original order, so the `rows` output values
    are laid out exactly as `out_shape`'s row-major buffer.
    """
    rank = len(t._shape)
    rset = set(reduce_dims)
    kept = [i for i in range(rank) if i not in rset]
    if keepdim:
        out_shape = tuple(1 if i in rset else t._shape[i] for i in range(rank))
    else:
        out_shape = tuple(t._shape[i] for i in kept)
    cols = 1
    for i in reduce_dims:
        cols *= t._shape[i]
    rows = t._numel // cols if cols else t._numel
    if reduce_dims:
        contig = _tc(fast_aten_permute(t, kept + sorted(reduce_dims)))
    else:
        contig = _tc(t)
    return contig, rows, cols, out_shape


@no_type_check
def fast_aten_mean(input, dim=None, keepdim=False, *, dtype=None):
    a = _t(input)
    if a is None:
        return NOT_HANDLED
    if dtype is not None:
        target = _torch_dtype_to_max(dtype)
        if target is None or target not in _FLOAT_DTYPES:
            return NOT_HANDLED
        if a._dtype != target:
            if a._dtype not in _CAST_DTYPES or target not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, target)
    if a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, rdims, keepdim)
    if cols == 0:
        return NOT_HANDLED  # mean of an empty dim is nan; torch warns/errors
    out = _alloc(out_shape, a._dtype, a._device)
    if out._numel > 0:
        eager_kernels.nn_ops.MeanRows(
            out._ptr, contig._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device)
        )
    return out


@no_type_check
def fast_aten_sum(input, dim=None, keepdim=False, *, dtype=None):
    a = _t(input)
    if a is None:
        return NOT_HANDLED
    if dtype is not None:
        target = _torch_dtype_to_max(dtype)
        if target is None:
            return NOT_HANDLED
        if a._dtype != target:
            if a._dtype not in _CAST_DTYPES or target not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, target)
    elif not a._dtype.is_float():
        # torch promotes bool/integer sums to int64.
        if a._dtype != DType.int64:
            if a._dtype not in _CAST_DTYPES:
                return NOT_HANDLED
            a = _cast_tensor(a, DType.int64)
    if a._dtype not in (DType.float32, DType.float16, DType.bfloat16, DType.int64):
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, rdims, keepdim)
    if rows * cols == 0:
        # Empty reduction: torch defines sum over 0 elements as 0.
        filled = fast_filled(out_shape, 0, a._dtype, a._device)
        return NOT_HANDLED if filled is None else filled
    out = _alloc(out_shape, a._dtype, a._device)
    eager_kernels.reduction_ops.SumRows(
        out._ptr, contig._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def _amax_amin(input, dim, keepdim, kernel_name):
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, rdims, keepdim)
    if cols == 0:
        return NOT_HANDLED  # torch errors on amax/amin over an empty dim
    out = _alloc(out_shape, a._dtype, a._device)
    if out._numel > 0:
        getattr(eager_kernels.reduction_ops, kernel_name)(
            out._ptr, contig._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device)
        )
    return out


@no_type_check
def fast_aten_amax(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "MaxRowsR")


@no_type_check
def fast_aten_amin(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "MinRows")


@no_type_check
def fast_aten_min(input):
    # Values-only full reduction: aten::min(Tensor) -> Tensor.
    a = _tc(input)
    if a is None or a._numel == 0 or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    out = _alloc((), a._dtype, a._device)
    eager_kernels.reduction_ops.MinRows(
        out._ptr, a._ptr, 1, a._numel, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def fast_aten_min_dim(input, dim, keepdim=False):
    """aten::min.dim -> (values, indices) along `dim` (first-min-wins)."""
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(a._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, [dim % rank], keepdim)
    if cols == 0:
        return NOT_HANDLED
    values = _alloc(out_shape, a._dtype, a._device)
    indices = _alloc(out_shape, DType.int64, a._device)
    if rows > 0:
        eager_kernels.reduction_ops.MinMaxIdxRows(
            values._ptr,
            indices._ptr,
            contig._ptr,
            rows,
            cols,
            1,
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return values, indices


@no_type_check
def _argreduce(input, dim, keepdim, is_min):
    a = _t(input)
    if a is None or a._numel == 0 or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    if dim is None:
        contig = _tc(a)
        rows, cols = 1, a._numel
        out_shape = tuple([1] * rank) if keepdim else ()
    else:
        if not isinstance(dim, int) or rank == 0 or not -rank <= dim < rank:
            return NOT_HANDLED
        contig, rows, cols, out_shape = _reduce_to_rows(a, [dim % rank], keepdim)
        if cols == 0:
            return NOT_HANDLED
    out = _alloc(out_shape, DType.int64, a._device)
    if out._numel > 0:
        kernel = (
            eager_kernels.reduction_ops.ArgminRows
            if is_min
            else eager_kernels.nn_ops.ArgmaxRows
        )
        kernel(out._ptr, contig._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device))
    return out


@no_type_check
def fast_aten_argmax(input, dim=None, keepdim=False):
    return _argreduce(input, dim, keepdim, is_min=False)


@no_type_check
def fast_aten_argmin(input, dim=None, keepdim=False):
    return _argreduce(input, dim, keepdim, is_min=True)


@no_type_check
def fast_aten_max(input, *args, **kwargs):
    # Only the values-only overload max(Tensor) -> Tensor.
    if args or kwargs:
        return NOT_HANDLED
    a = _tc(input)
    if a is None or a._numel == 0 or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    out = _alloc((), a._dtype, a._device)
    eager_kernels.nn_ops.MaxRows(
        out._ptr, a._ptr, 1, a._numel, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def fast_aten_var(input, dim=None, *, correction=1, keepdim=False):
    a = _t(input)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    if correction is None:
        correction = 1
    if not isinstance(correction, int | float) or isinstance(correction, bool):
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, rdims, keepdim)
    if cols == 0:
        return NOT_HANDLED
    out = _alloc(out_shape, a._dtype, a._device)
    if out._numel > 0:
        eager_kernels.reduction_ops.VarRows(
            out._ptr,
            contig._ptr,
            rows,
            cols,
            float(correction),
            a._dtype.value,
            _ctx_ptr(a._device),
        )
    return out


@no_type_check
def _any_all(input, dim, keepdim, is_all):
    a = _t(input)
    if a is None or a._dtype not in _ANYALL_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    # Fast full-reduce bool path (the existing single-block AllBool/AnyBool).
    if dim is None and not keepdim and a._dtype == DType.bool:
        c = _tc(a)
        if 0 < c._numel < (1 << 22):
            out = _alloc((), DType.bool, a._device)
            fn = (
                eager_kernels.nn_ops.AllBool if is_all else eager_kernels.nn_ops.AnyBool
            )
            fn(out._ptr, c._ptr, c._numel, _ctx_ptr(a._device))
            return out
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=False)
    if rdims is None:
        return NOT_HANDLED
    contig, rows, cols, out_shape = _reduce_to_rows(a, rdims, keepdim)
    out = _alloc(out_shape, DType.bool, a._device)
    if out._numel > 0:
        # cols == 0 is valid: any -> False, all -> True (kernel handles it).
        fn = (
            eager_kernels.reduction_ops.AllRows
            if is_all
            else eager_kernels.reduction_ops.AnyRows
        )
        fn(out._ptr, contig._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device))
    return out


@no_type_check
def fast_aten_all(input, dim=None, keepdim=False):
    return _any_all(input, dim, keepdim, is_all=True)


@no_type_check
def fast_aten_any(input, dim=None, keepdim=False):
    return _any_all(input, dim, keepdim, is_all=False)


@no_type_check
def fast_aten__log_softmax(input, dim, half_to_float=False):
    t = _t(input)
    if (
        t is None
        or t._numel == 0
        or t._dtype not in _FLOAT_DTYPES
        or half_to_float
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0:
        return fast_filled((), 0.0, t._dtype, t._device)
    dim %= rank
    if dim != rank - 1:
        # log_softmax(x, d) = log_softmax(x.transpose(d, -1), -1).T; both
        # transposes are zero-copy, the inner one materializes once.
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__log_softmax(swapped, rank - 1, half_to_float)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    a = _tc(t)
    cols = a._shape[-1]
    rows = a._numel // cols
    out = _alloc(a._shape, a._dtype, a._device)
    eager_kernels.reduction_ops.LogSoftmaxRows(
        out._ptr, a._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def fast_aten_cumsum(input, dim, *, dtype=None):
    a = _tc(input) if dtype is None else None
    if (
        a is None
        or a._numel == 0
        or a._dtype not in (DType.int64, DType.int32, DType.float32)
    ):
        return NOT_HANDLED
    rank = len(a._shape)
    if not isinstance(dim, int) or rank == 0 or dim % rank != rank - 1:
        return NOT_HANDLED
    cols = a._shape[-1]
    rows = a._numel // cols
    out = _alloc(a._shape, a._dtype, a._device)
    eager_kernels.nn_ops.CumsumRows(
        out._ptr, a._ptr, rows, cols, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


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
    a = _tc(input)
    kernel = _pair(kernel_size)
    strides = _pair(stride) if stride not in (None, []) else kernel
    pads = _pair(padding)
    dils = _pair(dilation)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and not ceil_mode
        and None not in (kernel, strides, pads, dils)
    ):
        n, c, in_h, in_w = a._shape
        kh, kw = kernel
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            indices = _alloc((n, c, out_h, out_w), DType.int64, a._device)
            eager_kernels.nn_ops.MaxPool2dWithIndices(
                out._ptr,
                indices._ptr,
                a._ptr,
                (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, n * c),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out, indices
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# NN tail: average / adaptive-average pool, group norm, bilinear upsample, and
# the lowered SDPA variants (flash / efficient / math), all inference forward.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    a = _tc(input)
    kernel = _pair(kernel_size)
    strides = _pair(stride) if stride not in (None, []) else kernel
    pads = _pair(padding)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and not ceil_mode
        and None not in (kernel, strides, pads)
        and (
            divisor_override is None
            or (isinstance(divisor_override, int) and divisor_override != 0)
        )
    ):
        n, c, in_h, in_w = a._shape
        kh, kw = kernel
        sh, sw = strides
        ph, pw = pads
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            div = divisor_override if divisor_override is not None else 0
            eager_kernels.nn_ops.AvgPool2d(
                out._ptr,
                a._ptr,
                (
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    1 if count_include_pad else 0,
                    div,
                    n * c,
                ),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out
    return NOT_HANDLED


@no_type_check
def fast_aten__adaptive_avg_pool2d(input, output_size):
    a = _tc(input)
    osize = _pair(output_size)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and osize is not None
    ):
        n, c, in_h, in_w = a._shape
        out_h, out_w = osize
        if out_h > 0 and out_w > 0:
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            eager_kernels.nn_ops.AdaptiveAvgPool2d(
                out._ptr,
                a._ptr,
                (in_h, in_w, out_h, out_w, n * c),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out
    return NOT_HANDLED


@no_type_check
def fast_aten_native_group_norm(input, weight, bias, N, C, HxW, group, eps):
    a = _tc(input)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and isinstance(group, int)
        and group > 0
        and isinstance(C, int)
        and C % group == 0
        and a._numel == N * C * HxW
    ):
        cpg = C // group
        cols = cpg * HxW
        rows = N * group
        gamma = (
            _tc(weight)
            if weight is not None
            else fast_filled((C,), 1.0, a._dtype, a._device)
        )
        beta = (
            _tc(bias)
            if bias is not None
            else fast_filled((C,), 0.0, a._dtype, a._device)
        )
        if (
            gamma is None
            or beta is None
            or gamma._dtype != a._dtype
            or beta._dtype != a._dtype
            or tuple(gamma._shape) != (C,)
            or tuple(beta._shape) != (C,)
        ):
            return NOT_HANDLED
        out = _alloc(a._shape, a._dtype, a._device)
        mean = _alloc((N, group), DType.float32, a._device)
        rstd = _alloc((N, group), DType.float32, a._device)
        eager_kernels.nn_ops.GroupNorm(
            out._ptr,
            mean._ptr,
            rstd._ptr,
            a._ptr,
            gamma._ptr,
            beta._ptr,
            (float(eps), rows, cols, HxW, group, cpg),
            a._dtype.value,
            _ctx_ptr(a._device),
        )
        return out, mean, rstd
    return NOT_HANDLED


@no_type_check
def _area_pixel_scale(in_size, out_size, align_corners, scale):
    """torch area_pixel_compute_scale for one axis."""
    if align_corners:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    if scale is not None and scale > 0:
        return 1.0 / scale
    return in_size / out_size


@no_type_check
def fast_aten_upsample_bilinear2d(
    input, output_size, align_corners, scales_h=None, scales_w=None
):
    a = _tc(input)
    osize = _pair(output_size)
    if (
        a is not None
        and a._numel > 0
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and osize is not None
    ):
        n, c, in_h, in_w = a._shape
        out_h, out_w = osize
        if out_h > 0 and out_w > 0:
            ratio_h = _area_pixel_scale(in_h, out_h, align_corners, scales_h)
            ratio_w = _area_pixel_scale(in_w, out_w, align_corners, scales_w)
            out = _alloc((n, c, out_h, out_w), a._dtype, a._device)
            eager_kernels.nn_ops.UpsampleBilinear2d(
                out._ptr,
                a._ptr,
                (
                    float(ratio_h),
                    float(ratio_w),
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    n * c,
                    1 if align_corners else 0,
                ),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out
    return NOT_HANDLED


@no_type_check
def _sdpa_math_forward(query, key, value, is_causal, scale):
    """Decomposed bmm + scale/causal softmax + bmm; returns (out, probs) both
    as (B, H, Sq, *) views. Used by the math SDPA variant (needs the softmax
    probabilities as a second output). Returns NOT_HANDLED for shapes/dtypes
    the fast kernels don't cover."""
    q = _tc(query)
    k = _tc(key)
    v = _tc(value)
    if (
        q is None
        or k is None
        or v is None
        or q._dtype != k._dtype
        or q._dtype != v._dtype
        or q._dtype not in _FLOAT_DTYPES
        or len(q._shape) != 4
        or tuple(k._shape) != tuple(v._shape)
        or tuple(q._shape[:2]) != tuple(k._shape[:2])
        or q._shape[3] != k._shape[3]
        or 0 in q._shape
        or 0 in k._shape
    ):
        return NOT_HANDLED
    b, h, q_len, head_dim = q._shape
    kv_len = k._shape[2]
    scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    ctx = _ctx_ptr(q._device)
    dt = q._dtype.value
    scores = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
    eager_kernels.matmul_ops.Bmm(
        scores._ptr, q._ptr, k._ptr, (b * h, q_len, kv_len, head_dim, 1), dt, ctx
    )
    probs = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
    eager_kernels.nn_ops.SoftmaxRows(
        probs._ptr,
        scores._ptr,
        b * h * q_len,
        kv_len,
        float(scale_val),
        1 if is_causal else 0,
        q_len,
        dt,
        ctx,
    )
    out = _alloc((b * h, q_len, head_dim), q._dtype, q._device)
    eager_kernels.matmul_ops.Bmm(
        out._ptr, probs._ptr, v._ptr, (b * h, q_len, head_dim, kv_len, 0), dt, ctx
    )
    out_shape = (b, h, q_len, head_dim)
    out4 = _view_of(out, out_shape, _row_major_strides(out_shape), out._offset)
    probs_shape = (b, h, q_len, kv_len)
    probs4 = _view_of(
        probs, probs_shape, _row_major_strides(probs_shape), probs._offset
    )
    return out4, probs4


@no_type_check
def fast_aten__scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    *,
    scale=None,
    enable_gqa=False,
):
    if (
        dropout_p != 0.0
        or dropout_mask is not None
        or attn_mask is not None
        or enable_gqa
    ):
        return NOT_HANDLED
    result = _sdpa_math_forward(query, key, value, is_causal, scale)
    if result is NOT_HANDLED:
        return NOT_HANDLED
    out, probs = result
    return out, probs


@no_type_check
def fast_aten__scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    if dropout_p != 0.0:
        return NOT_HANDLED
    out = fast_aten_scaled_dot_product_attention(
        query, key, value, None, 0.0, is_causal, scale, False
    )
    if out is NOT_HANDLED:
        return NOT_HANDLED
    q = _t(query)
    b, h, sq, _ = q._shape
    sk = _t(key)._shape[2]
    dev = q._device
    # Auxiliary returns are only consumed by the backward pass; inference only
    # needs the primary output. logsumexp is (B, H, Sq) float32 (matching the
    # real kernel's shape/dtype); cum_seq_q/k are None as the dense CUDA path
    # returns; rng_state/unused/debug_attn_mask are zero placeholders.
    logsumexp = fast_filled((b, h, sq), 0.0, DType.float32, dev)
    rng_state = fast_filled((2,), 0, DType.int64, dev)
    unused = fast_filled((), 0, DType.int64, dev)
    debug_attn_mask = _alloc((0,), q._dtype, dev)
    return (out, logsumexp, None, None, sq, sk, rng_state, unused, debug_attn_mask)


@no_type_check
def fast_aten__scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias,
    compute_log_sumexp,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    if dropout_p != 0.0 or attn_bias is not None:
        return NOT_HANDLED
    out = fast_aten_scaled_dot_product_attention(
        query, key, value, None, 0.0, is_causal, scale, False
    )
    if out is NOT_HANDLED:
        return NOT_HANDLED
    q = _t(query)
    b, h, sq, _ = q._shape
    dev = q._device
    lse_len = sq if compute_log_sumexp else 0
    log_sumexp = fast_filled((b, h, lse_len), 0.0, DType.float32, dev)
    philox_seed = fast_filled((), 0, DType.int64, dev)
    philox_offset = fast_filled((), 0, DType.int64, dev)
    return (out, log_sumexp, philox_seed, philox_offset)


# ---------------------------------------------------------------------------
# Matmul family (GPU: pure-Mojo GEMM; CPU: correctness-grade loops)
# ---------------------------------------------------------------------------


@no_type_check
def _fast_matmul(a, b) -> TorchMaxTensor | None:
    """C = A @ B for 2D contiguous same-dtype float tensors."""
    if a is None or b is None:
        return None
    if a._device != b._device:
        return None
    if a._dtype != b._dtype or a._dtype not in _FLOAT_DTYPES:
        return None
    if len(a._shape) != 2 or len(b._shape) != 2:
        return None
    m, k = a._shape
    k2, n = b._shape
    if k != k2 or 0 in (m, n, k):
        return None
    out = _alloc((m, n), a._dtype, a._device)
    eager_kernels.matmul_ops.Matmul(
        out._ptr, a._ptr, b._ptr, (m, n, k, 0), a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def fast_aten_mm(x, y):
    out = _fast_matmul(_tc(x), _tc(y))
    if out is not None:
        return out
    return NOT_HANDLED


@no_type_check
def fast_aten_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    if beta == 1 and alpha == 1:
        bias = _tc(input)
        a = _tc(mat1)
        b = _tc(mat2)
        if (
            bias is not None
            and a is not None
            and b is not None
            and len(bias._shape) == 1
            and len(b._shape) == 2
            and bias._shape[0] == b._shape[1]
            and bias._dtype == a._dtype
            and a._device == b._device
            and a._dtype == b._dtype
            and a._dtype in _FLOAT_DTYPES
            and len(a._shape) == 2
            and a._shape[1] == b._shape[0]
            and 0 not in a._shape
            and 0 not in b._shape
        ):
            m, k = a._shape
            n = b._shape[1]
            out = _alloc((m, n), a._dtype, a._device)
            eager_kernels.matmul_ops.MatmulBias(
                out._ptr,
                a._ptr,
                b._ptr,
                bias._ptr,
                (m, n, k, 0),
                a._dtype.value,
                _ctx_ptr(a._device),
            )
            return out
    return NOT_HANDLED


@no_type_check
def fast_aten_linear(input, weight, bias=None):
    # linear(input, weight, bias) = input @ weight^T + bias. Registering the
    # composite op (instead of letting torch decompose it into t() + mm)
    # matters because the GEMM kernel reads B transposed for free, so the
    # weight is never materialized in transposed layout.
    a = _tc(input)
    w = _tc(weight)
    if (
        a is not None
        and w is not None
        and a._device == w._device
        and a._dtype == w._dtype
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) >= 2
        and len(w._shape) == 2
        and a._shape[-1] == w._shape[1]
        and 0 not in a._shape
        and 0 not in w._shape
    ):
        n, k = w._shape
        bias_t = None
        if bias is not None:
            bias_t = _tc(bias)
            if (
                bias_t is None
                or bias_t._dtype != a._dtype
                or tuple(bias_t._shape) != (n,)
            ):
                return NOT_HANDLED
        m = a._numel // k
        out_shape = tuple(a._shape[:-1]) + (n,)
        out = _alloc(out_shape, a._dtype, a._device)
        ctx = _ctx_ptr(a._device)
        if bias_t is not None:
            eager_kernels.matmul_ops.MatmulBias(
                out._ptr, a._ptr, w._ptr, bias_t._ptr, (m, n, k, 1), a._dtype.value, ctx
            )
        else:
            eager_kernels.matmul_ops.Matmul(
                out._ptr, a._ptr, w._ptr, (m, n, k, 1), a._dtype.value, ctx
            )
        return out
    return NOT_HANDLED


@no_type_check
def fast_aten_bmm(input, mat2):
    a = _tc(input)
    b = _tc(mat2)
    if (
        a is not None
        and b is not None
        and a._device == b._device
        and a._dtype == b._dtype
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 3
        and len(b._shape) == 3
        and a._shape[0] == b._shape[0]
        and a._shape[2] == b._shape[1]
        and 0 not in a._shape
        and 0 not in b._shape
    ):
        batch, m, k = a._shape
        n = b._shape[2]
        out = _alloc((batch, m, n), a._dtype, a._device)
        eager_kernels.matmul_ops.Bmm(
            out._ptr,
            a._ptr,
            b._ptr,
            (batch, m, n, k, 0),
            a._dtype.value,
            _ctx_ptr(a._device),
        )
        return out
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Convolution, pure Mojo: batched im2col + the pure GEMM with the torch
# (K,C,R,S) weight used as-is and NCHW output — no layout permutes.
# Grouped convolutions slice the channel-major im2col rows and weights per
# group with element offsets.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    a = _tc(input)
    w = _tc(weight)
    bias_t = _tc(bias) if bias is not None else None
    strides = _pair(list(stride))
    pads = _pair(list(padding))
    dils = _pair(list(dilation))
    if (
        a is not None
        and w is not None
        and not transposed
        and (bias is None or bias_t is not None)
        and a._dtype == w._dtype
        and a._dtype in _FLOAT_DTYPES
        and len(a._shape) == 4
        and len(w._shape) == 4
        and isinstance(groups, int)
        and groups >= 1
        and None not in (strides, pads, dils)
        and 0 not in a._shape
    ):
        n, c, in_h, in_w = a._shape
        out_c, c_per_group, kh, kw = w._shape
        sh, sw = strides
        ph, pw = pads
        dh, dw = dils
        out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
        if c_per_group * groups == c and out_h > 0 and out_w > 0:
            if bias_t is not None and (
                bias_t._dtype != a._dtype or tuple(bias_t._shape) != (out_c,)
            ):
                return NOT_HANDLED
            ctx = _ctx_ptr(a._device)
            cols = out_h * out_w
            ckk = c * kh * kw
            if (kh, kw, sh, sw, ph, pw, dh, dw) == (1, 1, 1, 1, 0, 0, 1, 1):
                # 1x1 stride-1 conv: NCHW input already is the col matrix.
                col_ptr = a._ptr
            else:
                col = _alloc((n, ckk, cols), a._dtype, a._device)
                eager_kernels.conv_ops.Im2col(
                    col._ptr,
                    a._ptr,
                    (in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, c, n),
                    a._dtype.value,
                    ctx,
                )
                col_ptr = col._ptr
            out = _alloc((n, out_c, cols), a._dtype, a._device)
            if groups == 1:
                eager_kernels.matmul_ops.Bmm(
                    out._ptr,
                    w._ptr,
                    col_ptr,
                    (n, out_c, cols, ckk, 0, 1),  # a_shared=1: broadcast weights
                    a._dtype.value,
                    ctx,
                )
            else:
                # Channel-major im2col rows make each group a contiguous
                # (crs_g, cols) slice; run one offset GEMM per (sample, group).
                crs_g = c_per_group * kh * kw
                oc_g = out_c // groups
                for s in range(n):
                    for g in range(groups):
                        eager_kernels.matmul_ops.Matmul(
                            out._ptr,
                            w._ptr,
                            col_ptr,
                            (
                                oc_g,
                                cols,
                                crs_g,
                                0,
                                (s * out_c + g * oc_g) * cols,
                                g * oc_g * crs_g,
                                (s * c + g * c_per_group) * kh * kw * cols,
                            ),
                            a._dtype.value,
                            ctx,
                        )
            if bias_t is not None:
                eager_kernels.conv_ops.BiasAddChan(
                    out._ptr,
                    bias_t._ptr,
                    (cols, out_c, n * out_c * cols),
                    a._dtype.value,
                    ctx,
                )
            return _view_of(
                out,
                (n, out_c, out_h, out_w),
                _row_major_strides((n, out_c, out_h, out_w)),
                out._offset,
            )
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Attention: decomposed scaled dot product attention (bmm + fused
# scale/causal softmax + bmm), with a fused single-kernel decode path.
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
    q = _tc(query)
    k = _tc(key)
    v = _tc(value)
    if (
        q is not None
        and k is not None
        and v is not None
        and attn_mask is None
        and dropout_p == 0.0
        and q._dtype == k._dtype == v._dtype
        and q._dtype in _FLOAT_DTYPES
        and len(q._shape) == 4
        and tuple(k._shape) == tuple(v._shape)
        and tuple(q._shape[:2]) == tuple(k._shape[:2])
        and q._shape[3] == k._shape[3]
        and 0 not in q._shape
        and 0 not in k._shape
    ):
        b, h, q_len, head_dim = q._shape
        kv_len = k._shape[2]
        scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        ctx = _ctx_ptr(q._device)
        dtype_val = q._dtype.value
        if (
            _on_gpu(q)
            and q_len == 1
            and not is_causal
            and head_dim % 4 == 0
            and head_dim <= 256
            and kv_len <= 4096
        ):
            # Decode step: one fused kernel instead of bmm+softmax+bmm
            # (single launch, no scratch buffers, coalesced K/V reads).
            out = _alloc((b, h, 1, head_dim), q._dtype, q._device)
            eager_kernels.nn_ops.AttnDecode(
                out._ptr,
                q._ptr,
                k._ptr,
                v._ptr,
                (b * h, kv_len, head_dim, float(scale_val)),
                dtype_val,
                ctx,
            )
            return out
        scores = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
        # scores = q @ k^T (transpose_b=1)
        eager_kernels.matmul_ops.Bmm(
            scores._ptr,
            q._ptr,
            k._ptr,
            (b * h, q_len, kv_len, head_dim, 1),
            dtype_val,
            ctx,
        )
        probs = _alloc((b * h, q_len, kv_len), q._dtype, q._device)
        eager_kernels.nn_ops.SoftmaxRows(
            probs._ptr,
            scores._ptr,
            b * h * q_len,
            kv_len,
            float(scale_val),
            1 if is_causal else 0,
            q_len,
            dtype_val,
            ctx,
        )
        out = _alloc((b * h, q_len, head_dim), q._dtype, q._device)
        eager_kernels.matmul_ops.Bmm(
            out._ptr,
            probs._ptr,
            v._ptr,
            (b * h, q_len, head_dim, kv_len, 0),
            dtype_val,
            ctx,
        )
        out_shape = (b, h, q_len, head_dim)
        return _view_of(out, out_shape, _row_major_strides(out_shape), out._offset)
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Softmax (the SDPA SoftmaxRows kernel with scale=1, no causal mask).
# Non-trailing dims go through a zero-copy transpose + materialize.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten__softmax(input, dim, half_to_float=False):
    t = _t(input)
    if (
        t is None
        or t._numel == 0
        or t._dtype not in _FLOAT_DTYPES
        or half_to_float
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    rank = len(t._shape)
    if rank == 0:
        return fast_filled((), 1.0, t._dtype, t._device)
    dim %= rank
    if dim != rank - 1:
        # softmax(x, d) = softmax(x.transpose(d, -1), -1).transpose(d, -1);
        # both transposes are zero-copy, the inner one materializes once.
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__softmax(swapped, rank - 1, half_to_float)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    a = _tc(t)
    cols = a._shape[-1]
    rows = a._numel // cols
    out = _alloc(a._shape, a._dtype, a._device)
    eager_kernels.nn_ops.SoftmaxRows(
        out._ptr, a._ptr, rows, cols, 1.0, 0, 1, a._dtype.value, _ctx_ptr(a._device)
    )
    return out


@no_type_check
def fast_aten_softmax(input, dim=-1, dtype=None):
    if dtype is not None:
        return NOT_HANDLED
    return fast_aten__softmax(input, dim, False)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_embedding(
    input, weight, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    # `input` is the weight table, `weight` the indices (aten naming).
    table = _tc(input)
    idx = _tc(weight)
    if (
        table is not None
        and idx is not None
        and table._dtype in _FLOAT_DTYPES
        and idx._dtype in (DType.int32, DType.int64)
        and len(table._shape) == 2
        and idx._numel > 0
    ):
        row_len = table._shape[1]
        out_shape = tuple(idx._shape) + (row_len,)
        out = _alloc(out_shape, table._dtype, table._device)
        eager_kernels.nn_ops.Gather0(
            out._ptr,
            table._ptr,
            idx._ptr,
            idx._dtype.value,
            idx._numel,
            row_len,
            table._dtype.value,
            _ctx_ptr(table._device),
        )
        return out
    return NOT_HANDLED


# ---------------------------------------------------------------------------
# Filled factories (full / ones / zeros / scalar_tensor): one allocation
# plus one Fill kernel. The registrations resolve torch dtype/device.
# ---------------------------------------------------------------------------


@no_type_check
def fast_filled(shape, value, dtype: DType, device):
    """A TorchMaxTensor of `shape` filled with `value`, or None."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int | float):
        return None
    if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
        return None
    if dtype not in _FILL_DTYPES:
        return None
    out = _alloc(tuple(shape), dtype, device)
    if out._numel > 0:
        eager_kernels.elementwise_ops.Fill(
            out._ptr, float(value), out._numel, dtype.value, _ctx_ptr(device)
        )
    return out


# ---------------------------------------------------------------------------
# Scalar readback
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten__local_scalar_dense(tensor):
    t = _t(tensor)
    if t is None or t._numel != 1:
        return NOT_HANDLED
    return eager_kernels.tensor_holder.read_scalar(
        _ctx_ptr(t._device), t._ptr, t._dtype.value
    )


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
