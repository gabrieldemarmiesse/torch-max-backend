"""ATen-signature-compatible fast implementations for max_device eager mode.

Each function here is registered in `max_device_aten_ops.py` (through
`_eager_impl`). Tensors are `TorchMojoTensor`s: a Mojo `TensorHolder`
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
import weakref
from typing import no_type_check

from max.dtype import DType

from torch_max_backend import eager_kernels, is_running_tests
from torch_max_backend.eager_kernels import _ctx_ptr
from torch_max_backend.max_device.torch_max_tensor import (
    TorchMojoTensor,
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


# Hugging Face GPT-2 spells its tanh GELU out as seven eager ATen ops:
#
#   0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))
#
# On a large decode batch, dispatching and allocating those intermediates is
# substantially slower than the single GeluTanhSpec kernel.  The small state
# machine below delays only that exact, gfx942 FP32, rank-3/width-3072 chain.
# A value consumed by anything other than the next expected operation is
# materialized through the ordinary kernels, so this remains a transparent
# eager optimization rather than a model-specific API requirement.
_GELU_POW3 = 1
_GELU_CUBE_SCALE = 2
_GELU_INNER = 3
_GELU_INNER_SCALE = 4
_GELU_TANH = 5
_GELU_TANH_PLUS_ONE = 6
_GELU_HALF_X = 7
_GELU_CUBE_COEFF = 0.044715
_GELU_INNER_COEFF = math.sqrt(2.0 / math.pi)
_materializing_deferred_gelu = False


@no_type_check
def _deferred_gelu_marker(x):
    if isinstance(x, TorchMojoTensor):
        return x.__dict__.get("_deferred_gpt2_gelu")
    return None


@no_type_check
def _looks_like_gpt2_gelu_input(x):
    return (
        isinstance(x, TorchMojoTensor)
        and _deferred_gelu_marker(x) is None
        and x._dtype == DType.float32
        and len(x._shape) == 3
        and x._shape[-1] == 3072
        and x._device.architecture_name == "gfx942"
    )


@no_type_check
def _deferred_gelu_view(base, stage):
    out = TorchMojoTensor._view_of(
        base, base._shape, base._strides, base._offset, base._is_contiguous
    )
    out._deferred_gpt2_gelu = (stage, base)
    return out


@no_type_check
def _materialize_deferred_gelu(x):
    cached = x.__dict__.get("_deferred_gpt2_gelu_value")
    if cached is not None:
        return cached

    stage, base = x._deferred_gpt2_gelu
    global _materializing_deferred_gelu
    old = _materializing_deferred_gelu
    _materializing_deferred_gelu = True
    try:
        half_x = fast_aten_mul(base, 0.5)
        pow3 = fast_aten_pow(base, 3.0)
        cube_scale = fast_aten_mul(pow3, _GELU_CUBE_COEFF)
        inner = fast_aten_add(base, cube_scale)
        inner_scale = fast_aten_mul(inner, _GELU_INNER_COEFF)
        tanh_value = fast_aten_tanh(inner_scale)
        tanh_plus_one = fast_aten_add(tanh_value, 1.0)
        values = {
            _GELU_POW3: pow3,
            _GELU_CUBE_SCALE: cube_scale,
            _GELU_INNER: inner,
            _GELU_INNER_SCALE: inner_scale,
            _GELU_TANH: tanh_value,
            _GELU_TANH_PLUS_ONE: tanh_plus_one,
            _GELU_HALF_X: half_x,
        }
        result = values[stage]
    finally:
        _materializing_deferred_gelu = old
    x._deferred_gpt2_gelu_value = result
    return result


@no_type_check
def _t(x) -> TorchMojoTensor | None:
    """x as a TorchMojoTensor (any layout), or None."""
    if not isinstance(x, TorchMojoTensor):
        return None
    if _deferred_gelu_marker(x) is not None:
        return _materialize_deferred_gelu(x)
    return x


@no_type_check
def _tc(x) -> TorchMojoTensor | None:
    """x as a *contiguous* TorchMojoTensor (materializing views), or None."""
    x = _t(x)
    if x is None:
        return None
    return x if x._is_contiguous else x._materialize_contiguous()


_alloc = TorchMojoTensor._alloc
_view_of = TorchMojoTensor._view_of

# GPT-2's Conv1D weights are stored KxN and reach eager mode through addmm,
# while the fast gfx942 MFMA schedule consumes the transposed NxK layout.
# Cache that one-time materialization by tensor identity/version so decode
# steps use the faster transposed-B kernel without recopying model weights.
_GFX942_GPT2_WEIGHT_T_CACHE = {}
_GFX942_GPT2_WEIGHT_SHAPES = {(768, 768), (768, 2304), (768, 3072), (3072, 768)}


@no_type_check
def _gfx942_cached_gpt2_weight_t(weight):
    key = id(weight)
    version = weight._version
    cached = _GFX942_GPT2_WEIGHT_T_CACHE.get(key)
    if cached is not None:
        source_ref, cached_version, transposed = cached
        if source_ref() is weight and cached_version == version:
            return transposed

    transposed_view = _view_of(
        weight,
        (weight._shape[1], weight._shape[0]),
        (weight._strides[1], weight._strides[0]),
        weight._offset,
    )
    transposed = transposed_view._materialize_contiguous()

    def remove(_source_ref, *, cache_key=key):
        _GFX942_GPT2_WEIGHT_T_CACHE.pop(cache_key, None)

    _GFX942_GPT2_WEIGHT_T_CACHE[key] = (
        weakref.ref(weight, remove),
        version,
        transposed,
    )
    return transposed


# ---------------------------------------------------------------------------
# TensorSpec fast paths (docs/tensor_spec_design.md). A spec is the
# Mojo-side descriptor of a tensor's layout; spec ops live next to their
# kernels (binary/comparison in logic_ops, relu/batch_norm still in
# tensor_holder) and do checks, broadcast, output alloc and kernel launch
# in one boundary call, raising a real NotImplementedError when the inputs
# don't qualify — the callers below treat any exception as "take the
# classic path". make_spec (the constructor) stays in tensor_holder.
# ---------------------------------------------------------------------------


@no_type_check
def _spec_of(t):
    """t's cached Mojo TensorSpec, built on first use."""
    spec = t.__dict__.get("_spec")
    if spec is None:
        spec = eager_kernels.tensor_holder.make_spec(
            t._ptr,
            len(t._shape),
            _pad8(t._shape, 1),
            _pad8(t._strides, 0),
            t._offset,
            t._dtype.value,
            t._itemsize,
            t._numel,
            1 if t._is_contiguous else 0,
            _ctx_ptr(t._device),
        )
        t._spec = spec
    return spec


@no_type_check
def _wrap_spec_result(result, dtype, device):
    """Mint the torch wrapper for a spec op's (holder, spec, shape, ptr)."""
    holder, spec, shape, ptr = result
    out = TorchMojoTensor._make(
        holder, ptr, shape, _row_major_strides(shape), 0, dtype, device, contiguous=True
    )
    out._spec = spec
    return out


@no_type_check
def _try_spec_binary(spec_fn_name, lhs, rhs, out_dtype=None):
    """Broadcast binary through a logic_ops spec op, or None.

    Python keeps what Python must do — scalar embedding, torch's promotion
    rules, dim-spec sanity — but the intermediates stay spec-to-spec: a
    scalar operand becomes a 0-d FillSpec result and a promoted operand a
    CastSpec result whose SPECS feed the binary entry directly. No wrapper
    is minted for them; their holders live in locals until the launch is
    enqueued (the stream-ordered free then lands after the kernel).
    rank>4 operands are pre-materialized (the spec's flat path needs
    contiguity there). `out_dtype` overrides the wrapper dtype for ops
    whose output differs (comparisons -> bool)."""
    a = _t(lhs)
    b = _t(rhs)
    spec_a = spec_b = None
    keep_a = keep_b = None
    if a is not None and b is not None:
        if a._device != b._device:
            return None
        device = a._device
        dtype = a._dtype
        if len(a._shape) > 4 or len(b._shape) > 4:
            a = _tc(a)
            b = _tc(b)
        if a._dtype != b._dtype:
            # torch's promotion rules (the only pairs the loops hit).
            if a._dtype == DType.bool and b._dtype in _CAST_DTYPES:
                cast_a, dtype = True, b._dtype
            elif b._dtype == DType.bool and a._dtype in _CAST_DTYPES:
                cast_a, dtype = False, a._dtype
            elif a._dtype == DType.int32 and b._dtype == DType.int64:
                cast_a, dtype = True, DType.int64
            elif a._dtype == DType.int64 and b._dtype == DType.int32:
                cast_a, dtype = False, DType.int64
            else:
                return None
            try:
                if cast_a:
                    keep_a, spec_a, _, _ = eager_kernels.data_movement_ops.CastSpec(
                        _spec_of(a), dtype.value
                    )
                else:
                    keep_b, spec_b, _, _ = eager_kernels.data_movement_ops.CastSpec(
                        _spec_of(b), dtype.value
                    )
            except Exception:
                return None
    elif a is not None:
        device = a._device
        dtype = a._dtype
        value = _scalar_embed(rhs, dtype)
        if value is None:
            return None
        try:
            keep_b, spec_b, _, _ = eager_kernels.elementwise_ops.FillSpec(
                _pad8((), 1), 0, 1, value, dtype.value, _ctx_ptr(device)
            )
        except Exception:
            return None
    elif b is not None:
        # Scalar-first calls, e.g. rsub-style `1 - tensor`.
        device = b._device
        dtype = b._dtype
        value = _scalar_embed(lhs, dtype)
        if value is None:
            return None
        try:
            keep_a, spec_a, _, _ = eager_kernels.elementwise_ops.FillSpec(
                _pad8((), 1), 0, 1, value, dtype.value, _ctx_ptr(device)
            )
        except Exception:
            return None
    else:
        return None
    try:
        result = getattr(eager_kernels.logic_ops, spec_fn_name)(
            spec_a if spec_a is not None else _spec_of(a),
            spec_b if spec_b is not None else _spec_of(b),
        )
    except Exception:
        return None
    _ = keep_a, keep_b  # intermediates must outlive the enqueued launch
    return _wrap_spec_result(result, out_dtype or dtype, device)


@no_type_check
def _try_spec_unary(spec_fn_name, x, out_dtype=None, module_name="elementwise_ops"):
    """Contiguous unary through a spec op, or None.

    `out_dtype` overrides the wrapper dtype for the bool-output ops
    (isnan / logical_not)."""
    a = _t(x)
    if a is None:
        return None
    try:
        result = getattr(getattr(eager_kernels, module_name), spec_fn_name)(_spec_of(a))
    except Exception:
        return None
    return _wrap_spec_result(result, out_dtype or a._dtype, a._device)


@no_type_check
def _try_spec_reduce(
    spec_fn_name, a, rdims, keepdim, *extra, out_dtype=None, module_name="reduction_ops"
):
    """Trailing-dims reduction through a spec op, or None. `a` is already a
    TorchMojoTensor (dtype promotion happened upstream); the spec op raises
    on non-trailing dims / strided input and the classic path takes over."""
    try:
        result = getattr(getattr(eager_kernels, module_name), spec_fn_name)(
            _spec_of(a), tuple(rdims), 1 if keepdim else 0, *extra
        )
    except Exception:
        return None
    return _wrap_spec_result(result, out_dtype or a._dtype, a._device)


@no_type_check
def _wrap_spec_pair(result, dtype0, dtype1, device):
    """Mint two torch wrappers from a two-group spec result."""
    return (
        _wrap_spec_result(result[0], dtype0, device),
        _wrap_spec_result(result[1], dtype1, device),
    )


@no_type_check
def _try_spec_matmul(spec_fn_name, tensors, transpose_b):
    """Matmul-family spec op over already-typed operands, or None. The spec
    raises on non-contiguous operands; the classic path materializes them."""
    ts = [_t(x) for x in tensors]
    if any(t is None for t in ts):
        return None
    try:
        result = getattr(eager_kernels.matmul_ops, spec_fn_name)(
            *[_spec_of(t) for t in ts], transpose_b
        )
    except Exception:
        return None
    return _wrap_spec_result(result, ts[0]._dtype, ts[0]._device)


@no_type_check
def _try_spec_scalar(spec_fn_name, x, scalar):
    """Contiguous tensor-with-float-scalar through a spec op, or None."""
    if not isinstance(scalar, int | float) or isinstance(scalar, bool):
        return None
    a = _t(x)
    if a is None:
        return None
    try:
        result = getattr(eager_kernels.elementwise_ops, spec_fn_name)(
            _spec_of(a), float(scalar)
        )
    except Exception:
        return None
    return _wrap_spec_result(result, a._dtype, a._device)


@no_type_check
def _try_spec_int_scalar(spec_fn_name, x, scalar):
    """Contiguous tensor-with-int-scalar through a spec op, or None."""
    if not isinstance(scalar, int) or isinstance(scalar, bool):
        return None
    a = _t(x)
    if a is None:
        return None
    try:
        result = getattr(eager_kernels.elementwise_ops, spec_fn_name)(
            _spec_of(a), scalar
        )
    except Exception:
        return None
    return _wrap_spec_result(result, a._dtype, a._device)


@no_type_check
def _on_gpu(t: TorchMojoTensor) -> bool:
    return t._device.label == "gpu"


@no_type_check
def _copy_into(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
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
def _scalar_tensor_0d(value, dtype, device) -> TorchMojoTensor:
    """A 0-d tensor holding `value`, for stride-0 broadcast operands."""
    result = eager_kernels.elementwise_ops.FillSpec(
        _pad8((), 1), 0, 1, float(value), dtype.value, _ctx_ptr(device)
    )
    return _wrap_spec_result(result, dtype, device)


@no_type_check
def _cast_tensor(x: TorchMojoTensor, dtype: DType) -> TorchMojoTensor:
    """Dtype cast through CastSpec (strided inputs materialize Mojo-side).

    Callers pre-gate on _CAST_DTYPES; anything else propagates the spec's
    NotImplementedError (the classic kernel silently wrote garbage there)."""
    result = eager_kernels.data_movement_ops.CastSpec(_spec_of(_t(x)), dtype.value)
    return _wrap_spec_result(result, dtype, x._device)


@no_type_check
def _promoted_pair(a: TorchMojoTensor, b: TorchMojoTensor):
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
def _scalar_embed(value, dtype: DType) -> float | None:
    """`value` validated for lossless embedding into `dtype`, as a float,
    or None."""
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
    return float(value)


@no_type_check
def _resolve_scalar(value, dtype: DType, device) -> TorchMojoTensor | None:
    """A 0-d stride-0 tensor holding a Python scalar in `dtype`, or None
    when the value doesn't embed losslessly."""
    v = _scalar_embed(value, dtype)
    if v is None:
        return None
    return _scalar_tensor_0d(v, dtype, device)


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
    scaled = _try_spec_scalar("MulScalarSpec", b, alpha)
    if scaled is None:
        scaled = _try_spec_int_scalar("MulScalarIntSpec", b, alpha)
    return scaled


@no_type_check
def _matches_float(value, expected):
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    )


@no_type_check
def fast_aten_add(input, other, alpha=1):
    if not _materializing_deferred_gelu and alpha == 1:
        input_marker = _deferred_gelu_marker(input)
        other_marker = _deferred_gelu_marker(other)
        for marker, plain in ((input_marker, other), (other_marker, input)):
            if marker is None:
                continue
            stage, base = marker
            if stage == _GELU_CUBE_SCALE and plain is base:
                return _deferred_gelu_view(base, _GELU_INNER)
            if stage == _GELU_TANH and _matches_float(plain, 1.0):
                return _deferred_gelu_view(base, _GELU_TANH_PLUS_ONE)

    if alpha != 1:
        other = _scaled_operand(other, alpha)
        if other is None:
            return NOT_HANDLED
    result = _try_spec_scalar("AddScalarSpec", input, other)
    if result is None:
        result = _try_spec_int_scalar("AddScalarIntSpec", input, other)
    if result is None:
        result = _try_spec_binary("AddSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


_fast_aten_add_default = fast_aten_add


@no_type_check
def fast_aten_add_apple(input, other, alpha=1):
    """Metal specialization for equal-shape contiguous tensor addition."""
    a = _t(input)
    b = _t(other)
    if (
        alpha == 1
        and a is not None
        and b is not None
        and a._device.api == "metal"
        and a._device == b._device
        and a._is_contiguous
        and b._is_contiguous
        and a._dtype == b._dtype
        and a._dtype in _FLOAT_DTYPES
        and a._shape == b._shape
    ):
        out = _alloc(a._shape, a._dtype, a._device)
        if a._numel > 0:
            eager_kernels.elementwise_ops.Add(
                out._ptr, a._ptr, b._ptr, a._numel, a._dtype.value, _ctx_ptr(a._device)
            )
        return out
    return _fast_aten_add_default(input, other, alpha)


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
    result = None
    if isinstance(other, int | float) and not isinstance(other, bool):
        # sub-by-scalar reuses the AddScalar spec with a negated scalar.
        result = _try_spec_scalar("AddScalarSpec", input, -other)
        if result is None and isinstance(other, int):
            result = _try_spec_int_scalar("AddScalarIntSpec", input, -other)
    if result is None:
        result = _try_spec_binary("SubSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_mul(input, other):
    if not _materializing_deferred_gelu:
        input_marker = _deferred_gelu_marker(input)
        other_marker = _deferred_gelu_marker(other)
        if input_marker is not None and other_marker is not None:
            input_stage, input_base = input_marker
            other_stage, other_base = other_marker
            if input_base is other_base and {input_stage, other_stage} == {
                _GELU_HALF_X,
                _GELU_TANH_PLUS_ONE,
            }:
                return fast_aten_gelu(input_base, approximate="tanh")

        for marker, scalar in ((input_marker, other), (other_marker, input)):
            if marker is None:
                continue
            stage, base = marker
            if stage == _GELU_POW3 and _matches_float(scalar, _GELU_CUBE_COEFF):
                return _deferred_gelu_view(base, _GELU_CUBE_SCALE)
            if stage == _GELU_INNER and _matches_float(scalar, _GELU_INNER_COEFF):
                return _deferred_gelu_view(base, _GELU_INNER_SCALE)

        if isinstance(input, TorchMojoTensor) and _matches_float(other, 0.5):
            if _looks_like_gpt2_gelu_input(input):
                return _deferred_gelu_view(input, _GELU_HALF_X)
        if isinstance(other, TorchMojoTensor) and _matches_float(input, 0.5):
            if _looks_like_gpt2_gelu_input(other):
                return _deferred_gelu_view(other, _GELU_HALF_X)

    result = _try_spec_scalar("MulScalarSpec", input, other)
    if result is None:
        result = _try_spec_int_scalar("MulScalarIntSpec", input, other)
    if result is None:
        result = _try_spec_binary("MulSpec", input, other)
    if result is not None:
        return result
    return NOT_HANDLED


@no_type_check
def fast_aten_div(input, other, *, rounding_mode=None):
    if rounding_mode is not None:
        return NOT_HANDLED
    a = _t(input)
    if a is not None and a._dtype in _FLOAT_DTYPES:
        result = _try_spec_binary("DivSpec", input, other)
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
    result = fast_filled(a._shape, value, a._dtype, a._device)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_fill__scalar(input, value):
    """In-place fill of input (any strides). Returns None when unavailable."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    a = _t(input)
    if a is None:
        return None
    if a._dtype == DType.float64 and a._device.api == "metal":
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
    result = _try_spec_binary("MaximumSpec", x, y)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_minimum(x, y):
    result = _try_spec_binary("MinimumSpec", x, y)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_relu(tensor):
    return _unary_spec_op("ReluSpec", tensor)


@no_type_check
def fast_aten_exp(input):
    return _unary_spec_op("ExpSpec", input)


@no_type_check
def fast_aten_tanh(x):
    if not _materializing_deferred_gelu:
        marker = _deferred_gelu_marker(x)
        if marker is not None:
            stage, base = marker
            if stage == _GELU_INNER_SCALE:
                return _deferred_gelu_view(base, _GELU_TANH)
    return _unary_spec_op("TanhSpec", x)


@no_type_check
def fast_aten_pow(x, y):
    if (
        not _materializing_deferred_gelu
        and _matches_float(y, 3.0)
        and _looks_like_gpt2_gelu_input(x)
    ):
        return _deferred_gelu_view(x, _GELU_POW3)
    result = _try_spec_scalar("PowScalarSpec", x, y)
    return result if result is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Unary elementwise suite
#
# Spec-only (no classic fallback, design doc §2.4): the spec entries cover
# the full classic gates — float64 included (CPU device; the kernels
# comptime-refuse f64 on GPU), strided inputs via Mojo-side temporaries,
# int dtypes for the direct ops, bool via uint8 for the bool-output ops.
# ---------------------------------------------------------------------------


@no_type_check
def _unary_spec_op(spec, x):
    """Float-only unary with no classic fallback: the spec entry provably
    covers the classic gate (_FLOAT_DTYPES, strided via Mojo-side
    temporaries), so the classic chain was deleted (design doc §2.4)."""
    result = _try_spec_unary(spec, x)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_abs(x):
    return _unary_spec_op("AbsSpec", x)


@no_type_check
def fast_aten_neg(x):
    return _unary_spec_op("NegSpec", x)


@no_type_check
def fast_aten_sign(x):
    return _unary_spec_op("SignSpec", x)


@no_type_check
def _int_unary_identity(x):
    """ceil/floor on integer tensors is the identity in torch; return a copy."""
    a = _t(x)
    if a is not None and a._dtype in _BITWISE_DTYPES and a._dtype != DType.bool:
        return a._materialize_contiguous()
    return None


@no_type_check
def fast_aten_ceil(x):
    result = _int_unary_identity(x)
    if result is not None:
        return result
    return _unary_spec_op("CeilSpec", x)


@no_type_check
def fast_aten_floor(x):
    result = _int_unary_identity(x)
    if result is not None:
        return result
    return _unary_spec_op("FloorSpec", x)


@no_type_check
def fast_aten_acos(x):
    return _unary_spec_op("AcosSpec", x)


@no_type_check
def fast_aten_asinh(x):
    return _unary_spec_op("AsinhSpec", x)


@no_type_check
def fast_aten_atanh(x):
    return _unary_spec_op("AtanhSpec", x)


@no_type_check
def fast_aten_cos(x):
    return _unary_spec_op("CosSpec", x)


@no_type_check
def fast_aten_cosh(x):
    return _unary_spec_op("CoshSpec", x)


@no_type_check
def fast_aten_erf(x):
    return _unary_spec_op("ErfSpec", x)


@no_type_check
def fast_aten_log(x):
    return _unary_spec_op("LogSpec", x)


@no_type_check
def fast_aten_log1p(x):
    return _unary_spec_op("Log1pSpec", x)


@no_type_check
def fast_aten_reciprocal(x):
    return _unary_spec_op("ReciprocalSpec", x)


@no_type_check
def fast_aten_rsqrt(x):
    return _unary_spec_op("RsqrtSpec", x)


@no_type_check
def fast_aten_sigmoid(x):
    return _unary_spec_op("SigmoidSpec", x)


@no_type_check
def fast_aten_silu(x):
    return _unary_spec_op("SiluSpec", x)


@no_type_check
def fast_aten_sin(x):
    return _unary_spec_op("SinSpec", x)


@no_type_check
def fast_aten_sinh(x):
    return _unary_spec_op("SinhSpec", x)


@no_type_check
def fast_aten_sqrt(x):
    return _unary_spec_op("SqrtSpec", x)


@no_type_check
def fast_aten_tan(x):
    return _unary_spec_op("TanSpec", x)


@no_type_check
def fast_aten_gelu(input, approximate="none"):
    if approximate == "none":
        spec = "GeluNoneSpec"
    elif approximate == "tanh":
        spec = "GeluTanhSpec"
    else:
        return NOT_HANDLED
    return _unary_spec_op(spec, input)


@no_type_check
def fast_aten_isnan(x):
    result = _try_spec_unary("IsNanSpec", x, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_logical_not(x):
    result = _try_spec_unary("LogicalNotSpec", x, DType.bool)
    return result if result is not None else NOT_HANDLED


# ---------------------------------------------------------------------------
# Comparisons and bitwise/logic ops (broadcast-strided kernels). These are
# the generation-loop bookkeeping ops: stopping criteria, attention-mask
# prep, position ids.
# ---------------------------------------------------------------------------


@no_type_check
def fast_aten_eq(input, other):
    result = _try_spec_binary("EqSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_ne(input, other):
    result = _try_spec_binary("NeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_lt(input, other):
    result = _try_spec_binary("LtSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_le(input, other):
    result = _try_spec_binary("LeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_gt(input, other):
    result = _try_spec_binary("GtSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_ge(input, other):
    result = _try_spec_binary("GeSpec", input, other, DType.bool)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_and(input, other):
    result = _try_spec_binary("BitwiseAndSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_or(input, other):
    result = _try_spec_binary("BitwiseOrSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_xor(input, other):
    result = _try_spec_binary("BitwiseXorSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_bitwise_not(input):
    a = _tc(input)
    if a is None or a._dtype not in _BITWISE_DTYPES:
        return NOT_HANDLED
    # bitwise_not on bool is logical negation (0<->1), NOT a byte complement
    # (~0 == 255 in uint8), so route bool through the logical-not path.
    if a._dtype == DType.bool:
        return fast_aten_logical_not(a)
    out = _alloc(a._shape, a._dtype, a._device)
    if out._numel > 0:
        eager_kernels.logic_ops.BitwiseNot(
            out._ptr, a._ptr, out._numel, a._dtype.value, _ctx_ptr(a._device)
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
    if el._numel > 0 and te._numel == 0:
        # x in {} is always False (True under invert).
        return fast_filled(el._shape, 1.0 if invert else 0.0, DType.bool, el._device)
    out = _alloc(el._shape, DType.bool, el._device)
    if el._numel > 0:
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
    result = _try_spec_binary("RemainderSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_floor_divide(input, other):
    # floor(input / other), float and int dtypes.
    result = _try_spec_binary("FloorDivSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_pow_tensor_tensor(input, exponent):
    # Float-only (the kernel raises on ints, which would leave the output
    # unwritten); gate here so unsupported dtypes fall through cleanly.
    a = _t(input)
    if a is None or a._dtype not in _FLOAT_DTYPES:
        return NOT_HANDLED
    result = _try_spec_binary("PowSpec", input, exponent)
    return result if result is not None else NOT_HANDLED


@no_type_check
def _try_logical(spec_fn_name, input, other):
    """Mixed-dtype logical_and / logical_xor: reduce each operand to bool
    (the nonzero test) spec-to-spec via CastSpec, then the bool spec path
    (uint8 dispatch). Same-dtype pairs already went through the spec
    directly; no wrappers are minted for the bool intermediates."""
    a = _t(input)
    b = _t(other)
    if a is None or b is None or a._device != b._device:
        return None
    if a._dtype not in _CAST_DTYPES or b._dtype not in _CAST_DTYPES:
        return None
    keep_a = keep_b = None
    try:
        if a._dtype == DType.bool:
            spec_a = _spec_of(a)
        else:
            keep_a, spec_a, _, _ = eager_kernels.data_movement_ops.CastSpec(
                _spec_of(a), DType.bool.value
            )
        if b._dtype == DType.bool:
            spec_b = _spec_of(b)
        else:
            keep_b, spec_b, _, _ = eager_kernels.data_movement_ops.CastSpec(
                _spec_of(b), DType.bool.value
            )
        result = getattr(eager_kernels.logic_ops, spec_fn_name)(spec_a, spec_b)
    except Exception:
        return None
    _ = keep_a, keep_b  # intermediates must outlive the enqueued launch
    return _wrap_spec_result(result, DType.bool, a._device)


@no_type_check
def fast_aten_logical_and(input, other):
    result = _try_spec_binary("LogicalAndSpec", input, other, DType.bool)
    if result is None:
        result = _try_logical("LogicalAndSpec", input, other)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_logical_xor(input, other):
    result = _try_spec_binary("LogicalXorSpec", input, other, DType.bool)
    if result is None:
        result = _try_logical("LogicalXorSpec", input, other)
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
def _binary_operands(input, other):
    """Resolve (lhs, rhs) TorchMojoTensors with equal dtypes for the ternary
    broadcast kernels (where / masked_fill). Either operand may be a Python
    scalar (which becomes a 0-d stride-0 tensor of the tensor operand's
    dtype), or None if unresolvable."""
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
        scalar = _resolve_scalar(input, b._dtype, b._device)
        return None if scalar is None else (scalar, b)
    return None


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
        return _view_of(t, sizes, _row_major_strides(sizes), t._offset, contiguous=True)
    new_strides = _compute_view_strides(t._shape, t._strides, sizes)
    if new_strides is None:
        # PyTorch's reshape reads the TensorImpl's (fake-contiguous) strides
        # and routes copy-requiring reshapes here too — materialize.
        c = t._materialize_contiguous()
        return _view_of(c, sizes, _row_major_strides(sizes), 0, contiguous=True)
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
    if t._is_contiguous:
        # Straight D2D memcpy; the strided gather kernel below costs ~2x
        # the bandwidth (per-element index math on 50k-column logits).
        out = _alloc(t._shape, t._dtype, t._device)
        _copy_into(out, t)
        return out
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
def _is_gpt2_kv_append(first, second, dim):
    return (
        first is not None
        and second is not None
        and dim == 2
        and first._dtype == second._dtype == DType.float32
        and first._device == second._device
        and first._device.architecture_name == "gfx942"
        and len(first._shape) == len(second._shape) == 4
        and first._shape[0] == second._shape[0]
        and first._shape[0] >= 64
        and first._shape[1] == second._shape[1] == 12
        and first._shape[3] == second._shape[3] == 64
        and second._shape[2] == 1
    )


@no_type_check
def _gpt2_kv_append(first, second):
    """Append one GPT-2 K/V token into a branch-safe reserved allocation.

    The previous logical tensor is never overwritten. Only bytes beyond its
    shape are touched, and consuming its append token invalidates that token;
    reusing an older cache branch therefore allocates/copies instead of
    clobbering the newer branch.
    """
    old_len = first._shape[2]
    new_len = old_len + 1
    capacity = first.__dict__.get("_gpt2_kv_capacity", 0)
    appendable = first.__dict__.get("_gpt2_kv_appendable", False)

    if appendable and new_len <= capacity:
        first._gpt2_kv_appendable = False
        out_shape = first._shape[:2] + (new_len,) + first._shape[3:]
        out = _view_of(
            first, out_shape, first._strides, first._offset, contiguous=False
        )
        out._gpt2_kv_capacity = capacity
        out._gpt2_kv_appendable = True
        slot_offset = out._offset + old_len * out._strides[2]
        slot = _view_of(out, second._shape, out._strides, slot_offset)
        _copy_strided_into(slot, second)
        return out

    # Reserve through the normal 200-token benchmark on the first append;
    # double after that for longer generation without quadratic recopying.
    capacity = max(256, capacity * 2, new_len)
    storage_shape = first._shape[:2] + (capacity,) + first._shape[3:]
    storage = _alloc(storage_shape, first._dtype, first._device)
    storage_strides = storage._strides
    out_shape = first._shape[:2] + (new_len,) + first._shape[3:]
    out = _view_of(storage, out_shape, storage_strides, 0, contiguous=False)
    old_slot = _view_of(out, first._shape, out._strides, 0)
    new_slot = _view_of(out, second._shape, out._strides, old_len * out._strides[2])
    _copy_strided_into(old_slot, first)
    _copy_strided_into(new_slot, second)
    out._gpt2_kv_capacity = capacity
    out._gpt2_kv_appendable = True
    return out


@no_type_check
def fast_aten_cat(tensors, dim=0):
    # PyTorch's cat skips legacy "empty" (1-D, size-0) tensors, e.g.
    # uninitialized KV-caches.
    real = [x for x in tensors if not _is_legacy_empty(x)]
    if not real or not isinstance(dim, int):
        return NOT_HANDLED
    ins = [_t(x) for x in real]
    first = ins[0]
    if first is None or first._dtype not in _COPYABLE_DTYPES:
        return NOT_HANDLED
    rank = len(first._shape)
    if rank == 0 or not -rank <= dim < rank or rank > _MAX_RANK:
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
    if len(ins) == 2 and _is_gpt2_kv_append(ins[0], ins[1], dim):
        return _gpt2_kv_append(ins[0], ins[1])
    out_shape = list(first._shape)
    out_shape[dim] = sum(b._shape[dim] for b in ins)
    inner = math.prod(out_shape[dim + 1 :])
    outer = math.prod(out_shape[:dim])
    out = _alloc(out_shape, first._dtype, first._device)
    dst_stride = out_shape[dim] * inner
    ctx = _ctx_ptr(first._device)
    if (
        len(ins) == 2
        and ins[0]._is_contiguous
        and ins[1]._is_contiguous
        and outer > 0
        and inner > 0
    ):
        # Both source rows contiguous: one fused kernel fills each output
        # row from both inputs (no second launch, no bubble between the
        # two copies). Raises (e.g. row lengths not vector-aligned, outer
        # over the grid cap, CPU device) -> per-input copy loop below.
        len1 = ins[0]._shape[dim] * inner
        len2 = ins[1]._shape[dim] * inner
        if len1 > 0 and len2 > 0:
            try:
                eager_kernels.data_movement_ops.Cat2(
                    out._ptr,
                    ins[0]._ptr,
                    ins[1]._ptr,
                    outer,
                    len1,
                    len2,
                    out._itemsize,
                    ctx,
                )
                return out
            except NotImplementedError:
                pass
    offset = 0
    for b in ins:
        copy_len = b._shape[dim] * inner
        if copy_len > 0 and outer > 0:
            if b._is_contiguous:
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
            else:
                # Strided input (e.g. the new-token K/V head-transpose in a
                # KV-cache append): gather it straight into its output slot
                # instead of materializing a contiguous copy first. The slot
                # is out[..., pos:pos+len, ...]: same strides as out, offset
                # pos * out_strides[dim] == the accumulated flat offset.
                slot = _view_of(out, b._shape, out._strides, offset)
                _copy_strided_into(slot, b)
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
        return TorchMojoTensor._from_cpu(result, t._device)

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
    if a._dtype == DType.float64 and a._device.api == "metal":
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
    return TorchMojoTensor._from_cpu(result, t._device)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@no_type_check
def _fast_batch_norm_inference(input, weight, bias, running_mean, running_var, eps):
    a = _t(input)
    stats = [_t(x) for x in (running_mean, running_var, weight, bias)]
    if a is not None and all(s is not None for s in stats):
        try:
            result = eager_kernels.nn_ops.BatchNormSpec(
                _spec_of(a),
                _spec_of(stats[0]),
                _spec_of(stats[1]),
                _spec_of(stats[2]),
                _spec_of(stats[3]),
                float(eps),
            )
        except Exception:
            pass
        else:
            out = _wrap_spec_result(result, a._dtype, a._device)
            # Inference mode returns empty (0,) tensors for the saved stats.
            return (
                out,
                _alloc((0,), a._dtype, a._device),
                _alloc((0,), a._dtype, a._device),
            )
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
    k = len(normalized_shape)
    if (
        a is None
        or a._numel == 0
        or a._dtype not in _FLOAT_DTYPES
        or k < 1
        or len(a._shape) < k
        or tuple(a._shape[-k:]) != tuple(normalized_shape)
    ):
        return NOT_HANDLED
    cols = 1
    for s in normalized_shape:
        cols *= s
    rows = a._numel // cols
    # weight/bias are optional (no-affine layer norm): default to 1s / 0s.
    if weight is not None:
        gamma = _tc(weight)
        if gamma is None or gamma._dtype != a._dtype or gamma._numel != cols:
            return NOT_HANDLED
    else:
        gamma = fast_filled((cols,), 1.0, a._dtype, a._device)
    if bias is not None:
        beta = _tc(bias)
        if beta is None or beta._dtype != a._dtype or beta._numel != cols:
            return NOT_HANDLED
    else:
        beta = fast_filled((cols,), 0.0, a._dtype, a._device)
    # Not spec-converted: the classic prologue here is already thin, and
    # building three (holder, spec, shape, ptr) result groups measurably
    # costs more than the removed Python work (+4us/call measured A/B on
    # (6, 768); contrast min.dim, whose heavy classic prologue makes the
    # two-group spec a -34% win).
    out = _alloc(a._shape, a._dtype, a._device)
    stat_shape = tuple(a._shape[:-k]) + (1,) * k
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
def _reduced_shape(shape, reduce_dims, keepdim):
    """The reduction output shape (keepdim already applied)."""
    rset = set(reduce_dims)
    if keepdim:
        return tuple(1 if i in rset else s for i, s in enumerate(shape))
    return tuple(s for i, s in enumerate(shape) if i not in rset)


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
    result = _try_spec_reduce("MeanSpec", a, rdims, keepdim, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


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
    result = _try_spec_reduce("SumSpec", a, rdims, keepdim)
    if result is not None:
        return result
    if a._numel == 0:
        # Empty reduction (the spec raises): torch defines it as 0.
        out_shape = _reduced_shape(a._shape, rdims, keepdim)
        filled = fast_filled(out_shape, 0, a._dtype, a._device)
        return NOT_HANDLED if filled is None else filled
    return NOT_HANDLED


@no_type_check
def _amax_amin(input, dim, keepdim, spec_name):
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = _norm_reduce_dims(dim, rank, empty_is_all=True)
    if rdims is None:
        return NOT_HANDLED
    result = _try_spec_reduce(spec_name, a, rdims, keepdim)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_amax(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "AmaxSpec")


@no_type_check
def fast_aten_amin(input, dim=(), keepdim=False):
    return _amax_amin(input, dim, keepdim, "AminSpec")


@no_type_check
def fast_aten_min(input):
    # Values-only full reduction: aten::min(Tensor) -> Tensor.
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    result = _try_spec_reduce("AminSpec", t, range(len(t._shape)), False)
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_min_dim(input, dim, keepdim=False):
    """aten::min.dim -> (values, indices) along `dim` (first-min-wins)."""
    a = _t(input)
    if a is None or a._dtype not in _ROW_REDUCE_DTYPES or not isinstance(dim, int):
        return NOT_HANDLED
    rank = len(a._shape)
    if rank == 0 or not -rank <= dim < rank:
        return NOT_HANDLED
    try:
        result = eager_kernels.reduction_ops.MinDimSpec(
            _spec_of(a), (dim % rank,), 1 if keepdim else 0
        )
    except Exception:
        result = None
    if result is not None:
        return _wrap_spec_pair(result, a._dtype, DType.int64, a._device)
    return NOT_HANDLED


@no_type_check
def _argreduce(input, dim, keepdim, is_min):
    a = _t(input)
    if a is None or a._numel == 0 or a._dtype not in _ROW_REDUCE_DTYPES:
        return NOT_HANDLED
    rank = len(a._shape)
    rdims = list(range(rank)) if dim is None else None
    if rdims is None and isinstance(dim, int) and rank > 0 and -rank <= dim < rank:
        rdims = [dim % rank]
    if rdims is not None:
        spec_name = "ArgminSpec" if is_min else "ArgmaxSpec"
        module_name = "reduction_ops" if is_min else "nn_ops"
        result = _try_spec_reduce(
            spec_name, a, rdims, keepdim, out_dtype=DType.int64, module_name=module_name
        )
        if result is not None:
            return result
    return NOT_HANDLED


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
    t = _t(input)
    if t is None:
        return NOT_HANDLED
    result = _try_spec_reduce(
        "MaxSpec", t, range(len(t._shape)), False, module_name="nn_ops"
    )
    return result if result is not None else NOT_HANDLED


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
    result = _try_spec_reduce("VarSpec", a, rdims, keepdim, float(correction))
    return result if result is not None else NOT_HANDLED


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
    result = _try_spec_reduce(
        "AllSpec" if is_all else "AnySpec", a, rdims, keepdim, out_dtype=DType.bool
    )
    return result if result is not None else NOT_HANDLED


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
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    # half_to_float: half input, float32 output (torch computes in fp32).
    if half_to_float:
        if t._dtype not in (DType.float16, DType.bfloat16):
            return NOT_HANDLED
        t = _cast_tensor(t, DType.float32)
    rank = len(t._shape)
    if rank == 0:
        return fast_filled((), 0.0, t._dtype, t._device)
    dim %= rank
    if dim != rank - 1:
        # log_softmax(x, d) = log_softmax(x.transpose(d, -1), -1).T; both
        # transposes are zero-copy, the inner one materializes once.
        # (t is already fp32 here if half_to_float was set, so pass False.)
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__log_softmax(swapped, rank - 1, False)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    result = _try_spec_unary("LogSoftmaxSpec", t, module_name="reduction_ops")
    return result if result is not None else NOT_HANDLED


@no_type_check
def fast_aten_cumsum(input, dim, *, dtype=None):
    a = _t(input) if dtype is None else None
    if (
        a is None
        or a._numel == 0
        or a._dtype not in (DType.int64, DType.int32, DType.float32)
    ):
        return NOT_HANDLED
    rank = len(a._shape)
    if not isinstance(dim, int) or rank == 0 or dim % rank != rank - 1:
        return NOT_HANDLED
    result = _try_spec_unary("CumsumSpec", a, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


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
        # Not spec-converted for the same reason as native_layer_norm
        # above (three-group result construction outweighs the thin
        # classic prologue).
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
    out4 = _view_of(
        out, out_shape, _row_major_strides(out_shape), out._offset, contiguous=True
    )
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
def fast_aten_mm(x, y):
    out = _try_spec_matmul("MatmulSpec", (x, y), 0)
    return out if out is not None else NOT_HANDLED


@no_type_check
def fast_aten_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    # beta/alpha scaling isn't implemented by the fast path (falls through).
    if beta == 1 and alpha == 1:
        a = _t(mat1)
        weight = _t(mat2)
        bias = _t(input)
        if (
            a is not None
            and weight is not None
            and bias is not None
            and a._dtype == weight._dtype == bias._dtype == DType.float32
            and len(a._shape) == len(weight._shape) == 2
            and a._shape[0] >= 64
            and tuple(weight._shape) in _GFX942_GPT2_WEIGHT_SHAPES
            and tuple(bias._shape) == (weight._shape[1],)
            and a._shape[1] == weight._shape[0]
            and a._device.architecture_name == "gfx942"
        ):
            weight_t = _gfx942_cached_gpt2_weight_t(weight)
            out = _try_spec_matmul("MatmulBiasSpec", (a, weight_t, bias), 1)
            if out is not None:
                return out

        out = _try_spec_matmul("MatmulBiasSpec", (mat1, mat2, input), 0)
        if out is not None:
            return out
    return NOT_HANDLED


@no_type_check
def fast_aten_linear(input, weight, bias=None):
    # linear(input, weight, bias) = input @ weight^T + bias. Registering the
    # composite op (instead of letting torch decompose it into t() + mm)
    # matters because the GEMM kernel reads B transposed for free, so the
    # weight is never materialized in transposed layout.
    if bias is None:
        out = _try_spec_matmul("MatmulSpec", (input, weight), 1)
    else:
        out = _try_spec_matmul("MatmulBiasSpec", (input, weight, bias), 1)
    return out if out is not None else NOT_HANDLED


@no_type_check
def fast_aten_bmm(input, mat2):
    out = _try_spec_matmul("BmmSpec", (input, mat2), 0)
    return out if out is not None else NOT_HANDLED


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
    q = _t(query)
    k = _t(key)
    v = _t(value)
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
            and q._strides[3] == 1
            and k._strides[3] == v._strides[3] == 1
            and k._strides[2] == v._strides[2] == head_dim
        ):
            # Decode step: one fused kernel instead of bmm+softmax+bmm
            # (single launch, no scratch buffers, coalesced K/V reads).
            # q reads through its (batch, head) strides, so the per-head
            # transpose view of the fused qkv projection is NOT
            # materialized first.
            try:
                result = eager_kernels.nn_ops.AttnDecodeSpec(
                    _spec_of(q), _spec_of(k), _spec_of(v), float(scale_val)
                )
            except Exception:
                result = None
            if result is not None:
                return _wrap_spec_result(result, q._dtype, q._device)
            out = _alloc((b, h, 1, head_dim), q._dtype, q._device)
            eager_kernels.nn_ops.AttnDecode(
                out._ptr,
                q._ptr,
                k._ptr,
                v._ptr,
                (
                    b * h,
                    kv_len,
                    head_dim,
                    float(scale_val),
                    h,
                    q._strides[0],
                    q._strides[1],
                    k._strides[0],
                    k._strides[1],
                    k._strides[2],
                    v._strides[0],
                    v._strides[1],
                    v._strides[2],
                ),
                dtype_val,
                ctx,
            )
            return out
        k = k if k._is_contiguous else k._materialize_contiguous()
        v = v if v._is_contiguous else v._materialize_contiguous()
        q = q if q._is_contiguous else q._materialize_contiguous()
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
        return _view_of(
            out, out_shape, _row_major_strides(out_shape), out._offset, contiguous=True
        )
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
        or not isinstance(dim, int)
    ):
        return NOT_HANDLED
    # half_to_float: half input, float32 output (torch computes in fp32).
    if half_to_float:
        if t._dtype not in (DType.float16, DType.bfloat16):
            return NOT_HANDLED
        t = _cast_tensor(t, DType.float32)
    rank = len(t._shape)
    if rank == 0:
        return fast_filled((), 1.0, t._dtype, t._device)
    dim %= rank
    if dim != rank - 1:
        # softmax(x, d) = softmax(x.transpose(d, -1), -1).transpose(d, -1);
        # both transposes are zero-copy, the inner one materializes once.
        # (t is already fp32 here if half_to_float was set, so pass False.)
        swapped = fast_aten_transpose(t, dim, rank - 1)
        result = fast_aten__softmax(swapped, rank - 1, False)
        if result is NOT_HANDLED:
            return NOT_HANDLED
        return fast_aten_transpose(result, dim, rank - 1)
    result = _try_spec_unary("SoftmaxSpec", t, module_name="nn_ops")
    return result if result is not None else NOT_HANDLED


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
    """A TorchMojoTensor of `shape` filled with `value`, or None."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int | float):
        return None
    if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
        return None
    if dtype not in _FILL_DTYPES:
        return None
    if dtype == DType.float64 and device.api == "metal":
        return None
    shape = tuple(shape)
    result = eager_kernels.elementwise_ops.FillSpec(
        _pad8(shape, 1),
        len(shape),
        math.prod(shape),
        float(value),
        dtype.value,
        _ctx_ptr(device),
    )
    return _wrap_spec_result(result, dtype, device)


# What the Arange kernel dispatches on (_FILL_DTYPES minus bool, which
# torch.arange rejects anyway).
_ARANGE_DTYPES = _FILL_DTYPES[:-1]


@no_type_check
def fast_arange(numel, start, step, dtype: DType, device):
    """A 1-D TorchMojoTensor holding start + i*step (device kernel), or None.

    The caller resolves torch's arange semantics (numel, dtype). Inputs arrive
    through Float64, while the kernel uses PyTorch's device-specific
    accumulator type; integers beyond the exact-f64 range must not reach here.
    """
    if dtype not in _ARANGE_DTYPES:
        return None
    if dtype == DType.float64 and device.api == "metal":
        return None
    out = _alloc((numel,), dtype, device)
    if numel > 0:
        eager_kernels.elementwise_ops.Arange(
            out._ptr, float(start), float(step), numel, dtype.value, _ctx_ptr(device)
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
