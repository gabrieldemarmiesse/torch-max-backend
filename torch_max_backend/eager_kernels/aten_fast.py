"""ATen-signature-compatible fast implementations for max_device eager mode.

Each function here has the same signature as its counterpart in
`aten_functions.py` and is registered in `max_device_aten_ops.py` *instead
of* it when the fast eager path is enabled. When the inputs qualify
(realized, contiguous, same shape/dtype/device), the op runs as a single
Mojo kernel call — no graph building, no MLIR passes, no interpreter.
Anything else falls back to the regular `aten_functions` implementation,
so behavior is unchanged for the cases the fast path doesn't cover yet
(broadcasting, scalars, mixed dtypes, alpha != 1, ...).

Only the eager (max_device) path uses this module; the torch.compile
backend keeps using `aten_functions` directly.
"""

from max.dtype import DType
from max.experimental.tensor import Tensor as MaxEagerTensor

from torch_max_backend import aten_functions, eager_kernels
from torch_max_backend.eager_kernels import elementwise_ops
from torch_max_backend.types import MaxTensor, Scalar

# The Mojo kernels raise (instead of falling back) on dtypes they don't
# support; gate float-only ops here so those inputs take the regular path.
_FLOAT_DTYPES = (DType.float16, DType.bfloat16, DType.float32)


def _is_float_tensor(x) -> bool:
    return isinstance(x, MaxEagerTensor) and x.dtype in _FLOAT_DTYPES


def _try_binary(mojo_fn, lhs, rhs) -> MaxEagerTensor | None:
    if not (isinstance(lhs, MaxEagerTensor) and isinstance(rhs, MaxEagerTensor)):
        return None
    try:
        return eager_kernels.binary_op(mojo_fn, lhs, rhs)
    except eager_kernels.FastPathUnavailable:
        return None


def _try_unary(mojo_fn, x) -> MaxEagerTensor | None:
    if not isinstance(x, MaxEagerTensor):
        return None
    try:
        return eager_kernels.unary_op(mojo_fn, x)
    except eager_kernels.FastPathUnavailable:
        return None


def fast_aten_add(
    input: MaxTensor, other: MaxTensor | Scalar, alpha: Scalar = 1
) -> MaxTensor:
    if alpha == 1:
        result = _try_binary(elementwise_ops.Add, input, other)
        if result is not None:
            return result
    return aten_functions.aten_add(input, other, alpha)


def fast_aten_sub(
    input: MaxTensor | int | float, other: MaxTensor | Scalar, alpha: Scalar = 1
) -> MaxTensor:
    if alpha == 1:
        result = _try_binary(elementwise_ops.Sub, input, other)
        if result is not None:
            return result
    return aten_functions.aten_sub(input, other, alpha)


def fast_aten_mul(input: MaxTensor, other: MaxTensor | Scalar) -> MaxTensor:
    result = _try_binary(elementwise_ops.Mul, input, other)
    if result is not None:
        return result
    return aten_functions.aten_mul(input, other)


def fast_aten_div(
    input: MaxTensor, other: MaxTensor | Scalar, *, rounding_mode: str | None = None
) -> MaxTensor:
    if rounding_mode is None and _is_float_tensor(input):
        result = _try_binary(elementwise_ops.Div, input, other)
        if result is not None:
            return result
    return aten_functions.aten_div(input, other, rounding_mode=rounding_mode)


def fast_aten_maximum(x: MaxTensor, y: MaxTensor) -> MaxTensor:
    result = _try_binary(elementwise_ops.Max, x, y)
    if result is not None:
        return result
    return aten_functions.aten_maximum(x, y)


def fast_aten_minimum(x: MaxTensor, y: MaxTensor) -> MaxTensor:
    result = _try_binary(elementwise_ops.Min, x, y)
    if result is not None:
        return result
    return aten_functions.aten_minimum(x, y)


def fast_aten_relu(tensor: MaxTensor) -> MaxTensor:
    result = _try_unary(elementwise_ops.Relu, tensor)
    if result is not None:
        return result
    return aten_functions.aten_relu(tensor)


def fast_aten_exp(input: MaxTensor) -> MaxTensor:
    if _is_float_tensor(input):
        result = _try_unary(elementwise_ops.Exp, input)
        if result is not None:
            return result
    return aten_functions.aten_exp(input)
