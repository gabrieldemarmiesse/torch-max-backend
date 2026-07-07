import math
from collections.abc import Callable
from typing import Any, no_type_check

import max.driver
import numpy as np
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor as MaxEagerTensor
from max.experimental.torch.torch import torch_dtype_to_max

from torch_max_backend import aten_functions
from torch_max_backend.flags import fast_eager_enabled
from torch_max_backend.max_device.torch_max_tensor import (
    TorchMaxTensor,
    find_equivalent_max_device,
)

# Global registry for functions to register
_aten_ops_registry: list[tuple[str, Callable]] = []

# The flag is env-var driven and constant for the process lifetime;
# per-call ops (view, factories) are hot enough that re-reading os.environ
# on each call shows up in profiles.
_FAST_EAGER = fast_eager_enabled()


def register_aten_op(op_name: str):
    """Decorator to mark a function for aten op registration.

    Args:
        op_name: The aten operation name (e.g., "aten::add.Tensor")

    Usage:
        @register_aten_op("aten::add.Tensor")
        def max_device_aten_add(input, other, alpha=1):
            return execute_with_max_graph(aten.add, (input, other, alpha), {})
    """

    def decorator(func: Callable) -> Callable:
        _aten_ops_registry.append((op_name, func))
        return func

    return decorator


_PASSTHROUGH_TYPES = (
    int,
    float,
    str,
    bool,
    type(None),
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)


@no_type_check
def convert_all_torch_max_tensors_to_lazy(x: Any) -> Any:
    """Recursively convert all TorchMaxTensor instances in x to their max_data"""
    # Scalars/None dominate op argument lists; test them first.
    if isinstance(x, _PASSTHROUGH_TYPES):
        return x
    if isinstance(x, TorchMaxTensor):
        if not hasattr(x, "_max_data"):
            raise RuntimeError(
                "TorchMaxTensor does not have _max_data attribute, this is a bug"
            )
        return x._max_data
    if isinstance(x, torch.Tensor):
        if x.device.type == "max_device":
            raise RuntimeError(
                "Found a raw torch.Tensor on max_device, "
                "expected it to be wrapped in TorchMaxTensor. This is a bug in the torch-max-backend."
            )
        else:
            raise RuntimeError(
                f"Cannot perform operations that mix the devices max_device and {x.device.type},"
                f" found a raw torch.Tensor on {x.device.type} when calling the max backend. "
                f"It has the shape {x.shape} and dtype {x.dtype}. "
                f"Please convert the tensor to max_device using .to('max_device') "
                f"before passing it to the max backend."
            )
    elif isinstance(x, list | tuple):
        return type(x)(convert_all_torch_max_tensors_to_lazy(item) for item in x)
    elif isinstance(x, dict):
        return {
            key: convert_all_torch_max_tensors_to_lazy(value)
            for key, value in x.items()
        }
    else:
        raise TypeError(
            f"Unsupported type to automatically convert to lazy tensors: {type(x)}"
        )


@no_type_check
def convert_all_lazy_to_torch_max_tensors(x: Any) -> Any:
    if isinstance(x, MaxEagerTensor):
        return TorchMaxTensor._from_max_data(x)
    elif isinstance(x, _PASSTHROUGH_TYPES + (NotImplementedError,)):
        return x
    elif isinstance(x, list | tuple):
        return type(x)(convert_all_lazy_to_torch_max_tensors(item) for item in x)
    elif isinstance(x, dict):
        return {
            key: convert_all_lazy_to_torch_max_tensors(value)
            for key, value in x.items()
        }
    else:
        raise TypeError(
            f"Unsupported type to automatically convert to TorchMaxTensor: {type(x)}"
        )


def wrap_for_max_device(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        args, kwargs = convert_all_torch_max_tensors_to_lazy((args, kwargs))
        result = func(*args, **kwargs)
        return convert_all_lazy_to_torch_max_tensors(result)

    return wrapper


def _eager_impl(fast_name: str, default: Callable) -> Callable:
    """Use the Mojo-extension fast implementation for eager mode if enabled.

    The fast implementations receive the TorchMaxTensor arguments directly
    (no generic conversion walk) and return `aten_fast.NOT_HANDLED`
    whenever the inputs don't qualify, in which case the call falls back
    to the graph-based `default` — so registering them is always
    behavior-preserving. Only the eager (max_device) registrations use
    this; the torch.compile backend is untouched.

    The import of the Mojo extensions (which compiles them on a cold
    cache, ~30s) is deferred to the first call of a fast-path op, so
    `import torch_max_backend` and torch.compile-only workloads never
    pay for it.
    """
    slow = wrap_for_max_device(default)
    if not _FAST_EAGER:
        return slow

    fast_fn: Callable | None = None
    not_handled = None

    def lazy_dispatcher(*args, **kwargs):
        nonlocal fast_fn, not_handled
        if fast_fn is None:
            from torch_max_backend.eager_kernels import aten_fast

            fast_fn = getattr(aten_fast, fast_name)
            not_handled = aten_fast.NOT_HANDLED
        result = fast_fn(*args, **kwargs)
        if result is not_handled:
            return slow(*args, **kwargs)
        return result

    return lazy_dispatcher


# ----------------------------------------------------------------------------------
# List of registered aten ops for max_device
# ----------------------------------------------------------------------------------
_aten_fast_module = None


def _fast_module():
    """The aten_fast module when the fast eager path is enabled, else None.

    Imported lazily: the first import triggers the (cached) Mojo kernel
    compilation, which pure torch.compile workloads should never pay for.
    The module is cached in a global afterwards; per-call `import` of an
    already-loaded module still costs ~1µs, which the hottest ops (view)
    pay hundreds of times per generated token.
    """
    global _aten_fast_module
    if _aten_fast_module is None:
        if not _FAST_EAGER:
            return None
        from torch_max_backend.eager_kernels import aten_fast

        _aten_fast_module = aten_fast
    return _aten_fast_module


register_aten_op("aten::_local_scalar_dense")(
    _eager_impl(
        "fast_aten__local_scalar_dense", aten_functions.aten__local_scalar_dense
    )
)
register_aten_op("aten::_adaptive_avg_pool2d")(
    wrap_for_max_device(aten_functions.aten__adaptive_avg_pool2d)
)
register_aten_op("aten::_adaptive_avg_pool2d_backward")(
    wrap_for_max_device(aten_functions.aten__adaptive_avg_pool2d_backward)
)


@register_aten_op("aten::_copy_from")
@no_type_check
def max_device__copy_from(self: TorchMaxTensor, dest: TorchMaxTensor) -> TorchMaxTensor:
    if self.device.type == dest.device.type and self.device.type == "max_device":
        dest_max_device = find_equivalent_max_device(dest.device)
        copied_data = self._max_data.to(dest_max_device)
        dest._max_data = copied_data
        return dest

    if self.device.type == "max_device" and dest.device.type == "cpu":
        cpu_tensor = self._max_data.to(CPU())
        x = torch.from_dlpack(cpu_tensor)
        dest.copy_(x)
        return dest

    elif self.device.type == "cpu" and dest.device.type == "max_device":
        self = TorchMaxTensor._from_max_data(
            MaxEagerTensor(storage=max.driver.Buffer.from_dlpack(self.detach()))
        )
        dest._max_data = self._max_data.to(dest._max_data.device)
        return dest
    else:
        raise RuntimeError(
            f"invalid configuration, trying to copy from "
            f"{self.device.type}:{self.device.index} to {dest.device.type}:{dest.device.index}"
        )


@register_aten_op("aten::_to_copy")
@no_type_check
def max_device__to_copy(
    tensor: TorchMaxTensor | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TorchMaxTensor:
    if isinstance(tensor, TorchMaxTensor):
        result = tensor._max_data
    else:
        result = MaxEagerTensor(storage=max.driver.Buffer.from_dlpack(tensor.detach()))
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        if max_dtype != result.dtype:
            result = _cast_max_tensor(result, max_dtype)
    if device is not None and device.type == "cpu":
        fast = _fast_module()
        if fast is not None:
            buffer = fast._buffer_or_none(result)
            if buffer is not None and buffer.dtype != DType.bfloat16:
                # Driver-level D2H copy (stream-ordered with the enqueued
                # kernels), bypassing the graph-based Tensor.to(CPU()).
                return torch.from_numpy(buffer.to_numpy())
        return torch.from_dlpack(result.to(CPU()))
    if device is not None:
        target = find_equivalent_max_device(device)
        if not (result.real and result.driver_tensor.device == target):
            result = result.to(target)
    return TorchMaxTensor._from_max_data(result)


@no_type_check
def _cast_max_tensor(result: MaxEagerTensor, max_dtype) -> MaxEagerTensor:
    """Dtype cast, through the fast Cast kernel when inputs qualify."""
    fast = _fast_module()
    if fast is not None:
        buffer = fast._buffer_or_none(result)
        if (
            buffer is not None
            and buffer.num_elements > 0
            and buffer.dtype in fast._CAST_DTYPES
            and max_dtype in fast._CAST_DTYPES
        ):
            from torch_max_backend import eager_kernels

            out = max.driver.Buffer(max_dtype, buffer.shape, buffer.device)
            eager_kernels.data_movement_ops.Cast(
                out, buffer, eager_kernels._ctx_ptr(buffer.device)
            )
            return MaxEagerTensor(storage=out)
    return result.cast(max_dtype)


register_aten_op("aten::_log_softmax")(
    wrap_for_max_device(aten_functions.aten__log_softmax)
)
register_aten_op("aten::_native_batch_norm_legit_no_training")(
    _eager_impl(
        "fast_aten__native_batch_norm_legit_no_training",
        aten_functions.aten__native_batch_norm_legit_no_training,
    )
)

register_aten_op("aten::_scaled_dot_product_efficient_attention")(
    wrap_for_max_device(aten_functions.aten__scaled_dot_product_efficient_attention)
)

register_aten_op("aten::_scaled_dot_product_flash_attention")(
    wrap_for_max_device(aten_functions.aten__scaled_dot_product_flash_attention)
)
register_aten_op("aten::_scaled_dot_product_attention_math")(
    wrap_for_max_device(aten_functions.aten__scaled_dot_product_attention_math)
)
register_aten_op("aten::scaled_dot_product_attention")(
    _eager_impl(
        "fast_aten_scaled_dot_product_attention",
        aten_functions.aten_scaled_dot_product_attention,
    )
)
register_aten_op("aten::_softmax")(wrap_for_max_device(aten_functions.aten__softmax))


register_aten_op("aten::abs")(wrap_for_max_device(aten_functions.aten_abs))
register_aten_op("aten::acos")(wrap_for_max_device(aten_functions.aten_acos))
register_aten_op("aten::add.Tensor")(
    _eager_impl("fast_aten_add", aten_functions.aten_add)
)


@register_aten_op("aten::add_.Tensor")
@no_type_check
def max_device_add_(
    self: TorchMaxTensor, other: TorchMaxTensor | int | float, alpha: float = 1.0
) -> TorchMaxTensor:
    fast = _fast_module()
    if fast is not None:
        # In-place kernel writing into self's existing buffer.
        result = fast.fast_aten_add_(self, other, alpha)
        if result is not None:
            return self
    rhs = other._max_data if isinstance(other, TorchMaxTensor) else other
    self._max_data = aten_functions.aten_add(self._max_data, rhs, alpha)
    return self


register_aten_op("aten::addcdiv")(wrap_for_max_device(aten_functions.aten_addcdiv))
register_aten_op("aten::addcmul")(wrap_for_max_device(aten_functions.aten_addcmul))
register_aten_op("aten::addmm")(
    _eager_impl("fast_aten_addmm", aten_functions.aten_addmm)
)

register_aten_op("aten::alias")(wrap_for_max_device(aten_functions.aten_alias))
register_aten_op("aten::amax")(wrap_for_max_device(aten_functions.aten_amax))
register_aten_op("aten::amin")(wrap_for_max_device(aten_functions.aten_amin))
register_aten_op("aten::any")(_eager_impl("fast_aten_any", aten_functions.aten_any))
register_aten_op("aten::all")(_eager_impl("fast_aten_all", aten_functions.aten_all))
register_aten_op("aten::all.dim")(wrap_for_max_device(aten_functions.aten_all))
register_aten_op("aten::all.dims")(wrap_for_max_device(aten_functions.aten_all))


def _host_arange_array(start, end, step, dtype: torch.dtype | None):
    """torch.arange(start, end, step, dtype=dtype) built as a numpy array.

    torch's CPU arange opens an OpenMP parallel region, which costs
    milliseconds per call on machines where the OMP pool is oversubscribed;
    numpy stays single-threaded and takes microseconds. Returns None when
    the result can't be built with numpy (bfloat16, step of 0, ...) so
    callers can fall back to torch.
    """
    if dtype is None:
        any_float = any(isinstance(v, float) for v in (start, end, step))
        dtype = torch.get_default_dtype() if any_float else torch.int64
    try:
        np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    except TypeError:
        return None
    if step == 0:
        return None
    if all(isinstance(v, int) for v in (start, end, step)):
        n = -((start - end) // step)  # exact ceil((end - start) / step)
        values = np.arange(n, dtype=np.int64)
    else:
        n = math.ceil((float(end) - float(start)) / float(step))
        values = np.arange(n, dtype=np.float64)
    if n <= 0:
        return None
    return (values * step + start).astype(np_dtype, copy=False)


@register_aten_op("aten::arange")
@no_type_check
def max_device_arange(
    start, end=None, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    if _FAST_EAGER:
        # Build on the host with exact torch semantics, then one H2D copy.
        # Buffer.to() completes before returning, so the host array (whose
        # memory the dlpack buffer aliases) may safely die afterwards; an
        # async copy (e.g. inplace_copy_from) would race its free.
        if end is None:
            start, end = 0, start
        arr = _host_arange_array(start, end, step, dtype)
        if arr is None:
            cpu_tensor = torch.arange(start, end, step, dtype=dtype)
        else:
            cpu_tensor = arr
        host = max.driver.Buffer.from_dlpack(cpu_tensor)
        out = host.to(find_equivalent_max_device(device))
        return TorchMaxTensor._from_buffer(out)
    return wrap_for_max_device(aten_functions.aten_arange)(
        start,
        end,
        step,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


@register_aten_op("aten::arange.start_out")
@no_type_check
def max_device_arange_start_out(start, end, step=1, *, out) -> TorchMaxTensor:
    # torch.arange(start, end, step, device=...) dispatches to the out
    # variant with a pre-allocated `out` of the right size and dtype.
    arr = _host_arange_array(start, end, step, out.dtype)
    if arr is None:
        cpu_tensor = torch.arange(start, end, step, dtype=out.dtype)
    else:
        cpu_tensor = arr
    host = max.driver.Buffer.from_dlpack(cpu_tensor)
    out._max_data = MaxEagerTensor(
        storage=host.to(find_equivalent_max_device(out.device))
    )
    return out


register_aten_op("aten::argmax")(
    _eager_impl("fast_aten_argmax", aten_functions.aten_argmax)
)
register_aten_op("aten::argmin")(wrap_for_max_device(aten_functions.aten_argmin))
register_aten_op("aten::asinh")(wrap_for_max_device(aten_functions.aten_asinh))
register_aten_op("aten::atanh")(wrap_for_max_device(aten_functions.aten_atanh))

register_aten_op("aten::avg_pool2d")(
    wrap_for_max_device(aten_functions.aten_avg_pool2d)
)

register_aten_op("aten::bitwise_and.Scalar")(
    _eager_impl("fast_aten_bitwise_and", aten_functions.aten_bitwise_and_scalar)
)
register_aten_op("aten::bitwise_and.Tensor")(
    _eager_impl("fast_aten_bitwise_and", aten_functions.aten_bitwise_and)
)
register_aten_op("aten::bitwise_not")(
    _eager_impl("fast_aten_bitwise_not", aten_functions.aten_bitwise_not)
)
register_aten_op("aten::bitwise_or.Scalar")(
    _eager_impl("fast_aten_bitwise_or", aten_functions.aten_bitwise_or_scalar)
)
register_aten_op("aten::bitwise_or.Tensor")(
    _eager_impl("fast_aten_bitwise_or", aten_functions.aten_bitwise_or)
)
register_aten_op("aten::bitwise_xor.Scalar")(
    _eager_impl("fast_aten_bitwise_xor", aten_functions.aten_bitwise_xor_scalar)
)
register_aten_op("aten::bitwise_xor.Tensor")(
    _eager_impl("fast_aten_bitwise_xor", aten_functions.aten_bitwise_xor)
)
register_aten_op("aten::bmm")(_eager_impl("fast_aten_bmm", aten_functions.aten_bmm))

register_aten_op("aten::cat")(_eager_impl("fast_aten_cat", aten_functions.aten_cat))

register_aten_op("aten::ceil")(wrap_for_max_device(aten_functions.aten_ceil))
register_aten_op("aten::clamp")(wrap_for_max_device(aten_functions.aten_clamp))
register_aten_op("aten::clone")(wrap_for_max_device(aten_functions.aten_clone))

register_aten_op("aten::convolution")(
    _eager_impl("fast_aten_convolution", aten_functions.aten_convolution)
)

register_aten_op("aten::cos")(wrap_for_max_device(aten_functions.aten_cos))
register_aten_op("aten::cosh")(wrap_for_max_device(aten_functions.aten_cosh))

register_aten_op("aten::cumsum")(
    _eager_impl("fast_aten_cumsum", aten_functions.aten_cumsum)
)

register_aten_op("aten::detach")(wrap_for_max_device(aten_functions.aten_detach))

register_aten_op("aten::div.Tensor")(
    _eager_impl("fast_aten_div", aten_functions.aten_div)
)

register_aten_op("aten::embedding")(
    _eager_impl("fast_aten_embedding", aten_functions.aten_embedding)
)

register_aten_op("aten::empty_permuted")(
    wrap_for_max_device(aten_functions.aten_empty_permuted)
)


@register_aten_op("aten::empty.memory_format")
@no_type_check
def max_device_empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = torch_dtype_to_max(dtype)
    device = find_equivalent_max_device(device)
    if _FAST_EAGER:
        # torch.empty is uninitialized memory: a bare allocation, no kernel.
        return TorchMaxTensor._from_buffer(
            max.driver.Buffer(dtype, tuple(size), device)
        )
    return TorchMaxTensor._from_max_data(
        MaxEagerTensor.zeros(size, dtype=dtype, device=device)
    )


@register_aten_op("aten::empty_strided.memory_format")
@register_aten_op("aten::empty_strided")
@no_type_check
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = torch_dtype_to_max(dtype)
    device = find_equivalent_max_device(device)
    if _FAST_EAGER:
        return TorchMaxTensor._from_buffer(
            max.driver.Buffer(dtype, tuple(size), device)
        )
    return TorchMaxTensor._from_max_data(
        MaxEagerTensor.zeros(size, dtype=dtype, device=device)
    )


register_aten_op("aten::empty_like")(
    wrap_for_max_device(aten_functions.aten_empty_like)
)

register_aten_op("aten::eq")(_eager_impl("fast_aten_eq", aten_functions.aten_eq))
register_aten_op("aten::eq.Scalar")(_eager_impl("fast_aten_eq", aten_functions.aten_eq))
register_aten_op("aten::eq.Tensor")(_eager_impl("fast_aten_eq", aten_functions.aten_eq))

register_aten_op("aten::erf")(wrap_for_max_device(aten_functions.aten_erf))
register_aten_op("aten::exp")(_eager_impl("fast_aten_exp", aten_functions.aten_exp))
register_aten_op("aten::expand")(wrap_for_max_device(aten_functions.aten_expand))

register_aten_op("aten::fill.Scalar")(
    _eager_impl("fast_aten_fill_scalar", aten_functions.aten_fill_scalar)
)


@register_aten_op("aten::fill_.Scalar")
@no_type_check
def max_device_fill__scalar(self: TorchMaxTensor, value: float) -> TorchMaxTensor:
    if _FAST_EAGER:
        from torch_max_backend.eager_kernels import aten_fast

        result = aten_fast.fast_aten_fill__scalar(self, value)
        if result is not None:
            return result
    self._max_data = aten_functions.aten_fill__scalar(self._max_data, value)
    return self


register_aten_op("aten::floor")(wrap_for_max_device(aten_functions.aten_floor))
register_aten_op("aten::floordiv")(wrap_for_max_device(aten_functions.aten_floordiv))


@no_type_check
def _fast_filled_tensor(size, value, dtype, device):
    """Filled-tensor factory on the fast path (alloc + Fill), or None."""
    fast = _fast_module()
    if fast is None:
        return None
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        return None
    return fast.fast_filled(size, value, max_dtype, find_equivalent_max_device(device))


_slow_full = wrap_for_max_device(aten_functions.aten_full)


@register_aten_op("aten::full")
@no_type_check
def max_device_full(
    size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    if dtype is not None:
        resolved = dtype
    elif isinstance(fill_value, bool):
        resolved = torch.bool
    elif isinstance(fill_value, int):
        resolved = torch.int64
    else:
        resolved = torch.get_default_dtype()
    result = _fast_filled_tensor(size, fill_value, resolved, device)
    if result is not None:
        return result
    return _slow_full(
        size,
        fill_value,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


register_aten_op("aten::full_like")(wrap_for_max_device(aten_functions.aten_full_like))

register_aten_op("aten::ge")(_eager_impl("fast_aten_ge", aten_functions.aten_ge))
register_aten_op("aten::ge.Scalar")(_eager_impl("fast_aten_ge", aten_functions.aten_ge))
register_aten_op("aten::ge.Tensor")(_eager_impl("fast_aten_ge", aten_functions.aten_ge))

register_aten_op("aten::gelu")(wrap_for_max_device(aten_functions.aten_gelu))
register_aten_op("aten::gelu_backward")(
    wrap_for_max_device(aten_functions.aten_gelu_backward)
)
register_aten_op("aten::gt")(_eager_impl("fast_aten_gt", aten_functions.aten_gt))
register_aten_op("aten::gt.Scalar")(_eager_impl("fast_aten_gt", aten_functions.aten_gt))
register_aten_op("aten::gt.Tensor")(_eager_impl("fast_aten_gt", aten_functions.aten_gt))

register_aten_op("aten::index.Tensor")(wrap_for_max_device(aten_functions.aten_index))
register_aten_op("aten::isin.Tensor_Tensor")(
    _eager_impl("fast_aten_isin", aten_functions.aten_isin)
)
register_aten_op("aten::isin.Tensor_Tensor_out")(
    wrap_for_max_device(aten_functions.aten_isin)
)
register_aten_op("aten::isnan")(wrap_for_max_device(aten_functions.aten_isnan))

register_aten_op("aten::le")(_eager_impl("fast_aten_le", aten_functions.aten_le))
register_aten_op("aten::le.Scalar")(_eager_impl("fast_aten_le", aten_functions.aten_le))
register_aten_op("aten::le.Tensor")(_eager_impl("fast_aten_le", aten_functions.aten_le))

register_aten_op("aten::linear")(
    _eager_impl("fast_aten_linear", aten_functions.aten_linear)
)

register_aten_op("aten::log")(wrap_for_max_device(aten_functions.aten_log))
register_aten_op("aten::log1p")(wrap_for_max_device(aten_functions.aten_log1p))

register_aten_op("aten::logical_and")(
    wrap_for_max_device(aten_functions.aten_logical_and)
)
register_aten_op("aten::logical_not")(
    wrap_for_max_device(aten_functions.aten_logical_not)
)
register_aten_op("aten::logical_xor")(
    wrap_for_max_device(aten_functions.aten_logical_xor)
)
register_aten_op("aten::lt")(_eager_impl("fast_aten_lt", aten_functions.aten_lt))
register_aten_op("aten::lt.Scalar")(_eager_impl("fast_aten_lt", aten_functions.aten_lt))
register_aten_op("aten::lt.Tensor")(_eager_impl("fast_aten_lt", aten_functions.aten_lt))

register_aten_op("aten::masked_fill.Scalar")(
    _eager_impl("fast_aten_masked_fill", aten_functions.aten_masked_fill)
)
register_aten_op("aten::masked_fill.Tensor")(
    _eager_impl("fast_aten_masked_fill", aten_functions.aten_masked_fill)
)


@register_aten_op("aten::masked_fill_.Scalar")
def max_device_masked_fill__scalar(
    self: TorchMaxTensor, mask: TorchMaxTensor, value: int | float
) -> TorchMaxTensor:
    # in-place masked fill
    fast = _fast_module()
    if fast is not None:
        result = fast.fast_aten_masked_fill_(self, mask, value)
        if result is not None:
            return result
    self._max_data = aten_functions.aten_masked_fill(
        self._max_data, mask._max_data, value
    )
    return self


@register_aten_op("aten::masked_fill_.Tensor")
def max_device_masked_fill__tensor(
    self: TorchMaxTensor, mask: TorchMaxTensor, value: TorchMaxTensor
) -> TorchMaxTensor:
    # in-place masked fill
    fast = _fast_module()
    if fast is not None:
        result = fast.fast_aten_masked_fill_(self, mask, value)
        if result is not None:
            return result
    self._max_data = aten_functions.aten_masked_fill(
        self._max_data, mask._max_data, value._max_data
    )
    return self


register_aten_op("aten::max")(_eager_impl("fast_aten_max", aten_functions.aten_max))

register_aten_op("aten::max_pool2d_with_indices")(
    _eager_impl(
        "fast_aten_max_pool2d_with_indices", aten_functions.aten_max_pool2d_with_indices
    )
)

register_aten_op("aten::maximum")(
    _eager_impl("fast_aten_maximum", aten_functions.aten_maximum)
)
register_aten_op("aten::mean")(_eager_impl("fast_aten_mean", aten_functions.aten_mean))
# Registering the base name only covers the default overload; mean.dim would
# otherwise get decomposed by PyTorch into a chain of sum/div/... ops.
register_aten_op("aten::mean.dim")(
    _eager_impl("fast_aten_mean", aten_functions.aten_mean)
)


@register_aten_op("aten::mean.out")
def max_device_mean_out(
    input: TorchMaxTensor,
    dim,
    keepdim: bool = False,
    *,
    dtype=None,
    out: TorchMaxTensor,
) -> TorchMaxTensor:
    out._max_data = aten_functions.aten_mean_out(
        input._max_data, dim, keepdim, dtype=dtype, out=out._max_data
    )
    return out


register_aten_op("aten::min")(wrap_for_max_device(aten_functions.aten_min))


@register_aten_op("aten::min.dim_min")
def max_device_min_dim_min(
    input: TorchMaxTensor,
    dim: int,
    keepdim: bool = False,
    min: TorchMaxTensor | None = None,
    min_indices: TorchMaxTensor | None = None,
) -> tuple[TorchMaxTensor, TorchMaxTensor]:
    """
    Out-variant of min operation along a dimension for max_device.
    Computes minimum values and indices, writing to pre-allocated output tensors.
    """
    # Convert input to MaxEagerTensor
    input_max = input._max_data

    # Compute results using aten_functions
    values, indices = aten_functions.aten_min_dim_min(
        input_max, dim=dim, keepdim=keepdim, min=None, min_indices=None
    )

    # Copy results into pre-allocated output tensors
    if min is not None:
        min._max_data = values
    if min_indices is not None:
        min_indices._max_data = indices

    # Return the output tensors (wrapped as TorchMaxTensor)
    return (min, min_indices)


register_aten_op("aten::minimum")(
    _eager_impl("fast_aten_minimum", aten_functions.aten_minimum)
)

register_aten_op("aten::mul.Tensor")(
    _eager_impl("fast_aten_mul", aten_functions.aten_mul)
)

register_aten_op("aten::mm")(_eager_impl("fast_aten_mm", aten_functions.aten_mm))

register_aten_op("aten::native_batch_norm")(
    _eager_impl("fast_aten_native_batch_norm", aten_functions.aten_native_batch_norm)
)
register_aten_op("aten::native_group_norm")(
    wrap_for_max_device(aten_functions.aten_native_group_norm)
)
register_aten_op("aten::native_layer_norm")(
    _eager_impl("fast_aten_native_layer_norm", aten_functions.aten_native_layer_norm)
)


@register_aten_op("aten::normal_")
def max_device_normal_(
    self: TorchMaxTensor, mean: float = 0.0, std: float = 1.0, generator=None
) -> TorchMaxTensor:
    self._max_data = aten_functions.aten_normal_(self._max_data, mean, std, generator)
    return self


register_aten_op("aten::ne")(_eager_impl("fast_aten_ne", aten_functions.aten_ne))
register_aten_op("aten::ne.Scalar")(_eager_impl("fast_aten_ne", aten_functions.aten_ne))
register_aten_op("aten::ne.Tensor")(_eager_impl("fast_aten_ne", aten_functions.aten_ne))
register_aten_op("aten::neg")(wrap_for_max_device(aten_functions.aten_neg))

register_aten_op("aten::nonzero")(wrap_for_max_device(aten_functions.aten_nonzero))


_slow_ones = wrap_for_max_device(aten_functions.aten_ones)


@register_aten_op("aten::ones")
@no_type_check
def max_device_ones(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    result = _fast_filled_tensor(size, 1, resolved, device)
    if result is not None:
        return result
    return _slow_ones(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


register_aten_op("aten::ones_like")(wrap_for_max_device(aten_functions.aten_ones_like))

register_aten_op("aten::permute")(wrap_for_max_device(aten_functions.aten_permute))

register_aten_op("aten::pow.Tensor_Scalar")(
    _eager_impl("fast_aten_pow", aten_functions.aten_pow)
)

register_aten_op("aten::pow.Tensor_Tensor")(
    wrap_for_max_device(aten_functions.aten_pow)
)

register_aten_op("aten::relu")(_eager_impl("fast_aten_relu", aten_functions.aten_relu))


@register_aten_op("aten::relu_")
@no_type_check
def max_device_relu_(self: TorchMaxTensor) -> TorchMaxTensor:
    # in-place relu
    self._max_data = aten_functions.aten_relu(self._max_data)
    return self


register_aten_op("aten::reciprocal")(
    wrap_for_max_device(aten_functions.aten_reciprocal)
)
register_aten_op("aten::remainder")(wrap_for_max_device(aten_functions.aten_remainder))
register_aten_op("aten::repeat")(wrap_for_max_device(aten_functions.aten_repeat))
register_aten_op("aten::rsqrt")(wrap_for_max_device(aten_functions.aten_rsqrt))

_slow_scalar_tensor = wrap_for_max_device(aten_functions.aten_scalar_tensor)


@register_aten_op("aten::scalar_tensor")
@no_type_check
def max_device_scalar_tensor(
    s, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.float32 if dtype is None else dtype
    result = _fast_filled_tensor((), s, resolved, device)
    if result is not None:
        return result
    return _slow_scalar_tensor(s, dtype, layout, device)


register_aten_op("aten::scatter.src")(
    wrap_for_max_device(aten_functions.aten_scatter_src)
)
register_aten_op("aten::scatter.value")(
    wrap_for_max_device(aten_functions.aten_scatter_value)
)

register_aten_op("aten::select.int")(
    _eager_impl("fast_aten_select", aten_functions.aten_select)
)
register_aten_op("aten::select_scatter")(
    wrap_for_max_device(aten_functions.aten_select_scatter)
)
register_aten_op("aten::sigmoid")(wrap_for_max_device(aten_functions.aten_sigmoid))
register_aten_op("aten::sign")(wrap_for_max_device(aten_functions.aten_sign))
register_aten_op("aten::silu")(wrap_for_max_device(aten_functions.aten_silu))
register_aten_op("aten::sin")(wrap_for_max_device(aten_functions.aten_sin))
register_aten_op("aten::sinh")(wrap_for_max_device(aten_functions.aten_sinh))

register_aten_op("aten::slice.Tensor")(
    _eager_impl("fast_aten_slice", aten_functions.aten_slice)
)

register_aten_op("aten::softmax.int")(wrap_for_max_device(aten_functions.aten_softmax))

register_aten_op("aten::split.Tensor")(
    _eager_impl("fast_aten_split", aten_functions.aten_split)
)
register_aten_op("aten::split_with_sizes")(
    wrap_for_max_device(aten_functions.aten_split_with_sizes)
)

register_aten_op("aten::sqrt")(wrap_for_max_device(aten_functions.aten_sqrt))
register_aten_op("aten::squeeze.dim")(wrap_for_max_device(aten_functions.aten_squeeze))

register_aten_op("aten::stack")(wrap_for_max_device(aten_functions.aten_stack))

register_aten_op("aten::sub.Tensor")(
    _eager_impl("fast_aten_sub", aten_functions.aten_sub)
)
register_aten_op("aten::sum.dim_IntList")(wrap_for_max_device(aten_functions.aten_sum))
register_aten_op("aten::t")(_eager_impl("fast_aten_t", aten_functions.aten_t))
register_aten_op("aten::tan")(wrap_for_max_device(aten_functions.aten_tan))
register_aten_op("aten::tanh")(_eager_impl("fast_aten_tanh", aten_functions.aten_tanh))

register_aten_op("aten::transpose.int")(
    _eager_impl("fast_aten_transpose", aten_functions.aten_transpose)
)

register_aten_op("aten::tril")(wrap_for_max_device(aten_functions.aten_tril))
register_aten_op("aten::triu")(wrap_for_max_device(aten_functions.aten_triu))

register_aten_op("aten::unbind.int")(wrap_for_max_device(aten_functions.aten_unbind))
register_aten_op("aten::unsqueeze")(
    _eager_impl("fast_aten_unsqueeze", aten_functions.aten_unsqueeze)
)

register_aten_op("aten::upsample_bilinear2d")(
    wrap_for_max_device(aten_functions.aten_upsample_bilinear2d)
)

register_aten_op("aten::var.correction")(wrap_for_max_device(aten_functions.aten_var))


_slow_view = wrap_for_max_device(aten_functions.aten_view)


@register_aten_op("aten::view")
@no_type_check
def max_device_view(self: TorchMaxTensor, size) -> TorchMaxTensor:
    # view is by far the most frequent op in transformer forwards; skip the
    # generic argument-conversion walk and alias the driver buffer directly.
    if _FAST_EAGER:
        aten_fast = _aten_fast_module or _fast_module()

        buffer = aten_fast._buffer_or_none(self)
        if buffer is not None and buffer.num_elements > 0:
            sizes = aten_fast._resolve_sizes(size, buffer.num_elements)
            if sizes is not None:
                return TorchMaxTensor._from_buffer(buffer.view(buffer.dtype, sizes))
    return _slow_view(self, size)


register_aten_op("aten::_unsafe_view")(
    _eager_impl("fast_aten__unsafe_view", aten_functions.aten__unsafe_view)
)

register_aten_op("aten::where.self")(
    _eager_impl("fast_aten_where", aten_functions.aten_where)
)

_slow_zeros = wrap_for_max_device(aten_functions.aten_zeros)


@register_aten_op("aten::zeros")
@no_type_check
def max_device_zeros(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    result = _fast_filled_tensor(size, 0, resolved, device)
    if result is not None:
        return result
    return _slow_zeros(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
