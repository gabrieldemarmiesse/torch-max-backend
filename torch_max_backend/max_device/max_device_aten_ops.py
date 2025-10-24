from collections.abc import Callable
from typing import Any

import max.driver
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor as MaxEagerTensor

from torch_max_backend import aten_functions
from torch_max_backend.max_device.torch_max_tensor import (
    TorchMaxTensor,
    find_equivalent_max_device,
)

# Global registry for functions to register
_aten_ops_registry: list[tuple[str, Callable]] = []


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


def convert_all_torch_max_tensors_to_lazy(x: Any) -> Any:
    """Recursively convert all TorchMaxTensor instances in x to their max_data"""
    if isinstance(x, TorchMaxTensor):
        if not hasattr(x, "_max_data"):
            raise RuntimeError(
                "TorchMaxTensor does not have _max_data attribute, this is a bug"
            )
        return x._max_data
    elif isinstance(x, list | tuple):
        return type(x)(convert_all_torch_max_tensors_to_lazy(item) for item in x)
    elif isinstance(x, dict):
        return {
            key: convert_all_torch_max_tensors_to_lazy(value)
            for key, value in x.items()
        }
    elif isinstance(
        x, int | float | str | bool | type(None) | torch.dtype | torch.device
    ):
        return x
    else:
        raise TypeError(
            f"Unsupported type to automatically convert to lazy tensors: {type(x)}"
        )


def convert_all_lazy_to_torch_max_tensors(x: Any) -> Any:
    if isinstance(x, MaxEagerTensor):
        return TorchMaxTensor._from_max_data(x)
    elif isinstance(x, list | tuple):
        return type(x)(convert_all_lazy_to_torch_max_tensors(item) for item in x)
    elif isinstance(x, dict):
        return {
            key: convert_all_lazy_to_torch_max_tensors(value)
            for key, value in x.items()
        }
    elif isinstance(
        x, int | float | str | bool | type(None) | torch.dtype | torch.device
    ):
        return x
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


# ----------------------------------------------------------------------------------
# List of registered aten ops for max_device
# ----------------------------------------------------------------------------------

register_aten_op("aten::_adaptive_avg_pool2d")(
    wrap_for_max_device(aten_functions.aten__adaptive_avg_pool2d)
)


@register_aten_op("aten::_copy_from")
def max_device__copy_from(self: TorchMaxTensor, dest: TorchMaxTensor) -> TorchMaxTensor:
    if self.device == dest.device and self.device.type == "max_device":
        dest._max_data = self._max_data
        return dest

    if self.device.type == "max_device" and dest.device.type == "cpu":
        cpu_tensor = self._max_data.to(CPU())
        x = torch.from_dlpack(cpu_tensor)
        dest.copy_(x)
        return dest

    elif self.device.type == "cpu" and dest.device.type == "max_device":
        self = TorchMaxTensor._from_max_data(
            MaxEagerTensor(storage=max.driver.Tensor.from_dlpack(self.detach()))
        )
        dest._max_data = self._max_data.to(dest._max_data.device)
        return dest
    else:
        raise RuntimeError(
            f"invalid configuration {self.device.type}, {dest.device.type}"
        )


register_aten_op("aten::_log_softmax")(
    wrap_for_max_device(aten_functions.aten__log_softmax)
)
register_aten_op("aten::_native_batch_norm_legit_no_training")(
    wrap_for_max_device(aten_functions.aten__native_batch_norm_legit_no_training)
)

register_aten_op("aten::_scaled_dot_product_efficient_attention")(
    wrap_for_max_device(aten_functions.aten__scaled_dot_product_efficient_attention)
)

register_aten_op("aten::_scaled_dot_product_flash_attention")(
    wrap_for_max_device(aten_functions.aten__scaled_dot_product_flash_attention)
)

register_aten_op("aten::_softmax")(wrap_for_max_device(aten_functions.aten__softmax))


register_aten_op("aten::abs")(wrap_for_max_device(aten_functions.aten_abs))
register_aten_op("aten::acos")(wrap_for_max_device(aten_functions.aten_acos))
register_aten_op("aten::add.Tensor")(wrap_for_max_device(aten_functions.aten_add))


@register_aten_op("aten::add_.Tensor")
def max_device_add_(
    self: TorchMaxTensor, other: TorchMaxTensor, alpha: float = 1.0
) -> TorchMaxTensor:
    self._max_data = aten_functions.aten_add(self._max_data, other._max_data, alpha)
    return self


register_aten_op("aten::addcdiv")(wrap_for_max_device(aten_functions.aten_addcdiv))
register_aten_op("aten::addcmul")(wrap_for_max_device(aten_functions.aten_addcmul))
register_aten_op("aten::addmm")(wrap_for_max_device(aten_functions.aten_addmm))

register_aten_op("aten::alias")(wrap_for_max_device(aten_functions.aten_alias))
register_aten_op("aten::amax")(wrap_for_max_device(aten_functions.aten_amax))
register_aten_op("aten::amin")(wrap_for_max_device(aten_functions.aten_amin))
register_aten_op("aten::any")(wrap_for_max_device(aten_functions.aten_any))
register_aten_op("aten::arange")(wrap_for_max_device(aten_functions.aten_arange))

register_aten_op("aten::argmax")(wrap_for_max_device(aten_functions.aten_argmax))
register_aten_op("aten::argmin")(wrap_for_max_device(aten_functions.aten_argmin))
register_aten_op("aten::asinh")(wrap_for_max_device(aten_functions.aten_asinh))
register_aten_op("aten::atanh")(wrap_for_max_device(aten_functions.aten_atanh))

register_aten_op("aten::avg_pool2d")(
    wrap_for_max_device(aten_functions.aten_avg_pool2d)
)

register_aten_op("aten::bitwise_and.Scalar")(
    wrap_for_max_device(aten_functions.aten_bitwise_and_scalar)
)
register_aten_op("aten::bitwise_and.Tensor")(
    wrap_for_max_device(aten_functions.aten_bitwise_and)
)
register_aten_op("aten::bitwise_not")(
    wrap_for_max_device(aten_functions.aten_bitwise_not)
)
register_aten_op("aten::bitwise_or.Scalar")(
    wrap_for_max_device(aten_functions.aten_bitwise_or_scalar)
)
register_aten_op("aten::bitwise_or.Tensor")(
    wrap_for_max_device(aten_functions.aten_bitwise_or)
)
register_aten_op("aten::bitwise_xor.Scalar")(
    wrap_for_max_device(aten_functions.aten_bitwise_xor_scalar)
)
register_aten_op("aten::bitwise_xor.Tensor")(
    wrap_for_max_device(aten_functions.aten_bitwise_xor)
)
register_aten_op("aten::bmm")(wrap_for_max_device(aten_functions.aten_bmm))

register_aten_op("aten::cat")(wrap_for_max_device(aten_functions.aten_cat))

register_aten_op("aten::ceil")(wrap_for_max_device(aten_functions.aten_ceil))
register_aten_op("aten::clamp")(wrap_for_max_device(aten_functions.aten_clamp))
register_aten_op("aten::clone")(wrap_for_max_device(aten_functions.aten_clone))

register_aten_op("aten::convolution")(
    wrap_for_max_device(aten_functions.aten_convolution)
)

register_aten_op("aten::cos")(wrap_for_max_device(aten_functions.aten_cos))
register_aten_op("aten::cosh")(wrap_for_max_device(aten_functions.aten_cosh))

register_aten_op("aten::cumsum")(wrap_for_max_device(aten_functions.aten_cumsum))

register_aten_op("aten::detach")(wrap_for_max_device(aten_functions.aten_detach))

register_aten_op("aten::div.Tensor")(wrap_for_max_device(aten_functions.aten_div))

register_aten_op("aten::embedding")(wrap_for_max_device(aten_functions.aten_embedding))

register_aten_op("aten::empty_permuted")(
    wrap_for_max_device(aten_functions.aten_empty_permuted)
)


@register_aten_op("aten::empty.memory_format")
def max_device_empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    return TorchMaxTensor._from_max_data(
        MaxEagerTensor.zeros(size, dtype=dtype, device=device)
    )


@register_aten_op("aten::empty_strided.memory_format")
@register_aten_op("aten::empty_strided")
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    return TorchMaxTensor._from_max_data(
        MaxEagerTensor.zeros(size, dtype=dtype, device=device)
    )


register_aten_op("aten::eq")(wrap_for_max_device(aten_functions.aten_eq))
register_aten_op("aten::exp")(wrap_for_max_device(aten_functions.aten_exp))
register_aten_op("aten::expand")(wrap_for_max_device(aten_functions.aten_expand))

register_aten_op("aten::fill.Scalar")(
    wrap_for_max_device(aten_functions.aten_fill_scalar)
)
register_aten_op("aten::floor")(wrap_for_max_device(aten_functions.aten_floor))
register_aten_op("aten::floordiv")(wrap_for_max_device(aten_functions.aten_floordiv))
register_aten_op("aten::full")(wrap_for_max_device(aten_functions.aten_full))
register_aten_op("aten::full_like")(wrap_for_max_device(aten_functions.aten_full_like))

register_aten_op("aten::ge")(wrap_for_max_device(aten_functions.aten_ge))
register_aten_op("aten::gelu")(wrap_for_max_device(aten_functions.aten_gelu))
register_aten_op("aten::gt")(wrap_for_max_device(aten_functions.aten_gt))

register_aten_op("aten::index.Tensor")(wrap_for_max_device(aten_functions.aten_index))
register_aten_op("aten::isnan")(wrap_for_max_device(aten_functions.aten_isnan))

register_aten_op("aten::le")(wrap_for_max_device(aten_functions.aten_le))
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
register_aten_op("aten::lt")(wrap_for_max_device(aten_functions.aten_lt))

register_aten_op("aten::masked_fill.Scalar")(
    wrap_for_max_device(aten_functions.aten_masked_fill)
)
register_aten_op("aten::max")(wrap_for_max_device(aten_functions.aten_max))

register_aten_op("aten::max_pool2d_with_indices")(
    wrap_for_max_device(aten_functions.aten_max_pool2d_with_indices)
)

register_aten_op("aten::maximum")(wrap_for_max_device(aten_functions.aten_maximum))
register_aten_op("aten::mean")(wrap_for_max_device(aten_functions.aten_mean))
register_aten_op("aten::min")(wrap_for_max_device(aten_functions.aten_min))
register_aten_op("aten::minimum")(wrap_for_max_device(aten_functions.aten_minimum))

register_aten_op("aten::mul.Tensor")(wrap_for_max_device(aten_functions.aten_mul))

register_aten_op("aten::mm")(wrap_for_max_device(aten_functions.aten_mm))

register_aten_op("aten::native_group_norm")(
    wrap_for_max_device(aten_functions.aten_native_group_norm)
)
register_aten_op("aten::native_layer_norm")(
    wrap_for_max_device(aten_functions.aten_native_layer_norm)
)

register_aten_op("aten::ne")(wrap_for_max_device(aten_functions.aten_ne))
register_aten_op("aten::neg")(wrap_for_max_device(aten_functions.aten_neg))

register_aten_op("aten::nonzero")(wrap_for_max_device(aten_functions.aten_nonzero))
register_aten_op("aten::ones")(wrap_for_max_device(aten_functions.aten_ones))

register_aten_op("aten::permute")(wrap_for_max_device(aten_functions.aten_permute))

register_aten_op("aten::pow.Tensor_Scalar")(
    wrap_for_max_device(aten_functions.aten_pow)
)

register_aten_op("aten::relu")(wrap_for_max_device(aten_functions.aten_relu))


@register_aten_op("aten::relu_")
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

register_aten_op("aten::scalar_tensor")(
    wrap_for_max_device(aten_functions.aten_scalar_tensor)
)

register_aten_op("aten::scatter.src")(
    wrap_for_max_device(aten_functions.aten_scatter_src)
)
register_aten_op("aten::scatter.value")(
    wrap_for_max_device(aten_functions.aten_scatter_value)
)

register_aten_op("aten::select.int")(wrap_for_max_device(aten_functions.aten_select))
register_aten_op("aten::select_scatter")(
    wrap_for_max_device(aten_functions.aten_select_scatter)
)
register_aten_op("aten::sigmoid")(wrap_for_max_device(aten_functions.aten_sigmoid))
register_aten_op("aten::sign")(wrap_for_max_device(aten_functions.aten_sign))
register_aten_op("aten::sin")(wrap_for_max_device(aten_functions.aten_sin))
register_aten_op("aten::sinh")(wrap_for_max_device(aten_functions.aten_sinh))

register_aten_op("aten::slice.Tensor")(wrap_for_max_device(aten_functions.aten_slice))

register_aten_op("aten::softmax.int")(wrap_for_max_device(aten_functions.aten_softmax))

register_aten_op("aten::split.Tensor")(wrap_for_max_device(aten_functions.aten_split))
register_aten_op("aten::split_with_sizes")(
    wrap_for_max_device(aten_functions.aten_split_with_sizes)
)

register_aten_op("aten::sqrt")(wrap_for_max_device(aten_functions.aten_sqrt))
register_aten_op("aten::squeeze.dim")(wrap_for_max_device(aten_functions.aten_squeeze))

register_aten_op("aten::stack")(wrap_for_max_device(aten_functions.aten_stack))

register_aten_op("aten::sub.Tensor")(wrap_for_max_device(aten_functions.aten_sub))
register_aten_op("aten::sum.dim_IntList")(wrap_for_max_device(aten_functions.aten_sum))
register_aten_op("aten::t")(wrap_for_max_device(aten_functions.aten_t))
register_aten_op("aten::tanh")(wrap_for_max_device(aten_functions.aten_tanh))

register_aten_op("aten::transpose.int")(
    wrap_for_max_device(aten_functions.aten_transpose)
)

register_aten_op("aten::tril")(wrap_for_max_device(aten_functions.aten_tril))
register_aten_op("aten::triu")(wrap_for_max_device(aten_functions.aten_triu))

register_aten_op("aten::unbind.int")(wrap_for_max_device(aten_functions.aten_unbind))
register_aten_op("aten::unsqueeze")(wrap_for_max_device(aten_functions.aten_unsqueeze))

register_aten_op("aten::view")(wrap_for_max_device(aten_functions.aten_view))

register_aten_op("aten::where.self")(wrap_for_max_device(aten_functions.aten_where))

register_aten_op("aten::zeros")(wrap_for_max_device(aten_functions.aten_zeros))
