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

register_aten_op("aten::add.Tensor")(wrap_for_max_device(aten_functions.aten_add))
register_aten_op("aten::sub.Tensor")(wrap_for_max_device(aten_functions.aten_sub))
register_aten_op("aten::mul.Tensor")(wrap_for_max_device(aten_functions.aten_mul))
register_aten_op("aten::sum.dim_IntList")(wrap_for_max_device(aten_functions.aten_sum))
register_aten_op("aten::sqrt")(wrap_for_max_device(aten_functions.aten_sqrt))
register_aten_op("aten::arange")(wrap_for_max_device(aten_functions.aten_arange))
register_aten_op("aten::full")(wrap_for_max_device(aten_functions.aten_full))
register_aten_op("aten::ones")(wrap_for_max_device(aten_functions.aten_ones))
register_aten_op("aten::zeros")(wrap_for_max_device(aten_functions.aten_zeros))
register_aten_op("aten::pow.Tensor_Scalar")(
    wrap_for_max_device(aten_functions.aten_pow)
)
register_aten_op("aten::max_pool2d_with_indices")(
    wrap_for_max_device(aten_functions.aten_max_pool2d_with_indices)
)
register_aten_op("aten::_adaptive_avg_pool2d")(
    wrap_for_max_device(aten_functions.aten__adaptive_avg_pool2d)
)

register_aten_op("aten::convolution")(
    wrap_for_max_device(aten_functions.aten_convolution)
)
register_aten_op("aten::t")(wrap_for_max_device(aten_functions.aten_t))
register_aten_op("aten::addmm")(wrap_for_max_device(aten_functions.aten_addmm))
register_aten_op("aten::view")(wrap_for_max_device(aten_functions.aten_view))
register_aten_op("aten::detach")(wrap_for_max_device(aten_functions.aten_detach))
register_aten_op("aten::relu")(wrap_for_max_device(aten_functions.aten_relu))


@register_aten_op("aten::relu_")
def max_device_relu_(self: TorchMaxTensor) -> TorchMaxTensor:
    # in-place relu
    self._max_data = aten_functions.aten_relu(self._max_data)
    return self


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


@register_aten_op("aten::_copy_from")
def max_device__copy_from(self: TorchMaxTensor, dest: TorchMaxTensor) -> TorchMaxTensor:
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
