from collections.abc import Callable

import max.driver
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor as MaxEagerTensor

from torch_max_backend.torch_max_tensor import (
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


@register_aten_op("aten::add.Tensor")
def max_device_aten_add(
    input: TorchMaxTensor, other: TorchMaxTensor, alpha=1
) -> TorchMaxTensor:
    return TorchMaxTensor._from_max_data(input._max_data + other._max_data * alpha)


@register_aten_op("aten::sub.Tensor")
def max_device_aten_sub(
    input: TorchMaxTensor, other: TorchMaxTensor, alpha=1
) -> TorchMaxTensor:
    return TorchMaxTensor._from_max_data(input._max_data - other._max_data * alpha)


@register_aten_op("aten::mul.Tensor")
def max_device_aten_mul(input: TorchMaxTensor, other: TorchMaxTensor) -> TorchMaxTensor:
    return TorchMaxTensor._from_max_data(input._max_data * other._max_data)


@register_aten_op("aten::sum.dim_IntList")
def max_device_aten_sum(
    input: TorchMaxTensor,
    dim: list[int] | int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> TorchMaxTensor:
    pass


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


@register_aten_op("aten::sqrt")
def max_device_aten_sqrt(x: TorchMaxTensor):
    return TorchMaxTensor._from_max_data(F.sqrt(x._max_data))


@register_aten_op("aten::arange")
def max_device_aten_arange_start_out(
    start,
    end=None,
    step=1,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    max_eager_tensor = F.range(start, end, step, dtype=dtype, device=device)
    return TorchMaxTensor._from_max_data(max_eager_tensor)


@register_aten_op("aten::full")
def max_device_aten_full(
    size: list[int],
    fill_value: int | float,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    max_eager_tensor = MaxEagerTensor.full(size, fill_value, dtype=dtype, device=device)
    return TorchMaxTensor._from_max_data(max_eager_tensor)


@register_aten_op("aten::ones")
def max_device_aten_ones(
    size: list[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    max_eager_tensor = MaxEagerTensor.ones(size, dtype=dtype, device=device)
    return TorchMaxTensor._from_max_data(max_eager_tensor)


@register_aten_op("aten::zeros")
def max_device_aten_zeros(
    size: list[int],
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> TorchMaxTensor:
    dtype = torch.float32 if dtype is None else dtype
    dtype = DType.from_torch(dtype)
    device = find_equivalent_max_device(device)
    max_eager_tensor = MaxEagerTensor.zeros(size, dtype=dtype, device=device)
    return TorchMaxTensor._from_max_data(max_eager_tensor)


@register_aten_op("aten::pow.Tensor_Scalar")
def max_device_aten_pow(input: TorchMaxTensor, exponent) -> TorchMaxTensor:
    return TorchMaxTensor._from_max_data(F.pow(input._max_data, exponent))
