from collections.abc import Callable

import max.driver
import torch
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor as MaxEagerTensor

from torch_max_backend import (
    MAPPING_TORCH_ATEN_TO_MAX,
    get_accelerators,
    torch_max_device_module,
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


class TorchMaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data, similar to MyDeviceTensor in trying_stuff.py"""

    _max_data: MaxEagerTensor

    @staticmethod
    def __new__(cls, size, dtype, max_data=None, requires_grad=False):
        # Use a meta Tensor as the wrapper (following trying_stuff.py pattern)
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = TorchMaxTensor
        return res

    def __init__(
        self, size, dtype, max_data: MaxEagerTensor | None = None, requires_grad=False
    ):
        self._max_data = max_data

    def __repr__(self):
        if hasattr(self, "_max_data"):
            return "MaxTensor(" + repr(self._max_data) + ")"
        return super().__repr__()

    @classmethod
    def _from_max_data(cls, max_data: MaxEagerTensor) -> "TorchMaxTensor":
        shape = tuple(max_data.shape)

        dtype = max_data.dtype.to_torch()
        return TorchMaxTensor(shape, dtype=dtype, max_data=max_data)


def get_max_equivalent(func) -> Callable:
    """Get the MAX equivalent of a torch operation"""
    if func in MAPPING_TORCH_ATEN_TO_MAX:
        return MAPPING_TORCH_ATEN_TO_MAX[func]
    elif (
        hasattr(func, "overloadpacket")
        and func.overloadpacket in MAPPING_TORCH_ATEN_TO_MAX
    ):
        return MAPPING_TORCH_ATEN_TO_MAX[func.overloadpacket]
    else:
        raise NotImplementedError(
            f"Operation {func} not implemented for TorchMaxTensor"
        )


def get_ordered_accelerators():
    """Get accelerators ordered with GPUs first, then CPU last"""
    accelerators = list(get_accelerators())

    # Separate GPU and CPU accelerators
    gpu_accelerators = [acc for acc in accelerators if acc.label == "gpu"]
    cpu_accelerators = [acc for acc in accelerators if acc.label == "cpu"]

    # Order: GPUs first, then CPU last
    return gpu_accelerators + cpu_accelerators


def find_equivalent_torch_device(device: max.driver.Device) -> torch.device:
    if device.label == "cpu":
        return torch_max_device_module.cpu()
    elif device.label == "gpu":
        return torch.device(f"max_device:{device.id}")


def find_equivalent_max_device(device: torch.device) -> max.driver.Device:
    """Find the equivalent MAX device for a given torch device

    Device mapping:
    - max_device:0 (or max_device) -> First GPU (or CPU if no GPUs)
    - max_device:1, max_device:2, ... -> Additional GPUs
    - max_device:<last_index> -> CPU device
    """
    ordered_accelerators = get_ordered_accelerators()

    if device.type == "max_device":
        # max_device with specific index
        if device.index is None:
            # Default to first accelerator (first GPU or CPU if no GPUs)
            return ordered_accelerators[0]
        else:
            if device.index < len(ordered_accelerators):
                return ordered_accelerators[device.index]
            else:
                raise ValueError(f"Invalid max_device index {device.index}")
    elif device.type == "cpu":
        # Find CPU accelerator (should be last in ordered list)
        for acc in reversed(ordered_accelerators):  # Check from the end
            if acc.label == "cpu":
                return acc
        # If no CPU found, return last accelerator as fallback
        return ordered_accelerators[-1]
    elif device.type in ("cuda", "hip"):
        # Find GPU accelerator (should be first in ordered list)
        # TODO: allow setting the default device index globally like with cuda
        gpu_index = device.index if device.index is not None else 0
        gpu_accelerators = [acc for acc in ordered_accelerators if acc.label == "gpu"]
        if gpu_index < len(gpu_accelerators):
            return gpu_accelerators[gpu_index]
        raise RuntimeError(f"GPU index {gpu_index} not available in MAX")
    else:
        raise NotImplementedError(f"Cannot convert {device.type} to MAX device")


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
def max_device__copy_from(self, dest):
    if self.device.type == "max_device" and dest.device.type == "cpu":
        x = torch.from_numpy(self._max_data.to_numpy())
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
