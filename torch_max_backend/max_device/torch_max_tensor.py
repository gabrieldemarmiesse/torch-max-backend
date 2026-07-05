import functools
from collections.abc import Callable
from typing import no_type_check

import max.driver
import torch
from max.driver import CPU
from max.experimental.tensor import Tensor as MaxEagerTensor
from max.experimental.torch import max_dtype_to_torch

from torch_max_backend.max_device import torch_max_device_module


class TorchMaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data, similar to MyDeviceTensor in trying_stuff.py

    The MAX data can be stored in two forms: an eager `MaxEagerTensor`
    (`_max_data`), or — for outputs of the fast kernel path — just the
    realized `max.driver.Buffer` (`_buffer`). The `_max_data` property
    builds the MaxEagerTensor wrapper on first access, so the many
    fast-path tensors that only ever feed other fast ops never pay for
    one (~1.7 µs per construction).
    """

    @staticmethod
    def __new__(cls, size, dtype, max_data=None, requires_grad=False):
        # Use a meta Tensor as the wrapper (following trying_stuff.py pattern)
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = TorchMaxTensor
        return res

    @no_type_check
    def __init__(self, size, dtype, max_data=None, requires_grad=False):
        self._max_data_ = max_data
        self._buffer = None

    @property
    @no_type_check
    def _max_data(self) -> MaxEagerTensor:
        max_data = self._max_data_
        if max_data is None and self._buffer is not None:
            max_data = MaxEagerTensor(storage=self._buffer)
            self._max_data_ = max_data
        return max_data

    @_max_data.setter
    @no_type_check
    def _max_data(self, value):
        self._max_data_ = value
        self._buffer = None

    def __repr__(self):
        if hasattr(self, "_max_data_"):
            return "MaxTensor(" + repr(self._max_data) + ")"
        return super().__repr__()

    @property
    def device(self):
        if hasattr(self, "_max_data_"):
            if self._buffer is not None:
                max_device = self._buffer.device
            else:
                max_device = self._max_data.device
            if max_device == CPU():
                return torch_max_device_module.cpu()
            else:
                return torch.device(f"max_device:{max_device.id}")
        return super().device

    @classmethod
    @no_type_check
    def _from_buffer(cls, buffer: max.driver.Buffer) -> "TorchMaxTensor":
        """Wrap a realized contiguous driver buffer (fast path outputs)."""
        result = TorchMaxTensor(tuple(buffer.shape), max_dtype_to_torch(buffer.dtype))
        result._buffer = buffer
        return result

    @classmethod
    @no_type_check
    def _from_max_data(cls, max_data: MaxEagerTensor) -> "TorchMaxTensor":
        if max_data.real:
            # The driver buffer's shape/dtype are plain C-level values;
            # max_data.shape would build graph Shape/Dim wrappers per call.
            buffer = max_data.driver_tensor
            shape = tuple(buffer.shape)
            dtype = max_dtype_to_torch(buffer.dtype)
        else:
            shape = tuple(max_data.shape)
            dtype = max_dtype_to_torch(max_data.dtype)
        return TorchMaxTensor(shape, dtype=dtype, max_data=max_data)

    __torch_function__ = torch._C._disabled_torch_function_impl


def get_max_equivalent(func) -> Callable:
    """Get the MAX equivalent of a torch operation"""
    from torch_max_backend.torch_compile_backend.compiler import (
        MAPPING_TORCH_ATEN_TO_MAX,
    )

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


@functools.cache
def get_ordered_accelerators():
    """Get accelerators ordered with GPUs first, then CPU last"""
    from torch_max_backend.torch_compile_backend.compiler import get_accelerators

    accelerators = list(get_accelerators())

    # Separate GPU and CPU accelerators
    gpu_accelerators = [acc for acc in accelerators if acc.label == "gpu"]
    cpu_accelerators = [acc for acc in accelerators if acc.label == "cpu"]

    # Order: GPUs first, then CPU last
    return gpu_accelerators + cpu_accelerators


def find_equivalent_torch_device(device: max.driver.Device) -> torch.device:
    if device.label == "cpu":
        return torch.device("cpu")
    elif device.label == "gpu":
        return torch.device(f"max_device:{device.id}")


@functools.cache
@no_type_check
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
