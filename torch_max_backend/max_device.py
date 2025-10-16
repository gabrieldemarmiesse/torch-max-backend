from collections.abc import Callable

import max.driver
import numpy as np
import torch
from max import engine
from max.graph import Graph, TensorType
from torch.ops import aten
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

from torch_max_backend import (
    MAPPING_TORCH_ATEN_TO_MAX,
    get_accelerators,
    torch_max_device_module,
)


class UseStockImplementation(Exception):
    pass


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
        raise NotImplementedError(f"Operation {func} not implemented for MaxTensor")


class MaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data, similar to MyDeviceTensor in trying_stuff.py"""

    _max_data: max.driver.Tensor

    @staticmethod
    def __new__(cls, size, dtype, max_data=None, requires_grad=False):
        # Use a meta Tensor as the wrapper (following trying_stuff.py pattern)
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = MaxTensor
        return res

    def __init__(
        self,
        size,
        dtype,
        max_data: max.driver.Tensor | None = None,
        requires_grad=False,
    ):
        self._max_data = max_data

    def __repr__(self):
        if hasattr(self, "_max_data"):
            return "MaxTensor(" + repr(self._max_data.to_numpy()) + ")"
        return super().__repr__()

    def __sub__(self, other):
        if hasattr(self, "_max_data"):
            return torch.sub(self, other)
        return super().__sub__(self, other)


def max_device_aten_add(input, other, alpha=1):
    return execute_with_max_graph(aten.add, (input, other, alpha), {})


torch.library.impl("aten::add.Tensor", "privateuseone")(max_device_aten_add)


def max_device_aten_sub(input, other, alpha=1):
    return execute_with_max_graph(aten.sub, (input, other, alpha), {})


torch.library.impl("aten::sub.Tensor", "privateuseone")(max_device_aten_sub)


def max_device_aten_mul(input, other):
    return execute_with_max_graph(aten.mul, (input, other), {})


torch.library.impl("aten::mul.Tensor", "privateuseone")(max_device_aten_mul)


def max_device_aten_sum(
    input,
    dim: list[int] | int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    return execute_with_max_graph(aten.sum, (input, dim, keepdim), dict(dtype=dtype))


torch.library.impl("aten::sum.dim_IntList", "privateuseone")(max_device_aten_sum)


def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)

    else:
        return obj


class InputsManager:
    def __init__(self):
        self.input_tensors = list[max.driver.Tensor]()  # Fix type
        self.input_specs = list[TensorType]()
        self.placeholder_map = dict[int, int]()

    def collect_tensors(self, arg, path=""):
        if isinstance(arg, MaxTensor):
            if id(arg) not in self.placeholder_map:
                idx = len(self.input_specs)
                input_type = TensorType(
                    dtype=arg._max_data.dtype,
                    shape=list(arg._max_data.shape),
                    device=arg._max_data.device,
                )
                self.input_specs.append(input_type)
                self.input_tensors.append(arg._max_data)
                self.placeholder_map[id(arg)] = idx
            # important for hashing
            return f"placeholder_{self.placeholder_map[id(arg)]}_{self.input_specs[self.placeholder_map[id(arg)]]}"
        elif isinstance(arg, list | tuple):
            return type(arg)(
                self.collect_tensors(x, f"{path}[{i}]") for i, x in enumerate(arg)
            )
        elif isinstance(arg, dict):
            return {k: self.collect_tensors(v, f"{path}[{k}]") for k, v in arg.items()}
        else:
            return arg

    def collect_tensors_from_args(self, args, kwargs):
        return self.collect_tensors(args), self.collect_tensors(kwargs)


models_cache = {}


def create_model_with_cache(
    inputs_manager, processed_args, processed_kwargs, func
) -> tuple[engine.Model, bool]:
    cache_key = hash(make_hashable((func, processed_args, processed_kwargs)))
    if cache_key in models_cache:
        return models_cache[cache_key]
    model, is_tuple = create_model(
        inputs_manager, processed_args, processed_kwargs, func
    )
    models_cache[cache_key] = (model, is_tuple)
    return model, is_tuple


def create_model(inputs_manager, processed_args, processed_kwargs, func):
    # Build and execute graph
    with Graph("max_op_graph", input_types=inputs_manager.input_specs) as graph:
        # Replace placeholders with actual graph inputs
        def replace_placeholders(arg):
            if isinstance(arg, str) and arg.startswith("placeholder_"):
                idx = int(arg.split("_")[1])
                return graph.inputs[idx]
            elif isinstance(arg, list | tuple):
                return type(arg)(replace_placeholders(x) for x in arg)
            elif isinstance(arg, dict):
                return {k: replace_placeholders(v) for k, v in arg.items()}
            else:
                return arg

        graph_args, graph_kwargs = replace_placeholders(
            (processed_args, processed_kwargs)
        )

        # Get MAX equivalent function and execute
        func_to_use = get_max_equivalent(func)

        out = func_to_use(*graph_args, **graph_kwargs)
        # Handle output
        if isinstance(out, tuple):
            graph.output(*out)
            is_tuple = True
        else:
            graph.output(out)
            is_tuple = False

    # Execute the graph
    session = engine.InferenceSession(devices=get_ordered_accelerators())
    return session.load(graph), is_tuple


def execute_with_max_graph(func, args, kwargs):
    """Execute a torch operation using MAX graph compilation"""
    # Collect input tensors and create placeholders
    inputs_manager = InputsManager()

    # First pass: collect tensors
    processed_args, processed_kwargs = inputs_manager.collect_tensors_from_args(
        args, kwargs
    )
    model, is_tuple = create_model_with_cache(
        inputs_manager, processed_args, processed_kwargs, func
    )
    # Convert input tensors to proper MAX format
    max_inputs = []
    for tensor_data in inputs_manager.input_tensors:
        if isinstance(tensor_data, np.ndarray):
            # For numpy arrays, we need to pass them directly
            max_inputs.append(tensor_data)
        else:
            # Already in proper format
            max_inputs.append(tensor_data)

    output = model.execute(*max_inputs)

    # Convert output back to MaxTensor
    if is_tuple:
        return tuple(make_max_tensor_from_max(o) for o in output)
    else:
        return make_max_tensor_from_max(output[0])


def make_max_tensor_from_max(tensor: max.driver.Tensor) -> MaxTensor:
    """Convert a max.driver.Tensor to a MaxTensor"""
    shape = tuple(tensor.shape)

    dtype = tensor.dtype.to_torch()
    return MaxTensor(shape, dtype=dtype, max_data=tensor)


def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    a = execute_with_max_graph(
        aten.empty_strided,
        (),
        dict(
            size=size,
            stride=stride,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        ),
    )
    return a


torch.library.impl("aten::empty_strided.memory_format", "privateuseone")(empty_strided)
torch.library.impl("aten::empty_strided", "privateuseone")(empty_strided)


def max_device__copy_from(self, dest):
    if self.device.type == "max_device" and dest.device.type == "cpu":
        x = torch.from_numpy(self._max_data.to_numpy())
        dest.copy_(x)
        return dest

    elif self.device.type == "cpu" and dest.device.type == "max_device":
        self = make_max_tensor_from_max(max.driver.Tensor.from_dlpack(self.detach()))
        dest._max_data = self._max_data.to(dest._max_data.device)
        return dest
    else:
        raise RuntimeError(
            f"invalid configuration {self.device.type}, {dest.device.type}"
        )


torch.library.impl("aten::_copy_from", "privateuseone")(max_device__copy_from)


def max_device_empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    print("called memory format")
    return execute_with_max_graph(
        aten.empty.memory_format,
        (),
        dict(
            size=size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            memory_format=memory_format,
        ),
    )


torch.library.impl("aten::empty.memory_format", "privateuseone")(
    max_device_empty_memory_format
)


def max_device_aten_sqrt(x):
    return execute_with_max_graph(aten.sqrt, (x,), {})


torch.library.impl("aten::sqrt", "privateuseone")(max_device_aten_sqrt)


def max_device_aten_arange_start_out(
    start,
    end=None,
    step=1,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
):
    return execute_with_max_graph(
        aten.arange,
        (),
        dict(
            start=start,
            end=end,
            step=step,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        ),
    )


torch.library.impl("aten::arange", "privateuseone")(max_device_aten_arange_start_out)


def max_device_aten_pow(input, exponent):
    return execute_with_max_graph(aten.pow, (input, exponent), {})


torch.library.impl("aten::pow.Tensor_Scalar", "privateuseone")(max_device_aten_pow)

_registered = False


def register_max_devices():
    """Enable the max_device globally"""
    global _registered
    if _registered:
        # Already registered
        return

    _setup_privateuseone_for_python_backend("max_device")
    _registered = True
