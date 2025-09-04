import torch
from torch.overrides import TorchFunctionMode
from max.dtype import DType
from max.graph import Graph, TensorType
from max import engine
import max.driver
from torch.ops import aten
from collections.abc import Callable
from torch_max_backend import get_accelerators, MAPPING_TORCH_ATEN_TO_MAX
import numpy as np
from max.graph.type import DeviceRef
from torch_max_backend import torch_max_device_module
from line_profiler import profile
from torch.utils._python_dispatch import TorchDispatchMode
import os
from max.dtype import DType
from max.torch.torch import max_device_ref
import os
import max.graph.ops as max_ops
from max.dtype import DType
from torch.ops import aten
import torch
from max.graph.type import DeviceRef
import max.graph.type as max_type
from max.graph import StaticDim, Dim, TensorValue
import numpy as np
import math
from torch._decomp import core_aten_decompositions
from torch._ops import OpOverloadPacket, OpOverload
from typing import Literal
from torch_max_backend.flags import verbose_enabled
from max.graph import TensorType
from torch_max_backend.aten_functions import DECOMPOSITION_TABLE

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
        return torch.device(f"max_device:{device.index}")

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

FILTERS = {}

def map_to(func):
    def decorator(func_to_map):
        if os.environ.get("TORCH_MAX_BACKEND_BEARTYPE", "1") == "1":
            from beartype import beartype
            func_to_map = beartype(func_to_map)

        FILTERS[func] = func_to_map
        return func_to_map

    return decorator


@map_to(aten.arange)
def aten_arange(
    start,
    end= None,
    step = 1,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,):
    if device is None:
        raise UseStockImplementation()
    if device is not None and device.type != "max_device":
        raise UseStockImplementation()


class MaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data, similar to MyDeviceTensor in trying_stuff.py"""


    @staticmethod
    def __new__(cls, shape, dtype, max_data=None, requires_grad=False):
        # Use a meta Tensor as the wrapper (following trying_stuff.py pattern)
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=dtype, device="meta"),
            require_grad=requires_grad,
        )

    def __init__(self, shape, dtype, max_data: max.driver.Tensor | None = None, requires_grad=False):
        # Store the MAX engine data
        self._max_data = max_data
        self._shape = shape
        self._dtype = dtype

    @property
    def device(self):
        return find_equivalent_torch_device(self._max_data.device)

    def __repr__(self):
        st = super().__repr__()
        st = st.replace("device='meta'", "device='max_device'")
        # Could add more detailed representation if needed
        return st


class DispatchMax(TorchDispatchMode):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to MAX implementations"""
        if kwargs is None:
            kwargs = {}

        # Allow us to check if Pytorch is trying to build a normal tensor
        if func.overloadpacket in FILTERS:
            try:
                FILTERS[func.overloadpacket](*args, **kwargs)
            except UseStockImplementation:
                return func(*args, **kwargs)
        
        # If using only normal torch tensors, we let torch handle the computing too
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, MaxTensor):
                return func(*args, **kwargs)

        # Try to use the general MAX graph execution
        try:
            result = execute_with_max_graph(func, args, kwargs)
        except NotImplementedError:
            raise RuntimeError(
                f"No implementation for 'max_device' for {func}, args={args}, kwargs={kwargs}"
            )
        return result


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
        self.input_tensors = list[max.driver.Tensor]() # Fix type
        self.input_specs = list[TensorType]()
        self.placeholder_map = dict[int, int]()

    def collect_tensors(self, arg, path=""):
        if isinstance(arg, MaxTensor):
            if id(arg) not in self.placeholder_map:
                idx = len(self.input_specs)
                input_type = TensorType(
                        dtype=DType.from_torch(arg._dtype),
                        shape=list(arg._shape),
                        device=arg._max_data.device,
                    )
                self.input_specs.append(input_type)
                self.input_tensors.append(arg._max_data)
                self.placeholder_map[id(arg)] = idx
            # important for hashing
            return f"placeholder_{self.placeholder_map[id(arg)]}_{input_type}"
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

        graph_args, graph_kwargs = replace_placeholders((processed_args, processed_kwargs))

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
    processed_args, processed_kwargs = inputs_manager.collect_tensors_from_args(args, kwargs)
    cache_key = hash(make_hashable((func, processed_args, processed_kwargs)))
    if cache_key in models_cache:
        model, is_tuple = models_cache[cache_key]
    else:
        model, is_tuple = create_model(inputs_manager, processed_args, processed_kwargs, func)
        models_cache[cache_key] = (model, is_tuple)
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


class MaxDeviceMode(TorchFunctionMode):
    """Mode to handle factory functions and device conversions (following trying_stuff.py pattern)"""

    IMPLEMENTATIONS = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        def super_fn(*args, **kwargs):
            # Disable torch_function to avoid wrapping behavior
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

        if func in self.IMPLEMENTATIONS:
            return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})

        # No-op for non-factory functions
        return super_fn(*args, **kwargs or {})


def implements_factory(func):
    """Decorator to register factory function implementations"""

    def _inner_fn(impl):
        MaxDeviceMode.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn





@implements_factory(torch.Tensor.to)
def to(super_fn, self, *args, **kwargs):
    """Handle tensor.to() conversions - supporting device, dtype, and combined calls"""
    # Parse arguments - .to() can be called with device, dtype, or both
    device = kwargs.get("device")
    if len(args) >= 1:
        if isinstance(args[0], str) or isinstance(args[0], torch.device):
            device = args[0]

    result = self.to(*args, **kwargs)

    # Should we go back to pure pytorch? We check it here:
    if isinstance(device, str):
        device = torch.device(device)
    if device is not None and device.type != "max_device" and isinstance(result, MaxTensor):
        # We have to convert it to a pure pytorch tensor
        result = torch.from_dlpack(result._max_data)
    return result


# Global mode holder
_max_device_mode = None
_max_device_function_mode = None

def rename_privateuse_backend():
    torch.utils.rename_privateuse1_backend("max_device")


def register_device_module():
    torch._register_device_module("max_device", torch_max_device_module)


def generate_methods_for_privateuse_backend():
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_packed_sequence=True, for_storage=False
    )


def register_max_devices():
    """Enable the max_device globally"""
    global _max_device_mode
    global _max_device_function_mode
    if _max_device_mode is not None:
        # Already registered
        return

    rename_privateuse_backend()
    register_device_module()
    generate_methods_for_privateuse_backend()
    _max_device_mode = DispatchMax()
    _max_device_function_mode = MaxDeviceMode()
    _max_device_mode.__enter__()
    _max_device_function_mode.__enter__()

