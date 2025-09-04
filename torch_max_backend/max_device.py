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


class MaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data, similar to MyDeviceTensor in trying_stuff.py"""

    IMPLEMENTATIONS = {}

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

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to MAX implementations"""
        if kwargs is None:
            kwargs = {}

        if func in cls.IMPLEMENTATIONS:
            try:

                def super_fn(*args, **kwargs):
                    return super(cls, MaxTensor).__torch_dispatch__(
                        func, types, args, kwargs
                    )

                return cls.IMPLEMENTATIONS[func](super_fn, *args, **kwargs)
            except Exception as e:
                print(f"Error in MaxTensor dispatch for {func}: {e}")
                raise e

        # Try to use the general MAX graph execution
        try:
            return execute_with_max_graph(func, args, kwargs)
        except NotImplementedError:
            raise RuntimeError(
                f"No implementation for 'max_device' for {func}, args={args}, kwargs={kwargs}"
            )


def implements(func):
    """Decorator to register implementations for MaxTensor operations"""

    def _inner_fn(impl):
        MaxTensor.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn

class InputsManager:
    def __init__(self):
        self.input_tensors = list[max.driver.Tensor]() # Fix type
        self.input_specs = list[TensorType]()
        self.placeholder_map = dict[int, int]()

    def collect_tensors(self, arg, path=""):
        if isinstance(arg, MaxTensor):
            if id(arg) not in self.placeholder_map:
                idx = len(self.input_specs)
                # Determine device based on the actual data location
                # For now, use CPU since we're storing numpy arrays

                device_ref = arg._max_data.device

                self.input_specs.append(
                    TensorType(
                        dtype=DType.from_torch(arg._dtype),
                        shape=list(arg._shape),
                        device=device_ref,
                    )
                )
                self.input_tensors.append(arg._max_data)
                self.placeholder_map[id(arg)] = idx
            return f"placeholder_{self.placeholder_map[id(arg)]}"
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


@profile
def execute_with_max_graph(func, args, kwargs):
    """Execute a torch operation using MAX graph compilation"""
    # Collect input tensors and create placeholders
    inputs_manager = InputsManager()
    
    # First pass: collect tensors
    processed_args, processed_kwargs = inputs_manager.collect_tensors_from_args(args, kwargs)

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
    model = session.load(graph)

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
    dtype = torch.float32  # TODO: Get proper dtype from tensor
    return MaxTensor(shape, dtype=dtype, max_data=tensor)


# Register some basic operations (following trying_stuff.py pattern)
@implements(aten.add.Tensor)
def add(super_fn, t1, t2):
    """Implementation of tensor addition"""
    return execute_with_max_graph(aten.add.Tensor, (t1, t2), {})


@implements(aten.mul.Tensor)
def mul(super_fn, t1, t2):
    """Implementation of tensor multiplication"""
    return execute_with_max_graph(aten.mul.Tensor, (t1, t2), {})


@implements(aten.detach.default)
@implements(aten.alias.default)
def detach(super_fn, self):
    """Pass through for detach/alias operations"""
    return super_fn(self)


class MaxDeviceMode(TorchFunctionMode):
    """Mode to handle factory functions and device conversions (following trying_stuff.py pattern)"""

    IMPLEMENTATIONS = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        def super_fn(*args, **kwargs):
            # Disable torch_function to avoid wrapping behavior
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

        if func in self.IMPLEMENTATIONS:
            try:
                return self.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})
            except Exception as e:
                print(f"Error in MaxDeviceMode for {func}: {e}")
                raise e

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
    device = None
    dtype = None

    # Handle positional arguments
    if args:
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str) or isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg
            elif hasattr(arg, "device"):  # Another tensor
                device = arg.device
                dtype = getattr(arg, "dtype", None)
        elif len(args) == 2:
            device, dtype = args

    # Handle keyword arguments
    if "device" in kwargs:
        device = kwargs["device"]
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]

    # If neither device nor dtype specified, pass through
    if device is None and dtype is None:
        return super_fn(self, *args, **kwargs)

    # Handle device conversion to max_device
    if device and (
        device == "max_device"
        or (isinstance(device, str) and device.startswith("max_device"))
    ):
        if isinstance(self, MaxTensor):
            # Already a MaxTensor, handle dtype conversion if needed
            if dtype and dtype != self._dtype:
                # Convert dtype by converting to CPU, changing dtype, then back

                if isinstance(self._max_data, np.ndarray):
                    cpu_tensor = torch.from_numpy(self._max_data.copy())
                    converted = cpu_tensor.to(dtype=dtype)
                    np_data = converted.detach().cpu().numpy()
                    return MaxTensor(converted.shape, dtype=dtype, max_data=np_data)
            return self

        # Convert regular tensor to MaxTensor
        tensor_to_convert = self
        if dtype:
            tensor_to_convert = self.to(dtype=dtype)
        np_data = tensor_to_convert.detach().cpu().numpy()
        return MaxTensor(
            tensor_to_convert.shape, dtype=tensor_to_convert.dtype, max_data=np_data
        )

    elif isinstance(self, MaxTensor):
        # Convert MaxTensor back to regular tensor

        if isinstance(self._max_data, np.ndarray):
            result = torch.from_numpy(self._max_data.copy())
        elif isinstance(self._max_data, max.driver.Tensor):
            # Convert MAX tensor back to numpy then torch
            np_data = self._max_data.to_numpy()
            # Copy to ensure writable array
            result = torch.from_numpy(np_data.copy())
        else:
            # Unknown data type - this should not happen
            raise RuntimeError(f"Unknown MaxTensor data type: {type(self._max_data)}")

        # Apply device and/or dtype conversion
        return result.to(*args, **kwargs)
    else:
        # For non-MaxTensor, just pass through unless it's a dtype-only conversion
        if device is None and dtype:
            # This is a dtype-only conversion, let PyTorch handle it
            return super_fn(self, *args, **kwargs)
        return super_fn(self, *args, **kwargs)


# Factory functions for creating tensors directly on max_device
def get_factory_wrapper(np_func):
    """Wrapper for numpy-based factory functions - following trying_stuff.py pattern"""

    def inner(super_fn, *args, **kwargs):
        # Check device as string, supporting both "max_device" and "max_device:N"
        device_str = str(kwargs.get("device", None))
        if device_str == "max_device" or device_str.startswith("max_device:"):
            # Default dtype depends on the function - arange defaults to int64, others to float32
            if np_func == np.arange:
                default_dtype = torch.int64
            else:
                default_dtype = torch.float32
            dtype = kwargs.get("dtype", default_dtype)

            if np_func == np.random.rand:
                # Special case for rand which takes size as positional args
                np_data = np_func(*args)
            elif np_func == np.arange:
                # arange takes end value and should match the requested dtype
                return execute_with_max_graph(aten.arange, args, kwargs)
            else:
                # For functions like empty
                if args:
                    if isinstance(args[0], list | tuple):
                        shape = args[0]
                    else:
                        shape = args  # multiple args for shape
                    np_data = np_func(shape)
                else:
                    np_data = np_func((1,))  # default shape

            return MaxTensor(np_data.shape, dtype=dtype, max_data=np_data)
        else:
            return super_fn(*args, **kwargs)

    return inner


implements_factory(torch.rand)(get_factory_wrapper(np.random.rand))
implements_factory(torch.empty)(get_factory_wrapper(np.empty))


@implements_factory(torch.arange)
def arange_equivalent(super_fn, *args, device=None, **kwargs):
    """Special handling for torch.arange to support max_device"""
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "max_device":
            kwargs["device"] = torch.device(device)
            return execute_with_max_graph(aten.arange, args, kwargs)

    return super_fn(*args, device=device, **kwargs)

# Add support for torch.tensor with device argument
@implements_factory(torch.tensor)
def tensor(super_fn, data, *args, **kwargs):
    """Handle torch.tensor with max_device"""
    device = kwargs.get("device", None)
    if isinstance(device, str) and (
        device == "max_device" or device.startswith("max_device:")
    ):
        # First create on CPU with proper dtype
        kwargs_cpu = kwargs.copy()
        kwargs_cpu["device"] = "cpu"
        cpu_tensor = super_fn(data, *args, **kwargs_cpu)

        # Then convert to MaxTensor
        np_data = cpu_tensor.numpy()
        return MaxTensor(cpu_tensor.shape, cpu_tensor.dtype, max_data=np_data)

    return super_fn(data, *args, **kwargs)


# Global mode holder
_max_device_mode = None


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
    if _max_device_mode is not None:
        # Already registered
        return

    rename_privateuse_backend()
    register_device_module()
    generate_methods_for_privateuse_backend()
    _max_device_mode = MaxDeviceMode()
    _max_device_mode.__enter__()
