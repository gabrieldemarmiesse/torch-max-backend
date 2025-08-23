import torch
import max.graph.ops as ops
from max.dtype import DType
from max.graph.type import DeviceRef
from max.graph import Graph, TensorType
from max import engine
import numpy as np
from torch_max_backend import get_accelerators
from torch_max_backend import MAPPING_TORCH_ATEN_TO_MAX
import max.driver
from torch.ops import aten
from collections.abc import Callable
from typing import Literal


class Placeholder:
    def __init__(self, index):
        self.index = index


def get_max_equivalent(func) -> Callable:
    if func in MAPPING_TORCH_ATEN_TO_MAX:
        return MAPPING_TORCH_ATEN_TO_MAX[func]
    elif func.overloadpacket in MAPPING_TORCH_ATEN_TO_MAX:
        return MAPPING_TORCH_ATEN_TO_MAX[func.overloadpacket]
    else:
        raise NotImplementedError(f"Operation {func} not implemented for MaxTensor")


class Dispatcher:
    def __init__(self):
        self.input_tensors = []
        self.list_of_input_specs = []
        self.graph = None

    def traversal(self, arg):
        if isinstance(arg, torch.Tensor):
            # First pass on the arguments
            # create an input spec
            input_type = TensorType(
                dtype=DType.from_torch(arg.dtype),
                shape=list(arg.shape),
                device=DeviceRef.GPU(),
            )
            self.list_of_input_specs.append(input_type)
            self.input_tensors.append(arg._max_data)
            return Placeholder(len(self.list_of_input_specs) - 1)
        elif isinstance(arg, Placeholder):
            # Second pass on the arguments
            return self.graph.inputs[arg.index]
        elif isinstance(arg, int | float):
            return arg
        elif isinstance(arg, list):
            return [self.traversal(x) for x in arg]
        elif isinstance(arg, tuple):
            return tuple(self.traversal(x) for x in arg)
        elif isinstance(arg, dict):
            return {k: self.traversal(v) for k, v in arg.items()}
        elif isinstance(arg, torch.dtype):
            return arg
        elif isinstance(arg, torch.layout):
            return arg
        elif isinstance(arg, torch.device):
            return arg
        elif arg is None:
            return arg
        else:
            raise NotImplementedError(f"Argument type {type(arg)} not supported")

    def run_with_max_graph(self, tensor, func, types, args, kwargs: dict):
        new_args_with_placeholders = self.traversal(args)
        new_kwargs_with_placeholders = self.traversal(kwargs)
        with Graph("add_graph", input_types=self.list_of_input_specs) as graph:
            self.graph = graph
            replaced_args = self.traversal(new_args_with_placeholders)
            replaced_kwargs = self.traversal(new_kwargs_with_placeholders)

            func_to_use = get_max_equivalent(func)
            out = func_to_use(*replaced_args, **replaced_kwargs)
            # can be a tuple or a single tensor
            if isinstance(out, tuple):
                graph.output(*out)
                is_tuple = True
            else:
                graph.output(out)
                is_tuple = False

            session = engine.InferenceSession(devices=list(get_accelerators()))
            model = session.load(graph)
            output = model.execute(*self.input_tensors)

            if is_tuple:
                return tuple(make_max_tensor_from_max(o) for o in output)
            else:
                return make_max_tensor_from_max(output[0])

    @staticmethod
    def execute_with_max(tensor, func, types, args, kwargs=None):
        dispatcher = Dispatcher()
        return dispatcher.run_with_max_graph(tensor, func, types, args, kwargs)


class MaxTensor(torch.Tensor):
    """Custom tensor subclass that holds MAX engine data"""

    @staticmethod
    def __new__(cls, data, max_data=None, device=None):
        # Create tensor with proper device
        if isinstance(data, torch.Tensor):
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                data.shape,
                dtype=data.dtype,
                device=device or torch.device("max_gpu"),
                requires_grad=data.requires_grad,
            )
        else:
            # data is a shape tuple
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                data,
                dtype=torch.float32,  # TODO fix this
                device=device or torch.device("max_gpu"),
                requires_grad=False,
            )
        r._max_data = max_data
        return r

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == aten._to_copy.default and kwargs.get("device") != torch.device(
            "max_gpu"
        ):
            if kwargs.get("device") == torch.device("cpu"):
                # TODO: transfer on the cpu with max²²
                return torch.from_dlpack(args[0]._max_data).to("cpu")
            else:
                raise NotImplementedError("Transfer to non-CPU devices not implemented")
        return Dispatcher.execute_with_max(self, func, types, args, kwargs)


def get_max_device_module(device_name: Literal["max_cpu", "max_gpu"]):
    class MaxDeviceModule:
        @staticmethod
        def _is_in_bad_fork():
            return False

        @staticmethod
        def manual_seed_all(seed):
            np.random.seed(seed)

        @staticmethod
        def device_count():
            return 1  # TODO: change

        @staticmethod
        def get_rng_state(device=None):
            return torch.tensor(np.random.get_state()[1])

        @staticmethod
        def set_rng_state(new_state, device=None):
            if isinstance(new_state, torch.Tensor):
                new_state = new_state.cpu().numpy()
            np_state = ("MT19937", new_state, 624, 0, 0.0)
            np.random.set_state(np_state)

        @staticmethod
        def is_available():
            return True  # TODO change

        @staticmethod
        def current_device():
            return 0  # TODO change

        @staticmethod
        def get_amp_supported_dtype():
            return [torch.float16, torch.bfloat16]  # TODO change

        # TODO: necessary?
        def max_gpu(self):
            print("hello")


class Custom:
    def __init__(self, t):
        self.t = t


def register_max_ops(device_name: Literal["max_cpu", "max_gpu"]):
    private_use_name = "PrivateUse1" if device_name == "max_cpu" else "PrivateUse2"
    max_graph_device = DeviceRef.CPU() if device_name == "max_cpu" else DeviceRef.GPU()

    @torch.library.impl("aten::arange", private_use_name)
    def arange_max(end, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: arange called with end={end}, device={device}")
        if dtype is None:
            dtype = torch.int64
        # Create the computation graph
        with Graph("arange_graph", input_types=tuple()) as graph:
            out = ops.range(
                0, end, 1, device=max_graph_device, dtype=DType.from_torch(dtype)
            )
            graph.output(out)

        # Execute on MAX engine
        session = engine.InferenceSession(devices=list(get_accelerators()))
        model = session.load(graph)
        output = model.execute()[0]

        # Return MaxTensor with GPU data
        result = MaxTensor((int(end),), max_data=output, device=torch.device("max_gpu"))
        print(f"DEBUG: Created MaxTensor with shape {result.shape} (data kept on GPU)")
        return result


def make_max_tensor_from_max(tensor: max.driver.Tensor) -> MaxTensor:
    """Convert a max.driver.Tensor to a MaxTensor"""
    shape = tuple(tensor.shape)
    max_data = tensor
    return MaxTensor(shape, max_data=max_data, device=torch.device("max_gpu"))


def rename_privateuse_backend(device_name: Literal["max_cpu", "max_gpu"]):
    if device_name == "max_cpu":
        torch.utils.rename_privateuse1_backend("max_cpu")
    elif device_name == "max_gpu":
        torch.utils.rename_privateuse2_backend("max_gpu")


def generate_methods_for_privateuse_backend(device_name: Literal["max_cpu", "max_gpu"]):
    if device_name == "max_cpu":
        torch.utils.generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_packed_sequence=True,
            for_storage=False,
        )
    elif device_name == "max_gpu":
        torch.utils.generate_methods_for_privateuse2_backend(
            for_tensor=True,
            for_module=True,
            for_packed_sequence=True,
            for_storage=False,
        )


def _register(device_name: Literal["max_cpu", "max_gpu"]):
    device_module = get_max_device_module(device_name)
    rename_privateuse_backend(device_name)

    torch._register_device_module(device_name, device_module)

    register_max_ops(device_name)

    generate_methods_for_privateuse_backend(device_name)


registered = False


def register_max_devices():
    global registered
    if registered:
        return
    _register("max_cpu")
    _register("max_gpu")
    registered = True
