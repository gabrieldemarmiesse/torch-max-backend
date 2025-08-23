import torch
import max.graph.ops as ops
from max.dtype import DType
from max.graph.type import DeviceRef
from max.graph import Graph, TensorType
from max import engine
from torch_max_backend import get_accelerators
from torch_max_backend import MAPPING_TORCH_ATEN_TO_MAX
import max.driver
from torch.ops import aten
from collections.abc import Callable
from max.torch.torch import max_device_ref
from torch_max_backend import torch_max_device_module
from line_profiler import profile

device_name = "max_device"


def current_torch_device() -> torch.device:
    return torch.device(f"max_device:{torch_max_device_module.current_device()}")


def torch_device_to_max_device(x: torch.device) -> DeviceRef:
    if x.type == "max_device":
        if x.index is None:
            index = torch_max_device_module.current_device()
        else:
            index = x.index
        if index == torch_max_device_module.device_count() - 1:
            return DeviceRef.CPU()
        else:
            return DeviceRef.GPU(index)
    else:
        return max_device_ref(x)


def max_device_to_torch_device(x: max.driver.Device) -> torch.device:
    if x.label == "cpu":
        return torch_max_device_module.cpu()
    elif x.label == "gpu":
        return torch.device(f"max_device:{x.id}")
    else:
        raise ValueError(f"unrecognized device type {x.label}")


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

    @profile
    def traversal(self, arg):
        if isinstance(arg, torch.Tensor):
            # First pass on the arguments
            # create an input spec
            input_type = TensorType(
                dtype=DType.from_torch(arg.dtype),
                shape=list(arg.shape),
                device=torch_device_to_max_device(arg.device),
            )
            self.list_of_input_specs.append(input_type)
            self.input_tensors.append(getattr(arg, "_max_data", arg))
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

    @profile
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
    @profile
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
            raise ValueError("data should not be a torch.Tensor")
        else:
            # data is a shape tuple
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                data,
                dtype=torch.float32,  # TODO fix this
                device=device,
                requires_grad=False,
            )
        r._max_data = max_data
        return r

    @profile
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == aten._to_copy.default:
            device = kwargs.get("device")
            if device.type != "max_device":
                if kwargs.get("device") == torch.device("cpu"):
                    # TODO: transfer on the cpu with max
                    return torch.from_dlpack(args[0]._max_data).to("cpu")
                else:
                    raise NotImplementedError(
                        "Transfer to non-CPU devices not implemented"
                    )
        return Dispatcher.execute_with_max(self, func, types, args, kwargs)


def register_max_ops():
    private_use_name = "PrivateUse1"

    @torch.library.impl("aten::arange", private_use_name)
    @profile
    def arange_max(end, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: arange called with end={end}, device={device}")
        if dtype is None:
            dtype = torch.int64
        if device is None:
            device = current_torch_device()
            # Create the computation graph
        with Graph("arange_graph", input_types=tuple()) as graph:
            out = ops.range(
                0,
                end,
                1,
                device=torch_device_to_max_device(device),
                dtype=DType.from_torch(dtype),
            )
            graph.output(out)

        # Execute on MAX engine
        accelerators = list(get_accelerators())
        session = engine.InferenceSession(devices=accelerators)
        model = session.load(graph)
        output = model.execute()[0]

        # Return MaxTensor with GPU data
        result = make_max_tensor_from_max(output)
        print(f"DEBUG: Created MaxTensor with shape {result.shape} (data kept on GPU)")
        return result


def make_max_tensor_from_max(tensor: max.driver.Tensor) -> MaxTensor:
    """Convert a max.driver.Tensor to a MaxTensor"""
    shape = tuple(tensor.shape)
    return MaxTensor(
        shape, max_data=tensor, device=max_device_to_torch_device(tensor.device)
    )


def rename_privateuse_backend():
    torch.utils.rename_privateuse1_backend(device_name)


def _register_device_module():
    torch._register_device_module(device_name, torch_max_device_module)


def generate_methods_for_privateuse_backend():
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_packed_sequence=True, for_storage=False
    )


@profile
def _register():
    rename_privateuse_backend()
    _register_device_module()
    register_max_ops()
    generate_methods_for_privateuse_backend()


registered = False


def register_max_devices():
    global registered
    if registered:
        return
    _register()
    registered = True
