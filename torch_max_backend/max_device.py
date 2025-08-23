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


class Placeholder:
    def __init__(self, index):
        self.index = index


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
                # TODO: transfer on the cpu with max
                return torch.from_dlpack(args[0]._max_data).to("cpu")
            else:
                raise NotImplementedError("Transfer to non-CPU devices not implemented")
        input_tensors = []
        list_of_input_specs = []

        def extract_tensors(arg):
            if isinstance(arg, torch.Tensor):
                # create an input spec
                input_type = TensorType(
                    dtype=DType.from_torch(arg.dtype),
                    shape=list(arg.shape),
                    device=DeviceRef.GPU(),
                )
                list_of_input_specs.append(input_type)
                input_tensors.append(arg._max_data)

                return Placeholder(len(list_of_input_specs) - 1)
            elif isinstance(arg, int | float):
                return arg
            elif isinstance(arg, list):
                return [extract_tensors(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple(extract_tensors(x) for x in arg)
            elif isinstance(arg, dict):
                return {k: extract_tensors(v) for k, v in arg.items()}
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

        new_args_with_placeholders = extract_tensors(args)
        new_kwargs_with_placeholders = extract_tensors(kwargs)

        with Graph("add_graph", input_types=list_of_input_specs) as graph:

            def replace_with_inputs(arg):
                if isinstance(arg, Placeholder):
                    return graph.inputs[arg.index]
                elif isinstance(arg, int | float):
                    return arg
                elif isinstance(arg, list):
                    return [replace_with_inputs(x) for x in arg]
                elif isinstance(arg, tuple):
                    return tuple(replace_with_inputs(x) for x in arg)
                elif isinstance(arg, dict):
                    return {k: replace_with_inputs(v) for k, v in arg.items()}
                elif isinstance(arg, torch.dtype):
                    return arg
                elif isinstance(arg, torch.layout):
                    return arg
                elif isinstance(arg, torch.device):
                    return arg
                elif arg is None:
                    return arg
                else:
                    raise NotImplementedError(
                        f"Argument type {type(arg)} not supported"
                    )

            replaced_args = replace_with_inputs(new_args_with_placeholders)
            replaced_kwargs = replace_with_inputs(new_kwargs_with_placeholders)

            if func in MAPPING_TORCH_ATEN_TO_MAX:
                func_to_use = MAPPING_TORCH_ATEN_TO_MAX[func]
            elif func.overloadpacket in MAPPING_TORCH_ATEN_TO_MAX:
                func_to_use = MAPPING_TORCH_ATEN_TO_MAX[func.overloadpacket]
            else:
                raise NotImplementedError(
                    f"Operation {func} not implemented for MaxTensor"
                )
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
        output = model.execute(*input_tensors)

        if is_tuple:
            return tuple(make_max_tensor_from_max(o) for o in output)
        else:
            return make_max_tensor_from_max(output[0])


class MaxDeviceModule:
    @staticmethod
    def _is_in_bad_fork():
        return False

    @staticmethod
    def manual_seed_all(seed):
        np.random.seed(seed)

    @staticmethod
    def device_count():
        return 1

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
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16, torch.bfloat16]

    def max_gpu(self):
        print("hello")


class Custom:
    def __init__(self, t):
        self.t = t


def register_max_ops():
    @torch.library.impl("aten::arange", "PrivateUse1")
    def arange_max(end, dtype=None, layout=None, device=None, pin_memory=None):
        print(f"DEBUG: arange called with end={end}, device={device}")
        if dtype is None:
            dtype = torch.int64
        # Create the computation graph
        with Graph("arange_graph", input_types=tuple()) as graph:
            out = ops.range(
                0, end, 1, device=DeviceRef.GPU(), dtype=DType.from_torch(dtype)
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


class MaxDeviceBackend:
    def __init__(self):
        self.device_module = MaxDeviceModule()

    def register(self):
        """Register the max backend with PyTorch."""
        # Step 1: Rename privateuse1 backend to "max_gpu"
        torch.utils.rename_privateuse1_backend("max_gpu")

        # Step 2: Register the device module
        torch._register_device_module("max_gpu", self.device_module)

        # Step 3: Register operations for the PrivateUse1 dispatch key
        register_max_ops()

        # Step 4: Generate helper methods for tensors
        torch.utils.generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_packed_sequence=True,
            for_storage=False,
        )
