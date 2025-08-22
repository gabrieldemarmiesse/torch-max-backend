import torch
import max.graph.ops as ops
from max.dtype import DType
from max.graph.type import DeviceRef
from max.graph import Graph, TensorType
from max import engine
import numpy as np
from torch_max_backend import get_accelerators


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
                dtype=torch.float32,
                device=device or torch.device("max_gpu"),
                requires_grad=False,
            )
        r._max_data = max_data
        return r

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Handle specific operations
        if func == torch.ops.aten.add.Tensor:
            return self._add_impl(args[1])
        elif func == torch.ops.aten.mul.Tensor:
            return self._mul_impl(args[1])
        elif func == torch.ops.aten.sqrt.default:
            return self._sqrt_impl()
        elif func == torch.ops.aten._to_copy.default:
            return self._to_impl(kwargs.get("device"))
        else:
            raise NotImplementedError(f"Operation {func} not implemented for MaxTensor")

    def _add_impl(self, other):
        """Custom add implementation"""

        if not isinstance(other, MaxTensor):
            raise RuntimeError("Can only add MaxTensor to MaxTensor")

        # Get MAX data from tensors
        lhs_data = self._max_data
        rhs_data = other._max_data

        if lhs_data is None or rhs_data is None:
            raise RuntimeError("Tensors don't have MAX data")

        # Create computation graph
        input_type = TensorType(
            dtype=DType.float32, shape=list(self.shape), device=DeviceRef.GPU()
        )
        with Graph("add_graph", input_types=(input_type, input_type)) as graph:
            lhs, rhs = graph.inputs
            out = ops.add(lhs, rhs)
            graph.output(out)

        # Execute on MAX engine
        session = engine.InferenceSession(devices=list(get_accelerators()))
        model = session.load(graph)
        output = model.execute(lhs_data, rhs_data)[0]

        # Create result MaxTensor
        result = MaxTensor(self.shape, max_data=output, device=torch.device("max_gpu"))
        return result

    def _mul_impl(self, other):
        """Custom multiply implementation"""
        print("DEBUG: MaxTensor multiply implementation called")

        if not isinstance(other, MaxTensor):
            raise RuntimeError("Can only multiply MaxTensor with MaxTensor")

        # Get MAX data from tensors
        lhs_data = self._max_data
        rhs_data = other._max_data

        if lhs_data is None or rhs_data is None:
            raise RuntimeError("Tensors don't have MAX data")

        # Create computation graph
        input_type = TensorType(
            dtype=DType.float32, shape=list(self.shape), device=DeviceRef.GPU()
        )
        with Graph("mul_graph", input_types=(input_type, input_type)) as graph:
            lhs, rhs = graph.inputs
            out = ops.mul(lhs, rhs)
            graph.output(out)

        # Execute on MAX engine
        session = engine.InferenceSession(devices=list(get_accelerators()))
        model = session.load(graph)
        output = model.execute(lhs_data, rhs_data)[0]

        # Create result MaxTensor
        result = MaxTensor(self.shape, max_data=output, device=torch.device("max_gpu"))
        print(f"DEBUG: Multiply completed, result shape: {result.shape}")
        return result

    def _sqrt_impl(self):
        """Custom sqrt implementation"""

        # Get MAX data from tensor
        input_data = self._max_data

        if input_data is None:
            raise RuntimeError("Tensor doesn't have MAX data")

        # Create computation graph
        input_type = TensorType(
            dtype=DType.from_torch(self.dtype),
            shape=list(self.shape),
            device=DeviceRef.GPU(),
        )
        with Graph("sqrt_graph", input_types=(input_type,)) as graph:
            (input_tensor,) = graph.inputs
            out = ops.sqrt(input_tensor)
            graph.output(out)

        # Execute on MAX engine
        session = engine.InferenceSession(devices=list(get_accelerators()))
        model = session.load(graph)
        output = model.execute(input_data)[0]

        # Create result MaxTensor
        result = MaxTensor(self.shape, max_data=output, device=torch.device("max_gpu"))
        return result

    def _to_impl(self, device):
        """Custom to() implementation"""
        if device is not None and device.type == "cpu":
            print("DEBUG: Converting MaxTensor to CPU")
            max_data = self._max_data
            if max_data is None:
                return torch.zeros(self.shape, dtype=self.dtype, device="cpu")

            try:
                cpu_array = max_data.to_numpy()
                return torch.from_numpy(cpu_array).float()
            except Exception as e:
                print(f"GPU->CPU transfer failed: {e}")
                return torch.zeros(self.shape, dtype=self.dtype, device="cpu")

        # For other devices, return a copy
        return MaxTensor(self.shape, max_data=self._max_data, device=device)


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
    @torch.library.impl("aten::empty.memory_format", "PrivateUse1")
    def empty_max(
        size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
    ):
        # Create a meta tensor first, then create the storage for max_gpu
        meta_tensor = torch.empty(size, dtype=dtype or torch.float32, device="meta")
        # Now we need to make this tensor appear as max_gpu
        # For now, return meta tensor - PyTorch will handle device assignment
        meta_tensor._max_data = None
        return meta_tensor

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
