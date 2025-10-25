from max.experimental import functional as F
from max.experimental.tensor import Tensor as MaxEagerTensor
from max.graph import TensorType, TensorValue

MaxTensor = TensorValue | MaxEagerTensor


def _register_kernels() -> None:
    """Register custom Mojo kernels in the global graph."""
    import max.experimental.tensor

    import torch_max_backend.torch_compile_backend.compiler

    max.experimental.tensor.GRAPH.graph._import_kernels(
        torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels
    )


def adaptive_avg_pool2d_backward(
    grad_output: MaxTensor, input_tensor_reshaped: MaxTensor
) -> MaxTensor:
    """Custom Mojo kernel for adaptive_avg_pool2d_backward operation."""
    _register_kernels()

    return F.custom(
        name="adaptive_avg_pool2d_backward",
        device=input_tensor_reshaped.device,
        values=[grad_output, input_tensor_reshaped],
        out_types=[
            TensorType(
                dtype=input_tensor_reshaped.dtype,
                shape=input_tensor_reshaped.shape,
                device=input_tensor_reshaped.device,
            )
        ],
    )[0]
