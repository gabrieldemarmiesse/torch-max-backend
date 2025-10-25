from max.experimental import functional as F
from max.graph import TensorType

from torch_max_backend.types import MaxTensor, Scalar


def _register_kernels() -> None:
    """Register custom Mojo kernels in the global graph."""
    import max.experimental.tensor

    import torch_max_backend.torch_compile_backend.compiler

    max.experimental.tensor.GRAPH.graph._import_kernels(
        torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels
    )


def bitwise_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_and",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def bitwise_and_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and_scalar operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_and_scalar",
        device=input.device,
        values=[input],
        parameters=dict(other=other),
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


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


def bitwise_not(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_not operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_not",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def bitwise_or(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_or",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def bitwise_or_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or_scalar operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_or_scalar",
        device=input.device,
        values=[input],
        parameters=dict(other=other),
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def bitwise_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_xor",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def bitwise_xor_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor_scalar operation.
    """
    _register_kernels()

    return F.custom(
        name="bitwise_xor_scalar",
        device=input.device,
        values=[input],
        parameters=dict(other=other),
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]


def ceil(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for ceil operation.
    """
    _register_kernels()

    return F.custom(
        name="ceil",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
    )[0]
