from max.experimental import functional as F
from max.graph import TensorType

import torch_max_backend
from torch_max_backend.types import MaxTensor, Scalar


def bitwise_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and operation.
    """

    return F.custom(
        name="bitwise_and",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_and_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and_scalar operation.
    """
    if isinstance(other, bool):
        return F.custom(
            name="bitwise_and_scalar_bool",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]
    else:
        return F.custom(
            name="bitwise_and_scalar",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]


def bitwise_not(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_not operation.
    """

    return F.custom(
        name="bitwise_not",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_or(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or operation.
    """

    return F.custom(
        name="bitwise_or",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_or_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or_scalar operation.
    """
    if isinstance(other, bool):
        return F.custom(
            name="bitwise_or_scalar_bool",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]
    else:
        return F.custom(
            name="bitwise_or_scalar",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]


def bitwise_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor operation.
    """

    return F.custom(
        name="bitwise_xor",
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]


def bitwise_xor_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor_scalar operation.
    """
    if isinstance(other, bool):
        return F.custom(
            name="bitwise_xor_scalar_bool",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]
    else:
        return F.custom(
            name="bitwise_xor_scalar",
            device=input.device,
            values=[input],
            parameters=dict(other=other),
            out_types=[
                TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
            ],
            custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
        )[0]


def ceil(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for ceil operation.
    """

    return F.custom(
        name="ceil",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]


def gelu_backward(
    grad_output: MaxTensor, input: MaxTensor, *, approximate: str = "none"
) -> MaxTensor:
    """
    Custom Mojo kernel for gelu_backward operation.
    """
    return F.custom(
        name="gelu_backward",
        device=input.device,
        values=[grad_output, input],
        parameters=dict(approximate=approximate),
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=torch_max_backend.torch_compile_backend.compiler.paths_to_mojo_kernels,
    )[0]
