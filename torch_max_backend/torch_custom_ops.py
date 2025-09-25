import torch
import torch_max_backend
import torch_max_backend.compiler
from max.torch import CustomOpLibrary
from pathlib import Path
from max.graph import TensorType
import max.graph.ops as max_ops
from collections.abc import Callable


def make_torch_op_from_mojo(
    path_to_kernels: Path, mojo_custom_op_str: str, register_fake_fn: Callable
):
    ops = CustomOpLibrary(path_to_kernels)
    mojo_custom_op = ops.__getattr__(mojo_custom_op_str)

    def compiler_fn(*args, **kwargs):
        # We split args into outputs and inputs
        # We assume outputs are first
        # TODO: make more flexible
        # inspect the signature of mojo_custom_op ?
        out_args = args[: mojo_custom_op.num_outputs]
        in_args = args[mojo_custom_op.num_outputs :]

        out_types = [
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device) for x in out_args
        ]
        return (
            "Not handled yet",
            *max_ops.custom(
                mojo_custom_op.name,
                device=args[0].device,
                values=list(in_args),
                parameters=kwargs,
                out_types=out_types,
            ),
        )

    if torch_max_backend.compiler._global_max_objects is not None:
        # TODO: make more flexible
        # torch_max_backend.compiler._global_max_objects = None ?
        raise ValueError("Must be called before any compilation")

    def mojo_custom_op_with_signature(*args, **kwargs):
        return mojo_custom_op(*args, **kwargs)

    mojo_custom_op_with_signature.__signature__ = mojo_custom_op.torch_signature

    torch_max_backend.compiler.paths_to_mojo_kernels.append(path_to_kernels)
    torch_max_backend.MAPPING_TORCH_ATEN_TO_MAX[
        f"{path_to_kernels.name}.{mojo_custom_op_str}"
    ] = compiler_fn

    torch_custom_op = torch.library.custom_op(
        f"{path_to_kernels.name}::{mojo_custom_op_str}", mutates_args=("img_out",)
    )(mojo_custom_op_with_signature)

    def fn(*args, **kwargs):
        output_tensors = register_fake_fn(*args, **kwargs)
        if isinstance(output_tensors, torch.Tensor):
            output_tensors = (output_tensors,)
            single_output = True
        else:
            single_output = False

        torch_custom_op(*output_tensors, *args, **kwargs)
        if single_output:
            return output_tensors[0]
        return output_tensors

    return fn
