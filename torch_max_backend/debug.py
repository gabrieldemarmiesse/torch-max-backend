"""We put here all functions that can compare the Max graph
with the PyTorch graph.

Notably we can run the Max graph at the same time as the PyTorch graph
and compare each intermediate result.

This is not public yet but still useful for debugging.
"""

import torch
from max import engine
from torch_max_backend.flags import debug_graph
from typing import Any
from pathlib import Path
from max.graph import TensorValue
import max.graph.ops as max_ops
from max.driver.tensor import load_max_tensor
from torch_max_backend.utils import get_error_message

output_directory = Path("/tmp/.torch_max_backend_debug")
output_directory_max = output_directory / "max"
output_directory_max.mkdir(parents=True, exist_ok=True)


def set_print_options(session: engine.InferenceSession):
    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=output_directory_max
    )


def add_prints(node_idx: int, func_name: str, func_output: Any):
    if not debug_graph():
        return
    if isinstance(func_output, TensorValue):
        func_output = [func_output]

    for i, output_tensor in enumerate(func_output):
        label = get_tensor_label(node_idx, i, func_name)
        max_ops.print(output_tensor, label)


def get_tensor_label(node_idx: int, i: int, func_name: str) -> str:
    return f"node-idx-{node_idx:09}-output-{i}-function-{func_name}"


def pp(x) -> str:
    if isinstance(x, torch.Tensor):
        return repr(x)[:-1] + f", shape={x.shape})"
    if isinstance(x, list):
        return "[" + ", ".join(pp(y) for y in x) + "]"
    if isinstance(x, tuple):
        return "(" + ", ".join(pp(y) for y in x) + ")"
    return repr(x)


def make_debug_function(node_idx, old_func, node):
    def new_function(*func_args, **func_kwargs):
        print(f"Debugging node {node_idx} function {old_func}")
        result = old_func(*func_args, **func_kwargs)
        to_check = result
        if isinstance(to_check, torch.Tensor):
            to_check = [to_check]
        for output_idx, tensor in enumerate(to_check):
            if isinstance(tensor, torch.Tensor):
                # Load the corresponding tensor from MAX
                filename = output_directory_max / (
                    get_tensor_label(node_idx, output_idx, str(old_func)) + ".max"
                )
                if not filename.exists():
                    print(f"Debugging file {filename} does not exist, skipping.")
                    continue
                loaded_tensor = torch.from_dlpack(load_max_tensor(filename))
                true_tensor_from_torch = tensor.to("cpu")
                # Check dtype
                if loaded_tensor.dtype != true_tensor_from_torch.dtype:
                    raise ValueError(
                        f"The output tensor of node {node_idx} function {old_func} with args {func_args} "
                        f", kwargs {func_kwargs} and output {output_idx} has different dtypes between Max and PyTorch. "
                        f"Max dtype is {loaded_tensor.dtype}, PyTorch dtype is {true_tensor_from_torch.dtype}. "
                        f"This is likely a bug in the Max backend. Please open an issue."
                    )
                # Check shape
                if loaded_tensor.shape != true_tensor_from_torch.shape:
                    raise ValueError(
                        f"The output tensor of node {node_idx} function {old_func} with args {func_args} "
                        f", kwargs {func_kwargs} and output {output_idx} has different shapes between Max and PyTorch. "
                        f"Max shape is {loaded_tensor.shape}, PyTorch shape is {true_tensor_from_torch.shape}. "
                        f"This is likely a bug in the Max backend. Please open an issue."
                    )
                # Check values
                try:
                    torch.testing.assert_close(
                        loaded_tensor,
                        true_tensor_from_torch,
                        equal_nan=True,
                        rtol=100000,
                        atol=100000,
                    )
                except AssertionError:
                    print(
                        "error coming from",
                        get_error_message(node, node_idx, func_args, func_kwargs),
                    )
                    raise ValueError(
                        f"The output tensor of node {node_idx} function {old_func} with args {pp(func_args)} "
                        f", kwargs {pp(func_kwargs)} and output {output_idx} has different values between Max and PyTorch. "
                        f"Expected {true_tensor_from_torch} but got {loaded_tensor}. "
                        f"This is likely a bug in the Max backend. Please open an issue."
                    )

        return result

    return new_function


def debug_graph_if_required(gm: torch.fx.GraphModule, args):
    if not debug_graph():
        return

    # Let's insert checks in the graph
    for node_idx, node in enumerate(gm.graph.nodes):
        if node.op == "call_function" or node.op == "call_method":
            node.target = make_debug_function(node_idx, node.target, node)

    gm(*args)
    raise ValueError("Found no error in the debugged graph.")
