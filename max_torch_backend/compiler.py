import torch
from max import mlir
from max.graph import KernelLibrary
from max.torch.torch import CustomOpLibrary

from .mappings import MAPPING_TORCH_TO_MOJO_FUNCTIONS
from .ops import MyMaxOp


def my_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm.graph.print_tabular()

    # Use meta tensors (no memory allocation, no computation)
    # Meta tensors only track shape/dtype/device metadata
    meta_inputs = [torch.empty_like(inp, device="meta") for inp in example_inputs]
    with torch.no_grad():
        meta_outputs = gm(*meta_inputs)
        if isinstance(meta_outputs, torch.Tensor):
            meta_outputs = [meta_outputs]

    def max_add_i_want_to_use(*args):
        mapping_names_to_tensors = {}

        def get_tensor_or_value(something):
            if isinstance(something, torch.fx.Node):
                return mapping_names_to_tensors[something.name]
            elif isinstance(something, int):
                return something

        args_index = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                mapping_names_to_tensors[node.name] = args[args_index]
                args_index += 1
            elif node.op == "call_function":
                func_args = [get_tensor_or_value(x) for x in node.args]
                func_kwags = {k: get_tensor_or_value(v) for k, v in node.kwargs.items()}
                if node.target not in MAPPING_TORCH_TO_MOJO_FUNCTIONS:
                    raise ValueError(
                        f"Function {node.target} not supported by the Max backend yet."
                    )
                tensor = MAPPING_TORCH_TO_MOJO_FUNCTIONS[node.target](
                    *func_args, **func_kwags
                )
                mapping_names_to_tensors[node.name] = tensor
            elif node.op == "output":
                return tuple(get_tensor_or_value(x) for x in node.args[0])

    op = MyMaxOp(
        max_add_i_want_to_use,
        max_add_i_want_to_use.__name__,
        CustomOpLibrary(KernelLibrary(mlir.Context())),
        input_types=None,
        output_types=None,
        num_outputs=len(meta_outputs),
        num_inputs=len(example_inputs),
    )
    custom_op_def = op.custom_op_def()

    def torch_add_with_max(*args) -> torch.Tensor:
        results = [torch.empty_like(x, device=args[0].device) for x in meta_outputs]
        custom_op_def(*results, *args)
        return results

    print("compiler done!:!!!!!!!!!")
    return torch_add_with_max
