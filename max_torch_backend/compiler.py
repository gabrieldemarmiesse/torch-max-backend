import torch
from max import mlir
from max.graph import KernelLibrary
from max.torch.torch import CustomOpLibrary

from .mappings import MAPPING_TORCH_TO_MOJO_FUNCTIONS
from .ops import CompiledFunctionMaxOp


class TensorsBook:
    def __init__(self):
        self.tensors = {}

    def __setitem__(self, name: str, tensor):
        self.tensors[name] = tensor

    def convert_to_max(self, something):
        if isinstance(something, torch.fx.Node):
            return self.tensors[something.name]
        elif isinstance(something, int):
            return something
        elif isinstance(something, torch.fx.immutable_collections.immutable_list):
            return [self.convert_to_max(x) for x in something]
        raise ValueError(f"Unsupported type: {type(something)}")


class CustomOpFunction:
    def __init__(self, gm: torch.fx.GraphModule):
        self.gm = gm

    def __call__(self, *args):
        tensor_book = TensorsBook()
        args_index = 0
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                tensor_book[node.name] = args[args_index]
                args_index += 1
            elif node.op == "call_function":
                func_args = [tensor_book.convert_to_max(x) for x in node.args]
                func_kwags = {
                    k: tensor_book.convert_to_max(v) for k, v in node.kwargs.items()
                }
                if node.target not in MAPPING_TORCH_TO_MOJO_FUNCTIONS:
                    raise ValueError(
                        f"Function {node.target} not supported by the Max backend yet."
                    )
                tensor = MAPPING_TORCH_TO_MOJO_FUNCTIONS[node.target](
                    *func_args, **func_kwags
                )
                tensor_book[node.name] = tensor
            elif node.op == "output":
                return tuple(tensor_book.convert_to_max(x) for x in node.args[0])


class MaxCompiler:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs
        gm.graph.print_tabular()

        # Use meta tensors (no memory allocation, no computation)
        # Meta tensors only track shape/dtype/device metadata
        meta_inputs = [torch.empty_like(inp, device="meta") for inp in example_inputs]
        with torch.no_grad():
            self.meta_outputs = gm(*meta_inputs)
            if isinstance(self.meta_outputs, torch.Tensor):
                self.meta_outputs = [self.meta_outputs]

        op = CompiledFunctionMaxOp(
            CustomOpFunction(gm),
            "CustomOpFromTheCompiler",
            CustomOpLibrary(KernelLibrary(mlir.Context())),
            input_types=None,
            output_types=None,
            num_outputs=len(self.meta_outputs),
            num_inputs=len(example_inputs),
        )
        self.custom_op_def = op.custom_op_def()

    def __call__(self, *args):
        results = [
            torch.empty_like(x, device=args[0].device) for x in self.meta_outputs
        ]
        self.custom_op_def(*results, *args)
        return results
