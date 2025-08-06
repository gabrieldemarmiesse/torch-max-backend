import operator

import max.graph.ops
import torch

IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.pow,
    torch.remainder,
]


MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    torch.abs: max.graph.ops.abs,
    torch.cos: max.graph.ops.cos,
    torch.sin: max.graph.ops.sin,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func
