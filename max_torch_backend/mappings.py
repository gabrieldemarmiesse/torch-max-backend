import operator

import max.graph.ops
import torch
import torch.nn.functional as F



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


def torch_cat_equivalent(tensors: list, dim=0):
    return max.graph.ops.concat(tensors, axis=dim)


def torch_conv2d_equivalent(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    raise NotImplementedError(
        "The MAX equivalent of F.conv2d is not implemented yet. Feel free to contribute!"
    )


MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    torch.abs: max.graph.ops.abs,
    torch.cos: max.graph.ops.cos,
    torch.sin: max.graph.ops.sin,
    torch.cat: torch_cat_equivalent,
    F.conv2d: torch_conv2d_equivalent,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func
