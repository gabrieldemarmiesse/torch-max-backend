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
    if groups != 1:
        raise NotImplementedError("Grouped convolution is not supported yet.")
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, str):
        raise ValueError("Padding must be an int or a tuple of ints.")
    elif isinstance(padding, tuple):
        padding = (padding[0], padding[1], padding[0], padding[1])
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return max.graph.ops.conv2d(
        input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation,
        input_layout=max.graph.type.ConvInputLayout.NCHW, 
        filter_layout=max.graph.type.FilterLayout.FCRS,
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
