import operator

import max.graph.ops
import torch
import torch.nn.functional as F
import torch.amp.autocast_mode

IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    operator.getitem,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.pow,
    torch.remainder,
    str,
]


def torch_cat_equivalent(tensors: list, dim=0):
    return max.graph.ops.concat(tensors, axis=dim)


def torch_conv2d_equivalent(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    if groups != 1:
        raise NotImplementedError("Grouped convolution is not supported yet.")

    if dilation != 1:
        raise NotImplementedError("Dilation is not supported yet.")

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, str):
        raise ValueError("Padding must be an int or a tuple of ints.")
    elif isinstance(padding, tuple):
        # PyTorch padding=(pad_h, pad_w) -> MAX padding=(pad_h_before, pad_h_after, pad_w_before, pad_w_after)
        padding = (padding[0], padding[0], padding[1], padding[1])
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    # NCHW: [batch, channels, height, width] -> NHWC: [batch, height, width, channels]
    input_nhwc = input.permute([0, 2, 3, 1])

    # Convert weight from PyTorch OIHW: [out_channels, in_channels, kernel_h, kernel_w]
    # to MAX RSCF: [kernel_h, kernel_w, in_channels, out_channels]
    weight_rscf = weight.permute([2, 3, 1, 0])

    result = max.graph.ops.conv2d(
        input_nhwc,
        weight_rscf,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_layout=max.graph.type.ConvInputLayout.NHWC,
        filter_layout=max.graph.type.FilterLayout.RSCF,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    # NHWC: [batch, height, width, channels] -> NCHW: [batch, channels, height, width]
    return result.permute([0, 3, 1, 2])


def torch_embedding_equivalent(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    # Note: padding_idx affects gradient computation during training, not forward pass
    # During inference, we simply perform the lookup as normal
    # The padding_idx behavior (zero gradients) is handled by PyTorch's autograd system
    if max_norm is not None:
        raise NotImplementedError(
            "max_norm is not supported yet in this embedding implementation"
        )
    if scale_grad_by_freq:
        raise NotImplementedError(
            "scale_grad_by_freq is not supported yet in this embedding implementation"
        )
    if sparse:
        raise NotImplementedError(
            "sparse gradients are not supported yet in this embedding implementation"
        )

    # Handle scalar indices by reshaping to have at least one dimension
    # PyTorch embedding returns the selected row directly for scalar input
    # but MAX gather may need proper shape handling
    original_shape = input.shape
    if len(original_shape) == 0:  # Scalar tensor
        input_reshaped = max.graph.ops.unsqueeze(input, axis=0)
        result = max.graph.ops.gather(weight, input_reshaped, axis=0)
        # Remove the added dimension: [1, embedding_dim] -> [embedding_dim]
        return max.graph.ops.squeeze(result, axis=0)
    else:
        # Use gather to select rows from weight matrix based on input indices
        # axis=0 means we're gathering along the first dimension (vocab dimension)
        return max.graph.ops.gather(weight, input, axis=0)


def torch_autocast_equivalent(*args, **kwargs):
    pass


def torch_float_equivalent(tensor):
    return max.graph.ops.cast(tensor, dtype=max.graph.type.DType.float32)


def torch_expand_equivalent(tensor, *size):
    # Convert size tuple to list and handle -1 values
    target_shape = []

    # Get current tensor shape - we need this to handle -1 values
    current_shape = tensor.shape

    # Pad the current shape with 1s if target has more dimensions
    if len(size) > len(current_shape):
        padded_current_shape = [1] * (len(size) - len(current_shape)) + list(
            current_shape
        )
    else:
        padded_current_shape = list(current_shape)

    # Process each dimension in the target size
    for i, dim_size in enumerate(size):
        if dim_size == -1:
            # Keep current dimension size
            if i < len(padded_current_shape):
                target_shape.append(padded_current_shape[i])
            else:
                # This shouldn't happen in well-formed expand calls
                target_shape.append(1)
        else:
            target_shape.append(dim_size)

    return max.graph.ops.broadcast_to(tensor, target_shape)


MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    torch.abs: max.graph.ops.abs,
    torch.cos: max.graph.ops.cos,
    torch.sin: max.graph.ops.sin,
    torch.cat: torch_cat_equivalent,
    F.conv2d: torch_conv2d_equivalent,
    F.embedding: torch_embedding_equivalent,
    torch.amp.autocast_mode._enter_autocast: torch_autocast_equivalent,
    # methods are given as strings in the graph
    "float": torch_float_equivalent,
    "expand": torch_expand_equivalent,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func
