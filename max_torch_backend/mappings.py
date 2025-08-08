import operator

import max.graph.ops
import torch
import torch.nn.functional as F
import torch.amp.autocast_mode
from max.graph.type import DeviceRef
from max.torch.torch import max_device_ref
from max.dtype import DType

IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    operator.getitem,
    operator.matmul,
    operator.neg,
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


def torch_to_equivalent(tensor, *args, **kwargs):
    # Let's support simple stuff for now.
    # TODO: refactor this, this is so ugly
    kwargs = kwargs.copy()
    device = None
    dtype = None
    if len(args) > 1:
        raise ValueError(
            f"Only one argument is supported for torch.to equivalent for now. got {args}"
        )
    device = kwargs.pop("device", None)
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        dtype = DType.from_torch(dtype)

    # Handle device string conversion
    if isinstance(device, str):
        if device == "cpu":
            device = DeviceRef.CPU()
        elif device == "cuda":
            device = DeviceRef.GPU()
        else:
            raise ValueError(f"Unsupported device string: {device}")
    elif isinstance(device, torch.device):
        device = max_device_ref(device)

    if kwargs:
        raise ValueError(
            f"Unsupported arguments for torch.to equivalent: {kwargs}. Only 'device' and 'dtype' are supported."
        )
    if args:
        first_arg = args[0]
        if first_arg == "cpu":
            device = DeviceRef.CPU()
        elif first_arg == "cuda":
            device = DeviceRef.GPU()
        elif isinstance(first_arg, torch.device):
            device = max_device_ref(first_arg)
        elif isinstance(first_arg, torch.dtype):
            dtype = DType.from_torch(first_arg)

    result = tensor
    if device is not None:
        result = max.graph.ops.transfer_to(result, device=device)
    if dtype is not None:
        result = max.graph.ops.cast(result, dtype=dtype)
    if device is None and dtype is None:
        raise ValueError(
            "Either 'device' or 'dtype' must be specified for torch.to equivalent."
        )
    return result


def torch_transpose_equivalent(tensor, dim0, dim1):
    # Get the current tensor dimensions
    ndim = len(tensor.shape)

    # Handle negative dimensions
    if dim0 < 0:
        dim0 = ndim + dim0
    if dim1 < 0:
        dim1 = ndim + dim1

    # Validate dimensions
    if dim0 < 0 or dim0 >= ndim:
        raise ValueError(
            f"Dimension {dim0} out of range for tensor with {ndim} dimensions"
        )
    if dim1 < 0 or dim1 >= ndim:
        raise ValueError(
            f"Dimension {dim1} out of range for tensor with {ndim} dimensions"
        )

    # If dimensions are the same, no change needed
    if dim0 == dim1:
        return tensor

    # Create permutation list - swap dim0 and dim1
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

    return max.graph.ops.permute(tensor, perm)


def torch_mean_equivalent(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        max_dtype = DType.from_torch(dtype)
        input = max.graph.ops.cast(input, dtype=max_dtype)

    result = input

    if dim is None:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Multiple dimensions reduction - reduce each dimension one by one
    # Sort dimensions in descending order to avoid index shifting issues
    for axis in dim:
        result = max.graph.ops.mean(result, axis=axis)

    # Handle keepdim=False - MAX's mean keeps dimensions by default, so we need to squeeze
    if not keepdim:
        # Remove multiple dimensions - need to be careful about index shifting
        # Sort original dimensions and squeeze from highest to lowest
        dims_to_squeeze = sorted(dim, reverse=True)
        for axis in dims_to_squeeze:
            result = max.graph.ops.squeeze(result, axis=axis)

    return result


def torch_linear_equivalent(input, weight, bias=None):
    weight_t = max.graph.ops.permute(weight, [1, 0])  # Transpose weight
    result = max.graph.ops.matmul(input, weight_t)

    if bias is not None:
        result = result + bias

    return result


def torch_contiguous_equivalent(tensor):
    return tensor


def torch_view_equivalent(tensor, *shape):
    if len(shape) == 1 and isinstance(shape[0], tuple | list):
        target_shape = list(shape[0])
    else:
        target_shape = list(shape)
    return max.graph.ops.reshape(tensor, target_shape)


def torch_unsqueeze_equivalent(tensor, dim):
    return max.graph.ops.unsqueeze(tensor, axis=dim)


def torch_log_api_usage_once_equivalent(*args, **kwargs):
    """
    No-op function for torch._C.PyCapsule._log_api_usage_once.
    This is an internal PyTorch function used for API usage logging
    that we can safely ignore in the MAX backend.
    """
    pass


MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    torch.abs: max.graph.ops.abs,
    torch.cos: max.graph.ops.cos,
    torch.sin: max.graph.ops.sin,
    torch.rsqrt: max.graph.ops.rsqrt,
    torch.sqrt: max.graph.ops.sqrt,
    torch.mean: torch_mean_equivalent,
    torch.cat: torch_cat_equivalent,
    F.conv2d: torch_conv2d_equivalent,
    F.embedding: torch_embedding_equivalent,
    F.linear: torch_linear_equivalent,
    F.relu: max.graph.ops.relu,
    # TODO: Use noop function
    torch.amp.autocast_mode._enter_autocast: torch_autocast_equivalent,
    torch.amp.autocast_mode._exit_autocast: torch_autocast_equivalent,
    torch._C._log_api_usage_once: torch_log_api_usage_once_equivalent,
    torch._C._nn.linear: torch_linear_equivalent,
    # methods are given as strings in the graph
    "float": torch_float_equivalent,
    "expand": torch_expand_equivalent,
    "to": torch_to_equivalent,
    "transpose": torch_transpose_equivalent,
    "view": torch_view_equivalent,
    "contiguous": torch_contiguous_equivalent,
    "unsqueeze": torch_unsqueeze_equivalent,
    "abs": max.graph.ops.abs,
    "cos": max.graph.ops.cos,
    "sin": max.graph.ops.sin,
    "sqrt": max.graph.ops.sqrt,
    "rsqrt": max.graph.ops.rsqrt,
    "pow": operator.pow,
    "mean": torch_mean_equivalent,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func
