import operator

import max.graph.ops as max_ops
import torch
from max.graph.type import DeviceRef
from max.torch.torch import max_device_ref
from max.dtype import DType
from max.graph import StaticDim, Dim
import max.graph.type as max_type
import numpy as np
import math
from torch.ops import aten


def torch_cat_equivalent(tensors: list, dim=0):
    return max_ops.concat(tensors, axis=dim)


def torch_stack_equivalent(tensors: list, dim=0):
    return max_ops.stack(tensors, axis=dim)


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
    elif isinstance(padding, tuple | list):
        if len(padding) == 2:
            # PyTorch padding=(pad_h, pad_w) -> MAX padding=(pad_h_before, pad_h_after, pad_w_before, pad_w_after)
            padding = (padding[0], padding[0], padding[1], padding[1])
        elif len(padding) == 4:
            # Already in MAX format
            padding = tuple(padding)
        else:
            raise ValueError(f"Unsupported padding length: {len(padding)}")
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    # NCHW: [batch, channels, height, width] -> NHWC: [batch, height, width, channels]
    input_nhwc = input.permute([0, 2, 3, 1])

    # Convert weight from PyTorch OIHW: [out_channels, in_channels, kernel_h, kernel_w]
    # to MAX RSCF: [kernel_h, kernel_w, in_channels, out_channels]
    weight_rscf = weight.permute([2, 3, 1, 0])

    result = max_ops.conv2d(
        input_nhwc,
        weight_rscf,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_layout=max_type.ConvInputLayout.NHWC,
        filter_layout=max_type.FilterLayout.RSCF,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    # NHWC: [batch, height, width, channels] -> NCHW: [batch, channels, height, width]
    return result.permute([0, 3, 1, 2])


def torch_aten_embedding_equivalent(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    # For some reason with aten, input and weight are inverted.
    return torch_embedding_equivalent(
        weight,
        input,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )


def torch_embedding_equivalent(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
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
        input_reshaped = max_ops.unsqueeze(input, axis=0)
        result = max_ops.gather(weight, input_reshaped, axis=0)
        # Remove the added dimension: [1, embedding_dim] -> [embedding_dim]
        return max_ops.squeeze(result, axis=0)
    else:
        # Use gather to select rows from weight matrix based on input indices
        # axis=0 means we're gathering along the first dimension (vocab dimension)
        return max_ops.gather(weight, input, axis=0)


def torch_autocast_equivalent(*args, **kwargs):
    pass


def torch_float_equivalent(tensor):
    return max_ops.cast(tensor, dtype=max_type.DType.float32)


def torch_aten_expand_equivalent(tensor, size: list[int]):
    return torch_expand_equivalent(tensor, *size)


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

    return max_ops.broadcast_to(tensor, target_shape)


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
    kwargs.pop("layout", None)  # Ignore layout for now
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
        result = max_ops.transfer_to(result, device=device)
    if dtype is not None:
        result = max_ops.cast(result, dtype=dtype)
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

    return max_ops.permute(tensor, perm)


def torch_mean_equivalent(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        max_dtype = DType.from_torch(dtype)
        input = max_ops.cast(input, dtype=max_dtype)

    result = input

    if dim is None:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Multiple dimensions reduction - reduce each dimension one by one
    # Sort dimensions in descending order to avoid index shifting issues
    for axis in dim:
        result = max_ops.mean(result, axis=axis)

    # Handle keepdim=False - MAX's mean keeps dimensions by default, so we need to squeeze
    if not keepdim:
        # Remove multiple dimensions - need to be careful about index shifting
        # Sort original dimensions and squeeze from highest to lowest
        dims_to_squeeze = sorted(dim, reverse=True)
        for axis in dims_to_squeeze:
            result = max_ops.squeeze(result, axis=axis)

    return result


def torch_contiguous_equivalent(tensor):
    return tensor


def torch_view_equivalent(tensor, *shape):
    if len(shape) == 1 and isinstance(shape[0], tuple | list):
        target_shape = list(shape[0])
    else:
        target_shape = list(shape)
    return max_ops.reshape(tensor, target_shape)


def torch_unsqueeze_equivalent(tensor, dim):
    return max_ops.unsqueeze(tensor, axis=dim)


def relu_equivalent(tensor, inplace: bool = False):
    # inplace has no meaning in max since it's graph-based
    return max_ops.relu(tensor)


def torch_max_pool2d_with_indices_equivalent(
    input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
) -> tuple[max_ops.TensorType, max_ops.TensorType]:
    # the first output is the values, the second output is the indices
    # most of the time people just want the values so we'll implement that
    # for now.
    values = torch_max_pool2d_equivalent(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=False,
    )
    # TODO: Add indices
    return (values,)


def torch_max_pool2d_equivalent(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if return_indices:
        raise NotImplementedError("return_indices=True is not supported yet")

    if not stride:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW (PyTorch default) to NHWC (MAX requirement)
    input_nhwc = input.permute([0, 2, 3, 1])

    result = max_ops.max_pool2d(
        input_nhwc,
        kernel_size=kernel_size,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        ceil_mode=ceil_mode,
    )

    # Convert result back from NHWC to NCHW for PyTorch compatibility
    return result.permute([0, 3, 1, 2])


def torch_adaptive_avg_pool2d_equivalent(input, output_size):
    # For now, we'll implement this using global average pooling for (1, 1) output
    # and regular avg pooling for other sizes
    if output_size == (1, 1) or output_size == 1:
        # Global average pooling - take mean over spatial dimensions
        return torch_mean_equivalent(input, dim=(2, 3), keepdim=True)
    else:
        # For other output sizes, we'll use avg_pool2d with calculated kernel size and stride
        # Get input spatial dimensions (assuming NCHW format)
        input_h, input_w = input.shape[2], input.shape[3]

        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size

        # Calculate kernel size and stride to achieve the desired output size
        kernel_h = input_h // output_h
        kernel_w = input_w // output_w
        stride_h = input_h // output_h
        stride_w = input_w // output_w

        # Convert input from NCHW to NHWC for MAX
        input_nhwc = input.permute([0, 2, 3, 1])

        result = max_ops.avg_pool2d(
            input_nhwc,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(0, 0),
            ceil_mode=False,
            count_boundary=True,
        )

        # Convert result back from NHWC to NCHW
        return result.permute([0, 3, 1, 2])


def torch_tril_equivalent(input: max_ops.TensorType, diagonal: int = 0, *, out=None):
    # Max doesn't have tril built-in, so we get around this. It should be pretty
    # easy to implement on cpu and gpu though.
    shape = input.shape

    for i in range(len(shape)):
        if not isinstance(shape[i], StaticDim):
            raise ValueError(f"Input dims must be static, got shape {shape}")

    shape_ints = [int(dim) for dim in shape]

    numpy_mask = np.ones(shape_ints, dtype=input.dtype.to_numpy())
    numpy_mask = np.tril(numpy_mask, k=diagonal)
    mask_in_graph = max_ops.constant(numpy_mask, dtype=input.dtype, device=input.device)
    result = input * mask_in_graph
    return result


def torch_triu_equivalent(input: max_ops.TensorType, diagonal: int = 0, *, out=None):
    # For dynamic shapes, we can't pre-compute a mask. Instead we use a different approach.
    # For now, let's check if we can handle static dims, otherwise return input unchanged
    # TODO: Implement dynamic triu using coordinate-based masking
    shape = input.shape

    try:
        # Try to handle static dimensions
        for i in range(len(shape)):
            if not isinstance(shape[i], StaticDim):
                # For dynamic shapes, just return the input unchanged for now
                # This is not correct but will allow the graph to compile
                # TODO: Implement proper dynamic triu
                return input

        shape_ints = [int(dim) for dim in shape]

        numpy_mask = np.ones(shape_ints, dtype=input.dtype.to_numpy())
        numpy_mask = np.triu(numpy_mask, k=diagonal)
        mask_in_graph = max_ops.constant(
            numpy_mask, dtype=input.dtype, device=input.device
        )
        result = input * mask_in_graph
        return result
    except Exception:
        # Fallback: return input unchanged
        return input


def torch_split_equivalent(
    input: max_ops.TensorType, split_size: int | list[int], dim: int = 0
) -> list[max_ops.TensorType]:
    if isinstance(split_size, int):
        shape = int(input.shape[dim])
        new_split_size = [split_size] * (shape // split_size)
        if shape % split_size != 0:
            new_split_size.append(shape % split_size)
    else:
        new_split_size = split_size
    return max_ops.split(input, new_split_size, dim)


def torch_unbind_equivalent(
    input: max_ops.TensorType, dim: int = 0
) -> list[max_ops.TensorType]:
    """
    Equivalent to torch.unbind - removes a tensor dimension and returns a tuple of all slices along that dimension.
    """
    # Get the size of the dimension to unbind
    shape = input.shape
    if dim < 0:
        dim = len(shape) + dim

    size = int(shape[dim])

    # Use split with size 1 to get individual slices, then squeeze
    split_sizes = [1] * size
    split_tensors = max_ops.split(input, split_sizes, dim)

    # Squeeze each tensor to remove the dimension we split along
    result = []
    for tensor in split_tensors:
        squeezed = max_ops.squeeze(tensor, axis=dim)
        result.append(squeezed)

    return result


def torch_select_equivalent(input: max_ops.TensorType, dim: int, index: int):
    """
    Equivalent to torch.select - selects a slice of the tensor along the given dimension at the given index.
    """
    nb_dims = len(input.shape)
    slices = [slice(None)] * nb_dims
    slices[dim] = index
    return input[slices]


def torch_repeat_interleave_equivalent(
    input: max_ops.TensorType, repeats: int, dim: int = 0
) -> max_ops.TensorType:
    """
    Equivalent to torch.repeat_interleave - repeats elements of a tensor along a dimension.
    Each element is repeated 'repeats' times before moving to the next element.
    """
    # Handle negative dim
    if dim < 0:
        dim = len(input.shape) + dim

    # Get the current shape
    shape = input.shape

    # Create a new shape where the specified dimension is expanded
    new_shape = list(shape)
    new_shape[dim] = int(new_shape[dim]) * repeats

    # Use expand to repeat elements along the dimension
    # First, add a new dimension after the target dim, then expand and reshape
    expanded_shape = list(shape)
    expanded_shape.insert(dim + 1, repeats)

    # Add the new dimension
    unsqueezed = max_ops.unsqueeze(input, axis=dim + 1)

    # Expand along the new dimension
    expanded = max_ops.broadcast_to(unsqueezed, expanded_shape)

    # Reshape to merge the repeated dimension
    result = max_ops.reshape(expanded, new_shape)

    return result


def torch_amax_equivalent(input, dim=None, keepdim=False, *, out=None):
    # If only input is provided, we find the maximum along the specified dimension
    if not dim:
        dim = [i for i in range(len(input.shape))]
    elif isinstance(dim, int):
        dim = [dim]

    # Similar to mean, we can only reduce dimensions one at a time
    result = input
    for axis in dim:
        result = max_ops.max(result, axis=axis)
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = max_ops.squeeze(result, axis=axis)
    return result


def torch_amin_equivalent(input, dim=None, keepdim=False, *, out=None):
    # If only input is provided, we find the minimum along the specified dimension
    if not dim:
        dim = [i for i in range(len(input.shape))]
    elif isinstance(dim, int):
        dim = [dim]

    # Similar to mean, we can only reduce dimensions one at a time
    result = input
    for axis in dim:
        result = max_ops.min(result, axis=axis)
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = max_ops.squeeze(result, axis=axis)
    return result


def torch_argmax_equivalent(input, dim=None, keepdim=False, *, out=None):
    # If dim is None, return argmax of flattened tensor
    if dim is None:
        # Flatten the tensor and compute argmax along axis 0
        flattened = max_ops.reshape(input, [-1])
        result = max_ops.argmax(flattened, axis=0)
        if keepdim:
            # Return tensor with same number of dimensions as input, all size 1
            result_shape = [1] * len(input.shape)
            result = max_ops.reshape(result, result_shape)
        else:
            # Return scalar (0-dimensional tensor)
            result = max_ops.squeeze(result, axis=0)
    else:
        # Compute argmax along specified dimension
        result = max_ops.argmax(input, axis=dim)
        if not keepdim:
            # Squeeze the reduced dimension
            result = max_ops.squeeze(result, axis=dim)
    return result


def torch_argmin_equivalent(input, dim=None, keepdim=False, *, out=None):
    # If dim is None, return argmin of flattened tensor
    if dim is None:
        # Flatten the tensor and compute argmin along axis 0
        flattened = max_ops.reshape(input, [-1])
        result = max_ops.argmin(flattened, axis=0)
        if keepdim:
            # Return tensor with same number of dimensions as input, all size 1
            result_shape = [1] * len(input.shape)
            result = max_ops.reshape(result, result_shape)
        else:
            # Return scalar (0-dimensional tensor)
            result = max_ops.squeeze(result, axis=0)
    else:
        # Compute argmin along specified dimension
        result = max_ops.argmin(input, axis=dim)
        if not keepdim:
            # Squeeze the reduced dimension
            result = max_ops.squeeze(result, axis=dim)
    return result


def torch_max_equivalent(*args, **kwargs):
    """
    Implements torch.max with 3 variants:
    1. torch.max(input) - single maximum value
    2. torch.max(input, dim, keepdim=False) - (values, indices) tuple along dimension
    3. torch.max(input, other) - element-wise maximum
    """
    if len(args) == 1:
        # Variant 1: torch.max(input) - single maximum value
        input_tensor = args[0]
        # Check if dim is specified in kwargs
        if "dim" in kwargs:
            dim = kwargs["dim"]
            keepdim = kwargs.get("keepdim", False)
            # Get both values and indices
            values = torch_amax_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            indices = torch_argmax_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            return (values, indices)
        else:
            return torch_amax_equivalent(input_tensor, dim=None, keepdim=False)

    elif len(args) == 2:
        input_tensor, second_arg = args

        # Check if second argument is a tensor (element-wise max)
        if hasattr(second_arg, "shape") and hasattr(second_arg, "dtype"):
            # Variant 3: torch.max(input, other) - element-wise maximum
            return max_ops.max(input_tensor, second_arg)
        else:
            # Variant 2: torch.max(input, dim) - (values, indices) tuple along dimension
            dim = second_arg
            keepdim = kwargs.get("keepdim", False)

            # Get both values and indices
            values = torch_amax_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            indices = torch_argmax_equivalent(input_tensor, dim=dim, keepdim=keepdim)

            # Return as tuple (PyTorch returns namedtuple, but tuple should work)
            return (values, indices)

    elif len(args) == 3:
        # Variant 2: torch.max(input, dim, keepdim)
        input_tensor, dim, keepdim = args
        values = torch_amax_equivalent(input_tensor, dim=dim, keepdim=keepdim)
        indices = torch_argmax_equivalent(input_tensor, dim=dim, keepdim=keepdim)
        return (values, indices)

    else:
        raise ValueError(f"torch.max expects 1-3 arguments, got {len(args)}")


def torch_min_equivalent(*args, **kwargs):
    """
    Implements torch.min with 3 variants:
    1. torch.min(input) - single minimum value
    2. torch.min(input, dim, keepdim=False) - (values, indices) tuple along dimension
    3. torch.min(input, other) - element-wise minimum
    """
    if len(args) == 1:
        # Variant 1: torch.min(input) - single minimum value
        input_tensor = args[0]
        # Check if dim is specified in kwargs
        if "dim" in kwargs:
            dim = kwargs["dim"]
            keepdim = kwargs.get("keepdim", False)
            # Get both values and indices
            values = torch_amin_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            indices = torch_argmin_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            return (values, indices)
        else:
            return torch_amin_equivalent(input_tensor, dim=None, keepdim=False)

    elif len(args) == 2:
        input_tensor, second_arg = args

        # Check if second argument is a tensor (element-wise min)
        if hasattr(second_arg, "shape") and hasattr(second_arg, "dtype"):
            # Variant 3: torch.min(input, other) - element-wise minimum
            return max_ops.min(input_tensor, second_arg)
        else:
            # Variant 2: torch.min(input, dim) - (values, indices) tuple along dimension
            dim = second_arg
            keepdim = kwargs.get("keepdim", False)

            # Get both values and indices
            values = torch_amin_equivalent(input_tensor, dim=dim, keepdim=keepdim)
            indices = torch_argmin_equivalent(input_tensor, dim=dim, keepdim=keepdim)

            # Return as tuple (PyTorch returns namedtuple, but tuple should work)
            return (values, indices)

    elif len(args) == 3:
        # Variant 2: torch.min(input, dim, keepdim)
        input_tensor, dim, keepdim = args
        values = torch_amin_equivalent(input_tensor, dim=dim, keepdim=keepdim)
        indices = torch_argmin_equivalent(input_tensor, dim=dim, keepdim=keepdim)
        return (values, indices)

    else:
        raise ValueError(f"torch.min expects 1-3 arguments, got {len(args)}")


def torch_clamp_equivalent(input, min=None, max=None, *, out=None):
    """
    Implements torch.clamp by clamping all elements in input to the range [min, max].
    Uses max_ops.max and max_ops.min to implement clamp as:
    clamp(x, min, max) = min(max(x, min), max)
    """
    result = input

    # Apply lower bound if min is provided
    if min is not None:
        result = max_ops.max(result, min)

    # Apply upper bound if max is provided
    if max is not None:
        result = max_ops.min(result, max)

    return result


def torch_arange_equivalent(
    start,
    end=None,
    step=1,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    if isinstance(start, float):
        raise ValueError("We don't support float start values for torch.arange")
    if isinstance(step, float):
        raise ValueError("We don't support float step values for torch.arange")
    if isinstance(end, float):
        raise ValueError("We don't support float end values for torch.arange")
    if dtype is None:
        dtype = torch.int64
    dtype = DType.from_torch(dtype)

    if device is None:
        device = torch.get_default_device()
    device = max_device_ref(device)

    if end is None:
        # Single argument form: torch.arange(end)
        end = start
        start = 0

    # Calculate output dimension for max_ops.range
    # The length is ceil((end - start) / step) as per PyTorch docs
    out_dim = end - start
    if step != 1:
        out_dim = int(math.ceil(out_dim / step))

    # Use max_ops.range to create the sequence
    result = max_ops.range(
        Dim(start),
        Dim(end),
        Dim(step),
        out_dim=Dim(out_dim),
        device=device,
        dtype=dtype,
    )
    # TODO: Remove this when the bug is addressed in MAX, range doesn't produce the correct dtype
    # https://github.com/modular/modular/issues/5178
    return max_ops.cast(result, dtype=dtype)


def torch_full_equivalent(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    if dtype is None:
        dtype = torch.float32
    dtype = DType.from_torch(dtype)

    if device is None:
        device = torch.get_default_device()
    device = max_device_ref(device)

    # Create a scalar constant with the fill value
    scalar = max_ops.constant(np.array(fill_value), dtype=dtype, device=device)

    # Broadcast the scalar to the target shape
    return max_ops.broadcast_to(scalar, size)


def torch_full_like_equivalent(
    input,
    fill_value,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=None,
):
    # If dtype is not specified, use the input tensor's dtype
    if dtype is None:
        target_dtype = input.dtype
    else:
        target_dtype = DType.from_torch(dtype)

    # If device is not specified, use the input tensor's device
    if device is None:
        target_device = input.device
    else:
        target_device = max_device_ref(device)

    # Get the shape from the input tensor
    target_shape = input.shape

    # Create a scalar constant with the fill value
    scalar = max_ops.constant(
        np.array(fill_value), dtype=target_dtype, device=target_device
    )

    # Broadcast the scalar to the target shape
    return max_ops.broadcast_to(scalar, target_shape)


def torch_native_layer_norm_equivalent(input, normalized_shape, weight, bias, eps):
    # expects a tuple or list for some reason
    # surely for the backward pass,
    # for the moment we only output the first one.
    return (
        torch_layer_norm_equivalent(
            input, normalized_shape, weight=weight, bias=bias, eps=eps
        ),
    )


def torch_layer_norm_equivalent(
    input, normalized_shape, weight=None, bias=None, eps=1e-5
):
    # Layer norm normalizes over the last len(normalized_shape) dimensions
    # Calculate mean and variance over these dimensions
    axis_to_reduce = list(
        range(len(input.shape) - len(normalized_shape), len(input.shape))
    )

    # Calculate mean
    mean = torch_mean_equivalent(input, dim=axis_to_reduce, keepdim=True)

    # Calculate variance: Var(X) = E[(X - mean)^2]
    centered = input - mean
    variance = torch_mean_equivalent(
        centered * centered, dim=axis_to_reduce, keepdim=True
    )

    # Normalize: (x - mean) / sqrt(variance + eps)
    normalized = centered / max_ops.sqrt(variance + eps)

    # Apply scale and shift if provided
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias

    return normalized


def torch_gelu_equivalent(input, approximate="none"):
    if approximate == "tanh":
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = math.sqrt(2.0 / math.pi)
        inner = coeff * (input + 0.044715 * input * input * input)
        return 0.5 * input * (1.0 + max_ops.tanh(inner))
    else:
        # Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        # Since MAX might not have erf, use the tanh approximation
        coeff = math.sqrt(2.0 / math.pi)
        inner = coeff * (input + 0.044715 * input * input * input)
        return 0.5 * input * (1.0 + max_ops.tanh(inner))


def torch_sum_equivalent(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        max_dtype = DType.from_torch(dtype)
        input = max_ops.cast(input, dtype=max_dtype)

    result = input

    if not dim:
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    # Sum over each dimension
    for axis in sorted(dim, reverse=True):
        result = max_ops.sum(result, axis=axis)

    # Handle keepdim=False - squeeze the reduced dimensions
    if not keepdim:
        # MAX's sum keeps dimensions by default, so we need to squeeze
        for axis in sorted(dim, reverse=True):
            result = max_ops.squeeze(result, axis=axis)

    return result


def torch_aten__softmax_equivalent(input, dim, half_to_float):
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = None
    return torch_softmax_equivalent(input, dim=dim, dtype=dtype)


def torch_softmax_equivalent(input, dim=-1, dtype=None):
    if dtype is not None:
        max_dtype = DType.from_torch(dtype)
        input = max_ops.cast(input, dtype=max_dtype)

    # Handle negative dim
    if dim < 0:
        dim = len(input.shape) + dim

    # Manual implementation
    # Compute max along the specified axis for numerical stability, keeping dimensions
    x_max = torch_amax_equivalent(input, dim=[dim], keepdim=True)

    # Subtract max for numerical stability
    x_shifted = input - x_max

    # Compute exponential
    x_exp = max_ops.exp(x_shifted)

    # Sum along the axis, keeping dimensions for broadcasting
    x_sum = torch_sum_equivalent(x_exp, dim=[dim], keepdim=True)

    # Divide to get softmax
    return x_exp / x_sum


def torch_masked_fill_equivalent(input, mask, value):
    return max_ops.where(mask, value, input)


def torch_mse_loss_equivalent(
    input, target, size_average=None, reduce=None, reduction="mean"
):
    # Compute squared differences
    diff = input - target
    squared_diff = diff * diff

    # Handle reduction types
    if reduction == "none":
        return squared_diff
    elif reduction == "mean":
        return torch_mean_equivalent(squared_diff)
    elif reduction == "sum":
        return torch_sum_equivalent(squared_diff)
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def torch_t_equivalent(input):
    return torch_transpose_equivalent(input, 0, 1)


def torch_addmm_equivalent(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    # addmm computes: beta * input + alpha * mat1 @ mat2
    matmul_result = operator.matmul(mat1, mat2)

    # Apply scaling factors
    if alpha != 1.0:
        matmul_result = operator.mul(matmul_result, alpha)

    if beta != 1.0:
        scaled_input = operator.mul(input, beta)
    else:
        scaled_input = input

    return operator.add(scaled_input, matmul_result)


def torch_aten_convolution_equivalent(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    # aten.convolution is more general than F.conv2d
    # For now, we only support the 2D case that maps to F.conv2d
    if transposed:
        raise NotImplementedError("Transposed convolution is not supported yet")
    if any(p != 0 for p in output_padding):
        raise NotImplementedError("Output padding is not supported yet")

    return torch_conv2d_equivalent(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def torch_div_equivalent(input, other, *, rounding_mode=None):
    # Handle torch.div with different rounding modes
    if rounding_mode is None:
        return operator.truediv(input, other)
    elif rounding_mode == "floor":
        return operator.floordiv(input, other)
    elif rounding_mode == "trunc":
        # Truncation towards zero (not implemented in operator, need custom logic)
        result = operator.truediv(input, other)
        return max_ops.trunc(result)
    else:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")


def torch_foreach_add_equivalent(tensors, others, alpha=1.0):
    """
    Equivalent to torch._foreach_add.List - element-wise addition of two lists of tensors.
    Computes: tensors[i] + alpha * others[i] for each i
    """
    if len(tensors) != len(others):
        raise ValueError(
            f"Expected len(tensors) == len(others), but got {len(tensors)} and {len(others)}"
        )

    result = []
    for tensor, other in zip(tensors, others):
        if alpha == 1.0:
            result.append(tensor + other)
        else:
            result.append(tensor + alpha * other)

    return result


def torch_slice_equivalent(input, dim, start: int, end: int, step: int = 1):
    if end == 2**63 - 1:  # MAX_INT64
        end = None
    slices = [slice(None)] * len(input.shape)
    slices[dim] = slice(start, end, step)
    return input[*slices]


def torch_split_with_sizes_equivalent(input, split_sizes, dim=0):
    result = []
    start = 0
    for size in split_sizes:
        end = start + size
        result.append(torch_slice_equivalent(input, dim, start, end))
        start = end
    return result


def torch_scalar_tensor_equivalent(
    value: float | int,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    device: torch.device = None,
):
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.get_default_device()

    return max_ops.constant(
        value, dtype=DType.from_torch(dtype), device=max_device_ref(device)
    )


def torch_aten_where_equivalent(input, condition, other):
    return max_ops.where(input, condition, other)


def identity(x):
    return x


def torch_clone_equivalent(input, memory_format=None):
    return input


def torch_squeeze_equivalent(input, dim):
    if isinstance(dim, int):
        dim = [dim]
    result = input
    for d in sorted(dim, reverse=True):
        result = max_ops.squeeze(input, axis=d)
    return result


def torch_bmm_equivalent(input, mat2):
    """
    Batch matrix multiplication equivalent to torch.bmm.

    Args:
        input: 3D tensor of shape [batch_size, n, m]
        mat2: 3D tensor of shape [batch_size, m, p]

    Returns:
        3D tensor of shape [batch_size, n, p]
    """
    # MAX's matmul handles batch dimensions automatically through broadcasting
    return max_ops.matmul(input, mat2)


def torch_exp_equivalent(input):
    return max_ops.exp(input)


def torch_group_norm_equivalent(input, num_groups, weight=None, bias=None, eps=1e-5):
    # input shape: [N, C, H, W]
    N, C, H, W = input.shape

    # Ensure number of channels is divisible by number of groups
    if int(C) % num_groups != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by number of groups ({num_groups})"
        )

    channels_per_group = int(C) // num_groups

    # Reshape input to [N, num_groups, channels_per_group, H, W]
    reshaped = max_ops.reshape(
        input, [int(N), num_groups, channels_per_group, int(H), int(W)]
    )

    # Calculate mean and variance for each group
    # Normalize over dimensions: channels_per_group, H, W (dims 2, 3, 4)
    axis_to_reduce = [2, 3, 4]

    # Calculate mean
    mean = torch_mean_equivalent(reshaped, dim=axis_to_reduce, keepdim=True)

    # Calculate variance: Var(X) = E[(X - mean)^2]
    centered = reshaped - mean
    variance = torch_mean_equivalent(
        centered * centered, dim=axis_to_reduce, keepdim=True
    )

    # Normalize: (x - mean) / sqrt(variance + eps)
    normalized = centered / max_ops.sqrt(variance + eps)

    # Reshape back to original shape [N, C, H, W]
    normalized = max_ops.reshape(normalized, [int(N), int(C), int(H), int(W)])

    # Apply scale and shift if provided
    if weight is not None:
        # weight shape: [C] - broadcast to [N, C, H, W]
        weight_reshaped = max_ops.reshape(weight, [1, int(C), 1, 1])
        normalized = normalized * weight_reshaped

    if bias is not None:
        # bias shape: [C] - broadcast to [N, C, H, W]
        bias_reshaped = max_ops.reshape(bias, [1, int(C), 1, 1])
        normalized = normalized + bias_reshaped

    return normalized


def torch_native_group_norm_equivalent(input, weight, bias, N, C, HxW, group, eps):
    """
    Equivalent to aten.native_group_norm.
    This is the low-level operation that F.group_norm gets compiled to.
    Returns (normalized_output, mean, rstd) tuple but we only return the first element for simplicity.
    """
    # Reshape input from [N*C, HxW] back to [N, C, H, W] format
    # First, calculate H and W from HxW
    HW = int(HxW)
    # For simplicity, assume square spatial dimensions
    H = W = int(HW**0.5)
    if H * W != HW:
        # If not square, try to factor HxW into reasonable H and W
        # For now, use 1D spatial dimension
        H, W = HW, 1

    # Reshape input to [N, C, H, W]
    input_reshaped = max_ops.reshape(input, [int(N), int(C), H, W])

    # Use the regular group_norm implementation
    result = torch_group_norm_equivalent(input_reshaped, group, weight, bias, eps)

    # Return just the normalized output (native_group_norm returns a tuple)
    return (result,)


def torch_logical_not_equivalent(input):
    """
    Equivalent to torch.logical_not.
    PyTorch's logical_not treats any non-zero value as True and returns the logical negation.
    MAX's logical_not requires boolean input, so we need to convert first.
    """
    # Convert input to boolean (non-zero -> True, zero -> False)
    input_bool = max_ops.not_equal(input, 0)
    # Apply logical not
    return max_ops.logical_not(input_bool)


def torch_logical_and_equivalent(input, other):
    """
    Equivalent to torch.logical_and.
    Computes element-wise logical AND of two tensors.
    Both inputs are converted to boolean first if they aren't already.
    """
    # Convert both inputs to boolean if they aren't already
    if input.dtype != max_type.DType.bool:
        input_bool = max_ops.not_equal(input, 0)
    else:
        input_bool = input

    if other.dtype != max_type.DType.bool:
        other_bool = max_ops.not_equal(other, 0)
    else:
        other_bool = other

    # Apply logical and
    return max_ops.logical_and(input_bool, other_bool)


def torch_any_equivalent(input, dim=None, keepdim=False, *, out=None):
    """
    Equivalent to torch.any.
    Tests if any elements in the input are True (non-zero).
    Uses max() on boolean tensor since True > False.
    """
    # Convert input to boolean first (non-zero values become True)
    input_bool = max_ops.not_equal(input, 0)

    if dim is None:
        # Return True if any element is True (reduce all dimensions)
        dim = tuple(range(len(input.shape)))
    elif isinstance(dim, int):
        dim = (dim,)

    # Handle negative dimensions
    dim = [x if x >= 0 else len(input.shape) + x for x in dim]

    result = input_bool
    # Use max() to implement any() since True > False
    for axis in sorted(dim, reverse=True):
        result = max_ops.max(result, axis=axis)

    # Handle keepdim=False
    if not keepdim:
        # Squeeze the reduced dimensions
        for axis in sorted(dim, reverse=True):
            result = max_ops.squeeze(result, axis=axis)

    return result


IDENTICAL_FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
    operator.matmul,
    operator.neg,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.ifloordiv,
    operator.ipow,
    operator.imod,
    operator.getitem,
    str,
    max,
    min,
]


MAPPING_TORCH_TO_MOJO_FUNCTIONS = {
    aten.t: torch_t_equivalent,
    aten.addmm: torch_addmm_equivalent,
    aten.mse_loss: torch_mse_loss_equivalent,
    aten._foreach_add: torch_foreach_add_equivalent,
    aten.sub: operator.sub,
    aten.mul: operator.mul,
    aten.add: operator.add,
    aten.le: operator.le,
    aten.lt: operator.lt,
    aten.ge: operator.ge,
    aten.gt: operator.gt,
    aten.eq: operator.eq,
    aten.ne: operator.ne,
    aten.neg: operator.neg,
    aten.div: torch_div_equivalent,
    aten.floordiv: operator.floordiv,
    aten.permute: max_ops.permute,
    aten.pow: operator.pow,
    aten.mean: torch_mean_equivalent,
    aten.mm: operator.matmul,
    aten.bmm: torch_bmm_equivalent,
    aten.sum: torch_sum_equivalent,
    aten.view: torch_view_equivalent,
    aten.argmax: torch_argmax_equivalent,
    aten.full: torch_full_equivalent,
    aten.full_like: torch_full_like_equivalent,
    aten.remainder: operator.mod,
    aten.abs: max_ops.abs,
    aten.cos: max_ops.cos,
    aten.sin: max_ops.sin,
    aten.rsqrt: max_ops.rsqrt,
    aten.sqrt: max_ops.sqrt,
    aten.cat: torch_cat_equivalent,
    aten.stack: torch_stack_equivalent,
    aten.min: torch_min_equivalent,
    aten.max: torch_max_equivalent,
    aten.amax: torch_amax_equivalent,
    aten.amin: torch_amin_equivalent,
    aten.clamp: torch_clamp_equivalent,
    aten.arange: torch_arange_equivalent,
    aten.layer_norm: torch_layer_norm_equivalent,
    aten.native_layer_norm: torch_native_layer_norm_equivalent,
    aten.gelu: torch_gelu_equivalent,
    aten.softmax: torch_softmax_equivalent,
    aten._softmax: torch_aten__softmax_equivalent,
    aten.masked_fill: torch_masked_fill_equivalent,
    aten.split: torch_split_equivalent,
    aten.tril: torch_tril_equivalent,
    aten.triu: torch_triu_equivalent,
    aten.unbind: torch_unbind_equivalent,
    aten.repeat_interleave: torch_repeat_interleave_equivalent,
    aten.minimum: max_ops.min,
    aten.maximum: max_ops.max,
    aten.squeeze: torch_squeeze_equivalent,
    aten.unsqueeze: torch_unsqueeze_equivalent,
    aten.argmin: torch_argmin_equivalent,
    aten.argmax: torch_argmax_equivalent,
    aten.relu: relu_equivalent,
    aten.embedding: torch_aten_embedding_equivalent,
    aten.convolution: torch_aten_convolution_equivalent,
    aten._adaptive_avg_pool2d: torch_adaptive_avg_pool2d_equivalent,
    aten.select: torch_select_equivalent,
    aten._to_copy: torch_to_equivalent,
    aten.slice: torch_slice_equivalent,
    aten.expand: torch_aten_expand_equivalent,
    aten.alias: identity,
    aten.split_with_sizes: torch_split_with_sizes_equivalent,
    aten.scalar_tensor: torch_scalar_tensor_equivalent,
    aten.where: torch_aten_where_equivalent,
    aten.sigmoid: max_ops.sigmoid,
    aten.max_pool2d_with_indices: torch_max_pool2d_with_indices_equivalent,
    aten.clone: torch_clone_equivalent,
    aten.exp: torch_exp_equivalent,
    aten.native_group_norm: torch_native_group_norm_equivalent,
    aten.logical_not: torch_logical_not_equivalent,
    aten.logical_and: torch_logical_and_equivalent,
    aten.any: torch_any_equivalent,
}

for func in IDENTICAL_FUNCTIONS:
    MAPPING_TORCH_TO_MOJO_FUNCTIONS[func] = func
