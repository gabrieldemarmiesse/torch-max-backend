from compiler import register
from itertools import product
from math import ceildiv
from os import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, ManagedTensorSlice
from utils.index import IndexList
from gpu import global_idx
from gpu.host import DeviceBuffer
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor, RuntimeLayout


fn _adaptive_avg_pool2d_backward_cpu[
    dtype: DType,
    rank: Int,
](
    grad_input: OutputTensor[dtype=dtype, rank=rank],
    grad_output: InputTensor[dtype=dtype, rank=rank],
) raises:
    """CPU implementation of adaptive average pool 2D backward pass.

    Based on PyTorch's non-atomic backward implementation:
    pytorch/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu
    (adaptive_average_gradinput function)

    Iterates over INPUT positions to avoid needing atomic operations,
    accumulating contributions from all output positions that used this input.
    """
    var batch_size = grad_input.dim_size(0)
    var channels = grad_input.dim_size(1)
    var input_height = grad_input.dim_size(2)
    var input_width = grad_input.dim_size(3)
    var output_height = grad_output.dim_size(2)
    var output_width = grad_output.dim_size(3)

    # Initialize grad_input to zeros
    for n, c, ih, iw in product(
        range(batch_size),
        range(channels),
        range(input_height),
        range(input_width),
    ):
        var indices = IndexList[rank](n, c, ih, iw)
        grad_input[indices] = Scalar[dtype](0)

    # Iterate over input positions (not output positions)
    # This avoids needing to read from grad_input
    for n, c, ih, iw in product(
        range(batch_size),
        range(channels),
        range(input_height),
        range(input_width),
    ):
        # Find which output positions contribute to this input position
        var ostartH = (ih * output_height) // input_height
        var oendH = (
            (ih + 1) * output_height + input_height - 1
        ) // input_height
        var ostartW = (iw * output_width) // input_width
        var oendW = ((iw + 1) * output_width + input_width - 1) // input_width

        var accumulated_grad = Scalar[dtype](0.0)

        # Accumulate gradients from all contributing output positions
        for oh, ow in product(range(ostartH, oendH), range(ostartW, oendW)):
            # Compute the input region for this output position
            var ih_start = (oh * input_height) // output_height
            var ih_end = (
                (oh + 1) * input_height + output_height - 1
            ) // output_height
            var iw_start = (ow * input_width) // output_width
            var iw_end = (
                (ow + 1) * input_width + output_width - 1
            ) // output_width

            # Compute region size
            var kh = ih_end - ih_start
            var kw = iw_end - iw_start
            var region_size = kh * kw

            # Get gradient from output using IndexList
            var grad_output_indices = IndexList[rank](n, c, oh, ow)
            var grad_val = grad_output[grad_output_indices]

            # Accumulate weighted gradient
            accumulated_grad += grad_val / Scalar[dtype](region_size)

        # Write accumulated gradient to input using IndexList
        var grad_input_indices = IndexList[rank](n, c, ih, iw)
        grad_input[grad_input_indices] = accumulated_grad


fn _adaptive_avg_pool2d_backward_gpu[
    dtype: DType,
    rank: Int,
](
    grad_input: OutputTensor[dtype=dtype, rank=rank],
    grad_output: InputTensor[dtype=dtype, rank=rank],
    batch_size: Int,
    channels: Int,
    input_height: Int,
    input_width: Int,
    output_height: Int,
    output_width: Int,
    ctx_ptr: DeviceContextPtr,
) raises:
    """GPU implementation of adaptive average pool 2D backward pass.

    Based on PyTorch's atomic backward implementation:
    aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu
    (atomic_adaptive_average_gradinput function)

    This kernel parallelizes over OUTPUT positions. Each thread processes one or more
    output positions and uses atomic operations to safely accumulate gradients to
    input positions, handling race conditions when multiple outputs map to the same input.

    Key correspondence with PyTorch:
    - Iterates over output positions
    - Uses START_IND/END_IND for input region bounds
    - Computes grad_delta = gradOutput / (kH * kW)
    - Uses atomic add for gradient accumulation (gpuAtomicAddNoReturn)
    """

    alias block_dim = 256

    @parameter
    fn kernel[
        dtype: DType
    ](
        grad_input_ptr: UnsafePointer[Scalar[dtype]],
        grad_output_ptr: UnsafePointer[Scalar[dtype]],
        batch_size: Int,
        channels: Int,
        input_height: Int,
        input_width: Int,
        output_height: Int,
        output_width: Int,
    ):
        # Global thread index
        var tid = global_idx.x

        # Total number of output elements across all batches and channels
        var total_output_elements = (
            batch_size * channels * output_height * output_width
        )

        if tid >= UInt(total_output_elements):
            return

        # Compute which output position this thread is processing
        var tid_int = Int(tid)
        var ow = tid_int % output_width
        var tid_remaining = tid_int // output_width
        var oh = tid_remaining % output_height
        tid_remaining = tid_remaining // output_height
        var c = tid_remaining % channels
        var n = tid_remaining // channels

        # Compute input region bounds using adaptive pooling formula
        var ih_start = (oh * input_height) // output_height
        var ih_end = (
            (oh + 1) * input_height + output_height - 1
        ) // output_height
        var iw_start = (ow * input_width) // output_width
        var iw_end = ((ow + 1) * input_width + output_width - 1) // output_width

        # Compute region size
        var kh = ih_end - ih_start
        var kw = iw_end - iw_start
        var region_size = kh * kw

        # Get gradient value at this output position
        var grad_output_idx = (
            n * (channels * output_height * output_width)
            + c * (output_height * output_width)
            + oh * output_width
            + ow
        )
        var grad_val = grad_output_ptr[grad_output_idx]

        # Compute gradient delta (divided by region size for averaging)
        var grad_delta = grad_val / Scalar[dtype](region_size)

        # Distribute gradient to all input positions in this region
        # Use atomic add to handle race conditions (multiple threads may write to same input position)
        for ih, iw in product(range(ih_start, ih_end), range(iw_start, iw_end)):
            var grad_input_idx = (
                n * (channels * input_height * input_width)
                + c * (input_height * input_width)
                + ih * input_width
                + iw
            )
            _ = Atomic.fetch_add(grad_input_ptr + grad_input_idx, grad_delta)

    var total_output_elements = (
        batch_size * channels * output_height * output_width
    )
    var grid_dim = ceildiv(total_output_elements, block_dim)

    var device_ctx = ctx_ptr.get_device_context()

    # Convert to LayoutTensor for device buffer creation
    alias layout = Layout.row_major[rank]()
    var grad_input_layout = grad_input.to_layout_tensor()
    var grad_output_layout = grad_output.to_layout_tensor()

    # Create device buffers from LayoutTensor pointers
    var grad_input_device = DeviceBuffer[dtype](
        device_ctx, grad_input_layout.ptr, grad_input.size(), owning=False
    )
    var grad_output_device = DeviceBuffer[dtype](
        device_ctx, grad_output_layout.ptr, grad_output.size(), owning=False
    )

    # Initialize grad_input to zeros
    device_ctx.enqueue_memset(grad_input_device, Scalar[dtype](0))

    # Launch the kernel
    device_ctx.enqueue_function_checked[kernel[dtype], kernel[dtype]](
        grad_input_device,
        grad_output_device,
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@compiler.register("adaptive_avg_pool2d_backward")
struct AdaptiveAvgPool2dBackwardKernel:
    """High-performance Mojo kernel for adaptive average pooling 2D backward pass.

    This kernel distributes gradients from output positions back to the input
    positions that contributed to them. For each output position, it:
    1. Computes which input region was averaged (using adaptive pooling formula).
    2. Divides the gradient by the region size (averaging).
    3. Accumulates the gradient to all input positions in that region.

    The kernel uses parallel execution on GPU with atomic operations to handle
    race conditions when multiple output positions contribute to the same input position.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,  # Should be 4 for [N, C, H, W]
        target: StaticString,
    ](
        grad_input: OutputTensor[dtype=dtype, rank=rank],
        grad_output: InputTensor[dtype=dtype, rank=rank],
        input_tensor: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        """Execute the adaptive average pool 2D backward kernel.

        Args:
            grad_input: Output tensor for input gradients [N, C, H_in, W_in].
            grad_output: Input tensor with output gradients [N, C, H_out, W_out].
            input_tensor: Original input tensor (for shape info) [N, C, H_in, W_in].
            ctx: Device context for execution.
        """
        # Get dimensions at runtime
        var batch_size = grad_input.shape()[0]
        var channels = grad_input.shape()[1]
        var input_height = grad_input.shape()[2]
        var input_width = grad_input.shape()[3]
        var output_height = grad_output.shape()[2]
        var output_width = grad_output.shape()[3]

        @parameter
        if is_cpu[target]():
            _adaptive_avg_pool2d_backward_cpu[dtype, rank](
                grad_input, grad_output
            )
        else:
            _adaptive_avg_pool2d_backward_gpu[dtype, rank](
                grad_input,
                grad_output,
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                ctx,
            )
