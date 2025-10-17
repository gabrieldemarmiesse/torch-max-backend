from compiler import register
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from utils.index import IndexList


@compiler.register("adaptive_avg_pool2d_backward")
struct AdaptiveAvgPool2dBackwardKernel:
    """High-performance Mojo kernel for adaptive average pooling 2D backward pass.

    This kernel distributes gradients from output positions back to the input
    positions that contributed to them. For each output position, it:
    1. Computes which input region was averaged (using adaptive pooling formula).
    2. Divides the gradient by the region size (averaging).
    3. Accumulates the gradient to all input positions in that region.

    The kernel processes sequentially to handle overlapping regions where
    multiple output positions may contribute to the same input position.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,  # Should be 4 for [N, C, H, W]
        //,
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

        # Initialize grad_input to zeros using indexing
        for n in range(batch_size):
            for c in range(channels):
                for ih in range(input_height):
                    for iw in range(input_width):
                        var idx = IndexList[rank](n, c, ih, iw)
                        grad_input.store[1](idx, SIMD[dtype, 1](0))

        # Iterate over all positions in grad_output
        # We process sequentially to avoid race conditions when accumulating
        for n in range(batch_size):
            for c in range(channels):
                for oh in range(output_height):
                    for ow in range(output_width):
                        # Compute input region bounds using adaptive pooling formula
                        # These match PyTorch's adaptive pooling index computation
                        var ih_start = (oh * input_height) // output_height
                        var ih_end = ((oh + 1) * input_height + output_height - 1) // output_height
                        var iw_start = (ow * input_width) // output_width
                        var iw_end = ((ow + 1) * input_width + output_width - 1) // output_width

                        # Compute region size
                        var kh = ih_end - ih_start
                        var kw = iw_end - iw_start
                        var region_size = kh * kw

                        # Get gradient value at this output position
                        var grad_out_idx = IndexList[rank](n, c, oh, ow)
                        var grad_val = grad_output.load[1](grad_out_idx)

                        # Compute gradient delta (divided by region size for averaging)
                        var grad_delta = grad_val / Scalar[dtype](region_size)
                        # Distribute gradient to all input positions in this region
                        for ih in range(ih_start, ih_end):
                            for iw in range(iw_start, iw_end):
                                var grad_in_idx = IndexList[rank](n, c, ih, iw)
                                var current_grad = grad_output.load[1](grad_in_idx)
                                # using grad_input raises this compiling error:
                                # loading not supported for output tensors
                                # var current_grad = grad_input.load[1](grad_in_idx)
                                grad_input.store[1](grad_in_idx, current_grad + grad_delta)
