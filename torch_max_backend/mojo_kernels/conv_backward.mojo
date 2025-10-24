# ===----------------------------------------------------------------------=== #
# Convolution Backward Kernels for PyTorch MAX Backend
#
# Implements grad_weight computation for 2D convolution backward pass.
# Based on PyTorch's CUDA implementation: one thread per weight element
# ===----------------------------------------------------------------------=== #

from compiler import register
from gpu.id import global_idx
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from utils.index import IndexList


@compiler.register("conv2d_backward_weight")
struct Conv2dBackwardWeightKernel:
    """
    Computes grad_weight for 2D convolution backward pass.

    GPU Parallelization Strategy (following PyTorch CUDA):
    - Total threads = F × C × R × S (one thread per weight element)
    - Each thread independently computes one grad_weight element
    - Thread accumulates over all (N, HO, WO) positions
    - Uses linear thread indexing with multi-dimensional decomposition

    Algorithm:
        grad_weight[f, c, r, s] = Σ(grad_output[n, f, ho, wo] * input[n, c, h, w])
        where h = ho * stride_h - pad_h + r * dil_h
              w = wo * stride_w - pad_w + s * dil_w

    Layouts:
        grad_output: NCHW [N, F, HO, WO]
        input: NCHW [N, C, H, W]
        grad_weight: OIHW [F, C, R, S]

    Tensor shapes are passed as compile-time parameters from Python.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank_grad_out: Int,  # Should be 4 for NCHW
        rank_input: Int,     # Should be 4 for NCHW
        rank_grad_weight: Int,  # Should be 4 for OIHW
        //,
        target: StaticString,
        # Tensor dimensions (passed from Python)
        N: Int,   # Batch size
        F: Int,   # Output channels
        C: Int,   # Input channels
        H: Int,   # Input height
        W: Int,   # Input width
        HO: Int,  # Output height
        WO: Int,  # Output width
        R: Int,   # Kernel height
        S: Int,   # Kernel width
        # Convolution parameters
        stride_h: Int,
        stride_w: Int,
        pad_h: Int,
        pad_w: Int,
        dil_h: Int,
        dil_w: Int,
    ](
        grad_weight: OutputTensor[dtype=dtype, rank=rank_grad_weight],
        grad_output: InputTensor[dtype=dtype, rank=rank_grad_out],
        input: InputTensor[dtype=dtype, rank=rank_input],
        ctx: DeviceContextPtr,
    ) raises:
        """
        Execute the grad_weight computation with proper GPU parallelization.

        All tensor dimensions are passed as compile-time parameters.
        """

        @parameter
        if target == "cpu":
            # CPU implementation: Nested loops parallelized at outer level
            # Nested loops - outer loops can be parallelized by compiler
            for f in range(F):
                for c in range(C):
                    for r in range(R):
                        for s in range(S):
                            var accumulator = Scalar[dtype](0)

                            # Accumulate over batch and spatial dimensions
                            for n in range(N):
                                for ho in range(HO):
                                    for wo in range(WO):
                                        # Compute input position
                                        var h = ho * stride_h - pad_h + r * dil_h
                                        var w = wo * stride_w - pad_w + s * dil_w

                                        # Bounds check
                                        if h >= 0 and h < H and w >= 0 and w < W:
                                            var grad_out_idx = IndexList[rank_grad_out](n, f, ho, wo)
                                            var input_idx = IndexList[rank_input](n, c, h, w)

                                            accumulator += grad_output[grad_out_idx] * input[input_idx]

                            # Store result using bracket notation
                            var weight_idx = IndexList[rank_grad_weight](f, c, r, s)
                            grad_weight[weight_idx] = accumulator
        else:
            # GPU implementation: One thread per weight element
            # Get global thread ID (linear indexing across all weight elements)
            var tid = Int(global_idx.x)

            # Total number of weight elements
            alias total_elements = F * C * R * S

            # Bounds check
            if tid >= total_elements:
                return

            # Decompose linear thread ID to 4D weight coordinates [f, c, r, s]
            # Using row-major order: index = f*(C*R*S) + c*(R*S) + r*S + s
            var s = tid % S
            var temp = tid // S
            var r = temp % R
            temp = temp // R
            var c = temp % C
            var f = temp // C

            # Accumulate gradient for this weight element
            var accumulator = Scalar[dtype](0)

            # Loop over batch and spatial output dimensions
            for n in range(N):
                for ho in range(HO):
                    for wo in range(WO):
                        # Compute corresponding input position
                        var h = ho * stride_h - pad_h + r * dil_h
                        var w = wo * stride_w - pad_w + s * dil_w

                        # Bounds check for input position
                        if h >= 0 and h < H and w >= 0 and w < W:
                            # Use direct indexing (bracket notation)
                            var grad_out_idx = IndexList[rank_grad_out](n, f, ho, wo)
                            var grad_out_val = grad_output[grad_out_idx]

                            var input_idx = IndexList[rank_input](n, c, h, w)
                            var input_val = input[input_idx]

                            # Accumulate: grad_weight += grad_output * input
                            accumulator += grad_out_val * input_val

            # Store result using bracket notation
            var weight_idx = IndexList[rank_grad_weight](f, c, r, s)
            grad_weight[weight_idx] = accumulator
