"""
Element-wise Kernel Template

This template provides a starting point for implementing element-wise operations
that process each element independently.

Replace OPERATION_NAME, operation(), and configure parameters as needed.
"""

from gpu import global_idx, barrier, WARP_SIZE
from gpu.host import DeviceContext, DeviceBuffer
from math import ceildiv

# ============================================================================
# Kernel Implementation
# ============================================================================

fn OPERATION_NAME_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """
    Element-wise operation kernel.

    Args:
        output: Output buffer
        input: Input buffer
        len: Number of elements to process
    """
    var tid = global_idx.x

    # Bounds check
    if tid >= UInt(len):
        return

    # Perform element-wise operation
    output[tid] = operation(input[tid])


@always_inline
fn operation[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """
    The actual element-wise operation to perform.

    Replace this with your specific operation.
    Examples:
        return x * x              # Square
        return max(x, 0)          # ReLU
        return 1.0 / (1.0 + exp(-x))  # Sigmoid
    """
    return x  # TODO: Replace with actual operation


# ============================================================================
# Vectorized Variant (Optional - for better performance)
# ============================================================================

fn OPERATION_NAME_vectorized_kernel[
    dtype: DType,
    simd_width: Int,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """
    Vectorized element-wise operation kernel.
    Processes simd_width elements per thread.
    """
    var tid = global_idx.x
    var idx = tid * UInt(simd_width)

    if idx >= UInt(len):
        return

    # Check if we have full vector or need to handle tail
    if idx + UInt(simd_width) <= UInt(len):
        # Full vector
        var vec_in = input.load[width=simd_width](idx)
        var vec_out = vectorized_operation[dtype, simd_width](vec_in)
        output.store[width=simd_width](idx, vec_out)
    else:
        # Tail: process remaining elements scalar
        for i in range(simd_width):
            var elem_idx = idx + UInt(i)
            if elem_idx < UInt(len):
                output[elem_idx] = operation[dtype](input[elem_idx])


@always_inline
fn vectorized_operation[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Vectorized version of the operation.
    Must operate on SIMD vectors.
    """
    return x  # TODO: Replace with vectorized operation


# ============================================================================
# Binary Element-wise Template
# ============================================================================

fn BINARY_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input1: UnsafePointer[Scalar[dtype]],
    input2: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """
    Binary element-wise operation: output[i] = op(input1[i], input2[i])
    """
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    output[tid] = binary_operation(input1[tid], input2[tid])


@always_inline
fn binary_operation[dtype: DType](
    x: Scalar[dtype],
    y: Scalar[dtype]
) -> Scalar[dtype]:
    """
    Binary operation.
    Examples: x + y, x * y, max(x, y), pow(x, y)
    """
    return x + y  # TODO: Replace with actual operation


# ============================================================================
# Scalar-Tensor Element-wise Template
# ============================================================================

fn SCALAR_TENSOR_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    scalar: Scalar[dtype],
    len: Int,
):
    """
    Element-wise operation with scalar: output[i] = op(input[i], scalar)
    """
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    output[tid] = scalar_operation(input[tid], scalar)


@always_inline
fn scalar_operation[dtype: DType](
    x: Scalar[dtype],
    scalar: Scalar[dtype]
) -> Scalar[dtype]:
    """
    Operation with scalar.
    Examples: x + scalar, x * scalar, pow(x, scalar)
    """
    return x * scalar  # TODO: Replace with actual operation


# ============================================================================
# Fused Operations Template
# ============================================================================

fn FUSED_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    alpha: Scalar[dtype],
    beta: Scalar[dtype],
    len: Int,
):
    """
    Fused operation to reduce memory traffic.
    Example: output[i] = relu(alpha * input[i] + beta)
    """
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    var x = input[tid]

    # Fused computation
    var temp = alpha * x + beta
    output[tid] = max(temp, 0.0)  # TODO: Replace with fused operation


# ============================================================================
# Launch Configuration Helper
# ============================================================================

fn get_launch_config(size: Int, block_size: Int = 256) -> (Int, Int):
    """
    Calculate grid and block dimensions for kernel launch.

    Args:
        size: Total number of elements
        block_size: Threads per block (default 256)

    Returns:
        (grid_dim, block_dim) tuple
    """
    var grid_dim = ceildiv(size, block_size)
    return (grid_dim, block_size)


fn get_vectorized_launch_config(
    size: Int,
    simd_width: Int,
    block_size: Int = 256
) -> (Int, Int):
    """
    Calculate launch config for vectorized kernel.

    Args:
        size: Total number of elements
        simd_width: Elements processed per thread
        block_size: Threads per block

    Returns:
        (grid_dim, block_dim) tuple
    """
    var num_threads = ceildiv(size, simd_width)
    var grid_dim = ceildiv(num_threads, block_size)
    return (grid_dim, block_size)


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of how to launch the kernel.
    """
    alias dtype = DType.float32
    alias size = 1000000

    with DeviceContext() as ctx:
        # Allocate buffers
        var input_device = ctx.enqueue_create_buffer[dtype](size)
        var output_device = ctx.enqueue_create_buffer[dtype](size)

        # Initialize input (example)
        # ... copy data to input_device ...

        # Get launch configuration
        var grid_dim, block_dim = get_launch_config(size)

        # Launch kernel
        alias kernel = OPERATION_NAME_kernel[dtype]
        ctx.enqueue_function_checked[kernel, kernel](
            output_device,
            input_device,
            size,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        # Copy result back
        # ... copy from output_device ...

        ctx.synchronize()


# ============================================================================
# Common Activation Functions (Reference Implementations)
# ============================================================================

@always_inline
fn relu_impl[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """ReLU: max(0, x)"""
    return max(x, 0)


@always_inline
fn leaky_relu_impl[dtype: DType](
    x: Scalar[dtype],
    negative_slope: Scalar[dtype] = 0.01
) -> Scalar[dtype]:
    """Leaky ReLU: x if x > 0 else negative_slope * x"""
    return (x > 0).select(x, negative_slope * x)


@always_inline
fn sigmoid_impl[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + exp(-x))


@always_inline
fn tanh_impl[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return math.tanh(x)


@always_inline
fn gelu_impl[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    alias inv_SQRT_2 = 0.70710678118654752440
    var val_half = 0.5 * x
    var erf_res = math.erf(x * inv_SQRT_2)
    return val_half.fma(erf_res, val_half)


@always_inline
fn silu_impl[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """SiLU/Swish: x * sigmoid(x)"""
    return x / (1.0 + exp(-x))
