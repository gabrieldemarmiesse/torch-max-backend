"""
Reduction Kernel Templates

This file provides templates for implementing reduction operations (sum, max, min, etc.)
at warp-level, block-level, and multi-block levels.
"""

from gpu import (
    global_idx,
    block_idx,
    thread_idx,
    block_dim,
    grid_dim,
    barrier,
    lane_id,
    WARP_SIZE,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
import gpu.warp as warp
from memory import stack_allocation
from math import ceildiv

# ============================================================================
# Warp-level Reduction Template
# ============================================================================

fn warp_reduce_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """
    Warp-level reduction kernel.
    Each warp reduces WARP_SIZE (32) elements to one result.

    Args:
        output: Output buffer (size = ceildiv(size, WARP_SIZE))
        input: Input buffer (size elements)
        size: Number of elements
    """
    var tid = global_idx.x

    if tid >= UInt(size):
        return

    # Each thread loads one value
    var value = input[tid]

    # Perform warp reduction
    var warp_result = warp_reduce_op[dtype](value)

    # First thread in warp writes result
    if lane_id() == 0:
        var warp_idx = tid / UInt(WARP_SIZE)
        output[warp_idx] = warp_result


@always_inline
fn warp_reduce_op[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """
    Warp reduction operation.

    Available warp primitives:
        warp.sum(value)      # Sum reduction
        warp.max(value)      # Max reduction
        warp.min(value)      # Min reduction

    Replace with appropriate operation.
    """
    return warp.sum(value)  # TODO: Change to desired reduction


# ============================================================================
# Block-level Reduction Template
# ============================================================================

@always_inline
fn block_reduce_OPERATION[
    dtype: DType,
    max_warps_per_block: Int,
](val: Scalar[dtype]) -> Scalar[dtype]:
    """
    Reduce across all threads in a block.

    Two-stage reduction:
    1. Reduce within each warp (using warp primitive)
    2. Reduce across warp results (first warp only)

    Args:
        val: Value from each thread
        max_warps_per_block: Maximum warps in block (e.g., 256 threads = 8 warps)

    Returns:
        Reduced value (same across all threads in block)
    """
    # Shared memory for warp results
    var warp_results = stack_allocation[
        max_warps_per_block,
        dtype,
        address_space = AddressSpace.SHARED
    ]()

    var broadcast = stack_allocation[
        1, dtype,
        address_space = AddressSpace.SHARED
    ]()

    # Stage 1: Reduce within warp
    var warp_result = warp_reduce_op[dtype](val)

    var warp_idx = thread_idx.x // UInt(WARP_SIZE)
    var lane_idx = lane_id()

    # First thread of each warp stores result
    if lane_idx == 0:
        warp_results[warp_idx] = warp_result

    barrier()

    # Stage 2: First warp reduces across warp results
    if warp_idx == 0 and lane_idx < UInt(max_warps_per_block):
        var final_result = cross_warp_reduce_op[dtype, max_warps_per_block](
            warp_results[lane_idx]
        )

        if lane_idx == 0:
            broadcast[0] = final_result

    barrier()

    return broadcast[0]


@always_inline
fn cross_warp_reduce_op[dtype: DType, num_warps: Int](
    value: Scalar[dtype]
) -> Scalar[dtype]:
    """
    Reduce across warp results.

    For sum: warp.lane_group_sum[num_lanes=num_warps](value)
    For max: warp.lane_group_max[num_lanes=num_warps](value)
    For min: warp.lane_group_min[num_lanes=num_warps](value)
    """
    return warp.lane_group_sum[num_lanes=num_warps](value)  # TODO: Change as needed


# ============================================================================
# Multi-block Global Reduction Template
# ============================================================================

fn global_reduce_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """
    First-pass kernel: Each block reduces its portion.

    This kernel is launched with multiple blocks. Each block produces
    one partial result. A second kernel launch then reduces these
    partial results to a final answer.

    Args:
        output: Partial results (one per block)
        input: Input data
        size: Total number of input elements
    """
    alias BLOCK_SIZE = 256
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE

    var tid = global_idx.x
    var bid = block_idx.x

    # Load value (or reduction identity if out of bounds)
    var value = get_identity[dtype]()
    if tid < UInt(size):
        value = input[tid]

    # Reduce within block
    var block_result = block_reduce_OPERATION[dtype, MAX_WARPS](value)

    # First thread writes block's result
    if thread_idx.x == 0:
        output[bid] = block_result


@always_inline
fn get_identity[dtype: DType]() -> Scalar[dtype]:
    """
    Return identity element for the reduction operation.

    Examples:
        Sum: 0
        Product: 1
        Max: -inf
        Min: +inf
    """
    return Scalar[dtype](0)  # TODO: Change based on operation


# ============================================================================
# Multi-element per Thread Template
# ============================================================================

fn reduce_multiple_per_thread_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """
    Each thread processes multiple elements before block reduction.
    This reduces synchronization overhead.

    Args:
        output: Partial results (one per block)
        input: Input data
        size: Total number of input elements
    """
    alias ELEMENTS_PER_THREAD = 4
    alias BLOCK_SIZE = 256
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE

    var tid = global_idx.x
    var thread_local = get_identity[dtype]()

    # Each thread accumulates ELEMENTS_PER_THREAD values
    for i in range(ELEMENTS_PER_THREAD):
        var idx = tid * ELEMENTS_PER_THREAD + i
        if idx < UInt(size):
            thread_local = combine[dtype](thread_local, input[idx])

    # Now reduce thread_local across block
    var block_result = block_reduce_OPERATION[dtype, MAX_WARPS](thread_local)

    if thread_idx.x == 0:
        output[block_idx.x] = block_result


@always_inline
fn combine[dtype: DType](
    a: Scalar[dtype],
    b: Scalar[dtype]
) -> Scalar[dtype]:
    """
    Combine two values according to reduction operation.

    Examples:
        Sum: a + b
        Max: max(a, b)
        Min: min(a, b)
        Product: a * b
    """
    return a + b  # TODO: Change based on operation


# ============================================================================
# Row-wise Reduction Template
# ============================================================================

fn row_reduce_OPERATION_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],  # Shape: (M,)
    input: UnsafePointer[Scalar[dtype]],   # Shape: (M, N)
    M: Int,
    N: Int,
):
    """
    Reduce each row independently: output[i] = reduce(input[i, :])

    Launch with grid_dim=M, block_dim=256

    Args:
        output: Row reduction results (M elements)
        input: Input matrix (M x N)
        M: Number of rows
        N: Number of columns
    """
    alias BLOCK_SIZE = 256
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE

    var row_idx = block_idx.x
    var tid = thread_idx.x

    if row_idx >= UInt(M):
        return

    # Thread-local accumulator
    var thread_result = get_identity[dtype]()

    # Stride across row
    var base_idx = row_idx * UInt(N)
    var col_idx = tid

    while col_idx < UInt(N):
        thread_result = combine[dtype](
            thread_result,
            input[base_idx + col_idx]
        )
        col_idx += block_dim.x

    # Reduce across threads in block
    var row_result = block_reduce_OPERATION[dtype, MAX_WARPS](thread_result)

    if thread_idx.x == 0:
        output[row_idx] = row_result


# ============================================================================
# ArgMax/ArgMin Template (Reduction with Index)
# ============================================================================

struct IndexedValue[dtype: DType]:
    """Value paired with its index for argmax/argmin"""
    var value: Scalar[dtype]
    var index: Int

    @always_inline
    fn __init__(inout self, value: Scalar[dtype], index: Int):
        self.value = value
        self.index = index

    @always_inline
    fn max_by_value(self, other: Self) -> Self:
        """Return the IndexedValue with larger value"""
        if self.value > other.value:
            return self
        return other

    @always_inline
    fn min_by_value(self, other: Self) -> Self:
        """Return the IndexedValue with smaller value"""
        if self.value < other.value:
            return self
        return other


fn argmax_kernel[dtype: DType](
    max_values: UnsafePointer[Scalar[dtype]],
    max_indices: UnsafePointer[Int],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """
    Find maximum value and its index.

    Note: This is a simplified template. Full implementation requires
    custom warp reduction for IndexedValue type.

    Args:
        max_values: Output maximum values (one per block)
        max_indices: Output indices of maximum values
        input: Input data
        size: Number of elements
    """
    alias BLOCK_SIZE = 256

    # Shared memory for values and indices
    var shared_values = stack_allocation[
        BLOCK_SIZE, dtype,
        address_space = AddressSpace.SHARED
    ]()

    var shared_indices = stack_allocation[
        BLOCK_SIZE, DType.int32,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gid = global_idx.x

    # Load value and index
    var value = Scalar[dtype](-1e38)  # -inf
    var index = -1

    if gid < UInt(size):
        value = input[gid]
        index = Int(gid)

    shared_values[tid] = value
    shared_indices[tid] = index

    barrier()

    # Reduction tree
    var offset = BLOCK_SIZE // 2
    while offset > 0:
        if tid < offset:
            if shared_values[tid + offset] > shared_values[tid]:
                shared_values[tid] = shared_values[tid + offset]
                shared_indices[tid] = shared_indices[tid + offset]

        offset //= 2
        barrier()

    # First thread writes result
    if tid == 0:
        max_values[block_idx.x] = shared_values[0]
        max_indices[block_idx.x] = shared_indices[0]


# ============================================================================
# Variance/Standard Deviation Template (Two-pass)
# ============================================================================

fn variance_kernel_pass1[dtype: DType](
    mean_output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """First pass: Compute mean"""
    alias BLOCK_SIZE = 256
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE

    var tid = global_idx.x
    var value = Scalar[dtype](0)

    if tid < UInt(size):
        value = input[tid]

    var sum = block_reduce_OPERATION[dtype, MAX_WARPS](value)

    if thread_idx.x == 0:
        # Store partial sum for mean computation
        mean_output[block_idx.x] = sum


fn variance_kernel_pass2[dtype: DType](
    variance_output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    mean: Scalar[dtype],
    size: Int,
):
    """Second pass: Compute variance given mean"""
    alias BLOCK_SIZE = 256
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE

    var tid = global_idx.x
    var squared_deviation = Scalar[dtype](0)

    if tid < UInt(size):
        var deviation = input[tid] - mean
        squared_deviation = deviation * deviation

    var sum_sq_dev = block_reduce_OPERATION[dtype, MAX_WARPS](squared_deviation)

    if thread_idx.x == 0:
        variance_output[block_idx.x] = sum_sq_dev


# ============================================================================
# Launch Configuration Helpers
# ============================================================================

fn get_reduction_launch_config(
    size: Int,
    block_size: Int = 256
) -> (Int, Int):
    """
    Get launch configuration for single-pass reduction.

    Returns:
        (grid_dim, block_dim)
    """
    var grid_dim = ceildiv(size, block_size)
    return (grid_dim, block_size)


fn get_multi_element_launch_config(
    size: Int,
    elements_per_thread: Int = 4,
    block_size: Int = 256
) -> (Int, Int):
    """
    Get launch configuration for multi-element per thread reduction.

    Returns:
        (grid_dim, block_dim)
    """
    var elements_per_block = block_size * elements_per_thread
    var grid_dim = ceildiv(size, elements_per_block)
    return (grid_dim, block_size)


# ============================================================================
# Example Usage
# ============================================================================

def example_multi_block_reduction():
    """
    Example: Complete multi-block reduction to single value.
    """
    alias dtype = DType.float32
    alias size = 1000000
    alias BLOCK_SIZE = 256

    with DeviceContext() as ctx:
        # Allocate input
        var input_device = ctx.enqueue_create_buffer[dtype](size)

        # First pass: reduce to partial results
        var num_blocks = ceildiv(size, BLOCK_SIZE)
        var partial_device = ctx.enqueue_create_buffer[dtype](num_blocks)

        alias kernel1 = global_reduce_OPERATION_kernel[dtype]
        ctx.enqueue_function_checked[kernel1, kernel1](
            partial_device,
            input_device,
            size,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )

        # Second pass: reduce partial results to final value
        var result_device = ctx.enqueue_create_buffer[dtype](1)

        alias kernel2 = global_reduce_OPERATION_kernel[dtype]
        ctx.enqueue_function_checked[kernel2, kernel2](
            result_device,
            partial_device,
            num_blocks,
            grid_dim=1,
            block_dim=BLOCK_SIZE,
        )

        # Copy result to host
        # ...

        ctx.synchronize()
