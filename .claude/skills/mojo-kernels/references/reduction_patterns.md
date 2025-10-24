# Reduction Kernel Patterns

Reductions combine multiple values into a single result (sum, max, min, mean, etc.). Unlike element-wise operations, reductions require communication between threads.

## Warp-level Reduction

Fastest reduction strategy for small data (≤32 elements per group).

### Basic Warp Sum

```mojo
fn warp_sum_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """Sum using warp primitive - each warp produces one result"""
    var tid = global_idx.x

    if tid >= UInt(size):
        return

    # Each thread starts with one value
    var value = input[tid]

    # Warp-level sum reduction (across 32 threads)
    var warp_sum = warp.sum(value)

    # Only first thread in each warp writes result
    if lane_id() == 0:
        var warp_idx = tid / UInt(WARP_SIZE)
        output[warp_idx] = warp_sum
```

### Available Warp Primitives

```mojo
# Sum reduction
var sum = warp.sum(value)

# Max/Min reduction
var max_val = warp.max(value)
var min_val = warp.min(value)

# Broadcast value from lane 0 to all lanes
var broadcasted = warp.broadcast(value, source_lane=0)

# Shuffle: exchange data between lanes
var shuffled = warp.shuffle(value, src_lane)

# Partial warp reduction (for non-32 sizes)
var partial_sum = warp.lane_group_sum[num_lanes=16](value)
```

## Block-level Reduction

For larger data, reduce across entire thread block using shared memory.

### Two-stage Block Reduction

```mojo
@always_inline
fn block_reduce_sum[dtype: DType, max_warps_per_block: Int](
    val: Scalar[dtype]
) -> Scalar[dtype]:
    """Reduce across all threads in a block

    Strategy:
    1. Each warp reduces to one value (warp primitive)
    2. First warp reduces across warp results
    3. Result is broadcast to all threads
    """

    # Shared memory for inter-warp communication
    var warp_results = stack_allocation[
        max_warps_per_block,
        dtype,
        address_space = AddressSpace.SHARED
    ]()

    var broadcast = stack_allocation[
        1, dtype,
        address_space = AddressSpace.SHARED
    ]()

    # Stage 1: Reduce within each warp
    var warp_sum = warp.sum(val)

    var warp_idx = thread_idx.x // UInt(WARP_SIZE)
    var lane_idx = lane_id()

    # First thread of each warp stores warp result
    if lane_idx == 0:
        warp_results[warp_idx] = warp_sum

    # Synchronize to ensure all warp results are written
    barrier()

    # Stage 2: Final reduction across warp results
    # Only first warp participates
    if warp_idx == 0 and lane_idx < UInt(max_warps_per_block):
        var block_sum = warp.lane_group_sum[num_lanes=max_warps_per_block](
            warp_results[lane_idx]
        )

        # First thread writes final result
        if lane_idx == 0:
            broadcast[0] = block_sum

    # Synchronize to ensure result is written before read
    barrier()

    # All threads now have access to result
    return broadcast[0]
```

### Multi-block Global Reduction

For arrays larger than one block:

```mojo
fn global_sum_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
    block_size: Int,
):
    """Each block reduces its portion, results combined in second kernel"""

    var tid = global_idx.x
    var bid = block_idx.x

    # Each thread loads one element (or 0 if out of bounds)
    var value = Scalar[dtype](0)
    if tid < UInt(size):
        value = input[tid]

    # Reduce within block
    alias MAX_WARPS = 8  # For 256 threads: 256/32 = 8 warps
    var block_result = block_reduce_sum[dtype, MAX_WARPS](value)

    # First thread of each block writes partial result
    if thread_idx.x == 0:
        output[bid] = block_result
```

Then launch second kernel to sum block results:
```mojo
# First kernel: size -> num_blocks partial sums
var num_blocks = ceildiv(size, BLOCK_SIZE)
ctx.enqueue_function_checked[global_sum_kernel, global_sum_kernel](
    partial_results, input, size, BLOCK_SIZE,
    grid_dim=num_blocks,
    block_dim=BLOCK_SIZE,
)

# Second kernel: num_blocks -> 1 final sum
ctx.enqueue_function_checked[global_sum_kernel, global_sum_kernel](
    final_result, partial_results, num_blocks, BLOCK_SIZE,
    grid_dim=1,
    block_dim=BLOCK_SIZE,
)
```

## Reduction with Multiple Elements per Thread

For better efficiency, each thread can process multiple elements:

```mojo
fn reduce_multiple_per_thread[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    alias ELEMENTS_PER_THREAD = 4
    alias BLOCK_SIZE = 256

    var tid = global_idx.x
    var thread_sum = Scalar[dtype](0)

    # Each thread sums ELEMENTS_PER_THREAD elements
    for i in range(ELEMENTS_PER_THREAD):
        var idx = tid * ELEMENTS_PER_THREAD + i
        if idx < UInt(size):
            thread_sum += input[idx]

    # Now reduce thread_sum across block
    alias MAX_WARPS = BLOCK_SIZE // WARP_SIZE
    var block_result = block_reduce_sum[dtype, MAX_WARPS](thread_sum)

    if thread_idx.x == 0:
        output[block_idx.x] = block_result
```

Launch configuration:
```mojo
var elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD
var num_blocks = ceildiv(size, elements_per_block)
```

## Row-wise and Column-wise Reductions

For 2D data, reduce along specific dimension:

### Row-wise Reduction

```mojo
fn row_reduce_sum(
    output: UnsafePointer[Float32],  # Shape: (M,)
    input: UnsafePointer[Float32],   # Shape: (M, N)
    M: Int,
    N: Int,
):
    """Sum each row: output[i] = sum(input[i, :])"""

    var row_idx = block_idx.x
    var tid = thread_idx.x

    if row_idx >= UInt(M):
        return

    # Thread-local sum for this row
    var thread_sum = 0.0

    # Stride across row, each thread processes every BLOCK_SIZE-th element
    var idx = row_idx * UInt(N) + tid
    var stride = block_dim.x

    while tid < UInt(N):
        thread_sum += input[idx]
        tid += stride
        idx += stride

    # Reduce across threads in block
    alias MAX_WARPS = 8
    var row_sum = block_reduce_sum[DType.float32, MAX_WARPS](thread_sum)

    if thread_idx.x == 0:
        output[row_idx] = row_sum
```

Launch: One block per row
```mojo
ctx.enqueue_function_checked[row_reduce_sum, row_reduce_sum](
    output, input, M, N,
    grid_dim=M,
    block_dim=256,
)
```

### Column-wise Reduction (Atomic)

```mojo
fn col_reduce_sum_atomic(
    output: UnsafePointer[Float32],  # Shape: (N,)
    input: UnsafePointer[Float32],   # Shape: (M, N)
    M: Int,
    N: Int,
):
    """Sum each column using atomics"""

    var tid = global_idx.x
    var total_size = M * N

    if tid >= UInt(total_size):
        return

    var col_idx = tid % UInt(N)

    # Each thread atomically adds its value to output column
    atomic_add(output + col_idx, input[tid])
```

Note: Atomic operations can be slower due to contention.

## MAX Framework Reduction Patterns

The MAX framework provides high-level reduction primitives:

```mojo
@parameter
fn reduce_fn[dtype: DType, width: Int, reduction_idx: Int](
    lhs: SIMD[dtype, width],
    rhs: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    return lhs + rhs  # or max, min, etc.

@parameter
fn input_fn[dtype: DType, width: Int, rank: Int](
    coords: IndexList[rank]
) -> SIMD[dtype, width]:
    return input_buffer.load[width=width](coords)

@parameter
fn output_fn[dtype: DType, width: Int, rank: Int](
    coords: IndexList[rank],
    val: StaticTuple[SIMD[dtype, width], num_reductions],
):
    output_buffer[coords] = val[0]

reduce_launch[
    num_reductions,
    input_fn,
    output_fn,
    reduce_fn,
    rank,
    dtype
](shape, axis, init_value, ctx)
```

This handles:
- Optimal block/grid configuration
- Multi-stage reduction
- Edge cases and alignment
- Vectorization

## Common Reduction Operations

### Sum
```mojo
var result = warp.sum(value)
var result = block_reduce_sum[dtype, MAX_WARPS](value)
```

### Max/Min
```mojo
var max_val = warp.max(value)
var min_val = warp.min(value)

# For block-level, modify block_reduce to use max/min instead of sum
```

### Mean
```mojo
var sum = block_reduce_sum[dtype, MAX_WARPS](value)
var mean = sum / Scalar[dtype](N)
```

### Variance
```mojo
# Two-pass: first compute mean, then variance
var mean = compute_mean(...)

var deviation = value - mean
var squared_dev = deviation * deviation
var variance = block_reduce_sum[dtype, MAX_WARPS](squared_dev) / Scalar[dtype](N)
```

### ArgMax (Index of Maximum)

```mojo
struct IndexedValue[dtype: DType]:
    var value: Scalar[dtype]
    var index: Int

    fn max(self, other: Self) -> Self:
        if self.value > other.value:
            return self
        return other

fn argmax_warp_reduce(val: IndexedValue[dtype]) -> IndexedValue[dtype]:
    # Warp shuffle-based reduction with index tracking
    # ... implementation using warp.shuffle ...
```

## Performance Considerations

### Memory Access Patterns

- **Coalesced loads**: Consecutive threads load consecutive addresses
- **Shared memory**: Minimize bank conflicts (pad arrays if needed)
- **Atomic operations**: Use sparingly; prefer structured reductions

### Synchronization Overhead

- **Barrier**: Expensive; minimize usage
- **Warp primitives**: No explicit sync needed within warp
- **Minimize communication**: More work per thread = fewer sync points

### Occupancy

- **Shared memory usage**: More shared memory = fewer active blocks
- **Register usage**: Keep local variables minimal
- **Block size**: Typically 256 threads balances occupancy and reduction efficiency

### Launch Configuration

```mojo
# Good block sizes for reductions
alias BLOCK_SIZE = 256  # 8 warps
alias BLOCK_SIZE = 512  # 16 warps (if registers permit)

# Grid size depends on reduction strategy
var num_blocks = ceildiv(size, BLOCK_SIZE)  # First pass
# Second pass reduces num_blocks partial results
```

## Testing Template

```python
import torch
import pytest

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [32, 1000, 10000, 1000000])
@pytest.mark.parametrize("reduction", ["sum", "max", "min", "mean"])
def test_reduction(dtype, size, reduction):
    input_tensor = torch.randn(size, dtype=dtype, device='cuda')

    expected = getattr(torch, reduction)(input_tensor)
    result = custom_reduction_kernel(input_tensor, reduction)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

## Common Pitfalls

1. **Missing synchronization**: Not calling `barrier()` between stages
2. **Incorrect warp count**: Not accounting for partial last warp
3. **Bank conflicts**: Accessing shared memory with stride-32 patterns
4. **Race conditions**: Multiple threads writing to same location without atomics
5. **Incorrect initialization**: Not initializing reduction identity (0 for sum, -inf for max)
6. **Overflow**: Not using accumulation type for large reductions

## Quick Reference

**Warp reduction** (fastest):
- Up to 32 elements
- No shared memory needed
- `warp.sum()`, `warp.max()`, `warp.min()`

**Block reduction** (medium):
- Up to 1024 elements (typical block size)
- Uses shared memory
- Two-stage: warp → block

**Multi-block reduction** (large arrays):
- Unlimited size
- Multiple kernel launches
- Each block → partial result → final reduction

**Use high-level MAX patterns when available** for optimal performance without manual tuning.
