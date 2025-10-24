# Shared Memory Patterns

Shared memory is fast, on-chip memory shared by all threads in a block. It enables efficient inter-thread communication and data reuse.

## Shared Memory Basics

### Allocation

```mojo
# Static allocation with stack_allocation
var shared = stack_allocation[
    SIZE,                              # Number of elements
    DType.float32,                     # Data type
    address_space = AddressSpace.SHARED  # Memory space
]()
```

### Properties

- **Speed**: ~100x faster than global memory
- **Scope**: Shared within thread block only
- **Size**: Typically 48-96 KB per SM (varies by GPU)
- **Lifetime**: Exists only during kernel execution
- **Banks**: Organized into 32 banks (on NVIDIA)

## Bank Conflicts

Shared memory is divided into banks. When multiple threads access different addresses in the same bank simultaneously, a bank conflict occurs.

### Conflict-free Access

```mojo
# Good: Sequential access (threads 0-31 access indices 0-31)
var shared = stack_allocation[256, DType.float32, address_space = AddressSpace.SHARED]()
var tid = thread_idx.x
shared[tid] = value  # Conflict-free
```

### Bank Conflicts

```mojo
# Bad: Strided access with stride = number of banks (32)
shared[tid * 32] = value  # All threads access same bank!
```

### Padding to Avoid Conflicts

```mojo
# Without padding: conflicts on stride-32 access
var shared = stack_allocation[32 * 32, DType.float32, address_space = AddressSpace.SHARED]()

# With padding: add extra element per row
alias PADDED_WIDTH = 32 + 1
var shared = stack_allocation[32 * PADDED_WIDTH, DType.float32, address_space = AddressSpace.SHARED]()

# Access: shared[row * PADDED_WIDTH + col]
```

## Synchronization

Always synchronize after writing to shared memory before reading:

```mojo
# Write to shared memory
shared[thread_idx.x] = input[global_idx.x]

# MUST synchronize before reading
barrier()

# Now safe to read shared memory
var value = shared[some_other_index]
```

**Critical**: All threads in a block must execute the same `barrier()` call. Divergent control flow can cause deadlock:

```mojo
# WRONG: Deadlock if some threads don't reach barrier
if thread_idx.x < 64:
    shared[thread_idx.x] = value
    barrier()  # Only some threads reach this!

# RIGHT: All threads reach barrier
if thread_idx.x < 64:
    shared[thread_idx.x] = value
barrier()  # All threads reach this
```

## Common Patterns

### Pattern 1: Matrix Transpose

Transpose a tile using shared memory to achieve coalesced access:

```mojo
fn transpose_tile_kernel(
    output: UnsafePointer[Float32],  # Shape: (N, M)
    input: UnsafePointer[Float32],   # Shape: (M, N)
    M: Int,
    N: Int,
):
    alias TILE_SIZE = 32
    alias PADDED_SIZE = 33  # Avoid bank conflicts

    # Shared memory for tile
    var tile = stack_allocation[
        TILE_SIZE * PADDED_SIZE,
        DType.float32,
        address_space = AddressSpace.SHARED
    ]()

    var tx = thread_idx.x
    var ty = thread_idx.y

    var row = block_idx.y * TILE_SIZE + ty
    var col = block_idx.x * TILE_SIZE + tx

    # Coalesced read from input
    if row < UInt(M) and col < UInt(N):
        tile[ty * PADDED_SIZE + tx] = input[row * UInt(N) + col]

    barrier()

    # Transpose indices for output
    row = block_idx.x * TILE_SIZE + ty
    col = block_idx.y * TILE_SIZE + tx

    # Coalesced write to output (transposed)
    if row < UInt(N) and col < UInt(M):
        output[row * UInt(M) + col] = tile[tx * PADDED_SIZE + ty]
```

Launch with 2D blocks:
```mojo
var grid_x = ceildiv(N, TILE_SIZE)
var grid_y = ceildiv(M, TILE_SIZE)

ctx.enqueue_function_checked[transpose_tile_kernel, transpose_tile_kernel](
    output, input, M, N,
    grid_dim=(grid_x, grid_y),
    block_dim=(TILE_SIZE, TILE_SIZE),
)
```

### Pattern 2: Parallel Prefix Sum (Scan)

Compute cumulative sum using shared memory:

```mojo
fn block_scan_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    len: Int,
):
    alias BLOCK_SIZE = 256

    var shared = stack_allocation[
        BLOCK_SIZE,
        DType.float32,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gid = global_idx.x

    # Load data into shared memory
    if gid < UInt(len):
        shared[tid] = input[gid]
    else:
        shared[tid] = 0.0

    barrier()

    # Up-sweep (reduce)
    var offset = 1
    while offset < BLOCK_SIZE:
        var index = (tid + 1) * offset * 2 - 1
        if index < BLOCK_SIZE:
            shared[index] += shared[index - offset]

        offset *= 2
        barrier()

    # Down-sweep (distribute)
    # ... (implementation continues)

    if gid < UInt(len):
        output[gid] = shared[tid]
```

### Pattern 3: Stencil Computation

Use shared memory to cache neighbors for reuse:

```mojo
fn stencil_1d_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
    radius: Int,
):
    alias BLOCK_SIZE = 256

    # Shared memory: block + 2*radius for halos
    var shared_size = BLOCK_SIZE + 2 * radius
    var shared = stack_allocation[
        shared_size,
        DType.float32,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gid = global_idx.x
    var local_idx = tid + radius  # Offset for halo region

    # Load main data
    if gid < UInt(size):
        shared[local_idx] = input[gid]

    # Load left halo (first threads)
    if tid < radius:
        var left_idx = Int(gid) - radius
        shared[tid] = (left_idx >= 0).select(input[left_idx], 0.0)

    # Load right halo (last threads)
    if tid >= BLOCK_SIZE - radius:
        var right_idx = Int(gid) + radius
        var right_local = local_idx + radius
        shared[right_local] = (right_idx < size).select(input[right_idx], 0.0)

    barrier()

    # Compute stencil
    if gid < UInt(size):
        var sum = 0.0
        for r in range(-radius, radius + 1):
            sum += shared[local_idx + r]

        output[gid] = sum / Float32((2 * radius + 1))
```

### Pattern 4: Histogram

Compute histogram using shared memory bins with atomics:

```mojo
fn histogram_kernel(
    histogram: UnsafePointer[Int],  # Output: (NUM_BINS,)
    input: UnsafePointer[Float32],  # Input values [0, 1)
    size: Int,
):
    alias NUM_BINS = 256
    alias BLOCK_SIZE = 256

    # Shared memory for local histogram
    var shared_hist = stack_allocation[
        NUM_BINS,
        DType.int32,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x

    # Initialize shared histogram
    if tid < NUM_BINS:
        shared_hist[tid] = 0

    barrier()

    # Each thread processes multiple elements
    var gid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    var idx = gid
    while idx < UInt(size):
        var value = input[idx]
        var bin = Int(value * Float32(NUM_BINS))
        bin = min(bin, NUM_BINS - 1)  # Clamp to valid range

        # Atomic increment to shared memory
        atomic_add(shared_hist + bin, 1)

        idx += stride

    barrier()

    # Merge shared histogram to global
    if tid < NUM_BINS:
        if shared_hist[tid] > 0:
            atomic_add(histogram + tid, shared_hist[tid])
```

### Pattern 5: Reduction with Shared Memory

Already covered in reduction_patterns.md, but key structure:

```mojo
fn block_reduce_with_shared[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    alias BLOCK_SIZE = 256

    var shared = stack_allocation[
        BLOCK_SIZE,
        dtype,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    shared[tid] = val

    barrier()

    # Reduction tree in shared memory
    var offset = BLOCK_SIZE // 2
    while offset > 0:
        if tid < offset:
            shared[tid] += shared[tid + offset]

        offset //= 2
        barrier()

    # First thread has result
    return shared[0]
```

## 2D Stencil Pattern

Convolution-like operations require 2D halo:

```mojo
fn stencil_2d_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    width: Int,
    height: Int,
):
    alias TILE_X = 16
    alias TILE_Y = 16
    alias RADIUS = 1

    # Shared memory: tile + halo
    alias SHARED_X = TILE_X + 2 * RADIUS
    alias SHARED_Y = TILE_Y + 2 * RADIUS

    var shared = stack_allocation[
        SHARED_X * SHARED_Y,
        DType.float32,
        address_space = AddressSpace.SHARED
    ]()

    var tx = thread_idx.x
    var ty = thread_idx.y

    var gx = block_idx.x * TILE_X + tx
    var gy = block_idx.y * TILE_Y + ty

    # Load main tile
    var shared_x = tx + RADIUS
    var shared_y = ty + RADIUS

    if gx < UInt(width) and gy < UInt(height):
        shared[shared_y * SHARED_X + shared_x] = input[gy * UInt(width) + gx]

    # Load halos (4 edges + 4 corners)
    # Top edge
    if ty < RADIUS and gy >= RADIUS:
        var halo_y = ty
        var halo_gy = gy - RADIUS
        if gx < UInt(width):
            shared[halo_y * SHARED_X + shared_x] = input[halo_gy * UInt(width) + gx]

    # Bottom edge
    if ty >= TILE_Y - RADIUS and gy + RADIUS < UInt(height):
        var halo_y = shared_y + RADIUS
        var halo_gy = gy + RADIUS
        if gx < UInt(width):
            shared[halo_y * SHARED_X + shared_x] = input[halo_gy * UInt(width) + gx]

    # Left/right edges and corners similar...

    barrier()

    # Compute 3x3 stencil
    if gx < UInt(width) and gy < UInt(height):
        var sum = 0.0

        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                var sy = shared_y + dy
                var sx = shared_x + dx
                sum += shared[sy * SHARED_X + sx]

        output[gy * UInt(width) + gx] = sum / 9.0
```

## Memory Access Patterns

### Coalesced Load/Store

```mojo
# Good: Coalesced
shared[tid] = global_input[gid]

# Bad: Strided (if stride != 1)
shared[tid] = global_input[gid * stride]
```

### Avoiding Bank Conflicts

```mojo
# Matrix stored row-major
# Access: shared[row][col]

# Without padding: stride = width
alias WIDTH = 32
var index = row * WIDTH + col  # Bank conflicts if WIDTH = 32

# With padding: stride = width + 1
alias PADDED_WIDTH = 33
var index = row * PADDED_WIDTH + col  # No conflicts
```

## Size Limitations

Check GPU shared memory limits:
```mojo
# Typical limits:
# - 48 KB per SM (older GPUs)
# - 64-96 KB per SM (newer GPUs)
# - Can configure L1 cache vs shared memory split

alias MAX_SHARED = 48 * 1024  # bytes
alias ELEMENT_SIZE = 4  # Float32
alias MAX_ELEMENTS = MAX_SHARED / ELEMENT_SIZE  # 12,288 elements
```

If you need more:
- Use multiple kernel launches
- Process data in tiles
- Use global memory for overflow

## Performance Tips

### 1. Minimize Synchronization

```mojo
# Bad: Unnecessary barriers
for i in range(N):
    shared[tid] = compute(i)
    barrier()  # Barrier inside loop!

# Good: Single barrier
for i in range(N):
    shared[tid] = compute(i)
barrier()  # Single barrier after loop
```

### 2. Maximize Reuse

Load data once, use multiple times:
```mojo
# Load tile into shared memory
shared[local_idx] = input[global_idx]
barrier()

# Reuse for multiple operations
var result1 = operation1(shared[...])
var result2 = operation2(shared[...])
var result3 = operation3(shared[...])
```

### 3. Avoid Bank Conflicts

Use padding or reorganize access patterns:
```mojo
# Add padding to multiples of 32
alias PADDED = (SIZE + 31) & ~31  # Round up to multiple of 32
var shared = stack_allocation[PADDED + 1, ...]()  # +1 avoids conflict
```

### 4. Optimize for Occupancy

Monitor shared memory usage:
```mojo
# Each block uses X bytes of shared memory
# GPU has Y bytes per SM
# Max active blocks per SM = Y / X (up to hardware limit)
#
# Example:
# - 48 KB shared per SM
# - Each block uses 16 KB
# - Max 3 blocks per SM active simultaneously
```

## Debugging Shared Memory

### Common Issues

1. **Race conditions**: Missing `barrier()`
2. **Deadlock**: Divergent control flow before `barrier()`
3. **Out of bounds**: Accessing shared[index >= SIZE]
4. **Bank conflicts**: Performance degradation
5. **Excessive usage**: Reduces occupancy

### Debugging Techniques

```mojo
# Bounds checking
debug_assert(index < SIZE, "Shared memory out of bounds")

# Initialization checking (for debugging)
if thread_idx.x == 0:
    for i in range(SIZE):
        shared[i] = -999.0  # Sentinel value

barrier()

# After computation, check for unmodified sentinel values
```

## Testing Shared Memory Kernels

```python
import torch
import pytest

def test_stencil_shared_memory():
    size = 1000
    radius = 2

    input_tensor = torch.randn(size, device='cuda')

    # Reference implementation (CPU)
    expected = torch.zeros_like(input_tensor)
    for i in range(size):
        start = max(0, i - radius)
        end = min(size, i + radius + 1)
        expected[i] = input_tensor[start:end].mean()

    # GPU kernel with shared memory
    result = stencil_kernel(input_tensor, radius)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

## Quick Reference

**Allocation**:
```mojo
var shared = stack_allocation[SIZE, dtype, address_space = AddressSpace.SHARED]()
```

**Synchronization**:
```mojo
barrier()  # All threads in block must reach this
```

**Bank conflicts**: Pad arrays when stride = 32:
```mojo
alias PADDED_WIDTH = WIDTH + 1
```

**Size limits**: 48-96 KB per block (GPU dependent)

**Use cases**:
- Reductions
- Matrix operations (transpose, multiply)
- Stencils (convolution, pooling)
- Histogram
- Sorting
- Prefix sum

**Best practices**:
- Always `barrier()` between write and read
- Pad arrays to avoid bank conflicts
- Minimize synchronization points
- Monitor shared memory usage for occupancy
