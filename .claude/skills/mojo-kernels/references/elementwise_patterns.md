# Element-wise Kernel Patterns

Element-wise operations process each element independently, making them highly parallelizable and straightforward to implement on GPUs.

## Basic Element-wise Kernel

```mojo
fn elementwise_add_kernel(
    output: UnsafePointer[Float32],
    input1: UnsafePointer[Float32],
    input2: UnsafePointer[Float32],
    len: Int,
):
    """Add two arrays element-wise: output[i] = input1[i] + input2[i]"""
    var tid = global_idx.x

    # Bounds check - critical for correctness
    if tid >= UInt(len):
        return

    output[tid] = input1[tid] + input2[tid]
```

## Vectorized Element-wise Kernel

For better performance, process multiple elements per thread:

```mojo
fn vectorized_elementwise_kernel[
    dtype: DType,
    simd_width: Int,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Process simd_width elements per thread"""
    var tid = global_idx.x
    var idx = tid * UInt(simd_width)

    if idx >= UInt(len):
        return

    # Load vector
    var vec_in = input.load[width=simd_width](idx)

    # Process
    var vec_out = operation(vec_in)

    # Store result
    output.store[width=simd_width](idx, vec_out)
```

Launch configuration must account for vectorization:
```mojo
alias simd_width = 4
var block_dim = 256
var grid_dim = ceildiv(length, block_dim * simd_width)
```

## Fused Element-wise Operations

Combine multiple operations to reduce memory traffic:

```mojo
fn fused_operation_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    alpha: Float32,
    beta: Float32,
    len: Int,
):
    """Fused operation: output[i] = relu(alpha * input[i] + beta)"""
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    var x = input[tid]
    var result = alpha * x + beta
    output[tid] = max(result, 0.0)  # ReLU
```

Benefits:
- Single kernel launch overhead
- Reduced global memory accesses (one load, one store vs. multiple)
- Better cache utilization

## Broadcasting Element-wise Operations

Handle broadcasting for operations between tensors of different shapes:

```mojo
fn broadcast_add_kernel(
    output: UnsafePointer[Float32],
    input1: UnsafePointer[Float32],  # Shape: (M, N)
    input2: UnsafePointer[Float32],  # Shape: (N,)
    M: Int,
    N: Int,
):
    """Add with broadcasting: output[i,j] = input1[i,j] + input2[j]"""
    var tid = global_idx.x
    var total_size = M * N

    if tid >= UInt(total_size):
        return

    var i = tid / UInt(N)  # Row index
    var j = tid % UInt(N)  # Column index

    output[tid] = input1[tid] + input2[j]
```

## Strided Element-wise Operations

Handle non-contiguous tensors with stride information:

```mojo
fn strided_elementwise_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    output_stride: Int,
    input_stride: Int,
    len: Int,
):
    """Process with explicit strides"""
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    var out_idx = tid * UInt(output_stride)
    var in_idx = tid * UInt(input_stride)

    output[out_idx] = operation(input[in_idx])
```

Note: Strided access may reduce memory coalescing efficiency.

## Complex Element-wise Example: GELU Activation

```mojo
fn gelu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    alias inv_SQRT_2 = 0.70710678118654752440

    var x = input[tid]
    var val_half = 0.5 * x
    var erf_res = math.erf(x * inv_SQRT_2)

    # Fused multiply-add for efficiency
    output[tid] = val_half.fma(erf_res, val_half)
```

## Performance Considerations

### Memory Bandwidth

Element-wise operations are typically memory-bound:
- **Arithmetic intensity**: Operations / bytes transferred
- **Strategy**: Fuse operations to increase arithmetic intensity

### Coalescing

Ensure consecutive threads access consecutive memory:
```mojo
# Good: Coalesced access
output[tid] = input[tid]

# Bad: Strided access (if stride != 1)
output[tid * stride] = input[tid * stride]
```

### Launch Configuration

```mojo
# Typical configuration for element-wise
alias BLOCK_SIZE = 256  # Multiple of warp size (32)
var num_blocks = ceildiv(length, BLOCK_SIZE)

ctx.enqueue_function_checked[kernel, kernel](
    output, input, length,
    grid_dim=num_blocks,
    block_dim=BLOCK_SIZE,
)
```

### Vectorization Trade-offs

Pros:
- Higher arithmetic intensity
- Better instruction-level parallelism

Cons:
- Edge case handling complexity
- Requires length divisible by vector width (or tail loop)

## Common Patterns

### Unary Operations
```mojo
output[tid] = func(input[tid])
```
Examples: sqrt, exp, log, sin, cos, tanh, abs, neg

### Binary Operations
```mojo
output[tid] = func(input1[tid], input2[tid])
```
Examples: add, sub, mul, div, pow, max, min

### Ternary Operations
```mojo
output[tid] = cond[tid] ? true_val[tid] : false_val[tid]
```
Example: where, select

### Scalar-Tensor Operations
```mojo
output[tid] = func(input[tid], scalar)
```
Examples: add constant, multiply by scalar

## Testing Template

```python
import torch
import pytest

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("size", [1, 100, 1000, 10000])
def test_elementwise_operation(dtype, size):
    input_tensor = torch.randn(size, dtype=dtype, device='cuda')

    # Reference implementation
    expected = torch.operation(input_tensor)

    # Custom kernel
    output = custom_kernel(input_tensor)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)
```

## MAX Framework Integration

The MAX framework provides high-level `elementwise` pattern:

```mojo
@always_inline
@parameter
fn operation[simd_width: Int, rank: Int](idx: IndexList[rank]):
    output_tensor.store(
        idx,
        my_func(input_tensor.load[width=simd_width](idx))
    )

elementwise[operation, pack_size, target="gpu"](dims, ctx)
```

This handles:
- Optimal launch configuration
- Vectorization
- Edge case handling
- Multi-dimensional indexing

Use high-level patterns when available; write custom kernels for operations not covered by MAX primitives.
