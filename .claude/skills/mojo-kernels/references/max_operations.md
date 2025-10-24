# MAX Framework Operations Reference

This document catalogs available operations in the MAX framework that can be used when implementing ATen operations for the PyTorch backend.

## Discovery Process

When implementing a new ATen operation:

1. **Search kernel sources**: Look in `../modular/max/kernels/src/`
2. **Check examples**: Review `../modular/max/examples/` and `../modular/max/kernels/benchmarks/`
3. **Examine tests**: Look at `../modular/max/kernels/test/` for usage patterns
4. **Read documentation**: Check function signatures and docstrings

## Activation Functions

Location: `../modular/max/kernels/src/nn/activations.mojo`

### ReLU
```mojo
@always_inline
fn relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """Rectified Linear Unit: max(0, x)"""
    return max(x, 0)
```

### GELU
```mojo
@always_inline
fn gelu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """Gaussian Error Linear Unit: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    alias inv_SQRT_2 = 0.70710678118654752440
    var val_half = 0.5 * x
    var erf_res = math.erf(x * inv_SQRT_2)
    return val_half.fma(erf_res, val_half)
```

### Leaky ReLU
```mojo
@always_inline
fn leaky_relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    negative_slope: Scalar[dtype]
) -> SIMD[dtype, simd_width]:
    """Leaky ReLU: x if x > 0 else negative_slope * x"""
    return x.ge(0).select(x, negative_slope * x)
```

### ELU
```mojo
@always_inline
fn elu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    alpha: Scalar[dtype] = 1.0
) -> SIMD[dtype, simd_width]:
    """Exponential Linear Unit: x if x > 0 else alpha * (exp(x) - 1)"""
    return x.ge(0).select(x, alpha * (exp(x) - 1))
```

### SiLU / Swish
```mojo
@always_inline
fn silu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """SiLU/Swish: x * sigmoid(x)"""
    return x / (1.0 + exp(-x))
```

### Sigmoid
```mojo
@always_inline
fn sigmoid[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + exp(-x))
```

### Tanh
```mojo
@always_inline
fn tanh[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """Hyperbolic tangent (from math library)"""
    return math.tanh(x)
```

## Normalization Operations

Location: `../modular/max/kernels/src/nn/normalization.mojo`, `softmax.mojo`

### Softmax
```mojo
fn softmax[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Softmax along specified dimension.

    Computes: exp(x[i]) / sum(exp(x[j])) for j in dimension
    Uses numerically stable variant: exp(x[i] - max) / sum(exp(x[j] - max))
    """
```

### Log Softmax
```mojo
fn log_softmax[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Log of softmax: log(exp(x[i]) / sum(exp(x[j])))
    Numerically stable: (x[i] - max) - log(sum(exp(x[j] - max)))
    """
```

### Layer Normalization
```mojo
fn layer_norm[dtype: DType](
    input: LayoutTensor[dtype],
    normalized_shape: List[Int],
    weight: Optional[LayoutTensor[dtype]],
    bias: Optional[LayoutTensor[dtype]],
    eps: Float64,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Layer normalization: (x - mean) / sqrt(variance + eps)
    Optionally scale by weight and shift by bias
    """
```

### Batch Normalization
```mojo
fn batch_norm[dtype: DType](
    input: LayoutTensor[dtype],
    running_mean: LayoutTensor[dtype],
    running_var: LayoutTensor[dtype],
    weight: Optional[LayoutTensor[dtype]],
    bias: Optional[LayoutTensor[dtype]],
    training: Bool,
    momentum: Float64,
    eps: Float64,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Batch normalization for training and inference"""
```

## Linear Algebra Operations

Location: `../modular/max/kernels/src/linalg/`

### Matrix Multiplication
```mojo
fn matmul[dtype: DType](
    lhs: LayoutTensor[dtype],
    rhs: LayoutTensor[dtype],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Matrix multiplication: C = A @ B

    Handles:
    - 2D x 2D: Standard matmul
    - 3D x 3D: Batched matmul
    - ND x ND: Batched matmul with broadcasting
    """
```

### Batched Matrix Multiplication
```mojo
fn bmm[dtype: DType](
    input: LayoutTensor[dtype],
    mat2: LayoutTensor[dtype],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Batched matrix multiplication
    Input: (B, N, M), (B, M, P)
    Output: (B, N, P)
    """
```

### Matrix-Vector Multiplication
```mojo
fn mv[dtype: DType](
    mat: LayoutTensor[dtype],
    vec: LayoutTensor[dtype],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """
    Matrix-vector product
    Input: (N, M) matrix, (M,) vector
    Output: (N,) vector
    """
```

### Grouped Matrix Multiplication
```mojo
fn grouped_matmul[dtype: DType](
    lhs: List[LayoutTensor[dtype]],
    rhs: List[LayoutTensor[dtype]],
    ctx: DeviceContext
) -> List[LayoutTensor[dtype]]:
    """Multiple independent matmuls executed efficiently"""
```

## Reduction Operations

Location: `../modular/max/kernels/src/algorithm/_gpu/reduction.mojo`

### Reduce Launch (Generic)
```mojo
fn reduce_launch[
    num_reductions: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int](IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]),
    reduce_fn: fn[dtype: DType, width: Int, reduction_idx: Int](SIMD[dtype, width], SIMD[dtype, width]) -> SIMD[dtype, width],
    rank: Int,
    dtype: DType,
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    ctx: DeviceContext
):
    """Generic reduction framework supporting custom operations"""
```

### Sum
```mojo
fn sum[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Optional[Int] = None,
    keepdim: Bool = False,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Sum along dimension(s)"""
```

### Mean
```mojo
fn mean[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Optional[Int] = None,
    keepdim: Bool = False,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Mean along dimension(s)"""
```

### Max/Min
```mojo
fn max[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    keepdim: Bool = False,
    ctx: DeviceContext
) -> (LayoutTensor[dtype], LayoutTensor[DType.int64]):
    """Maximum values and indices along dimension"""

fn min[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    keepdim: Bool = False,
    ctx: DeviceContext
) -> (LayoutTensor[dtype], LayoutTensor[DType.int64]):
    """Minimum values and indices along dimension"""
```

## Convolution Operations

Location: `../modular/max/kernels/src/nn/conv.mojo`

### Conv2D
```mojo
fn conv2d[dtype: DType](
    input: LayoutTensor[dtype],     # (N, C_in, H, W)
    weight: LayoutTensor[dtype],    # (C_out, C_in, kH, kW)
    bias: Optional[LayoutTensor[dtype]],
    stride: (Int, Int),
    padding: (Int, Int),
    dilation: (Int, Int),
    groups: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:           # (N, C_out, H_out, W_out)
    """2D convolution"""
```

### Conv Transpose
```mojo
fn conv_transpose2d[dtype: DType](
    input: LayoutTensor[dtype],
    weight: LayoutTensor[dtype],
    bias: Optional[LayoutTensor[dtype]],
    stride: (Int, Int),
    padding: (Int, Int),
    output_padding: (Int, Int),
    groups: Int,
    dilation: (Int, Int),
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """2D transposed convolution (deconvolution)"""
```

## Pooling Operations

Location: `../modular/max/kernels/src/nn/pooling.mojo`

### Max Pool 2D
```mojo
fn max_pool2d[dtype: DType](
    input: LayoutTensor[dtype],
    kernel_size: (Int, Int),
    stride: Optional[(Int, Int)] = None,
    padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1),
    ceil_mode: Bool = False,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """2D max pooling"""
```

### Average Pool 2D
```mojo
fn avg_pool2d[dtype: DType](
    input: LayoutTensor[dtype],
    kernel_size: (Int, Int),
    stride: Optional[(Int, Int)] = None,
    padding: (Int, Int) = (0, 0),
    ceil_mode: Bool = False,
    count_include_pad: Bool = True,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """2D average pooling"""
```

## Attention Operations

Location: `../modular/max/kernels/src/nn/attention.mojo`

### Flash Attention
```mojo
fn flash_attention[dtype: DType](
    query: LayoutTensor[dtype],     # (B, N_heads, N, D)
    key: LayoutTensor[dtype],       # (B, N_heads, N, D)
    value: LayoutTensor[dtype],     # (B, N_heads, N, D)
    dropout_p: Float64 = 0.0,
    is_causal: Bool = False,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:           # (B, N_heads, N, D)
    """
    Flash Attention: Memory-efficient attention

    Computes softmax(Q @ K^T / sqrt(D)) @ V
    without materializing full attention matrix
    """
```

## Tensor Manipulation

Location: Various files in `../modular/max/kernels/src/`

### Reshape / View
```mojo
fn reshape[dtype: DType](
    input: LayoutTensor[dtype],
    new_shape: List[Int]
) -> LayoutTensor[dtype]:
    """Reshape tensor to new shape"""
```

### Transpose / Permute
```mojo
fn transpose[dtype: DType](
    input: LayoutTensor[dtype],
    dim0: Int,
    dim1: Int
) -> LayoutTensor[dtype]:
    """Transpose two dimensions"""

fn permute[dtype: DType](
    input: LayoutTensor[dtype],
    dims: List[Int]
) -> LayoutTensor[dtype]:
    """Permute dimensions according to dims"""
```

### Concat / Stack
```mojo
fn concat[dtype: DType](
    tensors: List[LayoutTensor[dtype]],
    dim: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Concatenate tensors along dimension"""

fn stack[dtype: DType](
    tensors: List[LayoutTensor[dtype]],
    dim: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Stack tensors along new dimension"""
```

### Split / Chunk
```mojo
fn split[dtype: DType](
    input: LayoutTensor[dtype],
    split_size: Int,
    dim: Int,
    ctx: DeviceContext
) -> List[LayoutTensor[dtype]]:
    """Split tensor into chunks"""

fn chunk[dtype: DType](
    input: LayoutTensor[dtype],
    chunks: Int,
    dim: Int,
    ctx: DeviceContext
) -> List[LayoutTensor[dtype]]:
    """Split tensor into specific number of chunks"""
```

### Padding
```mojo
fn pad[dtype: DType](
    input: LayoutTensor[dtype],
    padding: List[Int],  # (left, right, top, bottom, ...)
    mode: String = "constant",
    value: Scalar[dtype] = 0,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Pad tensor with specified mode"""
```

## Element-wise Math Operations

Available via `math` module and SIMD operations:

### Trigonometric
```mojo
math.sin(x)
math.cos(x)
math.tan(x)
math.asin(x)
math.acos(x)
math.atan(x)
math.atan2(y, x)
```

### Exponential / Logarithmic
```mojo
math.exp(x)
math.log(x)      # Natural log
math.log2(x)
math.log10(x)
math.exp2(x)     # 2^x
math.expm1(x)    # exp(x) - 1
math.log1p(x)    # log(1 + x)
```

### Power / Root
```mojo
math.pow(x, y)
math.sqrt(x)
math.rsqrt(x)    # 1 / sqrt(x)
math.cbrt(x)     # Cube root
```

### Rounding
```mojo
math.floor(x)
math.ceil(x)
math.round(x)
math.trunc(x)
```

### Other
```mojo
math.abs(x)
math.erf(x)      # Error function
math.erfc(x)     # Complementary error function
math.copysign(x, y)
math.fma(x, y, z)  # Fused multiply-add: x * y + z
```

## SIMD Operations

For vectorized operations:

```mojo
# Reductions
vec.reduce_add()
vec.reduce_mul()
vec.reduce_max()
vec.reduce_min()
vec.reduce_and()
vec.reduce_or()

# Comparisons (return SIMD[DType.bool, width])
vec.eq(other)    # ==
vec.ne(other)    # !=
vec.lt(other)    # <
vec.le(other)    # <=
vec.gt(other)    # >
vec.ge(other)    # >=

# Select based on condition
condition.select(true_val, false_val)

# Type conversions
vec.cast[new_dtype]()
```

## Broadcasting Operations

Location: `../modular/max/kernels/src/nn/broadcast.mojo`

```mojo
fn broadcast_to[dtype: DType](
    input: LayoutTensor[dtype],
    shape: List[Int],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Broadcast tensor to new shape following NumPy rules"""
```

## Indexing and Slicing

```mojo
fn gather[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    index: LayoutTensor[DType.int64],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Gather values along dimension using indices"""

fn scatter[dtype: DType](
    input: LayoutTensor[dtype],
    dim: Int,
    index: LayoutTensor[DType.int64],
    src: LayoutTensor[dtype],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    """Scatter values along dimension using indices"""
```

## Usage Patterns

### Composing Operations

Many ATen operations can be implemented by composing MAX primitives:

```mojo
# Example: Implementing softmax backward
fn softmax_backward[dtype: DType](
    grad_output: LayoutTensor[dtype],
    output: LayoutTensor[dtype],  # Softmax forward result
    dim: Int,
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    # grad_input = output * (grad_output - (output * grad_output).sum(dim))

    var prod = output * grad_output
    var sum_grad = sum[dtype](prod, dim=dim, keepdim=True, ctx)
    var grad_input = output * (grad_output - sum_grad)

    return grad_input
```

### Finding Equivalent Operations

1. **Check activation functions** for element-wise operations
2. **Look in linalg** for matrix operations
3. **Examine reduction.mojo** for aggregation operations
4. **Review nn/** directory for neural network primitives
5. **Search benchmarks/** for usage examples

### When No Direct Equivalent Exists

You may need to:
1. **Write custom kernel**: Using patterns from this skill
2. **Compose primitives**: Combine multiple MAX operations
3. **Use algorithm patterns**: `elementwise`, `reduce_launch`, etc.
4. **Adapt existing kernels**: Modify similar operations

## Performance Considerations

### Fused Operations

MAX often provides fused variants that are more efficient:
- Use `gelu` instead of composing `erf`, `mul`, `add`
- Use `silu` instead of `x * sigmoid(x)`
- Use `flash_attention` instead of manual attention computation

### Memory Layout

Operations may have layout preferences:
- Matmul expects contiguous or transposed layouts
- Convolutions may prefer NCHW or NHWC
- Use `to_contiguous()` if needed, but prefer avoiding copies

### Device Context

Always pass `DeviceContext` for GPU operations:
```mojo
with DeviceContext() as ctx:
    result = operation[dtype](input, ctx)
```

## Documentation Locations

- **Source code**: `../modular/max/kernels/src/`
- **Tests**: `../modular/max/kernels/test/`
- **Benchmarks**: `../modular/max/kernels/benchmarks/`
- **Examples**: `../modular/max/examples/`

When in doubt, search these directories for similar operations and usage patterns.
