# Common ATen Operation Implementation Patterns

This guide provides reusable patterns for implementing common types of ATen operations in the MAX backend.

## Pattern Categories

- [Element-wise Operations](#element-wise-operations)
- [Reduction Operations](#reduction-operations)
- [Shape Manipulation](#shape-manipulation)
- [Linear Algebra](#linear-algebra)
- [Tensor Creation and Combination](#tensor-creation-and-combination)
- [Special Patterns](#special-patterns)

## Element-wise Operations

### Simple Unary Operation

Operations that apply a function to each element independently.

**ATen signature**: `operation(Tensor self) -> Tensor`

**Pattern**:
```python
# relu(Tensor self) -> Tensor
@map_to(aten.relu)
def aten_relu(self: TensorValue) -> TensorValue:
    return max_ops.relu(self)
```

**Examples**: relu, sigmoid, tanh, abs, neg, exp, log, sqrt, sin, cos

### Unary with Parameter

Element-wise operation with a scalar parameter.

**ATen signature**: `operation(Tensor self, Scalar alpha) -> Tensor`

**Pattern**:
```python
# leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
@map_to(aten.leaky_relu)
def aten_leaky_relu(
    self: TensorValue,
    negative_slope: float = 0.01,
) -> TensorValue:
    return max_ops.leaky_relu(self, negative_slope=negative_slope)
```

**Examples**: leaky_relu, elu, threshold, clamp

### Binary Element-wise

Operations between two tensors (supports broadcasting).

**ATen signature**: `operation(Tensor self, Tensor other) -> Tensor`

**Pattern**:
```python
# add(Tensor self, Tensor other) -> Tensor
@map_to(aten.add)
def aten_add(self: TensorValue, other: TensorValue) -> TensorValue:
    return max_ops.add(self, other)
```

**Examples**: add, sub, mul, div, pow, maximum, minimum

### Binary with Scalar Parameter

Binary operation with an additional scalar multiplier.

**ATen signature**: `operation(Tensor self, Tensor other, Scalar alpha=1) -> Tensor`

**Pattern**:
```python
# add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
@map_to(aten.add)
def aten_add(
    self: TensorValue,
    other: TensorValue,
    alpha: float = 1.0,
) -> TensorValue:
    if alpha != 1.0:
        other = max_ops.mul(other, alpha)
    return max_ops.add(self, other)
```

**Examples**: add (with alpha), sub (with alpha)

### In-place Variants

In-place operations (name ends with underscore).

**ATen signature**: `operation_(Tensor self) -> Tensor`

**Pattern**:
```python
# relu_(Tensor self) -> Tensor
@map_to(aten.relu_)
def aten_relu_(self: TensorValue) -> TensorValue:
    # For graph compilation, can often use same implementation as out-of-place
    # Graph compiler may optimize
    return max_ops.relu(self)
```

**Note**: In graph compilation mode, in-place operations may be equivalent to out-of-place since the graph is optimized as a whole.

## Reduction Operations

### Full Tensor Reduction

Reduce entire tensor to a single scalar.

**ATen signature**: `operation(Tensor self) -> Tensor`

**Pattern**:
```python
# sum(Tensor self) -> Tensor
@map_to(aten.sum)
def aten_sum(self: TensorValue) -> TensorValue:
    return max_ops.sum(self)
```

**Examples**: sum, mean, max, min, prod

### Reduction Along Dimension

Reduce along specific dimension(s).

**ATen signature**: `operation(Tensor self, int dim, bool keepdim=False) -> Tensor`

**Pattern**:
```python
# sum(Tensor self, int dim, bool keepdim=False) -> Tensor
@map_to(aten.sum)
def aten_sum(
    self: TensorValue,
    dim: int,
    keepdim: bool = False,
) -> TensorValue:
    # Handle negative dimension
    if dim < 0:
        dim = len(self.shape) + dim
    return max_ops.sum(self, axis=dim, keepdim=keepdim)
```

**Examples**: sum, mean, max, min, any, all

### Optional Dimension Reduction

Reduce all dimensions or specific dimension.

**ATen signature**: `operation(Tensor self, int? dim=None, bool keepdim=False) -> Tensor`

**Pattern**:
```python
# sum(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
@map_to(aten.sum)
def aten_sum(
    self: TensorValue,
    dim: int | None = None,
    keepdim: bool = False,
) -> TensorValue:
    if dim is None:
        # Reduce all dimensions
        return max_ops.sum(self)

    # Handle negative dimension
    if dim < 0:
        dim = len(self.shape) + dim

    return max_ops.sum(self, axis=dim, keepdim=keepdim)
```

### Reduction Returning Multiple Values

Operations that return both value and indices.

**ATen signature**: `operation(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)`

**Pattern**:
```python
# max(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
@map_to(aten.max)
def aten_max(
    self: TensorValue,
    dim: int,
    keepdim: bool = False,
) -> tuple[TensorValue, TensorValue]:
    if dim < 0:
        dim = len(self.shape) + dim

    values, indices = max_ops.max_with_indices(self, axis=dim, keepdim=keepdim)
    return values, indices
```

**Examples**: max, min, median, mode

## Shape Manipulation

### Reshape/View

Change tensor shape without copying data.

**ATen signature**: `view(Tensor self, int[] size) -> Tensor`

**Pattern**:
```python
# view(Tensor self, int[] size) -> Tensor
@map_to(aten.view)
def aten_view(self: TensorValue, size: list[int]) -> TensorValue:
    return max_ops.reshape(self, size)
```

**Examples**: view, reshape, flatten

### Transpose

Swap two dimensions.

**ATen signature**: `transpose(Tensor self, int dim0, int dim1) -> Tensor`

**Pattern**:
```python
# transpose(Tensor self, int dim0, int dim1) -> Tensor
@map_to(aten.transpose)
def aten_transpose(self: TensorValue, dim0: int, dim1: int) -> TensorValue:
    # Handle negative dimensions
    ndim = len(self.shape)
    if dim0 < 0:
        dim0 = ndim + dim0
    if dim1 < 0:
        dim1 = ndim + dim1

    return max_ops.transpose(self, dim0, dim1)
```

### Permute

Reorder all dimensions.

**ATen signature**: `permute(Tensor self, int[] dims) -> Tensor`

**Pattern**:
```python
# permute(Tensor self, int[] dims) -> Tensor
@map_to(aten.permute)
def aten_permute(self: TensorValue, dims: list[int]) -> TensorValue:
    return max_ops.permute(self, dims)
```

### Squeeze/Unsqueeze

Remove or add size-1 dimensions.

**ATen signature**: `squeeze(Tensor self, int? dim=None) -> Tensor`

**Pattern**:
```python
# squeeze(Tensor self, int? dim=None) -> Tensor
@map_to(aten.squeeze)
def aten_squeeze(self: TensorValue, dim: int | None = None) -> TensorValue:
    if dim is None:
        return max_ops.squeeze(self)

    if dim < 0:
        dim = len(self.shape) + dim

    return max_ops.squeeze(self, axis=dim)
```

```python
# unsqueeze(Tensor self, int dim) -> Tensor
@map_to(aten.unsqueeze)
def aten_unsqueeze(self: TensorValue, dim: int) -> TensorValue:
    # Note: unsqueeze allows dim = len(shape) to append
    ndim = len(self.shape)
    if dim < 0:
        dim = ndim + dim + 1

    return max_ops.expand_dims(self, axis=dim)
```

**Examples**: squeeze, unsqueeze, flatten

## Linear Algebra

### Matrix Multiplication

**ATen signature**: `matmul(Tensor self, Tensor other) -> Tensor`

**Pattern**:
```python
# matmul(Tensor self, Tensor other) -> Tensor
@map_to(aten.matmul)
def aten_matmul(self: TensorValue, other: TensorValue) -> TensorValue:
    return max_ops.matmul(self, other)
```

**Examples**: matmul, mm, bmm, addmm

### Batched Matrix Multiplication

**ATen signature**: `bmm(Tensor self, Tensor mat2) -> Tensor`

**Pattern**:
```python
# bmm(Tensor self, Tensor mat2) -> Tensor
@map_to(aten.bmm)
def aten_bmm(self: TensorValue, mat2: TensorValue) -> TensorValue:
    # bmm is batched matrix multiplication
    # Input: (B, N, M) and (B, M, P)
    # Output: (B, N, P)
    return max_ops.bmm(self, mat2)
```

### Matrix-Vector Product

**ATen signature**: `mv(Tensor self, Tensor vec) -> Tensor`

**Pattern**:
```python
# mv(Tensor self, Tensor vec) -> Tensor
@map_to(aten.mv)
def aten_mv(self: TensorValue, vec: TensorValue) -> TensorValue:
    # Matrix-vector product
    # Input: (N, M) and (M,)
    # Output: (N,)
    return max_ops.mv(self, vec)
```

## Tensor Creation and Combination

### Concatenation

Concatenate tensors along a dimension.

**ATen signature**: `cat(Tensor[] tensors, int dim=0) -> Tensor`

**Pattern**:
```python
# cat(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.cat)
def aten_cat(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    if dim < 0 and len(tensors) > 0:
        dim = len(tensors[0].shape) + dim

    return max_ops.concat(tensors, axis=dim)
```

**Examples**: cat, concat

### Stack

Stack tensors along new dimension.

**ATen signature**: `stack(Tensor[] tensors, int dim=0) -> Tensor`

**Pattern**:
```python
# stack(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.stack)
def aten_stack(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    if dim < 0 and len(tensors) > 0:
        # For stack, can insert at -1 (before last dim)
        dim = len(tensors[0].shape) + dim + 1

    return max_ops.stack(tensors, axis=dim)
```

### Split/Chunk

Split tensor into multiple tensors.

**ATen signature**: `split(Tensor self, int split_size, int dim=0) -> Tensor[]`

**Pattern**:
```python
# split(Tensor self, int split_size, int dim=0) -> Tensor[]
@map_to(aten.split)
def aten_split(
    self: TensorValue,
    split_size: int,
    dim: int = 0,
) -> list[TensorValue]:
    if dim < 0:
        dim = len(self.shape) + dim

    return max_ops.split(self, split_size=split_size, axis=dim)
```

**Examples**: split, chunk

## Special Patterns

### Softmax

Softmax activation along dimension.

**ATen signature**: `softmax(Tensor self, int dim, int? dtype=None) -> Tensor`

**Pattern**:
```python
# softmax(Tensor self, int dim, int? dtype=None) -> Tensor
@map_to(aten.softmax)
def aten_softmax(
    self: TensorValue,
    dim: int,
    dtype: int | None = None,
) -> TensorValue:
    if dim < 0:
        dim = len(self.shape) + dim

    result = max_ops.softmax(self, axis=dim)

    # Handle dtype if specified (usually None)
    # Note: dtype is PyTorch dtype enum, may need conversion
    if dtype is not None:
        # Convert if needed
        pass

    return result
```

### Log Softmax

Numerically stable log(softmax(x)).

**ATen signature**: `log_softmax(Tensor self, int dim, int? dtype=None) -> Tensor`

**Pattern**:
```python
# log_softmax(Tensor self, int dim, int? dtype=None) -> Tensor
@map_to(aten.log_softmax)
def aten_log_softmax(
    self: TensorValue,
    dim: int,
    dtype: int | None = None,
) -> TensorValue:
    if dim < 0:
        dim = len(self.shape) + dim

    # Use numerically stable implementation
    result = max_ops.log_softmax(self, axis=dim)

    return result
```

### Convolution

2D convolution operation.

**ATen signature**: `conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor`

**Pattern**:
```python
# conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
@map_to(aten.conv2d)
def aten_conv2d(
    input: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> TensorValue:
    result = max_ops.conv2d(
        input,
        weight,
        bias=bias,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
    )
    return result
```

### Embedding Lookup

**ATen signature**: `embedding(Tensor weight, Tensor indices) -> Tensor`

**Pattern**:
```python
# embedding(Tensor weight, Tensor indices) -> Tensor
@map_to(aten.embedding)
def aten_embedding(
    weight: TensorValue,
    indices: TensorValue,
) -> TensorValue:
    # Embedding is essentially a lookup operation
    return max_ops.embedding(weight, indices)
```

## Handling Parameter Conversions

### PyTorch dim ↔ MAX axis

PyTorch uses `dim`, MAX often uses `axis`:

```python
# Always convert in function
@map_to(aten.operation)
def aten_operation(self: TensorValue, dim: int) -> TensorValue:
    return max_ops.operation(self, axis=dim)  # Note: axis not dim
```

### Negative Dimension Handling

```python
def aten_operation(self: TensorValue, dim: int) -> TensorValue:
    # Normalize negative dimension
    if dim < 0:
        dim = len(self.shape) + dim
    return max_ops.operation(self, axis=dim)
```

### List to Tuple Conversion

Some MAX ops expect tuples:

```python
def aten_operation(self: TensorValue, sizes: list[int]) -> TensorValue:
    return max_ops.operation(self, shape=tuple(sizes))
```

### Optional Parameter Handling

```python
def aten_operation(
    self: TensorValue,
    param: float | None = None,
) -> TensorValue:
    if param is None:
        param = 1.0  # Default value
    return max_ops.operation(self, param=param)
```

## Composing Operations

When no direct MAX equivalent exists, compose multiple operations:

```python
# Example: log_softmax = log(softmax(x))
@map_to(aten.log_softmax)
def aten_log_softmax(self: TensorValue, dim: int) -> TensorValue:
    if dim < 0:
        dim = len(self.shape) + dim

    # Compose: log(softmax(x))
    softmax_result = max_ops.softmax(self, axis=dim)
    return max_ops.log(softmax_result)
```

```python
# Example: silu = x * sigmoid(x)
@map_to(aten.silu)
def aten_silu(self: TensorValue) -> TensorValue:
    sigmoid_result = max_ops.sigmoid(self)
    return max_ops.mul(self, sigmoid_result)
```

## Error Handling

### Validate Inputs

```python
def aten_operation(self: TensorValue, dim: int) -> TensorValue:
    ndim = len(self.shape)

    # Validate dimension
    if dim < -ndim or dim >= ndim:
        raise ValueError(f"Dimension {dim} out of range for {ndim}D tensor")

    if dim < 0:
        dim = ndim + dim

    return max_ops.operation(self, axis=dim)
```

### Handle Edge Cases

```python
def aten_cat(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    # Handle empty list
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    # Handle single tensor
    if len(tensors) == 1:
        return tensors[0]

    if dim < 0:
        dim = len(tensors[0].shape) + dim

    return max_ops.concat(tensors, axis=dim)
```

## Best Practices Summary

1. **Always handle negative dimensions**: Normalize to positive
2. **Check for direct MAX equivalents first**: Don't reimplement
3. **Use composition when needed**: Combine multiple MAX ops
4. **Map parameter names correctly**: dim → axis, etc.
5. **Handle optional parameters**: Provide appropriate defaults
6. **Validate inputs for complex operations**: Check dimensions, empty tensors
7. **Follow alphabetical order**: Keep `aten_functions.py` sorted
8. **Add signature comments**: Include ATen signature
9. **Type hints accurately**: Use beartype method if unsure
10. **Test thoroughly**: Multiple dtypes, shapes, edge cases

For specific MAX operations available, see the `mojo-gpu-kernels` skill reference on MAX operations.
