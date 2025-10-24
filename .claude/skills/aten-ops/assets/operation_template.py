"""
ATen Operation Implementation Template

This template provides a starting point for implementing PyTorch ATen operations
in the MAX backend.

Usage:
1. Copy this template
2. Replace placeholders with actual operation details
3. Follow the 8-step workflow from the aten-ops skill

NOTE: This is a template file showing multiple patterns. The duplicate function
names are intentional for demonstration purposes. Copy only the relevant pattern
when implementing actual operations.
"""
# ruff: noqa: F811, F821

import max.graph.ops as max_ops
from max.graph import TensorValue
from torch.ops import aten

from torch_max_backend.aten_functions import map_to

# =============================================================================
# Element-wise Operation Template
# =============================================================================


# OPERATION_NAME(Tensor self) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue) -> TensorValue:
    """
    Brief description of what this operation does.

    Maps to max_ops.EQUIVALENT_OPERATION
    """
    return max_ops.EQUIVALENT_OPERATION(self)


# =============================================================================
# Element-wise with Parameter Template
# =============================================================================


# OPERATION_NAME(Tensor self, Scalar alpha=DEFAULT_VALUE) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue, alpha: float = DEFAULT_VALUE) -> TensorValue:
    """
    Brief description of operation with parameter.

    Args:
        self: Input tensor
        alpha: Parameter description (default: DEFAULT_VALUE)

    Returns:
        Result tensor
    """
    return max_ops.EQUIVALENT_OPERATION(self, alpha=alpha)


# =============================================================================
# Binary Operation Template
# =============================================================================


# OPERATION_NAME(Tensor self, Tensor other) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue, other: TensorValue) -> TensorValue:
    """
    Binary operation between two tensors.

    Supports broadcasting.

    Args:
        self: First tensor
        other: Second tensor

    Returns:
        Result tensor
    """
    return max_ops.EQUIVALENT_OPERATION(self, other)


# =============================================================================
# Binary with Scalar Multiplier Template
# =============================================================================


# OPERATION_NAME(Tensor self, Tensor other, Scalar alpha=1.0) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(
    self: TensorValue, other: TensorValue, alpha: float = 1.0
) -> TensorValue:
    """
    Binary operation with scalar multiplier.

    Computes: self OP (other * alpha)

    Args:
        self: First tensor
        other: Second tensor
        alpha: Multiplier for second tensor (default: 1.0)

    Returns:
        Result tensor
    """
    if alpha != 1.0:
        other = max_ops.mul(other, alpha)
    return max_ops.EQUIVALENT_OPERATION(self, other)


# =============================================================================
# Reduction Operation Template
# =============================================================================


# OPERATION_NAME(Tensor self, int dim, bool keepdim=False) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(
    self: TensorValue, dim: int, keepdim: bool = False
) -> TensorValue:
    """
    Reduction operation along dimension.

    Args:
        self: Input tensor
        dim: Dimension to reduce (supports negative indexing)
        keepdim: Keep reduced dimension as size 1 (default: False)

    Returns:
        Reduced tensor
    """
    # Handle negative dimension
    if dim < 0:
        dim = len(self.shape) + dim

    return max_ops.EQUIVALENT_OPERATION(self, axis=dim, keepdim=keepdim)


# =============================================================================
# Optional Dimension Reduction Template
# =============================================================================


# OPERATION_NAME(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(
    self: TensorValue, dim: int | None = None, keepdim: bool = False
) -> TensorValue:
    """
    Reduction operation with optional dimension.

    If dim is None, reduces all dimensions to scalar.
    Otherwise, reduces along specified dimension.

    Args:
        self: Input tensor
        dim: Dimension to reduce, or None for all (default: None)
        keepdim: Keep reduced dimension as size 1 (default: False)

    Returns:
        Reduced tensor or scalar
    """
    if dim is None:
        # Reduce all dimensions
        return max_ops.EQUIVALENT_OPERATION(self)

    # Handle negative dimension
    if dim < 0:
        dim = len(self.shape) + dim

    return max_ops.EQUIVALENT_OPERATION(self, axis=dim, keepdim=keepdim)


# =============================================================================
# Shape Manipulation Template
# =============================================================================


# OPERATION_NAME(Tensor self, int[] size) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue, size: list[int]) -> TensorValue:
    """
    Shape manipulation operation.

    Args:
        self: Input tensor
        size: New shape

    Returns:
        Reshaped tensor
    """
    return max_ops.EQUIVALENT_OPERATION(self, size)


# =============================================================================
# Transpose Template
# =============================================================================


# OPERATION_NAME(Tensor self, int dim0, int dim1) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue, dim0: int, dim1: int) -> TensorValue:
    """
    Transpose two dimensions.

    Args:
        self: Input tensor
        dim0: First dimension (supports negative indexing)
        dim1: Second dimension (supports negative indexing)

    Returns:
        Transposed tensor
    """
    # Handle negative dimensions
    ndim = len(self.shape)
    if dim0 < 0:
        dim0 = ndim + dim0
    if dim1 < 0:
        dim1 = ndim + dim1

    return max_ops.EQUIVALENT_OPERATION(self, dim0, dim1)


# =============================================================================
# Concatenation Template
# =============================================================================


# OPERATION_NAME(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    """
    Concatenate list of tensors along dimension.

    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate (default: 0)

    Returns:
        Concatenated tensor
    """
    # Handle empty list
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    # Handle single tensor
    if len(tensors) == 1:
        return tensors[0]

    # Handle negative dimension
    if dim < 0:
        dim = len(tensors[0].shape) + dim

    return max_ops.EQUIVALENT_OPERATION(tensors, axis=dim)


# =============================================================================
# Composed Operation Template
# =============================================================================


# OPERATION_NAME(Tensor self, int dim) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(self: TensorValue, dim: int) -> TensorValue:
    """
    Operation composed of multiple MAX ops.

    Example: log_softmax = log(softmax(x))

    Args:
        self: Input tensor
        dim: Dimension parameter

    Returns:
        Result tensor
    """
    # Handle negative dimension
    if dim < 0:
        dim = len(self.shape) + dim

    # Compose operations
    intermediate = max_ops.FIRST_OPERATION(self, axis=dim)
    result = max_ops.SECOND_OPERATION(intermediate)

    return result


# =============================================================================
# Operation with Optional Tensor Template
# =============================================================================


# OPERATION_NAME(Tensor self, Tensor? other=None) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(
    self: TensorValue, other: TensorValue | None = None
) -> TensorValue:
    """
    Operation with optional tensor parameter.

    Args:
        self: Input tensor
        other: Optional second tensor (default: None)

    Returns:
        Result tensor
    """
    if other is None:
        # Behavior when other is not provided
        return max_ops.EQUIVALENT_OPERATION(self)

    # Behavior when other is provided
    return max_ops.EQUIVALENT_OPERATION(self, other)


# =============================================================================
# Complex Operation Template
# =============================================================================


# OPERATION_NAME(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding) -> Tensor
@map_to(aten.OPERATION_NAME)
def aten_OPERATION_NAME(
    input: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None,
    stride: list[int],
    padding: list[int],
) -> TensorValue:
    """
    Complex operation with multiple parameters.

    Example: convolution, linear layer

    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor (default: None)
        stride: Stride values
        padding: Padding values

    Returns:
        Result tensor
    """
    result = max_ops.EQUIVALENT_OPERATION(
        input, weight, bias=bias, stride=tuple(stride), padding=tuple(padding)
    )
    return result


# =============================================================================
# In-place Operation Template
# =============================================================================


# OPERATION_NAME_(Tensor self) -> Tensor
@map_to(aten.OPERATION_NAME_)
def aten_OPERATION_NAME_(self: TensorValue) -> TensorValue:
    """
    In-place variant of OPERATION_NAME.

    Note: In graph compilation mode, may be equivalent to out-of-place
    version as the graph compiler optimizes as a whole.

    Args:
        self: Input tensor (modified in-place)

    Returns:
        Modified tensor (same as self)
    """
    # For graph compilation, often same as out-of-place
    return max_ops.EQUIVALENT_OPERATION(self)


# =============================================================================
# Helper Functions (if needed)
# =============================================================================


def _normalize_dimension(dim: int, ndim: int) -> int:
    """
    Normalize negative dimension to positive.

    Args:
        dim: Dimension index (can be negative)
        ndim: Number of dimensions in tensor

    Returns:
        Positive dimension index
    """
    if dim < 0:
        dim = ndim + dim

    if dim < 0 or dim >= ndim:
        raise ValueError(f"Dimension {dim} out of range for {ndim}D tensor")

    return dim


def _validate_shape_compatibility(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> bool:
    """
    Check if two shapes are compatible for broadcasting.

    Args:
        shape1: First shape
        shape2: Second shape

    Returns:
        True if shapes are broadcastable
    """
    # Implementation of broadcasting rules
    # ...
    return True


# =============================================================================
# Usage Notes
# =============================================================================

"""
When implementing a new ATen operation:

1. Choose the appropriate template above
2. Replace OPERATION_NAME with actual operation name
3. Replace EQUIVALENT_OPERATION with MAX operation
4. Update function signature with correct types
5. Add comprehensive docstring
6. Handle edge cases (negative dims, None values, etc.)
7. Add validation if needed
8. Write unit tests before implementing
9. Run tests to verify
10. Run linter: uvx pre-commit run --all-files

Common type hints:
- TensorValue: Single tensor
- list[TensorValue]: List of tensors
- int, float, bool: Scalar values
- int | None, float | None: Optional scalars
- list[int]: Integer sequences (shapes, strides)
- TensorValue | None: Optional tensor

Common patterns:
- Negative dimension handling: if dim < 0: dim = len(self.shape) + dim
- PyTorch dim → MAX axis: Use axis=dim in MAX calls
- Lists to tuples: Some MAX ops need tuple(list_param)
- Optional parameters: Provide sensible defaults, handle None
"""
