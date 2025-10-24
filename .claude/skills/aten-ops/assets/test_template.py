"""
ATen Operation Test Template

This template provides patterns for testing PyTorch ATen operations
in the torch-max-backend project.

Usage:
1. Copy relevant test pattern
2. Replace placeholders with actual operation details
3. Add to tests/test_aten_functions.py (middle of file)
4. Run: uv run pytest tests/test_aten_functions.py::test_aten_OPERATION -v
"""

import pytest
import torch

from torch_max_backend import max_backend

# =============================================================================
# Basic Element-wise Operation Test
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (10,),  # 1D small
        (1000,),  # 1D large
        (10, 20),  # 2D
        (5, 10, 15),  # 3D
    ],
)
def test_aten_OPERATION_NAME(dtype, shape):
    """Test aten::OPERATION_NAME implementation."""
    # Create input tensor
    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference implementation (PyTorch)
    expected = torch.OPERATION_NAME(input_tensor)

    # Compile with MAX backend
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x)

    # Execute
    result = compiled_fn(input_tensor)

    # Verify
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Binary Operation Test with Broadcasting
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 20), (10, 20)),  # Same shape
        ((10, 20), (20,)),  # Matrix + vector
        ((5, 1, 10), (10,)),  # 3D + 1D broadcast
        ((5, 1), (1, 10)),  # Row + column broadcast
        ((1, 10, 20), (5, 10, 20)),  # Batch broadcast
    ],
)
def test_aten_OPERATION_NAME_broadcasting(dtype, shapes):
    """Test aten::OPERATION_NAME with broadcasting."""
    shape1, shape2 = shapes
    x = torch.randn(shape1, dtype=dtype)
    y = torch.randn(shape2, dtype=dtype)

    # Reference
    expected = torch.OPERATION_NAME(x, y)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        return torch.OPERATION_NAME(a, b)

    result = compiled_fn(x, y)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Reduction Operation Test
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 10, 15)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_aten_OPERATION_NAME_reduction(dtype, shape, dim, keepdim):
    """Test aten::OPERATION_NAME reduction along dimension."""
    # Skip invalid dimensions for shape
    if abs(dim) >= len(shape):
        pytest.skip(f"Dimension {dim} invalid for shape {shape}")

    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference
    expected = torch.OPERATION_NAME(input_tensor, dim=dim, keepdim=keepdim)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x, dim=dim, keepdim=keepdim)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Optional Dimension Reduction Test
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 10, 15)])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
def test_aten_OPERATION_NAME_optional_dim(dtype, shape, dim):
    """Test aten::OPERATION_NAME with optional dimension."""
    # Skip invalid dimensions
    if dim is not None and abs(dim) >= len(shape):
        pytest.skip(f"Dimension {dim} invalid for shape {shape}")

    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference
    if dim is None:
        expected = torch.OPERATION_NAME(input_tensor)
    else:
        expected = torch.OPERATION_NAME(input_tensor, dim=dim)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        if dim is None:
            return torch.OPERATION_NAME(x)
        return torch.OPERATION_NAME(x, dim=dim)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Operation with Parameter Test
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(10,), (10, 20)])
@pytest.mark.parametrize("param_value", [0.01, 0.1, 0.5, 1.0])
def test_aten_OPERATION_NAME_with_param(dtype, shape, param_value):
    """Test aten::OPERATION_NAME with parameter."""
    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference
    expected = torch.OPERATION_NAME(input_tensor, param_value)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x, param_value)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Edge Cases Test
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (0,),  # Empty 1D
        (10, 0),  # Empty dimension
        (1,),  # Single element
        (1, 1),  # All size-1
        (1, 1, 1, 1),  # 4D all size-1
    ],
)
def test_aten_OPERATION_NAME_edge_cases(shape):
    """Test aten::OPERATION_NAME edge cases."""
    dtype = torch.float32
    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference
    expected = torch.OPERATION_NAME(input_tensor)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Concatenation Test
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tensors", [2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_aten_OPERATION_NAME_concat(dtype, num_tensors, dim):
    """Test aten::OPERATION_NAME concatenation."""
    shape = (5, 10, 15)

    # Skip invalid dimensions
    if abs(dim) >= len(shape):
        pytest.skip(f"Dimension {dim} invalid for shape {shape}")

    # Create list of tensors
    tensors = [torch.randn(shape, dtype=dtype) for _ in range(num_tensors)]

    # Reference
    expected = torch.OPERATION_NAME(tensors, dim=dim)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(tensor_list):
        return torch.OPERATION_NAME(tensor_list, dim=dim)

    result = compiled_fn(tensors)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Numerical Stability Test
# =============================================================================


def test_aten_OPERATION_NAME_numerical_stability():
    """Test aten::OPERATION_NAME numerical stability."""
    # Test with large values that might cause overflow/underflow
    input_tensor = torch.randn(10, 20) * 100  # Large values

    # Reference
    expected = torch.OPERATION_NAME(input_tensor)

    # Compiled
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x)

    result = compiled_fn(input_tensor)

    # May need looser tolerance for numerical operations
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


# =============================================================================
# In-place Operation Test
# =============================================================================


def test_aten_OPERATION_NAME_inplace():
    """Test aten::OPERATION_NAME_ (in-place variant)."""
    dtype = torch.float32
    shape = (10, 20)

    # Create input
    input_tensor = torch.randn(shape, dtype=dtype)
    input_copy = input_tensor.clone()

    # Reference (in-place)
    expected = torch.OPERATION_NAME_(input_tensor)

    # Compiled (in-place, reset input)
    input_tensor = input_copy.clone()

    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME_(x)

    result = compiled_fn(input_tensor)

    # Verify result
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    # Verify in-place modification (if applicable in graph mode)
    # Note: In graph compilation, this may not be true in-place
    # torch.testing.assert_close(input_tensor, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Multiple Operations in Graph Test
# =============================================================================


def test_aten_OPERATION_NAME_in_graph():
    """Test aten::OPERATION_NAME as part of larger graph."""
    x = torch.randn(10, 20)
    y = torch.randn(10, 20)

    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        # Multiple operations in single graph
        c = torch.add(a, b)
        d = torch.OPERATION_NAME(c)
        e = torch.mul(d, 2.0)
        return e

    # Reference
    c_ref = torch.add(x, y)
    d_ref = torch.OPERATION_NAME(c_ref)
    expected = torch.mul(d_ref, 2.0)

    # Compiled
    result = compiled_fn(x, y)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Tolerance Helper for Different Dtypes
# =============================================================================


def get_tolerance(dtype):
    """Get appropriate tolerance for dtype."""
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-7}
    elif dtype == torch.float16:
        return {"rtol": 1e-3, "atol": 1e-5}
    elif dtype == torch.bfloat16:
        return {"rtol": 1e-2, "atol": 1e-4}
    else:
        return {"rtol": 1e-5, "atol": 1e-7}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_aten_OPERATION_NAME_with_dtype_tolerance(dtype):
    """Test aten::OPERATION_NAME with dtype-specific tolerance."""
    input_tensor = torch.randn(10, 20, dtype=dtype)

    expected = torch.OPERATION_NAME(input_tensor)

    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.OPERATION_NAME(x)

    result = compiled_fn(input_tensor)

    # Use dtype-specific tolerance
    tol = get_tolerance(dtype)
    torch.testing.assert_close(result, expected, **tol)


# =============================================================================
# Test Class for Grouping Related Tests
# =============================================================================


class TestAtenOperationNameVariants:
    """Group related tests for aten::OPERATION_NAME and variants."""

    def test_basic(self):
        """Test basic functionality."""
        x = torch.randn(10, 20)
        expected = torch.OPERATION_NAME(x)

        @torch.compile(backend=max_backend)
        def compiled_fn(input):
            return torch.OPERATION_NAME(input)

        result = compiled_fn(x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_with_parameter(self):
        """Test with parameter."""
        x = torch.randn(10, 20)
        expected = torch.OPERATION_NAME(x, param=0.5)

        @torch.compile(backend=max_backend)
        def compiled_fn(input):
            return torch.OPERATION_NAME(input, param=0.5)

        result = compiled_fn(x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_inplace(self):
        """Test in-place variant."""
        x = torch.randn(10, 20)
        expected = torch.OPERATION_NAME_(x.clone())

        @torch.compile(backend=max_backend)
        def compiled_fn(input):
            return torch.OPERATION_NAME_(input)

        result = compiled_fn(x.clone())
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


# =============================================================================
# Usage Notes
# =============================================================================

"""
When writing tests for a new ATen operation:

1. Choose appropriate test pattern(s) from above
2. Replace OPERATION_NAME with actual operation
3. Add to tests/test_aten_functions.py (middle of file)
4. Run specific test: uv run pytest tests/test_aten_functions.py::test_name -v
5. Parametrize for multiple dtypes and shapes
6. Include edge cases
7. Use appropriate tolerance for dtype
8. Test should FAIL initially (operation not implemented)
9. Implement operation
10. Test should PASS after implementation

Running tests:
- Single test: uv run pytest tests/test_aten_functions.py::test_aten_op -v
- With filter: uv run pytest tests/test_aten_functions.py::test_aten_op -v -k "float32"
- With debug: TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_aten_op -v

Tolerance guidelines:
- float32: rtol=1e-5, atol=1e-7
- float16: rtol=1e-3, atol=1e-5 (less precise)
- bfloat16: rtol=1e-2, atol=1e-4 (less precise than float16)
- Numerical ops may need looser tolerance: rtol=1e-4, atol=1e-6

Best practices:
- Test multiple data types
- Test various shapes (1D, 2D, 3D, large)
- Test edge cases (empty, size-1, etc.)
- Test broadcasting if applicable
- Test optional parameters
- Use descriptive test names
- Add docstrings explaining what is tested
"""
