# ATen Operations Testing Guide

Comprehensive guide for writing effective tests for ATen operations in the torch-max-backend project.

## Testing Philosophy

We follow **test-driven development (TDD)**:
1. Write tests first
2. See them fail
3. Implement the operation
4. See tests pass
5. Refactor if needed

## Test Structure

### Basic Test Template

```python
import torch
import pytest
from torch_max_backend import max_backend


@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16,
])
@pytest.mark.parametrize("shape", [
    (10,),           # 1D small
    (1000,),         # 1D large
    (10, 20),        # 2D
    (5, 10, 15),     # 3D
    (2, 3, 4, 5),    # 4D
])
def test_aten_operation_name(dtype, shape):
    """Test aten::operation_name implementation."""
    # Setup
    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference (PyTorch)
    expected = torch.operation_name(input_tensor)

    # Compile with MAX backend
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.operation_name(x)

    # Execute
    result = compiled_fn(input_tensor)

    # Verify
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

## Parametrization Strategies

### Data Types

Test multiple data types to ensure dtype handling:

```python
@pytest.mark.parametrize("dtype", [
    torch.float32,    # Standard floating point
    torch.float16,    # Half precision
    torch.bfloat16,   # Brain floating point (ML optimized)
    # Add others if operation supports them
    # torch.float64,  # Double precision
    # torch.int32,    # Integer (for specific ops)
    # torch.int64,    # Long integer
])
```

**Tolerance by dtype**:
- `float32`: `rtol=1e-5, atol=1e-7`
- `float16`: `rtol=1e-3, atol=1e-5` (less precise)
- `bfloat16`: `rtol=1e-2, atol=1e-4` (less precise than float16)

### Shapes

Test various dimensions and sizes:

```python
@pytest.mark.parametrize("shape", [
    # 1D
    (1,),            # Single element
    (10,),           # Small
    (1000,),         # Large

    # 2D
    (1, 1),          # Minimal
    (10, 20),        # Small rectangular
    (100, 100),      # Square
    (1, 1000),       # Row vector
    (1000, 1),       # Column vector

    # 3D
    (5, 10, 15),     # Small 3D
    (2, 100, 50),    # Batch of matrices

    # 4D
    (2, 3, 4, 5),    # Small 4D
    (1, 3, 224, 224), # Image-like (batch, channels, height, width)
])
```

### Edge Cases

Always test edge cases:

```python
@pytest.mark.parametrize("shape", [
    (0,),             # Empty tensor
    (1,),             # Single element
    (10, 0),          # Empty dimension
    (1, 1, 1, 1),     # All size-1 dimensions
])
def test_aten_operation_edge_cases(shape):
    # ...
```

## Testing Different Operation Types

### Element-wise Operations

```python
def test_aten_relu():
    """Test aten::relu - element-wise activation."""
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 10, 15)])
    def run_test(dtype, shape):
        # Include negative values to test ReLU behavior
        input_tensor = torch.randn(shape, dtype=dtype)

        expected = torch.relu(input_tensor)

        @torch.compile(backend=max_backend)
        def compiled_fn(x):
            return torch.relu(x)

        result = compiled_fn(input_tensor)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    run_test()
```

### Reduction Operations

```python
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dim", [0, 1, -1, None])
@pytest.mark.parametrize("keepdim", [False, True])
def test_aten_sum(dtype, dim, keepdim):
    """Test aten::sum with various reduction dimensions."""
    shape = (5, 10, 15)
    input_tensor = torch.randn(shape, dtype=dtype)

    if dim is None:
        expected = torch.sum(input_tensor)
    else:
        expected = torch.sum(input_tensor, dim=dim, keepdim=keepdim)

    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        if dim is None:
            return torch.sum(x)
        return torch.sum(x, dim=dim, keepdim=keepdim)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### Operations with Optional Parameters

```python
@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0, None])
def test_aten_add_with_alpha(alpha):
    """Test aten::add with alpha parameter."""
    x = torch.randn(10, 20)
    y = torch.randn(10, 20)

    if alpha is None:
        expected = torch.add(x, y)
    else:
        expected = torch.add(x, y, alpha=alpha)

    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        if alpha is None:
            return torch.add(a, b)
        return torch.add(a, b, alpha=alpha)

    result = compiled_fn(x, y)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### Broadcasting Operations

```python
@pytest.mark.parametrize("shapes", [
    ((10, 20), (20,)),       # Matrix + vector
    ((5, 1, 10), (10,)),     # 3D + 1D
    ((5, 1), (1, 10)),       # Row + column
    ((1, 10, 20), (5, 10, 20)), # Batch broadcast
])
def test_aten_add_broadcasting(shapes):
    """Test aten::add with broadcasting."""
    shape1, shape2 = shapes
    x = torch.randn(shape1)
    y = torch.randn(shape2)

    expected = torch.add(x, y)

    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        return torch.add(a, b)

    result = compiled_fn(x, y)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### In-place Operations

```python
def test_aten_relu_inplace():
    """Test aten::relu_ (in-place variant)."""
    input_tensor = torch.randn(10, 20)
    input_copy = input_tensor.clone()

    # Reference
    expected = torch.relu_(input_tensor)

    # Compiled (reset input)
    input_tensor = input_copy.clone()

    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.relu_(x)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
    # Verify in-place modification
    torch.testing.assert_close(input_tensor, expected, rtol=1e-5, atol=1e-7)
```

## Running Tests

### Run Specific Test

```bash
# Single test function
uv run pytest tests/test_aten_functions.py::test_aten_operation -v

# Specific parametrization
uv run pytest tests/test_aten_functions.py::test_aten_operation -v -k "float32"

# Multiple filters
uv run pytest tests/test_aten_functions.py::test_aten_operation -v -k "float32 and 10"
```

### Run Test File

```bash
# All tests in file
uv run pytest tests/test_aten_functions.py -v

# With parallel execution
uv run pytest tests/test_aten_functions.py -n 4 -v
```

### Run with Debug Info

```bash
# Show PyTorch FX graphs
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Show timing
TORCH_MAX_BACKEND_PROFILE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Disable type checking
TORCH_MAX_BACKEND_BEARTYPE=0 uv run pytest tests/test_aten_functions.py::test_op -v
```

## Assertion Strategies

### Exact Equality (Rare)

```python
torch.testing.assert_equal(result, expected)
```

Use only for:
- Integer operations
- Boolean operations
- Exact mathematical properties

### Close with Tolerance

```python
torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

**Parameters**:
- `rtol`: Relative tolerance (relative to magnitude of values)
- `atol`: Absolute tolerance (fixed threshold)

**Formula**: `|result - expected| <= atol + rtol * |expected|`

### Tolerance by Precision

```python
if dtype == torch.float32:
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
elif dtype == torch.float16:
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-5)
elif dtype == torch.bfloat16:
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-4)
```

### Shape Assertions

```python
assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
```

### Type Assertions

```python
assert result.dtype == expected.dtype, f"Dtype mismatch: {result.dtype} vs {expected.dtype}"
```

## Common Test Patterns

### Test Multiple Operations Together

```python
def test_aten_complex_graph():
    """Test multiple operations in single graph."""
    x = torch.randn(10, 20)
    y = torch.randn(10, 20)

    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        c = torch.add(a, b)
        d = torch.relu(c)
        return torch.sum(d)

    expected = torch.sum(torch.relu(torch.add(x, y)))
    result = compiled_fn(x, y)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### Test with Different Devices

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_aten_operation_devices(device):
    """Test operation on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    input_tensor = torch.randn(10, 20, device=device)

    expected = torch.operation(input_tensor)

    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.operation(x)

    result = compiled_fn(input_tensor)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### Test Numerical Stability

```python
def test_aten_log_softmax_stability():
    """Test log_softmax numerical stability with large values."""
    # Large values to test numerical stability
    x = torch.randn(10, 20) * 100

    expected = torch.nn.functional.log_softmax(x, dim=-1)

    @torch.compile(backend=max_backend)
    def compiled_fn(input):
        return torch.nn.functional.log_softmax(input, dim=-1)

    result = compiled_fn(x)

    # May need looser tolerance for numerical operations
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)
```

## Debugging Failed Tests

### Print Intermediate Values

```python
def test_aten_operation_debug():
    x = torch.randn(10, 20)

    @torch.compile(backend=max_backend)
    def compiled_fn(input):
        # Can't print inside compiled function
        return torch.operation(input)

    result = compiled_fn(x)
    expected = torch.operation(x)

    # Print for debugging
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: {expected.shape}")
    print(f"Max difference: {(result - expected).abs().max()}")

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

### Use Verbose Mode

```bash
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_op -v -s
```

This shows:
- PyTorch FX graph structure
- Operation mappings
- Compilation details

### Isolate the Issue

```python
# Test operation in isolation
def test_single_operation():
    x = torch.randn(10, 20)

    # Direct call (no compilation)
    direct_result = torch.operation(x)

    # Compiled call
    @torch.compile(backend=max_backend)
    def compiled_fn(input):
        return torch.operation(input)

    compiled_result = compiled_fn(x)

    torch.testing.assert_close(compiled_result, direct_result)
```

## Best Practices

1. **Test before implementing**: Write tests first, see them fail
2. **Parametrize extensively**: Test multiple dtypes, shapes, parameters
3. **Test edge cases**: Empty tensors, size-1 dims, extreme values
4. **Use appropriate tolerances**: Adjust for dtype precision
5. **Test broadcasting**: Various shape combinations
6. **Test optional parameters**: All combinations of optional args
7. **Descriptive test names**: Name tests clearly: `test_aten_operation_name`
8. **Add docstrings**: Explain what the test verifies
9. **Run specific tests during development**: Don't run full suite
10. **Use fixtures for common setups**: Reduce code duplication

## Test Organization

### File Placement

- All ATen operation tests: `tests/test_aten_functions.py`
- Place new tests in the middle of the file (avoid merge conflicts)
- Keep tests alphabetically organized within sections

### Test Naming

```python
# Good
def test_aten_relu():
def test_aten_relu_inplace():
def test_aten_sum_with_dim():
def test_aten_add_broadcasting():

# Bad
def test_relu():  # Missing aten prefix
def test_1():     # Not descriptive
def test_my_test():  # Not clear what it tests
```

### Grouping Related Tests

```python
class TestAtenReluVariants:
    """Group related ReLU tests together."""

    def test_relu_basic(self):
        # ...

    def test_relu_inplace(self):
        # ...

    def test_relu_numerical_stability(self):
        # ...
```

## Performance Testing

While not required for every operation, consider adding performance tests for critical ops:

```python
import time

def test_aten_matmul_performance():
    """Benchmark matmul performance (optional)."""
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    @torch.compile(backend=max_backend)
    def compiled_fn(a, b):
        return torch.matmul(a, b)

    # Warmup
    for _ in range(10):
        compiled_fn(x, y)

    # Benchmark
    start = time.time()
    for _ in range(100):
        result = compiled_fn(x, y)
    elapsed = time.time() - start

    print(f"Matmul 100 iterations: {elapsed:.3f}s")
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Pre-commit hooks (linting)

Ensure tests pass locally before pushing:

```bash
# Run tests
uv run pytest tests/test_aten_functions.py::test_your_op -v

# Run linter
uvx pre-commit run --all-files
```
