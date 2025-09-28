# Contributing to Torch MAX Backend

Thank you for your interest in contributing to the Torch MAX Backend project! This guide will help you get started with development and understand our development workflow.

## Prerequisites

- **Python**: >=3.11 required
- **uv**: Package manager for dependency management
- **MAX SDK**: Modular's MAX framework
- **Git**: For version control

## Getting Started

### 1. Fork and Clone the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone git@github.com:youraccount/torch-max-backend.git
cd torch-max-backend/
```

### 2. Set Up Development Environment

Create and activate a virtual environment:

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install development dependencies
uv pip install -e .[dev]
```

### 3. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
uvx pre-commit install
```

### 4. Verify Setup

Run the test suite to ensure everything is working:

```bash
# Run all tests with parallel execution
uv run pytest -n 15

# Run with verbose output to see more details
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_compiler.py
```

## Development Workflow

### Project Structure

```
torch-max-backend/
├── torch_max_backend/          # Main package
│   ├── __init__.py             # Package exports
│   ├── compiler.py             # Core compiler implementation
│   ├── aten_functions.py       # PyTorch ATen operation implementations
│   ├── max_device.py           # Device management
│   ├── flags.py                # Environment variable handling
│   └── utils.py                # Utility functions
├── tests/                      # Test suite
│   ├── test_compiler.py        # Basic compilation tests
│   ├── test_aten_functions.py  # ATen function tests
│   └── conftest.py             # Pytest fixtures
├── demo_scripts/               # Example implementations
├── docs/                       # Documentation
└── pyproject.toml              # Project configuration
```

### Code Quality Standards

We maintain high code quality using several tools:

- **Ruff**: Linting and formatting (>=0.12.7)
- **Beartype**: Runtime type checking
- **Pre-commit**: Automated code quality checks

### Code Formatting and Linting

```bash
# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Run all pre-commit checks
uvx pre-commit run --all-files
```

## Adding Support for New Operations

We use **test-driven development** to add support for new PyTorch operations. Follow these steps:

### Step 1: Research the Operation

Explore the [PyTorch ATen native codebase](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) to understand:
- Function signature
- Input and output meanings
- Expected behavior

### Step 2: Write Unit Tests

1. Add unit tests in `tests/test_aten_functions.py`
2. Place tests in the middle of the file to avoid merge conflicts
3. Use `pytest.mark.parametrize` to test various data types

### Step 3: Run Tests (Should Fail)

```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op
```

You should see an error indicating the operation is not supported.

### Step 4: Find Operation Signature

1. Locate the operation signature in `torch_max_backend/aten_functions.py`
2. If not present, add it following alphabetical order
3. The signature should be in a comment above the implementation

### Step 5: Research MAX Implementation

Check the [Max Graph Ops](https://docs.modular.com/max/api/python/graph/ops/) available looking for:

- Equivalent MAX function
- Composable functions to implement the operation
- Examples in existing MAX models

If there is "backward" in the name, it's likely that you'll have to implement it in Mojo and put it in the "mojo_kernels" directory. If not, it's likely MAX already have something that can do it in.

When there is no MAX alternative, the best alternative would be to migrate to Mojo a C++ function, by looking for the signature in the Pytorch codebase.

### Step 6: Implement the Operation

Write the ATen operation implementation using MAX functions.

This is an example of currently implemented `aten.cat()` operation:
```python
# cat(Tensor[] tensors, int dim=0) -> Tensor
@map_to(aten.cat)
def aten_cat(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    return max_ops.concat(tensors, axis=dim)
```

### Step 7: Verify Implementation

```bash
# Run your specific tests
uv run pytest tests/test_aten_functions.py::test_your_new_op

# Run full test suite
uv run pytest -n 15

# Run code quality checks
uvx pre-commit run --all-files
```

## Type Hints

Finding correct type hints can be challenging. Here's our approach:

1. Add an obviously wrong type hint (e.g., `datetime.timezone`)
2. Run a unit test that calls the function
3. Beartype will throw an error with the correct type name
4. Replace with the correct type from Beartype's error message
5. Verify with unit tests and full test suite

## Debugging Tools

### Environment Variables

- `TORCH_MAX_BACKEND_PROFILE=1`: Enable timing profiling
- `TORCH_MAX_BACKEND_VERBOSE=1`: Show graph structures
- `TORCH_MAX_BACKEND_BEARTYPE=0`: Disable type checking (for debugging)

Values accepted: "1", "true", "yes" (case-insensitive)

### Example Usage

```bash
# Debug compilation with verbose output
TORCH_MAX_BACKEND_VERBOSE=1 uv run python your_script.py

# Profile performance
TORCH_MAX_BACKEND_PROFILE=1 uv run python your_script.py
```

## Testing Strategy

### Test Coverage Areas

- **Basic Operations**: Arithmetic operations on available devices
- **Device Support**: CPU and CUDA compatibility
- **Compilation**: `torch.compile` integration
- **Error Handling**: Unsupported operations

### Test Fixtures

- `tensor_shapes`: Common tensor shapes for testing
- `devices`: Available devices from MAX accelerator detection

### Running Specific Tests

```bash
# Test specific operation
uv run pytest tests/test_aten_functions.py::test_aten_amin_multiple_dims -v

# Test on specific device
uv run pytest tests/test_aten_functions.py -k "cuda" -v

# Test with forked processes (isolation)
uv run pytest tests/test_aten_functions.py --forked
```

## Keeping Your Fork Updated

To stay synchronized with the upstream repository:

```bash
# Add upstream remote (one-time setup)
git remote add upstream https://github.com/gabrieldemarmiesse/torch-max-backend.git

# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Push updated branch
git push --force-with-lease origin your-branch-name
```

Happy contributing! 🚀
