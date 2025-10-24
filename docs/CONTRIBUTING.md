# Contributing to Torch MAX Backend

Thank you for your interest in contributing to the Torch MAX Backend project! This guide will help you get started with development and understand our development workflow.

## Prerequisites

- **uv**: Package manager for dependency management
- **Git**: For version control

## Getting Started

### 1. Fork and Clone the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone git@github.com:youraccount/torch-max-backend.git
cd torch-max-backend/
```

### 2. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
uvx pre-commit install
```

### 3. Verify Setup

Run the test suite to ensure everything is working:

```bash
# Run all tests with parallel execution
uv run pytest -n 15

# Run a test file
uv run pytest tests/test_compiler.py
```

## Development Workflow

### Code Quality Standards

We maintain high code quality using several tools:

- **Ruff**: Linting and formatting (>=0.12.7)
- **Beartype**: Runtime type checking
- **Pre-commit**: Automated code quality checks

### Using the Mojo GPU Kernels Skill

The project includes a comprehensive **Mojo GPU Kernels Skill** to accelerate GPU kernel development for the PyTorch backend.

#### What the Skill Provides

- **Kernel patterns**: Element-wise operations, reductions (warp/block/multi-block), shared memory operations
- **Code templates**: Ready-to-use starter code for common kernel types
- **MAX operations catalog**: Comprehensive reference of available MAX framework operations
- **Performance guidance**: Memory coalescing, occupancy optimization, bank conflict avoidance
- **Workflow integration**: Specific guidance for torch-max-backend development process

#### Quick Start with the Skill

When working with Claude Code, simply reference the skill in your prompts:

```
Use the mojo-kernels skill to help me implement the aten::relu operation.
```

```
I need to write a reduction kernel for sum. Use the mojo-kernels skill to show me the best pattern.
```

```
How do I optimize this kernel for better memory coalescing? Use the mojo-kernels skill.
```

#### When to Use the Skill

**The skill helps with:**
- Implementing new ATen operations that require custom kernels
- Writing GPU kernels in Mojo or optimizing existing ones
- Finding the right kernel pattern (element-wise, reduction, shared memory)
- Implementation with proper type hints and error handling

## Using the ATen Operations Skill

The project includes a comprehensive **ATen Operations Skill** that guides you through implementing PyTorch ATen operations.

### Quick Start with the Skill

```
Use the aten-ops skill to help me implement aten::relu
```

The skill provides:
- **8-step workflow**: Complete test-driven development process
- **Implementation patterns**: Ready-to-use code templates for all operation types
- **Testing strategies**: Parametrized tests for multiple dtypes and shapes
- **Type hints guidance**: Beartype method for finding correct types
- **MAX operation discovery**: Integration with mojo-kernels skill

## Adding Support for New Operations

We use **test-driven development** to add support for new PyTorch operations. Follow these steps (or use the aten-ops skill for guided assistance):

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

**💡 Tip**: Use the **mojo-kernels skill** to quickly find MAX operations:
```
Use the mojo-kernels skill to find MAX operations for implementing [operation_name]
```

The skill includes a comprehensive catalog of MAX operations in its references.

If there is "backward" in the name, it's likely that you'll have to implement it in Mojo and put it in the "mojo_kernels" directory. If not, it's likely MAX already have something that can do it in.

When there is no MAX alternative, the best alternative would be to migrate to Mojo a C++ function, by looking for the signature in the Pytorch codebase. **The mojo-kernels skill provides templates and patterns for writing custom Mojo kernels.**

### Step 6: Implement the Operation

Write the ATen operation implementation using MAX functions.

**💡 Tip**: Use the **aten-ops skill** for implementation patterns:
```
Use the aten-ops skill to show me the implementation pattern for [operation type]
```

The skill provides templates for:
- Element-wise operations
- Reductions
- Shape manipulation
- Linear algebra
- Tensor combination

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
