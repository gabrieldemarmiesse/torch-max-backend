# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations.

## Dependencies and Setup

- **Python**: >=3.11 required (per pyproject.toml)
- **Key Dependencies**: 
  - `max` (Modular's MAX framework)
  - `torch` (PyTorch)
  - `tabulate` (for formatted output)
- **Development Dependencies**:
  - `pytest>=8.4.1` with plugins (`pytest-xdist`, `pytest-forked`, `pytest-split`)
  - `ruff>=0.12.7` (for linting/formatting)
  - `transformers>=4.54.1`, `accelerate>=1.10.0` (for model examples)
  - `torchvision>=0.22.1`, `pillow>=11.3.0` (for vision tasks)
- **Package Manager**: Uses `uv` for dependency management

## Common Commands

```bash
# Run tests (with parallel execution)
uv run pytest -n 15

# Run specific test file
uv run pytest tests/test_compiler.py

# Run with profiling enabled
TORCH_MAX_BACKEND_PROFILE=1 uv run pytest tests/test_compiler.py

# Run with verbose output (shows graph structures)
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_compiler.py

# Run linter/formatter
uv run ruff check .
uv run ruff format .

# Or use pre-commit for all checks
uvx pre-commit run --all-files
```

## Project Structure

```
torch-max-backend/
├── torch_max_backend/                      # Main package
│   ├── __init__.py                        # Package exports
│   ├── aten_functions.py                  # ATen operation implementations (graph and eager mode)
│   ├── flags.py                           # Environment variable handling for profiling/verbose
│   ├── profiler.py                        # Profiling utilities
│   ├── testing.py                         # Testing helper functions
│   ├── torch_compile_backend/             # Torch compilation backend
│   │   ├── __init__.py
│   │   ├── compiler.py                    # Core compiler implementation
│   │   ├── debug.py                       # Debugging utilities
│   │   └── utils.py                       # Utility functions
│   ├── max_device/                        # MAX device implementation (eager mode)
│   │   ├── __init__.py
│   │   ├── torch_max_tensor.py            # TorchMaxTensor wrapper class
│   │   ├── torch_max_device_module.py     # MAX device module
│   │   ├── max_device_aten_ops.py         # ATen ops registration for eager execution
│   │   ├── log_aten_calls.py              # Logging for aten calls
│   │   └── register.py                    # Device registration
│   ├── mojo_kernels/                      # Mojo kernel implementations
│   │   ├── __init__.mojo
│   │   ├── bitwise.mojo                   # Bitwise operations
│   │   ├── math.mojo                      # Math operations
│   │   └── pooling.mojo                   # Pooling operations
│   └── custom_torch_ops_in_mojo/          # Custom PyTorch ops in Mojo
│       ├── __init__.py
│       └── torch_custom_ops.py
├── tests/                                  # Test suite
│   ├── __init__.py
│   ├── conftest.py                        # Pytest fixtures
│   ├── test_compiler.py                   # Compilation tests
│   ├── test_aten_functions.py             # ATen operation tests
│   ├── test_high_level_ops.py             # High-level operation tests
│   └── test_max_device.py                 # MAX device tests
├── demo_scripts/                           # Example model implementations (GPT-2, VGG, DenseNet, Gemma3, etc.)
├── docs/                                   # Documentation
│   └── CONTRIBUTING.md                    # Contribution guidelines
├── pyproject.toml                         # Project configuration
├── uv.lock                               # Dependency lock file
├── CLAUDE.md                            # This file
└── README.md                           # Project documentation and usage examples
```

## Architecture

The project implements PyTorch operations using MAX in two modes:

1. **Torch Compile Backend (Graph Mode)**: Compiles PyTorch FX graphs to MAX operations
2. **MAX Device (Eager Mode)**: Executes operations eagerly on the custom `max_device` PyTorch device

### Key Components

#### `torch_max_backend/aten_functions.py`
- **Central ATen Operation Implementations**: Contains implementations for all supported PyTorch ATen operations
- **Dual-Mode Support**: Functions work in both graph compilation mode (using `TensorValue`) and eager mode (using `MaxEagerTensor`)
- **Type Flexibility**: Operations accept `MaxTensor = TensorValue | MaxEagerTensor` for unified implementation
- **Mapping Dictionary**: `MAPPING_TORCH_ATEN_TO_MAX` maps PyTorch ATen ops to implementation functions
- **Decomposition Support**: `DECOMPOSITION_TABLE` contains operations that decompose into simpler ops

#### `torch_max_backend/torch_compile_backend/compiler.py`
- **`max_backend`**: Main compiler function for `torch.compile(backend=max_backend)`
- **Graph Compilation Process**:
  - Accepts FX GraphModule and example inputs
  - Optionally prints graph structure for debugging (controlled by `TORCH_MAX_BACKEND_VERBOSE`)
  - Uses meta tensors to track shapes without memory allocation
  - Creates runtime function that executes graph nodes
  - Returns wrapped function compatible with PyTorch

#### `torch_max_backend/max_device/`
- **`torch_max_tensor.py`**: Defines `TorchMaxTensor` class that wraps MAX eager tensors
- **`max_device_aten_ops.py`**: Registers ATen operations for eager execution on `max_device`
- **`register.py`**: Handles device registration with PyTorch
- **Registration Pattern**:
  ```python
  register_aten_op("aten::operation_name")(
      wrap_for_max_device(aten_functions.aten_operation_name)
  )
  ```
- **`wrap_for_max_device`**: Automatically converts between `TorchMaxTensor` and `MaxEagerTensor`

#### `torch_max_backend/mojo_kernels/`
- **Custom Mojo Implementations**: Contains optimized Mojo kernels for operations
- **Kernel Categories**:
  - `bitwise.mojo`: Bitwise operations
  - `math.mojo`: Mathematical operations
  - `pooling.mojo`: Pooling operations

#### `torch_max_backend/flags.py`
- **Environment Variable Support**:
  - `TORCH_MAX_BACKEND_PROFILE` / `PYTORCH_MAX_BACKEND_PROFILE`: Enable timing profiling
  - `TORCH_MAX_BACKEND_VERBOSE` / `PYTORCH_MAX_BACKEND_VERBOSE`: Enable verbose graph output
  - Both accept values: "1", "true", "yes" (case-insensitive)

### Execution Modes

#### Graph Mode (torch.compile)
1. PyTorch function decorated with `@torch.compile(backend=max_backend)`
2. FX graph generated and passed to `max_backend`
3. Graph nodes processed sequentially using `MAPPING_TORCH_ATEN_TO_MAX`
4. Operations execute on `TensorValue` (symbolic graph tensors)
5. Compiled function returns PyTorch tensors

#### Eager Mode (max_device)
1. Tensors created on `torch.device("max_device")`
2. Operations dispatched through PyTorch's dispatcher to registered ATen ops
3. `wrap_for_max_device` converts `TorchMaxTensor` → `MaxEagerTensor`
4. Operations execute immediately on `MaxEagerTensor`
5. Results converted back to `TorchMaxTensor`

## Testing

### Test Files
- **`test_aten_functions.py`**: Comprehensive tests for individual ATen operations
  - Tests each operation with multiple data types and shapes
  - Validates both graph mode and eager mode execution
  - Uses `pytest.mark.parametrize` for extensive coverage
- **`test_compiler.py`**: Tests for the torch.compile backend
  - Verifies graph compilation and execution
  - Tests integration with PyTorch's compilation pipeline
- **`test_max_device.py`**: Tests for the custom max_device
  - Validates device creation and tensor operations
  - Tests tensor movement between devices
- **`test_high_level_ops.py`**: Tests for high-level composed operations

### Test Coverage
- **ATen Operations**: Individual PyTorch ATen operations
- **Device Support**: Tests run on CPU and max_device (with GPU support if available)
- **Compilation**: Verifies that `@torch.compile(backend=max_backend)` works correctly
- **Eager Execution**: Validates operations on tensors created with `device="max_device"`
- **Error Handling**: Tests for unsupported operations raise appropriate errors

### Test Fixtures
- `tensor_shapes`: Common tensor shapes for testing (various sizes and dimensions)
- `devices`: Available devices determined by MAX accelerator detection
- Various dtype fixtures for comprehensive type testing

## Current Limitations

1. **Limited Operation Support**: Only operations implemented in `aten_functions.py` and registered in `max_device_aten_ops.py` are supported
2. **Partial Coverage**: Not all PyTorch ATen operations are implemented yet
3. **GPU Compatibility**: Not all NVIDIA/AMD GPUs are supported by MAX - use `get_accelerators()` to check available devices
4. **Error Handling**: Unsupported operations will raise errors indicating missing implementations

## Development Notes

- **Code Quality**: Uses Ruff for linting/formatting with Python 3.11+ target and pyupgrade rules
- **Testing Strategy**: Tests use `pytest-forked` for process isolation and `pytest-xdist` for parallelization
- **Debugging Tools**:
  - Environment variables for profiling and verbose output
  - Graph visualization when `TORCH_MAX_BACKEND_VERBOSE=1`
- **Model Examples**: `demo_scripts/` contains examples showing real-world usage:
  - GPT-2, Gemma3 (LLM models)
  - VGG, DenseNet (vision models)
  - `no_graph_breaks.py` (example demonstrating graph compilation without breaks)
- **Reference Materials**:
  - The directory `../pytorch/` contains PyTorch source for `torch.compile` internals and ATen operation definitions
  - The directory `../modular/max` contains MAX graph implementation examples and API reference

## Usage Examples

The backend can be used in two ways:

### Graph Mode (torch.compile)
Use the MAX backend with `torch.compile` for ahead-of-time graph compilation:

```python
from torch_max_backend import max_backend
import torch

@torch.compile(backend=max_backend)
def my_function(x, y):
    return x + y * 2

# Call with regular PyTorch tensors
x = torch.randn(10, 10)
y = torch.randn(10, 10)
result = my_function(x, y)
```

### Eager Mode (max_device)
Use tensors on the custom `max_device` for eager execution:

```python
import torch

# Create tensors on max_device
x = torch.randn(10, 10, device="max_device")
y = torch.randn(10, 10, device="max_device")

# Operations execute immediately on MAX
result = x + y * 2

# Move back to CPU if needed
result_cpu = result.cpu()
```

Device compatibility should be checked using `get_accelerators()` before GPU usage.


## To add support for an op

To add support for a new ATen operation, follow this test-driven development process:

### Step 1: Research the Operation
Ask a subagent to explore the PyTorch codebase `../pytorch` and look for:
- The signature of the ATen function
- The meaning of inputs and outputs
- Any important behavioral details
- Request a full report with this information

### Step 2: Write Unit Tests
Write unit tests in `test_aten_functions.py` using this op directly:
- Place tests somewhere in the middle of the file to avoid merge conflicts
- Use `pytest.mark.parametrize` to test multiple input data types and shapes
- Test edge cases and different parameter combinations

### Step 3: Run Tests (Expected to Fail)
Run the unit tests:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```
You should see an error message explaining that the ATen op is not supported.

### Step 4: Add Operation Signature to aten_functions.py
- Find the alphabetically correct position in `aten_functions.py`
- Add a comment with the full ATen operation signature
- **IMPORTANT**: The file is sorted alphabetically and must remain this way

Example:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
```

### Step 5: Research MAX Implementation
Ask a subagent to explore the directory `../modular/max` to find:
- MAX functions that do something similar (sometimes there are direct equivalents)
- Functions that can be composed to re-implement the operation
- Check models created with MAX for usage examples
- Look in `kernels.py` for complex operation implementations
- Request a full report of useful functions with descriptions of inputs/outputs

### Step 6: Implement the Operation
Write the ATen operation implementation in `aten_functions.py` just below the signature comment:

**Important**: The implementation must support **both execution modes**:
- **Graph Mode**: Works with `TensorValue` (symbolic tensors)
- **Eager Mode**: Works with `MaxEagerTensor` (actual tensors)

Use the type hint `MaxTensor = TensorValue | MaxEagerTensor` for tensor parameters.

Example implementation:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def aten__log_softmax(
    self: MaxTensor, dim: int, half_to_float: bool
) -> MaxTensor:
    # Implementation using MAX operations that works for both modes
    return F.log_softmax(self, axis=dim)
```

### Step 7: Register for Eager Mode Execution
Add the operation to `torch_max_backend/max_device/max_device_aten_ops.py`:

**Registration Pattern**:
```python
register_aten_op("aten::_log_softmax")(
    wrap_for_max_device(aten_functions.aten__log_softmax)
)
```

Place the registration in alphabetical order within the file. The `wrap_for_max_device` wrapper automatically:
- Converts `TorchMaxTensor` inputs to `MaxEagerTensor`
- Executes the operation
- Converts results back to `TorchMaxTensor`

**Note**: For operations requiring custom device handling (like `aten::_copy_from`), you can implement a custom function directly instead of using `wrap_for_max_device`.

### Step 8: Re-run Tests
Run the unit tests again and verify they pass:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```

Test both execution modes if applicable:
- Graph mode via `torch.compile(backend=max_backend)`
- Eager mode via tensors on `torch.device("max_device")`

### Step 9: Run Linter
Make sure to run the linter:
```bash
uvx pre-commit run --all-files
```

**Do not run the whole test suite** as it takes too long. Only run tests for the specific operation you added.

### Summary: Two-Part Implementation
When adding an operation, you need to update **two files**:
1. **`aten_functions.py`**: Core implementation (works for both modes)
2. **`max_device_aten_ops.py`**: Registration for eager mode execution

This ensures the operation works in both `torch.compile()` and on the `max_device`.


## To find the correct type hints for a function
It may be hard to find the correct type hints for a function. What you should do in this case is:
1) Add an obviously wrong type hint, for example datetime.timezone in an aten function.
2) Run an existing unit test that calls this function.
3) Beartype will throw an error and give the name of the type being actually passed to the function.
4) Replace the type hint by the type given by beartype.
5) Run the unit test again to check that it works.
6) Run the whole test suite to verify that the type hint shouldn't be wider.
