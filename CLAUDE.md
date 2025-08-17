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
max-torch-backend/
├── torch_max_backend/       # Main package
│   ├── __init__.py         # Package exports (max_backend, get_accelerators)
│   ├── compiler.py         # Core compiler implementation
│   ├── mappings.py         # PyTorch to MAX/Mojo operation mappings
│   └── flags.py            # Environment variable handling for profiling/verbose
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   └── test_compiler.py   # Basic compilation tests
├── pretrained_models/      # Example model implementations
│   ├── run_gpt2.py        # GPT-2 model example
│   └── run_vgg.py         # VGG model example
├── ressources/             # Reference materials
│   └── aten_ops.txt       # Complete list of PyTorch ATen operation signatures
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
├── CLAUDE.md            # This file
└── README.md           # Project documentation and usage examples
```

## Architecture

The project implements a custom PyTorch compiler backend (`max_backend`, aliased as `max_backend` in README examples) that:

1. **Graph Analysis**: Takes PyTorch FX graphs and analyzes their structure
2. **Operation Mapping**: Maps PyTorch operations to Mojo/MAX equivalents via `MAPPING_TORCH_TO_MOJO_FUNCTIONS`
3. **Custom Operations**: Creates MAX custom operations (`MyMaxOp`) that wrap the compiled functions
4. **Runtime Bridge**: Provides a bridge between PyTorch tensors and MAX execution

### Key Components

#### `torch_max_backend/compiler.py`
- **`max_backend`**: Main compiler function exported to users
- **`get_accelerators`**: Function to query available MAX accelerators (CPU + GPU if supported)
- **Compilation Process**: 
  - Accepts FX GraphModule and example inputs
  - Optionally prints graph structure for debugging (controlled by `TORCH_MAX_BACKEND_VERBOSE`)
  - Uses meta tensors to track shapes without memory allocation
  - Creates runtime function that executes graph nodes
  - Returns wrapped function compatible with PyTorch

#### `torch_max_backend/flags.py`
- **Environment Variable Support**:
  - `TORCH_MAX_BACKEND_PROFILE` / `PYTORCH_MAX_BACKEND_PROFILE`: Enable timing profiling
  - `TORCH_MAX_BACKEND_VERBOSE` / `PYTORCH_MAX_BACKEND_VERBOSE`: Enable verbose graph output
  - Both accept values: "1", "true", "yes" (case-insensitive)

#### `torch_max_backend/mappings.py`
- **`MAPPING_TORCH_TO_MOJO_FUNCTIONS`**: Dictionary mapping PyTorch ops to MAX/Mojo equivalents
- **Supported Operations**:
  - Arithmetic: `add`, `sub`, `mul`, `truediv`, `floordiv`, `pow`, `mod`
  - Math functions: `abs`, `cos`, `sin`
  - Both `operator` module and `torch` module variants supported

### Compilation Flow

1. PyTorch function decorated with `@torch.compile(backend=max_backend)` or `@torch.compile(backend=max_backend)`
2. FX graph generated and passed to `max_backend`
3. Graph nodes processed sequentially:
   - `placeholder` nodes map to function arguments
   - `call_function` nodes execute mapped operations via `MAPPING_TORCH_TO_MOJO_FUNCTIONS`
   - `output` nodes return results as tuple
4. Custom MAX operation created with:
   - Runtime function
   - CustomOpLibrary with KernelLibrary and MLIR context
   - Input/output type information from meta tensors
5. Compiled function allocates output tensors and executes custom op on specified device

## Testing

### Test Coverage
- **Basic Operations**: Arithmetic operations on available devices
- **Device Support**: Tests run on CPU and CUDA (if available via `get_accelerators()`)
- **Compilation**: Verifies that `@torch.compile(backend=max_backend)` works correctly
- **Error Handling**: Tests for unsupported operations raise appropriate errors

### Test Fixtures  
- `tensor_shapes`: Common tensor shapes for testing (various sizes and dimensions)
- `devices`: Available devices determined by MAX accelerator detection

## Current Limitations

1. **Limited Operation Support**: Only operations listed in `mappings.py` are supported
2. **No Complex Operations**: Matrix multiplication, reductions, reshaping not yet implemented  
3. **GPU Compatibility**: Not all NVIDIA/AMD GPUs are supported by MAX - use `get_accelerators()` to check
4. **Error Handling**: Raises ValueError for unsupported operations

## Development Notes

- **Code Quality**: Uses Ruff for linting/formatting with Python 3.11+ target and pyupgrade rules
- **Testing Strategy**: Tests use `pytest-forked` for process isolation and `pytest-xdist` for parallelization
- **Debugging Tools**: 
  - Environment variables for profiling and verbose output
  - Graph visualization when `TORCH_MAX_BACKEND_VERBOSE=1`
- **Model Examples**: `pretrained_models/` contains GPT-2 and VGG examples showing real-world usage
- **Reference Materials**: 
  - `ressources/aten_ops.txt` contains complete PyTorch ATen operation signatures
  - The directory `../modular/` contains MAX graph implementation examples
  - The directory `../pytorch/` contains PyTorch source for `torch.compile` internals

## Usage Examples from README

The backend is used via standard `torch.compile` syntax:

```python
from torch_max_backend import max_backend  # or max_backend
import torch

@torch.compile(backend=max_backend)
def my_function(x, y):
    return x + y * 2
```

Device compatibility should be checked using `get_accelerators()` before GPU usage.


## To add support for an op
To add support for an op, the process is the following:
We use test-driven dev
1) Find a basic torch function that's very close to the aten op you want to use.
2) Write a few unit tests in test_compiler.py using this op (somewhere in the middle of the file to avoid conflicts).
3) Run those unit tests. You should see an error message explaining that the aten op is not supported.
4) Find in aten_functions.py the comment giving the signature of the aten op.
5) Look into the directory `../modular/max` to find if functions exist in MAX to do something similar (sometimes they have direct equivalents).
6) Just below it, write the aten op implementation with the max functions you just found.
7) Re-run the unit tests and make sure they're passing. Do not hesistate to use pytest.mark.parametrize 
   to test many input data types.
8) When you're done, make sure the whole test suite is passing with `uv run pytest -n 15` 
   and the linter with `uvx pre-commit run --all-files`.

## To find the correct type hints for a function
It may be hard to find the correct type hints for a function. What you should do in this case is:
1) Add an obviously wrong type hint, for example datetime.timezone in an aten function.
2) Run an existing unit test that calls this function.
3) Beartype will throw an error and give the name of the type being actually passed to the function.
4) Replace the type hint by the type given by beartype.
5) Run the unit test again to check that it works.
6) Run the whole test suite to verify that the type hint shouldn't be wider.
