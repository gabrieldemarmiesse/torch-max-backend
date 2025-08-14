# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations.

## Dependencies and Setup

- **Python**: >=3.12 required
- **Key Dependencies**: 
  - `modular>=25.4.0` (from Modular's nightly index)
  - `torch>=2.7.0`
  - `tabulate>=0.9.0`
- **Development Dependencies**:
  - `pytest>=8.4.1` (for testing)
  - `ruff>=0.12.7` (for linting/formatting)
- **Package Manager**: Uses `uv` for dependency management
- **Package Index**: Configured to use Modular's nightly Python index at `https://dl.modular.com/public/nightly/python/simple/`

## Common Commands

```bash
# Run tests
uv run pytest -v -n 5

# Run specific test file
uv run pytest tests/test_compiler.py


# Run linter/formatter
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
max-torch-backend/
├── torch_max_backend/       # Main package
│   ├── __init__.py         # Package exports
│   ├── compiler.py         # Core compiler implementation
│   ├── mappings.py         # PyTorch to MAX/Mojo operation mappings
│   └── ops.py              # Custom MAX operation wrapper
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_compiler.py   # Basic compilation tests
│   └── test_unsupported_ops.py  # Tests for unsupported operations
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
├── CLAUDE.md            # This file
└── README.md           # Project documentation
```

## Architecture

The project implements a custom PyTorch compiler backend (`my_compiler`) that:

1. **Graph Analysis**: Takes PyTorch FX graphs and analyzes their structure
2. **Operation Mapping**: Maps PyTorch operations to Mojo/MAX equivalents via `MAPPING_TORCH_TO_MOJO_FUNCTIONS`
3. **Custom Operations**: Creates MAX custom operations (`MyMaxOp`) that wrap the compiled functions
4. **Runtime Bridge**: Provides a bridge between PyTorch tensors and MAX execution

### Key Components

#### `torch_max_backend/compiler.py`
- **`my_compiler`**: Main compiler function that:
  - Accepts FX GraphModule and example inputs
  - Prints graph structure for debugging
  - Uses meta tensors to track shapes without memory allocation
  - Creates runtime function `max_add_i_want_to_use` that executes graph nodes
  - Returns wrapped function compatible with PyTorch

#### `torch_max_backend/ops.py`
- **`MyMaxOp`**: Custom MAX operation class that:
  - Extends `MaxOp` from `max.torch.torch`
  - Dynamically generates torch signatures based on input/output counts
  - Uses inspect.Signature to define parameter structure

#### `torch_max_backend/mappings.py`
- **`MAPPING_TORCH_TO_MOJO_FUNCTIONS`**: Dictionary mapping PyTorch ops to MAX/Mojo equivalents
- **Supported Operations**:
  - Arithmetic: `add`, `sub`, `mul`, `truediv`, `floordiv`, `pow`, `mod`
  - Math functions: `abs`, `cos`, `sin`
  - Both `operator` module and `torch` module variants supported

### Compilation Flow

1. PyTorch function decorated with `@torch.compile(backend=my_compiler)`
2. FX graph generated and passed to `my_compiler`
3. Graph nodes processed sequentially:
   - `placeholder` nodes map to function arguments
   - `call_function` nodes execute mapped operations
   - `output` nodes return results as tuple
4. Custom MAX operation created via `MyMaxOp` with:
   - Runtime function
   - CustomOpLibrary with KernelLibrary and MLIR context
   - Input/output type information
5. Compiled function allocates output tensors and executes custom op

## Testing

### Test Coverage
- **Basic Operations**: Addition on CPU and CUDA devices
- **Unsupported Operations**: 
  - Mathematical: `exp`, `log`, `sqrt`, `tanh`
  - Matrix operations: `matmul`
  - Tensor operations: `cat`, `mean`, `max`
  - Shape operations: `reshape` (marked as xfail)

### Test Fixtures
- `tensor_shapes`: Common tensor shapes for testing
- `devices`: Available devices (CPU, CUDA if available)

## Current Limitations

1. **Limited Operation Support**: Only basic arithmetic and trigonometric functions
2. **No Complex Operations**: Matrix multiplication, reductions, reshaping not yet supported
3. **Debug Output**: Compiler prints tabular graph representation (should be configurable)
4. **Error Handling**: Raises ValueError for unsupported operations

## Development Notes

- Uses Ruff for code formatting with:
  - Target Python 3.12
  - pyupgrade rules enabled
  - Magic trailing comma skipped in formatting
- Project configured for Modular's nightly builds
- Tests parametrized for multiple devices when available
- The directory `../modular/` contains the python code for the max graph. You can look inside to find out how the max graph is implemented. You can find examples of graphs in `modular/max/pipelines/architectures`.
- The directory `../pytorch/` contains the PyTorch source code. It might be helpful to look inside as 
  many things in `torch.compile` are not documented correctly.
  

The signature of all aten ops are in ressources/aten_ops.txt. Feel free to look inside if you need.
