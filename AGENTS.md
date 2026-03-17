# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations. It also has support for eager mode.

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

Always use uv to run commands to ensure the correct environment is activated. Never run python directly. Never run the whole test suite because it's too slow. Only run tests for the specific files you are working on.


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
  - The directory `./modular` contains MAX graph implementation examples and API reference


## To add support for an op

To add support for a new ATen operation, follow this test-driven development process:

### Step 1: Research the Operation
Ask a subagent to explore the PyTorch codebase `../pytorch` and look for:
- The signature of the ATen function
- The meaning of inputs and outputs
- Any important behavioral details
- Request a full report with this information

You can skip this step if the user provided the signature and the details of the operation in the initial request.

### Step 2: Write Unit Tests
Write unit tests in `test_aten_functions.py` using this op directly:
- Place tests somewhere in the middle of the file to avoid merge conflicts
- Use `pytest.mark.parametrize` to test multiple input data types and shapes
- Test edge cases and different parameter combinations

You shoud check in the unit test that the aten function has been called with this pattern:

```python
def test_aten_min_no_dim(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])
```

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
