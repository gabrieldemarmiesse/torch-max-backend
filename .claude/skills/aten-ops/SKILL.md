---
name: aten-ops
description: Guide for implementing and reviewing PyTorch ATen operations in the MAX backend, including test-driven development workflow, MAX operation discovery, type hints, and best practices
---

# ATen Operations Implementation Skill

This skill provides comprehensive guidance for implementing PyTorch ATen operations using Modular's MAX framework in the torch-max-backend project.

## Using This Skill

Reference the skill in your prompts:

```
Use the aten-ops skill to help me implement aten::relu
```

```
What pattern should I use for a reduction operation? Use the aten-ops skill.
```

The skill provides:
- **8-step TDD workflow**: Complete test-driven development process
- **Implementation patterns**: Templates for all operation types in `assets/`
- **Testing strategies**: Comprehensive testing guide in `references/`
- **Type hints guidance**: Beartype method for finding correct types

## Documentation Management

**IMPORTANT**: When working on ATen operations:

- **DO NOT commit** intermediate specification documents, research notes, or planning documentation
- **Store all intermediate docs** in `.claude/docs/` directory (this directory is git-ignored)
- Only commit final implementation code, tests, and essential reference materials in the skill directories
- Intermediate docs include: operation research notes, implementation plans, MAX operation discovery reports, type hint explorations

**Examples of what goes where**:
- `.claude/docs/`: Research on `aten::log_softmax` signature, notes on MAX equivalent functions, type hint discovery notes
- Committed to repo: Final `aten_functions.py` implementation, unit tests in `test_aten_functions.py`, templates in `assets/`, guides in `references/`

## Overview

ATen (A Tensor Library) is PyTorch's foundational tensor operation library. This skill guides you through the complete workflow of adding support for ATen operations in the MAX backend using test-driven development.

## Core Workflow: The 8-Step Process

### Step 1: Research the ATen Operation

**Goal**: Understand signature, semantics, and behavior

**How**: Ask subagent to explore `../pytorch` for the ATen function signature and semantics

**Where to look**:
- `../pytorch/aten/src/ATen/native/` - **C++ implementations** (CPU/CUDA reference code)
- `../pytorch/aten/src/ATen/native/native_functions.yaml` - Declarations
- `../pytorch/torch/` - Python API

**Finding C++ reference implementation**:
- **CUDA kernels**: `../pytorch/aten/src/ATen/native/cuda/` (e.g., `SoftMax.cu`, `Activation.cu`, `Reduce.cu`)
- **CPU kernels**: `../pytorch/aten/src/ATen/native/cpu/`
- **Example**: For `log_softmax`, see `pytorch/aten/src/ATen/native/cuda/SoftMax.cu (log_softmax_cuda_out)`
- **Why**: Understanding the C++ implementation helps you translate the logic to Mojo/MAX equivalents

### Step 2: Write Unit Tests

**Location**: `tests/test_aten_functions.py` (middle of file to avoid conflicts)

**Pattern**: Parametrize with multiple dtypes (float32, float16, bfloat16) and shapes (1D, 2D, 3D)

**Template**: See `assets/test_template.py` for complete patterns

### Step 3: Run Tests (Expect Failure)

**Command**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v
```

**Expected output**:
```
ValueError: Unsupported operation: aten.operation_name
```

This confirms the operation isn't implemented yet.

### Step 4: Locate or Add Operation Signature

**Location**: `torch_max_backend/aten_functions.py` (alphabetically sorted!)

**Format**: Add signature comment above function with `@map_to(aten.operation_name)` decorator

**Example**: See `assets/operation_template.py` for patterns

### Step 5: Research MAX Implementation

**How**: Ask subagent to search `../modular/max` for equivalent functions, or use `mojo-kernels` skill

**Where to look**:
- `../modular/max/graph/ops/` - High-level ops
- `../modular/max/kernels/src/` - Kernel implementations
- `../modular/max/examples/` - Usage patterns

**Decision tree**:
- Direct equivalent → use it: `max_ops.relu(self)`
- Can compose → combine: `max_ops.log(max_ops.softmax(self, axis=dim))`
- Backward op or no equivalent → custom Mojo kernel (use `mojo-kernels` skill)

### Step 6: Implement the Operation

**Location**: `torch_max_backend/aten_functions.py` (below signature comment)

**Key points**:
- Handle negative dimensions: `if dim < 0: dim = len(self.shape) + dim`
- Map PyTorch `dim` to MAX `axis`
- Use correct type hints: `TensorValue`, `int`, `float`, `bool`, `list[int]`, `int | None`
- **Document C++ reference**: Add comment linking to PyTorch implementation
  - Example: `# Based on: pytorch/aten/src/ATen/native/cuda/Activation.cu (relu_kernel)`
  - This helps future developers understand the reference implementation

**Patterns**: See `assets/operation_template.py` and `references/common_patterns.md`

### Step 7: Verify Implementation

**Run tests**: `uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v`

**Verify**: All tests pass, correct values/shapes, all dtypes work

**If tests fail**: Check MAX signature, dimension mapping, dtype handling

### Step 8: Run Linter

**Command**: `uvx pre-commit run --all-files`

**Important**: Do NOT run full test suite (too slow)

## Type Hints Discovery Process

### The Beartype Method

1. Add wrong type hint (e.g., `datetime.timezone`)
2. Run test → beartype reveals correct type
3. Replace with correct type
4. Verify

**Common types**: `TensorValue`, `list[TensorValue]`, `int`, `float`, `bool`, `list[int]`, `int | None`, `float | None`

## Operation Categories

**Categories**: Element-wise, Reductions, Shape operations, Linear algebra, Tensor creation/manipulation

**Examples and patterns**: See `references/common_patterns.md` for detailed implementations

**Key characteristics**:
- Element-wise: Independent per-element processing
- Reductions: Reduce dimensions (may return tuple)
- Shape: Change layout without modifying data
- Linear algebra: Matrix/tensor operations
- Creation: Combine/create tensors (broadcasting)

## Common Patterns

**Key patterns**:
- Negative dimensions: `if dim < 0: dim = len(self.shape) + dim`
- Optional parameters: Check for `None`, provide default
- Dimension mapping: PyTorch `dim` → MAX `axis`
- Broadcasting: MAX handles automatically
- In-place ops: Often same as out-of-place in graph mode

**Full examples**: See `references/common_patterns.md`

## Debugging

**Environment variables**:
- `TORCH_MAX_BACKEND_PROFILE=1` - Show timing
- `TORCH_MAX_BACKEND_VERBOSE=1` - Show FX graphs

**Common issues**:
- Type mismatch → Use beartype method
- Shape errors → Verify dimension/axis handling
- MAX op not found → Check name/compose ops/custom kernel
- Numerical differences → Adjust tolerance for float16

## Best Practices

1. Test-driven development (write tests first)
2. Parametrize tests (multiple dtypes, shapes)
3. Keep `aten_functions.py` alphabetically sorted
4. Add signature comments
5. Document C++ reference implementation in comments
6. Check for MAX equivalents first
7. Handle edge cases (empty tensors, negative dims)
8. Use beartype method for type hints
9. Lint before commit: `uvx pre-commit run --all-files`
10. Don't run full test suite during development
11. Store intermediate documentation in `.claude/docs/`, not in version control

## Integration with mojo-kernels Skill

For custom kernels: `Use the mojo-kernels skill to help me implement a custom kernel for [operation]`

## Quick Reference

**Commands**:
- Research: Ask subagent to explore `../pytorch` for signature
- Find MAX: Ask subagent to search `../modular/max` or use `mojo-kernels` skill
- Test: `uv run pytest tests/test_aten_functions.py::test_aten_op_name -v`
- Lint: `uvx pre-commit run --all-files`
- Debug: `TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest ...`

## Checklist

- [ ] Researched ATen op signature
- [ ] Found C++ reference implementation in PyTorch source
- [ ] Written parametrized tests (multiple dtypes, shapes)
- [ ] Tests fail with "Unsupported operation"
- [ ] Added to `aten_functions.py` (alphabetical order)
- [ ] Added signature comment
- [ ] Researched MAX equivalents
- [ ] Implemented with correct type hints
- [ ] Documented C++ reference in code comments
- [ ] All tests pass
- [ ] Tested float32, float16, bfloat16
- [ ] Tested edge cases
- [ ] Linter passes

## Resources

**Within this skill**:
- `assets/operation_template.py` - Implementation patterns
- `assets/test_template.py` - Test patterns
- `references/testing_guide.md` - Testing strategies
- `references/common_patterns.md` - All operation categories

**External**:
- [PyTorch ATen source](../pytorch/aten/)
- [MAX ops](../modular/max/graph/ops/)
- [MAX kernels](../modular/max/kernels/src/)
