---
name: mojo-kernels
description: Guide for writing efficient GPU kernels in Mojo using the MAX framework, including patterns for element-wise operations, reductions, shared memory usage, and PyTorch backend integration
---

# Mojo GPU Kernel Development Skill

This skill provides comprehensive guidance for writing GPU kernels in Mojo using Modular's MAX framework, with specific focus on integration with the PyTorch backend.

## Using This Skill

Reference the skill in your prompts:

```
Use the mojo-kernels skill to help me implement a custom activation function.
```

```
I need to write a GPU kernel for matrix transpose. Use the mojo-kernels skill.
```

The skill provides:
- **Kernel patterns**: Element-wise, reductions, shared memory operations
- **Code templates**: Ready-to-use templates in `assets/` directory
- **Detailed references**: In-depth guides in `references/` directory
- **MAX operations catalog**: Complete reference of available MAX operations

## Core Concepts

### MAX GPU Kernel Architecture

Mojo GPU kernels are functions that execute on GPU devices via the MAX framework. Key differences from CUDA/HIP:
- Regular function syntax (no `__global__` decorator)
- Compile-time parameters for type safety
- Explicit address space annotations
- **Use LayoutTensor for tensor operations** - preferred high-level abstraction

**Reference**: See [MAX kernels documentation](../modular/max/kernels/src/) for implementation examples

### Development Workflow

When implementing a new ATen operation for the PyTorch backend:

1. **Research the operation**: Explore PyTorch source (`../pytorch`) to understand the ATen function signature and semantics
2. **Write tests first**: Add parametrized unit tests in `test_aten_functions.py`
3. **Find MAX equivalents**: Search `../modular/max/kernels` for similar operations or composable primitives
4. **Implement the kernel**: Add the operation to `aten_functions.py` with proper type hints
5. **Verify and iterate**: Run tests, fix type errors using beartype feedback, verify with multiple data types
6. **Lint before committing**: Run `uvx pre-commit run --all-files`

## Essential Imports

**Reference**: See [MAX kernels imports](../modular/max/kernels/src/) for comprehensive examples

Key modules:
- **gpu**: `barrier`, `global_idx`, `thread_idx`, `warp`, `WARP_SIZE`
- **gpu.host**: `DeviceContext`, `DeviceBuffer`
- **layout**: `LayoutTensor`, `RuntimeLayout`
- **algorithm**: `elementwise`, `reduce_launch`
- **math**: `ceildiv`, `exp`, `log`, `sqrt`, `tanh`

## Kernel Patterns

### Pattern 1: Element-wise Operation

**Use case**: Operations that process each element independently (activations, arithmetic)

**Quick example**:
```mojo
var tid = global_idx.x
if tid >= UInt(len):
    return
output[tid] = operation(input[tid])
```

**Full details**: See `references/elementwise_patterns.md` and [MAX kernels examples](../modular/max/kernels/src/)

### Pattern 2: Warp-level Reduction

**Use case**: Fast reductions within a warp (32 threads)

**Available warp operations**: `warp.sum()`, `warp.max()`, `warp.min()`, `warp.broadcast()`, `warp.shuffle()`

**Full details**: See `references/reduction_patterns.md`

### Pattern 3: Block-level Reduction

**Use case**: Reductions across entire thread block using shared memory

**Key points**:
- Use `stack_allocation` with `AddressSpace.SHARED`
- Call `barrier()` after writes before reads
- Two-level: warp reduction then across warps

**Full details**: See `references/reduction_patterns.md` and [MAX reduction kernels](../modular/max/kernels/src/)

### Pattern 4: Shared Memory Operations

**Use case**: Operations requiring neighbor access (convolution, pooling, stencil)

**Full details**: See `references/shared_memory_patterns.md`

## Type Handling

### SIMD Types

Key operations: `SIMD[DType.float32, 8]()`, `vec.reduce_add()`, `vec.cast[DType]()`, compile-time `[dtype: DType]` parameters

### Type Hints for ATen Operations

**Method**: Add wrong type hint (e.g., `datetime.timezone`), run test, beartype reveals correct type

**Common types**: `TensorValue`, `int`, `float`, `bool`, `list[int]`, `int | None`

## Device Context Management

**Pattern**:
```mojo
with DeviceContext() as ctx:
    var buffer = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(buffer, host_data)
    ctx.enqueue_function_checked[kernel, kernel](buffer, size, grid_dim=..., block_dim=...)
    ctx.enqueue_copy(host_result, buffer)
    ctx.synchronize()
```

**Reference**: See [MAX device context examples](../modular/max/kernels/src/)

## Activation Functions

**Examples**: `relu`, `gelu`, `leaky_relu`, `silu` - use `@always_inline` and SIMD types

**Full implementations**: See `assets/activation_templates.mojo` and [MAX models](../modular/max/examples/)

## Performance Considerations

- **Occupancy**: Block size 128-256 (multiple of 32), grid size `ceildiv(N, block_size)`
- **Memory coalescing**: Consecutive threads → consecutive addresses
- **Warp divergence**: Use `select()` over `if/else`, uniform control flow
- **Shared memory**: 32 banks - pad to avoid conflicts

**Deep dive**: See reference files and [MAX optimization guides](../modular/max/kernels/)

## Common Patterns from MAX

MAX provides high-level patterns that generate optimized kernels:

- **Element-wise**: `elementwise[operation, pack_size, target="gpu"](dims, ctx)`
- **Reductions**: `reduce_launch[..., reduce_fn, ...](shape, axis, init_value, ctx)`

**Examples**: See [MAX algorithm patterns](../modular/max/kernels/src/) and reference files

## Available MAX Operations

Categories: activations, normalization, reductions, linear algebra, convolution, pooling, attention, utilities

**Full catalog**: See `references/max_operations.md` and [MAX ops documentation](../modular/max/graph/ops/)

## Debugging and Profiling

**Environment variables**:
- `TORCH_MAX_BACKEND_PROFILE=1` - Enable timing
- `TORCH_MAX_BACKEND_VERBOSE=1` - Show graph structures

**Kernel debugging**:
- Always include bounds check: `if tid >= UInt(size): return`
- Verify `barrier()` calls match across threads
- Use beartype for type mismatches

**Testing**: Parametrize with multiple dtypes and shapes (see `assets/` templates)

## Best Practices

1. Use `LayoutTensor[dtype]` over raw pointers
2. Always bounds-check: `if tid >= UInt(size): return`
3. Use compile-time `@parameter` for specialization
4. Prefer warp primitives (`warp.sum()`)
5. Call `barrier()` after shared writes before reads
6. Use `with DeviceContext() as ctx:`
7. Test across dtypes and shapes
8. Lint: `uvx pre-commit run --all-files`

## Common Pitfalls

- Missing `barrier()` after shared memory writes
- Not using `ceildiv()` for grid dimension
- Using `if/else` instead of `select()` (warp divergence)
- Shared memory bank conflicts (stride-32 patterns)

## Quick Reference

### Templates & References
- **Templates**: `assets/elementwise_template.mojo`, `assets/reduction_template.mojo`, `assets/activation_templates.mojo`
- **Patterns**: `references/elementwise_patterns.md`, `references/reduction_patterns.md`, `references/shared_memory_patterns.md`
- **MAX ops**: `references/max_operations.md`

### Common Calculations
- Grid: `ceildiv(total_elements, block_size)`
- Thread: `global_idx.x`
- Warp: `thread_idx.x // UInt(WARP_SIZE)`
- Lane: `lane_id()`
- Sync: `barrier()` (block), implicit (warp)

### Pattern Selection
- **Element-wise**: No thread communication needed
- **Warp reduction**: ≤32 elements
- **Block reduction**: ≤1024 elements, shared memory
- **Shared memory**: Neighbor access, data reuse

### External Resources
- [MAX kernels source](../modular/max/kernels/src/)
- [MAX graph ops](../modular/max/graph/ops/)
- [MAX examples](../modular/max/examples/)
- [PyTorch ATen source](../pytorch/aten/)
