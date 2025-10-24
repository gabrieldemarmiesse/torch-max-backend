# PyTorch convolution_backward Implementation Research

## Complete Research Report: Understanding PyTorch's Backward Convolution Architecture

This document provides a comprehensive analysis of how PyTorch implements `convolution_backward` to support your MAX backend implementation.

---

## Executive Summary

PyTorch implements `convolution_backward` as a **monolithic operation** that computes all three gradients (grad_input, grad_weight, grad_bias) together, NOT as separate decomposed operations.

### Key Findings:

1. **Single Backward Operation**: PyTorch returns `(grad_input, grad_weight, grad_bias)` from one function
2. **Backend Selection**: Sophisticated routing selects among ~15 different backends based on tensor properties
3. **Output Masking**: `bool[3] output_mask` controls which gradients to compute
4. **Bias Special Case**: PyTorch computes grad_bias as a fallback (sum reduction) if backend doesn't
5. **No Decomposition**: Internal only - external backends should NOT split into separate operations

---

## Question-by-Question Analysis

### 1. Decomposition Strategy: Direct Implementation

**Does PyTorch decompose convolution_backward into separate operations?**

NO. PyTorch provides:
- `convolution_backward`: Main entry point (returns 3-tuple)
- `convolution_backward_overrideable`: For out-of-source backends
- Individual backend operations: Internal dispatcher stubs, NOT for external backends

Evidence from `/pytorch/aten/src/ATen/native/native_functions.yaml` (lines 1714-1728):

```yaml
- func: convolution_backward(Tensor grad_output, Tensor input, Tensor weight, 
    SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, 
    bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) 
    -> (Tensor, Tensor, Tensor)
  dispatch:
    CompositeExplicitAutograd, CUDA: convolution_backward
  autogen: convolution_backward.out
```

### 2. Backend Delegation: Switch-Based Router

**How does PyTorch delegate the three gradient computations?**

PyTorch uses a sophisticated backend selection and delegation pattern:

**Step 1: Backend Selection** (`Convolution.cpp:2029`)
```cpp
ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, 
                                          /*need_backward=*/ true, params);
```

**Step 2: Backend-Specific Computation** (`Convolution.cpp:2035-2241`)
```cpp
switch(backend) {
  case ConvBackend::Cudnn:
    std::tie(backend_grad_input, backend_grad_weight) = 
      cudnn_convolution_backward_stub(...);
    break;
  case ConvBackend::Mps:
    std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
      at::mps_convolution_backward(...);
    break;
  // ... ~13 other backends
}
```

**Backends Supported**: CudaDepthwise2d/3d, Cudnn, Miopen, MPS, MKL-DNN, Slow2d/3d, Overrideable, Empty

**Step 3: Bias Gradient Fallback** (`Convolution.cpp:2255-2259`)
```cpp
if (output_mask[2]) {
  if (!backend_grad_bias.defined()) {
    // Calculate bias gradients outside of the backend for those that don't support it.
    backend_grad_bias = grad_output.sum(
        (dim == 3) ? IntArrayRef{0, 2, 3, 4} : IntArrayRef{0, 2, 3});
  }
}
```

### 3. Native Functions: Two Public, Many Internal

**Which native functions exist?**

Public API:
- `convolution_backward`: Main backward operation
- `convolution_backward_overrideable`: For external backends

Internal (dispatcher stubs, not for external backends):
- `cudnn_convolution_backward`, `cudnn_convolution_transpose_backward`
- `miopen_convolution_backward`, `miopen_convolution_transpose_backward`
- `slow_conv2d_backward`, `slow_conv3d_backward`, `slow_conv_dilated2d_backward`
- Backend-specific depthwise operations
- MPS and MKL-DNN specific operations

### 4. Decompositions: None in Python

**Does PyTorch decompose convolution_backward in Python?**

NO decompositions found in:
- `torch/_decomp/decompositions.py`
- `torch/_refs/__init__.py` or subdirectories
- `torch/_refs/nn/functional/__init__.py`

The operation is implemented directly in C++ at `/pytorch/aten/src/ATen/native/Convolution.cpp`.

### 5. Separate Operations: Only Internally

**Are there individual backward operations like convolution_backward_input, etc.?**

Only internally:
- `_slow_conv2d_backward` (returns all three gradients)
- Backend-specific stubs (internal use only)

NOT exposed as public operations. External backends should implement `convolution_backward_overrideable`.

### 6. PyTorch's Architectural Pattern

**How does PyTorch expect backends to implement this?**

One monolithic function returning tuple:

```cpp
std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output_,      // Gradient from upstream
    const Tensor& input_,            // Original input (SAVED in forward)
    const Tensor& weight_,           // Original weight (SAVED in forward)
    const at::OptionalIntArrayRef bias_sizes_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask  // CRITICAL: selective computation
) {
    // 1. Validate and preprocess
    // 2. Select backend
    // 3. Call backend-specific implementation
    // 4. Fallback for bias gradient
    // 5. Return (grad_input, grad_weight, grad_bias)
}
```

---

## Deep Dive: Architectural Components

### Output Mask: The Key Optimization

The `bool[3] output_mask` parameter enables selective gradient computation:

```
output_mask[0] = compute gradient w.r.t. input
output_mask[1] = compute gradient w.r.t. weight
output_mask[2] = compute gradient w.r.t. bias
```

**Critical Rule**: When `output_mask[i] = False`, return **undefined tensor** (not zeros):

```python
if output_mask[0]:
    grad_input = compute_grad_input()
else:
    grad_input = torch.Tensor()  # undefined, not zeros
```

Why? PyTorch's autograd checks `tensor.defined()`:
- If undefined: skips gradient accumulation
- If zeros: wastes computation accumulating zeros

### Bias Gradient is Always a Sum

```cpp
// If backend didn't compute grad_bias:
backend_grad_bias = grad_output.sum(
    (dim == 3) ? IntArrayRef{0, 2, 3, 4} : IntArrayRef{0, 2, 3});
```

This is **always** the same formula:
- **2D Convolution** (4D tensor NCHW): sum over dims {0, 2, 3}
- **3D Convolution** (5D tensor NCDHW): sum over dims {0, 2, 3, 4}

Many backends don't support bias gradient computation, so PyTorch provides automatic fallback.

### Backend Capability Matrix

```
Backend         grad_input   grad_weight  grad_bias
─────────────────────────────────────────────────────
CuDNN           native       native       fallback
MiOpen (AMD)    native       native       native
MPS (Apple)     native       native       native
MKL-DNN (Intel) native       native       native
Slow CPU        native       native       native
Empty           zeros        zeros        zeros
External        native       native       native
```

---

## Mathematical Formulas: What Gets Computed

### Forward Pass (Reference)
```
output[n,c_out,h_out,w_out] = sum over (h_k, w_k, c_in):
    input[n, c_in, h_out*stride_h + h_k, w_out*stride_w + w_k] * 
    weight[c_out, c_in, h_k, w_k] + bias[c_out]
```

### Backward Gradient 1: grad_input
```
grad_input[n, c_in, h, w] = sum over (c_out, h_k, w_k):
    grad_output[n, c_out, h_out, w_out] * 
    weight[c_out, c_in, h_k, w_k]
    
where h_out = (h - h_k) / stride_h (must be integer and in bounds)
      w_out = (w - w_k) / stride_w (must be integer and in bounds)
```

Intuition: This is essentially a transposed convolution of grad_output with weight.

### Backward Gradient 2: grad_weight
```
grad_weight[c_out, c_in, h_k, w_k] = sum over (n, h_out, w_out):
    input[n, c_in, h_out*stride_h + h_k, w_out*stride_w + w_k] *
    grad_output[n, c_out, h_out, w_out]
```

Intuition: This is a convolution of input with grad_output (treating grad_output as the kernel).

### Backward Gradient 3: grad_bias
```
grad_bias[c_out] = sum over (n, h_out, w_out):
    grad_output[n, c_out, h_out, w_out]
```

Intuition: Simply sum grad_output over batch and spatial dimensions.

---

## How Autograd Calls convolution_backward

```
1. User calls loss.backward()
   
2. PyTorch autograd engine:
   a) Finds saved tensors: input, weight, bias_sizes, stride, padding, dilation, groups
   b) Determines output_mask based on requires_grad:
      - output_mask[0] = input.requires_grad
      - output_mask[1] = weight.requires_grad
      - output_mask[2] = bias.requires_grad (if bias exists)
   
3. Calls: convolution_backward(
      grad_output,           // from upstream layer
      saved_input,
      saved_weight,
      saved_bias_sizes,
      stride, padding, dilation,
      transposed,
      output_padding,
      groups,
      output_mask           // [True, True, True] or selective
   )
   
4. Returns: (grad_input, grad_weight, grad_bias)
   
5. Autograd accumulates to leaf tensors:
   input.grad += grad_input
   weight.grad += grad_weight
   bias.grad += grad_bias (if defined)
```

---

## File Reference Guide

| File | Lines | Purpose |
|------|-------|---------|
| `Convolution.cpp` | 1708-1718 | Native function definitions for forward convolution |
| `Convolution.cpp` | 1714-1718 | Native function definition for convolution_backward |
| `Convolution.cpp` | 1920-1953 | `_convolution_backward_nogroup_backend` helper |
| `Convolution.cpp` | 1955-1976 | Documentation and signature of convolution_backward |
| `Convolution.cpp` | 1976-2263 | **MAIN IMPLEMENTATION** of convolution_backward |
| `Convolution.cpp` | 2029 | Backend selection call |
| `Convolution.cpp` | 2035-2241 | Backend-specific switch statement |
| `Convolution.cpp` | 2255-2259 | Bias gradient fallback computation |
| `native_functions.yaml` | 1714-1728 | Operation registration and dispatch |
| `README.md` | 315-324 | Documentation on CompositeExplicitAutograd |

---

## Key Implementation Takeaways for MAX Backend

### DO:
1. Implement single function: `convolution_backward` returning `(grad_input, grad_weight, grad_bias)`
2. Support full signature with stride, padding, dilation, groups, transposed
3. Check `output_mask` and skip expensive computations when mask[i] = False
4. Return **undefined tensors** when mask is False (use `Tensor()` or equivalent)
5. Support optional grad_bias computation (or skip and let PyTorch do sum)

### DON'T:
1. Decompose into separate operations
2. Return zeros when output_mask[i] = False (return undefined)
3. Expect multiple backward calls
4. Ignore output_mask parameter (critical optimization)
5. Manually compute grad_bias if simple sum reduction works

---

## Grouped Convolution Handling

Weight shape: `(out_channels, in_channels // groups, *kernel_size)`

PyTorch optionally splits computation by groups in backward, but most backends handle it internally.

---

## Transposed Convolution

Same interface, but:
- `transposed = true`
- `output_padding` parameter is used
- Different spatial dimension mapping in forward and backward

---

## Why This Monolithic Design?

1. **Efficiency**: Backends can compute grad_input and grad_weight with shared intermediates
2. **Output Masking**: Skip expensive gradients when not needed
3. **Simplicity**: One backward per forward operation
4. **Consistency**: All backends use same interface
5. **Optimization**: PyTorch can automatically optimize away unused gradients

---

## External Backend Support: convolution_backward_overrideable

```yaml
- func: convolution_backward_overrideable(
    Tensor grad_output, Tensor input, Tensor weight,
    SymInt[] stride, SymInt[] padding, SymInt[] dilation,
    bool transposed, SymInt[] output_padding, SymInt groups,
    bool[3] output_mask
  ) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
  dispatch:
    CompositeExplicitAutograd: convolution_backward_overrideable
```

This is called when `ConvBackend::Overrideable` is selected (for out-of-source backends).

---

## Absolute File Paths for Reference

- **Main Implementation**: `/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/Convolution.cpp`
- **Native Definitions**: `/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/native_functions.yaml`
- **ATen Documentation**: `/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/README.md`

---

## Conclusion

PyTorch's `convolution_backward` is a sophisticated operation that:
1. Routes to backend-specific implementations
2. Supports selective gradient computation via output_mask
3. Provides fallback for bias gradient computation
4. Integrates seamlessly with autograd and torch.compile

For your MAX backend, implement a single monolithic function that respects output_mask and returns the three gradients as a tuple. PyTorch will handle backend selection and optimization automatically.

