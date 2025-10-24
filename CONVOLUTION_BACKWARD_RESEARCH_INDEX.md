# PyTorch convolution_backward Research - Document Index

## Quick Navigation

This index helps you navigate the comprehensive research on PyTorch's `convolution_backward` implementation.

### Main Document
- **CONVOLUTION_BACKWARD_RESEARCH.md** (this directory)
  - 374 lines, 13KB
  - Complete architectural analysis
  - All 6 research questions answered
  - Implementation guidelines for MAX backend
  - File references and takeaways

### Quick Answer Guide

If you have limited time, read these sections:

1. **Executive Summary** (CONVOLUTION_BACKWARD_RESEARCH.md)
   - Key findings at a glance
   - Takes 2-3 minutes to read

2. **Questions 1-6 Analysis** (CONVOLUTION_BACKWARD_RESEARCH.md)
   - Direct answers to your research questions
   - Code examples and location references
   - Takes 10-15 minutes to read

3. **Key Implementation Takeaways** (CONVOLUTION_BACKWARD_RESEARCH.md)
   - DO and DON'T rules
   - Critical pattern: output_mask
   - Bias gradient special case
   - Takes 5 minutes to read

### Deep Dive Documents

For comprehensive understanding, additional documents are available:

#### Mathematical & Implementation Details
- **convolution_backward_math_and_code.md**
  - Mathematical formulas for grad_input, grad_weight, grad_bias
  - Backend-specific behavior with code examples
  - Implementation patterns and edge cases
  - Reference: slow_conv2d_backward implementation

#### Architecture & Design Patterns
- **convolution_backward_architecture.md**
  - High-level architecture flow diagrams
  - Backend capability matrix
  - Data flow comparisons
  - Output mask pattern examples
  - Grouped and transposed convolution handling
  - Design rationale (why monolithic)

#### Executive Summary
- **RESEARCH_SUMMARY.md**
  - Quick answers to all 6 questions
  - Critical implementation rules
  - Implementation template
  - File reference table

---

## Research Questions Addressed

### 1. Decomposition Strategy
**Question**: Does PyTorch implement convolution_backward directly, or decompose it into separate operations?

**Answer**: Direct monolithic implementation, NOT decomposed.
- Single operation returns `(grad_input, grad_weight, grad_bias)` tuple
- Location: `/pytorch/aten/src/ATen/native/Convolution.cpp:1976-2263`
- No decompositions in `torch/_decomp/` or `torch/_refs/`

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 1

---

### 2. Backend Delegation
**Question**: How does PyTorch delegate the three gradient computations?

**Answer**: Backend selection function + switch statement with ~15 backends.
- `select_conv_backend()` at line 2029
- Routes to appropriate backend (CuDNN, MiOpen, MPS, MKL-DNN, Slow CPU, etc.)
- PyTorch provides automatic fallback for grad_bias (sum reduction)

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 2

---

### 3. Native Functions
**Question**: What native functions exist for convolution_backward?

**Answer**: Two public, many internal.
- **Public**: `convolution_backward`, `convolution_backward_overrideable`
- **Internal**: Backend-specific stubs (NOT for external backends)

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 3

---

### 4. Decompositions
**Question**: Are there decompositions in torch/_decomp or torch/_refs?

**Answer**: None. Implemented directly in C++.
- No decompositions found
- Direct C++ implementation in Convolution.cpp
- Registered as CompositeExplicitAutograd

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 4

---

### 5. Separate Operations
**Question**: Are there individual operations like convolution_backward_input, etc.?

**Answer**: Only internal stubs, NOT exposed.
- `_slow_conv2d_backward` for CPU reference
- Backend-specific operations (internal use only)
- External backends should NOT implement separately

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 5

---

### 6. PyTorch's Architecture Pattern
**Question**: How does PyTorch expect backends to implement this?

**Answer**: One monolithic function with output_mask parameter.
- Single function returning 3-tuple
- Critical: `bool[3] output_mask` for selective computation
- Return undefined tensors when mask[i] = False
- Bias gradient always computed as sum over batch and spatial dimensions

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → Question 6

---

## Key Architectural Insights

### The Monolithic Design
- All three gradients computed together
- Enables shared computation between grad_input and grad_weight
- Allows output masking to skip expensive gradients
- One backward operation per forward operation

**Document**: convolution_backward_architecture.md → "Why Not Decompose"

### Output Mask Pattern
```python
output_mask[0] = compute gradient w.r.t. input
output_mask[1] = compute gradient w.r.t. weight
output_mask[2] = compute gradient w.r.t. bias

# CRITICAL: Return undefined tensors when mask[i] = False
if output_mask[0]:
    grad_input = compute_grad_input()
else:
    grad_input = torch.Tensor()  # undefined, NOT zeros
```

**Document**: convolution_backward_architecture.md → "Output Mask Pattern"

### Bias Gradient is Special
- PyTorch computes as fallback if backend doesn't
- Always computed as simple sum reduction
- `grad_bias = grad_output.sum(dim={0, 2, 3})` for 2D convolution
- `grad_bias = grad_output.sum(dim={0, 2, 3, 4})` for 3D convolution

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → "Bias Gradient is Always a Sum"

---

## Implementation Guide

### For Your MAX Backend

Follow this pattern:

```python
def max_convolution_backward(
    grad_output, input, weight,
    bias_sizes, stride, padding,
    dilation, transposed, output_padding,
    groups, output_mask
) -> (Tensor, Tensor, Tensor):
    
    # Step 1: grad_input (if requested)
    if output_mask[0]:
        grad_input = compute_grad_input(...)
    
    # Step 2: grad_weight (if requested)
    if output_mask[1]:
        grad_weight = compute_grad_weight(...)
    
    # Step 3: grad_bias (if requested, optional)
    if output_mask[2]:
        grad_bias = grad_output.sum(dim=spatial_and_batch)
    
    return (grad_input, grad_weight, grad_bias)
```

**Document**: RESEARCH_SUMMARY.md → "Implementation Template"

### Critical Rules

DO:
1. Implement as single function returning tuple
2. Check output_mask and skip unnecessary computations
3. Return undefined tensors when mask[i] = False
4. Support all parameters (stride, padding, dilation, groups, transposed)

DON'T:
1. Decompose into separate operations
2. Return zeros when output_mask[i] = False
3. Expect multiple backward calls
4. Ignore output_mask parameter

**Document**: RESEARCH_SUMMARY.md → "Critical Implementation Rules"

---

## File References

### PyTorch Source Files

| Location | Purpose |
|----------|---------|
| `/pytorch/aten/src/ATen/native/Convolution.cpp:1976-2263` | Main convolution_backward implementation |
| `/pytorch/aten/src/ATen/native/Convolution.cpp:2029` | Backend selection |
| `/pytorch/aten/src/ATen/native/Convolution.cpp:2035-2241` | Backend dispatch switch statement |
| `/pytorch/aten/src/ATen/native/Convolution.cpp:2255-2259` | Bias gradient fallback |
| `/pytorch/aten/src/ATen/native/native_functions.yaml:1714-1728` | Operation registration |
| `/pytorch/aten/src/ATen/native/README.md:315-324` | CompositeExplicitAutograd documentation |

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → "File Reference Guide"

### Absolute Paths (from your working directory)

```
/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/Convolution.cpp
/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/native_functions.yaml
/home/stream/Projects/torch-max-backend/../pytorch/aten/src/ATen/native/README.md
```

---

## Mathematical Formulas

### Gradient Computations

**grad_input**: Essentially transposed convolution of grad_output with weight
```
grad_input[n, c_in, h, w] = sum over (c_out, h_k, w_k):
    grad_output[n, c_out, h_out, w_out] * weight[c_out, c_in, h_k, w_k]
```

**grad_weight**: Convolution of input with grad_output
```
grad_weight[c_out, c_in, h_k, w_k] = sum over (n, h_out, w_out):
    input[n, c_in, h_out*stride_h + h_k, w_out*stride_w + w_k] *
    grad_output[n, c_out, h_out, w_out]
```

**grad_bias**: Simple sum over batch and spatial dimensions
```
grad_bias[c_out] = sum over (n, h_out, w_out):
    grad_output[n, c_out, h_out, w_out]
```

**Document**: convolution_backward_math_and_code.md → "Mathematical Foundations"

---

## Backend Capability Matrix

```
Backend         grad_input   grad_weight  grad_bias
────────────────────────────────────────────────────
CuDNN           native       native       fallback
MiOpen (AMD)    native       native       native
MPS (Apple)     native       native       native
MKL-DNN (Intel) native       native       native
Slow CPU        native       native       native
Empty           zeros        zeros        zeros
External        native       native       native
```

**Document**: convolution_backward_architecture.md → "Backend Capability Matrix"

---

## Why This Architecture?

1. **Efficiency**: Backends can share computation between grad_input and grad_weight
2. **Flexibility**: Output masking allows skipping expensive gradients
3. **Simplicity**: One backward operation per forward operation
4. **Consistency**: All backends follow same interface
5. **Optimization**: PyTorch can automatically optimize away unused gradients

**Document**: convolution_backward_architecture.md → "Why Not Decompose"

---

## Integration with PyTorch

### torch.compile Integration
When user decorates with `@torch.compile(backend=max_backend)`:
1. Forward pass traced to FX graph
2. If backward needed, PyTorch autograd calls convolution_backward
3. Your implementation must respect output_mask for optimization
4. Return undefined tensors when mask[i] = False

**Document**: convolution_backward_architecture.md → "Integration with torch.compile"

### Autograd Flow
1. Forward saves tensors and parameters
2. Backward determines output_mask from requires_grad
3. Calls convolution_backward with grad_output and saved tensors
4. Returns gradients which are accumulated to leaf tensors

**Document**: CONVOLUTION_BACKWARD_RESEARCH.md → "How Autograd Calls convolution_backward"

---

## Reading Recommendations

### For Quick Understanding (15 minutes)
1. Read this index
2. Read "Executive Summary" in CONVOLUTION_BACKWARD_RESEARCH.md
3. Read "Key Implementation Takeaways" in CONVOLUTION_BACKWARD_RESEARCH.md
4. Read implementation template in RESEARCH_SUMMARY.md

### For Complete Understanding (1-2 hours)
1. Read CONVOLUTION_BACKWARD_RESEARCH.md (main document)
2. Read architecture diagrams in convolution_backward_architecture.md
3. Read mathematical formulas in convolution_backward_math_and_code.md
4. Study RESEARCH_SUMMARY.md implementation template

### For Deep Technical Dive (2-3 hours)
1. Read all documents above
2. Study PyTorch source files referenced
3. Look at slow_conv2d_backward implementation for reference
4. Understand backend selection logic

---

## Next Steps for Implementation

1. **Register operation** in your native_functions.yaml
   - Copy PyTorch's convolution_backward signature
   - Specify your backend dispatch

2. **Implement the math**
   - grad_input: transposed convolution of grad_output with weight
   - grad_weight: convolution of input with grad_output
   - grad_bias: (optional) sum of grad_output

3. **Handle output_mask**
   - Check mask for each gradient
   - Skip computation when mask[i] = False
   - Return undefined tensors (not zeros)

4. **Test comprehensively**
   - All gradients computed
   - Selective mask combinations
   - Transposed convolution
   - Grouped convolution
   - Various tensor shapes and dtypes

5. **Optimize**
   - Share computation between grad_input and grad_weight where possible
   - Profile backward pass
   - Leverage your backend's kernels

---

## Document Statistics

| Document | Location | Size | Purpose |
|----------|----------|------|---------|
| Main Research | CONVOLUTION_BACKWARD_RESEARCH.md | 13KB, 374 lines | Complete architectural analysis |
| Mathematics | convolution_backward_math_and_code.md | ~8KB | Formulas and implementation patterns |
| Architecture | convolution_backward_architecture.md | ~12KB | Diagrams and design patterns |
| Summary | RESEARCH_SUMMARY.md | ~8KB | Quick reference |
| Index | CONVOLUTION_BACKWARD_RESEARCH_INDEX.md | This file | Navigation guide |

**Total Research**: ~50KB of comprehensive documentation

---

## Questions or Further Research?

Refer back to this index to find the relevant document section for your question.

All documents are self-contained but cross-reference each other for deep dives.

The main document (CONVOLUTION_BACKWARD_RESEARCH.md) is the primary reference.

