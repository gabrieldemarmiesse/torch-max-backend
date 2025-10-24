"""
Activation Function Templates

Common activation functions used in neural networks, optimized for GPU execution.
These can be used as-is or as reference implementations.
"""

from math import exp, tanh as math_tanh, erf, sqrt
from gpu import global_idx

# ============================================================================
# ReLU (Rectified Linear Unit)
# ============================================================================

@always_inline
fn relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    ReLU: max(0, x)

    Gradient: 1 if x > 0, else 0
    """
    return max(x, 0)


fn relu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """ReLU kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = max(input[tid], Scalar[dtype](0))


# ============================================================================
# Leaky ReLU
# ============================================================================

@always_inline
fn leaky_relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    negative_slope: Scalar[dtype] = 0.01
) -> SIMD[dtype, simd_width]:
    """
    Leaky ReLU: x if x > 0 else negative_slope * x

    Allows small gradient when x < 0 to prevent "dying ReLU"
    """
    return x.ge(0).select(x, negative_slope * x)


fn leaky_relu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    negative_slope: Scalar[dtype],
    len: Int,
):
    """Leaky ReLU kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = (x > 0).select(x, negative_slope * x)


# ============================================================================
# PReLU (Parametric ReLU)
# ============================================================================

fn prelu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],  # Per-channel negative slope
    channels: Int,
    elements_per_channel: Int,
):
    """
    PReLU: x if x > 0 else weight[channel] * x

    weight is learned during training
    """
    var tid = global_idx.x
    var total_size = channels * elements_per_channel

    if tid >= UInt(total_size):
        return

    var channel = tid / UInt(elements_per_channel)
    var x = input[tid]
    var slope = weight[channel]

    output[tid] = (x > 0).select(x, slope * x)


# ============================================================================
# ELU (Exponential Linear Unit)
# ============================================================================

@always_inline
fn elu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    alpha: Scalar[dtype] = 1.0
) -> SIMD[dtype, simd_width]:
    """
    ELU: x if x > 0 else alpha * (exp(x) - 1)

    Smooth transition, outputs can be negative
    """
    return x.ge(0).select(x, alpha * (exp(x) - 1))


fn elu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    alpha: Scalar[dtype],
    len: Int,
):
    """ELU kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = (x > 0).select(x, alpha * (exp(x) - 1))


# ============================================================================
# GELU (Gaussian Error Linear Unit)
# ============================================================================

@always_inline
fn gelu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    GELU: 0.5 * x * (1 + erf(x / sqrt(2)))

    Smooth, non-monotonic activation used in transformers
    Approximates x * Φ(x) where Φ is Gaussian CDF
    """
    alias inv_SQRT_2 = 0.70710678118654752440

    var val_half = 0.5 * x
    var erf_res = erf(x * inv_SQRT_2)

    # Use fused multiply-add for efficiency
    return val_half.fma(erf_res, val_half)


@always_inline
fn gelu_tanh_approx[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    GELU approximation using tanh (faster, less accurate):
    0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    alias SQRT_2_OVER_PI = 0.7978845608028654

    var x_cubed = x * x * x
    var inner = SQRT_2_OVER_PI * (x + 0.044715 * x_cubed)
    var tanh_res = math_tanh(inner)

    return 0.5 * x * (1.0 + tanh_res)


fn gelu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """GELU kernel (exact version)"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    alias inv_SQRT_2 = 0.70710678118654752440
    var x = input[tid]
    var val_half = 0.5 * x
    var erf_res = erf(x * inv_SQRT_2)

    output[tid] = val_half.fma(erf_res, val_half)


# ============================================================================
# SiLU / Swish
# ============================================================================

@always_inline
fn silu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))

    Used in modern architectures, smooth and unbounded above
    """
    return x / (1.0 + exp(-x))


fn silu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """SiLU kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = x / (1.0 + exp(-x))


# ============================================================================
# Sigmoid
# ============================================================================

@always_inline
fn sigmoid[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Sigmoid: 1 / (1 + exp(-x))

    Squashes input to (0, 1) range
    """
    return 1.0 / (1.0 + exp(-x))


fn sigmoid_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Sigmoid kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = 1.0 / (1.0 + exp(-input[tid]))


# ============================================================================
# Tanh (Hyperbolic Tangent)
# ============================================================================

@always_inline
fn tanh[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Squashes input to (-1, 1) range
    """
    return math_tanh(x)


fn tanh_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Tanh kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = math_tanh(input[tid])


# ============================================================================
# Softplus
# ============================================================================

@always_inline
fn softplus[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    beta: Scalar[dtype] = 1.0,
    threshold: Scalar[dtype] = 20.0
) -> SIMD[dtype, simd_width]:
    """
    Softplus: (1/beta) * log(1 + exp(beta * x))

    Smooth approximation of ReLU
    For numerical stability, returns x when beta * x > threshold
    """
    var beta_x = beta * x
    # For large values, softplus(x) ≈ x
    return (beta_x > threshold).select(
        x,
        (1.0 / beta) * log(1.0 + exp(beta_x))
    )


fn softplus_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    beta: Scalar[dtype],
    threshold: Scalar[dtype],
    len: Int,
):
    """Softplus kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    var beta_x = beta * x

    output[tid] = (beta_x > threshold).select(
        x,
        (1.0 / beta) * log(1.0 + exp(beta_x))
    )


# ============================================================================
# Mish
# ============================================================================

@always_inline
fn mish[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

    Smooth, non-monotonic, unbounded above
    """
    return x * math_tanh(log(1.0 + exp(x)))


fn mish_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Mish kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = x * math_tanh(log(1.0 + exp(x)))


# ============================================================================
# Hardswish
# ============================================================================

@always_inline
fn hardswish[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Hardswish: x * ReLU6(x + 3) / 6

    Piecewise linear approximation of Swish, faster to compute
    """
    var relu6 = min(max(x + 3, 0), 6)
    return x * relu6 / 6.0


fn hardswish_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Hardswish kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    var relu6 = min(max(x + 3, 0), 6)
    output[tid] = x * relu6 / 6.0


# ============================================================================
# Hardsigmoid
# ============================================================================

@always_inline
fn hardsigmoid[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    Hardsigmoid: ReLU6(x + 3) / 6

    Piecewise linear approximation of sigmoid
    """
    return min(max(x + 3, 0), 6) / 6.0


fn hardsigmoid_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """Hardsigmoid kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = min(max(x + 3, 0), 6) / 6.0


# ============================================================================
# GLU (Gated Linear Unit)
# ============================================================================

fn glu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    dim: Int,
    len: Int,
):
    """
    GLU: input[..., :dim] * sigmoid(input[..., dim:])

    Input is split in half along dimension, first half gated by sigmoid of second half
    Output size is half of input size
    """
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    var first_half = input[tid]
    var second_half = input[tid + UInt(len)]

    var gate = 1.0 / (1.0 + exp(-second_half))
    output[tid] = first_half * gate


# ============================================================================
# Threshold
# ============================================================================

@always_inline
fn threshold[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    threshold: Scalar[dtype],
    value: Scalar[dtype]
) -> SIMD[dtype, simd_width]:
    """
    Threshold: x if x > threshold else value
    """
    return x.gt(threshold).select(x, value)


fn threshold_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    threshold: Scalar[dtype],
    value: Scalar[dtype],
    len: Int,
):
    """Threshold kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = (x > threshold).select(x, value)


# ============================================================================
# ReLU6
# ============================================================================

@always_inline
fn relu6[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    """
    ReLU6: min(max(0, x), 6)

    Used in mobile networks for quantization-friendly activation
    """
    return min(max(x, 0), 6)


fn relu6_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    """ReLU6 kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = min(max(input[tid], 0), 6)


# ============================================================================
# CELU (Continuously Differentiable ELU)
# ============================================================================

@always_inline
fn celu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    alpha: Scalar[dtype] = 1.0
) -> SIMD[dtype, simd_width]:
    """
    CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    """
    var pos_part = max(x, 0)
    var neg_part = min(Scalar[dtype](0), alpha * (exp(x / alpha) - 1))
    return pos_part + neg_part


fn celu_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    alpha: Scalar[dtype],
    len: Int,
):
    """CELU kernel"""
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    var pos_part = max(x, Scalar[dtype](0))
    var neg_part = min(Scalar[dtype](0), alpha * (exp(x / alpha) - 1))
    output[tid] = pos_part + neg_part


# ============================================================================
# Quick Reference
# ============================================================================

"""
Activation Function Selection Guide:

1. ReLU: Default choice, fast, works well for most cases
   - Problem: Can "die" (all negative gradients → no learning)

2. Leaky ReLU / PReLU: Prevents dying ReLU
   - Leaky: Fixed slope for negative values
   - PReLU: Learnable slope

3. ELU: Smooth, negative values allowed
   - Better than ReLU for some tasks
   - Slower (uses exp)

4. GELU: Used in transformers (BERT, GPT)
   - Smooth, non-monotonic
   - More expensive (uses erf or tanh approximation)

5. SiLU/Swish: Used in EfficientNet, modern CNNs
   - Smooth, self-gated
   - x * sigmoid(x)

6. Sigmoid/Tanh: Classic, for output layers
   - Sigmoid: Binary classification
   - Tanh: Range (-1, 1)

7. Hardswish/Hardsigmoid: Mobile-friendly
   - Piecewise linear approximations
   - Faster, quantization-friendly

8. Mish: Experimental, claimed improvements
   - Similar to Swish but smoother
   - More expensive

Performance ranking (fastest to slowest):
ReLU < Leaky ReLU < Sigmoid < Tanh < SiLU < GELU (approx) < GELU (exact) < Mish
"""
