import math
from compiler import register
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils import IndexList


@compiler.register("gelu_backward")
struct GeluBackwardKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
        approximate: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        grad_output: InputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[
            width: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, width]:
            var x = input.load[width](idx)
            var grad_out = grad_output.load[width](idx)
            var result = SIMD[dtype, width]()

            @parameter
            if approximate == "none":
                # Exact GELU backward using error function
                # Formula: grad = dy * (CDF + x * PDF)
                # where CDF = 0.5 * (1 + erf(x * M_SQRT1_2))
                #       PDF = (M_2_SQRTPI * M_SQRT1_2 * 0.5) * exp(-0.5 * x²)

                # Constants from PyTorch implementation
                alias M_SQRT1_2 = 0.7071067811865476  # sqrt(1/2) = 1/sqrt(2)
                alias PDF_CONSTANT = 0.39894228040143276  # M_2_SQRTPI * M_SQRT1_2 * 0.5

                # Compute CDF term: 0.5 * (1 + erf(x * M_SQRT1_2))
                cdf = 0.5 * (1.0 + math.erf(x * M_SQRT1_2))

                # Compute PDF term: PDF_CONSTANT * exp(-0.5 * x²)
                x_squared = x * x
                pdf = PDF_CONSTANT * math.exp(-0.5 * x_squared)

                # Gradient: grad_out * (CDF + x * PDF)
                result = grad_out * (cdf + x * pdf)
            elif approximate == "tanh":
                # Tanh approximation backward
                # Formula: grad = dy * (left_derivative + right_derivative)
                # See PyTorch CUDA implementation for details

                # Constants from PyTorch implementation
                alias k_Beta = 0.7978845608028654  # sqrt(2) * sqrt(2/π) * 0.5
                alias k_Kappa = 0.044715

                # Compute inner = kBeta * (x + kKappa * x³)
                x_squared = x * x
                x_cubed = x_squared * x
                inner = k_Beta * (x + k_Kappa * x_cubed)
                tanh_inner = math.tanh(inner)

                # Left term derivatives
                left = 0.5 * x
                right = 1.0 + tanh_inner
                left_derivative = 0.5 * right

                # Right term derivatives
                tanh_derivative = 1.0 - tanh_inner * tanh_inner
                inner_derivative = k_Beta * (1.0 + 3.0 * k_Kappa * x_squared)
                right_derivative = left * tanh_derivative * inner_derivative

                # Total gradient
                result = grad_out * (left_derivative + right_derivative)
            else:
                # This should never be reached as we validate approximate mode in Python
                # with a clear error message. Return zeros as safe fallback.
                result = SIMD[dtype, width](0)
            return result

        foreach[
            func,
            target=target,
        ](output, ctx)
