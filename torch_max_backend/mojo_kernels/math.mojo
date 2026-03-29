import compiler
from std.math import ceil
from tensor import ElementwiseUnaryOp


@compiler.register("ceil")
struct CeilKernel(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "ceil requires floating point dtype"
        return ceil(x)
