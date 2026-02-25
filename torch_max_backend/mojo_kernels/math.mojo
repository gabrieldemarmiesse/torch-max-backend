import compiler
from math import ceil
from tensor import ElementwiseUnaryOp


@compiler.register("ceil")
struct CeilKernel(ElementwiseUnaryOp):
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "ceil requires floating point dtype"
        return ceil(x)
