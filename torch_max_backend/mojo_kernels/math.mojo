from compiler import register
from math import ceil
from tensor import ElementwiseUnaryOp


@compiler.register("ceil")
struct CeilKernel(ElementwiseUnaryOp):
    @staticmethod
    fn elementwise[
        dtype: DType,
        width: Int,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return ceil(x)
