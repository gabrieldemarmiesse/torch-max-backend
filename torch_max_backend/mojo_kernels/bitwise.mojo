import extensibility as compiler
from extensibility import ElementwiseBinaryOp, ElementwiseUnaryOp


@compiler.register("bitwise_and")
struct BitwiseAndKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs & rhs


@compiler.register("bitwise_or")
struct BitwiseOrKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs | rhs


@compiler.register("bitwise_xor")
struct BitwiseXorKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs ^ rhs


@compiler.register("bitwise_not")
struct BitwiseNotKernel(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return ~x
