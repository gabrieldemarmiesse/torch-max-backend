from compiler import register
from gpu.host import DeviceContext
from gpu.id import block_idx
from layout import LayoutTensor
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor, foreach

from utils.index import IndexList


@compiler.register("bitwise_and")
struct BitwiseAndKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        y: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_bitwise_and[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) & y.load[width](idx)

        foreach[elementwise_bitwise_and, target=target](output, ctx)


@compiler.register("bitwise_and_scalar")
struct BitwiseAndScalarKernel:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int, //, target: StaticString, other: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        alias other_as_scalar = Scalar[dtype](other)

        @parameter
        @always_inline
        fn elementwise_bitwise_and[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) & other_as_scalar

        foreach[elementwise_bitwise_and, target=target](output, ctx)


@compiler.register("bitwise_or")
struct BitwiseOrKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        y: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_bitwise_or[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) | y.load[width](idx)

        foreach[elementwise_bitwise_or, target=target](output, ctx)


@compiler.register("bitwise_or_scalar")
struct BitwiseOrScalarKernel:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int, //, target: StaticString, other: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        alias other_as_scalar = Scalar[dtype](other)

        @parameter
        @always_inline
        fn elementwise_bitwise_or[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) | other_as_scalar

        foreach[elementwise_bitwise_or, target=target](output, ctx)


@compiler.register("bitwise_xor")
struct BitwiseXorKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        y: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_bitwise_xor[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) ^ y.load[width](idx)

        foreach[elementwise_bitwise_xor, target=target](output, ctx)


@compiler.register("bitwise_xor_scalar")
struct BitwiseXorScalarKernel:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int, //, target: StaticString, other: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        alias other_as_scalar = Scalar[dtype](other)

        @parameter
        @always_inline
        fn elementwise_bitwise_xor[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) ^ other_as_scalar

        foreach[elementwise_bitwise_xor, target=target](output, ctx)


@compiler.register("bitwise_not")
struct BitwiseNotKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        x: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_bitwise_not[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return ~x.load[width](idx)

        foreach[elementwise_bitwise_not, target=target](output, ctx)
