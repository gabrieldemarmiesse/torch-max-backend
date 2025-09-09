
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
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype = dtype, rank = rank],
        x: InputTensor[dtype = dtype, rank = rank],
        y: InputTensor[dtype = dtype, rank = rank],
        ctx: DeviceContextPtr,
    ) raises:

        @parameter
        @always_inline
        fn elementwise_bitwise_and[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) & y.load[width](idx)

        foreach[elementwise_bitwise_and, target=target](output, ctx)
