from compiler import register
from tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList


@register("grayscale")
struct Grayscale:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[dtype = DType.float32, rank=2],
        img_in: InputTensor[dtype = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn color_to_grayscale[
            simd_width: Int
        ](idx: IndexList[img_out.rank]) -> SIMD[DType.float32, simd_width]:
            @parameter
            fn load(
                idx: IndexList[img_in.rank],
            ) -> SIMD[DType.float32, simd_width]:
                return img_in.load[simd_width](idx).cast[DType.float32]()

            row = idx[0]
            col = idx[1]

            # Load RGB values
            r = load(IndexList[3](row, col, 0))
            g = load(IndexList[3](row, col, 1))
            b = load(IndexList[3](row, col, 2))

            # Apply standard grayscale conversion formula
            gray = 0.21 * r + 0.71 * g + 0.07 * b
            return min(gray, 255)

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)
